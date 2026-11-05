"""Offline vLLM execution using the same pinned model and rendered prompts."""
import hashlib
import os
from .local_hf import local_digest, render_prompts, resolve_source


class VLLMModel:
    def __init__(self, spec, memory_fraction, tensor_parallel_size=1, max_sequences=1024):
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        import vllm
        from transformers import AutoTokenizer
        self.spec = spec
        source = resolve_source(spec)
        if spec.get('local_path') and local_digest(source) != spec['artifact_hash']:
            raise ValueError('Local model export changed since run creation')
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=spec.get('revision'))
        self.engine = vllm.LLM(model=source, revision=spec.get('revision'), tokenizer_revision=spec.get('revision'),
            dtype=spec.get('dtype', 'bfloat16'), tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=memory_fraction, max_model_len=spec.get('context_limit', 4096),
            max_num_seqs=max_sequences, enable_prefix_caching=True, language_model_only=True,
            compilation_config={'cudagraph_capture_sizes':[size for size in (1,2,4,8,16,32,64,128,256,512,1024) if size <= max_sequences]})
        self.version = vllm.__version__
        self.tp = tensor_parallel_size
        self.memory_fraction = memory_fraction

    def close(self):
        self.engine.llm_engine.engine_core.shutdown()

    def generate_batch(self, payloads):
        from vllm import SamplingParams
        if any(p['model'] != self.spec for p in payloads):
            raise ValueError('A batch must have identical model configuration')
        prompts = render_prompts(self.tokenizer, payloads, self.spec)
        parameters = self.spec['parameters']
        tokens = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        if any(len(ids) + parameters.get('max_tokens', 64) > self.spec.get('context_limit', 4096) for ids in tokens):
            raise ValueError('Input exceeds context budget; no truncation performed')
        params = SamplingParams(temperature=parameters.get('temperature', 0),
            top_p=parameters.get('top_p', 1), max_tokens=parameters.get('max_tokens', 64),
            seed=parameters.get('seed'))
        outputs = self.engine.generate([{'prompt_token_ids':ids} for ids in tokens], params, use_tqdm=False)
        results = []
        for prompt, output in zip(prompts, outputs):
            answer = output.outputs[0]
            results.append({'response':answer.text, 'metadata':{
                'engine':'vllm', 'vllm':self.version, 'tensor_parallel_size':self.tp,
                'memory_fraction':self.memory_fraction, 'dtype':self.spec.get('dtype','bfloat16'),
                'revision':self.spec.get('revision'), 'artifact_hash':self.spec.get('artifact_hash'),
                'input_tokens':len(output.prompt_token_ids), 'output_tokens':len(answer.token_ids),
                'generated_token_ids':list(answer.token_ids), 'finish_reason':answer.finish_reason,
                'rendered_chat_prompt':prompt, 'raw_response':answer.text,
                'prompt_format':'chat_template' if self.tokenizer.chat_template else 'plain_text',
                'chat_template_sha256':hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest() if self.tokenizer.chat_template else None,
                'chat_template_kwargs':self.spec.get('chat_template_kwargs',{})}})
        return results
