"""Local Transformers inference with pinned weights and bounded GPU allocation."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=16)
def _hash_files(signature):
    digest = hashlib.sha256()
    for name, size, modified in signature:
        path = Path(name)
        digest.update(path.name.encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def local_digest(directory):
    root = Path(directory).resolve()
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {'.json', '.jinja', '.safetensors', '.bin', '.py', '.model'})
    if not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError('Local HF export has no weight files')
    return _hash_files(tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files))


def pin_model(model):
    if model.local_path:
        model.local_path = str(Path(model.local_path).resolve())
        model.artifact_hash = local_digest(model.local_path)
    else:
        from huggingface_hub import HfApi
        model.revision = HfApi().model_info(model.name, revision=model.revision).sha
    return model


class LocalModel:
    def __init__(self, spec, device='cuda:0', memory_fraction=.20):
        import torch
        import transformers
        self.torch, self.spec, self.device = torch, spec, device
        if device.startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        source = spec.get('local_path') or spec['name']
        if spec.get('local_path') and local_digest(source) != spec['artifact_hash']:
            raise ValueError('Local model export changed since run creation')
        kwargs = {'revision': spec.get('revision')} if not spec.get('local_path') else {}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(source, **kwargs)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = transformers.AutoConfig.from_pretrained(source, **kwargs)
        architecture = config.architectures[0]
        model_cls = getattr(transformers, architecture)
        self.model = model_cls.from_pretrained(source, dtype=getattr(torch, spec.get('dtype', 'bfloat16')),
            attn_implementation='sdpa', **kwargs).to(device).eval()
        self.is_prefix_lm = bool(getattr(config, 'prefix_lm', False))
        text_config = getattr(config, 'text_config', config)
        self.context_limit = min(spec.get('context_limit', 4096), getattr(text_config, 'max_position_embeddings', 4096))
        self.runtime = {'torch': torch.__version__, 'transformers': transformers.__version__, 'device': device,
            'architecture': architecture, 'dtype': spec.get('dtype', 'bfloat16'), 'prefix_lm': self.is_prefix_lm,
            'revision': spec.get('revision'), 'artifact_hash': spec.get('artifact_hash'), 'memory_fraction': memory_fraction}

    def generate_batch(self, payloads):
        torch = self.torch
        parameters = payloads[0]['model']['parameters']
        if any(p['model'] != payloads[0]['model'] for p in payloads):
            raise ValueError('A local batch must use identical model configuration')
        prompts = [self.tokenizer.apply_chat_template(p['messages'], tokenize=False, add_generation_prompt=True,
            **self.spec.get('chat_template_kwargs', {'enable_thinking': False})) for p in payloads]
        inputs = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt')
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        budget = parameters.get('max_tokens', 64)
        if max(lengths) + budget > self.context_limit:
            raise ValueError(f'Input exceeds context budget ({max(lengths)} + {budget} > {self.context_limit}); no truncation performed')
        inputs = {key: value.to(self.device) for key, value in inputs.items() if key in {'input_ids', 'attention_mask'}}
        if self.is_prefix_lm:
            inputs['token_type_ids'] = inputs['attention_mask'].clone()
        temperature = parameters.get('temperature', 0)
        options = {'max_new_tokens': budget, 'do_sample': temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id, 'use_cache': True}
        if temperature > 0:
            options.update(temperature=temperature, top_p=parameters.get('top_p', 1))
        if parameters.get('seed') is not None:
            torch.manual_seed(parameters['seed'])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **options)
        generated = outputs[:, inputs['input_ids'].shape[1]:].cpu().tolist()
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        results = []
        for prompt, ids, length in zip(prompts, generated, lengths):
            end = next((i+1 for i, token in enumerate(ids) if token in eos), len(ids))
            ids = ids[:end]
            results.append({'response': self.tokenizer.decode(ids, skip_special_tokens=True), 'metadata': {
                **self.runtime, 'raw_response': self.tokenizer.decode(ids, skip_special_tokens=False),
                'generated_token_ids': ids, 'rendered_chat_prompt': prompt, 'input_tokens': length, 'output_tokens': len(ids),
                'finish_reason': 'stop' if ids and ids[-1] in eos else 'length',
                'chat_template_sha256': hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                'chat_template_kwargs': self.spec.get('chat_template_kwargs', {})}})
        return results


@lru_cache(maxsize=1)
def cached_local_model(spec_json):
    return LocalModel(json.loads(spec_json))
