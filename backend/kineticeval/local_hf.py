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
        model.revision = HfApi().model_info(model.model_identifier or model.name, revision=model.revision).sha
    return model


def resolve_source(spec):
    if spec.get('local_path'):
        return spec['local_path']
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        source = snapshot_download(spec['name'], revision=spec['revision'], local_files_only=True)
        root = Path(source)
        weights = list(root.glob('*.safetensors')) or list(root.glob('pytorch_model*.bin'))
        indexes = list(root.glob('*.index.json'))
        complete = bool(weights)
        for index in indexes:
            mapping = json.loads(index.read_text()).get('weight_map', {})
            if mapping and not all((root / name).exists() for name in set(mapping.values())):
                complete = False
        if complete:
            # A previous text-only loader may have cached weights/tokenizers but
            # omitted processor metadata required by vLLM. Complete metadata
            # from the same pinned revision without downloading weights again.
            return snapshot_download(spec['name'], revision=spec['revision'],
                allow_patterns=['*.json','*.model','*.tiktoken','*.txt','*.jinja','tokenizer*','vocab*','merges*'])
    except LocalEntryNotFoundError:
        pass
    return snapshot_download(spec['name'], revision=spec['revision'],
        allow_patterns=['*.json','*.safetensors','*.bin','*.model','*.tiktoken','*.txt','*.jinja','tokenizer*','vocab*','merges*'])


def runtime_identity(spec):
    # Generation parameters and chat formatting do not require reloading weights.
    return {key: value for key, value in spec.items() if key not in {'parameters', 'chat_template_kwargs'}}


def render_prompts(tokenizer, payloads, spec):
    if tokenizer.chat_template:
        return [tokenizer.apply_chat_template(payload['messages'], tokenize=False, add_generation_prompt=True,
            **spec.get('chat_template_kwargs', {'enable_thinking': False})) for payload in payloads]
    prompts = []
    for payload in payloads:
        messages = payload['messages']
        if len(messages) != 1 or messages[0]['role'] != 'user':
            raise ValueError('Model has no chat template; only a single user prompt can be passed unchanged')
        prompts.append(messages[0]['content'])
    return prompts


def validate_prompt_batch(tokenizer, claims, spec):
    """Reject incompatible requests independently before allocating a GPU model."""
    valid, errors = [], []
    for claim in claims:
        try:
            prompt = render_prompts(tokenizer, [claim['payload']], spec)[0]
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if count + spec['parameters'].get('max_tokens', 64) > spec.get('context_limit', 4096):
                raise ValueError('Input exceeds context budget; no truncation performed')
            valid.append(claim)
        except Exception as exc:
            errors.append((claim, exc))
    return valid, errors


class LocalModel:
    def __init__(self, spec, device='cuda:0', memory_fraction=.20, compile_model=True):
        import torch
        import transformers
        self.torch, self.spec, self.device = torch, spec, device
        if device.startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        source = resolve_source(spec)
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
        if compile_model:
            self.model.forward = torch.compile(self.model.forward, dynamic=True)
        self.is_prefix_lm = bool(getattr(config, 'prefix_lm', False))
        text_config = getattr(config, 'text_config', config)
        self.context_limit = min(spec.get('context_limit', 4096), getattr(text_config, 'max_position_embeddings', 4096))
        self.runtime = {'engine': 'transformers', 'compiled': compile_model, 'tensor_parallel_size': 1, 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': device,
            'architecture': architecture, 'dtype': spec.get('dtype', 'bfloat16'), 'prefix_lm': self.is_prefix_lm,
            'revision': spec.get('revision'), 'artifact_hash': spec.get('artifact_hash'), 'memory_fraction': memory_fraction}

    def generate_batch(self, payloads):
        torch = self.torch
        parameters = payloads[0]['model']['parameters']
        if any(p['model'] != payloads[0]['model'] for p in payloads):
            raise ValueError('A local batch must use identical model configuration')
        prompts = render_prompts(self.tokenizer, payloads, self.spec)
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
                'chat_template_sha256': hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest() if self.tokenizer.chat_template else None,
                'prompt_format': 'chat_template' if self.tokenizer.chat_template else 'plain_text',
                'chat_template_kwargs': self.spec.get('chat_template_kwargs', {})}})
        return results


@lru_cache(maxsize=1)
def cached_local_model(spec_json):
    return LocalModel(json.loads(spec_json))
