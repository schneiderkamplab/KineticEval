"""Inspect integration. Private registry access is isolated and contract-tested."""
import importlib.metadata
import json
import math
from .components import load, fingerprint


def verify(config):
    if config.get('inspect_version') and importlib.metadata.version('inspect-ai') != config['inspect_version']:
        raise ValueError('Inspect version changed; import a new evaluation revision')
    if config.get('fingerprint') and fingerprint(config['factory']) != config['fingerprint']:
        raise ValueError('Inspect task code changed; import a new revision')


def task_from(config):
    verify(config)
    return load(config['factory'])(**config.get('kwargs', {}))


def state_for(sample, messages, model='mockllm/model', response=''):
    from inspect_ai.model import ModelName, ModelOutput, ChatMessageUser, ChatMessageSystem, ChatMessageAssistant
    from inspect_ai.solver import TaskState
    from inspect_ai.scorer import Target
    types = {'user': ChatMessageUser, 'system': ChatMessageSystem, 'assistant': ChatMessageAssistant}
    return TaskState(model=ModelName(model), sample_id=sample['id'], epoch=1, input=sample['input'],
        messages=[types[m['role']](content=m['content']) for m in messages],
        target=Target(sample['target']), choices=sample.get('choices') or None,
        metadata=sample.get('metadata', {}), output=ModelOutput.from_content(model, response))


async def import_task(factory, kwargs=None, limit=100):
    from inspect_ai._util.registry import registry_info, registry_params
    from inspect_ai.scorer._scorer import scorer_metrics
    from inspect_ai.model import ModelOutput
    from .schemas import Definition, Example, Component
    config = {'factory': factory, 'kwargs': kwargs or {}, 'fingerprint': fingerprint(factory),
              'inspect_version': importlib.metadata.version('inspect-ai')}
    task = task_from(config)
    if task.config.model_dump(exclude_none=True).keys() - {'temperature', 'max_tokens', 'top_p', 'seed'}:
        raise ValueError('Task generation configuration includes unsupported v1 options')
    if any(getattr(task, field, None) for field in ['setup', 'cleanup', 'sandbox', 'model_roles', 'tools']):
        raise ValueError('Task uses setup, cleanup, sandbox, or model roles; only single-generation QA imports are supported')
    solver = task.solver
    def flatten(value):
        try:
            info = registry_info(value)
        except Exception as exc:
            raise ValueError('Unregistered solver cannot be safely converted') from exc
        if info.name == 'inspect_ai/chain':
            return [leaf for child in value for leaf in flatten(child)]
        if info.name not in {'inspect_ai/generate', 'inspect_ai/prompt_template', 'inspect_ai/system_message', 'inspect_ai/multiple_choice'}:
            raise ValueError(f'Unsupported solver {info.name}; no approximate import was performed')
        if info.name == 'inspect_ai/multiple_choice' and registry_params(value).get('shuffle'):
            raise ValueError('Shuffled choices require a custom importer')
        return [value]
    solvers = flatten(solver)
    generators = {'inspect_ai/generate', 'inspect_ai/multiple_choice'}
    names = [registry_info(step).name for step in solvers]
    if sum(name in generators for name in names) != 1 or names[-1] not in generators:
        raise ValueError('Supported QA imports require one generation as the final solver step')
    config['solver_specs'] = [{'name': registry_info(s).name, 'params': registry_params(s)} for s in solvers]
    generation_config = task.config.model_dump(exclude_none=True)
    examples = []
    for i, sample in enumerate(task.dataset):
        if i >= limit:
            break
        if not isinstance(sample.target, str):
            raise ValueError('V1 requires a single string reference answer')
        if isinstance(sample.input, str):
            messages = [{'role': 'user', 'content': sample.input}]
        else:
            messages = [{'role': m.role, 'content': m.content} for m in sample.input]
        if any(not isinstance(m['content'], str) or m['role'] not in {'user', 'system', 'assistant'} for m in messages):
            raise ValueError('Only text QA messages are supported')
        data = Example(id=str(sample.id if sample.id is not None else i), input=sample.input if isinstance(sample.input, str) else messages[-1]['content'],
            target=sample.target, choices=sample.choices or [], metadata=sample.metadata or {},
            subset=str((sample.metadata or {}).get('subset', (sample.metadata or {}).get('subject', 'default')))).model_dump()
        state = state_for(data, messages)
        captured = []
        async def capture(state, **options):
            options = {k: v for k, v in options.items() if v is not None and k != 'tool_calls'}
            if options.keys() - {'temperature', 'max_tokens', 'top_p', 'seed'}:
                raise ValueError('Solver uses unsupported generation configuration')
            generation_config.update(options)
            captured.append([{'role': m.role, 'content': m.content} for m in state.messages])
            state.output = ModelOutput.from_content('mockllm/model', 'ANSWER: A')
            return state
        for step in solvers:
            state = await step(state, capture)
        if len(captured) != 1:
            raise ValueError('Import requires exactly one generation per example')
        data['messages'] = captured[0]
        examples.append(Example(**data))
    scorers = task.scorer if isinstance(task.scorer, list) else [task.scorer] if task.scorer else []
    if len(scorers) != 1:
        raise ValueError('V1 imports require exactly one scorer')
    metric_values = task.metrics or scorer_metrics(scorers[0])
    if isinstance(metric_values, dict) or any(isinstance(m, dict) for m in metric_values):
        raise ValueError('Dictionary-valued Inspect metrics require a custom adapter')
    def describe(value):
        import inspect
        module = inspect.getmodule(value)
        from pathlib import Path
        import hashlib
        return {'name': registry_info(value).name, 'params': registry_params(value), 'module': module.__name__,
                'code_hash': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    config['scorer_spec'] = describe(scorers[0])
    config['metric_specs'] = [describe(m) for m in metric_values]
    # Ensure only serializable, reconstructable component definitions are accepted.
    try:
        json.dumps(config)
        reconstruct('scorer', config['scorer_spec'])
        for spec in config['metric_specs']:
            reconstruct('metric', spec)
    except Exception as exc:
        raise ValueError('Inspect components must have serializable, reconstructable registry parameters') from exc
    config['metric_names'] = [registry_info(m).name for m in metric_values]
    definition = Definition(name=task.name or factory, description=task.description or '', examples=examples,
        source={'kind': 'inspect', **config, 'limit': limit, 'generation_config': generation_config,
                'solver': [{'name': registry_info(s).name, 'params': registry_params(s)} for s in solvers]},
        scorer=Component(name='inspect', config=config), metrics=[Component(name='inspect', config=config)])
    return definition


async def generate(model, messages, parameters, endpoint=None):
    from inspect_ai.model import get_model, GenerateConfig, ChatMessageUser, ChatMessageSystem, ChatMessageAssistant
    types = {'user': ChatMessageUser, 'system': ChatMessageSystem, 'assistant': ChatMessageAssistant}
    result = await get_model(model, base_url=endpoint).generate([types[m['role']](content=m['content']) for m in messages], config=GenerateConfig(**parameters))
    if result.error:
        raise ValueError('Inspect provider returned a generation error')
    return {'response': result.completion, 'metadata': result.model_dump(mode='json', exclude={'choices'})}


async def score_response(payload):
    from inspect_ai.scorer import Target
    config = payload['component']['config']
    verify(config)
    scorer = reconstruct('scorer', config['scorer_spec'])
    state = state_for(payload['sample'], payload['messages'], payload['model']['name'], payload['response'])
    # multiple_choice marks selected Choice objects after generation. Replay its
    # postprocessing against the stored output, then restore the actual prompt.
    for spec in payload['component']['config'].get('solver_specs', []):
        if spec['name'] == 'inspect_ai/multiple_choice':
            from inspect_ai.solver import multiple_choice
            actual_messages = list(state.messages)
            import copy
            state.messages = copy.deepcopy(actual_messages)
            async def stored_generation(value, **kwargs):
                return value
            state = await multiple_choice(**spec['params'])(state, stored_generation)
            state.messages = actual_messages
    result = await scorer(state, Target(payload['sample']['target']))
    value = result.value
    if value == 'C': value = 1.0
    elif value in ('I', 'N'): value = 0.0
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('V1 requires a scalar numeric or C/I Inspect score')
    return {'value': float(value), 'answer': result.answer, 'explanation': result.explanation,
            'inspect_score': result.model_dump(mode='json'), 'component': payload['component']}


def metrics(component, scores):
    from inspect_ai.scorer import SampleScore, Score
    from inspect_ai.scorer._scorer import scorer_metrics
    from inspect_ai._util.registry import registry_info
    verify(component['config'])
    funcs = [reconstruct('metric', spec) for spec in component['config']['metric_specs']]
    samples = [SampleScore(score=Score(**s.get('inspect_score', {'value': s['value']})), sample_id=i) for i, s in enumerate(scores)]
    return {registry_info(fn).name: fn(samples) for fn in funcs}


def reconstruct(kind, spec):
    import importlib
    from inspect_ai._util.registry import registry_create
    module = importlib.import_module(spec['module'])
    from pathlib import Path
    import hashlib
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != spec['code_hash']:
        raise ValueError('Inspect component code changed; import a new revision')
    return registry_create(kind, spec['name'], **spec['params'])
