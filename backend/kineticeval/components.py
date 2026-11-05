"""Versioned installed components. No generated code is evaluated."""
import importlib
import inspect
import hashlib
import re
import math
from .schemas import Finding


def load(ref):
    module, name = ref.split(':', 1)
    # Resolve immutable component references recorded before the project rename.
    prefix, _, suffix = module.partition('.')
    if prefix in {'dynaeval', 'smarteval'}:
        module = 'kineticeval' + ('.' + suffix if suffix else '')
    return getattr(importlib.import_module(module), name)


def fingerprint(ref):
    obj = load(ref)
    module = inspect.getmodule(obj)
    path = getattr(module, '__file__', None)
    if path:
        with open(path, 'rb') as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    raise ValueError(f'Cannot fingerprint component {ref}')


def normalize(value):
    return value.strip().rstrip('.!').strip().casefold()


def extract(response, choices=None):
    value = normalize(response)
    if value in ('yes', 'no', 'a', 'b', 'c', 'd'):
        return value
    match = re.fullmatch(r'(?:the (?:correct )?answer is|det rigtige svar er|answer\s*:)\s*([A-Z])[.!]?', response.strip(), re.I)
    if match and (not choices or ord(match[1].upper()) - 65 < len(choices)):
        return match[1].casefold()
    return None


def qa(payload, config):
    response, target = payload['response'], payload['sample']['target']
    prompt = '\n'.join(m['content'] for m in payload['messages'])
    value = extract(response, payload['sample'].get('choices'))
    semantic = True if normalize(response) == normalize(target) else None if value is None else value == normalize(target)
    # Equivalents are language-scoped and configurable, never first-letter matching.
    equivalents = config.get('equivalents', {'yes': ['affirmative'], 'no': ['negative']})
    for answer, words in equivalents.items():
        if normalize(response) in [normalize(w) for w in words]:
            semantic = normalize(answer) == normalize(target)
    expected = payload.get('expected_format', 'none')
    explicit = bool(re.search(r'(?:only|exactly|single|kun)\s+(?:one\s+)?(?:letter|[a-d](?:\s*[,/]\s*[a-d])*|yes|no|bogstav)|(?:answer|respond|output)\s+(?:with\s+)?(?:only\s+)?(?:yes\s+or\s+no|a,?\s*b,?\s*c,?\s*(?:or\s+)?d)', prompt, re.I))
    if expected == 'answer_letter':
        explicit = bool(re.search(r"format:.*ANSWER:.*LETTER", prompt, re.I))
    compliance = None
    if expected == 'answer_letter' and explicit:
        compliance = bool(re.fullmatch(r'ANSWER: [A-Z]', response))
    if expected == 'letter' and explicit:
        compliance = bool(re.fullmatch(r'[A-Z]', response))
    if expected == 'yes_no' and explicit:
        compliance = response in ('yes', 'no')
    labels = []
    score = payload['score']['value']
    if semantic is False:
        labels.append('wrong answer')
    elif semantic and score < 1:
        labels.append('scoring too harsh' if normalize(response) == normalize(target) else 'semantically correct')
        if expected != 'none' and not explicit:
            labels.append('insufficient instructions')
    if semantic is None:
        labels.append('other')
    scorer_correct = True if score == 1 and semantic else False if semantic and score < 1 and compliance is not False else None
    if semantic is False and compliance is True:
        scorer_correct = score < 1
    return Finding(semantic_correct=semantic, format_compliance=compliance,
        instruction_adequacy=True if explicit else None,
        scorer_correct=scorer_correct,
        explanation='Conservative QA extraction; absence of a recognized instruction remains unresolved.',
        evidence=[response], confidence='high' if semantic is not None else 'low', labels=labels).model_dump()


async def score(payload):
    component = payload['component']
    name = component['name']
    response, target = payload['response'], payload['sample']['target']
    if name == 'exact':
        value = float(response == target)
    elif name == 'normalized':
        value = float(normalize(response) == normalize(target))
    elif name == 'mcq':
        value = float(extract(response, payload['sample'].get('choices')) == normalize(target))
    elif name == 'inspect':
        from .inspect_adapter import score_response
        return await score_response(payload)
    elif name.startswith('python:'):
        fn = load(name.removeprefix('python:'))
        result = fn(payload, component['config'])
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict) or not isinstance(result.get('value'), (int, float)) or not math.isfinite(result['value']):
            raise ValueError('Custom scorer must return a numeric value in a dict')
        return result
    else:
        raise ValueError(f'Unknown scorer: {name}')
    return {'value': value, 'answer': response, 'explanation': name, 'component': component}


def diagnose(payload):
    findings = []
    combined = Finding().model_dump()
    conflicts = []
    for component in payload['components']:
        name = component['name']
        if name == 'qa':
            finding = qa(payload, component['config'])
        elif name == 'aliases':
            aliases = component['config'].get('aliases', {})
            answer = aliases.get(normalize(payload['response']))
            finding = Finding(semantic_correct=None if answer is None else normalize(answer) == normalize(payload['sample']['target']),
                explanation='Reviewed explicit alias mapping', evidence=[payload['response']]).model_dump()
        elif name.startswith('python:'):
            finding = Finding.model_validate(load(name.removeprefix('python:'))(payload, component['config'])).model_dump()
        else:
            raise ValueError(f'Unknown heuristic: {name}')
        findings.append({'component': component, 'finding': finding})
        for key in ('semantic_correct', 'format_compliance', 'instruction_adequacy', 'scorer_correct'):
            if finding[key] is not None:
                if combined[key] is not None and combined[key] != finding[key]:
                    conflicts.append(key)
                combined[key] = finding[key]
        combined['labels'] = sorted(set(combined['labels'] + finding['labels']))
    for key in conflicts:
        combined[key] = None
    return {'finding': combined, 'heuristics': findings, 'conflicts': sorted(set(conflicts)),
        'unresolved': [k for k in ('semantic_correct', 'format_compliance', 'instruction_adequacy', 'scorer_correct') if combined[k] is None]}


def aggregate(payload):
    import math
    values = payload['scores']
    result = {}
    for component in payload['components']:
        name = component['name']
        if not values:
            continue
        if name == 'accuracy':
            result[name] = sum(s['value'] for s in values) / len(values)
        elif name == 'inspect':
            from .inspect_adapter import metrics
            result.update(metrics(component, values))
        elif name.startswith('python:'):
            result[name] = load(name.removeprefix('python:'))(values, component['config'])
        else:
            raise ValueError(f'Unknown metric {name}')
    return {'metrics': {k: v if isinstance(v, (int, float)) and math.isfinite(v) else None for k, v in result.items()},
            'scored': len(values)}
