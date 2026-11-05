"""Generation and judge providers. Credentials never enter stored payloads."""
import json
import os
import httpx
from .schemas import JudgeResult, ReportResult


class UCloud:
    def __init__(self, transport=None, endpoint=None):
        self.base_url = os.environ.get('UCLOUD_BASE_URL', '').rstrip('/')
        if endpoint and endpoint != self.base_url:
            raise ValueError('UCloud endpoint differs from the pinned configuration; align worker environment or create a new run')
        self.key = os.environ.get('UCLOUD_API_KEY', '')
        self.transport = transport
        if not self.base_url or not self.key:
            raise ValueError('Set UCLOUD_BASE_URL (including /v1) and UCLOUD_API_KEY in the worker environment')

    async def models(self):
        async with httpx.AsyncClient(transport=self.transport, timeout=30) as client:
            r = await client.get(f'{self.base_url}/models', headers={'Authorization': f'Bearer {self.key}'})
            if r.status_code != 200:
                raise ValueError(f'UCloud model discovery failed (HTTP {r.status_code})')
            return [row['id'] for row in r.json()['data']]

    async def chat(self, model, messages, parameters):
        async with httpx.AsyncClient(transport=self.transport, timeout=180) as client:
            r = await client.post(f'{self.base_url}/chat/completions', headers={'Authorization': f'Bearer {self.key}'},
                json={'model': model, 'messages': messages, **parameters})
            if r.status_code != 200:
                raise ValueError(f'UCloud inference failed (HTTP {r.status_code}); check model and quota')
            data = r.json()
            content = data['choices'][0]['message'].get('content')
            if not isinstance(content, str):
                raise ValueError('Provider did not return text content')
            return {'response': content, 'metadata': {'usage': data.get('usage'), 'model': data.get('model'), 'id': data.get('id')}}


async def generate(payload):
    model, messages = payload['model'], payload['messages']
    if model['backend'] == 'inspect':
        from .inspect_adapter import generate as inspect_generate
        return await inspect_generate(model['name'], messages, model['parameters'], model.get('endpoint'))
    if model['backend'] == 'ucloud':
        return await UCloud(endpoint=model.get('endpoint')).chat(model['name'], messages, model['parameters'])
    if model['backend'] == 'hf':
        import asyncio
        from .local_hf import cached_local_model
        return (await asyncio.to_thread(lambda: cached_local_model(json.dumps(model, sort_keys=True)).generate_batch([payload])))[0]
    params = model['parameters']
    prompt = messages[-1]['content']
    response = params.get('responses', {}).get(prompt, params.get('response', 'A'))
    if params.get('style') == 'sentence':
        response = f'The answer is {response}.'
    return {'response': response, 'metadata': {'provider': 'deterministic fixture', 'model': model['name']}}


def parse_json(text):
    value = text.strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(value)


async def judge(payload, report=False):
    from .judge_protocol import checked_protocol
    protocol = checked_protocol(payload,report)
    protocol_ref = {'sha256':protocol['sha256'], 'version':protocol['version']}
    config = payload['config']
    schema = ReportResult if report else JudgeResult
    if config['provider'] == 'fake':
        if report:
            return ReportResult(summary=('Fixture investigation: review recurring symptoms.' if payload['evidence'].get('purpose') == 'automatic_pattern_investigation' else 'Fixture report: review scoring and unresolved cases.'), findings=['This deterministic report is for workflow testing only.']).model_dump()
        finding = payload['evidence']['diagnostic']['finding']
        return JudgeResult(**{**finding, 'explanation': 'Deterministic fixture judge; not an independent correctness assessment.'}).model_dump()
    instruction = protocol['instruction']
    messages = [{'role': 'system', 'content': instruction},
        {'role': 'user', 'content': json.dumps(payload['evidence'], ensure_ascii=False)}]
    attempts = []
    for attempt in range(protocol['max_attempts']):
        result = await UCloud(endpoint=config.get('endpoint')).chat(config['model'], messages,
            {'temperature': 0, 'max_tokens': config['max_tokens']})
        attempts.append(result['metadata'])
        try:
            parsed = schema.model_validate(parse_json(result['response'])).model_dump()
            if not report:
                response = payload['evidence']['response']
                if any(excerpt not in response for excerpt in parsed['evidence']):
                    raise ValueError(protocol['evidence_error'])
            return {**parsed, '_protocol':protocol_ref, '_provider': {**result['metadata'], 'attempts': attempts}}
        except (ValueError, TypeError) as exc:
            if attempt == protocol['max_attempts'] - 1:
                raise
            messages.extend([{'role': 'assistant', 'content': result['response']},
                {'role': 'user', 'content': protocol['repair_prefix'] + str(exc)[:1500] + protocol['repair_suffix']}])
