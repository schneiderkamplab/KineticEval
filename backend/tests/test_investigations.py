import asyncio
from sqlalchemy import select
from kineticeval import providers
from kineticeval.db import Database, Job
from kineticeval.service import Service
from kineticeval.worker import Worker
from kineticeval.fixtures import demo_definition


def test_scheduler_automatically_investigates_truncation_and_records_recommendations(tmp_path, monkeypatch):
    observed = []
    async def generate(payload):
        return {'response':'Unfinished explanation', 'metadata':{'finish_reason':'length',
            'output_tokens':64,'rendered_chat_prompt':'rendered input'}}
    original = providers.judge
    async def judge(payload, report=False):
        if payload['evidence'].get('purpose') == 'automatic_pattern_investigation':
            observed.append(payload)
            assert report
            return {'summary':'Investigate truncation', 'findings':['All sampled responses reached the token limit.'],
                'suggestions':[{'kind':'experiment','title':'Compare token budgets','rationale':'Test the truncation hypothesis',
                    'patch':{'parameters':{'max_tokens':256}},'example_ids':[]}]}
        return await original(payload, report)
    monkeypatch.setattr(providers, 'generate', generate)
    monkeypatch.setattr(providers, 'judge', judge)
    service = Service(Database(tmp_path/'investigations.db'))
    definition = demo_definition()
    definition['judge'] = {'enabled':True,'provider':'fake'}
    revision = service.create_evaluation(definition)['revision_id']
    run = service.create_run({'revision_id':revision,'models':[{'name':'any-model','backend':'fake',
        'parameters':{'max_tokens':64}}]})['id']
    asyncio.run(Worker(service.db).drain())
    result = service.results(run)
    assert result['run']['status'] == 'completed'
    assert len(observed) == 1
    evidence = observed[0]['evidence']
    assert evidence['triggers']['token_limit'] == 3
    assert evidence['model_config']['parameters']['max_tokens'] == 64
    assert evidence['results'][0]['generation']['metadata']['rendered_chat_prompt'] == 'rendered input'
    assert len(result['investigations']) == 1
    assert result['proposals'][0]['status'] == 'pending'
    assert result['proposals'][0]['proposal']['origin_job'] == result['investigations'][0]['id']
    assert result['run']['config']['models'][0]['parameters']['max_tokens'] == 64
    # Resuming scheduling does not duplicate investigation or proposals.
    service.control(run, 'resume')
    asyncio.run(Worker(service.db).drain())
    assert len(observed) == 1
    assert len(service.results(run)['proposals']) == 1


def test_disabled_judge_never_schedules_investigation(tmp_path, monkeypatch):
    async def generate(payload):
        return {'response':'unfinished','metadata':{'finish_reason':'length'}}
    monkeypatch.setattr(providers, 'generate', generate)
    service = Service(Database(tmp_path/'disabled.db'))
    rev = service.create_evaluation(demo_definition())['revision_id']
    run = service.create_run({'revision_id':rev,'models':[{'name':'any','backend':'fake'}]})['id']
    asyncio.run(Worker(service.db).drain())
    assert service.results(run)['investigations'] == []
