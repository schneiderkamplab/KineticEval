import asyncio
import pytest
import httpx
from kineticeval.inspect_adapter import import_task, score_response, metrics, generate
from kineticeval.providers import UCloud
from kineticeval.importers import from_jsonl


def test_inspect_import_scoring_metrics_generation():
    definition = asyncio.run(import_task('kineticeval.fixtures:inspect_fixture'))
    assert definition.examples[0].messages[-1]['content'] == 'What is 2 + 2?\nAnswer briefly.'
    payload = {'component': definition.scorer.model_dump(), 'sample': definition.examples[0].model_dump(),
        'messages': definition.examples[0].messages, 'model': {'name': 'mockllm/model'}, 'response': '4'}
    result = asyncio.run(score_response(payload))
    assert result['value'] == 1
    assert metrics(definition.metrics[0].model_dump(), [result])['inspect_ai/mean'] == 1
    result = asyncio.run(generate('mockllm/model', [{'role': 'user', 'content': 'Hello'}], {}))
    assert isinstance(result['response'], str)


def test_ucloud_contract(monkeypatch):
    monkeypatch.setenv('UCLOUD_BASE_URL', 'https://test.invalid/v1')
    monkeypatch.setenv('UCLOUD_API_KEY', 'fixture-key')
    def handler(request):
        assert request.headers['Authorization'] == 'Bearer fixture-key'
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'fixture-model'}]})
        return httpx.Response(200, json={'choices': [{'message': {'content': 'A'}}], 'model': 'fixture-model'})
    provider = UCloud(httpx.MockTransport(handler))
    assert asyncio.run(provider.models()) == ['fixture-model']
    assert asyncio.run(provider.chat('fixture-model', [{'role': 'user', 'content': 'Q'}], {}))['response'] == 'A'


def test_jsonl_gsm8k_transform():
    result = from_jsonl('math', '{"question":"2+2?","answer":"Reasoning #### 4"}',
        {'input': 'question', 'target': 'answer', 'target_transform': 'gsm8k'})
    assert result.examples[0].target == '4'


def test_inspect_mcq_postprocessing_preserved():
    definition = asyncio.run(import_task('kineticeval.fixtures:inspect_mcq_fixture'))
    payload = {'component': definition.scorer.model_dump(), 'sample': definition.examples[0].model_dump(),
        'messages': definition.examples[0].messages, 'model': {'name': 'mockllm/model'}, 'response': 'ANSWER: B'}
    assert asyncio.run(score_response(payload))['value'] == 1
    assert asyncio.run(score_response({**payload, 'response': 'ANSWER: A'}))['value'] == 0
    assert asyncio.run(score_response({**payload, 'response': ''}))['value'] == 0


def test_judge_structured_output_and_evidence_validation(monkeypatch):
    import json
    from kineticeval import providers
    async def response(self, model, messages, parameters):
        assert 'untrusted data' in messages[0]['content']
        return {'response':json.dumps({'semantic_correct':True,'evidence':['not in response']}),'metadata':{}}
    monkeypatch.setenv('UCLOUD_BASE_URL','https://fixture.invalid/v1')
    monkeypatch.setenv('UCLOUD_API_KEY','fixture-key')
    monkeypatch.setattr(providers.UCloud,'chat',response)
    payload={'config':{'provider':'ucloud','model':'model','max_tokens':1000},'evidence':{'response':'A'},'dimensions':['semantic_correct']}
    with pytest.raises(ValueError,match='evidence'):
        asyncio.run(providers.judge(payload))


def test_report_output_becomes_reviewable_proposals(tmp_path,monkeypatch):
    from kineticeval import providers
    from kineticeval.db import Database
    from kineticeval.service import Service
    from kineticeval.worker import Worker
    from kineticeval.fixtures import demo_definition
    original=providers.judge
    async def report(payload, report=False):
        result=await original(payload,report)
        if report:
            result['suggestions']=[{'kind':'prompt','title':'Clarify output','rationale':'Avoid hidden format expectations',
                'patch':{'prompt_template':'{input}\n{choices}\nAnswer with only one letter.'},'example_ids':['capital']}]
        return result
    monkeypatch.setattr(providers,'judge',report)
    s=Service(Database(tmp_path/'report.db'))
    d=demo_definition();d['judge']={'enabled':True,'provider':'fake'}
    rev=s.create_evaluation(d)['revision_id']
    run=s.create_run({'revision_id':rev,'models':[{'name':'fixture','backend':'fake'}]})['id']
    asyncio.run(Worker(s.db).drain())
    proposal=s.results(run)['proposals'][0]
    assert proposal['status']=='pending'
    from kineticeval.proposals import review
    new=review(s,proposal['id'],'approve')['revision_id']
    assert new!=rev
    assert s.results(run)['definition']['prompt_template']=='{input}\n{choices}'


@pytest.mark.skipif(__import__('os').environ.get('KINETICEVAL_LIVE_TESTS') != '1', reason='Live UCloud checks are opt-in')
def test_live_ucloud_judge():
    import os
    from kineticeval.providers import judge
    from kineticeval.schemas import JudgeConfig
    available=asyncio.run(UCloud().models())
    model=os.environ.get('UCLOUD_JUDGE_MODEL')
    assert model in available, 'Set UCLOUD_JUDGE_MODEL to an ID returned by /models'
    config=JudgeConfig(enabled=True,model=model).model_dump()
    result=asyncio.run(judge({'config':config,'dimensions':['semantic_correct'], 'evidence':{
        'sample':{'input':'What is 2 + 2?','target':'4'}, 'messages':[{'role':'user','content':'What is 2 + 2?'}],
        'response':'The answer is 4.', 'score':{'value':0}, 'diagnostic':{'finding':{'semantic_correct':None}}}}))
    assert result['semantic_correct'] is True


def test_inspect_rescoring_does_not_reload_dataset(monkeypatch):
    from kineticeval import inspect_adapter
    definition=asyncio.run(import_task('kineticeval.fixtures:inspect_fixture'))
    def fail(config): raise AssertionError('Task factory must not be called while rescoring')
    monkeypatch.setattr(inspect_adapter,'task_from',fail)
    payload={'component':definition.scorer.model_dump(),'sample':definition.examples[0].model_dump(),
        'messages':definition.examples[0].messages,'model':{'name':'mockllm/model'},'response':'4'}
    result=asyncio.run(score_response(payload))
    assert result['value']==1
    assert metrics(definition.metrics[0].model_dump(),[result])['inspect_ai/mean']==1


def test_judge_repairs_invalid_evidence_with_bounded_feedback(monkeypatch):
    import json
    from kineticeval import providers
    calls = []
    async def response(self, model, messages, parameters):
        calls.append(list(messages))
        evidence = ['invented quote'] if len(calls) == 1 else ['A']
        return {'response': json.dumps({'semantic_correct': True, 'evidence': evidence}), 'metadata': {'id':str(len(calls))}}
    monkeypatch.setenv('UCLOUD_BASE_URL', 'https://fixture.invalid/v1')
    monkeypatch.setenv('UCLOUD_API_KEY', 'fixture-key')
    monkeypatch.setattr(providers.UCloud, 'chat', response)
    result = asyncio.run(providers.judge({'config':{'provider':'ucloud','model':'model','max_tokens':1000},
        'evidence':{'response':'A'},'dimensions':['semantic_correct']}))
    assert result['evidence'] == ['A']
    assert len(calls) == 2
    assert 'failed validation' in calls[1][-1]['content']
    assert len(result['_provider']['attempts']) == 2
