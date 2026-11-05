import asyncio
from kineticeval.db import Database,Revision
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition
from kineticeval.worker import Worker


def test_run_analysis_is_overridable_and_frozen(tmp_path):
    service=Service(Database(tmp_path/'settings.db'))
    revision=service.create_evaluation(demo_definition())['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake'}],
        'analysis_overrides':{'scorer':{'name':'normalized','version':'1','config':{}}}})['id']
    with service.db.session() as s:
        row=s.get(Revision,revision)
        row.definition={**row.definition,'scorer':{'name':'exact','version':'1','config':{}}}
    asyncio.run(Worker(service.db).drain())
    result=service.results(run)
    assert result['definition']['scorer']['name']=='normalized'
    assert result['run']['status']=='completed'


def test_suggestion_api_cannot_change_inputs_and_run_overrides_win(tmp_path):
    from fastapi.testclient import TestClient
    from kineticeval.api import create_app
    client=TestClient(create_app(tmp_path/'suggestions.db'))
    revision=client.post('/api/evaluations',json=demo_definition()).json()['revision_id']
    path=f'/api/revisions/{revision}'
    assert client.post(path+'/suggestions',json={'prompt_template':'changed'}).status_code==422
    saved=client.post(path+'/suggestions',json={'scorer':{'name':'normalized'},'generation_defaults':{'max_tokens':128}}).json()
    assert saved['revision_id']==revision and saved['created'] is False
    run=client.post('/api/runs',json={'revision_id':revision,'models':[{'name':'fake','backend':'fake'}],'analysis_overrides':{'scorer':{'name':'exact'}}}).json()
    assert client.get('/api/runs/'+run['id']).json()['definition']['scorer']['name']=='exact'
    assert client.get(path+'/suggestions').json()['scorer']['name']=='normalized'
    assert 'scorer' not in client.get(path+'/inputs').json()


def test_generation_default_precedence_and_snapshot(tmp_path,monkeypatch):
    from kineticeval import local_hf
    monkeypatch.setattr(local_hf,'pin_model',lambda model:model)
    service=Service(Database(tmp_path/'precedence.db'))
    definition=demo_definition();definition['source']={'kind':'inline','generation_config':{'max_tokens':32,'temperature':.1}}
    definition['generation_defaults']={'max_tokens':64,'temperature':.2}
    revision=service.create_evaluation(definition)['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'local','backend':'hf','parameters':{'max_tokens':128}}],
        'analysis_overrides':{'generation_defaults':{'temperature':.3}}})['id']
    model=service.results(run)['run']['config']['models'][0]
    assert model['parameters']=={'max_tokens':128,'temperature':.3}


def test_single_run_is_grouped_in_a_result(tmp_path):
    from kineticeval.library import results
    service=Service(Database(tmp_path/'single-result.db'))
    revision=service.create_evaluation(demo_definition())['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake'}]})
    collection=next(r for r in results(service.db) if r['id']==run['result_id'])
    assert collection['run_ids']==[run['id']]
