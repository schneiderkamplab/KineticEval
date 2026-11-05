from fastapi.testclient import TestClient
from kineticeval.api import create_app


def test_named_variants_and_result_lifecycle(tmp_path):
    client=TestClient(create_app(tmp_path/'library.db'))
    for name in ['Thinking','Direct']:
        r=client.post('/api/models',json={'name':name,'spec':{'name':'org/model','backend':'hf'}})
        assert r.status_code==200
    assert len(client.get('/api/models').json())==2
    result=client.post('/api/results',json={'name':'Comparison','run_ids':[]}).json()
    assert client.get('/api/results').json()[0]['id']==result['id']
    assert client.delete('/api/results/'+result['id']).status_code==200
    assert client.get('/api/results').json()==[]
    assert client.post('/api/results',json={'name':'Invalid','run_ids':['missing']}).status_code==404


def test_launch_atomic_idempotent_and_retains_runs_on_collection_delete(tmp_path):
    from kineticeval.fixtures import demo_definition
    client=TestClient(create_app(tmp_path/'launch.db'))
    revisions=[client.post('/api/evaluations',json=demo_definition()).json()['revision_id'] for _ in range(3)]
    body={'name':'Three evaluations','launch_key':'launch-1','runs':[{'revision_id':r,'models':[{'name':f'fake-{i}','backend':'fake'} for i in range(3)]} for r in revisions]}
    result=client.post('/api/launches',json=body)
    assert result.status_code==200
    assert len(result.json()['run_ids'])==3
    assert client.post('/api/launches',json=body).json()==result.json()
    assert len(client.get('/api/runs').json())==3
    assert client.post('/api/launches',json={**body,'name':'Changed'}).status_code==422
    bad={**body,'launch_key':'bad','runs':[body['runs'][0],{'revision_id':'missing','models':[{'name':'fake','backend':'fake'}]}]}
    assert client.post('/api/launches',json=bad).status_code==404
    assert len(client.get('/api/runs').json())==3
    client.delete('/api/results/'+result.json()['id'])
    assert len(client.get('/api/runs').json())==3


def test_named_variants_share_identical_generations_but_keep_rows(tmp_path):
    from kineticeval.fixtures import demo_definition
    client=TestClient(create_app(tmp_path/'variants.db'))
    revision=client.post('/api/evaluations',json=demo_definition()).json()['revision_id']
    run=client.post('/api/runs',json={'revision_id':revision,'models':[
        {'name':label,'model_identifier':'physical-model','backend':'fake'} for label in ['variant one','variant two']]}).json()
    data=client.get('/api/runs/'+run['id']).json()
    assert len(data['items'])==6
    assert len({i['generation']['id'] for i in data['items']})==3
    assert {i['model']['name'] for i in data['items']}=={'variant one','variant two'}


def test_editing_collection_keeps_launch_idempotency(tmp_path):
    from kineticeval.fixtures import demo_definition
    client=TestClient(create_app(tmp_path/'edit-idempotency.db'))
    revision=client.post('/api/evaluations',json=demo_definition()).json()['revision_id']
    body={'name':'Original','launch_key':'key','runs':[{'revision_id':revision,'models':[{'name':'fake','backend':'fake'}]}]}
    result=client.post('/api/launches',json=body).json()
    client.post('/api/results/'+result['id'],json={'name':'Edited','run_ids':result['run_ids'],'comparison':{'allow_mismatch':True}})
    assert client.post('/api/launches',json=body).json()['id']==result['id']
    assert len(client.get('/api/runs').json())==1


def test_result_retry_resumes_all_runs_sharing_failures_and_preserves_success(tmp_path):
    from sqlalchemy import select
    from kineticeval.db import Database, Job, Run
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    from kineticeval.library import launch, retry_result
    db=Database(tmp_path/'retry.db');service=Service(db)
    revision=service.create_evaluation(demo_definition())['revision_id']
    request={'revision_id':revision,'models':[{'name':'fake','backend':'fake'}]}
    result=launch(service,{'name':'Shared','launch_key':'shared','runs':[request,request]})
    with db.session() as s:
        jobs=list(s.scalars(select(Job)))
        jobs[0].status='failed';jobs[0].error='temporary failure'
        failed_id=jobs[0].id
        jobs[1].status='succeeded';jobs[1].result={'response':'preserved'}
        success_id=jobs[1].id
        for rid in result['run_ids']:
            r=s.get(Run,rid);r.status='completed_with_errors'
            r.audit={'failure_review':{'selected':['stable']}}
    response=retry_result(service,result['id'])
    assert response['runs_retried']==2 and response['jobs_retried']==1
    with db.session() as s:
        assert s.get(Job,failed_id).status=='pending'
        assert s.get(Job,success_id).result=={'response':'preserved'}
        for rid in result['run_ids']:
            r=s.get(Run,rid)
            assert r.status=='active'
            assert r.audit=={'failure_review':{'selected':['stable']}}
    assert retry_result(service,result['id'])['jobs_retried']==0
    client=TestClient(create_app(tmp_path/'retry.db'))
    assert client.post('/api/results/'+result['id']+'/retry').json()['jobs_retried']==0
    assert client.post('/api/results/missing/retry').status_code==404
