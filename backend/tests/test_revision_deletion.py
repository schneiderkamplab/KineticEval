import asyncio
import pytest
from sqlalchemy import select
from kineticeval.db import Database,Run,Job,Revision,Item,Attempt,Evaluation
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition
from kineticeval.worker import Worker
from kineticeval.revision_deletion import preview,remove


def test_delete_preserves_shared_cache_and_surviving_revision(tmp_path):
    service=Service(Database(tmp_path/'delete.db'))
    original=service.create_evaluation(demo_definition())
    first=original['revision_id']; eid=original['id']
    second=service.revise(first,{'sampler':{'seed':43}})['revision_id']
    a=service.create_run({'revision_id':first,'models':[{'name':'fake','backend':'fake'}]})['id']
    asyncio.run(Worker(service.db).drain())
    b=service.create_run({'revision_id':second,'models':[{'name':'fake','backend':'fake'}]})['id']
    impact=preview(service.db,eid,[first])
    assert impact['counts']['shared_jobs']==3
    assert impact['counts']['runs']==1
    remove(service.db,eid,[first],impact['token'])
    with service.db.session() as s:
        assert s.get(Run,a) is None
        assert s.get(Run,b) is not None
        assert s.get(Revision,second).parent_id is None
        for item in s.scalars(select(Item).where(Item.run_id==b)):
            assert s.get(Job,item.generation_id).status=='succeeded'
    asyncio.run(Worker(service.db).drain())
    impact=preview(service.db,eid,[second])
    assert remove(service.db,eid,[second],impact['token'])['evaluation_deleted']
    with service.db.session() as s:
        assert s.get(Evaluation,eid) is None
        assert not s.scalars(select(Job)).all()
        assert not s.scalars(select(Attempt)).all()


def test_stale_preview_and_foreign_revision_rejected(tmp_path):
    service=Service(Database(tmp_path/'delete.db'))
    a=service.create_evaluation(demo_definition());b=service.create_evaluation(demo_definition())
    with pytest.raises(ValueError):preview(service.db,a['id'],[b['revision_id']])
    impact=preview(service.db,a['id'],[a['revision_id']])
    service.create_run({'revision_id':a['revision_id'],'models':[{'name':'fake','backend':'fake'}]})
    with pytest.raises(ValueError,match='changed'):remove(service.db,a['id'],[a['revision_id']],impact['token'])


def test_active_claim_is_fenced_after_deletion(tmp_path):
    service=Service(Database(tmp_path/'active.db'))
    a=service.create_evaluation(demo_definition())
    service.create_run({'revision_id':a['revision_id'],'models':[{'name':'fake','backend':'fake'}]})
    claim=service.db.claim()
    impact=preview(service.db,a['id'],[a['revision_id']])
    assert impact['counts']['active_runs']==1
    remove(service.db,a['id'],[a['revision_id']],impact['token'])
    with pytest.raises(ValueError,match='Stale'):service.db.finish(claim['id'],claim['token'],result={})
    assert service.db.claim() is None


def test_api_requires_preview_token(tmp_path):
    from fastapi.testclient import TestClient
    from kineticeval.api import create_app
    client=TestClient(create_app(tmp_path/'api.db'))
    a=client.post('/api/evaluations',json=demo_definition()).json()
    path=f"/api/evaluations/{a['id']}"
    body={'revision_ids':[a['revision_id']]}
    assert client.post(path+'/delete-revisions',json=body).status_code==422
    impact=client.post(path+'/deletion-preview',json=body).json()
    assert client.post(path+'/delete-revisions',json={**body,'token':impact['token']}).status_code==200


def test_multi_revision_preview_deduplicates_and_removes_experiment_records(tmp_path):
    from kineticeval.db import Proposal,Experiment
    service=Service(Database(tmp_path/'multi.db'))
    a=service.create_evaluation(demo_definition())
    second=service.revise(a['revision_id'],{'sampler':{'seed':43}})['revision_id']
    one=service.create_run({'revision_id':a['revision_id'],'models':[{'name':'fake','backend':'fake'}]})['id']
    two=service.create_run({'revision_id':second,'models':[{'name':'fake','backend':'fake'}]})['id']
    with service.db.session() as s:
        s.add(Proposal(id='p',run_id=one,proposal={},revision_id=second))
        s.add(Experiment(id='e',proposal_id='p',baseline_id=one,plan={},preview={},runs={'arm':two}))
    impact=preview(service.db,a['id'],[a['revision_id'],second])
    assert impact['counts']['jobs']==3
    assert impact['counts']['shared_jobs']==0
    assert impact['counts']['experiments']==1
    remove(service.db,a['id'],[a['revision_id'],second],impact['token'])
    with service.db.session() as s:
        assert s.get(Experiment,'e') is None
        assert s.get(Proposal,'p') is None


def test_deleting_revision_removes_memberships_not_collection(tmp_path):
    from kineticeval.library import save_result,results
    service=Service(Database(tmp_path/'membership.db'))
    e=service.create_evaluation(demo_definition())
    run=service.create_run({'revision_id':e['revision_id'],'models':[{'name':'fake','backend':'fake'}]})['id']
    collection=save_result(service.db,{'name':'Keep collection','run_ids':[run]})
    impact=preview(service.db,e['id'],[e['revision_id']])
    assert impact['counts']['result_memberships']==2
    remove(service.db,e['id'],[e['revision_id']],impact['token'])
    assert results(service.db)[0]['id']==collection['id']
    assert results(service.db)[0]['run_ids']==[]
