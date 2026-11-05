import asyncio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from kineticeval.db import Database, Proposal, Run, Job, Experiment
from kineticeval.service import Service
from kineticeval.worker import Worker
from kineticeval.fixtures import demo_definition
from kineticeval import experiments
from kineticeval.api import create_app


def setup(tmp_path):
    service=Service(Database(tmp_path/'experiments.db'))
    revision=service.create_evaluation(demo_definition())['revision_id']
    baseline=service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake','parameters':{'max_tokens':64}}]})['id']
    asyncio.run(Worker(service.db).drain())
    with service.db.session() as s:
        s.add(Proposal(id='proposal',run_id=baseline,proposal={'kind':'experiment','title':'Token budget test',
            'rationale':'Fixture recommendation','origin_job':'fixture-inspector',
            'patch':{'proposed_settings':{'model':'fixture','arms':[{'name':'baseline','max_tokens':64},
                {'name':'larger','max_tokens':128}]}},'example_ids':['capital']}))
    return service,baseline


def counts(service):
    with service.db.session() as s:
        return tuple(s.scalar(select(func.count()).select_from(table)) for table in [Run,Job])


def test_preview_no_execution_approval_atomic_idempotent_and_paired_results(tmp_path):
    service,baseline=setup(tmp_path)
    seeded=experiments.seed(service,'proposal')
    assert len(seeded['plan']['example_ids'])==3  # citation IDs do not silently choose experiment scope
    before=counts(service)
    draft=experiments.prepare(service,'proposal',seeded['plan'])
    assert counts(service)==before
    assert draft['status']=='prepared'
    assert draft['preview']['arms'][0]['reusable']==3
    assert draft['preview']['arms'][1]['pending_or_new']==3
    approved=experiments.approve(service,draft['id'])
    after=counts(service)
    assert experiments.approve(service,draft['id'])['runs']==approved['runs']
    assert counts(service)==after
    assert after[0]==before[0]+2
    with service.db.session() as s:
        assert all(s.get(Run,r).parent_id==baseline for r in approved['runs'].values())
        assert s.get(Run,baseline).config['models'][0]['parameters']['max_tokens']==64
    asyncio.run(Worker(service.db).drain())
    compared=experiments.results(service,draft['id'])
    assert all(r['paired_scored']==3 and r['paired_delta']==0 for r in compared['comparisons'])
    assert compared['preview']['origin_job']=='fixture-inspector'


def test_partial_scope_prompt_revision_and_invalid_plans(tmp_path):
    service,baseline=setup(tmp_path)
    plan=experiments.seed(service,'proposal')['plan']
    plan['example_ids']=['capital']
    plan['arms']=[{'name':'prompt','prompt_template':'{input}\n{choices}\nOnly a letter.'}]
    draft=experiments.prepare(service,'proposal',plan)
    assert draft['preview']['arms'][0]['prompt_changed']
    approved=experiments.approve(service,draft['id'])
    with service.db.session() as s:
        child=s.get(Run,approved['runs']['prompt'])
        assert child.revision_id!=s.get(Run,baseline).revision_id
        assert child.manifest['ids']==['capital']
    for bad in [{'name':'bad','parameters':{'max_tokens':0}}, {'name':'bad','transform':'rotate'},
                {'name':'bad','chat_template_kwargs':{'enable_thinking':'false'}}]:
        with pytest.raises(ValueError):experiments.Plan.model_validate({**plan,'arms':[bad]})


def test_approval_rolls_back_all_arms_and_rejected_proposal_cannot_launch(tmp_path,monkeypatch):
    service,baseline=setup(tmp_path)
    draft=experiments.prepare(service,'proposal',experiments.seed(service,'proposal')['plan'])
    before=counts(service)
    original=service._create_run
    called=0
    def fail(s,req):
        nonlocal called
        called+=1
        if called==2:raise ValueError('Injected transaction failure')
        return original(s,req)
    monkeypatch.setattr(service,'_create_run',fail)
    with pytest.raises(ValueError):experiments.approve(service,draft['id'])
    assert counts(service)==before
    with service.db.session() as s:
        assert s.get(Experiment,draft['id']).status=='prepared'
        s.get(Proposal,'proposal').status='rejected'
    with pytest.raises(ValueError,match='reviewed'):experiments.approve(service,draft['id'])


def test_routes_prepare_without_approving(tmp_path):
    service,baseline=setup(tmp_path)
    client=TestClient(create_app(str(tmp_path/'experiments.db')))
    seeds=client.get('/api/proposals/proposal/experiment').json()
    response=client.post('/api/proposals/proposal/prepare-experiment',json=seeds['plan'])
    assert response.status_code==200,response.text
    draft=response.json()
    assert client.get('/api/experiment-proposals').status_code==200
    assert client.get('/api/experiments/'+draft['id']).json()['status']=='prepared'
    assert counts(service)[0]==1
    assert client.post('/api/experiments/'+draft['id']+'/approve').status_code==200


def test_local_worker_handles_parameter_arms_with_same_weights(tmp_path,monkeypatch):
    from kineticeval import local_hf,local_worker
    monkeypatch.setattr(local_hf,'pin_model',lambda model:model)
    observed=[]
    class Engine:
        def __init__(self,spec,*args):self.spec=spec
        def generate_batch(self,payloads):
            observed.append(self.spec['parameters']['max_tokens'])
            return [{'response':'A','metadata':{'output_tokens':1}} for _ in payloads]
    monkeypatch.setattr(local_worker,'LocalModel',Engine)
    service=Service(Database(tmp_path/'local.db'))
    revision=service.create_evaluation(demo_definition())['revision_id']
    for budget in (64,128):
        service.create_run({'revision_id':revision,'example_ids':['capital'],'models':[{
            'name':'same-weights','backend':'hf','revision':'pinned','parameters':{'max_tokens':budget}}]})
    asyncio.run(local_worker.run(service.db,'same-weights'))
    assert observed==[64,128]
    with service.db.session() as s:
        assert s.scalar(select(func.count()).select_from(Job).where(Job.kind=='generation',Job.status=='failed'))==0


@pytest.mark.parametrize('patch',[
    {'steps':[{'name':'shuffle','settings':{'choice_order':'random'}}]},
    {'proposed_settings':{'model':'fixture'},'transform':'rotate'},
    {'proposed_settings':{'model':'fixture','arms':[{'name':'shuffle','choice_order':'random'}]}},
    {'proposed_settings':{'model':'fixture','choice_order':'random'}},
    {'proposed_settings':{'model':'fixture'}},
])
def test_unsupported_or_unchanged_proposals_cannot_be_prepared(tmp_path,patch):
    service,_=setup(tmp_path)
    with service.db.session() as s:
        p=s.get(Proposal,'proposal');p.proposal={**p.proposal,'patch':patch}
    assert experiments.seed(service,'proposal')['blocked'] is True
    before=counts(service)
    # A crafted direct API request with a valid parameter change cannot hide unsupported operations.
    with pytest.raises(ValueError):
        experiments.prepare(service,'proposal',{'model':'fixture','example_ids':['capital'],
            'arms':[{'name':'unchanged'}]})
    assert counts(service)==before


def test_approval_revalidates_legacy_preview(tmp_path):
    service,_=setup(tmp_path)
    draft=experiments.prepare(service,'proposal',experiments.seed(service,'proposal')['plan'])
    with service.db.session() as s:
        p=s.get(Proposal,'proposal');p.proposal={**p.proposal,'patch':{'steps':[{'name':'shuffle'}]}}
    before=counts(service)
    with pytest.raises(ValueError,match='Unsupported experiment operations'):
        experiments.approve(service,draft['id'])
    assert counts(service)==before


def test_noop_with_explicit_identical_values_is_rejected_but_cached_changes_are_valid(tmp_path):
    service,_=setup(tmp_path)
    plan=experiments.seed(service,'proposal')['plan']
    plan['arms']=[{'name':'renamed','parameters':{'max_tokens':64,'temperature':0,'top_p':1},'chat_template_kwargs':{'enable_thinking':False}}]
    with pytest.raises(ValueError,match='no effective change'):
        experiments.prepare(service,'proposal',plan)
