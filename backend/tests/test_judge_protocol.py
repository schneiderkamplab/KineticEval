import asyncio
import copy
import json
import pytest
from kineticeval import judge_protocol, providers
from kineticeval.db import Database, Job
from kineticeval.service import Service, job_for
from kineticeval.fixtures import demo_definition


def setup(tmp_path):
    service=Service(Database(tmp_path/'protocol.db'))
    rev=service.create_evaluation(demo_definition())['revision_id']
    runs=[service.create_run({'revision_id':rev,'models':[{'name':'fixture','backend':'fake'}]})['id'] for _ in range(4)]
    return service,runs


def payload():
    return {'config':{'provider':'ucloud','model':'judge','max_tokens':1000},
        'evidence':{'response':'A'},'dimensions':['semantic_correct']}


@pytest.mark.parametrize('kind',['judge','report','investigation'])
def test_protocol_reuses_identical_work_but_changed_prompt_and_schema_invalidate(tmp_path,monkeypatch,kind):
    service,runs=setup(tmp_path)
    with service.db.session() as s:
        first=job_for(s,runs[0],kind,payload())
        original=copy.deepcopy(first.payload['protocol'])
        same=job_for(s,runs[1],kind,payload())
        assert first.id==same.id
    monkeypatch.setattr(judge_protocol,'EXTRA_INSTRUCTIONS',' Different instructions.')
    with service.db.session() as s:
        changed=job_for(s,runs[2],kind,payload())
        assert changed.id!=first.id
        assert changed.payload['protocol']['sha256']!=original['sha256']
        # A resume is bound to its original protocol, rather than silently retargeted.
        resumed=job_for(s,runs[0],kind,payload())
        assert resumed.id==first.id
    schema_type=judge_protocol.JudgeResult if kind=='judge' else judge_protocol.ReportResult
    schema=copy.deepcopy(schema_type.model_json_schema());schema['description']='changed schema'
    monkeypatch.setattr(schema_type,'model_json_schema',classmethod(lambda cls: schema))
    with service.db.session() as s:
        changed_schema=job_for(s,runs[3],kind,payload())
        assert changed_schema.id!=changed.id


def test_worker_sends_pinned_prompt_and_records_protocol(monkeypatch):
    p=payload();p['protocol']=judge_protocol.protocol_for(p)
    expected=p['protocol']['instruction']
    monkeypatch.setattr(judge_protocol,'EXTRA_INSTRUCTIONS','newly deployed prompt')
    monkeypatch.setenv('UCLOUD_BASE_URL','https://fixture.invalid/v1')
    monkeypatch.setenv('UCLOUD_API_KEY','fixture-key')
    async def chat(self,model,messages,parameters):
        assert messages[0]['content']==expected
        return {'response':json.dumps({'semantic_correct':True,'evidence':['A']}),'metadata':{}}
    monkeypatch.setattr(providers.UCloud,'chat',chat)
    result=asyncio.run(providers.judge(p))
    assert result['_protocol']['sha256']==p['protocol']['sha256']
    p['protocol']['instruction']='tampered'
    with pytest.raises(ValueError,match='fingerprint'):
        asyncio.run(providers.judge(p))


def test_legacy_jobs_never_satisfy_new_run_cache(tmp_path,monkeypatch):
    service,runs=setup(tmp_path)
    from kineticeval.core import identity
    from kineticeval.db import RunJob
    p=payload();old_id=identity({'kind':'judge','payload':p,'engine_version':'1'})
    with service.db.session() as s:
        s.add(Job(id=old_id,kind='judge',payload=p,status='succeeded',result={'semantic_correct':True}))
        s.add(RunJob(run_id=runs[0],job_id=old_id))
    with service.db.session() as s:
        assert job_for(s,runs[0],'judge',p).id==old_id
        assert job_for(s,runs[1],'judge',p).id!=old_id
    before=judge_protocol.checked_protocol(p)
    monkeypatch.setattr(judge_protocol,'EXTRA_INSTRUCTIONS','new prompt')
    after=judge_protocol.checked_protocol(p)
    assert before==after
    assert before['version']=='legacy-unversioned'
    monkeypatch.setattr(judge_protocol,'original_instruction',lambda *args:'entirely new instructions')
    assert judge_protocol.checked_protocol(p)==before
