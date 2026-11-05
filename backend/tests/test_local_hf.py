from pathlib import Path
from kineticeval.local_hf import local_digest
from kineticeval.schemas import ModelSpec
from kineticeval.components import diagnose


def test_local_weight_identity_changes_with_content(tmp_path):
    weights=tmp_path/'model.safetensors'
    weights.write_bytes(b'first')
    (tmp_path/'config.json').write_text('{}')
    first=local_digest(tmp_path)
    weights.write_bytes(b'second')
    assert local_digest(tmp_path)!=first


def test_hf_model_contract():
    spec=ModelSpec(name='org/model',backend='hf',revision='commit',parameters={'max_tokens':64})
    assert spec.chat_template_kwargs=={'enable_thinking':False}
    assert spec.dtype=='bfloat16'


def test_unambiguous_mcq_failure_can_be_fully_covered():
    payload={'response':'ANSWER: D','sample':{'target':'A','choices':['a','b','c','d']},'score':{'value':0},
        'messages':[{'role':'user','content':"The entire content of your response should be of the following format: 'ANSWER: $LETTER'."}],
        'expected_format':'answer_letter','components':[{'name':'qa','version':'2','config':{}}]}
    result=diagnose(payload)
    assert result['finding']['semantic_correct'] is False
    assert result['finding']['scorer_correct'] is True
    assert result['unresolved']==[]


def test_model_grouped_claims_are_disjoint(tmp_path):
    from kineticeval.db import Database
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    db=Database(tmp_path/'queue.db');service=Service(db)
    revision=service.create_evaluation(demo_definition())['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'one','backend':'fake'},{'name':'two','backend':'fake'}]})['id']
    first=db.claim_many(kind='generation',model='one',limit=2,run_id=run)
    second=db.claim_many(kind='generation',model='one',limit=2,run_id=run)
    assert len(first)==2 and len(second)==1
    assert not {c['id'] for c in first}&{c['id'] for c in second}
    assert all(c['payload']['model']['name']=='one' for c in first+second)
    assert len(db.claim_many(kind='generation',model='two',limit=10))==3


def test_candidate_final_marker_rejects_conflicts_and_negation():
    from kineticeval.qa_candidates import qa_with_final_marker
    base={'sample':{'target':'A','choices':['first','second','third','fourth']},'messages':[],
          'score':{'value':0},'expected_format':'none'}
    for response,expected in [('Reasoning about B.\nANSWER: A',True),('ANSWER: A\nANSWER: B',None),
            ('The answer is not A.',None),('Not ANSWER: A',None),('ANSWER: A\nActually not.',None),
            ('Reasoning.\nANSWER: D',False)]:
        assert qa_with_final_marker({**base,'response':response},{})['semantic_correct'] is expected


def test_batch_completion_rejects_stale_claim_atomically(tmp_path):
    import pytest
    from kineticeval.db import Database, Job
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    db = Database(tmp_path/'batch.db')
    service = Service(db)
    rev = service.create_evaluation(demo_definition())['revision_id']
    service.create_run({'revision_id':rev,'models':[{'name':'one','backend':'fake'}]})
    claims = db.claim_many(kind='generation', limit=2)
    completions = [{'id':c['id'],'token':c['token'],'result':{'response':'A'}} for c in claims]
    completions[1]['token'] = 'stale'
    with pytest.raises(ValueError, match='Stale'):
        db.finish_many(completions)
    with db.session() as session:
        assert all(session.get(Job,c['id']).status == 'running' for c in claims)
    completions[1]['token'] = claims[1]['token']
    db.finish_many(completions)
    with db.session() as session:
        assert all(session.get(Job,c['id']).status == 'succeeded' for c in claims)


def test_auto_worker_drains_resident_weights_before_switching(tmp_path, monkeypatch):
    import asyncio
    import sys
    import weakref
    from types import SimpleNamespace
    from kineticeval import local_hf, local_worker
    from kineticeval.db import Database, Job
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    from sqlalchemy import select
    monkeypatch.setattr(local_hf, 'pin_model', lambda model: model)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    loaded, generated, references = [], [], []
    class Engine:
        def __init__(self, spec, *args):
            assert all(ref() is None for ref in references), 'Previous weights must be released first'
            self.spec = spec
            loaded.append((spec['name'], spec['revision']))
            references.append(weakref.ref(self))
        def generate_batch(self, payloads):
            assert all(p['model'] == self.spec for p in payloads)
            generated.append((self.spec['name'], self.spec['parameters']['max_tokens']))
            return [{'response':'A','metadata':{'output_tokens':1}} for p in payloads]
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(chat_template=None, encode=lambda text, **kw:[1])
    Engine.tokenizer = tokenizer
    monkeypatch.setattr(local_worker, 'load_tokenizer', lambda spec: tokenizer)
    monkeypatch.setattr(local_worker, 'LocalModel', Engine)
    db = Database(tmp_path/'auto.db')
    service = Service(db)
    revision = service.create_evaluation(demo_definition())['revision_id']
    for name, weight, budget in [('a','v1',64), ('b','v1',64), ('a','v1',128), ('a','v2',64)]:
        service.create_run({'revision_id':revision,'models':[{'name':name,'backend':'hf',
            'revision':weight,'parameters':{'max_tokens':budget}}]})
    fake = service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake'}]})['id']
    asyncio.run(local_worker.run(db, batch_size=2))
    assert loaded == [('a','v1'), ('b','v1'), ('a','v2')]
    assert generated[:4] == [('a',64), ('a',64), ('a',128), ('a',128)]
    assert len(db.claim_many(kind='generation',run_id=fake,limit=10)) == 3
    with db.session() as s:
        assert len(s.scalars(select(Job).where(Job.kind=='generation',Job.status=='succeeded')).all()) == 12


def test_auto_claims_are_disjoint_and_ignore_cancelled_runs(tmp_path, monkeypatch):
    from kineticeval import local_hf
    from kineticeval.db import Database
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    monkeypatch.setattr(local_hf, 'pin_model', lambda model: model)
    db = Database(tmp_path/'claims.db')
    service = Service(db)
    revision = service.create_evaluation(demo_definition())['revision_id']
    run = service.create_run({'revision_id':revision,'models':[{'name':'hf','backend':'hf'}]})['id']
    first = db.claim_many(kind='generation',backend='hf',limit=2)
    second = db.claim_many(kind='generation',backend='hf',limit=2)
    assert len(first)==2 and len(second)==1
    assert not {c['id'] for c in first} & {c['id'] for c in second}

    cancelled = service.create_run({'revision_id':revision,'models':[{'name':'other','backend':'hf'}]})['id']
    service.control(cancelled, 'cancel')
    assert db.claim_many(kind='generation',backend='hf',limit=2) == []


def test_load_failure_does_not_repeat_download_or_block_other_models(tmp_path, monkeypatch):
    import asyncio
    from kineticeval import local_hf, local_worker
    from kineticeval.db import Database, Job
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    from sqlalchemy import select
    monkeypatch.setattr(local_hf, 'pin_model', lambda model: model)
    loads = []
    class Engine:
        def __init__(self, spec, *args):
            loads.append(spec['name'])
            if spec['name']=='bad':
                raise ValueError('Unsupported architecture')
            self.spec = spec
        def generate_batch(self, payloads):
            return [{'response':'A','metadata':{'output_tokens':1}} for p in payloads]
    from types import SimpleNamespace
    tokenizer = SimpleNamespace(chat_template=None, encode=lambda text, **kw:[1])
    Engine.tokenizer = tokenizer
    monkeypatch.setattr(local_worker, 'load_tokenizer', lambda spec: tokenizer)
    monkeypatch.setattr(local_worker, 'LocalModel', Engine)
    db = Database(tmp_path/'failure.db')
    service = Service(db)
    revision = service.create_evaluation(demo_definition())['revision_id']
    service.create_run({'revision_id':revision,'models':[{'name':name,'backend':'hf'} for name in ['bad','good']]})
    asyncio.run(local_worker.run(db, batch_size=1))
    assert loads == ['bad','good']
    with db.session() as s:
        assert len(s.scalars(select(Job).where(Job.status=='failed')).all()) == 3
        assert len(s.scalars(select(Job).where(Job.status=='succeeded')).all()) == 3


def test_analysis_worker_does_not_claim_generation(tmp_path):
    import asyncio
    from kineticeval.db import Database, Job
    from kineticeval.service import Service
    from kineticeval.worker import Worker
    from kineticeval.fixtures import demo_definition
    from sqlalchemy import select
    db = Database(tmp_path/'analysis.db')
    service = Service(db)
    revision = service.create_evaluation(demo_definition())['revision_id']
    service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake'}]})
    assert asyncio.run(Worker(db,analysis_only=True).step()) is False
    with db.session() as s:
        assert all(job.status=='pending' for job in s.scalars(select(Job)).all())


def test_base_model_gets_exact_single_user_prompt():
    import pytest
    from types import SimpleNamespace
    from kineticeval.local_hf import render_prompts
    tokenizer = SimpleNamespace(chat_template=None)
    assert render_prompts(tokenizer, [{'messages':[{'role':'user','content':'Question\nANSWER: '}]}], {}) == ['Question\nANSWER: ']
    with pytest.raises(ValueError, match='single user'):
        render_prompts(tokenizer, [{'messages':[{'role':'system','content':'instruction'},{'role':'user','content':'Q'}]}], {})


def test_cached_weights_still_complete_processor_metadata(tmp_path, monkeypatch):
    import huggingface_hub
    from kineticeval.local_hf import resolve_source
    (tmp_path / 'model.safetensors').touch()
    calls = []
    def snapshot(name, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    assert resolve_source({'name':'model', 'revision':'pinned'}) == str(tmp_path)
    assert calls[0]['local_files_only'] is True
    assert calls[1]['revision'] == 'pinned'
    assert '*.json' in calls[1]['allow_patterns']
    assert '*.safetensors' not in calls[1]['allow_patterns']


def test_prompt_preflight_isolates_invalid_requests_without_changing_valid_ones():
    from types import SimpleNamespace
    from kineticeval.local_hf import validate_prompt_batch
    tokenizer = SimpleNamespace(chat_template=None, encode=lambda text, **kwargs:list(text))
    good = {'id':'good', 'payload':{'messages':[{'role':'user','content':'Q'}]}}
    bad = {'id':'bad', 'payload':{'messages':[{'role':'system','content':'S'},{'role':'user','content':'Q'}]}}
    long = {'id':'long', 'payload':{'messages':[{'role':'user','content':'x'*100}]}}
    valid, errors = validate_prompt_batch(tokenizer, [bad, good, long], {'parameters':{'max_tokens':8},'context_limit':64})
    assert valid == [good]
    assert [claim['id'] for claim, _ in errors] == ['bad','long']
    assert 'single user' in str(errors[0][1])
    assert 'no truncation' in str(errors[1][1])
    assert good['payload']['messages'][0]['content'] == 'Q'


def test_worker_rejects_bad_prompt_without_poisoning_other_evaluations(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from sqlalchemy import select
    from kineticeval import local_hf, local_worker
    from kineticeval.db import Database, Job
    from kineticeval.service import Service
    from kineticeval.fixtures import demo_definition
    monkeypatch.setattr(local_hf, 'pin_model', lambda model:model)
    tokenizer = SimpleNamespace(chat_template=None, encode=lambda text, **kwargs:[1])
    monkeypatch.setattr(local_worker, 'load_tokenizer', lambda spec:tokenizer)
    loads = []
    class Engine:
        def __init__(self, spec, *args):
            self.spec, self.tokenizer = spec, tokenizer
            loads.append(spec['name'])
        def generate_batch(self, payloads):
            assert all(len(payload['messages']) == 1 for payload in payloads)
            return [{'response':'A','metadata':{'output_tokens':1}} for _ in payloads]
    monkeypatch.setattr(local_worker, 'LocalModel', Engine)
    db = Database(tmp_path/'mixed.db');service = Service(db)
    data = demo_definition()
    data['examples'][0]['messages'] = [{'role':'system','content':'Instruction'},{'role':'user','content':'Q'}]
    revision = service.create_evaluation(data)['revision_id']
    service.create_run({'revision_id':revision,'models':[{'name':'base','backend':'hf'}]})
    asyncio.run(local_worker.run(db, batch_size=3))
    assert loads == ['base']
    with db.session() as session:
        assert len(session.scalars(select(Job).where(Job.status=='failed')).all()) == 1
        assert len(session.scalars(select(Job).where(Job.status=='succeeded')).all()) == 2
