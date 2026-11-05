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
