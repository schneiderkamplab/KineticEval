from kineticeval.db import Database, Job, JobActivity
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition
from kineticeval.progress import snapshot


def test_progress_counts_reuse_and_fences_stale_activity(tmp_path):
    db = Database(tmp_path / 'progress.db')
    service = Service(db)
    revision = service.create_evaluation(demo_definition())['revision_id']
    request = {'revision_id': revision, 'models': [{'name':'test', 'backend':'fake'}]}
    run = service.create_run(request)['id']
    initial = snapshot(db, run)
    assert initial['stages'][0]['counts'] == {'pending': 3}
    assert initial['stages'][1]['counts'] == {'waiting': 3}
    claim = db.claim_many(kind='generation', run_id=run)[0]
    db.activity([claim], 'preparing_model')
    assert snapshot(db, run)['activities'][0]['phase'] == 'preparing_model'
    db.activity([{**claim, 'token':'stale'}], 'wrong')
    assert snapshot(db, run)['activities'][0]['phase'] == 'preparing_model'
    with db.session() as s:
        s.get(Job, claim['id']).lease = 0
    assert snapshot(db, run)['activities'][0]['stale'] is True
    new_claim = db.claim_many(kind='generation', run_id=run)[0]
    assert snapshot(db, run)['activities'][0]['phase'] == 'generation'
    db.finish(new_claim['id'], new_claim['token'], result={'response':'A'})
    progress = snapshot(db, run)
    assert progress['stages'][0]['counts']['succeeded'] == 1
    assert progress['activities'] == []
    second = service.create_run(request)['id']
    assert snapshot(db, second)['reused'] == 1
    failed = db.claim_many(kind='generation', run_id=run)[0]
    db.finish(failed['id'], failed['token'], error='test failure')
    assert snapshot(db, run)['stages'][0]['counts']['failed'] == 1
    assert snapshot(db, run)['models'][0]['total'] == 3
