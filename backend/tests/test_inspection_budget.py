import asyncio
from sqlalchemy import select,func
from kineticeval.db import Database,Item,Job,RunJob,Run
from kineticeval.service import Service
from kineticeval.worker import Worker
from kineticeval.fixtures import demo_definition


def setup(tmp_path,limit=None):
    service=Service(Database(tmp_path/'budget.db'))
    definition=demo_definition()
    definition['inspector']={'enabled':True,'provider':'fake','audit_fraction':1,'audit_min':0,'audit_max':100,'failure_inspector_limit':limit}
    revision=service.create_evaluation(definition)['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'bad','backend':'fake','parameters':{'response':'???'}}]})['id']
    return service,run


async def prepare(service,run):
    worker=Worker(service.db)
    for _ in range(30):
        service.schedule()
        claims=service.db.claim_many(kinds=['generation','scoring','diagnostic','metrics'],run_id=run)
        if not claims:break
        await worker.execute(claims[0])
    service.schedule()


def test_cap_includes_failure_audits_and_unresolved_cases(tmp_path):
    service,run=setup(tmp_path,1)
    asyncio.run(Worker(service.db).drain())
    result=service.results(run)
    assert sum(item['inspector'] is not None for item in result['items'])==1
    assert result['run']['audit']['strata'][0]['sample_size']==1
    assert result['run']['audit']['strata'][0]['probability']==1/3
    assert result['run']['status']=='completed'


def test_runtime_cap_preserves_finished_assessments_and_unlinks_pending(tmp_path):
    service,run=setup(tmp_path)
    asyncio.run(prepare(service,run))
    claim=service.db.claim_many(kind='inspector',run_id=run)[0]
    asyncio.run(Worker(service.db).execute(claim))
    before=[item['generation']['id'] for item in service.results(run)['items']]
    service.control(run,'cancel')
    applied=service.set_failure_inspector_limit(run,1,resume=True)
    assert applied['failure_review']['groups'][0]['retained_started']==1
    assert service.db.claim_many(kind='inspector',run_id=run)==[]
    asyncio.run(Worker(service.db).drain())
    result=service.results(run)
    assert [item['generation']['id'] for item in result['items']]==before
    assert [item['inspector']['id'] for item in result['items'] if item['inspector']]==[claim['id']]
    assert result['run']['audit']['strata'][0]['probability'] is None
    assert result['definition']['inspector']['failure_inspector_limit'] is None


def test_limit_below_started_work_never_discards_evidence(tmp_path):
    service,run=setup(tmp_path)
    asyncio.run(prepare(service,run))
    claims=service.db.claim_many(kind='inspector',run_id=run,limit=2)
    for claim in claims:asyncio.run(Worker(service.db).execute(claim))
    applied=service.set_failure_inspector_limit(run,0,resume=True)
    assert applied['failure_review']['groups'][0]['over_limit'] is True
    assert applied['failure_review']['groups'][0]['selected']==2
    assert service.db.claim_many(kind='inspector',run_id=run)==[]


def test_retry_keeps_capped_cohort_and_reuses_completed_assessments(tmp_path):
    service,run=setup(tmp_path)
    asyncio.run(prepare(service,run))
    claim=service.db.claim_many(kind='inspector',run_id=run)[0]
    asyncio.run(Worker(service.db).execute(claim))
    applied=service.set_failure_inspector_limit(run,2,resume=True)
    selected=set(applied['failure_review']['selected'])
    service.control(run,'retry')
    service.schedule()
    result=service.results(run)
    assert set(result['run']['audit']['failure_review']['selected'])==selected
    assert sum(item['inspector'] is not None for item in result['items'])==2
    asyncio.run(Worker(service.db).drain())
    result=service.results(run)
    assert sum(item['inspector'] is not None for item in result['items'])==2
    assert claim['id'] in [item['inspector']['id'] for item in result['items'] if item['inspector']]
