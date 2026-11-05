import asyncio
import copy
import pytest
from sqlalchemy import select
from fastapi.testclient import TestClient
from kineticeval.db import Database, Job, Run, Item, Proposal
from kineticeval.service import Service
from kineticeval.worker import Worker
from kineticeval.fixtures import demo_definition
from kineticeval.api import create_app


@pytest.fixture
def service(tmp_path):
    return Service(Database(tmp_path / 'test.db'))


def complete(service):
    asyncio.run(Worker(service.db).drain())


def setup_run(service, inspector=False):
    data = demo_definition()
    data['inspector'] = {'enabled': inspector, 'provider': 'fake'}
    rev = service.create_evaluation(data)['revision_id']
    run = service.create_run({'revision_id': rev, 'models': [
        {'name': 'literal', 'backend': 'fake'}, {'name': 'verbose', 'backend': 'fake', 'parameters': {'style': 'sentence'}}]})['id']
    return rev, run


def test_complete_revision_reuse_and_partial(service):
    rev, run = setup_run(service, True)
    complete(service)
    results = service.results(run)
    assert results['run']['status'] == 'completed'
    assert len(results['items']) == 6
    assert results['report']['status'] == 'succeeded'
    assert sum(x['sample_size'] for x in results['run']['audit']['strata']) == 6
    corrected = service.revise(rev, {'overlays': {'planet': {'target': 'A'}}})['revision_id']
    preview = service.preview(rev, {'overlays': {'planet': {'target': 'A'}}})
    assert preview['generation_changes'] == []
    assert preview['reference_changes'] == ['planet']
    rerun = service.create_run({'revision_id': corrected, 'models': results['run']['config']['models'],
        'parent_id': run, 'example_ids': ['planet']})['id']
    complete(service)
    revised = service.results(rerun)
    assert all(i['reused'] for i in revised['items'])
    assert revised['run']['manifest']['partial']
    assert not revised['summaries'][0]['full_evaluation']
    assert revised['items'][0]['score']['result']['value'] == 1
    assert next(i for i in results['items'] if i['example_id'] == 'planet')['score']['result']['value'] == 0
    assert service.results(run)['definition']['overlays'] == {}


def test_resume_cancel_and_fresh(service):
    rev, run = setup_run(service)
    service.control(run, 'cancel')
    assert service.db.claim() is None
    service.control(run, 'resume')
    asyncio.run(Worker(service.db).step())
    complete(service)
    first = service.results(run)
    repeat = service.create_run(first['run']['config'])['id']
    complete(service)
    assert all(i['reused'] for i in service.results(repeat)['items'])
    new = service.create_run({**first['run']['config'], 'fresh': True})['id']
    assert not any(i['reused'] for i in service.results(new)['items'])


def test_stage_failure_retry(service, monkeypatch):
    from kineticeval import components
    rev, run = setup_run(service)
    original = components.score
    async def failure(payload): raise ValueError('test scoring outage')
    monkeypatch.setattr(components, 'score', failure)
    complete(service)
    assert service.results(run)['run']['status'] == 'completed_with_errors'
    generations = [i['generation']['id'] for i in service.results(run)['items']]
    monkeypatch.setattr(components, 'score', original)
    service.control(run, 'retry')
    complete(service)
    assert service.results(run)['run']['status'] == 'completed'
    assert generations == [i['generation']['id'] for i in service.results(run)['items']]


def test_atomic_claim_and_recovery(service):
    _, run = setup_run(service)
    old = service.db.claim(-1)
    new = service.db.claim()
    assert old['id'] == new['id']
    with pytest.raises(ValueError): service.db.finish(old['id'], old['token'], result={})
    service.db.finish(new['id'], new['token'], result={'response': 'A'})
    assert len(service.attempts(new['id'])) == 2


def test_api(tmp_path):
    client = TestClient(create_app(tmp_path / 'api.db'))
    rev = client.post('/api/evaluations', json=demo_definition()).json()['revision_id']
    assert client.post(f'/api/revisions/{rev}/preview', json={}).status_code == 200
    assert client.post('/api/runs', json={'revision_id': rev, 'models': [{'name': 'demo', 'backend': 'fake'}]}).status_code == 200
    assert client.get('/api/pending').json()['demo']
    assert client.post('/api/evaluations', json=demo_definition(), headers={'Origin': 'https://evil.example'}).status_code == 403


def test_proposal_gate(service):
    from kineticeval.proposals import review
    rev, run = setup_run(service)
    with service.db.session() as s:
        s.add(Proposal(id='candidate', run_id=run, proposal={'kind': 'heuristic', 'title': 'Alias', 'rationale': 'test',
            'patch': {'heuristics': [{'name': 'qa'}, {'name': 'aliases', 'config': {'aliases': {'affirmative': 'yes'}}}]}}))
    with pytest.raises(ValueError): review(service, 'candidate', 'approve')
    cases = [{'response': 'affirmative', 'target': 'yes', 'expected_semantic': True},
             {'response': 'not affirmative', 'target': 'yes', 'expected_semantic': None}]
    assert review(service, 'candidate', 'validate', cases)['validation']['passed']
    assert review(service, 'candidate', 'approve')['status'] == 'approved'


def test_dependency_specific_invalidation(service):
    rev, run = setup_run(service)
    complete(service)
    original = service.results(run)
    models = original['run']['config']['models']
    scorer_rev = service.revise(rev, {'scorer': {'name': 'normalized'}})['revision_id']
    score_run = service.create_run({'revision_id': scorer_rev, 'models': models})['id']
    complete(service)
    rescored = service.results(score_run)
    assert [i['generation']['id'] for i in original['items']] == [i['generation']['id'] for i in rescored['items']]
    assert [i['score']['id'] for i in original['items']] != [i['score']['id'] for i in rescored['items']]
    heuristic_rev = service.revise(rev, {'heuristics': [{'name': 'qa', 'version': '2'}]})['revision_id']
    heuristic_run = service.create_run({'revision_id': heuristic_rev, 'models': models, 'analysis_overrides':{'scorer':original['definition']['scorer']}})['id']
    complete(service)
    rediagnosed = service.results(heuristic_run)
    assert [i['score']['id'] for i in original['items']] == [i['score']['id'] for i in rediagnosed['items']]
    assert [i['diagnostic']['id'] for i in original['items']] != [i['diagnostic']['id'] for i in rediagnosed['items']]
    prompt_rev = service.revise(rev, {'prompt_template': '{input}\n{choices}\nReturn only the letter.'})['revision_id']
    prompt_run = service.create_run({'revision_id': prompt_rev, 'models': models})['id']
    assert not any(i['reused'] for i in service.results(prompt_run)['items'])


def test_inspector_failure_retry_and_disagreement(service, monkeypatch):
    from kineticeval import providers
    rev, run = setup_run(service, True)
    original = providers.inspector
    async def fail(payload, report=False):
        if not report: raise ValueError('temporary inspector failure')
        return await original(payload, report)
    monkeypatch.setattr(providers, 'inspector', fail)
    complete(service)
    assert service.results(run)['run']['status'] == 'completed_with_errors'
    async def disagree(payload, report=False):
        result = await original(payload, report)
        if not report: result['semantic_correct'] = False
        return result
    monkeypatch.setattr(providers, 'inspector', disagree)
    service.control(run, 'retry')
    complete(service)
    result = service.results(run)
    assert result['run']['status'] == 'completed'
    assert any('semantic_correct' in row['disagreements'] for row in result['items'])
    assert all(row['inspector']['status'] == 'succeeded' for row in result['items'])


def test_partial_heuristic_coverage_and_negation():
    from kineticeval.components import diagnose
    base = {'sample': {'target': 'A', 'choices': ['Paris','Berlin','Rome','Madrid']},
        'score': {'value': 0}, 'expected_format': 'letter', 'components': [{'name': 'qa', 'version': '1', 'config': {}}],
        'messages': [{'role': 'user', 'content': 'Answer with only one letter.'}]}
    response = diagnose({**base, 'response': 'Det rigtige svar er A.'})
    assert response['finding']['semantic_correct'] is True
    assert response['finding']['format_compliance'] is False
    assert 'scorer_correct' in response['unresolved']
    for text in ['Not A, B', 'A or D', 'The answer is not A.', 'D is wrong; A is right']:
        assert diagnose({**base, 'response': text})['finding']['semantic_correct'] is None


def test_imported_overlay_preserves_template(service):
    from kineticeval.inspect_adapter import import_task
    definition = asyncio.run(import_task('kineticeval.fixtures:inspect_fixture'))
    rev = service.create_evaluation(definition)['revision_id']
    preview = service.preview(rev, {'overlays': {'one': {'input': 'What is 3 + 3?', 'target': '6'}}})
    assert preview['examples'][0]['messages'][-1]['content'] == 'What is 3 + 3?\nAnswer briefly.'


def test_metrics_are_durable_and_metric_only_revision_reuses_scores(service, monkeypatch):
    from kineticeval import components
    rev, run = setup_run(service)
    complete(service)
    original = service.results(run)
    def broken(payload): raise ValueError('new incompatible metric implementation')
    monkeypatch.setattr(components, 'aggregate', broken)
    assert service.results(run)['summaries'][0]['metrics'] == original['summaries'][0]['metrics']
    metric_rev = service.revise(rev, {'metrics': [{'name': 'accuracy', 'version': '2'}]})['revision_id']
    new = service.create_run({'revision_id': metric_rev, 'models': original['run']['config']['models']})['id']
    complete(service)
    result = service.results(new)
    assert result['run']['status'] == 'completed_with_errors'
    assert all(i['reused'] for i in result['items'])
    assert [i['score']['id'] for i in result['items']] == [i['score']['id'] for i in original['items']]


def test_pending_excludes_canceled_and_subset_reruns(service):
    rev, run = setup_run(service)
    service.control(run, 'cancel')
    assert service.pending() == {}
    partial = service.create_run({'revision_id': rev, 'parent_id': run, 'subsets': ['science'],
        'models': [{'name':'literal','backend':'fake'}]})['id']
    result = service.results(partial)
    assert result['run']['manifest']['ids'] == ['planet']
    assert len(service.pending()['literal']) == 1


def test_endpoint_participates_in_identity(service, monkeypatch):
    rev, _ = setup_run(service)
    monkeypatch.setenv('UCLOUD_BASE_URL','https://first.invalid/v1')
    run1 = service.create_run({'revision_id': rev, 'models':[{'name':'model','backend':'ucloud'}]})['id']
    monkeypatch.setenv('UCLOUD_BASE_URL','https://second.invalid/v1')
    run2 = service.create_run({'revision_id': rev, 'models':[{'name':'model','backend':'ucloud'}]})['id']
    assert service.results(run1)['items'][0]['generation']['id'] != service.results(run2)['items'][0]['generation']['id']


def test_worker_interruptions_preserve_completed_stages(service):
    rev, run = setup_run(service, True)
    # Simulate process loss after claiming each stage, then reconstruct the worker.
    interrupted = set()
    for _ in range(200):
        service.schedule()
        with service.db.session() as s:
            kinds = s.scalars(select(Job.kind).where(Job.status == 'pending')).all()
        candidate = next((k for k in kinds if k not in interrupted), None)
        if candidate:
            claim = service.db.claim(lease_seconds=-1)
            if claim:
                interrupted.add(claim['kind'])
        if not asyncio.run(Worker(service.db).step()):
            break
    result = service.results(run)
    assert result['run']['status'] == 'completed'
    assert {'generation','scoring','diagnostic','inspector','report','metrics'} <= interrupted
    assert all(row['generation']['status'] == 'succeeded' for row in result['items'])


def test_audit_sampling_probabilities(service):
    definition = demo_definition()
    definition['examples'] = [{'id':str(i),'input':f'Question {i}', 'target':'A' if i<100 else 'B'} for i in range(120)]
    definition['inspector'] = {'enabled':True,'provider':'fake','audit_fraction':.1,'audit_min':2,'audit_max':20,'seed':5}
    rev=service.create_evaluation(definition)['revision_id']
    run=service.create_run({'revision_id':rev,'models':[{'name':'fixture','backend':'fake'}]})['id']
    complete(service)
    result=service.results(run)
    strata={row['outcome']:row for row in result['run']['audit']['strata']}
    assert strata['success']['sample_size']==10
    assert strata['failure']['sample_size']==2
    assert strata['success']['probability']==.1
    assert len(result['run']['audit']['selected'])==12
    assert sum(bool(row['inspector']) for row in result['items'])>=12


def test_installed_custom_components(service, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples'))
    definition={'name':'numeric QA','examples':[{'id':'one','input':'2+2?','target':'4'}],
        'scorer':{'name':'python:custom_components:numeric_score'},
        'heuristics':[{'name':'qa'},{'name':'python:custom_components:answered'}],
        'metrics':[{'name':'python:custom_components:mean_score'}]}
    rev=service.create_evaluation(definition)['revision_id']
    run=service.create_run({'revision_id':rev,'models':[{'name':'fixture','backend':'fake','parameters':{'response':'4.0'}}]})['id']
    complete(service)
    result=service.results(run)
    assert result['run']['status']=='completed'
    assert result['summaries'][0]['metrics']['python:custom_components:mean_score']==1


def test_concurrent_handlers_and_reproducible_audit(service):
    rev, run = setup_run(service, True)
    async def drain():
        await asyncio.gather(*(Worker(service.db).drain() for _ in range(3)))
    asyncio.run(drain())
    first = service.results(run)
    assert first['run']['status']=='completed'
    second_id = service.create_run(first['run']['config'])['id']
    asyncio.run(drain())
    second = service.results(second_id)
    def selected(result):
        return {(r['example_id'],r['model']['name'],r['repeat']) for r in result['items'] if r['audited']}
    assert selected(first)==selected(second)
    assert all(len(service.attempts(row['generation']['id']))==1 for row in second['items'])


def test_library_summary_omits_examples_and_revision_loads_on_demand(service):
    rev, run = setup_run(service)
    client = TestClient(create_app(str(service.db.engine.url.database)))
    listing = client.get('/api/evaluations?summary=true').json()
    summary = listing[0]['revisions'][0]['definition']
    assert 'examples' not in summary
    assert summary['example_count'] == 3
    detail = client.get('/api/revisions/'+rev).json()
    assert len(detail['examples']) == 3
    assert len(client.get('/api/evaluations').json()[0]['revisions'][0]['definition']['examples']) == 3


def test_evaluation_overview_counts_actual_corrections_and_selected_subsets(service):
    rev,_=setup_run(service)
    new=service.revise(rev,{'overlays':{'capital':{'target':'B'},'sum':{'target':'A'}},
        'sampler':{'limit':1,'seed':42,'subsets':['geography','science']}})['revision_id']
    data=service.evaluations(summary=True)[0]['revisions'][0]['definition']
    assert data['overview']=={'total':3,'corrected':1,'selected':1,
        'subsets':{'geography':1,'arithmetic':1,'science':1},'type':'Multiple-choice QA'}
    assert 'examples' not in data


def test_library_name_tracks_latest_revision_without_rewriting_history(service):
    rev,run=setup_run(service)
    service.revise(rev,{'name':'Updated evaluation name','description':'Updated description'})
    evaluation=service.evaluations(summary=True)[0]
    assert evaluation['name']=='Updated evaluation name'
    assert len(evaluation['revisions'])==1
    assert service.results(run)['definition']['name']=='QA diagnostic laboratory'


def test_revision_names_are_display_metadata(service):
    revision, run = setup_run(service)
    before = service.results(run)
    assert service.name_revision(revision, '  smoke  ')['name'] == 'smoke'
    for summary in (False, True):
        assert service.evaluations(summary)[0]['revisions'][0]['name'] == 'smoke'
    assert service.results(run) == before
    child = service.revise(revision, {'sampler': {'seed':43}, 'description': 'Full version'})['revision_id']
    revisions = {r['id']: r for r in service.evaluations()[0]['revisions']}
    assert revisions[child]['name'] == ''
    assert revisions[revision]['name'] == 'smoke'
    service.name_revision(revision, '')
    assert next(r for r in service.evaluations()[0]['revisions'] if r['id'] == revision)['name'] == ''
    with pytest.raises(KeyError):
        service.name_revision('missing', 'full')


def test_revision_name_api(tmp_path):
    client = TestClient(create_app(tmp_path / 'names.db'))
    revision = client.post('/api/evaluations', json=demo_definition()).json()['revision_id']
    assert client.post(f'/api/revisions/{revision}/name', json={'name': 'full'}).status_code == 200
    assert client.post(f'/api/revisions/{revision}/name', json={'name': 'x' * 201}).status_code == 422
    assert client.post('/api/revisions/missing/name', json={'name': 'full'}).status_code == 404


def test_scoring_is_scheduled_before_generation_finishes(service):
    revision, run = setup_run(service)
    claim = service.db.claim_many(kind='generation',run_id=run,limit=1)[0]
    service.db.finish(claim['id'],claim['token'],result={'response':'A'})
    service.schedule()
    scoring = service.db.claim_many(kind='scoring',run_id=run,limit=10)
    assert len(scoring) == 1
    asyncio.run(Worker(service.db).execute(scoring[0]))
    service.schedule()
    assert len(service.db.claim_many(kind='diagnostic',run_id=run,limit=10)) == 1
    with service.db.session() as s:
        assert s.scalar(select(Job).where(Job.kind=='generation',Job.status=='pending')) is not None


def test_analysis_worker_round_robins_runs_and_prioritizes_investigations(service):
    from kineticeval.service import job_for
    first_revision, first = setup_run(service)
    second = service.create_run({'revision_id':first_revision,'models':[{'name':'other','backend':'fake'}]})['id']
    with service.db.session() as session:
        for run in (first,second):
            for index in range(3):
                job_for(session,run,'inspector',{'protocol':{},'run':run,'index':index})
        job_for(session,second,'investigation',{'protocol':{},'run':second})
    worker=Worker(service.db,analysis_only=True)
    seen=[]
    async def no_schedule(): pass
    async def compute(claim):
        seen.append((claim['payload']['run'],claim['kind']))
        service.db.finish(claim['id'],claim['token'],result={})
    worker.schedule=no_schedule
    worker.execute=compute
    async def steps():
        for _ in range(4):await worker.step()
    asyncio.run(steps())
    assert seen[0]==(second,'investigation')
    assert [run for run,kind in seen]==[second,first,second,first]


def test_canceled_claim_never_calls_inspector_and_can_resume(service, monkeypatch):
    from kineticeval.service import job_for
    from kineticeval import providers
    revision, run = setup_run(service)
    with service.db.session() as s:
        job_for(s,run,'inspector',{'protocol':{},'config':{'provider':'fake'}})
    claim=service.db.claim_many(kind='inspector',run_id=run)[0]
    service.control(run,'cancel')
    async def unexpected(*args,**kwargs):
        raise AssertionError('Canceled work must not call provider')
    monkeypatch.setattr(providers,'inspector',unexpected)
    asyncio.run(Worker(service.db).execute(claim))
    with service.db.session() as s:
        assert s.get(Job,claim['id']).status=='pending'
    service.control(run,'resume')
    assert service.db.claim_many(kind='inspector',run_id=run)


def test_cancellation_prevents_inspector_repair_request(service, monkeypatch):
    from kineticeval import providers
    revision, run = setup_run(service,True)
    worker=Worker(service.db)
    async def prepare():
        for _ in range(30):
            service.schedule()
            claims=service.db.claim_many(kinds=['generation','scoring','diagnostic','metrics'],run_id=run)
            if not claims:break
            await worker.execute(claims[0])
    asyncio.run(prepare())
    service.schedule()
    claim=service.db.claim_many(kind='inspector',run_id=run)[0]
    claim['payload']['config']['provider']='ucloud'
    calls=[]
    class Provider:
        def __init__(self,**kwargs):pass
        async def chat(self,*args,**kwargs):
            calls.append(1)
            service.control(run,'cancel')
            return {'response':'invalid JSON','metadata':{}}
    monkeypatch.setattr(providers,'UCloud',Provider)
    asyncio.run(worker.execute(claim))
    assert len(calls)==1
    with service.db.session() as s:
        assert s.get(Job,claim['id']).status=='pending'


def test_metrics_do_not_wait_for_other_models(service):
    _, run = setup_run(service)
    claims = service.db.claim_many(kind='generation', run_id=run, limit=100)
    for claim in claims:
        if claim['payload']['model']['name'] == 'literal':
            service.db.finish(claim['id'], claim['token'], result={'response': 'A'})
    for _ in range(4):
        service.schedule()
        for claim in service.db.claim_many(kinds=['scoring', 'diagnostic', 'metrics'], run_id=run, limit=100):
            asyncio.run(Worker(service.db).execute(claim))
    result = service.results(run)
    assert result['run']['status'] == 'active'
    assert result['summaries'][0]['metrics']
    assert not result['summaries'][1]['metrics']


def test_cpu_lane_scores_while_inspector_slots_are_occupied(service):
    from kineticeval.service import job_for
    _, run = setup_run(service)
    generation = service.db.claim_many(kind='generation', run_id=run, limit=1)[0]
    service.db.finish(generation['id'], generation['token'], result={'response':'A'})
    with service.db.session(immediate=True) as session:
        for n in range(4):
            job_for(session, run, 'inspector', {'config':{'provider':'fake'},'number':n})
    inspectors = service.db.claim_many(kind='inspector', run_id=run, limit=4)
    assert len(inspectors) == 4
    worker = Worker(service.db, analysis_only=True)
    assert asyncio.run(worker.cpu_step())
    with service.db.session() as session:
        assert session.scalar(select(Job).where(Job.kind=='scoring',Job.status=='succeeded')) is not None
        assert all(session.get(Job, claim['id']).status=='running' for claim in inspectors)
