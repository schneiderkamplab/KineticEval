"""Evaluation lifecycle, dependency planning, and partial-run accounting."""
import copy
import math
import os
import random
import uuid
from collections import Counter
from sqlalchemy import select
from sqlalchemy.orm import defer
from .core import identity
from .db import Evaluation, Revision, RevisionLabel, Run, Item, Job, RunJob, Attempt, Proposal
from .schemas import Definition, RunRequest
from .components import fingerprint


def uid():
    return uuid.uuid4().hex


def require(session, cls, key):
    value = session.get(cls, key)
    if value is None:
        raise KeyError(f'{cls.__name__} not found: {key}')
    return value


def provider_endpoint(backend, name, configured=None):
    from urllib.parse import urlparse
    variable = 'UCLOUD_BASE_URL' if backend == 'ucloud' else name.split('/')[0].upper() + '_BASE_URL'
    endpoint = configured or os.environ.get(variable)
    if backend == 'ucloud' and not endpoint:
        raise ValueError('Configure UCLOUD_BASE_URL before creating UCloud work')
    if endpoint:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Provider endpoint must be an HTTP(S) URL without credentials, query, or fragment')
        return endpoint.rstrip('/')
    return None


def pin(definition):
    data = definition.model_dump()
    if data['inspector']['enabled'] and data['inspector']['provider'] == 'ucloud':
        data['inspector']['endpoint'] = provider_endpoint('ucloud', '', data['inspector'].get('endpoint'))
    for component in [data['scorer'], *data['metrics'], *data['heuristics']]:
        if component['name'].startswith('python:'):
            component['config']['_fingerprint'] = fingerprint(component['name'].removeprefix('python:'))
    return data


def materialize(definition):
    rows = []
    for original in definition['examples']:
        row = {**original, **definition['overlays'].get(original['id'], {})}
        # A corrected imported input must replace its frozen user prompt explicitly.
        if 'input' in definition['overlays'].get(original['id'], {}) and 'messages' not in definition['overlays'][original['id']]:
            if original.get('messages'):
                row['messages'] = copy.deepcopy(original['messages'])
                matches = [m for m in row['messages'] if original['input'] in m['content']]
                if not matches:
                    raise ValueError('Correcting this imported input requires an explicit messages overlay')
                for message in matches:
                    message['content'] = message['content'].replace(original['input'], row['input'])
        if definition['sampler']['subsets'] and row['subset'] not in definition['sampler']['subsets']:
            continue
        rows.append(row)
    random.Random(definition['sampler']['seed']).shuffle(rows)
    return rows[:definition['sampler']['limit']] if definition['sampler']['limit'] else rows


def definition_overview(definition):
    rows = [{**r, **definition['overlays'].get(r['id'], {})} for r in definition['examples']]
    total = len(rows)
    corrected = sum(any(value != row.get(key) for key, value in definition['overlays'].get(row['id'], {}).items()) for row in definition['examples'])
    choices = sum(bool(row['choices']) for row in rows)
    kind = 'Multiple-choice QA' if choices == total else 'Mixed question answering' if choices else 'Yes/no QA' if definition['expected_format'] == 'yes_no' else 'Short-answer QA'
    counts = dict(Counter(row['subset'] for row in rows))
    included = sum(not definition['sampler']['subsets'] or row['subset'] in definition['sampler']['subsets'] for row in rows)
    return {'total': total, 'corrected': corrected, 'selected': min(included, definition['sampler']['limit'] or included), 'subsets': counts, 'type':kind}


def render(definition, sample):
    choices = '\n'.join(f'{chr(65+i)}. {choice}' for i, choice in enumerate(sample['choices']))
    if sample.get('messages'):
        messages = copy.deepcopy(sample['messages'])
        if definition['prompt_template'] != '{input}':
            last = next((m for m in reversed(messages) if m['role'] == 'user'), None)
            if last is None:
                raise ValueError('Imported example has no user message')
            last['content'] = definition['prompt_template'].format(input=last['content'], choices=choices)
    else:
        messages = [{'role': 'user', 'content': definition['prompt_template'].format(input=sample['input'], choices=choices)}]
    if definition['system_prompt']:
        messages.insert(0, {'role': 'system', 'content': definition['system_prompt']})
    return messages


def physical_model(spec):
    physical = dict(spec)
    physical['name'] = physical.pop('model_identifier', None) or physical['name']
    physical.pop('configuration_id', None)
    if physical.get('chat_template') is None:
        physical.pop('chat_template', None)
    return physical


def job_for(s, run_id, kind, payload, model=''):
    if kind == 'generation':
        payload = {**payload, 'model': physical_model(payload['model'])}
        model = payload['model']['name']
    if kind in {'inspector','report','investigation'} and 'config' in payload:
        # Selection budgets do not change the inspector's answer to identical evidence.
        payload = {**payload, 'config': {k:v for k,v in payload['config'].items() if k != 'failure_inspector_limit'}}
    if kind in {'inspector', 'report', 'investigation'} and 'protocol' not in payload:
        # Existing run jobs keep their pinned protocol, including legacy provenance.
        # New runs always receive the current protocol and cannot hit legacy cache keys.
        indexes = s.info.setdefault('inspector_protocol_jobs', {})
        index_key = (run_id, kind)
        if index_key not in indexes:
            candidates = s.scalars(select(Job).join(RunJob,RunJob.job_id==Job.id)
                .where(RunJob.run_id==run_id, Job.kind==kind)).all()
            indexes[index_key] = {identity({k:v for k,v in job.payload.items() if k!='protocol'}):job for job in candidates}
        existing = indexes[index_key].get(identity(payload))
        if existing is not None:
            return existing
        from .inspector_protocol import protocol_for
        payload = {**payload, 'protocol':protocol_for(payload,report=kind!='inspector')}
    key = identity({'kind': kind, 'payload': payload, 'engine_version': '1'})
    job = s.get(Job, key)
    if job is None:
        job = Job(id=key, kind=kind, payload=payload, model=model)
        s.add(job)
        s.flush()
    if s.get(RunJob, (run_id, key)) is None:
        s.add(RunJob(run_id=run_id, job_id=key))
        s.flush()
    return job


class Service:
    def __init__(self, db):
        self.db = db

    def create_evaluation(self, definition):
        data = pin(Definition.model_validate(definition))
        with self.db.session() as s:
            ev = Evaluation(id=uid(), name=data['name'])
            rev = Revision(id=uid(), evaluation_id=ev.id, definition=data)
            s.add_all([ev, rev])
            return {'id': ev.id, 'revision_id': rev.id}

    def revise(self, revision_id, patch):
        from .revisions import save_revision
        with self.db.session(immediate=True) as s:
            previous = require(s, Revision, revision_id)
            if 'overlays' in patch:
                patch = {**patch, 'overlays': {**previous.definition['overlays'], **patch['overlays']}}
            data = pin(Definition.model_validate({**previous.definition, **patch}))
            rev, created = save_revision(s, previous, data)
            return {'revision_id': rev.id, 'evaluation_id': rev.evaluation_id, 'created': created}

    def preview(self, revision_id, patch=None):
        with self.db.session() as s:
            old = require(s, Revision, revision_id).definition
            new = Definition.model_validate({**old, **(patch or {})}).model_dump()
            before = {row['id']: (row, render(old, row)) for row in materialize(old)}
            rows = materialize(new)
            changed = [row['id'] for row in rows if row['id'] not in before or render(new, row) != before[row['id']][1]]
            targets = [row['id'] for row in rows if row['id'] in before and row['target'] != before[row['id']][0]['target']]
            return {'count': len(rows), 'generation_changes': changed, 'reference_changes': targets,
                'scoring_changed': old['scorer'] != new['scorer'], 'heuristics_changed': old['heuristics'] != new['heuristics'],
                'inspector_changed': old['inspector'] != new['inspector'],
                'examples': [{'sample': r, 'messages': render(new, r)} for r in rows[:5]]}

    def create_run(self, request):
        req = RunRequest.model_validate(request)
        # Weight hashing and Hub lookups must not hold the database write lock.
        for model in req.models:
            if model.backend == 'hf':
                from .local_hf import pin_model
                pin_model(model)
        with self.db.session(immediate=True) as s:
            result = self._create_run(s, req)
            from .db import ResultCollection
            revision = require(s, Revision, req.revision_id)
            collection = ResultCollection(id=uid(), name=revision.definition['name'] + (' · partial rerun' if req.parent_id else ' · run'), run_ids=[result['id']])
            s.add(collection)
            return {**result, 'result_id': collection.id}

    def _create_run(self, s, req):
        """Create jobs inside the caller transaction; model artifacts must already be pinned."""
        revision = require(s, Revision, req.revision_id)
        definition = copy.deepcopy(revision.definition)
        allowed = {"scorer", "metrics", "heuristics", "inspector", "expected_format", "generation_defaults"}
        if set(req.analysis_overrides) - allowed:
            raise ValueError("Unsupported run analysis setting")
        overrides = copy.deepcopy(req.analysis_overrides)
        if 'generation_defaults' in overrides:
            overrides['generation_defaults'] = {**definition.get('generation_defaults',{}), **overrides['generation_defaults']}
        definition.update(overrides)
        definition = pin(Definition.model_validate(definition))
        defaults = {**definition['source'].get('generation_config', {}), **definition.get('generation_defaults',{})}
        for model in req.models:
            model.endpoint = provider_endpoint(model.backend, model.model_identifier or model.name, model.endpoint) if model.backend not in {'fake', 'hf'} else None
            if model.backend != 'fake':
                model.parameters = {**defaults, **model.parameters}
                type(model).model_validate(model.model_dump())
        rows = materialize(definition)
        full_ids = [r['id'] for r in rows]
        if req.parent_id:
            parent = require(s, Run, req.parent_id)
            parent_rev = require(s, Revision, parent.revision_id)
            if parent_rev.evaluation_id != revision.evaluation_id:
                raise ValueError('Partial reruns must belong to the same evaluation')
        if req.example_ids is not None:
            if not req.example_ids or set(req.example_ids) - set(full_ids):
                raise ValueError('Select nonempty example IDs within the revision sample manifest')
            rows = [r for r in rows if r['id'] in req.example_ids]
        if req.subsets is not None:
            if not req.subsets or set(req.subsets) - {r['subset'] for r in rows}:
                raise ValueError('Select existing nonempty subsets')
            rows = [r for r in rows if r['subset'] in req.subsets]
        if not rows:
            raise ValueError('Sampler selected no examples')
        if len({m.name for m in req.models}) != len(req.models):
            raise ValueError('Model names must be unique within a run')
        run = Run(id=uid(), revision_id=revision.id, parent_id=req.parent_id, config={**req.model_dump(), 'effective_definition':copy.deepcopy(definition)},
            manifest={'ids': [r['id'] for r in rows], 'full_ids': full_ids, 'sampler': definition['sampler'],
                'partial': len(rows) != len(full_ids), 'source': definition['source']})
        s.add(run)
        s.flush()
        fresh = uid() if req.fresh else None
        for row in rows:
            messages = render(definition, row)
            for model in req.models:
                for repeat in range(req.repeats):
                    payload = {'model': model.model_dump(), 'messages': messages, 'repeat': repeat, 'fresh': fresh}
                    job = job_for(s, run.id, 'generation', payload, model.name)
                    s.add(Item(id=uid(), run_id=run.id, example_id=row['id'], model=model.model_dump(), repeat=repeat,
                        sample=row, messages=messages, generation_id=job.id, reused=job.status == 'succeeded'))
        return {'id': run.id}

    def control(self, run_id, action, job_ids=None):
        with self.db.session(immediate=True) as s:
            run = require(s, Run, run_id)
            if action == 'cancel':
                run.status = 'canceled'
            elif action in {'resume', 'retry'}:
                run.status = 'active'
                if action == 'retry':
                    jobs = s.scalars(select(Job).join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run_id)).all()
                    known = {j.id for j in jobs}
                    if job_ids and set(job_ids) - known:
                        raise ValueError('Job does not belong to this run')
                    # A retry must not select a new cohort and spend the failure
                    # budget again. Preserve an explicitly capped selection.
                    if not run.audit or 'failure_review' not in run.audit:
                        run.audit = None
                    for old_id in (run.metric_jobs or {}).values():
                        link = s.get(RunJob, (run.id, old_id))
                        if link: s.delete(link)
                    run.metric_jobs = {}
                    if run.report_job:
                        link = s.get(RunJob, (run.id, run.report_job))
                        if link: s.delete(link)
                        run.report_job = None
                    for job in jobs:
                        if job.status == 'failed' and (job_ids is None or job.id in job_ids):
                            job.status, job.error = 'pending', None
            else:
                raise ValueError('Unknown action')
            return {'status': run.status}

    def schedule(self, run_ids=None):
        """Advance each run in its own transaction; new runs get a turn first."""
        with self.db.session() as s:
            query = select(Run.id).where(Run.status == 'active').order_by(Run.created.desc())
            if run_ids is not None:
                query = query.where(Run.id.in_(run_ids))
            ids = list(s.scalars(query))
        for run_id in ids:
            self._schedule_run(run_id)

    def _schedule_run(self, run_id):
        with self.db.session(immediate=True) as s:
            run = require(s, Run, run_id)
            if run.status != 'active':
                return
            definition = run.config.get('effective_definition') or require(s, Revision, run.revision_id).definition
            # Retain strong references: otherwise SQLAlchemy expires weakly held jobs
            # and repeated per-example lookups turn into thousands of SQL reads.
            job_cache = {job.id: job for job in s.scalars(select(Job).options(defer(Job.payload))
                .join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run.id)).all()}
            items = s.scalars(select(Item).where(Item.run_id == run.id)).all()
            for item in items:
                if item.diagnostic_id is not None:
                    continue
                gen = s.get(Job, item.generation_id)
                if gen.status != 'succeeded':
                    continue
                evidence = {'sample': item.sample, 'messages': item.messages, 'response': gen.result['response'], 'model': item.model}
                score = s.get(Job, item.score_id) if item.score_id else job_for(s, run.id, 'scoring', {**evidence, 'component': definition['scorer'], 'generation_id': gen.id})
                item.score_id = score.id
                if score.status != 'succeeded':
                    continue
                diag = job_for(s, run.id, 'diagnostic', {**evidence, 'score': score.result, 'score_id': score.id,
                    'components': definition['heuristics'], 'expected_format': definition['expected_format']})
                item.diagnostic_id = diag.id
            s.flush()
            terminal = {'succeeded', 'failed'}
            def stage_done(item):
                gen = s.get(Job, item.generation_id)
                if gen.status == 'failed': return True
                if not item.score_id: return False
                score = s.get(Job, item.score_id)
                if score.status == 'failed': return True
                return bool(item.diagnostic_id and s.get(Job, item.diagnostic_id).status in terminal)
            metric_jobs = dict(run.metric_jobs or {})
            for model in run.config['models']:
                if model['name'] in metric_jobs or not all(stage_done(i) for i in items if i.model['name'] == model['name']):
                    continue
                scores = [s.get(Job, i.score_id) for i in items if i.model['name'] == model['name'] and i.score_id]
                scores = [score for score in scores if score.status == 'succeeded']
                metric = job_for(s, run.id, 'metrics', {'components': definition['metrics'],
                    'scores': [score.result for score in scores], 'score_ids': [score.id for score in scores]})
                metric_jobs[model['name']] = metric.id
            run.metric_jobs = metric_jobs
            if not all(stage_done(i) for i in items):
                return
            if any(s.get(Job, key).status not in terminal for key in metric_jobs.values()):
                return
            from .investigations import schedule_investigations
            investigations = schedule_investigations(s, run, items, definition, job_for)
            limit = run.config.get('inspector_budget',{}).get('failure_inspector_limit', definition['inspector'].get('failure_inspector_limit'))
            if run.audit is None:
                run.audit = self._audit(s, run, items, {**definition['inspector'],'failure_inspector_limit':limit})
                if limit is not None:
                    from .inspection_policy import select_failures
                    run.audit = {**run.audit,'failure_review':select_failures(s,items,run.audit,limit)}
            allowed_failures = set(run.audit['failure_review']['selected']) if 'failure_review' in run.audit else None
            for item in items:
                if item.inspector_id:
                    continue
                if not item.diagnostic_id:
                    continue
                diag = s.get(Job, item.diagnostic_id)
                score = s.get(Job, item.score_id)
                if diag.status != 'succeeded' or not definition['inspector']['enabled']:
                    continue
                if score.result['value'] < 1 and allowed_failures is not None and item.id not in allowed_failures:
                    continue
                audited = item.id in run.audit['selected']
                if audited or (score.result['value'] < 1 and diag.result['unresolved']):
                    gen = s.get(Job, item.generation_id)
                    evidence = {'sample': item.sample, 'messages': item.messages, 'response': gen.result['response'],
                        'score': score.result, 'diagnostic': diag.result}
                    job = job_for(s, run.id, 'inspector', {'config': definition['inspector'], 'evidence': evidence,
                        'dimensions': ['semantic_correct', 'format_compliance', 'instruction_adequacy', 'scorer_correct'] if audited else diag.result['unresolved'],
                        'diagnostic_id': diag.id, 'purpose': 'audit' if audited else 'unresolved'})
                    item.inspector_id = job.id
            s.flush()
            if any(j.status not in terminal for j in investigations):
                return
            if all(not i.inspector_id or s.get(Job, i.inspector_id).status in terminal for i in items):
                if definition['inspector']['enabled']:
                    if run.report_job is None:
                        evidence = self._report_evidence(s, run, items)
                        report = job_for(s, run.id, 'report', {'config': definition['inspector'], 'evidence': evidence})
                        run.report_job = report.id
                    report = s.get(Job, run.report_job)
                    if report.status not in terminal:
                        return
                    if report.status == 'succeeded':
                        for idx, proposal in enumerate(report.result['suggestions']):
                            key = identity({'run': run.id, 'report': report.id, 'index': idx})
                            if s.get(Proposal, key) is None:
                                s.add(Proposal(id=key, run_id=run.id, proposal=proposal))
                jobs = s.scalars(select(Job).join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run.id)).all()
                run.status = 'completed_with_errors' if any(j.status == 'failed' for j in jobs) else 'completed'


    def _audit(self, s, run, items, config):
        rng = random.Random(config['seed'])
        groups = {}
        selected, strata = {}, []
        for item in items:
            if item.score_id:
                score = s.get(Job, item.score_id)
                if score.status == 'succeeded':
                    key = (item.model['name'], 'success' if score.result['value'] >= 1 else 'failure')
                    groups.setdefault(key, []).append(item.id)
        item_keys = {item.id: (item.example_id, item.repeat) for item in items}
        for (model, outcome), ids in sorted(groups.items()):
            ids.sort(key=lambda key: item_keys[key])
            n = min(len(ids), config['audit_max'], max(config['audit_min'], math.ceil(len(ids)*config['audit_fraction']))) if config['enabled'] else 0
            if outcome == 'failure' and config.get('failure_inspector_limit') is not None:
                n = min(n, config['failure_inspector_limit'])
            chosen = rng.sample(ids, n)
            for key in chosen:
                selected[key] = {'probability': n/len(ids), 'model': model, 'stratum': outcome}
            strata.append({'model': model, 'outcome': outcome, 'population': len(ids), 'sample_size': n, 'probability': n/len(ids)})
        return {'seed': config['seed'], 'selected': selected, 'strata': strata}

    def set_failure_inspector_limit(self, run_id, limit, resume=False):
        import time
        from .inspection_policy import select_failures
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError('Failure limit must be a non-negative integer or null')
        with self.db.session(immediate=True) as s:
            run = require(s, Run, run_id)
            if run.status in {'completed','completed_with_errors'}:
                raise ValueError('This run has finished; set a limit in a new evaluation revision')
            run.status = 'canceled'
            history = list(run.config.get('inspector_budget_history',[]))
            history.append({'changed_at':time.time(),'previous':run.config.get('inspector_budget'),
                'failure_inspector_limit':limit})
            run.config = {**run.config,'inspector_budget':{'failure_inspector_limit':limit},'inspector_budget_history':history}
            review = None
            if run.audit is not None:
                items = s.scalars(select(Item).where(Item.run_id == run_id)).all()
                job_cache = {job.id:job for job in s.scalars(select(Job).options(defer(Job.payload))
                    .join(RunJob,RunJob.job_id==Job.id).where(RunJob.run_id==run_id)).all()}
                audit = copy.deepcopy(run.audit)
                if 'original_selection' in audit:
                    audit.update(copy.deepcopy(audit['original_selection']))
                audit.setdefault('original_selection',{'selected':copy.deepcopy(audit['selected']),'strata':copy.deepcopy(audit['strata'])})
                review = select_failures(s,items,audit,limit,preserve_started=True)
                allowed = set(review['selected'])
                for item in items:
                    score = s.get(Job,item.score_id) if item.score_id else None
                    if not score or score.status != 'succeeded' or score.result['value'] >= 1 or item.id in allowed:
                        continue
                    audit['selected'].pop(item.id,None)
                    if item.inspector_id:
                        link = s.get(RunJob,(run_id,item.inspector_id))
                        if link:s.delete(link)
                        item.inspector_id = None
                for row in audit['strata']:
                    if row['outcome']=='failure':
                        row['sample_size'] = sum(v['model']==row['model'] and v['stratum']=='failure' for v in audit['selected'].values())
                        row['probability'] = None
                for value in audit['selected'].values():
                    if value['stratum']=='failure':value['probability']=None
                audit.update(failure_review=review,selection_note='Failure cap changed after execution began; retained attempts are not a fresh random audit sample.')
                run.audit = audit
                if run.report_job:
                    link=s.get(RunJob,(run_id,run.report_job))
                    if link:s.delete(link)
                    run.config={**run.config,'previous_report_jobs':run.config.get('previous_report_jobs',[])+[run.report_job]}
                    run.report_job=None
            if resume:
                run.status='active'
            return {'id':run_id,'status':run.status,'failure_inspector_limit':limit,'failure_review':review}

    def _report_evidence(self, s, run, items):
        rows = []
        for item in items:
            rows.append({'example_id': item.example_id, 'model': item.model['name'],
                'sample': item.sample, 'messages': item.messages,
                'response': s.get(Job, item.generation_id).result,
                'score': s.get(Job, item.score_id).result if item.score_id else None,
                'diagnostic': s.get(Job, item.diagnostic_id).result if item.diagnostic_id else None,
                'inspector': s.get(Job, item.inspector_id).result if item.inspector_id else None})
        return {'revision_id': run.revision_id, 'manifest': run.manifest, 'audit': {k:({key:value for key,value in v.items() if key!='selected'} if k=='failure_review' else v)
                for k,v in (run.audit or {}).items() if k not in {'selected','original_selection'}},
            'metrics': {model: s.get(Job, key).result for model, key in (run.metric_jobs or {}).items()},
            'investigations': [j.result for j in s.scalars(select(Job).join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run.id, Job.kind == 'investigation')).all()],
            'total_results': len(rows), 'included_results': len(rows[:200]), 'results': rows[:200],
            'note': 'Results truncated to first 200 when necessary. Audit is stratified, not a full independent rescore.'}

    def name_revision(self, revision_id, name):
        with self.db.session() as s:
            require(s, Revision, revision_id)
            label = s.get(RevisionLabel, revision_id)
            if label is None:
                label = RevisionLabel(revision_id=revision_id, name=name.strip())
                s.add(label)
            else:
                label.name = name.strip()
            return {'revision_id': revision_id, 'name': label.name}

    def evaluations(self, summary=False):
        with self.db.session() as s:
            labels = {label.revision_id: label.name for label in s.scalars(select(RevisionLabel)).all()}
            result = []
            for ev in s.scalars(select(Evaluation).order_by(Evaluation.created.desc())).all():
                revisions = s.scalars(select(Revision).where(Revision.evaluation_id == ev.id).order_by(Revision.created.desc())).all()
                result.append({'id': ev.id, 'name': revisions[0].definition['name'] if revisions else ev.name, 'revisions': [{'id': r.id, 'name': labels.get(r.id, ''), 'parent_id': r.parent_id, 'created': r.created,
                    'definition': ({**{k:v for k,v in r.definition.items() if k != 'examples'},
                        'example_count': len(r.definition['examples']), 'overview': definition_overview(r.definition)} if summary else r.definition)} for r in revisions]})
            return result

    def runs(self):
        with self.db.session() as s:
            return [{'id': r.id, 'revision_id': r.revision_id, 'parent_id': r.parent_id, 'status': r.status,
                'created': r.created, 'manifest': r.manifest, 'config': {k:v for k,v in r.config.items() if k!='effective_definition'}} for r in s.scalars(select(Run).order_by(Run.created.desc())).all()]

    def results(self, run_id, summary=False):
        with self.db.session() as s:
            run = require(s, Run, run_id)
            definition = run.config.get('effective_definition') or require(s, Revision, run.revision_id).definition
            items = s.scalars(select(Item).where(Item.run_id == run_id)).all()
            # Keep a strong identity-map cache to avoid per-cell database queries
            # when displaying full benchmark runs. Request payloads are not needed.
            job_cache = {job.id: job for job in s.scalars(select(Job).options(defer(Job.payload)).join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run_id)).all()}
            rows = []
            def artifact(key):
                job = s.get(Job, key) if key else None
                return None if job is None else {'id': job.id, 'status': job.status, 'result': job.result, 'error': job.error}
            for item in items:
                diag = s.get(Job, item.diagnostic_id) if item.diagnostic_id else None
                inspector = s.get(Job, item.inspector_id) if item.inspector_id else None
                disagreements = []
                if diag and diag.result and inspector and inspector.result:
                    for dimension in ('semantic_correct', 'format_compliance', 'instruction_adequacy', 'scorer_correct'):
                        a, b = diag.result['finding'][dimension], inspector.result[dimension]
                        if a is not None and b is not None and a != b:
                            disagreements.append(dimension)
                rows.append({'disagreements': disagreements, 'id': item.id, 'example_id': item.example_id, 'model': item.model, 'repeat': item.repeat,
                    'sample': item.sample, 'messages': item.messages, 'reused': item.reused,
                    'generation': artifact(item.generation_id), 'score': artifact(item.score_id),
                    'diagnostic': artifact(item.diagnostic_id), 'inspector': artifact(item.inspector_id),
                    'audited': (run.audit or {}).get('selected', {}).get(item.id)})
            summaries = []
            for model in run.config['models']:
                model_rows = [r for r in rows if r['model']['name'] == model['name']]
                scores = [r['score']['result'] for r in model_rows if r['score'] and r['score']['status'] == 'succeeded']
                values = [v['value'] for v in scores]
                labels = Counter(label for r in model_rows if r['diagnostic'] and r['diagnostic']['result'] for label in r['diagnostic']['result']['finding']['labels'])
                semantic = [r['diagnostic']['result']['finding']['semantic_correct'] for r in model_rows if r['diagnostic'] and r['diagnostic']['result']]
                known = [v for v in semantic if v is not None]
                metric_job = s.get(Job, (run.metric_jobs or {}).get(model['name'])) if (run.metric_jobs or {}).get(model['name']) else None
                metrics = metric_job.result['metrics'] if metric_job and metric_job.result else {}
                metric_errors = [metric_job.error] if metric_job and metric_job.error else []
                inspected = [r['inspector']['result']['semantic_correct'] for r in model_rows if r['inspector'] and r['inspector']['result'] and r['inspector']['result']['semantic_correct'] is not None]
                summaries.append({'model': model['name'], 'total': len(model_rows), 'scored': len(scores),
                    'execution': {stage:dict(Counter((row[field]['status'] if row[field] else 'waiting') for row in model_rows if field!='inspector' or row[field])) for stage,field in [('generation','generation'),('scoring','score'),('diagnostic','diagnostic'),('inspector','inspector')]},
                    'metric_status': metric_job.status if metric_job else 'waiting',
                    'execution_errors': list(dict.fromkeys(row[field]['error'] for row in model_rows for field in ('generation','score','diagnostic','inspector') if row[field] and row[field].get('error'))),
                    'scored_correct': sum(v>=1 for v in values), 'coverage': len(scores)/len(model_rows), 'metrics': metrics, 'metric_errors': metric_errors,
                    'full_evaluation': not run.manifest['partial'] and len(scores) == len(model_rows),
                    'inspector_correct': sum(v is True for v in inspected), 'inspector_assessed': len(inspected),
                    'disagreements': sum(bool(r['disagreements']) for r in model_rows),
                    'semantic_correct': sum(v is True for v in known), 'semantic_known': len(known),
                    'heuristic_checked': len(semantic), 'semantic_incorrect': sum(v is False for v in known),
                    'semantic_unresolved': len(semantic) - len(known), 'labels': dict(labels),
                    'reused': sum(r['reused'] for r in model_rows),
                    'truncated': sum(bool(r['generation']['result'] and r['generation']['result'].get('metadata', {}).get('finish_reason') == 'length') for r in model_rows)})
            return {'run': {'id': run.id, 'evaluation_id': require(s, Revision, run.revision_id).evaluation_id, 'parent_id': run.parent_id, 'revision_id': run.revision_id, 'status': run.status, 'manifest': run.manifest,
                'config': {k:v for k,v in run.config.items() if k!='effective_definition'}, 'audit': run.audit if not summary else None}, 'definition': ({k:v for k,v in definition.items() if k not in {'examples','overlays'}} if summary else definition), 'items': [] if summary else rows, 'summaries': summaries,
                'report': artifact(run.report_job), 'investigations': [artifact(j.id) for j in job_cache.values() if j.kind == 'investigation'],
                'proposals': [{'id': p.id, 'proposal': p.proposal, 'status': p.status,
                    'validation': p.validation, 'revision_id': p.revision_id} for p in s.scalars(select(Proposal).where(Proposal.run_id == run_id)).all()]}

    def attempts(self, job_id):
        with self.db.session() as s:
            return [{'id': a.id, 'started': a.started, 'finished': a.finished, 'error': a.error}
                for a in s.scalars(select(Attempt).where(Attempt.job_id == job_id).order_by(Attempt.started)).all()]

    def pending(self):
        with self.db.session() as s:
            grouped = {}
            for job in s.scalars(select(Job).join(RunJob, RunJob.job_id == Job.id).join(Run, Run.id == RunJob.run_id).where(Job.kind == 'generation', Job.status == 'pending', Run.status == 'active').distinct()).all():
                grouped.setdefault(job.model, []).append({'id': job.id, 'inputs': job.payload})
            return grouped
