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
from .db import Evaluation, Revision, Run, Item, Job, RunJob, Attempt, Proposal
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
    if data['judge']['enabled'] and data['judge']['provider'] == 'ucloud':
        data['judge']['endpoint'] = provider_endpoint('ucloud', '', data['judge'].get('endpoint'))
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


def job_for(s, run_id, kind, payload, model=''):
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
        with self.db.session() as s:
            previous = require(s, Revision, revision_id)
            if 'overlays' in patch:
                patch = {**patch, 'overlays': {**previous.definition['overlays'], **patch['overlays']}}
            data = pin(Definition.model_validate({**previous.definition, **patch}))
            rev = Revision(id=uid(), evaluation_id=previous.evaluation_id, parent_id=previous.id, definition=data)
            s.add(rev)
            return {'revision_id': rev.id, 'evaluation_id': rev.evaluation_id}

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
                'judge_changed': old['judge'] != new['judge'],
                'examples': [{'sample': r, 'messages': render(new, r)} for r in rows[:5]]}

    def create_run(self, request):
        req = RunRequest.model_validate(request)
        # Weight hashing and Hub lookups must not hold the database write lock.
        for model in req.models:
            if model.backend == 'hf':
                from .local_hf import pin_model
                pin_model(model)
        with self.db.session(immediate=True) as s:
            return self._create_run(s, req)

    def _create_run(self, s, req):
        """Create jobs inside the caller transaction; model artifacts must already be pinned."""
        revision = require(s, Revision, req.revision_id)
        definition = revision.definition
        defaults = definition['source'].get('generation_config', {})
        for model in req.models:
            model.endpoint = provider_endpoint(model.backend, model.name, model.endpoint) if model.backend not in {'fake', 'hf'} else None
            if model.backend != 'fake':
                model.parameters = {**defaults, **model.parameters}
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
        run = Run(id=uid(), revision_id=revision.id, parent_id=req.parent_id, config=req.model_dump(),
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
                    run.audit = None
                    for old_id in (run.metric_jobs or {}).values():
                        link = s.get(RunJob, (run.id, old_id))
                        if link: s.delete(link)
                    run.metric_jobs = {}
                    if run.report_job:
                        link = s.get(RunJob, (run.id, run.report_job))
                        if link: s.delete(link)
                        run.report_job = None
                    jobs = s.scalars(select(Job).join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run_id)).all()
                    known = {j.id for j in jobs}
                    if job_ids and set(job_ids) - known:
                        raise ValueError('Job does not belong to this run')
                    for job in jobs:
                        if job.status == 'failed' and (job_ids is None or job.id in job_ids):
                            job.status, job.error = 'pending', None
            else:
                raise ValueError('Unknown action')
            return {'status': run.status}

    def schedule(self, run_ids=None):
        """Idempotently advance all active pipelines from durable checkpoints."""
        with self.db.session(immediate=True) as s:
            query = select(Run).where(Run.status == 'active')
            if run_ids is not None:
                query = query.where(Run.id.in_(run_ids))
            for run in s.scalars(query).all():
                definition = require(s, Revision, run.revision_id).definition
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
                if not all(stage_done(i) for i in items):
                    continue
                metric_jobs = {}
                for model in run.config['models']:
                    scores = [s.get(Job, i.score_id) for i in items if i.model['name'] == model['name'] and i.score_id]
                    scores = [score for score in scores if score.status == 'succeeded']
                    metric = job_for(s, run.id, 'metrics', {'components': definition['metrics'],
                        'scores': [score.result for score in scores], 'score_ids': [score.id for score in scores]})
                    metric_jobs[model['name']] = metric.id
                run.metric_jobs = metric_jobs
                if any(s.get(Job, key).status not in terminal for key in metric_jobs.values()):
                    continue
                from .investigations import schedule_investigations
                investigations = schedule_investigations(s, run, items, definition, job_for)
                if run.audit is None:
                    run.audit = self._audit(s, run, items, definition['judge'])
                for item in items:
                    if not item.diagnostic_id:
                        continue
                    diag = s.get(Job, item.diagnostic_id)
                    score = s.get(Job, item.score_id)
                    if diag.status != 'succeeded' or not definition['judge']['enabled']:
                        continue
                    audited = item.id in run.audit['selected']
                    if audited or (score.result['value'] < 1 and diag.result['unresolved']):
                        gen = s.get(Job, item.generation_id)
                        evidence = {'sample': item.sample, 'messages': item.messages, 'response': gen.result['response'],
                            'score': score.result, 'diagnostic': diag.result}
                        job = job_for(s, run.id, 'judge', {'config': definition['judge'], 'evidence': evidence,
                            'dimensions': ['semantic_correct', 'format_compliance', 'instruction_adequacy', 'scorer_correct'] if audited else diag.result['unresolved'],
                            'diagnostic_id': diag.id, 'purpose': 'audit' if audited else 'unresolved'})
                        item.judge_id = job.id
                s.flush()
                if any(j.status not in terminal for j in investigations):
                    continue
                if all(not i.judge_id or s.get(Job, i.judge_id).status in terminal for i in items):
                    if definition['judge']['enabled']:
                        if run.report_job is None:
                            evidence = self._report_evidence(s, run, items)
                            report = job_for(s, run.id, 'report', {'config': definition['judge'], 'evidence': evidence})
                            run.report_job = report.id
                        report = s.get(Job, run.report_job)
                        if report.status not in terminal:
                            continue
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
            chosen = rng.sample(ids, n)
            for key in chosen:
                selected[key] = {'probability': n/len(ids), 'model': model, 'stratum': outcome}
            strata.append({'model': model, 'outcome': outcome, 'population': len(ids), 'sample_size': n, 'probability': n/len(ids)})
        return {'seed': config['seed'], 'selected': selected, 'strata': strata}

    def _report_evidence(self, s, run, items):
        rows = []
        for item in items:
            rows.append({'example_id': item.example_id, 'model': item.model['name'],
                'sample': item.sample, 'messages': item.messages,
                'response': s.get(Job, item.generation_id).result,
                'score': s.get(Job, item.score_id).result if item.score_id else None,
                'diagnostic': s.get(Job, item.diagnostic_id).result if item.diagnostic_id else None,
                'judge': s.get(Job, item.judge_id).result if item.judge_id else None})
        return {'revision_id': run.revision_id, 'manifest': run.manifest, 'audit': run.audit,
            'metrics': {model: s.get(Job, key).result for model, key in (run.metric_jobs or {}).items()},
            'investigations': [j.result for j in s.scalars(select(Job).join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run.id, Job.kind == 'investigation')).all()],
            'total_results': len(rows), 'included_results': len(rows[:200]), 'results': rows[:200],
            'note': 'Results truncated to first 200 when necessary. Audit is stratified, not a full independent rescore.'}

    def evaluations(self, summary=False):
        with self.db.session() as s:
            result = []
            for ev in s.scalars(select(Evaluation).order_by(Evaluation.created.desc())).all():
                revisions = s.scalars(select(Revision).where(Revision.evaluation_id == ev.id).order_by(Revision.created.desc())).all()
                result.append({'id': ev.id, 'name': ev.name, 'revisions': [{'id': r.id, 'parent_id': r.parent_id, 'created': r.created,
                    'definition': ({**{k:v for k,v in r.definition.items() if k != 'examples'},
                        'example_count': len(r.definition['examples'])} if summary else r.definition)} for r in revisions]})
            return result

    def runs(self):
        with self.db.session() as s:
            return [{'id': r.id, 'revision_id': r.revision_id, 'parent_id': r.parent_id, 'status': r.status,
                'created': r.created, 'manifest': r.manifest, 'config': r.config} for r in s.scalars(select(Run).order_by(Run.created.desc())).all()]

    def results(self, run_id):
        with self.db.session() as s:
            run = require(s, Run, run_id)
            definition = require(s, Revision, run.revision_id).definition
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
                judge = s.get(Job, item.judge_id) if item.judge_id else None
                disagreements = []
                if diag and diag.result and judge and judge.result:
                    for dimension in ('semantic_correct', 'format_compliance', 'instruction_adequacy', 'scorer_correct'):
                        a, b = diag.result['finding'][dimension], judge.result[dimension]
                        if a is not None and b is not None and a != b:
                            disagreements.append(dimension)
                rows.append({'disagreements': disagreements, 'id': item.id, 'example_id': item.example_id, 'model': item.model, 'repeat': item.repeat,
                    'sample': item.sample, 'messages': item.messages, 'reused': item.reused,
                    'generation': artifact(item.generation_id), 'score': artifact(item.score_id),
                    'diagnostic': artifact(item.diagnostic_id), 'judge': artifact(item.judge_id),
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
                judged = [r['judge']['result']['semantic_correct'] for r in model_rows if r['judge'] and r['judge']['result'] and r['judge']['result']['semantic_correct'] is not None]
                summaries.append({'model': model['name'], 'total': len(model_rows), 'scored': len(scores),
                    'coverage': len(scores)/len(model_rows), 'metrics': metrics, 'metric_errors': metric_errors,
                    'full_evaluation': not run.manifest['partial'] and len(scores) == len(model_rows),
                    'judge_correct': sum(v is True for v in judged), 'judge_assessed': len(judged),
                    'disagreements': sum(bool(r['disagreements']) for r in model_rows),
                    'semantic_correct': sum(v is True for v in known), 'semantic_known': len(known), 'labels': dict(labels),
                    'reused': sum(r['reused'] for r in model_rows),
                    'truncated': sum(bool(r['generation']['result'] and r['generation']['result'].get('metadata', {}).get('finish_reason') == 'length') for r in model_rows)})
            return {'run': {'id': run.id, 'revision_id': run.revision_id, 'status': run.status, 'manifest': run.manifest,
                'config': run.config, 'audit': run.audit}, 'definition': definition, 'items': rows, 'summaries': summaries,
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
