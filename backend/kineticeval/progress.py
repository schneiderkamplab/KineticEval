"""Lightweight progress snapshots; counts never load prompts or generation bodies."""
import time
from sqlalchemy import select, func
from .db import Run, Item, Job, RunJob, JobActivity
from .service import require


def snapshot(db, run_id):
    with db.session() as s:
        run = require(s, Run, run_id)
        stages = []
        for kind, column in [('generation', Item.generation_id), ('scoring', Item.score_id), ('diagnostic', Item.diagnostic_id)]:
            rows = s.execute(select(Job.status, func.count(Item.id)).select_from(Item)
                .outerjoin(Job, Job.id == column).where(Item.run_id == run_id).group_by(Job.status)).all()
            counts = {status or 'waiting': count for status, count in rows}
            stages.append({'kind': kind, 'counts': counts, 'total': sum(counts.values()), 'dynamic': False})
        for kind, status, count in s.execute(select(Job.kind, Job.status, func.count(Job.id))
                .join(RunJob, RunJob.job_id == Job.id).where(RunJob.run_id == run_id,
                Job.kind.notin_(['generation', 'scoring', 'diagnostic'])).group_by(Job.kind, Job.status)):
            stage = next((stage for stage in stages if stage['kind'] == kind), None)
            if stage is None:
                stage = {'kind': kind, 'counts': {}, 'total': 0, 'dynamic': True}
                stages.append(stage)
            stage['counts'][status] = count
            stage['total'] += count
        models = []
        for name, status, count in s.execute(select(Job.model, Job.status, func.count(Item.id)).select_from(Item)
                .join(Job, Job.id == Item.generation_id).where(Item.run_id == run_id).group_by(Job.model, Job.status)):
            model = next((model for model in models if model['name'] == name), None)
            if model is None:
                model = {'name': name, 'counts': {}, 'total': 0}
                models.append(model)
            model['counts'][status] = count
            model['total'] += count
        activities = []
        for kind, model, lease, phase, updated in s.execute(select(Job.kind, Job.model, Job.lease, JobActivity.phase, JobActivity.updated)
                .join(RunJob, RunJob.job_id == Job.id)
                .outerjoin(JobActivity, (JobActivity.job_id == Job.id) & (JobActivity.token == Job.token))
                .where(RunJob.run_id == run_id, Job.status == 'running')):
            activity = {'kind': kind, 'model': model, 'phase': phase or kind,
                'stale': lease is None or lease < time.time()}
            if activity not in activities:
                activities.append(activity)
        return {'status': run.status, 'stages': stages, 'models': models, 'activities': activities,
            'reused': s.scalar(select(func.count(Item.id)).where(Item.run_id == run_id, Item.reused.is_(True))),
            'updated': time.time()}
