"""Deterministic failure-review budgets, including audits and retained attempts."""
import random
from sqlalchemy import select
from .db import Job, Attempt


def select_failures(session, items, audit, limit, preserve_started=False):
    attempted = set()
    if preserve_started:
        job_ids = [item.inspector_id for item in items if item.inspector_id]
        if job_ids:
            attempted = set(session.scalars(select(Attempt.job_id).where(Attempt.job_id.in_(job_ids)).distinct()))
    groups = {}
    for item in items:
        if not item.score_id or not item.diagnostic_id:
            continue
        score, diag = session.get(Job,item.score_id), session.get(Job,item.diagnostic_id)
        if score.status != 'succeeded' or score.result['value'] >= 1 or diag.status != 'succeeded':
            continue
        if item.id in audit['selected'] or diag.result['unresolved']:
            groups.setdefault(item.model['name'],[]).append(item)
    selected, summaries = [], []
    for model, rows in sorted(groups.items()):
        rows.sort(key=lambda item:(item.example_id,item.repeat))
        retained = [item for item in rows if preserve_started and item.inspector_id and
            (item.inspector_id in attempted or session.get(Job,item.inspector_id).status in {'running','succeeded','failed'})]
        chosen = {item.id for item in retained}
        rng = random.Random(f"{audit['seed']}:failure-review:{model}")
        for pool in ([item for item in rows if item.id in audit['selected'] and item.id not in chosen],
                     [item for item in rows if item.id not in audit['selected'] and item.id not in chosen]):
            n = len(pool) if limit is None else min(len(pool),max(0,limit-len(chosen)))
            chosen.update(item.id for item in rng.sample(pool,n))
        selected.extend(sorted(chosen))
        summaries.append({'model':model,'eligible_failures':len(rows),'selected':len(chosen),
            'retained_started':len(retained),'limit':limit,'over_limit':limit is not None and len(chosen)>limit})
    return {'limit_per_model':limit,'selected':selected,'groups':summaries,
        'selection_method':'retained started work, then seeded sampling' if preserve_started else 'seeded sampling; audits included'}
