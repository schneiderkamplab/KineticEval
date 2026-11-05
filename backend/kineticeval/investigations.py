"""Automatic, model-independent investigation of repeated evaluation symptoms."""
import random
from collections import defaultdict
from .db import Job, Proposal
from .core import identity


def schedule_investigations(session, run, items, definition, job_for):
    if not definition['inspector']['enabled']:
        return []
    groups = defaultdict(list)
    for item in items:
        generation = session.get(Job, item.generation_id)
        if generation.status != 'succeeded':
            continue
        score = session.get(Job, item.score_id) if item.score_id else None
        diagnostic = session.get(Job, item.diagnostic_id) if item.diagnostic_id else None
        groups[item.model['name']].append((item, generation, score, diagnostic))
    jobs = []
    for model, rows in sorted(groups.items()):
        truncated = [r for r in rows if r[1].result.get('metadata', {}).get('finish_reason') == 'length']
        unresolved = [r for r in rows if r[2] and r[2].result and r[2].result['value'] < 1
                      and r[3] and r[3].result and r[3].result['unresolved']]
        triggers = {name: len(group) for name, group in [('token_limit', truncated), ('unresolved_failure', unresolved)]
                    if len(group) >= 3 and len(group) / len(rows) >= .2}
        if not triggers:
            continue
        # Sample symptoms and controls separately, deterministically; never the first N rows.
        affected = {r[0].id for r in truncated + unresolved}
        rng = random.Random(definition['inspector']['seed'])
        selected = []
        strata = []
        for name, group in [('affected', [r for r in rows if r[0].id in affected]),
                            ('controls', [r for r in rows if r[0].id not in affected])]:
            group.sort(key=lambda r: (r[0].example_id, r[0].repeat))
            sample = rng.sample(group, min(12, len(group)))
            selected.extend(sample)
            strata.append({'name': name, 'population': len(group), 'sample_size': len(sample)})
        evidence = {'purpose': 'automatic_pattern_investigation', 'model': model, 'triggers': triggers,
            'population': len(rows), 'sampling': strata, 'model_config': rows[0][0].model,
            'prompt_template': definition['prompt_template'], 'expected_format': definition['expected_format'],
            'scorer': definition['scorer'], 'heuristics': definition['heuristics'],
            'results': [{'example_id': i.example_id, 'sample': i.sample, 'messages': i.messages,
                'generation': g.result, 'score': sc.result if sc else None,
                'diagnostic': d.result if d else None} for i, g, sc, d in selected],
            'note': 'A symptom-triggered sample, not a full independent rescore. Settings are requested settings, not proof of effective model behavior.'}
        job = job_for(session, run.id, 'investigation', {'config': definition['inspector'], 'evidence': evidence})
        jobs.append(job)
        if job.status == 'succeeded':
            for index, proposal in enumerate(job.result['suggestions']):
                key = identity({'run': run.id, 'investigation': job.id, 'index': index})
                if session.get(Proposal, key) is None:
                    session.add(Proposal(id=key, run_id=run.id, proposal={**proposal, 'origin_job': job.id}))
    return jobs
