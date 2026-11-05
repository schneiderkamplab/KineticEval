"""Transactional revision deletion with explicit dependency previews."""
from collections import Counter
from sqlalchemy import select, delete, update
from .db import Evaluation, Revision, RevisionLabel, Run, Item, RunJob, Job, Attempt, JobActivity, Proposal, Experiment, ResultCollection
from .core import identity


def plan(s, evaluation_id, revision_ids):
    revisions=s.scalars(select(Revision).where(Revision.evaluation_id==evaluation_id)).all()
    selected=set(revision_ids)
    if not selected or not selected <= {r.id for r in revisions}:
        raise ValueError('Select existing revisions from this evaluation')
    runs=set(s.scalars(select(Run.id).where(Run.revision_id.in_(selected))))
    jobs=set(s.scalars(select(RunJob.job_id).where(RunJob.run_id.in_(runs))))
    shared=set(s.scalars(select(RunJob.job_id).where(RunJob.job_id.in_(jobs),~RunJob.run_id.in_(runs))))
    exclusive=jobs-shared
    proposals=set(s.scalars(select(Proposal.id).where(Proposal.run_id.in_(runs))))
    experiments={e.id for e in s.scalars(select(Experiment)).all()
        if e.baseline_id in runs or e.proposal_id in proposals or runs.intersection(e.runs.values())}
    items=set(s.scalars(select(Item.id).where(Item.run_id.in_(runs))))
    attempts=set(s.scalars(select(Attempt.id).where(Attempt.job_id.in_(exclusive))))
    memberships={r.id:sorted(runs.intersection(r.run_ids)) for r in s.scalars(select(ResultCollection)).all() if runs.intersection(r.run_ids)}
    ids={'revisions':sorted(selected),'runs':sorted(runs),'items':sorted(items),'jobs':sorted(exclusive),
         'shared_jobs':sorted(shared),'proposals':sorted(proposals),'experiments':sorted(experiments),'attempts':sorted(attempts),'result_memberships':memberships}
    kinds=dict(Counter(s.scalars(select(Job.kind).where(Job.id.in_(exclusive)))))
    counts={k:len(v) for k,v in ids.items()}
    counts['active_runs']=len(s.scalars(select(Run.id).where(Run.id.in_(runs),Run.status=='active')).all())
    return {'counts':counts,'job_types':kinds,'token':identity(ids),'ids':ids}


def preview(db,evaluation_id,revision_ids):
    with db.session() as s:
        result=plan(s,evaluation_id,revision_ids)
        rows=[]
        for revision in s.scalars(select(Revision).where(Revision.evaluation_id==evaluation_id).order_by(Revision.created.desc())).all():
            label=s.get(RevisionLabel,revision.id)
            impact=plan(s,evaluation_id,[revision.id])
            rows.append({'id':revision.id,'name':label.name if label else '', 'created':revision.created,
                         'counts':impact['counts'],'job_types':impact['job_types']})
        return {**{k:v for k,v in result.items() if k!='ids'},'revisions':rows}


def remove(db,evaluation_id,revision_ids,token):
    with db.session(immediate=True) as s:
        result=plan(s,evaluation_id,revision_ids)
        if result['token']!=token:
            raise ValueError('Dependent data changed. Review the refreshed deletion preview before deleting.')
        ids=result['ids']
        for collection_id in ids['result_memberships']:
            collection=s.get(ResultCollection,collection_id)
            collection.run_ids=[key for key in collection.run_ids if key not in set(ids['runs'])]
        # Remove queue consumers and fence exclusive jobs in the same transaction.
        for cls,column,key in [(Experiment,Experiment.id,'experiments'),(Proposal,Proposal.id,'proposals'),
            (Item,Item.id,'items'),(RunJob,RunJob.run_id,'runs'),(Attempt,Attempt.id,'attempts'),
            (JobActivity,JobActivity.job_id,'jobs'),(Job,Job.id,'jobs'),(Run,Run.id,'runs'),
            (RevisionLabel,RevisionLabel.revision_id,'revisions'),(Revision,Revision.id,'revisions')]:
            s.execute(delete(cls).where(column.in_(ids[key])))
        # Surviving independent revisions/runs remain usable without dangling links.
        s.execute(update(Revision).where(Revision.parent_id.in_(ids['revisions'])).values(parent_id=None))
        s.execute(update(Run).where(Run.parent_id.in_(ids['runs'])).values(parent_id=None))
        s.execute(update(Proposal).where(Proposal.revision_id.in_(ids['revisions'])).values(revision_id=None))
        remaining=s.scalar(select(Revision.id).where(Revision.evaluation_id==evaluation_id).limit(1))
        if remaining is None:s.execute(delete(Evaluation).where(Evaluation.id==evaluation_id))
        return {'deleted':result['counts'],'evaluation_deleted':remaining is None}
