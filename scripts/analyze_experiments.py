"""Watch approved experiment runs; no generation or approval is performed here."""
import asyncio
import getpass
import json
import os
from sqlalchemy import select, func
from kineticeval.db import Database, Experiment, Run, Item, Job
from kineticeval.service import Service
from kineticeval.worker import Worker

async def main():
    os.environ.setdefault('UCLOUD_BASE_URL','https://ai.cloud.sdu.dk/v1')
    if not os.environ.get('UCLOUD_API_KEY'):
        os.environ['UCLOUD_API_KEY']=getpass.getpass('Inference key for approved experiment audits: ')
    db=Database(os.environ.get('KINETICEVAL_DB','kineticeval.db'));service=Service(db);worker=Worker(db)
    while True:
        with db.session() as s:
            ids=[rid for e in s.scalars(select(Experiment).where(Experiment.status=='approved')).all() for rid in e.runs.values()]
            active=s.scalars(select(Run.id).where(Run.id.in_(ids),Run.status=='active')).all()
        for run_id in active:
            with db.session() as s:
                remaining=s.scalar(select(func.count()).select_from(Item).join(Job,Job.id==Item.generation_id)
                    .where(Item.run_id==run_id,Job.status.in_(['pending','running'])))
            if remaining:continue
            for kind in ['scoring','diagnostic','metrics','investigation','inspector','report']:
                service.schedule(run_ids=[run_id])
                while claims:=db.claim_many(kind=kind,run_id=run_id,limit=4,lease_seconds=600):
                    if kind in {'scoring','diagnostic','metrics'}:await worker.execute_batch(claims)
                    else:await asyncio.gather(*(worker.execute(c) for c in claims))
            service.schedule(run_ids=[run_id])
            with db.session() as s:status=s.get(Run,run_id).status
            print(json.dumps({'run':run_id,'status':status}),flush=True)
        await asyncio.sleep(5)

if __name__=='__main__':asyncio.run(main())
