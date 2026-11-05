"""Analyze ready runs in stage batches, avoiding per-response full-run scans."""
import asyncio
import json
import time
from pathlib import Path
from sqlalchemy import select, func
from kineticeval.db import Database, Run, Item, Job
from kineticeval.service import Service
from kineticeval.worker import Worker

db=Database('kineticeval.db')
service=Service(db)
worker=Worker(db)
manifest=json.loads(Path('artifacts/campaign.json').read_text())
run_ids=list(manifest['pilot'].values())+list(manifest['full'].values())

async def main():
    while True:
        with db.session() as s:
            active=s.scalars(select(Run.id).where(Run.id.in_(run_ids),Run.status=='active')).all()
        if not active: break
        for run_id in active:
            with db.session() as s:
                remaining=s.scalar(select(func.count()).select_from(Item).join(Job,Job.id==Item.generation_id)
                    .where(Item.run_id==run_id,Job.status.in_(['pending','running'])))
            if remaining: continue
            service.schedule(run_ids=[run_id])
            for kind in ['scoring','diagnostic','metrics','investigation','judge','report']:
                while True:
                    claims=db.claim_many(kind=kind,run_id=run_id,limit=32,lease_seconds=600)
                    if not claims: break
                    await worker.execute_batch(claims) if kind in {'scoring','diagnostic','metrics'} else await asyncio.gather(*(worker.execute(claim) for claim in claims))
                service.schedule(run_ids=[run_id])
            result=service.results(run_id)
            if result['run']['status']!='active':
                output=Path('artifacts')/f'results-{run_id}.json'
                output.write_text(json.dumps(result,ensure_ascii=False))
                print(json.dumps({'run':run_id,'status':result['run']['status'],'summaries':result['summaries']}),flush=True)
        await asyncio.sleep(10)
asyncio.run(main())
