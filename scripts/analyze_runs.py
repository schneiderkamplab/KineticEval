"""Resume CPU analysis for explicit run IDs using the pinned Inspect environment."""
import argparse
import asyncio
import json
from pathlib import Path
from kineticeval.db import Database
from kineticeval.service import Service
from kineticeval.worker import Worker

async def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');args=parser.parse_args()
    db=Database('kineticeval.db');service=Service(db);worker=Worker(db)
    for run_id in args.runs:
        service.control(run_id,'retry')
        for kind in ['scoring','diagnostic','metrics','investigation','judge','report']:
            service.schedule(run_ids=[run_id])
            while claims:=db.claim_many(kind=kind,run_id=run_id,limit=32):
                await asyncio.gather(*(worker.execute(claim) for claim in claims))
        service.schedule(run_ids=[run_id])
        result=service.results(run_id)
        Path(f'artifacts/results-{run_id}.json').write_text(json.dumps(result))
        print(json.dumps({'run':run_id,'status':result['run']['status'],'summaries':result['summaries']}),flush=True)
asyncio.run(main())
