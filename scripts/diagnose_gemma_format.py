"""Bounded diagnostic variants; never overwrite the original benchmark results."""
import asyncio
import json
from pathlib import Path
from kineticeval.db import Database
from kineticeval.service import Service
from kineticeval.local_hf import LocalModel
from kineticeval.worker import Worker

async def main():
    db=Database('kineticeval.db'); service=Service(db); worker=Worker(db)
    campaign=json.loads(Path('artifacts/campaign.json').read_text())
    baseline=service.results(campaign['pilot']['winogrande'])
    examples=baseline['run']['manifest']['ids'][:20]
    model=next(m for m in campaign['models'] if m['name']=='google/gemma-4-E2B-it')
    revision=campaign['revisions']['winogrande']['revision_id']
    longer={**model,'parameters':{**model['parameters'],'max_tokens':256}}
    long_run=service.create_run({'revision_id':revision,'models':[longer],'example_ids':examples,'parent_id':baseline['run']['id']})['id']
    clarified=service.revise(revision,{'prompt_template':"{input}\n\nReturn only ANSWER: A or ANSWER: B. Do not explain."})['revision_id']
    clear_run=service.create_run({'revision_id':clarified,'models':[model],'example_ids':examples,'parent_id':baseline['run']['id']})['id']
    manifest={'longer_budget':long_run,'clearer_prompt':clear_run,'baseline':baseline['run']['id'],'examples':examples}
    Path('artifacts/gemma-diagnostics.json').write_text(json.dumps(manifest,indent=2))
    engine=await asyncio.to_thread(LocalModel,longer)
    for run_id,spec in [(long_run,longer),(clear_run,model)]:
        claims=db.claim_many(kind='generation',model=model['name'],limit=20,lease_seconds=600,run_id=run_id)
        engine.spec=spec
        outputs=await asyncio.to_thread(engine.generate_batch,[c['payload'] for c in claims])
        for claim,result in zip(claims,outputs):db.finish(claim['id'],claim['token'],result=result)
        for kind in ['scoring','diagnostic','metrics']:
            service.schedule(run_ids=[run_id])
            while claims:=db.claim_many(kind=kind,run_id=run_id,limit=32):
                await asyncio.gather(*(worker.execute(claim) for claim in claims))
        service.schedule(run_ids=[run_id])
        result=service.results(run_id)
        Path(f'artifacts/results-{run_id}.json').write_text(json.dumps(result))
        print(json.dumps({'run':run_id,'summaries':result['summaries']}),flush=True)
asyncio.run(main())
