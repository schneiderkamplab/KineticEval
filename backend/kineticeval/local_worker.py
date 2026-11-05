"""Model-grouped GPU worker. Set CUDA_VISIBLE_DEVICES before launching."""
import argparse
import asyncio
import contextlib
import json
import time
from .db import Database
from .local_hf import LocalModel
from .worker import safe_error


async def run(db, model_name, batch_size=16, memory_fraction=.20, watch=False):
    engine = None
    completed = 0
    started = time.time()
    while True:
        claims = db.claim_many(kind='generation', model=model_name, limit=batch_size, lease_seconds=600)
        if not claims:
            if watch:
                await asyncio.sleep(5)
                continue
            break
        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                for claim in claims:
                    db.heartbeat(claim['id'], claim['token'], lease_seconds=600)
        task = asyncio.create_task(heartbeat())
        try:
            spec = claims[0]['payload']['model']
            if spec['backend'] != 'hf':
                raise ValueError('Local worker only accepts HF generation requests')
            if engine is None:
                engine = await asyncio.to_thread(LocalModel, spec, 'cuda:0', memory_fraction)
            if engine.spec != spec:
                fixed = lambda value: {k:v for k,v in value.items() if k not in {'parameters', 'chat_template_kwargs'}}
                if fixed(engine.spec) != fixed(spec):
                    raise ValueError('Weight or runtime configuration changed; start a separate worker')
                engine.spec = spec
            outputs = await asyncio.to_thread(engine.generate_batch, [c['payload'] for c in claims])
            db.finish_many([{'id': claim['id'], 'token': claim['token'], 'result': output}
                for claim, output in zip(claims, outputs)])
            completed += len(claims)
            print(json.dumps({'model':model_name, 'completed':completed,'elapsed':round(time.time()-started,1),
                'batch':len(claims),'output_tokens':sum(r['metadata']['output_tokens'] for r in outputs)}), flush=True)
        except Exception as exc:
            error = safe_error(exc)
            for claim in claims:
                db.finish(claim['id'], claim['token'], error=error)
            print(json.dumps({'model':model_name,'error':error}),flush=True)
            raise
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--db',default='kineticeval.db')
    parser.add_argument('--model',required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--memory-fraction',type=float,default=.20)
    parser.add_argument('--watch',action='store_true',help='Wait for newly approved work without loading weights while idle')
    args=parser.parse_args()
    if not 1 <= args.batch_size <= 256 or not 0 < args.memory_fraction <= .4:
        parser.error('Batch size must be 1–256; memory fraction must be in (0, .4]')
    asyncio.run(run(Database(args.db),args.model,args.batch_size,args.memory_fraction,args.watch))


if __name__=='__main__': main()
