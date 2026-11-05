"""Model-grouped GPU worker. Set CUDA_VISIBLE_DEVICES before launching."""
import argparse
import asyncio
import contextlib
import json
import gc
import time
from .db import Database
from .local_hf import LocalModel, runtime_identity
from .worker import safe_error


async def run(db, model_name=None, batch_size=16, memory_fraction=.20, watch=False):
    engine = None
    load_failures = {}
    completed = 0
    started = time.time()
    while True:
        claims = db.claim_many(kind='generation', model=model_name, limit=batch_size, lease_seconds=600, backend='hf', preferred_spec=engine.spec if engine is not None else None)
        if not claims:
            load_failures.clear()
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
            if engine is not None and runtime_identity(engine.spec) != runtime_identity(spec):
                db.activity(claims, 'unloading_model')
                old_name = engine.spec['name']
                del engine
                engine = None
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(json.dumps({'event':'model_unloaded','model':old_name}), flush=True)
            if engine is None:
                db.activity(claims, 'preparing_model')
                identity = json.dumps(runtime_identity(spec), sort_keys=True)
                if identity in load_failures:
                    raise RuntimeError('Model could not be loaded for this queue pass: ' + load_failures[identity])
                try:
                    engine = await asyncio.to_thread(LocalModel, spec, 'cuda:0', memory_fraction)
                except Exception as exc:
                    load_failures[identity] = safe_error(exc)
                    raise
                print(json.dumps({'event':'model_loaded','model':spec['name']}), flush=True)
            engine.spec = spec
            db.activity(claims, 'generating')
            outputs = await asyncio.to_thread(engine.generate_batch, [c['payload'] for c in claims])
            if len(outputs) != len(claims):
                raise ValueError('Model returned an incomplete batch')
            db.finish_many([{'id': claim['id'], 'token': claim['token'], 'result': output}
                for claim, output in zip(claims, outputs)])
            completed += len(claims)
            print(json.dumps({'model':spec['name'], 'completed':completed,'elapsed':round(time.time()-started,1),
                'batch':len(claims),'output_tokens':sum(r['metadata']['output_tokens'] for r in outputs)}), flush=True)
        except Exception as exc:
            error = safe_error(exc)
            for claim in claims:
                try:
                    db.finish(claim['id'], claim['token'], error=error)
                except ValueError:
                    pass  # An expired/reclaimed lease must never overwrite its replacement.
            print(json.dumps({'model':spec['name'],'error':error}),flush=True)
            # A bad batch must not stop unrelated queued models.
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--db',default='kineticeval.db')
    parser.add_argument('--model',help='Optional restriction; by default consume all pending local HF models')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--memory-fraction',type=float,default=.20)
    parser.add_argument('--watch',action='store_true',help='Wait for newly approved work without loading weights while idle')
    args=parser.parse_args()
    if not 1 <= args.batch_size <= 256 or not 0 < args.memory_fraction <= .4:
        parser.error('Batch size must be 1–256; memory fraction must be in (0, .4]')
    asyncio.run(run(Database(args.db),args.model,args.batch_size,args.memory_fraction,args.watch))


if __name__=='__main__': main()
