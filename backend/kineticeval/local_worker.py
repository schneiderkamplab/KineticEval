"""Model-grouped GPU worker. Set CUDA_VISIBLE_DEVICES before launching."""
import argparse
import asyncio
import contextlib
import json
import gc
import signal
import os
import sys
from pathlib import Path
import time
from .db import Database
from .local_hf import LocalModel, runtime_identity, resolve_source
from .worker import safe_error


async def run(db, model_name=None, batch_size=16, memory_fraction=.20, watch=False, engine_kind="transformers", tp=1, runtime_id=None, adaptive=False):
    engine = None
    load_failures = {}
    completed = 0
    started = time.time()
    stopping = False
    def stop():
        nonlocal stopping
        stopping = True
    loop = asyncio.get_running_loop()
    if watch:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop)
    while not stopping:
        claims = db.claim_many(kind='generation', model=model_name, limit=min(batch_size,int(os.environ.get('KINETICEVAL_MAX_BATCH','1024'))), lease_seconds=600, backend='hf', preferred_spec=engine.spec if engine is not None else None, runtime_id=runtime_id)
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
                    selected_engine = engine_kind
                    if selected_engine == 'auto':
                        from transformers import AutoConfig
                        source = await asyncio.to_thread(resolve_source, spec)
                        config = await asyncio.to_thread(AutoConfig.from_pretrained, source)
                        try:
                            from vllm import ModelRegistry
                            supported = set(ModelRegistry.get_supported_archs())
                            selected_engine = 'vllm' if any(a in supported for a in config.architectures or []) else 'transformers'
                        except ImportError:
                            selected_engine = 'transformers'
                    if selected_engine == 'vllm':
                        from .vllm_engine import VLLMModel
                        engine = await asyncio.to_thread(VLLMModel, spec, memory_fraction, tp, 1024)
                        if adaptive:
                            batch_size = 1024
                    else:
                        if tp != 1:
                            raise ValueError('This model has no supported vLLM TP implementation; refusing to substitute pipeline parallelism')
                        if adaptive:
                            engine = await asyncio.to_thread(LocalModel, spec, 'cuda:0', memory_fraction, True)
                        else:
                            engine = await asyncio.to_thread(LocalModel, spec, 'cuda:0', memory_fraction)
                except Exception as exc:
                    load_failures[identity] = safe_error(exc)
                    raise
                print(json.dumps({'event':'model_loaded','model':spec['name']}), flush=True)
            engine.spec = spec
            db.activity(claims, 'generating')
            generation_started = time.monotonic()
            async def generate_adaptive():
                import torch
                from .gpu_policy import next_batch_size
                nonlocal batch_size
                remaining = [c['payload'] for c in claims]
                outputs = []
                while remaining:
                    chunk = remaining[:batch_size]
                    torch.cuda.reset_peak_memory_stats()
                    baseline = torch.cuda.memory_allocated()
                    try:
                        result = await asyncio.to_thread(engine.generate_batch, chunk)
                    except torch.OutOfMemoryError:
                        gc.collect()
                        torch.cuda.empty_cache()
                        if batch_size <= 1:
                            raise
                        batch_size = max(1, batch_size // 2)
                        continue
                    peak = torch.cuda.max_memory_allocated()
                    for output in result:
                        output['metadata'].update(batch_size=len(chunk), peak_memory_bytes=peak)
                    outputs.extend(result)
                    remaining = remaining[len(chunk):]
                    budget = torch.cuda.get_device_properties(0).total_memory * memory_fraction
                    batch_size = next_batch_size(len(chunk), torch.cuda.max_memory_allocated(), baseline, budget)
                return outputs
            if adaptive and selected_engine == 'transformers':
                outputs = await generate_adaptive()
            else:
                outputs = await asyncio.to_thread(engine.generate_batch, [c['payload'] for c in claims])
            for output in outputs:
                output['metadata'].setdefault('batch_size', len(claims))
                output['metadata']['visible_gpus'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                output['metadata']['batch_generation_seconds'] = time.monotonic() - generation_started
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

    if engine is not None and hasattr(engine, 'close'):
        await asyncio.to_thread(engine.close)


def configure_cuda_toolkit():
    os.environ['PATH'] = str(Path(sys.base_prefix)/'bin') + os.pathsep + os.environ.get('PATH','')
    if os.environ.get('CUDA_HOME'):
        return
    base = Path(sys.base_prefix).resolve()
    target = base/'targets'/'x86_64-linux'
    if (base/'bin'/'nvcc').exists() and (target/'include'/'cuda.h').exists():
        # Conda splits compiler binaries from headers. Present a complete toolkit
        # without modifying the shared environment or mixing pip CUDA versions.
        toolkit = (Path('artifacts')/('cuda-toolkit-' + base.name)).resolve()
        toolkit.mkdir(parents=True, exist_ok=True)
        for name, source in {'bin':base/'bin','include':target/'include','lib':target/'lib',
                'lib64':target/'lib','nvvm':base/'nvvm','targets':base/'targets'}.items():
            link = toolkit/name
            if not link.is_symlink():
                link.symlink_to(source, target_is_directory=True)
        os.environ['CUDA_HOME'] = str(toolkit)
        os.environ['PATH'] = str(toolkit/'bin') + os.pathsep + str(base/'nvvm'/'bin') + os.pathsep + os.environ['PATH']
        return
    candidates = []
    candidates.extend(candidate for entry in sys.path for candidate in
        (Path(entry)/'nvidia'/'cu13', Path(entry)/'nvidia'/'cuda_nvcc'))
    for candidate in candidates:
        if (candidate/'bin'/'nvcc').exists() and (candidate/'include'/'cuda.h').exists():
            os.environ['CUDA_HOME'] = str(candidate)
            os.environ['PATH'] = str(candidate/'bin') + os.pathsep + os.environ.get('PATH','')
            return


def main():
    configure_cuda_toolkit()
    parser=argparse.ArgumentParser()
    parser.add_argument('--db',default='kineticeval.db')
    parser.add_argument('--model',help='Optional restriction; by default consume all pending local HF models')
    parser.add_argument('--engine',choices=['auto','vllm','transformers'],default='auto')
    parser.add_argument('--tp',type=int,default=1)
    parser.add_argument('--runtime-id')
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--memory-fraction',type=float,default=.20)
    parser.add_argument('--watch',action='store_true',help='Wait for newly approved work without loading weights while idle')
    args=parser.parse_args()
    if not 1 <= args.batch_size <= 256 or not 0 < args.memory_fraction <= .95:
        parser.error('Batch size must be 1–256; memory fraction must be in (0, .95]')
    asyncio.run(run(Database(args.db),args.model,args.batch_size,args.memory_fraction,args.watch,args.engine,args.tp,args.runtime_id,args.adaptive))


if __name__=='__main__': main()
