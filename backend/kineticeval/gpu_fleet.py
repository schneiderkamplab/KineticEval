"""Manage GPU groups and resident model workers across the local generation queue."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from sqlalchemy import select, func
from .db import Database, Job, RunJob, Run
from .core import identity
from .local_hf import runtime_identity
from .gpu_policy import memory_budget, tensor_parallel_size


def pending_groups(db):
    with db.session() as session:
        rows = session.execute(select(Job.payload['model'], func.count(func.distinct(Job.id)), func.min(Job.created))
            .join(RunJob, RunJob.job_id == Job.id).join(Run, Run.id == RunJob.run_id)
            .where(Run.status == 'active', Job.status == 'pending', Job.kind == 'generation',
                Job.payload['model']['backend'].as_string() == 'hf')
            .group_by(Job.payload['model']).order_by(func.min(Job.created))).all()
    groups = {}
    for spec, count, created in rows:
        key = identity(runtime_identity(spec))
        group = groups.setdefault(key, {'id':key,'spec':spec,'count':0,'created':created})
        group['count'] += count
    return list(groups.values())


def weight_bytes(spec):
    width = 4 if spec.get('dtype') == 'float32' else 2
    if spec.get('local_path'):
        root = Path(spec['local_path'])
        files = list(root.glob('*.safetensors')) or list(root.glob('pytorch_model*.bin'))
        if not files:
            raise ValueError('No local weight files found')
        return sum(file.stat().st_size for file in files)
    from huggingface_hub import HfApi
    info = HfApi().model_info(spec['name'], revision=spec.get('revision'), files_metadata=True)
    if info.safetensors and info.safetensors.total:
        return info.safetensors.total * width
    files = [file for file in info.siblings if file.rfilename.endswith('.safetensors')]
    if not files:
        files = [file for file in info.siblings if file.rfilename.endswith('.bin') and 'pytorch_model' in file.rfilename]
    if not files or any(file.size is None for file in files):
        raise ValueError('Cannot determine model size safely')
    return sum(file.size for file in files)


def gpu_memory():
    data = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'], text=True)
    return {int(index): (int(free)*1024**2, int(total)*1024**2)
        for index, free, total in (line.split(',') for line in data.strip().splitlines())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='kineticeval.db')
    parser.add_argument('--gpus', help='Comma-separated allowed GPU indices; defaults to all visible GPUs')
    parser.add_argument('--cap-gib', type=float, default=80)
    args = parser.parse_args()
    if args.cap_gib <= 0:
        parser.error('--cap-gib must be positive')
    db = Database(args.db)
    allowed = set(map(int,args.gpus.split(','))) if args.gpus else set(gpu_memory())
    if not allowed or not allowed.issubset(gpu_memory()):
        parser.error('--gpus must identify available GPUs')
    children, sizes, retry_after = [], {}, {}
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    Path('artifacts').mkdir(exist_ok=True)
    try:
        while not stopping:
            groups = pending_groups(db)
            requested = {group['id'] for group in groups}
            for child in list(children):
                if child['process'].poll() is not None:
                    children.remove(child)
                    retry_after[child['id']] = time.monotonic()+30
                elif child['id'] not in requested and any(group['id'] != child['id'] for group in groups):
                    # SIGTERM asks the worker to finish its batch before exiting.
                    child['process'].terminate()
            occupied = {gpu for child in children for gpu in child['gpus']}
            memory = gpu_memory()
            free = sorted(allowed - occupied, key=lambda gpu: memory[gpu][0], reverse=True)
            for group in groups:
                if time.monotonic() < retry_after.get(group['id'],0):
                    continue
                try:
                    if group['id'] not in sizes:
                        sizes[group['id']] = weight_bytes(group['spec'])
                    while free and group['count'] > 0:
                        budgets = [memory_budget(*memory[gpu], cap_gib=args.cap_gib) for gpu in free]
                        size = tensor_parallel_size(sizes[group['id']],budgets)
                        if size is None:
                            break
                        gpus = free[:size]
                        # vLLM takes a fraction of total VRAM, not of currently free VRAM.
                        fraction = min(budgets[i]/memory[gpu][1] for i,gpu in enumerate(gpus))
                        env = {**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
                            'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
                            'KINETICEVAL_MAX_BATCH':'256'}
                        command = [sys.executable,'-u','-m','kineticeval.local_worker','--db',args.db,
                            '--runtime-id',group['id'],'--engine','auto','--tp',str(size),
                            '--batch-size','8','--memory-fraction',str(fraction),'--adaptive','--watch']
                        with open(f"artifacts/gpu-{'-'.join(map(str,gpus))}-worker.log",'a') as log:
                            process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        children.append({'id':group['id'],'process':process,'gpus':gpus})
                        print(json.dumps({'event':'worker_started','model':group['spec']['name'],'gpus':gpus,
                            'tp':size,'memory_fraction':fraction,'budget_bytes_per_gpu':min(budgets[:size])}),flush=True)
                        free = free[size:]
                        group['count'] -= 8
                except Exception as exc:
                    from .worker import safe_error
                    print(json.dumps({'event':'dispatch_error','model':group['spec']['name'],'error':safe_error(exc)}),flush=True)
                    retry_after[group['id']] = time.monotonic()+60
            time.sleep(3)
    finally:
        for child in children:
            if child['process'].poll() is None:
                child['process'].terminate()
        for child in children:
            try:
                child['process'].wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(child['process'].pid, signal.SIGKILL)


if __name__ == '__main__':
    main()
