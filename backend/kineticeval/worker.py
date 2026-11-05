"""Separate resumable worker: python -m kineticeval.worker."""
import argparse
import asyncio
import contextlib
import os
import time
from .db import Database, InactiveClaim
from .service import Service
from . import components, providers


def verify_components(payload):
    values = [payload.get('component')] + payload.get('components', [])
    for component in filter(None, values):
        if component['name'].startswith('python:'):
            ref = component['name'].removeprefix('python:')
            if components.fingerprint(ref) != component['config'].get('_fingerprint'):
                raise ValueError('Installed component changed; create a new evaluation revision')


def safe_error(exc):
    message = str(exc)
    for key, value in os.environ.items():
        if value and len(value) > 5 and any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            message = message.replace(value, '[redacted]')
    return f'{type(exc).__name__}: {message[:1500]}'


class Worker:
    def __init__(self, db, analysis_only=False):
        self.analysis_only = analysis_only
        self.schedule_lock = asyncio.Lock()
        self.next_schedule = 0.0
        self.last_analysis_run = None
        self.db, self.service = db, Service(db)

    async def schedule(self):
        if not self.analysis_only:
            self.service.schedule()
            return
        async with self.schedule_lock:
            if time.monotonic() >= self.next_schedule:
                await asyncio.to_thread(self.service.schedule)
                self.next_schedule = time.monotonic() + 2

    async def step(self):
        await self.schedule()
        if self.analysis_only:
            claims = self.db.claim_many(limit=64, kinds=['scoring','diagnostic','metrics'])
            if claims:
                await self.execute_batch(claims)
                return True
        if self.analysis_only:
            from sqlalchemy import select
            from .db import Job, RunJob, Run
            with self.db.session() as session:
                run_ids = list(session.scalars(select(Run.id).join(RunJob, RunJob.run_id == Run.id)
                    .join(Job, Job.id == RunJob.job_id).where(Run.status == 'active', Job.status == 'pending',
                    Job.kind.in_(['inspector','investigation','report'])).distinct().order_by(Run.created.desc())))
            if self.last_analysis_run in run_ids:
                offset = run_ids.index(self.last_analysis_run) + 1
                run_ids = run_ids[offset:] + run_ids[:offset]
            claims = []
            for run_id in run_ids:
                claims = self.db.claim_many(limit=1, run_id=run_id, kinds=['investigation','report'])
                if not claims:
                    claims = self.db.claim_many(limit=1, run_id=run_id, kinds=['inspector'])
                if claims:
                    self.last_analysis_run = run_id
                    break
        else:
            claims = self.db.claim_many(limit=1)
        claim = claims[0] if claims else None
        if claim is None:
            return False
        await self.execute(claim)
        await self.schedule()
        return True

    async def execute(self, claim):
        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                if not self.db.heartbeat(claim['id'], claim['token']):
                    return
        task = asyncio.create_task(heartbeat())
        guard_token = None
        try:
            if claim['kind'] in {'inspector','investigation','report'}:
                guard = lambda: self.db.require_active_claim(claim['id'], claim['token'])
                guard()
                guard_token = providers.inspector_request_guard.set(guard)
            result = await self.compute(claim)
            self.db.finish(claim['id'], claim['token'], result=result)
        except InactiveClaim:
            self.db.abandon_claim(claim['id'], claim['token'])
        except Exception as exc:
            self.db.finish(claim['id'], claim['token'], error=safe_error(exc))
        finally:
            if guard_token is not None:
                providers.inspector_request_guard.reset(guard_token)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return True

    async def execute_batch(self, claims):
        """Compute short CPU stages and persist their results in one transaction."""
        if any(c['kind'] not in {'scoring', 'diagnostic', 'metrics'} for c in claims):
            raise ValueError('Batch execution is limited to CPU analysis stages')
        completions = []
        for claim in claims:
            completion = {'id': claim['id'], 'token': claim['token']}
            try:
                completion['result'] = await self.compute(claim)
            except Exception as exc:
                completion['error'] = safe_error(exc)
            completions.append(completion)
        self.db.finish_many(completions)

    async def compute(self, claim):
        payload = claim['payload']
        verify_components(payload)
        if claim['kind'] == 'generation':
            if payload['model']['backend'] == 'hf':
                result = await providers.generate(payload, progress=lambda phase: self.db.activity([claim], phase))
            else:
                result = await providers.generate(payload)
        elif claim['kind'] == 'scoring': result = await components.score(payload)
        elif claim['kind'] == 'metrics': result = components.aggregate(payload)
        elif claim['kind'] == 'diagnostic': result = components.diagnose(payload)
        elif claim['kind'] == 'inspector': result = await providers.inspector(payload)
        elif claim['kind'] in {'report', 'investigation'}: result = await providers.inspector(payload, report=True)
        else: raise ValueError('Unknown job kind')
        return result

    async def drain(self, max_steps=10000):
        for _ in range(max_steps):
            if not await self.step():
                return
        raise RuntimeError('Worker exceeded drain step limit')


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.environ.get('KINETICEVAL_DB', 'kineticeval.db'))
    parser.add_argument('--once', action='store_true', help='Drain available work then exit')
    parser.add_argument('--concurrency', type=int, default=1, help='Concurrent job handlers (1–32)')
    parser.add_argument('--analysis-only', action='store_true', help='Process scoring, checks, inspectors and reports; leave generation to GPU workers')
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        parser.error('--concurrency must be between 1 and 32')
    worker = Worker(Database(args.db), analysis_only=args.analysis_only)
    async def loop():
        if args.once:
            await worker.drain()
        else:
            while True:
                if not await worker.step():
                    await asyncio.sleep(1)
    await asyncio.gather(*(loop() for _ in range(args.concurrency)))


if __name__ == '__main__':
    asyncio.run(main())
