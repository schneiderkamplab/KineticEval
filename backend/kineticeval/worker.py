"""Separate resumable worker: python -m kineticeval.worker."""
import argparse
import asyncio
import contextlib
import os
from .db import Database
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
    def __init__(self, db):
        self.db, self.service = db, Service(db)

    async def step(self):
        self.service.schedule()
        claim = self.db.claim()
        if claim is None:
            return False
        await self.execute(claim)
        self.service.schedule()
        return True

    async def execute(self, claim):
        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                if not self.db.heartbeat(claim['id'], claim['token']):
                    return
        task = asyncio.create_task(heartbeat())
        try:
            result = await self.compute(claim)
            self.db.finish(claim['id'], claim['token'], result=result)
        except Exception as exc:
            self.db.finish(claim['id'], claim['token'], error=safe_error(exc))
        finally:
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
        if claim['kind'] == 'generation': result = await providers.generate(payload)
        elif claim['kind'] == 'scoring': result = await components.score(payload)
        elif claim['kind'] == 'metrics': result = components.aggregate(payload)
        elif claim['kind'] == 'diagnostic': result = components.diagnose(payload)
        elif claim['kind'] == 'judge': result = await providers.judge(payload)
        elif claim['kind'] in {'report', 'investigation'}: result = await providers.judge(payload, report=True)
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
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        parser.error('--concurrency must be between 1 and 32')
    worker = Worker(Database(args.db))
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
