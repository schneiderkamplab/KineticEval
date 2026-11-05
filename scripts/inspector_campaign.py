"""Resume campaign audits with a credential held only in this process's environment."""
import asyncio
import getpass
import json
import os
from pathlib import Path
from sqlalchemy import select, func
from kineticeval.db import Database, Run, Job, Attempt
from kineticeval.service import Service
from kineticeval.worker import Worker
from kineticeval.providers import UCloud

async def main():
    os.environ['UCLOUD_BASE_URL'] = 'https://ai.cloud.sdu.dk/v1'
    if not os.environ.get('UCLOUD_API_KEY'):
        os.environ['UCLOUD_API_KEY'] = getpass.getpass('Inference key: ')
    models = await UCloud().models()
    model = 'zai-org/GLM-5.3'
    if model not in models:
        raise ValueError('Configured inspector model unavailable')
    path = Path('artifacts/campaign.json')
    manifest = json.loads(path.read_text())
    manifest['inspector_status'] = 'connected; audits running'
    audits = manifest.setdefault('inspected', {})
    db = Database(manifest['db']); service = Service(db); worker = Worker(db)
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(path)
    save()
    for scope in ['pilot', 'full']:
        for benchmark, parent_id in manifest[scope].items():
            key = f'{scope}/{benchmark}'
            while True:
                with db.session() as s:
                    parent = s.get(Run, parent_id)
                    ready = parent.status == 'completed'
                    request = dict(parent.config)
                if ready: break
                await asyncio.sleep(30)
            if key not in audits:
                revision = service.revise(request['revision_id'], {'inspector': {
                    'enabled': True, 'provider': 'ucloud', 'model': model,
                    'endpoint': os.environ['UCLOUD_BASE_URL'], 'max_tokens': 4096,
                    'audit_fraction': .1, 'audit_min': 10, 'audit_max': 100, 'seed': 42}})
                request.update(revision_id=revision['revision_id'], parent_id=parent_id, fresh=False)
                audits[key] = service.create_run(request)['id']
                save()
            run_id = audits[key]
            print(json.dumps({'audit': key, 'run': run_id}), flush=True)
            for kind in ['scoring', 'diagnostic', 'metrics', 'investigation', 'inspector', 'report']:
                service.schedule(run_ids=[run_id])
                done = 0
                while claims := db.claim_many(kind=kind, run_id=run_id, limit=4 if kind in {'inspector','report','investigation'} else 32, lease_seconds=600):
                    if kind in {'scoring','diagnostic','metrics'}:
                        await worker.execute_batch(claims)
                    else:
                        await asyncio.gather(*(worker.execute(c) for c in claims))
                    done += len(claims)
                    print(json.dumps({'audit':key, 'stage':kind, 'processed':done}),flush=True)
            service.schedule(run_ids=[run_id])
            result = service.results(run_id)
            Path(f'artifacts/results-{run_id}.json').write_text(json.dumps(result, ensure_ascii=False))
            print(json.dumps({'audit':key,'status':result['run']['status']}),flush=True)
    manifest['inspector_status'] = 'audit processing finished; inspect run status for failures'
    save()

if __name__ == '__main__':
    asyncio.run(main())
