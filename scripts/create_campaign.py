"""Queue a reproducible pilot followed by full splits, reusing pilot generations."""
import json
import time
from pathlib import Path
from kineticeval.db import Database
from kineticeval.service import Service,materialize

path=Path('artifacts/campaign.json')
if path.exists():
    raise SystemExit('Campaign manifest exists; resume its runs instead of duplicating them')
models=[json.loads((Path('artifacts')/f'{name}-smoke.json').read_text())['model'] for name in ['mimir','gemma','qwen']]
service=Service(Database('kineticeval.db'))
manifest={'created':time.time(),'db':str(Path('kineticeval.db').resolve()),'models':models,'pilot':{},'full':{},'revisions':{},'inspector_status':'awaiting provider-scoped inference credential'}
for name in ['winogrande','hellaswag','mmlu']:
    definition=json.loads((Path('artifacts')/f'{name}-definition.json').read_text())
    created=service.create_evaluation(definition)
    manifest['revisions'][name]=created
    ids=[r['id'] for r in materialize(definition)][:100]
    run=service.create_run({'revision_id':created['revision_id'],'models':models,'example_ids':ids})
    manifest['pilot'][name]=run['id']
    print('Pilot queued',name,run['id'],flush=True)
path.write_text(json.dumps(manifest,indent=2))
for name in ['winogrande','hellaswag','mmlu']:
    run=service.create_run({'revision_id':manifest['revisions'][name]['revision_id'],'models':models,'parent_id':manifest['pilot'][name]})
    manifest['full'][name]=run['id']
    path.write_text(json.dumps(manifest,indent=2))
    print('Full run queued',name,run['id'],flush=True)
