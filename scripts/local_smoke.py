"""Verify pinned local models before starting benchmark generation."""
import argparse
import json
import time
from pathlib import Path
from kineticeval.schemas import ModelSpec
from kineticeval.local_hf import pin_model, LocalModel

parser=argparse.ArgumentParser()
parser.add_argument('model', choices=['mimir','gemma','qwen'])
args=parser.parse_args()
configs={
    'mimir': {'name':'danish-foundation-models/DFM-Mimir-v1.8','local_path':'/work/dfm/euroeval-runs/mimir-v1.8-3800000-validation/model'},
    'gemma': {'name':'google/gemma-4-E2B-it','revision':'4adf15748de9e1a0097b6274cee9c15836c41c37'},
    'qwen': {'name':'Qwen/Qwen3.5-2B','revision':'15852e8c16360a2fea060d615a32b45270f8a8fc'},
}
model=pin_model(ModelSpec(**configs[args.model], backend='hf', parameters={'temperature':0,'max_tokens':64})).model_dump()
print('Pinned model:',model['name'],model.get('revision') or model['artifact_hash'],flush=True)
started=time.time()
engine=LocalModel(model)
print('Loaded in',round(time.time()-started,1),'seconds',flush=True)
payload={'model':model,'messages':[{'role':'user','content':'What is 2 + 2?\nA) 4\nB) 5\nC) 6\nD) 7\nRespond with exactly ANSWER: A, ANSWER: B, ANSWER: C, or ANSWER: D.'}]}
results=engine.generate_batch([payload,payload])
path=Path('artifacts')/f'{args.model}-smoke.json'
path.write_text(json.dumps({'model':model,'results':results,'elapsed':time.time()-started},indent=2))
print(json.dumps({'model':model['name'],'responses':[r['response'] for r in results],'metadata':results[0]['metadata'],'elapsed':time.time()-started}),flush=True)
