import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
from kineticeval.inspect_adapter import import_task

parser=argparse.ArgumentParser()
parser.add_argument('benchmark',choices=['winogrande','hellaswag','mmlu'])
args=parser.parse_args()
async def main():
    name=args.benchmark
    definition=await import_task(f'kineticeval.benchmark_tasks:{name}_qa',limit=100000)
    definition.name={'winogrande':'WinoGrande','hellaswag':'HellaSwag','mmlu':'MMLU'}[name]+' · zero-shot generated answers'
    definition.description='Full labeled evaluation split, using Inspect Evals prompts and choice scoring. Native model chat templates; thinking disabled; greedy decoding; 64 output tokens.'
    definition.expected_format='answer_letter'
    definition.heuristics[0].version='2'
    definition.source['inspect_evals_version']=importlib.metadata.version('inspect-evals')
    definition.source['benchmark']=name
    definition.source['protocol']={'fewshot':0,'max_tokens':64,'thinking':False,'score_type':'generated-answer accuracy, not likelihood ranking'}
    module=__import__(f'inspect_evals.{name}.{name}',fromlist=['_'])
    definition.source['upstream_revisions']={key:getattr(module,key) for key in dir(module) if key.endswith('_REVISION')}
    path=Path('artifacts')/f'{name}-definition.json'
    path.write_text(definition.model_dump_json(indent=2))
    print(json.dumps({'benchmark':name,'examples':len(definition.examples),'subsets':len({e.subset for e in definition.examples}),'path':str(path),'source':definition.source['upstream_revisions']}),flush=True)
asyncio.run(main())
