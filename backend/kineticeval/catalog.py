"""Build the materialized definition database from version-controlled recipes."""
import argparse
import asyncio
import json
import re
import sqlite3
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from .core import identity
from .db import Base, Evaluation, Revision, RevisionLabel, ModelConfiguration
from .schemas import Definition, ModelSpec

CATALOG_MODELS = (Evaluation, Revision, RevisionLabel, ModelConfiguration)


def catalog_path(root, entry):
    path = (root / entry).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Catalog entries must be inside the catalog directory")
    if not path.exists():
        raise ValueError(f"Missing catalog entry: {entry}")
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def export_catalog(database, root):
    """Export recipes/settings, never HF example bodies or execution artifacts."""
    root=Path(root)
    with sqlite3.connect(database) as c:
        c.row_factory=sqlite3.Row
        evaluations=[dict(r) for r in c.execute('SELECT * FROM evaluations')]
        revisions=[dict(r) for r in c.execute('SELECT * FROM evaluation_revisions')]
        labels={r['revision_id']:r['name'] for r in c.execute('SELECT * FROM revision_labels')}
        models=[dict(r) for r in c.execute('SELECT * FROM model_configurations')]
    recipes={}
    folders={r['id']:re.sub(r'[^a-z0-9]+','-',r['name'].lower()).strip('-')+'-'+r['id'][:8] for r in evaluations}
    model_files=[]
    for row in revisions:
        definition=json.loads(row.pop('definition'))
        examples=definition.pop('examples')
        source=definition['source']
        if source.get('kind')=='inspect':
            loader={'kind':'inspect','factory':source['factory'],'kwargs':source.get('kwargs',{}),'limit':source.get('limit',len(examples))}
        elif source.get('kind')=='huggingface':
            loader={'kind':'huggingface',**{k:source.get(k) for k in ('path','config','split','revision','limit','mapping')}}
        else:
            raise ValueError(f"Revision {row['id']} needs a versioned non-HF data recipe before export")
        key=source.get('benchmark') or re.sub(r'[^a-z0-9]+', '-', source.get('factory', source.get('path', 'dataset')).split(':')[-1].lower()).strip('-')+'-'+identity(loader)[:8]
        recipe={**loader,'expected_count':len(examples),'examples_sha256':identity(examples)}
        if key in recipes and recipes[key]!=recipe:
            key+='-'+identity(recipe)[:8]
        recipes[key]=recipe
        row.update(definition=definition,dataset=key,label=labels.get(row['id'],''))
        write_json(root/'evaluations'/folders[row['evaluation_id']]/('revision-'+row['id'][:12]+'.json'),row)
    for key,recipe in recipes.items():write_json(root/'datasets'/(key+'.json'),recipe)
    for row in evaluations:write_json(root/'evaluations'/folders[row['id']]/'evaluation.json',row)
    for row in models:
        row['spec']=json.loads(row['spec'])
        template=row['spec'].get('chat_template')
        if template is not None:
            row['spec'].pop('chat_template')
            relative='templates/'+row['id']+'.jinja'
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(template)
            row['template_file']=relative
        filename=re.sub(r'[^a-z0-9]+','-',row['name'].lower()).strip('-')+'-'+row['id'][:8]+'.json'
        model_files.append('models/'+filename)
        write_json(root/'models'/filename,row)
    write_json(root/'catalog.json',{'format_version':1,'description':'Materialized definitions; generated responses and run history are stored separately.', 'evaluations':['evaluations/'+folder for folder in folders.values()], 'models':model_files, 'datasets':['datasets/'+key+'.json' for key in recipes]})
    return {'evaluations':len(evaluations),'revisions':len(revisions),'datasets':len(recipes),'models':len(models)}


async def load_examples(recipe):
    if recipe['kind']=='inspect':
        from .inspect_adapter import import_task
        definition=await import_task(recipe['factory'],recipe.get('kwargs'),recipe['limit'])
    elif recipe['kind']=='huggingface':
        from .importers import from_huggingface
        definition=from_huggingface(recipe['path'],recipe.get('config'),recipe['split'],recipe['revision'],recipe['limit'],recipe.get('mapping'))
    else:
        raise ValueError('Unsupported dataset recipe; refusing an approximate import')
    examples=[example.model_dump() for example in definition.examples]
    if len(examples)!=recipe['expected_count'] or identity(examples)!=recipe['examples_sha256']:
        raise ValueError('Dataset materialization differs from the pinned catalog count/hash')
    return examples


async def build_catalog(root, output, replace=False):
    root,output=Path(root),Path(output)
    if output.exists() and not replace:
        raise ValueError('Output already exists; build to a new file or explicitly use --replace with services stopped')
    manifest=json.loads((root/'catalog.json').read_text())
    if manifest['format_version']!=1:raise ValueError('Unsupported catalog format')
    data={}
    for path in sorted(catalog_path(root, entry) for entry in manifest['datasets']):
        data[path.stem]=await load_examples(json.loads(path.read_text()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='catalog-build-') as temp:
        staging=Path(temp)/'definitions.db'
        engine=create_engine(f'sqlite:///{staging}').execution_options(schema_translate_map={'definitions':None})
        Base.metadata.create_all(engine,tables=[model.__table__ for model in CATALOG_MODELS])
        counts={'datasets':len(data),'examples':sum(len(rows) for rows in data.values()),'evaluations':0,'revisions':0,'models':0}
        with Session(engine) as s:
            for directory in sorted(catalog_path(root, entry) for entry in manifest['evaluations']):
                if not directory.is_dir():
                    raise ValueError(f"Evaluation entry must be a directory: {directory.name}")
                metadata=json.loads((directory/'evaluation.json').read_text())
                s.add(Evaluation(**metadata));counts['evaluations']+=1
                for path in sorted(directory.glob('*.json')):
                    if path.name=='evaluation.json':continue
                    row=json.loads(path.read_text());dataset=row.pop('dataset');label=row.pop('label','')
                    definition={**row['definition'],'examples':data[dataset]}
                    Definition.model_validate(definition)
                    row['definition']=definition;s.add(Revision(**row));counts['revisions']+=1
                    if label:s.add(RevisionLabel(revision_id=row['id'],name=label))
            for path in sorted(catalog_path(root, entry) for entry in manifest['models']):
                row=json.loads(path.read_text());template=row.pop('template_file',None)
                if template:
                    file=(root/template).resolve()
                    if not file.is_relative_to(root.resolve()):raise ValueError('Template must be inside the catalog')
                    row['spec']['chat_template']=file.read_text()
                ModelSpec.model_validate(row['spec']);s.add(ModelConfiguration(**row));counts['models']+=1
            s.commit()
        engine.dispose()
        staging.replace(output)
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',default='catalog')
    parser.add_argument('--output',default='kineticeval.definitions.db')
    parser.add_argument('--replace',action='store_true')
    parser.add_argument('--export-from',help='Export current definitions to recipes instead of building')
    args=parser.parse_args()
    result=export_catalog(args.export_from,args.catalog) if args.export_from else asyncio.run(build_catalog(args.catalog,args.output,args.replace))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
