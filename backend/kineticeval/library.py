"""Named model configurations and collections independent of run lifetimes."""
import time
from sqlalchemy import select
from .db import ModelConfiguration, ResultCollection, Run
from .schemas import ModelSpec
from .service import require, uid


def model_dict(row):
    return {k:getattr(row,k) for k in ('id','name','spec','created','updated')}


def models(db):
    with db.session() as s:
        return [model_dict(r) for r in s.scalars(select(ModelConfiguration).order_by(ModelConfiguration.created)).all()]


def save_model(db, body, model_id=None):
    name=body.get('name','').strip()
    if not name:raise ValueError('Name the model configuration')
    spec=ModelSpec.model_validate(body['spec']).model_dump()
    with db.session(immediate=True) as s:
        row=require(s,ModelConfiguration,model_id) if model_id else ModelConfiguration(id=uid(),created=time.time())
        row.name,row.spec,row.updated=name,spec,time.time()
        s.add(row);s.flush()
        return model_dict(row)


def result_dict(row):
    return {k:getattr(row,k) for k in ('id','name','description','run_ids','comparison','created')}


def results(db):
    with db.session() as s:
        return [result_dict(r) for r in s.scalars(select(ResultCollection).order_by(ResultCollection.created.desc())).all()]


def save_result(db,body,result_id=None):
    name=body.get('name','').strip()
    if not name:raise ValueError('Name the Result')
    ids=list(dict.fromkeys(body.get('run_ids',[])))
    with db.session(immediate=True) as s:
        for key in ids:require(s,Run,key)
        row=require(s,ResultCollection,result_id) if result_id else ResultCollection(id=uid(),created=time.time())
        row.name,row.description,row.run_ids=name,body.get('description',''),ids
        row.comparison={**body.get('comparison',{}),**({'launch_fingerprint':row.comparison['launch_fingerprint']} if row.comparison and 'launch_fingerprint' in row.comparison else {})}
        s.add(row);s.flush()
        return result_dict(row)


def launch(service, body):
    """Resolve models outside the write lock; publish all runs atomically."""
    import copy
    from .schemas import RunRequest
    from .core import identity
    key=body.get('launch_key','').strip()
    if not key:raise ValueError('A launch key is required')
    name=body.get('name','').strip()
    if not name:raise ValueError('Name the Result')
    entries=body.get('runs',[])
    if not entries:raise ValueError('Select at least one evaluation')
    fingerprint=identity(body)
    with service.db.session() as s:
        existing=s.scalar(select(ResultCollection).where(ResultCollection.launch_key==key))
        if existing:
            if existing.comparison.get('launch_fingerprint')!=fingerprint:raise ValueError('Launch key already used with different settings')
            return result_dict(existing)
    requests=[]
    for entry in entries:
        request=RunRequest.model_validate(copy.deepcopy(entry))
        for model in request.models:
            if model.backend=='hf':
                from .local_hf import pin_model
                pin_model(model)
        requests.append(request)
    with service.db.session(immediate=True) as s:
        existing=s.scalar(select(ResultCollection).where(ResultCollection.launch_key==key))
        if existing:
            if existing.comparison.get('launch_fingerprint')!=fingerprint:raise ValueError('Launch key already used with different settings')
            return result_dict(existing)
        ids=[service._create_run(s,request)['id'] for request in requests]
        row=ResultCollection(id=uid(),name=name,description=body.get('description',''),run_ids=ids,
            comparison={'launch_fingerprint':fingerprint},launch_key=key,created=time.time())
        s.add(row);s.flush()
        return result_dict(row)
