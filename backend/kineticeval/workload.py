"""Read-only launch preview: exact cache counts and bounded inspection estimates."""
import copy
import math
from collections import Counter
from types import SimpleNamespace
from .core import identity
from .db import Job,Revision
from .schemas import RunRequest,Definition
from .service import require,pin,materialize,render,physical_model,provider_endpoint
from .inspection_policy import select_failures
from .inspector_protocol import protocol_for


def job_key(kind,payload):
    if kind in {'inspector','investigation','report'}:
        payload={**payload,'config':{k:v for k,v in payload['config'].items() if k!='failure_inspector_limit'}}
        payload={**payload,'protocol':protocol_for(payload,report=kind!='inspector')}
    return identity({'kind':kind,'payload':payload,'engine_version':'1'})


def classify(job):
    if job is None:return 'new'
    return {'succeeded':'reusable','pending':'queued','running':'running','failed':'failed'}[job.status]


def preview(service,body):
    entries=body.get('runs',[])
    if not entries:raise ValueError('Select at least one evaluation')
    requests=[RunRequest.model_validate(copy.deepcopy(entry)) for entry in entries]
    pinned={}
    for req in requests:
        for model in req.models:
            if model.backend=='hf':
                from .local_hf import pin_model
                key=identity(physical_model(model.model_dump()))
                if key not in pinned:pinned[key]=pin_model(model.model_copy(deep=True))
                model.revision=pinned[key].revision;model.local_path=pinned[key].local_path;model.artifact_hash=pinned[key].artifact_hash
    groups=[];generation_jobs={};inspection_jobs={};resolved=[];reports=0
    with service.db.session() as s:
        for index,req in enumerate(requests):
            revision=require(s,Revision,req.revision_id)
            definition=copy.deepcopy(revision.definition)
            allowed={'scorer','metrics','heuristics','inspector','expected_format','generation_defaults'}
            if set(req.analysis_overrides)-allowed:raise ValueError('Unsupported run analysis setting')
            overrides=copy.deepcopy(req.analysis_overrides)
            if 'generation_defaults' in overrides:overrides['generation_defaults']={**definition.get('generation_defaults',{}),**overrides['generation_defaults']}
            definition.update(overrides);definition=pin(Definition.model_validate(definition))
            rows=materialize(definition)
            if req.example_ids is not None:
                if not req.example_ids or set(req.example_ids)-{r['id'] for r in rows}:raise ValueError('Select example IDs in the revision sample')
                rows=[r for r in rows if r['id'] in req.example_ids]
            if req.subsets is not None:rows=[r for r in rows if r['subset'] in req.subsets]
            if not rows:raise ValueError('Sampler selected no examples')
            if len({m.name for m in req.models})!=len(req.models):raise ValueError('Model configuration names must be unique')
            defaults={**definition['source'].get('generation_config',{}),**definition.get('generation_defaults',{})}
            items=[];cache={};model_groups={}
            for model in req.models:
                model.endpoint=provider_endpoint(model.backend,model.model_identifier or model.name,model.endpoint) if model.backend not in {'hf','fake'} else None
                if model.backend!='fake':model.parameters={**defaults,**model.parameters}
                type(model).model_validate(model.model_dump())
                counts=Counter();score_counts=Counter();diag_counts=Counter()
                for row in rows:
                    messages=render(definition,row)
                    for repeat in range(req.repeats):
                        payload={'model':physical_model(model.model_dump()),'messages':messages,'repeat':repeat,'fresh':f'preview-fresh-{index}' if req.fresh else None}
                        gen_id=job_key('generation',payload)
                        gen=None if req.fresh else s.get(Job,gen_id)
                        counts[classify(gen)]+=1;generation_jobs[gen_id]=classify(gen)
                        item=SimpleNamespace(id=f'{index}:{model.name}:{row["id"]}:{repeat}',model=model.model_dump(),example_id=row['id'],repeat=repeat,sample=row,messages=messages,generation_id=gen_id,score_id=None,diagnostic_id=None,inspector_id=None)
                        items.append(item)
                        if not gen or gen.status!='succeeded':score_counts['unknown']+=1;diag_counts['unknown']+=1;continue
                        cache[gen.id]=gen
                        evidence={'sample':row,'messages':messages,'response':gen.result['response'],'model':item.model}
                        score_id=job_key('scoring',{**evidence,'component':definition['scorer'],'generation_id':gen.id})
                        score=s.get(Job,score_id);score_counts[classify(score)]+=1
                        if not score or score.status!='succeeded':diag_counts['unknown']+=1;continue
                        item.score_id=score.id;cache[score.id]=score
                        diag_id=job_key('diagnostic',{**evidence,'score':score.result,'score_id':score.id,'components':definition['heuristics'],'expected_format':definition['expected_format']})
                        diag=s.get(Job,diag_id);diag_counts[classify(diag)]+=1
                        if diag and diag.status=='succeeded':item.diagnostic_id=diag.id;cache[diag.id]=diag
                group={'evaluation':definition['name'],'revision_id':revision.id,'model':model.name,'responses':len(rows)*req.repeats,
                    'generation':dict(counts),'scoring':dict(score_counts),'heuristics':dict(diag_counts)}
                groups.append(group);model_groups[model.name]=group
            config=definition['inspector'];reports+=int(config['enabled']);complete=all(i.diagnostic_id for i in items)
            if config['enabled'] and complete:
                audit=service._audit(s,None,items,config)
                limit=config.get('failure_inspector_limit')
                allowed_ids=None if limit is None else set(select_failures(s,items,audit,limit)['selected'])
                for model_name,group in model_groups.items():
                    counts=Counter()
                    for item in (i for i in items if i.model['name']==model_name):
                        score,diag,gen=cache[item.score_id],cache[item.diagnostic_id],cache[item.generation_id]
                        audited=item.id in audit['selected'];failed=score.result['value']<1
                        if failed and allowed_ids is not None and item.id not in allowed_ids:continue
                        if not (audited or failed and diag.result['unresolved']):continue
                        payload={'config':config,'evidence':{'sample':item.sample,'messages':item.messages,'response':gen.result['response'],'score':score.result,'diagnostic':diag.result},'dimensions':['semantic_correct','format_compliance','instruction_adequacy','scorer_correct'] if audited else diag.result['unresolved'],'diagnostic_id':diag.id,'purpose':'audit' if audited else 'unresolved'}
                        key=job_key('inspector',payload);job=s.get(Job,key);counts[classify(job)]+=1;inspection_jobs[key]=classify(job)
                    group['inspection']={'exact':True,'counts':dict(counts),'maximum':sum(counts.values())}
            else:
                for group in model_groups.values():
                    total=group['responses'];limit=config.get('failure_inspector_limit')
                    failure_max=total if limit is None else min(total,limit)
                    success_max=min(total,config['audit_max'],max(config['audit_min'],math.ceil(total*config['audit_fraction'])))
                    group['inspection']={'exact':not config['enabled'],'counts':{},'minimum':0,'maximum':min(total,failure_max+success_max) if config['enabled'] else 0}
            for group in model_groups.values():
                group['investigation_maximum']=1 if config['enabled'] and group['responses']>=3 else 0
            req.analysis_overrides={key:definition[key] for key in allowed}
            resolved.append(req.model_dump())
    return {'groups':groups,'generation_jobs':dict(Counter(generation_jobs.values())),'known_inspection_jobs':dict(Counter(inspection_jobs.values())),
        'response_slots':sum(g['responses'] for g in groups),'reports_maximum':reports,
        'resolved_runs':resolved,'note':'Cache counts are a snapshot. Inspection ranges depend on future answers, scoring, and heuristics; investigations and reports are separate. HTTP repair attempts may add requests.'}
