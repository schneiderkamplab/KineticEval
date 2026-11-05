"""Immutable approval previews and atomic, traceable matched-example experiments."""
import copy
import time
from collections import Counter
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from .core import identity
from .db import Experiment, Proposal, Run, Revision, Item, Job
from .schemas import Definition, ModelSpec, RunRequest
from .service import require, uid, render, materialize, physical_model


class Arm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=80)
    parameters: dict[str, Any] = Field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str | None = None

    @model_validator(mode='after')
    def supported(self):
        if set(self.parameters) - {'max_tokens', 'temperature', 'top_p', 'seed'}:
            raise ValueError('Only max_tokens, temperature, top_p and seed can be varied')
        if set(self.chat_template_kwargs) - {'enable_thinking'}:
            raise ValueError('Only enable_thinking is supported as a template override')
        if 'enable_thinking' in self.chat_template_kwargs and type(self.chat_template_kwargs['enable_thinking']) is not bool:
            raise ValueError('enable_thinking must be boolean')
        for key in ('max_tokens', 'seed'):
            if key in self.parameters and type(self.parameters[key]) is not int:
                raise ValueError(f'{key} must be an integer')
        if not 1 <= self.parameters.get('max_tokens', 64) <= 32768:
            raise ValueError('max_tokens must be between 1 and 32768')
        for key in ('temperature', 'top_p'):
            value = self.parameters.get(key, 1)
            if type(value) not in (int, float) or not 0 <= value <= (2 if key == 'temperature' else 1):
                raise ValueError(f'Invalid {key}')
        if self.parameters.get('top_p', 1) == 0:
            raise ValueError('top_p must be greater than zero')
        return self


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str
    example_ids: list[str] = Field(min_length=1)
    arms: list[Arm] = Field(min_length=1, max_length=12)
    @model_validator(mode='after')
    def unique(self):
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError('Example IDs must be unique')
        if len({a.name for a in self.arms}) != len(self.arms):
            raise ValueError('Arm names must be unique')
        return self


class SupportedSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(min_length=1)
    revision: str | None = None
    dtype: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    arms: list[dict[str, Any]] | None = None
    prompt_template: str | None = None


def executable_arms(proposal):
    """Fail closed: prose-only steps and unknown transformations are not executable."""
    patch = proposal['patch']
    allowed = {'proposed_settings','hypothesis','hypotheses','experiment','comparison_protocol','advisory','auto_apply','kind'}
    unsupported = set(patch) - allowed
    if unsupported:
        raise ValueError('Unsupported experiment operations: ' + ', '.join(sorted(unsupported)))
    if 'proposed_settings' not in patch:
        raise ValueError('Proposal has no structured executable settings; analysis steps cannot be replaced with an unchanged rerun')
    settings = SupportedSettings.model_validate(patch['proposed_settings'])
    Arm(name='settings', parameters=settings.parameters, chat_template_kwargs=settings.chat_template_kwargs,
        prompt_template=settings.prompt_template)
    raw_arms = settings.arms if settings.arms is not None else [{'name':'follow-up'}]
    if not raw_arms:
        raise ValueError('Proposal has no executable arms')
    arms=[]
    for raw in raw_arms:
        raw=dict(raw)
        parameters={**settings.parameters, **raw.pop('parameters', {})}
        for key in ('max_tokens','temperature','top_p','seed'):
            if key in raw:parameters[key]=raw.pop(key)
        raw['parameters']=parameters
        raw['chat_template_kwargs']={**settings.chat_template_kwargs, **raw.get('chat_template_kwargs', {})}
        if settings.prompt_template is not None:raw.setdefault('prompt_template',settings.prompt_template)
        arms.append(Arm.model_validate(raw).model_dump(exclude_none=True))
    return settings, arms


def validate_effective_change(plan, base_model, definition):
    rows = [r for r in materialize(definition) if r['id'] in set(plan.example_ids)]
    for arm in plan.arms:
        parameters={**base_model['parameters'], **arm.parameters}
        kwargs={**base_model.get('chat_template_kwargs', {}), **arm.chat_template_kwargs}
        defaults = {'max_tokens':64, 'temperature':0, 'top_p':1, 'seed':None}
        before = {**defaults, **base_model['parameters']}
        after = {**defaults, **parameters}
        if after != before or kwargs != base_model.get('chat_template_kwargs', {}):
            return
        if arm.prompt_template is not None:
            amended={**definition,'prompt_template':arm.prompt_template}
            if any(render(amended,row)!=render(definition,row) for row in rows):return
    raise ValueError('Experiment makes no effective change to baseline inputs or generation settings; existing-result analysis does not require a follow-up run')


def validate_proposal_plan(proposal, plan, base_model, definition):
    settings, _ = executable_arms(proposal)
    if settings.model != plan.model:
        raise ValueError('Model must match the inspector proposal')
    if settings.revision is not None and settings.revision != base_model.get('revision'):
        raise ValueError('Proposed model revision differs from the pinned baseline')
    if settings.dtype is not None and settings.dtype != base_model.get('dtype'):
        raise ValueError('Proposed dtype differs from the pinned baseline')
    validate_effective_change(plan,base_model,definition)


def seed(service, proposal_id):
    with service.db.session() as s:
        proposal = require(s, Proposal, proposal_id)
        baseline = require(s, Run, proposal.run_id)
        if proposal.proposal['kind'] != 'experiment':
            raise ValueError('Select an experiment proposal')
        model = ''
        arms, warnings = [], []
        blocked = False
        try:
            settings, arms = executable_arms(proposal.proposal)
            model = settings.model
            base_model = next((m for m in baseline.config['models'] if m['name']==model), None)
            if base_model is None:raise ValueError('Proposed model is not in the baseline')
            definition = baseline.config.get('effective_definition') or require(s, Revision, baseline.revision_id).definition
            validate_proposal_plan(proposal.proposal, Plan(model=model,example_ids=baseline.manifest['ids'],arms=arms),base_model,definition)
        except ValueError as exc:
            blocked = True
            warnings.append(str(exc))
        return {'proposal': proposal.proposal, 'baseline_id':baseline.id, 'status':proposal.status,
            'models': baseline.config['models'], 'warnings':warnings, 'blocked':blocked,
            'plan':{'model':model,'example_ids':baseline.manifest['ids'], 'arms':arms},
            'scope_note':'Defaults to all examples in the baseline run. Inspector example_ids are evidence citations, not an executable sampling instruction. Review the scope against the proposed protocol.'}


def prepare(service, proposal_id, body):
    plan = Plan.model_validate(body)
    with service.db.session(immediate=True) as s:
        proposal = require(s, Proposal, proposal_id)
        if proposal.status != 'pending' or proposal.proposal['kind'] != 'experiment':
            raise ValueError('A pending experiment proposal is required')
        baseline = require(s, Run, proposal.run_id)
        if set(plan.example_ids) - set(baseline.manifest['ids']):
            raise ValueError('Follow-ups must use examples from the baseline run')
        base_model = next((m for m in baseline.config['models'] if m['name'] == plan.model), None)
        if base_model is None:
            raise ValueError('Select a model from the baseline run')
        old = require(s, Revision, baseline.revision_id)
        baseline_definition = baseline.config.get('effective_definition') or old.definition
        validate_proposal_plan(proposal.proposal,plan,base_model,baseline_definition)
        previews = []
        for arm in plan.arms:
            definition = copy.deepcopy(baseline_definition)
            if arm.prompt_template is not None: definition['prompt_template'] = arm.prompt_template
            definition = Definition.model_validate(definition).model_dump()
            model = ModelSpec.model_validate({**base_model,
                'parameters':{**base_model['parameters'], **arm.parameters},
                'chat_template_kwargs':{**base_model.get('chat_template_kwargs', {}), **arm.chat_template_kwargs}}).model_dump()
            if model['parameters'].get('max_tokens',64) >= model['context_limit']:
                raise ValueError('Output budget must leave room for the prompt within the context limit')
            rows = [r for r in materialize(definition) if r['id'] in set(plan.example_ids)]
            counts = Counter()
            for row in rows:
                for repeat in range(baseline.config['repeats']):
                    payload = {'model':physical_model(model), 'messages':render(definition,row), 'repeat':repeat, 'fresh':None}
                    key = identity({'kind':'generation','payload':payload,'engine_version':'1'})
                    job = s.get(Job,key)
                    counts['reusable' if job and job.status=='succeeded' else 'pending_or_new'] += 1
            previews.append({'name':arm.name,'model':model,'prompt_changed':definition['prompt_template']!=baseline_definition['prompt_template'],
                'prompt_template':definition['prompt_template'],'examples':len(rows),'repeats':baseline.config['repeats'],
                'reusable':counts['reusable'],'pending_or_new':counts['pending_or_new'],
                'output_token_ceiling':len(rows)*baseline.config['repeats']*model['parameters'].get('max_tokens',64),
                'prompt_preview':[{'example_id':r['id'],'messages':render(definition,r)} for r in rows[:2]]})
        record = Experiment(id=uid(),proposal_id=proposal_id,baseline_id=baseline.id,plan=plan.model_dump(),
            preview={'arms':previews,'baseline_model':base_model,'baseline_revision':old.id,
                'origin_job':proposal.proposal.get('origin_job'),'proposal_snapshot':proposal.proposal,
                'note':'Approval queues these exact arms. No generations run before approval. Token ceilings are not API cost estimates.'})
        s.add(record)
        s.flush()
        return describe(record)


def describe(record):
    return {key:getattr(record,key) for key in ('id','proposal_id','baseline_id','status','plan','preview','runs','created','approved_at')}


def approve(service, experiment_id):
    with service.db.session(immediate=True) as s:
        record = require(s, Experiment, experiment_id)
        if record.status == 'approved': return describe(record)
        if record.status != 'prepared': raise ValueError('Experiment is not awaiting approval')
        proposal = require(s, Proposal, record.proposal_id)
        if proposal.status != 'pending': raise ValueError('Proposal was already reviewed')
        baseline = require(s, Run, record.baseline_id)
        old = require(s, Revision, baseline.revision_id)
        baseline_definition = baseline.config.get('effective_definition') or old.definition
        plan = Plan.model_validate(record.plan)
        base_model = next(m for m in baseline.config['models'] if m['name']==plan.model)
        validate_proposal_plan(proposal.proposal,plan,base_model,baseline_definition)
        runs = {}
        for arm in record.preview['arms']:
            revision_id = old.id
            if arm['prompt_changed']:
                definition = {**baseline_definition,'prompt_template':arm['prompt_template']}
                from .revisions import save_revision
                new, _ = save_revision(s,old,definition)
                s.flush();revision_id = new.id
            req = RunRequest(revision_id=revision_id, analysis_overrides={k:baseline_definition[k] for k in ('scorer','metrics','heuristics','inspector','expected_format')}, models=[arm['model']], parent_id=baseline.id,
                example_ids=record.plan['example_ids'], repeats=baseline.config['repeats'])
            runs[arm['name']] = service._create_run(s,req)['id']
        from .db import ResultCollection
        s.add(ResultCollection(id=uid(),name=proposal.proposal['title'],description='Approved matched-example experiment',run_ids=list(runs.values())))
        record.runs = runs
        record.status = 'approved';record.approved_at = time.time();proposal.status = 'approved'
        return describe(record)


def results(service, experiment_id):
    with service.db.session() as s:
        record = require(s, Experiment, experiment_id)
        result = describe(record)
        selected = set(record.plan['example_ids'])
        def values(run_id):
            rows = s.scalars(select(Item).where(Item.run_id==run_id)).all()
            data = {}
            for item in rows:
                if item.example_id not in selected or item.model['name']!=record.plan['model']:continue
                score = s.get(Job,item.score_id) if item.score_id else None
                gen = s.get(Job,item.generation_id)
                data[(item.example_id,item.repeat)] = {'score':score.result['value'] if score and score.status=='succeeded' else None,
                    'truncated':(gen.result or {}).get('metadata',{}).get('finish_reason')=='length',
                    'tokens':(gen.result or {}).get('metadata',{}).get('output_tokens'),
                    'generated':gen.status=='succeeded'}
            return data
        baseline = values(record.baseline_id)
        comparisons=[]
        for name,run_id in record.runs.items():
            arm=values(run_id)
            pairs=[(baseline[key]['score'], value['score']) for key,value in arm.items()
                if key in baseline and value['score'] is not None and baseline[key]['score'] is not None]
            comparisons.append({'arm':name,'run_id':run_id,'status':s.get(Run,run_id).status,'total':len(arm),
                'paired_scored':len(pairs),'baseline_accuracy':sum(a for a,b in pairs)/len(pairs) if pairs else None,
                'followup_accuracy':sum(b for a,b in pairs)/len(pairs) if pairs else None,
                'paired_delta':sum(b-a for a,b in pairs)/len(pairs) if pairs else None,
                'generated':sum(v['generated'] for v in arm.values()),'token_limited':sum(v['truncated'] for v in arm.values())})
        result['comparisons']=comparisons
        return result


def listing(service):
    with service.db.session() as s:
        return [describe(row) for row in s.scalars(select(Experiment).order_by(Experiment.created.desc())).all()]
