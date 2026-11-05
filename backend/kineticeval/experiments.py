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
from .service import require, uid, render, materialize


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


def seed(service, proposal_id):
    with service.db.session() as s:
        proposal = require(s, Proposal, proposal_id)
        baseline = require(s, Run, proposal.run_id)
        if proposal.proposal['kind'] != 'experiment':
            raise ValueError('Select an experiment proposal')
        patch = proposal.proposal['patch']
        settings = patch.get('proposed_settings', {})
        model = settings.get('model', '')
        arms, warnings = [], []
        if 'transform' in patch:
            warnings.append('This proposal requires a data transformation. This editor supports generation settings and prompt templates only; no transformation has been applied.')
        else:
            for raw in settings.get('arms', [{'name':'follow-up', 'parameters': settings.get('parameters', {}),
                                               'chat_template_kwargs': settings.get('chat_template_kwargs', {})}]):
                allowed = {'name','parameters','chat_template_kwargs','prompt_template','max_tokens','temperature','top_p','seed'}
                if set(raw) - allowed:
                    warnings.append('An arm contains unsupported fields; configure supported arms explicitly.')
                    arms = []
                    break
                arm = {'name':raw.get('name','follow-up'),
                    'parameters':{**settings.get('parameters', {}), **raw.get('parameters', {}),
                        **{k:raw[k] for k in ('max_tokens','temperature','top_p','seed') if k in raw}},
                    'chat_template_kwargs':{**settings.get('chat_template_kwargs', {}), **raw.get('chat_template_kwargs', {})}}
                if 'prompt_template' in raw: arm['prompt_template'] = raw['prompt_template']
                arms.append(arm)
        return {'proposal': proposal.proposal, 'baseline_id':baseline.id, 'status':proposal.status,
            'models': baseline.config['models'], 'warnings':warnings,
            'plan':{'model':model,'example_ids':baseline.manifest['ids'], 'arms':arms},
            'scope_note':'Defaults to all examples in the baseline run. Judge example_ids are evidence citations, not an executable sampling instruction. Review the scope against the proposed protocol.'}


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
        previews = []
        for arm in plan.arms:
            definition = copy.deepcopy(old.definition)
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
                    payload = {'model':model, 'messages':render(definition,row), 'repeat':repeat, 'fresh':None}
                    key = identity({'kind':'generation','payload':payload,'engine_version':'1'})
                    job = s.get(Job,key)
                    counts['reusable' if job and job.status=='succeeded' else 'pending_or_new'] += 1
            previews.append({'name':arm.name,'model':model,'prompt_changed':definition['prompt_template']!=old.definition['prompt_template'],
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
        runs = {}
        for arm in record.preview['arms']:
            revision_id = old.id
            if arm['prompt_changed']:
                definition = {**old.definition,'prompt_template':arm['prompt_template']}
                new = Revision(id=uid(),evaluation_id=old.evaluation_id,parent_id=old.id,definition=definition)
                s.add(new);s.flush();revision_id = new.id
            req = RunRequest(revision_id=revision_id, models=[arm['model']], parent_id=baseline.id,
                example_ids=record.plan['example_ids'], repeats=baseline.config['repeats'])
            runs[arm['name']] = service._create_run(s,req)['id']
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
