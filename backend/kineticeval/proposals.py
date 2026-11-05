"""Reviewed proposals and regression-gated heuristic activation."""
from .db import Proposal, Run, Revision
from .service import require, uid, pin
from .schemas import Definition
from .components import diagnose


def review(service, proposal_id, action, cases=None):
    with service.db.session(immediate=True) as s:
        proposal = require(s, Proposal, proposal_id)
        if proposal.status != 'pending':
            raise ValueError('Proposal already reviewed')
        if action == 'reject':
            proposal.status = 'rejected'
            return {'status': 'rejected'}
        if action not in {'validate', 'approve'}:
            raise ValueError('Unknown review action')
        if proposal.proposal['kind'] == 'experiment':
            raise ValueError('Experiment recommendations are advisory; configure a separate run to test them')
        run = require(s, Run, proposal.run_id)
        revision = require(s, Revision, run.revision_id)
        patch = proposal.proposal['patch']
        permitted = {'overlay': {'overlays'}, 'prompt': {'prompt_template', 'system_prompt', 'expected_format'},
                     'scorer': {'scorer', 'metrics'}, 'heuristic': {'heuristics'}}[proposal.proposal['kind']]
        if not patch or set(patch) - permitted:
            raise ValueError('Proposal patch contains fields outside its declared kind')
        if 'overlays' in patch:
            patch = {**patch, 'overlays': {**revision.definition['overlays'], **patch['overlays']}}
        data = pin(Definition.model_validate({**revision.definition, **patch}))
        if cases is not None:
            if not cases or not any(c['expected_semantic'] is True for c in cases) or not any(c['expected_semantic'] is not True for c in cases):
                raise ValueError('Regression cases require positive examples and counterexamples')
            results = []
            for case in cases:
                payload = {'response': case['response'], 'sample': {'target': case['target'], 'choices': []},
                    'messages': [{'role': 'user', 'content': case.get('prompt', '')}], 'score': {'value': 0},
                    'expected_format': data['expected_format'], 'components': data['heuristics']}
                actual = diagnose(payload)['finding']['semantic_correct']
                results.append({**case, 'actual_semantic': actual, 'passed': actual == case['expected_semantic']})
            proposal.validation = {'cases': results, 'passed': all(c['passed'] for c in results)}
        if action == 'validate':
            return {'validation': proposal.validation}
        if proposal.proposal['kind'] == 'heuristic' and not (proposal.validation or {}).get('passed'):
            raise ValueError('Validate heuristic against positive examples and counterexamples before approval')
        new = Revision(id=uid(), evaluation_id=revision.evaluation_id, parent_id=revision.id, definition=data)
        s.add(new)
        proposal.status, proposal.revision_id = 'approved', new.id
        return {'status': 'approved', 'revision_id': new.id}
