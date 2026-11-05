"""Input identity and mutable suggestions, with historical run snapshots."""
import copy
from sqlalchemy import select
from .core import identity
from .db import Revision,Run

INPUT_KEYS=('source','examples','overlays','sampler','prompt_template','system_prompt')
# Import metadata about execution is provenance, not evaluation input identity.
SOURCE_DEFAULT_KEYS={'generation_config','protocol','scorer_spec','metric_specs','metric_names','solver_specs','solver'}


def input_identity(definition):
    inputs={k:copy.deepcopy(definition.get(k)) for k in INPUT_KEYS}
    overlays=inputs.pop('overlays') or {}
    inputs['examples']=[{**row,**overlays.get(row['id'],{})} for row in (inputs['examples'] or [])]
    inputs['source']={k:v for k,v in (inputs['source'] or {}).items() if k not in SOURCE_DEFAULT_KEYS}
    return identity(inputs)


def save_revision(s, previous, definition):
    from .service import uid
    key=input_identity(definition)
    candidates=s.scalars(select(Revision).where(Revision.evaluation_id==previous.evaluation_id).order_by(Revision.created)).all()
    existing=next((r for r in candidates if input_identity(r.definition)==key),None)
    if existing:
        # Legacy runs must retain their original defaults before any mutation.
        for run in s.scalars(select(Run).where(Run.revision_id==existing.id)).all():
            if 'effective_definition' not in run.config:
                run.config={**run.config,'effective_definition':copy.deepcopy(existing.definition)}
        existing.definition=copy.deepcopy(definition)
        return existing,False
    row=Revision(id=uid(),evaluation_id=previous.evaluation_id,parent_id=previous.id,definition=definition)
    s.add(row)
    return row,True
