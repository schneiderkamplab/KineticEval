"""Explicit, transactional input-revision migration. Run with writers quiesced."""
import copy
import time
from collections import defaultdict
from sqlalchemy import select,delete,JSON
from .db import Database,Base,Revision,RevisionLabel,Run,ModelConfiguration,MigrationRecord
from .revisions import input_identity
from .core import identity
from .service import uid

VERSION='input-revisions-v1'


def rewrite(value,mapping):
    if isinstance(value,str):return mapping.get(value,value)
    if isinstance(value,list):return [rewrite(v,mapping) for v in value]
    if isinstance(value,dict):return {mapping.get(k,k):rewrite(v,mapping) for k,v in value.items()}
    return value


def migrate(db):
    with db.session(immediate=True) as s:
        done=s.get(MigrationRecord,VERSION)
        if done:return {**done.details,'already_migrated':True}
        revisions=s.scalars(select(Revision).order_by(Revision.created)).all()
        old={r.id:r for r in revisions}
        labels={r.revision_id:r.name for r in s.scalars(select(RevisionLabel)).all()}
        groups=defaultdict(list)
        for r in revisions:groups[(r.evaluation_id,input_identity(r.definition))].append(r)
        mapping={}
        replacements=[]
        for members in groups.values():
            key=uid()
            for r in members:mapping[r.id]=key
            first,last=members[0],members[-1]
            replacements.append((key,first,last))
        runs=s.scalars(select(Run).order_by(Run.created)).all()
        for run in runs:
            if 'effective_definition' not in run.config:
                run.config={**run.config,'effective_definition':copy.deepcopy(old[run.revision_id].definition)}
        s.flush()
        # Rewrite structured references in every mapped table, including embedded
        # experiment previews and immutable historical payload references.
        for mapper in Base.registry.mappers:
            cls=mapper.class_
            if cls in {Revision,RevisionLabel,MigrationRecord}:continue
            columns=list(mapper.columns)
            relevant=[c for c in columns if isinstance(c.type,JSON) or c.name in {'revision_id','parent_id'}]
            if not relevant:continue
            for row in s.scalars(select(cls).execution_options(yield_per=100)):
                # Stream large job artifacts rather than loading the full cache.
                for column in relevant:
                    value=getattr(row,column.name)
                    updated=rewrite(value,mapping)
                    if updated!=value:setattr(row,column.name,updated)
        s.execute(delete(RevisionLabel));s.execute(delete(Revision))
        for key,first,last in replacements:
            parent=mapping.get(first.parent_id)
            if parent==key:parent=None
            s.add(Revision(id=key,evaluation_id=first.evaluation_id,parent_id=parent,
                definition=rewrite(copy.deepcopy(last.definition),mapping),created=first.created))
            if labels.get(last.id):s.add(RevisionLabel(revision_id=key,name=labels[last.id]))
        # Seed all distinct historical configurations without losing variants.
        known={identity(r.spec) for r in s.scalars(select(ModelConfiguration)).all()}
        added=0
        for run in runs:
            for spec in run.config['models']:
                if spec.get('backend')=='ucloud':continue
                spec=copy.deepcopy(spec)
                display=spec['name']
                spec['name']=spec.pop('model_identifier',None) or spec['name']
                spec.pop('configuration_id',None)
                fingerprint=identity(spec)
                if fingerprint in known:continue
                known.add(fingerprint);added+=1
                s.add(ModelConfiguration(id=uid(),name=f'{display} · {fingerprint[:6]}',spec=spec,created=run.created,updated=time.time()))
        details={'old_revisions':len(revisions),'input_revisions':len(replacements),'runs_preserved':len(runs),'models_seeded':added}
        s.add(MigrationRecord(id=VERSION,details=details))
        s.flush()
        valid=set(s.scalars(select(Revision.id)))
        if any(r.revision_id not in valid for r in runs):raise ValueError('Migration left a dangling run revision')
        return details


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database')
    args=parser.parse_args()
    print(json.dumps(migrate(Database(args.database))))
