import copy
from sqlalchemy import select
from kineticeval.db import Database,Revision,RevisionLabel,Run,Job,ModelConfiguration
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition
from kineticeval.migrate_inputs import migrate


def test_consolidation_rewrites_ids_preserves_snapshots_and_jobs(tmp_path):
    db=Database(tmp_path/'migration.db');service=Service(db)
    root=service.create_evaluation(demo_definition())
    with db.session() as s:
        original=s.get(Revision,root['revision_id'])
        data=copy.deepcopy(original.definition);data['scorer']={'name':'normalized','version':'1','config':{}}
        s.add(Revision(id='duplicate',evaluation_id=root['id'],parent_id=original.id,definition=data,created=original.created+1))
        s.add(RevisionLabel(revision_id='duplicate',name='full'))
    a=service.create_run({'revision_id':root['revision_id'],'models':[{'name':'fake','backend':'fake'}]})['id']
    b=service.create_run({'revision_id':'duplicate','models':[{'name':'fake','backend':'fake'}]})['id']
    with db.session() as s:before={j.id:copy.deepcopy(j.payload) for j in s.scalars(select(Job)).all()}
    result=migrate(db)
    assert result['input_revisions']==1
    with db.session() as s:
        revisions=s.scalars(select(Revision)).all()
        assert len(revisions)==1 and revisions[0].id not in {root['revision_id'],'duplicate'}
        assert revisions[0].parent_id is None
        assert s.get(Run,a).revision_id==s.get(Run,b).revision_id
        assert s.get(RevisionLabel,revisions[0].id).name=='full'
        assert {j.id:j.payload for j in s.scalars(select(Job)).all()}==before
        assert len(s.scalars(select(ModelConfiguration)).all())==1
    assert service.results(a)['definition']['scorer']['name']=='exact'
    assert service.results(b)['definition']['scorer']['name']=='normalized'
    assert migrate(db)['already_migrated']


def test_inspector_rename_preserves_protocol_and_job_ids(tmp_path):
    import sqlite3,json
    from kineticeval.migrate_inspector import migrate as rename
    from kineticeval.inspector_protocol import protocol_for,checked_protocol
    path=tmp_path/'old.db'
    connection=sqlite3.connect(path)
    connection.execute('CREATE TABLE items (id TEXT PRIMARY KEY,judge_id TEXT)')
    connection.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY,kind TEXT,payload JSON)')
    protocol=protocol_for({'dimensions':['semantic_correct']},legacy=True)
    payload={'config':{},'protocol':protocol,'definition':{'judge':{'failure_judge_limit':505}}}
    connection.execute('INSERT INTO jobs VALUES (?,?,?)',('original-id','judge',json.dumps(payload)))
    connection.execute('INSERT INTO items VALUES (?,?)',('item','original-id'))
    connection.commit();connection.close()
    rename(path)
    with sqlite3.connect(path) as connection:
        id,kind,text=connection.execute('SELECT * FROM jobs').fetchone()
        updated=json.loads(text)
        assert id=='original-id' and kind=='inspector'
        assert updated['protocol']==protocol
        assert updated['definition']['inspector']['failure_inspector_limit']==505
        assert connection.execute('SELECT inspector_id FROM items').fetchone()[0]=='original-id'
        checked_protocol(updated)
    assert rename(path)['already_migrated']
