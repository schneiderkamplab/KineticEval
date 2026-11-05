import sqlite3
import pytest
from sqlalchemy import select
from kineticeval.db import Database, Evaluation, Revision, ModelConfiguration, Run
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition
from kineticeval.storage import split_database


def tables(path):
    with sqlite3.connect(path) as c:
        return {row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_catalog_and_execution_have_distinct_files_and_atomic_rollback(tmp_path):
    path=tmp_path/'execution.db';db=Database(path);service=Service(db)
    revision=service.create_evaluation(demo_definition())['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake'}]})['id']
    assert 'evaluations' not in tables(path)
    assert {'evaluations','evaluation_revisions','revision_labels','model_configurations'} <= tables(db.definitions_path)
    assert 'runs' not in tables(db.definitions_path)
    with pytest.raises(RuntimeError):
        with db.session(immediate=True) as session:
            session.get(Evaluation,service.evaluations()[0]['id']).name='changed'
            session.get(Run,run).status='canceled'
            session.flush()
            raise RuntimeError('rollback both files')
    assert service.results(run)['run']['status']=='active'
    assert service.evaluations()[0]['name']==demo_definition()['name']


def test_legacy_split_preserves_ids_and_artifacts(tmp_path):
    path=tmp_path/'legacy.db'
    # Force the compatibility path, then construct a complete legacy database.
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE evaluations (id VARCHAR PRIMARY KEY,name VARCHAR NOT NULL,created FLOAT NOT NULL)')
    db=Database(path);service=Service(db)
    revision=service.create_evaluation(demo_definition())['revision_id']
    run=service.create_run({'revision_id':revision,'models':[{'name':'fixture','backend':'fake'}]})['id']
    before=service.results(run)
    db.engine.dispose()
    result=split_database(path)
    assert result['counts']['evaluation_revisions']==1
    assert 'evaluation_revisions' not in tables(path)
    restored=Service(Database(path))
    assert restored.results(run)==before
    assert split_database(path)['already_migrated']
