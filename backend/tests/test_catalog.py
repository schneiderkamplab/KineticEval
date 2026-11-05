import asyncio
import sqlite3
import pytest
from kineticeval.db import Database
from kineticeval.service import Service
from kineticeval.catalog import export_catalog,build_catalog
from kineticeval.inspect_adapter import import_task


def test_export_and_rebuild_preserve_catalog_without_execution_data(tmp_path):
    db=Database(tmp_path/'execution.db');service=Service(db)
    definition=asyncio.run(import_task('kineticeval.fixtures:inspect_fixture'))
    saved=service.create_evaluation(definition.model_dump())
    service.name_revision(saved['revision_id'],'full')
    run=service.create_run({'revision_id':saved['revision_id'],'models':[{'name':'fixture','backend':'fake'}]})
    root=tmp_path/'catalog';export_catalog(db.definitions_path,root)
    assert not any('What is 2 + 2?' in p.read_text() for p in root.rglob('*.json'))
    output=tmp_path/'rebuilt.db';result=asyncio.run(build_catalog(root,output))
    assert result['evaluations']==1 and result['revisions']==1
    with sqlite3.connect(output) as connection:
        tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables=={'evaluations','evaluation_revisions','revision_labels','model_configurations'}
        assert connection.execute('SELECT id FROM evaluation_revisions').fetchone()[0]==saved['revision_id']
        assert connection.execute('SELECT name FROM revision_labels').fetchone()[0]=='full'
    restored=Service(Database(tmp_path/'fresh-execution.db',definitions_path=output))
    assert restored.evaluations()[0]['id']==saved['id']
    assert restored.runs()==[]
    with pytest.raises(ValueError,match='Output already exists'):
        asyncio.run(build_catalog(root,output))


def test_missing_catalog_cannot_silently_replace_definitions_for_existing_runs(tmp_path):
    from kineticeval.fixtures import demo_definition
    path=tmp_path/'execution.db';db=Database(path);service=Service(db)
    revision=service.create_evaluation(demo_definition())['revision_id']
    service.create_run({'revision_id':revision,'models':[{'name':'fake','backend':'fake'}]})
    db.engine.dispose()
    from pathlib import Path
    Path(db.definitions_path).unlink()
    with pytest.raises(ValueError,match='Definitions database is missing'):
        Database(path)


def test_failed_rebuild_preserves_destination(tmp_path):
    import json
    root=tmp_path/'catalog';root.mkdir()
    (root/'catalog.json').write_text(json.dumps({'format_version':1,'datasets':[],
        'evaluations':['evaluations/missing'],'models':[]}))
    output=tmp_path/'definitions.db';output.write_bytes(b'original database')
    with pytest.raises(ValueError,match='Missing catalog entry'):
        asyncio.run(build_catalog(root,output,replace=True))
    assert output.read_bytes()==b'original database'
