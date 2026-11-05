"""Paths and explicit migration for the definition/execution storage boundary."""
import json
import sqlite3
import time
from pathlib import Path

DEFINITION_TABLES = ('evaluations', 'evaluation_revisions', 'revision_labels', 'model_configurations')
VERSION = 'split-storage-v1'


def definition_path(execution_path, configured=None):
    if configured:
        return str(configured)
    if str(execution_path) == ':memory:':
        return ':memory:'
    path = Path(execution_path)
    return str(path.with_name(path.stem + '.definitions' + (path.suffix or '.db')))


def legacy_layout(path):
    if str(path) == ':memory:' or not Path(path).is_file():
        return False
    with sqlite3.connect(path) as connection:
        return bool(connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluations'").fetchone())


def split_database(execution_path, definitions_path=None):
    """Move catalog tables without changing IDs. Stop all writers and back up first."""
    target = definition_path(execution_path, definitions_path)
    if Path(execution_path).resolve() == Path(target).resolve():
        raise ValueError('Definitions and execution must use different files')
    connection = sqlite3.connect(execution_path, timeout=300)
    try:
        connection.execute('PRAGMA busy_timeout=300000')
        # Rollback journals provide atomic commits across attached SQLite files.
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('ATTACH DATABASE ? AS definitions', (target,))
        connection.execute('PRAGMA definitions.journal_mode=DELETE')
        present = {row[0] for row in connection.execute("SELECT name FROM main.sqlite_master WHERE type='table'")}
        existing = {row[0] for row in connection.execute("SELECT name FROM definitions.sqlite_master WHERE type='table'")}
        if not set(DEFINITION_TABLES) & present:
            if not set(DEFINITION_TABLES) <= existing:
                raise ValueError('No complete definition catalog found')
            return {'already_migrated':True, 'definitions_path':target}
        if set(DEFINITION_TABLES) & existing:
            raise ValueError('Destination already contains catalog tables; refusing to overwrite')
        if not set(DEFINITION_TABLES) <= present:
            raise ValueError('Legacy catalog is incomplete; finish earlier migrations first')
        connection.execute('BEGIN IMMEDIATE')
        counts = {}
        for table in DEFINITION_TABLES:
            ddl = connection.execute('SELECT sql FROM main.sqlite_master WHERE type=? AND name=?', ('table', table)).fetchone()[0]
            body = ddl[ddl.index('('):]
            connection.execute(f'CREATE TABLE definitions."{table}" {body}')
            connection.execute(f'INSERT INTO definitions."{table}" SELECT * FROM main."{table}"')
            counts[table] = connection.execute(f'SELECT count(*) FROM main."{table}"').fetchone()[0]
            if connection.execute(f'SELECT * FROM main."{table}" EXCEPT SELECT * FROM definitions."{table}" LIMIT 1').fetchone():
                raise ValueError(f'Catalog verification failed for {table}')
            indexes = connection.execute("SELECT name,sql FROM main.sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)).fetchall()
            for name, sql in indexes:
                tail = sql[sql.upper().index(' ON '):]
                unique = 'UNIQUE ' if 'CREATE UNIQUE INDEX' in sql.upper() else ''
                connection.execute(f'CREATE {unique}INDEX definitions."{name}"{tail}')
        for table in reversed(DEFINITION_TABLES):
            connection.execute(f'DROP TABLE main."{table}"')
        connection.execute('INSERT INTO migration_records (id,completed,details) VALUES (?,?,?)',
            (VERSION, time.time(), json.dumps({'counts':counts, 'definition_file':Path(target).name})))
        connection.commit()
        return {'definitions_path':target, 'counts':counts}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('execution_database')
    parser.add_argument('--definitions-database')
    args=parser.parse_args()
    print(json.dumps(split_database(args.execution_database,args.definitions_database),indent=2))
