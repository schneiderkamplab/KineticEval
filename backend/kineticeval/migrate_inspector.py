"""Rename inspection storage without changing immutable protocol or response evidence."""
import json
import sqlite3
import time

KEYS={'judge':'inspector','judge_id':'inspector_id','judge_budget':'inspector_budget',
      'judge_budget_history':'inspector_budget_history','failure_judge_limit':'failure_inspector_limit',
      'judge_assessed':'inspector_assessed','judge_correct':'inspector_correct',
      'judge_config':'inspector_config','judge_changed':'inspector_changed','judged':'inspected',
      'judge_status':'inspector_status','judgement':'assessment','judgment':'assessment'}


def translate(value):
    if isinstance(value,list):return [translate(v) for v in value]
    if isinstance(value,dict):
        # Pinned instructions/schemas and their fingerprint must remain byte-for-byte.
        return {KEYS.get(k,k):(v if k in {'protocol','protocol_snapshot'} else translate(v)) for k,v in value.items()}
    return value


def migrate(path):
    connection=sqlite3.connect(str(path),timeout=300)
    try:
        connection.execute('BEGIN IMMEDIATE')
        connection.execute('CREATE TABLE IF NOT EXISTS migration_records (id VARCHAR PRIMARY KEY, completed FLOAT NOT NULL, details JSON NOT NULL)')
        if connection.execute("SELECT 1 FROM migration_records WHERE id='inspector-terminology-v1'").fetchone():
            connection.rollback();return {'already_migrated':True}
        columns={r[1] for r in connection.execute('PRAGMA table_info(items)')}
        if 'judge_id' in columns:connection.execute('ALTER TABLE items RENAME COLUMN judge_id TO inspector_id')
        changed=0
        tables=[r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            cols=[r[1] for r in connection.execute(f'PRAGMA table_info("{table}")') if r[2].upper()=='JSON']
            for column in cols:
                cursor=connection.execute(f'SELECT rowid,"{column}" FROM "{table}" WHERE "{column}" LIKE ?',('%judge%',))
                while batch:=cursor.fetchmany(100):
                    updates=[]
                    for rowid,raw in batch:
                        parsed=json.loads(raw);value=translate(parsed)
                        if value!=parsed:updates.append((json.dumps(value),rowid));changed+=1
                    connection.executemany(f'UPDATE "{table}" SET "{column}"=? WHERE rowid=?',updates)
        connection.execute("UPDATE jobs SET kind='inspector' WHERE kind='judge'")
        details={'changed_json_records':changed}
        connection.execute('INSERT INTO migration_records VALUES (?,?,?)',('inspector-terminology-v1',time.time(),json.dumps(details)))
        connection.commit();return details
    except BaseException:
        connection.rollback();raise
    finally:connection.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('database')
    print(json.dumps(migrate(parser.parse_args().database)))
