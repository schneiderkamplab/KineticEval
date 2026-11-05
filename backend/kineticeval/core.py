"""Durable immutable revisions and generation work claims."""
import hashlib
import json
import sqlite3
import time
import uuid


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS revisions (
                id TEXT PRIMARY KEY, definition TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY, model TEXT NOT NULL, inputs TEXT NOT NULL,
                status TEXT NOT NULL, output TEXT, token TEXT, lease REAL);
            CREATE TABLE IF NOT EXISTS attempts (
                token TEXT PRIMARY KEY, generation TEXT NOT NULL,
                started REAL NOT NULL, finished REAL, error TEXT);
        ''')

    def revision(self, definition):
        key = identity(definition)
        self.db.execute('INSERT OR IGNORE INTO revisions VALUES (?, ?)', (key, json.dumps(definition)))
        return key

    def request(self, model, messages, parameters=None, repeat=0):
        inputs = dict(model=model, messages=messages, parameters=parameters or {}, repeat=repeat)
        key = identity(inputs)
        self.db.execute('INSERT OR IGNORE INTO generations (id,model,inputs,status) VALUES (?,?,?,?)',
                        (key, model, json.dumps(inputs), 'pending'))
        return key

    def pending(self):
        grouped = {}
        for row in self.db.execute("SELECT * FROM generations WHERE status='pending'"):
            grouped.setdefault(row['model'], []).append(dict(row))
        return grouped

    def claim(self, model, lease_seconds=300):
        now = time.time()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute("UPDATE attempts SET finished=?, error='lease expired' WHERE finished IS NULL AND token IN (SELECT token FROM generations WHERE status='running' AND lease < ?)", (now, now))
            self.db.execute("UPDATE generations SET status='pending', token=NULL WHERE status='running' AND lease < ?", (now,))
            row = self.db.execute("SELECT * FROM generations WHERE model=? AND status='pending' LIMIT 1", (model,)).fetchone()
            if row is None:
                self.db.execute('COMMIT')
                return None
            token = uuid.uuid4().hex
            self.db.execute("UPDATE generations SET status='running',token=?,lease=? WHERE id=?", (token, now + lease_seconds, row['id']))
            self.db.execute('INSERT INTO attempts(token,generation,started) VALUES (?,?,?)', (token,row['id'],now))
            self.db.execute('COMMIT')
            return dict(id=row['id'], token=token, inputs=json.loads(row['inputs']))
        except Exception:
            self.db.execute('ROLLBACK')
            raise

    def finish(self, key, token, output=None, error=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            result = self.db.execute("UPDATE generations SET status=?,output=?,lease=NULL WHERE id=? AND token=? AND status='running'",
                ('failed' if error else 'succeeded', output, key, token))
            if result.rowcount != 1:
                raise ValueError('Stale or invalid claim')
            self.db.execute('UPDATE attempts SET finished=?,error=? WHERE token=?', (time.time(),error,token))
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK')
            raise

    def retry(self, key):
        return self.db.execute("UPDATE generations SET status='pending',token=NULL WHERE id=? AND status='failed'", (key,)).rowcount == 1
