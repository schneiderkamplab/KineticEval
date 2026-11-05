"""SQLite-backed durable queue and JSON artifacts using SQLAlchemy."""
import contextlib
import time
import uuid
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import JSON, String, Float, Integer


class InactiveClaim(Exception):
    pass


class Base(DeclarativeBase):
    pass


class Evaluation(Base):
    __tablename__ = 'evaluations'
    __table_args__ = {'schema': 'definitions'}
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    created: Mapped[float] = mapped_column(default=time.time)


class Revision(Base):
    __tablename__ = 'evaluation_revisions'
    __table_args__ = {'schema': 'definitions'}
    id: Mapped[str] = mapped_column(primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(index=True)
    parent_id: Mapped[str | None]
    definition: Mapped[dict] = mapped_column(JSON)
    created: Mapped[float] = mapped_column(default=time.time)


class RevisionLabel(Base):
    """Editable display metadata, separate from immutable evaluation definitions."""
    __tablename__ = 'revision_labels'
    __table_args__ = {'schema': 'definitions'}
    revision_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]


class Run(Base):
    __tablename__ = 'runs'
    id: Mapped[str] = mapped_column(primary_key=True)
    revision_id: Mapped[str] = mapped_column(index=True)
    parent_id: Mapped[str | None]
    status: Mapped[str] = mapped_column(default='active')
    config: Mapped[dict] = mapped_column(JSON)
    manifest: Mapped[dict] = mapped_column(JSON)
    audit: Mapped[dict | None] = mapped_column(JSON, default=None)
    report_job: Mapped[str | None]
    metric_jobs: Mapped[dict] = mapped_column(JSON, default=dict)
    created: Mapped[float] = mapped_column(default=time.time)


class Item(Base):
    __tablename__ = 'items'
    id: Mapped[str] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(index=True)
    example_id: Mapped[str]
    model: Mapped[dict] = mapped_column(JSON)
    repeat: Mapped[int]
    sample: Mapped[dict] = mapped_column(JSON)
    messages: Mapped[list] = mapped_column(JSON)
    generation_id: Mapped[str] = mapped_column(index=True)
    score_id: Mapped[str | None]
    diagnostic_id: Mapped[str | None]
    inspector_id: Mapped[str | None]
    reused: Mapped[bool] = mapped_column(default=False)


class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[str] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(index=True)
    model: Mapped[str] = mapped_column(default='')
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default='pending', index=True)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None]
    token: Mapped[str | None]
    lease: Mapped[float | None]
    created: Mapped[float] = mapped_column(default=time.time)


class JobActivity(Base):
    __tablename__ = 'job_activities'
    job_id: Mapped[str] = mapped_column(primary_key=True)
    token: Mapped[str]
    phase: Mapped[str]
    updated: Mapped[float]


class RunJob(Base):
    __tablename__ = 'run_jobs'
    run_id: Mapped[str] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(primary_key=True)


class Attempt(Base):
    __tablename__ = 'job_attempts'
    id: Mapped[str] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(index=True)
    started: Mapped[float]
    finished: Mapped[float | None]
    error: Mapped[str | None]


class Proposal(Base):
    __tablename__ = 'proposals'
    id: Mapped[str] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(index=True)
    proposal: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(default='pending')
    validation: Mapped[dict | None] = mapped_column(JSON)
    revision_id: Mapped[str | None]


class Experiment(Base):
    __tablename__ = 'experiments'
    id: Mapped[str] = mapped_column(primary_key=True)
    proposal_id: Mapped[str] = mapped_column(index=True)
    baseline_id: Mapped[str] = mapped_column(index=True)
    status: Mapped[str] = mapped_column(default='prepared')
    plan: Mapped[dict] = mapped_column(JSON)
    preview: Mapped[dict] = mapped_column(JSON)
    runs: Mapped[dict] = mapped_column(JSON, default=dict)
    created: Mapped[float] = mapped_column(default=time.time)
    approved_at: Mapped[float | None]


class Database:
    def __init__(self, path, definitions_path=None):
        import os
        from pathlib import Path
        from .storage import definition_path, legacy_layout
        self.path = str(path)
        self.definitions_path = definition_path(path, definitions_path or os.environ.get('KINETICEVAL_DEFINITIONS_DB'))
        self.legacy_layout = legacy_layout(path)
        if not self.legacy_layout and self.path != ':memory:' and Path(self.path).is_file() and not Path(self.definitions_path).is_file():
            import sqlite3
            with sqlite3.connect(self.path) as probe:
                has_runs = probe.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
                if has_runs and probe.execute('SELECT 1 FROM runs LIMIT 1').fetchone():
                    raise ValueError('Definitions database is missing; restore or rebuild the catalog before opening existing runs')
        if self.path != ':memory:' and Path(self.path).resolve() == Path(self.definitions_path).resolve():
            raise ValueError('Definitions and execution must use different files')
        self.engine = create_engine(f'sqlite:///{path}', connect_args={'check_same_thread': False, 'timeout': 300})
        if self.legacy_layout:
            self.engine = self.engine.execution_options(schema_translate_map={'definitions':None})
        @event.listens_for(self.engine, 'connect')
        def setup(connection, _):
            connection.execute('PRAGMA busy_timeout=300000')
            if self.legacy_layout:
                connection.execute('PRAGMA journal_mode=WAL')
            else:
                connection.execute('PRAGMA journal_mode=DELETE')
                connection.execute('ATTACH DATABASE ? AS definitions', (self.definitions_path,))
                connection.execute('PRAGMA definitions.journal_mode=DELETE')
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text('CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(kind,status,model,created)'))
            connection.execute(text('CREATE INDEX IF NOT EXISTS run_jobs_job ON run_jobs(job_id)'))

    @contextlib.contextmanager
    def session(self, immediate=False):
        with Session(self.engine, expire_on_commit=False) as session:
            if immediate:
                session.execute(text('BEGIN IMMEDIATE'))
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def claim(self, lease_seconds=300):
        claims = self.claim_many(lease_seconds=lease_seconds)
        return claims[0] if claims else None

    def claim_many(self, kind=None, model=None, limit=1, lease_seconds=300, run_id=None, backend=None, preferred_spec=None, kinds=None, runtime_id=None):
        from sqlalchemy import select
        now = time.time()
        with self.session(immediate=True) as s:
            expired = s.scalars(select(Job).where(Job.status == 'running', Job.lease < now)).all()
            for job in expired:
                attempt = s.get(Attempt, job.token)
                if attempt:
                    attempt.finished, attempt.error = now, 'lease expired; external request outcome unknown'
                job.status, job.token, job.lease = 'pending', None, None
            s.flush()
            query = select(Job).join(RunJob, RunJob.job_id == Job.id).join(Run, Run.id == RunJob.run_id).where(Job.status == 'pending', Run.status == 'active')
            if run_id:
                query = query.where(Run.id == run_id)
            if kinds is not None:
                query = query.where(Job.kind.in_(kinds))
            if kind:
                query = query.where(Job.kind == kind)
            if model:
                query = query.where(Job.model == model)
            if backend:
                query = query.where(Job.payload['model']['backend'].as_string() == backend)
            if kind == 'generation' and backend == 'hf':
                import json
                from sqlalchemy import func
                from .local_hf import runtime_identity
                specs = s.execute(query.with_only_columns(Job.payload['model'], func.min(Job.created))
                    .group_by(Job.payload['model']).order_by(func.min(Job.created))).all()
                if runtime_id:
                    from .core import identity
                    specs = [(spec, created) for spec, created in specs if identity(runtime_identity(spec)) == runtime_id]
                if not specs:
                    return []
                chosen = next((spec for spec, _ in specs if preferred_spec is not None
                    and runtime_identity(spec) == runtime_identity(preferred_spec)), specs[0][0])
                query = query.where(func.json(Job.payload['model']) == func.json(json.dumps(chosen)))
            jobs = s.scalars(query.distinct().order_by(Job.created).limit(limit)).all()
            if not jobs:
                return []
            if kind == 'generation':
                jobs = [job for job in jobs if job.payload['model'] == jobs[0].payload['model']]
            claims = []
            for job in jobs:
                token = uuid.uuid4().hex
                job.status, job.token, job.lease = 'running', token, now + lease_seconds
                s.add(Attempt(id=token, job_id=job.id, started=now))
                claims.append({'id': job.id, 'kind': job.kind, 'payload': job.payload, 'token': token})
            return claims

    def require_active_claim(self, key, token):
        from sqlalchemy import select
        with self.session() as s:
            active = s.scalar(select(Job.id).join(RunJob, RunJob.job_id == Job.id)
                .join(Run, Run.id == RunJob.run_id).where(Job.id == key, Job.token == token,
                Job.status == 'running', Job.lease > time.time(), Run.status == 'active').limit(1))
            if active is None:
                raise InactiveClaim('No active run authorizes this provider request')

    def abandon_claim(self, key, token):
        with self.session(immediate=True) as s:
            job = s.get(Job, key)
            if job and job.token == token and job.status == 'running':
                attempt = s.get(Attempt, token)
                if attempt:
                    attempt.finished, attempt.error = time.time(), 'Canceled before provider request'
                job.status, job.token, job.lease = 'pending', None, None

    def activity(self, claims, phase):
        with self.session(immediate=True) as s:
            for claim in claims:
                job = s.get(Job, claim['id'])
                if not job or job.token != claim['token'] or job.status != 'running':
                    continue
                s.merge(JobActivity(job_id=job.id, token=job.token, phase=phase, updated=time.time()))

    def heartbeat(self, key, token, lease_seconds=300):
        from sqlalchemy import update
        with self.session() as s:
            return s.execute(update(Job).where(Job.id == key, Job.token == token, Job.status == 'running')
                .values(lease=time.time() + lease_seconds)).rowcount == 1

    def finish(self, key, token, result=None, error=None):
        self.finish_many([{'id': key, 'token': token, 'result': result, 'error': error}])

    def finish_many(self, completions):
        """Commit a batch atomically; reject the entire batch on a stale claim."""
        with self.session(immediate=True) as s:
            checked = []
            for completion in completions:
                job = s.get(Job, completion['id'])
                if not job or job.token != completion['token'] or job.status != 'running':
                    raise ValueError('Stale worker claim')
                checked.append((job, completion))
            for job, completion in checked:
                error = completion.get('error')
                job.status = 'failed' if error else 'succeeded'
                job.result, job.error, job.lease = completion.get('result'), error, None
                attempt = s.get(Attempt, completion['token'])
                attempt.finished, attempt.error = time.time(), error


class ModelConfiguration(Base):
    __tablename__ = 'model_configurations'
    __table_args__ = {'schema': 'definitions'}
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    spec: Mapped[dict] = mapped_column(JSON)
    created: Mapped[float] = mapped_column(default=time.time)
    updated: Mapped[float] = mapped_column(default=time.time)


class ResultCollection(Base):
    __tablename__ = 'result_collections'
    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    description: Mapped[str] = mapped_column(default='')
    run_ids: Mapped[list] = mapped_column(JSON, default=list)
    comparison: Mapped[dict] = mapped_column(JSON, default=dict)
    launch_key: Mapped[str | None] = mapped_column(unique=True)
    created: Mapped[float] = mapped_column(default=time.time)


class MigrationRecord(Base):
    __tablename__ = 'migration_records'
    id: Mapped[str] = mapped_column(primary_key=True)
    completed: Mapped[float] = mapped_column(default=time.time)
    details: Mapped[dict] = mapped_column(JSON)
