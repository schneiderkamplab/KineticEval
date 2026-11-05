"""Local HTTP API and built dashboard."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, ValidationError
from .db import Database
from .service import Service
from .schemas import Definition, RunRequest
from . import importers


def create_app(db_path=None):
    app = FastAPI(title='KineticEval', version='0.1.0')
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=3)
    service = Service(Database(db_path or os.environ.get('KINETICEVAL_DB', 'kineticeval.db')))
    app.state.service = service

    @app.middleware('http')
    async def local_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        from urllib.parse import urlparse
        if origin and urlparse(origin).hostname not in {'127.0.0.1', 'localhost', '::1'}:
            return JSONResponse({'detail': 'Only local browser origins are allowed'}, status_code=403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from .worker import safe_error
        return JSONResponse({'detail': safe_error(exc)}, status_code=422)

    @app.exception_handler(ImportError)
    async def dependency_missing(request, exc):
        return JSONResponse({'detail': 'Optional integration dependency unavailable. Install kineticeval with the inspect and data extras; see README.'}, status_code=422)

    @app.get('/api/health')
    def health(): return {'status': 'ok'}

    @app.get('/api/evaluations')
    def evaluations(summary: bool = False): return service.evaluations(summary)

    @app.post('/api/evaluations')
    def create(definition: Definition): return service.create_evaluation(definition)

    @app.get('/api/revisions/{revision_id}')
    def revision(revision_id: str):
        from .db import Revision
        from .service import require
        with service.db.session() as s:
            return require(s, Revision, revision_id).definition

    @app.post('/api/revisions/{revision_id}')
    def revise(revision_id: str, patch: dict): return service.revise(revision_id, patch)

    @app.post('/api/revisions/{revision_id}/preview')
    def preview(revision_id: str, patch: dict): return service.preview(revision_id, patch)

    @app.post('/api/import/jsonl')
    def jsonl(body: dict):
        return service.create_evaluation(importers.from_jsonl(body['name'], body['content'], body.get('mapping')))

    @app.post('/api/import/huggingface')
    def hf(body: dict):
        limit = body.get('limit', 20)
        if not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError('Import limit must be 1–10000')
        return service.create_evaluation(importers.from_huggingface(**body))

    @app.post('/api/import/inspect')
    async def inspect_task(body: dict):
        from .inspect_adapter import import_task
        limit = body.get('limit', 100)
        if not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError('Import limit must be 1–10000')
        return service.create_evaluation(await import_task(**body))

    @app.get('/api/runs')
    def runs(): return service.runs()

    @app.post('/api/runs')
    def run(body: RunRequest): return service.create_run(body)

    @app.get('/api/runs/{run_id}')
    def results(run_id: str): return service.results(run_id)

    @app.post('/api/runs/{run_id}/{action}')
    def control(run_id: str, action: str, body: dict | None = None):
        return service.control(run_id, action, (body or {}).get('job_ids'))

    @app.get('/api/jobs/{job_id}/attempts')
    def attempts(job_id: str): return service.attempts(job_id)

    @app.get('/api/pending')
    def pending(): return service.pending()

    @app.get('/api/experiments')
    def experiments():
        from .experiments import listing
        return listing(service)

    @app.get('/api/experiment-proposals')
    def experiment_proposals():
        from sqlalchemy import select
        from .db import Proposal
        with service.db.session() as s:
            return [{'id':p.id,'run_id':p.run_id,'status':p.status,'proposal':p.proposal}
                for p in s.scalars(select(Proposal)).all() if p.proposal['kind']=='experiment']

    @app.get('/api/proposals/{proposal_id}/experiment')
    def experiment_seed(proposal_id: str):
        from .experiments import seed
        return seed(service, proposal_id)

    @app.post('/api/proposals/{proposal_id}/prepare-experiment')
    def prepare_experiment(proposal_id: str, body: dict):
        from .experiments import prepare
        return prepare(service, proposal_id, body)

    @app.get('/api/experiments/{experiment_id}')
    def experiment_result(experiment_id: str):
        from .experiments import results
        return results(service, experiment_id)

    @app.post('/api/experiments/{experiment_id}/approve')
    def approve_experiment(experiment_id: str):
        from .experiments import approve
        return approve(service, experiment_id)

    @app.post('/api/proposals/{proposal_id}/{action}')
    def review(proposal_id: str, action: str, body: dict | None = None):
        from .proposals import review
        return review(service, proposal_id, action, (body or {}).get('cases'))

    @app.get('/api/providers/ucloud/models')
    async def models():
        from .providers import UCloud
        return await UCloud().models()

    @app.get('/api/demo')
    def demo():
        from .fixtures import demo_definition
        return demo_definition()

    dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
    if dist.exists():
        app.mount('/', StaticFiles(directory=dist, html=True), name='dashboard')
    return app
