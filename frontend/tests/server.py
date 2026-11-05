"""Owned disposable server and worker for browser tests."""
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='kineticeval-e2e-') as directory:
    env = {**os.environ, 'KINETICEVAL_DB': str(Path(directory)/'test.db'), 'PYTHONPATH': str(root/'backend'), 'UCLOUD_BASE_URL': 'http://127.0.0.1:9/v1', 'UCLOUD_API_KEY': 'test-only'}
    # Initialize once before launching independent processes.
    sys.path.insert(0, str(root/'backend'))
    from kineticeval.db import Database
    Database(env['KINETICEVAL_DB'])
    children = [subprocess.Popen([sys.executable, '-m', 'uvicorn', 'kineticeval.api:create_app', '--factory', '--host', '127.0.0.1', '--port', '8123'], env=env, cwd=root),
                subprocess.Popen([sys.executable, '-m', 'kineticeval.worker'], env=env, cwd=root)]
    def stop(*args):
        for child in children: child.terminate()
        for child in children:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while all(p.poll() is None for p in children): time.sleep(.5)
    finally:
        stop()
