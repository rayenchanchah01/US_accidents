"""Serve the dashboard and run the pipeline from it, without installing a web framework.

Besides static files and live prediction, the server accepts a dataset upload and runs
the pipeline stages (clean, train, test) as background jobs, one at a time.
"""
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import webbrowser

from accidents.paths import DASHBOARD, DATA, LOGS, MODELS, REPORTS, ROOT

UPLOADS = DATA / 'uploads'
ACTIVE = DATA / 'active_source.json'
JOB_LOG = LOGS / 'pipeline.log'
MAX_UPLOAD = 20 * 1024 ** 3  # 20 GB; the full archive is about 3 GB.
STAGES = ('clean', 'train', 'test')
_job = {'stage': None, 'process': None, 'started': None, 'finished': None, 'returncode': None}
_job_lock = threading.Lock()


def active_source():
    try:
        info = json.loads(ACTIVE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    path = Path(info['path'])
    return {**info, 'bytes': path.stat().st_size} if path.exists() else None


def mtime(path):
    return path.stat().st_mtime if path.exists() else None


def stage_status(source):
    """Which stages are up to date for the active dataset (cheap checks, no hashing)."""
    cleaning = REPORTS / 'cleaning.json'
    cleaned = False
    if source and cleaning.exists():
        audit = json.loads(cleaning.read_text(encoding='utf-8'))
        cleaned = audit.get('source_file') == Path(source['path']).name and audit.get('source_bytes') == source['bytes']
    state = MODELS / 'training_state.json'
    trained = cleaned and (mtime(state) or 0) >= mtime(cleaning) and (MODELS / 'severity.joblib').exists()
    summary = DASHBOARD / 'data' / 'summary.json'
    tested = trained and (mtime(summary) or 0) >= mtime(state)
    return {'clean': cleaned, 'train': trained, 'test': tested}


def validate_dataset(path):
    """Reject files that the cleaning step cannot read, before they replace the active dataset."""
    import pandas as pd
    from accidents.cleaning import clean_chunk
    try:
        sample = pd.read_csv(path, nrows=500, low_memory=False)
    except Exception as error:
        raise ValueError(f'The file could not be read as CSV: {error}') from None
    try:
        clean_chunk(sample)
    except KeyError as error:
        raise ValueError(f'The file is missing the column {error}. Upload a US Accidents CSV.') from None
    except ValueError as error:
        raise ValueError(f'{error}. Upload a US Accidents CSV.') from None
    return len(sample.columns)


def job_snapshot():
    with _job_lock:
        process = _job['process']
        running = process is not None and process.poll() is None
        if process is not None and not running and _job['finished'] is None:
            _job['returncode'] = process.returncode
            _job['finished'] = datetime.now().isoformat(timespec='seconds')
            # A new model or new results are on disk: the next prediction reloads them.
            from accidents.predict import _cached_predictor
            _cached_predictor.cache_clear()
        info = {k: _job[k] for k in ('stage', 'started', 'finished', 'returncode')}
    log = JOB_LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-40:] if JOB_LOG.exists() else []
    return {**info, 'running': running, 'log': log}


def start_job(stage):
    if stage not in STAGES:
        raise ValueError('Unknown pipeline stage.')
    source = active_source()
    if not source:
        raise ValueError('Upload a dataset first.')
    done = stage_status(source)
    required = {'train': 'clean', 'test': 'train'}.get(stage)
    if required and not done[required]:
        raise ValueError(f'Run the "{required}" stage for this dataset first.')
    with _job_lock:
        if _job['process'] is not None and _job['process'].poll() is None:
            raise ValueError(f'The "{_job["stage"]}" stage is still running.')
        LOGS.mkdir(parents=True, exist_ok=True)
        log = JOB_LOG.open('w', encoding='utf-8')
        command = [sys.executable, '-u', 'run_all.py', '--source', source['path'], '--stage', stage]
        _job.update(stage=stage, started=datetime.now().isoformat(timespec='seconds'), finished=None, returncode=None,
                    # UTF-8 output: model names contain '·', which Windows' default codepage mangles.
                    process=subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
        log.close()  # The child process keeps its own handle.


class DashboardHandler(SimpleHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # Results change after every pipeline run and the page code changes with the project:
        # always revalidate, so the browser never runs stale scripts or shows stale exports.
        if not self.path.startswith('/api/'):
            self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/prediction-schema':
            self.infer(schema=True)
        elif path == '/api/pipeline':
            source = active_source()
            self.send_json(200, {'source': source, 'stages': stage_status(source), 'job': job_snapshot()})
        else:
            super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == '/api/upload':
            self.upload()
            return
        if path not in ('/api/predict', '/api/run'):
            self.send_json(404, {'error': 'Unknown API endpoint.'})
            return
        if self.headers.get_content_type() != 'application/json':
            self.send_json(415, {'error': 'Send inputs as application/json.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                self.send_json(413, {'error': 'Send one incident (at most 16 KB).'})
                return
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self.send_json(400, {'error': 'The input must be valid JSON.'})
            return
        if path == '/api/run':
            try:
                start_job(payload.get('stage') if isinstance(payload, dict) else None)
            except ValueError as error:
                self.send_json(409, {'error': str(error)})
            else:
                self.send_json(202, {'job': job_snapshot()})
            return
        self.infer(payload=payload)

    def upload(self):
        name = Path(self.headers.get('X-Filename', '')).name
        if not re.fullmatch(r'[\w .()-]{1,120}\.(csv|zip)', name, re.IGNORECASE):
            self.send_json(400, {'error': 'Choose a .csv file (or a .zip holding one CSV).'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_UPLOAD:
            self.send_json(413, {'error': 'The file is empty or larger than 20 GB.'})
            return
        with _job_lock:
            busy = _job['process'] is not None and _job['process'].poll() is None
        if busy:
            self.send_json(409, {'error': 'Wait for the running stage to finish before replacing the dataset.'})
            return
        UPLOADS.mkdir(parents=True, exist_ok=True)
        part = UPLOADS / (name + '.part')
        try:
            with part.open('wb') as out:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise ConnectionError('The upload was interrupted.')
                    out.write(chunk); remaining -= len(chunk)
            columns = validate_dataset(part)
        except (ValueError, ConnectionError) as error:
            part.unlink(missing_ok=True)
            self.send_json(400, {'error': str(error)})
            return
        target = UPLOADS / name
        part.replace(target)
        ACTIVE.write_text(json.dumps({'path': str(target), 'uploaded': datetime.now().isoformat(timespec='seconds'),
                                      'columns': columns}), encoding='utf-8')
        self.send_json(200, {'source': active_source()})

    def infer(self, *, schema=False, payload=None):
        try:
            from accidents.predict import get_predictor
            predictor = get_predictor()
        except Exception:
            logging.exception('Prediction model could not be loaded')
            self.send_json(503, {'error': 'The saved model is unavailable. Check the local model files and Python dependencies, then retry.'})
            return
        try:
            result = predictor.schema() if schema else predictor.predict(payload)
        except ValueError as error:
            self.send_json(400, {'error': str(error)})
        except Exception:
            logging.exception('Prediction failed')
            self.send_json(500, {'error': 'The prediction could not be completed. Check the server log and retry.'})
        else:
            self.send_json(200, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    handler = partial(DashboardHandler, directory=str(DASHBOARD))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        url = f'http://127.0.0.1:{args.port}'
        print(f'IMPACT dashboard: {url}\nPress Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
