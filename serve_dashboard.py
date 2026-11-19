"""Open the completed dashboard locally without installing a web framework."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from accidents.paths import DASHBOARD
import argparse
import json
import logging
from urllib.parse import urlsplit
import webbrowser


class DashboardHandler(SimpleHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path == '/api/prediction-schema':
            self.infer(schema=True)
        else:
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != '/api/predict':
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
        self.infer(payload=payload)

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
    dist = DASHBOARD
    if not (dist / 'data' / 'summary.json').exists():
        parser.error('Results are missing. Run python -m accidents.export after training.')
    handler = partial(DashboardHandler, directory=str(dist))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        url = f'http://127.0.0.1:{args.port}'
        print(f'IMPACT dashboard: {url}\nPress Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
