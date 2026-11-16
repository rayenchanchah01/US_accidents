"""Open the completed dashboard locally without installing a web framework."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from accidents.paths import DASHBOARD
import argparse
import webbrowser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    dist = DASHBOARD
    if not (dist / 'data' / 'summary.json').exists():
        parser.error('Results are missing. Run python -m accidents.export after training.')
    handler = partial(SimpleHTTPRequestHandler, directory=str(dist))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        url = f'http://127.0.0.1:{args.port}'
        print(f'IMPACT dashboard: {url}\nPress Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
