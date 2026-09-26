"""Serve the dependency-free review interface and its cached demo runs."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        "AgentOptim static demo: "
        f"http://127.0.0.1:{args.port}/static/index.html"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
