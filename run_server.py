#!/usr/bin/env python3
"""Start the mono3d-recon web server."""

import argparse
from src.server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mono3d-recon web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, debug=args.debug)
