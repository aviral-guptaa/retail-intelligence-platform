"""Entry point for the SIH Retail Intelligence web dashboard.

Runs the FastAPI dashboard that serves the webpage and runs the analytics
pipeline in the background for an uploaded video or the demo simulator.

Usage:
  python run_web.py [--host HOST] [--port PORT]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.loader import load_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="SIH Retail Intelligence web dashboard")
    ap.add_argument("--host", type=str, default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    settings = load_settings()
    if args.host:
        settings["api"]["host"] = args.host
    if args.port:
        settings["api"]["port"] = args.port

    from webserver.app import create_web_app
    import uvicorn

    app = create_web_app(settings)
    host = settings["api"]["host"]
    port = settings["api"]["port"]
    print(f"\n  Retail Intelligence dashboard -> http://{host}:{port}\n"
          f"  Live WS: ws://{host}:{port}/ws/live")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
