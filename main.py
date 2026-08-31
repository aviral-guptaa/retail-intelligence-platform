"""SIH 2026 - AI-Powered Retail Intelligence Platform.

Command-line entry point. Three modes:

  python main.py --mode demo            synthetic store (no camera/GPU needed)
  python main.py --mode video --source clip.mp4
  python main.py --mode live  --source 0
  python main.py --mode live  --source rtsp://user:pass@host:554/stream

Video/live run REAL inference through YoloDetector + the configured tracker
(ByteTrack when a YOLO checkpoint is present, IoU otherwise) with automatic
reconnect for flaky feeds - no synthetic detection in these modes.

The FastAPI dashboard + WebSocket feed starts automatically unless --no-api.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.services.inference_service import InferencePipeline
from config.loader import env, load_settings
from database.models import build_session, build_session_factory
from database.repository import Repository
from database.writer import BackgroundWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["demo", "video", "live"], default="demo",
                   help="demo = synthetic store, video = file, live = camera/RTSP")
    p.add_argument("--source", default=None,
                   help="video file path or RTSP URL or camera index (for video/live)")
    p.add_argument("--camera-id", default="store_01")
    p.add_argument("--display", action="store_true", help="show an OpenCV overlay window")
    p.add_argument("--no-api", action="store_true", help="run the analytics loop only")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--run-seconds", type=float, default=None,
                   help="stop after N seconds of demo (auto-stops API too)")
    p.add_argument("--logger", default=None)
    return p.parse_args()


def _resolve_source(args: argparse.Namespace):
    """Validate/normalise --source for video/live into an openable handle."""
    if args.mode == "demo":
        return None, None
    src = args.source if args.source is not None else 0
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    if args.mode == "video":
        is_rtsp = isinstance(src, str) and src.lower().startswith("rtsp://")
        if isinstance(src, (int, float)):
            logging.getLogger(__name__).error(
                "--mode video requires a file path (got a camera index); use --mode live for cameras")
            raise SystemExit(2)
        path = Path(src)
        if not path.is_file():
            logging.getLogger(__name__).error("video file not found: %s", path)
            raise SystemExit(2)
        return "video", str(path)
    # live: could be an index or an RTSP url
    if isinstance(src, int) and src < 0:
        logging.getLogger(__name__).error("invalid webcam index: %s", src)
        raise SystemExit(2)
    return "live", src


def main() -> int:
    args = parse_args()
    settings = load_settings()
    if args.run_seconds is not None:
        settings["demo"]["duration_seconds"] = args.run_seconds
        settings["demo"]["enabled"] = True
    if args.logger:
        settings["app"]["log_level"] = args.logger

    logging.basicConfig(
        level=getattr(logging, settings["app"]["log_level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    database_url = env("DATABASE_URL", "sqlite:///data/retail_intelligence.db")
    session = None
    writer = None
    try:
        _, session = build_session(database_url)
        writer = BackgroundWriter(build_session_factory(database_url),
                                  flush_interval=float(
                                      settings.get("database", {}).get("flush_interval_seconds", 2)))
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning("database unavailable, continuing in-memory: %s", exc)
    repo = Repository(session)

    pipeline = InferencePipeline(settings, repo, db_writer=writer, display=args.display)

    if args.mode == "demo":
        pipeline.add_camera(args.camera_id, mode="demo")
    else:
        mode, src = _resolve_source(args)
        pipeline.add_camera(args.camera_id, mode=mode, source=src)

    pipeline.start()

    if args.no_api:
        try:
            while not pipeline._stop_event.is_set():
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        pipeline.stop()
        return 0

    import threading

    import uvicorn

    from app.api import create_app
    from uvicorn import Config, Server

    app = create_app(pipeline)
    host = args.host or settings["api"]["host"]
    port = args.port or settings["api"]["port"]
    print(f"\n  Retail intelligence running -> http://{host}:{port}\n"
          f"  Live WS: ws://{host}:{port}/ws/live")
    server = Server(Config(app, host=host, port=port, log_level="warning"))

    def _watchdog():
        # Demo/video sources signal completion; shut uvicorn down so the CLI exits.
        pipeline._stop_event.wait()
        server.should_exit = True

    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())