"""Web dashboard backend for the SIH Retail Intelligence prototype.

Serves a static dashboard and lets a user start an analysis run by either
uploading a video file or launching the demo simulator. The full analytics
pipeline runs server-side; snapshots stream over a WebSocket and are also
available via REST (the same endpoints the CLI app exposes).

Run:
  python run_web.py            # http://<host>:<port>/
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

from app.api.routes import mjpeg_frames  # noqa: E402
from app.services.inference_service import InferencePipeline  # noqa: E402
from app.services.analytics_service import AnalyticsService   # noqa: E402
from app.api.websocket import hub                             # noqa: E402
from config.loader import load_settings                       # noqa: E402
from database.repository import Repository                     # noqa: E402

logger = logging.getLogger("webserver")

BROWSER_CODECS = {"h264", "avc1", "vp9", "av1", "hev1", "hvc1"}


def _probe_video_codec(path: Path) -> Optional[str]:
    """Return the codec name of the first video stream, or None if unsure."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    codec = (proc.stdout or "").strip().splitlines()
    return codec[0] if codec else None


def _transcode_for_web(src: Path) -> Path:
    """If the uploaded file uses a codec browsers can't play (e.g. mpeg4/Xvid),
    transcode it to H.264 MP4 so the dashboard's <video> can replay it. Falls
    back to the original file when ffmpeg is missing or anything fails."""
    try:
        codec = _probe_video_codec(src)
        if codec and codec.lower() in BROWSER_CODECS:
            return src
        web_dest = src.with_name(f"{src.stem}.web.mp4")
        if web_dest.is_file() and web_dest.stat().st_size > 0:
            return web_dest
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return src
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", str(web_dest)],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            logger.warning("transcode failed: %s", proc.stderr[-500:])
            return src
        return web_dest
    except Exception as exc:
        logger.warning("transcode error: %s", exc)
        return src


class RunManager:
    """Owns the current pipeline and supports swapping in a new video/demo run."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.pipeline: Optional[InferencePipeline] = None
        self.run_meta: Dict[str, Any] = {
            "id": None, "mode": None, "source": None, "started_ts": None,
            "finished": False, "error": None,
        }
        self._lock = threading.Lock()

    def _new_pipeline(self) -> InferencePipeline:
        repo = Repository(None)
        return InferencePipeline(self.settings, repo, db_writer=None, display=False)

    def stop_current(self) -> None:
        with self._lock:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as exc:
                    logger.warning("stop current run: %s", exc)
                self.pipeline = None

    def start(self, mode: str, source: Optional[str] = None,
              camera_id: str = "store_01", source_web: Optional[str] = None) -> Dict[str, Any]:
        self.stop_current()
        run_id = uuid.uuid4().hex[:12]

        def _on_done() -> None:
            # Set finished when a video source reaches its end.
            while True:
                p: Optional[InferencePipeline] = None
                with self._lock:
                    p = self.pipeline
                if p is None:
                    return
                svc = p.services.get(camera_id)
                finished = svc is not None and getattr(svc.source, "finished", False)
                if finished:
                    self.run_meta["finished"] = True
                    return
                time.sleep(0.5)

        try:
            if mode == "demo":
                pipeline = self._new_pipeline()
                with self._lock:
                    self.pipeline = pipeline
                pipeline.add_camera(camera_id, mode="demo")
                self.run_meta = {"id": run_id, "mode": "demo", "source": "simulator",
                                 "started_ts": time.time(), "finished": False, "error": None}
                threading.Thread(target=pipeline.start, daemon=True).start()
                threading.Thread(target=_on_done, daemon=True).start()
                return self.run_meta
            elif mode == "video":
                pipeline = self._new_pipeline()
                with self._lock:
                    self.pipeline = pipeline
                pipeline.add_camera(camera_id, mode="video", source=source)
                self.run_meta = {"id": run_id, "mode": "video", "source": source,
                                 "source_web": source_web,
                                 "started_ts": time.time(), "finished": False, "error": None}
                threading.Thread(target=pipeline.start, daemon=True).start()
                threading.Thread(target=_on_done, daemon=True).start()
                return self.run_meta
            else:
                raise ValueError(f"unknown mode: {mode}")
        except Exception as exc:
            logger.exception("start run %s failed", mode)
            self.run_meta["error"] = str(exc)
            raise


_app: Optional[FastAPI] = None


def create_web_app(settings: Optional[Dict[str, Any]] = None) -> FastAPI:
    global _app
    settings = settings or load_settings()
    manager = RunManager(settings)

    app = FastAPI(title="SIH Retail Intelligence Dashboard", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.manager = manager
    app.state.settings = settings

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # -------------------------------------------------------------- run control
    @app.post("/api/run/demo")
    def run_demo() -> Dict[str, Any]:
        meta = manager.start("demo")
        return {"status": "started", "run": meta}

    @app.post("/api/run/video")
    async def run_video(file: UploadFile = File(...)) -> Dict[str, Any]:
        # sanity: accept common video containers
        name = (file.filename or "upload").lower()
        if not name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
            raise HTTPException(415, "Unsupported video type; use mp4/mov/avi/mkv/webm/m4v")
        if file.size and file.size > 2 * 1024 * 1024 * 1024:  # 2GB guard
            raise HTTPException(413, "File too large")

        out_dir = ROOT / "data" / "uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{uuid.uuid4().hex}.{name.rsplit('.', 1)[-1]}"
        try:
            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer, length=1 << 20)
        finally:
            await file.close()
        web_dest = _transcode_for_web(dest)
        meta = manager.start("video", source=str(dest), source_web=str(web_dest))
        return {"status": "started", "run": meta}

    @app.post("/api/run/stop")
    def run_stop() -> Dict[str, Any]:
        manager.stop_current()
        manager.run_meta["finished"] = True
        return {"status": "stopped"}

    @app.get("/api/run/status")
    def run_status() -> Dict[str, Any]:
        p = manager.pipeline
        svc: Optional[AnalyticsService] = None
        if p and p.services:
            svc = next(iter(p.services.values()))
        return {
            "run": manager.run_meta,
            "running": p is not None and getattr(p, "_running", False),
            "live": svc is not None,
            "uptime_s": round(time.time() - manager.run_meta["started_ts"], 1)
                        if manager.run_meta["started_ts"] else 0,
        }

    # ----------------------------------------------------------- analytics (live)
    @app.get("/api/analytics/current")
    def analytics_current() -> Dict[str, Any]:
        p = manager.pipeline
        if not p or not p.services:
            return {"live": False}
        return {"live": True, **next(iter(p.services.values())).current()}

    @app.get("/api/run/media")
    def run_media() -> FileResponse:
        """Stream the currently-running uploaded video so the dashboard can show
        the actual footage (transcoded to a browser-playable codec when needed)."""
        source = manager.run_meta.get("source_web") or manager.run_meta.get("source")
        if manager.run_meta.get("mode") != "video" or not source or not Path(source).is_file():
            raise HTTPException(404, "no uploaded video in this run")
        return FileResponse(str(Path(source)), media_type="video/mp4", filename=Path(source).name)

    @app.get("/api/feedback")
    def feedback() -> Dict[str, Any]:
        p = manager.pipeline
        if not p or not p.services:
            return {"prediction_monitoring": {}, "alerts": []}
        svc = next(iter(p.services.values()))
        return {
            "prediction_monitoring": svc.health().get("prediction_monitoring", {}),
            "recommendation": svc.current().get("queues", {}).get("recommendation_detail"),
        }

    @app.get("/api/heatmap.png")
    def heatmap() -> Response:
        p = manager.pipeline
        if not p or not p.services:
            raise HTTPException(404, "no live run")
        svc = next(iter(p.services.values()))
        img = svc.heatmap_image()
        try:
            import cv2
            ok, buf = cv2.imencode(".png", img)
            if ok:
                return Response(content=buf.tobytes(), media_type="image/png")
        except Exception:
            pass
        return Response(content=img.tobytes(), media_type="application/octet-stream")

    @app.get("/video_stream")
    def video_stream(max_frames: Optional[int] = None) -> Response:
        """MJPEG push stream of the live, de-identified dashboard frame."""
        p = manager.pipeline
        if not p or not p.services:
            raise HTTPException(404, "no live run")
        svc = next(iter(p.services.values()))
        return StreamingResponse(
            mjpeg_frames(svc, max_width=960, max_frames=max_frames),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ------------------------------------------------------------------ websocket
    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        try:
            await hub.send(websocket, {"type": "welcome",
                                       "data": settings.get("app", {})})
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await hub.send(websocket, {"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(websocket)

    # background broadcaster (mirrors create_app in app/api/__init__.py)
    import asyncio

    async def _broadcast_loop() -> None:
        while True:
            await asyncio.sleep(1.0)
            p = manager.pipeline
            try:
                if p is not None and getattr(p, "_running", False):
                    await hub.broadcast({
                        "type": "live_snapshot",
                        "ts": time.time(),
                        "ts_epoch": time.time(),
                        "data": p.live_snapshot(),
                    })
            except Exception:
                logger.exception("broadcast failed")

    @app.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_broadcast_loop())

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    _app = app
    return app


def standalone_run() -> None:
    """Run like a CLI entry point (used by run_web.py)."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    settings = load_settings()
    app = create_web_app(settings)
    host = settings["api"]["host"]
    port = settings["api"]["port"]
    print(f"\n  Retail Intelligence dashboard -> http://{host}:{port}\n"
          f"  Live WS: ws://{host}:{port}/ws/live")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    standalone_run()
