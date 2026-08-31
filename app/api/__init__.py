"""FastAPI application factory + live WebSocket endpoint / broadcaster."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.websocket import hub
from app.services.inference_service import InferencePipeline
from config.loader import load_settings

logger = logging.getLogger(__name__)


def create_app(pipeline: InferencePipeline) -> FastAPI:
    app = FastAPI(title="SIH Retail Intelligence", version="0.1.0")
    settings = load_settings()
    app.state.pipeline = pipeline
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get("api", {}).get("cors_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    async def _broadcast_loop() -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                await hub.broadcast({
                    "type": "live_snapshot",
                    "ts": __import__("datetime").datetime.now().isoformat(),
                    "ts_epoch": __import__("time").time(),
                    "data": pipeline.live_snapshot(),
                })
            except Exception:
                logger.exception("broadcast failed")

    @app.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_broadcast_loop())

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        try:
            await hub.send(websocket, {"type": "welcome", "data": settings.get("app", {})})
            while True:
                await websocket.receive_text()  # ping/pong keepalive from client
                await hub.send(websocket, {"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(websocket)

    return app