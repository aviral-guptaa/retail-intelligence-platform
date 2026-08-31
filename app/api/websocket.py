"""WebSocket hub: pushes live snapshots to every connected dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketHub:
    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("ws client connected (%d)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("ws client disconnected (%d)", len(self._connections))

    async def send(self, ws: WebSocket, message: Dict[str, Any]) -> None:
        await ws.send_text(json.dumps(message))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        if not self._connections:
            return
        payload = json.dumps(message)
        dead = []
        async with self._lock:
            clients = list(self._connections)
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


# Singleton used by both the API layer and the inference pipeline.
hub = WebSocketHub()