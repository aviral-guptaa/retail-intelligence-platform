"""Footfall analytics: time-bucketed entry/exit counts and live occupancy."""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List

from app.schemas.models import ZoneEvent


class FootfallCounter:
    """Rolling footfall statistics, bucketed per minute.

    The pipeline feeds :meth:`update` with entry/exit events plus the number of
    people currently on the floor; this class maintains cumulative totals and a
    per-minute series that the API/dashboard can render.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.total_entries = 0
        self.total_exits = 0
        self.current_active = 0
        self._buckets: Dict[int, Dict[str, int]] = {}
        self._minutes: Deque[Dict[str, float]] = deque(maxlen=240)

    def update(self, events: List[ZoneEvent], active_count: int) -> None:
        for ev in events:
            if ev.event_type == "entry":
                self.total_entries += 1
            elif ev.event_type == "exit":
                self.total_exits += 1
        self.current_active = active_count
        now = time.time()
        bucket = int(now // 60)
        entry = self._buckets.setdefault(bucket, {"entry": 0, "exit": 0, "ts_min": bucket * 60})
        for ev in events:
            if ev.event_type == "entry":
                entry["entry"] += 1
            elif ev.event_type == "exit":
                entry["exit"] += 1

    def snapshot(self) -> Dict[str, float]:
        return {
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "current_active": self.current_active,
            "net_footfall": self.total_entries - self.total_exits,
            "camera_id": self.camera_id,
        }

    def series(self, minutes: int = 60) -> List[Dict[str, float]]:
        """Return entry/exit counts for the last N minute buckets."""
        now = int(time.time() // 60)
        out = []
        for b in range(now - minutes, now + 1):
            data = self._buckets.get(b, {"entry": 0, "exit": 0})
            out.append({"ts": b * 60, "entry": data["entry"], "exit": data["exit"]})
        return out