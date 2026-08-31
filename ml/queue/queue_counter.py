"""Queue zone detection and queue-length metrics."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from app.schemas.models import Track, ZoneEvent
from ml.geometry import point_in_polygon


class QueueCounter:
    """Counts shoppers inside each checkout queue polygon.

    Queue length must react to real crossings of the zone boundary, so the
    counter keeps a per-zone history array that the predictor and dashboard can
    consume. History samples are throttled by ``sample_interval_seconds``.
    """

    def __init__(self, queue_zones: Dict[str, List[np.ndarray]], settings: Dict, camera_id: str):
        self.zones = queue_zones
        self.camera_id = camera_id
        self.sample_interval = float(settings.get("sample_interval_seconds", 5))
        self.history_window = int(settings.get("history_window", 60))
        self._history: Dict[str, Deque[Tuple[float, int]]] = {
            z: deque(maxlen=self.history_window) for z in self.zones
        }
        self._last_sampled: Dict[str, float] = {}
        self._last_counts: Dict[str, int] = {}

    def update(self, tracks: List[Track], now: float) -> List[ZoneEvent]:
        """Return live per-zone queue counts and record throttled history."""
        events: List[ZoneEvent] = []
        for zname, poly in self.zones.items():
            count = sum(1 for tr in tracks if point_in_polygon(tr.center, poly))
            changed = self._last_counts.get(zname, -1) != count
            self._last_counts[zname] = count
            last = self._last_sampled.get(zname, 0.0)
            if now - last >= self.sample_interval:
                self._history[zname].append((now, count))
                self._last_sampled[zname] = now
            if changed:
                events.append(ZoneEvent(now, self.camera_id, "queue_change", float(count),
                                        {"queue": zname}))
        return events

    def counts(self) -> Dict[str, int]:
        return dict(self._last_counts)

    def total_queued(self) -> int:
        return sum(self._last_counts.values())

    def history(self) -> Dict[str, List[Tuple[float, int]]]:
        return {z: list(h) for z, h in self._history.items()}

    def growth_rate(self, queue_id: str, window: int = 10) -> float:
        """Linear slope (shoppers/second recently) for one queue."""
        hist = list(self._history.get(queue_id, []))
        if len(hist) < 2:
            return 0.0
        tail = hist[-window:]
        ts = np.array([t for t, _ in tail], dtype=float)
        vals = np.array([v for _, v in tail], dtype=float)
        if ts.max() == ts.min():
            return 0.0
        slope = np.polyfit(ts - ts[0], vals, 1)[0]
        return float(slope)

    def reset(self) -> None:
        self._history.clear()
        self._last_counts.clear()
        self._last_sampled.clear()