"""Measured (per-track) wait time behind checkout queues.

The rule-based estimator in :mod:`ml.queue.wait_time` answers "how long should
a shopper wait for a queue of length L" via queuing-theory arithmetic. This
module answers a *different* question from anonymous tracking data:

    "for the people who actually stood in this queue polygon, how long did each
    one really wait?"

It records the timestamp when an anonymous track-id first appears inside a
queue zone and finalizes the dwell (in seconds) when that id leaves either the
zone or the frame. Completed dwells accumulate per queue so we can report a
real distribution: count, mean, median, max, total minutes.

Limitations (documented, not hidden):
  * The tracker's anonymous ids may be recycled after a track is lost, so a
    single id can represent two different people. To reduce that artifact we
    clamp dwells at ``max_track_dwell_seconds`` (default one hour).
  * Dwell granularity is the analytics sampling rate: a "wait" ends on the
    first *observed* frame where the person is no longer in the zone.
  * This is measured *from the anonymous track stream*, never from faces.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

from app.schemas.models import Track, ZoneEvent
from ml.geometry import point_in_polygon


def _stats(dwells: List[float]) -> Dict[str, float]:
    if not dwells:
        return {"count": 0, "avg_wait_minutes": 0.0, "median_wait_minutes": 0.0,
                "max_wait_minutes": 0.0, "min_wait_minutes": 0.0,
                "total_wait_minutes": 0.0}
    ordered = sorted(dwells)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": n,
        "avg_wait_minutes": round(sum(dwells) / n / 60.0, 2),
        "median_wait_minutes": round(median / 60.0, 2),
        "max_wait_minutes": round(ordered[-1] / 60.0, 2),
        "min_wait_minutes": round(ordered[0] / 60.0, 2),
        "total_wait_minutes": round(sum(dwells) / 60.0, 2),
    }


class QueueWaitMeasurer:
    """Tracks how long anonymous track-ids stay inside each queue zone."""

    def __init__(self, queue_zones: Dict[str, List], settings: Dict, camera_id: str):
        self.zones = queue_zones
        self.camera_id = camera_id
        self.max_dwell = float(settings.get("max_track_dwell_seconds", 3600))
        self.window = int(settings.get("wait_history", 200))
        self._active: Dict[str, Dict[int, float]] = {
            z: {} for z in self.zones
        }
        self._completed: Dict[str, Deque[Tuple[float, float]]] = {
            z: deque(maxlen=self.window) for z in self.zones
        }

    def update(self, tracks: List[Track], now: float) -> List[ZoneEvent]:
        """Record zone entries/exits and emit ``queue_wait`` events on exit.

        ``now`` is the frame timestamp in unix seconds.
        """
        visible_ids = {t.id for t in tracks}
        inside: Dict[str, set] = {}
        for zname, poly in self.zones.items():
            inside[zname] = {t.id for t in tracks
                             if point_in_polygon(t.center, poly)}

        events: List[ZoneEvent] = []
        for zname in self.zones:
            active = self._active[zname]
            cur = inside[zname]
            # New arrivals into the queue zone.
            for tid in cur - set(active):
                active[tid] = now
            # Anyone who was inside and is no longer -> finalize their dwell.
            for tid in list(active):
                if tid not in cur:
                    entry = active.pop(tid)
                    dwell = max(0.0, now - entry)
                    if dwell <= self.max_dwell:
                        self._completed[zname].append((entry, dwell))
                        events.append(ZoneEvent(
                            now, self.camera_id, "queue_wait", round(dwell / 60.0, 2),
                            {"queue": zname, "track_id": int(tid),
                             "measured": True}))
        return events

    def stats(self, queue_id: str) -> Dict[str, float]:
        """Aggregate statistics of *completed* waits for one queue (minutes)."""
        dwells = [d for _, d in self._completed.get(queue_id, [])]
        return _stats(dwells)

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        return {q: self.stats(q) for q in self.zones}

    def active_in_zone(self, queue_id: str) -> int:
        return len(self._active.get(queue_id, {}))

    def reset(self) -> None:
        self._active = {z: {} for z in self.zones}
        self._completed = {z: deque(maxlen=self.window) for z in self.zones}