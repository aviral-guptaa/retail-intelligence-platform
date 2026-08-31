"""Directional entry/exit counting using a configurable virtual line."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from app.schemas.models import Detection, Track, ZoneEvent
from ml.geometry import segments_cross


class LineCounter:
    """Counts store entries/exits across a virtual line.

    For every tracked person we remember their previous center point. When the
    segment joining the previous and current center crosses the virtual line we
    record one crossing; the crossing direction decides entry vs exit.

    Anti double-count: a per-track cooldown (in frames) prevents one person
    whose bounding box wobbles over the line from being counted repeatedly - a
    new crossing for that track is only accepted once ``cooldown_frames`` have
    passed since their last counted crossing.

    The store-layout convention for which direction is an *entry* is
    configurable via ``entry_direction``:
      - "down" (default): moving downward on screen enters the floor;
      - "up": moving upward on screen enters the floor;
      - otherwise: raw traversal sign, whatever it is.

    Occupancy is derived as ``max(0, entries - exits)`` for store-level analytics.
    """

    def __init__(self, line_start: Tuple[float, float], line_end: Tuple[float, float],
                 camera_id: str, settings: Optional[Dict] = None):
        self.line_start = np.asarray(line_start, dtype=float)
        self.line_end = np.asarray(line_end, dtype=float)
        self.camera_id = camera_id
        self.settings = settings or {}
        self.cooldown_frames = max(0, int(self.settings.get("cooldown_frames", 0)))
        self.entry_direction = str(self.settings.get("entry_direction", "down")).lower()
        self.entries = 0
        self.exits = 0
        self.crossings = 0
        self._frame_no = 0
        self._last_center: Dict[int, np.ndarray] = {}
        self._last_crossings: Dict[int, int] = {}

    def _is_entry(self, cur: np.ndarray, prev: np.ndarray) -> bool:
        dy = cur[1] - prev[1]
        if self.entry_direction in ("up", "enter_up"):
            return dy < 0
        return dy >= 0  # default convention: "down"

    def update(self, tracks: List[Track]) -> List[ZoneEvent]:
        events: List[ZoneEvent] = []
        for tr in tracks:
            tid = tr.id if isinstance(tr, Track) else tr.track_id
            if tid is None:
                continue
            cur = tr.center
            prev = self._last_center.get(tid)
            if prev is not None and segments_cross(prev, cur, self.line_start, self.line_end):
                in_cooldown = (self.cooldown_frames > 0
                               and self._last_crossings.get(tid, -10 ** 9)
                               >= self._frame_no - self.cooldown_frames)
                if not in_cooldown:
                    self.crossings += 1
                    self._last_crossings[tid] = self._frame_no
                    if self._is_entry(cur, prev):
                        self.entries += 1
                        events.append(ZoneEvent(tr.ts, self.camera_id, "entry", 1.0,
                                                {"track_id": tid}))
                    else:
                        self.exits += 1
                        events.append(ZoneEvent(tr.ts, self.camera_id, "exit", 1.0,
                                                {"track_id": tid}))
            self._last_center[tid] = cur

        # Drop memory for tracks no longer present (prevents unbounded growth).
        active = {tr.id if isinstance(tr, Track) else tr.track_id for tr in tracks}
        for tid in list(self._last_center):
            if tid not in active:
                del self._last_center[tid]
        self._frame_no += 1
        return events

    def occupancy(self) -> int:
        """Store-level occupancy derived from net crossings, clamped at 0."""
        return max(0, self.entries - self.exits)

    def reset(self) -> None:
        self.entries = 0
        self.exits = 0
        self.crossings = 0
        self._frame_no = 0
        self._last_center.clear()
        self._last_crossings.clear()