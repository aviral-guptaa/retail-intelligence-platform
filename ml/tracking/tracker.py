"""Person tracking with persistent anonymous ids.

Greedy IoU association (SORT-style) so tracking works with any box source
(demo simulator, YOLO, ONNX) with no extra dependencies. An Ultralytics /
ByteTrack backend can be swapped in behind the same :class:`Tracker` interface
(see :mod:`ml.tracking.bytetrack` and :func:`ml.tracking.factory.create_tracker`).
"""
from __future__ import annotations

from itertools import count
from typing import Dict, List, Optional

import numpy as np

from app.schemas.models import Detection, Track
from ml.geometry import iou
from ml.tracking.base import BaseTracker

PERSON_CLASS = 0


class Tracker(BaseTracker):
    """Maps per-frame person detections to stable, anonymous temporary ids."""

    def __init__(self, tracking_settings: Dict, camera_id: str):
        super().__init__(tracking_settings, camera_id)
        self.max_age = int(tracking_settings.get("max_age_frames", 30))
        self.min_hits = int(tracking_settings.get("min_hits", 1))
        self.match_threshold = float(tracking_settings.get("iou_match_threshold", 0.30))
        self.maxlen = int(tracking_settings.get("track_buffer_points", 1200))
        self._tracks: Dict[int, Track] = {}
        self._next_id = count(1)

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """Associate detections to tracks, mint new ids, age out stale tracks.

        Association is greedy by IoU: the globally best (track, detection) pair
        is claimed first, then the next best, and so on. Unmatched detections
        create new tracks; unmatched tracks increment their missed counter and
        are dropped once it exceeds ``max_age``.
        """
        detections = [d for d in detections if d.class_id == PERSON_CLASS]

        # 1) Age every existing track (matched tracks reset below).
        for tr in self._tracks.values():
            tr.missed_frames += 1

        if not detections:
            self._prune()
            return list(self._tracks.values())

        # 2) Build & sort candidate (track, detection) pairs by IoU.
        candidates = [
            (iou(tr.bbox(), detections[di].bbox()), tid, di)
            for tid, tr in self._tracks.items()
            for di in range(len(detections))
        ]
        candidates.sort(key=lambda c: c[0], reverse=True)

        used_tracks: set = set()
        unmatched = set(range(len(detections)))

        # 3) Greedy matching.
        for _, tid, di in candidates:
            if tid in used_tracks or di not in unmatched or _ < self.match_threshold:
                continue
            used_tracks.add(tid)
            unmatched.discard(di)
            det = detections[di]
            tr = self._tracks[tid]
            tr.x1, tr.y1, tr.x2, tr.y2 = det.x1, det.y1, det.x2, det.y2
            tr.confidence = det.confidence
            tr.ts = det.ts
            tr.hit_streak += 1
            tr.missed_frames = 0
            tr.record_position(self.maxlen)

        # 4) New tracks for unmatched detections.
        for di in unmatched:
            det = detections[di]
            tr = Track(
                id=next(self._next_id), x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                confidence=det.confidence, class_id=det.class_id,
                class_name=det.class_name, ts=det.ts, hit_streak=1,
            )
            tr.record_position(self.maxlen)
            self._tracks[tr.id] = tr

        # 5) Drop stale tracks, then surface confirmed ones.
        self._prune()
        return [t for t in self._tracks.values() if t.hit_streak >= self.min_hits]

    def _prune(self) -> None:
        self._tracks = {
            tid: tr for tid, tr in self._tracks.items() if tr.missed_frames < self.max_age
        }

    def active_ids(self) -> List[int]:
        return sorted(self._tracks.keys())

    def reset(self) -> None:
        self._tracks.clear()


IoUTracker = Tracker  # explicit alias so callers can name the backend plainly