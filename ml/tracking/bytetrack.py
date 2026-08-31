"""ByteTrack tracking backend built on Ultralytics.

Ultralytics ships a ByteTrack re-implementation in its engine; we drive it
through ``YOLO.track(...)`` so the *same* checkpoint produces both detections
and tracks. This only runs for real cameras/video (a YOLO model + ultralytics
are required); demo mode keeps using the pure-IoU tracker.

ByteTrack handles occlusions and re-IDs people more reliably at checkout
bottlenecks than the pure-IoU fallback.
"""
from __future__ import annotations

import logging
from itertools import count
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.models import Detection, Track
from ml.tracking.base import BaseTracker

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


def _id_of(box) -> Optional[int]:
    """Extract the per-box track id assigned by ByteTrack (or None)."""
    if box.id is None:
        return None
    try:
        return int(box.id[0])
    except (TypeError, ValueError):
        return None


def _results_to_tracks(results, mint_id) -> List[Track]:
    """Map ultralytics track results to Track objects, preserving ByteTrack ids."""
    tracks: List[Track] = []
    if results is None:
        return tracks
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        names = r.names or {}
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != PERSON_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = _id_of(box)
            if track_id is None:
                track_id = next(mint_id)
            tracks.append(Track(
                id=track_id, x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(box.conf[0]), class_id=cls_id,
                class_name=str(names.get(cls_id, cls_id)),
                ts=float(__import__("time").time()), hit_streak=1,
            ))
    return tracks


class ByteTrackTracker(BaseTracker):
    """Ultralytics ByteTrack wrapper reusing the YOLO detector's model.

    Expects a :class:`ml.detection.yolo_detector.YoloDetector` whose ``model``
    attribute is a live Ultralytics YOLO instance. Every frame is passed to
    ``model.track(frame, persist=True, tracker='bytetrack.yaml', ...)``.
    """

    def __init__(self, tracking_settings: Dict[str, Any], camera_id: str, detector=None):
        super().__init__(tracking_settings, camera_id)
        self.detector = detector
        self.name = "bytetrack"
        self.min_hits = int(tracking_settings.get("min_hits", 1))
        self._mint = count(1)
        self._ready = detector is not None and getattr(detector, "model", None) is not None

    def _model_cfg(self) -> Dict[str, Any]:
        if self.detector is None:
            return {}
        return {
            "conf": self.detector.conf,
            "iou": self.detector.iou,
            "imgsz": self.detector.imgsz,
            "device": self.detector.device,
            "classes": self.detector.classes,
        }

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """Run ByteTrack on ``frame`` (detections ignored - the model re-detects).

        Returns Track objects; if the backend is unavailable it returns an empty
        list so the caller's fallback takes over gracefully.
        """
        if frame is None or self.detector is None or self.detector.model is None:
            if not self._ready:
                self._ready = self.detector is not None and getattr(self.detector, "model", None) is not None
            return []
        try:
            results = self.detector.model.track(
                frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                **self._model_cfg())
            tracks = _results_to_tracks(results, self._mint)
            return [t for t in tracks if t.hit_streak >= self.min_hits]
        except Exception as exc:  # tracker config missing, bad frame, etc.
            logger.warning("bytetrack failed on frame: %s", exc)
            return []

    def active_ids(self) -> List[int]:
        return []

    def reset(self) -> None:
        self._mint = count(1)