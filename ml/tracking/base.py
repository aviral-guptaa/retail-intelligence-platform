"""Tracking backend interface.

All trackers map per-frame person detections to stable, anonymous ids and
return a list of :class:`~app.schemas.models.Track`. Implementations live in
the same package: :class:`~ml.tracking.tracker.Tracker` (pure IoU, zero extra
deps) and :class:`~ml.tracking.bytetrack.ByteTrackTracker` (needs Ultralytics +
a YOLO checkpoint). Pick one through :func:`ml.tracking.factory.create_tracker`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from app.schemas.models import Detection, Track


class BaseTracker(ABC):
    """Common tracker contract shared by the IoU and ByteTrack backends."""

    def __init__(self, tracking_settings: dict, camera_id: str):
        self.settings = tracking_settings or {}
        self.camera_id = camera_id

    @abstractmethod
    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """Associate this frame's detections to tracks and return confirmed ones."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all internal track state (used on config reload)."""

    def active_ids(self) -> List[int]:
        """Return currently alive anonymous track ids."""
        return []