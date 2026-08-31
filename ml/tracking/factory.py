"""Tracker factory: pick IoU or ByteTrack from config.

``tracking.backend`` accepts ``iou``, ``bytetrack`` or ``auto``. ``auto`` uses
ByteTrack when a real YOLO model is attached to the detector, otherwise falls
back to the pure-IoU tracker (so the demo always works).
"""
from __future__ import annotations

import logging
from typing import Optional

from ml.tracking.base import BaseTracker

logger = logging.getLogger(__name__)


def create_tracker(backend: str, tracking_settings: dict, camera_id: str,
                   detector=None) -> BaseTracker:
    backend = (backend or "auto").lower()

    if backend == "bytetrack":
        from ml.tracking.bytetrack import ByteTrackTracker

        return ByteTrackTracker(tracking_settings, camera_id, detector=detector)
    if backend == "iou":
        from ml.tracking.tracker import Tracker

        return Tracker(tracking_settings, camera_id)

    # auto
    if detector is not None and getattr(detector, "model", None) is not None:
        from ml.tracking.bytetrack import ByteTrackTracker

        logger.info("tracking backend: bytetrack (auto)")
        return ByteTrackTracker(tracking_settings, camera_id, detector=detector)

    from ml.tracking.tracker import Tracker

    logger.info("tracking backend: iou (auto - no YOLO model attached)")
    return Tracker(tracking_settings, camera_id)