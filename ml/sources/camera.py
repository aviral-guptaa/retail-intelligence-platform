"""Real camera / video source with reconnect + frame throttling.

Replaces the raw ``cv2.VideoCapture`` the pipeline used to hand to
:class:`app.services.analytics_service.AnalyticsService` and exposes the same
``next_frame() -> (frame, detections)`` contract as the demo simulator, so the
analytics loop is source-agnostic.

Failure handling (per the spec): if a read fails the source enters
``RECONNECTING``, retries opening the device after ``retry_interval_seconds``
and, if a video file was configured, re-seeks it (loop). After
``reconnect_max_attempts`` (>0) failures it drops to ``OFFLINE`` and returns
``(None, [])`` so the rest of the pipeline keeps running.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.schemas.models import Detection

logger = logging.getLogger(__name__)

ONLINE = "ONLINE"
RECONNECTING = "RECONNECTING"
OFFLINE = "OFFLINE"
FINISHED = "FINISHED"


def _looks_like_rtsp(source) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


class CameraSource:
    """A resilient OpenCV capture wrapper that behaves like a pipeline source."""

    def __init__(self, source, camera_id: str = "store_01",
                 detector=None, processing: Optional[Dict[str, Any]] = None,
                 video_loop: bool = True):
        import cv2

        self.cv2 = cv2
        self.source = source
        self.camera_id = camera_id
        self.detector = detector
        self.video_loop = bool(video_loop)
        self.is_file = isinstance(source, str) and not _looks_like_rtsp(source) \
            and not source.isdigit()

        proc = processing or {}
        self.max_fps = float(proc.get("max_fps", 0) or 0)
        self.frame_skip = int(proc.get("frame_skip", 0) or 0)
        self.retry_interval = float(proc.get("retry_interval_seconds", 3))
        self.max_attempts = int(proc.get("reconnect_max_attempts", 0) or 0)

        self.status = RECONNECTING
        self._cap: Any = None
        self._attempts = 0
        self._last_read = 0.0
        self._last_frame_ts = 0.0
        self._connect_error: Optional[str] = None
        self._frames_received = 0
        self._finished = False

    # ------------------------------------------------------------ lifecycle
    def open(self) -> bool:
        src = self.source
        try:
            cap = self.cv2.VideoCapture(src)
            if cap is not None and cap.isOpened():
                self._cap = cap
                self.status = ONLINE
                self._connect_error = None
                self._attempts = 0
                logger.info("camera %s opened (source=%s)", self.camera_id, src)
                return True
            cap.release()
        except Exception as exc:  # pragma: no cover
            self._connect_error = str(exc)
            logger.warning("camera %s open failed: %s", self.camera_id, exc)
        self.status = RECONNECTING
        return False

    def _ensure_open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        # Throttle consecutive reconnect attempts.
        now = time.time()
        if now - self._last_frame_ts < self.retry_interval:
            return
        self._last_frame_ts = now
        if self.max_attempts > 0 and self._attempts >= self.max_attempts:
            self.status = OFFLINE
            return
        self._attempts += 1
        self.open()

    # ----------------------------------------------------------------- source
    def next_frame(self) -> Tuple[Optional[np.ndarray], List[Detection]]:
        """Return the next (frame, detections). Detections come from the real
        detector when attached - never synthetic."""
        self._throttle()
        if self._cap is None:
            self._ensure_open()
            if self._cap is None:
                return None, []
        if self.frame_skip > 0:
            for _ in range(self.frame_skip):
                ok = self._cap.grab()
                if not ok:
                    break
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._handle_failure()
            return None, []
        self._frames_received += 1
        self.status = ONLINE
        dets = self.detector.detect(frame) if self.detector is not None else []
        return frame, dets

    def _handle_failure(self) -> None:
        self.status = RECONNECTING
        if self.is_file:
            if self.video_loop:
                # Re-seek to the start of the file.
                try:
                    self._cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, _ = self._cap.read()
                    if ok:
                        self.status = ONLINE
                        return
                except Exception as exc:
                    logger.warning("video loop re-seek failed: %s", exc)
            else:
                self._finished = True
                self.status = FINISHED
                self._release()
                return
        # For live feeds, reopen the device after the retry interval.
        self._release()
        self._ensure_open()

    def _throttle(self) -> None:
        if self.max_fps <= 0:
            return
        interval = 1.0 / self.max_fps
        elapsed = time.time() - self._last_read
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_read = time.time()

    # ------------------------------------------------------------------ misc
    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def release(self) -> None:
        self._release()

    def health(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "source": str(self.source),
            "status": self.status,
            "is_file": self.is_file,
            "frames_received": self._frames_received,
            "connect_error": self._connect_error,
            "detector": self.detector.health() if self.detector is not None else {"backend": "none"},
        }

    @property
    def finished(self) -> bool:
        return self._finished