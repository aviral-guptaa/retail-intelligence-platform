"""Camera performance monitoring (no ML logic).

Each camera keeps a rolling account of what the pipeline actually observes:
frames processed, step latency (processing time per analytics pass), delivered
FPS, reconnect/error counts, state transitions (ONLINE / PROCESSING /
RECONNECTING / OFFLINE / ERROR / FINISHED) and the age of the newest analysed
frame. The numbers come straight from the loop timings - nothing is estimated.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from ml.sources.camera import CameraSource


class CameraMonitor:
    def __init__(self, camera_id: str, store_id: Optional[str] = None,
                 latency_window: int = 120, fps_window: int = 60):
        self.camera_id = camera_id
        self.store_id = store_id
        self.state = "ONLINE"
        self.frames_processed = 0
        self.errors = 0
        self.reconnects = 0
        self.latencies: Deque[float] = deque(maxlen=latency_window)
        self.frames_in_window: Deque[float] = deque(maxlen=fps_window)
        self.started_ts = time.time()
        self.last_frame_ts: Optional[float] = None
        self._last_was_down = False

    # ---------------------------------------------------------------- events
    def step_ok(self, elapsed: float, now: float) -> None:
        self.frames_processed += 1
        self.latencies.append(elapsed)
        self.frames_in_window.append(now)
        self.last_frame_ts = now

    def step_error(self, now: float) -> None:
        self.errors += 1
        now = now if now else time.time()

    def source_health(self, health: Dict[str, Any]) -> None:
        """Fold the camera source's own status into the monitor."""
        status = health.get("status", "ONLINE")
        if status == "RECONNECTING" and not self._last_was_down:
            self.reconnects += 1
            self._last_was_down = True
        if status == "ONLINE":
            self._last_was_down = False
        self.state = status if status in ("ONLINE", "RECONNECTING", "OFFLINE",
                                          "FINISHED", "ERROR") else "PROCESSING"

    # ---------------------------------------------------------------- reads
    def _fps(self, now: float) -> float:
        if not self.frames_in_window:
            return 0.0
        return round(len(self.frames_in_window) /
                     max(now - self.frames_in_window[0], 1e-9), 2)

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now if now is not None else time.time()
        n = len(self.latencies)
        avg_lat = sum(self.latencies) / n if n else 0.0
        latency = {
            "avg_ms": round(avg_lat * 1000.0, 1),
            "p95_ms": round(sorted(self.latencies)[int(n * 0.95)] * 1000.0, 1) if n else 0.0,
            "max_ms": round(max(self.latencies) * 1000.0, 1) if n else 0.0,
            "samples": n,
        }
        return {
            "camera_id": self.camera_id,
            "store_id": self.store_id,
            "state": self.state,
            "fps": self._fps(now),
            "frames_processed": self.frames_processed,
            "errors": self.errors,
            "reconnects": self.reconnects,
            "processing_latency_ms": latency,
            "last_frame_age_s": round(now - self.last_frame_ts, 1) if self.last_frame_ts else None,
            "last_frame_ts": self.last_frame_ts,
            "uptime_s": round(now - self.started_ts, 1),
        }


class PerformanceSummary:
    """Aggregates per-camera monitors into one /system/performance payload."""

    def __init__(self, monitors: Dict[str, CameraMonitor]):
        self.monitors = monitors

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now if now is not None else time.time()
        per_camera = {cid: mon.snapshot(now) for cid, mon in self.monitors.items()}
        states: Dict[str, int] = {}
        total_frames = 0
        avg_fps = 0.0
        if per_camera:
            for cam in per_camera.values():
                states[cam["state"]] = states.get(cam["state"], 0) + 1
                total_frames += cam["frames_processed"]
                avg_fps += cam["fps"]
            avg_fps = round(avg_fps / len(per_camera), 2)
        return {
            "now": now,
            "cameras": per_camera,
            "summary": {
                "camera_count": len(per_camera),
                "states": states,
                "total_frames": total_frames,
                "avg_fps": avg_fps,
                "total_errors": sum(c["errors"] for c in per_camera.values()),
                "total_reconnects": sum(c["reconnects"] for c in per_camera.values()),
            },
        }