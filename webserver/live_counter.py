"""Live webcam people counting for the Occupancy view.

A small standalone background thread: captures the local camera (cv2), runs
YOLOv8 person detection (COCO class 0 = "person" — the same approach as the
external ``queue_model`` repo's ``people_counter.py``) and exposes the latest
annotated JPEG frame plus rolling count stats.

Purely additive to the dashboard: it NEVER touches the analytics RunManager /
video pipeline. When no uploaded video is present, the Occupancy metrics fall
back to these live counts; when a video run is active, the video results win.
Nothing is recorded — frames live only in memory while the counter runs.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


class LivePeopleCounter:
    """Owns the webcam capture thread + YOLOv8 person counts."""

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        max_width: int = 960,
        history_len: int = 120,
        min_interval: float = 0.1,
    ) -> None:
        self.camera_index = camera_index
        self.conf = conf
        self.max_width = max_width
        self.history_len = history_len
        self.min_interval = min_interval
        self._model_path = model_path
        self._model: Any = None
        self._cap: Any = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_jpeg: Optional[bytes] = None
        self._last_count = 0

        self.running = False
        self.error: Optional[str] = None
        self.started_ts: Optional[float] = None
        self.count = 0
        self.peak = 0
        self.entries = 0
        self.exits = 0
        self.history: List = []

    # ------------------------------------------------------------ lifecycle
    def start(self, camera_index: Optional[int] = None) -> bool:
        """Open the camera and start the detection thread. Reusable across
        start/stop cycles (the model is loaded once). ``camera_index`` may be
        passed to switch to a different (e.g. external USB) webcam."""
        if self.running:
            return True
        if camera_index is not None:
            self.camera_index = int(camera_index)
        if self._model is None:
            try:
                from ultralytics import YOLO  # noqa: PLC0415

                self._model = YOLO(self._model_path)
            except Exception as exc:  # pragma: no cover - env dependent
                self._model = None
                self.error = f"could not load YOLO model ({exc})"
                logger.warning("live counter: %s", self.error)
                return False
        try:
            import cv2  # noqa: PLC0415

            self._cap = cv2.VideoCapture(self.camera_index)
        except Exception as exc:  # pragma: no cover
            self.error = f"could not open camera {self.camera_index}: {exc}"
            return False
        if not self._cap.isOpened():
            self.error = (f"camera {self.camera_index} unavailable — grant your "
                          "terminal Camera permission in System Settings")
            self._cap = None
            logger.warning("live counter: %s", self.error)
            return False

        with self._lock:
            self.count = 0
            self.peak = 0
            self.entries = 0
            self.exits = 0
            self._last_count = 0
            self.history = []
            self.error = None
            self.started_ts = time.time()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("live counter started (camera %s)", self.camera_index)
        return True

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2 * self.min_interval + 0.5)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # pragma: no cover
                pass
            self._cap = None
        with self._lock:
            self._frame_jpeg = None
        logger.info("live counter stopped")

    # --------------------------------------------------------------- stream
    def stream_frames(self, poll_sec: float = 0.1, max_frames: Optional[int] = None):
        """Yield the latest annotated JPEG as an MJPEG push stream.

        Repeats the newest cached frame; ``max_frames`` bounds the total parts
        for tests/clients that want a finite body.
        """
        sent = 0
        last: Optional[bytes] = None
        while True:
            if max_frames is not None and sent >= max_frames:
                break
            with self._lock:
                jpg = self._frame_jpeg
            if jpg is not None and (jpg is not last or max_frames is not None):
                last = jpg
                sent += 1
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(poll_sec)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self.history)
            fps = None
            if n >= 2:
                span = self.history[-1][0] - self.history[0][0]
                if span > 0.5:
                    fps = round((n - 1) / span, 1)
            return {
                "running": self.running,
                "camera_index": self.camera_index,
                "count": self.count,
                "peak": self.peak,
                "entries": self.entries,
                "exits": self.exits,
                "avg": self.avg_count(),
                "fps": fps,
                "error": self.error,
                "started_ts": self.started_ts,
                "history": [{"t": round(time.time() - t, 1), "c": c}
                            for t, c in self.history],
                "model": self._model is not None,
            }

    def devices(self, probe: int = 7) -> List[Dict[str, Any]]:
        """Probe local camera indices and report which are available.

        Skips the index currently owned by a running counter session.
        """
        import cv2  # noqa: PLC0415

        found: List[Dict[str, Any]] = []
        for i in range(probe):
            if self.running and i == self.camera_index:
                continue
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
            cap.release()
            if ok:
                found.append({
                    "index": i,
                    "name": f"Camera {i}" + (" (built-in)" if i == 0 else ""),
                    "resolution": f"{w}x{h}",
                })
        return found

    def avg_count(self) -> Optional[float]:
        n = len(self.history)
        if not n:
            return None
        return round(sum(c for _, c in self.history) / n, 1)

    # ---------------------------------------------------------------- loop
    def _loop(self) -> None:
        import cv2  # noqa: PLC0415

        last = time.time()
        while self.running:
            if self.min_interval > 0:
                time.sleep(self.min_interval)
            try:
                ok, frame = self._cap.read()
            except Exception as exc:  # pragma: no cover
                self.error = f"camera read failed: {exc}"
                break
            if not ok:
                self.error = "source ended or frame could not be read"
                break
            count, boxes = self.detect(frame)
            now = time.time()
            with self._lock:
                self.count = count
                self.peak = max(self.peak, count)
                if count > self._last_count:
                    self.entries += count - self._last_count
                elif count < self._last_count:
                    self.exits += self._last_count - count
                self._last_count = count
                self.history.append((now, count))
                if len(self.history) > self.history_len:
                    del self.history[0]
            try:
                annotated = self.annotate(frame, boxes, count)
                h, w = annotated.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / float(w)
                    annotated = cv2.resize(annotated,
                                           (self.max_width, max(1, int(h * scale))))
                okj, buf = cv2.imencode(".jpg", annotated)
                if okj:
                    with self._lock:
                        self._frame_jpeg = buf.tobytes()
            except Exception as exc:  # pragma: no cover
                logger.warning("live counter annotate: %s", exc)
            last = now

    # --------------------------------------------------------------- detect
    def detect(self, frame):
        """Run YOLOv8 person detection -> (person_count, boxes)."""
        if self._model is None:
            return 0, []
        results = self._model(frame, conf=self.conf, verbose=False)
        boxes = [b for b in results[0].boxes if int(b.cls[0]) == PERSON_CLASS]
        return len(boxes), boxes

    def annotate(self, frame, boxes, count):
        import cv2  # noqa: PLC0415

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "person", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        label = f"People: {count}"
        cv2.rectangle(frame, (0, 0), (220, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        return frame