"""Live webcam people counting for the Occupancy view.

Two modes:

1. **Browser capture** (primary — any device): any laptop opens the dashboard,
   captures its webcam via ``getUserMedia``, sends JPEG frames to
   ``POST /api/livecount/frame``.  Server runs YOLOv8 + IoU tracking + line
   counting and returns an annotated JPEG + stats.

2. **Server camera** (local fallback): the machine hosting the dashboard opens
   its webcam via ``cv2.VideoCapture`` in a background thread.  Used by tests
   and local-only setups.

Both paths never touch the analytics RunManager / video pipeline.  Nothing is
recorded — frames live only in memory while the counter runs.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PERSON_CLASS = 0

MODEL_BACKENDS = ("ultralytics", "onnx")


class _ModelHandle:
    """Thin adapter so both inference backends expose the same ``__call__``.

    ``ultralytics``: a YOLO instance (returns result objects with ``.boxes``).
    ``onnx``: an :class:`YoloDetector` in ONNX mode (returns Detection lists).
    """

    def __init__(self, backend: str, model: Any) -> None:
        self.backend = backend
        self._model = model

    def __call__(self, frame, conf: float, imgsz: int, verbose: bool = False,
                 classes: Optional[List[int]] = None):
        if self.backend == "onnx":
            return self._model.detect(frame)
        return self._model(frame, conf=conf, imgsz=imgsz, verbose=verbose,
                           classes=classes)


class LivePeopleCounter:
    """Owns the webcam capture thread + YOLOv8 person counts."""

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        imgsz: int = 960,
        max_width: int = 960,
        history_len: int = 120,
        min_interval: float = 0.1,
        line_start: Tuple[float, float] = (0.1, 0.72),
        line_end: Tuple[float, float] = (0.9, 0.72),
    ) -> None:
        self.camera_index = camera_index
        self._repo_root = str(Path(__file__).resolve().parent.parent)
        self.conf = conf
        self.imgsz = imgsz
        self.max_width = max_width
        self.history_len = history_len
        self.min_interval = min_interval
        self._model_path = model_path
        self._model: Any = None
        self.backend: Optional[str] = None
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

        self._line_start_rel = line_start
        self._line_end_rel = line_end

        self._tracker: Any = None
        self._tracker_lock = threading.Lock()
        self._line_counter: Any = None
        self._frame_times: deque = deque(maxlen=30)
        self._smooth_alpha = 0.35
        self._smoothed_count: float = 0.0

    # ------------------------------------------------------------ lifecycle
    def _ensure_model(self) -> bool:
        """Load a person-detection backend, preferring ultralytics but falling
        back to ONNX Runtime (lightweight, ships in the Render image)."""
        if self._model is not None:
            return True
        err_note = ""
        # 1) ultralytics .pt (local dev / full ML install)
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            self._model = _ModelHandle("ultralytics", YOLO(self._model_path))
            self.backend = "ultralytics"
            return True
        except Exception as exc:  # ultralytics absent or checkpoint unloadable
            err_note = str(exc)
        # 2) ONNX .onnx (lightweight Render path) or .pt->.onnx sibling
        onnx_path = self._resolve_onnx_path()
        if onnx_path is not None:
            try:
                from ml.detection.yolo_detector import YoloDetector  # noqa: PLC0415
                det = YoloDetector({"yolo_model": str(onnx_path),
                                    "conf_threshold": self.conf,
                                    "imgsz": self.imgsz,
                                    "classes": [PERSON_CLASS]})
                if det._ort_sess is not None:
                    self._model = _ModelHandle("onnx", det)
                    self.backend = "onnx"
                    self._model_path = str(onnx_path)
                    return True
                err_note = det._load_error or err_note
            except Exception as exc:  # pragma: no cover
                err_note = str(exc)
        self._model = None
        self.error = f"could not load YOLO model ({err_note})"
        logger.warning("live counter: %s", self.error)
        return False

    def _resolve_onnx_path(self) -> Optional[str]:
        from pathlib import Path
        p = Path(self._model_path)
        if p.suffix.lower() == ".onnx":
            return str(p) if p.exists() else None
        onnx_p = p.with_suffix(".onnx")
        if onnx_p.exists():
            return str(onnx_p)
        return None

    def _ensure_tracker(self, frame_shape: Tuple[int, ...]) -> None:
        if self._tracker is not None:
            return
        from ml.tracking.tracker import Tracker  # noqa: PLC0415
        from ml.shopper.line_counter import LineCounter  # noqa: PLC0415
        h, w = frame_shape[:2]
        ls = (self._line_start_rel[0] * w, self._line_start_rel[1] * h)
        le = (self._line_end_rel[0] * w, self._line_end_rel[1] * h)
        self._tracker = Tracker(
            {"max_age_frames": 30, "min_hits": 1, "iou_match_threshold": 0.30},
            "live_camera",
        )
        self._line_counter = LineCounter(
            ls, le, "live_camera",
            {"cooldown_frames": 8, "entry_direction": "up"},
        )

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Open the camera and start the detection thread. Reusable across
        start/stop cycles (the model is loaded once).  ``camera_index`` may be
        passed to switch to a different (e.g. external USB) webcam."""
        if self.running:
            return True
        if camera_index is not None:
            self.camera_index = int(camera_index)
        if not self._ensure_model():
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

    def _normalize_model_path(self, path: str) -> str:
        """Map a browser-supplied model name onto a real repo file so lazy
        ONNX-sibling resolution works from any CWD (Render ships no
        ultralytics and no bare `yolov8n.onnx` in the working directory)."""
        from pathlib import Path

        p = Path(path)
        if p.is_absolute():
            return str(p)
        # already repo-relative (e.g. "models/yolo/yolov8n.pt")
        cand = self._repo_root / p
        if cand.exists():
            return str(cand.resolve())
        # bare filename -> prefer the yolo models dir sibling
        cand = self._repo_root / "models" / "yolo" / p.name
        if cand.exists():
            return str(cand.resolve())
        # nothing on disk yet (ultralytics would download) — keep bare name so
        # the local ultralytics download path still applies.
        return str(cand)

    def reconfigure(self, conf: Optional[float] = None, imgsz: Optional[int] = None,
                    model_path: Optional[str] = None) -> None:
        """Runtime reconfiguration.  Model is reloaded lazily on next frame."""
        if conf is not None:
            self.conf = float(conf)
        if imgsz is not None:
            self.imgsz = int(imgsz)
        if model_path is not None:
            self._model_path = self._normalize_model_path(model_path)
            self._model = None  # force reload
            self.backend = None
            self._tracker = None
            self._line_counter = None
            self._ensure_model()
        logger.info("live counter reconfigured: conf=%.2f imgsz=%d model=%s",
                     self.conf, self.imgsz, self._model_path)

    def _resolve_model_path(self, path: str) -> str:
        """Accept a bare filename (e.g. 'yolov8n.pt' sent by the browser config
        dropdown) or a repo-relative path; returns an existing absolute path or
        the best-effort repo-rooted path so ONNX sibling lookup still works."""
        from pathlib import Path
        p = Path(path)
        if p.is_absolute():
            return str(p)
        repo = Path(self._repo_root)
        for cand in (repo / p, repo / "models" / "yolo" / p.name):
            if cand.exists():
                return str(cand.resolve())
        return str((repo / p).resolve())

    # --------------------------------------------------------------- stream
    def stream_frames(self, poll_sec: float = 0.1, max_frames: Optional[int] = None):
        """Yield the latest annotated JPEG as an MJPEG push stream."""
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
        """Probe local camera indices and report which are available."""
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

    # ------------------------------------------------------------ browser mode
    def process_frame(self, frame_bytes: bytes) -> Dict[str, Any]:
        """Process a JPEG frame sent from a browser.  Returns annotated frame
        (base64 JPEG) + live stats.  This is the primary entry point for the
        browser-capture mode (any-laptop access)."""
        import cv2  # noqa: PLC0415

        if not self._ensure_model():
            return {"error": self.error or "model not available", "frame": ""}

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "invalid frame", "frame": ""}

        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / float(w)
            frame = cv2.resize(frame, (self.max_width, max(1, int(h * scale))))
            h, w = frame.shape[:2]

        self._ensure_tracker((h, w, 3))

        results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        from app.schemas.models import Detection  # noqa: PLC0415

        raw_boxes = []
        if self.backend == "onnx":
            for d in results:
                raw_boxes.append((d.x1, d.y1, d.x2, d.y2, d.confidence))
        else:
            raw_boxes = [b for b in results[0].boxes if int(b.cls[0]) == PERSON_CLASS]

        detections: List[Detection] = []
        for box in raw_boxes:
            if self.backend == "onnx":
                x1, y1, x2, y2, c = box
            else:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                c = float(box.conf[0])
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            if frame_area > 0 and box_area / frame_area < 0.001:
                continue
            if y2 > h * 0.97:
                continue
            if (x2 - x1) < 8 or (y2 - y1) < 16:
                continue
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=c, class_id=PERSON_CLASS, class_name="person",
            ))

        with self._tracker_lock:
            tracks = self._tracker.update(detections, frame)
            person_tracks = [t for t in tracks if t.class_name == "person"]
            events = self._line_counter.update(person_tracks)

        raw_count = len(person_tracks)

        t0 = time.monotonic()
        if self._frame_times:
            dt = t0 - self._frame_times[-1]
            self._smoothed_count += self._smooth_alpha * (raw_count - self._smoothed_count)
        else:
            self._smoothed_count = float(raw_count)
        self._frame_times.append(t0)

        avg_fps = 0.0
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                avg_fps = round((len(self._frame_times) - 1) / span, 1)

        now = time.time()
        with self._lock:
            for ev in events:
                if ev.event_type == "entry":
                    self.entries += 1
                elif ev.event_type == "exit":
                    self.exits += 1
            self.count = raw_count
            self.peak = max(self.peak, raw_count)
            self.history.append((now, raw_count))
            if len(self.history) > self.history_len:
                del self.history[0]
            history_snap = [{"t": round(now - t, 1), "c": c} for t, c in self.history]
            avg = round(sum(c for _, c in self.history) / len(self.history), 1) if self.history else None

        annotated = self._annotate_tracks(frame, person_tracks, raw_count)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""

        return {
            "frame": frame_b64,
            "count": raw_count,
            "peak": self.peak,
            "entries": self.entries,
            "exits": self.exits,
            "avg": avg,
            "fps": avg_fps,
            "history": history_snap,
            "model": True,
        }

    # --------------------------------------------------------------- detect
    def detect(self, frame):
        """Run person detection -> (person_count, boxes)."""
        if self._model is None:
            return 0, []
        results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.backend == "onnx":
            boxes = results  # already Detection objects, person-only
        else:
            boxes = [b for b in results[0].boxes if int(b.cls[0]) == PERSON_CLASS]
        return len(boxes), boxes

    def _annotate_tracks(self, frame, tracks, count):
        import cv2  # noqa: PLC0415

        for tr in tracks:
            x1, y1, x2, y2 = map(int, [tr.x1, tr.y1, tr.x2, tr.y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{tr.id}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if self._line_counter is not None:
            h, w = frame.shape[:2]
            lp1 = tuple(map(int, self._line_counter.line_start))
            lp2 = tuple(map(int, self._line_counter.line_end))
            cv2.line(frame, lp1, lp2, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "counting line", (lp1[0] + 5, lp1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        label = f"People: {count}"
        cv2.rectangle(frame, (0, 0), (230, 42), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        return frame

    def annotate(self, frame, boxes, count):
        import cv2  # noqa: PLC0415

        for box in boxes:
            if hasattr(box, "xyxy"):  # ultralytics box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            else:                      # onnx Detection
                x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "person", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        label = f"People: {count}"
        cv2.rectangle(frame, (0, 0), (220, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        return frame

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
