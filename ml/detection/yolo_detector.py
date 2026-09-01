"""Person / product detection via YOLO, with a safe fallback.

When the demo is enabled the pipeline uses :class:`demo.simulator.DemoSimulator`
directly, so this module is only exercised for real cameras/video.

Two real inference backends are supported, selected by the checkpoint extension:

  * ``.pt``    -> Ultralytics YOLO (needs torch + ultralytics). Exposed as
                  :attr:`model` so the ByteTrack backend can reuse the weights.
  * ``.onnx``  -> ONNX Runtime (needs onnxruntime). Lightweight path preferred
                  for edge devices (Jetson / Pi) because it needs NO PyTorch at
                  runtime and runs on CUDA/TensorRT/CPU providers. ``predict()``
                  (used by ByteTrack) is unavailable on this backend - only
                  ``detect()``; the tracker factory therefore returns the IoU
                  tracker when an ONNX model is active.

If neither backend can load (missing libs / no checkpoint) ``ready`` is False
and ``detect()``/``predict()`` degrade gracefully to prevent pipeline crashes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.models import Detection
from config.loader import resolve

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


class YoloDetector:
    """Light wrapper around a YOLO checkpoint (Ultralytics .pt or ONNX)."""

    def __init__(self, model_settings: Dict[str, Any], camera_id: str = "store_01"):
        self.settings = model_settings
        self.camera_id = camera_id
        self.conf = float(model_settings.get("conf_threshold", 0.35))
        self.iou = float(model_settings.get("iou_threshold", 0.45))
        self.imgsz = int(model_settings.get("imgsz", 640))
        self.device = model_settings.get("device", "cpu")
        self.person_class_id = int(model_settings.get("person_class_id", PERSON_CLASS))
        self.product_class_id = int(model_settings.get("product_class_id", 999))
        # Which COCO class ids to keep; empty list means "keep everything".
        classes = model_settings.get("classes") or []
        self.classes: Optional[List[int]] = [int(c) for c in classes] or None
        self.model_path = resolve(model_settings.get("yolo_model", "models/yolo/yolov8n.pt"))
        self.backend = "none"
        self._model = None           # ultralytics model (pt backend)
        self._ort_sess: Any = None   # onnxruntime session (onnx backend)
        self._load_error: Optional[str] = None
        self._load()

    @property
    def is_onnx(self) -> bool:
        return self.backend == "onnx"

    def _load(self) -> None:
        path = self.model_path
        if not path.exists():
            self._load_error = (f"model not found at {path}; download it with "
                                f"`python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"` "
                                f"then re-run, or use demo mode")
            self.backend = "none"
            return
        try:
            if path.suffix.lower() == ".onnx":
                self._load_onnx(path)
            else:
                self._load_ultralytics(path)
        except ImportError:
            self._load_error = "required inference library not installed (see requirements.txt optional section)"
            self.backend = "none"

    def _load_ultralytics(self, path: Path) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(str(path))
            self.backend = "ultralytics"
            logger.info("Loaded YOLO %s on device=%s imgsz=%s", path.name, self.device, self.imgsz)
        except ImportError:
            self._load_error = "ultralytics is not installed (see requirements.txt optional section)"
            self.backend = "none"

    def _load_onnx(self, path: Path) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            self._load_error = "onnxruntime is not installed (pip install onnxruntime[-gpu])"
            self.backend = "none"
            return
        providers, opts = self._pick_providers(ort)
        try:
            self._ort_sess = ort.InferenceSession(str(path), providers=providers,
                                                  sess_options=opts)
            self.backend = "onnx"
            logger.info("Loaded ONNX %s (providers=%s)", path.name,
                        self._ort_sess.get_providers())
        except Exception as exc:  # pragma: no cover
            self._load_error = f"failed to load ONNX model: {exc}"
            self.backend = "none"
            self._ort_sess = None

    def _pick_providers(self, ort) -> tuple:
        """Choose a lightweight provider chain; CPU is always the fallback."""
        available = set(getattr(ort, "get_available_providers", lambda: [])())
        ordered = []
        if self.device in ("cuda", "0", "gpu"):
            for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
                if p in available:
                    ordered.append(p)
                else:
                    break
        ordered.append("CPUExecutionProvider")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = getattr(ort, "GraphOptimizationLevel", None) and \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ordered, opts

    def reload(self) -> None:
        """Re-attempt model loading (e.g. the checkpoint appeared after boot)."""
        self._model = None
        self._ort_sess = None
        self._load_error = None
        self._load()

    @property
    def model(self):
        """The underlying Ultralytics YOLO instance (None when unavailable/ONNX)."""
        return self._model

    def predict(self, frame, verbose: bool = False):
        """Raw Ultralytics predict - used by the ByteTrack backend.

        Not supported on the ONNX backend; returns None there so the tracker
        factory falls back to IoU tracking.
        """
        if self._model is None:
            return None
        return self._model.predict(frame, conf=self.conf, iou=self.iou,
                                   device=self.device, imgsz=self.imgsz,
                                   classes=self.classes, verbose=verbose)

    # ------------------------------------------------------------- ONNX path
    def _onnx_preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)           # CHW
        return np.expand_dims(img, axis=0)     # NCHW

    def _onnx_detect(self, frame: np.ndarray) -> List[Detection]:
        if self._ort_sess is None:
            return []
        try:
            import cv2
            inp = self._onnx_preprocess(frame)
            out = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: inp})[0]
            # YOLOv8 output: [1, 4+nc, 8400] as (cx, cy, w, h, scores...)
            pred = np.squeeze(out, axis=0)     # [4+nc, N]
            if pred.ndim == 2 and pred.shape[0] > pred.shape[1]:
                pred = pred.T                  # -> [N, 4+nc]
            h_img, w_img = frame.shape[:2]
            scale = max(w_img, h_img) / self.imgsz
            dets: List[Detection] = []
            for row in pred:
                cx, cy, w, h = row[:4]
                scores = row[4:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                if conf < self.conf:
                    continue
                if self.classes is not None and cls_id not in self.classes:
                    continue
                x1 = (cx - w / 2) * scale
                y1 = (cy - h / 2) * scale
                x2 = (cx + w / 2) * scale
                y2 = (cy + h / 2) * scale
                dets.append(Detection(x1, y1, x2, y2, conf, cls_id, str(cls_id)))
            return dets
        except Exception as exc:  # pragma: no cover
            logger.warning("ONNX detect failed: %s", exc)
            return []

    # ------------------------------------------------------------- entrypoint
    def detect(self, frame) -> List[Detection]:
        """Run person detection on one BGR frame and return :class:`Detection` boxes."""
        if self.backend == "onnx":
            all_dets = self._onnx_detect(frame)
            people = [d for d in all_dets if d.class_id == self.person_class_id]
            return people
        results = self.predict(frame)
        if results is None:
            return []
        dets: List[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            names = r.names or {}
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = str(names.get(cls_id, cls_id))
                if cls_id == self.person_class_id:
                    dets.append(Detection(x1, y1, x2, y2, conf, cls_id, cls_name))
        return dets

    def detect_products(self, frame) -> List[Detection]:
        """Run detection keeping only the configured product class (may be unset).

        Returns an empty list unless the loaded checkpoint actually predicts the
        ``product_class_id`` (a real product detector), so the shelf detector
        knows whether "detection" strategy is viable.
        """
        if self.backend == "onnx":
            if self.product_class_id < 0:
                return []
            return [d for d in self._onnx_detect(frame) if d.class_id == self.product_class_id]
        results = self.predict(frame)
        if results is None or self.product_class_id < 0:
            return []
        dets: List[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            names = r.names or {}
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == self.product_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append(Detection(x1, y1, x2, y2, float(box.conf[0]),
                                          cls_id, str(names.get(cls_id, cls_id))))
        return dets

    def health(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "model": str(self.model_path),
            "backend": self.backend,
            "ready": self._model is not None or self._ort_sess is not None,
            "load_error": self._load_error,
            "device": self.device,
            "imgsz": self.imgsz,
        }

    def uses_synthetic(self) -> bool:
        return self._model is None and self._ort_sess is None
