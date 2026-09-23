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
        # Optional SEPARATE product checkpoint (single "product" class) used for
        # honest planogram compliance. Not shared with the person tracker - a
        # shelf-product class id 0 is meaningless to COCO person detection.
        product_p = model_settings.get("product_model")
        self.product_model_path = resolve(product_p) if product_p else None
        self.product_imgsz = int(model_settings.get("product_imgsz", 960))
        self.product_conf = float(model_settings.get("product_conf_threshold", 0.30))
        self.backend = "none"
        self._model = None           # ultralytics model (pt backend)
        self._product_model = None   # ultralytics product checkpoint (planogram)
        self._product_model_exc: Optional[str] = None  # why product model is absent
        self._ort_sess: Any = None   # onnxruntime session (onnx backend)
        self._load_error: Optional[str] = None
        self._load()

    @property
    def is_onnx(self) -> bool:
        return self.backend == "onnx"

    @property
    def supports_products(self) -> bool:
        """True only when a REAL product detector is actually available.

        ``product_class_id`` defaults to the placeholder 999 ("set when a product
        model is trained"); planogram compliance MUST NOT run off a placeholder.
        The gate is additionally honest about the model LOADING: a configured
        ``product_model`` that is missing/broken/unloadable reports False so the
        pipeline reports MODEL_NOT_AVAILABLE instead of pretending compliance.
        """
        configured_id = self.product_class_id if self.product_class_id != 999 else None
        if configured_id is None:
            return False
        if self._product_model is not None:
            return True
        # Multi-class checkpoint path: the shared model itself emits the class.
        classes = self.settings.get("classes") or []
        return any(int(c) == self.product_class_id for c in classes)

    def _load(self) -> None:
        path = self.model_path
        use_onnx = False
        if not path.exists():
            # Fall back to the ONNX sibling (lightweight Render/edge images
            # ship only the .onnx export).
            onnx_p = path.with_suffix(".onnx")
            if onnx_p.exists():
                path = onnx_p
                use_onnx = True
            else:
                self._load_error = (f"model not found at {self.model_path}; "
                                    f"download it with "
                                    f"`python -c \"from ultralytics import YOLO; "
                                    f"YOLO('yolov8n.pt')\"` then re-run, or use demo mode")
                self.backend = "none"
                return
        try:
            if use_onnx or path.suffix.lower() == ".onnx":
                self._load_onnx(path)
            else:
                self._load_ultralytics(path)
            self._load_product()
        except ImportError:
            self._load_error = "required inference library not installed (see requirements.txt optional section)"
            self.backend = "none"

    def _load_product(self) -> None:
        """Load the separate product checkpoint (optional, honest gate).

        Missing file or unloadable weights -> ``_product_model`` stays None and
        ``supports_products`` is False (never report fake planogram compliance).
        """
        if self.product_model_path is None:
            return
        if not self.product_model_path.exists():
            self._product_model_exc = f"product model not found at {self.product_model_path}"
            logger.warning("product model configured but missing at %s",
                           self.product_model_path)
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._product_model = YOLO(str(self.product_model_path))
            logger.info("Loaded product detector %s (planogram gate OPEN)",
                        self.product_model_path.name)
        except ImportError:
            self._product_model_exc = "ultralytics not installed (product detector unavailable)"
        except Exception as exc:  # pragma: no cover
            self._product_model_exc = f"failed to load product model: {exc}"
            logger.warning("product model load failed: %s", exc)

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
            # The exported model has a fixed input size; use it so config
            # `imgsz` deviations never feed the wrong shape.
            inp = self._ort_sess.get_inputs()[0]
            shape = list(getattr(inp, "shape", None) or [])
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
                self.imgsz = int(shape[2])
            self.backend = "onnx"
            logger.info("Loaded ONNX %s (providers=%s, imgsz=%s)", path.name,
                        self._ort_sess.get_providers(), self.imgsz)
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
        self._product_model = None
        self._product_model_exc = None
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
        h, w = img.shape[:2]
        # letterbox: fit the frame inside imgsz preserving aspect ratio, then
        # center-pad to the square input the exported model expects.
        r = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114.0, dtype=np.float32)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        canvas /= 255.0
        canvas = canvas.transpose(2, 0, 1)        # CHW
        return np.expand_dims(canvas, axis=0)     # NCHW

    def _onnx_detect(self, frame: np.ndarray) -> List[Detection]:
        if self._ort_sess is None:
            return []
        try:
            import cv2
            inp = self._onnx_preprocess(frame)
            out = self._ort_sess.run(None, {self._ort_sess.get_inputs()[0].name: inp})[0]
            # YOLOv8 ONNX output: [1, 4+nc, N] as (cx, cy, w, h, scores...) with
            # the LAST axis holding the N grid/anchor candidates.
            pred = np.squeeze(out, axis=0)     # [4+nc, N]
            if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
                pred = pred.T                  # -> [N, 4+nc] (rows = candidates)
            h_img, w_img = frame.shape[:2]
            # undo the letterbox resize+pad applied in _onnx_preprocess:
            r = min(self.imgsz / w_img, self.imgsz / h_img)
            new_w, new_h = int(round(w_img * r)), int(round(h_img * r))
            pad_x = (self.imgsz - new_w) / 2.0
            pad_y = (self.imgsz - new_h) / 2.0
            boxes: List[np.ndarray] = []
            scores: List[float] = []
            cls_ids: List[int] = []
            for row in pred:
                cx, cy, w, h = row[:4]
                scores_all = row[4:]
                cls_id = int(np.argmax(scores_all))
                conf = float(scores_all[cls_id])
                if conf < self.conf:
                    continue
                if self.classes is not None and cls_id not in self.classes:
                    continue
                x1 = (cx - w / 2 - pad_x) / r
                y1 = (cy - h / 2 - pad_y) / r
                x2 = (cx + w / 2 - pad_x) / r
                y2 = (cy + h / 2 - pad_y) / r
                boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                scores.append(conf)
                cls_ids.append(cls_id)
            keep = self._nms(boxes, scores, self.iou) if boxes else []
            dets: List[Detection] = []
            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                cls_id = cls_ids[i]
                dets.append(Detection(x1, y1, x2, y2, scores[i], cls_id, str(cls_id)))
            return dets
        except Exception as exc:  # pragma: no cover
            logger.warning("ONNX detect failed: %s", exc)
            return []

    @staticmethod
    def _nms(boxes: List[np.ndarray], scores: List[float], iou_thr: float
             ) -> List[int]:
        """Class-agnostic NMS.  Returns kept indices (sorted by confidence)."""
        if not boxes:
            return []
        order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        keep: List[int] = []
        while order:
            i = order.pop(0)
            keep.append(i)
            order = [j for j in order
                     if YoloDetector._iou(boxes[i], boxes[j]) <= iou_thr]
        return keep

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (a_area + b_area - inter)

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
        """Run detection keeping only the configured product class.

        Uses the dedicated product checkpoint when one is loaded; otherwise
        falls back to filtering the shared checkpoint by ``product_class_id``.
        Returns an empty list when no real product capability is configured
        (``supports_products`` is False), so the shelf detector and planogram
        never run fictionally.
        """
        if self._product_model is not None:
            dets: List[Detection] = []
            for r in self._product_model.predict(frame, conf=self.product_conf, iou=self.iou,
                                                 device=self.device, imgsz=self.product_imgsz,
                                                 verbose=False):
                names = r.names or {}
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != self.product_class_id:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append(Detection(x1, y1, x2, y2, float(box.conf[0]),
                                          cls_id, str(names.get(cls_id, "product"))))
            return dets
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
            "product_model": str(self.product_model_path) if self.product_model_path else None,
            "product_ready": self._product_model is not None,
            "product_error": self._product_model_exc,
            "supports_products": self.supports_products,
        }

    def uses_synthetic(self) -> bool:
        return self._model is None and self._ort_sess is None
