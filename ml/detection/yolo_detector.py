"""Person / product detection via Ultralytics YOLO, with a safe fallback.

When the demo is enabled the pipeline uses :class:`demo.simulator.DemoSimulator`
directly, so this module is only exercised for real cameras/video.

The wrapped Ultralytics model is exposed as :attr:`model` so the ByteTrack
backend can call ``model.track(frame, persist=True, ...)`` and reuse the same
checkpoint without loading it twice.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.models import Detection
from config.loader import resolve

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


class YoloDetector:
    """Light wrapper around a pretrained Ultralytics YOLO checkpoint.

    The model is loaded lazily so importing this module always works - even on
    machines without ultralytics/torch installed (``ready`` will be False and
    ``detect()`` returns nothing, which keeps the demo pipeline alive).
    """

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
        self._model = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        path = self.model_path
        if not path.exists():
            self._load_error = (f"model not found at {path}; download it with "
                                f"`python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"` "
                                f"then re-run, or use demo mode")
            self.backend = "none"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(path))
            self.backend = "ultralytics"
            logger.info("Loaded YOLO %s on device=%s imgsz=%s", path.name, self.device, self.imgsz)
        except ImportError:
            self._load_error = "ultralytics is not installed (see requirements.txt optional section)"
            self.backend = "none"

    def reload(self) -> None:
        """Re-attempt model loading (e.g. the checkpoint appeared after boot)."""
        self._model = None
        self._load_error = None
        self._load()

    @property
    def model(self):
        """The underlying Ultralytics YOLO instance (None if unavailable)."""
        return self._model

    def predict(self, frame, verbose: bool = False):
        """Raw Ultralytics predict - used by the ByteTrack backend."""
        if self._model is None:
            return None
        return self._model.predict(frame, conf=self.conf, iou=self.iou,
                                   device=self.device, imgsz=self.imgsz,
                                   classes=self.classes, verbose=verbose)

    def detect(self, frame) -> List[Detection]:
        """Run person detection on one BGR frame and return :class:`Detection` boxes."""
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
            "ready": self._model is not None,
            "load_error": self._load_error,
            "device": self.device,
            "imgsz": self.imgsz,
        }

    def uses_synthetic(self) -> bool:
        return self._model is None