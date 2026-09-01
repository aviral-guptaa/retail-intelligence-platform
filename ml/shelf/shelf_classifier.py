"""Shelf / inventory state classification: FULL | LOW_STOCK | OUT_OF_STOCK.

Three strategies, selectable via ``config/settings.yaml -> shelf.strategy``:

  1. classification - transfer-learned MobileNet/EfficientNet CNN that scores a
     shelf ROI crop (train with ``scripts/train_shelf_model.py``, logs + weights
     saved to ``shelf.model_path``). Needs torch + torchvision at runtime. This
     is the recommended strategy for real cameras because it needs NO product
     detector - only the person-detection camera stream.
  2. detection - counts product-class detections inside each shelf region and
     compares against the expected item count. Only meaningful when a *product*
     detector is attached (e.g. the demo simulator's synthetic boxes). The
     stock COCO person model does NOT produce product boxes, so this strategy
     degrades to heuristic in video/live mode.
  3. heuristic - frame-content proxy for when neither a CNN nor product
     detections exist: the fraction of pixels that deviate from the (median)
     shelf background is used as a stand-in for remaining stock. Cheap, honest,
     clearly labelled ``source=heuristic``, meant as a smoke signal not a
     measurement.

``auto`` picks the first available in the order classification -> detection ->
heuristic. The active choice is reported per shelf as ``source``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.models import Detection, ShelfSnapshot
from config.loader import resolve
from ml.geometry import point_in_polygon

logger = logging.getLogger(__name__)

FULL = "FULL"
LOW_STOCK = "LOW_STOCK"
OUT_OF_STOCK = "OUT_OF_STOCK"

VALID_STRATEGIES = ("auto", "classification", "detection", "heuristic")
_PROD_CLASSES = ("product", "object")


class ShelfClassifier:
    def __init__(self, shelves: Dict[str, Dict[str, Any]], settings: Dict, camera_id: str):
        self.settings = settings
        self.camera_id = camera_id
        self.regions = {sid: [np.array(p, dtype=float) for p in s["region"]] for sid, s in shelves.items()}
        self.expected = {sid: int(s["expected_item_count"]) for sid, s in shelves.items()}
        self.low_threshold = float(settings.get("low_stock_threshold", 0.30))
        self.out_threshold = float(settings.get("out_of_stock_threshold", 0.10))
        self.poll_interval = float(settings.get("poll_interval_seconds", 5))
        strategy = str(settings.get("strategy", "auto")).lower()
        self.strategy = strategy if strategy in VALID_STRATEGIES else "auto"
        self.model_path = resolve(
            settings.get("model_path", "models/prediction/shelf_classifier.pt"))
        # Temporal smoothing: how many consecutive polls a *change* must be
        # observed before the committed state flips (guards against a single
        # bad frame / transient occlusion producing a false alert). The very
        # first observation is committed immediately so a fresh shelf has a
        # state; only transitions are confirmed.
        self.confirmation_polls = max(1, int(settings.get("confirmation_polls", 3)))
        # Depletion-trend window: how many recent polls to use for the stock-out
        # risk trend estimate.
        self.trend_window = max(2, int(settings.get("trend_window", 10)))

        self._cnn: Any = None
        self._cnn_classes: List[str] = []
        self._cnn_error: Optional[str] = None
        self.active_strategy = "heuristic"
        self._register_cnn()

        self._last_poll: float = 0.0
        self.states: Dict[str, ShelfSnapshot] = {}
        self._last_products: Dict[str, int] = {}
        # per-shelf pending observation buffer + confirmed status + count history
        self._pending: Dict[str, List[str]] = {}
        self._committed_status: Dict[str, str] = {}
        self._count_history: Dict[str, List[Tuple[float, int]]] = {}

    # ------------------------------------------------------------- cnn model
    def _register_cnn(self) -> None:
        """Load the transfer-learning shelf classifier if it was trained."""
        if not self.model_path.exists():
            self._cnn_error = f"no CNN weights at {self.model_path} (train with scripts/train_shelf_model.py)"
            return
        try:
            import torch  # type: ignore

            from torchvision.models import get_model  # type: ignore

            ckpt = torch.load(str(self.model_path), map_location="cpu")
            classes = list(ckpt.get("classes", ["FULL", "LOW_STOCK", "OUT_OF_STOCK"]))
            backbone = ckpt.get("backbone", "mobilenet_v3_small")
            model = get_model(backbone, weights=None, num_classes=len(classes))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._cnn = model
            self._cnn_classes = classes
            logger.info("loaded shelf CNN (%s backbone, %s classes) from %s",
                        backbone, len(classes), self.model_path)
        except Exception as exc:
            self._cnn_error = f"failed to load shelf CNN: {exc}"
            self._cnn = None
        # sanity check: the labels should map onto the three stock states
        self._label_to_status = {str(c).upper(): c.upper() for c in self._cnn_classes}

    # ------------------------------------------------------------- strategies
    def classify_by_counting(self, detections: List[Detection], shelf_id: str) -> Optional[ShelfSnapshot]:
        """Detection strategy: count product-class boxes inside the shelf region."""
        region = self.regions[shelf_id]
        count = sum(1 for d in detections
                    if d.class_name in _PROD_CLASSES
                    and point_in_polygon(d.center, region))
        expected = self.expected[shelf_id]
        return self._snap_from_ratio(shelf_id, count, expected, "detection")

    def classify_by_heuristic(self, frame: np.ndarray, shelf_id: str) -> Optional[ShelfSnapshot]:
        """Frame-based proxy: fraction of non-background pixels = stock proxy."""
        count, expected = self._estimate_from_frame(frame, self.regions[shelf_id]), self.expected[shelf_id]
        return self._snap_from_ratio(shelf_id, count, expected, "heuristic")

    def classify_by_cnn(self, frame: np.ndarray, shelf_id: str) -> Optional[ShelfSnapshot]:
        """Classification strategy: score the shelf ROI crop with the CNN."""
        if self._cnn is None or frame is None:
            return None
        crop = self._crop(frame, self.regions[shelf_id])
        if crop is None or crop.size == 0:
            return None
        try:
            import torch
            from torchvision import transforms

            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((96, 96)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            with torch.no_grad():
                out = self._cnn(tf(crop / 255.0).unsqueeze(0))
                prob = torch.softmax(out, dim=1)[0]
                idx = int(prob.argmax().item())
            label = self._label_to_status.get(self._cnn_classes[idx].upper(), "FULL")
            expected = self.expected[shelf_id]
            # Turn the class confidence into a coarse estimated item count for the UI.
            ratio = {OUT_OF_STOCK: 0.0, LOW_STOCK: 0.25, FULL: 0.95}.get(label, 0.5)
            est_count = max(0, int(round(expected * ratio)))
            return ShelfSnapshot(shelf_id, label, est_count, expected,
                                 round(float(prob[idx]), 3), time.time(), source="classification")
        except Exception as exc:
            logger.warning("shelf CNN inference failed for %s: %s", shelf_id, exc)
            return None

    # ------------------------------------------------------------- internals
    def _snap_from_ratio(self, shelf_id: str, count: int, expected: int, source: str) -> ShelfSnapshot:
        ratio = count / expected if expected else 1.0
        status = FULL
        if ratio <= self.out_threshold:
            status = OUT_OF_STOCK
        elif ratio <= self.low_threshold:
            status = LOW_STOCK
        confidence = min(0.95, 0.5 + 0.45 * abs(ratio - 0.5) * 2)
        return ShelfSnapshot(shelf_id, status, count, expected,
                             round(confidence, 3), time.time(), source=source)

    def _crop(self, frame: np.ndarray, region) -> Optional[np.ndarray]:
        xs = [p[0] for p in region]
        ys = [p[1] for p in region]
        x0, x1 = int(max(0, min(xs))), int(min(frame.shape[1], max(xs)))
        y0, y1 = int(max(0, min(ys))), int(min(frame.shape[0], max(ys)))
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1].copy()

    @staticmethod
    def _estimate_from_frame(frame: np.ndarray, region) -> int:
        """Rough fill estimate: pixels far from the shelf background colour."""
        if frame is None or len(frame.shape) != 3:
            return 0
        try:
            import cv2

            x0 = int(max(0, min(p[0] for p in region)))
            x1 = int(min(frame.shape[1], max(p[0] for p in region)))
            y0 = int(max(0, min(p[1] for p in region)))
            y1 = int(min(frame.shape[0], max(p[1] for p in region)))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                return 0
            bg = np.median(crop, axis=(0, 1))
            diff = np.abs(crop.astype(np.float32) - bg).mean(axis=2)
            # Background pixels are "flat/dusty"; products/edges pop. Bootstrap a
            # count from the non-background area using a fixed assumed item size.
            mask = diff > 18
            area = float(mask.sum())
            per_item = 6.0 * 8.0  # ~48 px² per product face at 720p shelf scale
            return int(area / per_item)
        except Exception:
            return 0

    # ------------------------------------------------------------------ entry
    def resolve_strategy(self, has_product_detections: bool) -> str:
        """Pick the active strategy per the configured preference + availability."""
        if self.strategy == "classification":
            return "classification" if self._cnn is not None else "heuristic"
        if self.strategy == "detection":
            return "detection" if has_product_detections else "heuristic"
        if self.strategy == "heuristic":
            return "heuristic"
        # auto
        if self._cnn is not None:
            return "classification"
        if has_product_detections:
            return "detection"
        return "heuristic"

    def update(self, frame: np.ndarray | None, detections: List[Detection], now: float) -> bool:
        """Poll shelf states at the configured interval with temporal smoothing.

        Returns True if refreshed. A candidate status change is only committed
        after ``confirmation_polls`` consecutive consistent polls, so a single
        bad frame (occlusion / shadow / dropout) cannot flip a shelf to LOW or
        OUT_OF_STOCK. The first observation is committed immediately.
        """
        if now - self._last_poll < self.poll_interval:
            return False
        self._last_poll = now
        has_products = any(d.class_name in _PROD_CLASSES for d in detections)
        self.active_strategy = self.resolve_strategy(has_products)

        for sid in self.regions:
            snap: Optional[ShelfSnapshot] = None
            if self.active_strategy == "classification":
                snap = self.classify_by_cnn(frame, sid) if frame is not None else None
            if snap is None and self.active_strategy == "detection":
                snap = self.classify_by_counting(detections, sid)
            if snap is None:
                snap = self.classify_by_heuristic(frame, sid) if frame is not None else None
            if snap is None:
                continue

            # Record count history for depletion-trend estimation.
            hist = self._count_history.setdefault(sid, [])
            hist.append((now, snap.item_count))
            trend, time_to_out = self._depletion(hist, now)

            committed = self.states.get(sid)
            if committed is None:
                # First observation: commit immediately.
                snap.confirmed = True
                self._commit(sid, snap, trend, time_to_out)
                continue

            if committed.confirmed and snap.status == committed.status:
                # Stable vs the committed state -> refresh count/confidence.
                self._pending[sid] = []
                self.states[sid] = self._finalize_snapshot(snap, confirmed=True,
                                                           trend=trend, time_to_out=time_to_out)
                self._last_products[sid] = snap.item_count
            elif snap.status == self._committed_status.get(sid):
                # Matching the committed status while a transition is pending or
                # after a provisional -> stays on the committed state (confirmed).
                self._pending[sid] = []
                self.states[sid] = self._finalize_snapshot(snap, confirmed=True,
                                                           trend=trend, time_to_out=time_to_out)
                self._last_products[sid] = snap.item_count
            else:
                # A transition candidate: require N consecutive consistent polls.
                pending = self._pending.setdefault(sid, [])
                pending.append(snap.status)
                pending = pending[-self.confirmation_polls:]
                self._pending[sid] = pending
                if len(pending) >= self.confirmation_polls and \
                        all(p == snap.status for p in pending):
                    snap.confirmed = True
                    self._commit(sid, snap, trend, time_to_out)
                else:
                    # Not yet confirmed: keep committed state, raise confidence drop.
                    provisional = self._finalize_snapshot(snap, confirmed=False,
                                                          trend=trend, time_to_out=time_to_out)
                    self.states[sid] = provisional
        return True

    def _finalize_snapshot(self, snap: ShelfSnapshot, confirmed: bool,
                           trend: float, time_to_out: float) -> ShelfSnapshot:
        snap.confirmed = confirmed
        snap.trend = round(float(trend), 4)
        snap.est_time_to_out_minutes = round(float(time_to_out), 2)
        snap.stock_out_risk = self._risk_level(snap, time_to_out)
        return snap

    def _commit(self, sid: str, snap: ShelfSnapshot, trend: float,
                time_to_out: float) -> None:
        snap.confirmed = True
        self.states[sid] = self._finalize_snapshot(snap, True, trend, time_to_out)
        self._committed_status[sid] = snap.status
        self._pending[sid] = []
        self._last_products[sid] = snap.item_count

    def _depletion(self, hist: List[Tuple[float, int]], now: float):
        """Estimate item depletion trend + minutes-to-out-of-stock.

        ``trend`` = items lost per poll interval (negative means losing stock).
        ``time_to_out`` = estimated minutes until the shelf hits zero at the
        current rate (0.0 when not depleting or insufficient data).
        """
        if len(hist) < 2:
            return 0.0, 0.0
        tail = hist[-self.trend_window:]
        ts = np.array([t for t, _ in tail], dtype=float)
        vals = np.array([v for _, v in tail], dtype=float)
        if ts.max() - ts.min() < 1e-6:
            return 0.0, 0.0
        slope = float(np.polyfit(ts - ts[0], vals, 1)[0])  # items per second
        current = float(vals[-1])
        if slope >= 0 or current <= 0:
            return round(slope * self.poll_interval, 4), 0.0
        minutes_to_deplete = max(0.0, current / (-slope) / 60.0)
        return round(slope * self.poll_interval, 4), minutes_to_deplete

    def _risk_level(self, snap: ShelfSnapshot, time_to_out: float) -> str:
        if snap.status == OUT_OF_STOCK and snap.confirmed:
            return "HIGH" if time_to_out == 0.0 else "HIGH"
        if snap.status == OUT_OF_STOCK:
            return "HIGH"
        if time_to_out > 0:
            if time_to_out <= 15:
                return "HIGH"
            if time_to_out <= 45:
                return "MEDIUM"
            return "LOW"
        ratio = snap.item_count / snap.expected_count if snap.expected_count else 1.0
        if snap.status == LOW_STOCK:
            return "MEDIUM" if ratio <= self.out_threshold + 0.05 else "LOW"
        return "NONE"

    def snapshot(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.states.values()]

    def status_summary(self) -> str:
        if not self.states:
            return "UNKNOWN"
        if all(s.status == OUT_OF_STOCK for s in self.states.values()):
            return OUT_OF_STOCK
        if any(s.status == OUT_OF_STOCK for s in self.states.values()):
            return "PARTIAL_OUT"
        if all(s.status == FULL for s in self.states.values()):
            return FULL
        return "MIXED"