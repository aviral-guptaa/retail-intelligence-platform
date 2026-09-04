"""Analytics service: orchestrates the per-camera ML modules into one snapshot.

Owns the source (demo simulator or camera), detector, tracker and all analytics
modules; the API layer only talks to this service, never to ML internals. All
frame/event loop wiring (retries, DB buffering, real-data logging) is pluggable
so the demo and the real pipeline run the exact same code.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.models import Detection, Track, ZoneEvent
from config.loader import resolve, load_zones
from database.repository import Repository
from ml.detection.yolo_detector import YoloDetector
from ml.queue.datalogger import QueueDataLogger
from ml.queue.evaluator import QueuePredictionEvaluator
from ml.queue.predictor import QueuePredictor
from ml.queue.queue_counter import QueueCounter
from ml.queue.wait_time import WaitTimeEstimator
from ml.shelf.planogram import PlanogramChecker
from ml.shelf.shelf_classifier import ShelfClassifier
from ml.shopper.dwell_time import ZoneDwellTracker
from ml.shopper.footfall import FootfallCounter
from ml.shopper.heatmap import HeatmapAccumulator
from ml.shopper.line_counter import LineCounter
from ml.tracking.factory import create_tracker

logger = logging.getLogger(__name__)

SHOPPING = "shopping_zone"
QUEUE = "queue_zone"
SHELF = "shelf_zone"
ENTRANCE = "entrance"
EXIT = "exit"

ZONE_TYPES = {SHOPPING, QUEUE, SHELF, ENTRANCE, EXIT}


def parse_zones(camera_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse zone config tolerantly: returns {zid: {polygon, zone_type}}.

    ``zone_type`` defaults to ``shopping_zone`` when missing; a legacy zone whose
    name contains 'checkout' is treated as a queue zone so old zones.json files
    keep working after the schema upgrade.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for zid, z in (camera_cfg.get("zones") or {}).items():
        poly = z.get("polygon")
        if not poly:
            continue
        ztype = str(z.get("zone_type", "")).lower()
        if ztype not in ZONE_TYPES:
            ztype = QUEUE if "checkout" in zid else SHOPPING
        out[zid] = {"polygon": [np.array(p, dtype=float) for p in poly],
                    "zone_type": ztype}
    return out


class AnalyticsService:
    """Single-camera analytics engine with a live ``current()`` snapshot."""

    def __init__(self, camera_id: str, settings: Dict[str, Any],
                 zones_config: Dict[str, Any],
                 repository: Optional[Repository] = None,
                 source=None, detector: Optional[YoloDetector] = None,
                 db_writer=None):
        self.camera_id = camera_id
        self.settings = settings
        self.repo = repository
        self.db_writer = db_writer
        self.camera_cfg = zones_config.get(camera_id, {})
        self.line = self.camera_cfg.get("entrance_line")

        zones_info = parse_zones(self.camera_cfg)
        self.zones = {zid: info["polygon"] for zid, info in zones_info.items()}
        self.zone_types = {zid: info["zone_type"] for zid, info in zones_info.items()}
        # Queue zones are whatever is explicitly tagged queue_zone.
        self.queue_zones = {k: v for k, v in self.zones.items()
                            if self.zone_types[k] == QUEUE}

        shelves = self.camera_cfg.get("shelves", {})
        self.shelf_specs = self._parse_shelves(shelves)
        self.shelf_list = [self.shelf_specs[sid]["region"] for sid in self.shelf_specs]

        # ---- ML modules ----------------------------------------------------
        demo = settings.get("demo", {})
        frame_w = int(demo.get("frame_width", 1280))
        frame_h = int(demo.get("frame_height", 720))
        zones_cfg = settings.get("zones", {})
        scale = int(zones_cfg.get("scale_for_heatmap", 4))
        decay = float(zones_cfg.get("heatmap_decay", 0.0) or 0.0)

        tracking_settings = settings.get("tracking", {})
        self.tracker = create_tracker(
            tracking_settings.get("backend", "auto"), tracking_settings, camera_id,
            detector=detector)

        self._build_zone_modules()
        self.footfall = FootfallCounter(camera_id)
        self.heatmap = HeatmapAccumulator(frame_w, frame_h, scale, camera_id, decay=decay)
        self._last_heatmap_ts = 0.0
        self._heat_export_path = resolve(zones_cfg.get("heatmap_export_path",
                                                       "data/processed/heatmap.png"))
        self._heat_export_interval = float(zones_cfg.get("heatmap_export_interval_seconds", 60))

        queue_settings = settings.get("queue", {})
        pred_settings = settings.get("prediction", {})
        self.wait_time = WaitTimeEstimator(queue_settings)
        self.predictor = QueuePredictor(pred_settings, queue_settings)
        self.pred_evaluator = QueuePredictionEvaluator(
            pred_settings.get("eval_path", "data/processed/prediction_eval.csv"),
            self.predictor.horizons, queue_id=camera_id)

        # ---- real-data + persistence ---------------------------------------
        self.data_logger = QueueDataLogger(
            pred_settings.get("log_path", "data/processed/queue_features.csv"),
            camera_id, self.queue_zones.keys(),
            sample_interval=float(queue_settings.get("sample_interval_seconds", 5)))

        db_settings = settings.get("database", {})
        self.snapshot_interval = float(db_settings.get("snapshot_interval_seconds", 10))
        self.trajectory_sampling = int(db_settings.get("trajectory_sampling_frames", 30))
        self._last_db_snapshot = 0.0
        self._frame_no = 0

        self.source = source
        self.detector = detector
        self._last_tracks: List[Track] = []
        self._last_frame: Optional[np.ndarray] = None
        self._frame_jpeg_cache: Dict[int, Optional[str]] = {}
        self.alert_status = "NORMAL"
        self.started = time.time()

    # ---------------------------------------------------------------- config
    @staticmethod
    def _parse_shelves(shelves: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            sid: {"region": [list(p) for p in s["region"]["polygon"]],
                  "expected_item_count": s.get("expected_item_count", 12),
                  "category": s.get("category", "unknown")}
            for sid, s in shelves.items()
        }

    def _build_zone_modules(self) -> None:
        """Create/recreate the modules that depend on zones/queue/shelves config.

        Called once during __init__ and again on reload_config() after a live
        zones POST update.  Modules that are camera-only (footfall, heatmap,
        queue/dwell/line/planogram) are shared; other state (tracker,
        data_logger, etc.) lives outside this method.
        """
        line_settings = self.settings.get("line_counter", {})
        self.line_counter = (LineCounter(self.line["start"], self.line["end"],
                                         self.camera_id, line_settings)
                             if self.line else None)
        self.dwell = ZoneDwellTracker(self.zones, self.camera_id)
        self.queue = QueueCounter(self.queue_zones, self.settings.get("queue", {}), self.camera_id)
        self.shelves = ShelfClassifier(
            {sid: {"region": s["region"], "expected_item_count": s["expected_item_count"]}
             for sid, s in self.shelf_specs.items()},
            self.settings.get("shelf", {}), self.camera_id)
        self.planogram = PlanogramChecker(self.camera_cfg.get("planogram", {}))

    # ---------------------------------------------------------------- pipeline
    def step(self) -> Dict[str, Any]:
        """Advance one frame: source -> detector -> tracker -> analytics modules."""
        now = time.time()
        events: List[ZoneEvent] = []

        if self.source is not None and hasattr(self.source, "next_frame"):
            frame, detections = self.source.next_frame()
            self._last_frame = frame if frame is not None else self._last_frame
        else:
            frame = self._last_frame
            detections = self.detector.detect(frame) if self.detector and frame is not None else []

        tracks = self.tracker.update(detections, frame)
        self._last_tracks = [t for t in tracks if t.class_name == "person"]

        events += (self.line_counter.update(self._last_tracks)
                   if self.line_counter else [])
        events += self.dwell.update(self._last_tracks, now)
        events += self.queue.update(self._last_tracks, now)
        self.footfall.update(events, active_count=len(self._last_tracks))
        self.heatmap.update(self._last_tracks)

        self.shelves.update(frame, detections, now)
        self._maybe_export_heatmap(now)

        self._log_queue_features(now)
        self._persist(now)

        self._frame_no += 1
        return self.current()

    # ------------------------------------------------------------- persistence
    def _log_queue_features(self, now: float) -> None:
        try:
            growth, _, _, _ = self._queue_metrics()
            self.data_logger.update(
                self.queue.history(), footfall=self.footfall.current_active,
                open_counters=int(self.settings.get("queue", {}).get("open_counters", 3)),
                now=now)
        except Exception as exc:  # never let logging break the pipeline
            logger.debug("queue datalog skipped: %s", exc)

    def _persist(self, now: float) -> None:
        if self.db_writer is None:
            return
        db_interval = self.snapshot_interval
        if db_interval <= 0 or now - self._last_db_snapshot >= db_interval:
            self._last_db_snapshot = now
            self._submit_snapshot(now)
        if self.trajectory_sampling > 0 and self._frame_no % self.trajectory_sampling == 0:
            for tid, tr in ((t.id, t) for t in self._last_tracks[:10]):
                self.db_writer.submit("position", timestamp=_utcnow(),
                                      camera_id=self.camera_id, track_id=int(tid),
                                      x=round(float(tr.center[0]), 2),
                                      y=round(float(tr.center[1]), 2))

    def _submit_snapshot(self, now: float) -> None:
        try:
            queues = self.current()["queues"]
            self.db_writer.submit("snapshot",
                                  timestamp=_utcnow(), camera_id=self.camera_id,
                                  footfall_count=self.footfall.current_active,
                                  entry_count=self.footfall.total_entries,
                                  exit_count=self.footfall.total_exits,
                                  queue_length=queues["total"],
                                  queue_growth_rate=queues["growth_rate"],
                                  predicted_queue_length=queues["predicted_max"],
                                  congestion_status=self.alert_status)
            for snap in self.shelves.states.values():
                self.db_writer.submit("snapshot",
                                      timestamp=_utcnow(), camera_id=self.camera_id,
                                      zone_id=None, shelf_id=snap.shelf_id,
                                      shelf_status=snap.status,
                                      congestion_status=self.alert_status)
        except Exception as exc:
            logger.debug("db snapshot skipped: %s", exc)

    def _maybe_export_heatmap(self, now: float) -> None:
        if self._heat_export_interval <= 0:
            return
        if now - self._last_heatmap_ts >= self._heat_export_interval:
            self._last_heatmap_ts = now
            if self.heatmap.save(self._heat_export_path):
                logger.debug("heatmap exported -> %s", self._heat_export_path)

    # ---------------------------------------------------------------- metrics
    def _queue_metrics(self):
        """Return (growth_rate, per_queue_predictions, global_predictions, waits)."""
        queue_counts = self.queue.counts()
        footfall = int(self.footfall.current_active)
        open_counters = int(self.settings.get("queue", {}).get("open_counters", 3))

        per_queue: Dict[str, Dict[str, float]] = {}
        for qid in queue_counts:
            hist = self.queue.history().get(qid, [])
            qpred = self.predictor.predict(hist, footfall=footfall,
                                           open_counters=open_counters)
            per_queue[qid] = qpred
            # runtime monitoring: record (made at now) forecasts per horizon
            horizon_vals = {h: float(qpred.get(f"{h}min", 0.0)) for h in self.predictor.horizons}
            self.pred_evaluator.record(
                time.time(), horizon_vals,
                [(t, float(v)) for t, v in self.queue.history().get(qid, [])],
                source=qpred.get("source", "fallback"))
        self.pred_evaluator.evaluate_ripe(time.time())

        global_preds: Dict[str, float] = {}
        sources = set()
        horizon_keys = {f"{h}min" for h in self.predictor.horizons} | {
            f"predicted_queue_length_{h}min" for h in self.predictor.horizons}
        for qp in per_queue.values():
            src = qp.get("source", "fallback")
            sources.add(src)
            for k, v in qp.items():
                if k in horizon_keys:
                    global_preds[k] = max(global_preds.get(k, 0.0), float(v))
        global_preds["source"] = "blend" if ("model" in sources or "blend" in sources) else "fallback"
        global_preds = self.predictor._decorate(global_preds)

        growth = 0.0
        if queue_counts:
            qid = max(queue_counts, key=queue_counts.get)
            growth = self.queue.growth_rate(qid)

        waits = {z: self.wait_time.estimate(c) for z, c in queue_counts.items()}
        return growth, per_queue, global_preds, waits

    # ---------------------------------------------------------------- snapshot
    def current(self) -> Dict[str, Any]:
        queue_counts = self.queue.counts()
        qtotal = self.queue.total_queued()
        growth, per_queue_preds, global_preds, wait = self._queue_metrics()

        from app.services.alert_service import AlertService  # avoid import cycle
        status, rec = AlertService.assess(qtotal, global_preds, self.settings.get("alerts", {}))
        self._emit_alerts(status, rec, qtotal)

        try:
            rec_detail = self.predictor.explain_recommendation(global_preds, qtotal)
        except Exception:
            rec_detail = None

        queues_list = [
            {
                "queue_id": qid,
                "length": int(count),
                "wait_minutes": wait.get(qid, 0.0),
                "predictions": per_queue_preds.get(qid, {}),
                "status": self._queue_status(global_preds.get("10min", 0.0), count),
            }
            for qid, count in queue_counts.items()
        ]

        occupancy = self.line_counter.occupancy() if self.line_counter else len(self._last_tracks)

        return {
            "camera_id": self.camera_id,
            "ts": time.time(),
            "uptime_s": round(time.time() - self.started, 1),
            "tracks": [t.to_dict() for t in self._last_tracks],
            "footfall": {
                **self.footfall.snapshot(),
                "occupancy": occupancy,
            },
            "dwell": {
                "avg_dwell_s": self.dwell.avg_dwell(),
                "current_dwell_s": self.dwell.current_dwell(),
                "occupancy": self.dwell.occupancy(),
            },
            "queues": {
                "counts": queue_counts,
                "total": qtotal,
                "growth_rate": growth,
                "wait_minutes": wait,
                "predictions": global_preds,
                "predicted_max": max((v for k, v in global_preds.items()
                                      if str(k).endswith("min") and str(k)[:-3].isdigit()),
                                     default=0.0),
                "queues": queues_list,
                "recommendation": rec,
                "recommendation_detail": rec_detail,
                "prediction_source": global_preds.get("source", "fallback"),
                "history": {z: [[ts, n] for ts, n in samples]
                            for z, samples in self.queue.history().items()},
            },
            "shelves": self.shelves.snapshot(),
            "congestion_status": status,
        }

    @staticmethod
    def _queue_status(pred_10min: float, length: int) -> str:
        if pred_10min >= 8 or length >= 8:
            return "HIGH"
        if pred_10min >= 4 or length >= 4:
            return "WARNING"
        return "NORMAL"

    def _emit_alerts(self, status, recommendation, qtotal) -> None:
        if self.alert_status != status:
            ts = _utcnow()
            if self.db_writer is not None:
                self.db_writer.submit("alert", timestamp=ts, camera_id=self.camera_id,
                                      alert_type="congestion", severity=status,
                                      message=recommendation)
            elif self.repo is not None:
                self.repo.add_alert(self.camera_id, "congestion", status, recommendation)
                self.repo.commit()
        self.alert_status = status
        logger.debug("queue=%s status=%s rec=%s", qtotal, status, recommendation)

    def reload_config(self, camera_cfg: Dict[str, Any]) -> None:
        """Re-derive zones/queues/shelves after a POST /config/zones update."""
        self.camera_cfg = camera_cfg
        self.line = camera_cfg.get("entrance_line")

        zones_info = parse_zones(self.camera_cfg)
        self.zones = {zid: info["polygon"] for zid, info in zones_info.items()}
        self.zone_types = {zid: info["zone_type"] for zid, info in zones_info.items()}
        self.queue_zones = {k: v for k, v in self.zones.items()
                            if self.zone_types[k] == QUEUE}
        self.shelf_specs = self._parse_shelves(camera_cfg.get("shelves", {}))
        self.shelf_list = [self.shelf_specs[sid]["region"] for sid in self.shelf_specs]

        self._build_zone_modules()
        logger.info("reloaded config for %s", self.camera_id)

    # ------------------------------------------------------------------ misc
    def heatmap_image(self) -> np.ndarray:
        return self.heatmap.to_image()

    def frame_jpeg_b64(self, max_width: int = 480) -> Optional[str]:
        """Encode the most recently analysed frame as a small base64 JPEG so the
        dashboard can show EXACTLY the frame the analytics were computed on
        (keeping the displayed video in sync with the predictions). The resize +
        encode result is cached per frame so repeated WebSocket clients do not
        re-encode the same frame."""
        frame = self._last_frame
        if frame is None:
            return None
        if self._frame_no in self._frame_jpeg_cache:
            return self._frame_jpeg_cache[self._frame_no]
        try:
            import base64, cv2 as _cv2
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = _cv2.resize(frame, (max_width, int(h * scale)),
                                    interpolation=_cv2.INTER_AREA)
            ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 68])
            if not ok:
                self._frame_jpeg_cache[self._frame_no] = None
                return None
            encoded = base64.b64encode(buf.tobytes()).decode("ascii")
            # keep the cache bounded to the last few frames
            if len(self._frame_jpeg_cache) > 30:
                self._frame_jpeg_cache.clear()
            self._frame_jpeg_cache[self._frame_no] = encoded
            return encoded
        except Exception:
            return None

    def health(self) -> Dict[str, Any]:
        source_info: Dict[str, Any] = {"status": "ONLINE", "backend": "demo"}
        if self.source is not None and hasattr(self.source, "health"):
            source_info = self.source.health()
        elif self.source is None:
            source_info = {"status": "ONLINE", "backend": "detector-only"}
        det_info = self.detector.health() if self.detector else {"backend": "none"}
        return {
            "camera_id": self.camera_id,
            "alive": True,
            "source": source_info,
            "tracker": getattr(self.tracker, "name", "iou"),
            "detector": det_info,
            "active_tracks": len(self._last_tracks),
            "entries": self.footfall.total_entries,
            "exits": self.footfall.total_exits,
            "occupancy": self.line_counter.occupancy() if self.line_counter else len(self._last_tracks),
            "prediction_monitoring": self.pred_evaluator.metrics(),
        }

    def shutdown(self) -> None:
        if self.source is not None and hasattr(self.source, "release"):
            self.source.release()
        try:
            self.data_logger.close()
        except Exception:
            pass
        try:
            self.pred_evaluator.close()
        except Exception:
            pass


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)