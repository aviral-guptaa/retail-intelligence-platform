"""Inference pipeline: runs the analytics loop in a background thread and
broadcasts snapshots over WebSockets to connected dashboards."""
from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from config.loader import load_zones
from demo.simulator import DemoSimulator
from database.repository import Repository
from database.writer import BackgroundWriter
from ml.sources.camera import CameraSource
from app.services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Drives a set of cameras; exposes live snapshots and a WS broadcaster."""

    def __init__(self, settings: Dict[str, Any], repository: Repository,
                 db_writer: Optional[BackgroundWriter] = None,
                 display: bool = False):
        self.settings = settings
        self.repo = repository
        self.db_writer = db_writer
        self.display = display
        self.services: Dict[str, AnalyticsService] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._csv_writer = None
        self._csv_path = None

    # --------------------------------------------------------------- builders
    def add_camera(self, camera_id: str, mode: str = "demo", source: str | int | None = None,
                   detector=None) -> AnalyticsService:
        zones_config = load_zones()
        if mode == "demo":
            sim = DemoSimulator(camera_id, self.settings, zones_config)
            svc = AnalyticsService(camera_id, self.settings, zones_config, self.repo,
                                   source=sim, db_writer=self.db_writer)
        else:
            from ml.detection.yolo_detector import YoloDetector

            det = detector or YoloDetector(self.settings.get("model", {}), camera_id)
            src = source if source is not None else 0
            processing = self.settings.get("processing", {})
            video_loop = bool(self.settings.get("video", {}).get("loop", True))
            cam = CameraSource(src, camera_id, detector=det,
                               processing=processing, video_loop=video_loop)
            cam.open()
            svc = AnalyticsService(camera_id, self.settings, zones_config, self.repo,
                                   source=cam, detector=det, db_writer=self.db_writer)
        self.services[camera_id] = svc
        return svc

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="inference", daemon=True)
        self._thread.start()
        logger.info("inference pipeline started")

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        for svc in self.services.values():
            svc.shutdown()
        if self.db_writer is not None:
            self.db_writer.shutdown()
        logger.info("inference pipeline stopped")

    def _run_loop(self) -> None:
        # Demo cadence comes from demo.fps; real feeds use processing.max_fps
        # throttling inside CameraSource, so the outer loop just sleeps briefly.
        fps = float(self.settings.get("demo", {}).get("fps", 10) or 10)
        processing = self.settings.get("processing", {})
        max_fps = float(processing.get("max_fps", 0) or 0)
        target_fps = max_fps if max_fps > 0 else fps
        interval = 1.0 / max(target_fps, 1e-9)

        while self._running:
            t0 = time.monotonic()
            snapshot = self.step_all()
            if self.display:
                self._show(snapshot)
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
            if self._stop_event.is_set():
                break
            for svc in self.services.values():
                if hasattr(svc.source, "finished") and svc.source.finished:
                    logger.info("source %s finished; pipeline stopping", svc.camera_id)
                    self._stop_event.set()
                    break

    def step_all(self) -> Dict[str, Any]:
        out: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        for cid, svc in self.services.items():
            t0 = time.monotonic()
            try:
                out[cid] = svc.step()
                svc.monitor.step_ok(time.monotonic() - t0, time.time())
            except Exception as exc:
                logger.exception("step failed for %s", cid)
                svc.monitor.step_error(time.monotonic() - t0)
                out[cid] = {"camera_id": cid, "error": str(exc)}
            # Fold the camera source status into the monitor (OFFLINE/RECONNECTING)
            # and raise a deduplicated CAMERA_OFFLINE alert through the unified store.
            try:
                svc.health()  # folds source_health into monitor
                st = svc.monitor.state
                store = svc.alert_store
                if st in ("OFFLINE", "RECONNECTING"):
                    store.emit("camera", "HIGH", f"Camera {cid} is {st}",
                               cid, store_id=svc.monitor.store_id,
                               source="monitor", dedup_key="camera_state")
                else:
                    store.resolve("camera_state", "camera")
            except Exception:
                pass
        return out

    def live_snapshot(self) -> Dict[str, Any]:
        snap = {cid: svc.current() for cid, svc in self.services.items()}
        for cid, svc in self.services.items():
            frame = getattr(svc, "frame_jpeg_b64", None)
            if callable(frame):
                jpg = frame()
                if jpg:
                    snap[cid] = {**snap[cid], "video_frame": jpg}
        return snap

    def single(self, camera_id: str) -> Dict[str, Any]:
        return self.services[camera_id].current()

    # ------------------------------------------------------- store / cameras
    def cameras(self) -> Dict[str, Any]:
        """Registry of known cameras with id, name, state, store and summary."""
        from config.loader import load_cameras
        meta = {c["id"]: c for c in load_cameras()}
        now = time.time()
        out = []
        for cid, svc in self.services.items():
            m = svc.monitor
            perf = m.snapshot(now)
            out.append({
                "id": cid,
                "name": meta.get(cid, {}).get("name", cid),
                "store_id": m.store_id,
                "state": perf["state"],
                "fps": perf["fps"],
                "frames": perf["frames_processed"],
                "reconnects": perf["reconnects"],
                "errors": perf["errors"],
                "last_frame_age_s": perf["last_frame_age_s"],
                "source_kind": meta.get(cid, {}).get("kind", "demo"),
                "zone_ids": meta.get(cid, {}).get("zone_ids", []),
                "shelf_ids": meta.get(cid, {}).get("shelf_ids", []),
            })
        return {"store_id": self.settings.get("app", {}).get("store_id"),
                "cameras": out}

    def performance(self) -> Dict[str, Any]:
        from ml.analytics.performance import PerformanceSummary
        return PerformanceSummary({cid: svc.monitor
                                   for cid, svc in self.services.items()}).snapshot()

    def stores(self) -> Dict[str, Any]:
        """Store-level analytics aggregated across its cameras (one store today)."""
        store_id = self.settings.get("app", {}).get("store_id", "store_01")
        totals = {"entries": 0, "exits": 0, "occupancy": 0, "unique": 0,
                  "queue_total": 0, "predicted_max": 0.0}
        by_status: Dict[str, int] = {}
        cameras_out = []
        for cid, svc in self.services.items():
            cur = svc.current()
            ff = cur.get("footfall", {})
            q = cur.get("queues", {})
            totals["entries"] += ff.get("total_entries", 0)
            totals["exits"] += ff.get("total_exits", 0)
            totals["occupancy"] += ff.get("occupancy", 0)
            totals["unique"] = max(totals["unique"], ff.get("unique_shoppers", 0))
            totals["queue_total"] += q.get("total", 0)
            totals["predicted_max"] = max(totals["predicted_max"],
                                          q.get("predicted_max", 0.0))
            st = cur.get("congestion_status", "NORMAL")
            by_status[st] = by_status.get(st, 0) + 1
            monitors = svc.monitor.snapshot()
            cameras_out.append({
                "id": cid,
                "state": monitors["state"],
                "congestion": st,
                "alerts_active": svc.alert_store.snapshot()["active_count"],
            })
        return {
            "store_id": store_id,
            "cameras": cameras_out,
            "totals": {
                "entries": totals["entries"],
                "exits": totals["exits"],
                "occupancy": totals["occupancy"],
                "unique_shoppers": totals["unique"],
                "queue_total": totals["queue_total"],
                "predicted_queue_max": round(totals["predicted_max"], 1),
            },
            "congestion_by_camera": by_status,
        }

    # ------------------------------------------------------------------- http
    def _show(self, snapshot: Dict[str, Any]) -> None:
        for cid, svc in self.services.items():
            frame = svc._last_frame
            if frame is None:
                continue
            overlay = frame.copy()
            status = snapshot[cid]["congestion_status"]
            q = snapshot[cid]["queues"]
            cv2.putText(overlay, f"QUEUE {q['total']}  {status}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(f"retail-intelligence {cid}", overlay)
        cv2.waitKey(1)

    # ---------------------------------------------------------------- logging
    def start_csv_log(self, path: Path) -> None:
        """Optional CSV dump of snapshots for offline prediction training."""
        self._csv_path = Path(path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_writer = open(self._csv_path, "w", newline="")
        self._csv_writer.write("ts,camera_id,entries,exits,active,occupancy,queue_length,growth_rate,predicted,status\n")

    def flush_csv(self, snapshot: Dict[str, Any]) -> None:
        if self._csv_writer is None:
            return
        for cid, row in snapshot.items():
            self._csv_writer.write(
                f"{row.get('ts', '')},{cid},"
                f"{row.get('footfall', {}).get('total_entries')},"
                f"{row.get('footfall', {}).get('total_exits')},"
                f"{row.get('footfall', {}).get('current_active')},"
                f"{row.get('footfall', {}).get('occupancy')},"
                f"{row.get('queues', {}).get('total')},"
                f"{row.get('queues', {}).get('growth_rate')},"
                f"{row.get('queues', {}).get('predictions', {}).get('10min', '')},"
                f"{row.get('congestion_status')}\n")

    def close_csv(self) -> None:
        if self._csv_writer is not None:
            self._csv_writer.close()
            self._csv_writer = None

    def __del__(self):
        self.close_csv()