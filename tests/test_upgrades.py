"""Upgrade-verification tests: cooldown, zone_type, dwell-in-progress, heatmap
decay, rule-based wait time, prediction source labels, shelf strategies, the
buffered DB writer, camera sources and the API snapshot shape."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from app.schemas.models import Detection, Track
from app.services.analytics_service import parse_zones
from ml.queue.datalogger import QueueDataLogger
from ml.queue.predictor import QueuePredictor
from ml.queue.wait_time import WaitTimeEstimator
from ml.shelf.shelf_classifier import ShelfClassifier
from ml.shopper.dwell_time import ZoneDwellTracker
from ml.shopper.heatmap import HeatmapAccumulator
from ml.shopper.line_counter import LineCounter
from ml.tracking.bytetrack import ByteTrackTracker
from ml.tracking.factory import create_tracker
from ml.tracking.tracker import Tracker
from ml.sources.camera import CameraSource


def _det(x, y, w=30, h=60, cls=0, name="person", track_id=None):
    d = Detection(x - w / 2, y - h, x + w / 2, y, 0.9, cls, name, ts=time.time())
    d.track_id = track_id
    return d


def _track(x, y, tid):
    d = _det(x, y)
    return Track(id=tid, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                 confidence=d.confidence, class_id=d.class_id,
                 class_name=d.class_name, ts=d.ts, hit_streak=1, missed_frames=0)


# ---------------------------------------------------------------- line counter
def test_line_counter_cooldown_blocks_repeat_crossings():
    lc = LineCounter((0, 50), (100, 50), "store_01", {"cooldown_frames": 10})
    lc.update([_track(50, 20, 1)])           # above the line (center y=-10)
    lc.update([_track(50, 80, 1)])           # entry crossing
    lc.update([_track(50, 20, 1)])           # back above -> within cooldown, no exit
    assert lc.entries == 1 and lc.exits == 0
    assert lc.occupancy() == 1
    for _ in range(12):
        lc.update([_track(50, 15, 1)])       # linger above the line, cooldown expires
    lc.update([_track(50, 80, 1)])           # crossing again, cooldown elapsed
    assert lc.entries == 2 and lc.exits == 0
    assert lc.occupancy() == 2


def test_line_counter_entry_direction_up():
    lc = LineCounter((0, 50), (100, 50), "c", {"entry_direction": "up"})
    lc.update([_track(50, 80, 1)])
    lc.update([_track(50, 20, 1)])       # moving up now counts as entry
    assert lc.entries == 1 and lc.exits == 0


# ------------------------------------------------------------------ zones
def test_parse_zones_uses_zone_type():
    cfg = {"zones": {
        "aisle": {"zone_type": "shopping_zone", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        "checkout_01": {"zone_type": "queue_zone", "polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
        "checkout_legacy": {"polygon": [[40, 0], [50, 0], [50, 10], [40, 10]]},
    }}
    parsed = parse_zones(cfg)
    assert parsed["aisle"]["zone_type"] == "shopping_zone"
    assert parsed["checkout_01"]["zone_type"] == "queue_zone"
    # legacy zone without a type but 'checkout' in the name -> queue zone
    assert parsed["checkout_legacy"]["zone_type"] == "queue_zone"
    assert parsed["checkout_01"]["polygon"][0].tolist() == [20.0, 0.0]


# ------------------------------------------------------------------ dwell
def test_dwell_current_in_progress():
    zone = {"Q": [np.array(p, float) for p in [[0, 0], [40, 0], [40, 40], [0, 40]]]}
    dz = ZoneDwellTracker(zone, "c")
    dz.update([_track(20, 60, 1)], now=100.0)       # center (20, 30) inside
    dz.update([_track(19, 59, 1)], now=105.0)
    assert dz.current_dwell(now=105.0)["Q"] > 4
    dz.update([_track(90, 60, 1)], now=106.0)        # leaves -> completed visit
    assert dz.avg_dwell()["Q"] > 0


# ------------------------------------------------------------------ heatmap
def test_heatmap_decay_and_save(tmp_path):
    hm = HeatmapAccumulator(100, 100, scale=4, decay=0.9)
    hm.update([_track(20, 70, 1)])                   # center (20, 40) on the grid
    before = float(hm.grid.max())
    hm.update([])                                    # decay applies with no tracks
    assert hm.grid.max() < before
    path = tmp_path / "h.png"
    assert hm.save(path) is True
    import cv2
    img = cv2.imread(str(path))
    assert img is not None and img.shape[0] == 100


# ------------------------------------------------------------------ wait time
def test_wait_time_rule_based():
    est = WaitTimeEstimator({"average_service_time_seconds": 30, "open_counters": 3})
    assert est.estimate(3, 3) == pytest.approx(0.5)     # 3 * 0.5min / 3
    assert est.estimate(6, 3) == pytest.approx(1.0)
    assert est.estimate(0) == 0.0
    assert est.explain()["kind"] == "rule_based"


# ------------------------------------------------------------------ predictor
def test_predictor_labels_source_and_verbose_keys():
    # Point at a nonexistent model so the fallback path is exercised deterministically.
    pr = QueuePredictor({"horizon_minutes": [5, 10],
                         "model_path": "models/prediction/does_not_exist.joblib"}, {})
    series = [(t, int(round(t / 10))) for t in range(0, 600, 10)]
    preds = pr.predict(series, footfall=10, open_counters=3)
    assert preds["source"] == "fallback"
    assert preds["predicted_queue_length_5min"] == preds["5min"]
    assert preds["predicted_queue_length_10min"] == preds["10min"]
    assert "congestion" in pr.recommendation(preds, current_queue=6).lower()


# ------------------------------------------------------------------ shelf
def test_shelf_strategy_auto_prefers_detection_with_products():
    shelves = {"a": {"region": [[0, 0], [40, 0], [40, 40], [0, 40]], "expected_item_count": 10}}
    sc = ShelfClassifier(shelves, {"strategy": "auto"}, "c")
    prods = [Detection(5 + i * 3, 5, 5 + i * 3 + 6, 11, 0.9, 999, "product")
             for i in range(10)]
    assert sc.resolve_strategy(has_product_detections=True) == "detection"
    sc.update(np.zeros((100, 100, 3), dtype=np.uint8), prods, 100.0)
    snap = sc.snapshot()[0]
    assert snap["source"] == "detection"
    assert snap["status"] == "FULL"


def test_shelf_strategy_auto_falls_to_heuristic_without_products():
    shelves = {"a": {"region": [[0, 0], [40, 0], [40, 40], [0, 40]], "expected_item_count": 10}}
    sc = ShelfClassifier(shelves, {"strategy": "auto"}, "c")
    assert sc.resolve_strategy(has_product_detections=False) == "heuristic"
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sc.update(frame, [], 100.0)
    assert sc.active_strategy == "heuristic"
    assert len(sc.snapshot()) == 1


# ------------------------------------------------------------------ tracker factory
def test_tracker_factory_backends():
    assert isinstance(create_tracker("iou", {}, "c"), Tracker)
    assert isinstance(create_tracker("auto", {}, "c"), Tracker)
    bytetrack = create_tracker("bytetrack", {}, "c")
    assert isinstance(bytetrack, ByteTrackTracker)
    assert bytetrack.update([], None) == []   # no model -> graceful empty


# ------------------------------------------------------------------ camera source
def test_camera_source_rejects_missing_file():
    src = CameraSource("/nonexistent/definitely_missing.mp4", processing={
        "max_fps": 0, "retry_interval_seconds": 0.01, "reconnect_max_attempts": 1},
        video_loop=False)
    frame, dets = src.next_frame()
    assert frame is None and dets == []
    assert src.status in ("RECONNECTING", "OFFLINE")
    src.release()


def test_camera_source_reads_real_video(tmp_path):
    import cv2

    path = str(tmp_path / "sample.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    for i in range(8):
        writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
    writer.release()

    src = CameraSource(path, processing={"max_fps": 0, "retry_interval_seconds": 0.01,
                                         "reconnect_max_attempts": 10}, video_loop=True)
    ok = src.open()
    assert ok
    frames = 0
    for _ in range(20):
        frame, dets = src.next_frame()
        if frame is not None:
            frames += 1
        if frames >= 9:   # more than the file has -> looped
            break
    assert frames >= 9
    src.release()


# ------------------------------------------------------------------ datalogger
def test_queue_datalogger_writes_rows(tmp_path):
    path = tmp_path / "features.csv"
    logger = QueueDataLogger(path, "camera_x", ["checkout_01"], sample_interval=0.01)
    logger.update({"checkout_01": [(1.0, 2), (2.0, 3)]}, footfall=7, open_counters=3, now=10.0)
    logger.update({"checkout_01": [(3.0, 4)]}, footfall=7, open_counters=3, now=10.1)
    logger.close()
    text = path.read_text()
    assert "camera_id,queue_id" in text.splitlines()[0]
    assert text.count("checkout_01") >= 2
    assert "target" in text.splitlines()[0]


# ------------------------------------------------------------------ DB writer
def test_background_writer_persists_and_shuts_down(tmp_path):
    from database.models import build_session_factory
    from database.writer import BackgroundWriter

    url = f"sqlite:///{tmp_path / 'w.db'}"
    factory = build_session_factory(url)
    writer = BackgroundWriter(factory, flush_interval=0.05)
    for i in range(5):
        writer.submit("snapshot", timestamp=__import__("datetime").datetime.utcnow(),
                      camera_id="c", footfall_count=i)
    writer.submit("alert", timestamp=__import__("datetime").datetime.utcnow(),
                  camera_id="c", alert_type="congestion", severity="HIGH",
                  message="test")
    writer.shutdown(flush_timeout=2.0)

    from database.models import AlertRecord, AnalyticsSnapshot

    session = factory()
    assert session.query(AnalyticsSnapshot).count() == 5
    assert session.query(AlertRecord).count() == 1
    session.close()


# ------------------------------------------------------------------ predictor: real models
def test_predictor_loads_per_horizon_models_and_blends(tmp_path):
    """The predictor auto-discovers the per-horizon joblib files and reports
    model/blend with intervals + confidence when trained models are present."""
    if not (Path("models/prediction/queue_model_5min.joblib").exists()):
        pytest.skip("no trained per-horizon models available in this checkout")
    pr = QueuePredictor({"horizon_minutes": [5, 10]}, {})
    assert any(m is not None for m in pr._models.values())
    series = [(t, int(round(t / 40))) for t in range(0, 3000, 10)]
    preds = pr.predict(series, footfall=10, open_counters=3)
    assert preds["source"] in ("model", "blend")
    assert preds["predicted_queue_length_5min"] == preds["5min"]
    assert preds["predicted_queue_length_10min"] == preds["10min"]
    # per-horizon explainability carries the linear-trend factor
    for h in pr.horizons:
        exp = preds[f"explain_{h}min"]
        assert "linear_trend_value" in exp
        assert "model_value" in exp
        iv = preds[f"interval_{h}min"]
        assert 0 <= iv["low"] <= iv["high"]
    assert 0.0 <= preds["confidence"] <= 1.0


def test_predictor_blend_weight_from_metrics_file():
    """The blend weight is read per-horizon from the metrics file."""
    if not Path("models/prediction/queue_metrics.json").exists():
        pytest.skip("no queue_metrics.json in this checkout")
    pr = QueuePredictor({"horizon_minutes": [5, 10]}, {})
    w = pr._blend_weight_for(5)
    assert 0.0 <= w <= 1.0


def test_predictor_fallback_uses_default_blend_weight():
    """Without a metrics file the neutral prior blend weight is used."""
    pr = QueuePredictor({"horizon_minutes": [5, 10],
                         "metrics_path": "models/prediction/missing_metrics.json",
                         "model_path": "models/prediction/does_not_exist.joblib"}, {})
    assert pr._blend_weight_for(5) == pytest.approx(0.55)


# ------------------------------------------------------------------ prediction evaluator
def test_prediction_evaluator_scores_ripe_samples(tmp_path):
    from ml.queue.evaluator import QueuePredictionEvaluator

    ev = QueuePredictionEvaluator(tmp_path / "eval.csv", horizons=[5])
    # history that grows to match a prediction of 10 at t=600 (5min horizon)
    hist = [(i * 10.0, float(i)) for i in range(65)]  # up to t=640
    ev.record(made_ts=300.0, horizons_values={5: 30.0}, queue_history=hist, source="blend")
    assert ev.metrics()["5min"]["samples"] == 0  # not ripe yet
    ev.evaluate_ripe(now=300.0 + 5 * 60 + 1)     # just past the 5-min horizon
    assert ev.metrics()["5min"]["samples"] == 1
    m = ev.metrics()["5min"]
    assert m["sources"].get("blend") == 1
    assert m["mae"] >= 0.0
    ev.close()
    assert (tmp_path / "eval.csv").exists()


def test_prediction_evaluator_skips_unresolvable(tmp_path):
    from ml.queue.evaluator import QueuePredictionEvaluator

    ev = QueuePredictionEvaluator(tmp_path / "eval2.csv", horizons=[5])
    # history ends before the horizon -> cannot resolve an actual -> dropped
    hist = [(i * 10.0, float(i)) for i in range(30)]  # only up to t=290
    ev.record(made_ts=300.0, horizons_values={5: 10.0}, queue_history=hist, source="fallback")
    ev.evaluate_ripe(now=300.0 + 5 * 60 + 1)
    assert ev.metrics()["5min"]["samples"] == 0


# ------------------------------------------------------------------ shelf: temporal smoothing + depletion
def _shelf_dets(n):
    return [Detection(5 + i * 3, 5, 5 + i * 3 + 6, 11, 0.9, 999, "product") for i in range(n)]


def test_shelf_confirmation_requires_consecutive_polls():
    from ml.shelf.shelf_classifier import OUT_OF_STOCK

    shelves = {"a": {"region": [[0, 0], [40, 0], [40, 40], [0, 40]], "expected_item_count": 10}}
    sc = ShelfClassifier(shelves, {"strategy": "detection", "confirmation_polls": 3,
                                   "poll_interval_seconds": 0.001}, "c")
    now = 1000.0
    sc.update(None, _shelf_dets(10), now)          # FULL, first observation committed
    assert sc.states["a"].confirmed is True
    assert sc.states["a"].status != OUT_OF_STOCK

    # a single low-count poll must NOT flip the state to LOW_STOCK
    sc.update(None, _shelf_dets(2), now + 0.01)    # 2/10 = 0.2 -> LOW candidate
    s = sc.states["a"]
    assert s.status == "LOW_STOCK"
    assert s.confirmed is False                    # provisional, not committed

    # one more consistent poll is still not enough (needs 3 consecutive)
    sc.update(None, _shelf_dets(2), now + 0.02)
    assert sc.states["a"].confirmed is False

    # the third consecutive consistent poll confirms the transition
    sc.update(None, _shelf_dets(2), now + 0.03)
    assert sc.states["a"].status == "LOW_STOCK"
    assert sc.states["a"].confirmed is True


def test_shelf_depletion_trend_and_time_to_out():
    shelves = {"a": {"region": [[0, 0], [40, 0], [40, 40], [0, 40]], "expected_item_count": 10}}
    sc = ShelfClassifier(shelves, {"strategy": "detection", "confirmation_polls": 2,
                                   "poll_interval_seconds": 1.0}, "c")
    now = 2000.0
    sc.update(None, _shelf_dets(10), now)          # 10/10 FULL
    sc.update(None, _shelf_dets(4), now + 1.0)     # 4/10 = 0.4 FULL
    sc.update(None, _shelf_dets(2), now + 2.0)     # 2/10 = 0.2 LOW_STOCK
    snap = sc.states["a"]
    assert snap.trend < 0.0          # depleting
    assert snap.est_time_to_out_minutes > 0.0
    assert snap.stock_out_risk in ("LOW", "MEDIUM", "HIGH")
    assert snap.status == "LOW_STOCK"


# ------------------------------------------------------------------ yolo detector: onnx/fallback handling
def test_yolo_detector_reports_backend_none_when_model_missing():
    from ml.detection.yolo_detector import YoloDetector

    det = YoloDetector({"yolo_model": "models/yolo/does_not_exist.pt"}, "c")
    assert det.backend == "none"
    assert det.is_onnx is False
    assert det.uses_synthetic() is True  # no model + no session
    assert det.detect(None) == []
    assert det.predict(None) is None


def test_yolo_detector_picks_onnx_backend_for_onnx_path(tmp_path):
    """An .onnx path routes to the ONNX backend (or degrades cleanly when the
    lib or file is missing - the contract is never a crash or a wrong backend)."""
    from ml.detection.yolo_detector import YoloDetector

    fake = tmp_path / "detector.onnx"
    fake.write_bytes(b"not-a-real-onnx")
    det = YoloDetector({"yolo_model": str(fake)}, "c")
    assert det.is_onnx is False if det.backend == "none" else True
    # If onnxruntime is present it will try to load and fail cleanly; otherwise
    # backend stays "none". Either way detect/predict never crash.
    assert det.detect(None) == []
    assert det.predict(None) is None


# ------------------------------------------------------------------ training: baselines + feature parity
def test_queue_metrics_include_baselines():
    """Training must record honest baselines (persistence/mean) so the model's
    skill can be judged against 'predict the same queue' rather than bare R2."""
    mp = Path("models/prediction/queue_metrics.json")
    if not mp.exists():
        pytest.skip("no queue_metrics.json in this checkout")
    import json
    d = json.loads(mp.read_text())
    assert d["models"]
    for m in d["models"]:
        assert "baseline_persistence_mae" in m
        assert "baseline_persistence_r2" in m
        assert "baseline_mean_mae" in m
        assert "beats_persistence" in m
    five = [m for m in d["models"] if m["horizon_minutes"] == 5][0]
    assert five["holdout_mae"] <= five["baseline_persistence_mae"] or five["beats_persistence"]


def test_predictor_feature_row_matches_training_features():
    """The runtime feature vector must align with the FEATURES list the models
    were trained on, and be the documented length, or predictions drift."""
    import scripts.train_queue_model as t
    from ml.queue.predictor import QueuePredictor
    assert len(t.FEATURES) == 12
    pr = QueuePredictor({"horizon_minutes": [5, 10],
                         "model_path": "models/prediction/does_not_exist.joblib"}, {})
    series = [(t * 10.0, int(round(t / 40))) for t in range(0, 600, 10)]
    row = pr._feature_row(series, footfall=10, open_counters=3, now=series[-1][0])
    assert len(row) == len(t.FEATURES)
    assert all(np.isfinite(row))