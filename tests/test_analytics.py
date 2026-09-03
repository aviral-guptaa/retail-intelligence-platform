"""Unit tests for zone dwell, queue counter and heatmap."""
import numpy as np
import time

from app.schemas.models import Detection
from ml.queue.queue_counter import QueueCounter
from ml.queue.predictor import QueuePredictor
from ml.shopper.dwell_time import ZoneDwellTracker
from ml.shopper.heatmap import HeatmapAccumulator
from ml.shelf.shelf_classifier import ShelfClassifier
from ml.shelf.planogram import PlanogramChecker


def _track(x, y, tid):
    d = Detection(x - 10, y - 20, x + 10, y, 0.9, 0, "person")
    from app.schemas.models import Track
    tr = Track(id=tid, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
               confidence=d.confidence, class_id=0, class_name="person")
    return tr


def test_dwell_time_counts_equivalently():
    zone = {"A": [np.array(p, float) for p in [[0, 0], [40, 0], [40, 40], [0, 40]]]}
    dz = ZoneDwellTracker(zone, "store_01")
    t = _track(20, 20, 1)
    dz.update([t])                       # enters
    time.sleep(0.05)
    t2 = _track(20, 20, 1)
    dz.update([t2])
    assert dz.occupancy() == {"A": 1}
    t3 = _track(90, 90, 1)               # leaves zone A
    dz.update([t3])
    assert dz.occupancy() == {}
    assert dz.avg_dwell()["A"] > 0


def test_queue_counter_counts_and_histories():
    zone = {"checkout_01": [np.array(p, float) for p in [[0, 0], [40, 0], [40, 40], [0, 40]]]}
    qc = QueueCounter(zone, {"sample_interval_seconds": 0.1, "history_window": 60}, "store_01")
    now = 100.0
    qc.update([_track(20, 20, 1), _track(20, 10, 2)], now)
    assert qc.counts()["checkout_01"] == 2
    assert qc.total_queued() == 2
    qc.update([_track(90, 90, 1)], now + 0.2)   # both left
    assert qc.total_queued() == 0
    assert len(qc.history()["checkout_01"]) == 2


def test_predictor_linear_fallback_forecasts():
    # Point at a nonexistent model so the linear fallback is exercised
    # deterministically regardless of any ambient model files on disk.
    pr = QueuePredictor({"horizon_minutes": [5, 10],
                         "model_path": "models/prediction/does_not_exist.joblib"}, {})
    series = [(t, int(round(t / 10))) for t in range(0, 600, 10)]  # growing queue
    preds = pr.predict(series, footfall=10, open_counters=3)
    assert preds["10min"] > preds["5min"] > 0
    assert "congestion" in pr.recommendation(preds, current_queue=6).lower()


def test_predictor_linear_fallback_constant_queue_no_crash():
    # Regression: a flat queue history used to produce identical x (queue length)
    # for polyfit -> "SVD did not converge". Must not crash and must forecast
    # sensibly. queue_history is [(ts, queue_len), ...].
    pr = QueuePredictor({"horizon_minutes": [5, 10],
                         "model_path": "models/prediction/does_not_exist.joblib"}, {})
    series = [(t * 10.0, 5) for t in range(30)]  # constant queue of 5
    preds = pr.predict(series, footfall=4, open_counters=2)
    # constant queue -> should stay near 5, not explode
    assert 0 < preds["5min"] <= 6
    assert preds["10min"] >= preds["5min"]


def test_predictor_linear_fallback_slope_uses_time_not_index():
    # The slope must be fit against elapsed TIME so that sparse vs dense history
    # yields the same per-minute growth. With the old swapped (queue,len) bug,
    # growth came out ~1.0 per sample regardless of timestamp spacing.
    pr = QueuePredictor({"horizon_minutes": [5, 10],
                         "model_path": "models/prediction/does_not_exist.joblib"}, {})
    # queue climbs 0->5 over 60 seconds (5 samples, 15s apart): 5/min growth? No:
    # over 60s that is 5 per minute -> ~5/min, forecast 5-min ahead ~ +25 -> large.
    series = [(t * 15.0, int(t)) for t in range(6)]  # 0,1,2,3,4,5 queue over 90s? last ts=75
    preds = pr.predict(series, footfall=4, open_counters=2)
    # growth = (5-0)/(75-0) per sec *60 = 4/min; monotonic in horizon
    assert preds["10min"] > preds["5min"]



def test_heatmap_accumulates_and_renders():
    hm = HeatmapAccumulator(100, 100, scale=4)
    hm.update([_track(20, 20, 1), _track(21, 21, 2)])
    img = hm.to_image()
    assert img.shape[0] == 100 and img.shape[1] == 100
    assert img.max() > 0


def test_shelf_classifier_statuses():
    shelves = {"a": {"region": [[0, 0], [40, 0], [40, 40], [0, 40]], "expected_item_count": 10}}
    sc = ShelfClassifier(shelves, {}, "store_01")
    # 0 products -> OUT_OF_STOCK
    snap = sc.classify_by_counting([], "a")
    assert snap.status == "OUT_OF_STOCK"
    prods = [Detection(5 + i * 3, 5, 5 + i * 3 + 6, 11, 0.9, 999, "product") for i in range(10)]
    snap2 = sc.classify_by_counting(prods, "a")
    assert snap2.status == "FULL"
    snaps = []
    sc.update(None, prods, 100.0)
    assert len(sc.snapshot()) == 1


def test_planogram_violation_on_empty_shelf():
    pc = PlanogramChecker({"a": {"expected_columns": 3, "expected_rows": 4}})
    region = [np.array(p, float) for p in [[0, 0], [40, 0], [40, 40], [0, 40]]]
    v = pc.check("a", region, [], 1.0)
    assert any(x.kind == "MISSING_ITEM" for x in v)