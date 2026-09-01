# SIH 2026 - AI-Powered Retail Intelligence Platform

Converts live video into privacy-focused retail insights: shopper footfall and
movement, per-zone dwell time, entry/exit counting, queue intelligence with
future-congestion prediction, and shelf FULL/LOW/OUT state - exposed live
through a FastAPI dashboard with WebSockets.

Built to spec (`SIH26179_Retail_Intelligence_Master_Project_Specification.docx`).
**Facial recognition is not used anywhere.** People are tracked with anonymous
temporary ids only.

## Hardware

The pipeline runs the same code in three modes:

| Mode | Input | ML required? | Good for |
|---|---|---|---|
| `demo` | built-in simulator (no camera/GPU/model) | no | development, dashboards, experiments |
| `video` | recorded `.mp4` / `.avi` footage | detection model optional | algorithm tuning without a camera |
| `live` | webcam `0` or RTSP URL | detection model recommended | the real deployment |

## Quick start (demo - no camera, no GPU, no model downloads)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --mode demo --display      # window + API on :8000
```

Point a browser at http://localhost:8000/docs and watch `/analytics/current`
and `/ws/live` update in real time. `--display` opens an OpenCV window showing
the simulated store floor, zones, queue and shelf products as they drain.

Other modes:

```bash
python main.py --mode video --source clip.mp4 --display   # recorded footage
python main.py --mode live  --source 0 --display          # webcam
python main.py --mode live  --source rtsp://...           # IP camera
python main.py --mode demo --no-api --run-seconds 30      # headless analytics
```

All real-source modes use `ml/sources/camera.py` (`CameraSource`) with automatic
reconnect, retry limits and an FPS cap, so a dead camera never takes the loop
down. Without a YOLO checkpoint, detection falls back to tracking raw
background-motion blobs; install `ultralytics` + a checkpoint for proper
person detection.

## What's implemented (per spec section 7-8 + 18)

| Module | Status | Notes |
|---|---|---|
| Person detection | io | `YoloDetector` (Ultralytics: `imgsz`, `classes`); fallback motion detection |
| Multi-object tracking | io | `BaseTracker` factory: `iou` (default) or `bytetrack` (needs ultralytics); persistent anonymous ids, stale-track expiry, per-track history for heatmaps |
| Entry/exit counting | io | directional virtual-line crossing, per-person cooldown to stop repeat counts, occupancy `= max(0, entries - exits)` |
| Zone analytics + dwell | io | named `zone_type` zones (`shopping/queue/shelf/entrance/exit`), avg dwell + current-dwell + occupancy per zone |
| Movement heatmaps | io | `/analytics/heatmap` returns a PNG; optional per-frame decay + periodic export to `data/processed/heatmap.png` |
| Queue detection + metrics | io | `queue_zone` polygons, per-checkout counts, history, growth, rule-based wait estimate |
| Congestion prediction | io | per-queue ML prediction (`model` / `blend` / `fallback` source labels) over 5/10 min |
| Wait time | io | rule-based `length × average_service_time_seconds / open_counters` (no ML needed); `explain()` returns the numbers |
| Shelf FULL/LOW/OUT | io | strategy `auto → classification (CNN) / detection / heuristic`; heuristic = no-ML default |
| Planogram compliance | opt | `PlanogramChecker` scaffolding |
| FastAPI + WebSockets | io | all section-12 endpoints; WS snapshots carry ISO `ts` + `ts_epoch` |
| Persistence | io | `BackgroundWriter` background batch-commit to SQLite (PostgreSQL ready via `DATABASE_URL`); throttled snapshots + trajectory points never block the loop |
| Edge export | scaffold | `scripts/export_onnx.py` + `deployment/edge/` + `scripts/benchmark.py` |

## Project layout

```
retail_intelligence/
├── main.py                  # CLI entry point (validates the source up front)
├── config/                  # settings.yaml, cameras.yaml, zones.json, loader
├── demo/                    # synthetic store simulator (demo source)
├── ml/
│   ├── detection/           # YoloDetector wrapper + motion fallback
│   ├── tracking/            # base + IoU + ByteTrack factory
│   ├── sources/             # CameraSource (video/live/RTSP), DemoSimulator adapter
│   ├── shopper/             # footfall, line_counter, dwell_time, heatmap
│   ├── queue/               # queue_counter, wait_time, predictor, datalogger
│   ├── shelf/               # shelf_classifier, planogram
│   └── geometry.py          # polygon / line / IoU primitives
├── app/
│   ├── api/                 # FastAPI routes + WebSocket hub
│   ├── services/            # analytics orchestrator, alerts, inference loop
│   └── schemas/             # shared DTOs
├── database/                # SQLAlchemy models + repository + BackgroundWriter
├── scripts/                 # training, dataset prep, benchmark, ONNX export
├── tests/                   # pytest suite (33 tests)
├── models/                  # yolo + prediction checkpoints
├── data/                    # raw / processed / training
└── deployment/              # docker + edge notes
```

## API endpoints

```
GET  /health                 engine + per-camera source/detector/tracker + db writer status
GET  /analytics/current      full live snapshot (tracks, footfall incl. occupancy, queues, shelves)
GET  /analytics/footfall     cumulative + per-minute entry/exit series
GET  /analytics/dwell        avg dwell / current-dwell / occupancy per zone
GET  /analytics/queues       per-queue counts, history, wait estimate, predictions, recommendation
GET  /analytics/shelves      per-shelf status + summary (+ CNN source when used)
GET  /analytics/heatmap      PNG heatmap of movement intensity
GET  /alerts                 persisted congestion alerts
GET  /config/zones           current zone polygons
POST /config/zones           hot-reload zone polygons (persisted to zones.json)
WS   /ws/live                live_snapshot broadcasts every second
```

Example snapshot (section 13 event shape):

```json
{
  "camera_id": "store_01", "ts": 1.7e9, "ts_epoch": 1.7e9,
  "footfall": {"entries": 31, "exits": 4, "occupancy": 27},
  "queues": {
    "prediction_source": "blend",
    "queues": [
      {"queue_id": "checkout_01", "length": 7, "wait_minutes": 1.1,
       "status": "WARNING", "predictions": {"5min": 6.4, "10min": 8.9},
       "recommendation": "Predicted congestion in ~10 minutes - open an additional counter."}
    ]
  },
  "congestion_status": "HIGH",
  "shelves": [{"shelf_id": "shelf_a", "status": "OUT_OF_STOCK", "item_count": 0, "source": "heuristic"}]
}
```

## Training + models

```bash
# Queue-length predictor: a SEPARATE model per forecast horizon, each tuned
# between RandomForest vs GradientBoosting via forward-chaining cross-val
# (no shuffling - respects the temporal order). Training also records honest
# baselines (persistence = "queue stays the same", naive mean) and a per-horizon
# blend weight + 80% CI, so the model is judged against what "no model" would do
# rather than a bare R² (which is naturally weak on noisy short-horizon queue
# data even for a good model).
# Synthetic mode (no data): produce a demo model quickly.
python scripts/train_queue_model.py --samples 2000

# REAL mode: train on a CSV of live features.
#   - collect data first:
#       python main.py --mode live --source 0      # pipeline logs to data/processed/queue_features.csv
#   - then train (per-queue timelines, target = queue length at horizon):
#       python scripts/train_queue_model.py --csv data/processed/queue_features.csv --horizons 5 10
#   Writes models/prediction/queue_model_{N}min.joblib (one per horizon),
#   plus queue_metrics.json (source=real) with per-horizon blend weights + CIs.
#   (The legacy single-file queue_model.joblib is superseded; delete it.)

# Shelf classifier (needs torch): ImageFolder training -> accuracy/F1 + .metrics.json
python scripts/train_shelf_model.py --data data/shelf --epochs 10
python scripts/make_shelf_dataset.py --out data/shelf     # build a small labelled set from a video

# Edge export (requires ultralytics + a downloaded yolov8n.pt)
python scripts/export_onnx.py --model models/yolo/yolov8n.pt --format onnx --half
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"   # download helper

# Benchmark FPS/latency/model-size/RAM (plus ONNX-throughput validation)
python scripts/benchmark.py --mode demo --seconds 30
```

`prediction.log_path` (default `data/processed/queue_features.csv`) is written
by the live pipeline (see `ml/queue/datalogger.py`) whenever
`prediction.log_path` config is set. The `QueuePredictor` reports its source:
`model` = trained model only, `blend` = model with bounded linear-trend fallback
using a validation-selected weight, `fallback` = trend-only (no model file yet).
Every prediction also carries per-horizon `interval_{N}min` (80% CI), a normed
`confidence` score, verbose `predicted_queue_length_{N}min` keys, and
`explain_{N}min` factor dicts so the dashboard can show *why*.

Runtime accuracy is tracked live by `ml/queue/evaluator.py`: each forecast is
resolved against the actual queue once the horizon elapses and MAE/RMSE per
horizon are written to `prediction.eval_path`
(`data/processed/prediction_eval.csv`) and surfaced via
`/api/health -> prediction_monitoring`. The runtime predictor builds the same
12-feature vector the models were trained on (see
`QueuePredictor._feature_row`), so deployed predictions stay aligned with
training.

Shelf snapshots are temporally smoothed: a status change (e.g. to LOW or OUT)
is only *committed* after `confirmation_polls` consecutive consistent polls, and
each shelf carries a depletion `trend`, `est_time_to_out_minutes` and
`stock_out_risk` derived from recent counts.

## Tests

```bash
python -m pytest tests/ -q
```

44 tests cover geometry, tracking id stability, entry/exit + cooldown,
occupancy, dwell, heatmap decay + export, queue counter + predictor source
labels + per-horizon models/CI/blend weight + training baselines/feature parity,
the runtime prediction evaluator, shelf classification + confirmation/depletion,
the YOLO ONNX/fallback backends, the background DB writer, CameraSource
(reconnect/loop/fps-cap), zones parsing and the API snapshot contract. API
tests need `httpx2` (skip cleanly otherwise).

## What needs real-world data / calibration

- **Zones & entrance line** - re-draw `config/zones.json` to the exact camera
  view. Zones carry `zone_type` (`shopping_zone`, `queue_zone`, `shelf_zone`,
  `entrance`, `exit`); verify the entry-line orientation (`line_counter.entry_direction`).
- **Queue service rate** - `config/settings.yaml → queue.average_service_time_seconds`
  drives the rule-based wait estimate; measure real checkout throughput on site.
- **Shelf item thresholds** - `shelf.low_stock_threshold` / `out_of_stock_threshold`
  vs `expected_item_count` per shelf; or train the CNN with `train_shelf_model.py`.
- **Product detector** - real shelf classification needs a product/object model
  (the demo reports products from the simulator).
- **Prediction model** - once `--csv` logs accumulate, re-train on site logs and
  validate MAE in `models/prediction/queue_metrics.json`.

## Edge deployment path

Laptop/colab experiments → export ONNX (FP16/INT8) → TensorRT/OpenVINO per
device → `scripts/benchmark.py` for FPS, latency percentiles, model size and
RAM/CPU/GPU utilisation. See `deployment/edge/README.md` and
`deployment/docker/` for containers.

## Integration checklist

1. Set `demo.enabled: false` and a real `cameras.yaml` source (webcam or RTSP).
2. Place a person-detection checkpoint at `models/yolo/yolov8n.pt` and install
   `ultralytics` (optional but recommended; set `tracking.backend: bytetrack`).
3. Re-publish `zones.json` (with `zone_type`) for each camera view.
4. Point the software team's dashboard at `/ws/live` + REST endpoints.
5. Log ~1 week of queue history, re-train the queue predictor, validate MAE.
6. Export + quantize the detector for the chosen edge device; benchmark.