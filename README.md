# SIH 2026 - AI-Powered Retail Intelligence Platform

Converts live video into privacy-focused retail insights: shopper footfall and
movement, per-zone dwell time, entry/exit counting, recognized queue
intelligence with future-congestion prediction, measured (per-person) checkout
wait times, and shelf FULL/LOW/OUT state - exposed live through a FastAPI
dashboard with WebSockets.

Built to spec (`SIH26179_Retail_Intelligence_Master_Project_Specification.docx`).
**Facial recognition is not used anywhere.** People are tracked with anonymous
temporary ids only; the optional appearance Re-ID derives a colour histogram of
the **body crop only** for cross-camera anonymous global ids (`g_N`).

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

### Web dashboard (upload a video + live dashboard)

A functional prototype dashboard where you upload a video (or run the demo
simulator) and the full pipeline runs server-side, streaming live metrics to the
page over WebSockets — occupancy, queue length, measured **and** estimated wait,
5/10-min forecasts with source/confidence, recommendations, shelf status with
source, a heatmap, the de-identified live frame (MJPEG), an active-alert feed,
per-camera health and hourly/daily historical charts:

```bash
python run_web.py       # -> http://127.0.0.1:8000/
```

`▶ Run demo` needs no camera; `Upload a video` runs the real vision pipeline on
your file (`python-multipart` is required for uploads). Uploaded clips are
**auto-deleted when the run finishes** unless `privacy.retain_uploaded_video`
is set (see `config/settings.yaml → privacy`); a startup sweep prunes stale
uploads per `retention_hours`.

## Genuine status (honest scorecard)

Every feature below is implemented and exercised by tests. Where a result
cannot be truthful yet (no trained model, no connected vendor system), the
platform reports that explicitly instead of fabricating numbers.

| Area | Status | Notes |
|---|---|---|
| Person detection | ✅ implemented | `YoloDetector` (Ultralytics: `imgsz`, `classes`); falls back to motion detection without a checkpoint |
| Multi-object tracking | ✅ implemented | `BaseTracker` factory: `iou` (default) or `bytetrack` (needs ultralytics); anonymous persistent ids |
| Entry/exit + occupancy | ✅ implemented | directional line crossing, per-person cooldown, `occupancy = max(0, entries-exits)` |
| Zone dwell + heatmaps | ✅ implemented | named zones, avg/current dwell per zone, PNG heatmap endpoint + periodic export |
| Queue detection | ✅ implemented | `queue_zone` polygons, per-checkout counts + history + growth |
| Wait time | ✅ implemented | rule-based **estimate** (`length × service_time / open_counters`) **and** per-person **measured** wait from queue-zone dwell (anonymous track ids) — both reported, never mixed |
| Congestion prediction (5/10 min) | ✅ implemented | per-horizon trained models, `model / blend / fallback` source labels, 80% CIs, confidence, `explain_` factors; live MAE tracker |
| Shelf FULL/LOW/OUT | ✅ implemented | `auto → classification (CNN) / detection / heuristic`; temporal confirmation, trend + time-to-out + risk |
| Planogram compliance | ✅ honest | integrated into the live pipeline; reports `MODEL_NOT_AVAILABLE` until a *product* model exists — never claims "OK" without one |
| Unified alerts | ✅ implemented | severity INFO/WARNING/HIGH, dedup, active + historical REST, optional webhook, congestion + shelf + CAMERA_OFFLINE |
| Stores & cameras | ✅ implemented | store aggregates, per-camera ONLINE/ERROR/RECONNECTING states, `/cameras`, `/system/performance` |
| Historical analytics | ✅ implemented | in-memory 1-min/1-hour buckets, hourly/daily summaries, JSON + CSV report downloads |
| Persistence | ✅ implemented | `BackgroundWriter` batch-commit to SQLite (Postgres via `DATABASE_URL`); snapshots, alerts, trajectories, queue/shelf events, stores, cameras, POS transactions |
| Privacy / retention | ✅ implemented | no face capture ever; uploads deleted on finish (configurable), startup sweep, `/api/privacy/*` |
| POS/ERP integration | ✅ adapter + honest | generic vendor-agnostic webhook ingest; conversion / ticket stats only when real data exists (`available: false` otherwise) |
| Edge/ONNX export | ✅ implemented | `scripts/export_onnx.py`, `scripts/benchmark.py`, `deployment/edge/` |

**What genuinely still needs trained models / data / credentials** (honest list):

- Queue predictor: shipped `models/prediction/queue_model_{5,10}min.joblib` were
  trained on a **realistic M/M/c synthetic** checkout dataset (positive R², beats
  the "queue stays the same" baseline). Real accuracy comes from re-training on
  `data/processed/queue_features.csv` collected in the live store.
- Shelf CNN: `models/prediction/shelf_classifier.pt` is **not trained** — the
  shelf module falls back to heuristic counting until you train it with
  `scripts/train_shelf_model.py`.
- Product detector for real planogram compliance needs a YOLO model fine-tuned
  on the store's products (the demo simulates products).
- POS/ERP: endpoints accept whatever your vendor pushes; nothing is wired to a
  specific vendor until you set `alerts.webhook_url` / feed the ingest endpoints.
- Zones, entrance line, service time and shelf thresholds need on-site
  calibration (documented in "What needs real-world data / calibration").
- Real webcam/RTSP footage needs a physical camera (unit tests run the
  demo/video modes in-process).

## Project layout

```
retail_intelligence/
├── main.py                  # CLI entry point (validates the source up front)
├── run_web.py               # web-dashboard entry point (upload video / demo)
├── config/                  # settings.yaml, cameras.yaml, zones.json, loader
├── demo/                    # synthetic store simulator (demo source)
├── ml/
│   ├── detection/           # YoloDetector wrapper + motion fallback
│   ├── tracking/            # base + IoU + ByteTrack factory
│   ├── sources/             # CameraSource (video/live/RTSP), DemoSimulator adapter
│   ├── shopper/             # footfall, line_counter, dwell_time, heatmap, reid
│   ├── queue/               # queue_counter, wait_time, measurer, predictor, datalogger, evaluator
│   ├── shelf/               # shelf_classifier, planogram
│   ├── analytics/           # history buckets, camera/performance monitor
│   ├── integrations/        # POS/ERP adapters (IntegrationHub)
│   └── geometry.py          # polygon / line / IoU primitives
├── app/
│   ├── api/                 # FastAPI routes + WebSocket hub + pydantic payloads
│   ├── services/            # analytics orchestrator, alert store, inference loop
│   └── schemas/             # DTOs (models.py) + API payload validation (api.py)
├── database/                # SQLAlchemy models + repository + BackgroundWriter
├── webserver/               # web dashboard backend + static/ dashboard
├── scripts/                 # training, dataset prep, benchmark, ONNX export
├── tests/                   # pytest suite (88 tests)
├── models/                  # yolo + prediction checkpoints
├── data/                    # raw / processed / uploads / training
└── deployment/              # docker + edge notes
```

## API endpoints

```
GET  /health                            engine + per-camera health + db writer status + prediction monitoring
GET  /analytics/current                 full live snapshot (tracks, footfall, queues, shelves, alerts, planogram)
GET  /analytics/footfall                cumulative + per-minute entry/exit series
GET  /analytics/dwell                   avg dwell / current-dwell / occupancy per zone
GET  /analytics/queues                  per-queue counts, history, wait estimate + measured, predictions
GET  /analytics/queues/events           persisted per-change queue events (DB mode)
GET  /analytics/shelves                 per-shelf status + summary (+ CNN source when used)
GET  /analytics/shelves/events          persisted committed shelf transitions (DB mode)
GET  /analytics/planogram               planogram compliance per shelf (MODEL_NOT_AVAILABLE without a product model)
GET  /analytics/heatmap                 PNG heatmap of movement intensity
GET  /analytics/history                 minute + hourly rolling history (JSON)
GET  /analytics/daily                   daily summaries + peak hours
GET  /analytics/report.csv              CSV report of minute + hourly history
GET  /analytics/current                 live snapshot
GET  /alerts/active                     active alerts with severity counts + webhook state
GET  /alerts/historical                 resolved/expired alert history
GET  /cameras                           registered cameras
GET  /cameras/{id}/health               per-camera health detail
GET  /stores                            store-level totals + congestion-by-camera
GET  /system/performance                per-camera fps/latency/frames/errors + summary
GET  /video_stream                      MJPEG push stream of the de-identified live frame
POST /integrations/pos/transactions     ingest POS transactions (pydantic-validated; batch)
GET  /integrations/pos/transactions     persisted transactions (DB mode)
GET  /integrations/pos/conversion       footfall-width conversion (only when data exists)
POST /integrations/erp/inventory        ingest ERP inventory snapshot (pydantic-validated)
GET  /integrations/erp/inventory        inventory stats (only when data exists)
GET  /integrations/status               integration connection status
GET  /config/zones                      current zone polygons
POST /config/zones                      hot-reload zone polygons (persisted to zones.json)
WS   /ws/live                           live_snapshot broadcasts (1/s) wrapped as {"type":"live_snapshot",...}
```

The web dashboard exposes the same surface under `/api/...` plus
`/api/privacy/status`, `/api/privacy/uploads` (DELETE), and file upload/run
controls.

Example snapshot (per-checkout shape with prediction + measured wait):

```json
{
  "camera_id": "store_01", "ts": 1.7e9, "ts_epoch": 1.7e9,
  "footfall": {"entries": 31, "exits": 4, "occupancy": 27, "unique_shoppers": 18},
  "queues": {
    "prediction_source": "blend",
    "queues": [
      {"queue_id": "checkout_01", "length": 7,
       "wait_minutes": 1.1,
       "measured_wait_minutes": {"avg_wait_minutes": 0.9, "max_wait_minutes": 1.4, "count": 3},
       "status": "WARNING",
       "predictions": {"5min": 6.4, "10min": 8.9,
                       "predicted_queue_length_5min": 6.4, "predicted_queue_length_10min": 8.9,
                       "interval_5min": {"low": 5.2, "high": 8.0}, "interval_10min": {"low": 6.9, "high": 11.2},
                       "confidence": 0.72, "explain_5min": {...}, "explain_10min": {...}},
       "recommendation": "Predicted congestion in ~10 minutes - open an additional counter.",
       "recommendation_detail": {"recommend_action": "open_counter", "text": "..."}}
    ]
  },
  "congestion_status": "HIGH",
  "shelves": [{"shelf_id": "shelf_a", "status": "OUT_OF_STOCK", "item_count": 0,
               "source": "heuristic", "confirmed": true, "stock_out_risk": "HIGH",
               "est_time_to_out_minutes": 0.0, "trend": -1.0}],
  "alerts": {"active": [...], "by_severity": {"INFO": 0, "WARNING": 2, "HIGH": 0},
             "webhook_enabled": false},
  "planogram": {"status": "MODEL_NOT_AVAILABLE", "product_model": false, "results": []}
}
```

## Training + models

```bash
# Queue-length predictor: a SEPARATE model per forecast horizon, each tuned
# between RandomForest vs GradientBoosting via forward-chaining cross-val
# (no shuffling - respects the temporal order). Training also records honest
# baselines (persistence = "queue stays the same", naive mean), a per-horizon
# blend weight + 80% CI, and evaluates on a leak-free holdout.
# Synthetic mode (no data): produce a demo model quickly.
python scripts/train_queue_model.py --samples 2000

# REAL mode: train on a CSV of live features.
#   - collect data first:
#       python main.py --mode live --source 0      # pipeline logs to data/processed/queue_features.csv
#   - or generate a REALISTIC synthetic retail-checkout dataset (proper M/M/c
#     queue dynamics, diurnal + weekday/weekend patterns) that mirrors the
#     on-site schema and is genuinely learnable:
#       python scripts/make_queue_dataset.py --stores 3 --zones 1 --days 21 --out data/processed/queue_sim.csv
#   - then train:
#       python scripts/train_queue_model.py --csv data/processed/queue_sim.csv --horizons 5 10
#   Writes models/prediction/queue_model_{N}min.joblib (one per horizon),
#   plus queue_metrics.json with per-horizon blend weights + CIs.
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
Every prediction carries per-horizon `interval_{N}min` (80% CI), a normed
`confidence` score, verbose `predicted_queue_length_{N}min` keys, and
`explain_{N}min` factor dicts so the dashboard can show *why*. The runtime
predictor builds the same 12-feature vector the models were trained on (see
`QueuePredictor._feature_row`), so deployed predictions stay aligned with
training.

Runtime accuracy is tracked live by `ml/queue/evaluator.py`: each forecast is
resolved against the actual queue once the horizon elapses and MAE/RMSE per
horizon are written to `prediction.eval_path`
(`data/processed/prediction_eval.csv`) and surfaced via
`/api/health -> prediction_monitoring`.

Wait time is measured per anonymous track from queue-zone dwell AND estimated
from the rule `length × average_service_time_seconds / open_counters`; both are
reported so the rule's calibration can be judged against reality.

Shelf snapshots are temporally smoothed: a status change (e.g. to LOW or OUT)
is only *committed* after `confirmation_polls` consecutive consistent polls, and
each shelf carries a depletion `trend`, `est_time_to_out_minutes` and
`stock_out_risk` derived from recent counts.

## Tests

```bash
python -m pytest tests/ -q
```

**88 tests** cover geometry, tracking id stability, entry/exit + cooldown,
occupancy, dwell, heatmap decay + export, queue counter + predictor source
labels + per-horizon models/CI/blend weight + training baselines/feature parity,
wait measuring, historical buckets, the unified alert store, camera/performance
monitoring, the planogram gate, POS/ERP adapters, retention/privacy, the pydantic
API boundary (422 validation + honest `persisted`), the new persisted entities
(stores/cameras/queue+shelf events/POS transactions), the YOLO ONNX/fallback
backends, the background DB writer, CameraSource (reconnect/loop/fps-cap), the
realistic dataset generator and the dashboard (index, demo run, live analytics,
privacy endpoints). API/dashboard tests need `httpx2` / `python-multipart`
(skip cleanly otherwise).

## What needs real-world data / calibration

- **Zones & entrance line** - re-draw `config/zones.json` to the exact camera
  view. Zones carry `zone_type` (`shopping_zone`, `queue_zone`, `shelf_zone`,
  `entrance`, `exit`); verify the entry-line orientation (`line_counter.entry_direction`).
- **Queue service rate** - `config/settings.yaml → queue.average_service_time_seconds`
  drives the rule-based wait estimate; measure real checkout throughput on site
  (the measured wait makes the mismatch visible).
- **Shelf item thresholds** - `shelf.low_stock_threshold` / `out_of_stock_threshold`
  vs `expected_item_count` per shelf; or train the CNN with `train_shelf_model.py`.
- **Product detector** - real planogram compliance needs a product/object model
  (until then the API reports `MODEL_NOT_AVAILABLE`).
- **Prediction model** - once `--csv` logs accumulate, re-train on site logs and
  validate MAE in `models/prediction/queue_metrics.json`.
- **POS/ERP credentials/payloads** - point the ingest endpoints at your real
  vendor payloads; nothing connects to external services until you configure it
  (`alerts.webhook_url`, the ingest endpoints).

## Edge deployment path

Laptop/colab experiments → export ONNX (FP16/INT8) → TensorRT/OpenVINO per
device → `scripts/benchmark.py` for FPS, latency percentiles, model size and
RAM/CPU/GPU utilisation. See `deployment/edge/README.md` and
`deployment/docker/` for containers.

### Running the ML model on a website / web service

The queue forecaster is a tiny (≈500 KB) scikit-learn GradientBoosting model —
it runs on any CPU and needs no GPU or torch. The REST+WebSocket API already
serves it; to embed it behind your own web service:

1. Ship `models/prediction/queue_model_{5,10}min.joblib` +
   `queue_metrics.json` to the server.
2. Reuse `ml/queue/predictor.py` (pure Python + numpy + sklearn + joblib — no
   FastAPI/camera dependency) to produce per-horizon forecasts.
3. Reproduce `_feature_row()` (12 features, fixed order) exactly — it interprets
   `(ts, queue_len)` samples your upstream supplies.
4. Start the API (default `:8000`) and point your frontend at `/api/*` + the
   `/ws/live` WebSocket; the containerised build is in `deployment/docker/`.

For the *vision* side (person/queue counting) on a server you'll need the YOLO
checkpoint + `ultralytics` and, for real-time frame processing, an appropriate
deployment target (see `deployment/edge/`).

## Integration checklist

1. Set `demo.enabled: false` and a real `cameras.yaml` source (webcam or RTSP).
2. Place a person-detection checkpoint at `models/yolo/yolov8n.pt` and install
   `ultralytics` (optional but recommended; set `tracking.backend: bytetrack`).
3. Re-publish `zones.json` (with `zone_type`) for each camera view.
4. Point the software team's dashboard at `/ws/live` + REST endpoints.
5. Log ~1 week of queue history, re-train the queue predictor, validate MAE.
6. Wire POS/ERP ingest to your vendor payloads; validate conversion + inventory.
7. Configure `alerts.webhook_url` to fan alerts out to your ops channel.
8. Export + quantize the detector for the chosen edge device; benchmark.