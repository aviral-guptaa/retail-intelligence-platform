# Web dashboard (prototype)

A functional web dashboard where you can **upload a video** (or launch the demo
simulator) and the full SIH analytics pipeline runs **server-side**, streaming
live metrics to the page — occupancy, queue length, 5/10-minute forecasts from
the trained model, confidence, counter-open recommendations, shelves, and a
movement heatmap.

This is the prototype front-end; the same analytics/endpoints are what the
hardware integration will feed into later.

## Run
```bash
source .venv/bin/activate
python run_web.py                 # -> http://127.0.0.1:8000/
```
(installs fine with core deps; `python-multipart` is needed for video upload,
it's in `requirements.txt`.)

Then in the browser:
- **▶ Run demo** — instantly starts a synthetic store (no camera/GPU needed).
- **Upload a video** — runs the real vision pipeline (`YoloDetector` fallback
  to motion if `ultralytics` isn't installed) on your file. mp4/mov/avi/mkv.

## How it works
- `POST /api/run/demo` / `POST /api/run/video` — start a run (uploads saved under
  `data/uploads/`).
- `GET /api/analytics/current` — REST snapshot (footfall, queues + predictions,
  shelves, congestion).
- `GET /api/heatmap.png` — current movement heatmap frame.
- `/ws/live` — WebSocket pushing a live snapshot every ~1s that the dashboard
  renders live. Also `GET /api/run/status` and `POST /api/run/stop`.

## Layout
- `webserver/app.py` — FastAPI app + `RunManager` (owns the running pipeline).
- `webserver/static/index.html` — the single-page dashboard (Chart.js via CDN).
- `run_web.py` — entry point.
- `tests/test_webserver.py` — in-process pipeline + REST checks.

## Integration note
Everything the dashboard displays comes from the standard
`AnalyticsService.current()` snapshot and `/ws/live` — the same contract the
CLI app and any future hardware firmware would emit, so swapping the prototype
backend for the on-site deployment is a drop-in.
