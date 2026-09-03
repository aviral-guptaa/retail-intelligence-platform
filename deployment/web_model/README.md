# Web deployment of the queue-forecast ML model

Standalone REST API that serves **only the trained queue model** (5/10-minute
queue-length forecast + confidence + explainable counter recommendation) with
no camera/vision/DB dependencies. Ideal for putting the ML on a website or
dashboard.

## It needs (already shipped in this zip)
- `models/prediction/queue_model_{5,10}min.joblib`
- `models/prediction/queue_metrics.json`
- `ml/queue/predictor.py` (pure Python + numpy + sklearn + joblib)

## Run locally
```bash
python -m venv .venv-web && source .venv-web/bin/activate
pip install -r deployment/web_model/requirements.txt
uvicorn deployment.web_model.app:app --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET /health` → `{status, source, horizons}`
- `POST /predict` — body:
  ```json
  {
    "history": [{"ts": 1699999999.0, "queue_len": 9}, ...],   // >=1, ideally >=30 chronological samples
    "footfall": 12,
    "open_counters": 3
  }
  ```
  Response includes `predicted_queue_length_5min/10min`, `confidence`,
  `intervals.{5,10}` (80% CI), `explain.{5,10}` and a `recommendation` block
  (`text`, `recommend_action`, `reason`, `factors`).

## Docker
```bash
docker build -f deployment/web_model/Dockerfile -t queue-forecaster .
docker run -p 8000:8000 queue-forecaster
```

## Test the live API
```bash
curl localhost:8000/health
curl -s localhost:8000/predict -H 'content-type: application/json' -d \
  '{"history":[{"ts":1,"queue_len":5},{"ts":11,"queue_len":7},{"ts":21,"queue_len":8},{"ts":31,"queue_len":9},{"ts":41,"queue_len":10},{"ts":51,"queue_len":11}],"footfall":10,"open_counters":3}'
```

Where your frontend gets queue samples: feed it from anyone (the full retail
pipeline's line counter, a manual feed, or a CSV). The server does the ML.
