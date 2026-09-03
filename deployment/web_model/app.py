"""Standalone web API for the trained queue-forecast model.

Self-contained: serves ONLY the ML model (per-horizon queue-length forecast +
recommendation) with no camera / vision / database dependencies, so it can be
deployed behind a website or dashboard by itself.

Expects the trained artifacts next to this file (or under the repo paths):
  models/prediction/queue_model_{5,10}min.joblib
  models/prediction/queue_metrics.json

Run:
  venv:  pip install fastapi uvicorn scikit-learn joblib numpy
         uvicorn app:app --host 0.0.0.0 --port 8000
  docker: see Dockerfile in this directory.
"""
from __future__ import annotations

import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.queue.predictor import QueuePredictor

app = FastAPI(
    title="Retail Queue Forecaster",
    version="1.0.0",
    description="Per-horizon queue-length forecast (5/10 min) with confidence "
                "intervals and an explainable counter-open recommendation.",
)

# Bootstrap the predictor from the packaged per-horizon models. Leave the
# model_path unset so QueuePredictor auto-discovers queue_model_{5,10}min.joblib
# + queue_metrics.json under models/prediction/ (resolve-able from repo root).
_predictor: QueuePredictor = QueuePredictor(
    {"horizon_minutes": [5, 10]},
    {"congestion_warning_queue": 4, "congestion_high_queue": 8},
)
_OVERALL_SOURCE = getattr(_predictor, "overall_source", "fallback")


class Sample(BaseModel):
    ts: float = Field(..., description="unix epoch seconds")
    queue_len: int = Field(..., ge=0, description="current queue length")


class PredictRequest(BaseModel):
    history: List[Sample] = Field(
        ...,
        min_length=1,
        description="chronological (ts, queue_len) samples, ideally >= 30",
    )
    footfall: int = Field(0, ge=0, description="current people on the floor")
    open_counters: int = Field(1, ge=1, description="open checkout counters")


class HealthResponse(BaseModel):
    status: str
    source: str
    horizons: List[int]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        source=_OVERALL_SOURCE,
        horizons=_predictor.horizons,
    )


@app.post("/predict")
def predict(req: PredictRequest):
    """Forecast queue length at each configured horizon + confidence + explanation."""
    history = [(s.ts, s.queue_len) for s in req.history]
    pred = _predictor.predict(
        history, footfall=req.footfall, open_counters=req.open_counters,
    )
    rec = _predictor.explain_recommendation(pred, history[-1][1])
    return {
        "source": pred.get("source"),
        "predicted_queue_length_5min": pred.get("predicted_queue_length_5min"),
        "predicted_queue_length_10min": pred.get("predicted_queue_length_10min"),
        "confidence": pred.get("confidence"),
        "intervals": {h: pred.get(f"interval_{h}min") for h in _predictor.horizons},
        "explain": {h: pred.get(f"explain_{h}min") for h in _predictor.horizons},
        "recommendation": rec,
    }
