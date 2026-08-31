"""Queue-length forecasting (the primary predictive ML component).

A trained RandomForest / GradientBoosting regressor is used when available
(see scripts/train_queue_model.py, which writes ``queue_metrics.json`` too).
Before enough real time-series data exists, or if no model is fitted, an online
bounded-linear fallback extrapolates the recent trend - so the whole pipeline
is demoable from the first frame.

Transparency: every prediction dict carries a ``source`` key:
  - "model"     -> the trained regressor alone;
  - "blend"     -> trained regressor blended with the online linear trend;
  - "fallback"  -> online trend only (no model / not enough history).
Horizon values are exposed both as ``"{n}min"`` (stable keys) and the verbose
spec-conformant ``predicted_queue_length_{n}min``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.loader import resolve

logger = logging.getLogger(__name__)

MODEL_BLEND = 0.55       # weight given to the ML prediction when blending


class QueuePredictor:
    """Predicts future queue lengths 5-10 minutes ahead plus an actionable message."""

    def __init__(self, prediction_settings: Dict, queue_settings: Dict):
        self.settings = prediction_settings
        self.horizons = sorted(int(h) for h in prediction_settings.get("horizon_minutes", [5, 10]))
        self.queue_settings = queue_settings
        self.model = None
        self.metrics: Dict = {}
        self.warning_queue = float(queue_settings.get("congestion_warning_queue", 4))
        self.high_queue = float(queue_settings.get("congestion_high_queue", 8))
        self._load_model()
        self._load_metrics()

    def _load_model(self) -> None:
        path = resolve(self.settings.get("model_path", "models/prediction/queue_model.joblib"))
        if Path(path).exists():
            try:
                import joblib

                self.model = joblib.load(path)
                logger.info("Loaded queue prediction model from %s", path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load queue model (%s); using linear fallback", exc)

    def _load_metrics(self) -> None:
        path = resolve(self.settings.get("metrics_path", "models/prediction/queue_metrics.json"))
        if Path(path).exists():
            try:
                self.metrics = json.loads(Path(path).read_text())
            except Exception:
                self.metrics = {}

    # ------------------------------------------------------------------ data
    @staticmethod
    def _feature_row(queue_history: List[Tuple[float, int]], footfall: int,
                     open_counters: int, now: float) -> np.ndarray:
        """Build the model feature vector per the SIH spec.

        [current_queue, mean_prev_5, mean_prev_10, growth_rate, footfall,
         hour_of_day, day_of_week, open_counters]
        """
        vals = np.array([v for _, v in queue_history], dtype=float) if queue_history else np.zeros(1)
        current = float(vals[-1] if len(vals) else 0.0)
        growth = 0.0
        if len(vals) >= 2:
            growth = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])  # per-sample trend
        mean5 = float(vals[-5:].mean()) if len(vals) else current
        mean10 = float(vals[-10:].mean()) if len(vals) else current
        ltm = time.localtime(now)
        return np.array([current, mean5, mean10, growth, float(footfall),
                         float(ltm.tm_hour), float(ltm.tm_wday), float(open_counters)])

    # ------------------------------------------------------------- inference
    def predict(self, queue_history: List[Tuple[float, int]], footfall: int,
                open_counters: int) -> Dict[str, float]:
        """Return predicted queue length per configured horizon + source label."""
        linear = self._linear_fallback(queue_history, footfall)
        if self.model is None:
            out = dict(linear)
            out["source"] = "fallback"
            return self._decorate(out)
        t_now = queue_history[-1][0] if queue_history else time.time()
        row = self._feature_row(queue_history, footfall, open_counters, t_now)
        try:
            base = max(0.0, float(self.model.predict([row])[0]))
        except Exception:  # feature mismatch with a freshly trained model
            base = 0.0
        model_pred = {f"{h}min": base * (h / 10.0) for h in self.horizons}
        out = {k: round(max(0.0, MODEL_BLEND * model_pred[k] + (1 - MODEL_BLEND) * linear[k]), 1)
               for k in linear}
        out["source"] = "blend"
        return self._decorate(out)

    @staticmethod
    def _decorate(preds: Dict[str, float]) -> Dict[str, float]:
        """Add the verbose predicted_queue_length_{n}min keys in place."""
        decorated = dict(preds)
        for key, value in list(preds.items()):
            if key.endswith("min") and key[:-3].isdigit():
                verbose = f"predicted_queue_length_{key}"
                if verbose not in decorated and isinstance(value, (int, float)):
                    decorated[verbose] = value
        return decorated

    def _linear_fallback(self, queue_history: List[Tuple[float, int]], footfall: int) -> Dict[str, float]:
        """Bounded extrapolation of the recent queue trend.

        Requires enough samples before trusting the slope, and caps the forecast
        so a momentary burst (or the ramp-up phase) cannot produce absurd
        predictions - the demo's bursty arrivals demand this.
        """
        if len(queue_history) < 6:
            val = queue_history[-1][1] if queue_history else 0.0
            return {f"{h}min": float(val) for h in self.horizons}

        ts = np.array([s for s, _ in queue_history], dtype=float)
        vals = np.array([v for _, v in queue_history], dtype=float)
        slope, intercept = np.polyfit(ts - ts[0], vals, 1)
        current = float(vals[-1])
        cap = current + 8.0     # no forecast more than ~8 above the live queue
        out: Dict[str, float] = {}
        for h in self.horizons:
            horizon_seconds = h * 60
            pred = slope * (horizon_seconds + ts[-1] - ts[0]) + intercept
            pred = float(np.clip(pred, 0.0, max(cap, current * 1.5)))
            out[f"{h}min"] = round(pred, 1)
        return out

    # -------------------------------------------------------- recommendation
    def recommendation(self, predictions: Dict[str, float], current_queue: int) -> str:
        """Actionable suggestion derived from the predicted queue lengths."""
        horizons = {k: v for k, v in predictions.items() if k.endswith("min")
                    and k[:-3].isdigit() and isinstance(v, (int, float))}
        if not horizons:
            return "No congestion expected in the near future."
        max_pred = max(horizons.values())
        aggressive = max_pred >= self.high_queue or \
            (max_pred >= self.warning_queue and current_queue >= self.warning_queue)
        if aggressive:
            horizon = min((int(k.replace("min", "")) for k, v in horizons.items()
                           if v == max_pred), default=5)
            return (f"Predicted congestion in ~{horizon} minutes "
                    f"(queue predicted at {max_pred:.0f}) - open an additional counter.")
        if current_queue >= self.warning_queue:
            return f"Queue at {current_queue:.0f} is forming; monitor dwell/heat."
        return "No congestion expected in the near future."

    def explain_features(self) -> Dict[str, str]:
        return {
            "current_queue": "live queue length",
            "queue_history": "recent queue samples used for trend features",
            "growth_rate": "linear slope of recent queue lengths",
            "footfall": "current people on the floor",
            "time_features": "time of day / day of week derived from timestamps",
            "open_counters": "number of open checkout counters",
        }