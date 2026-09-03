"""Queue-length forecasting (the primary predictive ML component).

A **per-horizon** set of trained regressors is used when available (see
``scripts/train_queue_model.py``, which writes one ``queue_model_{N}min.joblib``
per horizon plus ``queue_metrics.json``). Before enough real time-series data
exists, or if no model is fitted, an online bounded-linear fallback extends the
recent trend - so the pipeline is demoable from the first frame.

Transparency: every prediction dict carries a ``source`` key:
  - "model"     -> the trained regressors alone;
  - "blend"     -> trained regressors blended with the online linear trend;
  - "fallback"  -> online trend only (no model / not enough history).

Design (SIH "defensible ML"):

 * **Separate models per horizon.** 5-minute and 10-minute forecasts are
   different regression problems and are trained/scored independently.

 * **Validation-selected blend weight.** Rather than a hardcoded blend constant,
   the weight given to the ML prediction is stored per horizon by the training
   script (chosen to minimise holdout MAE). At runtime we reuse that stored
   weight, or fall back to 0.55 (a neutral prior) if the metrics file is
   missing/unparsable.

 * **Confidence / uncertainty.** Each horizon carries an 80% prediction interval
   (``low/high``) derived from the model's held-out residual quantiles, plus a
   normed confidence score. If the interval is unavailable the confidence falls
   back to a heuristic driven by history length.

 * **Explainable recommendations.** ``recommendation()`` returns a structured
   set of the real factors that drove the decision (current queue, growth rate,
   footfall, open counters, predicted queue) rather than a bare string, so the
   dashboard can show *why* another counter is suggested.

Horizon values are exposed both as ``"{n}min"`` (stable keys) and the verbose
spec-conformant ``predicted_queue_length_{n}min``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.loader import resolve

logger = logging.getLogger(__name__)

FALLBACK_BLEND = 0.55  # neutral prior used only if the metrics file is missing
CONF_P80 = 0.80


class QueuePredictor:
    """Predicts future queue lengths N minutes ahead plus an actionable message."""

    def __init__(self, prediction_settings: Dict, queue_settings: Dict):
        self.settings = prediction_settings
        self.horizons = sorted(int(h) for h in prediction_settings.get("horizon_minutes", [5, 10]))
        self.queue_settings = queue_settings
        self._models: Dict[int, Any] = {}          # horizon -> fitted regressor (or None)
        self.metrics: Dict[str, Any] = {}          # per-horizon training metrics
        self._horizon_meta: Dict[int, Dict[str, Any]] = {}
        self.overall_source = "fallback"
        self.warning_queue = float(queue_settings.get("congestion_warning_queue", 4))
        self.high_queue = float(queue_settings.get("congestion_high_queue", 8))
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        self._load_metrics_meta()
        self._load_models()

    def _metric_dir(self) -> Path:
        # Prefer an explicit models dir; otherwise derive from model_path's parent.
        models_dir = self.settings.get("models_dir")
        if models_dir:
            return resolve(str(models_dir))
        model_path = self.settings.get("model_path", "models/prediction/queue_model.joblib")
        return resolve(model_path).parent

    def _load_metrics_meta(self) -> None:
        path = resolve(self.settings.get("metrics_path",
                                         str(self._metric_dir() / "queue_metrics.json")))
        if not path.exists():
            return
        try:
            self.metrics = json.loads(path.read_text())
            for m in self.metrics.get("models", []):
                h = int(m.get("horizon_minutes", -1))
                if h in self.horizons:
                    self._horizon_meta[h] = m
        except Exception as exc:  # pragma: no cover
            logger.warning("could not parse queue metrics %s (%s); using defaults", path, exc)
            self.metrics = {}

    def _blend_weight_for(self, horizon: int) -> float:
        """Validation-selected blend weight, or the neutral prior when unknown."""
        meta = self._horizon_meta.get(horizon, {})
        w = meta.get("blend_weight")
        try:
            w = float(w)
            if 0.0 <= w <= 1.0:
                return w
        except (TypeError, ValueError):
            pass
        return FALLBACK_BLEND

    def _confidence_interval_for(self, horizon: int) -> Tuple[float, float]:
        meta = self._horizon_meta.get(horizon, {})
        ci = meta.get("confidence_interval_80")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                return float(ci[0]), float(ci[1])
            except (TypeError, ValueError):
                pass
        return -2.0, 2.0   # generous neutral band when unknown

    def _load_models(self) -> None:
        explicit = self.settings.get("model_path")
        if explicit:
            # An explicit model_path is authoritative: load exactly that file
            # (per-horizon auto-discovery and the legacy fallback are skipped).
            # If it's missing we deliberately report "fallback" rather than
            # silently scanning the directory for unrelated models.
            path = resolve(str(explicit))
            if path.exists():
                self._load_model_file(path, self.horizons[0])
                for h in self.horizons[1:]:
                    self._models.setdefault(h, self._models.get(self.horizons[0]))
            self.overall_source = "model" if any(m is not None for m in self._models.values()) else "fallback"
            return

        d = self._metric_dir()
        any_loaded = False
        # Auto-discovery: a per-horizon model per configured horizon, else the
        # legacy single-file queue_model.joblib applied to every horizon.
        for h in self.horizons:
            per_h = d / f"queue_model_{h}min.joblib"
            if per_h.exists():
                self._load_model_file(per_h, h)
                any_loaded = any_loaded or self._models.get(h) is not None
                continue
            legacy = d / "queue_model.joblib"
            if legacy.exists():
                self._load_model_file(legacy, h)
                any_loaded = any_loaded or self._models.get(h) is not None
        self.overall_source = "model" if any_loaded else "fallback"

    def _load_model_file(self, path: Path, horizon: int) -> None:
        try:
            import joblib
            self._models[horizon] = joblib.load(path)
            logger.info("Loaded queue model for %dmin from %s", horizon, path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load %dmin queue model (%s); using linear fallback", horizon, exc)
            self._models[horizon] = None

    # ------------------------------------------------------------------ data
    @staticmethod
    def _feature_row(queue_history: List[Tuple[float, int]], footfall: int,
                     open_counters: int, now: float) -> np.ndarray:
        """Build the model feature vector, matching scripts/train_queue_model.py.

        Order: [queue_now, mean5, mean10, mean30, std10, queue_delta,
                growth_rate, footfall, footfall_rate, hour, dow, open_counters].
        Windows are over the trailing history *samples*, mirroring the rolling
        windows in prepare_frame(). Means default to the current queue when too
        few samples exist; std/delta default to 0.
        """
        vals = np.array([v for _, v in queue_history], dtype=float) if queue_history else np.zeros(1)
        current = float(vals[-1] if len(vals) else 0.0)
        growth = 0.0
        if len(vals) >= 2:
            growth = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])  # per-sample trend
        mean5 = float(vals[-5:].mean()) if len(vals) else current
        mean10 = float(vals[-10:].mean()) if len(vals) else current
        mean30 = float(vals[-30:].mean()) if len(vals) else current
        std10 = float(vals[-10:].std(ddof=0)) if len(vals) >= 2 else 0.0
        delta = float(current - vals[-10]) if len(vals) > 10 else 0.0
        ltm = time.localtime(now)
        return np.array([current, mean5, mean10, mean30, std10, delta,
                         growth, float(footfall), float(footfall) / 60.0,
                         float(ltm.tm_hour), float(ltm.tm_wday), float(open_counters)])

    # ------------------------------------------------------------- inference
    def predict(self, queue_history: List[Tuple[float, int]], footfall: int,
                open_counters: int) -> Dict[str, Any]:
        """Return predicted queue length per configured horizon + metadata.

        Keys: ``"{N}min"``, ``predicted_queue_length_{N}min``, ``source``,
        ``confidence``, ``interval_{N}min`` (dict of low/high), and
        ``explain_{N}min`` with the input factors.
        """
        linear = self._linear_fallback(queue_history, footfall)
        has_model = any(m is not None for m in self._models.values())

        t_now = queue_history[-1][0] if queue_history else time.time()
        out: Dict[str, Any] = {}

        for h in self.horizons:
            model_pred = None
            model = self._models.get(h)
            if model is not None:
                row = self._feature_row(queue_history, footfall, open_counters, t_now)
                try:
                    model_pred = max(0.0, float(model.predict([row])[0]))
                except Exception:  # feature mismatch with a freshly trained model
                    model_pred = None

            if model_pred is None:
                value = float(linear[f"{h}min"])
                blend_w = 0.0
            else:
                blend_w = self._blend_weight_for(h)
                value = blend_w * model_pred + (1 - blend_w) * linear[f"{h}min"]

            value = round(max(0.0, value), 1)
            out[f"{h}min"] = value
            out[f"predicted_queue_length_{h}min"] = value
            out[f"explain_{h}min"] = {
                "current_queue": int(queue_history[-1][1]) if queue_history else 0,
                "history_points": len(queue_history),
                "growth_rate": float(self._growth_per_min(queue_history)),
                "footfall": int(footfall),
                "open_counters": int(open_counters),
                "model_used": model_pred is not None,
                "blend_weight": round(blend_w, 3),
                "linear_trend_value": round(float(linear[f"{h}min"]), 2),
                "model_value": round(float(model_pred), 2) if model_pred is not None else None,
            }
            lo, hi = self._confidence_interval_for(h)
            pred_lo = max(0.0, round(value + lo * blend_w, 1))
            pred_hi = max(0.0, round(value + hi * blend_w, 1))
            out[f"interval_{h}min"] = {"low": pred_lo, "high": pred_hi}

        if has_model:
            out["source"] = "model" if all(self._models.get(h) is not None for h in self.horizons) \
                else "blend"
            if out["source"] == "model":
                out["source"] = "blend" if self._any_blended(out) else "model"
        else:
            out["source"] = "fallback"

        out["confidence"] = self._confidence(out, queue_history)
        return self._decorate(out)

    @staticmethod
    def _any_blended(preds: Dict[str, Any]) -> bool:
        for hk in list(preds):
            exp = preds.get(f"explain_{hk}")
            if isinstance(exp, dict) and exp.get("blend_weight", 1.0) < 1.0:
                return True
        return False

    def _confidence(self, preds: Dict[str, Any], queue_history) -> float:
        """A normed 0..1 confidence combining interval tightness + data sufficiency."""
        widths = []
        present = False
        for h in self.horizons:
            iv = preds.get(f"interval_{h}min")
            if isinstance(iv, dict):
                present = True
                widths.append(max(0.0, float(iv.get("high", 0.0)) - float(iv.get("low", 0.0))))
        width = float(np.mean(widths)) if widths else 0.0
        n_hist = len(queue_history)
        # More history / tighter interval -> higher confidence.
        hist_score = min(1.0, n_hist / 30.0)
        width_score = 1.0 - min(1.0, width / 12.0)
        if not present:
            return round(hist_score * 0.5, 3)
        return round(0.6 * width_score + 0.4 * hist_score, 3)

    def _growth_per_min(self, queue_history: List[Tuple[float, int]]) -> float:
        """Linear slope of recent queue length expressed in shoppers/minute."""
        if not queue_history or len(queue_history) < 2:
            return 0.0
        ts = np.array([s for s, _ in queue_history], dtype=float)
        vals = np.array([v for _, v in queue_history], dtype=float)
        if ts.max() == ts.min():
            return 0.0
        slope, _ = np.polyfit(ts - ts[0], vals, 1)
        return float(slope) * 60.0

    @staticmethod
    def _decorate(preds: Dict[str, float]) -> Dict[str, float]:
        """Add the verbose predicted_queue_length_{n}min keys in place (no-op now)."""
        decorated = dict(preds)
        for key, value in list(preds.items()):
            if key.endswith("min") and key[:-3].isdigit():
                verbose = f"predicted_queue_length_{key}"
                if verbose in decorated and isinstance(value, (int, float)):
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
        # bound the forecast so a momentary burst cannot extrapolate to an
        # absurd value, but let the ceiling grow with the horizon so distinct
        # horizons are not flattened into a single number.
        out: Dict[str, float] = {}
        for h in self.horizons:
            horizon_seconds = h * 60
            pred = slope * (horizon_seconds + ts[-1] - ts[0]) + intercept
            ceiling = current + max(8.0, slope * horizon_seconds)
            pred = float(np.clip(pred, 0.0, max(ceiling, current * 1.5)))
            if out:  # keep forecasts monotonically non-decreasing in horizon
                prev = self.horizons[self.horizons.index(h) - 1]
                pred = max(pred, out[f"{prev}min"])
            out[f"{h}min"] = round(pred, 1)
        return out

    # -------------------------------------------------------- recommendation
    def recommendation(self, predictions: Dict[str, float],
                       current_queue: int) -> str:
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

    def explain_recommendation(self, predictions: Dict[str, float],
                               current_queue: int) -> Dict[str, Any]:
        """Explain the counter decision from the real driving factors.

        Returns a structured dict the dashboard can render as an explainable
        card: the action, the reason code, and the actual factors (current
        queue, forecast, growth rate, footfall, open counters) behind it.
        """
        horizons = {k: v for k, v in predictions.items() if k.endswith("min")
                    and k[:-3].isdigit() and isinstance(v, (int, float))}
        max_pred = max(horizons.values()) if horizons else 0.0
        horizon_of_max = min((int(k.replace("min", "")) for k, v in horizons.items()
                              if v == max_pred), default=10)
        growth = None
        footfall = None
        open_counters = None
        for h in self.horizons:
            exp = predictions.get(f"explain_{h}min")
            if isinstance(exp, dict):
                growth = exp.get("growth_rate")
                footfall = exp.get("footfall")
                open_counters = exp.get("open_counters")
                break

        high = self.high_queue
        warning = self.warning_queue
        aggressive = (max_pred >= high or
                      (max_pred >= warning and current_queue >= warning))

        factors = {
            "current_queue": int(current_queue),
            "predicted_queue": round(max_pred, 1),
            "predicted_horizon_minutes": int(horizon_of_max),
            "queue_growth_rate_per_min": growth,
            "footfall_current": footfall,
            "open_counters": open_counters,
            "high_threshold": high,
            "warning_threshold": warning,
        }

        if aggressive:
            reason = ("high_queue_forecast" if max_pred >= high
                      else "warning_forecast_with_live_queue")
            text = (f"Predicted congestion in ~{horizon_of_max} minutes "
                    f"(queue predicted at {max_pred:.0f}) - open an additional counter.")
        elif max_pred >= warning:
            reason = "warning_forecast"
            text = f"Watch queue: forecast reaches ~{max_pred:.0f} in ~{horizon_of_max} min."
        elif current_queue >= warning:
            reason = "live_queue_warning"
            text = f"Queue at {current_queue:.0f} is forming; monitor dwell/heat."
        else:
            reason = "no_congestion"
            text = "No congestion expected in the near future."

        return {
            "text": text,
            "recommend_action": "open_counter" if aggressive else ("monitor" if max_pred >= warning else "none"),
            "reason": reason,
            "factors": factors,
        }

    def explain_features(self) -> Dict[str, str]:
        return {
            "current_queue": "live queue length",
            "queue_history": "recent queue samples used for trend features",
            "growth_rate": "linear slope of recent queue lengths",
            "footfall": "current people on the floor",
            "time_features": "time of day / day of week derived from timestamps",
            "open_counters": "number of open checkout counters",
            "blend_weight": "validation-selected weight between ML prediction and linear trend",
            "interval_80": "80% prediction interval derived from held-out residuals",
        }
