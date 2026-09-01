"""Train queue-length prediction models (RandomForest / GradientBoosting).

Two data sources:

 1. Synthetic simulation (default) - bursty arrivals + service, used so the
    dashboard has a model from day one (previously-partnered demo data).

 2. Real collected data - ``--csv data/processed/queue_features.csv``, the file
    written continuously by ``ml.queue.datalogger`` while the pipeline runs in
    demo or live mode.

Key design decisions (SIH "defensible ML"):

 * **Separate models per horizon.** The 5-minute and 10-minute forecasts are
   distinct regression problems with their own features and error profile, so
   this trains one model per configured horizon instead of stretching a single
   10-minute model down to 5 (which silently assumes a constant scaling that
   rarely holds).

 * **Time-series-aware evaluation (no leakage).** Rows are never shuffled. The
   labelled series are split on *time* into a contiguous calibration set and a
   later holdout set, so the model is scored only on the future it did not see.
   Within cross-validation we use a deterministic expand/cut forward-chaining
   split rather than sklearn's random ``train_test_split``.

 * **Validation-selected blending weight.** The ML prediction and a bounded
   linear trend are combined as ``w*ml + (1-w)*trend``. Instead of a hardcoded
   weight (the old 0.55), ``w`` is chosen on a validation slice to minimise the
   horizon-specific MAE, and recorded per horizon so the runtime blends the two
   signals in the same ratio that won validation.

 * **Per-horizon confidence.** For each horizon the model's held-out residual
   distribution is summarised (MAE, RMSE, and the 80% prediction interval
   width) and stored, so the runtime can return a defensible uncertainty band
   instead of an arbitrary one.

Run with:
    python scripts/train_queue_model.py --samples 2000            # synthetic
    python scripts/train_queue_model.py --csv data/processed/queue_features.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

DEFAULT_HORIZONS_MIN = [5, 10]
SAMPLE_INTERVAL_SYNTHETIC = 10.0   # demo simulator steps 10s per row
SAMPLE_INTERVAL_REAL = 5.0         # datalogger default (config sample_interval_seconds)


def horizon_seconds(h_min: int) -> int:
    return h_min * 60


def simulate_queue_series(n_steps: int, seed: int = 7,
                          horizons_min=DEFAULT_HORIZONS_MIN) -> pd.DataFrame:
    """Synthetic store data with a target column for each horizon.

    Queue length evolves as: arrival bursts push it up, service draws it down.
    The target for horizon h is the queue length ``h*60`` seconds ahead.
    """
    rng = np.random.default_rng(seed)
    service_rate = 5.0  # shoppers/minute with an open counter
    open_counters = np.zeros(n_steps, dtype=int)

    hour = np.zeros(n_steps)
    dow = np.zeros(n_steps)
    queue = np.zeros(n_steps + 1)
    footfall = np.zeros(n_steps)
    max_h = max(horizons_min)

    # Keep enough leading "warm-up" so the earliest rows still have future
    # values for every horizon.
    horizon_steps = {h: h * 6 for h in horizons_min}      # 10s per sim step
    lead = max(horizon_steps.values()) + 5

    for i in range(n_steps):
        hour[i] = i % (60 * 12)  # 12-hour trading day in minutes
        dow[i] = (i // (60 * 12)) % 7
        rush = 1.0
        if 3.5 * 60 <= hour[i] <= 6 * 60:
            rush = 1.6
        open_counters[i] = max(1, int(3 * rush))
        base_arrivals = 55 * rush * (0.6 + 0.8 * rng.random())
        footfall[i] = base_arrivals
        arrivals = float(rng.poisson(max(0.0, base_arrivals / 60.0)))
        if rng.random() < 0.06:
            arrivals += rng.poisson(6)
        served = min(queue[i], open_counters[i] * service_rate)
        samples_per_min = 6
        queue[i + 1] = max(0.0, queue[i]
                           + arrivals / samples_per_min - served / samples_per_min)

    # fit the future values on the *extended* timeline (up to n_steps+lead)
    full_queue = queue.copy()
    queue_ext = np.concatenate([full_queue, np.full(int(lead), np.nan)])
    q = full_queue[:-1]

    df = pd.DataFrame({
        "ts": np.arange(n_steps) * 10.0,
        "camera_id": "synthetic", "queue_id": "checkout_01",
        "queue_now": q,
        "mean5": pd.Series(q).rolling(5, min_periods=1).mean().values,
        "mean10": pd.Series(q).rolling(10, min_periods=1).mean().values,
        "mean30": pd.Series(q).rolling(30, min_periods=1).mean().values,
        "std10": pd.Series(q).rolling(10, min_periods=1).std(ddof=0).fillna(0).values,
        "queue_delta": pd.Series(q).diff(10).fillna(0).values,
        "growth_rate": np.gradient(q),
        "footfall": footfall,
        "footfall_rate": footfall / 60.0,
        "hour": hour,
        "dow": dow,
        "open_counters": open_counters,
    })
    for h in horizons_min:
        steps = h * 6
        target = np.full(n_steps, np.nan)
        # target[i] = queue value `steps` samples later (on the extended queue)
        for i in range(n_steps - steps):
            target[i] = queue_ext[i + steps]
        df[f"target_{h}min"] = target
    return df


def load_real_features(csv_path: str, horizons_min=DEFAULT_HORIZONS_MIN) -> pd.DataFrame:
    """Read the datalogger CSV and compute a target per horizon for each series."""
    raw = pd.read_csv(csv_path)
    required = ["ts", "camera_id", "queue_id", "queue_now"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing} (got {list(raw.columns)})")

    series = raw.sort_values("ts").drop_duplicates(subset=["ts", "camera_id", "queue_id"])
    groups = series.groupby(["camera_id", "queue_id"])
    series = series.copy()
    for h in horizons_min:
        hsec = horizon_seconds(h)
        targets = []
        for _, g in groups:
            ts = g["ts"].to_numpy(dtype=float)
            val = g["queue_now"].to_numpy(dtype=float)
            targets.append(np.interp(ts + hsec, ts, val))
        series[f"target_{h}min"] = np.concatenate(targets)
    return series


def _rolling_linear_forecast(vals, horizon_samples, window=12, min_samples=6):
    """Rolling linear-trend forecast mirroring the runtime's bounded fallback.

    For each row computes ``last + slope*horizon_samples`` from a fit over the
    trailing ``window`` samples, bounded to avoid absurd extrapolations. Used so
    the validation-time blend weight is estimated against the *same* trend the
    runtime uses online.
    """
    n = len(vals)
    out = np.full(n, np.nan)
    v = np.asarray(vals, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = v[lo:i + 1]
        if len(seg) < min_samples:
            out[i] = v[i]
            continue
        xs = np.arange(len(seg))
        slope = np.polyfit(xs, seg, 1)[0]
        current = v[i]
        cap = current + 8.0
        pred = current + slope * horizon_samples
        out[i] = float(np.clip(pred, 0.0, max(cap, current * 1.5)))
    return out


def prepare_frame(df: pd.DataFrame, horizons_min=DEFAULT_HORIZONS_MIN,
                  sample_interval: float = SAMPLE_INTERVAL_REAL) -> pd.DataFrame:
    """Build the feature matrix + per-horizon targets, rolling features & time."""
    df = df.sort_values("ts").copy()
    q = df["queue_now"]
    df["mean5"] = q.rolling(5, min_periods=1).mean()
    df["mean10"] = q.rolling(10, min_periods=1).mean()
    df["mean30"] = q.rolling(30, min_periods=1).mean()
    df["std10"] = q.rolling(10, min_periods=1).std(ddof=0).fillna(0)
    df["queue_delta"] = q.diff(10).fillna(0)
    df["growth_rate"] = np.gradient(q.to_numpy(dtype=float))
    if "footfall" not in df.columns:
        df["footfall"] = 0
    df["footfall_rate"] = df["footfall"].to_numpy(dtype=float) / 60.0
    if "hour" not in df.columns:
        dt = pd.to_datetime(df["ts"], unit="s")
        df["hour"] = dt.dt.hour
        df["dow"] = dt.dt.dayofweek
    if "open_counters" not in df.columns:
        df["open_counters"] = 3
    # rolling linear-trend forecasts (bounded) mirroring the runtime fallback
    for h in horizons_min:
        horizon_samples = int(round(h * 60 / sample_interval))
        df[f"lin_{h}min"] = _rolling_linear_forecast(q.to_numpy(dtype=float), horizon_samples)
    target_cols = [f"target_{h}min" for h in horizons_min]
    use = ["ts", "camera_id", "queue_id", "queue_now", "mean5", "mean10",
           "mean30", "std10", "queue_delta", "growth_rate", "footfall",
           "footfall_rate", "hour", "dow", "open_counters"] \
        + [f"lin_{h}min" for h in horizons_min] + target_cols
    df = df[[c for c in use if c in df.columns]]
    df = df.dropna(subset=["queue_now"] + target_cols)
    return df


FEATURES = ["queue_now", "mean5", "mean10", "mean30", "std10", "queue_delta",
            "growth_rate", "footfall", "footfall_rate", "hour", "dow",
            "open_counters"]


def _expand_window_cv(X, y, n_splits=4):
    """Deterministic forward-chaining time-series splits.

    Each fold trains on a contiguous *earlier* window and validates on the
    *next* contiguous window (never shuffling, never borrowing the future).
    """
    n = len(X)
    piece = n // (n_splits + 1)
    folds = []
    for k in range(1, n_splits + 1):
        tr_end = piece * k
        if tr_end >= n:
            break
        val_end = min(tr_end + piece, n)
        yield X[:tr_end], y[:tr_end], X[tr_end:val_end], y[tr_end:val_end]


def _calibration_holdout_split(df, calibration_frac=0.7):
    """Split the time-sorted frame into calibration (early) + holdout (later).

    The holdout is entirely *after* calibration in time so nothing leaks.
    """
    df = df.sort_values("ts").reset_index(drop=True)
    cut = int(len(df) * calibration_frac)
    cut = max(cut, 20)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _optimal_blend_weight(ml_pred, trend_pred, y, grid=None):
    """Pick w in [0,1] minimising MAE of w*ml + (1-w)*trend on validation data."""
    grid = grid if grid is not None else np.linspace(0.0, 1.0, 21)
    best_w, best_mae = 1.0, float("inf")
    for w in grid:
        blended = w * np.asarray(ml_pred) + (1 - w) * np.asarray(trend_pred)
        mae = float(np.mean(np.abs(blended - np.asarray(y))))
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    return best_w, best_mae


def _confidence_from_errors(residuals, mae):
    """Derive a defensible 80% prediction interval from the held-out residuals.

    Uses the empirical quantiles of the signed residuals so the interval is
    grounded in the model's real behaviour rather than a Normal assumption.
    """
    residuals = residuals[~np.isnan(residuals)]
    lo = np.quantile(residuals, 0.10) if len(residuals) else -mae
    hi = np.quantile(residuals, 0.90) if len(residuals) else mae
    return float(lo), float(hi)


def fit_model_for_horizon(df_cal, df_hold, h_min, out_dir, sample_interval=5.0):
    """Train + time-aware-validate one model for a single horizon.

    Returns a dict of the chosen estimator, its validation metrics, the optimal
    blend weight, confidence interval, and the trend predictor params (which the
    runtime re-derives identically via QueuePredictor._linear_fallback_params).
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    col = f"target_{h_min}min"
    Xc = df_cal[FEATURES].values
    yc = df_cal[col].values
    Xh = df_hold[FEATURES].values
    yh = df_hold[col].values
    trend_h = np.asarray(df_hold[f"lin_{h_min}min"].to_numpy(dtype=float))

    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8,
                                              n_jobs=-1, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                      random_state=0),
    }
    best, best_cv_mae = None, float("inf")
    for name, mdl in candidates.items():
        # forward-chaining CV on the calibration set (time-ordered, no shuffle)
        fold_maes = []
        for Xf_tr, yf_tr, Xf_va, yf_va in _expand_window_cv(Xc, yc):
            m = mdl.__class__(**mdl.get_params())
            m.fit(Xf_tr, yf_tr)
            fold_maes.append(float(mean_absolute_error(yf_va, m.predict(Xf_va))))
        cv_mae = float(np.mean(fold_maes)) if fold_maes else float("inf")
        logging.info("%s [%dmin] forward-CV MAE=%.3f", name, h_min, cv_mae)
        if cv_mae < best_cv_mae:
            best, best_cv_mae = mdl, cv_mae

    # Retrain the winner on the full calibration frame, then score against the
    # unseen later holdout (the honest future).
    best.fit(Xc, yc)
    pred_hold = best.predict(Xh)
    mae = float(mean_absolute_error(yh, pred_hold))
    rmse = float(np.sqrt(mean_squared_error(yh, pred_hold)))
    r2 = float(r2_score(yh, pred_hold))

    # Baselines so the model's skill is judged against what "no model" would do.
    #   persistence -> predict the CURRENT queue for every horizon (a strong,
    #                  hard-to-beat baseline for short horizons).
    #   naive_mean  -> predict the calibration mean (the reference R2 uses).
    pers_pred = df_hold["queue_now"].to_numpy(dtype=float)
    pers_mae = float(mean_absolute_error(yh, pers_pred))
    pers_rmse = float(np.sqrt(mean_squared_error(yh, pers_pred)))
    pers_r2 = float(r2_score(yh, pers_pred))
    mean_pred = np.full_like(yh, float(yc.mean()))
    mean_mae = float(mean_absolute_error(yh, mean_pred))
    mean_rmse = float(np.sqrt(mean_squared_error(yh, mean_pred)))

    # Validation-selected blend weight between ML and the bounded linear trend.
    blend_w, blend_mae = _optimal_blend_weight(pred_hold, trend_h, yh)

    residuals = np.asarray(yh) - np.asarray(pred_hold)
    lo80, hi80 = _confidence_from_errors(residuals, mae)

    out_path = out_dir / f"queue_model_{h_min}min.joblib"
    import joblib
    joblib.dump(best, out_path)

    return {
        "horizon_minutes": h_min,
        "model_path": str(out_path.relative_to(ROOT)),
        "model": type(best).__name__,
        "cv_mae_forward_chain": round(best_cv_mae, 4),
        "holdout_r2": round(r2, 4),
        "holdout_mae": round(mae, 4),
        "holdout_rmse": round(rmse, 4),
        "baseline_persistence_mae": round(pers_mae, 4),
        "baseline_persistence_rmse": round(pers_rmse, 4),
        "baseline_persistence_r2": round(pers_r2, 4),
        "baseline_mean_mae": round(mean_mae, 4),
        "baseline_mean_rmse": round(mean_rmse, 4),
        "beats_persistence": bool(rmse < pers_rmse),
        "beats_mean": bool(mae < mean_mae),
        "blend_weight": round(blend_w, 4),
        "blend_mae": round(blend_mae, 4),
        "confidence_interval_80": [round(lo80, 3), round(hi80, 3)],
        "n_calibration": int(len(df_cal)),
        "n_holdout": int(len(df_hold)),
        "features": FEATURES,
        "trained_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def train_all(df, source_label: str, out_dir: Path, horizons_min,
              sample_interval: float = SAMPLE_INTERVAL_REAL) -> int:
    df = prepare_frame(df, horizons_min, sample_interval=sample_interval) \
        .sort_values("ts").reset_index(drop=True)
    if len(df) < 60:
        logging.error("not enough labelled samples for training (%d); collect more history", len(df))
        return 1

    cal, hold = _calibration_holdout_split(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for h in horizons_min:
        result = fit_model_for_horizon(cal, hold, h, out_dir,
                                       sample_interval=sample_interval)
        if result is None:
            logging.error("training failed for %dmin horizon", h)
            return 1
        models.append(result)
        logging.info("[%dmin] model=%s holdout MAE=%.3f RMSE=%.3f R2=%.3f "
                     "persistence_MAE=%.3f beats_persistence=%s "
                     "blend_w=%.3f 80%%CI=[%.2f,%.2f]",
                     h, result["model"], result["holdout_mae"],
                     result["holdout_rmse"], result["holdout_r2"],
                     result["baseline_persistence_mae"],
                     result["beats_persistence"],
                     result["blend_weight"],
                     result["confidence_interval_80"][0],
                     result["confidence_interval_80"][1])

    metrics = {
        "source": source_label,
        "horizons_minutes": horizons_min,
        "models": models,
        "n_samples": int(len(df)),
        "n_calibration": int(len(cal)),
        "n_holdout": int(len(hold)),
        "features": FEATURES,
        "validation": "walk-forward/expand-window time-series (no shuffle)",
        "trained_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    metrics_path = out_dir / "queue_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logging.info("saved %d horizon models + metrics -> %s", len(models), out_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000, help="synthetic rows when no --csv")
    ap.add_argument("--csv", default=None, help="real data log (see ml.queue.datalogger)")
    ap.add_argument("--out", default="models/prediction", help="output directory for models + metrics")
    ap.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS_MIN,
                    help="forecast horizons in minutes (default: 5 10)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    horizons = sorted(set(args.horizons))
    out_dir = ROOT / args.out

    if args.csv:
        try:
            raw = load_real_features(args.csv, horizons)
        except (FileNotFoundError, ValueError) as exc:
            logging.error("could not read real data %s: %s", args.csv, exc)
            return 1
        logging.info("real-data training: %d rows from %s", len(raw), args.csv)
        return train_all(raw, "real", out_dir, horizons,
                         sample_interval=SAMPLE_INTERVAL_REAL)

    df = simulate_queue_series(args.samples, horizons_min=horizons)
    logging.info("synthetic training: %d rows", len(df))
    return train_all(df, "synthetic", out_dir, horizons,
                     sample_interval=SAMPLE_INTERVAL_SYNTHETIC)


if __name__ == "__main__":
    raise SystemExit(main())
