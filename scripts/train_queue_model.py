"""Train the queue-length prediction model (RandomForest / GradientBoosting).

Two data sources:

 1. Synthetic simulation (default) - bursty arrivals + service, used so the
    dashboard has a model from day one (previously-partnered demo data). The
    metrics JSON records source="synthetic".

 2. Real collected data - ``--csv data/processed/queue_features.csv``, the file
    written continuously by ``ml.queue.datalogger`` while the pipeline runs in
    demo or live mode. For each (camera, queue) series the target is the queue
    length ``horizon_seconds`` ahead (interpolated on that queue's own timeline);
    features are [queue_now, mean5, mean10, growth_rate, footfall, hour, dow,
    open_counters]. Metrics + source="real" are stored so the API can tell the
    dashboard whether a live model is serving.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

HORIZON_SECONDS = 600  # 10 minutes ahead (sampling is handled per timeline)


def simulate_queue_series(n_steps: int, seed: int = 7) -> pd.DataFrame:
    """Synthetic 10-minute-horizon store data for training.

    Queue length evolves as: arrival bursts push it up, service draws it down.
    The target is the queue length ~10 minutes ahead (60 samples at 10s spacing).
    """
    rng = np.random.default_rng(seed)
    service_rate = 5.0  # shoppers/minute with an open counter
    open_counters = np.zeros(n_steps, dtype=int)

    hour = np.zeros(n_steps)
    dow = np.zeros(n_steps)
    queue = np.zeros(n_steps + 1)
    footfall = np.zeros(n_steps)
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
        samples_per_min = 6  # a "sample minute" steps the sim 10s per row
        queue[i + 1] = max(0.0, queue[i] + arrivals / samples_per_min - served / samples_per_min)

    q = queue[:-1]
    target = np.concatenate([queue[60:], np.full(59, np.nan)])
    df = pd.DataFrame({
        "ts": np.arange(n_steps) * 10.0,
        "camera_id": "synthetic", "queue_id": "checkout_01",
        "queue_now": q,
        "mean5": pd.Series(q).rolling(5, min_periods=1).mean().values,
        "mean10": pd.Series(q).rolling(10, min_periods=1).mean().values,
        "growth_rate": np.gradient(q),
        "footfall": footfall,
        "hour": hour,
        "dow": dow,
        "open_counters": open_counters,
        "target": target,
    })
    return df.dropna()


def load_real_features(csv_path: str) -> pd.DataFrame:
    """Read the datalogger CSV and compute horizon targets per queue series."""
    raw = pd.read_csv(csv_path)
    required = ["ts", "camera_id", "queue_id", "queue_now"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing} (got {list(raw.columns)})")

    series = raw.sort_values("ts").drop_duplicates(subset=["ts", "camera_id", "queue_id"])
    groups = series.groupby(["camera_id", "queue_id"])
    targets = []
    for _, g in groups:
        ts = g["ts"].to_numpy(dtype=float)
        val = g["queue_now"].to_numpy(dtype=float)
        target = np.interp(ts + HORIZON_SECONDS, ts, val)
        targets.append(target)
    series["target"] = np.concatenate(targets)
    return series.dropna(subset=["target"])


def prepare_frame(df: pd.DataFrame):
    """Build the feature matrix + target, rolling features and time columns."""
    df = df.sort_values("ts").copy()
    q = df["queue_now"]
    df["mean5"] = q.rolling(5, min_periods=1).mean()
    df["mean10"] = q.rolling(10, min_periods=1).mean()
    df["growth_rate"] = np.gradient(q.to_numpy(dtype=float))
    if "hour" not in df.columns:
        dt = pd.to_datetime(df["ts"], unit="s")
        df["hour"] = dt.dt.hour
        df["dow"] = dt.dt.dayofweek
    if "open_counters" not in df.columns:
        df["open_counters"] = 3
    df = df[df["target"].notna()].dropna(subset=["queue_now"])
    return df


def train_eval(df: pd.DataFrame, source_label: str, out_path: str) -> int:
    features = ["queue_now", "mean5", "mean10", "growth_rate", "footfall",
                "hour", "dow", "open_counters"]
    X, y = df[features].values, df["target"].values

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0),
    }
    best, best_r2 = None, -1.0
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        r2 = r2_score(yte, mdl.predict(Xte))
        mae = mean_absolute_error(yte, mdl.predict(Xte))
        logging.info("%s  R2=%.3f  MAE=%.2f", name, r2, mae)
        if r2 > best_r2:
            best, best_r2 = mdl, r2

    if best is None:
        logging.error("no model trained")
        return 1

    pred = best.predict(Xte)
    metrics = {
        "source": source_label,
        "r2": round(float(r2_score(yte, pred)), 4),
        "mae": round(float(mean_absolute_error(yte, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
        "n_samples": int(len(df)),
        "horizon_seconds": HORIZON_SECONDS,
        "features": features,
        "model": type(best).__name__,
        "trained_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out = Path(ROOT) / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(best, out)
    metrics_path = out.parent / "queue_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logging.info("saved best model (%s, R2=%.3f) -> %s", type(best).__name__, best_r2, out)
    logging.info("saved metrics (source=%s) -> %s", source_label, metrics_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000, help="synthetic rows when no --csv")
    ap.add_argument("--csv", default=None, help="real data log (see ml.queue.datalogger)")
    ap.add_argument("--out", default="models/prediction/queue_model.joblib")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.csv:
        try:
            raw = load_real_features(args.csv)
        except (FileNotFoundError, ValueError) as exc:
            logging.error("could not read real data %s: %s", args.csv, exc)
            return 1
        df = prepare_frame(raw)
        if len(df) < 50:
            logging.error("not enough labelled samples for real training (%d); "
                          "collect more queue history first", len(df))
            return 1
        logging.info("real-data training: %d rows from %s", len(df), args.csv)
        return train_eval(df, "real", args.out)

    df = simulate_queue_series(args.samples)
    logging.info("synthetic training: %d rows", len(df))
    return train_eval(df, "synthetic", args.out)


if __name__ == "__main__":
    raise SystemExit(main())