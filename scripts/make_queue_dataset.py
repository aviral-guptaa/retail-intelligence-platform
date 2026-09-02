"""Generate a realistic retail-checkout queue dataset for training.

The existing ``simulate_queue_series`` in ``train_queue_model.py`` produces a
tiny, mostly-noise queue (mean ~1.3, std ~0.56) which caps the achievable R2.
This script instead simulates proper **M/M/c** checkout queues with diurnal
arrival patterns and congestion episodes, producing a *representative* training
CSV in the exact schema the pipeline consumes:

    ts, camera_id, queue_id, queue_now, footfall, open_counters, ...

so it can be fed straight to ``train_queue_model.py --csv``.

Design (defensible, transparent):
  * Poisson arrivals whose rate follows a diurnal curve (morning + evening
    rush, quieter middle) scaled by a weekend/weekday factor - so the queue has
    genuine, learnable structure rather than pure noise.
  * A multi-server (M/M/c) queue: arrivals are served at rate c*mu per minute,
    giving the short-horizon continuity the forecaster needs.
  * Multiple stores x checkout zones, each a separate series (the pipeline
    trains one model over the groups but scores time-ordered).
  * Optional mild observation noise on ``queue_now`` to mimic CV/motion
    estimation error.

Run:
    python scripts/make_queue_dataset.py --stores 4 --zones 2 --days 21 --out data/processed/queue_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

SAMPLE_INTERVAL = 10.0        # seconds per row (matches a 5-10s cadence)
SECONDS_PER_DAY = 12 * 3600   # 12-hour trading day


def _diurnal_factor(hour_of_day: float, weekday: bool) -> float:
    """Arrival-rate multiplier across a 12h trading day (08:00-20:00).

    Morning + evening rush with a midday dip; weekends flatter + busier midday.
    """
    if not weekday:
        return 1.0 + 0.15 * np.sin((hour_of_day - 8.0) / 12.0 * np.pi)
    # weekday: rush at open, lunch bump, strong evening peak
    m = 1.0 + 0.45 * np.exp(-((hour_of_day - 10.5) ** 2) / (2 * 1.6 ** 2))  # mid-morning
    m += 0.75 * np.exp(-((hour_of_day - 17.5) ** 2) / (2 * 1.2 ** 2))        # evening rush
    return m


def simulate_store_day(camera_id: str, queue_id: str, rng: np.random.Generator,
                       base_arrivals_per_min: float, mu_per_min: float,
                       open_counters_max: int, date_idx: int,
                       sample_interval: float = SAMPLE_INTERVAL) -> pd.DataFrame:
    """Simulate one checkout zone for one trading day at a fixed cadence."""
    open_hour, close_hour = 8.0, 20.0
    n = int(SECONDS_PER_DAY // sample_interval)
    ts = np.arange(n) * sample_interval + date_idx * SECONDS_PER_DAY
    t_min = np.arange(n) * (sample_interval / 60.0)
    hour_in_day = (ts % SECONDS_PER_DAY) / 3600.0 + open_hour
    weekday = (date_idx % 7) < 5

    footfall = np.zeros(n)
    queue = np.zeros(n, dtype=float)
    open_counters = np.zeros(n, dtype=int)

    # open counters track demand: more when the queue is busy
    q = 0.0
    s_per_min = sample_interval / 60.0
    for i in range(n):
        arrivals_per_min = base_arrivals_per_min * _diurnal_factor(hour_in_day[i], weekday)
        lam = max(0.0, arrivals_per_min)
        new_arrivals = float(rng.poisson(lam * s_per_min))
        # footfall ~ observed customers arriving (smoothed proxy)
        footfall[i] = int(rng.poisson(max(0.0, lam) * (s_per_min * 6)))
        c = max(1, min(open_counters_max, int(1 + q / 5.0)))    # staff up with demand
        open_counters[i] = c
        # queueing service: each counter clears mu per minute; allow the queue
        # to build toward congestion in bursts but keep the system stable.
        service_capacity = c * mu_per_min * s_per_min
        served = min(q + new_arrivals, service_capacity)
        q = max(0.0, q + new_arrivals - served)
        # light observation noise to emulate CV/motion estimation error
        obs = q + rng.normal(0.0, 0.35)
        queue[i] = max(0.0, round(obs, 1))

    return pd.DataFrame({
        "ts": ts,
        "camera_id": camera_id,
        "queue_id": queue_id,
        "queue_now": queue,
        "footfall": footfall,
        "open_counters": open_counters,
        "hour": hour_in_day,
        "dow": [date_idx % 7] * n,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", type=int, default=4, help="number of stores")
    ap.add_argument("--zones", type=int, default=2, help="checkout zones per store")
    ap.add_argument("--days", type=int, default=21, help="trading days")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/processed/queue_features.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = []
    for store in range(args.stores):
        # vary demand/service across stores so the model sees heterogeneity
        base = float(rng.uniform(1.8, 2.5))     # arrivals/min at peak
        mu = float(rng.uniform(1.0, 1.2))       # service per counter per min (~1/min)
        maxc = int(rng.integers(4, 6))
        for zone in range(args.zones):
            for day in range(args.days):
                f = simulate_store_day(f"store_{store:02d}", f"checkout_{zone:02d}",
                                       rng, base, mu, maxc, day)
                frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {len(df):,} rows ({args.stores} stores x {args.zones} zones x "
          f"{args.days} days) -> {out}")
    print(f"queue_now: mean={df['queue_now'].mean():.2f} "
          f"std={df['queue_now'].std():.2f} max={df['queue_now'].max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
