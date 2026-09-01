"""Runtime prediction monitoring for queue-length forecasts.

The online predictor produces forecasts but, on its own, never learns whether
they were right. This evaluator records each prediction and, once the forecast
horizon has elapsed, looks up the *actual* queue length and accumulates a
running MAE / RMSE per horizon - the same metrics the training script reports,
but now tracked live so performance on the site can be watched over time.

Design:
  * A prediction is stored with a compact snapshot of the queue history at t0.
  * When ``now`` exceeds ``t0 + horizon``, the actual queue length at that time
    is recovered by interpolating the recorded history (which may include values
    already past the horizon).
  * Errors accumulate in-memory (for ``metrics()``) and are also appended to a
    CSV for offline analysis.

If history is insufficient to resolve an actual value, the sample is dropped
silently - this never throws into the analytics loop.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QueuePredictionEvaluator:
    """Tracks prediction-vs-actual error metrics over time."""

    def __init__(self, eval_path, horizons: List[int], queue_id: str = "all",
                 history_window: int = 120):
        self.eval_path = Path(eval_path)
        self.horizons = sorted(int(h) for h in horizons)
        self.queue_id = queue_id
        self.history_window = int(history_window)
        # made_ts -> (horizon, pred, history snapshot, source)
        self._pending: List[Tuple[float, int, float, List[Tuple[float, float]], str]] = []
        # horizon -> list of squared errors + absolute errors
        self._sq: Dict[int, List[float]] = {h: [] for h in self.horizons}
        self._abs: Dict[int, List[float]] = {h: [] for h in self.horizons}
        self._sources: Dict[int, Dict[str, int]] = {h: {} for h in self.horizons}
        self._fh = None
        self._file = None

    # ------------------------------------------------------------- recording
    def record(self, made_ts: float, horizons_values: Dict[int, float],
               queue_history: List[Tuple[float, float]], source: str) -> None:
        """Record predictions made at ``made_ts`` for several horizons."""
        for h in self.horizons:
            pred = horizons_values.get(h)
            if pred is None:
                continue
            hist = [(float(t), float(v)) for t, v in queue_history]
            self._pending.append((float(made_ts), int(h), float(pred), hist, str(source)))
        # keep only still-maturing samples bounded in size
        if len(self._pending) > 5000:
            self._pending = self._pending[-2000:]

    def evaluate_ripe(self, now: float, flush: bool = True) -> None:
        """Resolve and score any pending predictions whose horizon has elapsed."""
        still_pending: List[Tuple[float, int, float, List[Tuple[float, float]], str]] = []
        for made_ts, h, pred, hist, source in self._pending:
            if now < made_ts + h * 60:
                still_pending.append((made_ts, h, pred, hist, source))
                continue
            actual = self._resolve_actual(hist, made_ts + h * 60)
            if actual is None:
                continue
            err = float(actual - pred)
            self._abs[h].append(abs(err))
            self._sq[h].append(err * err)
            self._sources[h][source] = self._sources[h].get(source, 0) + 1
            if flush:
                self._append_row(made_ts, h, pred, actual, err, source)
        self._pending = still_pending

    @staticmethod
    def _resolve_actual(hist: List[Tuple[float, float]], t_target: float) -> Optional[float]:
        """Interpolate the queue value at ``t_target`` from a history snapshot."""
        if not hist:
            return None
        ts = np.array([t for t, _ in hist], dtype=float)
        vals = np.array([v for _, v in hist], dtype=float)
        if t_target <= ts.min() or t_target >= ts.max():
            return None   # cannot extrapolate beyond the recorded window
        return float(np.interp(t_target, ts, vals))

    # ---------------------------------------------------------------- output
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """Current MAE / RMSE / sample count per horizon."""
        out: Dict[str, Dict[str, float]] = {}
        for h in self.horizons:
            n = len(self._abs[h])
            out[f"{h}min"] = {
                "samples": n,
                "mae": round(float(np.mean(self._abs[h])), 4) if n else 0.0,
                "rmse": round(float(np.sqrt(np.mean(self._sq[h]))), 4) if n else 0.0,
                "sources": dict(self._sources[h]),
            }
        return out

    def _append_row(self, made_ts, h, pred, actual, err, source) -> None:
        try:
            self._ensure_open()
            self._fh.writerow([f"{made_ts:.1f}", h, f"{pred:.2f}", f"{actual:.2f}",
                               f"{err:.2f}", source])
        except Exception as exc:  # pragma: no cover
            logger.debug("eval row write skipped: %s", exc)

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        self.eval_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.eval_path.exists() or self.eval_path.stat().st_size == 0
        self._file = open(self.eval_path, "a", newline="")
        self._fh = csv.writer(self._file)
        if new:
            self._fh.writerow(["made_ts", "horizon_min", "predicted", "actual", "error", "source"])

    def close(self) -> None:
        try:
            self.evaluate_ripe(time.time(), flush=True)
        except Exception:
            pass
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._fh = None
