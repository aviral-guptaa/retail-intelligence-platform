"""Real-data collection for the queue-length predictor.

Every ``sample_interval_seconds`` the analytics service appends one CSV row per
queue zone with the SIH-spec feature inputs (live queue, rolling means, growth,
footfall, time-of-day, open counters). The target column is left empty at
collection time; ``scripts/train_queue_model.py --csv`` fills it by looking
``horizon_seconds`` into the future with each queue's own timeline, then fits
and evaluates the model.

This is the honest "real data pipeline" for the demo->pilot transition: demo
mode logs synthetic ground truth, but the exact same file format is written by
real camera runs - so retraining on site logs is a copy + one command.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_COLUMNS = ["ts", "camera_id", "queue_id", "queue_now", "mean5", "mean10",
            "growth_rate", "footfall", "hour", "dow", "open_counters", "target"]
_HORIZON_MIN_FOR_TARGET = 10  # the default training/forecast horizon


class QueueDataLogger:
    """Throttled CSV writer for queue feature rows."""

    def __init__(self, log_path: str | Path, camera_id: str,
                 queue_ids: Iterable[str], sample_interval: float = 5.0):
        self.path = Path(log_path)
        self.camera_id = camera_id
        self.queue_ids = list(queue_ids)
        self.sample_interval = float(sample_interval)
        self._fh = None
        self._last_logged: Dict[str, float] = {}

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        if new_file:
            self._fh.write(",".join(_COLUMNS) + "\n")

    def update(self, queue_histories: Dict[str, list], footfall: int,
               open_counters: int, now: Optional[float] = None) -> None:
        """Log a row per queue when its sampling interval has elapsed."""
        now = time.time() if now is None else now
        if not self.queue_ids:
            return
        self._ensure_open()
        ltm = time.localtime(now)
        for qid in self.queue_ids:
            hist = queue_histories.get(qid, [])
            if not hist:
                continue
            last_ts = self._last_logged.get(qid, 0.0)
            if now - last_ts < self.sample_interval:
                continue
            try:
                self._last_logged[qid] = now
                vals = [v for _, v in hist]
                current = float(vals[-1])
                mean5 = float(sum(vals[-5:]) / len(vals[-5:])) if vals[-5:] else current
                mean10 = float(sum(vals[-10:]) / len(vals[-10:])) if vals[-10:] else current
                growth = 0.0
                if len(vals) >= 2:
                    import numpy as _np
                    growth = float(_np.polyfit(_np.arange(len(vals)), _np.asarray(vals, dtype=float), 1)[0])
                row = [f"{now:.3f}", self.camera_id, qid, f"{current:.1f}", f"{mean5:.2f}",
                       f"{mean10:.2f}", f"{growth:.4f}", str(footfall), str(ltm.tm_hour),
                       str(ltm.tm_wday), str(open_counters), ""]
                self._fh.write(",".join(row) + "\n")
            except Exception as exc:
                logger.debug("queue datalog write skipped: %s", exc)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None