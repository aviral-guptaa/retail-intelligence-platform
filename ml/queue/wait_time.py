"""Estimated waiting time behind checkout queues.

Rule-based estimate (clearly NOT the ML queue-length prediction - see
:mod:`ml.queue.predictor` for the ML forecast and ``prediction.source``):

    wait_minutes = queue_length * average_service_time_seconds / 60 / open_counters

``average_service_time_seconds`` is the mean checkout duration per shopper with
one open counter - measure real throughput at the site and put it in
``config/settings.yaml -> queue.average_service_time_seconds``. This is pure
queuing-theory arithmetic; it calibrates instantly and has no training data.
"""
from __future__ import annotations

from typing import Dict


class WaitTimeEstimator:
    """Estimates how long the person at the back of a queue will wait."""

    def __init__(self, settings: Dict):
        # Explicitly use the queue-domain average service time (seconds per shopper).
        self.average_service_time_seconds = float(
            settings.get("average_service_time_seconds", 30))
        self.open_counters = int(settings.get("open_counters", 3))
        # Backwards-compatible convenience (still configurable if present).
        if "service_rate_per_minute" in settings:
            rate = float(settings["service_rate_per_minute"]) or 0.0
            if rate > 0:
                self.average_service_time_seconds = 60.0 / rate

    def estimate(self, queue_length: int, counters_open: int | None = None) -> float:
        """Waiting minutes for the newest arrival in a queue of ``queue_length``."""
        n = counters_open if counters_open else self.open_counters
        if n <= 0 or queue_length <= 0:
            return 0.0
        service_time_min = self.average_service_time_seconds / 60.0
        return round(queue_length * service_time_min / n, 2)

    def recommendation(self, queue_length: int, wait_minutes: float) -> str | None:
        if wait_minutes > 10:
            return f"Wait time ~{wait_minutes:.0f} min for a queue of {queue_length} - open an additional counter."
        return None

    def explain(self) -> Dict:
        """Show the formula that produced the estimate (for the dashboard)."""
        return {
            "formula": "queue_length * average_service_time_seconds / 60 / open_counters",
            "average_service_time_seconds": self.average_service_time_seconds,
            "open_counters": self.open_counters,
            "kind": "rule_based",
        }