"""Historical analytics store (in-memory, always on).

Every persisted snapshot already goes to the DB in the CLI flow, but the web
dashboard flow deliberately uses in-memory analytics (``Repository(None)``).
To make historical analytics work *identically in both flows* we keep a small,
bucketed, time-windowed history inside each analytics service:

  * 1-minute buckets  - kept ``minute_buckets`` (default 240 -> last 4 hours)
  * 1-hour buckets    - kept ``hour_buckets`` (default 168 -> last week)

Buckets hold honest aggregates of what the pipeline actually observed
(entries, exits, occupancy, active shoppers, queue length, predicted queue,
congestion status), and the window rolls as time advances so memory stays
bounded. No persistence, no fake smoothing - reports are computed directly
from these raw buckets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _empty() -> Dict[str, float]:
    return {"entries": 0.0, "exits": 0.0, "count": 0.0,
            "occ_sum": 0.0, "occ_max": 0.0, "unique_max": 0.0,
            "queue_sum": 0.0, "queue_max": 0.0, "pred_max": 0.0,
            "congested": 0.0}


class AnalyticsHistory:
    def __init__(self, camera_id: str,
                 minute_buckets: int = 240,
                 hour_buckets: int = 168):
        self.camera_id = camera_id
        self.minute_buckets = minute_buckets
        self.hour_buckets = hour_buckets
        self._minutes: Dict[int, Dict[str, float]] = {}
        self._hours: Dict[int, Dict[str, float]] = {}
        self._start_ts: Optional[float] = None

    # --------------------------------------------------------------- feeding
    def record(self, entries: int, exits: int, occupancy: int, unique: int,
               queue_total: int, pred_max: float, status: str, now: float) -> None:
        if self._start_ts is None:
            self._start_ts = now
        from datetime import datetime, timezone
        bucket_min = int(now // 60)
        bucket_hour = int(now // 3600)
        congested = 1.0 if status in ("WARNING", "HIGH") else 0.0
        for store, key in ((self._minutes, bucket_min), (self._hours, bucket_hour)):
            b = store.setdefault(key, _empty())
            b["entries"] += entries
            b["exits"] += exits
            b["count"] += 1.0
            b["occ_sum"] += occupancy
            b["occ_max"] = max(b["occ_max"], occupancy)
            b["unique_max"] = max(b["unique_max"], unique)
            b["queue_sum"] += queue_total
            b["queue_max"] = max(b["queue_max"], queue_total)
            b["pred_max"] = max(b["pred_max"], pred_max)
            b["congested"] += congested
        self._prune()

    def _prune(self) -> None:
        from datetime import datetime, timezone
        now_min = int(datetime.now(timezone.utc).timestamp()) // 60
        now_hour = int(datetime.now(timezone.utc).timestamp()) // 3600
        for key in [k for k in self._minutes if k < now_min - self.minute_buckets]:
            del self._minutes[key]
        for key in [k for k in self._hours if k < now_hour - self.hour_buckets]:
            del self._hours[key]

    # ---------------------------------------------------------------- reads
    def minute_series(self, minutes: int = 120) -> List[Dict[str, Any]]:
        """Last ``minutes`` of 1-minute buckets, oldest first."""
        keys = sorted(self._minutes)[-minutes:]
        out = []
        for k in keys:
            b = self._minutes[k]
            n = max(int(b["count"]), 1)
            out.append(self._bucketed(k * 60, b, n))
        return out

    def hourly_series(self, days: int = 7) -> List[Dict[str, Any]]:
        """Last ``days`` of 1-hour buckets, oldest first."""
        keys = sorted(self._hours)[-days * 24:]
        out = []
        for k in keys:
            b = self._hours[k]
            n = max(int(b["count"]), 1)
            out.append(self._bucketed(k * 3600, b, n))
        return out

    def daily_summary(self, days: int = 14) -> List[Dict[str, Any]]:
        """Per-calendar-day totals: visits, peak occupancy, busy hours."""
        from datetime import datetime, timezone

        by_day: Dict[str, Dict[str, float]] = {}
        for k, b in self._hours.items():
            day = datetime.fromtimestamp(k * 3600, tz=timezone.utc).date().isoformat()
            d = by_day.setdefault(day, _empty())
            for field in ("entries", "exits"):
                d[field] += b[field]
            d["queue_max"] = max(d["queue_max"], b["queue_max"])
            d["pred_max"] = max(d["pred_max"], b["pred_max"])
            d["congested"] += b["congested"]
            d["occ_max"] = max(d["occ_max"], b["occ_max"])
            d["unique_max"] = max(d["unique_max"], b["unique_max"])
            d["count"] += b["count"]
        days_sorted = sorted(by_day)[-days:]
        out = []
        for day in days_sorted:
            d = by_day[day]
            n = max(int(d["count"]), 1)
            out.append({
                "date": day,
                "visits": int(d["entries"]),
                "exits": int(d["exits"]),
                "peak_occupancy": int(d["occ_max"]),
                "peak_unique": int(d["unique_max"]),
                "peak_queue": int(d["queue_max"]),
                "peak_predicted_queue": round(d["pred_max"], 1),
                # fraction of sampled snapshots that were congested, as minutes
                "congested_minutes_est": round((d["congested"] / n) * 60.0, 1),
            })
        return out

    def peak_hours(self) -> List[Dict[str, Any]]:
        """Average hourly footfall per hour-of-day, ranked busiest first."""
        from collections import defaultdict
        from datetime import datetime, timezone

        acc = defaultdict(lambda: [0.0, 0])
        for k, b in self._hours.items():
            h = datetime.fromtimestamp(k * 3600, tz=timezone.utc).hour
            acc[h][0] += b["entries"]
            acc[h][1] += 1
        ranked = [
            {"hour": h, "avg_footfall": round(total / max(cnt, 1), 1)}
            for h, (total, cnt) in acc.items()
        ]
        return sorted(ranked, key=lambda r: r["avg_footfall"], reverse=True)

    def activity(self, days: int = 7) -> Dict[str, Any]:
        """Compact weekly activity summary (for dashboards)."""
        per_hour = {k: b for k, b in self._hours.items()
                    if list(sorted(self._hours))[-days * 24:][0] <= k}
        totals = _empty()
        for b in per_hour.values():
            totals["entries"] += b["entries"]
            totals["exits"] += b["exits"]
            totals["count"] += b["count"]
            totals["queue_max"] = max(totals["queue_max"], b["queue_max"])
            totals["occ_max"] = max(totals["occ_max"], b["occ_max"])
        n = max(int(totals["count"]), 1)
        return {
            "camera_id": self.camera_id,
            "period_hours": len(per_hour),
            "visits": int(totals["entries"]),
            "exits": int(totals["exits"]),
            "avg_footfall_per_hour": round(totals["entries"] / n, 1),
            "avg_occupancy": round(totals["occ_sum"] / n, 1),
            "peak_occupancy": int(totals["occ_max"]),
            "peak_queue": int(totals["queue_max"]),
            "peak_predicted_queue": round(totals["pred_max"], 1),
            "peak_hours": self.peak_hours(),
        }

    @staticmethod
    def _bucketed(ts: int, b: Dict[str, float], n: int) -> Dict[str, Any]:
        return {
            "ts": ts,
            "entries": int(b["entries"]),
            "exits": int(b["exits"]),
            "avg_occupancy": round(b["occ_sum"] / n, 2),
            "peak_occupancy": int(b["occ_max"]),
            "peak_unique": int(b["unique_max"]),
            "avg_queue": round(b["queue_sum"] / n, 2),
            "peak_queue": int(b["queue_max"]),
            "peak_predicted_queue": round(b["pred_max"], 1),
            "congested_minutes": round(b["congested"] / n, 2),
        }