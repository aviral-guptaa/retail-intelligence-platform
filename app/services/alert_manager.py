"""Unified alert store: every alert type (congestion, shelf, queue, camera)
flows through one place, tagged with severity but always persisted in memory
with a rolling active window. Optionally a generic webhook receives new
alerts in the background; with no ``webhook_url`` configured the webhook is
disabled and alerts are still recorded locally (honest no-op, plain Python,
no external dependency).

Severities: INFO < WARNING < HIGH. Each alert has an id, an optional expiry
(clear/ack) so the "active" set shrinks as conditions resolve.
"""
from __future__ import annotations

import itertools
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

_SEVERITIES = {"INFO": 0, "WARNING": 1, "HIGH": 2}


class Alert:
    def __init__(self, alert_type: str, severity: str, message: str,
                 camera_id: str, store_id: Optional[str] = None,
                 source: str = "rule", dedup_key: Optional[str] = None):
        self.id = next(_IDS)
        self.alert_type = alert_type
        self.severity = severity if severity in _SEVERITIES else "INFO"
        self.message = message
        self.camera_id = camera_id
        self.store_id = store_id
        self.source = source
        self.dedup_key = dedup_key
        self.created_ts = time.time()
        self.resolved_ts: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.resolved_ts is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "camera_id": self.camera_id,
            "store_id": self.store_id,
            "source": self.source,
            "created_ts": self.created_ts,
            "resolved_ts": self.resolved_ts,
            "active": self.active,
        }


_IDS = itertools.count(1)


class AlertStore:
    """In-memory unified alert store with optional webhook delivery."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        settings = settings or {}
        self.max_active = int(settings.get("max_active", 50))
        self.webhook_url = settings.get("webhook_url") or ""
        self.webhook_retry = float(settings.get("webhook_retry_seconds", 30))
        self._alerts: List[Alert] = []
        self._lock = threading.Lock()
        self._last_delivery: Dict[int, float] = {}
        self._delivered: set = set()

    # ------------------------------------------------------------ emitting
    def emit(self, alert_type: str, severity: str, message: str,
             camera_id: str, store_id: Optional[str] = None,
             source: str = "rule", dedup_key: Optional[str] = None) -> Optional[Alert]:
        """Record a new alert, deduplicated on ``dedup_key`` while active."""
        if dedup_key is not None:
            for a in self.active():
                if a.dedup_key == dedup_key and a.alert_type == alert_type:
                    return None          # still active, don't re-fire
        alert = Alert(alert_type, severity, message, camera_id, store_id, source,
                      dedup_key=dedup_key)
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > self.max_active:
                self._alerts = self._alerts[-self.max_active:]
        self._dispatch(alert)
        return alert

    def resolve(self, dedup_key: str, alert_type: str) -> None:
        """Clear any active alert matching the key/type (condition recovered)."""
        now = time.time()
        for a in list(self.active()):
            if a.dedup_key == dedup_key and a.alert_type == alert_type:
                a.resolved_ts = now
                self._delivered.discard(a.id)

    # ------------------------------------------------------------ webhook
    def _dispatch(self, alert: Alert) -> None:
        if not self.webhook_url:
            return
        threading.Thread(target=self._deliver, args=(alert,), daemon=True).start()

    def _deliver(self, alert: Alert) -> None:
        payload = alert.to_dict()
        encoded = __import__("json").dumps(payload).encode("utf-8")
        try:
            req = urllib.request.Request(
                self.webhook_url, data=encoded,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5):
                self._delivered.add(alert.id)
        except Exception:
            self._delivered.discard(alert.id)
            self._last_delivery[alert.id] = time.time()

    def retry_pending(self) -> int:
        """Best-effort background resend of undelivered alerts."""
        retried = 0
        for a in list(self._alerts):
            if a.id in self._delivered:
                continue
            last = self._last_delivery.get(a.id, 0.0)
            if time.time() - last >= self.webhook_retry:
                self._deliver(a)
                self._last_delivery[a.id] = time.time()
                retried += 1
        return retried

    # ---------------------------------------------------------------- reads
    def active(self, min_severity: str = "INFO") -> List[Alert]:
        out = [a for a in self._alerts if a.active
               and _SEVERITIES.get(a.severity, 0) >= _SEVERITIES[min_severity]]
        return sorted(out, key=lambda a: a.created_ts, reverse=True)

    def historical(self, limit: int = 100, alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        items = list(reversed(self._alerts))
        if alert_type:
            items = [a for a in items if a.alert_type == alert_type]
        return [a.to_dict() for a in items[:limit]]

    def snapshot(self) -> Dict[str, Any]:
        act = self.active()
        return {
            "active_count": len(act),
            "active": [a.to_dict() for a in act],
            "by_severity": {
                level: sum(1 for a in self._alerts
                           if a.severity == level and a.active)
                for level in ("INFO", "WARNING", "HIGH")
            },
            "total_emitted": len(self._alerts),
            "webhook_enabled": bool(self.webhook_url),
        }