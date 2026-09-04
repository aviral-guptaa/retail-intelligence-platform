"""Data-access helpers over the SQLAlchemy session.

The repository is thin on purpose: analytics are computed in the ML/services
layer and *persisted* here, keeping ML modules free of ORM imports.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from typing import Optional

logger = logging.getLogger(__name__)

from database.models import AlertRecord, AnalyticsSnapshot, TrajectorySample


class Repository:
    def __init__(self, session: Optional[Session]):
        self.session = session

    # ------------------------------------------------------------- snapshots
    def add_snapshot(self, **kwargs: Any) -> None:
        if self.session is None:
            return
        try:
            kwargs.setdefault("timestamp", _utcnow())
            self.session.add(AnalyticsSnapshot(**kwargs))
            self.session.flush()
        except Exception as exc:
            logger.warning("could not persist analytics snapshot: %s", exc)
            self.session.rollback()

    def recent_snapshots(self, limit: int = 200) -> List[Dict[str, Any]]:
        if self.session is None:
            return []
        rows = (self.session.query(AnalyticsSnapshot)
                .order_by(AnalyticsSnapshot.timestamp.desc())
                .limit(limit).all())
        return [_snap_to_dict(r) for r in rows]

    # ----------------------------------------------------------------- alerts
    def add_alert(self, camera_id: str, alert_type: str, severity: str, message: str) -> None:
        if self.session is None:
            return
        try:
            self.session.add(AlertRecord(
                timestamp=_utcnow(), camera_id=camera_id,
                alert_type=alert_type, severity=severity, message=message))
            self.session.flush()
        except Exception as exc:
            logger.warning("could not persist alert: %s", exc)
            self.session.rollback()

    def recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.session is None:
            return []
        rows = (self.session.query(AlertRecord)
                .order_by(AlertRecord.timestamp.desc())
                .limit(limit).all())
        return [
            {"timestamp": r.timestamp.isoformat(), "camera_id": r.camera_id,
             "alert_type": r.alert_type, "severity": r.severity, "message": r.message}
            for r in rows
        ]

    # -------------------------------------------------------------- position
    def add_position(self, camera_id: str, track_id: int, x: float, y: float) -> None:
        if self.session is None:
            return
        try:
            self.session.add(TrajectorySample(
                timestamp=_utcnow(), camera_id=camera_id,
                track_id=track_id, x=x, y=y))
        except Exception as exc:
            logger.warning("could not persist trajectory sample: %s", exc)
            self.session.rollback()

    def commit(self) -> None:
        if self.session is None:
            return
        try:
            self.session.commit()
        except Exception as exc:
            logger.warning("commit failed: %s", exc)
            try:
                self.session.rollback()
            except Exception:
                pass


def _snap_to_dict(row: AnalyticsSnapshot) -> Dict[str, Any]:
    return {
        "timestamp": row.timestamp.isoformat(),
        "camera_id": row.camera_id,
        "zone_id": row.zone_id,
        "footfall_count": row.footfall_count,
        "entry_count": row.entry_count,
        "exit_count": row.exit_count,
        "queue_length": row.queue_length,
        "queue_growth_rate": row.queue_growth_rate,
        "predicted_queue_length": row.predicted_queue_length,
        "congestion_status": row.congestion_status,
        "shelf_id": row.shelf_id,
        "shelf_status": row.shelf_status,
        "alert_type": row.alert_type,
    }


def _utcnow():
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)