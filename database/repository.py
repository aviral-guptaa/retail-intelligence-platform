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

from database.models import (AlertRecord, AnalyticsSnapshot, CameraRecord,
                             POSTransactionRecord, QueueEventRecord,
                             ShelfEventRecord, StoreRecord, TrajectorySample)


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

    # ---------------------------------------------------- store / camera
    def upsert_store(self, store_id: str, name: Optional[str] = None) -> None:
        if self.session is None:
            return
        try:
            row = (self.session.query(StoreRecord)
                   .filter(StoreRecord.store_id == store_id).first())
            if row is None:
                self.session.add(StoreRecord(store_id=store_id, name=name,
                                             created_at=_utcnow()))
            else:
                row.name = name or row.name
            self.session.flush()
        except Exception as exc:
            logger.warning("could not persist store: %s", exc)
            self.session.rollback()

    def upsert_camera(self, camera_id: str, store_id: Optional[str] = None,
                      source: Optional[str] = None,
                      state: str = "UNKNOWN") -> None:
        if self.session is None:
            return
        try:
            row = (self.session.query(CameraRecord)
                   .filter(CameraRecord.camera_id == camera_id).first())
            if row is None:
                self.session.add(CameraRecord(camera_id=camera_id,
                                              store_id=store_id or "store_01",
                                              source=source, state=state,
                                              created_at=_utcnow()))
            else:
                row.state = state or row.state
                if store_id:
                    row.store_id = store_id
            self.session.flush()
        except Exception as exc:
            logger.warning("could not persist camera: %s", exc)
            self.session.rollback()

    # ------------------------------------------------------------- events
    def recent_queue_events(self, camera_id: Optional[str] = None,
                            limit: int = 200) -> List[Dict[str, Any]]:
        if self.session is None:
            return []
        q = self.session.query(QueueEventRecord)
        if camera_id:
            q = q.filter(QueueEventRecord.camera_id == camera_id)
        rows = q.order_by(QueueEventRecord.timestamp.desc()).limit(limit).all()
        return [{"timestamp": r.timestamp.isoformat(), "camera_id": r.camera_id,
                 "queue_id": r.queue_id, "length": r.length,
                 "wait_minutes": r.wait_minutes,
                 "measured_wait_avg_minutes": r.measured_wait_avg_minutes}
                for r in rows]

    def recent_shelf_events(self, camera_id: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        if self.session is None:
            return []
        q = self.session.query(ShelfEventRecord)
        if camera_id:
            q = q.filter(ShelfEventRecord.camera_id == camera_id)
        rows = q.order_by(ShelfEventRecord.timestamp.desc()).limit(limit).all()
        return [{"timestamp": r.timestamp.isoformat(), "camera_id": r.camera_id,
                 "shelf_id": r.shelf_id, "status": r.status,
                 "item_count": r.item_count, "confidence": r.confidence,
                 "source": r.source} for r in rows]

    def recent_pos_transactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.session is None:
            return []
        rows = (self.session.query(POSTransactionRecord)
                .order_by(POSTransactionRecord.timestamp.desc())
                .limit(limit).all())
        return [{"transaction_id": r.transaction_id,
                 "timestamp": r.timestamp.isoformat(),
                 "store_id": r.store_id, "amount": r.amount,
                 "items": r.items_json} for r in rows]

    def add_pos_transaction(self, transaction_id: str, ts, store_id: str,
                            amount: float, items_json: Optional[str] = None) -> None:
        if self.session is None:
            return
        try:
            existing = (self.session.query(POSTransactionRecord)
                        .filter(POSTransactionRecord.transaction_id
                                == transaction_id).first())
            if existing is not None:
                return          # idempotent ingest
            self.session.add(POSTransactionRecord(
                transaction_id=transaction_id, timestamp=ts,
                store_id=store_id, amount=amount, items_json=items_json,
            ))
            self.session.flush()
        except Exception as exc:
            logger.warning("could not persist POS transaction: %s", exc)
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