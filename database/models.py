"""Persisted analytics: SQLAlchemy models (SQLite by default, PostgreSQL ready)."""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class AnalyticsSnapshot(Base):
    """One row per aggregate time bucket, mirroring the section-11 schema."""

    __tablename__ = "analytics_snapshots"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    camera_id = Column(String(64), index=True, nullable=False)
    zone_id = Column(String(64), index=True, default=None)
    footfall_count = Column(Integer, default=0)
    entry_count = Column(Integer, default=0)
    exit_count = Column(Integer, default=0)
    queue_length = Column(Integer, default=0)
    queue_growth_rate = Column(Float, default=0.0)
    predicted_queue_length = Column(Float, default=0.0)
    congestion_status = Column(String(16), default="NORMAL")
    shelf_id = Column(String(64), index=True, default=None)
    shelf_status = Column(String(16), default=None)
    alert_type = Column(String(32), default=None)


class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    camera_id = Column(String(64), nullable=False)
    alert_type = Column(String(32), nullable=False)      # congestion / shelf / queue
    severity = Column(String(16), nullable=False)        # INFO / WARNING / HIGH
    message = Column(String(512), nullable=False)


class TrajectorySample(Base):
    """Individual anonymised track positions used for heatmap reproducibility."""

    __tablename__ = "trajectory_samples"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    camera_id = Column(String(64), nullable=False)
    track_id = Column(Integer, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)


class StoreRecord(Base):
    """A retail store the platform monitors (one platform can serve many)."""

    __tablename__ = "stores"

    id = Column(Integer, primary_key=True)
    store_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), default=None)
    created_at = Column(DateTime, default=None)


class CameraRecord(Base):
    """A camera registered against a store, with its latest health state."""

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    camera_id = Column(String(64), unique=True, index=True, nullable=False)
    store_id = Column(String(64), index=True, default=None)
    source = Column(String(256), default=None)      # filename | RTSP URL | webcam index
    state = Column(String(16), default="UNKNOWN")   # ONLINE | OFFLINE | PROCESSING | ERROR
    last_online_at = Column(DateTime, default=None)
    created_at = Column(DateTime, default=None)


class QueueEventRecord(Base):
    """Per-checkout queue length / wait-time event (the base for forecasts)."""

    __tablename__ = "queue_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    camera_id = Column(String(64), index=True, nullable=False)
    queue_id = Column(String(64), default=None)
    length = Column(Float, default=0.0)
    wait_minutes = Column(Float, default=None)
    measured_wait_avg_minutes = Column(Float, default=None)


class ShelfEventRecord(Base):
    """Committed shelf-status transitions (FULL/LOW_STOCK/OUT_OF_STOCK)."""

    __tablename__ = "shelf_events"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    camera_id = Column(String(64), index=True, nullable=False)
    shelf_id = Column(String(64), index=True, nullable=False)
    status = Column(String(16), nullable=False)
    item_count = Column(Integer, default=0)
    confidence = Column(Float, default=None)
    source = Column(String(32), default="heuristic")


class POSTransactionRecord(Base):
    """POS transaction ingested from a retailer's webhook (adapter-only)."""

    __tablename__ = "pos_transactions"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(128), unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    store_id = Column(String(64), index=True, default=None)
    amount = Column(Float, default=None)
    items_json = Column(String(2048), default=None)


def build_session(database_url: str):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return engine, session_factory()


def build_session_factory(database_url: str):
    """Return a callable that yields a fresh Session (for background writers)."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory