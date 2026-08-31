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