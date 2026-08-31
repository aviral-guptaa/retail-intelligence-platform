"""Shared data structures used across the ML, services and API layers.

These are deliberately kept dependency-light (plain Python / numpy) so the ML
modules stay importable and testable in isolation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class Detection:
    """A single object detected in a frame (optionally carrying a track id)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    ts: float = field(default_factory=time.time)
    track_id: Optional[int] = None

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0])

    def bbox(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> Dict:
        cx, cy = self.center
        return {
            "track_id": self.track_id,
            "bbox": [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
            "center": [round(float(cx), 1), round(float(cy), 1)],
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "ts": self.ts,
        }

    @classmethod
    def from_track(cls, track: "Track") -> "Detection":
        return cls(
            x1=track.x1, y1=track.y1, x2=track.x2, y2=track.y2,
            confidence=track.confidence, class_id=track.class_id,
            class_name=track.class_name, ts=track.ts, track_id=track.id,
        )


@dataclass
class Track(Detection):
    """A persistently tracked person with a temporary, anonymous id."""

    id: int = 0
    hit_streak: int = 0
    missed_frames: int = 0
    positions: Deque[np.ndarray] = field(default_factory=lambda: __import__("collections").deque(maxlen=1200))

    def record_position(self, maxlen: int) -> None:
        if self.positions.maxlen != maxlen:
            self.positions = __import__("collections").deque(self.positions, maxlen=maxlen)
        self.positions.append(self.center)


@dataclass
class ZoneEvent:
    """Immutable event emitted by the analytics modules."""

    ts: float
    camera_id: str
    event_type: str            # entry | exit | zone_enter | zone_exit | queue_change | alert
    value: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "ts": self.ts,
            "camera_id": self.camera_id,
            "event_type": self.event_type,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass
class ShelfSnapshot:
    shelf_id: str
    status: str                  # FULL | LOW_STOCK | OUT_OF_STOCK
    item_count: int
    expected_count: int
    confidence: float
    ts: float = field(default_factory=time.time)
    source: str = "heuristic"    # classification | detection | heuristic

    def to_dict(self) -> Dict:
        return {
            "shelf_id": self.shelf_id,
            "status": self.status,
            "item_count": self.item_count,
            "expected_count": self.expected_count,
            "confidence": round(self.confidence, 3),
            "ts": self.ts,
            "source": self.source,
        }