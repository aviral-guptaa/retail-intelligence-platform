"""Privacy-preserving cross-camera person re-identification (Re-ID).

Inspired by the "Gods-Eye" surveillance repo, but with the face/AI part
deliberately REMOVED: identity here is an **anonymous appearance embedding**
(a colour-granularity histogram of clothing/body), never a face. This lets the
same anonymous shopper be recognised as they move between camera views /
checkouts — improving footfall, occupancy and dwell consistency — without ever
identifying a person. No face detection, no face encodings, no names.

Design
------
1. ``AppearanceEmbedder`` crops a track's bounding box from the frame and turns
   it into a small normalized HSV histogram (rotation/gait invariant; tolerant
   of partial occlusion), which cannot reconstruct a face.
2. ``AppearanceReIdTracker`` keeps a rolling gallery of *anonymous global ids*
   (``g_7``), each with its latest embedding. Each new track is matched against
   the gallery by histogram cosine similarity; above ``match_threshold`` it is
   assigned the existing id (embedding updated via exponential moving average),
   otherwise a fresh id is minted. Unseen ids expire after ``forget_seconds``.

The output is purely a string global-id plus a count of unique shoppers in a
rolling window — the dashboard can show "7 unique shoppers today" with zero
identity information.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AppearanceEmbedder:
    """Turns a person crop into a normalized HSV colour-granule histogram."""

    def __init__(self, hue_bins: int = 12, sat_bins: int = 5, val_bins: int = 5):
        self.hue_bins = hue_bins
        self.sat_bins = sat_bins
        self.val_bins = val_bins
        self._channels = [0, 1, 2]
        self._hist_size = [hue_bins, sat_bins, val_bins]
        self._ranges = [0, 180, 0, 256, 0, 256]

    def embed(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Return a normalized appearance embedding for ``bbox`` inside ``frame``.

        ``bbox`` is [x1, y1, x2, y2]. Returns None when the crop is degenerate
        (out of bounds / zero area) so callers can skip silently.
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(w, int(round(bbox[2])))
        y2 = min(h, int(round(bbox[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        try:
            import cv2
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], self._channels, None, self._hist_size, self._ranges)
            cv2.normalize(hist, hist)
            return hist.flatten().astype(np.float32)
        except Exception as exc:  # pragma: no cover - defensive (no cv2, bad crop)
            logger.debug("appearance embed skipped: %s", exc)
            return None

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class AppearanceReIdTracker:
    """Rolling anonymous identity gallery for appearance-based Re-ID."""

    def __init__(self, settings: Dict):
        self.threshold = float(settings.get("match_threshold", 0.86))
        self.forget_seconds = float(settings.get("forget_seconds", 900))
        self.unique_window_seconds = float(settings.get("unique_window_seconds", 3600))
        self.embedder = AppearanceEmbedder(
            hue_bins=int(settings.get("hue_bins", 12)),
            sat_bins=int(settings.get("sat_bins", 5)),
            val_bins=int(settings.get("val_bins", 5)),
        )
        self._gallery: Dict[str, Dict] = {}     # gid -> {embedding, last_seen, camera_ids}
        self._encounters: Dict[str, float] = {}  # gid -> first_seen_ts (for unique count)
        self._next_id = itertools.count(1)
        self._alpha = float(settings.get("embed_alpha", 0.5))

    # -------------------------------------------------------------- match
    def update(self, frame: np.ndarray, bbox: List[float],
               camera_id: str, now: float = None) -> Optional[str]:
        """Return the anonymous global id for this track (matching or minting)."""
        now = now or time.time()
        emb = self.embedder.embed(frame, bbox)
        if emb is None:
            return None
        gid = self.match(emb)
        if gid is None:
            gid = f"g_{next(self._next_id)}"
            self._gallery[gid] = {"embedding": emb.copy(),
                                  "last_seen": now, "camera_ids": {camera_id}}
            self._encounters[gid] = now
            logger.debug("reid mint %s (camera %s)", gid, camera_id)
        else:
            # EMA-update the stored embedding towards the freshest sighting.
            stored = self._gallery[gid]["embedding"]
            self._gallery[gid]["embedding"] = (1.0 - self._alpha) * stored + self._alpha * emb
            self._gallery[gid]["last_seen"] = now
            self._gallery[gid]["camera_ids"].add(camera_id)
        self._expire(now)
        return gid

    def match(self, embedding: np.ndarray) -> Optional[str]:
        """Best gallery match above threshold, or None."""
        best, best_sim = None, -1.0
        for gid, entry in self._gallery.items():
            sim = self.embedder.cosine(embedding, entry["embedding"])
            if sim > best_sim:
                best, best_sim = gid, sim
        return best if best_sim >= self.threshold else None

    # ------------------------------------------------------------- metrics
    def camera_ids(self, gid: str) -> List[str]:
        entry = self._gallery.get(gid)
        return sorted(entry["camera_ids"]) if entry else []

    def unique_shoppers(self, now: float = None) -> int:
        """Number of distinct anonymised shoppers seen in the rolling window."""
        now = now or time.time()
        return sum(1 for t in self._encounters.values() if now - t <= self.unique_window_seconds)

    def active_ids(self) -> List[str]:
        return sorted(self._gallery.keys())

    def seen_at_cameras(self, gid: str) -> List[str]:
        return self.camera_ids(gid)

    # --------------------------------------------------------------- expiry
    def _expire(self, now: float) -> None:
        stale = [gid for gid, e in self._gallery.items()
                 if now - e["last_seen"] > self.forget_seconds]
        for gid in stale:
            self._gallery.pop(gid, None)
            self._encounters.pop(gid, None)

    def reset(self) -> None:
        self._gallery.clear()
        self._encounters.clear()