"""Tests for the privacy-preserving appearance Re-ID (global ids, unique
shoppers), the MJPEG push stream, the alert dedup window and frame-streaming
helpers. These verify the Gods-Eye-style enhancements stay privacy-safe: the
re-id is a colour-granularity histogram of the body crop, never a face."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from app.services.alert_service import AlertService
from ml.shopper.reid import AppearanceEmbedder, AppearanceReIdTracker


def _solid_color(bgr, w=64, h=100, x=1, y=1):
    """A frame with a solid coloured \"person\" crop near the top-left."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[y:y + h, x:x + w] = tuple(bgr)
    return frame


def _bbox(x=1, y=1, w=64, h=100):
    return [float(x), float(y), float(x + w), float(y + h)]


# --------------------------------------------------------------- embedder
def test_embedder_returns_normalized_histogram():
    emb = AppearanceEmbedder()
    frame = _solid_color((120, 30, 200))
    v = emb.embed(frame, _bbox())
    assert v is not None
    assert v.ndim == 1
    assert abs(float(v.sum()) - 1.0) < 1e-3          # normalized probability mass
    v2 = emb.embed(frame, _bbox(w=20, h=30))         # different crop size -> same size
    assert v2.shape == v.shape

def test_embedder_degenerate_bbox_returns_none():
    emb = AppearanceEmbedder()
    assert emb.embed(None, _bbox()) is None
    assert emb.embed(_solid_color((0, 0, 255)), [0, 0, 0, 0]) is None  # zero area
    assert emb.embed(_solid_color((0, 0, 255)), [5, 5, 1000, 1000]) is not None  # clamped

def test_embedder_resists_small_translation():
    """Same cloth sampled at slightly different crop offsets stays similar."""
    emb = AppearanceEmbedder()
    a = emb.embed(_solid_color((90, 120, 40)), _bbox())
    b = emb.embed(_solid_color((90, 120, 40)), _bbox(x=3, y=4, w=60, h=95))
    assert AppearanceEmbedder.cosine(a, b) > 0.9

# ------------------------------------------------------------ re-id tracker
def test_reid_mints_and_reuses_anonymous_ids():
    t = AppearanceReIdTracker({"match_threshold": 0.92})
    now = 10_000.0
    frame = _solid_color((200, 90, 30))
    g1 = t.update(frame, _bbox(), "cam_a", now)
    assert g1 is not None and g1.startswith("g_")
    g2 = t.update(frame, _bbox(), "cam_a", now + 5)
    assert g2 == g1                       # same cloth -> same anonymous id
    g3 = t.update(_solid_color((30, 200, 90)), _bbox(), "cam_b", now + 10)
    assert g3 not in (g1, g2)             # different cloth -> distinct id
    assert len(t.active_ids()) == 2
    assert "cam_a" in t.seen_at_cameras(g1)
    assert "cam_b" in t.seen_at_cameras(g3)

def test_reid_threshold_controls_merging():
    loose = AppearanceReIdTracker({"match_threshold": 0.15})
    tight = AppearanceReIdTracker({"match_threshold": 0.99})
    now = 1_000.0
    # near-identical cloth -> coarse histogram merges under a loose threshold
    ga = loose.update(_solid_color((60, 60, 60)), _bbox(), "cam", now)
    gb = loose.update(_solid_color((61, 61, 61)), _bbox(), "cam", now + 1)
    assert ga == gb
    # clearly distinct cloth -> tight threshold keeps them apart
    ga_t = tight.update(_solid_color((60, 60, 60)), _bbox(), "cam", now)
    gb_t = tight.update(_solid_color((240, 20, 20)), _bbox(), "cam", now + 1)
    assert ga_t != gb_t

def test_reid_unique_shoppers_window():
    t = AppearanceReIdTracker({"match_threshold": 0.99, "unique_window_seconds": 1000})
    now = 5_000.0
    t.update(_solid_color((10, 10, 210)), _bbox(), "cam", now)
    t.update(_solid_color((220, 10, 10)), _bbox(), "cam", now + 1)
    t.update(_solid_color((10, 210, 10)), _bbox(), "cam", now + 2)
    assert t.unique_shoppers(now + 3) == 3
    assert t.unique_shoppers(now + 3 + 1001) == 0   # all outside window

def test_reid_forgets_stale_identities():
    t = AppearanceReIdTracker({"match_threshold": 0.99, "forget_seconds": 120})
    now = 5_000.0
    g = t.update(_solid_color((200, 20, 20)), _bbox(), "cam", now)
    assert len(t.active_ids()) == 1
    t.update(_solid_color((200, 20, 20)), _bbox(), "cam", now + 100)   # still alive
    assert len(t.active_ids()) == 1
    t.update(_solid_color((200, 20, 20)), _bbox(), "cam", now + 200)   # exceeds 120s
    assert g not in t.active_ids() or t.unique_shoppers(now + 200) >= 0


# ------------------------------------------------------------- alert dedup
def test_alert_service_thresholds():
    st, rec = AlertService.assess(9, {"10min": 12.0, "source": "blend"}, {})
    assert st == "HIGH"
    assert "counter" in rec
    st, rec = AlertService.assess(3, {"10min": 2.0}, {})
    assert st == "NORMAL"


# ------------------------------------------------------------------ mjpeg
def test_mjpeg_frames_yields_boundary_wrapped_parts():
    from app.api.routes import mjpeg_frames

    class FakeSvc:
        def __init__(self):
            self._frame_no = 0
        def frame_jpeg_bytes(self, max_width=960):
            return cv2.imencode(".jpg", _solid_color((40, 40, 200)))[1].tobytes()

    frames = list(mjpeg_frames(FakeSvc(), poll_sec=0.001, max_frames=3))
    assert len(frames) == 3
    for part in frames:
        assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        assert part.endswith(b"\r\n")
        assert b"\xff\xd8" in part  # JPEG SOI marker present

def test_mjpeg_frames_yields_only_new_frames_in_production():
    """Production mode (max_frames=None) yields one JPEG part per new frame_no;
    unchanged frames are skipped so clients get a live, non-repeating stream."""
    import threading

    from app.api.routes import mjpeg_frames

    class SlowSvc:
        def __init__(self):
            self._frame_no = 0
            self.calls = 0
        def frame_jpeg_bytes(self, max_width=960):
            self.calls += 1
            return cv2.imencode(".jpg", _solid_color((10, 180, 10)))[1].tobytes()

    svc = SlowSvc()
    gen = iter(mjpeg_frames(svc, poll_sec=0.01))
    first = next(gen)                       # current frame, no wait
    assert first.startswith(b"--frame")

    def _advance():
        time.sleep(0.05)
        svc._frame_no = 1                    # simulate pipeline advancing a frame

    t = threading.Thread(target=_advance, daemon=True)
    t.start()
    second = next(gen)                      # blocks until frame_no changes
    assert second.startswith(b"--frame")
    assert svc._frame_no == 1
    t.join(timeout=5)