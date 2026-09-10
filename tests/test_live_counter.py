"""Tests for the live webcam people counter (webserver/live_counter.py)
and its /api/livecount endpoints.

The counter itself is tested with a fake camera + mocked detector (no real
webcam, no YOLO inference). API tests use the standard in-process TestClient.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class FakeCap:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.closed = False

    def isOpened(self):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        self.i += 1
        return True, f

    def release(self):
        self.closed = True


class UnopenedCap(FakeCap):
    def isOpened(self):
        return False


def _frames(n=6):
    return [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(n)]


def _wait_until(pred, timeout=3.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_live_counter_counts_entries_exits_peak():
    from webserver.live_counter import LivePeopleCounter

    c = LivePeopleCounter(min_interval=0.0)
    c._model = object()
    c._calls = iter([(1, []), (3, []), (3, []), (2, []), (1, []), (1, [])])
    c.detect = lambda frame: next(c._calls, (0, []))
    cap = FakeCap(_frames())
    with patch("cv2.VideoCapture", return_value=cap):
        assert c.start() is True
        assert _wait_until(lambda: cap.i >= len(cap.frames))
        c.stop()
    assert cap.closed
    st = c.status()
    assert st["running"] is False
    assert st["count"] == 1
    assert st["peak"] == 3
    assert st["entries"] == 3  # 0->1 (+1), 1->3 (+2)
    assert st["exits"] == 2  # 3->2 (-1), 2->1 (-1)
    assert len(st["history"]) == len(cap.frames)
    assert st["model"] is True


def test_live_counter_handles_camera_unavailable():
    from webserver.live_counter import LivePeopleCounter

    c = LivePeopleCounter(min_interval=0.0)
    c._model = object()
    with patch("cv2.VideoCapture", return_value=UnopenedCap(_frames(2))):
        assert c.start() is False
    assert "permission" in (c.error or "").lower()


def test_live_counter_stream_frames():
    from webserver.live_counter import LivePeopleCounter

    c = LivePeopleCounter()
    with c._lock:
        c._frame_jpeg = b"FAKEJPEGBYTES"
    out = b"".join(c.stream_frames(max_frames=2))
    assert b"--frame" in out and b"FAKEJPEGBYTES" in out
    assert out.count(b"--frame") == 2


def test_live_counter_start_accepts_camera_index():
    from webserver.live_counter import LivePeopleCounter

    c = LivePeopleCounter(min_interval=0.0)
    c._model = object()
    c._calls = iter([(1, [])])
    c.detect = lambda frame: next(c._calls, (0, []))
    cap = FakeCap(_frames(n=1))
    with patch("cv2.VideoCapture", return_value=cap) as vc:
        assert c.start(camera_index=1) is True
        assert _wait_until(lambda: cap.i >= len(cap.frames))
        c.stop()
    assert vc.call_args.args[0] == 1
    assert c.status()["camera_index"] == 1


def test_live_counter_devices_probe():
    from webserver.live_counter import LivePeopleCounter

    class Cap:
        def __init__(self, index):
            self.index = index

        def isOpened(self):
            return self.index < 3

        def get(self, prop):
            return {3: 1280, 4: 720}.get(prop, 0)  # CAP_PROP_FRAME_WIDTH/HEIGHT

        def release(self):
            pass

    c = LivePeopleCounter()
    with patch("cv2.VideoCapture", side_effect=lambda i: Cap(i)):
        devs = c.devices(probe=5)
    assert [d["index"] for d in devs] == [0, 1, 2]
    assert devs[0]["name"].startswith("Camera 0")
    assert "1280x720" in devs[2]["resolution"]


def _client():
    from fastapi.testclient import TestClient  # noqa: PLC0415
    from webserver.app import create_web_app  # noqa: PLC0415
    from config.loader import load_settings  # noqa: PLC0415

    s = load_settings()
    s["demo"]["duration_seconds"] = 15
    s["demo"]["fps"] = 25
    app = create_web_app(s)
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    try:
        return _client()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"web server deps unavailable: {exc}")


def test_livecount_endpoints_idle(client):
    st = client.get("/api/livecount/status").json()
    assert st["running"] is False
    assert "count" in st and "peak" in st and "history" in st
    assert client.get("/api/livecount/stream").status_code == 404
    assert client.post("/api/livecount/stop").json()["ok"] is True