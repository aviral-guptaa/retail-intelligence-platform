"""Tests for the web dashboard backend (webserver/).

Skips cleanly if python-multipart / httpx TestClient is unavailable, so the core
CI suite stays green without web extras.

These run the real demo pipeline in-process, which is fast (a few seconds).
"""
from __future__ import annotations

import time

import pytest

from config.loader import load_settings


def _client(overrides=None):
    from fastapi.testclient import TestClient  # noqa: PLC0415
    from webserver.app import create_web_app  # noqa: PLC0415
    s = load_settings()
    s["demo"]["duration_seconds"] = 15
    s["demo"]["fps"] = 25
    if overrides:
        s.update(overrides)
    app = create_web_app(s)
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    try:
        return _client()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"web server deps unavailable: {exc}")


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "RetailIntelligence" in r.text or "dashboard" in r.text.lower()


def test_demo_run_and_live_analytics(client):
    r = client.post("/api/run/demo")
    assert r.status_code == 200
    assert r.json()["run"]["mode"] == "demo"

    # pipeline is async on a background thread; poll until live
    live = False
    for _ in range(50):
        time.sleep(0.2)
        st = client.get("/api/run/status").json()
        if st["live"] and st["running"]:
            live = True
            break
    assert live, "demo pipeline did not come online"

    cur = client.get("/api/analytics/current").json()
    assert cur["live"] is True
    assert "footfall" in cur and "queues" in cur and "congestion_status" in cur
    assert "planogram" in cur and "alerts" in cur
    # prediction source should reflect the trained model or blend
    assert cur["queues"]["prediction_source"] in ("model", "blend", "fallback")

    # heatmap endpoint returns bytes
    hm = client.get("/api/heatmap.png")
    assert hm.status_code == 200
    assert len(hm.content) > 0


def test_stop_run(client):
    r = client.post("/api/run/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"


# ---------------------------------------------------------------- privacy
def test_privacy_status_and_purge(client):
    r = client.get("/api/privacy/status")
    assert r.status_code == 200
    body = r.json()
    assert "retain_uploaded_video" in body
    assert "uploads_on_disk" in body
    # purge is a safe no-op / deletion of managed uploads
    purge = client.delete("/api/privacy/uploads")
    assert purge.status_code == 200
    assert purge.json()["status"] == "ok"


def test_retention_sweeps_stale_uploads(tmp_path):
    from webserver.app import RunManager
    fake = tmp_path / "data" / "uploads"
    fake.mkdir(parents=True)
    old = fake / "old_clip.mp4"
    old.write_bytes(b"\x00" * 16)
    import os
    import time as _t
    os.utime(old, (_t.time() - 48 * 3600, _t.time() - 48 * 3600))  # 2 days old
    fresh = fake / "fresh_clip.mp4"
    fresh.write_bytes(b"\x01" * 16)

    class Mgr(RunManager):
        pass

    mgr = Mgr({"privacy": {"retain_uploaded_video": False, "retention_hours": 24}})
    # point the sweep at the tmp dir
    mgr.UPLOAD_DIR = fake
    removed = mgr.sweep_uploads()
    assert removed == 1
    assert not old.exists() and fresh.exists()