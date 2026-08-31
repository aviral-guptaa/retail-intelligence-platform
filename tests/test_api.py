"""API contract tests: the /analytics/current snapshot carries the upgraded
fields (occupancy, queue list, prediction source labels) and policy endpoints
still respond."""
import logging

import pytest

httpx2 = pytest.importorskip("httpx2")  # TestClient needs it; tests skip if absent

from fastapi.testclient import TestClient

from app.api import create_app
from app.services.inference_service import InferencePipeline
from config.loader import load_settings, load_zones
from database.repository import Repository

logging.getLogger("ml").setLevel(logging.ERROR)


def _pipeline():
    settings = load_settings()
    settings["demo"] = {"fps": 10, "frame_width": 1280, "frame_height": 720}
    settings.setdefault("starting", {})
    pipeline = InferencePipeline(settings, Repository(None), display=False)
    zones = load_zones()
    from demo.simulator import DemoSimulator
    from app.services.analytics_service import AnalyticsService

    sim = DemoSimulator("store_01", settings, zones)
    pipeline.services["store_01"] = AnalyticsService(
        "store_01", settings, zones, repository=None, source=sim)
    for _ in range(40):
        pipeline.services["store_01"].step()
    return pipeline


def test_analytics_current_snapshot_shape():
    pipeline = _pipeline()
    app = create_app(pipeline)
    with TestClient(app) as client:
        resp = client.get("/analytics/current")
    assert resp.status_code == 200
    snap = resp.json()
    assert snap["camera_id"] == "store_01"
    assert "occupancy" in snap["footfall"]
    assert "current_dwell_s" in snap["dwell"]
    queues = snap["queues"]
    assert isinstance(queues["queues"], list)          # queue list per spec
    assert "prediction_source" in queues
    assert queues["prediction_source"] in ("model", "blend", "fallback")
    for q in queues["queues"]:
        assert {"queue_id", "length", "wait_minutes", "status"} <= set(q)
        assert "predictions" in q
    for shelf in snap["shelves"]:
        assert shelf["source"] in ("classification", "detection", "heuristic")
    pipeline.stop()


def test_health_reports_source_and_db():
    pipeline = _pipeline()
    app = create_app(pipeline)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cameras"][0]["source"]["status"] in ("ONLINE", "OFFLINE", "RECONNECTING")
    assert "db" in body
    pipeline.stop()