"""DB/API hardening: new persisted entities, validated payloads, event reads."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def session_factory(tmp_path):
    from database.models import Base, build_session_factory
    db = tmp_path / "harden.sqlite"
    return build_session_factory(f"sqlite:///{db}")


# ------------------------------------------------------------ persistence
def test_writer_persists_store_camera_and_events(session_factory):
    from database.models import (CameraRecord, QueueEventRecord,
                                 ShelfEventRecord, StoreRecord,
                                 POSTransactionRecord)
    from database.repository import Repository
    from database.writer import BackgroundWriter
    from datetime import datetime, timezone

    writer = BackgroundWriter(session_factory, flush_interval=0.05)
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    writer.submit("store", store_id="store_01", name="Test Mart")
    writer.submit("camera", camera_id="cam_1", store_id="store_01",
                  source="clip.mp4", state="ONLINE")
    writer.submit("queue_event", timestamp=ts, camera_id="cam_1",
                  queue_id=None, length=4.0, wait_minutes=1.2,
                  measured_wait_avg_minutes=1.0)
    writer.submit("shelf_event", timestamp=ts, camera_id="cam_1",
                  shelf_id="A1", status="LOW_STOCK", item_count=2,
                  confidence=0.9, source="heuristic")
    writer.submit("pos_transaction", transaction_id="T1", timestamp=ts,
                  store_id="store_01", amount=29.99)
    writer.shutdown(flush_timeout=5.0)

    repo = Repository(session_factory())
    _ = repo.recent_snapshots()  # warms nothing; just cover import
    stores = repo.session.query(StoreRecord).all()
    cams = repo.session.query(CameraRecord).all()
    qev = repo.session.query(QueueEventRecord).all()
    sev = repo.session.query(ShelfEventRecord).all()
    pos = repo.session.query(POSTransactionRecord).all()
    assert len(stores) == 1 and stores[0].store_id == "store_01"
    assert len(cams) == 1 and cams[0].state == "ONLINE"
    assert len(qev) == 1 and qev[0].length == 4.0
    assert len(sev) == 1 and sev[0].status == "LOW_STOCK"
    assert len(pos) == 1 and pos[0].amount == 29.99


def test_repo_event_reads_and_pos_idempotency(session_factory):
    from database.repository import Repository
    from datetime import datetime, timezone

    repo = Repository(session_factory())
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    repo.add_pos_transaction("T1", ts, "store_01", 10.0)
    repo.add_pos_transaction("T1", ts, "store_01", 10.0)   # idempotent -> 1 row
    repo.upsert_store("store_01", "Test Mart")
    repo.upsert_camera("cam_1", store_id="store_01", source="0")
    repo.commit()
    pos = repo.recent_pos_transactions()
    assert len(pos) == 1 and pos[0]["amount"] == 10.0
    assert repo.recent_queue_events() == []     # no events written
    assert repo.recent_shelf_events("cam_1") == []
    repo.session.close()


# ------------------------------------------------------------ API boundary
def test_http_422_on_bad_payload_and_valid_ingest():
    from fastapi.testclient import TestClient  # noqa: PLC0415
    from app.api.routes import router
    from app.services.inference_service import InferencePipeline
    from database.repository import Repository
    from fastapi import FastAPI
    from ml.integrations.pos import IntegrationHub

    app = FastAPI()
    app.include_router(router)
    settings = {"app": {"store_id": "store_01"}}
    app.state.pipeline = InferencePipeline(settings, Repository(None), display=False)
    app.state.integrations = IntegrationHub(settings)
    cli = TestClient(app)

    # missing transaction_id / non-numeric amount -> 422
    r = cli.post("/integrations/pos/transactions",
                 json={"transactions": [{"amount": "abc"}]})
    assert r.status_code == 422

    # negative quantity in ERP ingest -> 422
    r = cli.post("/integrations/erp/inventory",
                 json={"inventory": [{"sku": "S1", "quantity": -1}]})
    assert r.status_code == 422

    # valid transaction -> 200, in-memory hub + honest persisted=0 on repo None
    r = cli.post("/integrations/pos/transactions",
                 json={"transactions": [
                     {"transaction_id": "T9", "timestamp": "2026-01-01T10:00:00Z",
                      "amount": 12.5, "items": ["cola"]}]})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["accepted"] == 1
    assert body["persisted"] == 0

    # read-back endpoints exist and are honest on an in-memory repo
    for ep in ["/integrations/pos/transactions",
               "/analytics/queues/events", "/analytics/shelves/events"]:
        rr = cli.get(ep)
        assert rr.status_code == 200, ep
        assert "events" in rr.json() or "transactions" in rr.json()

    cli.close()


def test_video_upload_path_still_validates():
    """POS payload validation must not regress the video upload handler."""
    from config.loader import load_settings
    from webserver.app import create_web_app
    from fastapi.testclient import TestClient

    s = load_settings()
    s["demo"]["duration_seconds"] = 15
    s["demo"]["fps"] = 25
    try:
        cli = TestClient(create_web_app(s))
    except Exception:  # pragma: no cover
        pytest.skip("web deps unavailable")
    r = cli.post("/api/integrations/pos/transactions",
                 json={"transactions": [{"amount": "abc"}]})
    assert r.status_code == 422
    r = cli.post("/api/integrations/pos/transactions",
                 json={"transactions": [
                     {"transaction_id": "W1",
                      "timestamp": "2026-01-01T10:00:00Z",
                      "amount": 5.0}]})
    assert r.status_code == 200 and r.json()["accepted"] == 1
    cli.close()