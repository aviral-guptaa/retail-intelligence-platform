"""FastAPI routes exposing analytics, alerts and configuration.

Keep these handlers thin: they read from the pipeline / analytics service and
serialise to JSON. No ML logic lives here.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.schemas.api import InventoryIngest, POSTransactionIngest
from app.services.inference_service import InferencePipeline


def _naive_ts(iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return datetime.now(timezone.utc).replace(tzinfo=None)


def _items_json(row: Dict[str, Any]) -> Optional[str]:
    items = row.get("items")
    if not items:
        return None
    try:
        return json.dumps(items)[:2048]
    except Exception:
        return None


def _get_pipeline(request: Request) -> InferencePipeline:
    return request.app.state.pipeline


def _first_camera(pipeline: InferencePipeline, camera_id: Optional[str]) -> str:
    cams = list(pipeline.services.keys())
    return camera_id if camera_id in cams else (cams[0] if cams else "store_01")


router = APIRouter()


@router.get("/cameras")
def cameras(request: Request) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    return pipeline.cameras()


@router.get("/cameras/{camera_id}/health")
def camera_health(request: Request, camera_id: str) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services.get(camera_id)
    if svc is None:
        raise HTTPException(404, f"Unknown camera '{camera_id}'")
    return svc.health()


@router.get("/system/performance")
def system_performance(request: Request) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    return pipeline.performance()


@router.get("/stores")
def stores(request: Request) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    if not pipeline.services:
        raise HTTPException(503, "No cameras configured")
    return pipeline.stores()


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    pipeline: InferencePipeline = _get_pipeline(request)
    writer = getattr(pipeline, "db_writer", None)
    return {
        "status": "ok",
        "time": time.time(),
        "cameras": [svc.health() for cid, svc in pipeline.services.items()],
        "db": {
            "persistence": writer is not None,
            "dropped_rows": getattr(writer, "dropped", 0),
        } if writer is not None else {"persistence": False},
    }


@router.get("/analytics/current")
def analytics_current(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    if not pipeline.services:
        raise HTTPException(503, "No cameras configured")
    return pipeline.single(_first_camera(pipeline, camera_id))


@router.get("/analytics/footfall")
def analytics_footfall(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return {
        "camera_id": svc.camera_id,
        "current": svc.footfall.snapshot(),
        "series": svc.footfall.series(minutes=30),
    }


@router.get("/analytics/dwell")
def analytics_dwell(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return {"camera_id": svc.camera_id, "avg_dwell_s": svc.dwell.avg_dwell(),
            "occupancy": svc.dwell.occupancy()}


@router.get("/analytics/queues")
def analytics_queues(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return svc.current()["queues"]


@router.get("/analytics/shelves")
def analytics_shelves(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return {"camera_id": svc.camera_id, "status_summary": svc.shelves.status_summary(),
            "shelves": svc.shelves.snapshot()}


@router.get("/analytics/heatmap")
def analytics_heatmap(request: Request, camera_id: Optional[str] = None) -> Response:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    img = svc.heatmap_image()
    try:
        import cv2

        ok, buf = cv2.imencode(".png", img)
        if ok:
            return Response(content=buf.tobytes(), media_type="image/png")
    except Exception:
        pass
    return Response(content=img.tobytes(), media_type="application/octet-stream")


def mjpeg_frames(svc, max_width: int = 960, poll_sec: float = 0.10, max_frames: Optional[int] = None):
    """Yield raw JPEG frames as multipart/x-mixed-replace chunks for MJPEG push.

    Polls the analytics service for the most recent analysed frame and streams
    boundary-wrapped JPEG parts (Gods-Eye style motion stream, but re-using the
    already-cached encode so this adds no per-frame cost beyond a memcpy).
    ``max_frames`` caps the stream for tests/clients that want a bounded body.
    """
    import base64

    sent = 0
    last_no = -1
    while svc is not None:
        if max_frames is not None and sent >= max_frames:
            break
        no = getattr(svc, "_frame_no", 0)
        if no != last_no or max_frames is not None:
            raw = svc.frame_jpeg_bytes(max_width=max_width)
            if raw is not None:
                last_no = no
                sent += 1
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + raw + b"\r\n")
        time.sleep(poll_sec)


@router.get("/video_stream")
def video_stream(request: Request, camera_id: Optional[str] = None,
                 max_frames: Optional[int] = None) -> Response:
    """MJPEG push stream of the live (de-identified) frame for this camera."""
    pipeline = _get_pipeline(request)
    if not pipeline.services:
        raise HTTPException(503, "No cameras configured")
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return StreamingResponse(
        mjpeg_frames(svc, max_frames=max_frames),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/analytics/planogram")
def analytics_planogram(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return {"camera_id": svc.camera_id, **svc.planogram_status}


@router.get("/analytics/history")
def analytics_history(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    if not pipeline.services:
        raise HTTPException(503, "No cameras configured")
    return pipeline.services[_first_camera(pipeline, camera_id)].historical()


@router.get("/analytics/report.csv")
def analytics_report_csv(request: Request, camera_id: Optional[str] = None) -> Response:
    """Download historical analytics as CSV (minute + hourly buckets)."""
    pipeline = _get_pipeline(request)
    if not pipeline.services:
        raise HTTPException(503, "No cameras configured")
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return Response(content=svc.report_csv(),
                    media_type="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename=analytics_{svc.camera_id}.csv"})


@router.get("/analytics/daily")
def analytics_daily(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return {"camera_id": svc.camera_id, "daily": svc.history.daily_summary(),
            "peak_hours": svc.history.peak_hours()}


@router.get("/alerts")
def alerts(request: Request, limit: int = 50) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    return {"alerts": pipeline.repo.recent_alerts(limit)}


@router.get("/alerts/active")
def alerts_active(request: Request, camera_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    snap = svc.alert_store.snapshot()
    return {"alerts": snap["active"], "counts": snap["by_severity"],
            "webhook_enabled": snap["webhook_enabled"]}


@router.get("/alerts/historical")
def alerts_historical(request: Request, camera_id: Optional[str] = None,
                      limit: int = 100, alert_type: Optional[str] = None) -> Dict[str, Any]:
    pipeline = _get_pipeline(request)
    svc = pipeline.services[_first_camera(pipeline, camera_id)]
    return {"camera_id": svc.camera_id,
            "alerts": svc.alert_store.historical(limit=limit, alert_type=alert_type)}


@router.get("/analytics/queues/events")
def queue_events(request: Request, camera_id: Optional[str] = None,
                 limit: int = 200) -> Dict[str, Any]:
    """Persisted per-change queue events (empty when running without a DB writer)."""
    pipeline = _get_pipeline(request)
    sid = _first_camera(pipeline, camera_id)
    return {"camera_id": sid,
            "events": pipeline.repo.recent_queue_events(sid, limit=limit)}


@router.get("/analytics/shelves/events")
def shelf_events(request: Request, camera_id: Optional[str] = None,
                 limit: int = 100) -> Dict[str, Any]:
    """Persisted committed shelf-status transitions (empty without a DB writer)."""
    pipeline = _get_pipeline(request)
    sid = _first_camera(pipeline, camera_id)
    return {"camera_id": sid,
            "events": pipeline.repo.recent_shelf_events(sid, limit=limit)}


@router.post("/integrations/pos/transactions")
def pos_ingest(request: Request, payload: POSTransactionIngest) -> Dict[str, Any]:
    """Generic POS/webhook ingest point (vendor-agnostic, adapter only).

    Validated by pydantic: transaction_id/timestamp required, amount must be
    a non-negative number. Returns a 422 with field errors otherwise.
    """
    rows = [t.model_dump() for t in payload.transactions]
    hub = _get_integrations(request)
    result = hub.ingest_batch(rows)
    persisted = _persist_pos_rows(request, rows)
    result["persisted"] = persisted
    return result


@router.get("/integrations/pos/transactions")
def pos_transactions(request: Request, limit: int = 100) -> Dict[str, Any]:
    """POS transactions that were persisted to the DB (empty when running
    without a database writer, e.g. the in-memory web dashboard)."""
    pipeline = _get_pipeline(request)
    return {"transactions": pipeline.repo.recent_pos_transactions(limit)}


def _persist_pos_rows(request: Request, rows: List[Dict[str, Any]]) -> int:
    """Best-effort persistence of ingested transactions (writer or repo).

    Returns how many rows were actually written (0 in the in-memory dashboard,
    which has no DB session - callers surface this honestly).
    """
    pipeline = _get_pipeline(request)
    writer = getattr(pipeline, "db_writer", None)
    n = len(rows)
    if writer is not None:
        for r in rows:
            writer.submit("pos_transaction",
                          transaction_id=r["transaction_id"],
                          timestamp=_naive_ts(r["timestamp"]),
                          store_id=r.get("store_id") or "store_01",
                          amount=r.get("amount"),
                          items_json=_items_json(r))
        return n
    repo = getattr(pipeline, "repo", None)
    if repo is not None and getattr(repo, "session", None) is not None:
        for r in rows:
            repo.add_pos_transaction(
                r["transaction_id"], _naive_ts(r["timestamp"]),
                r.get("store_id") or "store_01", r.get("amount"),
                _items_json(r))
        repo.commit()
        return n
    return 0


@router.get("/integrations/pos/conversion")
def pos_conversion(request: Request) -> Dict[str, Any]:
    """Footfall-to-transaction conversion (only when POS data exists)."""
    pipeline = _get_pipeline(request)
    entries = 0
    for svc in pipeline.services.values():
        entries += svc.footfall.snapshot().get("total_entries", 0)
    return _get_integrations(request).conversion_rate(entries)


@router.post("/integrations/erp/inventory")
def erp_ingest(request: Request, payload: InventoryIngest) -> Dict[str, Any]:
    """Generic ERP/webhook inventory snapshot ingest (adapter only).

    Validated by pydantic: sku required, quantity must be a non-negative int.
    """
    rows = [r.model_dump() for r in payload.inventory]
    return _get_integrations(request).ingest_inventory_batch(rows)


@router.get("/integrations/erp/inventory")
def erp_inventory(request: Request) -> Dict[str, Any]:
    return _get_integrations(request).inventory_stats()


@router.get("/integrations/status")
def integrations_status(request: Request) -> Dict[str, Any]:
    hub = _get_integrations(request)
    out = hub.status()
    out.update({"pos_stats": hub.pos_stats(), "inventory": hub.inventory_stats()})
    return out


def _get_integrations(request: Request):
    hub = getattr(request.app.state, "integrations", None)
    if hub is None:                      # apps that never mounted the hub
        from ml.integrations.pos import IntegrationHub
        hub = IntegrationHub()
        request.app.state.integrations = hub
    return hub


@router.get("/config/zones")
def get_zones(request: Request) -> Dict[str, Any]:
    from config.loader import load_zones

    return load_zones()


@router.post("/config/zones")
def post_zones(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Hot-update the "zones" section for a camera and rewire live modules."""
    from config.loader import load_zones, CONFIG_DIR

    camera_id = payload.get("camera_id") or _first_camera(_get_pipeline(request), None)
    updated = load_zones()
    camera_cfg = dict(updated.get(camera_id, {}))
    camera_cfg["zones"] = payload.get("zones", camera_cfg.get("zones", {}))
    updated[camera_id] = camera_cfg
    import json

    (CONFIG_DIR / "zones.json").write_text(json.dumps(updated, indent=2))
    svc = _get_pipeline(request).services.get(camera_id)
    if svc is not None:
        svc.reload_config(camera_cfg)
    return {"status": "ok", "camera_id": camera_id, "zones": updated[camera_id]}