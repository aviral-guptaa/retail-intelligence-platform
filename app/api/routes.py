"""FastAPI routes exposing analytics, alerts and configuration.

Keep these handlers thin: they read from the pipeline / analytics service and
serialise to JSON. No ML logic lives here.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.services.inference_service import InferencePipeline


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


@router.post("/integrations/pos/transactions")
def pos_ingest(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generic POS/webhook ingest point (vendor-agnostic, adapter only)."""
    return _get_integrations(request).ingest_transaction(payload)


@router.get("/integrations/pos/conversion")
def pos_conversion(request: Request) -> Dict[str, Any]:
    """Footfall-to-transaction conversion (only when POS data exists)."""
    pipeline = _get_pipeline(request)
    entries = 0
    for svc in pipeline.services.values():
        entries += svc.footfall.snapshot().get("total_entries", 0)
    return _get_integrations(request).conversion_rate(entries)


@router.post("/integrations/erp/inventory")
def erp_ingest(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generic ERP/webhook inventory snapshot ingest (adapter only)."""
    return _get_integrations(request).ingest_inventory(payload)


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