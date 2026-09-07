"""Persistent, honest run results for the web dashboard.

Every finished/stopped analysis run is written to ``data/run_results/<run_id>.json``
(+ a heatmap PNG and a de-identified preview JPG). The metric detail pages keep
showing historical graphs and heatmaps AFTER the run has stopped by reading these
files back — no analytics are faked; everything is whatever the pipeline actually
computed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "data" / "run_results"

RUN_KINDS = ("entry_exit", "counter", "shelf")


def kind_label(kind: str) -> str:
    return {
        "entry_exit": "Entry / Exit",
        "counter": "Checkout queue",
        "shelf": "Shelf / Inventory",
    }.get(kind, kind or "entry_exit")


# ------------------------------------------------------------------ summary
def _fmean(vals: List[float]) -> float:
    vals = [v for v in vals if v]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def summary_from(cur: Dict[str, Any]) -> Dict[str, Any]:
    """Curated real minima from one live snapshot (the last frame's stats)."""
    f = cur.get("footfall", {}) or {}
    r = cur.get("reid", {}) or {}
    d = cur.get("dwell", {}) or {}
    q = cur.get("queues", {}) or {}
    preds = q.get("predictions", {}) or {}
    shelves = cur.get("shelves", []) or []
    sev = {s.get("status"): s for s in shelves if s.get("status")}

    def wait_avg(obj):
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and all(isinstance(v, (int, float)) for v in vals):
                return _fmean(vals)
        if isinstance(obj, (int, float)):
            return round(float(obj), 1)
        return 0.0

    measured = q.get("measured_wait_minutes") or {}
    meas = [m.get("avg_wait_minutes") for m in measured.values() if m and m.get("count")]
    return {
        "occupancy": int(f.get("occupancy", 0) or 0),
        "entries": int(f.get("total_entries", 0) or 0),
        "exits": int(f.get("total_exits", 0) or 0),
        "unique_shoppers": int(r.get("unique_shoppers", f.get("unique_shoppers", 0)) or 0),
        "reid_enabled": bool(r.get("enabled", False)),
        "active_identities": int(r.get("active_identities", 0) or 0),
        "avg_dwell_s": _fmean(list((d.get("avg_dwell_s") or {}).values())),
        "queue_total": int(q.get("total", 0) or 0),
        "wait_minutes": wait_avg(q.get("wait_minutes", 0)),
        "measured_wait_minutes": round(sum(meas) / len(meas), 1) if meas else 0.0,
        "prediction_source": q.get("prediction_source", "fallback"),
        "pred_5min": preds.get("5min"),
        "pred_10min": preds.get("10min"),
        "shelf_total": len(shelves),
        "shelf_full": sum(1 for s in shelves if s.get("status") in ("FULL",)),
        "shelf_low": sum(1 for s in shelves if s.get("status") in ("LOW", "LOW_STOCK")),
        "shelf_out": sum(1 for s in shelves if s.get("status") in ("OUT", "OUT_OF_STOCK")),
        "congestion_status": cur.get("congestion_status", "NORMAL"),
    }


def recommendation_from(cur: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One actionable recommendation, derived ONLY from real pipeline output."""
    if not cur:
        return None

    q = cur.get("queues", {}) or {}
    total = int(q.get("total", 0) or 0)
    preds = q.get("predictions", {}) or {}
    pred10 = preds.get("10min") or preds.get("predicted_queue_length_10min")
    pred5 = preds.get("5min") or preds.get("predicted_queue_length_5min")
    congestion = cur.get("congestion_status")
    rec_detail = q.get("recommendation_detail")
    shelves = cur.get("shelves", []) or []
    alerts = cur.get("alerts", {}) or {}

    def _queue(factors: Dict[str, Any] = None) -> Dict[str, Any]:
        text = ""
        if isinstance(rec_detail, str):
            text = rec_detail
        elif isinstance(rec_detail, dict):
            text = rec_detail.get("text") or rec_detail.get("reason") or ""
        wait = q.get("wait_minutes")
        if isinstance(wait, dict):
            wait = _fmean(list(wait.values()))
        detail_bits = {
            "current_queue": total,
            "predicted_5min": round(float(pred5), 1) if isinstance(pred5, (int, float)) else None,
            "predicted_10min": round(float(pred10), 1) if isinstance(pred10, (int, float)) else None,
            "estimated_wait_min": round(float(wait), 1) if isinstance(wait, (int, float)) else None,
        }
        if not text:
            if congestion == "HIGH":
                text = (f"Checkouts congested — queue at {total} shoppers"
                        f" (est {detail_bits['estimated_wait_min']} min).")
            elif congestion == "WARNING":
                text = (f"Queue building — {total} shoppers now,"
                        f" predicted {detail_bits['predicted_10min']} in 10 min.")
            else:
                text = f"Anticipate checkouts reaching {total} shoppers shortly."
        return {"action": "open_counter", "source": "queue", "text": text,
                "factors": detail_bits}

    def _restock(flag: str) -> Dict[str, Any]:
        ids = [s.get("shelf_id") or s.get("id") for s in shelves
               if s.get("status") in ({"out": ("OUT", "OUT_OF_STOCK"),
                                       "low": ("LOW", "LOW_STOCK")}[flag])]
        label = "OUT_OF_STOCK" if flag == "out" else "LOW_STOCK"
        return {"action": "restock", "source": "shelf",
                "text": f"{len(ids)} shelf/shelves {label}: {', '.join(ids)}.",
                "factors": {"shelves": ids}}

    if congestion in ("HIGH", "WARNING") or total >= 4 or (isinstance(pred10, (int, float)) and pred10 >= 4):
        return _queue()
    if any(s.get("status") in ("OUT", "OUT_OF_STOCK") for s in shelves):
        return _restock("out")
    if any(s.get("status") in ("LOW", "LOW_STOCK") for s in shelves):
        return _restock("low")
    high = [a for a in (alerts.get("active") or []) if a.get("severity") == "HIGH"]
    if high:
        a = high[0]
        return {"action": "inspect", "source": a.get("type", "alert"),
                "text": a.get("message", "High-severity alert active."),
                "factors": {"severity": "HIGH", "alert": a.get("type")}}
    return None


# ------------------------------------------------------------ persistence
def ensure_dir() -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR


def run_path(run_id: str) -> Path:
    return ensure_dir() / f"{run_id}.json"


def save_run(run_id: str, data: Dict[str, Any],
             heatmap_png: Optional[bytes] = None,
             frame_jpg: Optional[bytes] = None) -> bool:
    try:
        ensure_dir()
        run_path(run_id).write_text(json.dumps(data), encoding="utf-8")
        if heatmap_png:
            (RESULT_DIR / f"{run_id}.png").write_bytes(heatmap_png)
        if frame_jpg:
            (RESULT_DIR / f"{run_id}.jpg").write_bytes(frame_jpg)
        return True
    except OSError:
        return False


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    if not RESULT_DIR.is_dir():
        return []
    out = []
    for f in sorted(RESULT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime,
                    reverse=True)[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({
                "id": data.get("id"), "kind": data.get("kind"),
                "mode": data.get("mode"), "source": data.get("source"),
                "started_ts": data.get("started_ts"), "ended_ts": data.get("ended_ts"),
                "finished": data.get("finished", False),
                "frames": data.get("frames", 0),
                "summary": data.get("summary", {}),
                "recommendation": data.get("recommendation"),
            })
        except (OSError, ValueError):
            continue
    return out


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    p = run_path(run_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def media_path(run_id: str, ext: str) -> Optional[Path]:
    p = RESULT_DIR / f"{run_id}.{ext}"
    return p if p.is_file() else None