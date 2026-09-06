"""Generic integration adapters for POS / ERP.

Everything here is intentionally *adapter-scoped and format-agnostic*: the
endpoints accept whatever your POS or ERP pushes into them and store it as a
normalised, time-stamped record. There is NO hard-coded dependency on any
specific vendor. If an integration is not fed, the analytics report that
honestly ("not_configured"/"no data") instead of fabricating a number.

Metrics derived here - e.g. footfall-to-transaction conversion and ticket
totals - are ONLY reported once real transactions exist.
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


def _to_epoch(value: Any) -> float:
    """Accept either a numeric epoch (seconds) or an ISO-8601 timestamp."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    num = None
    try:
        num = float(text)
    except ValueError:
        pass
    if num is not None:
        return num
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        raise ValueError(f"unparseable timestamp: {text!r}") from None


class IntegrationHub:
    """Owns the POS transaction log and the (last-known) inventory snapshot."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None,
                 window: int = 5000):
        self.settings = settings or {}
        self._txs: Deque[Dict[str, Any]] = deque(maxlen=window)
        self._inventory: Dict[str, Dict[str, Any]] = {}
        self._last_inventory_ts: Optional[float] = None
        self._received = 0
        self._rejected = 0

    # --------------------------------------------------------------- POS
    def ingest_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a POS transaction payload and record it."""
        try:
            tid = str(payload.get("transaction_id") or payload.get("id") or
                      f"txn-{self._received + 1}")
            amount = float(payload.get("amount") or payload.get("total") or 0.0)
            raw_items = payload.get("items") or payload.get("item_count")
            items = (len(raw_items) if isinstance(raw_items, list)
                     else int(raw_items) if raw_items is not None else 1)
            method = str(payload.get("method") or payload.get("payment_method") or "unknown")
            ts = _to_epoch(payload.get("timestamp"))
        except (TypeError, ValueError):
            self._rejected += 1
            return {"ok": False, "error": "malformed payload"}
        self._received += 1
        self._txs.append({
            "transaction_id": tid,
            "amount": round(amount, 2),
            "items": max(items, 0),
            "method": method,
            "ts": ts,
        })
        return {"ok": True, "accepted": self._received, "rejected": self._rejected}

    def ingest_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest a validated list of transactions (pydantic already checked
        types at the API boundary; this only handles in-memory shape drift).
        Kept separate from :meth:`ingest_transaction` so the single-record path
        (tests / direct hub use) stays intact and battle-tested."""
        ok = bad = 0
        for row in rows:
            if isinstance(row, dict) and row.get("transaction_id"):
                out = self.ingest_transaction(row)
                ok += 1 if out.get("ok") else 0
                bad += 0 if out.get("ok") else 1
            else:
                bad += 1
        return {"ok": bad == 0, "accepted": ok, "rejected": bad,
                "stored": len(self._txs)}

    def ingest_inventory_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest a validated list of inventory rows."""
        return self.ingest_inventory({"inventory": rows})

    def pos_stats(self, hours: int = 1) -> Dict[str, Any]:
        cutoff = time.time() - hours * 3600
        recent = [t for t in self._txs if t["ts"] >= cutoff]
        if not recent:
            return {"available": False,
                    "message": "No POS transactions in window - POST /integrations/pos/transactions"}
        amounts = [t["amount"] for t in recent]
        return {
            "available": True,
            "window_hours": hours,
            "transactions": len(recent),
            "total_amount": round(sum(amounts), 2),
            "avg_ticket": round(sum(amounts) / len(amounts), 2),
            "items_sold": sum(t["items"] for t in recent),
        }

    def conversion_rate(self, footfall_entries: int) -> Dict[str, Any]:
        """footfall-to-transaction conversion, only when BOTH exist."""
        stats = self.pos_stats(hours=1)
        if not stats["available"]:
            return {"available": False,
                    "message": "No POS transactions - conversion unavailable"}
        if footfall_entries <= 0:
            return {"available": False,
                    "message": "No footfall entries measured in the window - conversion unavailable"}
        return {
            "available": True,
            "transactions": stats["transactions"],
            "footfall_entries": footfall_entries,
            "conversion_pct": round(stats["transactions"] / footfall_entries * 100.0, 2),
            "avg_ticket": stats["avg_ticket"],
        }

    # -------------------------------------------------------------- ERP
    def ingest_inventory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record a generic SKU/stock snapshot (adapter-only, no vendor logic)."""
        skus = payload.get("sku") or payload.get("inventory") or payload.get("items") or []
        updates = 0
        if isinstance(skus, list):
            for row in skus:
                sku = row.get("sku") or row.get("product_id")
                if not sku:
                    continue
                stock = row.get("stock")
                if stock is None:
                    stock = row.get("quantity")
                if stock is None:
                    continue
                self._inventory[str(sku)] = {
                    "sku": str(sku), "stock": int(stock),
                    "retail_price": row.get("retail_price"),
                    "updated_ts": time.time(),
                }
                updates += 1
        elif skus:
            return {"ok": False, "error": "inventory must be a list of {sku, stock}"}
        self._last_inventory_ts = time.time()
        return {"ok": updates > 0, "updated_skus": updates,
                "total_skus": len(self._inventory)}

    def inventory_stats(self) -> Dict[str, Any]:
        if not self._inventory:
            return {"available": False,
                    "message": "No inventory snapshot received - POST /integrations/erp/inventory"}
        stocks = [i["stock"] for i in self._inventory.values()]
        return {
            "available": True,
            "skus": len(self._inventory),
            "low_stock_skus": sum(1 for s in stocks if s <= 5),
            "out_of_stock_skus": sum(1 for s in stocks if s <= 0),
            "last_snapshot_ts": self._last_inventory_ts,
            "total_units": sum(stocks),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "pos_connected": self._received > 0,
            "pos_transactions_received": self._received,
            "erp_inventory_linked": len(self._inventory) > 0,
            "erp_skus_tracked": len(self._inventory),
        }