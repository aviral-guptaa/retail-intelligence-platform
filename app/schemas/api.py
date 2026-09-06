"""Pydantic request/response payloads for API boundary validation.

Kept separate from the dependency-light schemas in :mod:`models` so ML modules
stay free of pydantic; FastAPI routes validate against these and hand plain
dicts to the integration hub.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class POSTransaction(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0)
    store_id: Optional[str] = None
    items: Optional[List[str]] = None


class POSTransactionIngest(BaseModel):
    transactions: List[POSTransaction] = Field(..., min_length=1)


class InventoryRow(BaseModel):
    sku: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=0)
    location: Optional[str] = None
    store_id: Optional[str] = None


class InventoryIngest(BaseModel):
    inventory: List[InventoryRow] = Field(..., min_length=1)