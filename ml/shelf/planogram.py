"""Optional planogram compliance checking.

A planogram declares the expected category (and optionally grid position) of
products on each shelf. This module compares detected product positions inside
a shelf region against expectations and reports violations. Pure architectural
scaffolding for the prototype - the tight SKU-matching stage requires a product
detector with fine classes, which is out of scope for the initial demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from app.schemas.models import Detection
from ml.geometry import point_in_polygon


@dataclass
class PlanogramViolation:
    shelf_id: str
    kind: str            # MISSING_ITEM | WRONG_POSITION | COUNT_MISMATCH
    detail: str
    ts: float

    def to_dict(self) -> Dict[str, Any]:
        return {"shelf_id": self.shelf_id, "kind": self.kind,
                "detail": self.detail, "ts": self.ts}


class PlanogramChecker:
    def __init__(self, planogram: Dict[str, Dict[str, Any]]):
        self.planogram = planogram or {}

    def check(self, shelf_id: str, region, detections: List[Detection], now: float) -> List[PlanogramViolation]:
        expected = self.planogram.get(shelf_id)
        if not expected:
            return []
        violations: List[PlanogramViolation] = []
        products = [d for d in detections
                    if d.class_name in ("product", "object") and point_in_polygon(d.center, region)]
        expected_cols = int(expected.get("expected_columns", 3))
        expected_rows = int(expected.get("expected_rows", 4))
        expected_count = expected_cols * expected_rows
        if len(products) == 0 and expected_count > 0:
            violations.append(PlanogramViolation(
                shelf_id, "MISSING_ITEM", f"shelf {shelf_id} is empty against a {expected_count} unit planogram", now))
        elif len(products) < expected_count * 0.7:
            violations.append(PlanogramViolation(
                shelf_id, "COUNT_MISMATCH",
                f"expected ~{expected_count} units, detected {len(products)}", now))
        return violations