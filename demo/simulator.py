"""Synthetic retail demo.

Produces frames that look like a store floor plus ground-truth detections for
people and products. This lets the entire pipeline (tracking, entry/exit,
dwell, heatmap, queue, shelf) run end-to-end with no camera, model download or
GPU - ideal for an SIH prototyping/demo loop and for CI tests.

Scripted behaviours:
  - shoppers spawn above the entrance virtual line and walk through zones;
  - some follow a checkout/queue script and stand in a queue zone for a while;
  - bursts of shoppers arrive periodically to exercise congestion prediction;
  - shelf products are drawn as dots and deplete over time (FULL -> LOW -> OUT),
    with random restocks, to drive the shelf classifier.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.schemas.models import Detection
from config.loader import load_zones
from ml.geometry import dist, point_in_polygon

PERSON_CLASS = 0
PRODUCT_CLASS = 542  # @datasets category placeholder; products are 'object' in demo


def _rand_point_in_polygon(poly, rng: random.Random) -> Tuple[float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    for _ in range(200):
        x = rng.uniform(min(xs), max(xs))
        y = rng.uniform(min(ys), max(ys))
        if point_in_polygon([x, y], poly):
            return x, y
    return (sum(xs) / len(xs), sum(ys) / len(ys))


class _Shopper:
    __slots__ = ("id", "x", "y", "script", "stage", "stage_holds", "hold_until",
                 "speed", "gone_timer", "rng")

    def __init__(self, shopper_id: int, x: float, y: float, script: List[Tuple[Tuple[float, float], float]],
                 speed: float, rng: random.Random):
        self.id = shopper_id
        self.x, self.y = x, y
        self.script = [(np.array(p, dtype=float), hold) for p, hold in script]
        self.stage = 0
        self.stage_holds = 0.0
        self.hold_until = 0.0
        self.speed = speed
        self.gone_timer = 0.0
        self.rng = rng

    def update(self, dt: float, now: float) -> bool:
        """Advance by dt seconds; return True while the shopper is still alive."""
        if self.stage >= len(self.script):
            self.gone_timer += dt
            return self.gone_timer < 1.5   # linger just above the line then vanish
        target, hold = self.script[self.stage]
        if self.hold_until > now:
            # Standing in a queue: jitter a few pixels while waiting.
            self.x += self.rng.uniform(-2.5, 2.5)
            self.y += self.rng.uniform(-2.5, 2.5)
            return True
        dx, dy = target - np.array([self.x, self.y])
        step = self.speed * dt
        if dist([self.x, self.y], target) <= step:
            self.x, self.y = float(target[0]), float(target[1])
            self.hold_until = now + hold
            self.stage += 1
        else:
            norm = (dx * dx + dy * dy) ** 0.5
            self.x += dx / norm * step
            self.y += dy / norm * step
        return True

    @property
    def center(self) -> Tuple[float, float]:
        return self.x, self.y


class DemoSimulator:
    """Emulates a camera + detector. Call next_frame() to get (frame, detections)."""

    def __init__(self, camera_id: str, settings: Dict, zones_config: Dict):
        self.camera_id = camera_id
        self.settings = settings
        demo = settings.get("demo", {})
        self.width = int(demo.get("frame_width", 1280))
        self.height = int(demo.get("frame_height", 720))
        self.fps = float(demo.get("fps", 10))
        self.max_active = int(demo.get("max_active_shoppers", 22))
        self.spawn_interval = float(demo.get("spawn_interval_seconds", 1.6))
        self.duration = float(demo.get("duration_seconds", 600))
        self.rng = random.Random(42)
        self.camera_cfg = zones_config.get(camera_id, {})
        self.line = self.camera_cfg.get("entrance_line")
        zones = self.camera_cfg.get("zones", {})
        shelves = self.camera_cfg.get("shelves", {})
        self.zone_polys = {zid: [np.array(p, dtype=float) for p in z["polygon"]] for zid, z in zones.items()}
        self.shelf_regions = {
            sid: [np.array(p, dtype=float) for p in s["region"]["polygon"]]
            for sid, s in shelves.items()
        }
        self._line_a = np.array(self.line["start"], dtype=float)
        self._line_b = np.array(self.line["end"], dtype=float)
        self.shopper_id_counter = 1
        self.shoppers: Dict[int, _Shopper] = {}
        self.time = 0.0
        self._spawn_timer = 1.0          # first shopper arrives quickly
        self._next_burst = 12.0
        self._shelf_products: Dict[str, List[np.ndarray]] = {}
        self._init_products(shelves)
        self._restock_timer = 0.0
        self._drain_target: Optional[str] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_dets: List[Detection] = []

    # ------------------------------------------------------------------ setup
    def _init_products(self, shelves) -> None:
        for sid, spec in shelves.items():
            region = spec["region"]["polygon"]
            expected = int(spec.get("expected_item_count", 12))
            self._grid_products(sid, region, expected)

    # ------------------------------------------------------------------ state
    @property
    def finished(self) -> bool:
        # duration in seconds; -1 = run until stopped. Active shoppers are
        # transient demo state, so no need to drain them before finishing.
        return 0.0 < self.duration <= self.time

    # ----------------------------------------------------------------- detect
    def _spawn(self, now: float) -> None:
        if self.duration > 0 and now >= self.duration:
            return
        if len(self.shoppers) >= self.max_active:
            return
        line_x = (self._line_a[0] + self._line_b[0]) / 2.0
        x = self.rng.uniform(self._line_a[0] + 20, self._line_b[0] - 20)
        y = self._line_a[1] - 40.0
        entrance = (x, min(self._line_a[1], self._line_b[1]) + 12)
        by_pos = (line_x, self._line_a[1] - 60)
        exit_pt = (x, self._line_a[1] - 50)

        bev = _rand_point_in_polygon(self.zone_polys.get("beverages", []), self.rng)
        snack = _rand_point_in_polygon(self.zone_polys.get("snacks", []), self.rng)
        promo = _rand_point_in_polygon(self.zone_polys.get("promotional", []), self.rng)
        q1 = _rand_point_in_polygon(self.zone_polys.get("checkout_01", []), self.rng)

        r = self.rng.random()
        speed = self.rng.uniform(75.0, 130.0)
        if r < 0.30:      # queue shopper: browse then queue for a while
            script = [(entrance, 0), (bev, self.rng.uniform(1.5, 4.0)),
                      (q1, self.rng.uniform(6.0, 16.0)), (exit_pt, 0)]
        elif r < 0.62:    # browser: wander two zones
            script = [(entrance, 0), (bev, self.rng.uniform(2.0, 5.0)),
                      (snack, self.rng.uniform(2.0, 5.0)), (exit_pt, 0)]
        elif r < 0.85:    # promo dweller
            script = [(entrance, 0), (promo, self.rng.uniform(5.0, 12.0)), (exit_pt, 0)]
        else:             # quick run-through
            script = [(entrance, 0), (by_pos, 0), (exit_pt, 0)]

        sid = self.shopper_id_counter
        self.shopper_id_counter += 1
        self.shoppers[sid] = _Shopper(sid, x, y, script, speed, self.rng)

    def _burst(self) -> None:
        if self.duration > 0 and self.time >= self.duration:
            return
        for _ in range(self.rng.randint(4, 6)):
            self._spawn(self.time)

    def _update_products(self, dt: float) -> None:
        """Drive shelf states for the demo.

        One shelf drains item-by-item (guarantees a FULL -> LOW_STOCK ->
        OUT_OF_STOCK transition for the presentation), then refills and another
        shelf takes over; the second shelf gets random product churn.
        """
        if self._drain_target is None:
            self._drain_target = self.rng.choice(list(self._shelf_products))
        self._restock_timer -= dt
        if self._restock_timer > 0:
            return
        self._restock_timer = self.rng.uniform(4.0, 8.0)

        target_items = self._shelf_products[self._drain_target]
        if target_items:
            target_items.pop(self.rng.randrange(len(target_items)))
        else:
            self._init_products_from(self._drain_target)
            remaining = [s for s in self._shelf_products if s != self._drain_target]
            self._drain_target = self.rng.choice(remaining)

        other = self.rng.choice([s for s in self._shelf_products if s != self._drain_target])
        items = self._shelf_products[other]
        action = self.rng.random()
        if action < 0.7 and items:
            items.pop(self.rng.randrange(len(items)))
        elif action < 0.85 and len(items) < 14:
            items.append(self._shelf_products[other][-1] if items
                         else np.array([500, 300]))

    def _grid_products(self, shelf: str, poly, expected: int) -> None:
        xs0, ys0 = min(p[0] for p in poly), min(p[1] for p in poly)
        xspan = max(p[0] for p in poly) - xs0
        yspan = max(p[1] for p in poly) - ys0
        pad = 14
        rows = 2 if expected > 6 else 1
        cols = max(1, -(-expected // rows))
        positions = []
        for r in range(rows):
            for c in range(cols):
                px = xs0 + pad + (c / max(cols - 1, 1)) * (xspan - 2 * pad)
                py = ys0 + pad + (r / max(rows - 1, 1)) * (yspan - 2 * pad)
                positions.append(np.array([px, py]))
        self._shelf_products[shelf] = positions[:expected]

    def _init_products_from(self, shelf: str) -> None:
        zones_cfg = self.camera_cfg.get("shelves", {})
        expected = int(zones_cfg[shelf].get("expected_item_count", 12))
        self._grid_products(shelf, self.shelf_regions[shelf], expected)

    # ----------------------------------------------------------------- source
    def next_frame(self) -> Tuple[np.ndarray, List[Detection]]:
        dt = 1.0 / self.fps
        now = self.time

        # scheduling of spawns *before* the state update so metrics populate fast
        self._spawn_timer -= dt
        if self._spawn_timer <= 0:
            self._spawn_timer = self.spawn_interval
            self._spawn(now)
        if now >= self._next_burst:
            self._next_burst = now + self.rng.uniform(22.0, 38.0)
            self._burst()

        for sid in list(self.shoppers):
            if not self.shoppers[sid].update(dt, now):
                del self.shoppers[sid]

        self._update_products(dt)
        self.time += dt

        frame = self._render()
        dets = self._detections()
        self._prev_frame, self._prev_dets = frame, dets
        return frame, dets

    # ------------------------------------------------------------------ render
    def _render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 234, dtype=np.uint8)
        overlay = img.copy()

        for poly in self.zone_polys.values():
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2_polylines_filled(overlay, pts, (200, 220, 240), 0.35)

        for sid, poly in self.shelf_regions.items():
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2_polylines_filled(overlay, pts, (196, 176, 150), 0.9)
            for prod in self._shelf_products[sid]:
                cv2_circle(overlay, int(prod[0]), int(prod[1]), 5, (60, 200, 250), -1)

        img = overlay

        if self._line_a is not None:
            cv2_line(img, self._line_a, self._line_b, (60, 60, 60), 2)

        for shopper in self.shoppers.values():
            x, y = shopper.center
            cv2_circle(img, int(x), int(y), 11, (255, 120, 40), -1)
            cv2_circle(img, int(x), int(y), 11, (255, 255, 255), 1)
            cv2_puttext(img, f"P{shopper.id}", int(x) - 4, int(y) - 14, (40, 40, 40))

        cv2_puttext(img, f"t={self.time:6.1f}s  shoppers={len(self.shoppers)}", 12, 22, (0, 0, 0))
        return img

    def _detections(self) -> List[Detection]:
        dets: List[Detection] = []
        for shopper in self.shoppers.values():
            x, y = shopper.center
            dets.append(Detection(x - 13, y - 30, x + 13, y + 14, 0.99,
                                  PERSON_CLASS, "person", ts=self.time))
        for sid, prods in self._shelf_products.items():
            for prod in prods:
                dets.append(Detection(prod[0] - 4, prod[1] - 4, prod[0] + 4, prod[1] + 4,
                                      0.95, PRODUCT_CLASS, "product", ts=self.time))
        return dets

    def shelf_layout(self) -> Dict[str, Dict]:
        """Expose expected counts + regions for the shelf classifier."""
        zones_cfg = self.camera_cfg.get("shelves", {})
        out = {}
        for sid, region in self.shelf_regions.items():
            out[sid] = {
                "region": [list(map(float, p)) for p in region],
                "expected_item_count": int(zones_cfg[sid].get("expected_item_count", 12)),
                "category": zones_cfg[sid].get("category", "unknown"),
            }
        return out

    # ----------------------------------------------------------------- volume
    def current_volume(self) -> int:
        return len(self.shoppers)


# Tiny cv2 shims so the whole demo can run without opencv installed at import
# time inside tests that only need detections. Frames are still produced.
try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None


def _need_cv2():
    if _cv2 is None:
        raise RuntimeError("opencv-python is required to render demo frames")


def cv2_polylines_filled(img, pts, color, alpha):
    _need_cv2()
    _cv2.fillPoly(img, [pts], color)


def cv2_circle(img, x, y, radius, color, thickness):
    _need_cv2()
    _cv2.circle(img, (x, y), radius, color, thickness)


def cv2_line(img, a, b, color, thickness):
    _need_cv2()
    _cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thickness)


def cv2_puttext(img, text, x, y, color):
    _need_cv2()
    _cv2.putText(img, text, (x, y), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, _cv2.LINE_AA)