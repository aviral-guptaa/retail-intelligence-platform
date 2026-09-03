"""Generate a synthetic store CCTV clip that matches the configured zones so the
uploaded video drives the full analytics pipeline with REAL person detection.

Why: the demo simulator renders shoppers as plain coloured circles, which YOLO
cannot detect as people. This generator composites REAL person photos (cropped
out of stock images with YOLO) onto the same 1280x720 store floor the
`zones.json` (store_01) coordinates were calibrated for. People are rendered
LARGE (~200px) and kept apart so YOLO reliably detects them. Actors spawn at the
top entrance, cross the entry line, walk through the shopping zones and form a
queue at the checkout zones - so occupancy, footfall, queue length with
predictions and the heatmap all light up on the dashboard when this clip is
uploaded.

Usage:
  python scripts/make_store_clip.py --out data/uploads/store_clip.mp4 --seconds 20
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

W, H = 1280, 720

ENTRY_LINE = ([580, 60], [700, 60])
QUEUE_ZONES = {
    "checkout_01": [[1100, 200], [1270, 200], [1270, 700], [1100, 700]],
    "checkout_02": [[160, 200], [340, 200], [340, 700], [160, 700]],
}
SHOP_ZONE_MID = [(550, 330), (900, 330), (655, 540)]  # beverages, snacks, promo
SHELF_REGIONS = {
    "shelf_a": [[410, 230], [690, 230], [690, 430], [410, 430]],
    "shelf_b": [[770, 230], [1040, 230], [1040, 430], [770, 430]],
}

# Queue target anchors (top of each queue zone) so people line up in-zone.
QUEUE_ANCHOR = {
    "checkout_01": [(1120, 560), (1170, 580), (1220, 610), (1260, 640)],
    "checkout_02": [(185, 560), (235, 580), (285, 610), (325, 640)],
}


def _within(x, y, poly, ppad=25):
    pts = np.array(poly, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= -ppad


class Actor:
    def __init__(self, aid, sprite, rng, checkout):
        self.id = aid
        self.sprite = sprite
        self.checkout = checkout
        self.h = rng.uniform(190, 215)
        cx = rng.uniform(ENTRY_LINE[0][0], ENTRY_LINE[1][0])
        self.x, self.y = float(cx), float(rng.uniform(-20, 10))
        self.lane = rng.randint(0, len(SHOP_ZONE_MID) - 1)
        self.shop = SHOP_ZONE_MID[self.lane]
        self.anchor_i = rng.randint(0, len(QUEUE_ANCHOR[checkout]) - 1)
        self.anchor = QUEUE_ANCHOR[checkout][self.anchor_i]
        self.speed = rng.uniform(3.6, 5.2)
        self.phase = "enter"   # enter -> shop -> queue -> leave
        self.pause = 0.0

    def _move_toward(self, tx, ty, dt):
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 4:
            return True
        step = self.speed * dt
        if step >= dist:
            self.x, self.y = tx, ty
            return True
        self.x += step * dx / dist
        self.y += step * dy / dist
        return False


def step_actor(a: Actor, dt=1.0):
    if a.phase == "enter":
        if a._move_toward(a.x, 150, dt):
            a.phase = "shop"
    elif a.phase == "shop":
        if a._move_toward(a.shop[0], a.shop[1], dt):
            a.pause = 30
            a.phase = "queue"
    elif a.phase == "queue":
        if a.pause > 0:
            a.pause -= 1
        else:
            if a._move_toward(a.anchor[0], a.anchor[1], dt):
                a.phase = "leave"
    elif a.phase == "leave":
        a.x += a.speed * 0.9
        if a.x > 1500 or (a.checkout == "checkout_01" and a.x < 900):
            a.phase = "done"


def render_floor():
    img = np.full((H, W, 3), 234, dtype=np.uint8)
    ov = img.copy()
    zone_polys = list(SHELF_REGIONS.values()) + list(QUEUE_ZONES.values())
    for poly in zone_polys:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(ov, [pts], (200, 218, 236))
    for sid, poly in SHELF_REGIONS.items():
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(ov, [pts], (188, 170, 150))
        for _ in range(14):
            px = random.uniform(poly[0][0] + 15, poly[2][0] - 15)
            py = random.uniform(poly[0][1] + 15, poly[2][1] - 15)
            cv2.circle(ov, (int(px), int(py)), 5, (60, 200, 250), -1)
    img = ov
    cv2.line(img, tuple(ENTRY_LINE[0]), tuple(ENTRY_LINE[1]), (80, 80, 80), 3)
    for qz in QUEUE_ZONES.values():
        pts = np.array(qz, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (90, 140, 200), 2)
    for qz in QUEUE_ZONES.values():
        xs = [p[0] for p in qz]
        cv2.rectangle(img, (min(xs) - 20, 90), (min(xs) + 90, 200), (40, 40, 40), -1)
        cv2.putText(img, "REGISTER", (min(xs) - 10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def draw_sprite(img, a):
    scale = a.h / a.sprite.shape[0]
    spr = cv2.resize(a.sprite, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = spr.shape[:2]
    x = int(a.x - sw / 2); y = int(a.y - sh)
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + sw); y1 = min(H, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = spr[y0 - y:y1 - y, x0 - x:x1 - x]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data/uploads/store_clip.mp4"))
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-people", type=int, default=14)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sprite_dir = ROOT / "data/raw/cast"
    sprites = [cv2.imread(str(p)) for p in sorted(sprite_dir.glob("*.png"))]
    sprites = [s for s in sprites if s is not None]
    if not sprites:
        print("No person sprites in", sprite_dir)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, H))
    actors: list[Actor] = []
    n_frames = int(args.seconds * args.fps)
    spawn_every = max(4, int(n_frames * 0.4 / args.max_people))
    aid = 0
    max_people = args.max_people

    for f in range(n_frames):
        # spawn staggered so people enter over time and stay separated
        active = [a for a in actors if a.phase != "done"]
        if f % spawn_every == 0 and len(active) < max_people:
            actors.append(Actor(aid, rng.choice(sprites), rng,
                                rng.choice(list(QUEUE_ZONES.keys()))))
            aid += 1

        for a in actors[:]:
            if a.phase == "done":
                actors.remove(a)
                continue
            step_actor(a)

        img = render_floor()
        for a in sorted(actors, key=lambda a: a.y):
            draw_sprite(img, a)
        cv2.putText(img, "Store 01 - synthetic CCTV", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        vw.write(img)

    vw.release()
    print(f"wrote {out_path}: {n_frames} frames, {args.seconds}s@{args.fps}fps, "
          f"{aid} actors spawned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
