"""Generate a labelled shelf-image dataset for training the stock classifier.

Without hardware / real store photos this synthesises shelf crops with a wooden
background and product dots whose density matches each class, so the full
training pipeline can be exercised on a laptop. Replace the generated folders
with real store photos (one folder per class) and re-run train_shelf_model.py.

Output layout (as used by scripts/train_shelf_model.py --data):
    data/shelf/FULL/*.png  data/shelf/LOW_STOCK/*.png  data/shelf/OUT_OF_STOCK/*.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def render_shelf(size: int, item_count: int, rng: random.Random) -> np.ndarray:
    """A shelf face: dusty background plus ``item_count`` product fronts."""
    img = np.full((size, size, 3), (140, 132, 116), dtype=np.uint8)
    # subtle background noise
    noise = rng.randint(0, 8)
    bg = np.clip(img.astype(np.int16) + rng.randint(-noise, noise, img.shape), 0, 255)
    img = bg.astype(np.uint8)
    for _ in range(item_count):
        x = rng.randint(8, size - 24)
        y = rng.randint(8, size - 20)
        # product face (bright, colourful)
        color = (rng.randint(140, 255), rng.randint(90, 255), rng.randint(90, 255))
        cv2.rectangle(img, (x, y), (x + rng.randint(10, 14), y + rng.randint(8, 10)),
                      color, -1)
        cv2.rectangle(img, (x, y), (x + rng.randint(10, 14), y + rng.randint(8, 10)),
                      (20, 20, 20), 1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shelf")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--size", type=int, default=96)
    args = ap.parse_args()

    out = Path(ROOT := Path(__file__).resolve().parent.parent) / args.out
    counts = {"FULL": (10, 14), "LOW_STOCK": (3, 6), "OUT_OF_STOCK": (0, 1)}
    labels = list(counts)
    metadata = {"size": args.size, "labels": labels, "samples": {}, "generated": True}
    for cls in labels:
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        rng = random.Random(42)
        lo, hi = counts[cls]
        for i in range(args.per_class):
            count = rng.randint(lo, hi)
            img = render_shelf(args.size, count, rng)
            cv2.imwrite(str(d / f"{i:04d}.png"), img)
        metadata["samples"][cls] = args.per_class
        print(f"wrote {args.per_class} x {cls} -> {d}")

    (out / "dataset_metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2))
    print("replace these folders with real photos, then run:")
    print(f"  python scripts/train_shelf_model.py --data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())