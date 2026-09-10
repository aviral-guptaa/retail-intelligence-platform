"""Build a seeded YOLO-format SUBSET of the SKU-110K retail-shelf dataset.

SKU-110K (Trax Retail, CVPR 2019) is a single-class "product" detection set
densely packed with supermarket shelf items — a good large-scale upgrade path
for the planogram product detector (the current ``yolov8n-product.pt`` uses a
tiny 330-box ShellSense-derived set).

Usage:
    # 1. download (authors' S3 mirror, ~11.4 GB):
    #    curl -L -o data/sku110k_fixed.tar.gz \
    #         http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
    # 2. extract + build a deterministic subset:
    python scripts/build_sku110k_subset.py --tar data/sku110k_fixed.tar.gz
    # 3. train the product detector on it:
    python scripts/train_product_model.py --data data/sku110k_det/sku110k_det.yaml \
        --imgsz 640 --epochs 40 --batch 16 --patience 10 \
        --out models/yolo/yolov8n-sku110k.pt \
        --source "SKU-110K subset (single class: product)"

The subset keeps the original train/val/test split semantics (sampling WITHIN
each official split, never crossing them) and preserves class id 0 = product,
so it is trivially comparable to the ShellSense dataset.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_sku110k")

ANNO_FILES = {
    "train": "annotations/annotations_train.csv",
    "val": "annotations/annotations_val.csv",
    "test": "annotations/annotations_test.csv",
}


def _extract(tar_path: Path, root: Path) -> None:
    if (root / "annotations").exists() and (root / "images").exists():
        log.info("already extracted in %s", root)
        return
    root.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s (one pass, ~5-10 min)", tar_path.name, root)
    with tarfile.open(tar_path) as tar:
        tar.extractall(root, filter="data")
    log.info("extraction done")


def _locate_root(root: Path) -> Path:
    """Handle the tarball's top-level dir (SKU110K_fixed/) vs direct layout."""
    if (root / "annotations").is_dir() and (root / "images").is_dir():
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "annotations").is_dir() and (child / "images").is_dir():
            return child
    return root


def _image_relpath(image_file: str, images_dir: Path) -> Path | None:
    """Locate an annotation's image under images/{train,test,val}, return rel path."""
    for sub in ("train", "test", "val"):
        cand = images_dir / sub / image_file
        if cand.exists():
            return Path(sub) / image_file
    return None


def _annotations(anno_csv: Path, images_dir: Path):
    """Group SKU-110K CSV rows by image, skipping unlocatable images."""
    by_image: dict[str, list[list]] = {}
    with open(anno_csv, newline="") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 8:
                continue
            image_file = row[0]
            rel = _image_relpath(image_file, images_dir)
            if rel is None:
                continue
            by_image.setdefault(str(rel), []).append(row)
    return by_image


def _write_labels(rows, image_path: Path):
    """SKU-110K csv boxes (absolute px, w/h cols) -> YOLO txt (class 0)."""
    w = float(rows[0][6])
    h = float(rows[0][7])
    lines = []
    for r in rows:
        x1, y1, x2, y2 = (float(r[i]) for i in (1, 2, 3, 4))
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"0 {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
    label_path = image_path.parents[1] / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when possible (same volume -> zero-copy), else copy."""
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def build(args) -> int:
    tar_path = Path(args.tar)
    if not tar_path.exists():
        log.error("no tarball at %s (see download command in module docstring)", tar_path)
        return 1
    root = _locate_root(Path(args.root))
    images_dir = root / "images"
    annotations_dir = root / "annotations"

    _extract(tar_path, root)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    total = {"train": 0, "val": 0, "test": 0}
    boxes = {"train": 0, "val": 0, "test": 0}

    for split, n_keep in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        anno_csv = annotations_dir / ANNO_FILES[split]
        if not anno_csv.exists():
            log.warning("missing %s — skipping split", anno_csv)
            continue
        by_image = _annotations(anno_csv, images_dir)
        sample = sorted(by_image.keys())
        if n_keep and n_keep < len(sample):
            rng.shuffle(sample)
            sample = sorted(sample[:n_keep])
        for rel in sample:
            src_img = images_dir / rel
            dst = out / "images" / split / rel.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                _link_or_copy(src_img, dst)
            _write_labels(by_image[rel], dst)
            boxes[split] += len(by_image[rel])
        total[split] = len(sample)
        log.info("split %-5s images=%-5d boxes=%-7d", split, total[split], boxes[split])

    yaml_path = out / "sku110k_det.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: product\n"
    )
    log.info("wrote %s (%d train / %d val / %d test images)",
             yaml_path, total["train"], total["val"], total["test"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default="data/sku110k_fixed.tar.gz")
    ap.add_argument("--root", default="data/sku110k_fixed")
    ap.add_argument("--out", default="data/sku110k_det")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())