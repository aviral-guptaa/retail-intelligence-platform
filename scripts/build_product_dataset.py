"""Build a YOLO-format PRODUCT detection set from supermarket-shelf VOC XMLs.

Source: the free ShellSense supermarket-shelf imagery (Pinkslushie/
ShelfSense-Supermarket-Shelves-TensorFlow-Object-Detection). Its VOC XML
annotations contain two box classes:

  * "Misplacement" -> a real product sitting in the wrong hole on a shelf
  * "Out-of-stock" -> a bounded gap where products are missing (no product)

This builder keeps the "Misplacement" boxes as the single ``product`` class (id 0)
and rewrites them into the Ultralytics YOLO label format (normalized cx, cy, w, h).
Images without any product box are kept as pure-negative samples so the detector
learns to not fire on empty/gappy shelves.

Images are split at the IMAGE level into train/val (constantly seeded) so no
full-frame or crop leak exists between splits.

Output layout (consumed by ``scripts/train_product_model.py``):
    data/shelf_det/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt
      shelf_det.yaml
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_shelf_det")

IMG_EXT = (".jpg", ".jpeg", ".png")
PRODUCT = "product"


def _boxes(xml_path: Path):
    root = ET.parse(str(xml_path)).getroot()
    w = int(root.findtext("./size/width"))
    h = int(root.findtext("./size/height"))
    boxes = []
    for o in root.findall("object"):
        name = str(o.find("name").text).strip()
        bb = [int(o.find("bndbox/" + k).text) for k in
              ("xmin", "ymin", "xmax", "ymax")]
        x0, y0, x1, y1 = bb
        if name.lower() == "out-of-stock" or x1 <= x0 or y1 <= y0:
            continue
        boxes.append((w, h, x0, y0, x1, y1))
    return w, h, boxes


def _write_label(out_txt: Path, w: int, h: int, boxes) -> int:
    lines = []
    for _, _, x0, y0, x1, y1 in boxes:
        xc = (x0 + x1) / 2.0 / w
        yc = (y0 + y1) / 2.0 / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    if lines:
        out_txt.write_text("\n".join(lines) + "\n")
        return len(lines)
    out_txt.write_text("")       # pure-negative image
    return 0


def build(src: Path, out: Path, seed: int, val_frac: float) -> int:
    rng = random.Random(seed)
    xmls = sorted(Path(src).rglob("*.xml")) if src.is_dir() else []
    if not xmls:
        log.error("no .xml files under %s", src)
        return 1

    samples = []   # (img_path, src_y) with product boxes > 0
    neg = []
    for xml in xmls:
        img_p = xml.with_suffix(xml.suffix)
        if img_p.suffix in IMG_EXT and img_p.exists():
            img_path = img_p
        else:
            cand = [p for p in xml.parent.iterdir()
                    if p.suffix.lower() in IMG_EXT and p.stem == xml.stem]
            img_path = cand[0] if cand else None
        if img_path is None:
            continue
        w, h, boxes = _boxes(xml)
        (neg if not boxes else samples).append((img_path, w, h, boxes))

    rng.shuffle(samples)   # strictly image-level split, seeded & reproducible
    n_val = max(1, int(len(samples) * val_frac))
    val = samples[:n_val]
    train = samples[n_val:]
    rng.shuffle(neg)       # keep most negatives in train, a few in val
    neg_val = neg[: max(1, int(len(neg) * val_frac))]
    neg_train = neg[len(neg_val):]

    for split, items in (("train", train + neg_train), ("val", val + neg_val)):
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for src_img, w, h, boxes in items:
            dst = img_dir / src_img.name
            if not dst.exists():            # keep the largest raw pixels (EXIF here)
                shutil.copy2(str(src_img), str(dst))
            n_boxes += _write_label(lbl_dir / (dst.stem + ".txt"), w, h, boxes)
        log.info("%-5s images=%d product boxes=%d", split, len(items), n_boxes)

    yaml = out / "shelf_det.yaml"
    yaml.write_text(
        f"# ShellSense Misplacement boxes -> single 'product' class\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: {PRODUCT}\n")
    log.info("dataset -> %s", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="folder containing the ShellSense VOOC images + XMLs")
    ap.add_argument("--out", default="data/shelf_det",
                    help="output root (images/ labels/ shelf_det.yaml)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    if not args.src:
        log.error("provide --src pointing at the folder of shelf images + XMLs")
        return 1
    out = Path(__file__).resolve().parent.parent / args.out
    return build(Path(args.src), out, args.seed, args.val_frac)


if __name__ == "__main__":
    raise SystemExit(main())