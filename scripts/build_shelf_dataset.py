"""Build a shelf-status labelled set (FULL / LOW_STOCK / OUT_OF_STOCK) from
Object-Detection-style supermarket shelf annotations.

Source: the free ShellSense supermarket-shelf imagery (Pinkslushie/
ShelfSense-Supermarket-Shelves-TensorFlow-Object-Detection). Its VOC XML
annotations label two box classes:

  * "Out-of-stock"  -> a bounded gap where products are missing (empty shelf)
  * "Misplacement"  -> a product sitting in the wrong hole

We turn those box labels into a per-patch shelf STATUS classification that the
runtime ``ml/shelf/shelf_classifier.py`` CNN can consume:

  * OUT_OF_STOCK crop  - the interior of an "Out-of-stock" box (clearly empty)
  * FULL          crop - a patch sampled away from every annotation box
                         (where the shelf is conventionally stocked)
  * LOW_STOCK crop    - a patch overlapping a small part of an empty box but
                         still largely stocked (the transitional, "low" case).

To avoid data leakage, the FULL negatives are drawn only from the parts of each
image that do NOT intersect any annotation box, and OUT/LOW crops never come
from the same pixel region twice. Patches are scaled to a fixed square size so
``scripts/train_shelf_model.py`` can ingest the folder directly.

Output layout (ImageFolder-ready):
    data/shelf/{FULL,LOW_STOCK,OUT_OF_STOCK}/*.jpg
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_shelf")

IMG_EXT = (".jpg", ".jpeg", ".png")
PATCH = 96  # matches the training transform input resolution


def _boxes(xml_path: Path):
    root = ET.parse(str(xml_path)).getroot()
    boxes = []
    for o in root.findall("object"):
        name = str(o.find("name").text).strip()
        bb = [int(o.find("bndbox/" + k).text) for k in
              ("xmin", "ymin", "xmax", "ymax")]
        boxes.append((name, bb))
    return boxes


def _inside(pt, box):
    x, y = pt
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _pick_empty_patch(img_w, img_h, occupied, rng):
    """Randomly sample a PATCH-sized square that does not intersect any box."""
    for _ in range(80):
        x = rng.randint(0, max(1, img_w - PATCH - 1))
        y = rng.randint(0, max(1, img_h - PATCH - 1))
        patch_box = (x, y, x + PATCH, y + PATCH)
        bad = False
        for _, b in occupied:
            # expand each box slightly so a FULL patch never grazes a gap
            ex = (b[0] - 8, b[1] - 8, b[2] + 8, b[3] + 8)
            if (max(patch_box[0], ex[0]) < min(patch_box[2], ex[2]) and
                    max(patch_box[1], ex[1]) < min(patch_box[3], ex[3])):
                bad = True
                break
        if not bad:
            return patch_box
    return None


def _crop_resize(img, box, size=PATCH):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def build(src: Path, out: Path, seed: int, per_out: int) -> int:
    rng = random.Random(seed)
    xmls = sorted(Path(src).rglob("*.xml")) if src.is_dir() else []
    if not xmls:
        log.error("no .xml files under %s", src)
        return 1
    (out / "FULL").mkdir(parents=True, exist_ok=True)
    (out / "LOW_STOCK").mkdir(parents=True, exist_ok=True)
    (out / "OUT_OF_STOCK").mkdir(parents=True, exist_ok=True)

    n_full = n_low = n_out = 0
    for i, xml in enumerate(xmls):
        img_p = xml.with_suffix(xml.suffix)
        if img_p.suffix in IMG_EXT and img_p.exists():
            img_path = img_p
        else:
            cand = [p for p in xml.parent.iterdir()
                    if p.suffix.lower() in IMG_EXT and p.stem == xml.stem]
            img_path = cand[0] if cand else None
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("cannot read %s", img_path)
            continue
        h, w = img.shape[:2]

        oos = [bb for name, bb in _boxes(xml) if name.lower() == "out-of-stock"]
        allb = [(name, bb) for name, bb in _boxes(xml)]
        occupied = allb

        # OUT_OF_STOCK crops from every OOS box
        out_cnt = 0
        for bb in oos:
            crop = _crop_resize(img, bb)
            if crop is None:
                continue
            cv2.imwrite(str(out / "OUT_OF_STOCK" / f"img{i:04d}_{out_cnt}.jpg"), crop)
            n_out += 1
            out_cnt += 1

        # LOW_STOCK: patches adjacent to (touching) an OOS box, not fully empty
        low_cnt = 0
        for bb in oos:
            for _ in range(3):
                side = rng.choice(["top", "bottom", "left", "right"])
                x0, y0, x1, y1 = bb
                pw, ph = (x1 - x0), (y1 - y0)
                if side == "top":
                    box = (x0, max(0, y0 - PATCH), min(w, x0 + PATCH),
                           max(0, y0 - PATCH) + PATCH)
                elif side == "bottom":
                    box = (x0, min(h - PATCH, y1), min(w, x0 + PATCH),
                           min(h - PATCH, y1) + PATCH)
                elif side == "left":
                    box = (max(0, x0 - PATCH), y0, max(0, x0 - PATCH) + PATCH,
                           min(h, y0 + PATCH))
                else:
                    box = (min(w - PATCH, x1), y0, min(w - PATCH, x1) + PATCH,
                           min(h, y0 + PATCH))
                if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                    continue
                # keep only if it still contains substantial shelf area and is
                # adjacent to (not fully covering) the OOS gap
                crop = _crop_resize(img, box)
                if crop is None:
                    continue
                cv2.imwrite(str(out / "LOW_STOCK" / f"img{i:04d}_l{low_cnt}.jpg"), crop)
                n_low += 1
                low_cnt += 1

        # FULL: negative patches far away from every box in this image
        full_cnt = 0
        seen = set()
        target_full = min(per_out, max(1, out_cnt * 2))
        attempts = 0
        while full_cnt < target_full and attempts < 120:
            patch = _pick_empty_patch(w, h, occupied, rng)
            if patch is None:
                break
            key = (patch[0] // PATCH, patch[1] // PATCH)
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            crop = _crop_resize(img, patch)
            if crop is None:
                attempts += 1
                continue
            cv2.imwrite(str(out / "FULL" / f"img{i:04d}_f{full_cnt}.jpg"), crop)
            n_full += 1
            full_cnt += 1
            attempts += 1

    log.info("built %d OUT_OF_STOCK, %d LOW_STOCK, %d FULL -> %s",
             n_out, n_low, n_full, out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="folder containing the ShellSense VOOC images + XMLs")
    ap.add_argument("--out", default="data/shelf",
                    help="output ImageFolder root (creates FULL/LOW_STOCK/OUT_OF_STOCK)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-out", type=int, default=40,
                    help="target FULL crops per image (approx)")
    args = ap.parse_args()
    if not args.src:
        log.error("provide --src pointing at the folder of shelf images + XMLs")
        return 1
    src = Path(args.src)
    out = Path(__file__).resolve().parent.parent / args.out
    return build(src, out, args.seed, args.per_out)


if __name__ == "__main__":
    raise SystemExit(main())