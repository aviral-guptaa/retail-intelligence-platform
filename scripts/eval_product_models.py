"""Compare product-detector checkpoints honestly across datasets.

Answers "which checkpoint should ship" by measuring mAP of one or more
checkpoints on one or more YOLO datasets. Used after an SKU-110K fine-tune to
decide whether it beats the ShellSense-derived checkpoint on both:
  1. the SKU-110K held-out val (large-set generalization) and
  2. the ShellSense val (site-ish domain transfer).

    python scripts/eval_product_models.py \
        --checkpoints models/yolo/yolov8n-product.pt models/yolo/yolov8n-sku110k.pt \
        --datasets data/shelf_det/shelf_det.yaml data/sku110k_det/sku110k_det.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_product")


def main() -> int:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log.error("requires ultralytics: pip install ultralytics")
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rows: list[dict] = []
    for ckpt in args.checkpoints:
        p = Path(ROOT) / ckpt
        if not p.exists():
            log.error("no checkpoint %s", p)
            return 1
        model = YOLO(str(p))
        for ds in args.datasets:
            d = Path(ROOT) / ds
            if not d.exists():
                log.error("no dataset %s", d)
                return 1
            m = model.val(data=str(d), device=args.device, imgsz=args.imgsz, verbose=False)
            rows.append({"checkpoint": p.name, "dataset": d.name,
                         "mAP50": round(float(m.box.map50), 4),
                         "mAP50_95": round(float(m.box.map), 4),
                         "precision": round(float(m.box.mp), 4),
                         "recall": round(float(m.box.mr), 4)})
            log.info("%-28s on %-22s mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f",
                     p.name, d.name, rows[-1]["mAP50"], rows[-1]["mAP50_95"],
                     rows[-1]["precision"], rows[-1]["recall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())