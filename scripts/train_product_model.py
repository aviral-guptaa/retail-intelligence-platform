"""Fine-tune a YOLOv8 person checkpoint into a single-class PRODUCT detector.

For honest planogram compliance the pipeline needs a detector that actually
predicts shelf products (``detector.supports_products`` must be True and
``detect_products()`` must return real boxes), not the 999 placeholder.
This script fine-tunes the existing COCO ``yolov8n.pt`` on a fixed
single-``product`` class.

Dataset: ``scripts/build_product_dataset.py`` exports the ShellSense
"Misplacement" shelf boxes in YOLO format
(``data/shelf_det/{images,labels}/{train,val}``). Smaller, domain-matched
alternative to SKU-110K; if large-scale robustness is required, re-run with a
``--data`` pointing at an SKU-110K-derived set with the same class layout.

    python scripts/build_product_dataset.py --src <ShellSense angled_images>
    python scripts/train_product_model.py

Requires torch + ultralytics (see requirements.txt optional section).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_product")


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset_counts(data_path: Path) -> dict:
    """Report real image/box counts from the YOLO dataset directories."""
    root = data_path.parent if data_path.name.endswith(".yaml") else data_path
    out = {"n_train_images": 0, "n_val_images": 0,
           "train_boxes": 0, "val_boxes": 0}
    for key in ("train", "val"):
        lbl = root / "labels" / key
        if not lbl.is_dir():
            continue
        n_imgs = len(list((root / "images" / key).glob("*"))) if (root / "images" / key).is_dir() else 0
        n_box = 0
        for t in lbl.glob("*.txt"):
            if t.stat().st_size:
                n_box += sum(1 for l in t.read_text().splitlines() if l.strip())
        out[f"n_{key}_images"] = n_imgs
        out[f"{key}_boxes"] = n_box
    return out


def train(args) -> int:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log.error("training requires ultralytics + torch: "
                  "pip install torch torchvision ultralytics")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("no dataset yaml at %s (run scripts/build_product_dataset.py first)", data_path)
        return 1
    weights = Path(args.weights)
    if not weights.exists():
        log.error("no base weights at %s", weights)
        return 1

    device = _pick_device(args.device)
    log.info("fine-tuning %s -> %s product on device=%s imgsz=%s epochs=%d",
             weights.name, args.out, device, args.imgsz, args.epochs)
    model = YOLO(str(weights))

    results = model.train(data=str(data_path), epochs=args.epochs,
                          imgsz=args.imgsz, device=device,
                          batch=args.batch, workers=2, seed=args.seed,
                          patience=args.patience, project=str(ROOT / "runs/detect"),
                          name="product_det", exist_ok=True,
                          val=True)
    best = Path(ROOT) / "runs/detect/product_det" / "weights" / "best.pt"
    if not best.exists():
        log.error("training finished but best.pt missing; nothing to deploy")
        return 1

    out_path = Path(ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(best), str(out_path))

    metrics = model.val(data=str(data_path), device=device, imgsz=args.imgsz, verbose=False)
    counts = _dataset_counts(data_path)
    summary = {
        "source": args.source,
        "dataset": str(data_path), "epochs": args.epochs, "imgsz": args.imgsz,
        "device": device, "classes": ["product"],
        **counts,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    (out_path.with_suffix(".metrics.json")).write_text(json.dumps(summary, indent=2))
    log.info("deployed -> %s  mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f",
             out_path, summary["mAP50"], summary["mAP50_95"],
             summary["precision"], summary["recall"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shelf_det/shelf_det.yaml")
    ap.add_argument("--weights", default="models/yolo/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="larger than 640 so small shelf products stay visible")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="models/yolo/yolov8n-product.pt")
    ap.add_argument("--source",
                    default="ShellSense Misplacement boxes (single class: product)",
                    help="honest provenance label written to the .metrics.json")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())