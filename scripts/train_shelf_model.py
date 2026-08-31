"""Train a transfer-learning shelf-state classifier (FULL / LOW_STOCK / OUT_OF_STOCK).

Primary path: a labelled folder of shelf crops (generate placeholder data with
``scripts/make_shelf_dataset.py``, replace with real photos at the site):

    python scripts/train_shelf_model.py --data data/shelf --epochs 10

A MobileNetV3-Small (default) or EfficientNet-B0 head is fine-tuned and the
weights, class labels, backbone name and validation metrics are written to
``models/prediction/shelf_classifier.pt`` (+ ``.metrics.json``). The runtime
side ``ml/shelf/shelf_classifier.py`` reads the backbone name from the
checkpoint so no extra config is needed.

Requires torch + torchvision (see requirements.txt optional section).
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
log = logging.getLogger("train_shelf")

BACKBONES = ("mobilenet_v3_small", "efficientnet_b0")


def _build_model(backbone: str, num_classes: int):
    import torch
    from torchvision.models import get_model

    model = get_model(backbone, weights=None, num_classes=num_classes)
    return model


def _train_from_dir(data_dir: Path, args) -> int:
    try:
        import torch
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms
    except ImportError:
        log.error("training requires torch + torchvision: pip install torch torchvision")
        return 1

    tf = transforms.Compose([transforms.Resize((96, 96)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    full = datasets.ImageFolder(str(data_dir), transform=tf)
    if len(full.classes) < 2:
        log.error("expected FULL/LOW_STOCK/OUT_OF_STOCK subfolders in %s (got %s)",
                  data_dir, full.classes)
        return 1
    n = len(full)
    n_val = max(1, int(n * 0.2))
    train_ds, val_ds = random_split(full, [n - n_val, n_val])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = _build_model(args.backbone, len(full.classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total += yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
        scheduler.step()
        log.info("epoch %d  loss=%.4f  train_acc=%.3f", epoch, loss.item(), correct / total)

    model.eval()
    with torch.no_grad():
        preds, targets = [], []
        for xb, yb in val_loader:
            preds.extend(model(xb).argmax(1).tolist())
            targets.extend(yb.tolist())

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    acc = float(accuracy_score(targets, preds))
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, zero_division=0)
    cm = confusion_matrix(targets, preds).tolist()
    metrics = {
        "source": "real_images", "n_train": n - n_val, "n_val": n_val,
        "accuracy": round(acc, 4), "epochs": args.epochs, "backbone": args.backbone,
        "precision": [round(float(x), 4) for x in p],
        "recall": [round(float(x), 4) for x in r],
        "f1": [round(float(x), 4) for x in f1],
        "confusion_matrix": cm, "classes": full.classes,
    }
    log.info("val accuracy=%.3f  per-class F1=%s", acc, metrics["f1"])

    out_path = Path(ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": full.classes,
                "backbone": args.backbone}, str(out_path))
    (out_path.with_suffix(".metrics.json")).write_text(json.dumps(metrics, indent=2))
    log.info("saved model -> %s", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="folder with FULL/LOW_STOCK/OUT_OF_STOCK subfolders")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone", choices=BACKBONES, default="mobilenet_v3_small")
    ap.add_argument("--out", default="models/prediction/shelf_classifier.pt")
    args = ap.parse_args()

    data_dir = Path(args.data) if args.data else None
    if data_dir is None or not data_dir.exists():
        log.error("provide a labelled folder: --data data/shelf (generate it via "
                  "scripts/make_shelf_dataset.py) or point at real photos")
        return 1
    return _train_from_dir(data_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())