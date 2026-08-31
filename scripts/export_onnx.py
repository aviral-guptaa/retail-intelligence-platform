"""Export a YOLO checkpoint to ONNX for edge deployment (with FP16/INT8 notes).

Requires ultralytics. Typical usage:

  python scripts/export_onnx.py --model models/yolo/yolov8n.pt --format onnx --half

Then see deployment/edge/README.md for running ONNX on Jetson/Raspberry Pi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolo/yolov8n.pt")
    ap.add_argument("--format", choices=["onnx", "torchscript", "engine", "openvino"],
                    default="onnx")
    ap.add_argument("--half", action="store_true", help="FP16 export")
    ap.add_argument("--int8", action="store_true", dest="int8", help="INT8 (needs calibration)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is required: pip install ultralytics onnx onnxruntime")
        return 1

    model_path = Path(ROOT) / args.model
    if not model_path.exists():
        print(f"model not found: {model_path}\nDownload:\n"
              "  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
        return 1

    model = YOLO(str(model_path))
    out = model.export(format=args.format, imgsz=args.imgsz, half=args.half,
                       int8=args.int8, dynamic=False)
    print(f"exported -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())