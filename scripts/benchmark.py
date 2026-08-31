"""Edge / laptop performance benchmark for the analytics pipeline.

Measures what the spec's section-14 metric table requires: achieved FPS, end-to-end
step latency (avg / P50 / P95 / P99 / max), detector model size, ONNX Runtime
validity and process memory. Run in demo mode for a noise-free baseline, or
against a video/RTSP source for real figures.

    python scripts/benchmark.py --mode demo --seconds 20
    python scripts/benchmark.py --mode video --source clip.mp4 --seconds 30
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("benchmark")


def _rss_kb() -> int:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _model_size(detector) -> float:
    if detector is None:
        return 0.0
    try:
        return Path(detector.model_path).stat().st_size / 1024.0
    except OSError:
        return 0.0


def _benchmark_onnx(providers: bool) -> None:
    candidates = sorted(Path(ROOT / "models" / "yolo").glob("*.onnx"))
    if not candidates:
        print("no ONNX model found under models/yolo (export one with scripts/export_onnx.py)")
        return None
    path = candidates[0]
    print(f"ONNX: {path.name} ({path.stat().st_size / 1024.0:.0f} KB)")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed - skipping ONNX validation")
        return None
    sess = ort.InferenceSession(str(path))
    prov = sess.get_providers()
    print(f"onnxruntime ready | providers: {prov}")
    if providers:
        print("  (set providers=[\"CUDAExecutionProvider\",\"CPUExecutionProvider\"] at init)")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "video", "live"], default="demo")
    ap.add_argument("--source", default=None)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--camera-id", default="store_01")
    ap.add_argument("--onnx", action="store_true", help="validate the exported ONNX too")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)

    from config.loader import load_settings
    from database.repository import Repository
    from app.services.inference_service import InferencePipeline

    settings = load_settings()
    settings["demo"]["duration_seconds"] = -1
    if args.mode:  # keep the analytics loop busy for the whole benchmark
        settings["processing"]["max_fps"] = 0

    pipeline = InferencePipeline(settings, Repository(None), display=False)
    if args.mode == "demo":
        pipeline.add_camera(args.camera_id, mode="demo")
    else:
        src = int(args.source) if str(args.source).isdigit() else args.source
        if args.mode == "video" and isinstance(src, str):
            path = Path(src)
            if not path.is_file():
                print(f"video file not found: {path}")
                return 1
            src = str(path)
        pipeline.add_camera(args.camera_id, mode=args.mode, source=src)

    svc = pipeline.services[args.camera_id]
    detector = svc.detector

    latencies: list = []
    ops = 100
    print(f"warming up ({ops} steps) ...")
    for _ in range(ops):
        pipeline.step_all()

    print(f"benchmarking for {args.seconds:.0f}s ...")
    t0 = time.monotonic()
    frames = 0
    while time.monotonic() - t0 < args.seconds:
        s = time.monotonic()
        pipeline.step_all()
        latencies.append((time.monotonic() - s) * 1000.0)
        frames += 1
    elapsed = time.monotonic() - t0

    fps = frames / elapsed
    n = len(latencies)
    p = lambda q: round(statistics.quantiles(latencies, n=100)[q - 1], 2) if n else 0.0
    print("\n" + "=" * 52)
    print(f"  mode                    {args.mode}")
    print(f"  camera_id               {args.camera_id}")
    print(f"  detector backend        {getattr(detector, 'backend', 'none') if detector else 'demo'}")
    print(f"  achieved FPS            {fps:.1f}")
    print(f"  avg step latency        {statistics.mean(latencies):8.2f} ms" if n else "")
    if n:
        print(f"  p50 / p95 / p99 latency {p(50)} / {p(95)} / {p(99)} ms")
        print(f"  max step latency        {max(latencies):8.2f} ms")
        print(f"  samples                 {n}")
    print(f"  model size              {_model_size(detector):8.1f} KB")
    print(f"  peak RSS                {_rss_kb() / 1024.0:8.1f} MB")
    print("=" * 52)
    if args.onnx:
        _benchmark_onnx(providers=True)

    pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())