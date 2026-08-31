# Edge deployment notes

Targets: NVIDIA Jetson (Orin/Nano), Raspberry Pi 5, Intel NUC with a TPU/NPU.
The abstraction lives in `ml/detection/yolo_detector.py` (the `_load()` point
is where a device-specific runtime plugs in) and `scripts/export_onnx.py`.

## Recommended flow

1. Develop/validate the full pipeline on a laptop or Google Colab in demo mode.
2. Export the detector:

   ```bash
   python scripts/export_onnx.py --model models/yolo/yolov8n.pt --format onnx --half     # FP16
   # optional INT8 on Jetson (requires TensorRT):
   python scripts/export_onnx.py --model models/yolo/yolov8n.pt --format engine --int8
   ```

3. Run inference with ONNX Runtime on the edge. Full provider list differs per
   device; keep CPU as the guaranteed fallback:

   ```python
   import onnxruntime as ort
   providers = [
       ("CUDAExecutionProvider", {}),            # Jetson / NVIDIA GPU
       # "TensorrtExecutionProvider", "CPUExecutionProvider" (Jetson, engine file)
       # "OpenVINOExecutionProvider"             # Intel NUC with iGPU
       "CPUExecutionProvider",
   ]
   sess = ort.InferenceSession("models/yolo/yolov8n.onnx", providers=providers)
   # feed BGR frames at the exported imgsz; parse outputs into Detection boxes
   ```

4. Serve the same FastAPI endpoints on the device (WebSockets keep working).
5. Point `config/settings.yaml` at the exported model and mark the device:

   ```yaml
   model: { device: cpu, yolo_model: models/yolo/yolov8n.onnx, imgsz: 480 }
   processing: { max_fps: 12, retry_interval_seconds: 3, reconnect_max_attempts: 0 }
   ```

## Metrics to record (section 14) + benchmark script

`scripts/benchmark.py` automates the table below for any mode/source:

```bash
python scripts/benchmark.py --mode demo          # simulator, no camera
python scripts/benchmark.py --mode video --source clip.mp4
python scripts/benchmark.py --mode live --source 0
```

It reports: average FPS, mean + P50/P95/P99/max per-step latency, model size,
peak RSS (memory), and validates an ONNX export's throughput on the current
detector. For lower-level device telemetry use `/usr/bin/time -v` or
`tegrastats` / `vcgencmd measure_temp` (Pi).

| Metric | How |
|---|---|
| FPS / per-step latency | `scripts/benchmark.py` (P50/P95/P99/max) |
| Model size | file size of `.onnx` / `.engine` (in benchmark output) |
| Memory usage | peak RSS in benchmark output; `/usr/bin/time -v` or Jetson `tegrastats` |
| CPU/GPU/NPU util | `tegrastats` / `nvidia-smi` / `top` |
| Power / temperature | `tegrastats`, `vcgencmd measure_temp` (Pi) |

## Device-specific config points (set in `.env` / `settings.yaml`)

- `EDGE_DEVICE=jetson_orin_nano | rpi5 | nuc`
- `model.device=cuda` (Jetson) or `cpu`/NPU for others
- `model.yolo_model=...engine` when TensorRT is used
- Lower `frame_width/frame_height` or inference `imgsz` (e.g. 480) for FPS
- `processing.max_fps` caps system load on thin devices
- `tracking.backend=bytetrack` (needs ultralytics) or `iou` for a lightweight CPU path

## Offline operation

Demo mode runs fully offline. For real cameras on a closed network, keep static
DHCP leases/ONVIF URLs in `cameras.yaml` and the detector exported to ONNX so no
runtime model download is required on boot. Persistence goes to local SQLite by
default (`DATABASE_URL` swaps to PostgreSQL when one is reachable).