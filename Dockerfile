# SIH 2026 Retail Intelligence — web dashboard (deployed to Render)
# Light build by default so it succeeds on Render's free tier (fast, small, reliable).
# The app runs fully in `demo` mode + dashboard with only core deps.
# Real video person detection needs the ML deps (torch + ultralytics):
#   build with --build-arg SKIP_ML=0 (much slower, higher RAM).
FROM python:3.13-slim

ARG SKIP_ML=1

# opencv only needs libgl; keep minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Core runtime + dashboard (always installed, lightweight).
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        numpy \
        opencv-python-headless \
        PyYAML \
        python-dotenv \
        pandas \
        scikit-learn \
        joblib \
        fastapi \
        "uvicorn[standard]" \
        websockets \
        SQLAlchemy \
        pydantic \
        python-multipart

# Optional heavy ML stack for real person detection.
RUN if [ "$SKIP_ML" = "0" ]; then \
      pip install --no-cache-dir torch torchvision ultralytics onnxruntime; \
    fi

# Copy the app (respects .dockerignore; skips .venv, data, checkpoints).
COPY . .

RUN mkdir -p data/uploads data/raw data/processed models/yolo

# Best-effort YOLOv8n download (only when heavy ML is enabled).
RUN if [ "$SKIP_ML" = "0" ]; then \
      python - <<'PY' || true
import pathlib, urllib.request
dest = pathlib.Path("models/yolo/yolov8n.pt")
if not dest.exists():
    try:
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", dest)
        print("yolov8n downloaded", dest.stat().st_size)
    except Exception as e:
        print("yolov8n skipped:", e)
        dest.write_bytes(b"")
PY
    fi

EXPOSE 10000
CMD ["sh", "-c", "python run_web.py --host 0.0.0.0 --port ${PORT:-10000}"]