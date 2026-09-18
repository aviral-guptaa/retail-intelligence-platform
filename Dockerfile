# SIH 2026 Retail Intelligence — web dashboard (deployed to Render).
# Real video + live-computer person detection needs the ML deps (torch + ultralytics),
# so SKIP_ML defaults to 0 — the webcam people counter and uploaded-video analysis
# must work on Render. Build with --build-arg SKIP_ML=1 only for a stripped demo image.
FROM python:3.13-slim

ARG SKIP_ML=0

# opencv needs libgl; keep minimal.
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
COPY scripts/fetch_yolov8n.py scripts/fetch_yolov8n.py
RUN if [ "$SKIP_ML" = "0" ]; then \
      python scripts/fetch_yolov8n.py || true; \
    fi

EXPOSE 10000
CMD ["sh", "-c", "python run_web.py --host 0.0.0.0 --port ${PORT:-10000}"]