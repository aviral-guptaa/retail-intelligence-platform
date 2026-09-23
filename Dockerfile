# SIH 2026 Retail Intelligence — web dashboard (deployed to Render).
# Real person detection runs through the lightweight ONNX Runtime backend
# (yolov8n.onnx, ~13MB) so the image stays small enough for the free tier.
# torch + ultralytics are NOT installed here; that heavier (2GB+) path is for
# local development / training only.
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

# Lightweight real person detection: ONNX Runtime only (no torch). The .onnx
# export ships with the repo. SKIP_ML=1 produces a demo-only image (motion
# detection fallback).
RUN if [ "$SKIP_ML" = "0" ]; then \
      pip install --no-cache-dir onnxruntime; \
    fi

# Copy the app (respects .dockerignore; skips .venv, data, caches).
COPY . .

RUN mkdir -p data/uploads data/raw data/processed models/yolo

EXPOSE 10000
CMD ["sh", "-c", "python run_web.py --host 0.0.0.0 --port ${PORT:-10000}"]