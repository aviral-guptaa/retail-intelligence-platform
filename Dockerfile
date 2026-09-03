# SIH 2026 Retail Intelligence — web dashboard (deployed to Render)
#
# Full install (default): torch + ultralytics for REAL person detection.
#   docker build --build-arg SKIP_ML=1 ...   => lighter build (motion-detection fallback)
FROM python:3.13-slim

ARG SKIP_ML=0

# opencv / onnxruntime native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Core runtime + dashboard (always)
COPY requirements.txt ./

# Install everything when ML is enabled; otherwise only the core, non-ML deps.
RUN if [ "$SKIP_ML" = "1" ]; then \
      pip install --no-cache-dir \
        numpy opencv-python PyYAML python-dotenv pandas scikit-learn joblib \
        fastapi "uvicorn[standard]" websockets SQLAlchemy pydantic python-multipart; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the app
COPY . .

RUN mkdir -p data/uploads data/raw data/processed models/yolo

# Best-effort YOLOv8n download for real detection (only when ML enabled).
RUN if [ "$SKIP_ML" != "1" ]; then \
      python - <<'PY' || true
import pathlib, urllib.request
dest = pathlib.Path("models/yolo/yolov8n.pt")
if not dest.exists():
    try:
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        urllib.request.urlretrieve(url, dest)
        print("yolov8n downloaded", dest.stat().st_size)
    except Exception as e:
        print("yolov8n skipped:", e)
        dest.write_bytes(b"")
PY
    fi

EXPOSE 10000
CMD ["sh", "-c", "python run_web.py --host 0.0.0.0 --port ${PORT:-10000}"]