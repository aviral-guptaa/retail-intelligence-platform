# Standalone build of the queue-forecast web API.
#
# Build from the REPO ROOT so the model artifacts + predictor are included:
#   docker build -f deployment/web_model/Dockerfile -t queue-forecaster .
#   docker run -p 8000:8000 queue-forecaster
FROM python:3.11-slim

WORKDIR /app

COPY deployment/web_model/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The whole repo (model joblibs under models/prediction/ are gitignored, so copy
# them explicitly if present).
COPY . .
COPY models/prediction/queue_model_5min.joblib models/prediction/ 2>/dev/null || true
COPY models/prediction/queue_model_10min.joblib models/prediction/ 2>/dev/null || true
COPY models/prediction/queue_metrics.json models/prediction/ 2>/dev/null || true

EXPOSE 8000
CMD ["uvicorn", "deployment.web_model.app:app", "--host", "0.0.0.0", "--port", "8000"]
