FROM python:3.12-slim

ARG PRELOAD_YOLO_WEIGHTS=false
ARG YOLO_WEIGHTS_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data /app/models

COPY src ./src
COPY 01-App ./01-App
COPY README.md ./README.md
COPY feature-requests.txt ./feature-requests.txt

RUN if [ "$PRELOAD_YOLO_WEIGHTS" = "true" ]; then \
        python -c "from pathlib import Path; from logistics_approval_agent.config import ensure_yolo_weights; ensure_yolo_weights(Path('/app/models/yolov8m.pt'), '${YOLO_WEIGHTS_URL}')"; \
    fi

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/logistics_approval_agent/web/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
