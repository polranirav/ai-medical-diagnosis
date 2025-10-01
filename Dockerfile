# Multi-stage Dockerfile for AI Medical Diagnosis System

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first for caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt streamlit gradio

# Copy source
COPY src ./src
COPY configs ./configs
COPY frontend ./frontend
COPY models ./models
COPY results ./results
COPY README.md MODEL_CARD.md ./

# Expose ports
EXPOSE 8000 8501

# Default env vars
ENV MODEL_CKPT=models/exp_hydra/best.pt \
    API_PORT=8000 \
    STREAMLIT_PORT=8501 \
    PYTHONPATH=/app

# Launch script (FastAPI + optional Streamlit)
CMD ["bash", "-lc", "uvicorn src.api.app:app --host 0.0.0.0 --port $API_PORT & streamlit run frontend/streamlit_app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0"]
