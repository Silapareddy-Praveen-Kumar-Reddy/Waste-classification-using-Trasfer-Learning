# ── Stage: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Keeps Python from generating .pyc files and enables stdout/stderr flushing
# Disable oneDNN verbose logging; TF will still use CPU optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install system deps needed by TensorFlow / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure the uploads directory exists
RUN mkdir -p static/uploads

EXPOSE 8080

# --preload: model is loaded once in master process, then forked to workers
# This avoids 4 separate TF/Keras load_model() calls at startup
CMD ["gunicorn", \
     "-w", "4", \
     "-b", "0.0.0.0:8080", \
     "--preload", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
