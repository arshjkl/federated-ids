# Production Dockerfile for Federated IDS FastAPI Backend
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and datasets
COPY . .

# Ensure required storage directories exist
RUN mkdir -p artifacts/uploaded_datasets models/custom_trained data/mock/inference

# Expose ports for both Render (10000) and Railway/local (8000)
EXPOSE 10000 8000

# Start FastAPI server using dynamic $PORT assigned by Render/Railway (defaults to 10000 on Render)
CMD ["sh", "-c", "uvicorn src.dashboard.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
