# Production Dockerfile for Federated IDS FastAPI Backend
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

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

# Expose port
EXPOSE 8000

# Start FastAPI server using dynamic $PORT assigned by Railway/Cloud
CMD ["sh", "-c", "uvicorn src.dashboard.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
