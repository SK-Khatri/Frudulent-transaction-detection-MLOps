# ============================================================================
# Fraud Detection MLOps - Production Dockerfile
# Multi-stage build for optimized image size
# ============================================================================

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: Production ──────────────────────────────────────────────────────
FROM python:3.10-slim AS production

# Labels
LABEL maintainer="mlops-team"
LABEL description="Fraud Detection API - Credit Card Fraud Detection with XGBoost"
LABEL version="1.0.0"

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set up Kaggle credentials directory
ARG KAGGLE_USERNAME=""
ARG KAGGLE_KEY=""
RUN mkdir -p /home/appuser/.kaggle && \
    if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then \
        echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > /home/appuser/.kaggle/kaggle.json && \
        chmod 600 /home/appuser/.kaggle/kaggle.json; \
    fi && \
    chown -R appuser:appuser /home/appuser

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY azure_ml/ ./azure_ml/
COPY artifacts/ ./artifacts/

# Set ownership
RUN chown -R appuser:appuser /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/artifacts/model.pkl \
    SCALER_PATH=/app/artifacts/scaler.pkl \
    PREDICTION_THRESHOLD=0.5 \
    API_PORT=8000 \
    API_WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Switch to non-root user
USER appuser

EXPOSE ${API_PORT}

# Run with gunicorn for production
CMD ["python", "-m", "uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log"]
