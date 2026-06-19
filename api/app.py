"""
FastAPI Application for Fraud Detection
Production-ready REST API with Prometheus metrics integration.
"""

import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FraudDetector
from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse,
    BatchPredictionItem,
    HealthResponse,
    ModelInfoResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_COUNT = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["result"],
)
FRAUD_PROBABILITY = Histogram(
    "fraud_prediction_probability",
    "Distribution of fraud probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MODEL_LOADED = Gauge(
    "fraud_model_loaded",
    "Whether the model is loaded (1) or not (0)",
)
ACTIVE_REQUESTS = Gauge(
    "fraud_api_active_requests",
    "Number of active requests being processed",
)

# ─── Global State ─────────────────────────────────────────────────────────────
detector: FraudDetector = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global detector
    try:
        model_path = Path(os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "model.pkl")))
        scaler_path = Path(os.environ.get("SCALER_PATH", str(PROJECT_ROOT / "artifacts" / "scaler.pkl")))
        threshold = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))

        detector = FraudDetector(
            model_path=model_path,
            scaler_path=scaler_path,
            threshold=threshold,
        )
        MODEL_LOADED.set(1)
        logger.info("Model loaded successfully. Threshold=%.2f", threshold)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        MODEL_LOADED.set(0)

    yield

    logger.info("Shutting down fraud detection API.")


# ─── App Creation ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Production ML API for real-time credit card fraud detection. "
                "Powered by XGBoost with MLOps pipeline on Azure ML.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics for Prometheus."""
    ACTIVE_REQUESTS.inc()
    start = time.time()

    response = await call_next(request)

    duration = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)
    ACTIVE_REQUESTS.dec()

    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """API welcome endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    if detector is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            scaler_loaded=False,
            error="Model not loaded",
        )
    health = detector.health_check()
    return HealthResponse(**health)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get model metadata."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type="XGBClassifier",
        threshold=detector.threshold,
        version="1.0.0",
        features_count=33,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput) -> PredictionResponse:
    """
    Predict fraud probability for a single transaction.

    Returns fraud probability, binary classification, risk level, and latency.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        txn_dict = transaction.model_dump()
        result = detector.predict_single(txn_dict)

        # Track metrics
        label = "fraud" if result["is_fraud"] else "legit"
        PREDICTION_COUNT.labels(result=label).inc()
        FRAUD_PROBABILITY.observe(result["fraud_probability"])

        return PredictionResponse(**result)

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTransactionInput) -> BatchPredictionResponse:
    """
    Predict fraud for a batch of transactions (up to 10,000).
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start = time.time()
        txn_dicts = [t.model_dump() for t in batch.transactions]
        results = detector.predict_batch(txn_dicts)

        # Track metrics
        for r in results:
            label = "fraud" if r["is_fraud"] else "legit"
            PREDICTION_COUNT.labels(result=label).inc()
            FRAUD_PROBABILITY.observe(r["fraud_probability"])

        total_flagged = sum(1 for r in results if r["is_fraud"])
        processing_time = (time.time() - start) * 1000

        return BatchPredictionResponse(
            predictions=[BatchPredictionItem(**r) for r in results],
            total_transactions=len(results),
            total_flagged=total_flagged,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("Batch prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ─── Error Handlers ──────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", "8000")),
        workers=int(os.environ.get("API_WORKERS", "4")),
        log_level="info",
    )
