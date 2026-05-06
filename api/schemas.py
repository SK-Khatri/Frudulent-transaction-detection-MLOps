"""
Pydantic schemas for the Fraud Detection API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Single transaction input for fraud prediction."""

    Time: float = Field(..., description="Seconds elapsed from first transaction in dataset")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0, description="Transaction amount")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 406.0,
                    "V1": -2.3122265,
                    "V2": 1.9519925,
                    "V3": -1.6098706,
                    "V4": 3.9979055,
                    "V5": -0.5221878,
                    "V6": -1.4265453,
                    "V7": -2.5373865,
                    "V8": 1.3915842,
                    "V9": -2.7700898,
                    "V10": -2.7722723,
                    "V11": 3.2020297,
                    "V12": -2.8999076,
                    "V13": -0.5953246,
                    "V14": -4.2895437,
                    "V15": 0.3896918,
                    "V16": -1.1408076,
                    "V17": -2.8300028,
                    "V18": -0.0168224,
                    "V19": 0.4162922,
                    "V20": 0.1261578,
                    "V21": 0.5171164,
                    "V22": -0.0353889,
                    "V23": -0.4652112,
                    "V24": 0.3201983,
                    "V25": 0.0445191,
                    "V26": 0.1778389,
                    "V27": 0.2610920,
                    "V28": -0.1432051,
                    "Amount": 0.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Whether the transaction is classified as fraud")
    threshold: float = Field(..., description="Decision threshold used")
    risk_level: str = Field(..., description="Risk categorization: MINIMAL, LOW, MEDIUM, HIGH, CRITICAL")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class BatchTransactionInput(BaseModel):
    """Batch of transactions for prediction."""

    transactions: List[TransactionInput] = Field(..., min_length=1, max_length=10000)


class BatchPredictionItem(BaseModel):
    """Single item in a batch prediction response."""

    index: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[BatchPredictionItem]
    total_transactions: int
    total_flagged: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    scaler_loaded: bool
    test_prediction_latency_ms: Optional[float] = None
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_type: str
    threshold: float
    version: str
    features_count: int
