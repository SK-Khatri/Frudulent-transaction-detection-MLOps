"""
Tests for the Fraud Detection API
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_transaction():
    """A valid transaction payload for testing."""
    return {
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


class TestAPISchemas:
    """Test request/response schema validation."""

    def test_transaction_input_valid(self, sample_transaction):
        from api.schemas import TransactionInput
        txn = TransactionInput(**sample_transaction)
        assert txn.Time == 406.0
        assert txn.Amount == 0.0

    def test_transaction_input_missing_field(self, sample_transaction):
        from api.schemas import TransactionInput
        del sample_transaction["V1"]
        with pytest.raises(Exception):
            TransactionInput(**sample_transaction)

    def test_transaction_input_negative_amount(self, sample_transaction):
        from api.schemas import TransactionInput
        sample_transaction["Amount"] = -10.0
        with pytest.raises(Exception):
            TransactionInput(**sample_transaction)

    def test_prediction_response_schema(self):
        from api.schemas import PredictionResponse
        resp = PredictionResponse(
            fraud_probability=0.85,
            is_fraud=True,
            threshold=0.5,
            risk_level="CRITICAL",
            latency_ms=12.5,
        )
        assert resp.is_fraud is True
        assert resp.risk_level == "CRITICAL"

    def test_batch_transaction_input(self, sample_transaction):
        from api.schemas import BatchTransactionInput
        batch = BatchTransactionInput(
            transactions=[sample_transaction, sample_transaction]
        )
        assert len(batch.transactions) == 2

    def test_health_response(self):
        from api.schemas import HealthResponse
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            scaler_loaded=True,
            test_prediction_latency_ms=5.2,
        )
        assert resp.status == "healthy"
