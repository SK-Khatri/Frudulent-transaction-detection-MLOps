"""
Inference Module
Production inference engine with batch and single-transaction prediction,
caching, and monitoring hooks.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


class FraudDetector:
    """
    Production-grade fraud detection inference engine.
    Handles model loading, feature transformation, and prediction.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        threshold: float = 0.5,
    ):
        self.model_path = model_path or MODEL_PATH
        self.scaler_path = scaler_path or SCALER_PATH
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and scaler from disk."""
        logger.info("Loading model from %s", self.model_path)
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        logger.info("Loading scaler from %s", self.scaler_path)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        logger.info("Inference engine initialized.")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as training."""
        df = df.copy()

        df["Hour"] = (df["Time"] / 3600).astype(int) % 24
        df["Log_Amount"] = np.log1p(df["Amount"])
        df["Amount_Bin"] = pd.cut(
            df["Amount"],
            bins=[-1, 10, 50, 200, 1000, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)
        df["V1_V2"] = df["V1"] * df["V2"]
        df["V1_V3"] = df["V1"] * df["V3"]

        df.drop(columns=["Time", "Amount"], inplace=True, errors="ignore")
        if "Class" in df.columns:
            df.drop(columns=["Class"], inplace=True)

        return df

    def predict_single(self, transaction: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction: Dictionary with feature names as keys.

        Returns:
            Prediction result with probability and label.
        """
        start = time.time()

        df = pd.DataFrame([transaction])
        df = self._engineer_features(df)
        X = self.scaler.transform(df.values)

        proba = float(self.model.predict_proba(X)[0, 1])
        is_fraud = proba >= self.threshold
        latency_ms = (time.time() - start) * 1000

        result = {
            "fraud_probability": round(proba, 6),
            "is_fraud": bool(is_fraud),
            "threshold": self.threshold,
            "risk_level": self._get_risk_level(proba),
            "latency_ms": round(latency_ms, 2),
        }

        logger.info(
            "Prediction: prob=%.4f, fraud=%s, latency=%.1fms",
            proba, is_fraud, latency_ms,
        )

        return result

    def predict_batch(
        self,
        transactions: Union[pd.DataFrame, List[Dict[str, float]]],
    ) -> List[Dict[str, Any]]:
        """
        Predict fraud for a batch of transactions.

        Args:
            transactions: DataFrame or list of transaction dictionaries.

        Returns:
            List of prediction results.
        """
        start = time.time()

        if isinstance(transactions, list):
            df = pd.DataFrame(transactions)
        else:
            df = transactions.copy()

        df_eng = self._engineer_features(df)
        X = self.scaler.transform(df_eng.values)

        probas = self.model.predict_proba(X)[:, 1]
        predictions = (probas >= self.threshold).astype(int)

        total_latency = (time.time() - start) * 1000

        results = []
        for i in range(len(probas)):
            results.append({
                "index": i,
                "fraud_probability": round(float(probas[i]), 6),
                "is_fraud": bool(predictions[i]),
                "risk_level": self._get_risk_level(float(probas[i])),
            })

        logger.info(
            "Batch prediction: %d transactions, %d flagged, %.1fms total",
            len(results),
            sum(predictions),
            total_latency,
        )

        return results

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Categorize fraud probability into risk levels."""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.5:
            return "HIGH"
        elif probability >= 0.3:
            return "MEDIUM"
        elif probability >= 0.1:
            return "LOW"
        else:
            return "MINIMAL"

    def health_check(self) -> Dict[str, Any]:
        """Verify the inference engine is operational."""
        try:
            dummy = {f"V{i}": 0.0 for i in range(1, 29)}
            dummy["Time"] = 0.0
            dummy["Amount"] = 100.0
            result = self.predict_single(dummy)
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "scaler_loaded": self.scaler is not None,
                "test_prediction_latency_ms": result["latency_ms"],
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    detector = FraudDetector()

    # Health check
    health = detector.health_check()
    print(f"Health: {health}")

    # Example single prediction
    sample_txn = {f"V{i}": np.random.randn() for i in range(1, 29)}
    sample_txn["Time"] = 3600.0
    sample_txn["Amount"] = 149.62

    result = detector.predict_single(sample_txn)
    print(f"\nSingle prediction: {result}")

    # Example batch prediction
    batch = [
        {**{f"V{i}": np.random.randn() for i in range(1, 29)}, "Time": t, "Amount": a}
        for t, a in [(1000, 50), (2000, 200), (3000, 500)]
    ]
    results = detector.predict_batch(batch)
    print(f"\nBatch predictions ({len(results)} txns):")
    for r in results:
        print(f"  #{r['index']}: prob={r['fraud_probability']:.4f}, risk={r['risk_level']}")
