"""
Tests for the Preprocessing Module
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import FraudPreprocessor


@pytest.fixture
def sample_df():
    """Create a minimal synthetic dataset for testing."""
    np.random.seed(42)
    n = 1000
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.random.exponential(50, n)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][:50] = 1  # 5% fraud rate
    np.random.shuffle(data["Class"])
    return pd.DataFrame(data)


class TestFraudPreprocessor:
    def test_engineer_features_shape(self, sample_df):
        preprocessor = FraudPreprocessor()
        df = preprocessor.engineer_features(sample_df)
        # Should have dropped Time and Amount, added Hour, Log_Amount, Amount_Bin, V1_V2, V1_V3
        assert "Time" not in df.columns
        assert "Amount" not in df.columns
        assert "Hour" in df.columns
        assert "Log_Amount" in df.columns
        assert "Amount_Bin" in df.columns
        assert "V1_V2" in df.columns
        assert "V1_V3" in df.columns

    def test_engineer_features_hour_range(self, sample_df):
        preprocessor = FraudPreprocessor()
        df = preprocessor.engineer_features(sample_df)
        assert df["Hour"].min() >= 0
        assert df["Hour"].max() <= 23

    def test_fit_transform_output_shapes(self, sample_df):
        preprocessor = FraudPreprocessor(apply_smote=False)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_df)
        total = len(X_train) + len(X_test)
        assert total == len(sample_df)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_fit_transform_with_smote(self, sample_df):
        preprocessor = FraudPreprocessor(apply_smote=True, smote_ratio=0.5)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(sample_df)
        fraud_ratio = y_train.mean()
        # SMOTE should increase fraud representation
        assert fraud_ratio > 0.05

    def test_scaler_persistence(self, sample_df, tmp_path):
        preprocessor = FraudPreprocessor(apply_smote=False)
        preprocessor.fit_transform(sample_df)
        scaler_path = tmp_path / "scaler.pkl"
        preprocessor.save_scaler(scaler_path)
        assert scaler_path.exists()

        preprocessor2 = FraudPreprocessor()
        preprocessor2.load_scaler(scaler_path)
        assert preprocessor2._fitted is True

    def test_transform_requires_fit(self, sample_df):
        preprocessor = FraudPreprocessor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.transform(sample_df)

    def test_get_feature_names(self):
        preprocessor = FraudPreprocessor()
        names = preprocessor.get_feature_names()
        assert len(names) == 33  # 28 V-features + 5 engineered
        assert "V1" in names
        assert "Log_Amount" in names

    def test_get_stats(self, sample_df):
        preprocessor = FraudPreprocessor(apply_smote=False)
        X_train, _, y_train, _ = preprocessor.fit_transform(sample_df)
        stats = preprocessor.get_stats(X_train, y_train)
        assert "n_samples" in stats
        assert "n_features" in stats
        assert "fraud_ratio" in stats
        assert stats["n_samples"] == len(X_train)
