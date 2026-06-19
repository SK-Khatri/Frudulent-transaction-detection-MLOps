"""
Preprocessing Module
Handles feature engineering, scaling, class balancing, and train/test splitting
for the credit card fraud detection pipeline.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


class FraudPreprocessor:
    """End-to-end preprocessor for credit card fraud detection data."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        apply_smote: bool = True,
        smote_ratio: float = 0.5,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.apply_smote = apply_smote
        self.smote_ratio = smote_ratio
        self.scaler = RobustScaler()
        self._fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from the raw dataset.
        """
        df = df.copy()

        # Time-based features: hour of day (Time is seconds from first txn)
        df["Hour"] = (df["Time"] / 3600).astype(int) % 24

        # Log-transform Amount to reduce skew
        df["Log_Amount"] = np.log1p(df["Amount"])

        # Amount bins
        df["Amount_Bin"] = pd.cut(
            df["Amount"],
            bins=[-1, 10, 50, 200, 1000, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)

        # Interaction features between top PCA components
        df["V1_V2"] = df["V1"] * df["V2"]
        df["V1_V3"] = df["V1"] * df["V3"]

        # Drop original Time and Amount (replaced by engineered versions)
        df.drop(columns=["Time", "Amount"], inplace=True)

        logger.info("Feature engineering complete. New shape: %s", df.shape)
        return df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline: engineer features, scale, split, and balance.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        df = self.engineer_features(df)

        y = df["Class"].values
        X = df.drop(columns=["Class"]).values

        # Train/test split (stratified to preserve fraud ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info(
            "Train/Test split: train=%d, test=%d", len(X_train), len(X_test)
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self._fitted = True

        logger.info(
            "Pre-SMOTE class distribution: 0=%d, 1=%d",
            np.sum(y_train == 0),
            np.sum(y_train == 1),
        )

        # Apply SMOTE to training data only
        if self.apply_smote:
            smote = SMOTE(
                sampling_strategy=self.smote_ratio,
                random_state=self.random_state,
                n_jobs=-1,
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(
                "Post-SMOTE class distribution: 0=%d, 1=%d",
                np.sum(y_train == 0),
                np.sum(y_train == 1),
            )

        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted scaler.
        Used for inference.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")

        df = self.engineer_features(df)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        return self.scaler.transform(df.values)

    def save_scaler(self, path: Optional[Path] = None) -> Path:
        """Save the fitted scaler to disk."""
        path = path or SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info("Scaler saved to %s", path)
        return path

    def load_scaler(self, path: Optional[Path] = None) -> None:
        """Load a previously fitted scaler from disk."""
        path = path or SCALER_PATH
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        self._fitted = True
        logger.info("Scaler loaded from %s", path)

    def get_feature_names(self) -> list:
        """Return the feature names after engineering."""
        pca_features = [f"V{i}" for i in range(1, 29)]
        engineered = ["Hour", "Log_Amount", "Amount_Bin", "V1_V2", "V1_V3"]
        return pca_features + engineered

    def get_stats(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Return summary statistics for logging/monitoring."""
        return {
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "fraud_ratio": float(np.mean(y_train)),
            "mean_feature_values": X_train.mean(axis=0).tolist()[:5],
        }


def run_preprocessing(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to run the full preprocessing pipeline.
    """
    df = pd.read_csv(csv_path)
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessor.save_scaler()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_ingestion import load_dataset

    df = load_dataset()
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessor.save_scaler()

    stats = preprocessor.get_stats(X_train, y_train)
    print("\nPreprocessing complete:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    print(f"  Fraud ratio (train): {stats['fraud_ratio']:.4%}")
