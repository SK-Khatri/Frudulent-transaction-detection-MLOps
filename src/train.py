"""
Training Module
Trains an XGBoost classifier for fraud detection with hyperparameter tuning,
experiment tracking via MLflow, and model artifact persistence.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Default hyperparameters (tuned for fraud detection)
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,  # Already handled by SMOTE
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


class FraudModelTrainer:
    """Handles training, evaluation, and persistence of the fraud detection model."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or DEFAULT_PARAMS.copy()
        self.model: Optional[XGBClassifier] = None
        self.metrics: Dict[str, Any] = {}
        self.training_time: float = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> XGBClassifier:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.

        Returns:
            Trained XGBClassifier.
        """
        logger.info("Starting model training with params: %s", json.dumps(self.params, indent=2))

        self.model = XGBClassifier(**self.params)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = 50

        start = time.time()
        self.model.fit(X_train, y_train, **fit_params)
        self.training_time = time.time() - start

        logger.info("Training completed in %.2f seconds", self.training_time)
        return self.model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Perform stratified k-fold cross-validation.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        model = XGBClassifier(**self.params)

        scores = {
            "cv_f1": cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1),
            "cv_roc_auc": cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1),
            "cv_precision": cross_val_score(model, X, y, cv=skf, scoring="precision", n_jobs=-1),
            "cv_recall": cross_val_score(model, X, y, cv=skf, scoring="recall", n_jobs=-1),
        }

        cv_results = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in scores.items()}
        logger.info("Cross-validation results: %s", json.dumps(cv_results, indent=2))
        return cv_results

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision": float(average_precision_score(y_test, y_proba)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "threshold": threshold,
            "training_time_seconds": self.training_time,
            "n_test_samples": len(y_test),
            "fraud_rate_test": float(y_test.mean()),
        }

        logger.info("Evaluation metrics:\n%s", json.dumps(self.metrics, indent=2))
        logger.info(
            "\nClassification Report:\n%s",
            classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        )

        return self.metrics

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """Return feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        importances = self.model.feature_importances_
        if feature_names and len(feature_names) == len(importances):
            return dict(sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            ))
        return {f"feature_{i}": float(v) for i, v in enumerate(importances)}

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save the trained model to disk."""
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", path)
        return path

    def save_metrics(self, path: Optional[Path] = None) -> Path:
        """Save evaluation metrics to JSON."""
        path = path or METRICS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info("Metrics saved to %s", path)
        return path

    def load_model(self, path: Optional[Path] = None) -> XGBClassifier:
        """Load a trained model from disk."""
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return self.model

    def log_to_mlflow(
        self,
        experiment_name: str = "fraud-detection",
        run_name: Optional[str] = None,
    ) -> None:
        """Log parameters, metrics, and model to MLflow."""
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self.params)
            mlflow.log_metrics(
                {k: v for k, v in self.metrics.items() if isinstance(v, (int, float))}
            )
            if self.model is not None:
                mlflow.xgboost.log_model(self.model, "model")
            mlflow.log_artifact(str(METRICS_PATH))
            logger.info("Logged run to MLflow experiment '%s'", experiment_name)


def run_training(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Convenience function to train and evaluate a model.
    """
    trainer = FraudModelTrainer()
    trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save_model()
    trainer.save_metrics()
    return trainer.model, metrics


if __name__ == "__main__":
    from data_ingestion import load_dataset
    from preprocessing import FraudPreprocessor

    df = load_dataset()
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessor.save_scaler()

    model, metrics = run_training(X_train, X_test, y_train, y_test)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR AUC:    {metrics['average_precision']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
