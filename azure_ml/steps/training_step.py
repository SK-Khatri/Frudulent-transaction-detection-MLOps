"""
Azure ML Pipeline Step: Training
Trains an XGBoost model and logs metrics to MLflow.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")

    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    # Enable MLflow autologging
    mlflow.xgboost.autolog(log_models=False)

    with mlflow.start_run():
        mlflow.log_params(PARAMS)

        # Train
        model = XGBClassifier(**PARAMS)
        start = time.time()
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )
        training_time = time.time() - start

        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision": float(average_precision_score(y_test, y_proba)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "training_time_seconds": training_time,
        }

        mlflow.log_metrics(metrics)
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Copy scaler to model dir for deployment
        import shutil
        scaler_src = Path(args.artifacts_dir) / "scaler.pkl"
        scaler_dst = model_dir / "scaler.pkl"
        if scaler_src.exists():
            shutil.copy2(scaler_src, scaler_dst)
            print(f"Scaler copied to {scaler_dst}")

        mlflow.xgboost.log_model(model, "model")


if __name__ == "__main__":
    main()
