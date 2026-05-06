"""
Azure ML Pipeline Step: Evaluation
Runs quality gate checks and generates evaluation report.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import mlflow


# Quality gate thresholds
MIN_ROC_AUC = 0.95
MIN_PR_AUC = 0.70


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    # Load model
    model_path = model_dir / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Evaluating model on {len(X_test)} test samples")

    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    # Classification report
    report_text = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    print(f"\n{report_text}")

    # Quality gate
    roc_passed = metrics["roc_auc"] >= MIN_ROC_AUC
    pr_passed = metrics["average_precision"] >= MIN_PR_AUC
    gate_passed = roc_passed and pr_passed

    quality_gate = {
        "roc_auc_passed": roc_passed,
        "pr_auc_passed": pr_passed,
        "overall_passed": gate_passed,
        "thresholds": {"min_roc_auc": MIN_ROC_AUC, "min_pr_auc": MIN_PR_AUC},
    }

    report = {
        "metrics": metrics,
        "quality_gate": quality_gate,
        "classification_report": report_text,
    }

    # Save report
    report_path = report_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to {report_path}")

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))

    # Print gate result
    status = "PASSED ✓" if gate_passed else "FAILED ✗"
    print(f"\n{'='*50}")
    print(f"QUALITY GATE: {status}")
    print(f"  ROC AUC:  {metrics['roc_auc']:.4f} (min: {MIN_ROC_AUC}) → {'✓' if roc_passed else '✗'}")
    print(f"  PR AUC:   {metrics['average_precision']:.4f} (min: {MIN_PR_AUC}) → {'✓' if pr_passed else '✗'}")
    print(f"{'='*50}")

    if not gate_passed:
        print("WARNING: Model did not pass quality gates. Review before deploying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
