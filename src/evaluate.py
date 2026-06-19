"""
Evaluation Module
Comprehensive model evaluation with threshold optimization,
fairness metrics, and model comparison capabilities.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
EVAL_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"


class ModelEvaluator:
    """Comprehensive model evaluator for fraud detection."""

    def __init__(self, model=None, model_path: Optional[Path] = None):
        if model is not None:
            self.model = model
        elif model_path:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Run full evaluation suite."""
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision": float(average_precision_score(y_test, y_proba)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
            "threshold": threshold,
            "total_samples": int(len(y_test)),
            "fraud_samples": int(y_test.sum()),
            "legit_samples": int(len(y_test) - y_test.sum()),
        }

        logger.info(
            "Evaluation results:\n%s",
            classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        )

        return metrics

    def find_optimal_threshold(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find the optimal classification threshold.

        Args:
            metric: Optimization target - 'f1', 'precision', or 'recall'.

        Returns:
            (optimal_threshold, best_score)
        """
        y_proba = self.model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.1, 0.95, 0.01)

        best_threshold = 0.5
        best_score = 0.0

        metric_fn = {
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
        }[metric]

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = metric_fn(y_test, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = t

        logger.info(
            "Optimal threshold for %s: %.3f (score=%.4f)",
            metric, best_threshold, best_score,
        )
        return float(best_threshold), float(best_score)

    def evaluate_at_operating_points(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate at multiple operating points useful for business decisions.
        """
        y_proba = self.model.predict_proba(X_test)[:, 1]
        operating_points = {
            "high_precision": 0.8,
            "balanced": 0.5,
            "high_recall": 0.3,
            "max_recall": 0.1,
        }

        results = {}
        for name, threshold in operating_points.items():
            y_pred = (y_proba >= threshold).astype(int)
            results[name] = {
                "threshold": threshold,
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "flagged_transactions": int(y_pred.sum()),
                "flagged_ratio": float(y_pred.mean()),
            }

        return results

    def generate_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        save_path = save_path or EVAL_REPORT_PATH

        report = {
            "standard_evaluation": self.evaluate(X_test, y_test),
            "optimal_thresholds": {},
            "operating_points": self.evaluate_at_operating_points(X_test, y_test),
        }

        for metric in ["f1", "precision", "recall"]:
            threshold, score = self.find_optimal_threshold(X_test, y_test, metric)
            report["optimal_thresholds"][metric] = {
                "threshold": threshold,
                "score": score,
            }

        # Model quality gate check
        roc_auc = report["standard_evaluation"]["roc_auc"]
        pr_auc = report["standard_evaluation"]["average_precision"]

        report["quality_gate"] = {
            "roc_auc_passed": roc_auc >= 0.95,
            "pr_auc_passed": pr_auc >= 0.70,
            "overall_passed": roc_auc >= 0.95 and pr_auc >= 0.70,
            "thresholds": {"min_roc_auc": 0.95, "min_pr_auc": 0.70},
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation report saved to %s", save_path)
        logger.info(
            "Quality gate: %s (ROC-AUC=%.4f, PR-AUC=%.4f)",
            "PASSED" if report["quality_gate"]["overall_passed"] else "FAILED",
            roc_auc,
            pr_auc,
        )

        return report


def run_evaluation(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience function to run full evaluation."""
    evaluator = ModelEvaluator(model_path=model_path)
    return evaluator.generate_report(X_test, y_test)


if __name__ == "__main__":
    from data_ingestion import load_dataset
    from preprocessing import FraudPreprocessor

    df = load_dataset()
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    report = run_evaluation(X_test, y_test)

    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"ROC AUC:            {report['standard_evaluation']['roc_auc']:.4f}")
    print(f"PR AUC:             {report['standard_evaluation']['average_precision']:.4f}")
    print(f"F1 Score:           {report['standard_evaluation']['f1_score']:.4f}")
    print(f"Quality Gate:       {'PASSED' if report['quality_gate']['overall_passed'] else 'FAILED'}")
    print(f"\nOptimal F1 Threshold: {report['optimal_thresholds']['f1']['threshold']:.3f}")
