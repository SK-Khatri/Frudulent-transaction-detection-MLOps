#!/usr/bin/env python3
"""
Full local pipeline runner.
Executes the complete ML pipeline locally without Azure ML.
Useful for development and testing.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("  FRAUD DETECTION - LOCAL PIPELINE RUNNER")
    print("=" * 60)
    total_start = time.time()

    # Step 1: Data Ingestion
    print("\n[1/4] DATA INGESTION")
    print("-" * 40)
    from src.data_ingestion import download_dataset, validate_dataset

    csv_path = download_dataset()
    validate_dataset(csv_path)

    # Step 2: Preprocessing
    print("\n[2/4] PREPROCESSING")
    print("-" * 40)
    from src.preprocessing import FraudPreprocessor
    import pandas as pd

    df = pd.read_csv(csv_path)
    preprocessor = FraudPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    preprocessor.save_scaler()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 3: Training
    print("\n[3/4] TRAINING")
    print("-" * 40)
    from src.train import run_training

    model, metrics = run_training(X_train, X_test, y_train, y_test)

    # Step 4: Evaluation
    print("\n[4/4] EVALUATION")
    print("-" * 40)
    from src.evaluate import run_evaluation

    report = run_evaluation(X_test, y_test)

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  ROC AUC:       {report['standard_evaluation']['roc_auc']:.4f}")
    print(f"  PR AUC:        {report['standard_evaluation']['average_precision']:.4f}")
    print(f"  F1 Score:      {report['standard_evaluation']['f1_score']:.4f}")
    print(f"  Precision:     {report['standard_evaluation']['precision']:.4f}")
    print(f"  Recall:        {report['standard_evaluation']['recall']:.4f}")
    gate = report["quality_gate"]
    print(f"  Quality Gate:  {'PASSED ✓' if gate['overall_passed'] else 'FAILED ✗'}")
    print()
    print("  Artifacts saved to: artifacts/")
    print("  To serve the model: python -m api.app")
    print("=" * 60)


if __name__ == "__main__":
    main()
