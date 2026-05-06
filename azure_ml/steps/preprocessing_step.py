"""
Azure ML Pipeline Step: Preprocessing
Applies feature engineering, scaling, SMOTE, and train/test split.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
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
    df.drop(columns=["Time", "Amount"], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    csv_path = input_dir / "creditcard.csv"
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Raw data shape: {df.shape}")

    # Feature engineering
    df = engineer_features(df)

    y = df["Class"].values
    X = df.drop(columns=["Class"]).values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    scaler_path = artifacts_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # SMOTE on train set
    print(f"Pre-SMOTE: class_0={np.sum(y_train==0)}, class_1={np.sum(y_train==1)}")
    smote = SMOTE(sampling_strategy=0.5, random_state=42, n_jobs=-1)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Post-SMOTE: class_0={np.sum(y_train==0)}, class_1={np.sum(y_train==1)}")

    # Save processed data
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    main()
