"""
Azure ML Pipeline Step: Data Ingestion
Downloads the Kaggle Credit Card Fraud Detection dataset.
"""

import argparse
import os
import json
import zipfile
import sys
from pathlib import Path

def setup_kaggle():
    """Configure Kaggle credentials from environment."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    creds = {
        "username": os.environ["KAGGLE_USERNAME"],
        "key": os.environ["KAGGLE_KEY"],
    }
    with open(kaggle_json, "w") as f:
        json.dump(creds, f)
    os.chmod(str(kaggle_json), 0o600)
    print(f"Kaggle credentials configured at {kaggle_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_kaggle()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    dataset = "mlg-ulb/creditcardfraud"
    print(f"Downloading dataset: {dataset}")
    api.dataset_download_files(dataset, path=str(output_dir), unzip=False)

    zip_path = output_dir / "creditcardfraud.zip"
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    zip_path.unlink()

    csv_path = output_dir / "creditcard.csv"
    assert csv_path.exists(), f"Expected {csv_path} to exist after extraction"

    import pandas as pd
    df = pd.read_csv(csv_path, nrows=5)
    print(f"Dataset downloaded successfully: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample shape: {df.shape}")


if __name__ == "__main__":
    main()
