"""
Data Ingestion Module
Downloads the Credit Card Fraud Detection dataset from Kaggle,
validates integrity, and prepares it for the ML pipeline.
"""

import os
import sys
import json
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
DATASET_FILENAME = "creditcardfraud.zip"
CSV_FILENAME = "creditcard.csv"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def setup_kaggle_credentials() -> None:
    """
    Configure Kaggle API credentials from environment variables.
    Expects KAGGLE_USERNAME and KAGGLE_KEY to be set.
    Writes kaggle.json to ~/.kaggle/ if it does not exist.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if kaggle_json.exists():
        logger.info("kaggle.json already exists at %s", kaggle_json)
        return

    if not username or not key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set. "
            "Export them or add to your .env file."
        )

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    credentials = {"username": username, "key": key}

    with open(kaggle_json, "w") as f:
        json.dump(credentials, f)

    # Restrict permissions on Unix systems
    if sys.platform != "win32":
        os.chmod(kaggle_json, 0o600)

    logger.info("Kaggle credentials written to %s", kaggle_json)


def download_dataset(data_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download the Credit Card Fraud Detection dataset from Kaggle.

    Args:
        data_dir: Directory to store the dataset. Defaults to project data/ folder.
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded CSV file.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / CSV_FILENAME
    zip_path = data_dir / DATASET_FILENAME

    if csv_path.exists() and not force:
        logger.info("Dataset already exists at %s, skipping download.", csv_path)
        return csv_path

    setup_kaggle_credentials()

    # Import kaggle here so credentials are set first
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset '%s' to %s ...", KAGGLE_DATASET, data_dir)
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), unzip=False)

    # Unzip
    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    # Cleanup zip
    zip_path.unlink(missing_ok=True)
    logger.info("Dataset ready at %s", csv_path)

    return csv_path


def validate_dataset(csv_path: Path) -> bool:
    """
    Run basic validation checks on the downloaded dataset.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, nrows=5)
    expected_columns = ["Time", "Amount", "Class"]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    row_count = sum(1 for _ in open(csv_path)) - 1  # minus header
    logger.info("Dataset validated: %d rows, %d columns", row_count, len(df.columns))

    if row_count < 200000:
        logger.warning("Row count (%d) is lower than expected (~284807).", row_count)

    return True


def load_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the dataset into a pandas DataFrame.
    Downloads if not present.
    """
    if csv_path is None:
        csv_path = DEFAULT_DATA_DIR / CSV_FILENAME

    if not csv_path.exists():
        csv_path = download_dataset()

    validate_dataset(csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: shape=%s", df.shape)
    return df


if __name__ == "__main__":
    csv = download_dataset(force="--force" in sys.argv)
    validate_dataset(csv)
    df = load_dataset(csv)
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud ratio: {df['Class'].mean():.4%}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
