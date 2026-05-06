#!/bin/bash
# ============================================================================
# Dataset Download Script
# Downloads the Credit Card Fraud Detection dataset from Kaggle
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
DATASET="mlg-ulb/creditcardfraud"
CSV_FILE="${DATA_DIR}/creditcard.csv"

echo "================================================"
echo "  Fraud Detection - Dataset Download"
echo "================================================"

# Check credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
        echo "[ERROR] Kaggle credentials not found."
        echo "  Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,"
        echo "  or run: python scripts/setup_kaggle.py"
        exit 1
    fi
    mkdir -p "$HOME/.kaggle"
    echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" > "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
    echo "[OK] Kaggle credentials configured"
fi

# Check if dataset already exists
if [ -f "$CSV_FILE" ]; then
    ROW_COUNT=$(wc -l < "$CSV_FILE")
    echo "[OK] Dataset already exists: $CSV_FILE ($ROW_COUNT lines)"

    if [ "${1:-}" = "--force" ]; then
        echo "[INFO] Force flag set, re-downloading..."
        rm -f "$CSV_FILE"
    else
        echo "[SKIP] Use --force to re-download"
        exit 0
    fi
fi

# Download
mkdir -p "$DATA_DIR"
echo "[INFO] Downloading dataset: $DATASET"
kaggle datasets download -d "$DATASET" -p "$DATA_DIR" --force

# Unzip
ZIP_FILE="${DATA_DIR}/creditcardfraud.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "[INFO] Extracting $ZIP_FILE"
    unzip -o "$ZIP_FILE" -d "$DATA_DIR"
    rm -f "$ZIP_FILE"
fi

# Validate
if [ -f "$CSV_FILE" ]; then
    ROW_COUNT=$(wc -l < "$CSV_FILE")
    echo "[OK] Dataset ready: $CSV_FILE ($ROW_COUNT lines)"
else
    echo "[ERROR] Expected $CSV_FILE not found after extraction"
    exit 1
fi

echo ""
echo "[DONE] Dataset download complete."
