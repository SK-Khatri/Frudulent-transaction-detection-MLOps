#!/usr/bin/env python3
"""
Kaggle API Setup Script
Configures Kaggle credentials and verifies access.
Run this before using the data pipeline.
"""

import os
import sys
import json
from pathlib import Path


def main():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Check environment variables
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if kaggle_json.exists():
        print(f"[OK] kaggle.json already exists at {kaggle_json}")
        with open(kaggle_json) as f:
            creds = json.load(f)
        print(f"[OK] Username: {creds.get('username', 'N/A')}")
    elif username and key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        if sys.platform != "win32":
            os.chmod(str(kaggle_json), 0o600)
        print(f"[OK] Created {kaggle_json} from environment variables")
    else:
        print("[ERROR] Kaggle credentials not found.")
        print()
        print("Option 1: Set environment variables")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print()
        print("Option 2: Create kaggle.json manually")
        print(f"  mkdir -p {kaggle_dir}")
        print(f"  echo '{{\"username\":\"YOUR_USERNAME\",\"key\":\"YOUR_KEY\"}}' > {kaggle_json}")
        if sys.platform != "win32":
            print(f"  chmod 600 {kaggle_json}")
        print()
        print("Get your API key from: https://www.kaggle.com/settings → API → Create New Token")
        sys.exit(1)

    # Verify Kaggle API works
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[OK] Kaggle API authentication successful")

        # Verify dataset access
        datasets = api.dataset_list(search="creditcardfraud", sort_by="updated")
        if datasets:
            print(f"[OK] Can access Kaggle datasets (found {len(datasets)} results)")
        else:
            print("[WARN] No datasets found, but authentication works")
    except ImportError:
        print("[WARN] kaggle package not installed. Run: pip install kaggle")
    except Exception as e:
        print(f"[ERROR] Kaggle API error: {e}")
        sys.exit(1)

    print("\n[DONE] Kaggle setup complete. Ready for data ingestion.")


if __name__ == "__main__":
    main()
