#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMS Spam Dataset Downloader
Automatically downloads the SMS Spam Collection dataset from Kaggle
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if not kaggle_json.exists():
        print("=" * 70)
        print("⚠️  Kaggle API credentials not found!")
        print("=" * 70)
        print("\nTo download datasets from Kaggle, you need to set up API credentials:")
        print("\n1. Go to https://www.kaggle.com/")
        print("2. Sign in or create an account")
        print("3. Go to Account Settings (click on your profile → Settings)")
        print("4. Scroll to 'API' section")
        print("5. Click 'Create New Token' - this downloads kaggle.json")
        print("\n6. Place kaggle.json in one of these locations:")
        print(f"   - {kaggle_dir / 'kaggle.json'}")
        print(f"   - Current directory: {Path.cwd() / 'kaggle.json'}")
        print("\n7. On Linux/Mac, set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\n8. Then run this script again!")
        print("=" * 70)
        return False

    return True


def download_from_kaggle():
    """Download SMS Spam dataset from Kaggle"""
    print("\n" + "=" * 70)
    print("📦 Downloading SMS Spam Collection Dataset from Kaggle")
    print("=" * 70)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated successfully")

        # Create data directory if it doesn't exist
        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        dataset_name = 'uciml/sms-spam-collection-dataset'
        print(f"\n📥 Downloading dataset: {dataset_name}")
        print("   This may take a few moments...")

        api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )

        print("✓ Dataset downloaded successfully")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading from Kaggle: {e}")
        return False


def download_alternative():
    """Try alternative download methods"""
    print("\n" + "=" * 70)
    print("🌐 Trying alternative download method...")
    print("=" * 70)

    try:
        import urllib.request

        # Alternative URL (raw GitHub mirror)
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)

        output_file = data_dir / 'spam.csv'

        print(f"📥 Downloading from: {url}")
        urllib.request.urlretrieve(url, output_file)

        # Verify the download
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"✓ Dataset downloaded to: {output_file}")
            return True
        else:
            print("❌ Download failed - file is empty or missing")
            return False

    except Exception as e:
        print(f"❌ Alternative download failed: {e}")
        return False


def verify_dataset():
    """Verify that the dataset was downloaded and is valid"""
    print("\n" + "=" * 70)
    print("🔍 Verifying dataset...")
    print("=" * 70)

    data_file = Path('data/raw/spam.csv')

    if not data_file.exists():
        print(f"❌ Dataset file not found: {data_file}")
        return False

    try:
        # Try to load the dataset
        df = pd.read_csv(data_file, encoding='latin-1')

        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - File: {data_file}")
        print(f"  - Rows: {len(df):,}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - File size: {data_file.stat().st_size / 1024:.2f} KB")

        # Show first few rows
        print("\n📊 Dataset preview:")
        print(df.head())

        # Check column names
        print(f"\n📋 Columns: {list(df.columns)}")

        return True

    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False


def main():
    """Main function to download and verify dataset"""
    print("\n" + "=" * 70)
    print("🚀 SMS Spam Dataset Download Script")
    print("=" * 70)

    # Check if dataset already exists
    data_file = Path('data/raw/spam.csv')
    if data_file.exists():
        print(f"\n✓ Dataset already exists: {data_file}")
        response = input("\nDo you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("\n✓ Using existing dataset")
            verify_dataset()
            return

    # Method 1: Try Kaggle API
    if check_kaggle_credentials():
        if download_from_kaggle():
            if verify_dataset():
                print("\n" + "=" * 70)
                print("✅ Dataset download complete!")
                print("=" * 70)
                print("\nYou can now run the training notebooks or scripts.")
                print("Next steps:")
                print("  - Run: jupyter notebook")
                print("  - Or see: TRAINING.md")
                return

    # Method 2: Try alternative download
    print("\n💡 Trying alternative download method...")
    if download_alternative():
        if verify_dataset():
            print("\n" + "=" * 70)
            print("✅ Dataset download complete!")
            print("=" * 70)
            print("\nYou can now run the training notebooks or scripts.")
            print("Next steps:")
            print("  - Run: jupyter notebook")
            print("  - Or see: TRAINING.md")
            return

    # Method 3: Manual instructions
    print("\n" + "=" * 70)
    print("❌ Automatic download failed")
    print("=" * 70)
    print("\n📝 Manual download instructions:")
    print("\n1. Visit: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Extract the zip file")
    print("4. Move 'spam.csv' to: data/raw/spam.csv")
    print("\nOr try:")
    print("1. Visit: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
    print("2. Download the dataset")
    print("3. Extract and rename to 'spam.csv'")
    print("4. Place in: data/raw/spam.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
