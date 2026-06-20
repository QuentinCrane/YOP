#!/usr/bin/env python3
"""
Dataset download helper for night road detection training.

Some datasets require manual registration/download. This script provides:
1. Direct download for freely available datasets
2. Clear instructions and links for datasets requiring registration
3. Verification of downloaded datasets

Usage:
    python download_datasets.py --check        # Check what's already downloaded
    python download_datasets.py --exdark       # Try to download ExDark
    python download_datasets.py --instructions # Show download instructions
    python download_datasets.py --all          # Download everything possible
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "raw_datasets"

# Dataset info
DATASETS = {
    "bdd100k": {
        "name": "BDD100K",
        "url": "https://bdd-data.berkeley.edu/",
        "size": "~100GB",
        "requires_registration": True,
        "files_needed": ["images/100k/", "labels/"],
        "description": "Berkeley DeepDrive - largest diverse driving video dataset",
    },
    "nightowls": {
        "name": "NightOwls",
        "url": "https://www.nightowls-dataset.org/",
        "size": "~5GB",
        "requires_registration": False,
        "files_needed": ["images/", "annotations/"],
        "description": "Nighttime pedestrian detection dataset (279k frames)",
    },
    "exdark": {
        "name": "ExDark",
        "url": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        "size": "~1.5GB",
        "requires_registration": False,
        "files_needed": ["Images/", "Annotations/"],
        "description": "Exclusively Dark image dataset (7,363 low-light images)",
    },
    "nod": {
        "name": "NOD",
        "url": "https://arxiv.org/abs/2110.10364",
        "size": "~2GB",
        "requires_registration": True,
        "files_needed": ["images/", "annotations/"],
        "description": "Night Object Detection dataset",
    },
}


def check_downloads():
    """Check which datasets are already downloaded."""
    print("=" * 60)
    print("Dataset Download Status")
    print("=" * 60)

    for key, info in DATASETS.items():
        dataset_dir = RAW_DIR / key
        if dataset_dir.exists():
            files = list(dataset_dir.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            if file_count > 0:
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"\n[OK] {info['name']}")
                print(f"     Location: {dataset_dir}")
                print(f"     Files: {file_count}")
                print(f"     Size: {size_mb:.1f} MB")
            else:
                print(f"\n[EMPTY] {info['name']}")
                print(f"        Directory exists but no files found")
        else:
            print(f"\n[MISSING] {info['name']}")
            print(f"          URL: {info['url']}")
            print(f"          Size: {info['size']}")
            if info['requires_registration']:
                print(f"          *** Requires registration ***")


def show_instructions():
    """Show download instructions for all datasets."""
    print("=" * 60)
    print("Dataset Download Instructions")
    print("=" * 60)

    for key, info in DATASETS.items():
        print(f"\n{'─' * 40}")
        print(f"Dataset: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Size: {info['size']}")
        print(f"URL: {info['url']}")

        if info['requires_registration']:
            print(f"\n⚠️  This dataset requires registration!")
            print(f"   1. Visit: {info['url']}")
            print(f"   2. Create an account and agree to terms")
            print(f"   3. Download the dataset")
            print(f"   4. Extract to: {RAW_DIR / key}/")
        else:
            print(f"\n📥 Direct download available")
            print(f"   Download from: {info['url']}")
            print(f"   Extract to: {RAW_DIR / key}/")

        print(f"\n   Expected structure:")
        for f in info['files_needed']:
            print(f"     {RAW_DIR / key / f}")


def download_exdark():
    """Attempt to download ExDark dataset."""
    print("Attempting to download ExDark dataset...")
    print("Note: GitHub raw file downloads may require browser access.")
    print(f"\nPlease download manually from:")
    print(f"  https://github.com/cs-chan/Exclusively-Dark-Image-Dataset")
    print(f"\nOr use git clone:")
    print(f"  git clone https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.git {RAW_DIR / 'exdark_repo'}")


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--check", action="store_true", help="Check download status")
    parser.add_argument("--exdark", action="store_true", help="Download ExDark")
    parser.add_argument("--instructions", action="store_true", help="Show instructions")
    parser.add_argument("--all", action="store_true", help="Download all possible")

    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.check or args.all:
        check_downloads()

    if args.instructions or args.all:
        show_instructions()

    if args.exdark or args.all:
        download_exdark()

    if not any([args.check, args.exdark, args.instructions, args.all]):
        check_downloads()
        print("\nUse --instructions for detailed download guide")
        print("Use --exdark to attempt ExDark download")


if __name__ == "__main__":
    main()
