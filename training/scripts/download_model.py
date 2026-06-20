#!/usr/bin/env python3
"""
download_model.py -- Download a YOLO26n model, export it to TFLite FP16,
and copy the result into the Android assets directory.

This is a convenience placeholder script for quickly obtaining a deployable
TFLite model before a custom training run is available.  It downloads the
official yolo26n.pt weights from Ultralytics, exports them to TFLite FP16
at 512x320 resolution, and places the .tflite file where the Android app
expects it.

Prerequisites:
    pip install ultralytics

Usage:
    python download_model.py
    python download_model.py --output-dir exports/
    python download_model.py --skip-android-copy
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# The Ultralytics model to download (will be fetched automatically)
DEFAULT_MODEL_NAME = "yolo26n.pt"

# Export resolution (height, width) -- balanced preset for mobile
EXPORT_HEIGHT = 320
EXPORT_WIDTH = 512

# Relative path from the repo root to the Android assets directory
ANDROID_ASSETS_REL = os.path.join(
    "android-app", "app", "src", "main", "assets", "models"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_repo_root() -> Path:
    """Walk up from this script to locate the night-road-vision root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "android-app").is_dir() and (current / "training").is_dir():
            return current
        current = current.parent
    # Fallback: assume script is at training/scripts/
    return Path(__file__).resolve().parent.parent.parent


def download_model(model_name: str) -> str:
    """
    Download (or locate cached) YOLO weights via Ultralytics.

    Ultralytics automatically downloads the model on first use.
    Returns the local path to the .pt file.
    """
    from ultralytics import YOLO

    print(f"[1/3] Loading model '{model_name}' (will download if not cached)...")
    t0 = time.perf_counter()
    model = YOLO(model_name)
    elapsed = time.perf_counter() - t0
    print(f"      Model loaded in {elapsed:.1f}s")
    return model_name


def export_tflite_fp16(model_name: str, imgsz: tuple, output_dir: str) -> str:
    """
    Export the model to TFLite FP16 at the specified resolution.

    Returns the absolute path to the exported .tflite file.
    """
    from ultralytics import YOLO

    h, w = imgsz
    print(f"\n[2/3] Exporting to TFLite FP16 at {w}x{h}...")
    t0 = time.perf_counter()

    model = YOLO(model_name)
    export_path = model.export(
        format="tflite",
        imgsz=(h, w),
        half=True,
        int8=False,
    )
    elapsed = time.perf_counter() - t0

    export_path = str(export_path)

    # Move to the requested output directory
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, os.path.basename(export_path))

    if os.path.abspath(export_path) != os.path.abspath(dest):
        shutil.move(export_path, dest)
        export_path = dest

    # Rename to a clean, predictable filename
    final_name = f"yolo26n_{w}x{h}_fp16.tflite"
    final_path = os.path.join(output_dir, final_name)
    if export_path != final_path:
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(export_path, final_path)
        export_path = final_path

    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"      Exported: {export_path}")
    print(f"      Size: {size_mb:.2f} MB  |  Time: {elapsed:.1f}s")

    return export_path


def copy_to_android_assets(tflite_path: str, android_assets_dir: str) -> str:
    """
    Copy the .tflite file into the Android assets/models directory.

    Returns the destination path.
    """
    print(f"\n[3/3] Copying to Android assets...")
    os.makedirs(android_assets_dir, exist_ok=True)

    dest = os.path.join(android_assets_dir, os.path.basename(tflite_path))
    shutil.copy2(tflite_path, dest)

    # Also copy as the default model name the app expects
    default_name = "yolov8n_float32.tflite"
    default_dest = os.path.join(android_assets_dir, default_name)
    shutil.copy2(tflite_path, default_dest)

    print(f"      Copied to: {dest}")
    print(f"      Also as:   {default_dest}")
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO26n, export to TFLite FP16, and deploy to Android assets."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Ultralytics model name or path (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Directory to write the exported .tflite file (default: exports/).",
    )
    parser.add_argument(
        "--width", type=int, default=EXPORT_WIDTH,
        help=f"Export width in pixels (default: {EXPORT_WIDTH}).",
    )
    parser.add_argument(
        "--height", type=int, default=EXPORT_HEIGHT,
        help=f"Export height in pixels (default: {EXPORT_HEIGHT}).",
    )
    parser.add_argument(
        "--skip-android-copy",
        action="store_true",
        help="Do not copy the .tflite file to the Android assets directory.",
    )
    parser.add_argument(
        "--android-assets-dir",
        type=str,
        default=None,
        help="Override the Android assets directory path.",
    )
    args = parser.parse_args()

    # Check ultralytics
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("ERROR: ultralytics is not installed.", file=sys.stderr)
        print("Install with: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    repo_root = find_repo_root()
    print(f"Repository root: {repo_root}")

    output_dir = os.path.abspath(args.output_dir)
    imgsz = (args.height, args.width)

    # Step 1: Download
    model_path = download_model(args.model)

    # Step 2: Export
    tflite_path = export_tflite_fp16(model_path, imgsz, output_dir)

    # Step 3: Copy to Android assets
    if not args.skip_android_copy:
        if args.android_assets_dir:
            android_dir = os.path.abspath(args.android_assets_dir)
        else:
            android_dir = str(repo_root / ANDROID_ASSETS_REL)
        copy_to_android_assets(tflite_path, android_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  TFLite model: {tflite_path}")
    if not args.skip_android_copy:
        print(f"  Android assets: {android_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
