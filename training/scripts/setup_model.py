#!/usr/bin/env python3
"""
setup_model.py -- One-command model setup for night-road-vision.

Downloads a pretrained YOLO model, exports it to TFLite FP16 at multiple
resolutions, and copies the result into the Android app's assets directory.

Default workflow:
  1. Download yolo26n.pt (falls back to yolo11n.pt if unavailable)
  2. Export TFLite FP16 at 512x320 (balanced)
  3. Copy the .tflite file to the Android assets/models directory

Quick workflow (--quick):
  Same as above but skips multi-resolution exports and only produces
  the 512x320 balanced variant.

Prerequisites:
    pip install ultralytics

Usage:
    python setup_model.py                  # Full: download, export all presets, copy
    python setup_model.py --quick          # Fast: only 512x320 export
    python setup_model.py --skip-copy      # Export without copying to Android assets
    python setup_model.py --model yolo11n  # Force a specific model
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary model to download; fallback is tried if this fails.
PRIMARY_MODEL = "yolo26n.pt"
FALLBACK_MODEL = "yolo11n.pt"

# Resolution presets: (label, height, width)
# The "balanced" preset is always produced; the others are skipped in --quick.
RESOLUTION_PRESETS: List[Tuple[str, int, int]] = [
    ("eco",       256, 416),
    ("balanced",  320, 512),
    ("fine",      384, 640),
]

# The mandatory resolution (always exported even in --quick mode)
MANDATORY_PRESET = ("balanced", 320, 512)

# Android assets destination
ANDROID_ASSETS_DIR = os.path.join(
    "android-app", "app", "src", "main", "assets", "models",
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


def file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def resolve_model_name(requested: Optional[str] = None) -> str:
    """
    Determine which model to use.

    If *requested* is given, use it directly.  Otherwise try PRIMARY_MODEL
    and fall back to FALLBACK_MODEL if the primary cannot be loaded.
    """
    from ultralytics import YOLO

    if requested:
        print(f"Using requested model: {requested}")
        return requested

    # Try the primary model first.
    print(f"Attempting to load primary model: {PRIMARY_MODEL}")
    try:
        model = YOLO(PRIMARY_MODEL)
        # Trigger download / verification; ultralytics caches automatically.
        _ = model.model
        print(f"Primary model loaded successfully: {PRIMARY_MODEL}")
        return PRIMARY_MODEL
    except Exception as exc:
        print(f"Primary model unavailable ({exc}).")

    # Fall back.
    print(f"Falling back to: {FALLBACK_MODEL}")
    try:
        model = YOLO(FALLBACK_MODEL)
        _ = model.model
        print(f"Fallback model loaded successfully: {FALLBACK_MODEL}")
        return FALLBACK_MODEL
    except Exception as exc:
        print(f"ERROR: Neither {PRIMARY_MODEL} nor {FALLBACK_MODEL} could be loaded.", file=sys.stderr)
        print(f"  Last error: {exc}", file=sys.stderr)
        sys.exit(1)


def download_model(model_name: str) -> str:
    """
    Download (or locate cached) YOLO weights via Ultralytics.

    Ultralytics downloads the .pt file automatically on first use.
    Returns the local path to the .pt file.
    """
    from ultralytics import YOLO

    print(f"\n[1/3] Downloading model '{model_name}'...")
    t0 = time.perf_counter()
    model = YOLO(model_name)
    elapsed = time.perf_counter() - t0
    print(f"      Model ready in {elapsed:.1f}s")
    return model_name


def export_tflite_fp16(
    model_name: str,
    imgsz: Tuple[int, int],
    output_dir: str,
    label: str = "balanced",
) -> str:
    """
    Export the model to TFLite FP16 at the specified resolution.

    Returns the absolute path to the exported .tflite file.
    """
    from ultralytics import YOLO

    h, w = imgsz
    print(f"\n  Exporting TFLite FP16  |  {label}  ({w}x{h})...")
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

    # Move to the requested output directory if needed.
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, os.path.basename(export_path))

    if os.path.abspath(export_path) != os.path.abspath(dest):
        shutil.move(export_path, dest)
        export_path = dest

    # Rename to a clean, predictable filename.
    model_stem = Path(model_name).stem  # e.g. "yolo26n"
    final_name = f"{model_stem}_{w}x{h}_fp16.tflite"
    final_path = os.path.join(output_dir, final_name)
    if export_path != final_path:
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(export_path, final_path)
        export_path = final_path

    size_mb = file_size_mb(export_path)
    print(f"    -> {export_path}  ({size_mb:.2f} MB, {elapsed:.1f}s)")

    return export_path


def copy_to_android_assets(
    tflite_path: str,
    android_assets_dir: str,
    also_default_name: bool = True,
) -> str:
    """
    Copy the .tflite file into the Android assets/models directory.

    Also copies under the default name the Android app expects
    (yolo26n_512x320_fp16.tflite) if *also_default_name* is True.

    Returns the destination path.
    """
    print(f"\n[3/3] Copying to Android assets...")
    os.makedirs(android_assets_dir, exist_ok=True)

    dest = os.path.join(android_assets_dir, os.path.basename(tflite_path))
    shutil.copy2(tflite_path, dest)
    print(f"      Copied to: {dest}")

    # Copy as the canonical default model the Android app looks for.
    if also_default_name:
        default_name = "yolo26n_512x320_fp16.tflite"
        default_dest = os.path.join(android_assets_dir, default_name)
        if os.path.abspath(dest) != os.path.abspath(default_dest):
            shutil.copy2(tflite_path, default_dest)
            print(f"      Also as:   {default_dest}")

    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YOLO model, export to TFLite FP16, and deploy to Android assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python setup_model.py                  # Full export (eco/balanced/fine)\n"
            "  python setup_model.py --quick           # Only 512x320 balanced\n"
            "  python setup_model.py --model yolo11n   # Use a specific model\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            f"Ultralytics model name or path. "
            f"Default: try {PRIMARY_MODEL}, fall back to {FALLBACK_MODEL}."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only export the 512x320 balanced preset (skip eco and fine).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Directory to write the exported .tflite files (default: exports/).",
    )
    parser.add_argument(
        "--skip-copy",
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

    # ------------------------------------------------------------------
    # Check ultralytics is installed
    # ------------------------------------------------------------------
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
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Resolve and download the model
    # ------------------------------------------------------------------
    model_name = resolve_model_name(args.model)
    download_model(model_name)

    # ------------------------------------------------------------------
    # Step 2: Export to TFLite FP16
    # ------------------------------------------------------------------
    print(f"\n[2/3] Exporting to TFLite FP16...")
    t_export_start = time.perf_counter()

    if args.quick:
        # Quick mode: only the balanced 512x320 preset.
        presets = [MANDATORY_PRESET]
    else:
        presets = RESOLUTION_PRESETS

    exported_paths: List[str] = []
    for label, h, w in presets:
        tflite_path = export_tflite_fp16(model_name, (h, w), output_dir, label=label)
        exported_paths.append(tflite_path)

    total_export_time = time.perf_counter() - t_export_start
    print(f"\n  Export complete. {len(exported_paths)} model(s) in {total_export_time:.1f}s.")

    # ------------------------------------------------------------------
    # Step 3: Copy to Android assets
    # ------------------------------------------------------------------
    if not args.skip_copy:
        if args.android_assets_dir:
            android_dir = os.path.abspath(args.android_assets_dir)
        else:
            android_dir = str(repo_root / ANDROID_ASSETS_DIR)

        # Copy all exported models.
        for path in exported_paths:
            copy_to_android_assets(path, android_dir, also_default_name=False)

        # Also copy the balanced (512x320) as the canonical default name.
        balanced_path = exported_paths[0] if args.quick else exported_paths[1]
        copy_to_android_assets(balanced_path, android_dir, also_default_name=True)
    else:
        android_dir = None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"  Model source:   {model_name}")
    print(f"  Export dir:     {output_dir}")
    for p in exported_paths:
        print(f"  TFLite FP16:    {p}  ({file_size_mb(p):.2f} MB)")
    if android_dir:
        print(f"  Android assets: {android_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
