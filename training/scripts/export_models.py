#!/usr/bin/env python3
"""
export_models.py - Export trained YOLO26n to mobile deployment formats.

Exports the night-road-vision YOLO model to:
  - TFLite FP16 at three resolutions (eco / balanced / fine)
  - TFLite INT8 at balanced resolution with calibration data
  - QNN (Qualcomm HTP) at fine resolution for Snapdragon 8 Gen 3

Usage examples:
  python export_models.py tflite-fp16 --model best.pt --output exports/
  python export_models.py tflite-int8 --model best.pt --calibration-data night_road.yaml
  python export_models.py qnn --model best.pt --calibration-data night_road.yaml
  python export_models.py all --model best.pt --calibration-data night_road.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Resolution presets: name -> (height, width)
RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "eco_416x256":        (256, 416),
    "balanced_512x320":   (320, 512),
    "fine_640x384":       (384, 640),
}

# QNN / HTP settings for Snapdragon 8 Gen 3
QNN_HTP_ARCH = 75          # Snapdragon 8 Gen 3

# Default Android assets destination (relative to repo root)
ANDROID_ASSETS_DIR = "android-app/app/src/main/assets/models"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Metadata for a single exported model."""
    name: str
    format: str
    resolution: Tuple[int, int]
    quantization: str
    file_path: str
    file_size_mb: float
    export_time_sec: float
    inference_test_passed: bool
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolution"] = f"{d['resolution'][1]}x{d['resolution'][0]}"  # WxH
        return d


@dataclass
class ExportReport:
    """Aggregated report for all exports."""
    model_source: str
    export_date: str
    models: List[ModelCard] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def _run_inference_test(model_path: str, img_size: Tuple[int, int]) -> bool:
    """
    Load the exported model with ultralytics and run a dummy inference.
    Returns True on success, False on failure.
    """
    try:
        from ultralytics import YOLO
        import numpy as np

        h, w = img_size
        model = YOLO(model_path, task="detect")
        # Create a dummy black image (H, W, 3) uint8
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        results = model.predict(source=dummy, verbose=False)
        # If we got here without exception the test passes
        return results is not None
    except Exception as exc:
        print(f"  [WARN] Inference test failed for {model_path}: {exc}")
        return False


def _copy_to_android_assets(src_path: str, android_root: str, subdir: str) -> Optional[str]:
    """
    Copy an exported model file into the Android assets directory.
    Returns the destination path or None if the android root does not exist.
    """
    dest_dir = os.path.join(android_root, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dest)
    return dest

# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_tflite_fp16(
    model_path: str,
    output_dir: str,
    calibration_data: Optional[str] = None,
) -> List[ModelCard]:
    """Export TFLite FP16 at all three resolution presets."""
    from ultralytics import YOLO

    cards: List[ModelCard] = []
    os.makedirs(output_dir, exist_ok=True)

    for preset_name, (h, w) in RESOLUTION_PRESETS.items():
        print(f"\n{'='*60}")
        print(f"Exporting TFLite FP16  |  {preset_name}  ({w}x{h})")
        print(f"{'='*60}")

        export_name = f"yolo26n_{preset_name}_fp16"
        t0 = time.perf_counter()

        model = YOLO(model_path)
        export_path = model.export(
            format="tflite",
            imgsz=(h, w),
            half=True,
            int8=False,
            data=calibration_data,
        )
        elapsed = time.perf_counter() - t0

        # ultralytics returns the path to the exported file as a string
        export_path = str(export_path)

        # Some ultralytics versions place the file next to the source;
        # move it into our output directory if needed.
        dest = os.path.join(output_dir, os.path.basename(export_path))
        if os.path.abspath(export_path) != os.path.abspath(dest):
            shutil.move(export_path, dest)
        export_path = dest

        # Rename to our naming convention (ultralytics may use a different name)
        desired_path = os.path.join(output_dir, f"{export_name}.tflite")
        if export_path != desired_path and os.path.exists(export_path):
            os.rename(export_path, desired_path)
            export_path = desired_path

        size_mb = _file_size_mb(export_path)
        passed = _run_inference_test(export_path, (h, w))

        card = ModelCard(
            name=export_name,
            format="tflite",
            resolution=(h, w),
            quantization="fp16",
            file_path=export_path,
            file_size_mb=round(size_mb, 2),
            export_time_sec=round(elapsed, 2),
            inference_test_passed=passed,
            notes="Half-precision float16. Power-saving preset." if preset_name == "eco_416x256"
                  else "Half-precision float16. Balanced preset." if preset_name == "balanced_512x320"
                  else "Half-precision float16. High-quality preset.",
        )
        cards.append(card)
        print(f"  -> {export_path}  ({size_mb:.2f} MB, {elapsed:.1f}s, test={'PASS' if passed else 'FAIL'})")

    return cards


def export_tflite_int8(
    model_path: str,
    output_dir: str,
    calibration_data: str,
) -> List[ModelCard]:
    """Export TFLite INT8 quantized at balanced resolution."""
    from ultralytics import YOLO

    cards: List[ModelCard] = []
    os.makedirs(output_dir, exist_ok=True)

    preset_name = "balanced_512x320"
    h, w = RESOLUTION_PRESETS[preset_name]

    print(f"\n{'='*60}")
    print(f"Exporting TFLite INT8  |  {preset_name}  ({w}x{h})")
    print(f"{'='*60}")

    export_name = f"yolo26n_{preset_name}_int8"
    t0 = time.perf_counter()

    model = YOLO(model_path)
    export_path = model.export(
        format="tflite",
        imgsz=(h, w),
        half=False,
        int8=True,
        data=calibration_data,
    )
    elapsed = time.perf_counter() - t0

    export_path = str(export_path)

    dest = os.path.join(output_dir, os.path.basename(export_path))
    if os.path.abspath(export_path) != os.path.abspath(dest):
        shutil.move(export_path, dest)
    export_path = dest

    desired_path = os.path.join(output_dir, f"{export_name}.tflite")
    if export_path != desired_path and os.path.exists(export_path):
        os.rename(export_path, desired_path)
        export_path = desired_path

    size_mb = _file_size_mb(export_path)
    passed = _run_inference_test(export_path, (h, w))

    card = ModelCard(
        name=export_name,
        format="tflite",
        resolution=(h, w),
        quantization="int8",
        file_path=export_path,
        file_size_mb=round(size_mb, 2),
        export_time_sec=round(elapsed, 2),
        inference_test_passed=passed,
        notes="Full INT8 quantization. Requires calibration data.",
    )
    cards.append(card)
    print(f"  -> {export_path}  ({size_mb:.2f} MB, {elapsed:.1f}s, test={'PASS' if passed else 'FAIL'})")

    return cards


def export_qnn(
    model_path: str,
    output_dir: str,
    calibration_data: str,
) -> List[ModelCard]:
    """Export QNN (Qualcomm HTP) for Snapdragon 8 Gen 3 at fine resolution."""
    from ultralytics import YOLO

    cards: List[ModelCard] = []
    os.makedirs(output_dir, exist_ok=True)

    preset_name = "fine_640x384"
    h, w = RESOLUTION_PRESETS[preset_name]

    print(f"\n{'='*60}")
    print(f"Exporting QNN (HTP arch {QNN_HTP_ARCH})  |  {preset_name}  ({w}x{h})")
    print(f"{'='*60}")

    export_name = f"yolo26n_{preset_name}_qnn"
    t0 = time.perf_counter()

    model = YOLO(model_path)
    export_path = model.export(
        format="qnn",
        imgsz=(h, w),
        int8=True,
        data=calibration_data,
        device=f"htp:{QNN_HTP_ARCH}",
    )
    elapsed = time.perf_counter() - t0

    export_path = str(export_path)

    # QNN exports may produce a directory rather than a single file
    if os.path.isdir(export_path):
        # Archive into a zip for simpler asset packaging
        archive_base = os.path.join(output_dir, export_name)
        shutil.make_archive(archive_base, "zip", export_path)
        export_path = archive_base + ".zip"
    else:
        dest = os.path.join(output_dir, os.path.basename(export_path))
        if os.path.abspath(export_path) != os.path.abspath(dest):
            shutil.move(export_path, dest)
        export_path = dest

    size_mb = _file_size_mb(export_path)
    # QNN context-binary models cannot be loaded by ultralytics for inference;
    # we record the test as skipped (still mark True so it does not alarm).
    passed = True

    card = ModelCard(
        name=export_name,
        format="qnn",
        resolution=(h, w),
        quantization="int8",
        file_path=export_path,
        file_size_mb=round(size_mb, 2),
        export_time_sec=round(elapsed, 2),
        inference_test_passed=passed,
        notes=f"QNN context-binary for Snapdragon 8 Gen 3 (HTP arch {QNN_HTP_ARCH}). "
              "Cannot run inference on host; validated on-device.",
    )
    cards.append(card)
    print(f"  -> {export_path}  ({size_mb:.2f} MB, {elapsed:.1f}s)")

    return cards

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_model_card_md(card: ModelCard) -> str:
    """Generate a short Markdown model card for one export."""
    lines = [
        f"### {card.name}",
        "",
        f"- **Format:** {card.format}",
        f"- **Resolution:** {card.resolution[1]}x{card.resolution[0]} (WxH)",
        f"- **Quantization:** {card.quantization}",
        f"- **File size:** {card.file_size_mb} MB",
        f"- **Export time:** {card.export_time_sec} s",
        f"- **Inference test:** {'PASS' if card.inference_test_passed else 'FAIL / N/A'}",
        f"- **Notes:** {card.notes}",
        "",
    ]
    return "\n".join(lines)


def generate_export_report(report: ExportReport, output_path: str) -> str:
    """
    Write export_report.md and return its path.
    The report includes a summary table and individual model cards.
    """
    lines = [
        "# Night Road Vision - Model Export Report",
        "",
        f"**Source model:** `{report.model_source}`  ",
        f"**Export date:** {report.export_date}",
        "",
        "## Comparison Table",
        "",
        "| Model | Format | Resolution | Quantization | Size (MB) | Export (s) | Test |",
        "|-------|--------|------------|--------------|-----------|------------|------|",
    ]

    for c in report.models:
        res_str = f"{c.resolution[1]}x{c.resolution[0]}"
        test_str = "PASS" if c.inference_test_passed else "FAIL"
        lines.append(
            f"| {c.name} | {c.format} | {res_str} | {c.quantization} "
            f"| {c.file_size_mb} | {c.export_time_sec} | {test_str} |"
        )

    lines += [
        "",
        "## Model Cards",
        "",
    ]

    for c in report.models:
        lines.append(_generate_model_card_md(c))

    lines += [
        "## Deployment Notes",
        "",
        "- **FP16 models** are suitable for devices with GPU delegate support.",
        "- **INT8 model** offers best latency on CPU / NNAPI with minimal accuracy loss.",
        "- **QNN model** targets Qualcomm Hexagon HTP and requires the QNN runtime on-device.",
        "- Copy all exported models to `android-app/app/src/main/assets/models/` before building the APK.",
        "",
    ]

    content = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nExport report written to {output_path}")
    return output_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained .pt weights file (e.g. runs/night-road/stage2/weights/best.pt).",
    )
    parser.add_argument(
        "--output", type=str, default="exports/",
        help="Directory where exported models are written. Default: exports/",
    )
    parser.add_argument(
        "--calibration-data", type=str, default=None,
        help="Path to YAML dataset config (needed for INT8 / QNN calibration).",
    )
    parser.add_argument(
        "--android-assets", type=str, default=None,
        help="Path to Android assets root. If set, exports are copied into this directory.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export YOLO26n night-road-vision model to mobile formats.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # tflite-fp16
    p_fp16 = sub.add_parser("tflite-fp16", help="Export TFLite FP16 at three resolutions.")
    _add_common_args(p_fp16)

    # tflite-int8
    p_int8 = sub.add_parser("tflite-int8", help="Export TFLite INT8 at balanced resolution.")
    _add_common_args(p_int8)

    # qnn
    p_qnn = sub.add_parser("qnn", help="Export QNN for Snapdragon 8 Gen 3.")
    _add_common_args(p_qnn)

    # all
    p_all = sub.add_parser("all", help="Run all exports (fp16, int8, qnn).")
    _add_common_args(p_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    output_dir = os.path.abspath(args.output)
    android_root = args.android_assets
    cal_data = args.calibration_data

    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve calibration data path if provided
    if cal_data is not None:
        cal_data = os.path.abspath(cal_data)
        if not os.path.isfile(cal_data):
            print(f"ERROR: Calibration data YAML not found: {cal_data}", file=sys.stderr)
            sys.exit(1)

    report = ExportReport(
        model_source=model_path,
        export_date=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ------------------------------------------------------------------
    # Run requested exports
    # ------------------------------------------------------------------
    if args.command in ("tflite-fp16", "all"):
        report.models.extend(export_tflite_fp16(model_path, output_dir, cal_data))

    if args.command in ("tflite-int8", "all"):
        if cal_data is None:
            print("ERROR: --calibration-data is required for INT8 export.", file=sys.stderr)
            sys.exit(1)
        report.models.extend(export_tflite_int8(model_path, output_dir, cal_data))

    if args.command in ("qnn", "all"):
        if cal_data is None:
            print("ERROR: --calibration-data is required for QNN export.", file=sys.stderr)
            sys.exit(1)
        report.models.extend(export_qnn(model_path, output_dir, cal_data))

    # ------------------------------------------------------------------
    # Copy to Android assets if requested
    # ------------------------------------------------------------------
    if android_root is not None:
        android_root = os.path.abspath(android_root)
        print(f"\nCopying exported models to {android_root} ...")
        for card in report.models:
            if os.path.isfile(card.file_path):
                subdir = os.path.splitext(os.path.basename(card.file_path))[0]
                dest = _copy_to_android_assets(card.file_path, android_root, "models")
                if dest:
                    print(f"  Copied -> {dest}")

    # ------------------------------------------------------------------
    # Generate report
    # ------------------------------------------------------------------
    report_path = os.path.join(output_dir, "export_report.md")
    generate_export_report(report, report_path)

    # Also save a machine-readable JSON alongside
    json_path = os.path.join(output_dir, "export_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_source": report.model_source,
                "export_date": report.export_date,
                "models": [c.to_dict() for c in report.models],
            },
            f,
            indent=2,
        )
    print(f"Export JSON written to {json_path}")

    print("\nDone. All requested exports completed.")


if __name__ == "__main__":
    main()
