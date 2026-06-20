#!/usr/bin/env python3
"""
Two-stage training pipeline for YOLO26n on night road detection data.

Optimized for RTX 3090 24GB.

Stage 1: Public data pre-adaptation (BDD100K night + NOD + NightOwls + ExDark)
Stage 2: Phone-captured data fine-tune with mixed public + phone data

Usage:
    python train.py stage1 --data datasets/night_road/night_road.yaml
    python train.py stage2 --data datasets/night_road_phone/night_road.yaml \
                            --stage1-weights runs/night-road/stage1_public/weights/best.pt
    python train.py both --data-stage1 datasets/night_road/night_road.yaml \
                          --data-stage2 datasets/night_road_phone/night_road.yaml
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("night-road-train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PROJECT = "runs/night-road"
STAGE1_NAME = "stage1_public"
STAGE2_NAME = "stage2_finetune"

# RTX 3090 batch-size calibration range (nano model @ 640px)
BATCH_CALIBRATION_START = 48
BATCH_CALIBRATION_MIN = 8
BATCH_CALIBRATION_STEP = 8  # step down by 8 each failure

# Night-scene augmentation defaults
NIGHT_AUGMENTATION = {
    "hsv_h": 0.015,       # hue shift (subtle)
    "hsv_s": 0.5,         # saturation boost (street lights)
    "hsv_v": 0.4,         # value/brightness augmentation (headlights)
    "degrees": 5.0,       # rotation
    "translate": 0.1,     # translation
    "scale": 0.3,         # scale variation
    "shear": 2.0,         # shear
    "perspective": 0.0,   # no perspective for dashcam
    "flipud": 0.0,        # no vertical flip (road scenes)
    "fliplr": 0.5,        # horizontal flip OK
    "mosaic": 1.0,        # mosaic augmentation
    "mixup": 0.1,         # light mixup
    "copy_paste": 0.0,    # no copy-paste
    "erasing": 0.2,       # random erasing (partial occlusion)
    "crop_fraction": 0.1, # center crop fraction
}

# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def get_gpu_memory_info() -> dict:
    """Return GPU memory info in MB via nvidia-smi."""
    info = {"total": 0, "used": 0, "free": 0, "name": "Unknown"}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,name",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        ).strip()
        parts = [p.strip() for p in output.split(",")]
        if len(parts) >= 4:
            info["total"] = int(parts[0])
            info["used"] = int(parts[1])
            info["free"] = int(parts[2])
            info["name"] = parts[3]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        logger.warning("Could not query GPU memory: %s", exc)
    return info


def print_gpu_status(label: str = ""):
    """Print formatted GPU memory status."""
    info = get_gpu_memory_info()
    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sGPU: %s | Memory: %d MB total, %d MB used, %d MB free",
        prefix,
        info["name"],
        info["total"],
        info["used"],
        info["free"],
    )
    return info


# ---------------------------------------------------------------------------
# Batch-size auto-calibration
# ---------------------------------------------------------------------------

def calibrate_batch_size(
    model_path: str,
    data_yaml: str,
    imgsz: int = 640,
    start_batch: int = BATCH_CALIBRATION_START,
    min_batch: int = BATCH_CALIBRATION_MIN,
    step: int = BATCH_CALIBRATION_STEP,
) -> int:
    """
    Find the largest batch size that fits in GPU memory.

    Runs a short training (1 epoch, few iterations) starting from
    `start_batch` and steps down by `step` until it succeeds or
    reaches `min_batch`.
    """
    from ultralytics import YOLO

    logger.info("Starting batch-size calibration (start=%d, min=%d, step=%d)", start_batch, min_batch, step)
    print_gpu_status("before calibration")

    batch = start_batch
    while batch >= min_batch:
        logger.info("Testing batch_size=%d ...", batch)
        try:
            model = YOLO(model_path)
            model.train(
                data=data_yaml,
                epochs=1,
                imgsz=imgsz,
                batch=batch,
                amp=True,
                workers=4,       # fewer workers for calibration
                cache=False,     # no cache during calibration
                verbose=False,
                plots=False,
                save=False,
                exist_ok=True,
                max_det=50,
                # Limit iterations for speed -- use a tiny subset if possible
                # Ultralytics doesn't have a direct "max_steps" param, so we
                # rely on the epoch finishing quickly with small dataset subsets
            )
            logger.info("batch_size=%d succeeded.", batch)
            print_gpu_status("after calibration")
            return batch
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                logger.warning("batch_size=%d caused OOM. Reducing...", batch)
                # Free GPU memory
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                batch -= step
            else:
                raise
        finally:
            # Clean up model to free VRAM
            try:
                del model
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    logger.warning(
        "Could not find a working batch >= %d. Using minimum batch=%d.",
        min_batch,
        min_batch,
    )
    return min_batch


# ---------------------------------------------------------------------------
# Training config persistence
# ---------------------------------------------------------------------------

def save_training_config(save_dir: str, stage: str, config: dict):
    """Save the full training configuration alongside the weights."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    config_file = save_path / "training_config.json"

    config_record = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "gpu": get_gpu_memory_info(),
        "config": config,
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2, default=str)

    logger.info("Training config saved to %s", config_file)
    return config_file


# ---------------------------------------------------------------------------
# Stage 1: Public data pre-adaptation
# ---------------------------------------------------------------------------

def run_stage1(args) -> str:
    """
    Stage 1 -- Pre-train on public night-road datasets.

    Returns path to best.pt weights.
    """
    from ultralytics import YOLO

    logger.info("=" * 70)
    logger.info("STAGE 1: Public data pre-adaptation")
    logger.info("=" * 70)

    print_gpu_status("Stage 1 start")

    # Determine model source
    model_source = args.model if args.model else "yolo26n.pt"
    logger.info("Base model: %s", model_source)

    # Determine project/run directory
    project = args.project if args.project else DEFAULT_PROJECT
    run_name = args.name if args.name else STAGE1_NAME

    # Auto-calibrate batch size or use user-provided value
    if args.batch == -1:
        logger.info("Batch size set to -1 (auto). Running calibration...")
        batch_size = calibrate_batch_size(
            model_path=model_source,
            data_yaml=args.data,
            imgsz=args.imgsz,
            start_batch=BATCH_CALIBRATION_START,
            min_batch=BATCH_CALIBRATION_MIN,
        )
        logger.info("Calibrated batch size: %d", batch_size)
    else:
        batch_size = args.batch

    # Resume or fresh start
    resume_path = None
    if args.resume:
        resume_path = Path(project) / run_name / "weights" / "last.pt"
        if resume_path.exists():
            logger.info("Resuming training from %s", resume_path)
            model_source = str(resume_path)
        else:
            logger.warning("Resume requested but %s not found. Starting fresh.", resume_path)

    # Build training kwargs
    stage1_config = {
        "data": args.data,
        "epochs": args.epochs if args.epochs else 120,
        "imgsz": args.imgsz,
        "batch": batch_size,
        "amp": True,
        "optimizer": "auto",
        "patience": 30,
        "close_mosaic": 15,
        "cos_lr": True,
        "cache": "disk",
        "workers": args.workers,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": -1,  # only save best and last
        "plots": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "multi_scale": False,
        # Night-scene augmentation
        **NIGHT_AUGMENTATION,
    }

    # Logging
    if args.logger == "wandb":
        stage1_config["project"] = project  # wandb uses project
        os.environ["WANDB_PROJECT"] = "night-road-vision"
    elif args.logger == "tensorboard":
        pass  # tensorboard is default in ultralytics

    # Save config before training
    save_dir_1 = Path(project) / run_name
    save_training_config(str(save_dir_1), "stage1_public", stage1_config)

    # Load model
    logger.info("Loading model: %s", model_source)
    model = YOLO(model_source)

    # Train
    logger.info("Starting Stage 1 training for %d epochs, batch=%d, imgsz=%d",
                stage1_config["epochs"], batch_size, args.imgsz)
    print_gpu_status("Stage 1 pre-train")

    results = model.train(**stage1_config)

    # Locate best weights
    best_pt = save_dir_1 / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback: check runs directory
        alt = Path(project) / run_name / "weights" / "best.pt"
        if alt.exists():
            best_pt = alt
        else:
            logger.error("best.pt not found in %s/weights/", save_dir_1)
            raise FileNotFoundError(f"Stage 1 best.pt not found at {best_pt}")

    logger.info("Stage 1 complete. Best weights: %s", best_pt)
    print_gpu_status("Stage 1 end")

    return str(best_pt)


# ---------------------------------------------------------------------------
# Stage 2: Phone data fine-tune
# ---------------------------------------------------------------------------

def run_stage2(args, stage1_weights: str = None) -> str:
    """
    Stage 2 -- Fine-tune on phone-captured data mixed with public data.

    Returns path to best.pt weights.
    """
    from ultralytics import YOLO

    logger.info("=" * 70)
    logger.info("STAGE 2: Phone data fine-tune")
    logger.info("=" * 70)

    print_gpu_status("Stage 2 start")

    # Determine model source -- prefer stage1 weights
    if stage1_weights:
        model_source = stage1_weights
    elif args.stage1_weights:
        model_source = args.stage1_weights
    elif args.model:
        model_source = args.model
    else:
        # Default: look for stage1 output
        default_s1 = Path(args.project if args.project else DEFAULT_PROJECT) / STAGE1_NAME / "weights" / "best.pt"
        if default_s1.exists():
            model_source = str(default_s1)
            logger.info("Auto-detected Stage 1 weights: %s", model_source)
        else:
            model_source = "yolo26n.pt"
            logger.warning("No Stage 1 weights found. Using base model: %s", model_source)

    # Determine project/run directory
    project = args.project if args.project else DEFAULT_PROJECT
    run_name = getattr(args, "name_stage2", None) or (args.name if hasattr(args, "name") and args.name != STAGE1_NAME else STAGE2_NAME)

    # Auto-calibrate batch size
    if args.batch == -1:
        logger.info("Batch size set to -1 (auto). Running calibration...")
        batch_size = calibrate_batch_size(
            model_path=model_source,
            data_yaml=args.data_stage2 if hasattr(args, "data_stage2") and args.data_stage2 else args.data,
            imgsz=args.imgsz,
            start_batch=BATCH_CALIBRATION_START,
            min_batch=BATCH_CALIBRATION_MIN,
        )
        logger.info("Calibrated batch size: %d", batch_size)
    else:
        batch_size = args.batch

    # Resume or fresh start
    if args.resume:
        resume_path = Path(project) / run_name / "weights" / "last.pt"
        if resume_path.exists():
            logger.info("Resuming Stage 2 from %s", resume_path)
            model_source = str(resume_path)
        else:
            logger.warning("Resume requested but %s not found. Starting fresh.", resume_path)

    # Build training kwargs
    stage2_config = {
        "data": args.data_stage2 if hasattr(args, "data_stage2") and args.data_stage2 else args.data,
        "epochs": getattr(args, "epochs_stage2", None) or (args.epochs if args.epochs else 50),
        "imgsz": args.imgsz,
        "batch": batch_size,
        "amp": True,
        "lr0": 0.001,        # lower LR for fine-tuning
        "lrf": 0.01,         # final LR factor
        "optimizer": "auto",
        "patience": 20,
        "close_mosaic": 10,
        "cos_lr": True,
        "cache": "disk",
        "workers": args.workers,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "save": True,
        "save_period": -1,
        "plots": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        # Night-scene augmentation (slightly reduced for fine-tuning)
        **{
            **NIGHT_AUGMENTATION,
            "mosaic": 0.8,       # reduce mosaic for fine-tuning
            "mixup": 0.05,       # less mixup
            "scale": 0.2,        # less scale jitter
        },
    }

    # Logging
    if args.logger == "wandb":
        os.environ["WANDB_PROJECT"] = "night-road-vision"

    # Save config before training
    save_dir_2 = Path(project) / run_name
    save_training_config(str(save_dir_2), "stage2_finetune", stage2_config)

    # Load model
    logger.info("Loading model: %s", model_source)
    model = YOLO(model_source)

    # Train
    logger.info("Starting Stage 2 training for %d epochs, batch=%d, imgsz=%d, lr0=%.4f",
                stage2_config["epochs"], batch_size, args.imgsz, stage2_config["lr0"])
    print_gpu_status("Stage 2 pre-train")

    results = model.train(**stage2_config)

    # Locate best weights
    best_pt = save_dir_2 / "weights" / "best.pt"
    if not best_pt.exists():
        alt = Path(project) / run_name / "weights" / "best.pt"
        if alt.exists():
            best_pt = alt
        else:
            logger.error("best.pt not found in %s/weights/", save_dir_2)
            raise FileNotFoundError(f"Stage 2 best.pt not found at {best_pt}")

    logger.info("Stage 2 complete. Best weights: %s", best_pt)
    print_gpu_status("Stage 2 end")

    return str(best_pt)


# ---------------------------------------------------------------------------
# Both stages
# ---------------------------------------------------------------------------

def run_both(args):
    """Run Stage 1 then Stage 2 sequentially."""
    logger.info("=" * 70)
    logger.info("RUNNING BOTH STAGES")
    logger.info("=" * 70)

    # --- Stage 1 ---
    # Temporarily set args.data to data-stage1 for stage1
    original_data = args.data
    args.data = args.data_stage1
    stage1_best = run_stage1(args)
    args.data = original_data

    # --- Stage 2 ---
    run_stage2(args, stage1_weights=stage1_best)

    logger.info("Both stages complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage YOLO26n training pipeline for night road detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1 only
  python train.py stage1 --data datasets/night_road/night_road.yaml

  # Stage 2 only (with explicit stage1 weights)
  python train.py stage2 --data datasets/night_road_phone/night_road.yaml \\
                          --stage1-weights runs/night-road/stage1_public/weights/best.pt

  # Both stages sequentially
  python train.py both --data-stage1 datasets/night_road/night_road.yaml \\
                        --data-stage2 datasets/night_road_phone/night_road.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Training stage to run")
    subparsers.required = True

    # ---- Shared arguments function ----
    def add_common_args(p):
        p.add_argument(
            "--data",
            type=str,
            default=None,
            help="Path to dataset YAML config.",
        )
        p.add_argument(
            "--model",
            type=str,
            default=None,
            help="Path to model weights (default: yolo26n.pt for stage1).",
        )
        p.add_argument(
            "--imgsz",
            type=int,
            default=640,
            help="Image size for training (default: 640).",
        )
        p.add_argument(
            "--batch",
            type=int,
            default=-1,
            help="Batch size. -1 for auto-calibration (default: -1).",
        )
        p.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Number of epochs (default: 120 for stage1, 50 for stage2).",
        )
        p.add_argument(
            "--workers",
            type=int,
            default=8,
            help="DataLoader workers (default: 8).",
        )
        p.add_argument(
            "--project",
            type=str,
            default=DEFAULT_PROJECT,
            help=f"Project directory (default: {DEFAULT_PROJECT}).",
        )
        p.add_argument(
            "--name",
            type=str,
            default=None,
            help="Run name (default: stage-specific).",
        )
        p.add_argument(
            "--resume",
            action="store_true",
            help="Resume training from last.pt in the run directory.",
        )
        p.add_argument(
            "--logger",
            type=str,
            choices=["none", "wandb", "tensorboard"],
            default="tensorboard",
            help="Logging backend (default: tensorboard).",
        )

    # ---- Stage 1 ----
    stage1_parser = subparsers.add_parser("stage1", help="Stage 1: Public data pre-adaptation")
    add_common_args(stage1_parser)
    stage1_parser.set_defaults(func=run_stage1)

    # ---- Stage 2 ----
    stage2_parser = subparsers.add_parser("stage2", help="Stage 2: Phone data fine-tune")
    add_common_args(stage2_parser)
    stage2_parser.add_argument(
        "--stage1-weights",
        type=str,
        default=None,
        help="Path to Stage 1 best.pt weights (auto-detected if omitted).",
    )
    stage2_parser.add_argument(
        "--data-stage2",
        type=str,
        default=None,
        help="Path to Stage 2 dataset YAML (overrides --data for stage2).",
    )
    stage2_parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=None,
        help="Override epoch count for stage 2 (default: 50).",
    )
    stage2_parser.set_defaults(func=run_stage2)

    # ---- Both ----
    both_parser = subparsers.add_parser("both", help="Run Stage 1 then Stage 2")
    add_common_args(both_parser)
    both_parser.add_argument(
        "--data-stage1",
        type=str,
        required=True,
        help="Path to Stage 1 (public) dataset YAML.",
    )
    both_parser.add_argument(
        "--data-stage2",
        type=str,
        required=True,
        help="Path to Stage 2 (phone) dataset YAML.",
    )
    both_parser.add_argument(
        "--stage1-weights",
        type=str,
        default=None,
        help="Path to existing Stage 1 weights (skip stage1 if provided).",
    )
    both_parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=None,
        help="Override epoch count for stage 2 (default: 50).",
    )
    both_parser.set_defaults(func=run_both)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Night Road Vision Training Pipeline")
    logger.info("Command: %s", args.command)
    logger.info("Arguments: %s", vars(args))

    # Initial GPU check
    print_gpu_status("startup")

    # Check that data file exists (if provided)
    data_to_check = getattr(args, "data", None)
    if data_to_check and not Path(data_to_check).exists():
        logger.warning("Data YAML not found: %s (training will fail if path is wrong)", data_to_check)

    # Check ultralytics is available
    try:
        import ultralytics
        logger.info("Ultralytics version: %s", ultralytics.__version__)
    except ImportError:
        logger.error("ultralytics not installed. Install with: pip install ultralytics")
        sys.exit(1)

    # Check torch/CUDA
    try:
        import torch
        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
            logger.info("CUDA version: %s", torch.version.cuda)
    except ImportError:
        logger.warning("PyTorch not found. Training will likely fail.")

    # Dispatch
    start_time = time.time()

    try:
        if args.command == "stage1":
            best = run_stage1(args)
            logger.info("Stage 1 finished. Best weights: %s", best)
        elif args.command == "stage2":
            best = run_stage2(args)
            logger.info("Stage 2 finished. Best weights: %s", best)
        elif args.command == "both":
            run_both(args)
            logger.info("Both stages finished.")
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Total time: %.1f hours (%.0f seconds)", elapsed / 3600, elapsed)
    print_gpu_status("final")


if __name__ == "__main__":
    main()
