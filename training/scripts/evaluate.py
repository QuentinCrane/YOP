#!/usr/bin/env python3
"""
Comprehensive evaluation script for night road detection models.

Performs standard YOLO validation, slice-based evaluation on challenging
subsets, video-based temporal analysis, and multi-format model comparison.

Subcommands:
    standard  - Full evaluation on test split with plots
    slices    - Evaluate on specific challenging test subsets
    video     - Process video and compute temporal quality metrics
    compare   - Compare PyTorch, TFLite FP16, and TFLite INT8 models

Usage:
    python evaluate.py standard --model best.pt --data night_road.yaml
    python evaluate.py slices  --model best.pt --slices-dir evaluation/slices/
    python evaluate.py video   --model best.pt --video test.mp4
    python evaluate.py compare --models best.pt fp16.tflite int8.tflite
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLICES = {
    "night_person_small": {
        "description": "Small pedestrians at night",
        "yaml": "night_person_small.yaml",
    },
    "night_person_dark_clothes": {
        "description": "Pedestrians in dark clothing",
        "yaml": "night_person_dark_clothes.yaml",
    },
    "night_backlight": {
        "description": "Backlit scenes (car headlights)",
        "yaml": "night_backlight.yaml",
    },
    "night_rain": {
        "description": "Rainy night scenes",
        "yaml": "night_rain.yaml",
    },
    "parking_garage": {
        "description": "Underground parking",
        "yaml": "parking_garage.yaml",
    },
    "unseen_phone": {
        "description": "Phone model not in training",
        "yaml": "unseen_phone.yaml",
    },
    "daytime_general": {
        "description": "Daytime baseline",
        "yaml": "daytime_general.yaml",
    },
}

CLASS_NAMES_DEFAULT = ["person", "car"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_class_names(data_yaml: str | None) -> list[str]:
    """Load class names from a data YAML, falling back to defaults."""
    if data_yaml and Path(data_yaml).is_file():
        with open(data_yaml, "r", encoding="utf-8") as fh:
            data_cfg = yaml.safe_load(fh)
        names = data_cfg.get("names", CLASS_NAMES_DEFAULT)
        # names can be a dict {0: 'person', 1: 'car'} or a list
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        return names
    return CLASS_NAMES_DEFAULT


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_float(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def get_model_file_size(path: str) -> float:
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


# ---------------------------------------------------------------------------
# 1. Standard Evaluation
# ---------------------------------------------------------------------------


def run_standard_evaluation(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
    device: str = "",
    project: str = "runs/evaluate",
    name: str = "standard",
    save_txt: bool = False,
    save_conf: bool = False,
    plots: bool = True,
) -> dict[str, Any]:
    """
    Run yolo detect val on the test split and return structured results.

    Returns a dict with keys:
        mAP50, mAP50_95, per_class (list of dicts with precision/recall/map),
        and paths to generated plots.
    """
    print("=" * 70)
    print("STANDARD EVALUATION")
    print("=" * 70)
    print(f"  Model : {model_path}")
    print(f"  Data  : {data_yaml}")
    print(f"  ImgSz : {img_size}")
    print(f"  Device: {device or 'auto'}")
    print()

    model = YOLO(model_path)
    class_names = load_class_names(data_yaml)

    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device if device else None,
        project=project,
        name=name,
        save_txt=save_txt,
        save_conf=save_conf,
        plots=plots,
        split="test",
    )

    # Extract overall metrics
    output: dict[str, Any] = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class": [],
    }

    # Per-class metrics
    num_classes = len(results.box.ap_class_index)
    for idx in range(num_classes):
        cls_id = int(results.box.ap_class_index[idx])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        per_class_entry = {
            "class_id": cls_id,
            "class_name": cls_name,
            "precision": float(results.box.p[idx]) if idx < len(results.box.p) else 0.0,
            "recall": float(results.box.r[idx]) if idx < len(results.box.r) else 0.0,
            "mAP50": float(results.box.ap50[idx]) if idx < len(results.box.ap50) else 0.0,
            "mAP50_95": float(results.box.ap[idx]) if idx < len(results.box.ap) else 0.0,
        }
        output["per_class"].append(per_class_entry)

    # Plot paths (ultralytics saves them under project/name)
    plot_dir = Path(project) / name
    output["plot_dir"] = str(plot_dir)

    # Print summary table
    print()
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 70)
    for entry in output["per_class"]:
        print(
            f"{entry['class_name']:<20} "
            f"{format_float(entry['precision']):>10} "
            f"{format_float(entry['recall']):>10} "
            f"{format_float(entry['mAP50']):>10} "
            f"{format_float(entry['mAP50_95']):>10}"
        )
    print("-" * 70)
    print(
        f"{'OVERALL':<20} "
        f"{format_float(output['precision']):>10} "
        f"{format_float(output['recall']):>10} "
        f"{format_float(output['mAP50']):>10} "
        f"{format_float(output['mAP50_95']):>10}"
    )
    print("-" * 70)
    print()
    print(f"Plots saved to: {plot_dir}")

    return output


# ---------------------------------------------------------------------------
# 2. Slice Evaluation
# ---------------------------------------------------------------------------


def run_slice_evaluation(
    model_path: str,
    slices_dir: str,
    img_size: int = 640,
    batch_size: int = 16,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
    device: str = "",
    project: str = "runs/evaluate",
    class_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate on each slice YAML in slices_dir.

    Returns a dict mapping slice_name -> {
        description, person_recall, car_recall, mAP50, mAP50_95, ...
    }
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    slices_dir = Path(slices_dir)
    print("=" * 70)
    print("SLICE EVALUATION")
    print("=" * 70)
    print(f"  Model     : {model_path}")
    print(f"  Slices dir: {slices_dir}")
    print()

    model = YOLO(model_path)
    all_results: dict[str, dict[str, Any]] = {}

    for slice_name, slice_info in SLICES.items():
        yaml_file = slices_dir / slice_info["yaml"]

        if not yaml_file.is_file():
            print(f"  [SKIP] {slice_name}: YAML not found at {yaml_file}")
            all_results[slice_name] = {
                "description": slice_info["description"],
                "status": "missing_yaml",
                "person_recall": None,
                "car_recall": None,
                "mAP50": None,
                "mAP50_95": None,
            }
            continue

        print(f"  [{slice_name}] {slice_info['description']}")
        print(f"    YAML: {yaml_file}")

        try:
            results = model.val(
                data=str(yaml_file),
                imgsz=img_size,
                batch=batch_size,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device if device else None,
                project=project,
                name=f"slice_{slice_name}",
                plots=False,
                split="test",
                verbose=False,
            )

            # Per-class recall
            person_recall = 0.0
            car_recall = 0.0
            per_class_recall: dict[str, float] = {}

            for idx, cls_id in enumerate(results.box.ap_class_index):
                cls_id = int(cls_id)
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                r = float(results.box.r[idx]) if idx < len(results.box.r) else 0.0
                per_class_recall[cls_name] = r
                if cls_name == "person":
                    person_recall = r
                elif cls_name == "car":
                    car_recall = r

            entry = {
                "description": slice_info["description"],
                "status": "ok",
                "person_recall": person_recall,
                "car_recall": car_recall,
                "per_class_recall": per_class_recall,
                "mAP50": float(results.box.map50),
                "mAP50_95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            }
            all_results[slice_name] = entry

            print(f"    person Recall : {format_float(person_recall)}")
            print(f"    car Recall    : {format_float(car_recall)}")
            print(f"    mAP50         : {format_float(entry['mAP50'])}")
            print(f"    mAP50-95      : {format_float(entry['mAP50_95'])}")

        except Exception as exc:
            print(f"    [ERROR] {exc}")
            all_results[slice_name] = {
                "description": slice_info["description"],
                "status": "error",
                "error": str(exc),
                "person_recall": None,
                "car_recall": None,
                "mAP50": None,
                "mAP50_95": None,
            }

        print()

    # Summary table
    print("=" * 90)
    print(
        f"{'Slice':<30} {'person_R':>10} {'car_R':>10} {'mAP50':>10} {'mAP50-95':>10}"
    )
    print("-" * 90)
    for name, res in all_results.items():
        pr = format_float(res["person_recall"]) if res["person_recall"] is not None else "N/A"
        cr = format_float(res["car_recall"]) if res["car_recall"] is not None else "N/A"
        m50 = format_float(res["mAP50"]) if res["mAP50"] is not None else "N/A"
        m95 = format_float(res["mAP50_95"]) if res["mAP50_95"] is not None else "N/A"
        print(f"{name:<30} {pr:>10} {cr:>10} {m50:>10} {m95:>10}")
    print("=" * 90)

    return all_results


# ---------------------------------------------------------------------------
# 3. Video Evaluation
# ---------------------------------------------------------------------------


class VideoEvaluator:
    """
    Process a video through the model and compute temporal quality metrics:
      - Missed detection duration (gaps where target objects disappear)
      - Box flickering (bounding box appearing/disappearing frame-to-frame)
      - ID switches (tracker identity changes)
      - Box jitter (frame-to-frame bounding box position variance)
      - Detection latency
      - Frame-to-frame consistency
    """

    def __init__(
        self,
        model_path: str,
        video_path: str,
        output_path: str = "runs/evaluate/video_eval",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.6,
        img_size: int = 640,
        device: str = "",
        class_names: list[str] | None = None,
    ):
        self.model = YOLO(model_path)
        self.video_path = Path(video_path)
        self.output_dir = ensure_dir(output_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device if device else None
        self.class_names = class_names or CLASS_NAMES_DEFAULT

        # Per-frame tracking state
        self.frame_detections: list[list[dict]] = []  # detections per frame
        self.frame_times: list[float] = []  # inference time per frame
        self.track_history: dict[int, list[dict]] = {}  # track_id -> list of frame boxes
        self.frame_count = 0

        # Aggregated metrics
        self.metrics: dict[str, Any] = {}

    def _process_frame_results(self, result, frame_idx: int) -> None:
        """Extract detection info from a single frame result."""
        boxes = result.boxes
        frame_dets: list[dict] = []

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                track_id = int(boxes.id[i]) if boxes.id is not None else -1

                det = {
                    "bbox": xyxy,  # [x1, y1, x2, y2]
                    "conf": conf,
                    "class": cls_id,
                    "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    "track_id": track_id,
                    "frame": frame_idx,
                }
                frame_dets.append(det)

                # Update track history
                if track_id >= 0:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append(
                        {
                            "frame": frame_idx,
                            "bbox": xyxy,
                            "conf": conf,
                            "class": cls_id,
                        }
                    )

        self.frame_detections.append(frame_dets)

    def _compute_missed_detections(self) -> dict[str, Any]:
        """
        For each tracked object, compute total duration (in frames) where it
        was not detected despite being expected (gaps in track history).
        """
        total_gap_frames = 0
        gap_events: list[dict] = []

        for track_id, history in self.track_history.items():
            if len(history) < 2:
                continue

            frames = [h["frame"] for h in history]
            track_start = min(frames)
            track_end = max(frames)

            expected_frames = set(range(track_start, track_end + 1))
            detected_frames = set(frames)
            missing = sorted(expected_frames - detected_frames)

            if missing:
                # Group consecutive missing frames into gaps
                gap_start = missing[0]
                prev = missing[0]
                for f in missing[1:]:
                    if f == prev + 1:
                        prev = f
                        continue
                    gap_len = prev - gap_start + 1
                    total_gap_frames += gap_len
                    gap_events.append(
                        {
                            "track_id": track_id,
                            "start_frame": gap_start,
                            "end_frame": prev,
                            "duration": gap_len,
                        }
                    )
                    gap_start = f
                    prev = f
                # Last gap
                gap_len = prev - gap_start + 1
                total_gap_frames += gap_len
                gap_events.append(
                    {
                        "track_id": track_id,
                        "start_frame": gap_start,
                        "end_frame": prev,
                        "duration": gap_len,
                    }
                )

        return {
            "total_gap_frames": total_gap_frames,
            "num_gap_events": len(gap_events),
            "gap_events": gap_events,
        }

    def _compute_flickering(self) -> dict[str, Any]:
        """
        Count frames where a detection appears/disappears compared to the
        previous frame for the same track ID.
        """
        flicker_count = 0
        total_transitions = 0

        for track_id, history in self.track_history.items():
            if len(history) < 2:
                continue

            sorted_history = sorted(history, key=lambda h: h["frame"])
            for i in range(1, len(sorted_history)):
                prev_frame = sorted_history[i - 1]["frame"]
                curr_frame = sorted_history[i]["frame"]
                gap = curr_frame - prev_frame
                total_transitions += 1
                # A gap > 1 means the object disappeared for at least one frame
                if gap > 1:
                    flicker_count += 1

        flicker_rate = flicker_count / max(total_transitions, 1)
        return {
            "flicker_count": flicker_count,
            "total_transitions": total_transitions,
            "flicker_rate": flicker_rate,
        }

    def _compute_id_switches(self) -> dict[str, Any]:
        """
        Detect ID switches: when the spatial overlap between consecutive frames
        suggests the same object but the track ID changed.
        We look for new tracks that start near where old tracks ended.
        """
        id_switches = 0
        switch_events: list[dict] = []

        # Build a list of track endpoints: (end_frame, last_bbox, track_id)
        track_endpoints: list[tuple[int, list[float], int]] = []
        track_startpoints: list[tuple[int, list[float], int]] = []

        for track_id, history in self.track_history.items():
            sorted_history = sorted(history, key=lambda h: h["frame"])
            track_startpoints.append((sorted_history[0]["frame"], sorted_history[0]["bbox"], track_id))
            track_endpoints.append((sorted_history[-1]["frame"], sorted_history[-1]["bbox"], track_id))

        # For each track start, check if there is a recently ended track
        # whose bbox overlaps significantly
        for start_frame, start_bbox, start_tid in sorted(track_startpoints):
            best_iou = 0.0
            best_old_tid = -1
            for end_frame, end_bbox, end_tid in track_endpoints:
                if end_tid == start_tid:
                    continue
                # The old track must have ended close in time
                if start_frame - end_frame > 10 or start_frame <= end_frame:
                    continue
                iou = self._bbox_iou(start_bbox, end_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_old_tid = end_tid

            if best_iou > 0.3 and best_old_tid >= 0:
                id_switches += 1
                switch_events.append(
                    {
                        "frame": start_frame,
                        "old_track_id": best_old_tid,
                        "new_track_id": start_tid,
                        "iou": best_iou,
                    }
                )

        return {
            "id_switches": id_switches,
            "switch_events": switch_events,
        }

    @staticmethod
    def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _compute_box_jitter(self) -> dict[str, Any]:
        """
        Compute frame-to-frame bounding box position variance for each track.
        Jitter = mean of per-track std-dev of center-point displacement.
        """
        jitter_values: list[float] = []

        for track_id, history in self.track_history.items():
            if len(history) < 3:
                continue

            sorted_history = sorted(history, key=lambda h: h["frame"])
            centers = []
            for h in sorted_history:
                bbox = h["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                centers.append((cx, cy))

            # Compute frame-to-frame displacements
            dx_list = []
            dy_list = []
            for i in range(1, len(centers)):
                dx_list.append(centers[i][0] - centers[i - 1][0])
                dy_list.append(centers[i][1] - centers[i - 1][1])

            if dx_list:
                jitter = float(np.std(dx_list) + np.std(dy_list))
                jitter_values.append(jitter)

        mean_jitter = float(np.mean(jitter_values)) if jitter_values else 0.0
        return {
            "mean_jitter": mean_jitter,
            "per_track_jitter": jitter_values,
            "num_tracks_evaluated": len(jitter_values),
        }

    def _compute_frame_consistency(self) -> dict[str, Any]:
        """
        Frame-to-frame consistency: ratio of frames where the number of
        detections is the same as the previous frame (within tolerance).
        """
        if len(self.frame_detections) < 2:
            return {"consistency_ratio": 1.0, "stable_frames": 0, "total_transitions": 0}

        stable = 0
        total = len(self.frame_detections) - 1

        for i in range(1, len(self.frame_detections)):
            prev_count = len(self.frame_detections[i - 1])
            curr_count = len(self.frame_detections[i])
            # Allow +/- 1 difference
            if abs(curr_count - prev_count) <= 1:
                stable += 1

        return {
            "consistency_ratio": stable / max(total, 1),
            "stable_frames": stable,
            "total_transitions": total,
        }

    def run(self) -> dict[str, Any]:
        """Run full video evaluation and return metrics."""
        print("=" * 70)
        print("VIDEO EVALUATION")
        print("=" * 70)
        print(f"  Model : {self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else 'loaded'}")
        print(f"  Video : {self.video_path}")
        print()

        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Resolution : {width}x{height}")
        print(f"  FPS        : {fps}")
        print(f"  Frames     : {total_frames}")
        print()

        # Prepare annotated output video
        out_video_path = self.output_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        frame_idx = 0
        total_inference_ms = 0.0

        print("  Processing frames...", end="", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            results = self.model.track(
                source=frame,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml",
            )
            t1 = time.perf_counter()

            inference_ms = (t1 - t0) * 1000.0
            self.frame_times.append(inference_ms)
            total_inference_ms += inference_ms

            if results and len(results) > 0:
                result = results[0]
                self._process_frame_results(result, frame_idx)

                # Draw annotated frame
                annotated = result.plot()
                writer.write(annotated)
            else:
                self.frame_detections.append([])
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f" {frame_idx}", end="", flush=True)

        self.frame_count = frame_idx
        cap.release()
        writer.release()
        print(f" done ({self.frame_count} frames)")
        print()

        # Compute all temporal metrics
        missed = self._compute_missed_detections()
        flicker = self._compute_flickering()
        id_switch = self._compute_id_switches()
        jitter = self._compute_box_jitter()
        consistency = self._compute_frame_consistency()

        avg_inference_ms = total_inference_ms / max(self.frame_count, 1)

        self.metrics = {
            "video_path": str(self.video_path),
            "total_frames": self.frame_count,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "avg_inference_ms": round(avg_inference_ms, 2),
            "detection_latency_ms": round(avg_inference_ms, 2),
            "missed_detections": missed,
            "flickering": flicker,
            "id_switches": id_switch,
            "box_jitter": jitter,
            "frame_consistency": consistency,
            "annotated_video": str(out_video_path),
        }

        # Print summary
        print("-" * 70)
        print("VIDEO EVALUATION RESULTS")
        print("-" * 70)
        print(f"  Frames processed      : {self.frame_count}")
        print(f"  Avg inference time    : {avg_inference_ms:.2f} ms")
        print(f"  Detection latency     : {avg_inference_ms:.2f} ms")
        print(f"  Tracked objects       : {len(self.track_history)}")
        print(f"  Missed det gaps       : {missed['num_gap_events']} ({missed['total_gap_frames']} frames)")
        print(f"  Flicker rate          : {flicker['flicker_rate']:.4f} ({flicker['flicker_count']}/{flicker['total_transitions']})")
        print(f"  ID switches           : {id_switch['id_switches']}")
        print(f"  Mean box jitter       : {jitter['mean_jitter']:.4f} px")
        print(f"  Frame consistency     : {consistency['consistency_ratio']:.4f}")
        print(f"  Annotated video       : {out_video_path}")
        print("-" * 70)

        return self.metrics


# ---------------------------------------------------------------------------
# 4. Model Comparison
# ---------------------------------------------------------------------------


def run_model_comparison(
    model_paths: list[str],
    data_yaml: str = "",
    img_size: int = 640,
    batch_size: int = 16,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
    device: str = "",
    project: str = "runs/evaluate",
    class_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Compare multiple model formats (PyTorch FP32, TFLite FP16, TFLite INT8).

    For each model, reports:
      - person Recall
      - mAP50-95
      - file size (MB)
      - inference time (ms per image)
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print()

    comparison: list[dict[str, Any]] = []

    for model_path in model_paths:
        model_path = model_path.strip()
        if not model_path:
            continue

        print(f"  Evaluating: {model_path}")

        file_size_mb = get_model_file_size(model_path)
        ext = Path(model_path).suffix.lower()

        # Determine format label
        if ext == ".pt":
            fmt = "PyTorch FP32"
        elif ext == ".tflite" and "int8" in model_path.lower():
            fmt = "TFLite INT8"
        elif ext == ".tflite" and ("fp16" in model_path.lower() or "float16" in model_path.lower()):
            fmt = "TFLite FP16"
        elif ext == ".tflite":
            fmt = "TFLite"
        elif ext == ".onnx":
            fmt = "ONNX"
        elif ext in (".engine", ".trt"):
            fmt = "TensorRT"
        else:
            fmt = ext.lstrip(".").upper()

        entry: dict[str, Any] = {
            "model_path": model_path,
            "format": fmt,
            "file_size_mb": round(file_size_mb, 2),
        }

        try:
            model = YOLO(model_path)

            # Benchmark inference time
            dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            warmup_runs = 3
            benchmark_runs = 10

            # Warmup
            for _ in range(warmup_runs):
                model.predict(
                    source=dummy_img,
                    conf=conf_threshold,
                    imgsz=img_size,
                    device=device if device else None,
                    verbose=False,
                )

            # Benchmark
            times = []
            for _ in range(benchmark_runs):
                t0 = time.perf_counter()
                model.predict(
                    source=dummy_img,
                    conf=conf_threshold,
                    imgsz=img_size,
                    device=device if device else None,
                    verbose=False,
                )
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            avg_inference_ms = float(np.mean(times))
            entry["inference_ms"] = round(avg_inference_ms, 2)

            # Full validation if data YAML provided
            if data_yaml and Path(data_yaml).is_file():
                results = model.val(
                    data=data_yaml,
                    imgsz=img_size,
                    batch=batch_size,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device if device else None,
                    project=project,
                    name=f"compare_{Path(model_path).stem}",
                    plots=False,
                    split="test",
                    verbose=False,
                )

                entry["mAP50"] = round(float(results.box.map50), 4)
                entry["mAP50_95"] = round(float(results.box.map), 4)
                entry["precision"] = round(float(results.box.mp), 4)
                entry["recall"] = round(float(results.box.mr), 4)

                # Per-class recall
                for idx, cls_id in enumerate(results.box.ap_class_index):
                    cls_id = int(cls_id)
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    r = float(results.box.r[idx]) if idx < len(results.box.r) else 0.0
                    entry[f"{cls_name}_recall"] = round(r, 4)
            else:
                print("    [WARN] No data YAML provided, skipping mAP validation")

        except Exception as exc:
            print(f"    [ERROR] {exc}")
            entry["error"] = str(exc)
            entry["inference_ms"] = None
            entry["mAP50"] = None
            entry["mAP50_95"] = None

        comparison.append(entry)

    # Print comparison table
    print()
    print("=" * 100)
    header = f"{'Format':<15} {'Size (MB)':>10} {'Inf (ms)':>10}"
    # Dynamically add per-class recall columns
    recall_cols: list[str] = []
    for entry in comparison:
        for key in entry:
            if key.endswith("_recall") and key not in recall_cols:
                recall_cols.append(key)
    for col in recall_cols:
        header += f" {col:>12}"
    header += f" {'mAP50':>10} {'mAP50-95':>10}"

    print(header)
    print("-" * 100)

    for entry in comparison:
        row = f"{entry['format']:<15} {entry['file_size_mb']:>10.2f} "
        inf = entry.get("inference_ms")
        row += f"{inf:>10.2f} " if inf is not None else f"{'N/A':>10} "
        for col in recall_cols:
            val = entry.get(col)
            row += f"{val:>12.4f} " if val is not None else f"{'N/A':>12} "
        m50 = entry.get("mAP50")
        m95 = entry.get("mAP50_95")
        row += f"{m50:>10.4f} " if m50 is not None else f"{'N/A':>10} "
        row += f"{m95:>10.4f}" if m95 is not None else f"{'N/A':>10}"
        print(row)

    print("=" * 100)
    print()

    return comparison


# ---------------------------------------------------------------------------
# 5. Export Reports
# ---------------------------------------------------------------------------


def export_standard_csv(result: dict[str, Any], output_dir: str) -> str:
    """Export standard evaluation results to CSV."""
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "standard_results.csv"

    rows = []
    for entry in result.get("per_class", []):
        rows.append(
            {
                "class_id": entry["class_id"],
                "class_name": entry["class_name"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "mAP50": entry["mAP50"],
                "mAP50_95": entry["mAP50_95"],
            }
        )

    # Add overall row
    rows.append(
        {
            "class_id": -1,
            "class_name": "OVERALL",
            "precision": result["precision"],
            "recall": result["recall"],
            "mAP50": result["mAP50"],
            "mAP50_95": result["mAP50_95"],
        }
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


def export_slice_csv(results: dict[str, dict[str, Any]], output_dir: str) -> str:
    """Export slice evaluation results to CSV."""
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "slice_results.csv"

    rows = []
    for name, res in results.items():
        rows.append(
            {
                "slice": name,
                "description": res.get("description", ""),
                "person_recall": res.get("person_recall"),
                "car_recall": res.get("car_recall"),
                "mAP50": res.get("mAP50"),
                "mAP50_95": res.get("mAP50_95"),
                "status": res.get("status", ""),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


def export_comparison_csv(comparison: list[dict[str, Any]], output_dir: str) -> str:
    """Export model comparison results to CSV."""
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "comparison_results.csv"

    if not comparison:
        return str(csv_path)

    # Collect all keys
    all_keys: list[str] = []
    for entry in comparison:
        for key in entry:
            if key not in all_keys:
                all_keys.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)

    return str(csv_path)


def export_comparison_markdown(
    standard: dict[str, Any] | None,
    slices: dict[str, dict[str, Any]] | None,
    comparison: list[dict[str, Any]] | None,
    video: dict[str, Any] | None,
    output_dir: str,
) -> str:
    """Export a full comparison report as Markdown."""
    output_dir = ensure_dir(output_dir)
    md_path = output_dir / "evaluation_report.md"

    lines: list[str] = []
    lines.append("# Night Road Detection - Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # --- Standard Results ---
    if standard:
        lines.append("## 1. Standard Evaluation")
        lines.append("")
        lines.append("| Class | Precision | Recall | mAP50 | mAP50-95 |")
        lines.append("|-------|-----------|--------|-------|----------|")
        for entry in standard.get("per_class", []):
            lines.append(
                f"| {entry['class_name']} "
                f"| {format_float(entry['precision'])} "
                f"| {format_float(entry['recall'])} "
                f"| {format_float(entry['mAP50'])} "
                f"| {format_float(entry['mAP50_95'])} |"
            )
        lines.append(
            f"| **OVERALL** "
            f"| **{format_float(standard['precision'])}** "
            f"| **{format_float(standard['recall'])}** "
            f"| **{format_float(standard['mAP50'])}** "
            f"| **{format_float(standard['mAP50_95'])}** |"
        )
        lines.append("")

    # --- Slice Results ---
    if slices:
        lines.append("## 2. Slice Evaluation")
        lines.append("")
        lines.append("| Slice | Description | Person Recall | Car Recall | mAP50 | mAP50-95 |")
        lines.append("|-------|-------------|---------------|------------|-------|----------|")
        for name, res in slices.items():
            pr = format_float(res["person_recall"]) if res.get("person_recall") is not None else "N/A"
            cr = format_float(res["car_recall"]) if res.get("car_recall") is not None else "N/A"
            m50 = format_float(res["mAP50"]) if res.get("mAP50") is not None else "N/A"
            m95 = format_float(res["mAP50_95"]) if res.get("mAP50_95") is not None else "N/A"
            lines.append(f"| {name} | {res.get('description', '')} | {pr} | {cr} | {m50} | {m95} |")
        lines.append("")

    # --- Video Results ---
    if video:
        lines.append("## 3. Video Evaluation")
        lines.append("")
        lines.append(f"- **Video**: {video.get('video_path', 'N/A')}")
        lines.append(f"- **Resolution**: {video.get('resolution', 'N/A')}")
        lines.append(f"- **Total frames**: {video.get('total_frames', 0)}")
        lines.append(f"- **Avg inference**: {video.get('avg_inference_ms', 0):.2f} ms")
        lines.append(f"- **Detection latency**: {video.get('detection_latency_ms', 0):.2f} ms")
        missed = video.get("missed_detections", {})
        flicker = video.get("flickering", {})
        id_sw = video.get("id_switches", {})
        jit = video.get("box_jitter", {})
        cons = video.get("frame_consistency", {})
        lines.append(f"- **Missed detection gaps**: {missed.get('num_gap_events', 0)} ({missed.get('total_gap_frames', 0)} frames)")
        lines.append(f"- **Flicker rate**: {flicker.get('flicker_rate', 0):.4f}")
        lines.append(f"- **ID switches**: {id_sw.get('id_switches', 0)}")
        lines.append(f"- **Mean box jitter**: {jit.get('mean_jitter', 0):.4f} px")
        lines.append(f"- **Frame consistency**: {cons.get('consistency_ratio', 0):.4f}")
        lines.append("")

    # --- Model Comparison ---
    if comparison:
        lines.append("## 4. Model Comparison")
        lines.append("")

        # Build header
        recall_cols: list[str] = []
        for entry in comparison:
            for key in entry:
                if key.endswith("_recall") and key not in recall_cols:
                    recall_cols.append(key)

        header = "| Format | Size (MB) | Inference (ms)"
        for col in recall_cols:
            header += f" | {col}"
        header += " | mAP50 | mAP50-95 |"
        lines.append(header)

        sep = "|--------|-----------|----------------"
        for _ in recall_cols:
            sep += "|--------"
        sep += "|-------|----------|"
        lines.append(sep)

        for entry in comparison:
            inf = entry.get("inference_ms")
            inf_str = f"{inf:.2f}" if inf is not None else "N/A"
            m50 = entry.get("mAP50")
            m95 = entry.get("mAP50_95")
            row = f"| {entry['format']} | {entry['file_size_mb']:.2f} | {inf_str}"
            for col in recall_cols:
                val = entry.get(col)
                row += f" | {val:.4f}" if val is not None else " | N/A"
            row += f" | {m50:.4f}" if m50 is not None else " | N/A"
            row += f" | {m95:.4f} |" if m95 is not None else " | N/A |"
            lines.append(row)

        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(md_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for night road detection models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py standard --model best.pt --data night_road.yaml\n"
            "  python evaluate.py slices  --model best.pt --slices-dir evaluation/slices/\n"
            "  python evaluate.py video   --model best.pt --video test.mp4\n"
            "  python evaluate.py compare --models best.pt model_fp16.tflite model_int8.tflite\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # ---- standard ----
    sp_std = subparsers.add_parser("standard", help="Run standard evaluation on test split")
    sp_std.add_argument("--model", required=True, help="Path to model weights (.pt)")
    sp_std.add_argument("--data", required=True, help="Path to dataset YAML")
    sp_std.add_argument("--img-size", type=int, default=640, help="Inference image size")
    sp_std.add_argument("--batch-size", type=int, default=16, help="Batch size")
    sp_std.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    sp_std.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    sp_std.add_argument("--device", default="", help="Device (cpu, 0, 0,1, etc.)")
    sp_std.add_argument("--project", default="runs/evaluate", help="Output project directory")
    sp_std.add_argument("--name", default="standard", help="Run name")
    sp_std.add_argument("--save-txt", action="store_true", help="Save results as TXT")
    sp_std.add_argument("--save-conf", action="store_true", help="Save confidences in TXT")
    sp_std.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    sp_std.add_argument("--export-csv", action="store_true", help="Export results to CSV")
    sp_std.add_argument("--export-md", action="store_true", help="Export results to Markdown")

    # ---- slices ----
    sp_sli = subparsers.add_parser("slices", help="Evaluate on specific test subsets")
    sp_sli.add_argument("--model", required=True, help="Path to model weights (.pt)")
    sp_sli.add_argument("--slices-dir", required=True, help="Directory containing slice YAML configs")
    sp_sli.add_argument("--img-size", type=int, default=640, help="Inference image size")
    sp_sli.add_argument("--batch-size", type=int, default=16, help="Batch size")
    sp_sli.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    sp_sli.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    sp_sli.add_argument("--device", default="", help="Device")
    sp_sli.add_argument("--project", default="runs/evaluate", help="Output project directory")
    sp_sli.add_argument("--export-csv", action="store_true", help="Export results to CSV")
    sp_sli.add_argument("--export-md", action="store_true", help="Export results to Markdown")

    # ---- video ----
    sp_vid = subparsers.add_parser("video", help="Evaluate on video with temporal metrics")
    sp_vid.add_argument("--model", required=True, help="Path to model weights (.pt)")
    sp_vid.add_argument("--video", required=True, help="Path to input video file")
    sp_vid.add_argument("--output", default="runs/evaluate/video_eval", help="Output directory")
    sp_vid.add_argument("--img-size", type=int, default=640, help="Inference image size")
    sp_vid.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    sp_vid.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    sp_vid.add_argument("--device", default="", help="Device")
    sp_vid.add_argument("--export-csv", action="store_true", help="Export metrics to CSV")
    sp_vid.add_argument("--export-md", action="store_true", help="Export metrics to Markdown")

    # ---- compare ----
    sp_cmp = subparsers.add_parser("compare", help="Compare multiple model formats")
    sp_cmp.add_argument("--models", nargs="+", required=True, help="Paths to models to compare")
    sp_cmp.add_argument("--data", default="", help="Dataset YAML for mAP evaluation (optional)")
    sp_cmp.add_argument("--img-size", type=int, default=640, help="Inference image size")
    sp_cmp.add_argument("--batch-size", type=int, default=16, help="Batch size")
    sp_cmp.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    sp_cmp.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    sp_cmp.add_argument("--device", default="", help="Device")
    sp_cmp.add_argument("--project", default="runs/evaluate", help="Output project directory")
    sp_cmp.add_argument("--export-csv", action="store_true", help="Export comparison to CSV")
    sp_cmp.add_argument("--export-md", action="store_true", help="Export comparison to Markdown")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    output_dir = "runs/evaluate"

    # -----------------------------------------------------------------------
    if args.command == "standard":
        result = run_standard_evaluation(
            model_path=args.model,
            data_yaml=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            project=args.project,
            name=args.name,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            plots=not args.no_plots,
        )

        if args.export_csv:
            csv_path = export_standard_csv(result, output_dir)
            print(f"CSV exported to: {csv_path}")
        if args.export_md:
            md_path = export_comparison_markdown(
                standard=result, slices=None, comparison=None, video=None, output_dir=output_dir
            )
            print(f"Markdown report exported to: {md_path}")

    # -----------------------------------------------------------------------
    elif args.command == "slices":
        slices_result = run_slice_evaluation(
            model_path=args.model,
            slices_dir=args.slices_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            project=args.project,
        )

        if args.export_csv:
            csv_path = export_slice_csv(slices_result, output_dir)
            print(f"CSV exported to: {csv_path}")
        if args.export_md:
            md_path = export_comparison_markdown(
                standard=None, slices=slices_result, comparison=None, video=None, output_dir=output_dir
            )
            print(f"Markdown report exported to: {md_path}")

    # -----------------------------------------------------------------------
    elif args.command == "video":
        evaluator = VideoEvaluator(
            model_path=args.model,
            video_path=args.video,
            output_path=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            img_size=args.img_size,
            device=args.device,
        )
        video_result = evaluator.run()

        if args.export_csv:
            csv_path = output_dir + "/video_metrics.csv"
            ensure_dir(output_dir)
            # Flatten nested dicts for CSV
            flat: dict[str, Any] = {}
            for key, val in video_result.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if not isinstance(sub_val, (list, dict)):
                            flat[f"{key}.{sub_key}"] = sub_val
                elif not isinstance(val, (list, dict)):
                    flat[key] = val

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=flat.keys())
                writer.writeheader()
                writer.writerow(flat)
            print(f"CSV exported to: {csv_path}")

        if args.export_md:
            md_path = export_comparison_markdown(
                standard=None, slices=None, comparison=None, video=video_result, output_dir=output_dir
            )
            print(f"Markdown report exported to: {md_path}")

    # -----------------------------------------------------------------------
    elif args.command == "compare":
        comparison_result = run_model_comparison(
            model_paths=args.models,
            data_yaml=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            project=args.project,
        )

        if args.export_csv:
            csv_path = export_comparison_csv(comparison_result, output_dir)
            print(f"CSV exported to: {csv_path}")
        if args.export_md:
            md_path = export_comparison_markdown(
                standard=None, slices=None, comparison=comparison_result, video=None, output_dir=output_dir
            )
            print(f"Markdown report exported to: {md_path}")


if __name__ == "__main__":
    main()
