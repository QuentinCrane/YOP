#!/usr/bin/env python3
"""
Comprehensive data preparation script for night road detection training.

Handles the full pipeline:
  - BDD100K night-time label conversion to YOLO format
  - NightOwls pedestrian annotation conversion
  - ExDark low-light object annotation conversion
  - Video frame extraction with deduplication
  - Sequence-aware dataset splitting (train/val/test)
  - Dataset statistics reporting
  - YOLO YAML config generation

Usage examples:
  python prepare_datasets.py bdd100k --input /path/to/bdd100k --output datasets/night_road
  python prepare_datasets.py nightowls --input /path/to/nightowls --output datasets/night_road
  python prepare_datasets.py exdark --input /path/to/exdark --output datasets/night_road
  python prepare_datasets.py extract-frames --input /path/to/videos --output datasets/night_road --fps 2
  python prepare_datasets.py split --dataset datasets/night_road
  python prepare_datasets.py stats --dataset datasets/night_road
  python prepare_datasets.py create-yaml --dataset datasets/night_road
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("[WARN] tqdm not installed. Install with: pip install tqdm")
    print("       Falling back to basic progress reporting.")
    # Minimal fallback so the script still runs without tqdm
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable is not None else None)
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                if self.n % 100 == 0 or self.n == self.total:
                    print(f"\r{self.desc}: {self.n}/{self.total}", end="", flush=True)
            print()
        def update(self, n=1):
            self.n += n
        def set_postfix_str(self, s):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified class list for night road detection
CLASS_NAMES: List[str] = [
    "person",       # 0 - pedestrians, riders
    "rider",        # 1
    "bicycle",      # 2
    "motorcycle",   # 3
    "car",          # 4
    "bus",          # 5
    "truck",        # 6
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# BDD100K class mapping  (source name -> unified name)
BDD100K_CLASS_MAP: Dict[str, str] = {
    "pedestrian":   "person",
    "rider":        "rider",
    "bicycle":      "bicycle",
    "motorcycle":   "motorcycle",
    "car":          "car",
    "bus":          "bus",
    "truck":        "truck",
}

# Classes to skip from BDD100K
BDD100K_SKIP_CLASSES = {"train", "traffic light", "traffic sign", "other vehicle"}

# NightOwls only has pedestrians -> person
NIGHTOWLS_CLASS_MAP: Dict[str, str] = {
    "pedestrian": "person",
}

# ExDark class mapping  (source name -> unified name)
EXDARK_CLASS_MAP: Dict[str, str] = {
    "People":    "person",
    "Bicycle":   "bicycle",
    "Motorbike": "motorcycle",
    "Car":       "car",
    "Bus":       "bus",
    "Truck":     "truck",
    "pedestrian": "person",
    "bicycle":    "bicycle",
    "motorcycle": "motorcycle",
    "car":        "car",
    "bus":        "bus",
    "truck":      "truck",
}

NIGHT_KEYWORDS = {"night", "dawn", "dusk"}

DEFAULT_SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(directory: Path, recursive: bool = False) -> List[Path]:
    """Return sorted list of image files in *directory*."""
    if recursive:
        files = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def clip_box(x1: float, y1: float, x2: float, y2: float,
             img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Clip bounding box to image boundaries."""
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                 img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert [x1, y1, x2, y2] pixel coords to YOLO normalised format
    (x_center, y_center, width, height).  Returns None if box is < 1 px.
    """
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, img_w, img_h)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1 or bh < 1:
        return None
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return xc, yc, nw, nh


def write_yolo_label(label_path: Path, boxes: List[Tuple[int, float, float, float, float]]):
    """Write a YOLO-format TXT label file."""
    with open(label_path, "w") as f:
        for cls_id, xc, yc, w, h in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def get_image_size_fast(path: Path) -> Optional[Tuple[int, int]]:
    """
    Read image dimensions without decoding the full image.
    Supports JPEG and PNG.  Falls back to OpenCV if available.
    """
    suffix = path.suffix.lower()
    try:
        with open(path, "rb") as f:
            header = f.read(32)
            # PNG
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                w = struct.unpack(">I", header[16:20])[0]
                h = struct.unpack(">I", header[20:24])[0]
                return w, h
            # JPEG
            if header[:2] == b"\xff\xd8":
                f.seek(0)
                f.read(2)  # skip SOI
                while True:
                    marker = f.read(2)
                    if len(marker) < 2:
                        break
                    if marker[0] != 0xFF:
                        break
                    if marker[1] == 0xD9:  # EOI
                        break
                    if marker[1] in (0xC0, 0xC1, 0xC2):  # SOF
                        f.read(3)  # length + precision
                        h_val = struct.unpack(">H", f.read(2))[0]
                        w_val = struct.unpack(">H", f.read(2))[0]
                        return w_val, h_val
                    seg_len = struct.unpack(">H", f.read(2))[0]
                    f.read(seg_len - 2)
    except Exception:
        pass

    # Fallback: try OpenCV
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except ImportError:
        pass

    # Fallback: try PIL
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except ImportError:
        pass

    return None


def perceptual_hash(path: Path, hash_size: int = 16) -> Optional[str]:
    """
    Compute a simple perceptual hash for an image.
    Uses a down-sampled grayscale average hash.
    Returns a hex string or None on failure.
    """
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        bits = (resized > avg).flatten()
        hex_str = "".join(str(int(b)) for b in bits)
        return hex_str
    except ImportError:
        pass

    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
            pixels = list(im.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            return bits
    except ImportError:
        pass

    return None


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two equal-length bit strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ---------------------------------------------------------------------------
# BDD100K processing
# ---------------------------------------------------------------------------

def process_bdd100k(args: argparse.Namespace):
    """
    Read BDD100K labels JSON, filter night-time images, and convert to YOLO TXT.

    Expected layout:
        args.input/
            labels/
                bdd100k_labels_images_train.json   (or similar)
                bdd100k_labels_images_val.json
            images/
                100k/
                    train/
                    val/
    """
    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output) / "images" / "all")
    output_labels = ensure_dir(Path(args.output) / "labels" / "all")

    # Locate label JSON files
    label_dir = input_dir / "labels"
    if not label_dir.exists():
        # Some distributions put JSONs directly under input_dir
        label_dir = input_dir

    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON label files found under {label_dir}")
        sys.exit(1)

    print(f"[BDD100K] Found {len(json_files)} label file(s) in {label_dir}")

    # Locate image root
    img_root_candidates = [
        input_dir / "images" / "100k",
        input_dir / "images",
        input_dir,
    ]

    total_images = 0
    total_boxes = 0
    skipped_boxes = 0

    for jf in json_files:
        print(f"[BDD100K] Processing {jf.name} ...")
        with open(jf, "r", encoding="utf-8") as f:
            records = json.load(f)

        if isinstance(records, dict):
            records = records.get("images", records.get("frames", []))
            if not records:
                # Possibly nested: {image_name: {labels: [...]}}
                records = list(records.values())

        night_records = []
        for rec in records:
            # Determine time of day
            attrs = rec.get("attributes", {})
            tod = attrs.get("timeofday", "").lower().strip()
            if not tod:
                # Some formats store timeofday at top level
                tod = rec.get("timeofday", "").lower().strip()
            if tod in NIGHT_KEYWORDS:
                night_records.append(rec)

        print(f"  Total records: {len(records)}, night-time: {len(night_records)}")

        for rec in tqdm(night_records, desc="  BDD100K night"):
            # Image name
            img_name = rec.get("name", "")
            if not img_name:
                # Try nested image path
                img_name = rec.get("image_path", "")
            if not img_name:
                continue

            img_stem = Path(img_name).stem

            # Find the actual image file
            img_path = None
            for root in img_root_candidates:
                candidate = root / img_name
                if candidate.exists():
                    img_path = candidate
                    break
                # Try without subdirectory
                for sub in root.rglob(img_name):
                    if sub.exists():
                        img_path = sub
                        break
                if img_path:
                    break

            if img_path is None or not img_path.exists():
                # If image not found, try to get size from record
                img_w = rec.get("width", 1280)
                img_h = rec.get("height", 720)
            else:
                sz = get_image_size_fast(img_path)
                if sz is None:
                    img_w, img_h = rec.get("width", 1280), rec.get("height", 720)
                else:
                    img_w, img_h = sz

            # Collect labels
            labels = rec.get("labels", [])
            boxes = []
            for lbl in labels:
                cat = lbl.get("category", "").lower().strip()
                if cat in BDD100K_SKIP_CLASSES:
                    continue
                unified = BDD100K_CLASS_MAP.get(cat)
                if unified is None:
                    continue
                cls_id = CLASS_NAME_TO_ID[unified]

                bbox = lbl.get("box2d")
                if bbox is None:
                    continue
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                x2 = bbox.get("x2", 0)
                y2 = bbox.get("y2", 0)

                yolo_box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                if yolo_box is None:
                    skipped_boxes += 1
                    continue
                boxes.append((cls_id, *yolo_box))

            # Only write label if we have at least one box
            if not boxes:
                continue

            # Copy image if found
            if img_path and img_path.exists():
                dst_img = output_dir / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

            # Write label
            label_file = output_labels / f"{img_stem}.txt"
            write_yolo_label(label_file, boxes)
            total_images += 1
            total_boxes += len(boxes)

    print(f"[BDD100K] Done. Images written: {total_images}, boxes: {total_boxes}, "
          f"skipped (too small): {skipped_boxes}")


# ---------------------------------------------------------------------------
# NightOwls processing
# ---------------------------------------------------------------------------

def process_nightowls(args: argparse.Namespace):
    """
    Convert NightOwls annotations to YOLO format.

    Expected layout:
        args.input/
            annotations/
                *.json
            images/
                *.jpg  (or *.png)
    """
    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output) / "images" / "all")
    output_labels = ensure_dir(Path(args.output) / "labels" / "all")

    # Locate annotations
    ann_dir = input_dir / "annotations"
    if not ann_dir.exists():
        ann_dir = input_dir  # fallback

    json_files = sorted(ann_dir.glob("*.json"))
    if not json_files:
        # NightOwls may also use .mat or other formats; try .mat
        json_files = sorted(ann_dir.glob("*.mat"))
        if not json_files:
            print(f"[ERROR] No annotation files found under {ann_dir}")
            sys.exit(1)

    print(f"[NightOwls] Found {len(json_files)} annotation file(s) in {ann_dir}")

    # Locate image root
    img_dir_candidates = [
        input_dir / "images",
        input_dir / "JPEGImages",
        input_dir,
    ]

    total_images = 0
    total_boxes = 0

    for af in tqdm(json_files, desc="NightOwls files"):
        if af.suffix == ".json":
            with open(af, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # .mat files - try scipy
            try:
                from scipy.io import loadmat
                data = loadmat(str(af), squeeze_me=True)
            except ImportError:
                print("[WARN] scipy not installed, cannot read .mat files. "
                      "Install with: pip install scipy")
                continue
            except Exception as e:
                print(f"[WARN] Could not read {af}: {e}")
                continue

        # NightOwls JSON format: list of frames, each with image path and annotations
        # or dict with image_name -> annotations
        records = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            # Try common keys
            for key in ("annotations", "images", "frames", "data"):
                if key in data and isinstance(data[key], (list, dict)):
                    if isinstance(data[key], list):
                        records = data[key]
                    else:
                        records = list(data[key].values())
                    break
            if not records:
                # Treat dict values as records
                records = list(data.values())

        for rec in records:
            if not isinstance(rec, dict):
                continue

            # Image path
            img_name = rec.get("image_path", rec.get("name", rec.get("filename", "")))
            if not img_name:
                continue

            img_stem = Path(img_name).stem

            # Find image
            img_path = None
            for root in img_dir_candidates:
                candidate = root / img_name
                if candidate.exists():
                    img_path = candidate
                    break
                # Try just the filename
                candidate = root / Path(img_name).name
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None or not img_path.exists():
                img_w = rec.get("width", 640)
                img_h = rec.get("height", 480)
            else:
                sz = get_image_size_fast(img_path)
                img_w, img_h = sz if sz else (rec.get("width", 640), rec.get("height", 480))

            # Annotations - pedestrian bounding boxes
            anns = rec.get("annotations", rec.get("bboxes", rec.get("objects", [])))
            if isinstance(anns, dict):
                anns = list(anns.values())

            boxes = []
            for ann in anns:
                if isinstance(ann, dict):
                    cat = ann.get("category", ann.get("label", "pedestrian")).lower()
                    bbox = ann.get("bbox", ann.get("box2d", ann.get("rect", None)))
                elif isinstance(ann, (list, tuple)) and len(ann) >= 4:
                    cat = "pedestrian"
                    bbox = {"x1": ann[0], "y1": ann[1],
                            "x2": ann[0] + ann[2], "y2": ann[1] + ann[3]}
                else:
                    continue

                unified = NIGHTOWLS_CLASS_MAP.get(cat, "person")
                cls_id = CLASS_NAME_TO_ID[unified]

                if isinstance(bbox, dict):
                    x1 = bbox.get("x1", bbox.get("left", 0))
                    y1 = bbox.get("y1", bbox.get("top", 0))
                    x2 = bbox.get("x2", bbox.get("right", 0))
                    y2 = bbox.get("y2", bbox.get("bottom", 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                else:
                    continue

                yolo_box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                if yolo_box is None:
                    continue
                boxes.append((cls_id, *yolo_box))

            if not boxes:
                continue

            # Copy image
            if img_path and img_path.exists():
                dst_img = output_dir / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

            # Write label
            label_file = output_labels / f"{img_stem}.txt"
            write_yolo_label(label_file, boxes)
            total_images += 1
            total_boxes += len(boxes)

    print(f"[NightOwls] Done. Images written: {total_images}, boxes: {total_boxes}")


# ---------------------------------------------------------------------------
# ExDark processing
# ---------------------------------------------------------------------------

def process_exdark(args: argparse.Namespace):
    """
    Convert ExDark annotations to YOLO format.

    Expected layout (common variants):
        args.input/
            images/
                *.jpg
            annotations/
                *.txt   (one per image, each line: class x1 y1 w h)
            -- OR --
            Bicycle/
                *.jpg
            Car/
                *.jpg
            annotations/
                *.txt
    """
    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output) / "images" / "all")
    output_labels = ensure_dir(Path(args.output) / "labels" / "all")

    # Locate annotations
    ann_dir = input_dir / "annotations"
    ann_files = sorted(ann_dir.glob("*.txt")) if ann_dir.exists() else []

    if not ann_files:
        # Try alternative: annotations are .xml (PASCAL VOC style)
        ann_files = sorted(ann_dir.glob("*.xml")) if ann_dir.exists() else []

    if not ann_files:
        # ExDark sometimes puts annotations in a flat folder per-class
        for sub in input_dir.iterdir():
            if sub.is_dir() and sub.name.lower() in {v.lower() for v in EXDARK_CLASS_MAP}:
                for txt in sub.glob("*.txt"):
                    ann_files.append(txt)
        ann_files = sorted(ann_files)

    if not ann_files:
        print(f"[ERROR] No annotation files found under {input_dir}")
        sys.exit(1)

    print(f"[ExDark] Found {len(ann_files)} annotation file(s)")

    # Image search roots
    img_search_roots = [input_dir / "images", input_dir]
    # Also add class subdirectories
    for sub in input_dir.iterdir():
        if sub.is_dir() and sub.name.lower() in {v.lower() for v in EXDARK_CLASS_MAP}:
            img_search_roots.append(sub)

    total_images = 0
    total_boxes = 0

    for af in tqdm(ann_files, desc="ExDark files"):
        img_stem = af.stem

        # Find the corresponding image
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            for root in img_search_roots:
                candidate = root / f"{img_stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                break

        if img_path is None:
            # Try recursive search
            for root in img_search_roots:
                for candidate in root.rglob(f"{img_stem}.*"):
                    if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                        img_path = candidate
                        break
                if img_path:
                    break

        if img_path is None or not img_path.exists():
            continue

        sz = get_image_size_fast(img_path)
        if sz is None:
            continue
        img_w, img_h = sz

        boxes = []

        if af.suffix == ".txt":
            with open(af, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cat = parts[0]
                    # Format: class x1 y1 w h  (pixels)
                    try:
                        x1 = float(parts[1])
                        y1 = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                    except ValueError:
                        continue
                    x2 = x1 + bw
                    y2 = y1 + bh

                    unified = EXDARK_CLASS_MAP.get(cat)
                    if unified is None:
                        # Try case-insensitive match
                        for k, v in EXDARK_CLASS_MAP.items():
                            if k.lower() == cat.lower():
                                unified = v
                                break
                    if unified is None:
                        continue

                    cls_id = CLASS_NAME_TO_ID[unified]
                    yolo_box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                    if yolo_box is None:
                        continue
                    boxes.append((cls_id, *yolo_box))

        elif af.suffix == ".xml":
            # Minimal PASCAL VOC XML parsing without external libs
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(af))
                root = tree.getroot()
                for obj in root.findall("object"):
                    cat = obj.findtext("name", "")
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue
                    x1 = float(bndbox.findtext("xmin", "0"))
                    y1 = float(bndbox.findtext("ymin", "0"))
                    x2 = float(bndbox.findtext("xmax", "0"))
                    y2 = float(bndbox.findtext("ymax", "0"))

                    unified = EXDARK_CLASS_MAP.get(cat)
                    if unified is None:
                        for k, v in EXDARK_CLASS_MAP.items():
                            if k.lower() == cat.lower():
                                unified = v
                                break
                    if unified is None:
                        continue

                    cls_id = CLASS_NAME_TO_ID[unified]
                    yolo_box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                    if yolo_box is None:
                        continue
                    boxes.append((cls_id, *yolo_box))
            except Exception as e:
                print(f"[WARN] Failed to parse {af}: {e}")
                continue

        if not boxes:
            continue

        # Copy image
        dst_img = output_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Write label
        label_file = output_labels / f"{img_stem}.txt"
        write_yolo_label(label_file, boxes)
        total_images += 1
        total_boxes += len(boxes)

    print(f"[ExDark] Done. Images written: {total_images}, boxes: {total_boxes}")


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_frames(args: argparse.Namespace):
    """
    Extract frames from video files at a configurable FPS.
    Uses perceptual hashing to skip near-duplicate frames.

    Expected layout:
        args.input/
            *.mp4 / *.avi / *.mkv / ...
    """
    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output) / "images" / "all")
    output_labels_dir = ensure_dir(Path(args.output) / "labels" / "all")

    fps = args.fps
    dedup_threshold = args.dedup_threshold

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    video_files = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in video_extensions
    )

    if not video_files:
        print(f"[ERROR] No video files found in {input_dir}")
        sys.exit(1)

    print(f"[Extract] Found {len(video_files)} video(s), extracting at {fps} FPS")

    total_frames = 0
    sequence_registry: Dict[str, List[str]] = {}  # seq_id -> [frame_names]

    use_cv2 = False
    try:
        import cv2
        use_cv2 = True
        print("[Extract] Using OpenCV for frame extraction")
    except ImportError:
        print("[Extract] OpenCV not found, will try ffmpeg subprocess")

    for vf in video_files:
        seq_id = vf.stem
        print(f"  Processing: {vf.name} (sequence: {seq_id})")

        frame_dir = ensure_dir(output_dir / seq_id)
        seen_hashes: List[str] = []
        frame_idx = 0

        if use_cv2:
            cap = cv2.VideoCapture(str(vf))
            if not cap.isOpened():
                print(f"  [WARN] Cannot open {vf}")
                continue
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0
            frame_interval = max(1, int(round(video_fps / fps)))

            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=total_video_frames, desc=f"  {vf.stem}", unit="frame")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pbar.update(1)

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                fname = f"{seq_id}_{frame_idx:08d}.jpg"
                fpath = frame_dir / fname

                cv2.imwrite(str(fpath), frame)

                # Dedup
                ph = perceptual_hash(fpath)
                if ph is not None:
                    is_dup = False
                    for existing in seen_hashes:
                        if hamming_distance(ph, existing) <= dedup_threshold:
                            is_dup = True
                            break
                    if is_dup:
                        fpath.unlink(missing_ok=True)
                        frame_idx += 1
                        continue
                    seen_hashes.append(ph)

                # Also copy to flat "all" dir
                flat_dst = output_dir / fname
                if not flat_dst.exists():
                    shutil.copy2(fpath, flat_dst)

                if seq_id not in sequence_registry:
                    sequence_registry[seq_id] = []
                sequence_registry[seq_id].append(fname)
                total_frames += 1
                frame_idx += 1

            pbar.close()
            cap.release()

        else:
            # ffmpeg fallback
            import subprocess
            import tempfile

            tmpdir = Path(tempfile.mkdtemp(prefix="frames_"))

            cmd = [
                "ffmpeg", "-i", str(vf),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(tmpdir / f"{seq_id}_%08d.jpg"),
                "-y", "-hide_banner", "-loglevel", "error",
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError:
                print("  [ERROR] ffmpeg not found. Install ffmpeg or install opencv-python.")
                continue
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] ffmpeg failed for {vf}: {e}")
                continue

            extracted = sorted(tmpdir.glob("*.jpg"))
            for ef in tqdm(extracted, desc=f"  {vf.stem} dedup"):
                ph = perceptual_hash(ef)
                if ph is not None:
                    is_dup = False
                    for existing in seen_hashes:
                        if hamming_distance(ph, existing) <= dedup_threshold:
                            is_dup = True
                            break
                    if is_dup:
                        ef.unlink(missing_ok=True)
                        continue
                    seen_hashes.append(ph)

                fname = ef.name
                dst = frame_dir / fname
                shutil.move(str(ef), str(dst))

                flat_dst = output_dir / fname
                if not flat_dst.exists():
                    shutil.copy2(dst, flat_dst)

                if seq_id not in sequence_registry:
                    sequence_registry[seq_id] = []
                sequence_registry[seq_id].append(fname)
                total_frames += 1

            shutil.rmtree(tmpdir, ignore_errors=True)

    # Write sequence registry for the split command
    registry_path = Path(args.output) / "sequences.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(sequence_registry, f, indent=2)

    print(f"[Extract] Done. Total frames extracted: {total_frames}, "
          f"sequences: {len(sequence_registry)}")
    print(f"  Sequence registry saved to {registry_path}")


# ---------------------------------------------------------------------------
# Dataset splitting (sequence-aware)
# ---------------------------------------------------------------------------

def split_dataset(args: argparse.Namespace):
    """
    Split the dataset by sequence / video (not random frames from same video).
    Creates train/val/test directories with 70/15/15 ratio.
    Generates splits.csv tracking each image.
    """
    dataset_dir = Path(args.dataset)
    img_all = dataset_dir / "images" / "all"
    lbl_all = dataset_dir / "labels" / "all"

    if not img_all.exists():
        print(f"[ERROR] Image directory not found: {img_all}")
        sys.exit(1)

    # Load sequence registry if available
    registry_path = dataset_dir / "sequences.json"
    sequence_registry: Dict[str, List[str]] = {}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            sequence_registry = json.load(f)
        print(f"[Split] Loaded sequence registry with {len(sequence_registry)} sequences")

    # Gather all images
    all_images = list_images(img_all)
    if not all_images:
        print(f"[ERROR] No images found in {img_all}")
        sys.exit(1)
    print(f"[Split] Found {len(all_images)} total images")

    # Determine sequence membership for each image
    img_to_seq: Dict[str, str] = {}
    if sequence_registry:
        for seq_id, frame_names in sequence_registry.items():
            for fname in frame_names:
                img_to_seq[fname] = seq_id
    else:
        # No sequence registry - group by parent directory or filename prefix
        for img in all_images:
            # If image is in a subdirectory under "all", use that as sequence
            rel = img.relative_to(img_all)
            if len(rel.parts) > 1:
                seq_id = rel.parts[0]
            else:
                # Use filename prefix up to last underscore
                stem = img.stem
                parts = stem.rsplit("_", 1)
                seq_id = parts[0] if len(parts) > 1 else stem
            img_to_seq[img.name] = seq_id
            if seq_id not in sequence_registry:
                sequence_registry[seq_id] = []
            sequence_registry[seq_id].append(img.name)

    # Collect unique sequences
    all_sequences = sorted(set(img_to_seq.values()))
    if not all_sequences:
        # Every image is its own sequence
        all_sequences = [img.name for img in all_images]
        for img in all_images:
            img_to_seq[img.name] = img.name

    print(f"[Split] Unique sequences: {len(all_sequences)}")

    # Deterministic shuffle of sequences
    seed = args.seed
    import random
    rng = random.Random(seed)
    shuffled_seqs = list(all_sequences)
    rng.shuffle(shuffled_seqs)

    # Compute split boundaries
    n = len(shuffled_seqs)
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1.0 - train_ratio - val_ratio

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train_seqs = set(shuffled_seqs[:n_train])
    val_seqs = set(shuffled_seqs[n_train:n_train + n_val])
    test_seqs = set(shuffled_seqs[n_train + n_val:])

    seq_to_split: Dict[str, str] = {}
    for s in train_seqs:
        seq_to_split[s] = "train"
    for s in val_seqs:
        seq_to_split[s] = "val"
    for s in test_seqs:
        seq_to_split[s] = "test"

    # Create output directories
    for split in ("train", "val", "test"):
        ensure_dir(dataset_dir / "images" / split)
        ensure_dir(dataset_dir / "labels" / split)

    # Move / copy files and write splits.csv
    csv_path = dataset_dir / "splits.csv"
    split_counts = {"train": 0, "val": 0, "test": 0}

    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["image_path", "label_path", "source", "sequence_id", "split"])

        for img in tqdm(all_images, desc="Splitting"):
            seq_id = img_to_seq.get(img.name, img.stem)
            split = seq_to_split.get(seq_id, "train")

            # Destination paths
            dst_img = dataset_dir / "images" / split / img.name
            src_label = lbl_all / f"{img.stem}.txt"
            dst_label = dataset_dir / "labels" / split / f"{img.stem}.txt"

            # Move image
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            img.unlink(missing_ok=True)

            # Move label if it exists
            if src_label.exists():
                if not dst_label.exists():
                    shutil.copy2(src_label, dst_label)
                src_label.unlink(missing_ok=True)

            # Determine source from filename prefix or parent dir
            source = "unknown"
            name_lower = img.name.lower()
            if "bdd" in name_lower:
                source = "bdd100k"
            elif "nightowl" in name_lower:
                source = "nightowls"
            elif "exdark" in name_lower:
                source = "exdark"
            elif seq_id in sequence_registry:
                source = "video"

            writer.writerow([
                f"images/{split}/{img.name}",
                f"labels/{split}/{img.stem}.txt",
                source, seq_id, split,
            ])
            split_counts[split] += 1

    print(f"[Split] Done. Train: {split_counts['train']}, "
          f"Val: {split_counts['val']}, Test: {split_counts['test']}")
    print(f"  Splits CSV saved to {csv_path}")

    # Clean up empty "all" dirs if empty
    if img_all.exists() and not any(img_all.iterdir()):
        img_all.rmdir()
    if lbl_all.exists() and not any(lbl_all.iterdir()):
        lbl_all.rmdir()


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_stats(args: argparse.Namespace):
    """
    Print dataset statistics:
      - Total images, images per split
      - Instances per class
      - Average box size (normalised)
      - Small / medium / large target ratio
    """
    dataset_dir = Path(args.dataset)

    split_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = defaultdict(int)
    box_areas: List[float] = []
    total_boxes = 0

    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            continue

        images = list_images(img_dir)
        split_counts[split] = len(images)

        for img in images:
            label_file = lbl_dir / f"{img.stem}.txt"
            if not label_file.exists():
                continue
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                    except ValueError:
                        continue
                    if 0 <= cls_id < len(CLASS_NAMES):
                        class_counts[CLASS_NAMES[cls_id]] += 1
                    area = w * h
                    box_areas.append(area)
                    total_boxes += 1

    # Print summary
    total_images = sum(split_counts.values())
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total images:        {total_images}")
    for split, cnt in split_counts.items():
        print(f"    {split:8s}:          {cnt}")
    print(f"  Total instances:     {total_boxes}")
    print()

    if class_counts:
        print("  Instances per class:")
        for cls in CLASS_NAMES:
            cnt = class_counts.get(cls, 0)
            pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0
            print(f"    {cls:14s}: {cnt:6d}  ({pct:5.1f}%)")
        print()

    if box_areas:
        avg_w = sum(float(b) for b in box_areas) / len(box_areas)  # avg normalised area
        # Note: box_areas is w*h (both normalised), so sqrt gives a rough side length
        avg_side = (avg_w ** 0.5) if avg_w > 0 else 0.0
        print(f"  Average box area (norm): {avg_w:.6f}")
        print(f"  Average box side (norm): {avg_side:.6f}")
        print()

        # COCO-style size categories (normalised)
        #   small:  area < 32^2 / (640*640) ~ 0.0025
        #   medium: 0.0025 <= area < 96^2 / (640*640) ~ 0.0225
        #   large:  area >= 0.0225
        # But since we have normalised coords, thresholds are:
        small_thresh = 32 * 32 / (640 * 640)   # ~0.0025
        large_thresh = 96 * 96 / (640 * 640)   # ~0.0225
        small = sum(1 for a in box_areas if a < small_thresh)
        medium = sum(1 for a in box_areas if small_thresh <= a < large_thresh)
        large = sum(1 for a in box_areas if a >= large_thresh)
        print(f"  Size distribution (COCO-style, normalised):")
        print(f"    Small  (area < {small_thresh:.4f}):  {small:6d} ({small/total_boxes*100:5.1f}%)")
        print(f"    Medium ({small_thresh:.4f} <= area < {large_thresh:.4f}): {medium:6d} ({medium/total_boxes*100:5.1f}%)")
        print(f"    Large  (area >= {large_thresh:.4f}): {large:6d} ({large/total_boxes*100:5.1f}%)")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# YAML config generation
# ---------------------------------------------------------------------------

def create_yaml(args: argparse.Namespace):
    """
    Generate night_road.yaml config file for YOLO training.
    """
    dataset_dir = Path(args.dataset)
    yaml_path = dataset_dir / "night_road.yaml"

    # Compute absolute path for the dataset root
    abs_dataset = dataset_dir.resolve()

    # Build YAML content
    lines = [
        "# Night Road Detection Dataset Configuration",
        "# Generated by prepare_datasets.py",
        "",
        f"path: {abs_dataset.as_posix()}",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        "",
        "# Number of classes",
        f"nc: {len(CLASS_NAMES)}",
        "",
        "# Class names",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: '{name}'")

    # Append additional metadata
    lines.extend([
        "",
        "# Dataset metadata",
        "# Sources: BDD100K (night), NightOwls, ExDark, custom video",
        "# Class mapping:",
        "#   BDD100K: pedestrian->person, rider, bicycle, motorcycle, car, bus, truck",
        "#   NightOwls: pedestrian->person",
        "#   ExDark: People->person, Bicycle->bicycle, Motorbike->motorcycle, Car, Bus, Truck",
    ])

    yaml_content = "\n".join(lines) + "\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[YAML] Configuration written to {yaml_path}")
    print(yaml_content)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for night road detection training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_datasets.py bdd100k --input /data/bdd100k --output datasets/night_road
  python prepare_datasets.py nightowls --input /data/nightowls --output datasets/night_road
  python prepare_datasets.py exdark --input /data/exdark --output datasets/night_road
  python prepare_datasets.py extract-frames --input /data/videos --output datasets/night_road --fps 2
  python prepare_datasets.py split --dataset datasets/night_road
  python prepare_datasets.py stats --dataset datasets/night_road
  python prepare_datasets.py create-yaml --dataset datasets/night_road
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # -- bdd100k --
    p_bdd = subparsers.add_parser("bdd100k", help="Process BDD100K night-time labels")
    p_bdd.add_argument("--input", required=True, help="Path to BDD100K dataset root")
    p_bdd.add_argument("--output", required=True, help="Output dataset directory")

    # -- nightowls --
    p_no = subparsers.add_parser("nightowls", help="Process NightOwls annotations")
    p_no.add_argument("--input", required=True, help="Path to NightOwls dataset root")
    p_no.add_argument("--output", required=True, help="Output dataset directory")

    # -- exdark --
    p_ex = subparsers.add_parser("exdark", help="Process ExDark annotations")
    p_ex.add_argument("--input", required=True, help="Path to ExDark dataset root")
    p_ex.add_argument("--output", required=True, help="Output dataset directory")

    # -- extract-frames --
    p_ext = subparsers.add_parser("extract-frames", help="Extract frames from videos")
    p_ext.add_argument("--input", required=True, help="Path to directory with video files")
    p_ext.add_argument("--output", required=True, help="Output dataset directory")
    p_ext.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract (default: 2)")
    p_ext.add_argument("--dedup-threshold", type=int, default=10,
                       help="Perceptual hash Hamming distance threshold for dedup (default: 10)")

    # -- split --
    p_sp = subparsers.add_parser("split", help="Split dataset into train/val/test by sequence")
    p_sp.add_argument("--dataset", required=True, help="Dataset directory to split")
    p_sp.add_argument("--train-ratio", type=float, default=0.70, help="Training set ratio (default: 0.70)")
    p_sp.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    p_sp.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # -- stats --
    p_st = subparsers.add_parser("stats", help="Print dataset statistics")
    p_st.add_argument("--dataset", required=True, help="Dataset directory to analyse")

    # -- create-yaml --
    p_y = subparsers.add_parser("create-yaml", help="Generate YOLO night_road.yaml config")
    p_y.add_argument("--dataset", required=True, help="Dataset directory")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "bdd100k":         process_bdd100k,
        "nightowls":       process_nightowls,
        "exdark":          process_exdark,
        "extract-frames":  extract_frames,
        "split":           split_dataset,
        "stats":           compute_stats,
        "create-yaml":     create_yaml,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Running: {args.command}")
    print(f"{'='*60}\n")

    handler(args)

    print(f"\n[Done] Command '{args.command}' completed successfully.\n")


if __name__ == "__main__":
    main()
