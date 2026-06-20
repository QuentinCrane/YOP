# Night Road Detection - Training Guide

This document covers the full training pipeline for the night-road-vision YOLO26n
object detection model, from environment setup through model export. All commands,
time estimates, and batch-size recommendations target an **NVIDIA RTX 3090 24 GB**.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Model Export](#model-export)
7. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 24 GB | RTX 3090 24 GB (primary target) |
| System RAM | 16 GB | 32 GB+ |
| Storage | 100 GB free SSD | 200 GB+ free SSD |
| CUDA | 12.x | 12.x |
| cuDNN | 8.9+ | latest |

**Why these specs matter:**

- The RTX 3090's 24 GB VRAM supports batch sizes of 48 at 640 px with the nano
  model (see `configs/device_rtx3090.yaml`). Larger models or higher resolutions
  require smaller batches.
- 32 GB system RAM is recommended because the dataloader uses 8 workers with
  `cache=disk`, which buffers image paths and metadata in memory.
- SSD storage is critical: the `cache=disk` option writes preprocessed image tiles
  to disk, and BDD100K alone is ~100 GB.

---

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n night-road python=3.11 -y
conda activate night-road
```

### 2. Install PyTorch with CUDA 12.x

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Project Dependencies

```bash
pip install ultralytics>=8.3.0
pip install albumentations>=1.4.0
pip install opencv-python>=4.8.0
pip install pandas>=2.0.0
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
pip install numpy>=1.24.0
pip install pillow>=10.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

Optional (for experiment tracking):

```bash
pip install wandb          # Weights & Biases logging
pip install tensorboard    # TensorBoard logging (default)
```

### 4. Verify Installation

```bash
# Check CUDA and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"

# Check Ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Ultralytics OK')"

# Check GPU memory
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

Expected output for the first command:

```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
VRAM: 24.0 GB
```

---

## Dataset Preparation

The unified dataset uses 7 classes:

```
person (0), rider (1), bicycle (2), motorcycle (3), car (4), bus (5), truck (6)
```

All annotation sources are converted to YOLO format (one `.txt` per image with
`class_id x_center y_center width height` lines, normalized to 0-1).

### BDD100K

**What it provides:** Large-scale driving dataset with day/night images. Night
images contain pedestrians, cars, buses, trucks, riders, bicycles, and motorcycles
with bounding box annotations.

**Download:**

1. Go to https://bdd-data.berkeley.edu/ and register.
2. Download "100K Images" (train + val splits, ~100 GB).
3. Download "Detection 2020 Labels" (bounding box JSON files).

**Expected directory layout:**

```
/path/to/bdd100k/
    labels/
        bdd100k_labels_images_train.json
        bdd100k_labels_images_val.json
    images/
        100k/
            train/
            val/
```

**Filtering and conversion:**

```bash
cd "E:/Github Program/YLOP/night-road-vision"

python training/scripts/prepare_datasets.py bdd100k \
    --input /path/to/bdd100k \
    --output training/datasets/night_road
```

The script filters images whose JSON label has `"timeOfDay": "night"` (or
`"dawn"`/`"dusk"`). It maps BDD100K class names to the unified 7-class scheme
(`pedestrian` -> `person`, `car` -> `car`, etc.) and skips irrelevant classes
(`train`, `traffic light`, `traffic sign`, `other vehicle`).

**Expected count after filtering:** ~20,000-30,000 night images from the combined
train+val splits, depending on BDD100K version.

### NightOwls

**What it provides:** Nighttime pedestrian detection dataset. Contains only
pedestrian annotations.

**Download:**

1. Go to http://www.vision.rwth-aachen.de/page/nightowls
2. Download the dataset (images + annotations).

**Expected directory layout:**

```
/path/to/nightowls/
    images/
        train/
        val/
        test/
    annotations/
        *.json   (COCO-format pedestrian annotations)
```

**Conversion:**

```bash
python training/scripts/prepare_datasets.py nightowls \
    --input /path/to/nightowls \
    --output training/datasets/night_road
```

All NightOwls pedestrian annotations are mapped to class `person` (class 0).

### ExDark

**What it provides:** Low-light object detection dataset with 12 object classes
including people, bicycles, motorcycles, cars, buses, and trucks.

**Download:**

1. Go to https://github.com/cs-chan/ExDark-Dataset
2. Download the dataset and annotations.

**Expected directory layout:**

```
/path/to/exdark/
    images/
        People/
        Bicycle/
        Motorbike/
        Car/
        Bus/
        Truck/
        ...
    annotations/
        *.txt   (bounding box annotations)
```

**Conversion:**

```bash
python training/scripts/prepare_datasets.py exdark \
    --input /path/to/exdark \
    --output training/datasets/night_road
```

ExDark class names (`People`, `Bicycle`, `Motorbike`, `Car`, `Bus`, `Truck`) are
mapped to the unified scheme.

### Phone Data Collection

Phone-captured data is essential for Stage 2 fine-tuning. It bridges the domain gap
between professional datasets and real-world phone camera characteristics (noise
patterns, exposure, resolution, lens distortion).

**Recommended phones and settings:**

| Setting | Recommendation |
|---------|---------------|
| Resolution | 1080p or higher |
| Frame rate | 30 fps for video, or burst photo mode |
| Exposure | Auto (do not lock exposure) |
| Focus | Auto (continuous AF) |
| Stabilization | Enable OIS/EIS if available |
| Format | H.264/H.265 video or JPEG photos |

Test on at least 2-3 different phone models to improve generalization.

**Scene checklist:**

Record or photograph each of these scenarios:

- [ ] Urban street with streetlights (well-lit road, dark sidewalks)
- [ ] Residential area with sparse lighting
- [ ] Highway/expressway at night
- [ ] Parking lot (indoor and outdoor)
- [ ] Pedestrian crossing with people present
- [ ] Road with oncoming headlights (backlit scenario)
- [ ] Rainy night (wet road reflections)
- [ ] Foggy or misty night
- [ ] Tunnel entry/exit (light transition)
- [ ] Areas with cyclists/motorcyclists

**Video recording tips:**

- Mount the phone on the dashboard or hold steadily -- shaky footage produces
  motion-blurred frames that are hard to label.
- Record at least 5 minutes per scene to capture varied traffic patterns.
- Drive at normal speed; do not slow down for the camera.
- Avoid pointing directly at bright light sources (causes lens flare artifacts).

**Frame extraction settings:**

```bash
python training/scripts/prepare_datasets.py extract-frames \
    --input /path/to/phone_videos \
    --output training/datasets/night_road \
    --fps 2
```

The `--fps 2` flag extracts 2 frames per second. This is sufficient to capture
scene variety without creating near-duplicate frames. The script applies
perceptual-hash deduplication to remove frames that are too similar (Hamming
distance threshold).

**Expected output after extraction:** For a 5-minute video at 30 fps, extracting at
2 fps yields ~600 frames. After deduplication, expect ~400-500 unique frames.

### Running Data Preparation

After collecting all sources, run the full pipeline:

```bash
cd "E:/Github Program/YLOP/night-road-vision"

# Step 1: Process each source (run in any order)
python training/scripts/prepare_datasets.py bdd100k \
    --input /path/to/bdd100k \
    --output training/datasets/night_road

python training/scripts/prepare_datasets.py nightowls \
    --input /path/to/nightowls \
    --output training/datasets/night_road

python training/scripts/prepare_datasets.py exdark \
    --input /path/to/exdark \
    --output training/datasets/night_road

python training/scripts/prepare_datasets.py extract-frames \
    --input /path/to/phone_videos \
    --output training/datasets/night_road_phone \
    --fps 2

# Step 2: Split into train/val/test (sequence-aware)
python training/scripts/prepare_datasets.py split \
    --dataset training/datasets/night_road

python training/scripts/prepare_datasets.py split \
    --dataset training/datasets/night_road_phone

# Step 3: Generate dataset statistics
python training/scripts/prepare_datasets.py stats \
    --dataset training/datasets/night_road

python training/scripts/prepare_datasets.py stats \
    --dataset training/datasets/night_road_phone

# Step 4: Create YOLO YAML configs
python training/scripts/prepare_datasets.py create-yaml \
    --dataset training/datasets/night_road

python training/scripts/prepare_datasets.py create-yaml \
    --dataset training/datasets/night_road_phone
```

**Expected output structure:**

```
training/datasets/night_road/
    images/
        train/       # ~70% of images
        val/         # ~15% of images
        test/        # ~15% of images
    labels/
        train/       # matching YOLO .txt labels
        val/
        test/
    night_road.yaml  # YOLO dataset config
    stats.json       # dataset statistics

training/datasets/night_road_phone/
    images/
        train/
        val/
        test/
    labels/
        train/
        val/
        test/
    night_road.yaml
    stats.json
```

**Verifying data quality:**

1. Check `stats.json` for class distribution -- ensure `person` and `car` are
   well-represented. If `person` is below 5% of total annotations, consider
   collecting more pedestrian footage.
2. Spot-check 10-20 random images with their labels:

   ```bash
   python -c "
   from ultralytics.data.utils import visualize_image_annotations
   import random, glob
   imgs = glob.glob('training/datasets/night_road/images/train/*.jpg')
   for img in random.sample(imgs, 5):
       visualize_image_annotations(img, img.replace('images','labels').replace('.jpg','.txt'))
   "
   ```

3. Confirm no empty-label images slipped through (images with objects but no
   corresponding `.txt` file, or `.txt` files with invalid coordinates).

---

## Training Pipeline

The training pipeline has two stages, managed by `training/scripts/train.py`.

### Stage 1: Public Data Pre-adaptation

**Purpose:** Train on the combined public night-road dataset (BDD100K night +
NightOwls + ExDark) to build a strong baseline model that understands nighttime
visual features before exposure to phone-specific data.

**Command:**

```bash
cd "E:/Github Program/YLOP/night-road-vision"

python training/scripts/train.py stage1 \
    --data training/datasets/night_road/night_road.yaml \
    --batch -1 \
    --workers 8 \
    --project runs/night-road \
    --logger tensorboard
```

**Key parameters** (from `configs/hyperparams_stage1.yaml`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `yolo26n.pt` | Nano model, fast training, fits mobile deployment |
| Epochs | 120 | Sufficient for convergence on night data |
| Image size | 640 | Standard YOLO resolution |
| Batch size | -1 (auto) | Auto-calibrates to ~48 on RTX 3090 |
| Learning rate (lr0) | 0.01 | Standard for YOLO nano training |
| LR schedule | Cosine annealing | Smooth convergence, no sudden LR drops |
| Patience | 30 | Early stopping after 30 epochs without improvement |
| Close mosaic | 15 | Disables mosaic augmentation in last 15 epochs for cleaner fine-tuning |
| AMP (FP16) | Enabled | ~2x training speed on RTX 3090 with negligible accuracy loss |
| Cache | `disk` | Caches preprocessed images to SSD, reduces I/O bottleneck |

**Night-specific augmentation** (from `configs/augmentation_night.yaml`):

| Augmentation | Value | Why |
|-------------|-------|-----|
| HSV brightness (hsv_v) | 0.4 | Most critical -- simulates exposure variance in night scenes |
| HSV saturation (hsv_s) | 0.5 | Night scenes are naturally desaturated |
| HSV hue (hsv_h) | 0.015 | Subtle -- night lighting already shifts hue |
| Mosaic | 1.0 | Stitches 4 images for scene diversity |
| Mixup | 0.1 | Light blending simulates headlight glow |
| Random erasing | 0.2 | Simulates partial occlusion |
| Horizontal flip | 0.5 | Doubles effective data for road scenes |
| Vertical flip | 0.0 | Disabled -- physically impossible for roads |

**Expected training time on RTX 3090:**

| Dataset size | Batch | Time per epoch | Total (120 epochs) |
|-------------|-------|----------------|---------------------|
| ~25K images | 48 | ~3-4 min | **6-8 hours** |
| ~50K images | 48 | ~6-8 min | **12-16 hours** |

With `patience=30` early stopping, training typically finishes in 80-100 epochs
(~5-10 hours).

**What to monitor during training:**

```bash
# TensorBoard
tensorboard --logdir runs/night-road/stage1_public

# Or check the CSV logs directly
cat runs/night-road/stage1_public/results.csv
```

Key metrics to watch:

- **box_loss** should decrease steadily. If it plateaus or increases, the model
  may be overfitting.
- **cls_loss** should decrease. A spike after epoch ~105 (when mosaic closes) is
  normal and temporary.
- **mAP50** should increase. The final value for Stage 1 on public data is
  typically **0.45-0.60** for the nano model.
- **mAP50-95** is the stricter metric. A final value of **0.25-0.40** is typical.

**When to stop early:**

- `patience=30` handles this automatically. If mAP50-95 does not improve for 30
  consecutive epochs, training stops and `best.pt` is saved.
- If `box_loss` starts increasing while `mAP50` stalls, stop manually (Ctrl+C).
  The script saves `last.pt` on interrupt, and you can resume with `--resume`.

**Output:**

```
runs/night-road/stage1_public/
    weights/
        best.pt       # Best mAP50-95 checkpoint
        last.pt       # Final epoch checkpoint
    results.csv       # Per-epoch metrics
    results.png       # Loss/mAP plots
    confusion_matrix.png
    training_config.json  # Full config snapshot (auto-saved)
```

### Stage 2: Phone Data Fine-tune

**Purpose:** Fine-tune the Stage 1 model on phone-captured night-road data,
mixed with a portion of public data to prevent catastrophic forgetting.

**Command:**

```bash
python training/scripts/train.py stage2 \
    --data training/datasets/night_road_phone/night_road.yaml \
    --stage1-weights runs/night-road/stage1_public/weights/best.pt \
    --batch -1 \
    --workers 8 \
    --project runs/night-road \
    --logger tensorboard
```

**Key parameter differences from Stage 1** (from `configs/hyperparams_stage2.yaml`):

| Parameter | Stage 1 | Stage 2 | Rationale |
|-----------|---------|---------|-----------|
| Learning rate (lr0) | 0.01 | **0.001** | 10x lower to preserve learned features |
| Epochs | 120 | **50** | Fewer epochs needed; phone data is smaller |
| Patience | 30 | **20** | Tighter early stopping for smaller dataset |
| Close mosaic | 15 | **10** | Earlier mosaic cutoff for cleaner fine-tuning |
| Mosaic | 1.0 | **0.8** | Reduced -- phone data is already diverse |
| Mixup | 0.1 | **0.05** | Less aggressive blending |
| Scale jitter | 0.3 | **0.2** | Less scale variation for domain-specific tuning |

**Expected training time on RTX 3090:**

| Phone dataset size | Time per epoch | Total (50 epochs) |
|--------------------|----------------|---------------------|
| ~5K images | ~30-45 sec | **25-40 min** |
| ~10K images | ~1-1.5 min | **50-75 min** |

**Output:**

```
runs/night-road/stage2_finetune/
    weights/
        best.pt
        last.pt
    results.csv
    results.png
    training_config.json
```

### Running Both Stages Sequentially

```bash
python training/scripts/train.py both \
    --data-stage1 training/datasets/night_road/night_road.yaml \
    --data-stage2 training/datasets/night_road_phone/night_road.yaml \
    --batch -1 \
    --workers 8 \
    --project runs/night-road
```

This runs Stage 1 to completion, then automatically passes the best weights to
Stage 2. Total time: **6-10 hours** (dominated by Stage 1).

### Resuming Training

If training is interrupted:

```bash
# Resume Stage 1 from last checkpoint
python training/scripts/train.py stage1 \
    --data training/datasets/night_road/night_road.yaml \
    --resume \
    --project runs/night-road

# Resume Stage 2 from last checkpoint
python training/scripts/train.py stage2 \
    --data training/datasets/night_road_phone/night_road.yaml \
    --resume \
    --project runs/night-road
```

---

## Evaluation

### Standard Evaluation

Run full evaluation on the held-out test split:

```bash
python training/scripts/evaluate.py standard \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --data training/datasets/night_road_phone/night_road.yaml \
    --img-size 640 \
    --batch-size 16 \
    --conf-threshold 0.25 \
    --iou-threshold 0.6
```

**Output includes:**

- Overall mAP50, mAP50-95, Precision, Recall
- Per-class Precision, Recall, mAP50, mAP50-95
- Confusion matrix plot
- PR curve plot
- F1 curve plot

All plots are saved to `runs/evaluate/standard/`.

### Slice Evaluation

Evaluate on challenging subsets to identify weaknesses:

```bash
python training/scripts/evaluate.py slices \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --slices-dir training/evaluation/slices/ \
    --img-size 640 \
    --batch-size 16
```

**Available slices:**

| Slice | Description | What it tests |
|-------|-------------|---------------|
| `night_person_small` | Small pedestrians at night | Detection of distant/small people |
| `night_person_dark_clothes` | Dark-clothed pedestrians | Low-contrast person detection |
| `night_backlight` | Backlit scenes (headlights) | Handling bright light sources |
| `night_rain` | Rainy night scenes | Wet road reflections, rain artifacts |
| `parking_garage` | Underground parking | Indoor low-light detection |
| `unseen_phone` | Phone not in training | Cross-device generalization |
| `daytime_general` | Daytime baseline | Ensures daytime performance is not degraded |

Each slice requires its own YAML file in the `slices-dir`. Create slice YAML files
that point to the relevant subset of test images.

### Video Evaluation

Process a test video and compute temporal quality metrics:

```bash
python training/scripts/evaluate.py video \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --video /path/to/test_video.mp4 \
    --output runs/evaluate/video/
```

### Model Comparison

Compare PyTorch, TFLite FP16, and TFLite INT8 models side-by-side:

```bash
python training/scripts/evaluate.py compare \
    --models \
        runs/night-road/stage2_finetune/weights/best.pt \
        exports/yolo26n_balanced_512x320_fp16.tflite \
        exports/yolo26n_balanced_512x320_int8.tflite
```

### Interpreting Results

**Acceptance criteria** (from the design document):

| Class | Metric | Minimum Threshold | Stretch Goal |
|-------|--------|-------------------|-------------|
| person | Recall | **>= 0.70** | >= 0.80 |
| car | Recall | **>= 0.80** | >= 0.90 |
| Overall | mAP50-95 | >= 0.30 | >= 0.40 |

**What to do if thresholds are not met:**

- **person Recall < 0.70:** The model is missing too many pedestrians. Add more
  night pedestrian data, increase `hsv_v` augmentation, or lower the confidence
  threshold during evaluation (try `--conf-threshold 0.15`).
- **car Recall < 0.80:** Collect more varied night driving footage. Ensure the
  training data includes both parked and moving vehicles.
- **Low mAP50-95 but acceptable Recall:** The model detects objects but with
  imprecise bounding boxes. Increase training epochs or image size.

---

## Model Export

After evaluation passes the acceptance criteria, export the model for mobile
deployment.

### TFLite FP16 Export (Three Resolutions)

```bash
python training/scripts/export_models.py tflite-fp16 \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --output exports/ \
    --android-assets android-app/app/src/main/assets
```

This produces three FP16 TFLite models at different resolutions:

| Preset | Resolution | Use Case | Expected Size |
|--------|-----------|----------|---------------|
| `eco_416x256` | 416x256 | Battery-saving mode | ~3-4 MB |
| `balanced_512x320` | 512x320 | Default mode | ~4-5 MB |
| `fine_640x384` | 640x384 | High-quality mode | ~5-7 MB |

### TFLite INT8 Export

```bash
python training/scripts/export_models.py tflite-int8 \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --output exports/ \
    --calibration-data training/datasets/night_road_phone/night_road.yaml \
    --android-assets android-app/app/src/main/assets
```

INT8 quantization requires calibration data (the `--calibration-data` YAML). The
exporter samples images from the training set to calibrate quantization ranges.

| Preset | Resolution | Expected Size |
|--------|-----------|---------------|
| `balanced_512x320` | 512x320 | ~2-3 MB |

### QNN Export (Qualcomm Snapdragon 8 Gen 3)

```bash
python training/scripts/export_models.py qnn \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --output exports/ \
    --calibration-data training/datasets/night_road_phone/night_road.yaml \
    --android-assets android-app/app/src/main/assets
```

This produces a QNN context-binary targeting Hexagon HTP architecture 75
(Snapdragon 8 Gen 3). QNN models cannot be validated on a desktop GPU; they must
be tested on-device.

### Export All Formats

```bash
python training/scripts/export_models.py all \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --output exports/ \
    --calibration-data training/datasets/night_road_phone/night_road.yaml \
    --android-assets android-app/app/src/main/assets
```

### Verifying Exported Models

The export script automatically runs an inference test on each TFLite model using
a dummy image. Check the output for `test=PASS` lines:

```
  -> exports/yolo26n_balanced_512x320_fp16.tflite  (4.23 MB, 12.3s, test=PASS)
  -> exports/yolo26n_balanced_512x320_int8.tflite   (2.41 MB, 18.7s, test=PASS)
```

Manual verification:

```bash
python -c "
from ultralytics import YOLO
import numpy as np

# Test FP16 model
model = YOLO('exports/yolo26n_balanced_512x320_fp16.tflite', task='detect')
dummy = np.zeros((320, 512, 3), dtype=np.uint8)
results = model.predict(source=dummy, verbose=False)
print(f'FP16 model loaded OK, {len(results)} results')

# Test INT8 model
model = YOLO('exports/yolo26n_balanced_512x320_int8.tflite', task='detect')
results = model.predict(source=dummy, verbose=False)
print(f'INT8 model loaded OK, {len(results)} results')
"
```

### Export Report

After export completes, find the full report at:

```
exports/export_report.md    # Human-readable summary table
exports/export_report.json  # Machine-readable JSON
```

Copy exported models to Android assets before building the APK:

```
android-app/app/src/main/assets/models/
    yolo26n_eco_416x256_fp16.tflite
    yolo26n_balanced_512x320_fp16.tflite
    yolo26n_fine_640x384_fp16.tflite
    yolo26n_balanced_512x320_int8.tflite
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions (in order):**

1. **Reduce batch size.** Use an explicit value instead of `-1`:
   ```bash
   python training/scripts/train.py stage1 --batch 32 ...
   ```
   On RTX 3090 with YOLO26n at 640 px, batch 48 is typical. If OOM, try 32, then
   24, then 16.

2. **Reduce image size:**
   ```bash
   python training/scripts/train.py stage1 --imgsz 512 ...
   ```
   At 512 px, batch 64 typically fits on RTX 3090.

3. **Reduce dataloader workers:**
   ```bash
   python training/scripts/train.py stage1 --workers 4 ...
   ```
   Each worker loads images into CPU memory before transferring to GPU. Fewer
   workers = less host memory pressure.

4. **Close other GPU-consuming processes:**
   ```bash
   nvidia-smi  # Check for other processes using VRAM
   ```

### Slow Training

**Symptoms:** Training speed is significantly slower than expected.

**Diagnostics and solutions:**

1. **Check data loading bottleneck.** If GPU utilization (shown in `nvidia-smi`) is
   below 80%, the dataloader is the bottleneck. Increase workers:
   ```bash
   --workers 8   # or 12 if CPU has enough cores
   ```

2. **Enable disk caching** if not already active (it is enabled by default in
   Stage 1). The first epoch will be slow as it writes the cache, but subsequent
   epochs will be much faster:
   ```yaml
   cache: disk   # in training config
   ```

3. **Verify SSD speed.** If using an HDD, the `cache=disk` option will be slow.
   Move the dataset to an SSD.

4. **Check AMP is enabled.** FP16 training is ~2x faster than FP32 on RTX 3090:
   ```yaml
   amp: True   # should be True by default
   ```

### Poor Night Recall

**Symptoms:** Person or car recall on night test images is below the acceptance
threshold.

**Solutions:**

1. **Add more night-specific training data.** The most impactful fix. Phone-captured
   night footage from the target deployment scenario (similar camera, similar
   lighting conditions) is the highest-value data.

2. **Lower confidence threshold during inference/evaluation:**
   ```bash
   python training/scripts/evaluate.py standard \
       --conf-threshold 0.15 ...   # default is 0.25
   ```
   Lower thresholds increase recall at the cost of more false positives. On
   mobile, the app can use a lower threshold and let the user dismiss false
   detections.

3. **Increase brightness augmentation** in `configs/augmentation_night.yaml`:
   ```yaml
   hsv_v: 0.5   # increase from 0.4
   ```

4. **Check class balance.** If `person` annotations are < 5% of total, the model
   under-represents pedestrians. Use copy-paste augmentation or oversample
   person-heavy images.

5. **Train longer.** Increase epochs from 120 to 200 in Stage 1, or from 50 to 80
   in Stage 2.

### Overfitting

**Symptoms:** Training loss decreases but validation loss/mAP plateaus or worsens.

**Solutions:**

1. **Increase augmentation intensity.** The night augmentation config already uses
   aggressive settings, but you can further increase:
   ```yaml
   erasing: 0.3      # more occlusion simulation
   scale: 0.4        # more scale variation
   mixup: 0.15       # more image blending
   ```

2. **Verify early stopping is active.** The `patience` parameter (30 for Stage 1,
   20 for Stage 2) should trigger automatic stopping. Check that `best.pt` is
   from an earlier epoch, not the final one.

3. **Add more data.** Especially phone-captured data with varied scenes, weather
   conditions, and phone models.

4. **Use stronger regularization:**
   ```yaml
   weight_decay: 0.001   # increase from 0.0005
   ```

### Import Errors

**Symptoms:** `ModuleNotFoundError: No module named 'ultralytics'`

```bash
pip install --upgrade ultralytics
```

**Symptoms:** `RuntimeError: CUDA driver version is insufficient`

```bash
# Check CUDA version
nvidia-smi   # Shows driver CUDA version
python -c "import torch; print(torch.version.cuda)"
```

The driver CUDA version must be >= the PyTorch CUDA version. Update the NVIDIA
driver if needed.

### Corrupted Cache

**Symptoms:** Training crashes with file read errors on cached images.

```bash
# Delete the cache and re-train
rm -rf training/datasets/night_road/images/train.cache
rm -rf training/datasets/night_road/images/val.cache
```

The cache will be regenerated on the next training run.

---

## Quick Reference Card

```bash
# Full pipeline in 5 commands:

# 1. Setup environment
conda create -n night-road python=3.11 -y && conda activate night-road
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics albumentations opencv-python pandas pyyaml tqdm

# 2. Prepare data
python training/scripts/prepare_datasets.py bdd100k --input /data/bdd100k --output training/datasets/night_road
python training/scripts/prepare_datasets.py split --dataset training/datasets/night_road
python training/scripts/prepare_datasets.py create-yaml --dataset training/datasets/night_road

# 3. Train both stages
python training/scripts/train.py both \
    --data-stage1 training/datasets/night_road/night_road.yaml \
    --data-stage2 training/datasets/night_road_phone/night_road.yaml \
    --batch -1

# 4. Evaluate
python training/scripts/evaluate.py standard \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --data training/datasets/night_road_phone/night_road.yaml

# 5. Export
python training/scripts/export_models.py all \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --calibration-data training/datasets/night_road_phone/night_road.yaml \
    --android-assets android-app/app/src/main/assets
```
