# Night Road Vision

Real-time night-time road object detection for Android, powered by a custom-trained YOLO26n model exported to TFLite with GPU delegate support.

## Project Overview

Night Road Vision is an Android application that provides real-time object detection optimized for low-light / night driving conditions. It detects pedestrians, riders, cyclists, motorcycles, cars, buses, and trucks using a lightweight YOLO model running on-device via TensorFlow Lite.

### Architecture

```
night-road-vision/
  android-app/          # Jetpack Compose Android application
    app/src/main/java/com/nightroadvision/app/
      MainActivity.kt           # Entry point, permission handling, immersive mode
      MainViewModel.kt          # Pipeline orchestrator (camera -> inference -> tracking -> UI)
      NightRoadApp.kt           # Application class
      camera/
        CameraManager.kt        # CameraX lifecycle, preview, and frame analysis
      inference/
        InferenceEngine.kt      # TFLite interpreter with GPU delegate, YUV preprocessing, NMS
      tracking/
        ObjectTracker.kt        # ByteTrack-inspired multi-object tracker with EMA smoothing
      performance/
        PerformanceMonitor.kt   # Real-time FPS, latency percentiles, thermal monitoring
      ui/
        screen/
          MainScreen.kt         # Primary detection screen with camera preview and overlays
        overlay/
          DetectionOverlay.kt   # Bounding box canvas, performance HUD
        theme/
          Color.kt              # Dark-first color palette for night use
          Theme.kt              # Material 3 dark theme
          Type.kt               # Typography with monospace telemetry readouts
  training/
    configs/
      hyperparams_stage1.yaml       # Stage 1 hyperparameters (public data pre-adaptation)
      hyperparams_stage2.yaml       # Stage 2 hyperparameters (phone data fine-tune)
      device_rtx3090.yaml           # RTX 3090 device profile
      augmentation_night.yaml       # Night-scene augmentation settings
    scripts/
      train.py                      # Two-stage training pipeline
      prepare_datasets.py           # Dataset preparation (BDD100K, NightOwls, ExDark, video)
      export_models.py              # Model export to TFLite FP16/INT8 and QNN
      evaluate.py                   # Evaluation: standard, slice-based, video, multi-format
      download_model.py             # Quick-start: download model, export TFLite, deploy
  docs/                             # Documentation (placeholder)
  evaluation/                       # Evaluation results (placeholder)
```

### Key Features

- **7-class detection**: person, rider, bicycle, motorcycle, car, bus, truck
- **Per-class confidence thresholds**: lower thresholds for pedestrians/riders (safety-critical)
- **Three performance modes**: ECO (every 3rd frame), BALANCED (every 2nd), FINE (every frame)
- **Auto device calibration**: measures inference latency on first launch to select optimal mode
- **ByteTrack-inspired tracking**: persistent object IDs with EMA-smoothed bounding boxes
- **GPU delegate with CPU fallback**: TFLite GPU acceleration with automatic fallback to 4-thread CPU
- **Immersive fullscreen**: landscape orientation, hidden system bars, keep-screen-on
- **Real-time performance HUD**: detection FPS, preview FPS, inference latency, thermal status
- **Dark-first UI**: Material 3 dark theme optimized for night-vision preservation

### Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLO26n (nano) |
| Input resolution | 512x320 (balanced) |
| Quantization | FP16 (GPU), INT8 (CPU/NNAPI) |
| Output format | (1, N, 6) end-to-end or (1, C, M) traditional |
| Delegate | TFLite GPU with CPU fallback |

## Build Instructions

### Prerequisites

- Android Studio Ladybug (2024.2) or later
- JDK 11+
- Android SDK 35
- Python 3.10+ (for training/export scripts)
- CUDA-capable GPU with 24GB+ VRAM (for training)

### Android App

1. **Obtain a TFLite model**. Either:
   - Run the quick-start download script:
     ```bash
     cd training/scripts
     pip install ultralytics
     python download_model.py
     ```
   - Or train your own model (see Training below).

2. **Open the project** in Android Studio:
   ```
   File > Open > E:/Github Program/YLOP/night-road-vision/android-app
   ```

3. **Sync Gradle** and let dependencies download.

4. **Build and run** on a device with a camera:
   ```bash
   cd android-app
   ./gradlew installDebug
   ```

The app will request camera permission on first launch, auto-detect device capability, and begin real-time detection.

### Training (Optional)

Training requires an NVIDIA GPU with at least 24GB VRAM (RTX 3090 recommended).

1. **Install Python dependencies**:
   ```bash
   pip install ultralytics torch torchvision tqdm pandas pyyaml opencv-python
   ```

2. **Prepare datasets** (BDD100K night, NightOwls, ExDark, or custom video):
   ```bash
   cd training/scripts
   python prepare_datasets.py bdd100k --input /path/to/bdd100k --output datasets/night_road
   python prepare_datasets.py extract-frames --input /path/to/videos --output datasets/night_road --fps 2
   python prepare_datasets.py split --dataset datasets/night_road
   python prepare_datasets.py create-yaml --dataset datasets/night_road
   ```

3. **Run two-stage training**:
   ```bash
   python train.py both \
       --data-stage1 datasets/night_road/night_road.yaml \
       --data-stage2 datasets/night_road_phone/night_road.yaml
   ```

4. **Export to mobile formats**:
   ```bash
   python export_models.py all \
       --model runs/night-road/stage2_finetune/weights/best.pt \
       --calibration-data datasets/night_road/night_road.yaml \
       --android-assets ../../android-app/app/src/main/assets
   ```

5. **Evaluate**:
   ```bash
   python evaluate.py standard --model best.pt --data night_road.yaml
   python evaluate.py compare --models best.pt fp16.tflite int8.tflite
   ```

## Configuration

### Performance Modes

| Mode | Frame Skip | Best For |
|------|-----------|----------|
| ECO | Every 3rd frame | Low-end devices, thermal throttling |
| BALANCED | Every 2nd frame | Default for most devices |
| FINE | Every frame | High-end devices, maximum accuracy |

The app auto-selects the best mode based on measured inference latency:
- < 25ms average: FINE
- 25-50ms average: BALANCED
- > 50ms average: ECO

### Augmentation Settings

Night-scene augmentation is configured in `training/configs/augmentation_night.yaml` and includes HSV jitter, geometric transforms, mosaic, mixup, and random erasing -- all tuned for low-light road scenes.

## License

This project is provided as-is for educational and research purposes.
