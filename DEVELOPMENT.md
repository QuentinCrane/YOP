# Night Road Vision -- Developer Guide

An Android app for real-time object detection on nighttime road scenes using YOLO models on TFLite, optimized for Snapdragon 8 Gen 3 / Adreno 750.

---

## 1. Architecture Overview

```
MainActivity
  └─ MainScreen (Compose)
       ├─ PreviewView          ← CameraX live preview
       ├─ DetectionOverlay     ← Canvas drawn on top of preview
       └─ MainScreenViewModel
            ├─ CameraManager   ← CameraX frames, YUV->letterbox conversion
            ├─ InferenceEngine ← TFLite interpreter, coordinate mapping
            ├─ ModelManager    ← Model selection, persistence
            └─ PerformanceMonitor ← Thermal/memory status
```

**Data flow:** Camera frame -> letterbox to model input size -> TFLite inference -> parse output -> reverse letterbox to camera coordinates -> rotate to display coordinates -> draw overlay.

All detection coordinates are maintained in **camera sensor space** until the overlay applies rotation and scaling to match the on-screen preview.

---

## 2. Development Setup

### Prerequisites

- Android Studio (Ladybug or later)
- JDK 11+
- Android SDK 35 (compileSdk)
- A physical device (camera + GPU delegate testing requires real hardware)

### Build

```bash
cd android-app
./gradlew assembleDebug
```

The APK is produced at `android-app/app/build/outputs/apk/debug/app-debug.apk`.

### Key Dependencies

| Library | Purpose |
|---------|---------|
| CameraX 1.4.2 | Camera preview + image analysis |
| LiteRT 1.0.1 | TFLite successor (GPU delegate, NNAPI) |
| Compose BOM 2025.01 | UI toolkit |
| Material3 | Theming and components |

### Model Files

Place `.tflite` models in `android-app/app/src/main/assets/models/`. The app scans this directory at startup via `ModelManager.refreshInstalledModels()`. A `labels.txt` in `assets/` provides class names (falls back to hardcoded COCO 80).

---

## 3. Key Components

### InferenceEngine (`inference/InferenceEngine.kt`)

Core inference wrapper. Handles:
- **Delegate fallback chain:** GPU -> NNAPI -> CPU. Automatically selects the best available backend.
- **Model loading and hot-swap:** `loadModel(assetPath)` reinitializes the interpreter without restarting the app.
- **Letterbox calculation:** Computes `scale`, `padX`, `padY` for mapping between camera frame and model input.
- **Output parsing:** Supports two formats:
  - Post-processed `[1, 300, 6]` (YOLO26n): each row is `[x1, y1, x2, y2, conf, class_id]`.
  - Raw `[1, 84, 8400]` (YOLOv8): transposed anchor grid, requires class-score argmax and cx/cy/w/h to x1/y1/x2/y2 conversion.
- **NMS:** Built-in non-maximum suppression using IoU threshold.
- **Benchmarking:** `benchmarkGpu()` / `runBenchmark()` for comparing delegate performance.

### CameraManager (`camera/CameraManager.kt`)

Manages the CameraX pipeline:
- Binds `Preview` + `ImageAnalysis` use cases.
- Converts YUV_420_888 frames to NV21, compresses to JPEG, decodes to Bitmap, then letterboxes to a float32 `ByteBuffer` normalized to [0, 1].
- Exposes `targetModelWidth` / `targetModelHeight` for dynamic letterbox target (updated when model changes).
- Handles torch control and Camera2Interop exposure settings (manual ISO, shutter speed, focus).

### ModelManager (`model/ModelManager.kt`)

Manages available YOLO models:
- Scans `assets/models/` for installed `.tflite` files.
- Persists the selected model in `SharedPreferences`.
- Exposes `StateFlow<ModelInfo>` for reactive UI updates.
- Each `ModelInfo` carries metadata: input dimensions, parameter count, file size.

### DetectionOverlay (`ui/overlay/DetectionOverlay.kt`)

Compose `Canvas` that draws bounding boxes over the camera preview:
- Receives detections in **camera sensor coordinates**.
- Applies sensor rotation (typically 90 degrees for portrait) via `rotateCoords()`.
- Scales to fill the overlay canvas using FILL_CENTER logic (scale + crop).
- Draws per-class colored rectangles with confidence labels.

### ObjectTracker (`tracking/ObjectTracker.kt`)

ByteTrack-inspired multi-object tracker:
- Greedy IoU-based matching between frames.
- EMA smoothing on bounding boxes and confidence.
- Tracks are promoted to "confirmed" after 2 consecutive matches and dropped after 10 missed frames.

### PerformanceMonitor (`performance/PerformanceMonitor.kt`)

Monitors device thermal state and memory pressure. Recommends stepping down detection mode (FINE -> BALANCED -> ECO) under thermal throttling.

### MainScreenViewModel (`ui/screen/MainScreenViewModel.kt`)

Orchestrates the full pipeline. Key responsibilities:
- Initializes `InferenceEngine` and `ModelManager` on creation.
- Creates `CameraManager` and wires the frame callback.
- Implements keep-only-latest backpressure (drops frames if inference is still running).
- Exposes `StateFlow` for detections, settings, performance metrics, and model name.
- Handles model switching, backend selection, and torch toggling.

---

## 4. How to Add a New Model

1. **Export the model** as TFLite (`.tflite`). Both float32 and float16 are supported.

2. **Place the file** in `android-app/app/src/main/assets/models/` (e.g., `yolov8l.tflite`).

3. **Register the model** in `ModelManager.kt` by adding an entry to the `allModels` list:
   ```kotlin
   ModelInfo(
       id = "yolov8l",
       name = "YOLOv8l",
       path = "models/yolov8l.tflite",
       size = "~84 MB",
       sizeBytes = 84_000_000L,
       description = "Large -- highest accuracy, slowest.",
       parameterCount = "43.7M",
       inputWidth = 640,   // must match actual model input
       inputHeight = 640,
       numClasses = 80,
   )
   ```

4. **If the model uses a different output format**, update `InferenceEngine.readModelInputDimensions()` and add a new parser alongside `parsePostProcessedOutput` / `parseRawOutput`. The engine auto-detects format by comparing output tensor dimensions (`rows < cols` means raw YOLOv8-style).

5. **If the model uses different class labels**, place a `labels.txt` in `assets/` (one class name per line).

6. Rebuild. The model will appear in the model selector bottom sheet automatically.

---

## 5. Detection Pipeline

The end-to-end pipeline from camera frame to on-screen bounding box:

```
Camera sensor (1280x720, landscape, YUV_420_888)
       |
       v
  CameraManager.processFrame()
    - YUV -> NV21 -> JPEG -> Bitmap
    - Letterbox to model input size (e.g. 512x320 or 640x640)
    - Normalize pixels to float32 [0,1]
    - Produce ByteBuffer(1, H, W, 3)
       |
       v
  MainScreenViewModel.onCameraFrame()
    - Update actual camera dimensions on first frame
    - Apply frame skipping (skip N-1 of every N frames)
    - Keep-only-latest guard (drop if previous inference still running)
       |
       v
  InferenceEngine.runInference()
    - TFLite interpreter.run(input, output)
    - Parse output (post-processed or raw format)
    - Filter by confidence threshold
    - Reverse letterbox: cameraCoord = (modelCoord - pad) / scale
    - Clamp to camera bounds
    - Apply NMS (IoU threshold)
    - Return InferenceResult with detections in camera sensor coordinates
       |
       v
  MainScreenViewModel
    - Publishes detections to StateFlow
    - Updates FPS / latency metrics
       |
       v
  DetectionOverlay (Compose Canvas)
    - Rotate coordinates (sensor -> display): 90deg CW for portrait
    - Scale + offset to fill canvas (FILL_CENTER)
    - Draw colored bounding boxes + class/confidence labels
```

### Coordinate Spaces

| Space | Description |
|-------|-------------|
| Model input | Pixel coordinates in the letterboxed model input (e.g. 512x320) |
| Camera sensor | Pixel coordinates in the raw camera frame (e.g. 1280x720, landscape) |
| Display | Screen pixels after rotation and FILL_CENTER scaling |

The `LetterboxParams` data class holds `scale`, `padX`, `padY` needed to convert model-input coordinates back to camera-sensor coordinates.

---

## 6. Common Development Tasks

### Changing Detection Thresholds

Thresholds are in `InferenceSettings` (defined in `MainScreen.kt`):

```kotlin
data class InferenceSettings(
    val confidenceThreshold: Float = 0.25f,  // min confidence to keep
    val iouThreshold: Float = 0.45f,         // NMS overlap threshold
    val frameSkip: Int = 1,                  // process every Nth frame
    ...
)
```

These are passed to `InferenceEngine.runInference()` on every frame. The UI exposes sliders in the settings panel. Presets are also available:

- **ECO:** conf=0.6, iou=0.5, skip=3
- **BALANCED:** conf=0.45, iou=0.45, skip=1
- **FINE:** conf=0.25, iou=0.4, skip=0

### Adding New Detection Classes

1. Train or obtain a model with the desired classes.
2. Update `numClasses` in the `ModelInfo` entry.
3. Place a `labels.txt` in `assets/` with class names (one per line).
4. The `CLASS_NAMES` list in `InferenceEngine` is used as a fallback; `ModelManager.loadClassLabels()` prefers `labels.txt`.

### Switching Compute Backend at Runtime

Call from the ViewModel:

```kotlin
viewModel.selectBackend(InferenceEngine.DelegateType.GPU)
```

Or through settings by changing `backendPreference` in `InferenceSettings`. The engine's fallback chain is: GPU -> NNAPI -> CPU. Use `InferenceEngine.getAvailableBackends()` to query what the device supports.

### Adjusting GPU Precision

Change `GpuPrecision` in settings:
- `FP16` -- faster on Adreno 750 (2x FP16 throughput), slight accuracy loss.
- `FP32` -- full precision, slower.

Applied via `InferenceEngine.updateGpuConfig(GpuConfig(...))`.

### Running Benchmarks

```kotlin
val results = inferenceEngine.benchmarkGpu(warmupRuns = 3, timedRuns = 10)
// Returns: {"GPU" to 10, "NNAPI" to 13, "CPU" to 32}  (ms per inference)
```

Or the more detailed `runBenchmark()` which returns `List<BenchmarkResult>` with min/max/avg per delegate.

### Modifying the Overlay Appearance

Overlay drawing is in `DetectionOverlay.kt`. Key customization points:
- `getClassColor(classId)` -- per-class color palette (8 colors, cycles).
- Bounding box stroke width: `3f * density`.
- Label text size: `11sp`.
- Label background: semi-transparent black (`argb(200, 0, 0, 0)`).

### Adjusting Camera Exposure

The app supports manual exposure via Camera2Interop:

```kotlin
cameraManager.updateExposureSettings(
    mode = CameraManager.ExposureMode.MANUAL,
    iso = 1600,
    shutterNs = 33_000_000L,  // 33ms
    compensation = 0,
    focus = CameraManager.FocusMode.CONTINUOUS
)
```

This rebinds the camera to apply the new CaptureRequest options.

### Logging

The app uses a dual logging system:
- `android.util.Log` for logcat.
- `FileLogger` for persistent file-based logging (written to app internal storage). Use `FileLogger.i/d/w/e(TAG, message)`.

### Night Vision Theme

The dark theme uses a night-vision green accent (`#00FF41`) defined in `NightVisionColors` within `MainScreen.kt`. The color scheme is applied via `NightVisionTheme` wrapping Material3's `darkColorScheme`.

---

## File Structure

```
android-app/app/src/main/
├── assets/
│   ├── models/
│   │   ├── yolo26n_float16.tflite
│   │   ├── yolov8n.tflite
│   │   ├── yolov8s.tflite
│   │   └── yolov8m.tflite
│   └── labels.txt
└── java/com/nightroadvision/app/
    ├── MainActivity.kt              # Entry point, permission handling
    ├── NightRoadApp.kt              # Application class
    ├── FileLogger.kt                # Persistent file logger
    ├── CrashLogger.kt               # Crash reporting
    ├── camera/
    │   └── CameraManager.kt         # CameraX + YUV processing
    ├── inference/
    │   └── InferenceEngine.kt       # TFLite wrapper + coordinate mapping
    ├── model/
    │   └── ModelManager.kt          # Model registry + selection
    ├── tracking/
    │   └── ObjectTracker.kt         # Multi-object tracker
    ├── performance/
    │   └── PerformanceMonitor.kt    # Thermal + memory monitoring
    └── ui/
        ├── theme/
        │   ├── Color.kt
        │   ├── Type.kt
        │   └── Theme.kt
        ├── overlay/
        │   └── DetectionOverlay.kt  # Bounding box rendering
        └── screen/
            ├── MainScreen.kt        # Compose UI + settings data classes
            └── MainScreenViewModel.kt  # Pipeline orchestration
```
