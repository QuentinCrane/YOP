# YOP - Real-time YOLO Camera on phone

Android phone-side real-time road perception project focused on on-device YOLO inference, rider-facing alerts, and a matching training/export pipeline.

## What This Repo Contains

This repository currently has two main parts:

- `android-app/`: Android app built with Jetpack Compose, CameraX, LiteRT, and ONNX Runtime
- `training/`: dataset preparation, training, evaluation, and model export scripts for the YOLO pipeline

In the current codebase, the app uses YOLO models for live camera detection and can optionally load an `openpilot`-style `supercombo` ONNX model to enrich lead-vehicle perception when that asset is present.

## Current App Capabilities

- Real-time camera inference on Android phones
- YOLO model hot-switching through the app model manager
- LiteRT backend with GPU first, plus fallback behavior in the inference stack
- CameraX preview and analysis pipeline
- Object tracking for more stable overlay behavior
- Riding-risk evaluation with vibration and sound alerts
- Optional `supercombo` ONNX inference path for lead-vehicle merging
- Runtime performance monitoring and file-based logging

## Repository Structure

```text
night-road-vision/
├─ android-app/
│  ├─ app/
│  │  ├─ src/main/java/com/nightroadvision/app/
│  │  │  ├─ MainActivity.kt
│  │  │  ├─ NightRoadApp.kt
│  │  │  ├─ FileLogger.kt
│  │  │  ├─ CrashLogger.kt
│  │  │  ├─ camera/
│  │  │  ├─ inference/
│  │  │  ├─ model/
│  │  │  ├─ performance/
│  │  │  ├─ tracking/
│  │  │  ├─ alert/
│  │  │  └─ ui/
│  │  └─ src/main/assets/
│  │     ├─ labels.txt
│  │     └─ models/
│  └─ gradle/
├─ training/
│  ├─ configs/
│  ├─ docs/
│  ├─ scripts/
│  ├─ TRAINING_GUIDE.md
│  └─ DEPLOY_3090.md
├─ DEVELOPMENT.md
└─ OPENPILOT_HUD_RESEARCH.md
```

## Android Stack

- Kotlin + Jetpack Compose
- CameraX
- Google LiteRT (`com.google.ai.edge.litert`)
- ONNX Runtime Android
- Material 3

Key app settings from the current Gradle config:

- `compileSdk = 35`
- `targetSdk = 35`
- `minSdk = 26`
- Java/Kotlin target = `11`

## Included Model Assets

The repository currently includes multiple model assets under `android-app/app/src/main/assets/models/`, including lightweight YOLO variants that are suitable for running on-device.

The Gradle build also contains a lightweight asset-sync step so larger experimental assets are not packaged into every APK by default.

## Quick Start

### 1. Open the Android project

Open `android-app/` in Android Studio.

### 2. Build and install

On Windows:

```powershell
cd android-app
.\gradlew.bat installDebug
```

On macOS or Linux:

```bash
cd android-app
./gradlew installDebug
```

### 3. Grant permissions

The app currently requests:

- Camera permission
- Vibration permission

### 4. Run on a physical device

A real device is recommended because the project depends on camera input and mobile inference performance.

## Training Pipeline

The training workspace under `training/` includes scripts for:

- downloading or preparing datasets
- converting annotations into the YOLO workflow
- running training
- evaluating exported models
- exporting mobile assets for Android use

Useful entry points:

- [training/TRAINING_GUIDE.md](/E:/Github%20Program/YLOP/night-road-vision/training/TRAINING_GUIDE.md)
- [training/DEPLOY_3090.md](/E:/Github%20Program/YLOP/night-road-vision/training/DEPLOY_3090.md)
- [training/docs/DATASET_CARD.md](/E:/Github%20Program/YLOP/night-road-vision/training/docs/DATASET_CARD.md)
- [training/docs/MODEL_CARD.md](/E:/Github%20Program/YLOP/night-road-vision/training/docs/MODEL_CARD.md)

Example flow:

```bash
cd training/scripts
python prepare_datasets.py --help
python train.py --help
python export_models.py --help
python evaluate.py --help
```

## Notes About the Current State

- The repository name is still `night-road-vision`, but the project title in this README is now `YOP - Real-time YOLO Camera on phone`
- The Android app name shown in resources is currently `夜视路况`
- `supercombo` support is optional in code and depends on the corresponding ONNX model asset being available
- This repo already contains committed model files, so `.gitignore` should be kept careful not to hide the tracked assets you still want in source control

## Development Docs

- [DEVELOPMENT.md](/E:/Github%20Program/YLOP/night-road-vision/DEVELOPMENT.md)
- [OPENPILOT_HUD_RESEARCH.md](/E:/Github%20Program/YLOP/night-road-vision/OPENPILOT_HUD_RESEARCH.md)

## License

No explicit license file is present in the repository at the moment. If you plan to publish or collaborate externally, adding a license file would be a good next step.
