# YOLO26n-night-road Model Card

## Model Details
- Base model: YOLO26n (Nano)
- Architecture: YOLO26 Detect (end-to-end, NMS-free)
- Framework: Ultralytics / PyTorch → LiteRT (TFLite)
- Parameters: ~2.5M (Nano)
- Input: 640×640 (train), 512×320 / 640×384 / 416×256 (deploy)
- Output: Detection boxes with class + confidence

## Intended Use
- Real-time night road object detection on Android phones
- Person, rider, bicycle, motorcycle, car, bus, truck detection
- Experimental/entertainment tool, NOT a safety system

## Training Data
- BDD100K (night/dawn/dusk)
- NightOwls (night pedestrians)
- ExDark (low-light scenes)
- Phone-captured data (multiple devices)

## Training Procedure
- Stage 1: 120 epochs on public data, lr=0.01, batch=48, imgsz=640
- Stage 2: 50 epochs on phone data mix, lr=0.001, batch=48, imgsz=640
- Hardware: RTX 3090 24GB
- Mixed precision (FP16)

## Performance Targets
| Metric | Target |
|---|---|
| Night person Recall | ≥ 0.70 |
| Night car Recall | ≥ 0.80 |
| person AP50 | ≥ 0.75 |
| False alarms | ≤ 3/min |
| Inference FPS (mid-range) | ≥ 10 |
| P95 latency | ≤ 150ms |

## Exported Variants
| Variant | Size | Input | Format |
|---|---|---|---|
| eco_416x256_fp16 | ~3MB | 416×256 | TFLite FP16 |
| balanced_512x320_fp16 | ~3MB | 512×320 | TFLite FP16 |
| fine_640x384_fp16 | ~3MB | 640×384 | TFLite FP16 |
| balanced_512x320_int8 | ~2MB | 512×320 | TFLite INT8 |
| qnn_640x384 | ~3MB | 640×384 | QNN |

## Limitations
- Cannot detect in near-total darkness
- Struggles with very small distant pedestrians
- Motion blur degrades accuracy
- CameraX quality varies by device
- Not suitable for autonomous driving

## Ethical Considerations
- Not a replacement for driver attention
- Must display disclaimer in app
- No precise distance estimation
- Privacy: all processing on-device
