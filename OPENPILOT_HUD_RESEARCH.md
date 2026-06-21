# Openpilot HUD research for Night Road Vision

Research baseline: comma.ai/openpilot commit `ad5151b38b540e08a01c87029c3ddb0979ba4fdf`.

## What the current openpilot HUD actually does

### Camera-first composition

`AugmentedRoadView` renders the road camera first, clips every overlay to an inner
content rectangle, then draws model visualization, HUD, alerts and driver state.
The outer border changes with system state. This keeps the road image primary and
prevents overlays from escaping the camera viewport.

Night Road Vision adaptation:

- keep the camera full-screen;
- clip every detection and label to the preview;
- place status and controls at the edges;
- show a bottom alert only while a tracked target is risky;
- use the thin outer border as persistent state feedback.

### Lead-vehicle marker

Openpilot does not draw generic object-detection rectangles. For radar/model lead
vehicles, `ModelRenderer` draws:

- a yellow outer triangular glow;
- a red inner chevron;
- increasing fill opacity as relative distance decreases;
- additional opacity when relative velocity is negative;
- a size clamped to a narrow range so the marker never becomes enormous.

Night Road Vision adaptation:

- normal people and vehicles retain thin corner brackets and compact labels;
- only the selected central risk target receives the yellow/red chevron;
- the selected target comes from confirmed YOLO tracks;
- qualitative box size and growth replace unavailable radar distance/velocity.

### Alerts

`AlertRenderer` uses three visual severities:

- normal: dark neutral;
- user prompt: orange (`#DA6F25`);
- critical: red (`#C92231`).

Small and medium alerts occupy the lower part of the view; only genuinely critical
system alerts may take over the whole view. Stale alerts are rejected, and the mici
implementation briefly holds the previous alert to prevent visual flicker.

Night Road Vision adaptation:

- `CAUTION`: confirmed target in the central attention zone;
- `CRITICAL`: target is very large or grows quickly across frames;
- clear state requires several consecutive clear evaluations;
- no full-screen object warning, because it would hide the hazard itself.

### Audible and haptic behavior

Openpilot's `soundd` has distinct prompt, soft-warning and immediate-warning sounds.
Immediate warnings repeat and ramp toward maximum volume over four seconds. Volume is
adapted to measured ambient noise. Openpilot's dedicated hardware does not implement
Android phone vibration.

Night Road Vision adaptation:

- use Android `VibrationEffect` instead of copying the audio daemon;
- one short pulse for caution;
- two pulses for critical risk;
- per-severity cooldown plus a global minimum interval;
- alert only confirmed tracks and expose an explicit settings toggle.

## Model reuse assessment

| Openpilot model | Size | Inputs | Output | Suitable here? |
|---|---:|---|---|---|
| `driving_supercombo.onnx` | 60.9 MB | two 12-channel image tensors, recurrent features, desire, traffic convention, action | one 2576-value planning tensor | No |
| `big_driving_supercombo.onnx` | 195.5 MB | same multi-input driving context | one 2580-value planning tensor | No |
| `dmonitoring_model.onnx` | 7.5 MB | driver image and calibration | driver-monitoring tensor | No |

The driving models predict path, lane/road structure, motion and lead hypotheses. They
do not produce general `person`, `bicycle`, `motorcycle`, `car`, `bus` and `truck`
detections. They also depend on openpilot-specific camera warping and temporal state.
Using them would increase runtime and integration complexity without meeting this
project's object-detection requirement.

The bundled YOLO26n FP16 model is about 5.4 MB and directly produces the required
object detections. It remains the default lightweight path. The final accuracy target
should be a seven-class YOLO26n model trained on the night-road datasets and phone
CameraX footage specified in the project plan.

## Safety boundary

Bounding-box size is not a calibrated metric distance. The current implementation may
show qualitative far/medium/near state and box-growth risk, but must not display meters
until camera intrinsics and class-specific geometry are calibrated and validated.

## Upstream references

- `selfdrive/ui/onroad/augmented_road_view.py`
- `selfdrive/ui/onroad/model_renderer.py`
- `selfdrive/ui/onroad/alert_renderer.py`
- `selfdrive/ui/soundd.py`
- `cereal/log.capnp`
- repository `LICENSE` (MIT)
