# Models

Edge Impulse `.eim` files are **compiled per target**, so each model is provided for two platforms:

| File | Stage | Platform |
|------|-------|----------|
| `stage1-yolox-aarch64.eim` | 1 — object detection (YOLOX-Nano, BYOM) | Linux AARCH64 (Rubik Pi 3 / QCS6490) |
| `stage2-yolo11nseg-aarch64.eim` | 2 — instance segmentation (YOLO11n-seg, BYOM Freeform) | Linux AARCH64 (Rubik Pi 3 / QCS6490) |
| `stage1-yolox-macos-arm64.eim` | 1 — object detection | macOS (Apple Silicon) — for local dev |
| `stage2-yolo11nseg-macos-arm64.eim` | 2 — instance segmentation | macOS (Apple Silicon) — for local dev |

Make them executable after cloning: `chmod +x models/*.eim`
(On macOS, also clear quarantine if blocked: `xattr -d com.apple.quarantine models/*.eim`.)

## Rebuilding for another target

Both stages are deployed through **BYOM (Bring Your Own Model)**:

- **Stage 1 — YOLOX detector:** upload the ONNX with a YOLO/YOLOX output parser.
  🔗 *[placeholder: Edge Impulse project page]* · 🔗 *[placeholder: ONNX download]*
- **Stage 2 — YOLO11n-seg:** upload the ONNX with **Freeform** output, 640×640, `0..1` scaling (see the main [README](../README.md#5-stage-2--instance-segmentation-via-byom-freeform)).

In each project: **Deployment → select your target** (e.g. `Linux (AARCH64)`, `Raspberry Pi`, `macOS`) **→ Build**, then drop the resulting `.eim` here.
