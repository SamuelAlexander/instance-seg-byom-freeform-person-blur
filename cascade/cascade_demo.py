"""
Two-stage cascade demo: live webcam or video file with split-view visualization.

Left panel:  Full frame + Stage 1 bounding boxes (object detection)
Right panel: Cropped ROI + Stage 2 instance segmentation masks

Usage:
    # Live webcam
    python cascade_demo.py \
      --stage1 ../models/stage1-yolox-aarch64.eim \
      --stage2 ../models/stage2-yolo11nseg-aarch64.eim \
      --metadata ../model_metadata.json

    # Video file
    python cascade_demo.py \
      --stage1 ../models/stage1-yolox-aarch64.eim \
      --stage2 ../models/stage2-yolo11nseg-aarch64.eim \
      --metadata ../model_metadata.json \
      --video input.mp4 --output output.mp4

Controls:
    q / ESC  — quit
    s        — save current frame as screenshot
    SPACE    — pause/resume
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from postprocess import YOLOSegPostprocessor
from edge_impulse_linux.runner import ImpulseRunner


# Colors for visualization
STAGE1_BOX_COLOR = (255, 255, 255)  # white
STAGE1_ACTIVE_COLOR = (0, 255, 255)  # yellow — the bbox sent to Stage 2
INFO_BG_COLOR = (40, 40, 40)
np.random.seed(42)
INSTANCE_COLORS = np.random.randint(60, 255, size=(80, 3), dtype=np.uint8)


def pack_rgb_features(img_bgr, target_size):
    """Resize and pack RGB into EI feature format."""
    resized = cv2.resize(img_bgr, (target_size, target_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)
    return ((r << 16) | (g << 8) | b).flatten().astype(np.float32).tolist()


def run_stage1(runner, img_bgr, input_size):
    """Run Stage 1 object detection."""
    features = pack_rgb_features(img_bgr, input_size)
    t0 = time.time()
    result = runner.classify(features)
    dt = (time.time() - t0) * 1000
    return result["result"]["bounding_boxes"], dt


def run_stage2(runner, img_bgr, metadata, conf_thresh=0.25):
    """Run Stage 2 instance segmentation."""
    input_size = metadata.get("image_size", 640)
    num_classes = metadata.get("num_classes", 80)

    features = pack_rgb_features(img_bgr, input_size)
    t0 = time.time()
    result = runner.classify(features)
    dt = (time.time() - t0) * 1000

    freeform = result.get("result", {}).get("freeform", None)
    if freeform is None:
        return [], dt

    num_det_channels = 4 + num_classes + 32
    expected_det_size = num_det_channels * 8400

    flat_tensors = [np.array(t, dtype=np.float32) for t in freeform]
    if flat_tensors[0].size == expected_det_size:
        output0_flat, output1_flat = flat_tensors[0], flat_tensors[1]
    else:
        output0_flat, output1_flat = flat_tensors[1], flat_tensors[0]

    num_anchors = output0_flat.size // num_det_channels
    mask_h = mask_w = int(np.sqrt(output1_flat.size // 32))

    output0 = output0_flat.reshape(1, num_det_channels, num_anchors)
    output1 = output1_flat.reshape(1, mask_h, mask_w, 32).transpose(0, 3, 1, 2)

    pp = YOLOSegPostprocessor(
        num_classes=num_classes,
        conf_thresh=conf_thresh,
        iou_thresh=0.7,
        img_size=input_size,
    )
    instances = pp.process(output0, output1, orig_img_shape=img_bgr.shape[:2])
    return instances, dt


def draw_stage1(frame, bboxes, input_size, active_idx=0):
    """Draw Stage 1 bounding boxes on frame. Highlight the active one."""
    vis = frame.copy()
    h, w = frame.shape[:2]
    sx, sy = w / input_size, h / input_size

    for i, bb in enumerate(bboxes):
        x1 = int(bb["x"] * sx)
        y1 = int(bb["y"] * sy)
        x2 = int((bb["x"] + bb["width"]) * sx)
        y2 = int((bb["y"] + bb["height"]) * sy)

        is_active = (i == active_idx)
        color = STAGE1_ACTIVE_COLOR if is_active else STAGE1_BOX_COLOR
        thickness = 2 if is_active else 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"{bb['label']} {bb['value']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Title
    cv2.putText(vis, "Stage 1: Object Detection", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vis


def draw_stage2(crop, instances, class_names, alpha=0.5):
    """Draw Stage 2 segmentation masks on cropped image."""
    vis = crop.copy()

    for inst in instances:
        cid = inst["class_id"]
        color = INSTANCE_COLORS[cid % 80].tolist()
        mask = inst["mask"]
        score = inst["score"]
        name = class_names.get(cid, str(cid))

        # Resize mask to crop dimensions if needed
        if mask.shape[:2] != crop.shape[:2]:
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))

        mask_bool = mask > 127
        vis[mask_bool] = (
            np.array(color, dtype=np.uint8) * alpha +
            vis[mask_bool] * (1 - alpha)
        ).astype(np.uint8)

        # Draw contours for crisp edges
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 1)

        # Label
        bbox = inst["bbox"]
        bx1 = max(0, int(bbox[0] * crop.shape[1] / 640))
        by1 = max(0, int(bbox[1] * crop.shape[0] / 640))
        cv2.putText(vis, f"{name} {score:.2f}", (bx1, max(by1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Title
    cv2.putText(vis, "Stage 2: Instance Segmentation", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vis


def get_crop(frame, bbox, input_size, padding=0.1):
    """Crop a bounding box region from frame with padding."""
    h, w = frame.shape[:2]
    sx, sy = w / input_size, h / input_size

    x1 = int(bbox["x"] * sx)
    y1 = int(bbox["y"] * sy)
    x2 = int((bbox["x"] + bbox["width"]) * sx)
    y2 = int((bbox["y"] + bbox["height"]) * sy)

    # Add padding
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def letterbox(img, target_w, target_h):
    """Resize image to fit target dimensions while preserving aspect ratio.
    Pads with black bars."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_info_bar(width, s1_time, s2_time, fps, frame_num, total_dets, total_segs):
    """Draw info bar at the bottom."""
    bar_h = 40
    bar = np.full((bar_h, width, 3), INFO_BG_COLOR, dtype=np.uint8)

    texts = [
        f"FPS: {fps:.1f}",
        f"S1: {s1_time:.0f}ms",
        f"S2: {s2_time:.0f}ms",
        f"Det: {total_dets}",
        f"Seg: {total_segs}",
        f"Frame: {frame_num}",
    ]
    x = 10
    for t in texts:
        cv2.putText(bar, t, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        x += len(t) * 12 + 20

    return bar


def main():
    parser = argparse.ArgumentParser(description="Cascade demo with split-view")
    parser.add_argument("--stage1", type=str, required=True)
    parser.add_argument("--stage2", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="../model_metadata.json")
    parser.add_argument("--video", type=str, default=None, help="Input video file (default: webcam)")
    parser.add_argument("--output", type=str, default=None, help="Output video file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--panel-width", type=int, default=640, help="Width of each panel")
    parser.add_argument("--panel-height", type=int, default=480, help="Height of each panel")
    parser.add_argument("--skip", type=int, default=3,
                        help="Run Stage 2 every N frames (reuse last result otherwise). 1=every frame.")
    args = parser.parse_args()

    # Load metadata
    class_names = {}
    metadata = {}
    if os.path.exists(args.metadata):
        with open(args.metadata) as f:
            metadata = json.load(f)
        class_names = {int(k): v for k, v in metadata.get("class_names", {}).items()}

    # Initialize runners
    print("Loading Stage 1 model...")
    runner1 = ImpulseRunner(args.stage1)
    info1 = runner1.init(debug=False)
    s1_input = info1["model_parameters"]["image_input_width"]
    s1_labels = info1["model_parameters"]["labels"]
    print(f"  YoloX-Nano: {s1_input}x{s1_input}, {len(s1_labels)} classes")

    print("Loading Stage 2 model...")
    runner2 = ImpulseRunner(args.stage2)
    runner2.init(debug=False)
    print(f"  YOLO11n-seg: {metadata.get('image_size', 640)}x{metadata.get('image_size', 640)}")

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Webcam: device 0")

    if not cap.isOpened():
        print("ERROR: Could not open video source")
        runner1.stop()
        runner2.stop()
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pw, ph = args.panel_width, args.panel_height
    canvas_w = pw * 2
    canvas_h = ph + 40  # panels + info bar

    # Output video writer — match input FPS (default 30) so output plays at normal speed
    writer = None
    is_video = args.video is not None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = src_fps if is_video else 30
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (canvas_w, canvas_h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

    print(f"\nRunning cascade demo ({pw}x{ph} per panel)")
    if is_video and args.output:
        print(f"Processing video: {total_frames} frames → {args.output} @ {out_fps:.0f} FPS")
        print("(Stage 2 runs every frame for video output)")
    else:
        print("Controls: q/ESC=quit, s=screenshot, SPACE=pause")

    frame_num = 0
    paused = False
    fps_alpha = 0.9
    smoothed_fps = 0.0
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")

    # Cached Stage 2 results for frame skipping
    cached_right = np.full((ph, pw, 3), 30, dtype=np.uint8)
    cv2.putText(cached_right, "Stage 2: Waiting...", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
    cached_instances = []
    cached_s2_time = 0.0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.video:
                        print("End of video")
                    break
                frame_num += 1

            t_frame_start = time.time()

            # --- Stage 1 (runs every frame) ---
            bboxes, s1_time = run_stage1(runner1, frame, s1_input)

            # Pick the largest detection for the crop view
            active_idx = 0
            if len(bboxes) > 1:
                areas = [bb["width"] * bb["height"] for bb in bboxes]
                active_idx = int(np.argmax(areas))

            # Draw left panel
            left = draw_stage1(frame, bboxes, s1_input, active_idx)
            left = cv2.resize(left, (pw, ph))

            # --- Stage 2 (every frame for video output, every N frames for live) ---
            if is_video and args.output:
                run_s2 = True  # process every frame for smooth output video
            else:
                run_s2 = (frame_num % args.skip == 1) or args.skip == 1
            s2_time = cached_s2_time

            if len(bboxes) > 0:
                if run_s2:
                    instances, s2_time = run_stage2(runner2, frame, metadata, args.conf)
                    cached_s2_time = s2_time
                else:
                    instances = cached_instances

                # Get crop region from active Stage 1 bbox
                crop, crop_coords = get_crop(frame, bboxes[active_idx], s1_input)

                if crop.size > 0:
                    # Filter instances that overlap with crop region
                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                    crop_instances = []
                    for inst in instances:
                        full_mask = inst["mask"]
                        cropped_mask = full_mask[crop_y1:crop_y2, crop_x1:crop_x2]
                        if cropped_mask.sum() > 0:
                            crop_instances.append({
                                **inst,
                                "mask": cropped_mask,
                            })

                    right = draw_stage2(crop, crop_instances, class_names)
                    right = letterbox(right, pw, ph)
                else:
                    right = np.full((ph, pw, 3), 30, dtype=np.uint8)
                    cv2.putText(right, "Stage 2: No valid crop", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if run_s2:
                    cached_right = right.copy()
                    cached_instances = instances
            else:
                right = np.full((ph, pw, 3), 30, dtype=np.uint8)
                cv2.putText(right, "Stage 2: Waiting for detection...", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                instances = []

            # Compose canvas
            panels = np.hstack([left, right])

            t_frame_end = time.time()
            frame_fps = 1.0 / max(t_frame_end - t_frame_start, 0.001)
            smoothed_fps = fps_alpha * smoothed_fps + (1 - fps_alpha) * frame_fps if smoothed_fps > 0 else frame_fps

            info_bar = draw_info_bar(
                canvas_w, s1_time, s2_time, smoothed_fps,
                frame_num, len(bboxes), len(instances) if len(bboxes) > 0 else 0
            )
            canvas = np.vstack([panels, info_bar])

            # Write to output video
            if writer:
                writer.write(canvas)

            # For video file processing with output, skip display for speed
            # and show progress instead
            if is_video and args.output:
                if frame_num % 10 == 0 or frame_num == total_frames:
                    elapsed = time.time() - t_frame_start
                    pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                    print(f"\r  [{frame_num}/{total_frames}] {pct:.0f}% "
                          f"S1:{s1_time:.0f}ms S2:{s2_time:.0f}ms", end="", flush=True)
            else:
                # Live display for webcam or video without output
                cv2.imshow("Cascade Demo: Detection -> Segmentation", canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                elif key == ord("s"):
                    os.makedirs(screenshot_dir, exist_ok=True)
                    fname = os.path.join(screenshot_dir, f"cascade_frame_{frame_num:05d}.jpg")
                    cv2.imwrite(fname, canvas)
                    print(f"Screenshot saved: {fname}")
                elif key == ord(" "):
                    paused = not paused
                    print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
        print(f"Video saved to {args.output}")
    cv2.destroyAllWindows()
    runner1.stop()
    runner2.stop()
    print(f"Done. Processed {frame_num} frames.")


if __name__ == "__main__":
    main()
