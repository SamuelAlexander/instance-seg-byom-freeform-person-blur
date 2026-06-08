"""
Person blur application: Two-stage cascade for privacy/anonymization.

Uses instance segmentation to precisely blur all detected persons in a video
or live webcam feed. Demonstrates real-world value of pixel-accurate masks
over simple bounding-box blurring.

Layout (2x2 grid):
  Top-left:     Stage 1 — Object detection (person bboxes)
  Top-right:    Stage 2 — Cropped view of largest person with segmentation mask
  Bottom:       Application output — original frame with all persons blurred

Usage:
    # Live webcam
    python person_blur.py \
      --stage1 ../models/stage1-yolox-aarch64.eim \
      --stage2 ../models/stage2-yolo11nseg-aarch64.eim \
      --metadata ../model_metadata.json

    # Video file
    python person_blur.py \
      --stage1 ../models/stage1-yolox-aarch64.eim \
      --stage2 ../models/stage2-yolo11nseg-aarch64.eim \
      --metadata ../model_metadata.json \
      --video input.mp4 --output blurred_output.mp4

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


# COCO class ID for "person"
PERSON_CLASS_ID = 0

# Live preview window title
WINDOW_NAME = "Person Blur - Cascade Instance Segmentation"

# Visual settings
STAGE1_BOX_COLOR = (255, 255, 255)
STAGE1_ACTIVE_COLOR = (0, 255, 255)
PERSON_MASK_COLOR = (0, 120, 255)  # orange for person masks
INFO_BG_COLOR = (40, 40, 40)


def pack_rgb_features(img_bgr, target_size):
    """Resize and pack RGB into EI feature format."""
    resized = cv2.resize(img_bgr, (target_size, target_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)
    return ((r << 16) | (g << 8) | b).flatten().astype(np.float32).tolist()


def run_stage1(runner, img_bgr, input_size):
    """Run Stage 1 object detection. Returns only 'person' detections."""
    features = pack_rgb_features(img_bgr, input_size)
    t0 = time.time()
    result = runner.classify(features)
    dt = (time.time() - t0) * 1000
    all_bboxes = result["result"]["bounding_boxes"]
    person_bboxes = [bb for bb in all_bboxes if bb["label"] == "person"]
    return person_bboxes, dt


def run_stage2(runner, img_bgr, metadata, conf_thresh=0.25):
    """Run Stage 2 instance segmentation. Returns only 'person' instances."""
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
    all_instances = pp.process(output0, output1, orig_img_shape=img_bgr.shape[:2])
    person_instances = [inst for inst in all_instances if inst["class_id"] == PERSON_CLASS_ID]
    return person_instances, dt


def draw_stage1_panel(frame, bboxes, input_size, active_idx=0):
    """Draw Stage 1 panel with person bounding boxes."""
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
        label = f"person {bb['value']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


def draw_stage2_panel(crop, instances, alpha=0.5):
    """Draw Stage 2 panel with segmentation masks on cropped person."""
    vis = crop.copy()

    for inst in instances:
        mask = inst["mask"]
        score = inst["score"]

        if mask.shape[:2] != crop.shape[:2]:
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))

        mask_bool = mask > 127
        vis[mask_bool] = (
            np.array(PERSON_MASK_COLOR, dtype=np.uint8) * alpha +
            vis[mask_bool] * (1 - alpha)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, PERSON_MASK_COLOR, 2)

        bbox = inst["bbox"]
        bx1 = max(0, int(bbox[0] * crop.shape[1] / 640))
        by1 = max(0, int(bbox[1] * crop.shape[0] / 640))
        cv2.putText(vis, f"person {score:.2f}", (bx1, max(by1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PERSON_MASK_COLOR, 1)

    return vis


def draw_blurred_panel(frame, person_instances, blur_strength=51, passes=2):
    """Blur all person masks for privacy. More passes = stronger but slower."""
    vis = frame.copy()

    # Multi-pass Gaussian blur (kernel must be odd)
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    blurred_full = vis
    for _ in range(max(1, passes)):
        blurred_full = cv2.GaussianBlur(blurred_full, (k, k), 0)

    # Combine all person masks into one
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for inst in person_instances:
        mask = inst["mask"]
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        combined_mask = np.maximum(combined_mask, mask)

    # Smooth the mask edges slightly for natural blending
    combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 0)
    mask_float = (combined_mask / 255.0)[:, :, np.newaxis]

    # Blend: dark blurred where person, original everywhere else
    vis = (blurred_full * mask_float + vis * (1 - mask_float)).astype(np.uint8)

    return vis


def get_crop(frame, bbox, input_size, padding=0.1):
    """Crop a bounding box region from frame with padding."""
    h, w = frame.shape[:2]
    sx, sy = w / input_size, h / input_size

    x1 = int(bbox["x"] * sx)
    y1 = int(bbox["y"] * sy)
    x2 = int((bbox["x"] + bbox["width"]) * sx)
    y2 = int((bbox["y"] + bbox["height"]) * sy)

    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def letterbox(img, target_w, target_h):
    """Resize preserving aspect ratio, pad with black."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_panel_title(panel, title):
    """Draw a consistent title on a panel. Call after resize/letterbox."""
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return panel


def draw_info_bar(width, s1_time, s2_time, s3_time, fps, frame_num, n_persons):
    """Draw info bar."""
    bar_h = 36
    bar = np.full((bar_h, width, 3), INFO_BG_COLOR, dtype=np.uint8)

    texts = [
        f"FPS: {fps:.1f}",
        f"S1 Detection: {s1_time:.0f}ms",
        f"S2 Segmentation: {s2_time:.0f}ms",
        f"S3 Blur: {s3_time:.0f}ms",
        f"Persons: {n_persons}",
        f"Frame: {frame_num}",
    ]
    x = 10
    for t in texts:
        cv2.putText(bar, t, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        x += len(t) * 9 + 12

    return bar


def main():
    parser = argparse.ArgumentParser(description="Person blur using cascade instance segmentation")
    parser.add_argument("--stage1", type=str, required=True)
    parser.add_argument("--stage2", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="../model_metadata.json")
    parser.add_argument("--video", type=str, default=None, help="Input video (default: webcam)")
    parser.add_argument("--output", type=str, default=None, help="Output video file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--blur", type=int, default=51, help="Blur kernel size (odd number)")
    parser.add_argument("--blur-passes", type=int, default=2,
                        help="Gaussian blur passes. More = stronger anonymization but slower.")
    parser.add_argument("--skip", type=int, default=3,
                        help="Run Stage 2 every N frames for live mode. Ignored for video output.")
    args = parser.parse_args()

    # Load metadata
    metadata = {}
    if os.path.exists(args.metadata):
        with open(args.metadata) as f:
            metadata = json.load(f)

    # Initialize runners
    print("Loading Stage 1 (object detection)...")
    runner1 = ImpulseRunner(args.stage1)
    info1 = runner1.init(debug=False)
    s1_input = info1["model_parameters"]["image_input_width"]
    print(f"  Input: {s1_input}x{s1_input}")

    print("Loading Stage 2 (instance segmentation)...")
    runner2 = ImpulseRunner(args.stage2)
    runner2.init(debug=False)
    print(f"  Input: {metadata.get('image_size', 640)}x{metadata.get('image_size', 640)}")

    # Open video source
    is_video = args.video is not None
    if is_video:
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

    # Layout: top row = 2 panels, bottom row = full-width output, plus info bar
    # Top panels are half the width each
    top_pw = 480   # each top panel width
    top_ph = 360   # each top panel height
    canvas_w = top_pw * 2  # 960
    bot_ph = 360   # bottom panel height
    info_h = 36
    canvas_h = top_ph + bot_ph + info_h

    # Output video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = src_fps if is_video else 30
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (canvas_w, canvas_h))

    print(f"\nLayout: {canvas_w}x{canvas_h} (top: {top_pw}x{top_ph} x2, bottom: {canvas_w}x{bot_ph})")
    if is_video and args.output:
        print(f"Processing: {total_frames} frames -> {args.output} @ {out_fps:.0f} FPS")
    else:
        print("Controls: q/ESC=quit, s=screenshot, SPACE=pause")
    print("-" * 50)

    frame_num = 0
    paused = False
    fps_alpha = 0.9
    smoothed_fps = 0.0
    screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")

    # Cached Stage 2 results
    cached_instances = []
    cached_s2_time = 0.0

    # Create the live window explicitly. WINDOW_NORMAL avoids a Qt/XWayland
    # AUTOSIZE issue where the image area stays a tiny black box.
    show_window = not (is_video and args.output)
    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, canvas_w, canvas_h)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_video:
                        print("\nEnd of video")
                    break
                frame_num += 1

            t_start = time.time()

            # --- Stage 1: detect persons ---
            person_bboxes, s1_time = run_stage1(runner1, frame, s1_input)

            # Find largest person for crop view
            active_idx = 0
            if len(person_bboxes) > 1:
                areas = [bb["width"] * bb["height"] for bb in person_bboxes]
                active_idx = int(np.argmax(areas))

            # Top-left: Stage 1 panel
            top_left = draw_stage1_panel(frame, person_bboxes, s1_input, active_idx)
            top_left = cv2.resize(top_left, (top_pw, top_ph))
            draw_panel_title(top_left, "Stage 1: Object Detection")

            # --- Stage 2: segment persons ---
            if is_video and args.output:
                run_s2 = True
            else:
                run_s2 = (frame_num % args.skip == 1) or args.skip == 1
            s2_time = cached_s2_time

            if len(person_bboxes) > 0:
                if run_s2:
                    person_instances, s2_time = run_stage2(
                        runner2, frame, metadata, args.conf
                    )
                    cached_instances = person_instances
                    cached_s2_time = s2_time
                else:
                    person_instances = cached_instances

                # Top-right: cropped view of largest person with mask
                crop, crop_coords = get_crop(frame, person_bboxes[active_idx], s1_input)
                if crop.size > 0:
                    cx1, cy1, cx2, cy2 = crop_coords
                    crop_insts = []
                    for inst in person_instances:
                        cropped_mask = inst["mask"][cy1:cy2, cx1:cx2]
                        if cropped_mask.sum() > 0:
                            crop_insts.append({**inst, "mask": cropped_mask})
                    top_right = draw_stage2_panel(crop, crop_insts)
                    top_right = letterbox(top_right, top_pw, top_ph)
                else:
                    top_right = np.full((top_ph, top_pw, 3), 30, dtype=np.uint8)
            else:
                person_instances = cached_instances if not run_s2 else []
                if run_s2:
                    cached_instances = []
                top_right = np.full((top_ph, top_pw, 3), 30, dtype=np.uint8)

            draw_panel_title(top_right, "Stage 2: Instance Segmentation")

            # Bottom: blurred output (full width)
            t_blur_start = time.time()
            bottom = draw_blurred_panel(frame, person_instances, args.blur, args.blur_passes)
            s3_time = (time.time() - t_blur_start) * 1000
            bottom = letterbox(bottom, canvas_w, bot_ph)
            draw_panel_title(bottom, "Output: Gaussian Blur")

            # Compose canvas
            top_row = np.hstack([top_left, top_right])

            t_end = time.time()
            frame_fps = 1.0 / max(t_end - t_start, 0.001)
            smoothed_fps = fps_alpha * smoothed_fps + (1 - fps_alpha) * frame_fps if smoothed_fps > 0 else frame_fps

            info_bar = draw_info_bar(
                canvas_w, s1_time, s2_time, s3_time, smoothed_fps,
                frame_num, len(person_instances)
            )
            canvas = np.vstack([top_row, bottom, info_bar])

            # Output
            if writer:
                writer.write(canvas)

            if is_video and args.output:
                if frame_num % 10 == 0 or frame_num == total_frames:
                    pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                    print(f"\r  [{frame_num}/{total_frames}] {pct:.0f}% "
                          f"S1:{s1_time:.0f}ms S2:{s2_time:.0f}ms "
                          f"persons:{len(person_instances)}", end="", flush=True)
            else:
                cv2.imshow(WINDOW_NAME, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                # Window closed via the X button -> quit (otherwise imshow reopens it)
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord("s"):
                    os.makedirs(screenshot_dir, exist_ok=True)
                    fname = os.path.join(screenshot_dir, f"blur_frame_{frame_num:05d}.jpg")
                    cv2.imwrite(fname, canvas)
                    print(f"Screenshot saved: {fname}")
                elif key == ord(" "):
                    paused = not paused
                    print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nInterrupted")

    cap.release()
    if writer:
        writer.release()
        print(f"\nVideo saved to {args.output}")
    cv2.destroyAllWindows()
    runner1.stop()
    runner2.stop()
    print(f"Done. Processed {frame_num} frames.")


if __name__ == "__main__":
    main()
