"""
Two-stage cascade inference: Object Detection → Instance Segmentation.

Stage 1: YoloX-Nano (.eim, object detection) — fast detection, outputs bounding boxes
Stage 2: YOLO11n-seg (.eim, freeform BYOM) — instance segmentation, outputs pixel masks

The cascade runs Stage 1 first. If objects of interest are detected,
Stage 2 runs on the full image to produce instance segmentation masks.
Results are merged: Stage 1 boxes select which Stage 2 masks to keep.

Usage:
    python cascade_inference.py \
      --stage1 ../models/stage1-yolox-aarch64.eim \
      --stage2 ../models/stage2-yolo11nseg-aarch64.eim \
      --metadata ../model_metadata.json \
      --image test.jpg \
      --output result.jpg
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from postprocess import YOLOSegPostprocessor, visualize
from edge_impulse_linux.runner import ImpulseRunner


def pack_rgb_features(img_bgr, target_size):
    """Resize image and pack RGB into EI feature format (one float32 per pixel)."""
    resized = cv2.resize(img_bgr, (target_size, target_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)
    packed = ((r << 16) | (g << 8) | b).flatten().astype(np.float32).tolist()
    return packed


def run_stage1(runner, img_bgr, input_size):
    """Run Stage 1 object detection. Returns list of bounding boxes."""
    features = pack_rgb_features(img_bgr, input_size)
    result = runner.classify(features)
    bboxes = result["result"]["bounding_boxes"]
    timing = result.get("timing", {})
    return bboxes, timing


def run_stage2(runner, img_bgr, metadata, conf_thresh=0.25):
    """Run Stage 2 instance segmentation. Returns list of instance results."""
    input_size = metadata.get("image_size", 640)
    num_classes = metadata.get("num_classes", 80)

    features = pack_rgb_features(img_bgr, input_size)
    result = runner.classify(features)
    timing = result.get("timing", {})

    freeform = result.get("result", {}).get("freeform", None)
    if freeform is None:
        print("ERROR: No freeform output from Stage 2")
        return [], timing

    # Match tensors by size (EI may return them in arbitrary order)
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
    # EI freeform returns prototypes in NHWC — transpose to NCHW
    output1 = output1_flat.reshape(1, mask_h, mask_w, 32).transpose(0, 3, 1, 2)

    pp = YOLOSegPostprocessor(
        num_classes=num_classes,
        conf_thresh=conf_thresh,
        iou_thresh=0.7,
        img_size=input_size,
    )
    orig_shape = img_bgr.shape[:2]
    instances = pp.process(output0, output1, orig_img_shape=orig_shape)
    return instances, timing


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def merge_results(stage1_bboxes, stage2_instances, stage1_input_size, img_shape, class_names, iou_thresh=0.3):
    """
    Merge Stage 1 detections with Stage 2 instance masks.

    Stage 1 bboxes are in stage1 input coords (e.g. 416x416).
    Stage 2 bboxes are in stage2 input coords (e.g. 640x640).
    Both need to be scaled to original image coords for matching.
    """
    img_h, img_w = img_shape[:2]

    # Scale Stage 1 boxes to original image coords
    s1_scale_x = img_w / stage1_input_size
    s1_scale_y = img_h / stage1_input_size
    s1_boxes_orig = []
    for bb in stage1_bboxes:
        x1 = bb["x"] * s1_scale_x
        y1 = bb["y"] * s1_scale_y
        x2 = (bb["x"] + bb["width"]) * s1_scale_x
        y2 = (bb["y"] + bb["height"]) * s1_scale_y
        s1_boxes_orig.append({
            "box": [x1, y1, x2, y2],
            "label": bb["label"],
            "score": bb["value"],
        })

    # Stage 2 bboxes are already in model input coords — scale to original
    # (postprocessor doesn't auto-scale boxes, only masks)
    s2_input_size = 640  # YOLO11-seg input
    s2_scale_x = img_w / s2_input_size
    s2_scale_y = img_h / s2_input_size

    merged = []
    for inst in stage2_instances:
        s2_box_orig = [
            inst["bbox"][0] * s2_scale_x,
            inst["bbox"][1] * s2_scale_y,
            inst["bbox"][2] * s2_scale_x,
            inst["bbox"][3] * s2_scale_y,
        ]
        cls_name = class_names.get(inst["class_id"], str(inst["class_id"]))

        # Find matching Stage 1 detection
        best_iou = 0
        best_s1 = None
        for s1 in s1_boxes_orig:
            if s1["label"] == cls_name:
                iou = compute_iou(s1["box"], s2_box_orig)
                if iou > best_iou:
                    best_iou = iou
                    best_s1 = s1

        merged.append({
            "bbox": s2_box_orig,
            "bbox_model": inst["bbox"],
            "score": inst["score"],
            "class_id": inst["class_id"],
            "class_name": cls_name,
            "mask": inst["mask"],
            "stage1_match": best_s1 is not None and best_iou > iou_thresh,
            "stage1_iou": best_iou,
            "stage1_score": best_s1["score"] if best_s1 else None,
        })

    return merged, s1_boxes_orig


def main():
    parser = argparse.ArgumentParser(description="Two-stage cascade: detection → segmentation")
    parser.add_argument("--stage1", type=str, required=True, help="Stage 1 .eim (object detection)")
    parser.add_argument("--stage2", type=str, required=True, help="Stage 2 .eim (instance segmentation)")
    parser.add_argument("--metadata", type=str, default="../model_metadata.json")
    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Stage 2 confidence threshold")
    parser.add_argument("--output", type=str, default="cascade_result.jpg", help="Output image")
    args = parser.parse_args()

    # Load metadata
    class_names = {}
    metadata = {}
    if os.path.exists(args.metadata):
        with open(args.metadata) as f:
            metadata = json.load(f)
        class_names = {int(k): v for k, v in metadata.get("class_names", {}).items()}

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Could not load image: {args.image}")
        sys.exit(1)
    print(f"Image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    # --- Stage 1: Object Detection ---
    print("\n=== Stage 1: Object Detection (YoloX-Nano) ===")
    runner1 = ImpulseRunner(args.stage1)
    info1 = runner1.init(debug=False)
    s1_input = info1["model_parameters"]["image_input_width"]
    print(f"Input: {s1_input}x{s1_input}")

    t0 = time.time()
    bboxes, timing1 = run_stage1(runner1, img, s1_input)
    t1 = time.time()
    runner1.stop()

    print(f"Detected {len(bboxes)} objects ({(t1-t0)*1000:.0f}ms):")
    for bb in bboxes:
        print(f"  {bb['label']}: {bb['value']:.3f} @ ({bb['x']},{bb['y']},{bb['width']},{bb['height']})")

    if len(bboxes) == 0:
        print("No objects detected. Skipping Stage 2.")
        sys.exit(0)

    # --- Stage 2: Instance Segmentation ---
    print(f"\n=== Stage 2: Instance Segmentation (YOLO11n-seg) ===")
    runner2 = ImpulseRunner(args.stage2)
    runner2.init(debug=False)

    t2 = time.time()
    instances, timing2 = run_stage2(runner2, img, metadata, conf_thresh=args.conf)
    t3 = time.time()
    runner2.stop()

    print(f"Segmented {len(instances)} instances ({(t3-t2)*1000:.0f}ms):")
    for inst in instances:
        name = class_names.get(inst["class_id"], inst["class_id"])
        print(f"  {name}: {inst['score']:.3f}")

    # --- Merge ---
    print(f"\n=== Cascade Merge ===")
    merged, s1_boxes = merge_results(
        bboxes, instances, s1_input, img.shape, class_names
    )

    for m in merged:
        match_str = f"✓ IoU={m['stage1_iou']:.2f}" if m["stage1_match"] else "✗ no match"
        print(f"  {m['class_name']}: seg={m['score']:.3f} | Stage1 {match_str}")

    # --- Visualize ---
    vis = visualize(img, instances, class_names=class_names)

    # Draw Stage 1 boxes in a different color (white, dashed-like)
    for s1 in s1_boxes:
        x1, y1, x2, y2 = [int(v) for v in s1["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(vis, f"S1:{s1['label']} {s1['score']:.2f}", (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(f"\nTotal cascade time: {(t3-t0)*1000:.0f}ms")
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
