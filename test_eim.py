"""
Test YOLO11-seg .eim deployment with Edge Impulse Linux Python SDK.

Usage:
    python test_eim.py --eim model.eim --image test.jpg [--metadata model_metadata.json]

This script:
  1. Loads the .eim model via ImpulseRunner
  2. Preprocesses an input image
  3. Runs inference (freeform output)
  4. Parses the raw output tensors with our postprocessor
  5. Visualizes instance segmentation masks
"""

import argparse
import json
import sys
import os
import numpy as np
import cv2
from edge_impulse_linux.runner import ImpulseRunner
from postprocess import YOLOSegPostprocessor, visualize


def main():
    parser = argparse.ArgumentParser(description="Test YOLO11-seg .eim inference")
    parser.add_argument("--eim", type=str, required=True, help="Path to .eim file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--metadata", type=str, default="model_metadata.json",
                        help="Path to model_metadata.json")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="out/eim_result.jpg", help="Output image path")
    args = parser.parse_args()

    # Load metadata
    num_classes = 80
    img_size = 640
    class_names = {}
    if os.path.exists(args.metadata):
        with open(args.metadata) as f:
            meta = json.load(f)
        num_classes = meta.get("num_classes", 80)
        img_size = meta.get("image_size", 640)
        class_names = {int(k): v for k, v in meta.get("class_names", {}).items()}
        print(f"Metadata loaded: {num_classes} classes, {img_size}x{img_size} input")

    # Make .eim executable
    os.chmod(args.eim, 0o755)

    # Initialize runner
    print(f"Loading .eim model: {args.eim}")
    runner = ImpulseRunner(args.eim)
    model_info = runner.init(debug=False)

    print(f"Model info: {json.dumps(model_info, indent=2, default=str)[:500]}")

    # Get model parameters from hello response
    model_params = model_info.get("model_parameters", {})
    input_width = model_params.get("image_input_width", img_size)
    input_height = model_params.get("image_input_height", img_size)
    input_features = model_params.get("input_features_count", input_width * input_height * 3)
    print(f"Input: {input_width}x{input_height}, features: {input_features}")

    # Load and preprocess image
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Could not load image: {args.image}")
        sys.exit(1)
    orig_shape = img.shape[:2]
    print(f"Original image: {orig_shape[1]}x{orig_shape[0]}")

    # Resize to model input
    resized = cv2.resize(img, (input_width, input_height))

    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # EI image features: pack RGB into single float32 per pixel
    # Each pixel becomes (R << 16) | (G << 8) | B, cast to float32
    # This gives 320*320 = 102400 features for a 320x320 image
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)
    packed = (r << 16) | (g << 8) | b
    features = packed.flatten().astype(np.float32).tolist()

    print(f"Running inference ({len(features)} features)...")
    result = runner.classify(features)

    print(f"Inference result keys: {list(result.keys())}")
    if 'result' in result:
        print(f"Result keys: {list(result['result'].keys())}")

    # Extract freeform output tensors
    freeform = result.get('result', {}).get('freeform', None)
    if freeform is None:
        print("ERROR: No freeform output in result. Full result:")
        print(json.dumps(result, indent=2, default=str)[:1000])
        runner.stop()
        sys.exit(1)

    print(f"Freeform outputs: {len(freeform)} tensors")
    for i, tensor in enumerate(freeform):
        arr = np.array(tensor)
        print(f"  Tensor {i}: {arr.shape} (total elements: {arr.size})")

    # Reshape tensors to expected YOLO11-seg output shapes
    # output0 (detections): (1, 4+nc+32, 8400) = 974400 elements for 640 input
    # output1 (prototypes): (1, 32, mask_h, mask_w) = 819200 elements for 160x160
    # Note: EI freeform may return tensors in arbitrary order — match by size
    num_det_channels = 4 + num_classes + 32  # 116 for COCO
    expected_det_size = num_det_channels * 8400  # 974400

    flat_tensors = [np.array(t, dtype=np.float32) for t in freeform]
    if flat_tensors[0].size == expected_det_size:
        output0_flat, output1_flat = flat_tensors[0], flat_tensors[1]
    else:
        output0_flat, output1_flat = flat_tensors[1], flat_tensors[0]
        print("  (tensors were swapped, re-ordered by size)")

    num_anchors = output0_flat.size // num_det_channels
    mask_h = mask_w = int(np.sqrt(output1_flat.size // 32))

    print(f"Reshaping output0: ({num_det_channels}, {num_anchors})")
    print(f"Reshaping output1: ({mask_h}, {mask_w}, 32) NHWC → (32, {mask_h}, {mask_w}) NCHW")

    output0 = output0_flat.reshape(1, num_det_channels, num_anchors)
    # EIM freeform returns prototype masks in NHWC layout — transpose to NCHW
    output1 = output1_flat.reshape(1, mask_h, mask_w, 32).transpose(0, 3, 1, 2)

    # Run postprocessor
    pp = YOLOSegPostprocessor(
        num_classes=num_classes,
        conf_thresh=args.conf,
        iou_thresh=0.7,
        img_size=input_width,
    )
    results = pp.process(output0, output1, orig_img_shape=orig_shape)

    print(f"\nDetected {len(results)} instances:")
    for r in results:
        name = class_names.get(r["class_id"], r["class_id"])
        print(f"  {name}: {r['score']:.3f} bbox={[int(v) for v in r['bbox']]}")

    # Visualize
    vis = visualize(img, results, class_names=class_names)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(f"\nResult saved to {args.output}")

    # Cleanup
    runner.stop()
    print("Done.")


if __name__ == "__main__":
    main()
