"""
Postprocessing for YOLO11-seg freeform BYOM output.

YOLO11-seg ONNX produces two output tensors:
  - output0: (1, num_dets, 4 + num_classes + 32)
      First 4 values: x_center, y_center, width, height (in pixel coords)
      Next num_classes values: class confidence scores
      Last 32 values: mask coefficients
  - output1: (1, 32, mask_h, mask_w)
      Prototype masks (typically 160x160 for 640 input)

This script performs:
  1. Parse detection tensor → extract boxes, scores, class IDs, mask coefficients
  2. Apply confidence threshold + NMS
  3. Generate instance masks: coefficients @ prototypes → sigmoid → crop to bbox → resize
  4. Return structured results

Usage standalone:
    python postprocess.py --onnx model.onnx --image test.jpg --metadata model_metadata.json

Usage as library:
    from postprocess import YOLOSegPostprocessor
    pp = YOLOSegPostprocessor(num_classes=80, conf_thresh=0.25, iou_thresh=0.7)
    results = pp.process(output0, output1, orig_img_shape=(480, 640))
"""

import argparse
import json
import numpy as np
import cv2


class YOLOSegPostprocessor:
    def __init__(self, num_classes=80, conf_thresh=0.25, iou_thresh=0.7, img_size=640):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size

    def process(self, output0, output1, orig_img_shape=None):
        """
        Process YOLO11-seg raw outputs into instance segmentation results.

        Args:
            output0: np.ndarray shape (1, num_dets, 4+num_classes+32) or transposed
            output1: np.ndarray shape (1, 32, mask_h, mask_w)
            orig_img_shape: (height, width) of original image for mask resizing

        Returns:
            list of dicts with keys: bbox, score, class_id, mask
        """
        # Squeeze batch dim
        if output0.ndim == 3:
            output0 = output0[0]  # (num_dets, 4+nc+32)
        if output1.ndim == 4:
            protos = output1[0]  # (32, mask_h, mask_w)
        else:
            protos = output1

        # output0 might be (4+nc+32, num_dets) — transpose if needed
        # YOLO11 outputs (1, 4+nc+32, num_dets), so cols > rows means transposed
        if output0.shape[0] == (4 + self.num_classes + 32):
            output0 = output0.T  # → (num_dets, 4+nc+32)

        num_dets = output0.shape[0]

        # Split columns
        boxes_xywh = output0[:, :4]                            # (N, 4) x_center, y_center, w, h
        class_scores = output0[:, 4:4+self.num_classes]        # (N, nc)
        mask_coeffs = output0[:, 4+self.num_classes:]          # (N, 32)

        # Get best class per detection
        class_ids = np.argmax(class_scores, axis=1)            # (N,)
        scores = class_scores[np.arange(num_dets), class_ids]  # (N,)

        # Confidence filter
        mask = scores > self.conf_thresh
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]

        if len(scores) == 0:
            return []

        # Convert xywh to xyxy
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        # NMS
        keep = self._nms(boxes_xyxy, scores, self.iou_thresh)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        # Generate masks: coefficients @ prototypes
        masks = self._process_masks(protos, mask_coeffs, boxes_xyxy, orig_img_shape)

        # Build results
        results = []
        for i in range(len(scores)):
            results.append({
                "bbox": boxes_xyxy[i].tolist(),  # [x1, y1, x2, y2] in model input coords
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
                "mask": masks[i],  # binary mask at original image resolution
            })

        return results

    def _xywh_to_xyxy(self, boxes):
        """Convert (x_center, y_center, w, h) to (x1, y1, x2, y2)."""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy

    def _nms(self, boxes, scores, iou_thresh):
        """Non-maximum suppression using OpenCV."""
        # cv2.dnn.NMSBoxes expects (x, y, w, h)
        bboxes = []
        for b in boxes:
            bboxes.append([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])
        indices = cv2.dnn.NMSBoxes(bboxes, scores.tolist(), self.conf_thresh, iou_thresh)
        if len(indices) == 0:
            return np.array([], dtype=int)
        return indices.flatten()

    def _process_masks(self, protos, mask_coeffs, boxes_xyxy, orig_img_shape):
        """
        Generate instance masks from prototype masks and coefficients.

        protos: (32, mask_h, mask_w)
        mask_coeffs: (N, 32)
        boxes_xyxy: (N, 4) in model input pixel coords
        orig_img_shape: (H, W) for final resize, or None to keep at model input size

        Returns: list of binary masks (H, W) as uint8 (0 or 255)
        """
        mask_h, mask_w = protos.shape[1], protos.shape[2]
        # (N, 32) @ (32, mask_h*mask_w) → (N, mask_h*mask_w)
        masks_raw = mask_coeffs @ protos.reshape(32, -1)
        masks_raw = masks_raw.reshape(-1, mask_h, mask_w)

        # Sigmoid
        masks_raw = 1.0 / (1.0 + np.exp(-masks_raw))

        # Scale factor from model input to mask resolution
        scale_x = mask_w / self.img_size
        scale_y = mask_h / self.img_size

        result_masks = []
        for i in range(len(masks_raw)):
            mask = masks_raw[i]  # (mask_h, mask_w)

            # Crop mask to bounding box region (in mask coordinates)
            x1 = max(0, int(boxes_xyxy[i, 0] * scale_x))
            y1 = max(0, int(boxes_xyxy[i, 1] * scale_y))
            x2 = min(mask_w, int(boxes_xyxy[i, 2] * scale_x))
            y2 = min(mask_h, int(boxes_xyxy[i, 3] * scale_y))

            # Zero out everything outside the bbox
            cropped = np.zeros_like(mask)
            cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

            # Resize to model input size first
            cropped = cv2.resize(cropped, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)

            # Resize to original image size if provided
            if orig_img_shape is not None:
                cropped = cv2.resize(cropped, (orig_img_shape[1], orig_img_shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

            # Threshold to binary
            binary = (cropped > 0.5).astype(np.uint8) * 255
            result_masks.append(binary)

        return result_masks


def visualize(image, results, class_names=None, alpha=0.5):
    """Draw instance segmentation results on an image."""
    overlay = image.copy()
    # Generate distinct colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    for det in results:
        color = colors[det["class_id"] % 100].tolist()
        mask = det["mask"]
        bbox = det["bbox"]
        score = det["score"]
        cls = det["class_id"]

        # Apply mask overlay
        mask_bool = mask > 127
        overlay[mask_bool] = (
            np.array(color, dtype=np.uint8) * alpha +
            overlay[mask_bool] * (1 - alpha)
        ).astype(np.uint8)

        # Draw bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Scale bbox from model input to image coords
        img_h, img_w = image.shape[:2]
        # bbox is already in model-input coords, need to scale if image differs
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{class_names.get(cls, cls) if class_names else cls}: {score:.2f}"
        cv2.putText(overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg postprocessing demo")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--metadata", type=str, default=None, help="Path to model_metadata.json")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--output", type=str, default="result.jpg", help="Output image path")
    args = parser.parse_args()

    import onnxruntime as ort

    # Load metadata
    num_classes = 80
    img_size = 640
    class_names = {}
    if args.metadata:
        with open(args.metadata) as f:
            meta = json.load(f)
        num_classes = meta.get("num_classes", 80)
        img_size = meta.get("image_size", 640)
        class_names = {int(k): v for k, v in meta.get("class_names", {}).items()}

    # Load image
    img = cv2.imread(args.image)
    orig_shape = img.shape[:2]  # (H, W)

    # Preprocess: resize, normalize, HWC→CHW, add batch dim
    resized = cv2.resize(img, (img_size, img_size))
    blob = resized.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    blob = np.expand_dims(blob, 0)  # (1, 3, H, W)

    # Run inference
    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    output0 = outputs[0]  # detections
    output1 = outputs[1]  # prototypes

    print(f"output0 shape: {output0.shape}")
    print(f"output1 shape: {output1.shape}")

    # Postprocess
    pp = YOLOSegPostprocessor(
        num_classes=num_classes,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=img_size,
    )
    results = pp.process(output0, output1, orig_img_shape=orig_shape)

    print(f"Detected {len(results)} instances:")
    for r in results:
        name = class_names.get(r["class_id"], r["class_id"])
        print(f"  {name}: {r['score']:.3f} bbox={[int(v) for v in r['bbox']]}")

    # Visualize
    vis = visualize(img, results, class_names=class_names)
    cv2.imwrite(args.output, vis)
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
