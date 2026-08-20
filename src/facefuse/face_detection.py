"""YOLO face detection with margin-aware cropping and letterbox padding."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


class FaceDetection:
    def __init__(self, bbox, confidence, crop=None):
        self.bbox = bbox
        self.confidence = confidence
        self.crop = crop


class FaceDetector:
    def __init__(self, weights_path=None, conf_threshold=config.DETECTION_CONFIDENCE):
        # Imported here rather than at module scope so that pad_image and the
        # rest of the package stay usable without the torch/ultralytics stack.
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path or config.WEIGHTS_PATH))
        self.conf = conf_threshold

    def detect(self, image, extract_crops=True, margin=config.DETECTION_MARGIN):
        results = self.model(image, conf=self.conf, verbose=False)
        detections = []
        h, w = image.shape[:2]
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Add margin
                bw = x2 - x1
                bh = y2 - y1
                x1 = max(0, int(x1 - bw * margin))
                y1 = max(0, int(y1 - bh * margin))
                x2 = min(w, int(x2 + bw * margin))
                y2 = min(h, int(y2 + bh * margin))
                crop = image[y1:y2, x1:x2].copy() if extract_crops else None
                detections.append(
                    FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        crop=crop
                    )
                )
        return detections


def pad_image(image, target_size=config.EMBEDDING_INPUT_SIZE):
    """Resize the long edge to ``target_size`` and centre-pad to a square."""

    h, w = image.shape[:2]
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    else:
        new_h, new_w = h, w

    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image
    return padded