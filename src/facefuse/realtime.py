"""Live webcam recognition using the quality-adaptive hybrid recogniser."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from . import config
from .enhancement import DIEMPlus
from .face_detection import FaceDetector, pad_image
from .hybrid_recognition import (
    HandcraftedFeatureExtractor,
    HybridFaceDatabase,
    HybridFaceRecognizer,
)
from .recognition import FaceRecognizer

MATCH_COLOR = (0, 200, 0)
UNKNOWN_COLOR = (0, 0, 220)


def build_parser():
    p = argparse.ArgumentParser(
        prog="facefuse webcam",
        description="Run FaceFuse live on a webcam or a video file.",
    )
    p.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH)
    p.add_argument("--gallery", type=Path, default=config.GALLERY_DIR)
    p.add_argument("--source", default="0",
                   help="camera index (default 0) or a path to a video file")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    p.add_argument("--quality-scale", type=float, default=config.QUALITY_SCALE)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    print("Initialising FaceFuse...")
    detector = FaceDetector(args.weights)
    diem = DIEMPlus()
    recognizer = FaceRecognizer(device=args.device)
    handcrafted = HandcraftedFeatureExtractor()

    gallery = HybridFaceDatabase(recognizer, detector, diem, handcrafted)
    gallery.build(str(args.gallery))
    if not gallery.database:
        raise SystemExit(
            f"No identities enrolled from {args.gallery}. "
            "Add images as data/gallery/<person name>/*.jpg first."
        )

    hybrid = HybridFaceRecognizer(gallery, threshold=args.threshold,
                                  quality_scale=args.quality_scale)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    print("Running. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for face in detector.detect(frame, extract_crops=True):
                x1, y1, x2, y2 = face.bbox
                if face.crop.size == 0:
                    continue

                enhanced = diem.process(face.crop)
                emb = recognizer.get_embedding(pad_image(enhanced))
                if emb is None:
                    name, score = "Unknown", 0.0
                else:
                    hand_feat = handcrafted.extract(enhanced)
                    name, score = hybrid.identify(emb, hand_feat, enhanced)

                color = UNKNOWN_COLOR if name == "Unknown" else MATCH_COLOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {score:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("FaceFuse", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
