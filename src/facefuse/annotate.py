"""Recognise faces in still images and write an annotated copy.

This is the quickest way to see the whole pipeline produce a visible result:

    facefuse annotate photo.jpg --out annotated.jpg
"""

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
    get_image_quality,
)
from .recognition import FaceRecognizer

MATCH_COLOR = (0, 200, 0)
UNKNOWN_COLOR = (0, 0, 220)


def draw_result(frame, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(th + 6, y1)
    cv2.rectangle(frame, (x1, top - th - 6), (x1 + tw + 6, top), color, -1)
    cv2.putText(frame, label, (x1 + 3, top - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def build_parser():
    p = argparse.ArgumentParser(
        prog="facefuse annotate",
        description="Recognise every face in an image and save an annotated copy.",
    )
    p.add_argument("image", type=Path, help="input image")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: <output dir>/<name>_annotated.jpg)")
    p.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH)
    p.add_argument("--gallery", type=Path, default=config.GALLERY_DIR)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD)
    p.add_argument("--quality-scale", type=float, default=config.QUALITY_SCALE)
    p.add_argument("--detect-only", action="store_true",
                   help="draw detection boxes without enrolling a gallery")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = FaceDetector(args.weights)
    diem = DIEMPlus()
    faces = detector.detect(frame, extract_crops=True)
    print(f"Detected {len(faces)} face(s) in {args.image.name}")

    if args.detect_only:
        for face in faces:
            draw_result(frame, face.bbox, f"face {face.confidence:.2f}", MATCH_COLOR)
    else:
        recognizer = FaceRecognizer(device=args.device)
        handcrafted = HandcraftedFeatureExtractor()
        gallery = HybridFaceDatabase(recognizer, detector, diem, handcrafted)
        gallery.build(str(args.gallery))
        if not gallery.database:
            raise SystemExit(
                f"No identities enrolled from {args.gallery}. "
                "Add images as data/gallery/<person name>/*.jpg, "
                "or pass --detect-only."
            )
        hybrid = HybridFaceRecognizer(gallery, threshold=args.threshold,
                                      quality_scale=args.quality_scale)

        for face in faces:
            if face.crop.size == 0:
                continue
            enhanced = diem.process(face.crop)
            emb = recognizer.get_embedding(pad_image(enhanced))
            if emb is None:
                name, score = "Unknown", 0.0
            else:
                hand_feat = handcrafted.extract(enhanced)
                name, score = hybrid.identify(emb, hand_feat, enhanced)

            quality = get_image_quality(enhanced)
            w_deep = min(1.0, max(0.0, quality / hybrid.quality_scale))
            print(f"  {name:<20} score={score:.3f}  "
                  f"lap_var={quality:7.1f}  w_deep={w_deep:.2f}")
            color = UNKNOWN_COLOR if name == "Unknown" else MATCH_COLOR
            draw_result(frame, face.bbox, f"{name} {score:.2f}", color)

    out = args.out
    if out is None:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = config.OUTPUT_DIR / f"{args.image.stem}_annotated.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
