#!/usr/bin/env python3
"""Regenerate the README demo figures.

Uses ``skimage.data.astronaut()`` — a NASA photograph of Eileen Collins, public
domain — so the published assets carry no copyright or privacy encumbrance.
Nothing here depends on the private gallery.

    python scripts/make_demo_assets.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from facefuse.enhancement import DIEMPlus  # noqa: E402
from facefuse.face_detection import FaceDetector, pad_image  # noqa: E402
from facefuse.hybrid_recognition import (  # noqa: E402
    HandcraftedFeatureExtractor,
    get_image_quality,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

DEEP_COLOR = "#4c72b0"
HAND_COLOR = "#c44e52"
BOX_COLOR = (0, 200, 0)


def sample_image():
    """Public-domain NASA portrait shipped with scikit-image, as BGR."""
    from skimage.data import astronaut

    return cv2.cvtColor(astronaut(), cv2.COLOR_RGB2BGR)


def detection_figure(image, detector, out):
    faces = detector.detect(image, extract_crops=True)
    if not faces:
        raise SystemExit("No face detected in the sample image.")
    face = max(faces, key=lambda f: f.confidence)

    annotated = image.copy()
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 3)
    label = f"face {face.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), BOX_COLOR, -1)
    cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    enhanced = DIEMPlus().process(face.crop)
    padded = pad_image(enhanced)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    panels = [
        (annotated, f"1. detect\nYOLO11-L, conf {face.confidence:.2f}"),
        (face.crop, f"2. crop + 20% margin\n{face.crop.shape[1]}x{face.crop.shape[0]}"),
        (enhanced, "3. DIEM+ bilateral\nedge-preserving denoise"),
        (padded, "4. letterbox 320x320\nready for the embedder"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("FaceFuse preprocessing pipeline", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return face.crop


def quality_response_figure(crop, out, quality_scale=500.0):
    """Blur the same crop progressively and show the fusion weight respond."""
    handcrafted = HandcraftedFeatureExtractor()
    kernels = [1, 5, 11, 17, 23, 31]
    rows = []
    for k in kernels:
        blurred = crop if k == 1 else cv2.GaussianBlur(crop, (k, k), 0)
        variance = get_image_quality(blurred)
        w_deep = float(np.clip(variance / quality_scale, 0.0, 1.0))
        rows.append((k, blurred, variance, w_deep))
        handcrafted.extract(blurred)  # exercise the real extractor path

    fig = plt.figure(figsize=(14, 5.6))
    grid = fig.add_gridspec(2, len(rows), height_ratios=[1.15, 1.0], hspace=0.35)

    for i, (k, blurred, variance, _) in enumerate(rows):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        ax.set_title(f"blur k={k}\nvar={variance:.0f}", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(rows))
    deep = [r[3] for r in rows]
    hand = [1.0 - d for d in deep]
    ax.bar(x - 0.2, deep, width=0.4, label="deep embedding", color=DEEP_COLOR)
    ax.bar(x + 0.2, hand, width=0.4, label="LBP + HOG", color=HAND_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={r[0]}" for r in rows])
    ax.set_ylabel("fusion weight")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.set_title(f"As the crop blurs, weight shifts to the handcrafted branch "
                 f"(quality_scale = {quality_scale:.0f})", fontsize=11)

    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args(argv)

    ASSETS.mkdir(parents=True, exist_ok=True)
    image = sample_image()
    detector = FaceDetector(args.weights)

    crop = detection_figure(image, detector, ASSETS / "pipeline_stages.png")
    quality_response_figure(crop, ASSETS / "quality_response.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
