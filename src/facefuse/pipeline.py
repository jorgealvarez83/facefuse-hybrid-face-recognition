"""Full research pipeline: enrol, optimise, benchmark, plot.

Builds both the deep-only gallery and the hybrid gallery, tunes five
hyper-parameters with a genetic algorithm, then benchmarks the two recognisers
head to head on the probe set and writes the figures.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2

from . import config
from .benchmarking import Benchmarking
from .database import FaceDatabase
from .enhancement import DIEMPlus
from .face_detection import FaceDetector, pad_image
from .genetic_optimizer import GeneticOptimizer
from .hybrid_recognition import (
    HandcraftedFeatureExtractor,
    HybridFaceDatabase,
    HybridFaceRecognizer,
)
from .recognition import FaceRecognizer, identify
from .visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_quality_weighting,
)


def load_probe_set(probe_dir):
    """Load every image under ``probe_dir/<identity>/`` with its folder label."""
    images, labels = [], []
    for person in sorted(os.listdir(probe_dir)):
        person_path = os.path.join(probe_dir, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in sorted(os.listdir(person_path)):
            img = cv2.imread(os.path.join(person_path, img_name))
            if img is not None:
                images.append(img)
                labels.append(person)
    print(f"Loaded {len(images)} probe images from {len(set(labels))} identities")
    return images, labels


def predict_deep(image, detector, diem, recognizer, gallery, threshold):
    """Deep-only baseline: detect, enhance, embed, nearest gallery vector."""
    faces = detector.detect(image, extract_crops=True)
    if not faces:
        return "Unknown"
    crop = max(faces, key=lambda f: f.confidence).crop
    if crop.size == 0:
        return "Unknown"
    padded = pad_image(diem.process(crop))
    emb = recognizer.get_embedding(padded)
    if emb is None:
        return "Unknown"
    return identify(emb, gallery.database, threshold=threshold)[0]


def predict_hybrid(image, detector, diem, recognizer, handcrafted, hybrid_recognizer):
    """Hybrid path: fuse the deep score and the LBP+HOG score by image quality."""
    faces = detector.detect(image, extract_crops=True)
    if not faces:
        return "Unknown"
    crop = max(faces, key=lambda f: f.confidence).crop
    if crop.size == 0:
        return "Unknown"
    enhanced = diem.process(crop)
    emb = recognizer.get_embedding(pad_image(enhanced))
    if emb is None:
        return "Unknown"
    hand_feat = handcrafted.extract(enhanced)
    return hybrid_recognizer.identify(emb, hand_feat, enhanced)[0]


def build_parser():
    p = argparse.ArgumentParser(
        prog="facefuse benchmark",
        description=(
            "Enrol a gallery, tune the hybrid recogniser and benchmark it "
            "against the deep-only baseline."
        ),
    )
    p.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH,
                   help="YOLO face-detector weights (.pt)")
    p.add_argument("--gallery", type=Path, default=config.GALLERY_DIR,
                   help="enrolment images, one folder per identity")
    p.add_argument("--probe", type=Path, default=config.PROBE_DIR,
                   help="held-out evaluation images, same layout")
    p.add_argument("--output", type=Path, default=config.OUTPUT_DIR,
                   help="directory for figures and metrics.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                   help="execution provider for the embedding model")
    p.add_argument("--optimize", dest="optimize", action="store_true", default=True,
                   help="run the genetic hyper-parameter search (default)")
    p.add_argument("--no-optimize", dest="optimize", action="store_false",
                   help="skip the search and use the defaults from config.py")
    p.add_argument("--generations", type=int, default=config.GA_GENERATIONS)
    p.add_argument("--population", type=int, default=config.GA_POPULATION_SIZE)
    p.add_argument("--seed", type=int, default=0, help="seed for the genetic algorithm")
    p.add_argument("--no-show", action="store_true",
                   help="save figures without opening a window (use in CI/headless)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    show = not args.no_show
    args.output.mkdir(parents=True, exist_ok=True)

    print("=== FaceFuse pipeline ===")
    detector = FaceDetector(args.weights)
    diem = DIEMPlus()
    recognizer = FaceRecognizer(device=args.device)
    handcrafted = HandcraftedFeatureExtractor()

    deep_gallery = FaceDatabase(recognizer, detector, diem)
    deep_gallery.build(str(args.gallery))

    hybrid_gallery = HybridFaceDatabase(recognizer, detector, diem, handcrafted)
    hybrid_gallery.build(str(args.gallery))

    hybrid_recognizer = HybridFaceRecognizer(hybrid_gallery)
    probe_images, probe_labels = load_probe_set(args.probe)
    if not probe_images:
        raise SystemExit(f"No probe images found under {args.probe}")

    threshold = config.MATCH_THRESHOLD

    if args.optimize:
        print("\nRunning genetic hyper-parameter search...")

        def fitness(individual):
            thresh, q_scale, d, sigma_c, sigma_s = individual
            candidate = HybridFaceRecognizer(hybrid_gallery, threshold=thresh,
                                             quality_scale=q_scale)
            enhancer = DIEMPlus(int(d), sigma_c, sigma_s)
            correct = sum(
                predict_hybrid(img, detector, enhancer, recognizer,
                               handcrafted, candidate) == label
                for img, label in zip(probe_images, probe_labels)
            )
            return correct / len(probe_images)

        optimizer = GeneticOptimizer(
            population_size=args.population,
            generations=args.generations,
            mutation_rate=config.GA_MUTATION_RATE,
            seed=args.seed,
        )
        best_params, best_fitness = optimizer.optimize(fitness, config.GA_BOUNDS)
        threshold = float(best_params[0])
        hybrid_recognizer.threshold = threshold
        hybrid_recognizer.quality_scale = float(best_params[1])
        diem = DIEMPlus(int(best_params[2]), best_params[3], best_params[4])
        print(
            f"Best parameters: threshold={best_params[0]:.3f}, "
            f"quality_scale={best_params[1]:.1f}, d={int(best_params[2])}, "
            f"sigmaColor={best_params[3]:.1f}, sigmaSpace={best_params[4]:.1f}"
        )
        print(f"Search fitness (accuracy on the probe set): {best_fitness:.4f}")
        print("NOTE: the search maximises accuracy on this same probe set, so the "
              "tuned numbers below are optimistic. See 'Limitations' in the README.")

    deep_preds, deep_times = [], []
    hybrid_preds, hybrid_times = [], []

    print("\nBenchmarking...")
    for img in probe_images:
        start = time.time()
        deep_preds.append(
            predict_deep(img, detector, DIEMPlus(), recognizer, deep_gallery, threshold)
        )
        deep_times.append(time.time() - start)

        start = time.time()
        hybrid_preds.append(
            predict_hybrid(img, detector, diem, recognizer, handcrafted, hybrid_recognizer)
        )
        hybrid_times.append(time.time() - start)

    deep_metrics, hybrid_metrics = Benchmarking().benchmark(
        deep_preds, hybrid_preds, probe_labels, deep_times, hybrid_times
    )

    plot_accuracy_comparison(deep_metrics, hybrid_metrics,
                             save_path=args.output / "accuracy_comparison.png", show=show)
    plot_confusion_matrix(hybrid_metrics["confusion_matrix"], hybrid_metrics["labels"],
                          save_path=args.output / "confusion_matrix.png", show=show)
    plot_quality_weighting(hybrid_recognizer.quality_scale,
                           save_path=args.output / "quality_weighting.png", show=show)

    summary = {
        "identities": sorted(set(probe_labels)),
        "probe_images": len(probe_images),
        "optimized": bool(args.optimize),
        "parameters": {
            "threshold": hybrid_recognizer.threshold,
            "quality_scale": hybrid_recognizer.quality_scale,
            "bilateral_d": diem.diameter,
            "bilateral_sigma_color": diem.sigma_color,
            "bilateral_sigma_space": diem.sigma_space,
        },
        "deep_only": {k: float(v) for k, v in deep_metrics.items()
                      if k not in ("confusion_matrix", "labels")},
        "hybrid": {k: float(v) for k, v in hybrid_metrics.items()
                   if k not in ("confusion_matrix", "labels")},
        "seconds_per_image": {
            "deep_only": sum(deep_times) / len(deep_times),
            "hybrid": sum(hybrid_times) / len(hybrid_times),
        },
        "confusion_matrix": {
            "deep_only": {"labels": deep_metrics["labels"],
                          "matrix": deep_metrics["confusion_matrix"].tolist()},
            "hybrid": {"labels": hybrid_metrics["labels"],
                       "matrix": hybrid_metrics["confusion_matrix"].tolist()},
        },
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}")
    return summary


if __name__ == "__main__":
    main()
