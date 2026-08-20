"""Head-to-head comparison of the deep-only and hybrid recognisers."""

from __future__ import annotations

from .evaluation import SystemEvaluator


class Benchmarking:
    def benchmark(self, pure_preds, hybrid_preds, labels, pure_times, hybrid_times):
        evaluator = SystemEvaluator()

        pure_metrics = evaluator.evaluate(labels, pure_preds)
        hybrid_metrics = evaluator.evaluate(labels, hybrid_preds)

        pure_fps = evaluator.compute_fps(pure_times)
        hybrid_fps = evaluator.compute_fps(hybrid_times)

        print("\n=== BENCHMARK RESULTS ===")
        print("Deep-only baseline (InsightFace):")
        print(f"  Accuracy : {pure_metrics['accuracy']:.4f}")
        print(f"  Precision: {pure_metrics['precision']:.4f}")
        print(f"  Recall   : {pure_metrics['recall']:.4f}")
        print(f"  F1-score : {pure_metrics['f1']:.4f}")
        print(f"  FPS      : {pure_fps:.2f}")

        print("\nFaceFuse hybrid (deep + LBP/HOG + quality-adaptive fusion):")
        print(f"  Accuracy : {hybrid_metrics['accuracy']:.4f}")
        print(f"  Precision: {hybrid_metrics['precision']:.4f}")
        print(f"  Recall   : {hybrid_metrics['recall']:.4f}")
        print(f"  F1-score : {hybrid_metrics['f1']:.4f}")
        print(f"  FPS      : {hybrid_fps:.2f}")

        return pure_metrics, hybrid_metrics