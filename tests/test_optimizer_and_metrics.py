"""Genetic optimiser, evaluation metrics and geometry helpers."""

import numpy as np
import pytest

from facefuse.enhancement import DIEMPlus
from facefuse.evaluation import SystemEvaluator
from facefuse.face_detection import pad_image
from facefuse.genetic_optimizer import GeneticOptimizer


def test_optimizer_finds_the_optimum_of_a_smooth_objective():
    """Maximising -(x - 3)^2 - (y + 1)^2 should converge near (3, -1)."""
    optimizer = GeneticOptimizer(population_size=30, generations=40,
                                 mutation_rate=0.3, seed=0)
    best, fitness = optimizer.optimize(
        lambda ind: -((ind[0] - 3.0) ** 2) - ((ind[1] + 1.0) ** 2),
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        verbose=False,
    )
    assert best == pytest.approx([3.0, -1.0], abs=0.5)
    assert fitness > -0.5


def test_optimizer_respects_bounds():
    optimizer = GeneticOptimizer(population_size=12, generations=10,
                                 mutation_rate=1.0, seed=1)
    bounds = [(0.3, 0.7), (100.0, 2000.0)]
    best, _ = optimizer.optimize(lambda ind: float(ind[0] + ind[1]), bounds, verbose=False)
    for value, (low, high) in zip(best, bounds):
        assert low <= value <= high


def test_optimizer_is_reproducible_for_a_fixed_seed():
    def run(seed):
        return GeneticOptimizer(population_size=10, generations=5,
                                mutation_rate=0.5, seed=seed).optimize(
            lambda ind: -float(np.sum(ind ** 2)), [(-5.0, 5.0)] * 3, verbose=False)[0]

    np.testing.assert_allclose(run(42), run(42))
    assert not np.allclose(run(42), run(7))


def test_metrics_on_a_perfect_prediction():
    evaluator = SystemEvaluator()
    labels = ["a", "b", "a", "c"]
    metrics = evaluator.evaluate(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_metrics_on_a_partly_wrong_prediction():
    evaluator = SystemEvaluator()
    metrics = evaluator.evaluate(["a", "b", "a", "b"], ["a", "b", "b", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0.0 < metrics["f1"] < 1.0


def test_fps_is_the_inverse_of_mean_latency():
    evaluator = SystemEvaluator()
    assert evaluator.compute_fps([0.1, 0.1, 0.2]) == pytest.approx(3 / 0.4)
    assert evaluator.compute_fps([]) == 0.0


@pytest.mark.parametrize("shape", [(500, 200, 3), (100, 100, 3), (17, 640, 3)])
def test_pad_image_produces_a_square_without_distorting(shape):
    image = np.full(shape, 200, dtype=np.uint8)
    padded = pad_image(image, target_size=320)
    assert padded.shape == (320, 320, 3)
    assert padded.dtype == np.uint8


def test_pad_image_keeps_the_aspect_ratio_of_the_content():
    image = np.full((400, 200, 3), 255, dtype=np.uint8)
    padded = pad_image(image, target_size=320)
    content = padded.any(axis=2)
    heights = content.any(axis=1).sum()
    widths = content.any(axis=0).sum()
    assert heights / widths == pytest.approx(2.0, abs=0.05)


def test_enhancer_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    enhanced = DIEMPlus().process(image)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == image.dtype


def test_enhancer_parameters_are_configurable():
    enhancer = DIEMPlus(9, 75.0, 75.0)
    assert (enhancer.diameter, enhancer.sigma_color, enhancer.sigma_space) == (9, 75.0, 75.0)
