"""Quality-adaptive fusion and cosine matching."""

import numpy as np
import pytest

from facefuse.hybrid_recognition import HybridFaceRecognizer
from facefuse.recognition import cosine_similarity, identify


class FakeGallery:
    """Stands in for HybridFaceDatabase so the tests need no models or images."""

    def __init__(self, database):
        self.database = database


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def gallery():
    return FakeGallery({
        "alice": [(unit(1, 0, 0, 0), unit(0, 1, 0, 0))],
        "bob": [(unit(0, 0, 1, 0), unit(0, 0, 0, 1))],
    })


def sharp_crop(value=255):
    """A high-contrast checkerboard: large Laplacian variance."""
    crop = np.zeros((32, 32), dtype=np.uint8)
    crop[::2, ::2] = value
    crop[1::2, 1::2] = value
    return crop


def flat_crop():
    """A uniform patch: Laplacian variance is exactly zero."""
    return np.full((32, 32), 128, dtype=np.uint8)


def test_cosine_similarity_bounds():
    a = unit(1, 0)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, unit(0, 1)) == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity(a, unit(-1, 0)) == pytest.approx(-1.0)


def test_identify_returns_unknown_below_threshold():
    database = {"alice": [unit(1, 0, 0)]}
    name, score = identify(unit(0, 1, 0), database, threshold=0.5)
    assert name == "Unknown"
    assert score == pytest.approx(0.0, abs=1e-6)


def test_identify_returns_best_match_above_threshold():
    database = {"alice": [unit(1, 0, 0)], "bob": [unit(0, 1, 0)]}
    name, score = identify(unit(0.9, 0.1, 0), database, threshold=0.5)
    assert name == "alice"
    assert score > 0.9


def test_sharp_crop_trusts_the_deep_embedding(gallery):
    """With high sharpness the deep score dominates, even if LBP/HOG disagrees."""
    recognizer = HybridFaceRecognizer(gallery, threshold=0.4, quality_scale=1.0)
    name, _ = recognizer.identify(unit(1, 0, 0, 0), unit(0, 0, 0, 1), sharp_crop())
    assert name == "alice"


def test_flat_crop_falls_back_to_handcrafted_features(gallery):
    """Zero sharpness drives the deep weight to 0, so LBP/HOG decides."""
    recognizer = HybridFaceRecognizer(gallery, threshold=0.4, quality_scale=500.0)
    name, _ = recognizer.identify(unit(1, 0, 0, 0), unit(0, 0, 0, 1), flat_crop())
    assert name == "bob"


def test_unknown_when_nothing_clears_the_threshold(gallery):
    recognizer = HybridFaceRecognizer(gallery, threshold=0.99, quality_scale=1.0)
    name, score = recognizer.identify(unit(0, 1, 0, 0), unit(1, 0, 0, 0), sharp_crop())
    assert name == "Unknown"
    assert score < 0.99


def test_quality_scale_controls_the_crossover(gallery):
    """A larger quality_scale keeps more weight on the handcrafted branch."""
    strict = HybridFaceRecognizer(gallery, threshold=0.4, quality_scale=1e9)
    lenient = HybridFaceRecognizer(gallery, threshold=0.4, quality_scale=1e-9)
    crop = sharp_crop()
    assert strict.identify(unit(1, 0, 0, 0), unit(0, 0, 0, 1), crop)[0] == "bob"
    assert lenient.identify(unit(1, 0, 0, 0), unit(0, 0, 0, 1), crop)[0] == "alice"
