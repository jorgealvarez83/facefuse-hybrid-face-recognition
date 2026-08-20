"""Handcrafted feature extractor: LBP correctness and HOG shape."""

import numpy as np
import pytest

from facefuse.hybrid_recognition import HandcraftedFeatureExtractor, get_image_quality

LBP_BINS = 256
HOG_DIM = 3780  # 64x128 window, 16x16 blocks, 8x8 stride, 9 orientations


def reference_lbp(gray):
    """Straightforward scalar LBP, used as the ground truth for the fast path."""
    rows, cols = gray.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = gray[i, j]
            code = 0
            code |= (gray[i - 1, j - 1] >= center) << 7
            code |= (gray[i - 1, j] >= center) << 6
            code |= (gray[i - 1, j + 1] >= center) << 5
            code |= (gray[i, j - 1] >= center) << 4
            code |= (gray[i, j + 1] >= center) << 3
            code |= (gray[i + 1, j - 1] >= center) << 2
            code |= (gray[i + 1, j] >= center) << 1
            code |= (gray[i + 1, j + 1] >= center) << 0
            lbp[i, j] = code
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, 255))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    return hist


@pytest.fixture
def extractor():
    return HandcraftedFeatureExtractor()


@pytest.mark.parametrize("shape", [(32, 32), (48, 27), (3, 3)])
def test_vectorised_lbp_matches_scalar_reference(extractor, shape):
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
    np.testing.assert_array_equal(extractor._extract_lbp(gray), reference_lbp(gray))


def test_lbp_handles_degenerate_images(extractor):
    """Images too small for a 3x3 neighbourhood must not raise."""
    hist = extractor._extract_lbp(np.zeros((2, 2), dtype=np.uint8))
    assert hist.shape == (LBP_BINS,)
    assert hist[0] == pytest.approx(1.0)


def test_lbp_histogram_is_normalised(extractor):
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    assert extractor._extract_lbp(gray).sum() == pytest.approx(1.0, abs=1e-5)


def test_lbp_is_invariant_to_uniform_brightness_gain(extractor):
    """LBP compares neighbours to the centre, so a monotonic gain must not matter."""
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 128, size=(40, 40), dtype=np.uint8)
    brighter = (gray.astype(np.uint16) * 2).clip(0, 255).astype(np.uint8)
    np.testing.assert_array_equal(
        extractor._extract_lbp(gray), extractor._extract_lbp(brighter)
    )


def test_feature_vector_dimensions(extractor):
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, size=(80, 60, 3), dtype=np.uint8)
    features = extractor.extract(bgr)
    assert features.shape == (LBP_BINS + HOG_DIM,)
    assert np.isfinite(features).all()


def test_extract_accepts_grayscale_and_colour(extractor):
    rng = np.random.default_rng(2)
    bgr = rng.integers(0, 256, size=(70, 70, 3), dtype=np.uint8)
    gray = np.ascontiguousarray(bgr[:, :, 0])
    assert extractor.extract(bgr).shape == extractor.extract(gray).shape


def test_blurred_image_scores_lower_quality():
    import cv2

    rng = np.random.default_rng(5)
    sharp = rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert get_image_quality(blurred) < get_image_quality(sharp)
