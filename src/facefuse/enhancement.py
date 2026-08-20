"""DIEM+ enhancement: edge-preserving denoising ahead of feature extraction."""

from __future__ import annotations

import cv2

from . import config


class DIEMPlus:
    """Bilateral filter that smooths sensor noise while keeping facial edges.

    The three parameters are exposed because the genetic optimiser tunes them
    jointly with the recognition threshold (see :mod:`facefuse.genetic_optimizer`).
    """

    def __init__(
        self,
        diameter: int = config.BILATERAL_DIAMETER,
        sigma_color: float = config.BILATERAL_SIGMA_COLOR,
        sigma_space: float = config.BILATERAL_SIGMA_SPACE,
    ):
        self.diameter = int(diameter)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def process(self, image):
        return cv2.bilateralFilter(image, self.diameter, self.sigma_color, self.sigma_space)
