"""Filesystem layout and runtime defaults.

Every path is resolved relative to the repository root, so the project runs
unchanged on any machine and any operating system. Each default can be
overridden with an environment variable or a command-line flag.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/facefuse/config.py -> repository root
ROOT = Path(__file__).resolve().parents[2]


def _path(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


#: YOLO face-detector weights (see scripts/download_weights.py).
WEIGHTS_PATH = _path("FACEFUSE_WEIGHTS", ROOT / "models" / "yolo11l-face.pt")

#: Enrolled identities: data/gallery/<person name>/<image>.jpg
GALLERY_DIR = _path("FACEFUSE_GALLERY", ROOT / "data" / "gallery")

#: Held-out evaluation images, same folder-per-identity layout.
PROBE_DIR = _path("FACEFUSE_PROBE", ROOT / "data" / "probe")

#: Where pipeline figures and reports are written.
OUTPUT_DIR = _path("FACEFUSE_OUTPUT", ROOT / "outputs")

#: InsightFace model pack used for the 512-d embeddings.
EMBEDDING_MODEL = os.environ.get("FACEFUSE_EMBEDDING_MODEL", "buffalo_l")

# --- Recognition defaults -------------------------------------------------
DETECTION_CONFIDENCE = 0.55
DETECTION_MARGIN = 0.20
EMBEDDING_INPUT_SIZE = 320
MATCH_THRESHOLD = 0.50
QUALITY_SCALE = 500.0

# --- DIEM+ bilateral filter defaults --------------------------------------
BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 30.0
BILATERAL_SIGMA_SPACE = 30.0

# --- Genetic algorithm search space ---------------------------------------
GA_POPULATION_SIZE = 15
GA_GENERATIONS = 8
GA_MUTATION_RATE = 0.15
GA_BOUNDS = [
    (0.30, 0.70),      # match threshold
    (100.0, 2000.0),   # quality scale (Laplacian variance -> deep weight)
    (3.0, 9.0),        # bilateral diameter
    (10.0, 100.0),     # bilateral sigmaColor
    (10.0, 100.0),     # bilateral sigmaSpace
]
