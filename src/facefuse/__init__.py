"""FaceFuse - quality-adaptive hybrid face recognition.

Combines InsightFace deep embeddings with handcrafted LBP + HOG descriptors,
weighting the two modalities per-frame according to an image sharpness estimate.
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "face_detection",
    "enhancement",
    "recognition",
    "database",
    "hybrid_recognition",
    "genetic_optimizer",
    "evaluation",
    "benchmarking",
    "visualization",
]
