"""Deep face embeddings (InsightFace) and cosine-similarity identity matching."""

from __future__ import annotations

import numpy as np

from . import config


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify(embedding, database, threshold=config.MATCH_THRESHOLD):
    best_score = -1
    identity = "Unknown"
    for person, embeddings in database.items():
        for db_emb in embeddings:
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                identity = person
    if best_score < threshold:
        identity = "Unknown"
    return identity, best_score


class FaceRecognizer:
    """Wraps an InsightFace model pack and returns 512-d face embeddings.

    ``ctx_id`` follows the InsightFace convention: ``0`` selects the first GPU,
    ``-1`` forces CPU execution. ``device="auto"`` picks a GPU only when the
    installed onnxruntime actually exposes a CUDA provider, so the same code
    runs on a laptop without CUDA.
    """

    def __init__(self, det_size=None, device="auto", model_name=None):
        # Imported lazily so that the detection, handcrafted-feature, metrics
        # and plotting modules stay importable without the InsightFace stack.
        from insightface.app import FaceAnalysis

        size = det_size or (config.EMBEDDING_INPUT_SIZE, config.EMBEDDING_INPUT_SIZE)
        self.app = FaceAnalysis(name=model_name or config.EMBEDDING_MODEL)
        self.app.prepare(ctx_id=resolve_ctx_id(device), det_size=size)

    def get_embedding(self, face_image):
        faces = self.app.get(face_image)
        if len(faces) == 0:
            return None
        return faces[0].embedding


def resolve_ctx_id(device="auto") -> int:
    """Map ``auto`` / ``cpu`` / ``gpu`` (or an explicit index) to a ctx_id."""
    if isinstance(device, int):
        return device
    device = str(device).lower()
    if device == "cpu":
        return -1
    if device in ("gpu", "cuda"):
        return 0
    try:
        import onnxruntime

        return 0 if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else -1
    except Exception:
        return -1
