"""Quality-adaptive fusion of deep embeddings with LBP + HOG descriptors."""

from __future__ import annotations

import os

import cv2
import numpy as np

from . import config
from .face_detection import pad_image
from .recognition import cosine_similarity


def get_image_quality(image):
    """Sharpness proxy: variance of the Laplacian. Low value == blurred."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var


class HandcraftedFeatureExtractor:

    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            (64, 128),
            (16, 16),
            (8, 8),
            (8, 8),
            9
        )

    def extract(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        lbp_hist = self._extract_lbp(gray)
        resized = cv2.resize(gray, (64, 128))
        hog_feat = self.hog.compute(resized).flatten()

        return np.concatenate((lbp_hist, hog_feat))

    def _extract_lbp(self, gray):
        """256-bin uniform histogram of 8-neighbour Local Binary Patterns.

        Vectorised over the whole image; the border pixels keep a code of 0,
        matching the reference scalar implementation bit for bit.
        """
        rows, cols = gray.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)
        if rows > 2 and cols > 2:
            center = gray[1:-1, 1:-1]
            neighbours = (
                (gray[0:-2, 0:-2], 7),
                (gray[0:-2, 1:-1], 6),
                (gray[0:-2, 2:], 5),
                (gray[1:-1, 0:-2], 4),
                (gray[1:-1, 2:], 3),
                (gray[2:, 0:-2], 2),
                (gray[2:, 1:-1], 1),
                (gray[2:, 2:], 0),
            )
            code = np.zeros_like(center, dtype=np.uint8)
            for neighbour, shift in neighbours:
                code |= (neighbour >= center).astype(np.uint8) << shift
            lbp[1:-1, 1:-1] = code

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 255))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist


class HybridFaceDatabase:

    def __init__(self, recognizer, detector, diem, handcrafted_extractor):
        self.recognizer = recognizer
        self.detector = detector
        self.diem = diem
        self.handcrafted = handcrafted_extractor
        self.database = {}  

    def build(self, db_path):
        print("Building hybrid face database...")
        for person in os.listdir(db_path):
            print(f"Scanning: {person}")
            person_path = os.path.join(db_path, person)
            if not os.path.isdir(person_path):
                continue
            features_list = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = self.detector.detect(img, extract_crops=True)
                if not faces:
                    continue
                face = max(faces, key=lambda f: f.confidence)
                crop = face.crop
                if crop.size == 0:
                    continue
                enhanced = self.diem.process(crop)
                padded = pad_image(enhanced, target_size=320)
                emb = self.recognizer.get_embedding(padded)
                if emb is None:
                    print(f"InsightFace failed in {img_name}")
                    continue
                hand_feat = self.handcrafted.extract(enhanced)
                features_list.append((emb, hand_feat))
            if features_list:
                self.database[person] = features_list
            else:
                print(f"No features for {person}")
        print(f"Hybrid Database ready: {list(self.database.keys())}")


class HybridFaceRecognizer:

    def __init__(self, hybrid_database, threshold=config.MATCH_THRESHOLD,
                 quality_scale=config.QUALITY_SCALE):
        self.database = hybrid_database.database
        self.threshold = threshold
        self.quality_scale = quality_scale 

    def identify(self, embedding, hand_feat, enhanced_crop):
        quality = get_image_quality(enhanced_crop)
        weight_deep = min(1.0, max(0.0, quality / self.quality_scale))
        weight_hand = 1.0 - weight_deep

        best_score = -1
        identity = "Unknown"
        for person, feats_list in self.database.items():
            for db_emb, db_hand in feats_list:
                deep_sim = cosine_similarity(embedding, db_emb)
                hand_sim = cosine_similarity(hand_feat, db_hand)
                fused_score = weight_deep * deep_sim + weight_hand * hand_sim
                if fused_score > best_score:
                    best_score = fused_score
                    identity = person

        if best_score < self.threshold:
            identity = "Unknown"
        return identity, best_score