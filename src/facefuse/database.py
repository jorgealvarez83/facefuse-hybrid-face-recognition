"""Builds the deep-embedding gallery from folder-per-identity images."""

from __future__ import annotations

import os

import cv2

from .face_detection import pad_image


class FaceDatabase:
    def __init__(self, recognizer, detector, diem):
        self.recognizer = recognizer
        self.detector = detector
        self.diem = diem
        self.database = {}

    def build(self, db_path):
        print("Building face database...")
        for person in os.listdir(db_path):
            print(f"Scanning: {person}")
            person_path = os.path.join(db_path, person)
            if not os.path.isdir(person_path):
                continue
            embeddings = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = self.detector.detect(img, extract_crops=True)
                if not faces:
                    print(f"No face detected in {img_name}")
                    continue
                face = max(faces, key=lambda f: f.confidence)
                crop = face.crop
                if crop.size == 0:
                    continue
                enhanced = self.diem.process(crop)
                padded = pad_image(enhanced)
                emb = self.recognizer.get_embedding(padded)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    print(f"InsightFace failed to detect face in crop from {img_name}")
            if embeddings:
                self.database[person] = embeddings
            else:
                print(f"No embeddings for {person}")
        print(f"Database ready: {list(self.database.keys())}")