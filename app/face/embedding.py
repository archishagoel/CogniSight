from typing import List, Optional, Tuple
import numpy as np

from app.face.detector import FaceDetector
from app.utils.logging_config import logger


class FaceEmbedder:
    """Face Embedding Generator module using ArcFace (InsightFace MobileFaceNet / ResNet).

    Translates cropped/aligned face images into a 512-dimensional normalized numerical vector.
    """

    EMBEDDING_DIM = 512

    def __init__(self, detector: Optional[FaceDetector] = None):
        self.detector = detector if detector is not None else FaceDetector()

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2 normalizes a 1D embedding vector so its euclidean length equals 1.0."""
        norm = np.linalg.norm(embedding)
        if norm == 0 or np.isnan(norm):
            return embedding
        return (embedding / norm).astype(np.float32)

    def extract_embedding_from_raw_face(self, raw_face) -> Optional[np.ndarray]:
        """Extracts and normalizes embedding from an InsightFace Face object."""
        if raw_face is None or not hasattr(raw_face, "embedding") or raw_face.embedding is None:
            logger.warning("Raw face object has no embedding vector.")
            return None

        embedding = raw_face.embedding.astype(np.float32)
        if embedding.shape[0] != self.EMBEDDING_DIM:
            logger.warning(
                f"Unexpected embedding dimension: {embedding.shape[0]} (expected {self.EMBEDDING_DIM})"
            )

        return self.normalize_embedding(embedding)

    def generate_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects the primary face in an image frame and generates its normalized 512-D embedding.

        Args:
            frame: OpenCV image frame containing a face.

        Returns:
            Optional[np.ndarray]: 512-dimensional float32 normalized embedding vector, or None if no face detected.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame supplied for embedding generation.")
            return None

        raw_faces = self.detector.detect_raw(frame)
        if not raw_faces:
            logger.warning("No face detected in supplied frame for embedding generation.")
            return None

        # Sort detected faces by bounding box area to select the most prominent/largest face
        primary_face = max(
            raw_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        return self.extract_embedding_from_raw_face(primary_face)

    def generate_embeddings_for_all_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Generates normalized embeddings for ALL faces present in a single image frame."""
        if frame is None or frame.size == 0:
            return []

        raw_faces = self.detector.detect_raw(frame)
        results = []

        for face in raw_faces:
            emb = self.extract_embedding_from_raw_face(face)
            if emb is not None:
                box = tuple(face.bbox.astype(int).tolist())
                results.append((emb, box))

        return results
