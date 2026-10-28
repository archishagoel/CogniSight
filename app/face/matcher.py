from typing import List, Optional, Tuple
import numpy as np

from app.config.settings import settings
from app.face.schemas import BoundingBox, RecognitionResult
from app.utils.logging_config import logger


class FaceMatcher:
    """Similarity Matcher module using L2 Cosine Similarity for face embeddings.

    Compares query 512-D face embeddings against enrolled database candidates,
    evaluates maximum similarity against configurable decision thresholds, and returns
    structured RecognitionResult objects.
    """

    def __init__(self, threshold: float = settings.FACE_RECOGNITION_THRESHOLD):
        self.threshold = threshold

    @staticmethod
    def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two 1D numpy embedding vectors.

        For L2 normalized vectors (length = 1.0), cosine similarity equals the dot product.
        """
        if vec1 is None or vec2 is None or vec1.size == 0 or vec2.size == 0:
            return 0.0

        # Ensure 1D float32 arrays
        v1 = vec1.flatten().astype(np.float32)
        v2 = vec2.flatten().astype(np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Dot product divided by magnitudes
        similarity = float(np.dot(v1, v2) / (norm1 * norm2))

        # Clip bounds to [-1.0, 1.0] for floating point safety
        return float(np.clip(similarity, -1.0, 1.0))

    def match_face(
        self,
        query_embedding: np.ndarray,
        enrolled_candidates: List[Tuple[int, str, np.ndarray]],
        bbox: Optional[BoundingBox] = None,
        override_threshold: Optional[float] = None,
    ) -> RecognitionResult:
        """Matches a query face embedding against a list of enrolled person candidates.

        Args:
            query_embedding: 512-D float32 normalized face embedding.
            enrolled_candidates: List of (person_id, person_name, embedding_vector).
            bbox: BoundingBox of detected face.
            override_threshold: Optional custom threshold override.

        Returns:
            RecognitionResult: Result detailing whether face was recognized or unknown.
        """
        threshold = (
            override_threshold if override_threshold is not None else self.threshold
        )
        dummy_bbox = (
            bbox if bbox is not None else BoundingBox(x1=0, y1=0, x2=0, y2=0)
        )

        if query_embedding is None or len(enrolled_candidates) == 0:
            return RecognitionResult(
                bbox=dummy_bbox,
                confidence=0.0,
                recognized=False,
                person_id=None,
                name="Unknown Person",
                similarity=0.0,
            )

        best_person_id: Optional[int] = None
        best_name: Optional[str] = None
        max_similarity: float = -1.0

        # Compare query against all stored embeddings
        for p_id, p_name, stored_vector in enrolled_candidates:
            sim = self.calculate_cosine_similarity(query_embedding, stored_vector)
            if sim > max_similarity:
                max_similarity = sim
                best_person_id = p_id
                best_name = p_name

        # Apply recognition decision threshold
        if max_similarity >= threshold and best_person_id is not None:
            return RecognitionResult(
                bbox=dummy_bbox,
                confidence=max_similarity,
                recognized=True,
                person_id=best_person_id,
                name=best_name,
                similarity=max_similarity,
            )

        # Below threshold -> Unknown Person
        return RecognitionResult(
            bbox=dummy_bbox,
            confidence=max(0.0, max_similarity),
            recognized=False,
            person_id=None,
            name="Unknown Person",
            similarity=max(0.0, max_similarity),
        )
