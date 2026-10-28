from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.database.repository import PersonRepository
from app.face.detector import FaceDetector
from app.face.embedding import FaceEmbedder
from app.face.matcher import FaceMatcher
from app.face.schemas import BoundingBox, RecognitionResult
from app.utils.logging_config import logger


def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = boxA.area
    boxB_area = boxB.area

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return float(iou)


class FaceRecognizer:
    """Complete Face Recognition Pipeline Engine for CogniSight.

    Connects Face Detection -> Embedding Generation -> DB Candidate Search -> Similarity Matching.
    Supports multi-face tracking, temporal stability (smoothing), and frame-skipping optimizations.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        recognition_interval: int = settings.RECOGNITION_INTERVAL,
        smoothing_window: int = 5,
    ):
        self.detector = detector if detector is not None else FaceDetector()
        self.embedder = embedder if embedder is not None else FaceEmbedder(self.detector)
        self.matcher = matcher if matcher is not None else FaceMatcher()

        self.recognition_interval = recognition_interval
        self.smoothing_window = smoothing_window

        self.frame_counter: int = 0
        self.last_results: List[RecognitionResult] = []

        # History tracker: dict of track_id -> deque of recent RecognitionResult
        self.history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )
        self.next_track_id: int = 1

    def _match_track_id(
        self, current_box: BoundingBox, active_tracks: Dict[int, BoundingBox]
    ) -> int:
        """Assigns continuous track_id to detected bounding boxes using spatial IoU matching."""
        best_track_id = None
        max_iou = 0.30  # Minimum IoU threshold to associate face across frames

        for track_id, last_box in active_tracks.items():
            iou = compute_iou(current_box, last_box)
            if iou > max_iou:
                max_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = self.next_track_id
            self.next_track_id += 1

        return best_track_id

    def get_smoothed_result(
        self, track_id: int, latest_result: RecognitionResult
    ) -> RecognitionResult:
        """Applies majority voting / temporal smoothing over recent recognition history."""
        self.history[track_id].append(latest_result)
        window = list(self.history[track_id])

        if not window:
            return latest_result

        # Count occurrences of recognized identities
        votes: Dict[Optional[int], List[RecognitionResult]] = defaultdict(list)
        for res in window:
            votes[res.person_id].append(res)

        # Find person_id with most votes
        majority_person_id = max(votes.keys(), key=lambda pid: len(votes[pid]))
        majority_results = votes[majority_person_id]

        # Calculate average similarity for majority identity
        avg_similarity = float(
            np.mean([r.similarity for r in majority_results])
        )

        sample = majority_results[-1]
        return RecognitionResult(
            bbox=latest_result.bbox,
            confidence=avg_similarity,
            recognized=sample.recognized,
            person_id=sample.person_id,
            name=sample.name,
            similarity=avg_similarity,
        )

    def process_frame(
        self, frame: np.ndarray, db: Session
    ) -> List[RecognitionResult]:
        """Processes a single camera video frame through the full recognition pipeline.

        Performs neural embedding inference every `RECOGNITION_INTERVAL` frames for high performance.

        Args:
            frame: Raw OpenCV video frame (BGR).
            db: SQLAlchemy Database Session to load enrolled embeddings.

        Returns:
            List[RecognitionResult]: Recognition predictions for all faces in frame.
        """
        self.frame_counter += 1
        should_run_recognition = (
            self.frame_counter % self.recognition_interval == 1
            or not self.last_results
        )

        # 1. Detect all faces in frame
        detected_faces = self.detector.detect_faces(frame)
        if not detected_faces:
            self.last_results = []
            return []

        # Load enrolled database candidates
        enrolled_candidates = PersonRepository.get_all_enrolled_embeddings(db)

        current_results = []
        raw_faces = self.detector.detect_raw(frame)

        # If matching count aligns with raw face objects
        for i, face in enumerate(detected_faces):
            bbox = face.bbox

            if should_run_recognition and i < len(raw_faces):
                raw_face = raw_faces[i]
                emb = self.embedder.extract_embedding_from_raw_face(raw_face)
                result = self.matcher.match_face(
                    query_embedding=emb,
                    enrolled_candidates=enrolled_candidates,
                    bbox=bbox,
                )
            else:
                # Reuse spatial prediction or default unknown on non-inference frames
                matched_prev = None
                for prev_res in self.last_results:
                    if compute_iou(bbox, prev_res.bbox) > 0.30:
                        matched_prev = prev_res
                        break

                if matched_prev:
                    result = RecognitionResult(
                        bbox=bbox,
                        confidence=matched_prev.confidence,
                        recognized=matched_prev.recognized,
                        person_id=matched_prev.person_id,
                        name=matched_prev.name,
                        similarity=matched_prev.similarity,
                    )
                else:
                    result = RecognitionResult(
                        bbox=bbox,
                        confidence=0.0,
                        recognized=False,
                        person_id=None,
                        name="Unknown Person",
                        similarity=0.0,
                    )

            # Apply temporal smoothing
            track_id = self._match_track_id(bbox, {i: r.bbox for i, r in enumerate(self.last_results)})
            smoothed = self.get_smoothed_result(track_id, result)
            current_results.append(smoothed)

        self.last_results = current_results
        return current_results

    def draw_recognition(
        self, frame: np.ndarray, results: List[RecognitionResult]
    ) -> np.ndarray:
        """Visualizes recognition results with styled bounding boxes and labels."""
        output_frame = frame.copy()

        for res in results:
            bbox = res.bbox
            if res.recognized:
                color = (0, 255, 0)  # Green for Enrolled Known Person
                label = f"{res.name} ({res.similarity * 100:.0f}%)"
            else:
                color = (0, 140, 255)  # Orange for Unknown Person
                label = f"Unknown Person ({res.similarity * 100:.0f}%)"

            # Draw bounding box
            cv2.rectangle(
                output_frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                2,
            )

            # Label box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(bbox.y1 - 22, 0)
            cv2.rectangle(
                output_frame,
                (bbox.x1, y_label),
                (bbox.x1 + w + 10, y_label + h + 10),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                output_frame,
                label,
                (bbox.x1 + 5, y_label + h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return output_frame
