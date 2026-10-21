from typing import List, Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config.settings import settings
from app.face.schemas import BoundingBox, DetectedFace
from app.utils.logging_config import logger


class FaceDetector:
    """Face Detector module using InsightFace (SCRFD/RetinaFace backend).

    Responsible for detecting faces in raw OpenCV frames, extracting bounding boxes
    and 5-point facial landmarks.
    """

    def __init__(
        self,
        model_name: str = settings.INSIGHTFACE_MODEL_NAME,
        det_threshold: float = settings.DETECTION_THRESHOLD,
        det_size: Tuple[int, int] = (640, 640),
    ):
        self.det_threshold = det_threshold
        self.det_size = det_size
        logger.info(f"Initializing FaceDetector with model: {model_name}...")

        # Initialize InsightFace Analysis app with CPU execution provider
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_thresh=det_threshold, det_size=det_size)
        logger.info("FaceDetector successfully initialized.")

    def detect_raw(self, frame: np.ndarray) -> List:
        """Detect faces and return raw InsightFace Face objects."""
        if frame is None or frame.size == 0:
            logger.warning("Empty or invalid frame passed to face detector.")
            return []

        # InsightFace expects BGR image (standard OpenCV format)
        raw_faces = self.app.get(frame)
        return raw_faces

    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detect faces in frame and return structured DetectedFace Pydantic models.

        Args:
            frame: OpenCV image frame (BGR format)

        Returns:
            List[DetectedFace]: List of detected face bounding boxes & confidence scores.
        """
        raw_faces = self.detect_raw(frame)
        detected_faces = []

        for face in raw_faces:
            # Extract bounding box coordinates [x1, y1, x2, y2]
            box = face.bbox.astype(int)
            bbox = BoundingBox(
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
            )
            confidence = float(face.det_score)

            landmarks = (
                face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else None
            )

            detected_faces.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )

        return detected_faces

    def draw_detections(
        self, frame: np.ndarray, faces: List[DetectedFace]
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels on an OpenCV frame."""
        output_frame = frame.copy()

        for face in faces:
            bbox = face.bbox
            conf_str = f"Face: {face.confidence * 100:.1f}%"

            # Draw bounding box
            cv2.rectangle(
                output_frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                (0, 255, 0),  # Green box
                2,
            )

            # Draw label background box
            label_size, _ = cv2.getTextSize(
                conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            y1_label = max(bbox.y1 - 20, 0)
            cv2.rectangle(
                output_frame,
                (bbox.x1, y1_label),
                (bbox.x1 + label_size[0] + 10, y1_label + label_size[1] + 10),
                (0, 255, 0),
                cv2.FILLED,
            )

            # Write text label
            cv2.putText(
                output_frame,
                conf_str,
                (bbox.x1 + 5, y1_label + label_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA,
            )

        return output_frame
