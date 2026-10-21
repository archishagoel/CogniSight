from typing import List, Optional, Tuple
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.camera.camera import Camera
from app.config.settings import settings
from app.database.models import Person
from app.database.repository import PersonRepository
from app.face.detector import FaceDetector
from app.face.embedding import FaceEmbedder
from app.face.enroll import FaceEnrollmentService
from app.face.matcher import FaceMatcher
from app.face.schemas import BoundingBox, DetectedFace, RecognitionResult
from app.utils.logging_config import logger
from app.voice.voice_parser import VoiceTranscriptParser


import time


class RecognitionService:
    """Service layer decoupling business logic from FastAPI HTTP endpoints and hardware handlers."""

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
        enrollment_service: Optional[FaceEnrollmentService] = None,
    ):
        self.detector = detector if detector is not None else FaceDetector()
        self.embedder = embedder if embedder is not None else FaceEmbedder(self.detector)
        self.matcher = matcher if matcher is not None else FaceMatcher()
        self.enrollment_service = (
            enrollment_service
            if enrollment_service is not None
            else FaceEnrollmentService(self.detector, self.embedder)
        )

    def get_health_status(self) -> dict:
        """Returns service health status."""
        return {
            "status": "ok",
            "service": settings.APP_NAME,
            "version": "1.0.0",
            "model": settings.INSIGHTFACE_MODEL_NAME,
            "threshold": settings.FACE_RECOGNITION_THRESHOLD,
        }

    def get_all_persons(self, db: Session) -> List[Person]:
        """Retrieves list of all enrolled people."""
        return PersonRepository.get_all_persons(db)

    def get_person(self, db: Session, person_id: int) -> Optional[Person]:
        """Retrieves a single person by ID."""
        return PersonRepository.get_person_by_id(db, person_id)

    def create_person(self, db: Session, name: str) -> Person:
        """Creates a person profile record."""
        return PersonRepository.create_person(db, name)

    def delete_person(self, db: Session, person_id: int) -> bool:
        """Deletes a person profile and their face embeddings."""
        return PersonRepository.delete_person(db, person_id)

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decodes uploaded image byte stream into an OpenCV BGR numpy array."""
        if not image_bytes:
            return None

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    def capture_face_for_person(
        self, db: Session, person_id: int, image_bytes: Optional[bytes] = None
    ) -> Tuple[bool, str, int]:
        """Captures and stores face embedding for a person from webcam frame or uploaded bytes."""
        person = PersonRepository.get_person_by_id(db, person_id)
        if not person:
            return False, f"Person with ID {person_id} not found.", 0

        frame = None
        if image_bytes:
            frame = self.decode_image_bytes(image_bytes)
        else:
            # First check if there is a live frame in Camera.get_latest_frame() from the video stream
            frame = Camera.get_latest_frame()
            if frame is None or frame.size == 0:
                # Fallback: Try capturing frame with new camera manager instance
                camera = Camera()
                if camera.start():
                    try:
                        ret, grabbed_frame = camera.read_frame()
                        if ret and grabbed_frame is not None:
                            frame = grabbed_frame
                    finally:
                        camera.release()

        if frame is None or frame.size == 0:
            return False, "Could not acquire a valid image frame. Please ensure camera is available or upload a photo.", len(person.embeddings)

        raw_faces = self.detector.detect_raw(frame)
        if not raw_faces:
            return (
                False,
                "No face detected in frame. Please look directly at the camera.",
                len(person.embeddings),
            )

        # Pick largest face
        primary = max(
            raw_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        b = primary.bbox.astype(int)
        det_face = DetectedFace(
            bbox=BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3])),
            confidence=float(primary.det_score),
        )

        is_valid, reason = self.enrollment_service.validate_sample_quality(frame, det_face)
        if not is_valid:
            logger.warning(f"Face quality check notice: {reason}")
            if det_face.confidence < 0.40:
                return False, f"Quality check failed: {reason}", len(person.embeddings)

        emb = self.embedder.extract_embedding_from_raw_face(primary)
        if emb is None:
            return False, "Failed to extract 512-D face embedding.", len(person.embeddings)

        PersonRepository.add_face_embedding(db, person.id, emb)
        db.refresh(person)
        count = len(person.embeddings)

        msg = f"Successfully captured & stored 512-D face embedding for '{person.name}'!"
        logger.info(msg)
        return True, msg, count

    def auto_voice_enroll_from_transcript(
        self, db: Session, transcript: str
    ) -> Tuple[bool, str, dict]:
        """Automatically parses spoken audio transcript, creates person profile, and captures face embedding."""
        name, relationship = VoiceTranscriptParser.parse_name_and_relationship(transcript)

        person = PersonRepository.create_person(db, name)
        PersonRepository.update_profile(
            db,
            person.id,
            relationship=relationship,
            notes=f"Enrolled automatically via voice prompt ('{transcript}')",
        )

        # Attempt frame capture up to 5 times with short 0.2s pause to grab clear face embedding from live stream
        success = False
        msg = ""
        count = 0
        for _ in range(5):
            success, msg, count = self.capture_face_for_person(db, person.id)
            if success:
                break
            time.sleep(0.2)

        payload = {
            "person_id": person.id,
            "name": name,
            "relationship": relationship,
            "spoken_confirmation": (
                f"Nice to meet you {name}! I have enrolled you as a {relationship} in CogniSight memory."
            ),
        }

        return success, msg, payload

    def recognize_image_bytes(
        self, db: Session, image_bytes: bytes
    ) -> List[RecognitionResult]:
        """Runs face recognition on an uploaded image frame."""
        frame = self.decode_image_bytes(image_bytes)
        if frame is None or frame.size == 0:
            logger.warning("Invalid or corrupt image bytes passed to recognition service.")
            return []

        detected_faces = self.detector.detect_faces(frame)
        if not detected_faces:
            return []

        raw_faces = self.detector.detect_raw(frame)
        enrolled_candidates = PersonRepository.get_all_enrolled_embeddings(db)

        results = []
        for i, face in enumerate(detected_faces):
            bbox = face.bbox
            if i < len(raw_faces):
                emb = self.embedder.extract_embedding_from_raw_face(raw_faces[i])
                res = self.matcher.match_face(emb, enrolled_candidates, bbox=bbox)
            else:
                res = RecognitionResult(
                    bbox=bbox,
                    confidence=0.0,
                    recognized=False,
                    person_id=None,
                    name="Unknown Person",
                    similarity=0.0,
                )
            results.append(res)

        return results

    def enroll_person_from_image_bytes(
        self, db: Session, name: str, images_bytes_list: List[bytes]
    ) -> Tuple[bool, str, int, int]:
        """Enrolls a person using a list of uploaded image byte files."""
        frames = []
        for img_bytes in images_bytes_list:
            frame = self.decode_image_bytes(img_bytes)
            if frame is not None and frame.size > 0:
                frames.append(frame)

        if not frames:
            return False, "No valid image files provided for enrollment.", 0, 0

        return self.enrollment_service.enroll_from_frames(db, name, frames)
