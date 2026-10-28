import argparse
from typing import List, Optional, Tuple
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.camera.camera import Camera
from app.database.database import SessionLocal, init_db
from app.database.repository import PersonRepository
from app.face.detector import FaceDetector
from app.face.embedding import FaceEmbedder
from app.face.schemas import DetectedFace
from app.utils.logging_config import logger


class FaceEnrollmentService:
    """Service handling intentional, quality-checked person enrollment."""

    MIN_FACE_SIZE: int = 80  # Minimum width and height of face box in pixels
    MIN_CONFIDENCE: float = 0.60  # Minimum detection confidence
    MIN_LAPLACIAN_VAR: float = 20.0  # Minimum laplacian variance (blur threshold)

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
    ):
        self.detector = detector if detector is not None else FaceDetector()
        self.embedder = embedder if embedder is not None else FaceEmbedder(self.detector)

    def validate_sample_quality(
        self, frame: np.ndarray, face: DetectedFace
    ) -> Tuple[bool, str]:
        """Performs quality checks on a detected face sample frame.

        Checks:
        1. Bounding box size (must be large enough to contain clear biometric features)
        2. Detection confidence score
        3. Image blur check using Laplacian variance

        Returns:
            Tuple[bool, str]: (Is valid sample, Reason string)
        """
        bbox = face.bbox
        if bbox.width < self.MIN_FACE_SIZE or bbox.height < self.MIN_FACE_SIZE:
            return False, f"Face too small ({bbox.width}x{bbox.height}px, min {self.MIN_FACE_SIZE}px)"

        if face.confidence < self.MIN_CONFIDENCE:
            return False, f"Low confidence ({face.confidence*100:.1f}%, min {self.MIN_CONFIDENCE*100:.1f}%)"

        # Crop face region for blur assessment
        h, w, _ = frame.shape
        x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
        x2, y2 = min(w, bbox.x2), min(h, bbox.y2)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return False, "Invalid crop region"

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap_var < self.MIN_LAPLACIAN_VAR:
            return False, f"Frame too blurry (blur metric {lap_var:.1f}, min {self.MIN_LAPLACIAN_VAR})"

        return True, "Quality sample accepted"

    def enroll_from_frames(
        self, db: Session, person_name: str, frames: List[np.ndarray]
    ) -> Tuple[bool, str, int, int]:
        """Programmatic enrollment function from a list of image frames.

        Returns:
            Tuple[bool, str, int, int]: (Success, Message, Total frames processed, Valid samples)
        """
        valid_embeddings = []
        total_frames = len(frames)

        for frame in frames:
            raw_faces = self.detector.detect_raw(frame)
            if not raw_faces:
                continue

            # Pick largest face
            primary_face = max(
                raw_faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            # Convert to DetectedFace for validation
            from app.face.schemas import BoundingBox
            b = primary_face.bbox.astype(int)
            det_face = DetectedFace(
                bbox=BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3])),
                confidence=float(primary_face.det_score),
            )

            is_valid, _ = self.validate_sample_quality(frame, det_face)
            if is_valid:
                emb = self.embedder.extract_embedding_from_raw_face(primary_face)
                if emb is not None:
                    valid_embeddings.append(emb)

        if not valid_embeddings:
            return False, "No quality face samples were captured.", total_frames, 0

        # Combine embeddings via vector averaging and L2 normalization
        mean_embedding = np.mean(valid_embeddings, axis=0)
        final_embedding = FaceEmbedder.normalize_embedding(mean_embedding)

        # Database enrollment
        person = PersonRepository.create_person(db, person_name)
        PersonRepository.add_face_embedding(db, person.id, final_embedding)

        msg = f"Successfully enrolled '{person_name}' with {len(valid_embeddings)} valid samples."
        logger.info(msg)
        return True, msg, total_frames, len(valid_embeddings)

    def enroll_interactive(
        self, person_name: str, target_samples: int = 10
    ) -> bool:
        """Interactive webcam enrollment workflow."""
        init_db()
        db = SessionLocal()

        logger.info(f"Starting interactive enrollment for '{person_name}'...")
        logger.info("Notice: CogniSight extracts and stores 512-D face embedding vectors.")

        camera = Camera()
        if not camera.start():
            logger.error("Could not open camera for enrollment.")
            db.close()
            return False

        collected_embeddings = []
        captured_attempts = 0

        try:
            while len(collected_embeddings) < target_samples:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    break

                output_frame = frame.copy()
                raw_faces = self.detector.detect_raw(frame)

                status_msg = "Look directly at the camera"
                status_color = (255, 255, 0)  # Yellow

                if raw_faces:
                    primary = max(
                        raw_faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    b = primary.bbox.astype(int)
                    from app.face.schemas import BoundingBox
                    det_face = DetectedFace(
                        bbox=BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3])),
                        confidence=float(primary.det_score),
                    )

                    is_valid, reason = self.validate_sample_quality(frame, det_face)
                    captured_attempts += 1

                    if is_valid:
                        emb = self.embedder.extract_embedding_from_raw_face(primary)
                        if emb is not None:
                            collected_embeddings.append(emb)
                            status_msg = f"Sample {len(collected_embeddings)}/{target_samples} ACCEPTED!"
                            status_color = (0, 255, 0)  # Green
                    else:
                        status_msg = f"Rejected: {reason}"
                        status_color = (0, 0, 255)  # Red

                    # Draw box
                    cv2.rectangle(
                        output_frame,
                        (det_face.bbox.x1, det_face.bbox.y1),
                        (det_face.bbox.x2, det_face.bbox.y2),
                        status_color,
                        2,
                    )

                # Render overlay bar
                cv2.rectangle(
                    output_frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), cv2.FILLED
                )
                cv2.putText(
                    output_frame,
                    f"ENROLLING: {person_name} | {status_msg}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("CogniSight - Person Enrollment", output_frame)
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Enrollment cancelled by user.")
                    return False

            if collected_embeddings:
                mean_emb = np.mean(collected_embeddings, axis=0)
                final_emb = FaceEmbedder.normalize_embedding(mean_emb)

                person = PersonRepository.create_person(db, person_name)
                PersonRepository.add_face_embedding(db, person.id, final_emb)

                print("\n" + "=" * 50)
                print(f"ENROLLMENT REPORT for '{person_name}'")
                print("=" * 50)
                print(f"Total Attempts: {captured_attempts}")
                print(f"Valid Samples Accepted: {len(collected_embeddings)}")
                print(f"Embedding Vector Dimension: 512-D float32")
                print(f"Status: SUCCESS (Person ID = {person.id})")
                print("=" * 50 + "\n")
                return True

            return False
        finally:
            camera.release()
            cv2.destroyAllWindows()
            db.close()


def main():
    parser = argparse.ArgumentParser(description="CogniSight Person Enrollment CLI")
    parser.add_argument("--name", type=str, required=True, help="Full name of person to enroll")
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of valid facial samples to capture"
    )
    args = parser.parse_args()

    service = FaceEnrollmentService()
    service.enroll_interactive(person_name=args.name, target_samples=args.samples)


if __name__ == "__main__":
    main()
