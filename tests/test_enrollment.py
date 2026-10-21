import numpy as np
import pytest
from app.face.enroll import FaceEnrollmentService
from app.face.schemas import BoundingBox, DetectedFace


@pytest.fixture
def service():
    return FaceEnrollmentService()


def test_validate_sample_quality_too_small(service):
    """Test rejection of face bounding boxes that are too small."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    small_face = DetectedFace(
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),  # 40x40px (< 80px min)
        confidence=0.95,
    )

    is_valid, reason = service.validate_sample_quality(frame, small_face)
    assert is_valid is False
    assert "too small" in reason.lower()


def test_validate_sample_quality_low_confidence(service):
    """Test rejection of detections below confidence threshold."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    low_conf_face = DetectedFace(
        bbox=BoundingBox(x1=100, y1=100, x2=250, y2=250),  # 150x150px
        confidence=0.40,  # (< 0.60 min)
    )

    is_valid, reason = service.validate_sample_quality(frame, low_conf_face)
    assert is_valid is False
    assert "low confidence" in reason.lower()


def test_enroll_from_empty_frames(service, db_session):
    """Test programmatic enrollment when no valid faces are present."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    success, msg, total, valid = service.enroll_from_frames(
        db=db, person_name="Test User", frames=[blank_frame]
    )

    assert success is False
    assert valid == 0
    db.close()
