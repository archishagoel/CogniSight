import numpy as np
import pytest
from app.face.detector import FaceDetector
from app.face.schemas import BoundingBox, DetectedFace


@pytest.fixture(scope="module")
def detector():
    """Module-level fixture for FaceDetector to avoid reloading models in every test."""
    return FaceDetector()


def test_detector_empty_frame(detector):
    """Verify that detector gracefully handles empty/black frames without crashing."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect_faces(blank_frame)
    assert isinstance(faces, list)
    assert len(faces) == 0


def test_detector_invalid_input(detector):
    """Verify that detector handles invalid or empty numpy array gracefully."""
    empty_frame = np.array([], dtype=np.uint8)
    faces = detector.detect_faces(empty_frame)
    assert faces == []


def test_draw_detections(detector):
    """Verify that draw_detections properly overlays bounding boxes on image frames."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_face = DetectedFace(
        bbox=BoundingBox(x1=50, y1=50, x2=200, y2=200),
        confidence=0.95,
        landmarks=[[100, 100], [150, 100], [125, 125], [105, 150], [145, 150]],
    )

    annotated_frame = detector.draw_detections(frame, [fake_face])
    assert annotated_frame.shape == (480, 640, 3)
    # The output image should have modified pixel values where box was drawn
    assert not np.array_equal(frame, annotated_frame)
