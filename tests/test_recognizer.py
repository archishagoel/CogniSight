import numpy as np
import pytest
from app.face.recognizer import FaceRecognizer


@pytest.fixture
def recognizer():
    return FaceRecognizer(recognition_interval=3, smoothing_window=5)


def test_recognizer_empty_frame(recognizer, db_session):
    """Test process_frame on empty frame returns empty results."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = recognizer.process_frame(blank_frame, db_session)
    assert isinstance(results, list)
    assert len(results) == 0


def test_draw_recognition_results(recognizer):
    """Test overlaying recognition boxes on image frames."""
    from app.face.schemas import BoundingBox, RecognitionResult

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    known_res = RecognitionResult(
        bbox=BoundingBox(x1=50, y1=50, x2=200, y2=200),
        confidence=0.92,
        recognized=True,
        person_id=1,
        name="John",
        similarity=0.92,
    )
    unknown_res = RecognitionResult(
        bbox=BoundingBox(x1=300, y1=50, x2=450, y2=200),
        confidence=0.31,
        recognized=False,
        person_id=None,
        name="Unknown Person",
        similarity=0.31,
    )

    annotated = recognizer.draw_recognition(frame, [known_res, unknown_res])
    assert annotated.shape == (480, 640, 3)
    assert not np.array_equal(frame, annotated)
