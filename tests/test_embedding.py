import numpy as np
import pytest
from app.face.embedding import FaceEmbedder


def test_normalize_embedding():
    """Test L2 vector normalization calculation."""
    raw_vector = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)  # Length = 5.0
    normalized = FaceEmbedder.normalize_embedding(raw_vector)

    # L2 norm should now be 1.0
    norm = np.linalg.norm(normalized)
    assert pytest.approx(norm, 1e-5) == 1.0
    assert normalized[0] == pytest.approx(0.6, 1e-5)
    assert normalized[1] == pytest.approx(0.8, 1e-5)


def test_normalize_zero_vector():
    """Test normalization safety against zero vectors to avoid zero division errors."""
    zero_vector = np.zeros(512, dtype=np.float32)
    normalized = FaceEmbedder.normalize_embedding(zero_vector)
    assert np.array_equal(zero_vector, normalized)


def test_generate_embedding_empty_frame():
    """Test graceful handling of empty frame input."""
    embedder = FaceEmbedder()
    empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = embedder.generate_embedding(empty_frame)
    assert emb is None
