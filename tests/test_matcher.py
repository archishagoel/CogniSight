import numpy as np
import pytest
from app.face.embedding import FaceEmbedder
from app.face.matcher import FaceMatcher


@pytest.fixture
def matcher():
    return FaceMatcher(threshold=0.50)


def test_cosine_similarity_identical_vector(matcher):
    """Identical normalized vectors must yield cosine similarity of 1.0."""
    vec = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
    sim = matcher.calculate_cosine_similarity(vec, vec)
    assert pytest.approx(sim, 1e-5) == 1.0


def test_cosine_similarity_orthogonal_vectors(matcher):
    """Orthogonal vectors must yield cosine similarity of 0.0."""
    v1 = np.zeros(512, dtype=np.float32)
    v2 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0

    sim = matcher.calculate_cosine_similarity(v1, v2)
    assert pytest.approx(sim, 1e-5) == 0.0


def test_match_known_person_above_threshold(matcher):
    """Matching a query vector close to an enrolled vector returns recognized person."""
    john_vec = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
    # Query vector with slight noise (high similarity ~0.98)
    query_vec = FaceEmbedder.normalize_embedding(john_vec + np.random.randn(512) * 0.02)

    candidates = [(1, "John", john_vec)]
    result = matcher.match_face(query_vec, candidates)

    assert result.recognized is True
    assert result.name == "John"
    assert result.person_id == 1
    assert result.similarity >= 0.50


def test_match_unknown_person_below_threshold(matcher):
    """Matching an un-enrolled face vector returns Unknown Person when below threshold."""
    john_vec = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
    unknown_vec = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))

    candidates = [(1, "John", john_vec)]
    result = matcher.match_face(unknown_vec, candidates)

    # Statistical orthogonality in 512 dimensions means dot product of two random vectors is ~0.0
    assert result.recognized is False
    assert result.name == "Unknown Person"
    assert result.person_id is None
    assert result.similarity < 0.50


def test_match_empty_candidates(matcher):
    """Matching against an empty database returns Unknown Person."""
    query_vec = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
    result = matcher.match_face(query_vec, [])

    assert result.recognized is False
    assert result.name == "Unknown Person"
    assert result.similarity == 0.0
