import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.database.repository import PersonRepository


@pytest.fixture
def db_session():
    """Fixture providing an isolated in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_create_and_get_person(db_session):
    """Test creating a person and retrieving by ID/name."""
    person = PersonRepository.create_person(db_session, "Alice")
    assert person.id is not None
    assert person.name == "Alice"

    fetched_id = PersonRepository.get_person_by_id(db_session, person.id)
    assert fetched_id is not None
    assert fetched_id.name == "Alice"

    fetched_name = PersonRepository.get_person_by_name(db_session, "Alice")
    assert fetched_name is not None
    assert fetched_name.id == person.id


def test_add_and_retrieve_embedding(db_session):
    """Test embedding vector serialization and deserialization accuracy."""
    person = PersonRepository.create_person(db_session, "Bob")
    dummy_vector = np.random.randn(512).astype(np.float32)

    emb_record = PersonRepository.add_face_embedding(db_session, person.id, dummy_vector)
    assert emb_record.id is not None

    enrolled = PersonRepository.get_all_enrolled_embeddings(db_session)
    assert len(enrolled) == 1
    p_id, p_name, retrieved_vec = enrolled[0]

    assert p_id == person.id
    assert p_name == "Bob"
    assert retrieved_vec.shape == (512,)
    assert np.allclose(retrieved_vec, dummy_vector, atol=1e-6)


def test_delete_person_cascade(db_session):
    """Test person deletion with cascade deletion of associated embeddings."""
    person = PersonRepository.create_person(db_session, "Charlie")
    dummy_vector = np.zeros(512, dtype=np.float32)
    PersonRepository.add_face_embedding(db_session, person.id, dummy_vector)

    success = PersonRepository.delete_person(db_session, person.id)
    assert success is True

    enrolled = PersonRepository.get_all_enrolled_embeddings(db_session)
    assert len(enrolled) == 0
