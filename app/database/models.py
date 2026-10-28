from datetime import datetime, timezone
import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import relationship as orm_relationship

from app.database.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Person(Base):
    """SQLAlchemy model representing an enrolled person in CogniSight."""

    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # 1-to-many relationship with FaceEmbedding
    embeddings = orm_relationship(
        "FaceEmbedding", back_populates="person", cascade="all, delete-orphan"
    )

    # 1-to-1 relationship with PersonProfile
    profile = orm_relationship(
        "PersonProfile",
        back_populates="person",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Person(id={self.id}, name='{self.name}')>"


class FaceEmbedding(Base):
    """SQLAlchemy model representing a 512-D face embedding vector for an enrolled person."""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)  # Binary serialized float32 numpy vector
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    person = orm_relationship("Person", back_populates="embeddings")

    def set_vector(self, vector: np.ndarray) -> None:
        """Serializes numpy float32 embedding vector into binary storage."""
        vec_float32 = vector.astype(np.float32)
        self.embedding = vec_float32.tobytes()

    def get_vector(self) -> np.ndarray:
        """Deserializes binary storage into numpy float32 embedding vector."""
        return np.frombuffer(self.embedding, dtype=np.float32)

    def __repr__(self):
        return f"<FaceEmbedding(id={self.id}, person_id={self.person_id})>"


class PersonProfile(Base):
    """SQLAlchemy model storing personalized memory, relationship, and notes for a person."""

    __tablename__ = "person_profiles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(
        Integer, ForeignKey("persons.id"), nullable=False, unique=True, index=True
    )
    relationship = Column(String(100), nullable=True, default="Known Contact")
    notes = Column(Text, nullable=True, default="")
    last_seen = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    person = orm_relationship("Person", back_populates="profile")

    def __repr__(self):
        return f"<PersonProfile(person_id={self.person_id}, relationship='{self.relationship}')>"


class InteractionLog(Base):
    """SQLAlchemy model logging face recognition interaction events for context & memory analytics."""

    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True, index=True)
    person_name = Column(String(100), nullable=False)
    recognized = Column(Boolean, nullable=False, default=False)
    similarity = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    def __repr__(self):
        return f"<InteractionLog(id={self.id}, name='{self.person_name}', recognized={self.recognized})>"
