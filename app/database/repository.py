from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session

from app.database.models import FaceEmbedding, InteractionLog, Person, PersonProfile
from app.utils.logging_config import logger


def utc_now():
    return datetime.now(timezone.utc)


class PersonRepository:
    """Repository class encapsulating database CRUD operations for Persons, Embeddings, Memory Profiles, and Logs."""

    @staticmethod
    def create_person(db: Session, name: str) -> Person:
        """Creates and persists a new Person record along with a default memory profile."""
        existing = db.query(Person).filter(Person.name == name.strip()).first()
        if existing:
            return existing

        person = Person(name=name.strip())
        db.add(person)
        db.commit()
        db.refresh(person)

        # Check if profile already exists before creating default
        existing_prof = (
            db.query(PersonProfile).filter(PersonProfile.person_id == person.id).first()
        )
        if not existing_prof:
            profile = PersonProfile(
                person_id=person.id, relationship="Known Contact", notes=""
            )
            db.add(profile)
            db.commit()

        logger.info(f"Created person record in database: ID={person.id}, Name='{person.name}'")
        return person

    @staticmethod
    def get_person_by_id(db: Session, person_id: int) -> Optional[Person]:
        """Retrieves a person by their database primary key ID."""
        return db.query(Person).filter(Person.id == person_id).first()

    @staticmethod
    def get_person_by_name(db: Session, name: str) -> Optional[Person]:
        """Retrieves a person by their unique name."""
        return db.query(Person).filter(Person.name == name.strip()).first()

    @staticmethod
    def get_all_persons(db: Session) -> List[Person]:
        """Retrieves list of all enrolled people."""
        return db.query(Person).order_by(Person.id.asc()).all()

    @staticmethod
    def delete_person(db: Session, person_id: int) -> bool:
        """Deletes a person and all their associated face embeddings & profile."""
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            return False

        db.delete(person)
        db.commit()
        logger.info(f"Deleted person record: ID={person_id}, Name='{person.name}'")
        return True

    @staticmethod
    def add_face_embedding(
        db: Session, person_id: int, vector: np.ndarray
    ) -> FaceEmbedding:
        """Adds a 512-D face embedding vector for an enrolled person."""
        embedding_record = FaceEmbedding(person_id=person_id)
        embedding_record.set_vector(vector)
        db.add(embedding_record)
        db.commit()
        db.refresh(embedding_record)
        logger.info(
            f"Added face embedding record for Person ID={person_id}, Embedding ID={embedding_record.id}"
        )
        return embedding_record

    @staticmethod
    def get_all_enrolled_embeddings(
        db: Session,
    ) -> List[Tuple[int, str, np.ndarray]]:
        """Retrieves all enrolled face embeddings in memory for fast similarity search."""
        records = (
            db.query(FaceEmbedding, Person)
            .join(Person, FaceEmbedding.person_id == Person.id)
            .all()
        )

        results = []
        for emb_record, person_record in records:
            vector = emb_record.get_vector()
            results.append((person_record.id, person_record.name, vector))

        return results

    @staticmethod
    def get_or_create_profile(db: Session, person_id: int) -> PersonProfile:
        """Retrieves or creates a memory profile for a person."""
        profile = (
            db.query(PersonProfile).filter(PersonProfile.person_id == person_id).first()
        )
        if not profile:
            profile = PersonProfile(
                person_id=person_id, relationship="Known Contact", notes=""
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)
        return profile

    @staticmethod
    def update_profile(
        db: Session,
        person_id: int,
        relationship: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[PersonProfile]:
        """Updates personalized relationship and memory notes for a person."""
        profile = PersonRepository.get_or_create_profile(db, person_id)
        if relationship is not None:
            profile.relationship = relationship.strip()
        if notes is not None:
            profile.notes = notes.strip()

        db.commit()
        db.refresh(profile)
        logger.info(f"Updated profile for Person ID={person_id}: Relationship='{profile.relationship}'")
        return profile

    @staticmethod
    def update_last_seen(db: Session, person_id: int) -> None:
        """Updates last_seen timestamp when a person is recognized."""
        profile = PersonRepository.get_or_create_profile(db, person_id)
        profile.last_seen = utc_now()
        db.commit()

    @staticmethod
    def log_interaction(
        db: Session,
        person_name: str,
        recognized: bool,
        similarity: float,
        person_id: Optional[int] = None,
    ) -> InteractionLog:
        """Logs an interaction event into database."""
        log = InteractionLog(
            person_id=person_id,
            person_name=person_name,
            recognized=recognized,
            similarity=similarity,
            timestamp=utc_now(),
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log

    @staticmethod
    def get_recent_interactions(
        db: Session, limit: int = 10
    ) -> List[InteractionLog]:
        """Retrieves recent interaction log events."""
        return (
            db.query(InteractionLog)
            .order_by(InteractionLog.id.desc())
            .limit(limit)
            .all()
        )
