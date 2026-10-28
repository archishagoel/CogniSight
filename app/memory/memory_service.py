from typing import List, Optional
from sqlalchemy.orm import Session

from app.database.models import InteractionLog, PersonProfile
from app.database.repository import PersonRepository
from app.utils.logging_config import logger


class MemoryService:
    """Person Memory Layer Service managing personalized memory, relationships, notes, and last-seen history."""

    def __init__(self):
        pass

    def get_memory_profile(self, db: Session, person_id: int) -> Optional[PersonProfile]:
        """Retrieves personalized memory profile for an enrolled person."""
        return PersonRepository.get_or_create_profile(db, person_id)

    def update_memory_profile(
        self,
        db: Session,
        person_id: int,
        relationship: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[PersonProfile]:
        """Updates relationship metadata or personal memory notes for a person."""
        return PersonRepository.update_profile(db, person_id, relationship, notes)

    def record_recognition_event(
        self,
        db: Session,
        person_name: str,
        recognized: bool,
        similarity: float,
        person_id: Optional[int] = None,
    ) -> InteractionLog:
        """Records a recognition event in the interaction log and updates last-seen timestamp."""
        if recognized and person_id is not None:
            PersonRepository.update_last_seen(db, person_id)

        log = PersonRepository.log_interaction(
            db=db,
            person_name=person_name,
            recognized=recognized,
            similarity=similarity,
            person_id=person_id,
        )
        return log

    def get_recent_interactions(
        self, db: Session, limit: int = 10
    ) -> List[InteractionLog]:
        """Retrieves recent system interaction logs."""
        return PersonRepository.get_recent_interactions(db, limit=limit)
