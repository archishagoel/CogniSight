from dataclasses import dataclass
from typing import List, Optional
from sqlalchemy.orm import Session

from app.face.schemas import RecognitionResult
from app.memory.memory_service import MemoryService
from app.utils.logging_config import logger


@dataclass
class ContextState:
    """Dataclass representing synthesized real-time contextual state."""

    headline: str
    subtext: str
    spoken_announcement: str
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    relationship: Optional[str] = None
    recognized: bool = False


class ContextEngine:
    """CogniSight Context Engine.

    Synthesizes perception (face recognition predictions) with long-term person memory
    to produce human-understandable context summaries for visual display and voice assistance.
    """

    def __init__(self, memory_service: Optional[MemoryService] = None):
        self.memory_service = (
            memory_service if memory_service is not None else MemoryService()
        )

    def generate_context(
        self, db: Session, recognition_results: List[RecognitionResult]
    ) -> ContextState:
        """Generates ContextState by combining recognition results with memory profiles.

        Args:
            db: Database session.
            recognition_results: Predictions for faces detected in current frame.

        Returns:
            ContextState: Contextual headline, subtext, and spoken announcement sentence.
        """
        if not recognition_results:
            return ContextState(
                headline="Scanning Environment...",
                subtext="No person detected in camera view.",
                spoken_announcement="Scanning environment. No person in view.",
                recognized=False,
            )

        # Focus on primary / recognized face first
        recognized_matches = [r for r in recognition_results if r.recognized]
        target_res = recognized_matches[0] if recognized_matches else recognition_results[0]

        if target_res.recognized and target_res.person_id is not None:
            profile = self.memory_service.get_memory_profile(db, target_res.person_id)
            relationship = profile.relationship if profile and profile.relationship else "Known Contact"
            notes = profile.notes if profile and profile.notes else ""

            headline = f"{target_res.name} ({relationship})"
            subtext = f"Similarity: {target_res.similarity * 100:.1f}%"
            if notes:
                subtext += f" | Note: {notes}"

            spoken = f"{target_res.name}, your {relationship}, is in front of you."
            if notes:
                spoken += f" Remember: {notes}"

            # Record recognition event in memory interaction log
            self.memory_service.record_recognition_event(
                db=db,
                person_name=target_res.name,
                recognized=True,
                similarity=target_res.similarity,
                person_id=target_res.person_id,
            )

            return ContextState(
                headline=headline,
                subtext=subtext,
                spoken_announcement=spoken,
                person_id=target_res.person_id,
                person_name=target_res.name,
                relationship=relationship,
                recognized=True,
            )
        else:
            # Unknown Person
            headline = "Unknown Person Detected"
            subtext = f"Similarity below threshold ({target_res.similarity * 100:.1f}%)"
            spoken = "An unknown person is standing in front of you."

            # Log interaction
            self.memory_service.record_recognition_event(
                db=db,
                person_name="Unknown Person",
                recognized=False,
                similarity=target_res.similarity,
                person_id=None,
            )

            return ContextState(
                headline=headline,
                subtext=subtext,
                spoken_announcement=spoken,
                person_id=None,
                person_name="Unknown Person",
                relationship="Unknown",
                recognized=False,
            )
