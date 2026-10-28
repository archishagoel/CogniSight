import numpy as np
import pytest
from app.context.context_engine import ContextEngine
from app.database.repository import PersonRepository
from app.face.schemas import BoundingBox, RecognitionResult
from app.memory.memory_service import MemoryService
from app.voice.voice_assistant import VoiceAssistant


def test_memory_service_profile_update(db_session):
    """Test updating relationship and memory notes for a person."""
    memory_service = MemoryService()
    person = PersonRepository.create_person(db_session, "John")

    updated = memory_service.update_memory_profile(
        db=db_session,
        person_id=person.id,
        relationship="Brother / Caregiver",
        notes="Loves tea with 2 sugars.",
    )

    assert updated is not None
    assert updated.relationship == "Brother / Caregiver"
    assert updated.notes == "Loves tea with 2 sugars."

    # Verify retrieval
    prof = memory_service.get_memory_profile(db_session, person.id)
    assert prof.relationship == "Brother / Caregiver"


def test_context_engine_known_person(db_session):
    """Test ContextEngine generating headline, subtext, and spoken announcement for known person."""
    memory_service = MemoryService()
    context_engine = ContextEngine(memory_service)

    person = PersonRepository.create_person(db_session, "Sarah")
    memory_service.update_memory_profile(
        db=db_session,
        person_id=person.id,
        relationship="Doctor",
        notes="Dr. Sarah, Cardiologist",
    )

    rec_result = RecognitionResult(
        bbox=BoundingBox(x1=10, y1=10, x2=100, y2=100),
        confidence=0.92,
        recognized=True,
        person_id=person.id,
        name="Sarah",
        similarity=0.92,
    )

    context_state = context_engine.generate_context(db_session, [rec_result])

    assert context_state.recognized is True
    assert "Sarah" in context_state.headline
    assert "Doctor" in context_state.headline
    assert "Dr. Sarah" in context_state.subtext
    assert "Sarah, your Doctor, is in front of you" in context_state.spoken_announcement


def test_context_engine_unknown_person(db_session):
    """Test ContextEngine generating headline for unknown person."""
    memory_service = MemoryService()
    context_engine = ContextEngine(memory_service)

    rec_result = RecognitionResult(
        bbox=BoundingBox(x1=10, y1=10, x2=100, y2=100),
        confidence=0.30,
        recognized=False,
        person_id=None,
        name="Unknown Person",
        similarity=0.30,
    )

    context_state = context_engine.generate_context(db_session, [rec_result])

    assert context_state.recognized is False
    assert "Unknown Person" in context_state.headline
    assert "unknown person is standing" in context_state.spoken_announcement.lower()


def test_voice_assistant_cooldown():
    """Test VoiceAssistant rate-limiting cooldown logic."""
    va = VoiceAssistant(cooldown_seconds=5.0, enabled=False)

    # First attempt should succeed
    res1 = va.speak("Hello world")
    assert res1 is True

    # Immediate duplicate attempt should be throttled by cooldown
    res2 = va.speak("Hello world")
    assert res2 is False

    # Force speak should bypass cooldown
    res3 = va.speak("Hello world", force=True)
    assert res3 is True

    va.stop()
