from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas import (
    EnrollmentUploadResponse,
    HealthResponse,
    MemoryProfileResponse,
    MemoryProfileUpdate,
    PersonCreate,
    PersonResponse,
    SpeakRequest,
    VoiceEnrollRequest,
)
from app.database.database import get_db
from app.database.repository import PersonRepository
from app.face.schemas import RecognitionResult
from app.memory.memory_service import MemoryService
from app.services.cognisight_pipeline import CogniSightPipeline
from app.services.recognition_service import RecognitionService
from app.voice.voice_assistant import VoiceAssistant

router = APIRouter()
recognition_service = RecognitionService()
memory_service = MemoryService()
voice_assistant = VoiceAssistant()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns application health status and configuration details."""
    return recognition_service.get_health_status()


@router.get("/context", tags=["Context Engine"])
def get_current_context():
    """Retrieves current synthesized context state from active perception pipeline."""
    state = CogniSightPipeline.get_latest_context()
    if state is None:
        return {
            "headline": "Scanning Environment...",
            "subtext": "No person detected in camera view.",
            "spoken_announcement": "Scanning environment. No person in view.",
            "person_id": None,
            "person_name": None,
            "relationship": None,
            "recognized": False,
        }
    return {
        "headline": state.headline,
        "subtext": state.subtext,
        "spoken_announcement": state.spoken_announcement,
        "person_id": state.person_id,
        "person_name": state.person_name,
        "relationship": state.relationship,
        "recognized": state.recognized,
    }


@router.get("/persons", response_model=List[PersonResponse], tags=["Person Management"])
def list_persons(db: Session = Depends(get_db)):
    """Retrieves all enrolled person profiles."""
    persons = recognition_service.get_all_persons(db)
    results = []
    for p in persons:
        prof = PersonRepository.get_or_create_profile(db, p.id)
        resp = PersonResponse(
            id=p.id,
            name=p.name,
            created_at=p.created_at,
            updated_at=p.updated_at,
            embeddings_count=len(p.embeddings) if p.embeddings else 0,
            profile=MemoryProfileResponse.model_validate(prof) if prof else None,
        )
        results.append(resp)
    return results


@router.get(
    "/persons/{person_id}", response_model=PersonResponse, tags=["Person Management"]
)
def get_person(person_id: int, db: Session = Depends(get_db)):
    """Retrieves a single person profile by ID."""
    person = recognition_service.get_person(db, person_id)
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with ID {person_id} not found.",
        )
    prof = PersonRepository.get_or_create_profile(db, person.id)
    return PersonResponse(
        id=person.id,
        name=person.name,
        created_at=person.created_at,
        updated_at=person.updated_at,
        embeddings_count=len(person.embeddings) if person.embeddings else 0,
        profile=MemoryProfileResponse.model_validate(prof) if prof else None,
    )


@router.post(
    "/persons",
    response_model=PersonResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Person Management"],
)
def create_person(payload: PersonCreate, db: Session = Depends(get_db)):
    """Creates a new person profile."""
    person = recognition_service.create_person(db, payload.name)
    prof = PersonRepository.get_or_create_profile(db, person.id)
    return PersonResponse(
        id=person.id,
        name=person.name,
        created_at=person.created_at,
        updated_at=person.updated_at,
        embeddings_count=len(person.embeddings) if person.embeddings else 0,
        profile=MemoryProfileResponse.model_validate(prof) if prof else None,
    )


@router.post("/persons/{person_id}/capture_face", tags=["Person Management"])
async def capture_person_face(
    person_id: int,
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Captures and stores face embedding for a person and announces enrollment via voice."""
    image_bytes = await file.read() if file else None
    success, msg, count = recognition_service.capture_face_for_person(
        db, person_id, image_bytes
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=msg,
        )

    # Announce enrollment success via Voice Assistant
    person = recognition_service.get_person(db, person_id)
    if person:
        prof = PersonRepository.get_or_create_profile(db, person.id)
        rel = prof.relationship if prof else ""
        voice_assistant.announce_enrollment(person.name, rel)

    return {"status": "success", "message": msg, "embeddings_count": count}


@router.post("/enrollment/voice_transcript", tags=["Enrollment"])
def voice_enroll_transcript(
    payload: VoiceEnrollRequest, db: Session = Depends(get_db)
):
    """Automatically enrolls an unknown person from spoken audio transcript."""
    success, msg, data = recognition_service.auto_voice_enroll_from_transcript(
        db, payload.transcript
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=msg,
        )

    voice_assistant.speak(data["spoken_confirmation"], force=True)
    return {"status": "success", "message": msg, "data": data}


@router.put(
    "/persons/{person_id}/memory",
    response_model=MemoryProfileResponse,
    tags=["Person Memory"],
)
def update_person_memory(
    person_id: int, payload: MemoryProfileUpdate, db: Session = Depends(get_db)
):
    """Updates personalized relationship and memory notes for an enrolled person."""
    person = recognition_service.get_person(db, person_id)
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with ID {person_id} not found.",
        )

    updated_prof = memory_service.update_memory_profile(
        db=db,
        person_id=person_id,
        relationship=payload.relationship,
        notes=payload.notes,
    )

    voice_assistant.speak(
        f"Updated memory profile for {person.name}. Relationship set to {updated_prof.relationship}.",
        force=True,
    )

    return MemoryProfileResponse.model_validate(updated_prof)


@router.delete("/persons/{person_id}", tags=["Person Management"])
def delete_person(person_id: int, db: Session = Depends(get_db)):
    """Deletes a person profile and their associated face embeddings & memory."""
    success = recognition_service.delete_person(db, person_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with ID {person_id} not found.",
        )
    return {"status": "success", "message": f"Person ID {person_id} successfully deleted."}


@router.post(
    "/recognition/predict",
    response_model=List[RecognitionResult],
    tags=["Recognition"],
)
async def recognize_face(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    """Performs face recognition on an uploaded image file (JPEG/PNG)."""
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image file uploaded.",
        )

    results = recognition_service.recognize_image_bytes(db, contents)
    return results


@router.post(
    "/enrollment/upload",
    response_model=EnrollmentUploadResponse,
    tags=["Enrollment"],
)
async def enroll_person_upload(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """Enrolls a person using a batch of uploaded image sample files."""
    if not name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Person name must not be empty.",
        )

    image_bytes_list = []
    for f in files:
        data = await f.read()
        if data:
            image_bytes_list.append(data)

    success, msg, total, valid = recognition_service.enroll_person_from_image_bytes(
        db, name.strip(), image_bytes_list
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=msg,
        )

    voice_assistant.announce_enrollment(name.strip())

    return EnrollmentUploadResponse(
        success=success,
        message=msg,
        total_frames_processed=total,
        valid_samples_accepted=valid,
    )


@router.post("/voice/speak", tags=["Voice Assistant"])
def speak_announcement(payload: SpeakRequest):
    """Triggers Text-To-Speech voice assistant audio playback."""
    success = voice_assistant.speak(payload.text, force=True)
    return {
        "status": "queued" if success else "failed",
        "spoken_text": payload.text,
    }
