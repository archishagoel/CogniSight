from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from app.face.schemas import BoundingBox, RecognitionResult


class PersonCreate(BaseModel):
    """Request DTO for creating a new person profile."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "John"},
    )


class MemoryProfileUpdate(BaseModel):
    """Request DTO for updating a person's relationship and memory notes."""

    relationship: Optional[str] = Field(
        None, json_schema_extra={"example": "Brother / Caregiver"}
    )
    notes: Optional[str] = Field(
        None, json_schema_extra={"example": "Loves tea with 2 sugars."}
    )


class MemoryProfileResponse(BaseModel):
    """Response DTO for person memory profile."""

    id: int
    person_id: int
    relationship: Optional[str] = "Known Contact"
    notes: Optional[str] = ""
    last_seen: datetime

    model_config = {"from_attributes": True}


class PersonResponse(BaseModel):
    """Response DTO representing an enrolled person."""

    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    embeddings_count: int = 0
    profile: Optional[MemoryProfileResponse] = None

    model_config = {"from_attributes": True}


class EnrollmentUploadResponse(BaseModel):
    """Response DTO for image-based batch enrollment."""

    success: bool
    message: str
    total_frames_processed: int
    valid_samples_accepted: int


class VoiceEnrollRequest(BaseModel):
    """Request DTO for audio transcript auto-enrollment."""

    transcript: str = Field(
        ..., json_schema_extra={"example": "My name is Alex and I am a Caregiver"}
    )


class SpeakRequest(BaseModel):
    """Request DTO for TTS voice assistant announcement."""

    text: str = Field(
        ..., json_schema_extra={"example": "John is standing in front of you."}
    )


class HealthResponse(BaseModel):
    """Response DTO for system health check."""

    status: str
    service: str
    version: str
    model: str
    threshold: float
