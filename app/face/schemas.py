from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Coordinates of a detected face bounding box [x1, y1, x2, y2]."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class DetectedFace(BaseModel):
    """Schema representing a detected face in an image frame."""

    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    landmarks: Optional[List[List[float]]] = None  # 5 keypoints [[x, y], ...]

    model_config = {"arbitrary_types_allowed": True}


class RecognitionResult(BaseModel):
    """Schema representing the recognition result for a detected face."""

    bbox: BoundingBox
    confidence: float
    recognized: bool
    person_id: Optional[int] = None
    name: Optional[str] = "Unknown Person"
    similarity: float = 0.0
