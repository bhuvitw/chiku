import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.models.enums import Modality, StudyStatus


class StudyCreate(BaseModel):
    modality: Modality
    body_part: str | None = Field(default=None, max_length=64)


class StudyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    modality: Modality
    body_part: str | None
    status: StudyStatus
    created_at: datetime
    updated_at: datetime


class StudyImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    study_id: uuid.UUID
    original_filename: str | None
    content_type: str
    width: int
    height: int
    byte_size: int
    sha256: str
    created_at: datetime
    #: `stored_path` is deliberately absent: the server's filesystem layout is
    #: not the client's business (System Design §25).
