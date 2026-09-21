import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from backend.models.enums import JobStatus, JobType, PredictionLabel, StudyStatus

#: PRD §12. Carried on every result the API returns, so a client cannot render
#: a prediction without also having been handed the disclaimer.
DISCLAIMER = (
    "Research and educational use only. This is not a medical device and its "
    "output is not a diagnosis."
)


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    study_id: uuid.UUID
    job_type: JobType
    status: JobStatus
    progress: int
    error_code: str | None
    created_at: datetime
    completed_at: datetime | None


class StudyStatusRead(BaseModel):
    """What the processing view polls (System Design §18)."""

    study_id: uuid.UUID
    status: StudyStatus
    progress: int
    job: JobRead | None = None
    error_code: str | None = None
    message: str | None = None


class PredictionRead(BaseModel):
    """PRD FR-04.

    `confidence` and `reason` are mutually exclusive by state, and the route
    serialises with `exclude_none`, so an abstention carries no confidence
    field at all rather than a null one that a client might render as 0.
    """

    prediction: PredictionLabel
    model_version: str
    confidence: float | None = None
    reason: str | None = None
    message: str | None = None


class ResultsRead(BaseModel):
    study_id: uuid.UUID
    status: StudyStatus
    result: PredictionRead
    disclaimer: str = DISCLAIMER
