import uuid

from fastapi import APIRouter, status

from backend.api.deps import CurrentUser, DbSession
from backend.schemas.common import ErrorResponse
from backend.schemas.study import StudyCreate, StudyRead
from backend.services import studies as study_service

router = APIRouter(prefix="/studies", tags=["studies"])


@router.post("", response_model=StudyRead, status_code=status.HTTP_201_CREATED)
def create_study(payload: StudyCreate, db: DbSession, user: CurrentUser) -> StudyRead:
    study = study_service.create_study(db, payload, user)
    return StudyRead.model_validate(study)


@router.get("", response_model=list[StudyRead])
def list_studies(db: DbSession, user: CurrentUser) -> list[StudyRead]:
    return [StudyRead.model_validate(s) for s in study_service.list_studies(db, user)]


@router.get(
    "/{study_id}",
    response_model=StudyRead,
    responses={404: {"model": ErrorResponse}},
)
def get_study(study_id: uuid.UUID, db: DbSession, user: CurrentUser) -> StudyRead:
    return StudyRead.model_validate(study_service.get_study(db, study_id, user))
