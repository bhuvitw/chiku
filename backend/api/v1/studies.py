import uuid
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from backend.api.deps import CurrentUser, DbSession
from backend.errors import USER_FACING_MESSAGE, AppError, ErrorCode
from backend.models import JobStatus, PredictionLabel, StudyStatus
from backend.schemas.analysis import (
    JobRead,
    PredictionRead,
    ResultsRead,
    StudyStatusRead,
)
from backend.schemas.common import ErrorResponse
from backend.schemas.study import StudyCreate, StudyImageRead, StudyRead
from backend.services import analysis as analysis_service
from backend.services import studies as study_service
from backend.services import uploads as upload_service

router = APIRouter(prefix="/studies", tags=["studies"])

ERROR_RESPONSES: dict[int | str, dict] = {
    404: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


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


@router.post(
    "/{study_id}/upload",
    response_model=StudyImageRead,
    status_code=status.HTTP_201_CREATED,
    responses=ERROR_RESPONSES,
)
async def upload_image(
    study_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    file: Annotated[UploadFile, File()],
) -> StudyImageRead:
    study = study_service.get_study(db, study_id, user)
    record = upload_service.store_upload(
        db,
        study,
        data=await file.read(),
        original_filename=file.filename,
    )
    return StudyImageRead.model_validate(record)


@router.post(
    "/{study_id}/analyze",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERROR_RESPONSES,
)
def analyze_study(study_id: uuid.UUID, db: DbSession, user: CurrentUser) -> JobRead:
    """Accepted, not completed: 202 is honest whether the work ran inline or
    was handed to a worker, and the client polls /status either way."""
    study = study_service.get_study(db, study_id, user)
    job = analysis_service.start_analysis(db, study)
    return JobRead.model_validate(job)


@router.get(
    "/{study_id}/status",
    response_model=StudyStatusRead,
    responses={404: {"model": ErrorResponse}},
)
def study_status(study_id: uuid.UUID, db: DbSession, user: CurrentUser) -> StudyStatusRead:
    study = study_service.get_study(db, study_id, user)
    job = study.jobs[-1] if study.jobs else None
    error_code = job.error_code if job and job.status is JobStatus.FAILED else None
    return StudyStatusRead(
        study_id=study.id,
        status=study.status,
        progress=job.progress if job else 0,
        job=JobRead.model_validate(job) if job else None,
        error_code=error_code,
        message=USER_FACING_MESSAGE[ErrorCode(error_code)] if error_code else None,
    )


@router.get(
    "/{study_id}/results",
    response_model=ResultsRead,
    response_model_exclude_none=True,
    responses=ERROR_RESPONSES,
)
def study_results(study_id: uuid.UUID, db: DbSession, user: CurrentUser) -> ResultsRead:
    study = study_service.get_study(db, study_id, user)
    prediction = analysis_service.latest_prediction(db, study)

    if prediction is None:
        if study.status in (StudyStatus.FAILED, StudyStatus.UNSUPPORTED):
            job = study.jobs[-1] if study.jobs else None
            # Re-raise the failure through the same error contract the
            # original request used, so the copy is identical either way.
            code = (
                ErrorCode(job.error_code) if job and job.error_code else ErrorCode.PROCESSING_FAILED
            )
            raise AppError(code)
        raise AppError(
            ErrorCode.NOT_FOUND,
            message="No analysis result is available for this study yet.",
        )

    abstained = prediction.label is PredictionLabel.UNABLE_TO_ASSESS
    return ResultsRead(
        study_id=study.id,
        status=study.status,
        result=PredictionRead(
            prediction=prediction.label,
            model_version=prediction.model_version,
            confidence=None if abstained else prediction.confidence,
            reason=prediction.reason if abstained else None,
            # The abstention state reuses the LOW_CONFIDENCE copy, so the one
            # sentence the user sees is the same string the error contract
            # defines rather than a second, drifting copy of it.
            message=USER_FACING_MESSAGE[ErrorCode.LOW_CONFIDENCE] if abstained else None,
        ),
    )
