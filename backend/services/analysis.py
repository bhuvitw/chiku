"""Analysis orchestration: job lifecycle around one inference call.

The API contract is the point of this module. Whether the work runs inline or
on a Celery worker, the client does the same thing — POST /analyze, receive a
job, poll GET /status — so turning the worker on later changes no caller
(implementation-plan §2, System Design §17-18).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.errors import AppError, ErrorCode
from backend.inference import decide, get_predictor
from backend.models import Job, JobStatus, JobType, Prediction, Study, StudyStatus

logger = logging.getLogger(__name__)


def latest_image(db: Session, study: Study):
    from backend.models import StudyImage

    return db.scalar(
        select(StudyImage)
        .where(StudyImage.study_id == study.id)
        .order_by(StudyImage.created_at.desc())
    )


def start_analysis(db: Session, study: Study) -> Job:
    """Create the job, then dispatch it. Returns as soon as the job exists."""
    if latest_image(db, study) is None:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            message="Upload an image before requesting analysis.",
        )

    job = Job(study_id=study.id, job_type=JobType.ANALYZE, status=JobStatus.QUEUED, progress=0)
    db.add(job)
    study.status = StudyStatus.ANALYZING
    db.commit()
    db.refresh(job)

    if settings.async_jobs:
        from backend.workers.tasks import run_analysis_task

        run_analysis_task.delay(str(job.id))
    else:
        # Inline placeholder. The job is already terminal by the time the
        # client first polls, which the status endpoint handles like any
        # other state rather than special-casing.
        run_analysis(db, job.id)
        db.refresh(job)
    return job


def run_analysis(db: Session, job_id: uuid.UUID) -> Job:
    job = db.scalar(select(Job).where(Job.id == job_id))
    if job is None:
        raise AppError(ErrorCode.NOT_FOUND)
    study = db.scalar(select(Study).where(Study.id == job.study_id))
    if study is None:
        raise AppError(ErrorCode.NOT_FOUND)

    job.status = JobStatus.RUNNING
    job.progress = 10
    db.commit()

    try:
        image = latest_image(db, study)
        if image is None:
            raise AppError(ErrorCode.VALIDATION_ERROR, message="Study has no image to analyse.")

        path = Path(image.stored_path)
        if not path.exists():
            # The row says the file is there and it is not: a real failure, not
            # a modelling outcome. Fail loudly rather than analysing nothing.
            raise AppError(
                ErrorCode.PROCESSING_FAILED,
                message="The stored image could not be read.",
            )

        predictor = get_predictor()
        job.progress = 50
        db.commit()

        probability = predictor.probability(path)
        result = decide(
            probability,
            threshold=settings.decision_threshold,
            abstain_low=settings.abstention_low,
            abstain_high=settings.abstention_high,
            model_version=predictor.model_version,
        )

        db.add(
            Prediction(
                study_id=study.id,
                job_id=job.id,
                label=result.prediction,
                probability=result.probability,
                confidence=result.confidence,
                reason=result.reason,
                model_version=result.model_version,
                threshold=settings.decision_threshold,
                abstention_low=settings.abstention_low,
                abstention_high=settings.abstention_high,
            )
        )
        job.status = JobStatus.SUCCEEDED
        job.progress = 100
        job.completed_at = datetime.now(UTC)
        # An abstention is a completed analysis, not a failed one: the model
        # ran and its answer was "I cannot tell" (PRD FR-04).
        study.status = StudyStatus.COMPLETED
        db.commit()
    except AppError as error:
        db.rollback()
        _fail(db, job, study, error.code)
        raise
    except Exception:
        db.rollback()
        logger.exception("analysis job %s failed", job_id)
        _fail(db, job, study, ErrorCode.PROCESSING_FAILED)
        raise AppError(ErrorCode.PROCESSING_FAILED) from None

    db.refresh(job)
    return job


def _fail(db: Session, job: Job, study: Study, code: ErrorCode) -> None:
    job.status = JobStatus.FAILED
    job.error_code = code.value
    job.completed_at = datetime.now(UTC)
    # UNSUPPORTED is terminal-and-do-not-retry; FAILED invites a retry
    # (backend/models/enums.py).
    study.status = (
        StudyStatus.UNSUPPORTED if code is ErrorCode.UNSUPPORTED_MODALITY else StudyStatus.FAILED
    )
    db.commit()


def latest_prediction(db: Session, study: Study) -> Prediction | None:
    return db.scalar(
        select(Prediction)
        .where(Prediction.study_id == study.id)
        .order_by(Prediction.created_at.desc())
    )
