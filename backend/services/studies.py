import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.errors import AppError, ErrorCode
from backend.models import Study, StudyStatus, User
from backend.schemas.study import StudyCreate


def create_study(db: Session, payload: StudyCreate, user: User) -> Study:
    study = Study(
        user_id=user.id,
        modality=payload.modality,
        body_part=payload.body_part,
        status=StudyStatus.UPLOADED,
    )
    db.add(study)
    db.commit()
    db.refresh(study)
    return study


def list_studies(db: Session, user: User) -> list[Study]:
    return list(
        db.scalars(select(Study).where(Study.user_id == user.id).order_by(Study.created_at.desc()))
    )


def get_study(db: Session, study_id: uuid.UUID, user: User) -> Study:
    study = db.scalar(select(Study).where(Study.id == study_id, Study.user_id == user.id))
    if study is None:
        # Scoped to the requesting user: a study belonging to someone else is
        # indistinguishable from one that does not exist (System Design §25).
        raise AppError(ErrorCode.NOT_FOUND)
    return study
