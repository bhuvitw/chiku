import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base
from backend.models.enums import JobStatus, JobType

if TYPE_CHECKING:
    from backend.models.study import Study


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_jobs_progress_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("studies.id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[JobType] = mapped_column(
        sa.Enum(JobType, name="job_type", values_callable=lambda e: [m.value for m in e])
    )
    status: Mapped[JobStatus] = mapped_column(
        sa.Enum(JobStatus, name="job_status", values_callable=lambda e: [m.value for m in e]),
        default=JobStatus.QUEUED,
        index=True,
    )
    progress: Mapped[int] = mapped_column(sa.Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)

    study: Mapped["Study"] = relationship(back_populates="jobs")
