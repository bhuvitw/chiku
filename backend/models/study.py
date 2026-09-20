import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base
from backend.models.enums import Modality, StudyStatus

if TYPE_CHECKING:
    from backend.models.job import Job
    from backend.models.user import User


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    modality: Mapped[Modality] = mapped_column(
        sa.Enum(Modality, name="modality", values_callable=lambda e: [m.value for m in e])
    )
    body_part: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    status: Mapped[StudyStatus] = mapped_column(
        sa.Enum(StudyStatus, name="study_status", values_callable=lambda e: [m.value for m in e]),
        default=StudyStatus.UPLOADED,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()
    )

    user: Mapped["User"] = relationship(back_populates="studies")
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="study", cascade="all, delete-orphan", order_by="Job.created_at"
    )
