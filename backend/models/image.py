import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base

if TYPE_CHECKING:
    from backend.models.study import Study


class StudyImage(Base):
    """One stored, de-identified image belonging to a study (System Design §4).

    `stored_path` is a server-generated name, never the client's filename: an
    uploaded name is both untrusted input and a known carrier of identifiers
    (see `ml.data.deid.scan_filename`). The original is kept only as
    `original_filename` for display, and is not used to build a path.
    """

    __tablename__ = "study_images"

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("studies.id", ondelete="CASCADE"), index=True
    )
    stored_path: Mapped[str] = mapped_column(sa.String(512))
    original_filename: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(sa.String(64))
    width: Mapped[int] = mapped_column(sa.Integer)
    height: Mapped[int] = mapped_column(sa.Integer)
    byte_size: Mapped[int] = mapped_column(sa.BigInteger)
    #: Content hash of the stored (post-de-identification) file, so a result
    #: can be tied to the exact bytes the model saw.
    sha256: Mapped[str] = mapped_column(sa.String(64), index=True)
    #: What the de-identification pass found and stripped. Empty list means the
    #: scan ran and was clean — not that it was skipped.
    deid_findings: Mapped[list] = mapped_column(sa.JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )

    study: Mapped["Study"] = relationship(back_populates="images")
