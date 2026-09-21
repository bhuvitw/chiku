import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base
from backend.models.enums import PredictionLabel

if TYPE_CHECKING:
    from backend.models.study import Study


class Prediction(Base):
    """A model output for one study (PRD FR-04).

    `probability` is stored even when the result is `unable_to_assess`. The
    operating point and abstention band are configuration, not model weights,
    so a later recalibration has to be re-judged against the raw probability —
    storing only the label would make every past result unreinterpretable.

    `model_version` is mandatory and is never inferred at read time: a result
    means nothing without knowing which model produced it (System Design §20).
    """

    __tablename__ = "predictions"
    __table_args__ = (
        sa.CheckConstraint(
            "probability >= 0 AND probability <= 1", name="ck_predictions_probability_range"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid.uuid4)
    study_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("studies.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    label: Mapped[PredictionLabel] = mapped_column(
        sa.Enum(
            PredictionLabel,
            name="prediction_label",
            values_callable=lambda e: [m.value for m in e],
        )
    )
    probability: Mapped[float] = mapped_column(sa.Float)
    confidence: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    model_version: Mapped[str] = mapped_column(sa.String(128))
    #: The operating point this result was produced under, so a stored
    #: prediction stays interpretable after the thresholds move.
    threshold: Mapped[float] = mapped_column(sa.Float)
    abstention_low: Mapped[float] = mapped_column(sa.Float)
    abstention_high: Mapped[float] = mapped_column(sa.Float)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )

    study: Mapped["Study"] = relationship(back_populates="predictions")
