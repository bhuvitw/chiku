"""phase 2: study_images and predictions

Two more of the System Design §15 tables, added by the phase that first uses
them rather than upfront. Series, structures, annotations, meshes, reports and
findings still wait for Phase 4-6.

Revision ID: 0002_images_predictions
Revises: 0001_initial
Create Date: 2026-09-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_images_predictions"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "study_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("stored_path", sa.String(length=512), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("deid_findings", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_study_images_sha256"), "study_images", ["sha256"], unique=False)
    op.create_index(op.f("ix_study_images_study_id"), "study_images", ["study_id"], unique=False)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column(
            "label",
            sa.Enum(
                "possible_fracture",
                "no_fracture",
                "unable_to_assess",
                name="prediction_label",
            ),
            nullable=False,
        ),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("reason", sa.String(length=64), nullable=True),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("abstention_low", sa.Float(), nullable=False),
        sa.Column("abstention_high", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "probability >= 0 AND probability <= 1", name="ck_predictions_probability_range"
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_study_id"), "predictions", ["study_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_study_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_study_images_study_id"), table_name="study_images")
    op.drop_index(op.f("ix_study_images_sha256"), table_name="study_images")
    op.drop_table("study_images")

    # Postgres keeps enum types after their tables are gone.
    op.execute(sa.text("DROP TYPE IF EXISTS prediction_label"))
