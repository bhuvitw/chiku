"""initial schema: users, studies, jobs

Phase 0 creates only the three tables the scaffolding needs. The remaining
tables from System Design §15 (series, images, models, predictions, structures,
annotations, meshes, reports, findings) land in the phase that first uses them.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-20

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column(
            "role",
            sa.Enum("student", "clinician", "admin", name="user_role"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "studies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("modality", sa.Enum("xray", "mri", name="modality"), nullable=False),
        sa.Column("body_part", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "UPLOADED",
                "VALIDATING",
                "PREPROCESSING",
                "ANALYZING",
                "GENERATING_3D",
                "COMPLETED",
                "FAILED",
                "UNSUPPORTED",
                name="study_status",
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_studies_user_id"), "studies", ["user_id"])
    op.create_index(op.f("ix_studies_status"), "studies", ["status"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("study_id", sa.Uuid(), nullable=False),
        sa.Column(
            "job_type",
            sa.Enum("analyze", "generate_3d", name="job_type"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("QUEUED", "RUNNING", "SUCCEEDED", "FAILED", name="job_status"),
            nullable=False,
        ),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_jobs_progress_range"),
        sa.ForeignKeyConstraint(["study_id"], ["studies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_study_id"), "jobs", ["study_id"])
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("studies")
    op.drop_table("users")

    # Postgres keeps enum types after their tables are gone.
    for enum_name in ("job_status", "job_type", "study_status", "modality", "user_role"):
        op.execute(sa.text(f"DROP TYPE IF EXISTS {enum_name}"))
