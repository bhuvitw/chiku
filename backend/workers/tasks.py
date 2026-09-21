"""Worker tasks.

The task is a thin wrapper: it owns a database session and nothing else.
All the analysis logic lives in `backend.services.analysis` so that the inline
path and the Celery path execute the same code, and a bug cannot exist in one
but not the other (implementation-plan §2).
"""

import uuid

from backend.database.session import SessionLocal
from backend.workers.celery_app import celery_app


@celery_app.task(name="chiku.ping")
def ping() -> str:
    return "pong"


@celery_app.task(name="chiku.analyze", bind=True, max_retries=0)
def run_analysis_task(self, job_id: str) -> str:  # noqa: ANN001, ARG001
    from backend.services.analysis import run_analysis

    with SessionLocal() as db:
        # run_analysis already records the failure on the job row before it
        # re-raises, so a crashed task still leaves a status the client can
        # poll rather than a job stuck in RUNNING for ever.
        job = run_analysis(db, uuid.UUID(job_id))
        return job.status.value
