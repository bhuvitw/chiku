"""Worker task skeleton.

Phase 0 ships only a connectivity probe. The analysis tasks land in Phase 2
(X-ray) and Phase 4 (MRI), behind the same `jobs` row contract.
"""

from backend.workers.celery_app import celery_app


@celery_app.task(name="chiku.ping")
def ping() -> str:
    return "pong"
