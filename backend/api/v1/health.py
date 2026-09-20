import redis
from fastapi import APIRouter
from sqlalchemy import text

from backend.api.deps import DbSession
from backend.config import settings
from backend.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only — deliberately touches no dependency."""
    return HealthResponse(status="ok", environment=settings.environment)


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(db: DbSession) -> ReadinessResponse:
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "unavailable"

    try:
        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2).ping()
        queue = "ok"
    except Exception:
        queue = "unavailable"

    overall = "ok" if database == "ok" and queue == "ok" else "degraded"
    return ReadinessResponse(status=overall, database=database, redis=queue)
