from fastapi import APIRouter

from backend.api.v1 import health, studies

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(studies.router)
