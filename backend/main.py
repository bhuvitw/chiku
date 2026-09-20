import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.v1.router import api_router
from backend.config import settings
from backend.errors import DEFAULT_STATUS, USER_FACING_MESSAGE, AppError, ErrorCode

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("chiku")

_STATUS_TO_CODE = {
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.UNAUTHORIZED,
    404: ErrorCode.NOT_FOUND,
    422: ErrorCode.VALIDATION_ERROR,
}


def _error_response(code: ErrorCode, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code.value, "message": message}},
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Assisted Musculoskeletal Imaging Platform",
        description=(
            "Research prototype. Not a medical device; not for clinical use. "
            "All model output is AI-generated and must not be read as diagnosis."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        code = ErrorCode.VALIDATION_ERROR
        return _error_response(code, USER_FACING_MESSAGE[code], DEFAULT_STATUS[code])

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        return _error_response(code, USER_FACING_MESSAGE[code], exc.status_code)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Log the detail, return the generic copy: an internal failure must never
        # surface as anything a user could read as a finding (System Design §26).
        logger.exception("unhandled error: %s", exc)
        code = ErrorCode.INTERNAL_ERROR
        return _error_response(code, USER_FACING_MESSAGE[code], DEFAULT_STATUS[code])

    app.include_router(api_router)
    return app


app = create_app()
