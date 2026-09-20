"""Error contract for the API (System Design §3 and §26).

Every error leaves the API in one shape:

    {"error": {"code": "UNSUPPORTED_MODALITY", "message": "..."}}

so the frontend can map a failure to the exact user-facing copy required by
System Design §26 instead of rendering a generic "something went wrong".
"""

import enum


class ErrorCode(enum.StrEnum):
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


#: User-facing copy. The first four strings are fixed verbatim by System Design
#: §26 — they are product copy, not debug text, and must not drift.
USER_FACING_MESSAGE: dict[ErrorCode, str] = {
    ErrorCode.UNSUPPORTED_MODALITY: "This imaging study is not currently supported.",
    ErrorCode.LOW_IMAGE_QUALITY: "Image quality is insufficient for reliable analysis.",
    ErrorCode.LOW_CONFIDENCE: "The model could not make a confident prediction.",
    ErrorCode.PROCESSING_FAILED: "Analysis failed. Please retry.",
    ErrorCode.VALIDATION_ERROR: "The request could not be processed as submitted.",
    ErrorCode.NOT_FOUND: "The requested resource was not found.",
    ErrorCode.UNAUTHORIZED: "Authentication is required to access this resource.",
    ErrorCode.INTERNAL_ERROR: "An unexpected error occurred.",
}

DEFAULT_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNSUPPORTED_MODALITY: 415,
    ErrorCode.LOW_IMAGE_QUALITY: 422,
    ErrorCode.LOW_CONFIDENCE: 200,
    ErrorCode.PROCESSING_FAILED: 500,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.INTERNAL_ERROR: 500,
}


class AppError(Exception):
    """Raise this instead of HTTPException so responses keep one error shape."""

    def __init__(
        self,
        code: ErrorCode,
        *,
        status_code: int | None = None,
        message: str | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code or DEFAULT_STATUS[code]
        self.message = message or USER_FACING_MESSAGE[code]
        super().__init__(self.message)
