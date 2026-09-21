"""Ingestion: format detection → validation → de-identification → storage.

This is System Design §4 in order, and the order matters: nothing is stored
before it has been identified and validated, and nothing reaches the model
before its metadata has been stripped.

Phase 2 supports PNG and JPEG. DICOM is detected explicitly and rejected with
`UNSUPPORTED_MODALITY`, because the one thing implementation-plan §2 rules out
is handling it silently and wrongly.
"""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.errors import AppError, ErrorCode
from backend.models import Study, StudyImage
from ml.data.deid import scan_filename, scan_image, strip_metadata

#: A DICOM file carries "DICM" at byte 128. Cheaper and far more reliable than
#: trusting a client-supplied content type or a .dcm extension.
_DICOM_MAGIC_OFFSET = 128
_DICOM_MAGIC = b"DICM"

_PIL_FORMAT_TO_TYPE = {"PNG": "image/png", "JPEG": "image/jpeg"}


def _is_dicom(data: bytes) -> bool:
    return data[_DICOM_MAGIC_OFFSET : _DICOM_MAGIC_OFFSET + 4] == _DICOM_MAGIC


def _detect_format(path: Path) -> str:
    """Identify the image from its bytes, never from the declared content type.

    A client can send anything in the `Content-Type` header; what the decoder
    actually sees is the only thing the rest of the pipeline can rely on.
    """
    try:
        with Image.open(path) as image:
            return image.format or ""
    except UnidentifiedImageError:
        raise AppError(ErrorCode.UNSUPPORTED_MODALITY) from None


def store_upload(
    db: Session,
    study: Study,
    *,
    data: bytes,
    original_filename: str | None,
) -> StudyImage:
    if not data:
        raise AppError(ErrorCode.VALIDATION_ERROR, message="The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            ErrorCode.VALIDATION_ERROR,
            status_code=413,
            message="The uploaded file is larger than the maximum supported size.",
        )
    if _is_dicom(data):
        raise AppError(
            ErrorCode.UNSUPPORTED_MODALITY,
            message=(
                "DICOM is not supported yet. Export the image as PNG or JPEG and upload that."
            ),
        )

    with tempfile.TemporaryDirectory() as workspace:
        incoming = Path(workspace) / "incoming"
        incoming.write_bytes(data)

        image_format = _detect_format(incoming)
        if image_format not in _PIL_FORMAT_TO_TYPE:
            raise AppError(ErrorCode.UNSUPPORTED_MODALITY)

        with Image.open(incoming) as image:
            width, height = image.size
        if min(width, height) < settings.min_image_pixels:
            raise AppError(
                ErrorCode.LOW_IMAGE_QUALITY,
                message=(
                    f"Image quality is insufficient for reliable analysis: "
                    f"{width}x{height} is below the {settings.min_image_pixels}px minimum."
                ),
            )

        # Scan before stripping, so the record says what was actually present.
        findings = [asdict(f) for f in scan_image(incoming)]
        if original_filename:
            findings += [asdict(f) for f in scan_filename(Path(original_filename))]

        # Server-generated path: the client's filename is untrusted input and a
        # known identifier carrier, so it is stored for display and never used
        # to build a path.
        destination = settings.storage_dir / str(study.id) / f"{uuid.uuid4().hex}.png"
        strip_metadata(incoming, destination)

    stored = destination.read_bytes()
    record = StudyImage(
        study_id=study.id,
        stored_path=str(destination),
        original_filename=(original_filename or None),
        content_type=_PIL_FORMAT_TO_TYPE[image_format],
        width=width,
        height=height,
        byte_size=len(stored),
        sha256=hashlib.sha256(stored).hexdigest(),
        # Paths are made relative to the store so the record does not leak the
        # server's directory layout to anything that reads it back.
        deid_findings=[{**f, "path": Path(f["path"]).name} for f in findings],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
