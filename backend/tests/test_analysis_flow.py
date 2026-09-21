"""Phase 2 exit criterion: upload → analyse → result, end to end.

implementation-plan §2 is explicit that the failure paths are part of the
criterion, not an extra — so the abstention path and each error code are
exercised here rather than assumed to work because the happy path does.
"""

import io
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from backend.config import settings
from backend.errors import USER_FACING_MESSAGE, ErrorCode


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "inference_backend", "stub")
    monkeypatch.setattr(settings, "stub_probability", None)
    monkeypatch.setattr(settings, "async_jobs", False)


def png_bytes(width: int = 512, height: int = 512, metadata: dict | None = None) -> bytes:
    image = Image.new("L", (width, height), color=128)
    buffer = io.BytesIO()
    info = PngInfo()
    for key, value in (metadata or {}).items():
        info.add_text(key, value)
    image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def create_study(client: TestClient) -> str:
    response = client.post("/api/v1/studies", json={"modality": "xray", "body_part": "wrist"})
    assert response.status_code == 201
    return response.json()["id"]


def upload(client: TestClient, study_id: str, data: bytes, name: str = "wrist.png"):
    return client.post(
        f"/api/v1/studies/{study_id}/upload",
        files={"file": (name, data, "image/png")},
    )


def test_upload_analyze_results_round_trip(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "stub_probability", 0.97)
    study_id = create_study(client)

    uploaded = upload(client, study_id, png_bytes())
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["width"] == 512

    analyze = client.post(f"/api/v1/studies/{study_id}/analyze")
    assert analyze.status_code == 202
    assert analyze.json()["job_type"] == "analyze"

    status = client.get(f"/api/v1/studies/{study_id}/status").json()
    assert status["status"] == "COMPLETED"
    assert status["progress"] == 100

    results = client.get(f"/api/v1/studies/{study_id}/results")
    assert results.status_code == 200
    body = results.json()
    assert body["result"]["prediction"] == "possible_fracture"
    assert body["result"]["confidence"] == pytest.approx(0.97)
    assert body["result"]["model_version"] == "stub:v0"
    # PRD §12: a client cannot render a prediction without the disclaimer.
    assert "not a diagnosis" in body["disclaimer"]


def test_confident_negative_reports_confidence_in_its_own_class(client: TestClient, monkeypatch):
    """A 0.02 fracture probability is a *confident* no-fracture call, and the
    number shown to the user must reflect that rather than read as 2%."""
    monkeypatch.setattr(settings, "stub_probability", 0.02)
    study_id = create_study(client)
    upload(client, study_id, png_bytes())
    client.post(f"/api/v1/studies/{study_id}/analyze")

    result = client.get(f"/api/v1/studies/{study_id}/results").json()["result"]
    assert result["prediction"] == "no_fracture"
    assert result["confidence"] == pytest.approx(0.98)


def test_abstention_path_returns_unable_to_assess(client: TestClient, monkeypatch):
    """PRD FR-04: inside the band the product says so instead of guessing."""
    monkeypatch.setattr(settings, "stub_probability", 0.5)
    study_id = create_study(client)
    upload(client, study_id, png_bytes())
    client.post(f"/api/v1/studies/{study_id}/analyze")

    result = client.get(f"/api/v1/studies/{study_id}/results").json()["result"]
    assert result["prediction"] == "unable_to_assess"
    assert result["reason"] == "confidence_below_threshold"
    assert result["message"] == USER_FACING_MESSAGE[ErrorCode.LOW_CONFIDENCE]
    # Absent, not null: a null would render as 0% in a client that trusts the key.
    assert "confidence" not in result


def test_abstention_is_a_completed_study_not_a_failed_one(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "stub_probability", 0.5)
    study_id = create_study(client)
    upload(client, study_id, png_bytes())
    client.post(f"/api/v1/studies/{study_id}/analyze")

    status = client.get(f"/api/v1/studies/{study_id}/status").json()
    assert status["status"] == "COMPLETED"
    assert status["error_code"] is None


def test_dicom_is_rejected_explicitly_not_mishandled(client: TestClient):
    study_id = create_study(client)
    # A DICOM preamble: 128 bytes then the "DICM" magic.
    dicom = b"\0" * 128 + b"DICM" + b"\0" * 64

    response = client.post(
        f"/api/v1/studies/{study_id}/upload",
        files={"file": ("study.dcm", dicom, "application/dicom")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MODALITY"


def test_tiny_image_is_low_image_quality(client: TestClient):
    study_id = create_study(client)
    response = upload(client, study_id, png_bytes(width=32, height=32))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "LOW_IMAGE_QUALITY"


def test_non_image_upload_is_unsupported(client: TestClient):
    study_id = create_study(client)
    response = upload(client, study_id, b"this is not an image at all", name="notes.txt")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MODALITY"


def test_declared_content_type_is_not_trusted(client: TestClient):
    """The header says PNG; the bytes are a DICOM. The bytes win."""
    study_id = create_study(client)
    dicom = b"\0" * 128 + b"DICM" + b"\0" * 64
    response = client.post(
        f"/api/v1/studies/{study_id}/upload",
        files={"file": ("actually_dicom.png", dicom, "image/png")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MODALITY"


def test_analyze_without_an_image_is_rejected(client: TestClient):
    study_id = create_study(client)
    response = client.post(f"/api/v1/studies/{study_id}/analyze")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_results_before_analysis_is_not_found(client: TestClient):
    study_id = create_study(client)
    upload(client, study_id, png_bytes())
    response = client.get(f"/api/v1/studies/{study_id}/results")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_unknown_study_is_not_found(client: TestClient):
    response = client.get(f"/api/v1/studies/{uuid.uuid4()}/status")
    assert response.status_code == 404


def test_upload_strips_metadata_and_records_what_it_found(client: TestClient, db_session):
    """System Design §4: verify, don't trust — and store the evidence."""
    from backend.models import StudyImage

    study_id = create_study(client)
    payload = png_bytes(metadata={"Artist": "Dr Jane Doe", "Comment": "MRN 123456789012"})
    response = upload(client, study_id, payload, name="patient_jane_2019-04-02.png")
    assert response.status_code == 201

    record = db_session.get(StudyImage, uuid.UUID(response.json()["id"]))
    kinds = {finding["kind"] for finding in record.deid_findings}
    assert "metadata_text" in kinds
    assert any(kind.startswith("filename_") for kind in kinds)

    # The stored file carries none of it.
    with Image.open(record.stored_path) as stored:
        assert not any(key in (stored.info or {}) for key in ("Artist", "Comment"))


def test_stored_path_is_server_generated(client: TestClient, db_session):
    """An uploaded filename is untrusted input and an identifier carrier; it
    must not appear in the path the server writes to."""
    from backend.models import StudyImage

    study_id = create_study(client)
    response = upload(client, study_id, png_bytes(), name="../../patient_jane.png")
    assert response.status_code == 201

    record = db_session.get(StudyImage, uuid.UUID(response.json()["id"]))
    assert "patient_jane" not in record.stored_path
    assert ".." not in record.stored_path
    assert Path(record.stored_path).is_relative_to(settings.storage_dir)
    # Kept for display only.
    assert record.original_filename == "../../patient_jane.png"


def test_response_never_exposes_the_storage_path(client: TestClient):
    study_id = create_study(client)
    response = upload(client, study_id, png_bytes())
    assert "stored_path" not in response.json()
