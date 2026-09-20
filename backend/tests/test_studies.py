import uuid

from fastapi.testclient import TestClient


def test_create_study_persists_and_returns_it(client: TestClient) -> None:
    response = client.post("/api/v1/studies", json={"modality": "xray", "body_part": "wrist"})
    assert response.status_code == 201

    body = response.json()
    assert body["modality"] == "xray"
    assert body["body_part"] == "wrist"
    assert body["status"] == "UPLOADED"

    fetched = client.get(f"/api/v1/studies/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]


def test_list_studies_returns_created_study(client: TestClient) -> None:
    client.post("/api/v1/studies", json={"modality": "mri", "body_part": "knee"})
    response = client.get("/api/v1/studies")
    assert response.status_code == 200
    assert [s["body_part"] for s in response.json()] == ["knee"]


def test_missing_study_uses_the_error_envelope(client: TestClient) -> None:
    response = client.get(f"/api/v1/studies/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "NOT_FOUND",
            "message": "The requested resource was not found.",
        }
    }


def test_unsupported_modality_is_rejected_in_the_error_envelope(client: TestClient) -> None:
    # An unsupported modality must not create a study in an undefined state.
    response = client.post("/api/v1/studies", json={"modality": "ultrasound"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    assert client.get("/api/v1/studies").json() == []
