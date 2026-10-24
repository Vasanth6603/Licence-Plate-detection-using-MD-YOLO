"""Integration tests for the FastAPI application."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "classes" in data
    assert "img_size" in data


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code in (400, 422)


def test_predict_non_image_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
