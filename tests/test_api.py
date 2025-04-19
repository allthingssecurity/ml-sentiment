from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_endpoint_positive():
    response = client.post("/predict", json={"text": "This is fantastic!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data and "score" in data
    assert isinstance(data["label"], str)
    assert isinstance(data["score"], float)


def test_predict_endpoint_invalid():
    # Missing 'text' field
    response = client.post("/predict", json={})
    assert response.status_code == 422
