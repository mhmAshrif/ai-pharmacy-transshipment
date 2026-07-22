import pandas as pd
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_forecast_endpoint_returns_json():
    response = client.get('/api/forecast', params={'medicine': 'Amlodipine', 'district': 'Colombo'})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) > 0
