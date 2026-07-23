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


def test_optimizer_routes_return_manifests_and_dispatch_updates_status():
    manifests_response = client.get('/api/optimizer/manifests')
    assert manifests_response.status_code == 200
    manifests = manifests_response.json()
    assert isinstance(manifests, list)
    assert len(manifests) > 0

    first_manifest = manifests[0]
    dispatch_response = client.patch(f"/api/optimizer/manifests/{first_manifest['id']}/dispatch")
    assert dispatch_response.status_code == 200
    dispatched_payload = dispatch_response.json()
    assert dispatched_payload['manifest']['status'] == 'DISPATCHED'
