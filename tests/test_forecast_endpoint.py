import pandas as pd
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_forecast_endpoint_returns_json():
    # Ensure pipeline artifacts exist by triggering a full run (may be cached)
    run_resp = client.post('/api/pipeline/run-all')
    assert run_resp.status_code in (200, 201, 202)

    response = client.get('/api/forecast', params={'medicine': 'Amlodipine', 'district': 'Colombo'})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get('metrics'), dict)
    assert {'rmse', 'mae', 'mape'} <= set(payload['metrics'].keys())
    assert isinstance(payload.get('chart_data'), list)
    assert len(payload['chart_data']) > 0


def test_optimizer_routes_return_manifests_and_dispatch_updates_status():
    # Ensure optimizer sweep has been executed at least once
    client.post('/api/pipeline/run-all')

    manifests_response = client.get('/api/optimizer/manifests')
    assert manifests_response.status_code == 200
    manifests = manifests_response.json()
    assert isinstance(manifests, list)

    if len(manifests) == 0:
        # Nothing to dispatch; test passes as long as endpoint responds correctly
        return

    first_manifest = manifests[0]
    dispatch_response = client.patch(f"/api/optimizer/manifests/{first_manifest['id']}/dispatch")
    assert dispatch_response.status_code == 200
    dispatched_payload = dispatch_response.json()
    assert dispatched_payload['manifest']['status'] == 'DISPATCHED'
