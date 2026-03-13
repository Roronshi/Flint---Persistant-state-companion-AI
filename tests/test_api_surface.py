from fastapi.testclient import TestClient

from web.server import app


def test_api_surface_basic():
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        info = client.get('/api/info').json()
        assert info['name'] == 'flint'
        models = client.get('/api/models').json()
        assert 'models' in models
        save = client.post('/api/save').json()
        assert save['ok'] is True
        lora = client.get('/api/lora/status').json()
        assert 'can_run' in lora


def test_info_contains_onboarding_fields():
    with TestClient(app) as client:
        info = client.get('/api/info').json()
        assert 'training_method' in info
        assert 'recommended_model_preset' in info
        assert 'needs_model_upload' in info


def test_status_contains_system_health_fields():
    with TestClient(app) as client:
        status = client.get('/api/status').json()
        assert 'snapshot_status' in status
        assert 'reflection_status' in status
        assert 'idle_reasoning_status' in status
        assert 'lora_last_run' in status


def test_model_upload_rejects_empty_file():
    with TestClient(app) as client:
        files = {'file': ('empty.onnx', b'', 'application/octet-stream')}
        resp = client.post('/api/models/upload', files=files)
        data = resp.json()
        assert data['ok'] is False
        assert 'empty' in data['message'].lower()


def test_info_optionally_exposes_onnx_fields():
    with TestClient(app) as client:
        info = client.get('/api/info').json()
        # Keys may be absent in non-ONNX mode, but request must succeed.
        assert 'backend_kind' in info


from pathlib import Path

def test_beta_candidate_checklist_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'BETA_CANDIDATE_CHECKLIST.md').exists()

def test_readme_mentions_beta_candidate():
    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8', errors='ignore').lower()
    assert 'beta candidate' in readme
