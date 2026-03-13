from pathlib import Path

from core.session import ConversationDB
from services.model_registry import ModelRegistryService


def test_registry_scans_onnx_and_pth(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "a.onnx").write_bytes(b"x")
    (models_dir / "b.pth").write_bytes(b"y")
    db = ConversationDB(str(tmp_path / "test.db"))
    svc = ModelRegistryService(db)
    svc.models_dir = models_dir
    entries = svc.scan_models()
    exts = {e["format"] for e in entries}
    assert "onnx" in exts and "pth" in exts
