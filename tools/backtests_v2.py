from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from pathlib import Path

from core.session import ConversationDB
from services.reflection_service import ReflectionService
from services.state_service import StateService


def run_tests():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    test_db = data_dir / "test_backtest_v2.db"
    if test_db.exists():
        test_db.unlink()
    db = ConversationDB(str(test_db))
    companion_id = db.get_or_create_default_companion()

    snap1 = db.add_runtime_state_snapshot(companion_id, None, "/tmp/s1.bin", "one")
    snap2 = db.add_runtime_state_snapshot(companion_id, None, "/tmp/s2.bin", "two")
    latest = db.get_latest_runtime_state_snapshot(companion_id)
    assert latest and latest["id"] == snap2
    print("snapshot test: PASS")

    aid1 = db.add_adapter_version(companion_id, None, "adapters/a1.bin", "one")
    aid2 = db.add_adapter_version(companion_id, None, "adapters/a2.bin", "two")
    latest_a = db.get_latest_adapter_version(companion_id)
    assert latest_a and latest_a["id"] == aid2 and latest_a["version"] == 2
    print("adapter test: PASS")

    model_id = "rwkv::dummy-test"
    db.upsert_model(model_id, "dummy", "rwkv", "dummy", None, "{}")
    sid = db.new_session(companion_id, model_id)
    db.add_message(sid, "user", "Hello there", companion_id, model_id)
    db.add_message(sid, "assistant", "Hi there", companion_id, model_id)
    db.end_session(sid)
    rs = ReflectionService(db)
    rs.ingest_conversation_blocks(companion_id, model_id, block_size=2)
    rs.summarize_recent_blocks(companion_id, model_id)
    profile = db.get_active_initiative_profile(companion_id)
    created = rs.generate_reflections(companion_id, model_id, profile)
    rs.gate_reflections(companion_id, profile)
    rs.render_pending_outreach(companion_id)
    assert created >= 1
    assert len(db.get_visible_outreach(companion_id)) >= 1
    print("reflection pipeline: PASS")

    class DummyModel:
        def save_state(self, path: str):
            with open(path, "wb") as f:
                f.write(b"dummy")

    ss = StateService(db, DummyModel())
    sid3 = ss.save_runtime_snapshot(companion_id, model_id, notes="service")
    assert ss.latest_runtime_snapshot(companion_id, model_id)["id"] == sid3
    print("state service: PASS")

    try:
        os.remove(test_db)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    run_tests()
