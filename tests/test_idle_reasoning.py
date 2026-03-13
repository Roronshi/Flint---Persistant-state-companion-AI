from core.model import CompanionModel
from core.session import ConversationDB
from services.idle_reasoning import IdleReasoningService
from services.reflection_service import ReflectionService


def test_idle_reasoning_falls_back_to_reflection_pipeline(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "MODEL_PATH", "dummy")
    db = ConversationDB(str(tmp_path / "test.db"))
    companion_id = db.get_or_create_default_companion()
    model_id = "rwkv::dummy"
    db.upsert_model(model_id, "dummy", "rwkv", "dummy", None, "{}")
    sid = db.new_session(companion_id, model_id)
    db.add_message(sid, "user", "What about philosophy?", companion_id, model_id)
    db.add_message(sid, "assistant", "We can explore it.", companion_id, model_id)
    db.end_session(sid)
    rs = ReflectionService(db)
    rs.ingest_conversation_blocks(companion_id, model_id, block_size=2)
    rs.summarize_recent_blocks(companion_id, model_id)
    model = CompanionModel(model_path="dummy")
    idle = IdleReasoningService(db, model, rs)
    profile = db.get_active_initiative_profile(companion_id)
    result = idle.run(companion_id, model_id, profile)
    assert result["created"] >= 1
