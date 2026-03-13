from __future__ import annotations

from datetime import datetime
from typing import Optional

import config
from core.app_state import AppState
from core.model import GenerationResult
from core.session import Session


class ChatService:
    def __init__(self, state: AppState):
        self.state = state

    def ensure_session(self) -> Session:
        if not self.state.active_session:
            self.state.active_session = Session(
                self.state.db,
                companion_id=self.state.companion_id,
                model_id=self.state.active_model_id,
            )
        return self.state.active_session

    def register_turn(self, user_input: str, result: Optional[GenerationResult]) -> None:
        if not result or not result.text:
            return
        session = self.ensure_session()
        session.add_turn(user_input, result.text)
        self.state.total_tokens += result.tokens
        self.state.total_elapsed += result.elapsed

    def reset_conversation(self) -> Session:
        if self.state.model:
            self.state.model.reset_state()
            self.state.model.prime_system_prompt()
        if self.state.active_session:
            self.state.active_session.end()
        self.state.active_session = Session(
            self.state.db,
            companion_id=self.state.companion_id,
            model_id=self.state.active_model_id,
        )
        return self.state.active_session

    def status_payload(self) -> dict:
        session = self.state.active_session
        db_stats = self.state.db.stats() if self.state.db else {}
        model_info = self.state.db.get_model(self.state.active_model_id) if self.state.db else None
        initiative = self.state.db.get_active_initiative_profile(self.state.companion_id) if self.state.db else None
        return {
            "ready": self.state.startup_done,
            "session_id": session.session_id if session else None,
            "turn_count": session.turn_count if session else 0,
            "bot_name": config.BOT_NAME,
            "user_name": config.USER_NAME,
            "avg_tokens_per_second": round(self.state.avg_tokens_per_second, 1),
            "active_model_id": self.state.active_model_id,
            "active_model_name": model_info["name"] if model_info else None,
            "initiative_profile": initiative["profile_name"] if initiative else "normal",
            **db_stats,
        }
