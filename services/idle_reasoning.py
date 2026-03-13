from __future__ import annotations

import logging
from typing import Any, Dict, List

import config
from core.model import CompanionModel
from core.session import ConversationDB
from services.reflection_service import ReflectionService

log = logging.getLogger(__name__)


class IdleReasoningService:
    """Generates reflection candidates via model reasoning when available."""

    def __init__(self, db: ConversationDB, model: CompanionModel, reflection_service: ReflectionService):
        self.db = db
        self.model = model
        self.reflection_service = reflection_service

    def _build_context(self, companion_id: str) -> Dict[str, Any]:
        summaries = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=5)
        memories = self.db.get_semantic_memory(companion_id=companion_id, limit=8)
        open_loops: List[str] = []
        for row in summaries:
            open_loops.extend(row.get("open_loops", []))
        return {"summaries": summaries, "memories": memories, "open_loops": open_loops[:10]}

    def _prompt_from_context(self, bundle: Dict[str, Any]) -> str:
        summaries_txt = "\n".join(f"- {row['summary_text']}" for row in bundle["summaries"])
        memories_txt = "\n".join(f"- {row['title']}: {row['content']}" for row in bundle["memories"])
        loops_txt = "\n".join(f"- {item}" for item in bundle["open_loops"])
        return (
            f"You are {config.BOT_NAME}. Reflect quietly on prior conversations with {config.USER_NAME}.\n\n"
            f"Recent summaries:\n{summaries_txt or '- none'}\n\n"
            f"Semantic memory:\n{memories_txt or '- none'}\n\n"
            f"Open loops:\n{loops_txt or '- none'}\n\n"
            "Generate up to 3 short grounded open-ended questions. One per line."
        )

    def _parse_questions(self, text: str) -> List[str]:
        out = []
        for raw in text.splitlines():
            line = raw.strip().lstrip("-•0123456789. ")
            if not line:
                continue
            if len(line) < 12:
                continue
            out.append(line)
        return out[:3]

    def run(self, companion_id: str, model_id: str, initiative_profile: Dict[str, Any]) -> Dict[str, int]:
        if self.model.dummy or not getattr(self.model.backend, "supports_reasoning_mode", False):
            created = self.reflection_service.generate_reflections(companion_id, model_id, initiative_profile)
            gated = self.reflection_service.gate_reflections(companion_id, initiative_profile)
            visible = self.reflection_service.render_pending_outreach(companion_id)
            return {"created": created, "gated": gated, "visible": visible, "mode": 0}

        bundle = self._build_context(companion_id)
        prompt = self._prompt_from_context(bundle)
        result = self.model.generate_stateless(prompt, max_tokens=min(220, config.MAX_TOKENS))
        questions = self._parse_questions(result.text)
        created = 0
        for question in questions:
            self.db.add_reflection(
                companion_id=companion_id,
                model_id=model_id,
                reflection_type="idle_reasoning",
                input_bundle=bundle,
                reflection_text="Model-generated idle reflection",
                question_text=question,
                supporting_summary_ids=[row["id"] for row in bundle["summaries"]],
                supporting_memory_ids=[row["id"] for row in bundle["memories"]],
                novelty_score=0.65,
                relevance_score=0.7,
                groundedness_score=0.75,
                sensitivity_score=0.2,
                priority_score=0.7,
            )
            created += 1
        gated = self.reflection_service.gate_reflections(companion_id, initiative_profile)
        visible = self.reflection_service.render_pending_outreach(companion_id)
        log.info("Idle reasoning created=%s gated=%s visible=%s", created, gated, visible)
        return {"created": created, "gated": gated, "visible": visible, "mode": 1}
