from __future__ import annotations

import math
import random
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Sequence

from core.session import ConversationDB

_WORD_RE = re.compile(r"[A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-]{2,}")
_STOPWORDS = {
    "this", "that", "with", "have", "from", "your", "they", "them", "were", "been",
    "about", "would", "there", "their", "what", "when", "where", "which", "because",
    "så", "det", "den", "och", "att", "som", "för", "med", "inte", "har", "var",
    "jag", "mig", "dig", "you", "the", "and", "but", "our", "are", "was", "kan",
}


class ReflectionService:
    def __init__(self, db: ConversationDB):
        self.db = db

    def ingest_conversation_blocks(self, companion_id: str, model_id: str, block_size: int = 8) -> int:
        messages = self.db.get_messages_after_last_block(companion_id=companion_id)
        created = 0
        for i in range(0, len(messages), block_size):
            chunk = messages[i:i + block_size]
            if len(chunk) < 2:
                break
            self.db.create_conversation_block(
                companion_id=companion_id,
                model_id=model_id,
                session_id=chunk[0]["session_id"],
                start_message_id=chunk[0]["id"],
                end_message_id=chunk[-1]["id"],
                block_text="\n".join(f"{m['role']}: {m['content']}" for m in chunk),
                message_count=len(chunk),
                turn_count=max(1, len(chunk) // 2),
                block_type="recent",
            )
            created += 1
        return created

    def summarize_recent_blocks(self, companion_id: str, model_id: str, limit: int = 5) -> int:
        blocks = self.db.get_unsummarized_blocks(companion_id=companion_id, limit=limit)
        count = 0
        for block in blocks:
            summary_text, topics, open_loops, signals = self._summarize_text(block["block_text"])
            self.db.add_summary(
                companion_id=companion_id,
                model_id=model_id,
                source_block_id=block["id"],
                source_summary_ids=None,
                summary_level="block",
                summary_type="recent",
                summary_text=summary_text,
                key_topics=topics,
                open_loops=open_loops,
                signals=signals,
                coverage_start_at=block["created_at"],
                coverage_end_at=block["created_at"],
            )
            self.db.mark_block_summarized(block["id"])
            count += 1
        return count

    def synthesize_recent_period(self, companion_id: str, model_id: str) -> bool:
        summaries = self.db.get_recent_summaries(companion_id=companion_id, level="block", limit=5)
        if len(summaries) < 3:
            return False
        source_ids = [row["id"] for row in summaries]
        text = " ".join(row["summary_text"] for row in summaries)
        summary_text, topics, open_loops, signals = self._summarize_text(text)
        self.db.add_summary(
            companion_id=companion_id,
            model_id=model_id,
            source_block_id=None,
            source_summary_ids=source_ids,
            summary_level="period",
            summary_type="synthesis",
            summary_text=summary_text,
            key_topics=topics,
            open_loops=open_loops,
            signals=signals,
            coverage_start_at=summaries[-1]["created_at"],
            coverage_end_at=summaries[0]["created_at"],
        )
        return True

    def generate_reflections(self, companion_id: str, model_id: str, initiative_profile: Dict[str, object]) -> int:
        recent = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=5)
        if not recent:
            return 0
        older = self.db.sample_historical_summaries(
            companion_id=companion_id,
            exclude_ids=[row["id"] for row in recent],
            limit=10,
        )
        memories = self.db.get_semantic_memory(companion_id=companion_id, limit=8)
        bundle = {
            "recent_summary_ids": [row["id"] for row in recent],
            "historical_summary_ids": [row["id"] for row in older],
            "memory_ids": [row["id"] for row in memories],
            "initiative_profile": initiative_profile["profile_name"],
        }
        combined_topics = Counter()
        open_loops: List[str] = []
        for row in list(recent) + list(older):
            for topic in row["key_topics"]:
                combined_topics[topic] += 2 if row in recent else 1
            open_loops.extend(row["open_loops"])
        for memory in memories:
            combined_topics.update(self._extract_topics(memory["title"] + " " + memory["content"]))

        created = 0
        top_topics = [topic for topic, _ in combined_topics.most_common(3)]
        for topic in top_topics:
            matching_loops = [loop for loop in open_loops if topic.lower() in loop.lower()]
            question = self._build_question(topic, matching_loops)
            reflection = f"Theme '{topic}' has recurred across recent and older material."
            novelty = self._novelty_score(topic, recent, older)
            relevance = min(1.0, 0.4 + combined_topics[topic] / 6)
            groundedness = 0.75 if matching_loops else 0.6
            sensitivity = 0.2
            priority = (novelty + relevance + groundedness) / 3
            self.db.add_reflection(
                companion_id=companion_id,
                model_id=model_id,
                reflection_type="question_candidate",
                input_bundle=bundle,
                reflection_text=reflection,
                question_text=question,
                supporting_summary_ids=bundle["recent_summary_ids"] + bundle["historical_summary_ids"],
                supporting_memory_ids=bundle["memory_ids"],
                novelty_score=novelty,
                relevance_score=relevance,
                groundedness_score=groundedness,
                sensitivity_score=sensitivity,
                priority_score=priority,
            )
            created += 1
        return created

    def gate_reflections(self, companion_id: str, profile: Dict[str, object]) -> int:
        if profile["profile_name"] == "silent":
            return 0
        created = 0
        max_per_day = int(profile["outreach_max_per_day"])
        delivered_today = self.db.count_outreach_today(companion_id=companion_id)
        available_slots = max(0, max_per_day - delivered_today)
        if available_slots <= 0:
            return 0
        reflections = self.db.get_new_reflections(companion_id=companion_id, limit=available_slots * 3)
        for row in reflections:
            if row["groundedness_score"] < profile["minimum_groundedness_threshold"]:
                self.db.update_reflection_status(row["id"], "gated_out", "low_groundedness")
                continue
            if row["novelty_score"] < profile["minimum_novelty_threshold"]:
                self.db.update_reflection_status(row["id"], "gated_out", "low_novelty")
                continue
            if row["priority_score"] < profile["minimum_priority_threshold"]:
                self.db.update_reflection_status(row["id"], "gated_out", "low_priority")
                continue
            if self.db.recent_outreach_exists(companion_id, row["question_text"] or row["reflection_text"], hours=24):
                self.db.update_reflection_status(row["id"], "gated_out", "recently_used_topic")
                continue
            self.db.create_outreach_candidate(
                reflection_id=row["id"],
                companion_id=companion_id,
                model_id=row["model_id"],
                candidate_type="question",
                draft_text=row["question_text"] or row["reflection_text"],
                priority_score=row["priority_score"],
                channel="next_start",
            )
            self.db.update_reflection_status(row["id"], "gated_in", "accepted_high_priority")
            created += 1
            if created >= available_slots:
                break
        return created

    def render_pending_outreach(self, companion_id: str) -> int:
        # Draft text is already lightweight and user-facing for now.
        return self.db.mark_ready_outreach_visible(companion_id)

    def refresh_semantic_memory(self, companion_id: str, model_id: str) -> int:
        summaries = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=20)
        topics = Counter()
        for row in summaries:
            topics.update(row["key_topics"])
        updated = 0
        for topic, count in topics.most_common(5):
            self.db.upsert_semantic_memory(
                companion_id=companion_id,
                model_id=model_id,
                memory_type="theme",
                title=topic.title(),
                content=f"This theme has recurred in {count} summarized blocks or syntheses.",
                importance_score=min(1.0, 0.4 + count / 8),
            )
            updated += 1
        return updated

    def _summarize_text(self, text: str):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        summary_text = " ".join(sentences[:2])[:600] if sentences else text[:300]
        topics = self._extract_topics(text)[:6]
        open_loops = [s for s in sentences if "?" in s][:3]
        if not open_loops and sentences:
            open_loops = [sentences[-1][:160]]
        signals = {
            "topic_count": len(topics),
            "question_count": sum(1 for s in sentences if "?" in s),
            "length": len(text),
        }
        return summary_text, topics, open_loops, signals

    def _extract_topics(self, text: str) -> List[str]:
        words = [w.lower() for w in _WORD_RE.findall(text)]
        counts = Counter(w for w in words if w not in _STOPWORDS)
        return [word for word, _ in counts.most_common(6)]

    def _novelty_score(self, topic: str, recent: Sequence[Dict], older: Sequence[Dict]) -> float:
        recent_hits = sum(topic in row["key_topics"] for row in recent)
        older_hits = sum(topic in row["key_topics"] for row in older)
        novelty = 0.55 + (0.15 if recent_hits and older_hits else 0.0) - min(0.2, older_hits * 0.03)
        return max(0.1, min(1.0, novelty))

    def _build_question(self, topic: str, loops: Sequence[str]) -> str:
        if loops:
            seed = random.choice(list(loops))
            return f"I keep circling back to {topic}. Does this still feel unfinished: {seed[:180]}"
        return f"I’ve noticed {topic} returning more than once. Is there something there you want to explore further?"
