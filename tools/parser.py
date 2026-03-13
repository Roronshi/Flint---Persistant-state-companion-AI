# tools/parser.py — Conversation import for ChatGPT and Claude exports
#
# Usage:
#   python tools/parser.py --source chatgpt --file ~/Downloads/conversations.json
#   python tools/parser.py --source claude  --file ~/Downloads/conversations.json
#   python tools/parser.py --source claude  --file data/our_conversation.json --dry-run

import json
import logging
import argparse
import sqlite3
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.session import ConversationDB

log = logging.getLogger(__name__)


# ── Shared DB context ──────────────────────────────────────────────────────────

@contextmanager
def _raw_conn(db_path: str):
    """
    Bare sqlite3 connection used by import_to_db.
    Uses the same WAL + foreign_keys settings as ConversationDB._conn(),
    and guarantees the connection is always closed.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


# ── ChatGPT ────────────────────────────────────────────────────────────────────

def parse_chatgpt(filepath: str) -> list:
    """
    Parse ChatGPT conversations.json (OpenAI data export format).
    Returns a list of normalized session dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        log.warning("Unexpected ChatGPT export structure — expected a list.")
        return []

    sessions = []
    for convo in data:
        mapping = convo.get("mapping", {})
        if not mapping:
            continue
        messages = _walk_chatgpt_tree(mapping, convo.get("current_node"))
        if len(messages) >= 2:
            sessions.append({
                "id":         str(uuid.uuid4())[:8],
                "source":     "chatgpt",
                "started_at": messages[0]["timestamp"],
                "messages":   messages,
            })

    log.info(f"ChatGPT: {len(sessions)} conversations parsed.")
    return sessions


def _walk_chatgpt_tree(mapping: dict, start_node: str) -> list:
    """
    BFS walk of the ChatGPT message tree. Returns messages in conversation order.
    Uses deque (not recursion) — safe for deeply branched exports.
    """
    if not start_node:
        return []

    messages = []
    visited  = set()
    queue    = deque([start_node])

    while queue:
        node_id = queue.popleft()
        if not node_id or node_id in visited or node_id not in mapping:
            continue
        visited.add(node_id)

        node = mapping[node_id]
        msg  = node.get("message")

        if msg and msg.get("content"):
            parts   = msg["content"].get("parts", [])
            content = " ".join(p for p in parts if isinstance(p, str)).strip()
            role    = msg.get("author", {}).get("role", "")

            if content and role in ("user", "assistant"):
                ct = msg.get("create_time")
                ts = (
                    datetime.fromtimestamp(ct, tz=timezone.utc).isoformat()
                    if ct
                    else datetime.now(tz=timezone.utc).isoformat()
                )
                messages.append({
                    "role":      "user" if role == "user" else "bot",
                    "content":   content,
                    "timestamp": ts,
                })

        for child in node.get("children", []):
            queue.append(child)

    return messages


# ── Claude ─────────────────────────────────────────────────────────────────────

def parse_claude(filepath: str) -> list:
    """
    Parse Claude.ai conversations.json export.
    Handles the standard Claude export format and our own normalized format.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    else:
        conversations = [data]

    sessions = []
    for convo in conversations:
        messages     = []
        raw_messages = convo.get("messages", convo.get("chat_messages", []))

        for msg in raw_messages:
            role    = msg.get("role", msg.get("sender", ""))
            content = msg.get("content", "")

            # Claude exports content as a list of typed content blocks
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            content = content.strip()

            if content and role in ("human", "user", "assistant"):
                messages.append({
                    "role":      "user" if role in ("human", "user") else "bot",
                    "content":   content,
                    "timestamp": msg.get("created_at", datetime.now().isoformat()),
                })

        if len(messages) >= 2:
            sessions.append({
                "id":         str(uuid.uuid4())[:8],
                "source":     "claude",
                "started_at": messages[0]["timestamp"],
                "messages":   messages,
            })

    log.info(f"Claude: {len(sessions)} conversations parsed.")
    return sessions


# ── Database import ────────────────────────────────────────────────────────────

def import_to_db(sessions: list, db: ConversationDB) -> tuple[int, int]:
    """
    Import normalized sessions into SQLite.
    Each session is its own transaction — a crash won't leave partial imports.
    Sessions already present in the DB are skipped (idempotent).
    Returns (imported_count, skipped_count).
    Prints progress every 50 sessions for large imports.
    """
    imported = 0
    skipped  = 0

    with _raw_conn(db.db_path) as conn:
        for i, session in enumerate(sessions):
            # Progress for large imports
            if i > 0 and i % 50 == 0:
                print(f"  Progress: {i}/{len(sessions)} sessions processed "
                      f"({imported} imported, {skipped} skipped)...")

            if conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session["id"],)
            ).fetchone():
                skipped += 1
                continue

            try:
                with conn:  # per-session transaction
                    conn.execute(
                        """INSERT INTO sessions
                           (id, started_at, ended_at, turn_count, lora_version)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            session["id"],
                            session["started_at"],
                            session["started_at"],
                            len(session["messages"]),
                            "imported",
                        )
                    )
                    for msg in session["messages"]:
                        cursor = conn.execute(
                            """INSERT INTO messages
                               (session_id, role, content, timestamp)
                               VALUES (?, ?, ?, ?)""",
                            (
                                session["id"],
                                msg["role"],
                                msg["content"],
                                msg["timestamp"],
                            )
                        )
                        message_id = cursor.lastrowid
                        conn.execute(
                            """INSERT INTO messages_fts
                               (message_id, session_id, role, content)
                               VALUES (?, ?, ?, ?)""",
                            (message_id, session["id"], msg["role"], msg["content"])
                        )
                imported += 1
            except Exception as e:
                log.warning(f"Skipping session {session['id']}: {e}")
                skipped += 1

    log.info(f"Import complete: {imported} imported, {skipped} skipped.")
    return imported, skipped


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import conversation history into RWKV Companion"
    )
    parser.add_argument(
        "--source",
        choices=["chatgpt", "claude"],
        required=True,
        help="Format of the export file"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the export JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and preview without writing to the database"
    )
    args = parser.parse_args()

    filepath = Path(args.file).expanduser().resolve()
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"Parsing {args.source} export: {filepath.name}")

    if args.source == "chatgpt":
        sessions = parse_chatgpt(str(filepath))
    else:
        sessions = parse_claude(str(filepath))

    if not sessions:
        print("No conversations found. Check the file format.")
        sys.exit(0)

    total_messages = sum(len(s["messages"]) for s in sessions)
    print(f"\nParsed: {len(sessions)} conversations, {total_messages} messages")
    print(f"\nSample (first 2 turns):")
    for msg in sessions[0]["messages"][:2]:
        preview = msg["content"][:120].replace("\n", " ")
        print(f"  [{msg['role']}] {preview}...")

    if args.dry_run:
        print("\nDry run — nothing saved.")
        return

    print(f"\nImporting into {config.CONVERSATIONS_DB}...")
    db = ConversationDB()
    imported, skipped = import_to_db(sessions, db)
    print(f"\nDone. {imported} sessions imported, {skipped} skipped.")
    if imported > 0:
        print("Run '/lora now' in the chat to train immediately,")
        print("or they will be included in the next nightly run.")


if __name__ == "__main__":
    main()
