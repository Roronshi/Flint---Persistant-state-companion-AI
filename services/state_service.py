from __future__ import annotations

"""
state_service.py — Service for runtime state snapshots and adapter versioning

This module encapsulates logic for saving and retrieving persisted runtime
state snapshots (serialized model and conversation state) and managing
versioned LoRA adapters.  It wraps the underlying ConversationDB methods
for ease of use and ensures that snapshot files are stored under the
configured data directory.

The intent is to allow Flint to restore companions across restarts and
devices, and to track the evolution of LoRA adapters over time without
requiring manual interaction from the user.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from core.session import ConversationDB
from core.model import CompanionModel

log = logging.getLogger(__name__)

# Maximum number of snapshot files to keep on disk per companion/model pair.
# Older snapshots are deleted automatically after each save.
_SNAPSHOT_KEEP_LAST = 10


class StateService:
    def __init__(self, db: ConversationDB, model: Optional[CompanionModel] = None):
        self.db = db
        self.model = model
        # Ensure state directory exists
        self.states_dir = Path(config.BASE_DIR) / "data" / "states"
        self.states_dir.mkdir(parents=True, exist_ok=True)

    def save_runtime_snapshot(self, companion_id: str, model_id: Optional[str], notes: str = "") -> str:
        """
        Persist the current runtime state to disk and record it in the
        database.  Returns the snapshot id.  If no model is attached to
        this service, only the conversation state (if any) is saved; the
        snapshot still records its path and notes.
        """
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = f"state-{companion_id[:8]}-{timestamp}.bin"
        snapshot_path = str(self.states_dir / filename)
        # Attempt to save model state if available
        if self.model:
            try:
                self.model.save_state(snapshot_path)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Failed to save model state: %s", exc)
        # Ensure there is always a concrete file to export later, even if the model
        # had no in-memory state to serialize yet.
        if not Path(snapshot_path).exists():
            Path(snapshot_path).write_bytes(b"")
        snapshot_id = self.db.add_runtime_state_snapshot(
            companion_id=companion_id,
            model_id=model_id,
            snapshot_path=snapshot_path,
            notes=notes,
        )
        log.debug("Snapshot saved: %s", snapshot_path)
        self._prune_old_snapshots(companion_id, model_id)
        return snapshot_id

    def _prune_old_snapshots(self, companion_id: str, model_id: Optional[str]) -> None:
        """Delete snapshot files (and their DB rows) beyond _SNAPSHOT_KEEP_LAST."""
        rows = self.db.get_runtime_state_snapshots(companion_id, model_id, limit=1000)
        # Rows are ordered newest-first; anything beyond the keep window is stale.
        to_delete = rows[_SNAPSHOT_KEEP_LAST:]
        if not to_delete:
            return
        ids_to_delete = [r["id"] for r in to_delete]
        for row in to_delete:
            path = row.get("snapshot_path", "")
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as exc:
                    log.warning("Could not delete old snapshot file %s: %s", path, exc)
        self.db.delete_runtime_state_snapshots(ids_to_delete)
        log.debug("Pruned %d old snapshot(s).", len(to_delete))

    def register_adapter_version(self, companion_id: str, model_id: Optional[str], adapter_path: str, notes: str = "") -> str:
        """
        Record a new adapter version in the database.  The adapter_path
        should point to the LoRA adapter file relative to the project
        directory.  Returns the adapter id.
        """
        return self.db.add_adapter_version(
            companion_id=companion_id,
            model_id=model_id,
            adapter_path=adapter_path,
            notes=notes,
        )

    def latest_runtime_snapshot(self, companion_id: str, model_id: Optional[str] = None):
        """
        Return the most recently recorded runtime snapshot.  Returns a dict
        with keys matching the runtime_state_snapshots table or None.
        """
        return self.db.get_latest_runtime_state_snapshot(companion_id, model_id)

    def latest_adapter_version(self, companion_id: str, model_id: Optional[str] = None):
        """
        Return the most recent adapter version entry.  Returns a dict or None.
        """
        return self.db.get_latest_adapter_version(companion_id, model_id)