from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

from core.session import ConversationDB

log = logging.getLogger(__name__)


class ScheduledJob:
    def __init__(self, name: str, interval_seconds: int, fn: Callable[[], int | bool | None]):
        self.name = name
        self.interval_seconds = interval_seconds
        self.fn = fn
        self.last_run: Optional[datetime] = None
        self.last_success: Optional[datetime] = None
        self.last_error: Optional[str] = None


class SchedulerService:
    def __init__(self, db: ConversationDB):
        self.db = db
        self._jobs: Dict[str, ScheduledJob] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_job(self, name: str, interval_seconds: int, fn: Callable[[], int | bool | None]):
        self._jobs[name] = ScheduledJob(name, interval_seconds, fn)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="flint-scheduler")
        self._thread.start()
        log.info("Flint scheduler started with %d jobs", len(self._jobs))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def status(self) -> dict:
        return {
            name: {
                "interval_seconds": job.interval_seconds,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_success": job.last_success.isoformat() if job.last_success else None,
                "last_error": job.last_error,
            }
            for name, job in self._jobs.items()
        }

    def _loop(self):
        while not self._stop.is_set():
            now = datetime.now()
            for job in self._jobs.values():
                if job.last_run and now - job.last_run < timedelta(seconds=job.interval_seconds):
                    continue
                self._run_job(job, now)
            self._stop.wait(15)

    def _run_job(self, job: ScheduledJob, scheduled_for: datetime):
        run_id = self.db.log_job_run(job_type=job.name, companion_id=None, status="running", scheduled_for=scheduled_for.isoformat())
        job.last_run = datetime.now()
        try:
            result = job.fn()
            job.last_success = datetime.now()
            job.last_error = None
            self.db.finish_job_run(run_id, status="success", metadata={"result": result})
        except Exception as exc:  # pragma: no cover - defensive
            job.last_error = str(exc)
            log.exception("Background job %s failed", job.name)
            self.db.finish_job_run(run_id, status="failed", error_message=str(exc))
