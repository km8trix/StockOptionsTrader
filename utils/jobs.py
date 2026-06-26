"""
Background job manager for long-running work (e.g. backtests).

Jobs run on daemon threads. The submitted callable MUST accept a keyword
argument ``progress``: a callable taking a float 0-100 that the JobManager
injects so the job can report completion percentage (out-of-range values
are clamped to [0, 100] before being stored). The callable's return
value becomes the job result; a raised exception marks the job 'error' with
the exception message (the full traceback is logged server-side only and is
never stored in the job record).
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Maximum number of finished ('done'/'error') jobs retained in the registry.
#: When exceeded, the oldest finished jobs are evicted first.
MAX_FINISHED_JOBS = 50


class JobManager:
    """Thread-safe registry of background jobs executed on daemon threads.

    The clock is injectable (``now_fn``) so tests can pin timestamps.
    """

    def __init__(self, now_fn: Callable[[], datetime] = datetime.now):
        self._now_fn = now_fn
        self._lock = threading.Lock()
        # Insertion-ordered: oldest submission first.
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # Job ids in finish order (oldest finished first). Eviction goes by
        # finish order, NOT creation order, so a long-running job is never
        # evicted the instant it completes just because it was created early.
        self._finished_order: List[str] = []

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> str:
        """Run fn(*args, progress=<callable>, **kwargs) on a daemon thread.

        Returns the job id immediately; poll get(job_id) for status.
        """
        job_id = uuid.uuid4().hex
        record = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0.0,
            'result': None,
            'error': None,
            'created_at': self._now_fn().isoformat(),
            'finished_at': None,
        }
        with self._lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, fn, args, kwargs),
            name=f"job-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        logger.info("Submitted job %s (%s)", job_id, getattr(fn, '__name__', fn))
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return a snapshot dict for the job, or None for unknown ids."""
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_job(self, job_id: str, fn: Callable[..., Any],
                 args: tuple, kwargs: dict) -> None:
        """Daemon-thread body: run fn and record outcome."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:  # evicted before it ever ran; nothing to do
                return
            record['status'] = 'running'

        def progress(value: float) -> None:
            # Clamp to [0, 100] so a misbehaving job body can never expose
            # an out-of-range percentage via get().
            clamped = min(100.0, max(0.0, float(value)))
            with self._lock:
                rec = self._jobs.get(job_id)
                if rec is not None:
                    rec['progress'] = clamped

        try:
            result = fn(*args, progress=progress, **kwargs)
        except Exception as exc:
            # Full traceback server-side only; the record gets the message.
            logger.exception("Job %s failed", job_id)
            self._finish(job_id, status='error', error=str(exc))
            return

        self._finish(job_id, status='done', result=result)

    def _finish(self, job_id: str, status: str, result: Any = None,
                error: Optional[str] = None) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is not None:
                record['status'] = status
                record['result'] = result
                record['error'] = error
                record['finished_at'] = self._now_fn().isoformat()
                if status == 'done':
                    record['progress'] = 100.0
                self._finished_order.append(job_id)
            self._evict_finished_locked()

    def _evict_finished_locked(self) -> None:
        """Evict oldest-FINISHED jobs beyond MAX_FINISHED_JOBS. Lock held."""
        excess = len(self._finished_order) - MAX_FINISHED_JOBS
        if excess <= 0:
            return
        evicted, self._finished_order = (self._finished_order[:excess],
                                         self._finished_order[excess:])
        for jid in evicted:
            self._jobs.pop(jid, None)
            logger.debug("Evicted finished job %s", jid)


_manager_lock = threading.Lock()
_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Process-wide singleton JobManager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
