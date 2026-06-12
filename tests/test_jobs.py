"""Tests for utils.jobs.JobManager.

Offline and deterministic: jobs are plain callables synchronized with
threading.Event; no sleeps in the job bodies themselves, and every wait is
bounded by a timeout so a regression fails fast instead of hanging.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import pytest

from utils.jobs import MAX_FINISHED_JOBS, JobManager, get_job_manager

WAIT_TIMEOUT = 5.0


def wait_for(predicate, timeout=WAIT_TIMEOUT, interval=0.005):
    """Poll predicate until truthy or timeout; returns the last value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def wait_for_status(manager, job_id, status):
    record = wait_for(
        lambda: (manager.get(job_id) or {}).get('status') == status)
    assert record, (
        f"job {job_id} never reached status {status!r}: {manager.get(job_id)}")
    return manager.get(job_id)


class TestLifecycle:
    def test_pending_before_thread_starts(self, monkeypatch):
        # Suppress thread start so the freshly registered record is
        # observable deterministically, then start it manually.
        original_start = threading.Thread.start
        captured = []
        monkeypatch.setattr(threading.Thread, 'start',
                            lambda self: captured.append(self))

        manager = JobManager()
        job_id = manager.submit(lambda progress: 'ok')

        record = manager.get(job_id)
        assert record['status'] == 'pending'
        assert record['progress'] == 0.0
        assert record['result'] is None
        assert record['error'] is None
        assert record['created_at'] is not None
        assert record['finished_at'] is None

        assert len(captured) == 1
        original_start(captured[0])
        wait_for_status(manager, job_id, 'done')

    def test_running_then_done_with_result(self):
        manager = JobManager()
        release = threading.Event()

        def job(progress):
            release.wait(WAIT_TIMEOUT)
            return {'answer': 42}

        job_id = manager.submit(job)
        record = wait_for_status(manager, job_id, 'running')
        assert record['result'] is None
        assert record['finished_at'] is None

        release.set()
        record = wait_for_status(manager, job_id, 'done')
        assert record['result'] == {'answer': 42}
        assert record['error'] is None
        assert record['finished_at'] is not None
        assert record['progress'] == 100.0

    def test_positional_and_keyword_args_are_forwarded(self):
        manager = JobManager()

        def job(a, b, progress, scale=1):
            return (a + b) * scale

        job_id = manager.submit(job, 2, 3, scale=10)
        record = wait_for_status(manager, job_id, 'done')
        assert record['result'] == 50

    def test_injectable_clock_stamps_created_and_finished(self):
        ticks = [datetime(2026, 1, 1, 12, 0, 0),
                 datetime(2026, 1, 1, 12, 0, 5)]
        clock = iter(ticks)
        manager = JobManager(now_fn=lambda: next(clock))

        job_id = manager.submit(lambda progress: None)
        record = wait_for_status(manager, job_id, 'done')
        assert record['created_at'] == ticks[0].isoformat()
        assert record['finished_at'] == ticks[1].isoformat()


class TestErrors:
    def test_exception_sets_error_status_with_message_only(self):
        manager = JobManager()

        def job(progress):
            raise ValueError("boom: bad input")

        job_id = manager.submit(job)
        record = wait_for_status(manager, job_id, 'error')
        assert record['error'] == "boom: bad input"
        # The traceback is logged server-side only, never stored.
        assert 'Traceback' not in record['error']
        assert record['result'] is None
        assert record['finished_at'] is not None

    def test_unknown_id_returns_none(self):
        manager = JobManager()
        assert manager.get('not-a-job-id') is None


class TestProgress:
    def test_injected_progress_updates_get(self):
        manager = JobManager()
        checkpoint = threading.Event()
        release = threading.Event()

        def job(progress):
            progress(37.5)
            checkpoint.set()
            release.wait(WAIT_TIMEOUT)
            return 'done'

        job_id = manager.submit(job)
        assert checkpoint.wait(WAIT_TIMEOUT)
        assert manager.get(job_id)['progress'] == 37.5
        assert manager.get(job_id)['status'] == 'running'

        release.set()
        record = wait_for_status(manager, job_id, 'done')
        assert record['progress'] == 100.0

    def test_progress_is_clamped_to_0_100(self):
        manager = JobManager()
        below_seen = threading.Event()
        above_go = threading.Event()
        above_seen = threading.Event()
        release = threading.Event()

        def job(progress):
            progress(-50.0)
            below_seen.set()
            above_go.wait(WAIT_TIMEOUT)
            progress(150.0)
            above_seen.set()
            release.wait(WAIT_TIMEOUT)
            return 'done'

        job_id = manager.submit(job)
        assert below_seen.wait(WAIT_TIMEOUT)
        assert manager.get(job_id)['progress'] == 0.0

        above_go.set()
        assert above_seen.wait(WAIT_TIMEOUT)
        assert manager.get(job_id)['progress'] == 100.0
        # Clamping never finishes the job; only _finish does that.
        assert manager.get(job_id)['status'] == 'running'

        release.set()
        record = wait_for_status(manager, job_id, 'done')
        assert record['progress'] == 100.0


class TestListJobs:
    def test_newest_first_with_result_omitted(self):
        manager = JobManager()
        job_ids = []
        for i in range(3):
            job_id = manager.submit(lambda progress, i=i: f"result-{i}")
            wait_for_status(manager, job_id, 'done')
            job_ids.append(job_id)

        listed = manager.list_jobs()
        assert [r['job_id'] for r in listed] == list(reversed(job_ids))
        assert all(r['result'] is None for r in listed)
        assert all(r['status'] == 'done' for r in listed)
        # get() still serves the full result.
        assert manager.get(job_ids[-1])['result'] == 'result-2'

    def test_limit_caps_payload(self):
        manager = JobManager()
        for i in range(5):
            job_id = manager.submit(lambda progress: None)
            wait_for_status(manager, job_id, 'done')

        assert len(manager.list_jobs(limit=2)) == 2
        assert len(manager.list_jobs()) == 5


class TestEviction:
    def test_finished_jobs_capped_at_50_oldest_evicted(self):
        manager = JobManager()
        job_ids = []
        # Sequential submits (each awaited) make eviction order
        # deterministic: the oldest finished jobs go first.
        for i in range(MAX_FINISHED_JOBS + 5):
            job_id = manager.submit(lambda progress, i=i: i)
            wait_for_status(manager, job_id, 'done')
            job_ids.append(job_id)

        evicted, kept = job_ids[:5], job_ids[5:]
        for job_id in evicted:
            assert manager.get(job_id) is None
        for job_id in kept:
            assert manager.get(job_id) is not None
        assert len(manager.list_jobs(limit=100)) == MAX_FINISHED_JOBS

    def test_running_jobs_are_never_evicted(self):
        manager = JobManager()
        release = threading.Event()

        def long_job(progress):
            release.wait(WAIT_TIMEOUT)
            return 'long'

        long_id = manager.submit(long_job)
        wait_for_status(manager, long_id, 'running')

        for i in range(MAX_FINISHED_JOBS + 3):
            job_id = manager.submit(lambda progress: None)
            wait_for_status(manager, job_id, 'done')

        # Still present despite the finished-job churn around it.
        assert manager.get(long_id)['status'] == 'running'
        release.set()
        assert wait_for_status(manager, long_id, 'done')['result'] == 'long'


class TestSingleton:
    def test_get_job_manager_returns_process_wide_singleton(self):
        assert get_job_manager() is get_job_manager()
        assert isinstance(get_job_manager(), JobManager)
