"""Tests for utils/scheduler.py (LiveScheduler).

Deterministic: the tick/sleep behavior is exercised by calling the loop
body synchronously with an injected fake clock + fake sleep (the fake
sleep ADVANCES the fake clock, so virtual time moves only when the
scheduler decides to sleep). Real daemon threads are used only for the
lifecycle tests (start/stop idempotency, restart after pause, no thread
leak), where the default stop-event sleep makes stop() interrupt
instantly — no real waiting, no flaky timing assertions.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from utils.scheduler import MAX_SLEEP_CHUNK, LiveScheduler

NY = ZoneInfo("America/New_York")


def ny(y, m, d, hh, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


# A plain open Wednesday and the Saturday after (2026 calendar).
WED_OPEN = ny(2026, 6, 10, 10, 0)
SATURDAY = ny(2026, 6, 13, 12, 0)
MONDAY_OPEN = ny(2026, 6, 15, 9, 30)


class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeSleep:
    """Records every requested sleep and advances the fake clock by it."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


class FakeAudit:
    def __init__(self):
        self.rows = []

    def append(self, actor, event_type, payload):
        self.rows.append((actor, event_type, dict(payload)))

    def events(self):
        return [r[1] for r in self.rows]


class FakeSession:
    """evaluate_once stub: records the (virtual) call time, then returns
    the scripted result; optionally trips the scheduler's stop event after
    N calls so synchronous loop runs terminate deterministically."""

    def __init__(self, clock, result=None, stop_after=None):
        self.clock = clock
        self.result = result if result is not None else {
            "status": "ok", "reports": []}
        self.stop_after = stop_after
        self.calls: list[datetime] = []
        self.on_stop = lambda: None
        self.raises: Exception | None = None
        self.raise_times = 0  # raise for the first N calls, then succeed
        self.evaluated = threading.Event()

    def evaluate_once(self):
        self.calls.append(self.clock())
        self.evaluated.set()
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            self.on_stop()
        if self.raises is not None and (
                self.raise_times == 0 or len(self.calls) <= self.raise_times):
            raise self.raises
        return self.result


def make_sync(start=WED_OPEN, *, stop_after=None, result=None,
              interval_minutes=15, max_consecutive_errors=5, audit=None):
    """A scheduler wired for SYNCHRONOUS _run() with fake clock/sleep."""
    clock = FakeClock(start)
    sleep = FakeSleep(clock)
    session = FakeSession(clock, result=result, stop_after=stop_after)
    scheduler = LiveScheduler(
        session, interval_minutes=interval_minutes, clock=clock,
        sleep_fn=sleep, max_consecutive_errors=max_consecutive_errors,
        audit=audit)
    session.on_stop = scheduler._stop_event.set
    return scheduler, session, clock, sleep


def wait_for(predicate, timeout=5.0):
    """Poll a condition with a hard timeout (real-thread tests only)."""
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.01)
    return False


# ==========================================================================
# Tick cadence during market hours
# ==========================================================================

class TestTicksDuringHours:
    def test_ticks_at_exact_intervals(self):
        scheduler, session, _, _ = make_sync(WED_OPEN, stop_after=4)
        scheduler._run()

        assert session.calls == [
            ny(2026, 6, 10, 10, 0),
            ny(2026, 6, 10, 10, 15),
            ny(2026, 6, 10, 10, 30),
            ny(2026, 6, 10, 10, 45),
        ]

    def test_interval_is_configurable(self):
        scheduler, session, _, _ = make_sync(
            WED_OPEN, stop_after=3, interval_minutes=5)
        scheduler._run()
        assert session.calls == [
            ny(2026, 6, 10, 10, 0),
            ny(2026, 6, 10, 10, 5),
            ny(2026, 6, 10, 10, 10),
        ]

    def test_interval_sleeps_are_bounded_chunks(self):
        scheduler, _, _, sleep = make_sync(WED_OPEN, stop_after=2)
        scheduler._run()
        assert sleep.calls, "interval wait must actually sleep"
        assert all(c <= MAX_SLEEP_CHUNK for c in sleep.calls)
        assert sum(sleep.calls) == pytest.approx(15 * 60)

    def test_run_records_status_fields(self):
        scheduler, session, _, _ = make_sync(WED_OPEN, stop_after=3)
        scheduler._run()
        status = scheduler.status()
        assert status["runs_today"] == 3
        assert status["last_run"] == ny(2026, 6, 10, 10, 30).isoformat()
        assert status["last_result"] == session.result
        assert status["consecutive_errors"] == 0
        assert status["paused_reason"] is None

    def test_runs_today_resets_on_new_ny_trading_day(self):
        """A tick landing after the close sleeps to the NEXT open, and the
        per-day counter restarts with the New York date."""
        scheduler, session, _, _ = make_sync(
            ny(2026, 6, 10, 15, 50), stop_after=2)
        scheduler._run()

        assert session.calls == [
            ny(2026, 6, 10, 15, 50),
            ny(2026, 6, 11, 9, 30),  # next tick 16:05 is after the close
        ]
        assert scheduler.status()["runs_today"] == 1  # Thursday only


# ==========================================================================
# Outside hours: chunked sleep until next open
# ==========================================================================

class TestSleepsToOpen:
    def test_weekend_sleeps_to_monday_open_in_bounded_chunks(self):
        scheduler, session, _, sleep = make_sync(SATURDAY, stop_after=1)
        scheduler._run()

        # Never evaluated before the bell; first tick exactly at the open.
        assert session.calls == [MONDAY_OPEN]
        assert all(c <= MAX_SLEEP_CHUNK for c in sleep.calls)
        assert sum(sleep.calls) == pytest.approx(
            (MONDAY_OPEN - SATURDAY).total_seconds())

    def test_half_day_close_sleeps_to_next_session(self):
        """13:00 close on 2026-11-27 (day after Thanksgiving): a tick due
        at 13:05 must wait for Monday's open instead of evaluating."""
        scheduler, session, _, _ = make_sync(
            ny(2026, 11, 27, 12, 50), stop_after=2)
        scheduler._run()
        assert session.calls == [
            ny(2026, 11, 27, 12, 50),
            ny(2026, 11, 30, 9, 30),
        ]


# ==========================================================================
# Pause rules: session halts (kill switch / circuit breaker)
# ==========================================================================

class TestPauseOnHalt:
    @pytest.mark.parametrize("reason", [
        "kill_switch_engaged",            # top-of-cycle check
        "kill_switch_engaged_mid_loop",   # mid-execution engagement
        "daily_loss_circuit_breaker",     # E5 gate breach
        "circuit_breaker_error",          # gate unreadable (fail-closed)
        "execution_error",                # broker failure mid-execution
    ])
    def test_halted_result_pauses_with_the_sessions_reason(self, reason):
        audit = FakeAudit()
        scheduler, session, _, _ = make_sync(
            WED_OPEN, result={"status": "halted", "reason": reason,
                              "reports": []},
            audit=audit)
        scheduler._run()  # exits by itself — no stop event needed

        assert len(session.calls) == 1, "must NOT retry against a halt"
        status = scheduler.status()
        assert status["running"] is False
        assert status["paused_reason"] == reason
        assert status["last_result"]["status"] == "halted"
        assert ("scheduler", "scheduler_paused", {"reason": reason}) \
            in audit.rows

    def test_paused_scheduler_requires_manual_start(self):
        """After a kill-switch pause the loop is DEAD until start(); the
        manual restart clears the pause and evaluates again."""
        audit = FakeAudit()
        clock = FakeClock(WED_OPEN)
        session = FakeSession(clock, result={
            "status": "halted", "reason": "kill_switch_engaged",
            "reports": []})
        scheduler = LiveScheduler(session, interval_minutes=15, clock=clock,
                                  audit=audit)
        session.on_stop = scheduler._stop_event.set

        assert scheduler.start() is True
        assert wait_for(lambda: scheduler.status()["paused_reason"]
                        == "kill_switch_engaged")
        assert scheduler.status()["running"] is False
        # stop() on a paused (self-exited) scheduler is a no-op.
        assert scheduler.stop() is False

        # Operator resolves the halt, then MANUALLY restarts.
        session.result = {"status": "ok", "reports": []}
        session.evaluated.clear()
        assert scheduler.start() is True
        assert session.evaluated.wait(timeout=5)
        status = scheduler.status()
        assert status["running"] is True
        assert status["paused_reason"] is None

        assert scheduler.stop() is True
        assert scheduler._thread.is_alive() is False
        assert audit.events() == [
            "scheduler_started", "scheduler_paused",
            "scheduler_started", "scheduler_stopped"]


# ==========================================================================
# Error storm
# ==========================================================================

class TestErrorStorm:
    def test_pauses_after_max_consecutive_errors(self):
        audit = FakeAudit()
        scheduler, session, _, sleep = make_sync(
            WED_OPEN, max_consecutive_errors=3, audit=audit)
        session.raises = RuntimeError("broker exploded")
        scheduler._run()

        assert len(session.calls) == 3
        # Failed cycles still tick on the interval, never hot-spin.
        assert session.calls == [
            ny(2026, 6, 10, 10, 0),
            ny(2026, 6, 10, 10, 15),
            ny(2026, 6, 10, 10, 30),
        ]
        status = scheduler.status()
        assert status["running"] is False
        assert status["paused_reason"] == "error_storm"
        assert status["consecutive_errors"] == 3
        assert ("scheduler", "scheduler_paused", {"reason": "error_storm"}) \
            in audit.rows

    def test_success_resets_the_consecutive_counter(self):
        scheduler, session, _, _ = make_sync(
            WED_OPEN, stop_after=3, max_consecutive_errors=5)
        session.raises = RuntimeError("transient")
        session.raise_times = 2  # fail twice, then succeed
        scheduler._run()

        status = scheduler.status()
        assert status["paused_reason"] is None
        assert status["consecutive_errors"] == 0
        assert status["runs_today"] == 1  # only the successful cycle counts

    def test_calendar_failure_feeds_the_same_pause_path(self):
        """A clock outside the static 2026/2027 calendar makes
        is_market_open raise — counted like any other error, then paused
        (fail loud, never a silently dead thread)."""
        scheduler, session, _, _ = make_sync(
            datetime(2028, 1, 5, 12, 0, tzinfo=NY),
            max_consecutive_errors=2)
        scheduler._run()

        assert session.calls == []  # never evaluated against bad hours
        assert scheduler.status()["paused_reason"] == "error_storm"


# ==========================================================================
# Lifecycle: idempotent start/stop, audit, no thread leak
# ==========================================================================

def _scheduler_threads():
    return [t for t in threading.enumerate()
            if t.name == "live-scheduler" and t.is_alive()]


class TestLifecycle:
    def test_status_shape(self):
        scheduler, _, _, _ = make_sync()
        assert set(scheduler.status()) == {
            "running", "last_run", "last_result", "next_run_estimate",
            "runs_today", "consecutive_errors", "paused_reason"}

    @pytest.mark.parametrize("bad_interval", [0, -5])
    def test_rejects_non_positive_interval(self, bad_interval):
        clock = FakeClock(WED_OPEN)
        with pytest.raises(ValueError, match="interval_minutes"):
            LiveScheduler(FakeSession(clock), interval_minutes=bad_interval,
                          clock=clock)

    def test_start_stop_idempotent_and_no_thread_leak(self):
        """Real daemon thread on a closed market: the default sleep waits
        on the stop event, so stop() interrupts the weekend sleep
        immediately and JOINS the thread."""
        audit = FakeAudit()
        clock = FakeClock(SATURDAY)
        session = FakeSession(clock)
        scheduler = LiveScheduler(session, interval_minutes=15, clock=clock,
                                  audit=audit)
        before = len(_scheduler_threads())

        assert scheduler.start() is True
        assert scheduler.start() is False  # idempotent: second is a no-op
        assert len(_scheduler_threads()) == before + 1

        # While parked outside hours the estimate is the next open.
        assert wait_for(lambda: scheduler.status()["next_run_estimate"]
                        == MONDAY_OPEN.isoformat())
        assert session.calls == []  # weekend: never evaluated

        assert scheduler.stop() is True
        assert scheduler.stop() is False  # idempotent
        assert scheduler._thread.is_alive() is False  # joined, not leaked
        assert len(_scheduler_threads()) == before

        status = scheduler.status()
        assert status["running"] is False
        assert status["next_run_estimate"] is None
        assert audit.events() == ["scheduler_started", "scheduler_stopped"]

    def test_stop_before_start_is_a_quiet_noop(self):
        audit = FakeAudit()
        scheduler, _, _, _ = make_sync(audit=audit)
        assert scheduler.stop() is False
        assert audit.rows == []

    def test_works_without_an_audit_log(self):
        scheduler, session, _, _ = make_sync(WED_OPEN, stop_after=1,
                                             audit=None)
        scheduler._run()  # must not raise anywhere on the audit path
        assert len(session.calls) == 1


# ==========================================================================
# Opt-in controlled-deployment activation handshake
# ==========================================================================

class TestFirstCycleHandshake:
    def test_returns_the_first_successful_session_result(self):
        clock = FakeClock(WED_OPEN)
        result = {"status": "ok", "reports": [], "generated": 0}
        session = FakeSession(clock, result=result)
        scheduler = LiveScheduler(session, interval_minutes=15, clock=clock)

        assert scheduler.start() is True
        assert scheduler.wait_for_first_cycle(1.0) == result
        assert scheduler.stop() is True

    def test_surfaces_first_exception_and_one_error_policy_pauses(self):
        clock = FakeClock(WED_OPEN)
        session = FakeSession(clock)
        session.raises = RuntimeError("feed unavailable")
        scheduler = LiveScheduler(
            session, interval_minutes=15, clock=clock,
            max_consecutive_errors=1,
        )

        assert scheduler.start() is True
        outcome = scheduler.wait_for_first_cycle(1.0)
        assert outcome == {
            "status": "error",
            "reason": "evaluation_exception",
            "error_type": "RuntimeError",
            "error": "feed unavailable",
        }
        assert wait_for(
            lambda: scheduler.status()["paused_reason"] == "error_storm")
        assert scheduler.status()["running"] is False

    def test_closed_market_times_out_without_an_evaluation(self):
        clock = FakeClock(SATURDAY)
        session = FakeSession(clock)
        scheduler = LiveScheduler(session, interval_minutes=15, clock=clock)

        assert scheduler.start() is True
        with pytest.raises(TimeoutError, match="first cycle"):
            scheduler.wait_for_first_cycle(0.02)
        assert session.calls == []
        assert scheduler.stop() is True

    def test_opt_in_hold_prevents_a_second_cycle_until_released(self):
        clock = FakeClock(WED_OPEN)
        session = FakeSession(clock)
        scheduler = LiveScheduler(
            session, interval_minutes=15, clock=clock,
            hold_after_first_cycle=True,
        )

        assert scheduler.start() is True
        assert scheduler.wait_for_first_cycle(1.0)["status"] == "ok"
        _time.sleep(0.02)
        assert len(session.calls) == 1
        assert scheduler.status()["next_run_estimate"] is None
        assert scheduler.release_after_first_cycle() is True
        assert scheduler.release_after_first_cycle() is False
        assert scheduler.stop() is True

    @pytest.mark.parametrize("timeout", [0, -1, float("inf"), True])
    def test_requires_a_positive_finite_handshake_timeout(self, timeout):
        scheduler, _, _, _ = make_sync()
        with pytest.raises(ValueError, match="first-cycle timeout"):
            scheduler.wait_for_first_cycle(timeout)

    def test_hold_option_is_explicitly_boolean(self):
        clock = FakeClock(WED_OPEN)
        with pytest.raises(ValueError, match="hold_after_first_cycle"):
            LiveScheduler(
                FakeSession(clock), clock=clock,
                hold_after_first_cycle=1,
            )
