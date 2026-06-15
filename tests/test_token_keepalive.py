"""Tests for utils/token_keepalive.py (TokenKeepAliveScheduler).

Deterministic, mirroring tests/test_scheduler.py: branch logic is checked
by calling _renew_cycle() directly; loop cadence / pause behavior is
exercised by running _run() synchronously with an injected fake clock +
fake sleep (the fake sleep ADVANCES the clock, so virtual time only moves
when the loop sleeps); start/stop idempotency uses real daemon threads
where the default stop-event sleep makes stop() interrupt instantly.

The keep-alive loop NEVER trades — it is constructed with ONLY an auth
manager (no desk, broker, portfolio, or data feed), which every test here
relies on.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from utils.scheduler import MAX_SLEEP_CHUNK
from utils.token_keepalive import (
    KEEPALIVE_INTERVAL_MAX,
    KEEPALIVE_INTERVAL_MIN,
    TokenKeepAliveScheduler,
)

NY = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _no_leaked_keepalive_threads():
    """Backstop: fail loudly if any test leaves a token-keepalive daemon
    running — a leaked renew loop would mutate shared fakes in later tests.
    The real-thread tests stop() in a finally, so this should never trip."""
    yield
    alive = [t for t in threading.enumerate()
             if t.name == "token-keepalive" and t.is_alive()]
    assert not alive, (
        f"{len(alive)} token-keepalive thread(s) leaked — a test failed to "
        f"stop() its scheduler")


def ny(y, m, d, hh, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


WED_OPEN = ny(2026, 6, 10, 10, 0)


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


class FakeMarketHours:
    """is_market_open is either a fixed bool (always=) or a 'now >= open_at'
    gate (open_at=), so a closed->open transition is driven by the clock."""

    def __init__(self, always=None, open_at=None):
        self.always = always
        self.open_at = open_at

    def is_market_open(self, now):
        if self.always is not None:
            return self.always
        return now >= self.open_at

    def next_market_open(self, now):
        return self.open_at if self.open_at is not None else now


class FakeAuth:
    """Minimal EtradeAuthManager stand-in: status() -> {'state': ...} and
    renew() -> bool. renew() can raise, can flip the state to 'expired'
    after N calls (simulating the midnight roll), and runs an on_renew
    hook (used to stop the synchronous loop after a set number of cycles).
    """

    def __init__(self, state="connected", renew_result=True,
                 renew_raises=None, expire_after=None, status_raise_after=None):
        self.state = state
        self.renew_result = renew_result
        self.renew_raises = renew_raises
        self.expire_after = expire_after
        # status() raises on the Nth call and after (simulates a broken
        # manager / transient status failure mid-cycle).
        self.status_raise_after = status_raise_after
        self.renew_calls = 0
        self.status_calls = 0
        self.on_renew = lambda: None

    def status(self):
        self.status_calls += 1
        if (self.status_raise_after is not None
                and self.status_calls >= self.status_raise_after):
            raise RuntimeError("status boom")
        return {"state": self.state, "env": "sandbox"}

    def renew(self):
        self.renew_calls += 1
        self.on_renew()
        if self.renew_raises is not None:
            raise self.renew_raises
        if (self.expire_after is not None
                and self.renew_calls >= self.expire_after):
            self.state = "expired"
        return self.renew_result


def make_sync(auth, *, market=None, interval_minutes=15,
              max_consecutive_failures=5, audit=None, start=WED_OPEN):
    """A keep-alive scheduler wired for SYNCHRONOUS _run()."""
    clock = FakeClock(start)
    sleep = FakeSleep(clock)
    sched = TokenKeepAliveScheduler(
        auth, market_hours=market or FakeMarketHours(always=True),
        interval_minutes=interval_minutes,
        max_consecutive_failures=max_consecutive_failures,
        clock=clock, sleep_fn=sleep, audit=audit)
    return sched, clock, sleep


def wait_for(predicate, timeout=5.0):
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.01)
    return False


# ==========================================================================
# _renew_cycle branch logic (no threading)
# ==========================================================================

class TestRenewCycleBranches:
    def test_connected_renew_ok(self):
        auth = FakeAuth(state="connected", renew_result=True)
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) is None
        assert auth.renew_calls == 1
        st = sched.status()
        assert st["last_renew_ok"] is True
        assert st["renews_today"] == 1
        assert st["consecutive_failures"] == 0
        assert st["last_state"] == "connected"

    def test_connected_renew_raises_counts_failure(self):
        auth = FakeAuth(renew_raises=RuntimeError("boom"))
        audit = FakeAudit()
        sched, clock, _ = make_sync(auth, audit=audit)
        assert sched._renew_cycle(clock()) is None  # below storm threshold
        st = sched.status()
        assert st["consecutive_failures"] == 1
        assert st["last_renew_ok"] is False
        assert "keepalive_renew_failed" in audit.events()

    def test_connected_renew_false_still_connected_counts_failure(self):
        auth = FakeAuth(renew_result=False)  # stays connected
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) is None
        assert sched.status()["consecutive_failures"] == 1

    def test_connected_renew_false_then_expired_pauses(self):
        # renew() flips the state to expired (the midnight roll) and returns
        # False -> the cycle re-reads status and pauses for re-auth.
        auth = FakeAuth(renew_result=False, expire_after=1)
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) == "token_expired"
        assert sched.status()["last_state"] == "expired"
        # An expiry is NOT a renew failure — no failure counted.
        assert sched.status()["consecutive_failures"] == 0

    def test_expired_from_start_pauses_without_renewing(self):
        auth = FakeAuth(state="expired")
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) == "token_expired"
        assert auth.renew_calls == 0  # never renews a dead token

    @pytest.mark.parametrize("state",
                             ["disconnected", "unconfigured",
                              "pending_verifier"])
    def test_idle_states_skip_without_renew_or_pause(self, state):
        auth = FakeAuth(state=state)
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) is None  # idle, keep looping
        assert auth.renew_calls == 0
        assert sched.status()["last_state"] == state
        assert sched.status()["consecutive_failures"] == 0

    def test_storm_threshold_returns_pause_reason(self):
        auth = FakeAuth(renew_raises=RuntimeError("boom"))
        sched, clock, _ = make_sync(auth, max_consecutive_failures=3)
        assert sched._renew_cycle(clock()) is None  # 1
        assert sched._renew_cycle(clock()) is None  # 2
        assert sched._renew_cycle(clock()) == "renew_failure_storm"  # 3
        assert sched.status()["consecutive_failures"] == 3

    def test_success_resets_failure_counter(self):
        auth = FakeAuth(renew_raises=RuntimeError("boom"))
        sched, clock, _ = make_sync(auth)
        sched._renew_cycle(clock())
        assert sched.status()["consecutive_failures"] == 1
        auth.renew_raises = None  # now succeeds
        sched._renew_cycle(clock())
        assert sched.status()["consecutive_failures"] == 0


# ==========================================================================
# Loop cadence + pauses (synchronous _run)
# ==========================================================================

class TestLoop:
    def test_renews_each_cycle_while_open(self):
        auth = FakeAuth(state="connected", renew_result=True)
        sched, clock, sleep = make_sync(auth, interval_minutes=15)
        # Stop after 4 renews via the on_renew hook.
        auth.on_renew = lambda: (auth.renew_calls >= 4
                                 and sched._stop_event.set())
        sched._run()
        assert auth.renew_calls == 4
        assert sched.status()["renews_today"] == 4
        assert sched.status()["running"] is False
        assert sched.status()["paused_reason"] is None  # clean stop
        # Every sleep is a bounded chunk (<= 60s).
        assert sleep.calls and all(s <= 60.0 for s in sleep.calls)

    def test_pauses_immediately_on_expired(self):
        auth = FakeAuth(state="expired")
        audit = FakeAudit()
        sched, _, _ = make_sync(auth, audit=audit)
        sched._run()
        st = sched.status()
        assert st["running"] is False
        assert st["paused_reason"] == "token_expired"
        assert auth.renew_calls == 0
        assert "keepalive_paused" in audit.events()

    def test_pauses_on_renew_failure_storm(self):
        auth = FakeAuth(renew_raises=RuntimeError("boom"))
        audit = FakeAudit()
        sched, _, _ = make_sync(auth, max_consecutive_failures=3, audit=audit)
        sched._run()
        st = sched.status()
        assert st["paused_reason"] == "renew_failure_storm"
        assert st["consecutive_failures"] == 3
        assert auth.renew_calls == 3
        assert audit.events().count("keepalive_renew_failed") == 3
        assert "keepalive_paused" in audit.events()

    def test_does_not_renew_while_market_closed(self):
        auth = FakeAuth(state="connected", renew_result=True)
        # Opens 120s after start; closed before then.
        market = FakeMarketHours(open_at=WED_OPEN + timedelta(seconds=120))
        clock = FakeClock(WED_OPEN)
        sleep = FakeSleep(clock)
        sched = TokenKeepAliveScheduler(
            auth, market_hours=market, interval_minutes=15,
            clock=clock, sleep_fn=sleep)
        auth.on_renew = lambda: sched._stop_event.set()  # stop on first renew
        sched._run()
        assert auth.renew_calls == 1
        # The renew only happened once the clock reached the open time.
        assert clock() >= market.open_at


# ==========================================================================
# Construction validation
# ==========================================================================

class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1, KEEPALIVE_INTERVAL_MAX + 1, 120])
    def test_rejects_out_of_range_interval(self, bad):
        with pytest.raises(ValueError):
            TokenKeepAliveScheduler(FakeAuth(), interval_minutes=bad)

    @pytest.mark.parametrize("ok", [KEEPALIVE_INTERVAL_MIN, 15,
                                    KEEPALIVE_INTERVAL_MAX])
    def test_accepts_in_range_interval(self, ok):
        TokenKeepAliveScheduler(FakeAuth(), interval_minutes=ok)

    def test_rejects_zero_max_failures(self):
        with pytest.raises(ValueError):
            TokenKeepAliveScheduler(FakeAuth(), max_consecutive_failures=0)


# ==========================================================================
# Lifecycle (real daemon threads)
# ==========================================================================

class TestLifecycle:
    def test_start_stop_idempotent(self):
        # Market permanently closed so the loop just sleeps (no renew churn),
        # making this a pure thread-lifecycle test.
        auth = FakeAuth()
        audit = FakeAudit()
        sched = TokenKeepAliveScheduler(
            auth, market_hours=FakeMarketHours(always=False),
            interval_minutes=15, audit=audit)
        try:
            assert sched.start() is True
            assert wait_for(lambda: sched.status()["running"])
            assert sched.start() is False  # already running
            assert sched.stop() is True
            assert sched.status()["running"] is False
            assert sched.stop() is False  # already stopped
            assert "keepalive_started" in audit.events()
            assert "keepalive_stopped" in audit.events()
        finally:
            sched.stop()
        assert not any(t.name == "token-keepalive" and t.is_alive()
                       for t in threading.enumerate())

    def test_restart_after_pause(self):
        # Expired -> pauses on the first cycle; a manual start() resumes it.
        auth = FakeAuth(state="expired")
        sched = TokenKeepAliveScheduler(
            auth, market_hours=FakeMarketHours(always=True),
            interval_minutes=15)
        try:
            sched.start()
            assert wait_for(
                lambda: sched.status()["paused_reason"] == "token_expired")
            assert sched.status()["running"] is False
            # Re-auth happened (state connected again); start() clears pause.
            auth.state = "connected"
            assert sched.start() is True
            assert wait_for(lambda: sched.status()["running"])
            assert sched.status()["paused_reason"] is None
        finally:
            sched.stop()


# ==========================================================================
# Defensive paths + robustness (review hardening)
# ==========================================================================

class TestDefensiveAndRobustness:
    def test_first_status_raises_idles_without_crashing(self):
        # A broken manager whose status() raises surfaces as an idle skip
        # (state None -> the else branch), never a thread-killing exception.
        auth = FakeAuth(status_raise_after=1)
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) is None
        assert auth.renew_calls == 0

    def test_renew_declined_then_status_raises_counts_failure(self):
        # connected -> renew() False -> re-read status() raises -> counted as
        # a failure (last_state None), not an expiry, not a crash.
        auth = FakeAuth(renew_result=False, status_raise_after=2)
        sched, clock, _ = make_sync(auth)
        assert sched._renew_cycle(clock()) is None
        st = sched.status()
        assert st["consecutive_failures"] == 1
        assert st["last_state"] is None

    def test_calendar_error_pauses_loudly(self):
        # market_hours raising (e.g. a date outside the static NYSE calendar)
        # must PAUSE with a distinct reason, not silently kill the daemon.
        class BoomMarket:
            def is_market_open(self, now):
                raise ValueError("year out of calendar range")

            def next_market_open(self, now):
                raise ValueError("year out of calendar range")

        clock = FakeClock(WED_OPEN)
        audit = FakeAudit()
        sched = TokenKeepAliveScheduler(
            FakeAuth(), market_hours=BoomMarket(), interval_minutes=15,
            clock=clock, sleep_fn=FakeSleep(clock), audit=audit)
        sched._run()
        st = sched.status()
        assert st["running"] is False
        assert st["paused_reason"] == "loop_error"
        assert "keepalive_paused" in audit.events()

    def test_degenerate_calendar_does_not_busy_spin(self):
        # next_market_open not in the future while closed -> exactly one
        # bounded chunk sleep, then return (no hot loop).
        class DegenerateMarket:
            def is_market_open(self, now):
                return False

            def next_market_open(self, now):
                return now  # not in the future

        clock = FakeClock(WED_OPEN)
        sleep = FakeSleep(clock)
        sched = TokenKeepAliveScheduler(
            FakeAuth(), market_hours=DegenerateMarket(), interval_minutes=15,
            clock=clock, sleep_fn=sleep)
        sched._sleep_until_open(clock())
        assert sleep.calls == [MAX_SLEEP_CHUNK]

    def test_audit_failure_never_kills_loop(self):
        class RaisingAudit:
            def append(self, *_a, **_k):
                raise RuntimeError("audit down")

        clock = FakeClock(WED_OPEN)
        sched = TokenKeepAliveScheduler(
            FakeAuth(state="expired"),
            market_hours=FakeMarketHours(always=True), interval_minutes=15,
            clock=clock, sleep_fn=FakeSleep(clock), audit=RaisingAudit())
        sched._run()  # must not raise despite audit.append blowing up
        assert sched.status()["paused_reason"] == "token_expired"

    def test_renews_today_resets_across_et_midnight_in_status(self):
        clock = FakeClock(WED_OPEN)
        sched = TokenKeepAliveScheduler(
            FakeAuth(), market_hours=FakeMarketHours(always=True),
            interval_minutes=15, clock=clock, sleep_fn=FakeSleep(clock))
        sched._record_success(clock())
        assert sched.status()["renews_today"] == 1
        # Advance past the ET-midnight roll: status() must read 0 even though
        # no new renew has happened (the paused-across-midnight case).
        clock.now = clock.now + timedelta(days=1)
        assert sched.status()["renews_today"] == 0

    def test_stop_from_within_loop_thread_does_not_deadlock(self):
        # stop() called from the daemon thread itself must not self-join
        # deadlock (the `thread is not current_thread()` guard).
        auth = FakeAuth(state="connected", renew_result=True)
        sched = TokenKeepAliveScheduler(
            auth, market_hours=FakeMarketHours(always=True),
            interval_minutes=15)  # default event-backed sleep
        stopped = []
        auth.on_renew = lambda: stopped.append(sched.stop(join_timeout=1.0))
        try:
            sched.start()
            assert wait_for(lambda: stopped, timeout=5.0)
            assert stopped[0] is True  # stopped itself, no deadlock
            assert wait_for(lambda: not sched.status()["running"])
        finally:
            sched.stop()
