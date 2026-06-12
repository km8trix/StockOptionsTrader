"""
Market-hours scheduler — the DELIBERATE Phase 9 follow-up.

LiveScheduler wraps utils.live_session.LiveTradingSession.evaluate_once()
in a daemon-thread loop: while the NYSE regular session is open it calls
evaluate_once() every ``interval_minutes``; outside hours it sleeps in
bounded chunks (<= 60s each, so stop() is always responsive) until the
next open from utils.market_hours.MarketHours.

THE SCHEDULER NEVER BYPASSES THE SAFETY RAILS — it sits ABOVE them. The
kill switch, daily-loss circuit breaker, and execution-error handling all
live inside evaluate_once(), which converts every halt into a
``{'status': 'halted', 'reason': ...}`` result (it never raises for
those). When the scheduler sees a halted result it PAUSES: the loop ends,
status() reports running=False with the halt reason in paused_reason, an
audited 'scheduler_paused' row is written, and ONLY a manual start() (a
human decision, after resolving the halt) resumes the loop. An engaged
kill switch therefore stops automation after at most one more cycle —
the scheduler never spins retries against a thrown switch.

ERROR STORM: an exception escaping evaluate_once() (or the calendar) is
counted; after ``max_consecutive_errors`` consecutive failures the same
pause path fires with paused_reason='error_storm'. A single success
resets the counter.

Every lifecycle transition is audited (actor 'scheduler') when an
AuditLog is provided: scheduler_started / scheduler_stopped /
scheduler_paused.

Injectable for deterministic tests: ``clock`` (aware-datetime callable),
``sleep_fn`` (seconds callable; default waits on the internal stop event
so real sleeps abort instantly on stop()), and ``market_hours``.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from utils.market_hours import NYSE_TZ, MarketHours

logger = logging.getLogger(__name__)

#: Longest single sleep (seconds) — keeps stop() responsive even when a
#: plain time.sleep is injected, and bounds the lag after clock jumps.
MAX_SLEEP_CHUNK = 60.0


class LiveScheduler:
    """Market-hours evaluate_once() loop with pause-on-halt semantics.

    Args:
        session: LiveTradingSession (anything with ``evaluate_once()``
            returning the session's result dict).
        market_hours: MarketHours calendar; defaults to a fresh one.
        interval_minutes: minutes between evaluations while open (> 0).
        clock: callable -> AWARE datetime; defaults to UTC now.
        sleep_fn: callable(seconds); defaults to waiting on the internal
            stop event (instantly interruptible by stop()).
        max_consecutive_errors: consecutive evaluate_once() exceptions
            that trip the 'error_storm' pause (>= 1).
        audit: optional utils.audit.AuditLog for lifecycle rows.
    """

    def __init__(self, session,
                 market_hours: Optional[MarketHours] = None,
                 interval_minutes: float = 15,
                 clock: Optional[Callable[[], datetime]] = None,
                 sleep_fn: Optional[Callable[[float], None]] = None,
                 max_consecutive_errors: int = 5,
                 audit=None):
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be > 0")
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")
        self.session = session
        self.market_hours = market_hours or MarketHours()
        self.interval_minutes = interval_minutes
        self.max_consecutive_errors = max_consecutive_errors
        self.audit = audit
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep_fn = sleep_fn

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused_reason: Optional[str] = None
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[Dict] = None
        self._next_run_estimate: Optional[datetime] = None
        self._runs_today = 0
        self._runs_today_date: Optional[date] = None
        self._consecutive_errors = 0

    # ------------------------------------------------------------------
    # Lifecycle (thread-safe, idempotent)
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Start (or manually restart after a pause). Idempotent: returns
        False without side effects when the loop is already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("Scheduler already running; start() ignored")
                return False
            self._stop_event = threading.Event()
            self._running = True
            self._paused_reason = None
            self._consecutive_errors = 0
            self._thread = threading.Thread(
                target=self._run, name="live-scheduler", daemon=True)
            interval = self.interval_minutes
            self._thread.start()
        logger.info("Scheduler started (every %s min during market hours)",
                    interval)
        self._audit("scheduler_started", {"interval_minutes": interval})
        return True

    def stop(self, join_timeout: float = 10.0) -> bool:
        """Stop the loop and JOIN the thread. Idempotent: returns False
        when nothing was running (already stopped, or paused)."""
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                logger.info("Scheduler not running; stop() ignored")
                return False
            self._stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
            if thread.is_alive():  # pragma: no cover - defensive
                logger.error("Scheduler thread did not stop within %.1fs",
                             join_timeout)
        with self._lock:
            self._running = False
            self._next_run_estimate = None
        logger.info("Scheduler stopped")
        self._audit("scheduler_stopped", {})
        return True

    def status(self) -> Dict:
        """Snapshot for dashboards (gui /api/live/scheduler)."""
        with self._lock:
            running = (self._running and self._thread is not None
                       and self._thread.is_alive())
            return {
                "running": running,
                "last_run": (self._last_run.isoformat()
                             if self._last_run else None),
                "last_result": self._last_result,
                "next_run_estimate": (self._next_run_estimate.isoformat()
                                      if running and self._next_run_estimate
                                      else None),
                "runs_today": self._runs_today,
                "consecutive_errors": self._consecutive_errors,
                "paused_reason": self._paused_reason,
            }

    # ------------------------------------------------------------------
    # Loop body (daemon thread)
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    now = self._clock()
                    if not self.market_hours.is_market_open(now):
                        self._sleep_until_open(now)
                        continue
                    result = self.session.evaluate_once()
                except Exception:  # noqa: BLE001 - counted, never silent death
                    logger.exception("Scheduled evaluation failed")
                    if self._note_error():
                        self._pause("error_storm")
                        return
                    self._wait_interval(self._clock())
                    continue

                ran_at = now
                self._record_run(ran_at, result)
                if isinstance(result, dict) and result.get("status") == "halted":
                    # The session's rails (kill switch, circuit breaker,
                    # execution error) halted the cycle — automation must
                    # NOT keep retrying against a thrown rail.
                    self._pause(result.get("reason") or "session_halted")
                    return
                self._wait_interval(ran_at)
        finally:
            with self._lock:
                self._running = False
                self._next_run_estimate = None

    def _pause(self, reason: str) -> None:
        """Pause path: running goes false, the reason sticks, and ONLY a
        manual start() resumes — there is deliberately no auto-resume."""
        with self._lock:
            self._running = False
            self._paused_reason = reason
            self._next_run_estimate = None
        logger.warning("Scheduler PAUSED (%s) — manual start() required",
                       reason)
        self._audit("scheduler_paused", {"reason": reason})

    def _note_error(self) -> bool:
        """Count a consecutive failure; True when the storm threshold hit."""
        with self._lock:
            self._consecutive_errors += 1
            return self._consecutive_errors >= self.max_consecutive_errors

    def _record_run(self, ran_at: datetime, result) -> None:
        local_date = ran_at.astimezone(NYSE_TZ).date()
        with self._lock:
            if self._runs_today_date != local_date:
                self._runs_today_date = local_date
                self._runs_today = 0
            self._runs_today += 1
            self._last_run = ran_at
            self._last_result = result
            self._consecutive_errors = 0

    # ------------------------------------------------------------------
    # Sleeping (chunked <= MAX_SLEEP_CHUNK, responsive to stop())
    # ------------------------------------------------------------------
    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._sleep_fn is not None:
            self._sleep_fn(seconds)
        else:
            # Event.wait returns immediately once stop() sets the event.
            self._stop_event.wait(seconds)

    def _wait_interval(self, started_at: datetime) -> None:
        """Sleep until ``started_at + interval`` (re-reading the clock
        between bounded chunks so injected clocks and suspends both work)."""
        target = started_at + timedelta(minutes=self.interval_minutes)
        with self._lock:
            self._next_run_estimate = target
        while not self._stop_event.is_set():
            remaining = (target - self._clock()).total_seconds()
            if remaining <= 0:
                return
            self._sleep(min(remaining, MAX_SLEEP_CHUNK))

    def _sleep_until_open(self, now: datetime) -> None:
        """Outside market hours: chunked sleep until the next open."""
        next_open = self.market_hours.next_market_open(now)
        with self._lock:
            self._next_run_estimate = next_open
        logger.info("Market closed; next open %s", next_open.isoformat())
        while not self._stop_event.is_set():
            remaining = (next_open - self._clock()).total_seconds()
            if remaining <= 0:
                return
            self._sleep(min(remaining, MAX_SLEEP_CHUNK))

    # ------------------------------------------------------------------
    def _audit(self, event_type: str, payload: Dict) -> None:
        """Best-effort lifecycle audit — a failing audit write must never
        kill the scheduler thread, but it is never silent either."""
        if self.audit is None:
            return
        try:
            self.audit.append("scheduler", event_type, payload)
        except Exception:  # noqa: BLE001 - log loudly, keep running
            logger.exception("Failed to audit %s", event_type)
