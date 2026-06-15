"""
Token keep-alive scheduler — keeps an E*TRADE OAuth session warm.

DELIBERATELY DECOUPLED FROM THE TRADING LOOP. Keeping a token alive
(clearing E*TRADE's ~2h idle timeout via EtradeAuthManager.renew()) must
NOT require a desk, broker, portfolio, or data feed, and must NEVER place
an order. This loop calls renew() on a timer while the NYSE regular
session is open and does nothing else — so "consistent access" is a pure
session-liveness concern, not a self-materializing trading loop.

renew() is a no-op (returns False fast) unless there is an active token,
and refuses a day-stale token before any network call, so a keep-alive
cycle can never resurrect or mutate a token — it only clears the idle
timeout of a token the operator already connected. The MIDNIGHT-ET HARD
EXPIRY still forces a daily human re-auth; keep-alive cannot and must not
bypass that. When the token crosses the ET day boundary the loop PAUSES
with paused_reason='token_expired' (a loud signal for the morning
reconnect ritual), exactly like the trading scheduler pauses on a halt:
only a deliberate start() (or the connect flow) resumes it.

Off-hours the loop sleeps in bounded chunks (<= 60s) until the next open,
so stop() is always responsive and nothing renews when there is no
trading session to keep warm (the token dies at midnight regardless).

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
from utils.scheduler import MAX_SLEEP_CHUNK

logger = logging.getLogger(__name__)

#: Keep-alive cadence bounds (minutes). The upper bound stays WELL under
#: E*TRADE's ~2h (120 min) idle timeout — even at the maximum, a couple of
#: consecutive transient renew failures still leave margin before the token
#: idles out. A longer interval would defeat the loop's purpose.
KEEPALIVE_INTERVAL_MIN = 1
KEEPALIVE_INTERVAL_MAX = 60


class TokenKeepAliveScheduler:
    """Market-hours renew() loop with pause-on-expiry semantics.

    Args:
        auth_manager: an EtradeAuthManager (anything exposing ``status()``
            -> {'state': ...} per contract C16 and ``renew()`` -> bool).
        market_hours: MarketHours calendar; defaults to a fresh one.
        interval_minutes: minutes between renews while open
            (KEEPALIVE_INTERVAL_MIN..KEEPALIVE_INTERVAL_MAX).
        clock: callable -> AWARE datetime; defaults to UTC now.
        sleep_fn: callable(seconds); defaults to waiting on the internal
            stop event (instantly interruptible by stop()). An injected
            sleep_fn does NOT see the stop event, so it must return promptly
            (<= one chunk) for stop() to stay responsive — only tests inject
            one; production uses the event-backed default.
        max_consecutive_failures: consecutive renew failures that trip the
            'renew_failure_storm' pause (>= 1).
        audit: optional utils.audit.AuditLog for lifecycle + failure rows.
    """

    def __init__(self, auth_manager,
                 market_hours: Optional[MarketHours] = None,
                 interval_minutes: float = 15,
                 clock: Optional[Callable[[], datetime]] = None,
                 sleep_fn: Optional[Callable[[float], None]] = None,
                 max_consecutive_failures: int = 5,
                 audit=None):
        if not (KEEPALIVE_INTERVAL_MIN <= interval_minutes
                <= KEEPALIVE_INTERVAL_MAX):
            raise ValueError(
                f"interval_minutes must be between {KEEPALIVE_INTERVAL_MIN} "
                f"and {KEEPALIVE_INTERVAL_MAX} (under E*TRADE's 2h idle "
                f"timeout); got {interval_minutes}")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        self.auth_manager = auth_manager
        self.market_hours = market_hours or MarketHours()
        self.interval_minutes = interval_minutes
        self.max_consecutive_failures = max_consecutive_failures
        self.audit = audit
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep_fn = sleep_fn

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused_reason: Optional[str] = None
        self._last_renew: Optional[datetime] = None
        self._last_renew_ok: Optional[bool] = None
        self._last_state: Optional[str] = None
        self._next_run_estimate: Optional[datetime] = None
        self._renews_today = 0
        self._renews_today_date: Optional[date] = None
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Lifecycle (thread-safe, idempotent)
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Start (or manually restart after a pause). Idempotent: returns
        False without side effects when the loop is already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("Keep-alive already running; start() ignored")
                return False
            self._stop_event = threading.Event()
            self._running = True
            self._paused_reason = None
            self._consecutive_failures = 0
            self._thread = threading.Thread(
                target=self._run, name="token-keepalive", daemon=True)
            interval = self.interval_minutes
            self._thread.start()
        logger.info("Keep-alive started (renew every %s min during market "
                    "hours)", interval)
        self._audit("keepalive_started", {"interval_minutes": interval})
        return True

    def stop(self, join_timeout: float = 10.0) -> bool:
        """Stop the loop and JOIN the thread. Idempotent: returns False
        when nothing was running (already stopped, or paused)."""
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                logger.info("Keep-alive not running; stop() ignored")
                return False
            self._stop_event.set()
        if thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
            if thread.is_alive():  # pragma: no cover - defensive
                logger.error("Keep-alive thread did not stop within %.1fs",
                             join_timeout)
        with self._lock:
            # Only clear shared state if a concurrent start() hasn't already
            # installed a NEWER generation — otherwise we'd report
            # running=False while that fresh loop is live.
            if self._thread is thread:
                self._running = False
                self._next_run_estimate = None
        logger.info("Keep-alive stopped")
        self._audit("keepalive_stopped", {})
        return True

    def status(self) -> Dict:
        """Snapshot for dashboards (gui /api/live/keepalive)."""
        with self._lock:
            running = (self._running and self._thread is not None
                       and self._thread.is_alive())
            # Re-bucket on READ so a paused/idle loop across the ET-midnight
            # roll reports 0 — not yesterday's carryover (the count is only
            # written-back on the next successful renew, which may not come
            # until the morning re-auth).
            today = self._clock().astimezone(NYSE_TZ).date()
            renews_today = (self._renews_today
                            if self._renews_today_date == today else 0)
            return {
                "running": running,
                "last_renew": (self._last_renew.isoformat()
                               if self._last_renew else None),
                "last_renew_ok": self._last_renew_ok,
                "last_state": self._last_state,
                "renews_today": renews_today,
                "consecutive_failures": self._consecutive_failures,
                "next_run_estimate": (self._next_run_estimate.isoformat()
                                      if running and self._next_run_estimate
                                      else None),
                "paused_reason": self._paused_reason,
                "interval_minutes": self.interval_minutes,
            }

    # ------------------------------------------------------------------
    # Loop body (daemon thread)
    # ------------------------------------------------------------------
    def _run(self) -> None:
        current = threading.current_thread()
        try:
            while not self._stop_event.is_set():
                try:
                    now = self._clock()
                    if not self.market_hours.is_market_open(now):
                        self._sleep_until_open(now)
                        continue
                    pause_reason = self._renew_cycle(now)
                except Exception:  # noqa: BLE001 - calendar/clock error PAUSES loudly, never a silent thread death
                    logger.exception("Keep-alive loop body raised")
                    self._pause("loop_error")
                    return
                if pause_reason is not None:
                    self._pause(pause_reason)
                    return
                self._wait_interval(now)
        finally:
            with self._lock:
                # Clear shared state only if a concurrent start() hasn't
                # already installed a newer loop generation (the synchronous
                # test path has _thread is None and is allowed through).
                if self._thread is None or self._thread is current:
                    self._running = False
                    self._next_run_estimate = None

    def _renew_cycle(self, now: datetime) -> Optional[str]:
        """One renew attempt. Returns a pause-reason string, or None to
        keep looping. NEVER raises — a renew that blows up is counted as a
        failure, never a thread-killing exception."""
        state = self._safe_state()
        if state == "connected":
            try:
                ok = self.auth_manager.renew()
            except Exception as e:  # noqa: BLE001 - counted, never fatal
                logger.warning("Keep-alive renew raised: %s", e)
                self._note_failure(now, state,
                                   f"renew_raised:{type(e).__name__}")
                return self._storm_reason()
            if ok:
                self._record_success(now)
                return None
            # renew() declined: the most common cause is the midnight-ET
            # roll striking between cycles — re-read state to tell a hard
            # expiry (pause for re-auth) from a transient decline (count).
            state2 = self._safe_state()
            if state2 == "expired":
                self._record_expired(now)
                return "token_expired"
            self._note_failure(now, state2, "renew_declined")
            return self._storm_reason()
        if state == "expired":
            self._record_expired(now)
            return "token_expired"
        # disconnected / unconfigured / pending_verifier: nothing to keep
        # warm yet — idle (no renew, no failure) and wait for a connection.
        with self._lock:
            self._last_state = state
        return None

    def _storm_reason(self) -> Optional[str]:
        """'renew_failure_storm' once the consecutive-failure threshold is
        hit, else None (keep looping)."""
        with self._lock:
            hit = self._consecutive_failures >= self.max_consecutive_failures
        return "renew_failure_storm" if hit else None

    def _safe_state(self) -> Optional[str]:
        """auth_manager.status()['state'] — C16 promises status() never
        raises, but guard defensively so a broken manager can't kill the
        loop (it surfaces as a counted failure instead)."""
        try:
            return (self.auth_manager.status() or {}).get("state")
        except Exception as e:  # noqa: BLE001 - defensive
            logger.warning("Keep-alive could not read auth status: %s", e)
            return None

    def _pause(self, reason: str) -> None:
        """Pause path: running goes false, the reason sticks, and ONLY a
        manual start() (or the connect flow) resumes — no auto-resume.

        Generation-guarded: a superseded thread (one a concurrent start()
        has already replaced) must NOT stomp the fresh loop's state. The
        synchronous test path (no thread installed) is allowed through."""
        current = threading.current_thread()
        with self._lock:
            if self._thread is not None and self._thread is not current:
                return
            self._running = False
            self._paused_reason = reason
            self._next_run_estimate = None
        logger.warning("Keep-alive PAUSED (%s) — re-auth / manual start() "
                       "required", reason)
        self._audit("keepalive_paused", {"reason": reason})

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def _record_success(self, now: datetime) -> None:
        local_date = now.astimezone(NYSE_TZ).date()
        with self._lock:
            if self._renews_today_date != local_date:
                self._renews_today_date = local_date
                self._renews_today = 0
            self._renews_today += 1
            self._last_renew = now
            self._last_renew_ok = True
            self._last_state = "connected"
            self._consecutive_failures = 0

    def _note_failure(self, now: datetime, state: Optional[str],
                      reason: str) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_renew = now
            self._last_renew_ok = False
            self._last_state = state
            consecutive = self._consecutive_failures
        logger.warning("Keep-alive renew failed (%s); %d consecutive",
                       reason, consecutive)
        self._audit("keepalive_renew_failed",
                    {"reason": reason, "consecutive": consecutive})

    def _record_expired(self, now: datetime) -> None:
        with self._lock:
            self._last_renew = now
            self._last_renew_ok = False
            self._last_state = "expired"

    # ------------------------------------------------------------------
    # Sleeping (chunked <= MAX_SLEEP_CHUNK, responsive to stop())
    # ------------------------------------------------------------------
    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._sleep_fn is not None:
            self._sleep_fn(seconds)
        else:
            self._stop_event.wait(seconds)

    def _wait_interval(self, started_at: datetime) -> None:
        target = started_at + timedelta(minutes=self.interval_minutes)
        with self._lock:
            self._next_run_estimate = target
        while not self._stop_event.is_set():
            remaining = (target - self._clock()).total_seconds()
            if remaining <= 0:
                return
            self._sleep(min(remaining, MAX_SLEEP_CHUNK))

    def _sleep_until_open(self, now: datetime) -> None:
        next_open = self.market_hours.next_market_open(now)
        with self._lock:
            self._next_run_estimate = next_open
        if (next_open - now).total_seconds() <= 0:
            # Degenerate calendar: the market reads closed yet the next open
            # is not in the future. Don't hot-spin the core — wait one
            # bounded chunk (still stop()-interruptible) before re-checking.
            logger.warning("Keep-alive: next_market_open (%s) is not in the "
                           "future while closed; sleeping one chunk to avoid "
                           "a busy loop", next_open.isoformat())
            self._sleep(MAX_SLEEP_CHUNK)
            return
        logger.info("Keep-alive idle; market closed, next open %s",
                    next_open.isoformat())
        while not self._stop_event.is_set():
            remaining = (next_open - self._clock()).total_seconds()
            if remaining <= 0:
                return
            self._sleep(min(remaining, MAX_SLEEP_CHUNK))

    # ------------------------------------------------------------------
    def _audit(self, event_type: str, payload: Dict) -> None:
        """Best-effort lifecycle audit — a failing audit write must never
        kill the keep-alive thread, but it is never silent either."""
        if self.audit is None:
            return
        try:
            self.audit.append("keepalive", event_type, payload)
        except Exception:  # noqa: BLE001 - log loudly, keep running
            logger.exception("Failed to audit %s", event_type)
