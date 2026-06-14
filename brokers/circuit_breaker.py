"""
Daily-loss circuit breaker for live trading (Phase 9, E5).

One job: when today's account value has fallen limit_pct or more below
the start-of-day value, ENGAGE THE KILL SWITCH and audit the numbers.

WIRING (where check() actually runs in production):
  * DailyLossGate binds a breaker to a live account-value source and
    tracks the start-of-day value per US/Eastern calendar day. The gate
    is a zero-arg callable shaped exactly for EtradeClient's injected
    ``circuit_breaker`` hook (the client stays testable).
  * brokers.live_trader.LiveEtradeBroker builds that gate automatically
    when it constructs its own EtradeClient from an auth manager + kill
    switch (the GUI's construction path), so EVERY preview/place is
    gated.
  * utils.live_session.LiveTradingSession ALSO evaluates the gate at the
    top of evaluate_once() — before any intent executes — and halts the
    whole evaluation on a breach.

START-OF-DAY SEMANTICS: the baseline is the FIRST account value the gate
observes on each ET calendar day (a process that starts mid-day measures
from its first observation — the best a restartable process can do
without persisting opening prints). The date roll uses
zoneinfo('America/New_York'), so DST is handled by the tz database, and
the gate's clock is injectable for tests.

BOUNDARY SEMANTICS (pinned by test): the comparison is INCLUSIVE —
loss_pct >= limit_pct breaches. A day sitting at EXACTLY -2.000% is a
breach; -1.999% still trades. At the limit you stop; "one more trade to
get it back" is how drawdowns become disasters.

This DELIBERATELY diverges from the Phase 5 desk rail,
portfolio.risk_manager.RiskManager.check_daily_loss_limit, which is
STRICT (loss > limit violates; a loss exactly AT the limit still
trades). The two only disagree on the exact-boundary point, and the
divergence FAILS SAFE: this breaker is the stricter OUTER rail in live
trading, so a day pinned exactly at the limit halts here regardless of
what the desk rail would allow. Do not "align" the two conventions —
the asymmetry is intentional and documented on both sides.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Union
from zoneinfo import ZoneInfo

from utils.audit import AuditLog
from utils.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")

#: Default daily loss limit: 2% of start-of-day account value.
DEFAULT_LIMIT_PCT = 0.02


def extract_account_value(broker_balances: Union[Dict, float, int]) -> float:
    """Pull total account value from a BalanceResponse-shaped dict.

    Accepts (in priority order):
      * a plain number (already-extracted value),
      * {'BalanceResponse': {'Computed': {'RealTimeValues':
            {'totalAccountValue': x}}}} or the BalanceResponse inner dict,
      * {'portfolio_value': x} (ExecutionBroker.get_portfolio_status shape).

    Raises ValueError when no value can be found — a circuit breaker that
    silently reads 0.0 would instantly (and wrongly) kill the session.
    """
    if isinstance(broker_balances, (int, float)):
        return float(broker_balances)
    if isinstance(broker_balances, dict):
        inner = broker_balances.get("BalanceResponse", broker_balances)
        computed = inner.get("Computed", {})
        value = computed.get("RealTimeValues", {}).get("totalAccountValue")
        if value is not None:
            return float(value)
        if broker_balances.get("portfolio_value") is not None:
            return float(broker_balances["portfolio_value"])
    raise ValueError(
        "Could not extract a total account value from broker_balances")


class DailyLossCircuitBreaker:
    """Engages the kill switch when the day's loss reaches the limit.

    Args:
        kill_switch: the persistent KillSwitch to engage on breach.
        audit: AuditLog; defaults to the kill switch's.
        limit_pct: inclusive daily-loss limit as a fraction (0.02 = 2%).
    """

    def __init__(self, kill_switch: KillSwitch,
                 audit: Optional[AuditLog] = None,
                 limit_pct: float = DEFAULT_LIMIT_PCT,
                 clock: Optional[Callable[[], datetime]] = None):
        self.kill_switch = kill_switch
        self.audit = audit if audit is not None else kill_switch.audit
        self.limit_pct = limit_pct
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def check(self, broker_balances: Union[Dict, float, int],
              start_of_day_value: float,
              limit_pct: Optional[float] = None) -> Dict:
        """Evaluate the day's loss; engage the kill switch on breach.

        Returns {'breached': bool, 'loss_pct': float, 'limit_pct': float,
        'current_value': float, 'start_of_day_value': float}. loss_pct is
        POSITIVE for a loss ((start - current) / start). breached is True
        when loss_pct >= limit_pct (inclusive — see module docstring).
        """
        limit = self.limit_pct if limit_pct is None else limit_pct
        if start_of_day_value <= 0:
            raise ValueError("start_of_day_value must be positive")
        current_value = extract_account_value(broker_balances)
        loss_pct = (start_of_day_value - current_value) / start_of_day_value
        breached = loss_pct >= limit
        result = {
            "breached": breached,
            "loss_pct": loss_pct,
            "limit_pct": limit,
            "current_value": current_value,
            "start_of_day_value": start_of_day_value,
        }
        if breached:
            logger.error(
                "DAILY-LOSS CIRCUIT BREAKER: %.4f%% loss >= %.4f%% limit "
                "(start %.2f -> current %.2f); engaging kill switch",
                loss_pct * 100, limit * 100, start_of_day_value,
                current_value)
            self.audit.append("circuit_breaker", "circuit_breaker_breach", {
                "loss_pct": loss_pct,
                "limit_pct": limit,
                "current_value": current_value,
                "start_of_day_value": start_of_day_value,
                "checked_at": self._clock().isoformat(),
            })
            self.kill_switch.engage(
                reason=(f"daily-loss circuit breaker: {loss_pct:.4%} loss "
                        f">= {limit:.4%} limit"),
                actor="circuit_breaker")
        return result


class DailyLossGate:
    """Binds a DailyLossCircuitBreaker to a live account-value source.

    The gate is the zero-arg callable EtradeClient takes as its injected
    ``circuit_breaker`` hook: each call fetches the current account value
    via value_fn, lazily captures the start-of-day baseline on the first
    observation of each US/Eastern calendar day (audited as
    'start_of_day_captured'), and runs breaker.check() — which engages
    the kill switch itself on a breach. LiveTradingSession evaluates the
    same gate once per evaluate_once() (module docstring).

    FAIL-CLOSED by design: when value_fn raises (auth expired, broker
    dark) the exception PROPAGATES — a rail that cannot read the account
    value must block the order it was gating, never wave it through.

    Args:
        breaker: the DailyLossCircuitBreaker to evaluate.
        value_fn: callable() -> current account value — a number, a
            BalanceResponse-shaped dict, or a get_portfolio_status()
            dict (whatever extract_account_value accepts).
        clock: injectable callable -> aware datetime for the ET date
            roll; defaults to the breaker's clock.
    """

    def __init__(self, breaker: DailyLossCircuitBreaker,
                 value_fn: Callable[[], Union[Dict, float, int]],
                 clock: Optional[Callable[[], datetime]] = None):
        self.breaker = breaker
        self.value_fn = value_fn
        self._clock = clock or breaker._clock
        self._lock = threading.Lock()
        self._sod_date = None
        self._sod_value: Optional[float] = None

    @property
    def start_of_day_value(self) -> Optional[float]:
        """Today's captured baseline (None before the first observation)."""
        return self._sod_value

    def _persisted_baseline(self, date_iso: str) -> Optional[float]:
        """The start-of-day baseline already captured for ``date_iso`` (this
        env), recovered from the audit log so a mid-day process RESTART re-uses
        the morning's baseline instead of re-baselining from an already
        drawn-down value. Best-effort: any read failure returns None (the
        caller then captures fresh, preserving prior in-memory behavior)."""
        try:
            rows = self.breaker.audit.entries(
                event_type="start_of_day_captured", limit=500, ascending=False)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            return None
        env = getattr(self.breaker.audit, "env", None)
        for row in rows:
            payload = row.get("payload", {}) or {}
            if payload.get("date") == date_iso and (
                    env is None or row.get("env") == env):
                value = payload.get("value")
                if value is not None:
                    return float(value)
        return None

    def __call__(self) -> Dict:
        """Evaluate the rail; returns breaker.check()'s result dict
        ({'breached': bool, ...} — the shape EtradeClient's pre-trade
        hook consumes)."""
        current = extract_account_value(self.value_fn())
        today_et = self._clock().astimezone(EASTERN).date()
        with self._lock:
            if self._sod_date != today_et or self._sod_value is None:
                date_iso = today_et.isoformat()
                # Restart-safe: reuse a baseline already captured today (in this
                # env) rather than re-baselining from the current, possibly
                # drawn-down, value. Only capture+audit when none exists yet.
                persisted = self._persisted_baseline(date_iso)
                if persisted is not None:
                    self._sod_value = persisted
                    logger.info(
                        "Daily-loss baseline for %s reused from audit: %.2f",
                        date_iso, persisted)
                else:
                    self._sod_value = current
                    logger.info(
                        "Daily-loss baseline captured for %s: %.2f",
                        date_iso, current)
                    self.breaker.audit.append(
                        "circuit_breaker", "start_of_day_captured",
                        {"date": date_iso, "value": current})
                self._sod_date = today_et
            start_of_day = self._sod_value
        return self.breaker.check(current, start_of_day)
