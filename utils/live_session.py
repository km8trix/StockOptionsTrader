"""
Live trading session — desk -> data -> intents -> risk -> patient
execution -> audit (Phase 9, E5).

ONE manual entry point: evaluate_once(). There is deliberately NO
scheduler in this phase — automation (a market-hours loop calling
evaluate_once) is a follow-up the user turns on only after watching
manual evaluations behave. A session that starts trading by itself the
day it ships is how surprises happen.

PAPER-MODE PARITY: the same session wraps PaperTrader for rehearsal —
when no PatientExecutor is injected, approved intents are submitted
directly through broker.place_order (market), which is exactly the paper
path. Swap in the live broker + executor and nothing else changes.

Every step is audited in order: session_evaluate -> desk_intent (each
generated intent) -> execution_start/execution_report (approved intents
only — risk-blocked intents NEVER reach the executor) -> done. A kill
switch engaged before or during the loop halts the session cleanly with
a session_halted row.

EXECUTION FAILURES ARE A CLEAN, AUDITED STOP: a broker/auth exception
escaping the execution path (the midnight-ET token expiry striking while
an order is being worked, a dead transport, a gating broker) NEVER
propagates out of evaluate_once(). The PatientExecutor already converts
mid-work failures into terminal 'error' reports after a best-effort
cancel; _execute_intent adds a second rail catching anything that still
raises. Either way the audit trail closes properly — execution_report
(status 'error' + typed reason) then session_halted — and the remaining
intents do NOT run against a broker in an unknown state.

DAILY-LOSS CIRCUIT BREAKER (E5): the session evaluates the gate at the
top of every evaluate_once() — BEFORE any intent is generated or
executed. By default it discovers the gate LiveEtradeBroker auto-wires
onto itself (broker.circuit_breaker); pass circuit_breaker= explicitly
to override, or None stays None for brokers without one (paper parity:
FakeBroker/PaperTrader carry no gate, so nothing changes). A breach —
or a gate that cannot be evaluated at all — halts the evaluation
FAIL-CLOSED with a session_halted audit row; the breaker itself engages
the persistent kill switch, so every later cycle halts at the top check.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from core.models import OrderType
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class LiveTradingSession:
    """Wires a Desk to a broker through the safety rails.

    Args:
        desk: a desks.base.Desk (generate_intents + apply_risk).
        broker: ExecutionBroker (PaperTrader or LiveEtradeBroker).
        portfolio: the PortfolioManager backing the desk's risk checks.
        data_fn: callable() -> {symbol: indicator-enriched DataFrame}
            sliced through "now" — the session never fetches data itself.
        executor: optional PatientExecutor; None = direct market orders
            through the broker (paper-mode parity, see module docstring).
        audit / kill_switch: the shared safety rails.
        auth_manager: optional EtradeAuthManager; when present renew() is
            called opportunistically each evaluation (E*TRADE's 2h idle
            timeout) — failures surface in status, never crash the loop.
        reconcile_fn: optional callable(local_positions, local_cash,
            broker) -> C19 dict (defaults to brokers.reconcile.reconcile).
        circuit_breaker: zero-arg callable -> {'breached': bool, ...}
            (a brokers.circuit_breaker.DailyLossGate). Defaults to the
            broker's own ``circuit_breaker`` attribute when present
            (LiveEtradeBroker auto-wires one); evaluated at the top of
            every evaluate_once(), fail-closed (module docstring).
        clock: injectable callable -> aware datetime.
    """

    #: "use broker.circuit_breaker" sentinel — None disables explicitly.
    _AUTO = object()

    def __init__(self, desk, broker, portfolio,
                 data_fn: Callable[[], Dict],
                 executor=None,
                 audit: Optional[AuditLog] = None,
                 kill_switch: Optional[KillSwitch] = None,
                 auth_manager=None,
                 reconcile_fn: Optional[Callable] = None,
                 circuit_breaker=_AUTO,
                 clock: Optional[Callable[[], datetime]] = None):
        self.desk = desk
        self.broker = broker
        self.portfolio = portfolio
        self.data_fn = data_fn
        self.executor = executor
        self.audit = audit if audit is not None else AuditLog()
        self.kill_switch = kill_switch
        self.auth_manager = auth_manager
        if reconcile_fn is None:
            from brokers.reconcile import reconcile as reconcile_fn
        self.reconcile_fn = reconcile_fn
        if circuit_breaker is LiveTradingSession._AUTO:
            circuit_breaker = getattr(broker, "circuit_breaker", None)
        self.circuit_breaker = circuit_breaker
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.last_reconciliation: Optional[Dict] = None

    # ------------------------------------------------------------------
    def evaluate_once(self) -> Dict:
        """One full manual evaluation cycle (module docstring)."""
        now = self._clock()
        if self.kill_switch is not None and self.kill_switch.engaged():
            self.audit.append("live_session", "session_halted",
                              {"reason": "kill_switch_engaged"})
            logger.warning("Session halted: kill switch engaged")
            return {"status": "halted", "reason": "kill_switch_engaged",
                    "timestamp": now.isoformat(), "reports": []}

        if self.auth_manager is not None:
            # Opportunistic renew clears E*TRADE's 2h idle timeout; a
            # False just means reauth is needed and the API calls below
            # will surface it typed.
            try:
                self.auth_manager.renew()
            except Exception as e:  # noqa: BLE001 - renew must never kill the loop
                logger.warning("Opportunistic token renew failed: %s", e)

        # Daily-loss circuit breaker (E5) — evaluated BEFORE any intent
        # is generated or executed. The breaker engages the kill switch
        # itself on a breach; this halt covers the cycle that found it.
        halted_by_breaker = self._check_circuit_breaker(now)
        if halted_by_breaker is not None:
            return halted_by_breaker

        self.audit.append("live_session", "session_evaluate",
                          {"desk": self.desk.key,
                           "timestamp": now.isoformat()})
        all_data = self.data_fn()
        self.desk.set_clock(now)
        intents = self.desk.generate_intents(all_data, now, self.portfolio)
        for intent in intents:
            self.audit.append("live_session", "desk_intent", {
                "symbol": intent.asset.symbol,
                "action": intent.action,
                "size_fraction": intent.size_fraction,
                "quantity": intent.quantity,
                "reason": intent.reason,
            })
        approved = self.desk.apply_risk(intents, self.portfolio, all_data,
                                        now)

        reports: List[Dict] = []
        halted = False
        halt_reason: Optional[str] = None
        for intent in approved:
            # Mid-loop engagement (circuit breaker, operator) halts the
            # remainder of the evaluation cleanly.
            if self.kill_switch is not None and self.kill_switch.engaged():
                self.audit.append("live_session", "session_halted",
                                  {"reason": "kill_switch_engaged_mid_loop"})
                logger.warning("Session halted mid-loop: kill switch")
                halted = True
                halt_reason = "kill_switch_engaged_mid_loop"
                break
            report = self._execute_intent(intent)
            if report is None:
                continue
            reports.append(report)
            if report.get("status") == "error":
                # A broker/auth failure mid-execution (module docstring):
                # the working order's outcome is already recorded in the
                # execution_report row; the remaining intents must NOT
                # run against a broker in an unknown state.
                self.audit.append("live_session", "session_halted", {
                    "reason": "execution_error",
                    "symbol": intent.asset.symbol,
                    "error": report.get("error"),
                    "error_type": report.get("error_type"),
                })
                logger.error("Session halted: execution error on %s "
                             "(%s: %s)", intent.asset.symbol,
                             report.get("error_type"), report.get("error"))
                halted = True
                halt_reason = "execution_error"
                break

        result = {
            "status": "halted" if halted else "ok",
            "timestamp": now.isoformat(),
            "generated": len(intents),
            "approved": len(approved),
            "reports": reports,
        }
        if halted:
            result["reason"] = halt_reason
        return result

    # ------------------------------------------------------------------
    def _check_circuit_breaker(self, now: datetime) -> Optional[Dict]:
        """Run the daily-loss gate; a halted-result dict stops the cycle.

        FAIL-CLOSED: a gate that raises (balances unreachable, auth
        expired) halts the evaluation too — a rail that cannot be read
        must never be assumed clear. Returns None when trading may
        proceed (no gate wired, or no breach).
        """
        if self.circuit_breaker is None:
            return None
        try:
            result = self.circuit_breaker()
        except Exception as e:  # noqa: BLE001 - any failure halts, audited
            self.audit.append("live_session", "session_halted", {
                "reason": "circuit_breaker_error",
                "error": str(e),
            })
            logger.error("Session halted: daily-loss circuit breaker "
                         "could not be evaluated: %s", e)
            return {"status": "halted", "reason": "circuit_breaker_error",
                    "timestamp": now.isoformat(), "reports": []}
        if isinstance(result, dict) and result.get("breached"):
            self.audit.append("live_session", "session_halted", {
                "reason": "daily_loss_circuit_breaker",
                "loss_pct": result.get("loss_pct"),
                "limit_pct": result.get("limit_pct"),
            })
            logger.warning("Session halted: daily-loss circuit breaker "
                           "breached (%.4f%%)",
                           (result.get("loss_pct") or 0.0) * 100)
            return {"status": "halted",
                    "reason": "daily_loss_circuit_breaker",
                    "timestamp": now.isoformat(), "reports": []}
        return None

    # ------------------------------------------------------------------
    def _execute_intent(self, intent) -> Optional[Dict]:
        side = "BUY" if intent.action in ("BUY", "COVER") else "SELL"
        quantity = intent.quantity
        if quantity is None:
            quantity = self._size_from_fraction(intent)
            if quantity is None or quantity <= 0:
                self.audit.append("live_session", "execution_skipped", {
                    "symbol": intent.asset.symbol,
                    "action": intent.action,
                    "reason": "no price / zero size",
                })
                logger.info("Skipped %s %s: unsizable", intent.action,
                            intent.asset.symbol)
                return None
        self.audit.append("live_session", "execution_start", {
            "symbol": intent.asset.symbol,
            "action": intent.action,
            "side": side,
            "quantity": quantity,
        })
        try:
            if self.executor is not None:
                report = self.executor.execute(side, intent.asset, quantity)
            else:
                # Paper-mode parity path: direct market order.
                order_type = (OrderType.BUY if side == "BUY"
                              else OrderType.SELL)
                order_id = self.broker.place_order(intent.asset, order_type,
                                                   quantity, None)
                report = {"status": "submitted", "order_id": order_id}
        except Exception as e:  # noqa: BLE001 - audited halt, never a crash
            # Second rail under the executor's own mid-work handling
            # (module docstring): ANY exception escaping the execution
            # path becomes a terminal 'error' report so the audit trail
            # closes (execution_report + session_halted) instead of a
            # bare exception leaving 'execution_start' dangling with a
            # working order of unknown state.
            logger.error("Execution of %s %s raised %s: %s",
                         intent.action, intent.asset.symbol,
                         type(e).__name__, e)
            report = {"status": "error", "error": str(e),
                      "error_type": type(e).__name__}
        report_payload = {
            "symbol": intent.asset.symbol,
            "action": intent.action,
            "status": report.get("status"),
            "avg_fill": report.get("avg_fill"),
            "shortfall_per_unit": report.get("shortfall_per_unit"),
        }
        if report.get("status") == "error":
            report_payload["error"] = report.get("error")
            report_payload["error_type"] = report.get("error_type")
        self.audit.append("live_session", "execution_report",
                          report_payload)
        return report

    def _size_from_fraction(self, intent) -> Optional[int]:
        """Dollar sizing fallback when an intent carries no quantity:
        portfolio_value * desk allocation * size_fraction, divided by the
        live price x multiplier (contracts for options, shares for
        stock)."""
        price = self.broker.get_current_price(intent.asset.symbol)
        if price is None or price <= 0:
            return None
        dollars = (self.portfolio.get_portfolio_value()
                   * self.desk.capital_allocation * intent.size_fraction)
        return int(dollars // (price * intent.asset.multiplier))

    # ------------------------------------------------------------------
    def run_reconciliation(self, local_positions: Dict[str, float],
                           local_cash: float) -> Dict:
        """C19 wiring: reconcile, and on NOT-ok engage the kill switch —
        a book the broker disagrees with must stop trading immediately."""
        result = self.reconcile_fn(local_positions, local_cash, self.broker)
        self.last_reconciliation = result
        self.audit.append("live_session", "reconciliation", {
            "ok": result["ok"],
            "mismatches": result["mismatches"],
            "checked_at": result["checked_at"],
        })
        if not result["ok"] and self.kill_switch is not None:
            self.kill_switch.engage(
                reason=(f"reconciliation mismatch: "
                        f"{len(result['mismatches'])} difference(s) vs "
                        f"broker"),
                actor="reconciliation")
        return result
