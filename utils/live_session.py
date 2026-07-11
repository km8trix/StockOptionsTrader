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

try:
    from utils.market_hours import MarketHours
except Exception:  # noqa: BLE001 - market-hours guard is optional
    MarketHours = None

logger = logging.getLogger(__name__)


def _filled_quantity(report: Dict) -> Optional[int]:
    """Actual filled size from the executor report's fills, or None.

    A partial fill banks fewer units than the intended quantity, so the parity
    harness needs the realized size (not the intent size) to scale drift and
    commission. The PatientExecutor reports per-fill {qty, price}; the paper
    path carries no fills, so this returns None there (the harness then falls
    back to the recorded intended quantity).
    """
    fills = report.get("fills") if isinstance(report, dict) else None
    if not fills:
        return None
    total = sum(f.get("qty", 0) for f in fills if isinstance(f, dict))
    return total or None


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
        local_book: optional brokers.local_book.LocalBook (Step 5,
            OPT-IN — default None changes nothing). When set, every
            CONFIRMED fill is recorded into the persistent book (the
            executor path records the report's banked fills; the
            direct-market path confirms via broker.order_status), and
            run_reconciliation() called without arguments reconciles
            book.positions()/book.cash() against the broker — the
            restart-surviving zero-drift check.
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
                 local_book=None,
                 clock: Optional[Callable[[], datetime]] = None,
                 orchestrator=None,
                 enforce_market_hours: bool = False):
        # Fund mode: a FundOrchestrator drives N desks on the shared
        # portfolio. It exposes the read surface the session needs
        # (key/capital_allocation/set_clock) and a step() that returns the
        # netted, account-risk-approved intents. Provide exactly ONE of
        # desk= or orchestrator=.
        if (desk is None) == (orchestrator is None):
            raise ValueError(
                "Provide exactly one of desk= or orchestrator= to "
                "LiveTradingSession")
        self.desk = desk
        self.orchestrator = orchestrator
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
        # Step 5, OPT-IN (None = byte-identical behavior, pinned in tests).
        self.local_book = local_book
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # When True, evaluate_once() refuses to trade outside the NYSE regular
        # session (a manual call off-hours otherwise transmits — the scheduler
        # gates this for the autonomous loop, but a direct call does not).
        # Default False keeps paper-parity and existing callers unchanged.
        self.enforce_market_hours = enforce_market_hours
        self.last_reconciliation: Optional[Dict] = None

    @property
    def _driver(self):
        """The desk or orchestrator driving this session. Both expose
        key/capital_allocation/set_clock, so the session reads whichever is
        set off this one accessor."""
        return self.desk if self.desk is not None else self.orchestrator

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

        if (self.enforce_market_hours and MarketHours is not None
                and not MarketHours().is_market_open(now)):
            self.audit.append("live_session", "session_skipped",
                              {"reason": "market_closed"})
            logger.info("Session skipped: NYSE regular session is closed")
            return {"status": "market_closed", "reason": "market_closed",
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
                          {"desk": self._driver.key,
                           "timestamp": now.isoformat()})
        # Intent generation + risk is FAIL-CLOSED: a raise here (data fetch,
        # a desk's generate_intents, the orchestrator's net/apply_risk/
        # aggregator, or apply_risk) must NOT escape as a bare exception
        # leaving the audit trail open at session_evaluate — it becomes a
        # clean, audited session_halted, same as the execution-phase rails.
        # (Fund mode widens this surface to N desks + netting + aggregator;
        # the guard hardens both modes identically.)
        try:
            all_data = self.data_fn()
            self._driver.set_clock(now)
            if self.orchestrator is not None:
                # The orchestrator fans out to its desks, nets, and runs the
                # one account-wide apply_risk (+ aggregator) internally;
                # step() returns the already-approved intents. The netting/
                # conflict decisions are captured in orchestrator.notes.
                intents = self.orchestrator.step(all_data, now, self.portfolio)
                approved = intents
            else:
                intents = self.desk.generate_intents(all_data, now,
                                                     self.portfolio)
            for intent in intents:
                self.audit.append("live_session", "desk_intent", {
                    "symbol": intent.asset.symbol,
                    "action": intent.action,
                    "size_fraction": intent.size_fraction,
                    "quantity": intent.quantity,
                    "reason": intent.reason,
                })
            if self.orchestrator is None:
                approved = self.desk.apply_risk(intents, self.portfolio,
                                                all_data, now)
        except Exception as e:  # noqa: BLE001 - audited halt, never a crash
            self.audit.append("live_session", "session_halted", {
                "reason": "intent_generation_error",
                "error": str(e),
                "error_type": type(e).__name__,
            })
            logger.error("Session halted: intent generation/risk raised "
                         "%s: %s", type(e).__name__, e)
            return {"status": "halted", "reason": "intent_generation_error",
                    "timestamp": now.isoformat(), "reports": []}

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
            # Parity fields (Step 5, additive): let the read-only parity harness
            # replay this fill through the backtest cost model EXACTLY, without
            # recovering the reference price or guessing the asset class.
            # arrival_mid/commission come from the executor report (commission
            # is None today — the broker does not surface it yet; the field is a
            # forward-compatible hook). asset_type/quantity are known here.
            "arrival_mid": report.get("arrival_mid"),
            "asset_type": intent.asset.asset_type.value,
            "quantity": quantity,
            "filled_quantity": _filled_quantity(report),
            "commission": report.get("commission"),
        }
        if report.get("status") == "error":
            report_payload["error"] = report.get("error")
            report_payload["error_type"] = report.get("error_type")
        self.audit.append("live_session", "execution_report",
                          report_payload)
        if self.local_book is not None:
            self._record_fills_to_book(intent, side, report)
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _book_key(asset) -> str:
        """Reconcile-convention position key: plain symbol for stock,
        canonical str(Asset) for options (brokers.reconcile docstring)."""
        return (asset.symbol if asset.multiplier == 1 else str(asset))

    def _record_fills_to_book(self, intent, side: str, report: Dict) -> None:
        """Record CONFIRMED fills from this execution into the local book.

        Records what the broker CONFIRMS, never the intent:
          * executor path — the report's banked fills (per-slice qty/price;
            present on every terminal status, including 'partial'/'killed'/
            'error', because banked fills are real position changes);
          * direct-market path — broker.order_status(order_id). ANY
            confirmed filled_quantity > 0 is booked, whatever the status
            string says — E*TRADE reports 'PARTIAL' while working and a
            partial-then-cancelled order ends CANCELLED with real fills
            attached; those units are position changes and must not be
            dropped on the floor. A status with no fill yet gets one
            get_portfolio_status() nudge (ABC surface; PaperTrader
            processes pending fills there, a live broker treats it as a
            harmless snapshot read) before the second poll. The two-poll
            window is a KNOWN BOUND: a fill landing after the second
            poll is missed here and surfaces as drift — reconciliation
            is the backstop for late fills. Note the direct path is the
            paper-parity path — live wiring goes through the
            PatientExecutor (module docstring).

        Cash moves at fill_price x asset.multiplier per unit (options
        cash-flow the x100 contract multiplier; stock is x1). NEVER
        raises: a book-recording failure must not halt the session — the
        resulting drift is exactly what reconciliation exists to catch.
        """
        try:
            sign = 1.0 if side == "BUY" else -1.0
            key = self._book_key(intent.asset)
            multiplier = intent.asset.multiplier
            fills = report.get("fills") or []
            if fills:
                for fill in fills:
                    qty = float(fill.get("qty", 0) or 0)
                    price = fill.get("price")
                    if qty > 0 and price is not None:
                        self.local_book.record_fill(
                            key, sign * qty, float(price) * multiplier)
                return
            order_id = report.get("order_id")
            if order_id is None:
                return
            status = self.broker.order_status(order_id)
            if not self._confirmed_fill_qty(status):
                self.broker.get_portfolio_status()
                status = self.broker.order_status(order_id)
            qty = self._confirmed_fill_qty(status)
            price = status.get("avg_fill_price") if status else None
            if qty > 0 and price is not None:
                self.local_book.record_fill(
                    key, sign * qty, float(price) * multiplier)
            else:
                logger.warning(
                    "Local book: no confirmed fill for order %s "
                    "(status=%s) — any late fill surfaces at "
                    "reconciliation", order_id,
                    status.get("status") if status else None)
        except Exception as e:  # noqa: BLE001 - book must never halt the session
            logger.error("Local book recording failed (%s): %s — drift "
                         "surfaces at reconciliation", type(e).__name__, e)

    @staticmethod
    def _confirmed_fill_qty(status: Optional[Dict]) -> float:
        """Broker-confirmed filled units in a status dict (0.0 when none).
        Deliberately status-string-agnostic: partial fills ('PARTIAL',
        or CANCELLED-after-partial) are real position changes."""
        if status is None:
            return 0.0
        return float(status.get("filled_quantity") or 0)

    def _size_from_fraction(self, intent) -> Optional[int]:
        """Dollar sizing fallback when an intent carries no quantity:
        portfolio_value * desk allocation * size_fraction, divided by the
        live price x multiplier (contracts for options, shares for
        stock)."""
        price = self.broker.get_current_price(intent.asset.symbol)
        if price is None or price <= 0:
            return None
        # In fund mode the driver is the orchestrator (capital_allocation
        # 1.0) and intent.size_fraction is already account-absolute, so this
        # is portfolio_value * size_fraction — no double-scaling. Single-desk
        # mode uses the desk's own allocation, unchanged.
        dollars = (self.portfolio.get_portfolio_value()
                   * self._driver.capital_allocation * intent.size_fraction)
        return int(dollars // (price * intent.asset.multiplier))

    # ------------------------------------------------------------------
    def run_reconciliation(self,
                           local_positions: Optional[Dict[str, float]] = None,
                           local_cash: Optional[float] = None,
                           cash_tolerance: Optional[float] = None) -> Dict:
        """C19 wiring: reconcile, and on NOT-ok engage the kill switch —
        a book the broker disagrees with must stop trading immediately.

        Called with explicit local_positions AND local_cash this behaves
        as before (explicit args win over the book). Called with NO
        arguments it reads the wired local_book (Step 5) — the
        restart-surviving ledger — and raises ValueError when neither is
        available (reconciling an implicit empty book would report fake
        drift or fake cleanliness). Exactly ONE of the two provided is a
        caller bug: raises ValueError('provide both or neither').

        cash_tolerance=None keeps reconcile's default ($0.01); pass a
        fee-aware dollar tolerance for routine operational reconciles
        (brokers rarely report fees per fill — see
        brokers.local_book.LocalBook).

        FAIL-CLOSED: an exception from the book reads or from
        reconcile_fn engages the kill switch before re-raising
        (mirroring _check_circuit_breaker) — a reconciliation that
        CANNOT run must never be treated as clean.
        """
        # Caller-bug guards stay OUTSIDE the fail-closed rail: nothing was
        # read or reconciled yet, so there is nothing to fail closed on.
        if (local_positions is None) != (local_cash is None):
            raise ValueError(
                "run_reconciliation(): local_positions/local_cash — "
                "provide both or neither")
        if local_positions is None and self.local_book is None:
            raise ValueError(
                "run_reconciliation() needs explicit local_positions/"
                "local_cash when no local_book is wired")
        try:
            if local_positions is None:
                local_positions = self.local_book.positions()
                local_cash = self.local_book.cash()
            if cash_tolerance is None:
                # No kwarg: any injected reconcile_fn keeps its signature.
                result = self.reconcile_fn(local_positions, local_cash,
                                           self.broker)
            else:
                result = self.reconcile_fn(local_positions, local_cash,
                                           self.broker,
                                           cash_tolerance=cash_tolerance)
        except Exception as e:
            # FAIL-CLOSED (docstring): a book that cannot be read or a
            # reconcile that cannot run is indistinguishable from drift.
            logger.error("Reconciliation could not run (%s: %s) — "
                         "engaging the kill switch", type(e).__name__, e)
            if self.kill_switch is not None:
                self.kill_switch.engage(
                    reason=(f"reconciliation could not run: "
                            f"{type(e).__name__}: {e}"),
                    actor="reconciliation")
            raise
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
