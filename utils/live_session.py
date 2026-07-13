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
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Protocol

import pandas as pd

from core.models import Asset, AssetType, OrderType
from desks.base import DeskIntent
from portfolio.targets import (
    PortfolioSnapshot,
    build_order_deltas,
    filled_quantities_from_portfolio,
    reserved_deltas_from_risk_snapshot,
)
from portfolio.structures import StructureIntent
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch

class _MarketHoursInstance(Protocol):
    def is_market_open(self, dt: datetime) -> bool: ...


class _MarketHoursFactory(Protocol):
    def __call__(self) -> _MarketHoursInstance: ...


def _load_market_hours() -> Optional[_MarketHoursFactory]:
    """Load the optional market-hours guard without conflating a type alias."""
    try:
        from utils.market_hours import MarketHours
        return MarketHours
    except Exception:  # noqa: BLE001 - market-hours guard is optional
        return None


MarketHours = _load_market_hours()

logger = logging.getLogger(__name__)

_OPTION_BOOK_KEY = re.compile(
    r"^(?P<symbol>\S+) (?P<expiry>\d{4}-\d{2}-\d{2}) "
    r"\$(?P<strike>[0-9]+(?:\.[0-9]+)?) (?P<right>call|put)$",
    re.IGNORECASE,
)
_TERMINAL_ORDER_STATES = frozenset({
    "CANCELLED", "CANCELED", "EXECUTED", "FILLED", "REJECTED", "EXPIRED",
})


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
                 enforce_market_hours: bool = False,
                 execution_guard: Optional[Callable] = None):
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
            from brokers.reconcile import reconcile as default_reconcile
            reconcile_fn = default_reconcile
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
        # Optional final deployment boundary.  It runs after exact quantity
        # sizing but immediately before any executor/broker call, so immutable
        # manifest limits sit below strategy code.  Research and legacy paper
        # sessions leave it unset and remain byte-for-byte on their old path.
        if execution_guard is not None and not callable(execution_guard):
            raise TypeError("execution_guard must be callable or None")
        self.execution_guard = execution_guard
        self.last_reconciliation: Optional[Dict] = None
        self._target_snapshot_version = 0

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
            structures: List[StructureIntent] = []
            if self.orchestrator is not None:
                # The orchestrator fans out to its desks, nets, and runs the
                # one account-wide apply_risk (+ aggregator) internally;
                # step() returns the already-approved intents. The netting/
                # conflict decisions are captured in orchestrator.notes.
                intents = self.orchestrator.step(all_data, now, self.portfolio)
                approved = intents
            elif getattr(self.desk, "target_native_enabled", False):
                intents, cancellation_requested = self._target_intents(
                    all_data, now)
                if cancellation_requested:
                    return {
                        "status": "pending",
                        "reason": "target_order_cancellation_requested",
                        "timestamp": now.isoformat(),
                        "generated": 0,
                        "approved": 0,
                        "reports": [],
                    }
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
                    "intent_id": getattr(intent, "intent_id", None),
                })
            if self.orchestrator is None:
                approved = self.desk.apply_risk(intents, self.portfolio,
                                                all_data, now)
                structure_generator = getattr(
                    self.desk, "generate_structure_intents", None)
                if callable(structure_generator):
                    generated_structures = structure_generator(
                        all_data, now, self.portfolio)
                    structures = list(generated_structures)
                    if not all(isinstance(item, StructureIntent)
                               for item in structures):
                        raise ValueError(
                            "generate_structure_intents must return "
                            "StructureIntent values")
                    for structure in structures:
                        self.audit.append(
                            "live_session", "structure_intent", {
                                "intent_id": structure.intent_id,
                                "underlying": structure.legs[0].asset.symbol,
                                "quantity": structure.quantity,
                                "net_price": structure.net_price,
                                "max_loss": structure.max_loss,
                                "greeks": dict(structure.greeks),
                                "opening": structure.opening,
                                "legs": [
                                    {"asset": str(leg.asset),
                                     "action": leg.action.value,
                                     "ratio": leg.ratio}
                                    for leg in structure.legs
                                ],
                            })
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

        structure_reports: List[Dict] = []
        if not halted:
            for structure in structures:
                if (self.kill_switch is not None
                        and self.kill_switch.engaged()):
                    self.audit.append("live_session", "session_halted", {
                        "reason": "kill_switch_engaged_mid_loop",
                    })
                    halted = True
                    halt_reason = "kill_switch_engaged_mid_loop"
                    break
                report = self._execute_structure(structure)
                structure_reports.append(report)
                if report.get("status") == "error":
                    self.audit.append("live_session", "session_halted", {
                        "reason": "execution_error",
                        "intent_id": structure.intent_id,
                        "error": report.get("error"),
                        "error_type": report.get("error_type"),
                    })
                    halted = True
                    halt_reason = "execution_error"
                    break

        result: Dict = {
            "status": "halted" if halted else "ok",
            "timestamp": now.isoformat(),
            "generated": len(intents),
            "approved": len(approved),
            "reports": reports,
        }
        if halted:
            result["reason"] = halt_reason
        if structures:
            result["generated_structures"] = len(structures)
            result["structure_reports"] = structure_reports
        return result

    # ------------------------------------------------------------------
    # Target-native migration path
    # ------------------------------------------------------------------
    @staticmethod
    def _asset_from_book_key(key: str) -> Asset:
        """Decode LocalBook's reconciliation key without losing contracts."""
        text = str(key).strip()
        match = _OPTION_BOOK_KEY.fullmatch(text)
        if match is None:
            if not text or any(character.isspace() for character in text):
                raise ValueError(f"unsupported local-book position key {key!r}")
            return Asset(text.upper(), AssetType.STOCK)
        right = match.group("right").lower()
        return Asset(
            match.group("symbol").upper(),
            AssetType.CALL if right == "call" else AssetType.PUT,
            float(match.group("strike")),
            match.group("expiry"),
        )

    def _reservation_snapshot(self) -> Dict:
        gate = getattr(getattr(self.broker, "client", None),
                       "reservation_gate", None)
        snapshotter = getattr(gate, "snapshot", None)
        if not callable(snapshotter):
            return {"reservations": []}
        snapshot = snapshotter()
        if not isinstance(snapshot, dict):
            raise ValueError("reservation gate snapshot must be a mapping")
        return snapshot

    def _target_snapshot(self, now: datetime,
                         reservations: Dict) -> PortfolioSnapshot:
        """Build one filled-plus-working generation for target construction.

        A configured LocalBook is authoritative for live filled quantities and
        must have been explicitly bootstrapped.  Paper and isolated unit-test
        sessions keep using PortfolioManager, preserving their old lifecycle.
        """
        if self.local_book is not None:
            initialized = getattr(self.local_book, "is_initialized", None)
            if callable(initialized) and not initialized():
                raise RuntimeError(
                    "target-native execution requires an initialized local book")
            reader = getattr(self.local_book, "reconciliation_snapshot", None)
            if not callable(reader):
                raise ValueError(
                    "target-native local_book must expose reconciliation_snapshot")
            book = reader()
            if not book.get("initialized", True):
                raise RuntimeError(
                    "target-native execution requires an initialized local book")
            filled: Dict[Asset, int] = {}
            for key, raw_quantity in book.get("positions", {}).items():
                quantity = float(raw_quantity)
                if not quantity.is_integer():
                    raise ValueError(
                        f"local-book quantity for {key!r} is not a whole unit")
                if quantity:
                    asset = self._asset_from_book_key(key)
                    filled[asset] = filled.get(asset, 0) + int(quantity)
        else:
            filled = dict(filled_quantities_from_portfolio(self.portfolio))

        self._target_snapshot_version += 1
        return PortfolioSnapshot(
            filled_quantities=filled,
            reserved_deltas=reserved_deltas_from_risk_snapshot(reservations),
            version=self._target_snapshot_version,
            as_of=now,
        )

    @staticmethod
    def _target_action(delta) -> str:
        if delta.signed_quantity > 0:
            return "COVER" if delta.effective_quantity < 0 else "BUY"
        return "SELL" if delta.effective_quantity > 0 else "SHORT"

    def _cancel_obsolete_target_orders(self, targets, snapshot,
                                       reservations: Dict) -> bool:
        """Request cancellation before a changed target can reverse an order.

        Cancellation remains broker-authoritative: one status observation is
        applied to the reservation gate, and this evaluation always stops
        before transmitting a replacement.  A later evaluation re-diffs the
        confirmed filled + remaining state.
        """
        by_asset = {target.asset: target for target in targets}
        requested = False
        for reservation in reservations.get("reservations", []):
            if str(reservation.get("status", "")).upper() != "ACTIVE":
                continue
            reserved = reserved_deltas_from_risk_snapshot({
                "reservations": [reservation],
            })
            obsolete_assets = [
                asset for asset in reserved
                if asset in by_asset
                and by_asset[asset].target_quantity
                != snapshot.effective_quantity(asset)
            ]
            if not obsolete_assets:
                continue
            for order in reservation.get("orders", []):
                order_id = str(order.get("order_id") or "").strip()
                status = str(order.get("status") or "").upper()
                if not order_id or status in _TERMINAL_ORDER_STATES:
                    continue
                accepted = bool(self.broker.cancel_order(order_id))
                self.audit.append("live_session", "target_order_cancel", {
                    "order_id": order_id,
                    "accepted": accepted,
                    "assets": [str(asset) for asset in obsolete_assets],
                    "reservation_id": reservation.get("reservation_id"),
                })
                # A status call lets reservation-aware brokers record a
                # cancel/fill race.  Non-terminal confirmation is expected;
                # the next evaluation observes it again and still cannot place.
                status_reader = getattr(self.broker, "order_status", None)
                if callable(status_reader):
                    status_reader(order_id)
                requested = True
        return requested

    def _target_intents(self, all_data: Dict, now: datetime):
        """Construct exact legacy-shaped intents from a complete target set."""
        reservations = self._reservation_snapshot()
        snapshot = self._target_snapshot(now, reservations)
        # Backtests observe session D's completed bar and fill on D+1.  The
        # target-native live/paper path preserves that timing by evaluating the
        # desk at the newest completed bar handed in while the order snapshot
        # remains stamped with the actual execution instant ``now``.
        decision_dates = [pd.Timestamp(frame.index[-1])
                          for frame in all_data.values()
                          if frame is not None and not frame.empty]
        if not decision_dates:
            # Artifact-bound desks must prove the completed-bar decision input;
            # silently substituting the wall clock would turn a missing paper or
            # live snapshot into a same-session signal.  Keep the historical
            # target-native extension point usable for unbound/custom desks,
            # whose generate_targets implementations may not consume frames.
            if getattr(self.desk, "deployment_identity", None) is not None:
                raise ValueError(
                    "target-native execution received no market data")
            decision_date = pd.Timestamp(now)
        else:
            decision_date = max(decision_dates)
        targets = tuple(self.desk.generate_targets(
            all_data, decision_date, self.portfolio, snapshot))
        # Validate/coalesce the whole target set before broker mutation.
        build_order_deltas(targets, snapshot)
        for target in targets:
            self.audit.append("live_session", "target_position", {
                "asset": str(target.asset),
                "target_quantity": target.target_quantity,
                "owner": target.owner,
                "strategy": target.strategy,
                "reason": target.reason,
                "snapshot_version": snapshot.version,
                "decision_date": decision_date.isoformat(),
            })
        if self._cancel_obsolete_target_orders(
                targets, snapshot, reservations):
            return [], True

        intents = []
        for delta in build_order_deltas(targets, snapshot):
            size_fraction = delta.metadata.get("size_fraction")
            if size_fraction is None:
                raise ValueError(
                    f"target for {delta.asset} requires metadata.size_fraction")
            intent = DeskIntent(
                asset=delta.asset,
                action=self._target_action(delta),
                size_fraction=float(size_fraction),
                reason=(delta.reason
                        or f"target position {delta.target_quantity}"),
                quantity=delta.quantity,
                intent_id=delta.intent_id,
            )
            intents.append(intent)
            self.audit.append("live_session", "target_order_delta", {
                "asset": str(delta.asset),
                "signed_quantity": delta.signed_quantity,
                "target_quantity": delta.target_quantity,
                "effective_quantity": delta.effective_quantity,
                "phase": delta.phase.value,
                "intent_id": delta.intent_id,
                "snapshot_version": snapshot.version,
            })
        return intents, False

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
            if self.execution_guard is not None:
                self.execution_guard(
                    intent=intent,
                    side=side,
                    quantity=quantity,
                    now=self._clock(),
                )
            if self.executor is not None:
                execution_id = getattr(intent, "intent_id", None)
                if execution_id is None:
                    report = self.executor.execute(
                        side, intent.asset, quantity)
                else:
                    report = self.executor.execute(
                        side, intent.asset, quantity,
                        execution_id=execution_id)
            else:
                # Paper-mode parity path: direct market order.
                order_type = (OrderType.BUY if side == "BUY"
                              else OrderType.SELL)
                execution_id = getattr(intent, "intent_id", None)
                idempotent_placer = getattr(
                    self.broker, "place_order_with_client_id", None)
                if execution_id is not None and callable(idempotent_placer):
                    order_id = idempotent_placer(
                        intent.asset, order_type, quantity, None,
                        execution_id)
                else:
                    order_id = self.broker.place_order(
                        intent.asset, order_type, quantity, None)
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
        gate = getattr(getattr(self.broker, 'client', None),
                       'reservation_gate', None)
        if (self.local_book is not None
                and not bool(getattr(gate, 'books_fills', False))):
            self._record_fills_to_book(intent, side, report)
        return report

    def _execute_structure(self, structure: StructureIntent) -> Dict:
        """Transmit one canonical package as one patient/broker order."""
        legs = structure.execution_legs
        self.audit.append("live_session", "structure_execution_start", {
            "intent_id": structure.intent_id,
            "side": structure.side,
            "quantity": structure.quantity,
            "closing": structure.closing,
            "legs": [{"asset": str(item["asset"]),
                      "action": item["action"],
                      "ratio": item["ratio"]} for item in legs],
        })
        try:
            if self.executor is not None:
                report = self.executor.execute(
                    structure.side,
                    legs,
                    structure.quantity,
                    execution_id=structure.intent_id,
                    closing=structure.closing,
                )
            else:
                order_id = self.broker.place_structure(
                    legs,
                    structure.net_price,
                    structure.quantity,
                    closing=structure.closing,
                )
                report = {"status": "submitted", "order_id": order_id}
        except Exception as error:  # noqa: BLE001 - same audited rail as singles
            report = {"status": "error", "error": str(error),
                      "error_type": type(error).__name__}
        payload = {
            "intent_id": structure.intent_id,
            "status": report.get("status"),
            "quantity": structure.quantity,
            "filled_quantity": _filled_quantity(report),
            "avg_fill": report.get("avg_fill"),
            "arrival_mid": report.get("arrival_mid"),
            "shortfall_per_unit": report.get("shortfall_per_unit"),
            "closing": structure.closing,
        }
        if report.get("status") == "error":
            payload["error"] = report.get("error")
            payload["error_type"] = report.get("error_type")
        self.audit.append(
            "live_session", "structure_execution_report", payload)
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
            # PaperTrader surfaces both the raw market fill and the effective
            # cash fill after explicit costs.  Booking the latter makes the
            # independent broker ledger and LocalBook reconcile to the cent;
            # live brokers that do not expose it keep the historical raw path.
            price = (status.get("cash_fill_price",
                                status.get("effective_fill_price",
                                           status.get("avg_fill_price")))
                     if status else None)
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
                assert self.local_book is not None
                local_positions = self.local_book.positions()
                local_cash = self.local_book.cash()
            assert local_cash is not None
            assert self.reconcile_fn is not None
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
