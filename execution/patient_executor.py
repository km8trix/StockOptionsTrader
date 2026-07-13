"""
Patient execution engine (Phase 9, E4) — the default for every desk.

ALGORITHM (tested step-for-step in tests/test_patient_executor.py):

  t0     stop event / kill switch ALREADY engaged?  report 'killed'
         WITHOUT placing anything (defense-in-depth: in live wiring
         EtradeClient's pre-trade gate would refuse the order anyway,
         but the executor must never hand an order to a non-gating
         broker under an engaged switch, and the caller gets the
         documented 'killed' report instead of an exception).
         Otherwise fetch quote; place a limit at MID = (bid + ask) / 2,
         rounded to the $0.01 tick (round-half-up); poll fills
         immediately.
  each step_interval_s, while unfilled quantity remains:
    1. stop event / kill switch engaged?  cancel -> status 'killed'.
    2. poll fills; fully filled -> status 'filled'.
    3. edge_check() returned False?       cancel -> 'edge_decayed'.
    4. max_minutes elapsed?               cancel -> 'timeout' (or
       'partial' when some quantity already filled).
    5. otherwise REPRICE: re-fetch the quote and move the limit 25% of
       the REMAINING distance from the CURRENT limit toward the far
       touch (BUY: toward the ask; SELL: toward the bid), rounded to the
       tick and NEVER beyond the CURRENT far touch — if the market moved
       through our limit, the limit is capped AT the touch, not through
       it. The reprice is a cancel-and-replace for the remaining
       quantity; partial fills already banked keep working.

  CANCEL IS ONLY A REQUEST: every cancel polls order_status until the
  broker reports the order terminal (CANCELLED/EXECUTED/...), banking
  any fill that raced the cancel, BEFORE anything else happens — sizing
  a replacement off a pre-confirmation fill count double-executes the
  racing quantity. A cancel that never confirms is a mid-work failure
  (terminal 'error' report), never a blind replacement.

  NO MARKET-ORDER FALLBACK, deliberately: every desk here trades
  patient-edge strategies — paying the spread to force a fill erases the
  edge being captured. A desk would rather MISS than pay up; 'timeout'
  and 'edge_decayed' are successful outcomes of that policy.

  MID-WORK BROKER FAILURE (midnight-ET auth expiry mid-poll, transport
  death, a gating broker refusing a replacement): the executor NEVER
  lets a working order end with no recorded outcome. It attempts a
  best-effort cancel of the working order, banks any fills that raced
  the cancel, and returns a terminal 'error' report carrying the typed
  reason ('error' + 'error_type' keys) instead of propagating the
  exception — the session audits that report and halts.

ExecutionReport shape (contract E4):
    {'status': 'filled'|'partial'|'edge_decayed'|'timeout'|'killed'
               |'error',
     'fills': [{'qty', 'price', 'ts'}],
     'arrival_mid': float,
     'avg_fill': float|None,
     'shortfall_per_unit': float|None,
     'steps': [{'ts', 'limit'}]}
    plus, on status 'error' only: {'error': str, 'error_type': str}
    (the exception's message and class name, e.g. 'EtradeAuthExpired').

SHORTFALL SIGN CONVENTION: shortfall_per_unit is signed so POSITIVE is
WORSE on BOTH sides — BUY: avg_fill - arrival_mid (paid more than
arrival mid); SELL: arrival_mid - avg_fill (received less). A negative
shortfall means the patience earned price improvement.

Terminal-status precedence with partial fills: a full fill is 'filled';
the kill switch always reports 'killed'; edge decay always reports
'edge_decayed' (the report's fills list says what was banked); only the
deadline distinguishes 'partial' (some fills) from 'timeout' (none).

Broker protocol (duck-typed; both LiveEtradeBroker and PaperTrader comply
in full, so paper rehearsal can drive this engine exactly like live):
    place_order(asset, OrderType, quantity, limit_price) -> order_id
    place_structure(legs, net_price, contracts, closing=False) -> order_id
    place_structure_with_client_id(legs, net_price, contracts,
                                   client_order_id, closing=False) -> order_id
        (only needed when working multi-leg packages)
    cancel_order(order_id) -> bool
    order_status(order_id) -> {'status', 'filled_quantity',
                               'avg_fill_price'} | None

quote_fn(instrument_or_legs) -> {'bid': float, 'ask': float} — for a
multi-leg package these are the NET package bid/ask.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from core.models import Asset, AssetType, OrderType
from utils.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

TICK = 0.01

#: Cancel-confirmation polling: cancel_order is a REQUEST, not a state
#: change. PaperTrader confirms on the next poll (None / terminal);
#: live E*TRADE sits in CANCEL_REQUESTED briefly. 20 x 0.5s bounds the
#: wait at ~10s before the executor declares the order state unknown.
CANCEL_POLL_S = 0.5
CANCEL_CONFIRM_ATTEMPTS = 20
#: Statuses after which the order can no longer fill (both the E*TRADE
#: vocabulary and PaperTrader's); CANCEL_REQUESTED is deliberately NOT
#: here — that is exactly the window where a fill can race the cancel.
_TERMINAL_STATUSES = frozenset({
    "CANCELLED", "CANCELED", "EXECUTED", "FILLED", "REJECTED", "EXPIRED"})


class ControlledNotionalEnvelopeError(RuntimeError):
    """A controlled opening order escaped its durable notional envelope."""


def round_tick(price: float) -> float:
    """Round to the $0.01 tick, half-up, with an epsilon so float dust
    (114.74999...) lands on the hand-computable cent.

    Tradeoff (deliberate): the 1e-9 epsilon rescues representation dust
    — an intended 2.675 is stored as 2.67499...9 and still rounds up to
    2.68 — at the cost of treating values within 1e-9 below a half-cent
    as halves, which is safe because no real market price legitimately
    sits that close to a half-cent boundary.
    """
    return math.floor(price * 100.0 + 0.5 + 1e-9) / 100.0


class PatientExecutor:
    """Single-order patient worker (thread-safe, stoppable).

    Args:
        broker: see the module docstring's duck-typed protocol.
        quote_fn: callable(instrument_or_legs) -> {'bid','ask'}.
        clock: injectable callable -> aware datetime (tests freeze it).
        sleep_fn: injectable sleep (tests pass a virtual-time advancer).
        kill_switch: optional KillSwitch polled every step; engagement
            mid-work cancels the order and reports 'killed'.
    """

    def __init__(self, broker, quote_fn: Callable,
                 clock: Optional[Callable[[], datetime]] = None,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 kill_switch: Optional[KillSwitch] = None):
        if not callable(quote_fn):
            raise TypeError(
                "quote_fn must be an injected callable returning bid/ask; "
                "PatientExecutor never derives option or package prices "
                "from an underlying quote")
        self.broker = broker
        self.quote_fn = quote_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep_fn
        self._uses_default_sleep = sleep_fn is time.sleep
        self.kill_switch = kill_switch
        #: Permanent lifecycle stop. It is deliberately never cleared by
        #: execute(): a close/execute race must fail closed.
        self._stop_event = threading.Event()
        #: Per-order operator cancellation. Unlike stop(), this is cleared
        #: once that execution has reached a terminal report.
        self._cancel_event = threading.Event()
        #: Wakes the default real-time wait immediately on cancel/stop.
        self._wake_event = threading.Event()
        #: All public lifecycle/observability state is protected by one
        #: condition so snapshots and await_idle() cannot observe torn state.
        self._state = threading.Condition(threading.RLock())
        self._active = False
        self._working_order: Optional[Dict] = None
        # Controlled deployment authorizes notional durably before invoking
        # the executor.  The one-shot in-memory envelope binds that durable
        # reservation to the exact logical execution which consumes it.
        self._controlled_envelopes_required = False
        self._notional_envelopes: Dict[str, Dict] = {}

    def require_controlled_notional_envelopes(self) -> None:
        """Require a one-shot durable authorization for every execution.

        This is enabled only by the verified live composition root.  Legacy
        paper/research callers retain the normal patient-execution contract.
        Rebinding clears unconsumed process-local capabilities; the durable
        deployment store can safely reissue one for an idempotent intent.
        """
        with self._state:
            if self._active:
                raise RuntimeError(
                    "cannot enable controlled envelopes during execution")
            self._notional_envelopes.clear()
            self._controlled_envelopes_required = True

    def arm_controlled_notional_envelope(
            self, *, execution_id: str, side: str, symbol: str,
            quantity: int, opening: bool,
            max_notional: Optional[float]) -> None:
        """Arm the next exact execution from a persistent authorization.

        Opening notional is a hard aggregate budget across fills and every
        cancel/replace child.  Exit envelopes deliberately carry no budget:
        manifest capacity must never prevent reducing an existing position.
        """
        identity = str(execution_id or "").strip()
        normalized_side = str(side or "").strip().upper()
        normalized_symbol = str(symbol or "").strip().upper()
        if not identity or normalized_side not in {"BUY", "SELL"}:
            raise ValueError("invalid controlled execution identity")
        if not normalized_symbol:
            raise ValueError("controlled execution symbol is required")
        if isinstance(quantity, bool) or int(quantity) < 1:
            raise ValueError("controlled execution quantity must be positive")
        budget = None
        if opening:
            if max_notional is None:
                raise ValueError(
                    "opening execution requires a finite notional budget")
            try:
                budget = float(max_notional)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "opening execution requires a finite notional budget"
                ) from error
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError(
                    "opening execution requires a finite notional budget")
        envelope = {
            "side": normalized_side,
            "symbol": normalized_symbol,
            "quantity": int(quantity),
            "opening": bool(opening),
            "max_notional": budget,
        }
        with self._state:
            if not self._controlled_envelopes_required:
                raise RuntimeError(
                    "controlled notional envelopes are not enabled")
            if self._active:
                raise RuntimeError(
                    "cannot arm a controlled envelope during execution")
            pending = self._notional_envelopes.get(identity)
            if pending is not None and pending != envelope:
                raise ControlledNotionalEnvelopeError(
                    "execution id was re-armed with different economics")
            other = set(self._notional_envelopes) - {identity}
            if other:
                raise ControlledNotionalEnvelopeError(
                    "another controlled execution is awaiting consumption")
            self._notional_envelopes[identity] = envelope

    def stop(self) -> None:
        """Permanently stop this worker and abort in-flight work.

        ``stop`` is intentionally sticky. Use :meth:`cancel_current` to pull
        just the active order while keeping the executor reusable.
        """
        self._stop_event.set()
        self._wake_event.set()

    def close(self, timeout: Optional[float] = None) -> bool:
        """Idempotently stop the worker and wait for it to become idle.

        Returns ``False`` only when *timeout* expires while broker-side
        cancellation/confirmation is still in progress.
        """
        self.stop()
        return self.await_idle(timeout)

    def await_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until no execute() call is active; return on timeout."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        with self._state:
            return self._state.wait_for(lambda: not self._active,
                                        timeout=timeout)

    def cancel_current(self, order_id: Optional[str] = None) -> bool:
        """Request cancellation of the active execution.

        The worker thread remains the sole owner of broker polling and the
        cancel-confirm race. That prevents an API thread from cancelling and
        replacing concurrently with the execution loop. If *order_id* is
        supplied it must match the currently working broker order id.
        """
        with self._state:
            if not self._active:
                return False
            current_id = (self._working_order or {}).get("order_id")
            if order_id is not None:
                if current_id is None or str(order_id) != str(current_id):
                    return False
            self._cancel_event.set()
            self._wake_event.set()
            if self._working_order is not None:
                self._working_order["status"] = "cancel_requested"
            return True

    def working_orders(self) -> List[Dict]:
        """Return a thread-safe, JSON-safe snapshot of active work.

        A PatientExecutor works at most one order, so the result contains
        zero or one entries. A list keeps the contract convenient for the
        live ``/orders`` API and permits future multi-worker aggregation.
        """
        with self._state:
            if self._working_order is None:
                return []
            snapshot = deepcopy(self._working_order)
        expires_at = datetime.fromisoformat(snapshot["expires_at"])
        snapshot["remaining_seconds"] = max(
            0.0, (expires_at - self._clock()).total_seconds())
        return [snapshot]

    def _halted(self) -> bool:
        """True when the operator stop or the kill switch forbids work."""
        return (self._stop_event.is_set() or self._cancel_event.is_set() or (
            self.kill_switch is not None and self.kill_switch.engaged())
        )

    def _wait_step(self, seconds: float) -> None:
        """Sleep between execution steps, interruptibly in real time.

        Injected sleepers retain their deterministic one-call-per-step
        behavior for simulations. The default sleeper wakes immediately for
        stop/cancel and checks a database-backed kill switch at the cancel
        polling cadence instead of going dark for a full 30-second step.
        """
        if not self._uses_default_sleep:
            self._sleep(seconds)
            return
        deadline = time.monotonic() + max(0.0, seconds)
        while not self._halted():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._wake_event.wait(min(remaining, CANCEL_POLL_S)):
                return

    def _set_working(self, **updates) -> None:
        """Atomically update the active-order observability snapshot."""
        with self._state:
            if self._working_order is not None:
                self._working_order.update(deepcopy(updates))

    # ------------------------------------------------------------------
    def execute(self, side: str, instrument_or_legs, quantity: int,
                max_minutes: float = 10, step_interval_s: float = 30,
                edge_check: Optional[Callable[[], bool]] = None,
                execution_id: Optional[str] = None,
                closing: bool = False) -> Dict:
        """Work one order per the module-docstring algorithm.

        When *execution_id* is supplied, each child order receives a
        deterministic client order id. Re-running the same logical execution
        therefore reuses the same child ids for idempotent broker recovery,
        while cancel-and-replace children use distinct sequence suffixes.
        Brokers without the optional capability retain the original placement
        call unchanged.

        ``closing`` applies only to leg packages and tells the broker to map
        tracker-shaped SHORT/BUY legs to BUY_CLOSE/SELL_CLOSE. Explicit
        lifecycle actions embedded in a leg are preserved by LiveEtradeBroker.
        """
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side {side!r} must be BUY or SELL")
        if quantity <= 0:
            raise ValueError(f"quantity {quantity} must be positive")
        is_package = isinstance(instrument_or_legs, (list, tuple))
        if closing and not is_package:
            raise ValueError("closing=True is only valid for leg packages")
        if not math.isfinite(max_minutes) or max_minutes < 0:
            raise ValueError("max_minutes must be a non-negative finite number")
        if not math.isfinite(step_interval_s) or step_interval_s < 0:
            raise ValueError(
                "step_interval_s must be a non-negative finite number")
        if execution_id is not None:
            execution_id = str(execution_id).strip()
            if not execution_id:
                raise ValueError("execution_id must be non-empty when supplied")
        start = self._clock()
        expires_at = start + timedelta(minutes=max_minutes)
        initial_snapshot: Dict = {
            "order_id": None,
            "instrument": self._instrument_label(instrument_or_legs),
            "side": side,
            "quantity": quantity,
            "filled_quantity": 0.0,
            "remaining_quantity": quantity,
            "limit_price": None,
            "status": "starting",
            "steps": [],
            "started_at": start.isoformat(),
            "expires_at": expires_at.isoformat(),
            "remaining_seconds": max_minutes * 60.0,
        }
        with self._state:
            if self._active:
                raise RuntimeError(
                    "PatientExecutor is already working an order — one "
                    "worker, one order")
            notional_envelope = None
            if self._controlled_envelopes_required:
                if execution_id is None:
                    raise ControlledNotionalEnvelopeError(
                        "controlled execution has no durable notional envelope")
                notional_envelope = self._notional_envelopes.pop(
                    execution_id, None)
                if notional_envelope is None:
                    raise ControlledNotionalEnvelopeError(
                        "controlled execution has no durable notional envelope")
                actual_symbol = str(
                    getattr(instrument_or_legs, "symbol", "") or ""
                ).strip().upper()
                expected = (
                    notional_envelope["side"],
                    notional_envelope["symbol"],
                    notional_envelope["quantity"],
                )
                actual = (side, actual_symbol, int(quantity))
                if actual != expected:
                    raise ControlledNotionalEnvelopeError(
                        "controlled execution differs from its durable "
                        "authorization")
            self._active = True
            self._cancel_event.clear()
            if not self._stop_event.is_set():
                self._wake_event.clear()
            self._working_order = initial_snapshot
        try:
            return self._run(side, instrument_or_legs, quantity,
                             max_minutes, step_interval_s, edge_check, start,
                             execution_id, closing, notional_envelope)
        finally:
            with self._state:
                self._working_order = None
                self._active = False
                self._cancel_event.clear()
                if not self._stop_event.is_set():
                    self._wake_event.clear()
                self._state.notify_all()

    # ------------------------------------------------------------------
    def _run(self, side: str, instrument, quantity: int, max_minutes: float,
             step_interval_s: float, edge_check, start: datetime,
             execution_id: Optional[str] = None,
             closing: bool = False,
             notional_envelope: Optional[Dict] = None) -> Dict:
        deadline_s = max_minutes * 60.0
        quote = self._quote(instrument)
        arrival_mid = round_tick(
            (quote["bid"] + quote["ask"]) / 2.0)
        limit = arrival_mid
        self._set_working(limit_price=limit)

        fills: List[Dict] = []
        steps: List[Dict] = []
        remaining = quantity
        # t0 gate (module docstring): an ALREADY-engaged switch (or a stop
        # that raced in) means the initial order is never placed — report
        # 'killed' instead of handing an order to a non-gating broker or
        # surfacing the live client's KillSwitchEngaged as an exception.
        if self._halted():
            logger.warning("Patient execution killed before placement: "
                           "stop/kill switch already engaged")
            return self._report("killed", side, arrival_mid, fills, steps,
                                quantity)
        # Cumulative fill bookkeeping for the CURRENT order id.
        order_id: Optional[str] = None
        order_filled = 0.0
        order_avg: Optional[float] = None
        child_sequence = 0
        filled_notional = 0.0

        def _check_opening_budget(child_quantity: int,
                                  child_limit: float) -> None:
            """Bound aggregate fills plus this child's worst-case debit."""
            if (not notional_envelope
                    or not notional_envelope["opening"]):
                return
            multiplier = int(getattr(instrument, "multiplier", 1))
            projected = (filled_notional
                         + int(child_quantity) * float(child_limit) * multiplier)
            budget = float(notional_envelope["max_notional"])
            if (not math.isfinite(projected)
                    or projected > budget + 1e-9):
                raise ControlledNotionalEnvelopeError(
                    "opening child would exceed persistently authorized "
                    f"notional ({projected:.2f} > {budget:.2f})")

        def _place_child(child_quantity: int, child_limit: float) -> str:
            """Place one child with an execution-stable id when supported."""
            nonlocal child_sequence
            _check_opening_budget(child_quantity, child_limit)
            client_order_id = None
            if execution_id is not None:
                client_order_id = self._child_client_order_id(
                    execution_id, child_sequence)
            child_sequence += 1
            return self._place(side, instrument, child_quantity, child_limit,
                               client_order_id=client_order_id,
                               closing=closing)

        def _poll() -> Optional[Dict]:
            """Bank any new fills on the current order into `fills`.
            Returns the raw status dict (None when the broker no longer
            knows the id) so callers can check for a terminal state."""
            nonlocal remaining, order_filled, order_avg, filled_notional
            status = self.broker.order_status(order_id)
            if not status:
                return None
            cum_qty = float(status.get("filled_quantity", 0) or 0)
            cum_avg = status.get("avg_fill_price")
            if cum_qty <= order_filled:
                return status
            delta_qty = cum_qty - order_filled
            # Back out the marginal price of THIS delta from the moving
            # average so the fills list carries true per-slice prices.
            if cum_avg is None:
                delta_price = limit
            else:
                cumulative_avg = float(cum_avg)
                # EtradeClient reports a signed package net (credits are
                # negative), while the executor works a positive executable
                # package price on both BUY and SELL paths. Accounting sees
                # the raw signed status through the reservation callback;
                # normalize only this execution report/shortfall surface.
                if isinstance(instrument, (list, tuple)):
                    cumulative_avg = abs(cumulative_avg)
                if order_avg is None or order_filled == 0:
                    delta_price = cumulative_avg
                else:
                    delta_price = ((cum_qty * cumulative_avg
                                    - order_filled * order_avg) / delta_qty)
            fills.append({"qty": delta_qty, "price": round(delta_price, 4),
                          "ts": self._clock().isoformat()})
            order_filled = cum_qty
            order_avg = (abs(float(cum_avg))
                         if (cum_avg is not None
                             and isinstance(instrument, (list, tuple)))
                         else (float(cum_avg)
                               if cum_avg is not None else limit))
            if (notional_envelope
                    and notional_envelope["opening"]):
                multiplier = int(getattr(instrument, "multiplier", 1))
                filled_notional += (
                    float(delta_qty) * abs(float(delta_price)) * multiplier)
            remaining = quantity - int(round(sum(f["qty"] for f in fills)))
            self._set_working(
                filled_quantity=quantity - remaining,
                remaining_quantity=remaining,
                broker_status=str(status.get("status") or ""),
            )
            if (notional_envelope
                    and notional_envelope["opening"]):
                budget = float(notional_envelope["max_notional"])
                if (not math.isfinite(filled_notional)
                        or filled_notional > budget + 1e-9):
                    raise ControlledNotionalEnvelopeError(
                        "broker fills exceeded persistently authorized "
                        f"notional ({filled_notional:.2f} > {budget:.2f})")
            return status

        def _finish(status: str) -> Dict:
            self._set_working(status=status)
            return self._report(status, side, arrival_mid, fills, steps,
                                quantity)

        def _cancel_confirmed() -> None:
            """Cancel the working order and wait until the broker says it
            is DEAD. cancel_order is only a request — a live fill can land
            between the request and the broker killing the order, so each
            confirmation poll banks racing fills (updating `remaining`)
            before any replacement is sized. A cancel that never confirms
            raises, and the mid-work-failure handler turns that into a
            terminal 'error' report: working a replacement while the old
            order may still be live is how quantity double-executes."""
            self._set_working(status="cancel_requested")
            self.broker.cancel_order(order_id)
            for _ in range(CANCEL_CONFIRM_ATTEMPTS):
                status = _poll()
                if status is None or (str(status.get("status") or "").upper()
                                      in _TERMINAL_STATUSES):
                    return
                self._sleep(CANCEL_POLL_S)
            raise RuntimeError(
                f"cancel of order {order_id} not confirmed after "
                f"{CANCEL_CONFIRM_ATTEMPTS} polls — order state unknown, "
                "standing down")

        try:
            order_id = _place_child(remaining, limit)
            steps.append({"ts": self._clock().isoformat(), "limit": limit})
            self._set_working(order_id=str(order_id), status="working",
                              limit_price=limit, steps=steps)
            logger.info("Patient %s %d @ mid %.2f (order %s)", side,
                        quantity, limit, order_id)

            _poll()  # an aggressive mid can fill instantly
            if remaining <= 0:
                return _finish("filled")

            while True:
                self._wait_step(step_interval_s)
                # 1. operator stop / kill switch — cancel and stand down.
                if self._halted():
                    _cancel_confirmed()
                    logger.warning("Patient execution killed mid-work")
                    return _finish("killed")
                # 2. fills.
                _poll()
                if remaining <= 0:
                    return _finish("filled")
                # 3. edge gone? the desk's reason to trade no longer holds.
                if edge_check is not None and not edge_check():
                    _cancel_confirmed()
                    logger.info("Edge decayed; cancelled with %d remaining",
                                remaining)
                    return _finish("edge_decayed")
                # 4. deadline. NO market-order fallback (module docstring).
                elapsed = (self._clock() - start).total_seconds()
                if elapsed >= deadline_s:
                    _cancel_confirmed()
                    logger.info("Patient window expired; cancelled with %d "
                                "remaining", remaining)
                    return _finish("partial" if fills else "timeout")
                # 5. reprice 25% of the remaining distance toward the touch.
                quote = self._quote(instrument)
                if side == "BUY":
                    touch = quote["ask"]
                    target = round_tick(limit + 0.25 * (touch - limit))
                    new_limit = min(target, touch)
                else:
                    touch = quote["bid"]
                    target = round_tick(limit - 0.25 * (limit - touch))
                    new_limit = max(target, touch)
                if new_limit != limit:
                    _cancel_confirmed()  # banks racing fills before sizing
                    if remaining <= 0:
                        return _finish("filled")
                    limit = new_limit
                    order_id = _place_child(remaining, limit)
                    order_filled, order_avg = 0.0, None
                    self._set_working(order_id=str(order_id),
                                      status="working",
                                      limit_price=limit)
                    _poll()  # marketable replacements can fill instantly
                    if remaining <= 0:
                        steps.append({"ts": self._clock().isoformat(),
                                      "limit": limit})
                        self._set_working(steps=steps)
                        return _finish("filled")
                steps.append({"ts": self._clock().isoformat(),
                              "limit": limit})
                self._set_working(limit_price=limit, steps=steps)
        except Exception as error:  # noqa: BLE001 - terminal report, never a crash
            # MID-WORK FAILURE (module docstring): the midnight-ET auth
            # expiry striking between polls, a dead transport, a gating
            # broker refusing a replacement. NEVER leave a working order
            # with no recorded outcome: best-effort cancel, bank any
            # fills that raced the cancel, and return a terminal 'error'
            # report carrying the typed reason — the session audits it
            # and halts. The exception does not propagate.
            if order_id is not None:
                try:
                    self.broker.cancel_order(order_id)
                except Exception as cancel_error:  # noqa: BLE001
                    logger.error(
                        "Best-effort cancel of %s after a mid-work "
                        "failure itself failed (%s: %s) — confirm the "
                        "order state manually / at reconciliation",
                        order_id, type(cancel_error).__name__,
                        cancel_error)
                try:
                    _poll()
                except Exception:  # noqa: BLE001
                    pass  # fills-so-far stand as last known
            logger.error("Patient execution aborted mid-work: %s: %s",
                         type(error).__name__, error)
            report = _finish("error")
            report["error"] = str(error)
            report["error_type"] = type(error).__name__
            return report

    # ------------------------------------------------------------------
    @staticmethod
    def _instrument_label(instrument) -> str:
        """Stable, JSON-safe label for the working-orders surface."""
        if isinstance(instrument, (list, tuple)):
            labels = []
            for leg in instrument:
                if isinstance(leg, dict):
                    action = str(leg.get("action") or "").upper()
                    asset = leg.get("asset")
                    labels.append(" ".join(
                        part for part in (action, str(asset)) if part))
                else:
                    labels.append(str(leg))
            return " / ".join(labels) or "option package"
        return str(instrument)

    def _quote(self, instrument) -> Dict[str, float]:
        """Fetch and validate an exact-instrument/package bid/ask.

        The injected callable receives the option Asset itself or the entire
        leg package. There is deliberately no symbol extraction and no
        broker ``get_current_price`` fallback: an underlying stock quote is
        not a valid executable option/package quote.
        """
        is_package = isinstance(instrument, (list, tuple))
        is_option = (isinstance(instrument, Asset)
                     and instrument.asset_type in
                     (AssetType.CALL, AssetType.PUT))
        quote_kind = "option/package" if is_package or is_option else "asset"
        quote = self.quote_fn(instrument)
        if not isinstance(quote, dict):
            raise ValueError(
                f"injected quote_fn must return a {quote_kind} bid/ask dict")
        try:
            bid = float(quote["bid"])
            ask = float(quote["ask"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"injected quote_fn must return finite {quote_kind} bid/ask"
            ) from error
        if (not math.isfinite(bid) or not math.isfinite(ask)
                or bid < 0 or ask <= 0 or bid > ask):
            raise ValueError(
                f"injected quote_fn returned invalid {quote_kind} bid/ask: "
                f"bid={bid!r}, ask={ask!r}")
        return {"bid": bid, "ask": ask}

    def _place(self, side: str, instrument, quantity: int,
               limit: float,
               client_order_id: Optional[str] = None,
               closing: bool = False) -> str:
        if isinstance(instrument, (list, tuple)):
            # Multi-leg package: SELL works a credit (+net), BUY a debit
            # (-net) per build_spread_order's sign convention.
            net = limit if side == "SELL" else -limit
            idempotent_placer = getattr(
                self.broker, "place_structure_with_client_id", None)
            if client_order_id is not None and callable(idempotent_placer):
                return idempotent_placer(
                    list(instrument), net, quantity, client_order_id,
                    closing=closing)
            if closing:
                return self.broker.place_structure(
                    list(instrument), net, quantity, closing=True)
            return self.broker.place_structure(list(instrument), net, quantity)
        order_type = OrderType.BUY if side == "BUY" else OrderType.SELL
        idempotent_placer = getattr(
            self.broker, "place_order_with_client_id", None)
        if client_order_id is not None and callable(idempotent_placer):
            return idempotent_placer(
                instrument, order_type, quantity, limit, client_order_id)
        return self.broker.place_order(instrument, order_type, quantity,
                                       limit)

    @staticmethod
    def _child_client_order_id(execution_id: str, child_sequence: int) -> str:
        """Return an E*TRADE-safe identity for one execution child.

        Twelve hex characters bind the id to the logical execution; an
        eight-character hexadecimal sequence makes every replacement within
        that execution distinct. The result is deterministically 20
        alphanumeric characters, E*TRADE's documented maximum.
        """
        if not 0 <= child_sequence <= 0xFFFFFFFF:
            raise OverflowError("patient execution child sequence exhausted")
        execution_digest = hashlib.sha256(
            execution_id.encode("utf-8")).hexdigest()[:12]
        return f"{execution_digest}{child_sequence:08x}"

    def _report(self, status: str, side: str, arrival_mid: float,
                fills: List[Dict], steps: List[Dict],
                quantity: int) -> Dict:
        filled_qty = sum(f["qty"] for f in fills)
        avg_fill = (sum(f["qty"] * f["price"] for f in fills) / filled_qty
                    if filled_qty else None)
        if avg_fill is None:
            shortfall = None
        elif side == "BUY":
            shortfall = avg_fill - arrival_mid
        else:
            shortfall = arrival_mid - avg_fill
        return {
            "status": status,
            "fills": fills,
            "arrival_mid": arrival_mid,
            "avg_fill": avg_fill,
            "shortfall_per_unit": shortfall,
            "steps": steps,
        }


#: Public alias for the report dict shape (documentation/type hinting).
ExecutionReport = Dict
