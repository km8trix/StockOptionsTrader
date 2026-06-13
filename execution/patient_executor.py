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
    place_structure(legs, net_price, contracts)          -> order_id
        (only needed when working multi-leg packages)
    cancel_order(order_id) -> bool
    order_status(order_id) -> {'status', 'filled_quantity',
                               'avg_fill_price'} | None

quote_fn(instrument_or_legs) -> {'bid': float, 'ask': float} — for a
multi-leg package these are the NET package bid/ask.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from core.models import OrderType
from utils.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

TICK = 0.01


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
        self.broker = broker
        self.quote_fn = quote_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep_fn
        self.kill_switch = kill_switch
        #: set by stop() (or kill-switch engagement) to abort mid-work.
        self._stop_event = threading.Event()
        #: one order worked at a time — overlapping execute() calls on a
        #: single worker are an ambiguity factory.
        self._busy = threading.Lock()

    def stop(self) -> None:
        """Abort the in-flight execution at its next step ('killed')."""
        self._stop_event.set()

    def _halted(self) -> bool:
        """True when the operator stop or the kill switch forbids work."""
        return self._stop_event.is_set() or (
            self.kill_switch is not None and self.kill_switch.engaged())

    # ------------------------------------------------------------------
    def execute(self, side: str, instrument_or_legs, quantity: int,
                max_minutes: float = 10, step_interval_s: float = 30,
                edge_check: Optional[Callable[[], bool]] = None) -> Dict:
        """Work one order per the module-docstring algorithm."""
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side {side!r} must be BUY or SELL")
        if quantity <= 0:
            raise ValueError(f"quantity {quantity} must be positive")
        if not self._busy.acquire(blocking=False):
            raise RuntimeError(
                "PatientExecutor is already working an order — one worker, "
                "one order")
        try:
            self._stop_event.clear()
            return self._run(side, instrument_or_legs, quantity,
                             max_minutes, step_interval_s, edge_check)
        finally:
            self._busy.release()

    # ------------------------------------------------------------------
    def _run(self, side: str, instrument, quantity: int, max_minutes: float,
             step_interval_s: float, edge_check) -> Dict:
        start = self._clock()
        deadline_s = max_minutes * 60.0
        quote = self.quote_fn(instrument)
        arrival_mid = round_tick(
            (float(quote["bid"]) + float(quote["ask"])) / 2.0)
        limit = arrival_mid

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

        def _poll() -> None:
            """Bank any new fills on the current order into `fills`."""
            nonlocal remaining, order_filled, order_avg
            status = self.broker.order_status(order_id)
            if not status:
                return
            cum_qty = float(status.get("filled_quantity", 0) or 0)
            cum_avg = status.get("avg_fill_price")
            if cum_qty <= order_filled:
                return
            delta_qty = cum_qty - order_filled
            # Back out the marginal price of THIS delta from the moving
            # average so the fills list carries true per-slice prices.
            if cum_avg is None:
                delta_price = limit
            elif order_avg is None or order_filled == 0:
                delta_price = float(cum_avg)
            else:
                delta_price = ((cum_qty * float(cum_avg)
                                - order_filled * order_avg) / delta_qty)
            fills.append({"qty": delta_qty, "price": round(delta_price, 4),
                          "ts": self._clock().isoformat()})
            order_filled = cum_qty
            order_avg = float(cum_avg) if cum_avg is not None else limit
            remaining = quantity - int(round(sum(f["qty"] for f in fills)))

        def _finish(status: str) -> Dict:
            return self._report(status, side, arrival_mid, fills, steps,
                                quantity)

        try:
            order_id = self._place(side, instrument, remaining, limit)
            steps.append({"ts": self._clock().isoformat(), "limit": limit})
            logger.info("Patient %s %d @ mid %.2f (order %s)", side,
                        quantity, limit, order_id)

            _poll()  # an aggressive mid can fill instantly
            if remaining <= 0:
                return _finish("filled")

            while True:
                self._sleep(step_interval_s)
                # 1. operator stop / kill switch — cancel and stand down.
                if self._halted():
                    self.broker.cancel_order(order_id)
                    _poll()  # capture any race-the-cancel fills
                    logger.warning("Patient execution killed mid-work")
                    return _finish("killed")
                # 2. fills.
                _poll()
                if remaining <= 0:
                    return _finish("filled")
                # 3. edge gone? the desk's reason to trade no longer holds.
                if edge_check is not None and not edge_check():
                    self.broker.cancel_order(order_id)
                    _poll()
                    logger.info("Edge decayed; cancelled with %d remaining",
                                remaining)
                    return _finish("edge_decayed")
                # 4. deadline. NO market-order fallback (module docstring).
                elapsed = (self._clock() - start).total_seconds()
                if elapsed >= deadline_s:
                    self.broker.cancel_order(order_id)
                    _poll()
                    logger.info("Patient window expired; cancelled with %d "
                                "remaining", remaining)
                    return _finish("partial" if fills else "timeout")
                # 5. reprice 25% of the remaining distance toward the touch.
                quote = self.quote_fn(instrument)
                if side == "BUY":
                    touch = float(quote["ask"])
                    target = round_tick(limit + 0.25 * (touch - limit))
                    new_limit = min(target, touch)
                else:
                    touch = float(quote["bid"])
                    target = round_tick(limit - 0.25 * (limit - touch))
                    new_limit = max(target, touch)
                if new_limit != limit:
                    self.broker.cancel_order(order_id)
                    _poll()  # fills that raced the cancel still count
                    if remaining <= 0:
                        return _finish("filled")
                    limit = new_limit
                    order_id = self._place(side, instrument, remaining,
                                           limit)
                    order_filled, order_avg = 0.0, None
                    _poll()  # marketable replacements can fill instantly
                    if remaining <= 0:
                        steps.append({"ts": self._clock().isoformat(),
                                      "limit": limit})
                        return _finish("filled")
                steps.append({"ts": self._clock().isoformat(),
                              "limit": limit})
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
    def _place(self, side: str, instrument, quantity: int,
               limit: float) -> str:
        if isinstance(instrument, (list, tuple)):
            # Multi-leg package: SELL works a credit (+net), BUY a debit
            # (-net) per build_spread_order's sign convention.
            net = limit if side == "SELL" else -limit
            return self.broker.place_structure(list(instrument), net,
                                               quantity)
        order_type = OrderType.BUY if side == "BUY" else OrderType.SELL
        return self.broker.place_order(instrument, order_type, quantity,
                                       limit)

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
