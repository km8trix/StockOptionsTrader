"""
Local-book vs broker reconciliation (contract C19).

reconcile() compares what WE think we hold (the local book) against what
the BROKER says we hold. Position quantities are compared symbol-by-symbol
in NATIVE units — shares for equities, CONTRACTS for options (the broker's
portfolio reports option quantity in contracts; market values differ x100
from share-math, which is why comparisons happen on quantities and any
value math goes through position_market_value's explicit multiplier).
Cash must agree within $0.01.

ON MISMATCH THE CALLER ENGAGES THE KILL SWITCH — a book that disagrees
with the broker means something filled, expired, or got assigned that the
session does not know about, and trading on a wrong book compounds the
error. The wiring lives in utils.live_session.LiveTradingSession.

Option position keys: callers key option instruments with the canonical
``str(Asset)`` form ("SPY 2026-07-17 $440.0 put") so two distinct
contracts on one underlying never collapse into one key.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Cash agreement tolerance (dollars).
CASH_TOLERANCE = 0.01
#: Quantity agreement tolerance (float dust only — quantities are ints).
QTY_TOLERANCE = 1e-9


def position_market_value(quantity: float, price: float,
                          multiplier: int = 1) -> float:
    """Market value with an EXPLICIT contract multiplier (x100 options)."""
    return float(quantity) * float(price) * multiplier


def reconcile(local_positions: Dict[str, float], local_cash: float,
              broker, clock: Optional[Callable[[], datetime]] = None) -> Dict:
    """Compare the local book against broker.get_portfolio_status().

    Args:
        local_positions: {position key: signed quantity} — shares for
            stock, contracts for options (str(Asset) keys for options).
        local_cash: the local book's cash balance.
        broker: anything exposing get_portfolio_status() -> {'cash',
            'positions': [{'symbol', 'quantity', ...}]} (ExecutionBroker
            contract; LiveEtradeBroker maps option instruments to
            canonical keys).

    Returns (contract C19):
        {'ok': bool,
         'mismatches': [{'kind': 'position'|'cash', 'symbol': str|None,
                         'local': float, 'broker': float}],
         'checked_at': iso}
    """
    clock = clock or (lambda: datetime.now(timezone.utc))
    status = broker.get_portfolio_status()
    broker_positions: Dict[str, float] = {}
    for position in status.get("positions", []):
        key = position["symbol"]
        # Two broker rows on one key (should not happen with canonical
        # option keys) still reconcile on the summed quantity.
        broker_positions[key] = (broker_positions.get(key, 0.0)
                                 + float(position["quantity"]))

    mismatches: List[Dict] = []
    for key in sorted(set(local_positions) | set(broker_positions)):
        local_qty = float(local_positions.get(key, 0.0))
        broker_qty = float(broker_positions.get(key, 0.0))
        if abs(local_qty - broker_qty) > QTY_TOLERANCE:
            mismatches.append({"kind": "position", "symbol": key,
                               "local": local_qty, "broker": broker_qty})

    broker_cash = float(status.get("cash", 0.0))
    if abs(float(local_cash) - broker_cash) > CASH_TOLERANCE:
        mismatches.append({"kind": "cash", "symbol": None,
                           "local": float(local_cash),
                           "broker": broker_cash})

    ok = not mismatches
    if not ok:
        logger.error("Reconciliation FAILED with %d mismatch(es): %s",
                     len(mismatches), mismatches)
    else:
        logger.info("Reconciliation clean: %d position keys, cash within "
                    "$%.2f", len(broker_positions), CASH_TOLERANCE)
    return {"ok": ok, "mismatches": mismatches,
            "checked_at": clock().isoformat()}
