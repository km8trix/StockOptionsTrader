"""Coherent authoritative paper-ledger fixtures shared by deployment tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json

import numpy as np
import pandas as pd

from analysis.promotion import PromotionArtifact
from desks.foundation import FoundationDesk
from utils.market_hours import MarketHours


def _previous_session(day: date) -> date:
    calendar = MarketHours()
    candidate = day - timedelta(days=1)
    while not calendar.is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


@lru_cache(maxsize=32)
def _fitted_checkpoint(symbols: tuple[str, ...], signal_end: date) -> dict:
    """Build one real, safely restorable Foundation checkpoint."""
    index = pd.bdate_range(end=pd.Timestamp(signal_end), periods=140)
    step = np.arange(len(index), dtype=float)
    close = 100.0 + 0.04 * step + 2.0 * np.sin(step / 3.0)
    data = {}
    for offset, symbol in enumerate(symbols):
        values = close + float(offset)
        data[symbol] = pd.DataFrame({
            "open": values - 0.2,
            "high": values + 0.8,
            "low": values - 0.8,
            "close": values,
            "volume": 1_000.0 + (step % 11) * 10.0,
        }, index=index)
    desk = FoundationDesk()
    assert desk._controller.maybe_refit(data, pd.Timestamp(signal_end)) is True
    return desk.model_checkpoint_state()


def paper_model_checkpoint(
        universe: list[str], final_execution: datetime) -> dict:
    signal_end = _previous_session(final_execution.date())
    # Return a deep JSON copy so callers can mutate negative-test fixtures.
    return json.loads(json.dumps(_fitted_checkpoint(
        tuple(universe), signal_end), allow_nan=False))


def authoritative_paper_evidence(
        artifact: PromotionArtifact, *, fills: int = 2,
        final_session: date = date(2026, 7, 10)) -> dict:
    """Build twenty ordered cycles over fifteen real NYSE sessions."""
    if fills < 2 or fills % 2:
        raise ValueError("paper fixture fills must be an even round trip")
    calendar = MarketHours()
    session_days: list[date] = []
    candidate = final_session
    while len(session_days) < 15:
        if calendar.is_trading_day(candidate):
            session_days.append(candidate)
        candidate -= timedelta(days=1)
    session_days.reverse()
    session_times = [
        datetime.combine(day, datetime.min.time(), timezone.utc).replace(
            hour=14)
        for day in session_days
    ]
    execution_times: list[datetime] = []
    for index, session in enumerate(session_times):
        execution_times.append(session)
        if index < 5:
            execution_times.append(session + timedelta(hours=1))

    universe = list(artifact.payload["universe"])
    half = fills // 2
    final_orders = []
    for index in range(fills):
        side = "BUY" if index < half else "SELL"
        symbol = universe[index % len(universe)]
        final_orders.append({
            "order_id": f"paper-order-{index:04d}",
            "symbol": symbol,
            "asset_type": "stock",
            "side": side,
            "quantity": 1,
            "filled_quantity": 1,
            "status": "FILLED",
        })
    # Keep every symbol flat even when the fixture has a multi-name universe.
    for index in range(half, fills):
        final_orders[index]["symbol"] = final_orders[index - half]["symbol"]

    cycles = []
    for index, as_of in enumerate(execution_times):
        quote_time = as_of - timedelta(seconds=1)
        observed_at = as_of + timedelta(seconds=1)
        visible_orders = final_orders[:min(index + 1, len(final_orders))]
        cycles.append({
            "cycle_id": hashlib.sha256(
                as_of.isoformat().encode("utf-8")).hexdigest(),
            "as_of": as_of.isoformat(),
            "observed_at": observed_at.isoformat(),
            "started_at": observed_at.isoformat(),
            "completed_at": (as_of + timedelta(seconds=2)).isoformat(),
            "input_hash": hashlib.sha256(
                f"input-{index}".encode("utf-8")).hexdigest(),
            "execution_prices": {
                symbol: {
                    "price": 100.0 + index,
                    "observed_at": quote_time.isoformat(),
                    "source": "synthetic-contract-quote",
                }
                for symbol in universe
            },
            "pre_reconciliation": {"ok": True, "mismatches": []},
            "result": {"status": "ok", "reports": []},
            "post_reconciliation": {"ok": True, "mismatches": []},
            "orders": visible_orders,
            "error_count": 0,
        })
    state = paper_model_checkpoint(universe, execution_times[-1])
    return {
        "runner": "foundation_paper_rehearsal_v2",
        "run_id": "authoritative-paper-fixture",
        "started_at": (execution_times[0] - timedelta(minutes=1)).isoformat(),
        "universe": universe,
        "cycles": cycles,
        "model_checkpoint": {
            "cycle_id": cycles[-1]["cycle_id"],
            "sha256": state["sha256"],
            "state": state,
        },
        "final_reconciliation": {"ok": True, "mismatches": []},
        "broker": {
            "cash": 100_000.0,
            "positions": [],
            "orders": final_orders,
        },
        "local_book": {
            "cash": 100_000.0,
            "positions": {},
            "initialized": True,
        },
        "audit_verification": {"ok": True, "first_bad_seq": None},
    }


__all__ = ["authoritative_paper_evidence", "paper_model_checkpoint"]
