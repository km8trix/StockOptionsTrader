"""Step 5 operational shakedown — prove the live loop end to end.

THE CHECKLIST (each step prints PASS/FAIL; first FAIL exits nonzero):
  1. build the rig: PaperTrader (deterministic prices, no network) +
     synthetic two-symbol daily bars + a minimal inline desk +
     KillSwitch + LocalBook + LiveTradingSession
  2. submit -> fill -> order_status: evaluate_once places and fills an
     order; order_status(order_id) reports FILLED with qty/price
  3. PatientExecutor parity: one order worked through the polling
     contract against the same paper broker class
  4. LocalBook persisted: the sqlite file holds the position + cash
  5. reconcile-across-restart: discard EVERY in-memory object, rebuild
     from disk, run_reconciliation() -> ok=True, zero drift
  6. drift detection (fail-closed): corrupt one book quantity ->
     reconcile ok=False AND the kill switch ENGAGES
  7. breaker halt: operator kill switch halts evaluate_once; an
     injected breached circuit breaker halts it too
  8. summary table, exit 0

WHAT THE PAPER LEG DOES **NOT** PROVE (honesty box):
  * real OAuth 1.0a transport (token signing, midnight-ET expiry,
    renew) — nothing here touches the network;
  * real fills — paper fills all-or-nothing at the cached price with a
    synthetic 0.1% slippage the broker does not report per-fill (the
    book therefore books the raw fill price; that unreported-fee class
    of drift is exactly what reconciliation exists to catch on live);
  * E*TRADE's idempotency machine (clientOrderId retry/find-order) and
    order-state vocabulary beyond OPEN/FILLED/CANCELLED/REJECTED.
  Run ``--broker sandbox`` DURING MARKET HOURS to cover those.

RESTART REPLAY IS PAPER-ONLY SCAFFOLDING: a REAL broker holds its own
book server-side, so after a restart the broker side of reconcile is
simply queried. PaperTrader's book is in-memory and dies with the
process, so step 5 rebuilds the fresh paper broker's portfolio FROM the
LocalBook snapshot — that makes ok=True trivially true on paper, and is
why step 6 (a corrupted book MUST fail the reconcile and engage the
kill switch) is part of the same checklist: together they prove the
persistence, reload, and fail-closed wiring, which is what the paper
leg is for.

Usage (headless, rerunnable):
  .venv/bin/python scripts/step5_shakedown.py                # paper (default)
  ETRADE_ALLOW_NETWORK=1 ETRADE_ENV=sandbox \\
      .venv/bin/python scripts/step5_shakedown.py --broker sandbox
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from brokers.local_book import LocalBook  # noqa: E402
from brokers.paper_trader import PaperTrader  # noqa: E402
from core.models import Asset, AssetType, Position  # noqa: E402
from desks.base import DeskIntent  # noqa: E402
from execution.patient_executor import PatientExecutor  # noqa: E402
from utils.audit import AuditLog  # noqa: E402
from utils.kill_switch import KillSwitch  # noqa: E402
from utils.live_session import LiveTradingSession  # noqa: E402

#: Deterministic prices — the whole paper leg is arithmetic on these.
PRICES = {"SHAKE1": 100.0, "SHAKE2": 40.0}
INITIAL_CASH = 100_000.0
QTY1, QTY2 = 10, 5

_results: list[tuple[str, str]] = []


def _step(number: int, name: str, ok: bool, detail: str = "") -> None:
    verdict = "PASS" if ok else "FAIL"
    _results.append((f"{number}. {name}", verdict))
    print(f"[{verdict}] step {number}: {name}" + (f" — {detail}" if detail
                                                  else ""))
    if not ok:
        _summary()
        sys.exit(number)


def _summary() -> None:
    print("\n=== Step 5 shakedown summary ===")
    width = max(len(n) for n, _ in _results) if _results else 0
    for name, verdict in _results:
        print(f"  {name:<{width}}  {verdict}")


class DeterministicPaperTrader(PaperTrader):
    """PaperTrader with pinned prices — NO network, NO MarketDataHandler
    fetch. Everything else (order queue, fills, order_status, the
    portfolio book) is the real PaperTrader."""

    def get_current_price(self, symbol: str):
        return PRICES.get(symbol)


class ShakedownDesk:
    """Minimal deterministic desk: emits two BUY intents with explicit
    quantities on the first evaluation, then goes quiet. Deterministic
    intents matter more than realism here (module docstring)."""

    key = "shakedown"
    capital_allocation = 1.0

    def __init__(self):
        self._fired = False

    def set_clock(self, now):
        self.now = now

    def generate_intents(self, all_data, date, portfolio):
        if self._fired:
            return []
        self._fired = True
        return [
            DeskIntent(Asset("SHAKE1", AssetType.STOCK), "BUY", 0.01,
                       "shakedown", quantity=QTY1),
            DeskIntent(Asset("SHAKE2", AssetType.STOCK), "BUY", 0.01,
                       "shakedown", quantity=QTY2),
        ]

    def apply_risk(self, intents, portfolio, all_data, date):
        return list(intents)


def synthetic_bars() -> dict:
    """Two symbols, five deterministic daily bars each — no network."""
    idx = pd.bdate_range("2026-07-01", periods=5)
    out = {}
    for symbol, price in PRICES.items():
        out[symbol] = pd.DataFrame({
            "open": [price] * 5, "high": [price * 1.01] * 5,
            "low": [price * 0.99] * 5, "close": [price] * 5,
            "volume": [1_000_000] * 5,
        }, index=idx)
    return out


def _build_session(broker, book, audit, switch, circuit_breaker=None):
    return LiveTradingSession(
        desk=ShakedownDesk(), broker=broker, portfolio=broker.portfolio,
        data_fn=synthetic_bars, executor=None, audit=audit,
        kill_switch=switch, circuit_breaker=circuit_breaker,
        local_book=book, enforce_market_hours=False)


def _executor_parity_check() -> None:
    """Step 3: one order worked through PatientExecutor against the paper
    broker to prove the polling contract. Its own rig (locals die on
    return) — the main session's book/broker stay untouched."""
    parity_broker = DeterministicPaperTrader(initial_capital=INITIAL_CASH)
    price = PRICES["SHAKE1"]
    executor = PatientExecutor(
        parity_broker,
        quote_fn=lambda inst: {"bid": price - 0.02, "ask": price + 0.02},
        # Virtual sleep doubles as the paper fill pump: each executor step
        # processes pending orders, exactly the poll-advances-the-world
        # rhythm the live polling contract assumes.
        sleep_fn=lambda s: parity_broker.process_orders())
    report = executor.execute("BUY", Asset("SHAKE1", AssetType.STOCK), 5,
                              max_minutes=1, step_interval_s=0.01)
    filled_qty = sum(f["qty"] for f in report["fills"])
    ok = (report["status"] == "filled" and filled_qty == 5
          and report["avg_fill"] == price)
    _step(3, "PatientExecutor polling contract (paper parity)", ok,
          f"status={report['status']} filled={filled_qty} "
          f"avg={report['avg_fill']}")


def run_paper() -> int:
    tmp = tempfile.mkdtemp(prefix="step5_shakedown_")
    rails_db = os.path.join(tmp, "rails.db")
    book_db = os.path.join(tmp, "local_book.db")
    print(f"Scratch state in {tmp} (rails.db + local_book.db)\n")

    # -- 1. build the rig -------------------------------------------------
    audit = AuditLog(rails_db, env="sandbox")
    switch = KillSwitch(rails_db, audit=audit)
    if switch.engaged():  # fresh tmp db — never true; belt and braces
        switch.disengage(actor="step5_shakedown")
    book = LocalBook(book_db, env="sandbox")
    book.set_cash(INITIAL_CASH)
    broker = DeterministicPaperTrader(initial_capital=INITIAL_CASH)
    session = _build_session(broker, book, audit, switch)
    _step(1, "rig built (paper broker + rails + local book)", True,
          f"cash funded {INITIAL_CASH:,.0f}")

    # -- 2. submit -> fill -> order_status --------------------------------
    result = session.evaluate_once()
    reports = result.get("reports", [])
    ok = (result["status"] == "ok" and len(reports) == 2
          and all(r.get("order_id") for r in reports))
    _step(2, "evaluate_once placed orders", ok,
          f"status={result['status']} reports={len(reports)}")
    st1 = broker.order_status(reports[0]["order_id"])
    st2 = broker.order_status(reports[1]["order_id"])
    ok = (st1 is not None and st1["status"] == "FILLED"
          and st1["filled_quantity"] == QTY1
          and st1["avg_fill_price"] == PRICES["SHAKE1"]
          and st2 is not None and st2["status"] == "FILLED"
          and st2["filled_quantity"] == QTY2
          and st2["avg_fill_price"] == PRICES["SHAKE2"])
    _step(2, "order_status reports FILLED with qty/price", ok,
          f"SHAKE1={st1} SHAKE2={st2}")

    # -- 3. PatientExecutor parity ----------------------------------------
    _executor_parity_check()

    # -- 4. LocalBook persisted to sqlite ----------------------------------
    conn = sqlite3.connect(book_db)
    try:
        rows = dict(conn.execute(
            "SELECT symbol, quantity FROM local_book").fetchall())
        cash_row = conn.execute("SELECT cash FROM local_cash").fetchone()
    finally:
        conn.close()
    expected_cash = (INITIAL_CASH - QTY1 * PRICES["SHAKE1"]
                     - QTY2 * PRICES["SHAKE2"])
    ok = (rows.get("SHAKE1") == QTY1 and rows.get("SHAKE2") == QTY2
          and cash_row is not None
          and abs(cash_row[0] - expected_cash) < 1e-6)
    _step(4, "LocalBook persisted (positions + cash on disk)", ok,
          f"rows={rows} cash={cash_row and cash_row[0]}")

    # -- 5. RESTART: rebuild everything from disk, reconcile zero-drift ----
    # Discard EVERY in-memory object. A REAL broker holds its own book
    # server-side; PaperTrader's dies with the process, so the fresh
    # paper broker is rebuilt FROM the LocalBook snapshot (paper-only
    # scaffolding — see module docstring honesty box).
    del session, broker, book
    book2 = LocalBook(book_db, env="sandbox")          # fresh, same file
    snapshot = book2.snapshot()
    broker2 = DeterministicPaperTrader(initial_capital=0.0)
    for key, pos in snapshot["positions"].items():
        broker2.portfolio.add_position(Position(
            asset=Asset(key, AssetType.STOCK),
            quantity=int(pos["quantity"]),
            avg_entry_price=pos["avg_price"],
            current_price=PRICES.get(key, pos["avg_price"]),
            timestamp=datetime.now(timezone.utc)))
    broker2.portfolio.cash = snapshot["cash"]
    audit2 = AuditLog(rails_db, env="sandbox")
    switch2 = KillSwitch(rails_db, audit=audit2)
    session2 = _build_session(broker2, book2, audit2, switch2)
    recon = session2.run_reconciliation()              # book-aware, no args
    ok = (recon["ok"] is True and recon["mismatches"] == []
          and not switch2.engaged())
    _step(5, "reconcile-across-restart: zero drift", ok,
          f"ok={recon['ok']} mismatches={recon['mismatches']}")

    # -- 6. drift detection: corrupted book MUST fail closed ---------------
    conn = sqlite3.connect(book_db)
    try:
        conn.execute("UPDATE local_book SET quantity = quantity + 7 "
                     "WHERE symbol = 'SHAKE1'")
        conn.commit()
    finally:
        conn.close()
    recon = session2.run_reconciliation()
    ok = (recon["ok"] is False and len(recon["mismatches"]) == 1
          and switch2.engaged())
    _step(6, "drift detection engages the kill switch (fail-closed)", ok,
          f"ok={recon['ok']} mismatches={len(recon['mismatches'])} "
          f"kill_switch={switch2.engaged()}")
    switch2.disengage(actor="step5_shakedown")
    book2.record_fill("SHAKE1", -7, 0.0)               # undo the corruption

    # -- 7. breaker halt ----------------------------------------------------
    switch2.engage(reason="step5 operator drill", actor="step5_shakedown")
    halted = session2.evaluate_once()
    ok = (halted["status"] == "halted"
          and halted["reason"] == "kill_switch_engaged")
    _step(7, "operator kill switch halts evaluate_once", ok,
          f"status={halted['status']} reason={halted.get('reason')}")
    switch2.disengage(actor="step5_shakedown")
    breached_session = _build_session(
        broker2, book2, audit2, switch2,
        circuit_breaker=lambda: {"breached": True, "loss_pct": 0.05,
                                 "limit_pct": 0.02})
    halted = breached_session.evaluate_once()
    ok = (halted["status"] == "halted"
          and halted["reason"] == "daily_loss_circuit_breaker")
    _step(7, "breached circuit breaker halts evaluate_once", ok,
          f"status={halted['status']} reason={halted.get('reason')}")

    # -- 8. summary ----------------------------------------------------------
    _summary()
    print("\nAll paper-leg checks PASSED. Remember the honesty box: run "
          "--broker sandbox during market hours to cover the real "
          "transport.")
    return 0


def run_sandbox() -> int:
    """Sandbox leg: the same submit->status->cancel loop over the REAL
    OAuth transport. Requires ETRADE_ALLOW_NETWORK=1 and a valid token
    (connect via the GUI auth flow first). NOTE: E*TRADE sandbox order
    endpoints behave best DURING MARKET HOURS — off-hours responses can
    be stale or rejected."""
    print("NOTE: run this leg during NYSE market hours — sandbox order "
          "endpoints misbehave off-hours.\n")
    if os.environ.get("ETRADE_ALLOW_NETWORK") != "1":
        print("[FAIL] sandbox leg needs ETRADE_ALLOW_NETWORK=1 (the "
              "deliberate human act that authorizes network contact).")
        return 10
    from brokers.etrade_auth import EtradeAuthManager
    from brokers.etrade_client import EtradeClient
    from brokers.live_trader import LiveEtradeBroker

    auth = EtradeAuthManager()
    state = auth.status()["state"]
    if state != "connected":
        print(f"[FAIL] E*TRADE auth is '{state}', not 'connected' — the "
              "token is stale or missing. Reconnect via the GUI auth "
              "flow (Sandbox tab) and rerun.")
        return 11
    try:
        auth.renew()
        client = EtradeClient(auth)
        accounts = client.list_accounts()
        if not accounts:
            print("[FAIL] no sandbox accounts returned")
            return 12
        account_id_key = accounts[0]["accountIdKey"]
        broker = LiveEtradeBroker(client, account_id_key)
        print(f"[PASS] connected: accountIdKey={account_id_key}")
        status = broker.get_portfolio_status()
        print(f"[PASS] portfolio snapshot: cash={status.get('cash')} "
              f"positions={len(status.get('positions', []))}")
        # Far-from-market limit so this NEVER fills; cancelled in finally.
        from core.models import OrderType
        order_id = None
        try:
            order_id = broker.place_order(
                Asset("AAPL", AssetType.STOCK), OrderType.BUY, 1, 1.00)
            print(f"[PASS] order placed: {order_id}")
            st = broker.order_status(order_id)
            print(f"[PASS] order_status: {st}")
        finally:
            if order_id is not None:
                cancelled = broker.cancel_order(order_id)
                print(f"[{'PASS' if cancelled else 'FAIL'}] cancel: "
                      f"{cancelled}")
        print("\nSandbox leg complete — transport, order_status and "
              "cancel exercised for real.")
        return 0
    except Exception as e:  # noqa: BLE001 - operator-facing script
        print(f"[FAIL] sandbox leg raised {type(e).__name__}: {e}\n"
              "Most common causes: stale/expired token (reconnect via "
              "the GUI), market closed, or sandbox flakiness.")
        return 13


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 5 operational shakedown (module docstring)")
    parser.add_argument("--broker", choices=("paper", "sandbox"),
                        default="paper")
    args = parser.parse_args()
    return run_paper() if args.broker == "paper" else run_sandbox()


if __name__ == "__main__":
    sys.exit(main())
