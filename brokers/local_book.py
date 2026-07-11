"""
Persistent local trade ledger (Step 5 — the restart-surviving book).

WHY THIS EXISTS: reconciliation (brokers.reconcile, contract C19) compares
what WE think we hold against what the BROKER says we hold. That check is
only meaningful when "what we think we hold" survives a process restart —
PaperTrader's book is in-memory and LiveEtradeBroker deliberately keeps no
local book at all, so before this module a restarted session reconciled an
EMPTY book and learned nothing. LocalBook records every CONFIRMED fill in
SQLite so the session that comes back after a crash can ask "does my
ledger still match the broker?" and fail closed when it does not.

The book records fills as the broker CONFIRMS them (order_status /
executor fill reports), never intents — an intent that never filled must
not appear here. Cash moves by -signed_qty * price per fill; commissions,
fees, and paper-mode slippage are NOT modeled (the broker does not report
them per-fill today), which is exactly the class of drift reconciliation
exists to surface.

Position keys follow the reconcile convention: plain symbol for stock,
canonical ``str(Asset)`` for options ("SPY 2026-07-17 $440.0 put").

CONCURRENCY: same pattern as brokers.etrade_auth — one short-lived
sqlite3.connect per call, never a shared connection, so any thread (or a
second process) can use the book safely; the read-modify-write in
record_fill runs inside BEGIN IMMEDIATE so concurrent fills serialize.

Rows are scoped by env (sandbox/production share TRADING_DB_PATH).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: Quantities are ints in practice; anything under this is float dust and
#: the position row is deleted (mirrors brokers.reconcile.QTY_TOLERANCE).
ZERO_QTY_TOLERANCE = 1e-9


class LocalBook:
    """Restart-surviving {position, cash} ledger backed by SQLite.

    Args:
        db_path: SQLite path; defaults to env TRADING_DB_PATH, falling
            back to ``trading_data.db`` (same convention as KillSwitch /
            EtradeAuthManager, so the whole safety rail shares one file).
        env: row scope; defaults to env ETRADE_ENV, falling back to
            'sandbox'.
    """

    def __init__(self, db_path: Optional[str] = None,
                 env: Optional[str] = None):
        self.db_path = (db_path or os.environ.get("TRADING_DB_PATH")
                        or "trading_data.db")
        self.env = (env or os.environ.get("ETRADE_ENV", "sandbox")
                    ).strip().lower()
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS local_book (
                    env TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    PRIMARY KEY (env, symbol)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS local_cash (
                    env TEXT PRIMARY KEY,
                    cash REAL NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def record_fill(self, symbol: str, signed_qty: float,
                    price: float) -> None:
        """Apply one CONFIRMED fill: upsert the position, move the cash.

        Args:
            symbol: position key (reconcile convention — plain symbol for
                stock, canonical str(Asset) for options).
            signed_qty: +units bought, -units sold (shares/contracts).
            price: per-unit CASH price of the fill. For options the
                caller passes fill_price * contract multiplier so the
                cash leg is correct; quantity stays in native contracts.

        Cash moves by -signed_qty * price (buys spend, sells raise).
        avg_price: weighted average while adding in the same direction;
        unchanged while reducing; reset to the fill price when the
        position flips sign through zero. Rows that reach zero quantity
        are deleted so positions() mirrors the broker's open-positions
        view.
        """
        signed_qty = float(signed_qty)
        price = float(price)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT quantity, avg_price FROM local_book "
                "WHERE env = ? AND symbol = ?",
                (self.env, symbol)).fetchone()
            old_qty = float(row["quantity"]) if row is not None else 0.0
            old_avg = float(row["avg_price"]) if row is not None else 0.0
            new_qty = old_qty + signed_qty
            if abs(new_qty) <= ZERO_QTY_TOLERANCE:
                conn.execute(
                    "DELETE FROM local_book WHERE env = ? AND symbol = ?",
                    (self.env, symbol))
            else:
                if old_qty == 0.0 or (old_qty > 0) != (new_qty > 0):
                    new_avg = price          # fresh open / flipped through 0
                elif abs(new_qty) > abs(old_qty):
                    new_avg = ((old_avg * abs(old_qty)
                                + price * abs(signed_qty)) / abs(new_qty))
                else:
                    new_avg = old_avg        # reducing keeps the entry basis
                conn.execute(
                    "INSERT INTO local_book (env, symbol, quantity, "
                    "avg_price) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(env, symbol) DO UPDATE SET "
                    "quantity = excluded.quantity, "
                    "avg_price = excluded.avg_price",
                    (self.env, symbol, new_qty, new_avg))
            conn.execute(
                "INSERT INTO local_cash (env, cash) VALUES (?, ?) "
                "ON CONFLICT(env) DO UPDATE SET "
                "cash = local_cash.cash + excluded.cash",
                (self.env, -signed_qty * price))
            conn.commit()
        finally:
            conn.close()
        logger.info("LocalBook fill: %s %+g @ %.4f (cash %+.2f)",
                    symbol, signed_qty, price, -signed_qty * price)

    # ------------------------------------------------------------------
    def positions(self) -> Dict[str, float]:
        """{position key: signed quantity} — the reconcile() input shape."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT symbol, quantity FROM local_book WHERE env = ?",
                (self.env,)).fetchall()
        finally:
            conn.close()
        return {row["symbol"]: float(row["quantity"]) for row in rows}

    def cash(self) -> float:
        """The book's cash balance (0.0 until set_cash/record_fill)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT cash FROM local_cash WHERE env = ?",
                (self.env,)).fetchone()
        finally:
            conn.close()
        return float(row["cash"]) if row is not None else 0.0

    def set_cash(self, value: float) -> None:
        """Set cash absolutely (initial funding / operator correction)."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO local_cash (env, cash) VALUES (?, ?) "
                "ON CONFLICT(env) DO UPDATE SET cash = excluded.cash",
                (self.env, float(value)))
            conn.commit()
        finally:
            conn.close()

    def snapshot(self) -> Dict:
        """Full ledger state: {'positions': {key: {'quantity',
        'avg_price'}}, 'cash': float}."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT symbol, quantity, avg_price FROM local_book "
                "WHERE env = ?", (self.env,)).fetchall()
        finally:
            conn.close()
        return {
            "positions": {row["symbol"]: {
                "quantity": float(row["quantity"]),
                "avg_price": float(row["avg_price"]),
            } for row in rows},
            "cash": self.cash(),
        }

    def clear(self) -> None:
        """Delete every row for this env (tests / operator reset)."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM local_book WHERE env = ?", (self.env,))
            conn.execute("DELETE FROM local_cash WHERE env = ?", (self.env,))
            conn.commit()
        finally:
            conn.close()
