"""Persistent, account-scoped local trade ledger.

``LocalBook`` is the restart-surviving side of live reconciliation.  It keeps
positions, cash, initialization metadata, and the cumulative fill state of
broker orders in one SQLite database.  Each mutation uses a short-lived
connection and ``BEGIN IMMEDIATE`` so two pollers cannot book the same fill.

Rows are scoped by both E*TRADE environment and account id key.  Callers using
the historical ``LocalBook(db_path, env)`` constructor continue to share the
reserved legacy account scope; live callers should pass ``account_id_key``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

ZERO_QTY_TOLERANCE = 1e-9
LEGACY_ACCOUNT_SCOPE = "__legacy__"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalBook:
    """Restart-surviving position, cash, and order ledger.

    Args:
        db_path: SQLite path.  Defaults to ``TRADING_DB_PATH`` and then
            ``trading_data.db``.
        env: E*TRADE environment scope, defaulting to ``ETRADE_ENV`` or
            ``sandbox``.
        account_id_key: Broker account scope.  Omission intentionally uses a
            reserved legacy scope for compatibility with the original
            two-argument constructor.
    """

    def __init__(self, db_path: Optional[str] = None,
                 env: Optional[str] = None,
                 account_id_key: Optional[str] = None):
        self.db_path = (db_path or os.environ.get("TRADING_DB_PATH")
                        or "trading_data.db")
        self.env = (env or os.environ.get("ETRADE_ENV", "sandbox")
                    ).strip().lower()
        account = str(account_id_key or "").strip()
        self.account_id_key = account or LEGACY_ACCOUNT_SCOPE
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone() is not None

    @staticmethod
    def _columns(conn: sqlite3.Connection, name: str) -> set[str]:
        return {str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({name})")}

    @staticmethod
    def _create_position_table(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS local_book (
                env TEXT NOT NULL,
                account_id_key TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                PRIMARY KEY (env, account_id_key, symbol)
            )
        """)

    @staticmethod
    def _create_cash_table(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS local_cash (
                env TEXT NOT NULL,
                account_id_key TEXT NOT NULL,
                cash REAL NOT NULL,
                PRIMARY KEY (env, account_id_key)
            )
        """)

    def _migrate_legacy_tables(self, conn: sqlite3.Connection) -> None:
        """Upgrade the original env-only tables without losing their rows."""
        if (self._table_exists(conn, "local_book")
                and "account_id_key" not in self._columns(conn, "local_book")):
            conn.execute("ALTER TABLE local_book RENAME TO local_book_v1")
            self._create_position_table(conn)
            conn.execute("""
                INSERT INTO local_book
                    (env, account_id_key, symbol, quantity, avg_price)
                SELECT env, ?, symbol, quantity, avg_price
                FROM local_book_v1
            """, (LEGACY_ACCOUNT_SCOPE,))
            conn.execute("DROP TABLE local_book_v1")
        else:
            self._create_position_table(conn)

        if (self._table_exists(conn, "local_cash")
                and "account_id_key" not in self._columns(conn, "local_cash")):
            conn.execute("ALTER TABLE local_cash RENAME TO local_cash_v1")
            self._create_cash_table(conn)
            conn.execute("""
                INSERT INTO local_cash (env, account_id_key, cash)
                SELECT env, ?, cash FROM local_cash_v1
            """, (LEGACY_ACCOUNT_SCOPE,))
            conn.execute("DROP TABLE local_cash_v1")
        else:
            self._create_cash_table(conn)

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._migrate_legacy_tables(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS local_book_metadata (
                    env TEXT NOT NULL,
                    account_id_key TEXT NOT NULL,
                    initialized INTEGER NOT NULL DEFAULT 0,
                    initialized_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (env, account_id_key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS local_book_orders (
                    env TEXT NOT NULL,
                    account_id_key TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cumulative_booked_quantity REAL NOT NULL DEFAULT 0,
                    avg_price REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (env, account_id_key, order_id)
                )
            """)
            # A pre-upgrade ledger containing cash or positions was already a
            # real snapshot.  Record that fact explicitly during migration.
            now = _utc_now()
            conn.execute("""
                INSERT OR IGNORE INTO local_book_metadata
                    (env, account_id_key, initialized, initialized_at,
                     updated_at)
                SELECT env, account_id_key, 1, ?, ? FROM local_book
                UNION
                SELECT env, account_id_key, 1, ?, ? FROM local_cash
            """, (now, now, now, now))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @property
    def initialized(self) -> bool:
        """Whether this account scope has an authoritative local snapshot."""
        return self.is_initialized()

    def is_initialized(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT initialized FROM local_book_metadata
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
        finally:
            conn.close()
        return bool(row["initialized"]) if row is not None else False

    def metadata(self) -> Dict[str, Any]:
        """Return explicit lifecycle metadata for this account scope."""
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT initialized, initialized_at, updated_at
                FROM local_book_metadata
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
        finally:
            conn.close()
        return {
            "env": self.env,
            "account_id_key": self.account_id_key,
            "initialized": bool(row["initialized"]) if row else False,
            "initialized_at": (row["initialized_at"] if row else None),
            "updated_at": (row["updated_at"] if row else None),
        }

    @property
    def _scope(self) -> tuple[str, str]:
        return self.env, self.account_id_key

    def _mark_initialized(self, conn: sqlite3.Connection) -> None:
        now = _utc_now()
        conn.execute("""
            INSERT INTO local_book_metadata
                (env, account_id_key, initialized, initialized_at, updated_at)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(env, account_id_key) DO UPDATE SET
                initialized = 1,
                initialized_at = COALESCE(
                    local_book_metadata.initialized_at,
                    excluded.initialized_at),
                updated_at = excluded.updated_at
        """, (*self._scope, now, now))

    # ------------------------------------------------------------------
    # Position/cash primitives
    # ------------------------------------------------------------------
    def _apply_position(self, conn: sqlite3.Connection, symbol: str,
                        signed_qty: float, price: float) -> None:
        row = conn.execute("""
            SELECT quantity, avg_price FROM local_book
            WHERE env = ? AND account_id_key = ? AND symbol = ?
        """, (*self._scope, symbol)).fetchone()
        old_qty = float(row["quantity"]) if row is not None else 0.0
        old_avg = float(row["avg_price"]) if row is not None else 0.0
        new_qty = old_qty + signed_qty
        if abs(new_qty) <= ZERO_QTY_TOLERANCE:
            conn.execute("""
                DELETE FROM local_book
                WHERE env = ? AND account_id_key = ? AND symbol = ?
            """, (*self._scope, symbol))
            return
        if old_qty == 0.0 or (old_qty > 0) != (new_qty > 0):
            new_avg = price
        elif abs(new_qty) > abs(old_qty):
            new_avg = ((old_avg * abs(old_qty) + price * abs(signed_qty))
                       / abs(new_qty))
        else:
            new_avg = old_avg
        conn.execute("""
            INSERT INTO local_book
                (env, account_id_key, symbol, quantity, avg_price)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(env, account_id_key, symbol) DO UPDATE SET
                quantity = excluded.quantity,
                avg_price = excluded.avg_price
        """, (*self._scope, symbol, new_qty, new_avg))

    def _move_cash(self, conn: sqlite3.Connection, delta: float) -> None:
        conn.execute("""
            INSERT INTO local_cash (env, account_id_key, cash)
            VALUES (?, ?, ?)
            ON CONFLICT(env, account_id_key) DO UPDATE SET
                cash = local_cash.cash + excluded.cash
        """, (*self._scope, delta))

    def record_fill(self, symbol: str, signed_qty: float,
                    price: float, fees: float = 0.0) -> None:
        """Apply one confirmed fill and its cash movement atomically."""
        signed_qty = float(signed_qty)
        price = float(price)
        fees = float(fees)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._apply_position(conn, symbol, signed_qty, price)
            self._move_cash(conn, -signed_qty * price - fees)
            self._mark_initialized(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        logger.info("LocalBook fill: %s %+g @ %.4f (cash %+.2f, fees %.2f)",
                    symbol, signed_qty, price,
                    -signed_qty * price - fees, fees)

    def positions(self) -> Dict[str, float]:
        """Return ``{position key: signed quantity}`` for reconciliation."""
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT symbol, quantity FROM local_book
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchall()
        finally:
            conn.close()
        return {row["symbol"]: float(row["quantity"]) for row in rows}

    def cash(self) -> float:
        """Return cash, defaulting to zero before initialization."""
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT cash FROM local_cash
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
        finally:
            conn.close()
        return float(row["cash"]) if row is not None else 0.0

    def set_cash(self, value: float) -> None:
        """Set cash absolutely (initial funding / operator correction)."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                INSERT INTO local_cash (env, account_id_key, cash)
                VALUES (?, ?, ?)
                ON CONFLICT(env, account_id_key) DO UPDATE SET
                    cash = excluded.cash
            """, (*self._scope, float(value)))
            self._mark_initialized(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def snapshot(self) -> Dict[str, Any]:
        """Detailed position basis and cash from one SQLite snapshot."""
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            rows = conn.execute("""
                SELECT symbol, quantity, avg_price FROM local_book
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchall()
            cash_row = conn.execute("""
                SELECT cash FROM local_cash
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "positions": {row["symbol"]: {
                "quantity": float(row["quantity"]),
                "avg_price": float(row["avg_price"]),
            } for row in rows},
            "cash": float(cash_row["cash"]) if cash_row is not None else 0.0,
        }

    def reconciliation_snapshot(self) -> Dict[str, Any]:
        """Atomically read the exact input required by reconciliation.

        Unlike separate ``positions()`` and ``cash()`` calls, both values and
        the initialization marker are observed from the same SQLite snapshot.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            rows = conn.execute("""
                SELECT symbol, quantity FROM local_book
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchall()
            cash_row = conn.execute("""
                SELECT cash FROM local_cash
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
            meta = conn.execute("""
                SELECT initialized FROM local_book_metadata
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "positions": {row["symbol"]: float(row["quantity"])
                          for row in rows},
            "cash": float(cash_row["cash"]) if cash_row is not None else 0.0,
            "initialized": bool(meta["initialized"]) if meta else False,
        }

    @staticmethod
    def _normalize_positions(positions: Any) -> Dict[str, Dict[str, float]]:
        if isinstance(positions, Mapping):
            items: Iterable[tuple[Any, Any]] = positions.items()
        else:
            items = ((row["symbol"], row) for row in positions)
        normalized: Dict[str, Dict[str, float]] = {}
        for raw_symbol, raw_value in items:
            symbol = str(raw_symbol)
            if isinstance(raw_value, Mapping):
                quantity = float(raw_value.get("quantity", 0.0))
                avg_price = float(raw_value.get(
                    "avg_price", raw_value.get("current_price", 0.0)) or 0.0)
            else:
                quantity = float(raw_value)
                avg_price = 0.0
            if abs(quantity) > ZERO_QTY_TOLERANCE:
                normalized[symbol] = {"quantity": quantity,
                                      "avg_price": avg_price}
        return normalized

    def _replace_snapshot(self, conn: sqlite3.Connection, positions: Any,
                          cash: float) -> None:
        normalized = self._normalize_positions(positions)
        conn.execute("""
            DELETE FROM local_book
            WHERE env = ? AND account_id_key = ?
        """, self._scope)
        conn.executemany("""
            INSERT INTO local_book
                (env, account_id_key, symbol, quantity, avg_price)
            VALUES (?, ?, ?, ?, ?)
        """, [(*self._scope, symbol, value["quantity"], value["avg_price"])
              for symbol, value in normalized.items()])
        conn.execute("""
            INSERT INTO local_cash (env, account_id_key, cash)
            VALUES (?, ?, ?)
            ON CONFLICT(env, account_id_key) DO UPDATE SET cash = excluded.cash
        """, (*self._scope, float(cash)))
        self._mark_initialized(conn)

    @staticmethod
    def _snapshot_parts(positions: Any,
                        cash: Optional[float]) -> tuple[Any, float]:
        if cash is None and isinstance(positions, Mapping):
            if "positions" in positions and "cash" in positions:
                return positions["positions"], float(positions["cash"])
        if cash is None:
            raise TypeError("cash is required unless a full snapshot mapping "
                            "is supplied")
        return positions, float(cash)

    def bootstrap_snapshot(self, positions: Any,
                           cash: Optional[float] = None) -> bool:
        """Initialize from a broker snapshot once, transactionally.

        Returns ``True`` when this call initialized the scope and ``False``
        when a previous bootstrap or mutation already did so.  The check and
        replacement share one write transaction, making concurrent startup
        attempts harmless.
        """
        positions, cash_value = self._snapshot_parts(positions, cash)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""
                SELECT initialized FROM local_book_metadata
                WHERE env = ? AND account_id_key = ?
            """, self._scope).fetchone()
            if row is not None and bool(row["initialized"]):
                conn.rollback()
                return False
            self._replace_snapshot(conn, positions, cash_value)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def replace_snapshot(self, positions: Any,
                         cash: Optional[float] = None) -> None:
        """Transactionally replace this scope with an authoritative snapshot."""
        positions, cash_value = self._snapshot_parts(positions, cash)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._replace_snapshot(conn, positions, cash_value)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Short aliases make the lifecycle API natural without removing the
    # explicit snapshot-named methods used by integration code.
    bootstrap = bootstrap_snapshot
    replace = replace_snapshot

    # ------------------------------------------------------------------
    # Durable broker-order tracking
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_request(order_request: Mapping[str, Any]) -> str:
        return json.dumps(order_request, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)

    def track_order(self, order_id: Any, order_request: Mapping[str, Any],
                    status: str = "PENDING") -> Dict[str, Any]:
        """Durably register an order without resetting already-booked fills.

        Re-registering the same id and request is idempotent.  Reusing an id
        for a different request fails closed because it would make subsequent
        cumulative fill deltas ambiguous.
        """
        order_id = str(order_id)
        request_json = self._canonical_request(order_request)
        now = _utc_now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("""
                SELECT request_json FROM local_book_orders
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (*self._scope, order_id)).fetchone()
            if existing is not None and existing["request_json"] != request_json:
                raise ValueError(f"order {order_id!r} is already tracked with "
                                 "a different request")
            conn.execute("""
                INSERT INTO local_book_orders
                    (env, account_id_key, order_id, request_json, status,
                     cumulative_booked_quantity, avg_price, created_at,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?)
                ON CONFLICT(env, account_id_key, order_id) DO NOTHING
            """, (*self._scope, order_id, request_json,
                  str(status).upper(), now, now))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        tracked = self.tracked_order(order_id)
        assert tracked is not None
        return tracked

    @staticmethod
    def _decode_order(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "order_id": str(row["order_id"]),
            "account_id_key": str(row["account_id_key"]),
            "request": json.loads(row["request_json"]),
            "status": str(row["status"]),
            "cumulative_booked_quantity": float(
                row["cumulative_booked_quantity"]),
            "avg_price": (float(row["avg_price"])
                          if row["avg_price"] is not None else None),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def tracked_order(self, order_id: Any) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT * FROM local_book_orders
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (*self._scope, str(order_id))).fetchone()
        finally:
            conn.close()
        return self._decode_order(row) if row is not None else None

    def tracked_orders(self) -> list[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT * FROM local_book_orders
                WHERE env = ? AND account_id_key = ?
                ORDER BY created_at, order_id
            """, self._scope).fetchall()
        finally:
            conn.close()
        return [self._decode_order(row) for row in rows]

    @staticmethod
    def _request_body(order_request: Mapping[str, Any]) -> Mapping[str, Any]:
        for wrapper in ("PreviewOrderRequest", "PlaceOrderRequest"):
            wrapped = order_request.get(wrapper)
            if isinstance(wrapped, Mapping):
                return wrapped
        return order_request

    @staticmethod
    def _option_key(product: Mapping[str, Any]) -> str:
        expiry = (f"{int(product['expiryYear']):04d}-"
                  f"{int(product['expiryMonth']):02d}-"
                  f"{int(product['expiryDay']):02d}")
        right = ("call" if str(product.get("callPut", "")).upper() == "CALL"
                 else "put")
        return (f"{product['symbol']} {expiry} "
                f"${float(product['strikePrice'])} {right}")

    @staticmethod
    def _action_sign(action: Any) -> float:
        action = str(action or "").upper()
        if action.startswith("BUY"):
            return 1.0
        if action.startswith("SELL"):
            return -1.0
        raise ValueError(f"unsupported order action {action!r}")

    @staticmethod
    def _spread_package_shape(
            instruments: Iterable[Mapping[str, Any]]) -> tuple[int, list[int]]:
        """Return ``(package_count, normalized-source quantities)``.

        A spread request stores total contracts per leg. Their greatest common
        divisor is the number of package units; dividing each leg by that GCD
        yields its canonical integer ratio. For example, quantities ``6:9``
        are three packages of the ``2:3`` structure. Using the smallest leg
        quantity would incorrectly book fractional legs for this case.
        """
        quantities: list[int] = []
        for instrument in instruments:
            raw = float(instrument.get(
                "orderedQuantity", instrument.get("quantity", 0)) or 0)
            if (not math.isfinite(raw) or raw <= 0
                    or not raw.is_integer()):
                raise ValueError(
                    "spread instruments require positive integer quantities")
            quantities.append(int(raw))
        if not quantities:
            raise ValueError("tracked order request has no instruments")
        package_count = quantities[0]
        for quantity in quantities[1:]:
            package_count = math.gcd(package_count, quantity)
        return package_count, quantities

    @classmethod
    def _request_fill_capacity(
            cls, order_request: Mapping[str, Any]) -> Optional[float]:
        """Return the request's maximum cumulative fill in native units.

        For SPREADS the native unit is one package, never one leg. Builders
        express each leg's total requested contracts, so their greatest common
        divisor is the package count and the divided quantities are the
        canonical leg ratios. Unsupported order types are left to
        ``_apply_request_delta`` so its existing typed error is preserved.
        """
        request = cls._request_body(order_request)
        try:
            order_type = str(request["orderType"]).upper()
            instruments = request["Order"][0]["Instrument"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("malformed tracked order request") from exc
        if order_type not in {"EQ", "OPTN", "SPREADS"}:
            return None
        if not instruments:
            raise ValueError("tracked order request has no instruments")
        quantities = [float(inst.get("orderedQuantity",
                                     inst.get("quantity", 0)) or 0)
                      for inst in instruments]
        if any(not math.isfinite(qty) or qty <= 0 for qty in quantities):
            raise ValueError("tracked order instruments require positive "
                             "finite quantities")
        if order_type in {"EQ", "OPTN"}:
            if len(quantities) != 1:
                raise ValueError(
                    f"{order_type} request must have one instrument")
            return quantities[0]
        package_count, _ = cls._spread_package_shape(instruments)
        return float(package_count)

    def _apply_request_delta(self, conn: sqlite3.Connection,
                             order_request: Mapping[str, Any],
                             delta_qty: float, delta_price: float,
                             fees: float) -> None:
        request = self._request_body(order_request)
        try:
            order_type = str(request["orderType"]).upper()
            instruments = request["Order"][0]["Instrument"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("malformed tracked order request") from exc
        if not instruments:
            raise ValueError("tracked order request has no instruments")

        if order_type in {"EQ", "OPTN"}:
            if len(instruments) != 1:
                raise ValueError(f"{order_type} request must have one instrument")
            instrument = instruments[0]
            product = instrument["Product"]
            sign = self._action_sign(instrument.get("orderAction"))
            symbol = (str(product["symbol"]) if order_type == "EQ"
                      else self._option_key(product))
            multiplier = 1.0 if order_type == "EQ" else 100.0
            signed_qty = sign * delta_qty
            self._apply_position(conn, symbol, signed_qty, abs(delta_price))
            self._move_cash(
                conn, -signed_qty * abs(delta_price) * multiplier - fees)
            return

        if order_type != "SPREADS":
            raise ValueError(f"unsupported tracked order type {order_type!r}")

        package_quantity, requested = self._spread_package_shape(instruments)
        for instrument, quantity in zip(instruments, requested):
            product = instrument["Product"]
            signed_qty = (self._action_sign(instrument.get("orderAction"))
                          * delta_qty * quantity / package_quantity)
            # E*TRADE exposes the package net average through order_status,
            # not leg-level bases.  Reconciliation depends on quantity; retain
            # the absolute package price as the best available local basis.
            self._apply_position(conn, self._option_key(product), signed_qty,
                                 abs(delta_price))
        # EtradeClient reports a signed package net: BUY legs positive and
        # SELL legs negative.  Therefore a credit is negative and raises cash.
        self._move_cash(conn, -delta_qty * delta_price * 100.0 - fees)

    def apply_cumulative_order_fill(
            self, order_id: Any, cumulative_filled_quantity: float,
            avg_fill_price: Optional[float], *, status: str = "OPEN",
            order_request: Optional[Mapping[str, Any]] = None,
            fees: float = 0.0) -> float:
        """Book only the unobserved delta of a broker's cumulative fill.

        The order row, every affected position, and cash are updated in one
        transaction.  Replaying the same cumulative status returns ``0.0``
        and changes no accounting rows.  ``avg_fill_price`` is the broker's
        cumulative average; the method backs out the marginal delta price.
        """
        order_id = str(order_id)
        cumulative = float(cumulative_filled_quantity)
        fees = float(fees)
        if not math.isfinite(cumulative) or cumulative < 0:
            raise ValueError("cumulative filled quantity must be finite and "
                             "non-negative")
        if not math.isfinite(fees) or fees < 0:
            raise ValueError("fees must be finite and non-negative")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("""
                SELECT * FROM local_book_orders
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (*self._scope, order_id)).fetchone()
            if row is None:
                if order_request is None:
                    raise KeyError(f"order {order_id!r} is not tracked")
                request_json = self._canonical_request(order_request)
                now = _utc_now()
                conn.execute("""
                    INSERT INTO local_book_orders
                        (env, account_id_key, order_id, request_json, status,
                         cumulative_booked_quantity, avg_price, created_at,
                         updated_at)
                    VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """, (*self._scope, order_id, request_json,
                      str(status).upper(), now, now))
                row = conn.execute("""
                    SELECT * FROM local_book_orders
                    WHERE env = ? AND account_id_key = ? AND order_id = ?
                """, (*self._scope, order_id)).fetchone()
            elif order_request is not None:
                supplied = self._canonical_request(order_request)
                if supplied != row["request_json"]:
                    raise ValueError(f"order {order_id!r} request does not "
                                     "match the tracked request")

            assert row is not None
            booked = float(row["cumulative_booked_quantity"])
            request = json.loads(row["request_json"])
            capacity = self._request_fill_capacity(request)
            if (capacity is not None
                    and cumulative > capacity + ZERO_QTY_TOLERANCE):
                raise ValueError(
                    f"cumulative fill for order {order_id!r} exceeds "
                    f"requested native quantity {capacity:g}")
            if cumulative < booked - ZERO_QTY_TOLERANCE:
                raise ValueError(
                    f"cumulative fill for order {order_id!r} regressed "
                    f"from {booked:g} to {cumulative:g}")
            delta = cumulative - booked
            now = _utc_now()
            if delta <= ZERO_QTY_TOLERANCE:
                conn.execute("""
                    UPDATE local_book_orders SET status = ?, updated_at = ?
                    WHERE env = ? AND account_id_key = ? AND order_id = ?
                """, (str(status).upper(), now, *self._scope, order_id))
                conn.commit()
                return 0.0
            if avg_fill_price is None:
                raise ValueError("avg_fill_price is required for a positive "
                                 "cumulative fill")
            cumulative_avg = float(avg_fill_price)
            if not math.isfinite(cumulative_avg):
                raise ValueError("avg_fill_price must be finite")
            old_avg = (float(row["avg_price"])
                       if row["avg_price"] is not None else 0.0)
            delta_price = ((cumulative * cumulative_avg - booked * old_avg)
                           / delta)
            self._apply_request_delta(conn, request, delta, delta_price, fees)
            self._mark_initialized(conn)
            conn.execute("""
                UPDATE local_book_orders SET
                    status = ?, cumulative_booked_quantity = ?,
                    avg_price = ?, updated_at = ?
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (str(status).upper(), cumulative, cumulative_avg, now,
                  *self._scope, order_id))
            conn.commit()
            return delta
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def apply_order_status(self, order_id: Any,
                           order_status: Mapping[str, Any], *,
                           order_request: Optional[Mapping[str, Any]] = None,
                           fees: float = 0.0) -> float:
        """Convenience adapter for ``ExecutionBroker.order_status`` output."""
        return self.apply_cumulative_order_fill(
            order_id,
            float(order_status.get("filled_quantity", 0) or 0),
            order_status.get("avg_fill_price"),
            status=str(order_status.get("status") or "UNKNOWN"),
            order_request=order_request,
            fees=fees,
        )

    def clear(self) -> None:
        """Delete every ledger, metadata, and order row for this account."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for table in ("local_book", "local_cash", "local_book_metadata",
                          "local_book_orders"):
                conn.execute(
                    f"DELETE FROM {table} "
                    "WHERE env = ? AND account_id_key = ?", self._scope)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
