"""Paper execution broker with optional restart-safe state.

The legacy constructor remains an in-memory paper trader.  Supplying both
``db_path`` and ``session_id`` opts into an independent SQLite-backed broker
ledger: cash, positions, every order (including cancelled/rejected orders),
and the broker order-id counter are committed in one transaction after each
mutation.  ``resume=True`` reconstructs that broker state without consulting
the application's local book, which keeps reconciliation meaningful.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from brokers.base import ExecutionBroker
from core.models import Asset, AssetType, Order, OrderStatus, OrderType, Position
from data.market_data import MarketDataHandler
from portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)

_LEDGER_SCHEMA_VERSION = 1
_DEFAULT_TRANSACTION_COST_RATE = 0.001
_CLIENT_ORDER_ID = re.compile(r"[A-Za-z0-9]{1,20}")


class ConcurrentPaperSessionUpdate(RuntimeError):
    """A second broker instance changed the same persisted paper session."""


class PaperTrader(ExecutionBroker):
    """Simulated stock execution against current or injected market prices.

    Args:
        initial_capital: Starting paper cash.  Ignored in favour of the stored
            value when ``resume=True``.
        price_sanity_threshold: Optional warn-only fat-finger threshold from
            :class:`ExecutionBroker`.
        db_path/session_id: Both opt into durable broker state.  Neither keeps
            the historical in-memory behaviour.
        resume: Reopen an existing durable session.  A new durable session
            refuses to overwrite an existing id.
        clock: Injectable datetime clock.  The legacy default is local
            ``datetime.now``.
        price_provider: Optional ``provider(symbol)`` returning a numeric
            price, ``(price, as_of)``, or a mapping containing ``price`` and
            ``as_of``/``timestamp``/``date``.  Dated results allow strict
            freshness enforcement without network market data.
        max_price_age_business_days: When set, an undated, future-dated, or
            older quote returns ``None`` and therefore cannot fill an order.
            ``None`` preserves the historical warn-only stale-price policy.
        transaction_cost_rate: Cash-only simulated cost on each fill.  The
            historical 0.1% economics remain the default, but order status now
            reports the cost and an effective cash fill price explicitly.
    """

    def __init__(
            self, initial_capital: float = 100000,
            price_sanity_threshold: Optional[float] = None, *,
            db_path: Optional[str | os.PathLike[str]] = None,
            session_id: Optional[str] = None,
            resume: bool = False,
            clock: Optional[Callable[[], datetime]] = None,
            price_provider: Optional[Callable[[str], Any]] = None,
            max_price_age_business_days: Optional[int] = None,
            transaction_cost_rate: float = _DEFAULT_TRANSACTION_COST_RATE):
        self._lock = threading.RLock()
        self._clock = clock or datetime.now
        self._price_provider = price_provider
        self.price_sanity_threshold = price_sanity_threshold
        self.transaction_cost_rate = self._validated_cost_rate(
            transaction_cost_rate)
        self.max_price_age_business_days = self._validated_max_price_age(
            max_price_age_business_days)

        self.portfolio = PortfolioManager(initial_capital)
        self.market_data = MarketDataHandler()
        self.pending_orders: List[Order] = []
        self.order_id_counter = 0
        self.last_price_dates: Dict[str, date] = {}

        # Kept under the historical attribute name for compatibility.  It now
        # contains every terminal order, not just fills, so CANCELLED remains
        # queryable rather than becoming an unknown broker order.
        self.filled_orders: Dict[str, Order] = {}
        self.completed_orders = self.filled_orders
        self._order_details: Dict[str, Dict[str, Any]] = {}

        path_given = db_path is not None
        id_given = session_id is not None
        if path_given != id_given:
            raise ValueError(
                "db_path and session_id must be provided together")
        if resume and not path_given:
            raise ValueError(
                "resume=True requires db_path and session_id")

        self.db_path = os.fspath(db_path) if db_path is not None else None
        self.session_id = None
        self._persistence_revision: Optional[int] = None
        if self.db_path is not None:
            if self.db_path == ":memory:":
                raise ValueError(
                    "durable PaperTrader does not support ':memory:' databases")
            normalized_id = str(session_id or "").strip()
            if not normalized_id:
                raise ValueError("session_id must be a non-empty string")
            self.session_id = normalized_id
            self._init_ledger_tables()
            if resume:
                self._load_persisted_state()
            else:
                self._create_persisted_session()

    # ------------------------------------------------------------------
    # Durable broker ledger
    # ------------------------------------------------------------------
    @staticmethod
    def _validated_cost_rate(value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "transaction_cost_rate must be a finite non-negative number"
            ) from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(
                "transaction_cost_rate must be a finite non-negative number")
        return result

    @staticmethod
    def _validated_max_price_age(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(
                "max_price_age_business_days must be a non-negative integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_price_age_business_days must be a non-negative integer"
            ) from exc
        if (not math.isfinite(numeric) or not numeric.is_integer()
                or numeric < 0):
            raise ValueError(
                "max_price_age_business_days must be a non-negative integer")
        return int(numeric)

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("PaperTrader clock must return a datetime")
        return value

    def _connect_ledger(self) -> sqlite3.Connection:
        assert self.db_path is not None
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_ledger_tables(self) -> None:
        conn = self._connect_ledger()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_broker_sessions (
                    session_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    initial_capital REAL NOT NULL,
                    cash REAL NOT NULL,
                    order_id_counter INTEGER NOT NULL,
                    transaction_cost_rate REAL NOT NULL,
                    max_price_age_business_days INTEGER,
                    last_price_dates_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_broker_positions (
                    session_id TEXT NOT NULL,
                    asset_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    strike_price REAL,
                    expiration_date TEXT,
                    quantity INTEGER NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    owners_json TEXT,
                    PRIMARY KEY (session_id, asset_key),
                    FOREIGN KEY (session_id)
                        REFERENCES paper_broker_sessions(session_id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_broker_orders (
                    session_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    client_order_id TEXT,
                    request_json TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    strike_price REAL,
                    expiration_date TEXT,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    limit_price REAL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_price REAL,
                    filled_quantity INTEGER NOT NULL,
                    transaction_cost REAL NOT NULL,
                    effective_fill_price REAL,
                    cash_flow REAL NOT NULL,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    history_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, order_id),
                    UNIQUE (session_id, client_order_id),
                    FOREIGN KEY (session_id)
                        REFERENCES paper_broker_sessions(session_id)
                        ON DELETE CASCADE
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _create_persisted_session(self) -> None:
        assert self.session_id is not None
        now = self._now().isoformat()
        conn = self._connect_ledger()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT 1 FROM paper_broker_sessions WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    f"paper session {self.session_id!r} already exists; "
                    "pass resume=True to reopen it")
            conn.execute("""
                INSERT INTO paper_broker_sessions (
                    session_id, schema_version, initial_capital, cash,
                    order_id_counter, transaction_cost_rate,
                    max_price_age_business_days, last_price_dates_json,
                    revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, '{}', 0, ?, ?)
            """, (
                self.session_id, _LEDGER_SCHEMA_VERSION,
                float(self.portfolio.initial_capital),
                float(self.portfolio.cash), self.transaction_cost_rate,
                self.max_price_age_business_days, now, now,
            ))
            conn.commit()
            self._persistence_revision = 0
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _asset_key(asset: Asset) -> str:
        return json.dumps({
            "asset_type": asset.asset_type.value,
            "expiration_date": asset.expiration_date,
            "strike_price": asset.strike_price,
            "symbol": asset.symbol,
        }, allow_nan=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _asset_from_values(
            symbol: str, asset_type: str,
            strike_price: Optional[float],
            expiration_date: Optional[str]) -> Asset:
        return Asset(
            symbol=str(symbol), asset_type=AssetType(str(asset_type)),
            strike_price=(float(strike_price)
                          if strike_price is not None else None),
            expiration_date=expiration_date,
        )

    def _load_persisted_state(self) -> None:
        assert self.session_id is not None
        conn = self._connect_ledger()
        try:
            conn.execute("BEGIN")
            session = conn.execute("""
                SELECT * FROM paper_broker_sessions WHERE session_id = ?
            """, (self.session_id,)).fetchone()
            if session is None:
                raise ValueError(
                    f"paper session {self.session_id!r} does not exist")
            if int(session["schema_version"]) != _LEDGER_SCHEMA_VERSION:
                raise RuntimeError(
                    "unsupported paper broker ledger schema version "
                    f"{session['schema_version']}")
            position_rows = conn.execute("""
                SELECT * FROM paper_broker_positions
                WHERE session_id = ? ORDER BY asset_key
            """, (self.session_id,)).fetchall()
            order_rows = conn.execute("""
                SELECT * FROM paper_broker_orders
                WHERE session_id = ? ORDER BY created_at, order_id
            """, (self.session_id,)).fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        self.portfolio = PortfolioManager(float(session["initial_capital"]))
        self.portfolio.cash = float(session["cash"])
        self.order_id_counter = int(session["order_id_counter"])
        self.transaction_cost_rate = self._validated_cost_rate(
            session["transaction_cost_rate"])
        self.max_price_age_business_days = self._validated_max_price_age(
            session["max_price_age_business_days"])
        stored_dates = json.loads(session["last_price_dates_json"])
        self.last_price_dates = {
            str(symbol): date.fromisoformat(str(day))
            for symbol, day in stored_dates.items()
        }
        self._persistence_revision = int(session["revision"])

        for row in position_rows:
            asset = self._asset_from_values(
                row["symbol"], row["asset_type"], row["strike_price"],
                row["expiration_date"])
            owners_value = json.loads(row["owners_json"]) \
                if row["owners_json"] is not None else None
            self.portfolio.add_position(Position(
                asset=asset,
                quantity=int(row["quantity"]),
                avg_entry_price=float(row["avg_entry_price"]),
                current_price=float(row["current_price"]),
                timestamp=datetime.fromisoformat(row["opened_at"]),
                owners=(tuple(owners_value)
                        if owners_value is not None else None),
            ))

        self.pending_orders = []
        self.filled_orders = {}
        self.completed_orders = self.filled_orders
        self._order_details = {}
        for row in order_rows:
            asset = self._asset_from_values(
                row["symbol"], row["asset_type"], row["strike_price"],
                row["expiration_date"])
            order = Order(
                asset=asset,
                order_type=OrderType(row["order_type"]),
                quantity=int(row["quantity"]),
                price=(float(row["limit_price"])
                       if row["limit_price"] is not None else None),
                timestamp=datetime.fromisoformat(row["created_at"]),
                order_id=str(row["order_id"]),
                status=OrderStatus(row["status"]),
                filled_price=(float(row["filled_price"])
                              if row["filled_price"] is not None else None),
                filled_quantity=int(row["filled_quantity"]),
            )
            self._order_details[order.order_id] = {
                "client_order_id": row["client_order_id"],
                "request_json": str(row["request_json"]),
                "transaction_cost": float(row["transaction_cost"]),
                "effective_fill_price": (
                    float(row["effective_fill_price"])
                    if row["effective_fill_price"] is not None else None),
                "cash_flow": float(row["cash_flow"]),
                "completed_at": row["completed_at"],
                "updated_at": str(row["updated_at"]),
                "history": json.loads(row["history_json"]),
            }
            if order.status is OrderStatus.PENDING:
                self.pending_orders.append(order)
            else:
                self.filled_orders[order.order_id] = order

    def _all_orders(self) -> List[Order]:
        return [*self.pending_orders, *self.filled_orders.values()]

    def _memory_snapshot(self) -> Dict[str, Any]:
        """Serializable copy used to roll memory back if SQLite cannot commit."""
        return {
            "initial_capital": float(self.portfolio.initial_capital),
            "cash": float(self.portfolio.cash),
            "counter": self.order_id_counter,
            "last_price_dates": {
                symbol: day.isoformat()
                for symbol, day in self.last_price_dates.items()
            },
            "positions": [{
                "symbol": position.asset.symbol,
                "asset_type": position.asset.asset_type.value,
                "strike_price": position.asset.strike_price,
                "expiration_date": position.asset.expiration_date,
                "quantity": position.quantity,
                "avg_entry_price": position.avg_entry_price,
                "current_price": position.current_price,
                "timestamp": position.timestamp.isoformat(),
                "owners": list(position.owners)
                if position.owners is not None else None,
            } for position in self.portfolio.positions.values()],
            "orders": [{
                "symbol": order.asset.symbol,
                "asset_type": order.asset.asset_type.value,
                "strike_price": order.asset.strike_price,
                "expiration_date": order.asset.expiration_date,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "timestamp": order.timestamp.isoformat(),
                "order_id": order.order_id,
                "status": order.status.value,
                "filled_price": order.filled_price,
                "filled_quantity": order.filled_quantity,
                "details": deepcopy(self._order_details[order.order_id]),
            } for order in self._all_orders()],
        }

    def _restore_memory_snapshot(self, state: Mapping[str, Any]) -> None:
        self.portfolio = PortfolioManager(float(state["initial_capital"]))
        self.portfolio.cash = float(state["cash"])
        self.order_id_counter = int(state["counter"])
        self.last_price_dates = {
            str(symbol): date.fromisoformat(str(day))
            for symbol, day in state["last_price_dates"].items()
        }
        for value in state["positions"]:
            asset = self._asset_from_values(
                value["symbol"], value["asset_type"], value["strike_price"],
                value["expiration_date"])
            owners = value["owners"]
            self.portfolio.add_position(Position(
                asset=asset, quantity=int(value["quantity"]),
                avg_entry_price=float(value["avg_entry_price"]),
                current_price=float(value["current_price"]),
                timestamp=datetime.fromisoformat(value["timestamp"]),
                owners=tuple(owners) if owners is not None else None,
            ))
        self.pending_orders = []
        self.filled_orders = {}
        self.completed_orders = self.filled_orders
        self._order_details = {}
        for value in state["orders"]:
            asset = self._asset_from_values(
                value["symbol"], value["asset_type"], value["strike_price"],
                value["expiration_date"])
            order = Order(
                asset=asset,
                order_type=OrderType(value["order_type"]),
                quantity=int(value["quantity"]),
                price=value["price"],
                timestamp=datetime.fromisoformat(value["timestamp"]),
                order_id=str(value["order_id"]),
                status=OrderStatus(value["status"]),
                filled_price=value["filled_price"],
                filled_quantity=int(value["filled_quantity"]),
            )
            self._order_details[order.order_id] = deepcopy(value["details"])
            if order.status is OrderStatus.PENDING:
                self.pending_orders.append(order)
            else:
                self.filled_orders[order.order_id] = order

    def _persist_state(self) -> None:
        if self.db_path is None:
            return
        assert self.session_id is not None
        assert self._persistence_revision is not None
        expected_revision = self._persistence_revision
        now = self._now().isoformat()
        conn = self._connect_ledger()
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("""
                SELECT revision FROM paper_broker_sessions
                WHERE session_id = ?
            """, (self.session_id,)).fetchone()
            if current is None:
                raise RuntimeError(
                    f"persisted paper session {self.session_id!r} disappeared")
            if int(current["revision"]) != expected_revision:
                raise ConcurrentPaperSessionUpdate(
                    f"paper session {self.session_id!r} changed from revision "
                    f"{expected_revision} to {current['revision']}; reopen it "
                    "before writing")

            result = conn.execute("""
                UPDATE paper_broker_sessions SET
                    initial_capital = ?, cash = ?, order_id_counter = ?,
                    transaction_cost_rate = ?,
                    max_price_age_business_days = ?,
                    last_price_dates_json = ?, revision = revision + 1,
                    updated_at = ?
                WHERE session_id = ? AND revision = ?
            """, (
                float(self.portfolio.initial_capital),
                float(self.portfolio.cash), self.order_id_counter,
                self.transaction_cost_rate,
                self.max_price_age_business_days,
                json.dumps({
                    symbol: day.isoformat()
                    for symbol, day in self.last_price_dates.items()
                }, separators=(",", ":"), sort_keys=True),
                now, self.session_id, expected_revision,
            ))
            if result.rowcount != 1:
                raise ConcurrentPaperSessionUpdate(
                    f"paper session {self.session_id!r} changed while saving")

            conn.execute(
                "DELETE FROM paper_broker_positions WHERE session_id = ?",
                (self.session_id,))
            conn.executemany("""
                INSERT INTO paper_broker_positions (
                    session_id, asset_key, symbol, asset_type, strike_price,
                    expiration_date, quantity, avg_entry_price, current_price,
                    opened_at, owners_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                self.session_id, self._asset_key(position.asset),
                position.asset.symbol, position.asset.asset_type.value,
                position.asset.strike_price, position.asset.expiration_date,
                int(position.quantity), float(position.avg_entry_price),
                float(position.current_price), position.timestamp.isoformat(),
                (json.dumps(list(position.owners), separators=(",", ":"))
                 if position.owners is not None else None),
            ) for position in self.portfolio.positions.values()])

            conn.execute(
                "DELETE FROM paper_broker_orders WHERE session_id = ?",
                (self.session_id,))
            order_rows = []
            for order in self._all_orders():
                details = self._order_details[order.order_id]
                order_rows.append((
                    self.session_id, order.order_id,
                    details["client_order_id"], details["request_json"],
                    order.asset.symbol, order.asset.asset_type.value,
                    order.asset.strike_price, order.asset.expiration_date,
                    order.order_type.value, int(order.quantity), order.price,
                    order.timestamp.isoformat(), order.status.value,
                    order.filled_price, int(order.filled_quantity),
                    float(details["transaction_cost"]),
                    details["effective_fill_price"],
                    float(details["cash_flow"]), details["completed_at"],
                    details["updated_at"],
                    json.dumps(details["history"], allow_nan=False,
                               separators=(",", ":"), sort_keys=True),
                ))
            conn.executemany("""
                INSERT INTO paper_broker_orders (
                    session_id, order_id, client_order_id, request_json,
                    symbol, asset_type, strike_price, expiration_date,
                    order_type, quantity, limit_price, created_at, status,
                    filled_price, filled_quantity, transaction_cost,
                    effective_fill_price, cash_flow, completed_at, updated_at,
                    history_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?)
            """, order_rows)
            conn.commit()
            self._persistence_revision = expected_revision + 1
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _request_json(asset: Asset, order_type: OrderType, quantity: int,
                      limit_price: Optional[float]) -> str:
        return json.dumps({
            "asset": {
                "asset_type": asset.asset_type.value,
                "expiration_date": asset.expiration_date,
                "strike_price": asset.strike_price,
                "symbol": asset.symbol,
            },
            "limit_price": (float(limit_price)
                            if limit_price is not None else None),
            "order_type": order_type.value,
            "quantity": int(quantity),
        }, allow_nan=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _validate_order_request(
            asset: Asset, order_type: OrderType, quantity: int,
            limit_price: Optional[float]) -> None:
        if not isinstance(asset, Asset):
            raise ValueError("asset must be an Asset")
        if not isinstance(order_type, OrderType):
            raise ValueError("order_type must be an OrderType")
        if (isinstance(quantity, bool) or not isinstance(quantity, int)
                or quantity <= 0):
            raise ValueError("quantity must be a positive integer")
        if limit_price is not None:
            try:
                price = float(limit_price)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "limit_price must be a finite non-negative number or None"
                ) from exc
            if not math.isfinite(price) or price < 0:
                raise ValueError(
                    "limit_price must be a finite non-negative number or None")

    def _find_by_client_order_id(self, client_order_id: str) \
            -> Optional[Order]:
        for order in self._all_orders():
            details = self._order_details[order.order_id]
            if details["client_order_id"] == client_order_id:
                return order
        return None

    @staticmethod
    def _broker_status(status: OrderStatus) -> str:
        return "OPEN" if status is OrderStatus.PENDING else status.value.upper()

    def _new_order_details(
            self, order: Order, client_order_id: Optional[str],
            request_json: str) -> Dict[str, Any]:
        created_at = order.timestamp.isoformat()
        return {
            "client_order_id": client_order_id,
            "request_json": request_json,
            "transaction_cost": 0.0,
            "effective_fill_price": None,
            "cash_flow": 0.0,
            "completed_at": None,
            "updated_at": created_at,
            "history": [{"status": "OPEN", "at": created_at}],
        }

    def _transition_order(
            self, order: Order, status: OrderStatus,
            when: datetime) -> None:
        order.status = status
        details = self._order_details[order.order_id]
        timestamp = when.isoformat()
        details["updated_at"] = timestamp
        details["history"].append({
            "status": self._broker_status(status), "at": timestamp,
        })
        if status is not OrderStatus.PENDING:
            details["completed_at"] = timestamp

    # ------------------------------------------------------------------
    # ExecutionBroker order lifecycle
    # ------------------------------------------------------------------
    def place_order(self, asset: Asset, order_type: OrderType, quantity: int,
                    limit_price: Optional[float]) -> str:
        """Place an order using the historical non-idempotent interface."""
        return self.place_order_with_client_id(
            asset, order_type, quantity, limit_price, None)

    def place_order_with_client_id(
            self, asset: Asset, order_type: OrderType, quantity: int,
            limit_price: Optional[float],
            client_order_id: Optional[str]) -> str:
        """Place once for an explicit logical client order identity.

        Repeating the same id and canonical request returns the original broker
        order id, including after restart or terminal completion.  Reusing an
        id for a different request fails closed.
        """
        self._validate_order_request(
            asset, order_type, quantity, limit_price)
        normalized_client_id = (
            str(client_order_id) if client_order_id is not None else None)
        if (normalized_client_id is not None
                and _CLIENT_ORDER_ID.fullmatch(normalized_client_id) is None):
            raise ValueError(
                "client_order_id must be 1-20 alphanumeric characters")
        request_json = self._request_json(
            asset, order_type, quantity, limit_price)

        with self._lock:
            if normalized_client_id is not None:
                existing = self._find_by_client_order_id(normalized_client_id)
                if existing is not None:
                    details = self._order_details[existing.order_id]
                    if details["request_json"] != request_json:
                        raise ValueError(
                            f"client_order_id {normalized_client_id!r} is "
                            "already "
                            "bound to a different order request")
                    return existing.order_id

            self._check_price_sanity(asset.symbol, limit_price)
            before = self._memory_snapshot() if self.db_path is not None else None
            try:
                self.order_id_counter += 1
                order_id = f"ORD-{self.order_id_counter:06d}"
                now = self._now()
                order = Order(
                    asset=asset, order_type=order_type, quantity=quantity,
                    price=limit_price, timestamp=now, order_id=order_id)
                self._order_details[order_id] = self._new_order_details(
                    order, normalized_client_id, request_json)

                # Paper prices only stocks.  Rejecting options upfront avoids
                # silently using the underlying price without a multiplier.
                if asset.asset_type is not AssetType.STOCK:
                    logger.warning(
                        "PaperTrader rejects non-stock order %s (%s %s): no "
                        "option pricing in paper mode", order_id,
                        asset.asset_type.value, asset.symbol)
                    self._transition_order(order, OrderStatus.REJECTED, now)
                    self.filled_orders[order_id] = order
                else:
                    self.pending_orders.append(order)
                self._persist_state()
            except Exception:
                if before is not None:
                    self._restore_memory_snapshot(before)
                raise
            return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order while preserving its terminal status.

        Pending fills are processed first, matching the live cancel-vs-fill
        race: an order that became marketable is FILLED and uncancellable.
        """
        with self._lock:
            before = self._memory_snapshot() if self.db_path is not None else None
            try:
                changed = self._process_orders_locked()
                cancelled = False
                for order in self.pending_orders[:]:
                    if order.order_id != order_id:
                        continue
                    now = self._now()
                    self._transition_order(order, OrderStatus.CANCELLED, now)
                    self.pending_orders.remove(order)
                    self.filled_orders[order.order_id] = order
                    logger.info(
                        "Cancelled pending order %s (%s %d %s)", order_id,
                        order.order_type.value, order.quantity,
                        order.asset.symbol)
                    changed = True
                    cancelled = True
                    break
                if changed:
                    self._persist_state()
            except Exception:
                if before is not None:
                    self._restore_memory_snapshot(before)
                raise
        if not cancelled:
            logger.warning("cancel_order: no pending order with id %s", order_id)
        return cancelled

    def order_status(self, order_id: str) -> Optional[Dict]:
        """Return broker status plus explicit paper fill economics.

        ``avg_fill_price`` remains the raw market price expected by existing
        execution code.  ``transaction_cost`` is the total positive cash cost;
        ``effective_fill_price``/``cash_fill_price`` is the per-unit price that
        reproduces broker cash (higher for buys, lower for sells).
        """
        with self._lock:
            order = next((candidate for candidate in self.pending_orders
                          if candidate.order_id == order_id), None)
            if order is None:
                order = self.filled_orders.get(order_id)
            if order is None:
                return None
            details = self._order_details[order.order_id]
            transaction_cost = float(details["transaction_cost"])
            effective = details["effective_fill_price"]
            return {
                "status": self._broker_status(order.status),
                "filled_quantity": order.filled_quantity,
                "avg_fill_price": order.filled_price,
                "transaction_cost": transaction_cost,
                "fees": transaction_cost,
                "effective_fill_price": effective,
                "cash_fill_price": effective,
                "cash_flow": float(details["cash_flow"]),
                "client_order_id": details["client_order_id"],
                "completed_at": details["completed_at"],
                "status_history": deepcopy(details["history"]),
            }

    def orders_snapshot(self) -> List[Dict[str, Any]]:
        """Return the complete independent broker order ledger.

        Unlike ``get_portfolio_status`` this read never processes an order or
        fetches a price.  The paper qualification runner uses it to prove that
        every submitted id is known and terminal before evidence is sealed.
        """
        with self._lock:
            rows: List[Dict[str, Any]] = []
            for order in sorted(self._all_orders(),
                                key=lambda item: item.order_id):
                status = self.order_status(order.order_id)
                assert status is not None
                rows.append({
                    "order_id": order.order_id,
                    "symbol": order.asset.symbol,
                    "asset_type": order.asset.asset_type.value,
                    "side": order.order_type.value.upper(),
                    "quantity": order.quantity,
                    "limit_price": order.price,
                    **status,
                })
            return rows

    # ------------------------------------------------------------------
    # Prices and fills
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_quote_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return pd.Timestamp(value).date()
        except Exception as exc:
            raise ValueError(f"invalid quote as_of value {value!r}") from exc

    def _provider_quote(self, symbol: str) -> tuple[Optional[float], Optional[date]]:
        assert self._price_provider is not None
        raw = self._price_provider(symbol)
        if raw is None:
            return None, None
        as_of = None
        if isinstance(raw, Mapping):
            price_value = raw.get("price")
            as_of = raw.get("as_of", raw.get("timestamp", raw.get("date")))
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            price_value, as_of = raw
        else:
            price_value = raw
        if price_value is None:
            return None, self._coerce_quote_date(as_of)
        try:
            price = float(price_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"price provider returned an invalid price for {symbol}"
            ) from exc
        return price, self._coerce_quote_date(as_of)

    def _accept_quote_date(
            self, symbol: str, price: float, price_date: Optional[date],
            now: datetime) -> bool:
        if price_date is None:
            if self.max_price_age_business_days is not None:
                logger.warning(
                    "Refusing undated price for %s in strict freshness mode",
                    symbol)
                return False
            return True

        self.last_price_dates[symbol] = price_date
        age_bdays = int(np.busday_count(price_date, now.date()))
        if age_bdays > 3:
            logger.warning(
                "Stale price for %s: latest close %.2f is from %s "
                "(%d business days old)", symbol, price,
                price_date.isoformat(), age_bdays)
        maximum = self.max_price_age_business_days
        if maximum is not None and (age_bdays < 0 or age_bdays > maximum):
            logger.warning(
                "Refusing price for %s dated %s: age %d business days is "
                "outside strict maximum %d", symbol, price_date.isoformat(),
                age_bdays, maximum)
            return False
        return True

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Return the latest acceptable price for ``symbol``.

        The default path preserves the historical ten-calendar-day market-data
        window and warn-only age policy.  An injected provider avoids network
        access and may carry a date for fail-closed rehearsal freshness.
        """
        try:
            now = self._now()
            if self._price_provider is not None:
                price, price_date = self._provider_quote(symbol)
                if price is None:
                    return None
            else:
                start = now - timedelta(days=10)
                data = self.market_data.fetch_stock_data(
                    symbol, start.strftime("%Y-%m-%d"),
                    now.strftime("%Y-%m-%d"))
                if data is None or data.empty:
                    logger.warning(
                        "No price data for %s in the last 10 calendar days",
                        symbol)
                    return None
                price = float(data["close"].iloc[-1])
                price_date = self._coerce_quote_date(data.index[-1])

            if not math.isfinite(price) or price <= 0:
                logger.warning("Invalid current price for %s: %r", symbol, price)
                return None
            if not self._accept_quote_date(symbol, price, price_date, now):
                return None
            return price
        except Exception as exc:
            logger.warning("Failed to fetch current price for %s: %s",
                           symbol, exc)
            return None

    def process_orders(self) -> None:
        """Process every currently marketable order exactly once."""
        with self._lock:
            before = self._memory_snapshot() if self.db_path is not None else None
            try:
                changed = self._process_orders_locked()
                if changed:
                    self._persist_state()
            except Exception:
                if before is not None:
                    self._restore_memory_snapshot(before)
                raise

    def _process_orders_locked(self) -> bool:
        changed = False
        for order in self.pending_orders[:]:
            current_price = self.get_current_price(order.asset.symbol)
            if current_price is None:
                continue

            marketable = (
                order.price is None
                or (order.order_type is OrderType.BUY
                    and current_price <= order.price)
                or (order.order_type is OrderType.SELL
                    and current_price >= order.price)
            )
            if not marketable:
                continue

            now = self._now()
            executed = self._execute_order(order, current_price, now)
            if executed:
                order.filled_quantity = order.quantity
                order.filled_price = current_price
                self._transition_order(order, OrderStatus.FILLED, now)
            else:
                self._transition_order(order, OrderStatus.REJECTED, now)
            self.filled_orders[order.order_id] = order
            self.pending_orders.remove(order)
            changed = True
        return changed

    def _execute_order(
            self, order: Order, execution_price: float,
            now: Optional[datetime] = None) -> bool:
        """Execute and record explicit cash economics; return mutation status."""
        now = now or self._now()
        notional = order.quantity * execution_price
        transaction_cost = abs(notional) * self.transaction_cost_rate
        details = self._order_details[order.order_id]

        if order.order_type is OrderType.BUY:
            cash_flow = -notional - transaction_cost
            if self.portfolio.cash + cash_flow < 0:
                logger.warning(
                    "Paper BUY %s x%d rejected: cost %.2f exceeds cash %.2f",
                    order.asset.symbol, order.quantity, -cash_flow,
                    self.portfolio.cash)
                return False
            self.portfolio.cash += cash_flow
            existing_pos = self.portfolio.get_position(order.asset)
            if existing_pos:
                new_avg = (
                    existing_pos.avg_entry_price * existing_pos.quantity
                    + execution_price * order.quantity
                ) / (existing_pos.quantity + order.quantity)
                existing_pos.quantity += order.quantity
                existing_pos.avg_entry_price = new_avg
            else:
                self.portfolio.add_position(Position(
                    asset=order.asset, quantity=order.quantity,
                    avg_entry_price=execution_price,
                    current_price=execution_price, timestamp=now))
            effective_price = execution_price + (
                transaction_cost / order.quantity)
        elif order.order_type is OrderType.SELL:
            existing_pos = self.portfolio.get_position(order.asset)
            if not existing_pos or existing_pos.quantity < order.quantity:
                logger.warning(
                    "Paper SELL %s x%d rejected: position %s insufficient",
                    order.asset.symbol, order.quantity,
                    existing_pos.quantity if existing_pos else "missing")
                return False
            cash_flow = notional - transaction_cost
            self.portfolio.cash += cash_flow
            if existing_pos.quantity == order.quantity:
                self.portfolio.remove_position(order.asset)
            else:
                existing_pos.quantity -= order.quantity
            effective_price = execution_price - (
                transaction_cost / order.quantity)
        else:
            return False

        details["transaction_cost"] = transaction_cost
        details["effective_fill_price"] = effective_price
        details["cash_flow"] = cash_flow
        return True

    def get_portfolio_status(self) -> Dict:
        """Return current paper broker cash, value, and positions."""
        with self._lock:
            self.process_orders()
            result = {
                "timestamp": self._now().isoformat(),
                "cash": self.portfolio.cash,
                "portfolio_value": self.portfolio.get_portfolio_value(),
                "unrealized_pnl": self.portfolio.get_portfolio_pnl(),
                "positions": [{
                    "symbol": position.asset.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.avg_entry_price,
                    "current_price": position.current_price,
                    "pnl": position.pnl(),
                    "pnl_pct": position.pnl_pct(),
                } for position in self.portfolio.positions.values()],
                "pending_orders": len(self.pending_orders),
            }
            if self.session_id is not None:
                result["session_id"] = self.session_id
                result["revision"] = self._persistence_revision
            return result
