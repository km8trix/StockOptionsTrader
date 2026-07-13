"""Durable reservations for risk consumed by pending live orders.

The risk checks performed before an order is submitted need to include orders
that have not filled yet.  ``RiskReservationLedger`` owns that pending slice
of exposure.  Its rows are scoped by broker environment and account, and all
check-and-write operations use ``BEGIN IMMEDIATE`` so concurrent submitters
cannot both observe the same spare capacity.

Risk is represented by a small, deliberately instrument-neutral vector::

    {
        "gross": 1000.0,
        "cash_debit": 250.0,
        "per_name": {"SPY": 1000.0},
        "sector": {"ETF": 1000.0},
        "option": {"delta": 500.0, "vega": 12.0},
    }

The keyed dimensions accept arbitrary names, so the same component works for
equities, single options, and option packages.  Capacity maps use the same
shape.  A ``"*"`` entry provides a default limit for a keyed dimension;
unconfigured keys are unbounded.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional


_SCALAR_DIMENSIONS = ("gross", "cash_debit")
_KEYED_DIMENSIONS = ("per_name", "sector", "option")
_VECTOR_DIMENSIONS = frozenset((*_SCALAR_DIMENSIONS, *_KEYED_DIMENSIONS))
_RELEASING_ORDER_STATUSES = frozenset(
    {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "EXECUTED", "FILLED"}
)
_INACTIVE_ORDER_STATUSES = frozenset(
    {*_RELEASING_ORDER_STATUSES, "REPLACED"}
)
_EPSILON = 1e-9


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_vector() -> Dict[str, Any]:
    return {
        "gross": 0.0,
        "cash_debit": 0.0,
        "per_name": {},
        "sector": {},
        "option": {},
    }


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _normalise_risk_vector(
    value: Optional[Mapping[str, Any]],
    *,
    label: str,
) -> Dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - _VECTOR_DIMENSIONS
    if unknown:
        raise ValueError(f"{label} has unknown dimensions: {sorted(unknown)}")

    result = _empty_vector()
    for dimension in _SCALAR_DIMENSIONS:
        result[dimension] = _finite_nonnegative(
            value.get(dimension, 0.0), f"{label}.{dimension}"
        )
    for dimension in _KEYED_DIMENSIONS:
        entries = value.get(dimension, {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"{label}.{dimension} must be a mapping")
        normalised: Dict[str, float] = {}
        for raw_key, raw_amount in entries.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{label}.{dimension} keys cannot be empty")
            normalised[key] = _finite_nonnegative(
                raw_amount, f"{label}.{dimension}.{key}"
            )
        result[dimension] = normalised
    return result


def _normalise_capacity(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("capacity must be a mapping")
    unknown = set(value) - _VECTOR_DIMENSIONS
    if unknown:
        raise ValueError(f"capacity has unknown dimensions: {sorted(unknown)}")

    result: Dict[str, Any] = {}
    for dimension in _SCALAR_DIMENSIONS:
        if dimension in value and value[dimension] is not None:
            result[dimension] = _finite_nonnegative(
                value[dimension], f"capacity.{dimension}"
            )
    for dimension in _KEYED_DIMENSIONS:
        if dimension not in value or value[dimension] is None:
            continue
        entries = value[dimension]
        if not isinstance(entries, Mapping):
            raise ValueError(f"capacity.{dimension} must be a mapping")
        normalised: Dict[str, float] = {}
        for raw_key, raw_amount in entries.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"capacity.{dimension} keys cannot be empty")
            normalised[key] = _finite_nonnegative(
                raw_amount, f"capacity.{dimension}.{key}"
            )
        result[dimension] = normalised
    return result


def _json_dump(value: Any, label: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-safe") from exc


def _add_vectors(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    result = _empty_vector()
    for dimension in _SCALAR_DIMENSIONS:
        result[dimension] = float(left.get(dimension, 0.0)) + float(
            right.get(dimension, 0.0)
        )
    for dimension in _KEYED_DIMENSIONS:
        combined: Dict[str, float] = {}
        left_values = left.get(dimension, {})
        right_values = right.get(dimension, {})
        for key in sorted(set(left_values) | set(right_values)):
            combined[key] = float(left_values.get(key, 0.0)) + float(
                right_values.get(key, 0.0)
            )
        result[dimension] = combined
    return result


def _scale_vector(value: Mapping[str, Any], fraction: float) -> Dict[str, Any]:
    result = _empty_vector()
    for dimension in _SCALAR_DIMENSIONS:
        result[dimension] = float(value[dimension]) * fraction
    for dimension in _KEYED_DIMENSIONS:
        result[dimension] = {
            key: float(amount) * fraction
            for key, amount in value[dimension].items()
        }
    return result


class RiskCapacityExceeded(RuntimeError):
    """Raised when a new opening reservation would breach a configured limit."""

    def __init__(self, breaches: Iterable[Mapping[str, Any]]):
        self.breaches = [dict(breach) for breach in breaches]
        super().__init__(f"risk capacity exceeded: {self.breaches}")


class RiskReservationLedger:
    """SQLite ledger for pending-order risk reservations.

    ``capacity`` limits only configured dimensions.  ``base_usage`` represents
    current portfolio exposure and defaults to zero; active reservations from
    this ledger are always added inside the same write transaction.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        env: Optional[str] = None,
        account_id_key: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or os.environ.get("TRADING_DB_PATH") or "trading_data.db"
        self.env = (env or os.environ.get("ETRADE_ENV") or "sandbox").strip().lower()
        self.account_id_key = str(account_id_key or "").strip()
        if not self.env:
            raise ValueError("env cannot be empty")
        if not self.account_id_key:
            raise ValueError("account_id_key cannot be empty")
        self._init_tables()

    @property
    def _scope(self) -> tuple[str, str]:
        return self.env, self.account_id_key

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_tables(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS risk_reservations (
                    env TEXT NOT NULL,
                    account_id_key TEXT NOT NULL,
                    reservation_id TEXT NOT NULL,
                    instrument_type TEXT NOT NULL,
                    units REAL NOT NULL,
                    reducing INTEGER NOT NULL,
                    requested_risk_json TEXT NOT NULL,
                    remaining_risk_json TEXT NOT NULL,
                    intent_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    release_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (env, account_id_key, reservation_id)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS risk_reservation_orders (
                    env TEXT NOT NULL,
                    account_id_key TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    reservation_id TEXT NOT NULL,
                    cumulative_filled_units REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (env, account_id_key, order_id),
                    FOREIGN KEY (env, account_id_key, reservation_id)
                        REFERENCES risk_reservations (
                            env, account_id_key, reservation_id
                        )
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS ix_risk_reservation_orders_intent
                ON risk_reservation_orders (
                    env, account_id_key, reservation_id
                )
            """)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _normalise_identifier(value: Any, label: str) -> str:
        result = str(value or "").strip()
        if not result:
            raise ValueError(f"{label} cannot be empty")
        return result

    def _row(self, connection: sqlite3.Connection, reservation_id: str) -> Optional[sqlite3.Row]:
        return connection.execute("""
            SELECT * FROM risk_reservations
            WHERE env = ? AND account_id_key = ? AND reservation_id = ?
        """, (*self._scope, reservation_id)).fetchone()

    def _order_rows(
        self, connection: sqlite3.Connection, reservation_id: str
    ) -> list[sqlite3.Row]:
        return list(connection.execute("""
            SELECT order_id, cumulative_filled_units, status, created_at, updated_at
            FROM risk_reservation_orders
            WHERE env = ? AND account_id_key = ? AND reservation_id = ?
            ORDER BY created_at, order_id
        """, (*self._scope, reservation_id)))

    def _serialise_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> Dict[str, Any]:
        orders = self._order_rows(connection, str(row["reservation_id"]))
        return {
            "env": str(row["env"]),
            "account_id_key": str(row["account_id_key"]),
            "reservation_id": str(row["reservation_id"]),
            "instrument_type": str(row["instrument_type"]),
            "units": float(row["units"]),
            "reducing": bool(row["reducing"]),
            "requested_risk": json.loads(row["requested_risk_json"]),
            "remaining_risk": json.loads(row["remaining_risk_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "status": str(row["status"]),
            "release_reason": row["release_reason"],
            "orders": [
                {
                    "order_id": str(order["order_id"]),
                    "cumulative_filled_units": float(order["cumulative_filled_units"]),
                    "status": str(order["status"]),
                    "created_at": str(order["created_at"]),
                    "updated_at": str(order["updated_at"]),
                }
                for order in orders
            ],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _active_totals_in_connection(
        self,
        connection: sqlite3.Connection,
        *,
        exclude_reservation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        total = _empty_vector()
        if exclude_reservation_id is None:
            rows = connection.execute("""
                SELECT remaining_risk_json FROM risk_reservations
                WHERE env = ? AND account_id_key = ? AND status = 'ACTIVE'
            """, self._scope)
        else:
            rows = connection.execute("""
                SELECT remaining_risk_json FROM risk_reservations
                WHERE env = ? AND account_id_key = ? AND status = 'ACTIVE'
                    AND reservation_id != ?
            """, (*self._scope, exclude_reservation_id))
        for row in rows:
            total = _add_vectors(total, json.loads(row["remaining_risk_json"]))
        return total

    @staticmethod
    def _capacity_breaches(
        used: Mapping[str, Any],
        requested: Mapping[str, Any],
        capacity: Mapping[str, Any],
    ) -> list[Dict[str, Any]]:
        breaches: list[Dict[str, Any]] = []
        for dimension in _SCALAR_DIMENSIONS:
            if dimension not in capacity:
                continue
            current = float(used[dimension])
            addition = float(requested[dimension])
            limit = float(capacity[dimension])
            if current + addition > limit + _EPSILON:
                breaches.append({
                    "dimension": dimension,
                    "key": None,
                    "used": current,
                    "requested": addition,
                    "projected": current + addition,
                    "capacity": limit,
                })

        for dimension in _KEYED_DIMENSIONS:
            limits = capacity.get(dimension)
            if limits is None:
                continue
            for key, raw_addition in requested[dimension].items():
                if key in limits:
                    limit = float(limits[key])
                elif "*" in limits:
                    limit = float(limits["*"])
                else:
                    continue
                current = float(used[dimension].get(key, 0.0))
                addition = float(raw_addition)
                if current + addition > limit + _EPSILON:
                    breaches.append({
                        "dimension": dimension,
                        "key": key,
                        "used": current,
                        "requested": addition,
                        "projected": current + addition,
                        "capacity": limit,
                    })
        return breaches

    def _insert_binding(
        self,
        connection: sqlite3.Connection,
        reservation_id: str,
        order_id: str,
        now: str,
    ) -> None:
        existing = connection.execute("""
            SELECT reservation_id FROM risk_reservation_orders
            WHERE env = ? AND account_id_key = ? AND order_id = ?
        """, (*self._scope, order_id)).fetchone()
        if existing is not None:
            if str(existing["reservation_id"]) != reservation_id:
                raise ValueError(
                    f"order_id {order_id!r} is bound to a different reservation"
                )
            return
        connection.execute("""
            INSERT INTO risk_reservation_orders (
                env, account_id_key, order_id, reservation_id,
                cumulative_filled_units, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, 'PENDING', ?, ?)
        """, (*self._scope, order_id, reservation_id, now, now))

    def check_and_reserve(
        self,
        reservation_id: str,
        risk: Mapping[str, Any],
        capacity: Mapping[str, Any],
        *,
        units: float,
        instrument_type: str,
        reducing: bool = False,
        base_usage: Optional[Mapping[str, Any]] = None,
        order_ids: Iterable[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically check capacity and create an idempotent reservation.

        Reusing ``reservation_id`` with the same intent returns the existing
        row.  Reusing it with different intent data fails rather than silently
        mutating a live order's reservation.
        """
        reservation_id = self._normalise_identifier(reservation_id, "reservation_id")
        instrument_type = self._normalise_identifier(instrument_type, "instrument_type").upper()
        units = _finite_nonnegative(units, "units")
        if units <= 0:
            raise ValueError("units must be greater than zero")
        if not isinstance(reducing, bool):
            raise ValueError("reducing must be a boolean")
        normalised_risk = _normalise_risk_vector(risk, label="risk")
        normalised_capacity = _normalise_capacity(capacity)
        normalised_base = _normalise_risk_vector(base_usage, label="base_usage")
        normalised_metadata = dict(metadata or {})
        metadata_json = _json_dump(normalised_metadata, "metadata")
        intent = {
            "instrument_type": instrument_type,
            "units": units,
            "reducing": reducing,
            "requested_risk": normalised_risk,
            "metadata": normalised_metadata,
        }
        intent_json = _json_dump(intent, "intent")
        risk_json = _json_dump(normalised_risk, "risk")
        binding_ids = [
            self._normalise_identifier(order_id, "order_id")
            for order_id in order_ids
        ]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("order_ids cannot contain duplicates")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._row(connection, reservation_id)
            if existing is not None:
                if str(existing["intent_json"]) != intent_json:
                    raise ValueError(
                        f"reservation_id {reservation_id!r} already has a different intent"
                    )
                if str(existing["status"]) == "ACTIVE":
                    now = _utc_now()
                    for order_id in binding_ids:
                        self._insert_binding(connection, reservation_id, order_id, now)
                connection.commit()
                return self._serialise_row(connection, existing)

            reserved_risk = _empty_vector() if reducing else normalised_risk
            used = _add_vectors(
                normalised_base, self._active_totals_in_connection(connection)
            )
            breaches = self._capacity_breaches(
                used, reserved_risk, normalised_capacity
            )
            if breaches:
                raise RiskCapacityExceeded(breaches)

            now = _utc_now()
            remaining_json = _json_dump(reserved_risk, "remaining_risk")
            connection.execute("""
                INSERT INTO risk_reservations (
                    env, account_id_key, reservation_id, instrument_type,
                    units, reducing, requested_risk_json,
                    remaining_risk_json, intent_json, metadata_json, status,
                    release_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, ?)
            """, (
                *self._scope,
                reservation_id,
                instrument_type,
                units,
                int(reducing),
                risk_json,
                remaining_json,
                intent_json,
                metadata_json,
                now,
                now,
            ))
            for order_id in binding_ids:
                self._insert_binding(connection, reservation_id, order_id, now)
            row = self._row(connection, reservation_id)
            connection.commit()
            assert row is not None
            return self._serialise_row(connection, row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def bind_order(
        self,
        reservation_id: str,
        order_id: str,
        *,
        replaces_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bind a broker order, optionally replacing another one atomically."""
        reservation_id = self._normalise_identifier(reservation_id, "reservation_id")
        order_id = self._normalise_identifier(order_id, "order_id")
        if replaces_order_id is not None:
            replaces_order_id = self._normalise_identifier(
                replaces_order_id, "replaces_order_id"
            )
            if replaces_order_id == order_id:
                raise ValueError("replacement order_id must differ from the old order_id")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, reservation_id)
            if row is None:
                raise KeyError(f"unknown reservation_id {reservation_id!r}")
            if str(row["status"]) != "ACTIVE":
                raise ValueError(f"reservation {reservation_id!r} is not active")
            now = _utc_now()
            if replaces_order_id is not None:
                old = connection.execute("""
                    SELECT reservation_id FROM risk_reservation_orders
                    WHERE env = ? AND account_id_key = ? AND order_id = ?
                """, (*self._scope, replaces_order_id)).fetchone()
                if old is None or str(old["reservation_id"]) != reservation_id:
                    raise ValueError("replacement target is not bound to this reservation")
                connection.execute("""
                    UPDATE risk_reservation_orders
                    SET status = 'REPLACED', updated_at = ?
                    WHERE env = ? AND account_id_key = ? AND order_id = ?
                """, (now, *self._scope, replaces_order_id))
            self._insert_binding(connection, reservation_id, order_id, now)
            connection.execute("""
                UPDATE risk_reservations SET updated_at = ?
                WHERE env = ? AND account_id_key = ? AND reservation_id = ?
            """, (now, *self._scope, reservation_id))
            updated = self._row(connection, reservation_id)
            connection.commit()
            assert updated is not None
            return self._serialise_row(connection, updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def revalue(
        self,
        reservation_id: str,
        risk: Mapping[str, Any],
        capacity: Mapping[str, Any],
        *,
        base_usage: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically recheck and update risk at the latest market price.

        The reservation's current remaining risk is excluded from capacity
        usage, then the proposed full risk is scaled by its unfilled fraction.
        A capacity breach rolls back without changing either risk vector.
        """
        reservation_id = self._normalise_identifier(reservation_id, "reservation_id")
        normalised_risk = _normalise_risk_vector(risk, label="risk")
        normalised_capacity = _normalise_capacity(capacity)
        normalised_base = _normalise_risk_vector(base_usage, label="base_usage")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, reservation_id)
            if row is None:
                raise KeyError(f"unknown reservation_id {reservation_id!r}")
            if str(row["status"]) != "ACTIVE":
                raise ValueError(f"reservation {reservation_id!r} is not active")

            total_filled = sum(
                float(binding["cumulative_filled_units"])
                for binding in self._order_rows(connection, reservation_id)
            )
            units = float(row["units"])
            if total_filled > units + _EPSILON:
                raise ValueError(
                    "cumulative fills across replacement orders exceed reserved units"
                )
            fraction = 0.0 if bool(row["reducing"]) else max(
                0.0, 1.0 - total_filled / units
            )
            proposed_remaining = _scale_vector(normalised_risk, fraction)
            used = _add_vectors(
                normalised_base,
                self._active_totals_in_connection(
                    connection, exclude_reservation_id=reservation_id
                ),
            )
            breaches = self._capacity_breaches(
                used, proposed_remaining, normalised_capacity
            )
            if breaches:
                raise RiskCapacityExceeded(breaches)

            risk_json = _json_dump(normalised_risk, "risk")
            remaining_json = _json_dump(proposed_remaining, "remaining_risk")
            intent = json.loads(row["intent_json"])
            intent["requested_risk"] = normalised_risk
            intent_json = _json_dump(intent, "intent")
            if (
                str(row["requested_risk_json"]) == risk_json
                and str(row["remaining_risk_json"]) == remaining_json
                and str(row["intent_json"]) == intent_json
            ):
                connection.commit()
                return self._serialise_row(connection, row)

            now = _utc_now()
            connection.execute("""
                UPDATE risk_reservations
                SET requested_risk_json = ?, remaining_risk_json = ?,
                    intent_json = ?, updated_at = ?
                WHERE env = ? AND account_id_key = ? AND reservation_id = ?
            """, (
                risk_json,
                remaining_json,
                intent_json,
                now,
                *self._scope,
                reservation_id,
            ))
            updated = self._row(connection, reservation_id)
            connection.commit()
            assert updated is not None
            return self._serialise_row(connection, updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_order_update(
        self,
        order_id: str,
        *,
        cumulative_filled_units: float,
        status: str,
    ) -> Dict[str, Any]:
        """Apply a broker's cumulative fill counter exactly once.

        Remaining risk is recomputed from the sum of all bound order counters,
        rather than decremented from the last poll.  Retried status responses
        are therefore harmless, including across process restarts.
        """
        order_id = self._normalise_identifier(order_id, "order_id")
        cumulative = _finite_nonnegative(
            cumulative_filled_units, "cumulative_filled_units"
        )
        status = self._normalise_identifier(status, "status").upper()

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            order = connection.execute("""
                SELECT reservation_id, cumulative_filled_units, status
                FROM risk_reservation_orders
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (*self._scope, order_id)).fetchone()
            if order is None:
                raise KeyError(f"unknown order_id {order_id!r}")
            previous = float(order["cumulative_filled_units"])
            if cumulative + _EPSILON < previous:
                raise ValueError("cumulative_filled_units cannot decrease")
            previous_status = str(order["status"])
            if (
                previous_status in _INACTIVE_ORDER_STATUSES
                and status not in _INACTIVE_ORDER_STATUSES
            ):
                raise ValueError("a terminal order status cannot become active")

            reservation_id = str(order["reservation_id"])
            reservation = self._row(connection, reservation_id)
            assert reservation is not None
            now = _utc_now()
            connection.execute("""
                UPDATE risk_reservation_orders
                SET cumulative_filled_units = ?, status = ?, updated_at = ?
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (cumulative, status, now, *self._scope, order_id))

            bindings = self._order_rows(connection, reservation_id)
            total_filled = sum(
                float(binding["cumulative_filled_units"])
                for binding in bindings
            )
            units = float(reservation["units"])
            if total_filled > units + _EPSILON:
                raise ValueError(
                    "cumulative fills across replacement orders exceed reserved units"
                )

            reservation_status = str(reservation["status"])
            if reservation_status == "ACTIVE":
                any_active = any(
                    str(binding["status"]) not in _INACTIVE_ORDER_STATUSES
                    for binding in bindings
                )
                if status == "FILLED":
                    new_status = "FILLED"
                    reason = "order_filled"
                    remaining = _empty_vector()
                elif status in _RELEASING_ORDER_STATUSES and not any_active:
                    new_status = "RELEASED"
                    reason = f"order_{status.lower()}"
                    remaining = _empty_vector()
                else:
                    new_status = "ACTIVE"
                    reason = None
                    original = json.loads(reservation["requested_risk_json"])
                    fraction = 0.0 if bool(reservation["reducing"]) else max(
                        0.0, 1.0 - total_filled / units
                    )
                    remaining = _scale_vector(original, fraction)
                connection.execute("""
                    UPDATE risk_reservations
                    SET remaining_risk_json = ?, status = ?,
                        release_reason = ?, updated_at = ?
                    WHERE env = ? AND account_id_key = ? AND reservation_id = ?
                """, (
                    _json_dump(remaining, "remaining_risk"),
                    new_status,
                    reason,
                    now,
                    *self._scope,
                    reservation_id,
                ))
            updated = self._row(connection, reservation_id)
            connection.commit()
            assert updated is not None
            return self._serialise_row(connection, updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release(self, reservation_id: str, reason: str = "manual") -> Dict[str, Any]:
        """Idempotently release all remaining risk for one intent."""
        reservation_id = self._normalise_identifier(reservation_id, "reservation_id")
        reason = self._normalise_identifier(reason, "reason")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection, reservation_id)
            if row is None:
                raise KeyError(f"unknown reservation_id {reservation_id!r}")
            if str(row["status"]) == "ACTIVE":
                now = _utc_now()
                connection.execute("""
                    UPDATE risk_reservations
                    SET remaining_risk_json = ?, status = 'RELEASED',
                        release_reason = ?, updated_at = ?
                    WHERE env = ? AND account_id_key = ? AND reservation_id = ?
                """, (
                    _json_dump(_empty_vector(), "remaining_risk"),
                    reason,
                    now,
                    *self._scope,
                    reservation_id,
                ))
            updated = self._row(connection, reservation_id)
            connection.commit()
            assert updated is not None
            return self._serialise_row(connection, updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def reservation(self, reservation_id: str) -> Optional[Dict[str, Any]]:
        """Return one JSON-safe reservation snapshot, if present."""
        reservation_id = self._normalise_identifier(reservation_id, "reservation_id")
        connection = self._connect()
        try:
            row = self._row(connection, reservation_id)
            return self._serialise_row(connection, row) if row is not None else None
        finally:
            connection.close()

    def reservation_for_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Resolve a broker order id to its stable reservation."""
        order_id = self._normalise_identifier(order_id, "order_id")
        connection = self._connect()
        try:
            binding = connection.execute("""
                SELECT reservation_id FROM risk_reservation_orders
                WHERE env = ? AND account_id_key = ? AND order_id = ?
            """, (*self._scope, order_id)).fetchone()
            if binding is None:
                return None
            row = self._row(connection, str(binding["reservation_id"]))
            assert row is not None
            return self._serialise_row(connection, row)
        finally:
            connection.close()

    def active_reservations(self) -> list[Dict[str, Any]]:
        """Return all active reservations in stable creation order."""
        connection = self._connect()
        try:
            rows = connection.execute("""
                SELECT * FROM risk_reservations
                WHERE env = ? AND account_id_key = ? AND status = 'ACTIVE'
                ORDER BY created_at, reservation_id
            """, self._scope)
            return [self._serialise_row(connection, row) for row in rows]
        finally:
            connection.close()

    def active_totals(self) -> Dict[str, Any]:
        """Reconstruct active reservation totals directly from SQLite."""
        connection = self._connect()
        try:
            return self._active_totals_in_connection(connection)
        finally:
            connection.close()

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe account snapshot suitable for audit/log output."""
        connection = self._connect()
        try:
            rows = list(connection.execute("""
                SELECT * FROM risk_reservations
                WHERE env = ? AND account_id_key = ?
                ORDER BY created_at, reservation_id
            """, self._scope))
            return {
                "env": self.env,
                "account_id_key": self.account_id_key,
                "active_totals": self._active_totals_in_connection(connection),
                "reservations": [
                    self._serialise_row(connection, row) for row in rows
                ],
            }
        finally:
            connection.close()
