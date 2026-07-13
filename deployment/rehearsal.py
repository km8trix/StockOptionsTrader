"""Durable Foundation paper rehearsal and reconciliation evidence.

This is intentionally not a second backtest.  It runs the target-native live
session against a durable PaperTrader broker ledger while LocalBook records the
system's view independently.  Every cycle is idempotent, audited, and bracketed
by strict reconciliation.  Finalization seals all cycle, order, cost, book, and
audit facts into a content-addressed ``PaperValidationArtifact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
import sqlite3
from typing import Any, Mapping

import pandas as pd

from analysis.promotion import (
    PaperValidationArtifact,
    PromotionLevel,
    PromotionRegistry,
    canonical_json,
)
from brokers.local_book import LocalBook
from brokers.paper_trader import PaperTrader
from desks.foundation import FoundationDesk
from desks.registry import create_deployed_desk
from utils.audit import AuditLog
from utils.live_session import LiveTradingSession
from utils.market_hours import MarketHours, NYSE_TZ


_TERMINAL = frozenset({
    "FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "EXECUTED",
})
_MAX_PROSPECTIVE_SKEW_SECONDS = 300.0
_MAX_QUOTE_AGE_SECONDS = 300.0


def _wall_clock() -> datetime:
    """Authoritative observation clock (patched only by deterministic tests)."""
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime | str, *, field: str) -> datetime:
    try:
        parsed = (value if isinstance(value, datetime)
                  else datetime.fromisoformat(str(value)))
    except (TypeError, ValueError) as exc:
        raise RehearsalStateError(f"{field} must be an ISO-8601 timestamp") \
            from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RehearsalStateError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _index_market_date(value: Any) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise RehearsalStateError("market-data index cannot contain NaT")
    # A daily OHLCV index is a session *label*, not a wall-clock instant.
    # Preserve its written date even when a provider attaches a timezone.
    return timestamp.date()


class RehearsalStateError(RuntimeError):
    """A paper rehearsal lifecycle or evidence invariant failed."""


class _CycleMarket:
    def __init__(self) -> None:
        self.as_of: datetime | None = None
        self.frames: dict[str, pd.DataFrame] = {}
        self.execution_prices: dict[str, Mapping[str, Any]] = {}

    def set(self, as_of: datetime,
            frames: Mapping[str, pd.DataFrame],
            execution_prices: Mapping[str, Mapping[str, Any]]) -> None:
        normalized: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            key = str(symbol).strip().upper()
            if not key or not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError("every rehearsal symbol needs a non-empty frame")
            normalized[key] = frame
        if not normalized:
            raise ValueError("rehearsal cycle data cannot be empty")
        self.as_of = as_of
        self.frames = normalized
        self.execution_prices = {
            str(symbol).strip().upper(): dict(value)
            for symbol, value in execution_prices.items()
        }

    def data(self) -> dict[str, pd.DataFrame]:
        if self.as_of is None:
            raise RehearsalStateError("no rehearsal cycle is loaded")
        return self.frames

    def quote(self, symbol: str) -> Mapping[str, Any] | None:
        value = self.execution_prices.get(str(symbol).upper())
        if value is None:
            return None
        # PaperTrader's generic strict-price interface calls the timestamp
        # ``as_of``.  The evidence contract uses the less ambiguous
        # ``observed_at`` because this is a live D+1 quote, not a daily bar.
        return {
            "price": value["price"],
            "as_of": value["observed_at"],
        }

    def clock(self) -> datetime:
        if self.as_of is None:
            return datetime.now(timezone.utc)
        return self.as_of


@dataclass(frozen=True, slots=True)
class RehearsalStatus:
    run_id: str
    research_artifact_hash: str
    state: str
    cycles: int
    errors: int
    started_at: str
    finalized_at: str | None
    paper_artifact_hash: str | None


class FoundationPaperRehearsal:
    """Restart-safe paper execution for one exact approved artifact."""

    def __init__(
            self, *, registry: PromotionRegistry, db_path: str,
            run_id: str, research_artifact_hash: str | None = None,
            initial_capital: float = 100_000.0,
            resume: bool = False):
        self.registry = registry
        self.db_path = os.path.realpath(str(db_path))
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise ValueError("run_id is required")
        self.audit = AuditLog(self.db_path, env="paper")
        self.market = _CycleMarket()
        self._init_tables()

        if resume:
            row = self._run_row()
            stored_hash = str(row["research_artifact_hash"])
            if (research_artifact_hash is not None
                    and research_artifact_hash != stored_hash):
                raise RehearsalStateError(
                    "resume research artifact differs from persisted run")
            self.research_artifact_hash = stored_hash
        else:
            if research_artifact_hash is None:
                raise ValueError("research_artifact_hash is required for a new run")
            self.research_artifact_hash = str(research_artifact_hash)

        self.research = registry.require_approved(
            "foundation", self.research_artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE,
        )
        deployed_desk = create_deployed_desk(
            "foundation", artifact_hash=self.research_artifact_hash,
            promotion_registry=registry,
            required_level=PromotionLevel.PAPER_ELIGIBLE,
        )
        if not isinstance(deployed_desk, FoundationDesk):
            raise RehearsalStateError(
                "Foundation registry returned an unexpected desk type")
        self.desk = deployed_desk
        if resume:
            self._restore_model_checkpoint()
        broker_session = f"foundation-{self.run_id}"
        self.broker = PaperTrader(
            initial_capital=initial_capital,
            db_path=self.db_path,
            session_id=broker_session,
            resume=resume,
            clock=self.market.clock,
            price_provider=self.market.quote,
            max_price_age_business_days=0,
        )
        self.local_book = LocalBook(
            self.db_path, env="paper", account_id_key=self.run_id)

        if resume:
            if not self.local_book.is_initialized():
                raise RehearsalStateError(
                    "persisted rehearsal has an uninitialized LocalBook")
        else:
            self._create_run(initial_capital)
            status = self.broker.get_portfolio_status()
            self.local_book.bootstrap_snapshot(
                {row["symbol"]: row["quantity"]
                 for row in status.get("positions", [])},
                float(status["cash"]),
            )

        # PaperTrader owns broker truth and its PortfolioManager is the
        # strategy's current marked account view.  LocalBook remains a wholly
        # separate SQLite ledger and is never reconstructed from this object.
        self.session = LiveTradingSession(
            desk=self.desk,
            broker=self.broker,
            portfolio=self.broker.portfolio,
            data_fn=self.market.data,
            executor=None,
            audit=self.audit,
            kill_switch=None,
            local_book=self.local_book,
            clock=self.market.clock,
            enforce_market_hours=False,
        )
        # PortfolioSnapshot generation participates in deterministic target
        # intent ids.  Restore it from committed cycles so a crash/retry uses
        # the same client order id instead of creating a second broker order.
        self.session._target_snapshot_version = self.status().cycles

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS foundation_rehearsal_runs (
                    run_id TEXT PRIMARY KEY,
                    research_artifact_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    broker_session_id TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    state TEXT NOT NULL,
                    cycles INTEGER NOT NULL DEFAULT 0,
                    errors INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    finalized_at TEXT,
                    paper_artifact_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS foundation_rehearsal_cycles (
                    run_id TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    execution_prices_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    pre_reconciliation_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    post_reconciliation_json TEXT NOT NULL,
                    orders_json TEXT NOT NULL,
                    error_count INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    PRIMARY KEY (run_id, cycle_id),
                    FOREIGN KEY (run_id)
                        REFERENCES foundation_rehearsal_runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS foundation_rehearsal_checkpoints (
                    run_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_id)
                        REFERENCES foundation_rehearsal_runs(run_id)
                );
            """)
            # The earliest development schema did not separate the D+1 quote
            # from D's signal frame.  Keep such databases readable, but leave
            # their historical rows NULL so they fail qualification instead
            # of being silently upgraded into evidence they never captured.
            cycle_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(foundation_rehearsal_cycles)"
                ).fetchall()
            }
            if "execution_prices_json" not in cycle_columns:
                conn.execute(
                    "ALTER TABLE foundation_rehearsal_cycles "
                    "ADD COLUMN execution_prices_json TEXT"
                )
            if "observed_at" not in cycle_columns:
                conn.execute(
                    "ALTER TABLE foundation_rehearsal_cycles "
                    "ADD COLUMN observed_at TEXT"
                )
            if "state" not in cycle_columns:
                conn.execute(
                    "ALTER TABLE foundation_rehearsal_cycles "
                    "ADD COLUMN state TEXT NOT NULL DEFAULT 'COMPLETED'"
                )
            if "started_at" not in cycle_columns:
                conn.execute(
                    "ALTER TABLE foundation_rehearsal_cycles "
                    "ADD COLUMN started_at TEXT"
                )
                conn.execute(
                    "UPDATE foundation_rehearsal_cycles "
                    "SET started_at = completed_at WHERE started_at IS NULL"
                )
            conn.commit()
        finally:
            conn.close()

    def _create_run(self, initial_capital: float) -> None:
        now = _aware_utc(_wall_clock(), field="wall clock").isoformat()
        deployment_config = getattr(self.desk, "deployment_config", None)
        if deployment_config is None:
            raise RehearsalStateError(
                "deployed desk has no immutable configuration")
        config_hash = deployment_config.config_hash
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                    "SELECT 1 FROM foundation_rehearsal_runs WHERE run_id = ?",
                    (self.run_id,)).fetchone() is not None:
                raise RehearsalStateError(
                    "run_id already exists; use resume=True")
            conn.execute("""
                INSERT INTO foundation_rehearsal_runs
                    (run_id, research_artifact_hash, config_hash,
                     broker_session_id, initial_capital, state, started_at)
                VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
            """, (self.run_id, self.research_artifact_hash, config_hash,
                  f"foundation-{self.run_id}", float(initial_capital), now))
            self.audit._append_in_transaction(
                conn, "paper_rehearsal", "paper_rehearsal_started", {
                    "run_id": self.run_id,
                    "research_artifact_hash": self.research_artifact_hash,
                    "config_hash": config_hash,
                }, timestamp=now)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _restore_model_checkpoint(self) -> None:
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT cycle_id, checkpoint_json
                FROM foundation_rehearsal_checkpoints
                WHERE run_id = ?
            """, (self.run_id,)).fetchone()
            completed = conn.execute("""
                SELECT COUNT(*) AS n,
                       (SELECT cycle_id
                        FROM foundation_rehearsal_cycles
                        WHERE run_id = ? AND state = 'COMPLETED'
                        ORDER BY as_of DESC, cycle_id DESC LIMIT 1) AS latest
                FROM foundation_rehearsal_cycles
                WHERE run_id = ? AND state = 'COMPLETED'
            """, (self.run_id, self.run_id)).fetchone()
        finally:
            conn.close()
        count = int(completed["n"]) if completed is not None else 0
        if count == 0:
            if row is not None:
                raise RehearsalStateError(
                    "model checkpoint exists without a completed cycle")
            return
        if row is None or row["cycle_id"] != completed["latest"]:
            raise RehearsalStateError(
                "paper model checkpoint is missing or behind cycle evidence")
        try:
            checkpoint = json.loads(row["checkpoint_json"])
            self.desk.restore_model_checkpoint(checkpoint)
        except Exception as exc:
            raise RehearsalStateError(
                "paper model checkpoint failed integrity/compatibility "
                "validation") from exc

    def _run_row(self) -> sqlite3.Row:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM foundation_rehearsal_runs WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise RehearsalStateError("paper rehearsal run does not exist")
        return row

    def status(self) -> RehearsalStatus:
        row = self._run_row()
        return RehearsalStatus(
            run_id=self.run_id,
            research_artifact_hash=str(row["research_artifact_hash"]),
            state=str(row["state"]),
            cycles=int(row["cycles"]),
            errors=int(row["errors"]),
            started_at=str(row["started_at"]),
            finalized_at=row["finalized_at"],
            paper_artifact_hash=row["paper_artifact_hash"],
        )

    @staticmethod
    def _normalize_as_of(value: datetime | str) -> datetime:
        return _aware_utc(value, field="paper cycle as_of")

    @staticmethod
    def _input_hash(as_of: datetime,
                    data: Mapping[str, pd.DataFrame],
                    execution_prices: Mapping[str, Mapping[str, Any]]) -> str:
        digest = hashlib.sha256(as_of.isoformat().encode("utf-8"))
        for symbol in sorted(data):
            frame = data[symbol]
            digest.update(str(symbol).upper().encode("utf-8"))
            digest.update(pd.util.hash_pandas_object(
                frame, index=True).values.tobytes())
            digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
        digest.update(canonical_json({
            str(symbol).strip().upper(): dict(value)
            for symbol, value in execution_prices.items()
        }).encode("utf-8"))
        return digest.hexdigest()

    def _validate_cycle_data(
            self, as_of: datetime,
            data: Mapping[str, pd.DataFrame]) -> None:
        expected = set(self.research.payload["universe"])
        actual = {str(symbol).strip().upper() for symbol in data}
        if actual != expected or len(data) != len(actual):
            raise RehearsalStateError(
                "paper cycle universe differs from the research artifact "
                f"(missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)})")
        local_date = as_of.astimezone(NYSE_TZ).date()
        calendar = MarketHours()
        if not calendar.is_market_open(as_of):
            raise RehearsalStateError(
                "paper cycles must execute during the NYSE regular session")
        previous_session = local_date - timedelta(days=1)
        for _ in range(10):
            if calendar.is_trading_day(previous_session):
                break
            previous_session -= timedelta(days=1)
        else:  # pragma: no cover - calendar defensive bound
            raise RehearsalStateError(
                "cannot resolve the prior exchange session")
        required = {"open", "high", "low", "close", "volume", "macd",
                    "signal", "rsi", "volume_sma"}
        for symbol, frame in data.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise RehearsalStateError(
                    f"paper cycle data is empty for {symbol}")
            if len(frame) < 2:
                raise RehearsalStateError(
                    f"paper cycle data needs two or more rows for {symbol}")
            try:
                index = pd.DatetimeIndex(frame.index)
            except (TypeError, ValueError) as exc:
                raise RehearsalStateError(
                    f"paper cycle index is not datetime-like for {symbol}"
                ) from exc
            if index.hasnans or not index.is_unique \
                    or not index.is_monotonic_increasing:
                raise RehearsalStateError(
                    f"paper cycle index must be unique and increasing for "
                    f"{symbol}")
            if not required.issubset(frame.columns):
                raise RehearsalStateError(
                    f"paper cycle data lacks indicators for {symbol}")
            last = _index_market_date(index.max())
            if last != previous_session:
                raise RehearsalStateError(
                    f"paper signal data for {symbol} must end on prior "
                    f"exchange session {previous_session}")

    def _validate_execution_prices(
            self, as_of: datetime,
            values: Mapping[str, Mapping[str, Any]]) -> None:
        expected = set(self.research.payload["universe"])
        actual = {str(symbol).strip().upper() for symbol in values}
        if actual != expected or len(values) != len(actual):
            raise RehearsalStateError(
                "execution-price universe differs from research artifact")
        execution_date = as_of.astimezone(NYSE_TZ).date()
        for symbol, value in values.items():
            if not isinstance(value, Mapping):
                raise RehearsalStateError(
                    f"execution price for {symbol} must carry price/as_of")
            try:
                price = float(value["price"])
                observed_at = _aware_utc(
                    value["observed_at"],
                    field=f"execution price observed_at for {symbol}",
                )
                source = str(value["source"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise RehearsalStateError(
                    f"invalid execution price for {symbol}") from exc
            if not math.isfinite(price) or price <= 0:
                raise RehearsalStateError(
                    f"invalid execution price for {symbol}")
            if not source:
                raise RehearsalStateError(
                    f"execution price for {symbol} has no source")
            quote_age = (as_of - observed_at).total_seconds()
            if quote_age < 0 or quote_age > _MAX_QUOTE_AGE_SECONDS:
                raise RehearsalStateError(
                    f"execution price for {symbol} is future-dated or stale")
            if observed_at.astimezone(NYSE_TZ).date() != execution_date:
                raise RehearsalStateError(
                    f"execution price for {symbol} is not from execution date")

    @staticmethod
    def _cycle_id(as_of: datetime) -> str:
        return hashlib.sha256(as_of.isoformat().encode("utf-8")).hexdigest()

    def _validate_prospective(self, as_of: datetime) -> datetime:
        observed_at = _aware_utc(_wall_clock(), field="wall clock")
        lag = (observed_at - as_of).total_seconds()
        if lag < -1.0 or lag > _MAX_PROSPECTIVE_SKEW_SECONDS:
            raise RehearsalStateError(
                "new paper cycles must be observed prospectively within "
                "five minutes of wall clock")
        started_at = _aware_utc(
            str(self._run_row()["started_at"]), field="run started_at")
        if as_of < started_at:
            raise RehearsalStateError(
                "paper cycle cannot predate the rehearsal run")
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT MAX(as_of) AS latest
                FROM foundation_rehearsal_cycles
                WHERE run_id = ?
            """, (self.run_id,)).fetchone()
        finally:
            conn.close()
        if row is not None and row["latest"] is not None \
                and as_of <= _aware_utc(row["latest"], field="prior cycle"):
            raise RehearsalStateError(
                "paper cycle timestamps must increase strictly")
        return observed_at

    @staticmethod
    def _validate_started_recovery(
            as_of: datetime, existing: Mapping[str, Any]) -> datetime:
        """Admit a reserved cycle only while its original live window holds.

        ``STARTED`` is a crash-recovery reservation, not permission to replay
        yesterday's decision with yesterday's quote.  Re-check both the
        prospective cycle window and every persisted quote against the current
        wall clock before any reconciliation, model evaluation, or broker call.
        The input hash separately guarantees that a caller cannot replace the
        reserved quote with a newer one under the same cycle identity.
        """
        now = _aware_utc(_wall_clock(), field="wall clock")
        if not MarketHours().is_market_open(now):
            raise RehearsalStateError(
                "STARTED paper cycle is outside the live recovery window: "
                "NYSE regular trading is closed")

        cycle_age = (now - as_of).total_seconds()
        if cycle_age < -1.0 or cycle_age > _MAX_PROSPECTIVE_SKEW_SECONDS:
            raise RehearsalStateError(
                "STARTED paper cycle is outside the live recovery window: "
                "its reserved decision timestamp is stale")

        execution_date = as_of.astimezone(NYSE_TZ).date()
        for symbol, value in existing["execution_prices"].items():
            try:
                quote_at = _aware_utc(
                    value["observed_at"],
                    field=f"persisted execution price for {symbol}",
                )
            except (KeyError, TypeError) as exc:
                raise RehearsalStateError(
                    "STARTED paper cycle has invalid persisted quote evidence"
                ) from exc
            quote_age = (now - quote_at).total_seconds()
            if quote_age < -1.0 or quote_age > _MAX_QUOTE_AGE_SECONDS:
                raise RehearsalStateError(
                    "STARTED paper cycle is outside the live recovery window: "
                    f"its persisted quote for {symbol} is stale")
            if quote_at.astimezone(NYSE_TZ).date() != execution_date:
                raise RehearsalStateError(
                    "STARTED paper cycle has a persisted quote from a "
                    "different execution date")
        return _aware_utc(
            existing["observed_at"], field="cycle observed_at")

    def _existing_cycle(self, cycle_id: str) -> Mapping[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("""
                SELECT * FROM foundation_rehearsal_cycles
                WHERE run_id = ? AND cycle_id = ?
            """, (self.run_id, cycle_id)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        execution_prices_json = row["execution_prices_json"]
        observed_at = row["observed_at"]
        if not execution_prices_json or not observed_at:
            raise RehearsalStateError(
                "persisted cycle lacks prospective execution-price evidence"
            )
        state = str(row["state"])
        if state not in {"STARTED", "COMPLETED"}:
            raise RehearsalStateError(
                f"persisted cycle has unknown state {state!r}")
        return {
            "cycle_id": row["cycle_id"],
            "as_of": row["as_of"],
            "observed_at": observed_at,
            "input_hash": row["input_hash"],
            "execution_prices": json.loads(execution_prices_json),
            "pre_reconciliation": json.loads(row["pre_reconciliation_json"]),
            "result": json.loads(row["result_json"]),
            "post_reconciliation": json.loads(row["post_reconciliation_json"]),
            "orders": json.loads(row["orders_json"]),
            "error_count": int(row["error_count"]),
            "state": state,
            "idempotent": state == "COMPLETED",
        }

    def _reserve_cycle(
            self, *, cycle_id: str, as_of: datetime, observed_at: datetime,
            input_hash: str,
            execution_prices: Mapping[str, Mapping[str, Any]]) -> None:
        started_at = observed_at.isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                INSERT INTO foundation_rehearsal_cycles
                    (run_id, cycle_id, as_of, observed_at, input_hash,
                     execution_prices_json, state, pre_reconciliation_json,
                     result_json, post_reconciliation_json, orders_json,
                     error_count, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, 'STARTED', '{}', '{}', '{}',
                        '[]', 0, ?, NULL)
            """, (
                self.run_id, cycle_id, as_of.isoformat(), started_at,
                input_hash, canonical_json({
                    str(symbol).strip().upper(): dict(value)
                    for symbol, value in execution_prices.items()
                }), started_at,
            ))
            self.audit._append_in_transaction(
                conn, "paper_rehearsal", "paper_cycle_started", {
                    "run_id": self.run_id,
                    "cycle_id": cycle_id,
                    "as_of": as_of.isoformat(),
                    "observed_at": started_at,
                    "input_hash": input_hash,
                    "execution_prices": {
                        str(symbol).strip().upper(): dict(value)
                        for symbol, value in execution_prices.items()
                    },
                }, timestamp=started_at)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def run_cycle(
            self, as_of: datetime | str,
            data: Mapping[str, pd.DataFrame], *,
            execution_prices: Mapping[str, Mapping[str, Any]],
            ) -> Mapping[str, Any]:
        status = self.status()
        if status.state != "ACTIVE":
            raise RehearsalStateError(
                f"paper rehearsal is {status.state.lower()}")
        timestamp = self._normalize_as_of(as_of)
        self._validate_cycle_data(timestamp, data)
        self._validate_execution_prices(timestamp, execution_prices)
        input_hash = self._input_hash(timestamp, data, execution_prices)
        cycle_id = self._cycle_id(timestamp)
        existing = self._existing_cycle(cycle_id)
        if existing is not None:
            if existing["input_hash"] != input_hash:
                raise RehearsalStateError(
                    "cycle timestamp was reused with different input data")
            if existing["state"] == "COMPLETED":
                return existing
            observed_at = self._validate_started_recovery(
                timestamp, existing)
            self.audit.append(
                "paper_rehearsal", "paper_cycle_resumed", {
                    "run_id": self.run_id,
                    "cycle_id": cycle_id,
                    "input_hash": input_hash,
                })
        else:
            observed_at = self._validate_prospective(timestamp)
            self._reserve_cycle(
                cycle_id=cycle_id,
                as_of=timestamp,
                observed_at=observed_at,
                input_hash=input_hash,
                execution_prices=execution_prices,
            )

        self.market.set(timestamp, data, execution_prices)
        errors = 0
        pre: Mapping[str, Any]
        result: Mapping[str, Any]
        post: Mapping[str, Any]
        try:
            pre = self.session.run_reconciliation(cash_tolerance=0.01)
        except Exception as exc:
            pre = {"ok": False, "mismatches": [], "error": str(exc),
                   "error_type": type(exc).__name__}
            result = {"status": "halted",
                      "reason": "pre_reconciliation_error", "reports": []}
            post = pre
            errors += 1
        else:
            if pre.get("ok") is not True:
                result = {"status": "halted",
                          "reason": "pre_reconciliation_failed", "reports": []}
                post = pre
                errors += 1
            else:
                result = self.session.evaluate_once()
                self.broker.process_orders()
                try:
                    post = self.session.run_reconciliation(cash_tolerance=0.01)
                except Exception as exc:
                    post = {"ok": False, "mismatches": [], "error": str(exc),
                            "error_type": type(exc).__name__}
                    errors += 1
                if post.get("ok") is not True:
                    errors += 1
                if result.get("status") not in {"ok", "pending"}:
                    errors += 1
                errors += sum(
                    1 for report in result.get("reports", [])
                    if report.get("status") == "error")

        orders = self.broker.orders_snapshot()
        unknown = sum(1 for report in result.get("reports", [])
                      if report.get("order_id") is not None
                      and not any(order["order_id"] == report["order_id"]
                                  for order in orders))
        open_orders = sum(1 for order in orders
                          if order["status"] not in _TERMINAL)
        if unknown or open_orders:
            errors += unknown + open_orders

        # Cadence/model state is part of the same durable commit as the cycle.
        # The checkpoint contains primitive frames and deterministic refit
        # metadata only; no pickle or executable estimator bytes are loaded.
        checkpoint_json = canonical_json(self.desk.model_checkpoint_state())
        completed = _aware_utc(
            _wall_clock(), field="wall clock").isoformat()
        response = {
            "cycle_id": cycle_id,
            "as_of": timestamp.isoformat(),
            "observed_at": observed_at.isoformat(),
            "input_hash": input_hash,
            "execution_prices": {
                str(symbol).strip().upper(): dict(value)
                for symbol, value in execution_prices.items()
            },
            "pre_reconciliation": pre,
            "result": result,
            "post_reconciliation": post,
            "orders": orders,
            "error_count": errors,
            "state": "COMPLETED",
            "idempotent": False,
        }
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute("""
                UPDATE foundation_rehearsal_cycles
                SET state = 'COMPLETED',
                    pre_reconciliation_json = ?, result_json = ?,
                    post_reconciliation_json = ?, orders_json = ?,
                    error_count = ?, completed_at = ?
                WHERE run_id = ? AND cycle_id = ? AND state = 'STARTED'
            """, (
                canonical_json(pre), canonical_json(result),
                canonical_json(post), canonical_json(orders), errors,
                completed, self.run_id, cycle_id,
            ))
            if cursor.rowcount != 1:
                raise RehearsalStateError(
                    "paper cycle completion lost its STARTED reservation")
            conn.execute("""
                INSERT INTO foundation_rehearsal_checkpoints
                    (run_id, cycle_id, checkpoint_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    cycle_id = excluded.cycle_id,
                    checkpoint_json = excluded.checkpoint_json,
                    updated_at = excluded.updated_at
            """, (self.run_id, cycle_id, checkpoint_json, completed))
            conn.execute("""
                UPDATE foundation_rehearsal_runs
                SET cycles = cycles + 1, errors = errors + ?
                WHERE run_id = ? AND state = 'ACTIVE'
            """, (errors, self.run_id))
            self.audit._append_in_transaction(
                conn, "paper_rehearsal", "paper_cycle_completed", {
                    "run_id": self.run_id,
                    "cycle_id": cycle_id,
                    "errors": errors,
                    "pre_reconciliation_ok": pre.get("ok") is True,
                    "post_reconciliation_ok": post.get("ok") is True,
                    "order_count": len(orders),
                    "model_checkpoint_sha256": json.loads(
                        checkpoint_json)["sha256"],
                }, timestamp=completed)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return response

    def _cycles(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT * FROM foundation_rehearsal_cycles
                WHERE run_id = ? ORDER BY as_of, cycle_id
            """, (self.run_id,)).fetchall()
        finally:
            conn.close()
        cycles = []
        for row in rows:
            execution_prices_json = row["execution_prices_json"]
            if str(row["state"]) != "COMPLETED":
                raise RehearsalStateError(
                    "paper run has an incomplete cycle; resume it before "
                    "finalization")
            if not execution_prices_json or not row["observed_at"] \
                    or not row["started_at"] or not row["completed_at"]:
                raise RehearsalStateError(
                    "paper run contains a legacy cycle without separate "
                    "prospective execution-price evidence"
                )
            cycles.append({
                "cycle_id": row["cycle_id"],
                "as_of": row["as_of"],
                "observed_at": row["observed_at"],
                "input_hash": row["input_hash"],
                "execution_prices": json.loads(execution_prices_json),
                "pre_reconciliation": json.loads(
                    row["pre_reconciliation_json"]),
                "result": json.loads(row["result_json"]),
                "post_reconciliation": json.loads(
                    row["post_reconciliation_json"]),
                "orders": json.loads(row["orders_json"]),
                "error_count": int(row["error_count"]),
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            })
        return cycles

    def finalize(self) -> PaperValidationArtifact:
        status = self.status()
        if status.state == "FINALIZED":
            assert status.paper_artifact_hash is not None
            return self.registry.paper_validation_store.load(
                status.paper_artifact_hash)
        if status.state != "ACTIVE":
            raise RehearsalStateError(
                f"cannot finalize rehearsal in {status.state} state")
        cycles = self._cycles()
        orders = self.broker.orders_snapshot()
        final_reconciliation: Mapping[str, Any]
        try:
            final_reconciliation = self.session.run_reconciliation(
                cash_tolerance=0.01)
        except Exception as exc:
            final_reconciliation = {
                "ok": False, "mismatches": [], "error": str(exc),
                "error_type": type(exc).__name__,
            }
        reconciliation_checks = sum(
            1 for cycle in cycles
            for key in ("pre_reconciliation", "post_reconciliation")
            if key in cycle
        ) + 1
        reconciliation_failures = sum(
            1 for cycle in cycles
            for key in ("pre_reconciliation", "post_reconciliation")
            if cycle[key].get("ok") is not True
        ) + int(final_reconciliation.get("ok") is not True)
        unknown_orders = sum(
            1 for cycle in cycles for report in cycle["result"].get("reports", [])
            if report.get("order_id") is not None
            and not any(item["order_id"] == report["order_id"]
                        for item in orders)
        )
        open_orders = sum(1 for item in orders
                          if item["status"] not in _TERMINAL)
        fills = sum(1 for item in orders if item["status"] == "FILLED")
        transaction_cost = sum(float(item.get("transaction_cost", 0) or 0)
                               for item in orders)
        audit_verification = self.audit.verify_chain()
        run_started = _aware_utc(status.started_at, field="run started_at")
        prospective = bool(cycles)
        previous_as_of: datetime | None = None
        for cycle in cycles:
            cycle_as_of = _aware_utc(cycle["as_of"], field="cycle as_of")
            observed = _aware_utc(
                cycle["observed_at"], field="cycle observed_at")
            completed_at = _aware_utc(
                cycle["completed_at"], field="cycle completed_at")
            prospective = prospective and (
                run_started <= cycle_as_of <= observed <= completed_at
                and (observed - cycle_as_of).total_seconds()
                <= _MAX_PROSPECTIVE_SKEW_SECONDS
                and (previous_as_of is None or cycle_as_of > previous_as_of)
            )
            previous_as_of = cycle_as_of
        run_summary = {
            "cycles": len(cycles),
            "sessions": len({
                _aware_utc(cycle["as_of"], field="cycle as_of")
                .astimezone(NYSE_TZ).date().isoformat()
                for cycle in cycles
            }),
            "fills": fills,
            "errors": int(status.errors) + reconciliation_failures
                      + unknown_orders + open_orders,
            "prospective": prospective,
        }
        reconciliation_evidence = {
            "checks": reconciliation_checks,
            "failures": reconciliation_failures,
            "unknown_orders": unknown_orders,
            "open_orders": open_orders,
        }
        broker_status = self.broker.get_portfolio_status()
        local_snapshot = self.local_book.reconciliation_snapshot()
        conn = self._connect()
        try:
            checkpoint_row = conn.execute("""
                SELECT cycle_id, checkpoint_json
                FROM foundation_rehearsal_checkpoints
                WHERE run_id = ?
            """, (self.run_id,)).fetchone()
            checkpoint_audit_rows = conn.execute("""
                SELECT payload FROM audit_log
                WHERE event_type = 'paper_cycle_completed'
                ORDER BY seq DESC
            """).fetchall()
        finally:
            conn.close()
        model_checkpoint = None
        if checkpoint_row is not None:
            checkpoint_document = json.loads(checkpoint_row["checkpoint_json"])
            checkpoint_audit = None
            for audit_row in checkpoint_audit_rows:
                audit_payload = json.loads(audit_row["payload"])
                if (audit_payload.get("run_id") == self.run_id
                        and audit_payload.get("cycle_id")
                        == checkpoint_row["cycle_id"]):
                    checkpoint_audit = audit_payload
                    break
            if (not isinstance(checkpoint_audit, Mapping)
                    or checkpoint_audit.get("model_checkpoint_sha256")
                    != checkpoint_document.get("sha256")):
                raise RehearsalStateError(
                    "paper model checkpoint differs from its audited cycle seal")
            model_checkpoint = {
                "cycle_id": checkpoint_row["cycle_id"],
                "sha256": checkpoint_document["sha256"],
                "state": checkpoint_document,
            }
        deployment_config = getattr(self.desk, "deployment_config", None)
        if deployment_config is None:
            raise RehearsalStateError(
                "deployed desk lost its immutable configuration")
        evidence = {
            "runner": "foundation_paper_rehearsal_v2",
            "run_id": self.run_id,
            "started_at": status.started_at,
            "cycles": cycles,
            "final_reconciliation": final_reconciliation,
            "broker": {
                "session_id": self.broker.session_id,
                "revision": broker_status.get("revision"),
                "cash": broker_status["cash"],
                "positions": broker_status["positions"],
                "orders": orders,
                "transaction_cost": transaction_cost,
            },
            "local_book": local_snapshot,
            "audit_verification": audit_verification,
            "model_checkpoint": model_checkpoint,
            "deployment_config_hash": deployment_config.config_hash,
        }
        artifact = PaperValidationArtifact.create(
            research_artifact=self.research,
            run_summary=run_summary,
            reconciliation_evidence=reconciliation_evidence,
            audit_verified=audit_verification.get("ok") is True,
            policy=self.registry.paper_validation_policy,
            evidence=evidence,
        )
        self.registry.paper_validation_store.persist(artifact)
        completed = _aware_utc(
            _wall_clock(), field="wall clock").isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute("""
                UPDATE foundation_rehearsal_runs
                SET state = 'FINALIZED', finalized_at = ?,
                    paper_artifact_hash = ?
                WHERE run_id = ? AND state = 'ACTIVE'
            """, (completed, artifact.artifact_hash, self.run_id))
            if cursor.rowcount != 1:
                raise RehearsalStateError(
                    "concurrent paper rehearsal finalization")
            self.audit._append_in_transaction(
                conn, "paper_rehearsal", "paper_rehearsal_finalized", {
                    "run_id": self.run_id,
                    "paper_artifact_hash": artifact.artifact_hash,
                    "passed": artifact.passed,
                    "run_summary": run_summary,
                    "reconciliation_evidence": reconciliation_evidence,
                }, timestamp=completed)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return artifact


__all__ = [
    "FoundationPaperRehearsal",
    "RehearsalStateError",
    "RehearsalStatus",
]
