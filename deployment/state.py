"""Durable, audited state machine for one controlled live deployment.

Research and paper artifacts prove what was tested.  This module records the
separate operational decision to let that exact build trade one exact account
under explicit limits.  Every transition is compare-and-swap protected and is
written in the same SQLite transaction as its audit event.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
import sqlite3
from typing import Any, Callable, Sequence

from analysis.promotion import ArtifactIntegrityError, canonical_json


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(value: str, label: str) -> str:
    candidate = str(value).strip().lower()
    if len(candidate) != 64 or any(ch not in "0123456789abcdef"
                                   for ch in candidate):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return candidate


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _aware_utc(value: str, label: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


class DeploymentState(str, Enum):
    STAGED = "staged"
    RISK_APPROVED = "risk_approved"
    OPS_APPROVED = "ops_approved"
    ACTIVATED = "activated"
    ARMED = "armed"
    RUNNING = "running"
    PAUSED = "paused"
    REVOKED = "revoked"


class DeploymentStateError(RuntimeError):
    """A deployment transition or invariant failed closed."""


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    """Content-addressed authority and limits for one live account."""

    manifest_hash: str
    payload_json: str

    @classmethod
    def create(
            cls, *, research_artifact_hash: str, paper_evidence_hash: str,
            strategy_version: str, code_sha: str, config_hash: str,
            account_id_key: str, allowed_universe: Sequence[str],
            expires_at: str, created_by: str,
            environment: str = "production",
            max_order_notional: float = 2_500.0,
            max_daily_notional: float = 10_000.0,
            max_daily_orders: int = 5,
            max_gross_nav_multiple: float = 1.0,
            max_per_name_nav_fraction: float = 0.10,
            interval_minutes: float = 15.0,
            max_data_age_business_days: int = 1,
            require_flat_start: bool = True) -> DeploymentManifest:
        research_hash = _hash(research_artifact_hash,
                              "research_artifact_hash")
        paper_hash = _hash(paper_evidence_hash, "paper_evidence_hash")
        config_digest = _hash(config_hash, "config_hash")
        strategy_version = str(strategy_version).strip()
        code_sha = str(code_sha).strip().lower()
        account = str(account_id_key).strip()
        actor = str(created_by).strip()
        env = str(environment).strip().lower()
        if not strategy_version or not code_sha or not account or not actor:
            raise ValueError(
                "strategy_version, code_sha, account_id_key and created_by "
                "are required")
        if env != "production":
            raise ValueError("controlled live manifests require production")
        universe = sorted({str(symbol).strip().upper()
                           for symbol in allowed_universe
                           if str(symbol).strip()})
        if not universe:
            raise ValueError("allowed_universe cannot be empty")
        if isinstance(max_daily_orders, bool) or int(max_daily_orders) < 1:
            raise ValueError("max_daily_orders must be a positive integer")
        if isinstance(max_data_age_business_days, bool) \
                or int(max_data_age_business_days) < 1:
            raise ValueError(
                "max_data_age_business_days must be a positive integer")
        if type(require_flat_start) is not bool:
            raise ValueError("require_flat_start must be a boolean")
        payload = {
            "schema_version": 1,
            "strategy_id": "foundation",
            "research_artifact_hash": research_hash,
            "paper_evidence_hash": paper_hash,
            "strategy_version": strategy_version,
            "code_sha": code_sha,
            "config_hash": config_digest,
            "environment": env,
            "account_id_key": account,
            "allowed_universe": universe,
            "expires_at": _aware_utc(expires_at, "expires_at"),
            "created_by": actor,
            "limits": {
                "max_order_notional": _positive(
                    max_order_notional, "max_order_notional"),
                "max_daily_notional": _positive(
                    max_daily_notional, "max_daily_notional"),
                "max_daily_orders": int(max_daily_orders),
                "max_gross_nav_multiple": _positive(
                    max_gross_nav_multiple, "max_gross_nav_multiple"),
                "max_per_name_nav_fraction": _positive(
                    max_per_name_nav_fraction,
                    "max_per_name_nav_fraction"),
                "interval_minutes": _positive(
                    interval_minutes, "interval_minutes"),
                "max_data_age_business_days": int(
                    max_data_age_business_days),
            },
            "require_flat_start": require_flat_start,
        }
        payload_json = canonical_json(payload)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return cls(digest, payload_json)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def __getattr__(self, name: str) -> Any:
        direct = {
            "research_artifact_hash", "paper_evidence_hash",
            "strategy_version", "code_sha", "config_hash", "environment",
            "account_id_key", "allowed_universe", "expires_at",
            "created_by", "require_flat_start",
        }
        if name in direct:
            return self.payload[name]
        if name.startswith("max_") or name == "interval_minutes":
            limits = self.payload["limits"]
            if name in limits:
                return limits[name]
        raise AttributeError(name)

    def to_json(self) -> str:
        return canonical_json({"manifest_hash": self.manifest_hash,
                               "payload": self.payload}) + "\n"

    @classmethod
    def from_json(cls, value: str) -> DeploymentManifest:
        try:
            document = json.loads(value)
            claimed = _hash(document["manifest_hash"], "manifest_hash")
            payload = document["payload"]
            if int(payload["schema_version"]) != 1:
                raise ValueError("unsupported schema")
            rebuilt = cls.create(
                research_artifact_hash=payload["research_artifact_hash"],
                paper_evidence_hash=payload["paper_evidence_hash"],
                strategy_version=payload["strategy_version"],
                code_sha=payload["code_sha"],
                config_hash=payload["config_hash"],
                environment=payload["environment"],
                account_id_key=payload["account_id_key"],
                allowed_universe=payload["allowed_universe"],
                expires_at=payload["expires_at"],
                created_by=payload["created_by"],
                require_flat_start=payload["require_flat_start"],
                **payload["limits"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("invalid deployment manifest") from exc
        if claimed != rebuilt.manifest_hash:
            raise ArtifactIntegrityError("deployment manifest hash mismatch")
        return rebuilt


class DeploymentStore:
    """SQLite state machine with transition and audit in one transaction."""

    def __init__(self, db_path: str, audit,
                 clock: Callable[[], datetime] = _utc_now):
        self.db_path = os.path.realpath(str(db_path))
        self.audit = audit
        self._clock = clock
        audit_path = os.path.realpath(str(getattr(audit, "db_path", "")))
        if not audit_path or audit_path != self.db_path:
            raise ValueError("deployment store and audit must share one database")
        if not callable(getattr(audit, "_append_in_transaction", None)):
            raise ValueError("deployment store requires transactional AuditLog")
        self._init_tables()

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
                CREATE TABLE IF NOT EXISTS foundation_deployments (
                    manifest_hash TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    risk_approver TEXT,
                    ops_approver TEXT,
                    reason TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deployment_order_authorizations (
                    manifest_hash TEXT NOT NULL,
                    intent_id TEXT NOT NULL,
                    trading_date TEXT NOT NULL,
                    side TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    reference_price REAL NOT NULL,
                    notional REAL NOT NULL,
                    opening INTEGER NOT NULL,
                    authorized_at TEXT NOT NULL,
                    PRIMARY KEY (manifest_hash, intent_id),
                    FOREIGN KEY (manifest_hash)
                        REFERENCES foundation_deployments(manifest_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_deployment_daily_orders
                    ON deployment_order_authorizations
                    (manifest_hash, trading_date, opening);
            """)
            conn.commit()
        finally:
            conn.close()

    def stage(self, manifest: DeploymentManifest) -> dict[str, Any]:
        verified = DeploymentManifest.from_json(manifest.to_json())
        now = self._clock().astimezone(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT manifest_json FROM foundation_deployments "
                "WHERE manifest_hash = ?", (verified.manifest_hash,),
            ).fetchone()
            if existing is not None:
                if existing["manifest_json"] != verified.to_json():
                    raise ArtifactIntegrityError(
                        "manifest hash is stored with different bytes")
                conn.rollback()
                return self.get(verified.manifest_hash)
            conn.execute("""
                INSERT INTO foundation_deployments
                    (manifest_hash, manifest_json, state, version, updated_at)
                VALUES (?, ?, ?, 1, ?)
            """, (verified.manifest_hash, verified.to_json(),
                  DeploymentState.STAGED.value, now))
            self.audit._append_in_transaction(
                conn, verified.created_by, "deployment_staged", {
                    "manifest_hash": verified.manifest_hash,
                    "research_artifact_hash": verified.research_artifact_hash,
                    "paper_evidence_hash": verified.paper_evidence_hash,
                    "account_id_key": verified.account_id_key,
                }, timestamp=now)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self.get(verified.manifest_hash)

    def get(self, manifest_hash: str) -> dict[str, Any]:
        digest = _hash(manifest_hash, "manifest_hash")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM foundation_deployments WHERE manifest_hash = ?",
                (digest,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise DeploymentStateError("deployment manifest is not staged")
        manifest = DeploymentManifest.from_json(row["manifest_json"])
        if manifest.manifest_hash != digest:
            raise ArtifactIntegrityError("stored deployment manifest mismatch")
        return {
            "manifest": manifest,
            "state": DeploymentState(row["state"]),
            "version": int(row["version"]),
            "risk_approver": row["risk_approver"],
            "ops_approver": row["ops_approver"],
            "reason": row["reason"],
            "updated_at": row["updated_at"],
        }

    def records(self, states: Sequence[DeploymentState] | None = None
                ) -> list[dict[str, Any]]:
        """Return verified records, optionally filtered by durable state."""
        query = "SELECT manifest_hash FROM foundation_deployments"
        params: list[Any] = []
        if states:
            normalized = [DeploymentState(state).value for state in states]
            query += " WHERE state IN (" + ",".join("?" * len(normalized)) + ")"
            params.extend(normalized)
        query += " ORDER BY updated_at, manifest_hash"
        conn = self._connect()
        try:
            hashes = [row["manifest_hash"]
                      for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()
        return [self.get(item) for item in hashes]

    def _transition(self, manifest_hash: str, *,
                    allowed: Sequence[DeploymentState],
                    target: DeploymentState, actor: str,
                    event_type: str, reason: str | None = None,
                    require_distinct_risk_actor: bool = False) -> dict[str, Any]:
        actor = str(actor).strip()
        if not actor:
            raise ValueError("transition actor is required")
        digest = _hash(manifest_hash, "manifest_hash")
        now = self._clock().astimezone(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM foundation_deployments WHERE manifest_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                raise DeploymentStateError("deployment manifest is not staged")
            current = DeploymentState(row["state"])
            if current not in set(allowed):
                raise DeploymentStateError(
                    f"cannot transition deployment from {current.value} "
                    f"to {target.value}")
            if require_distinct_risk_actor and actor == row["risk_approver"]:
                raise DeploymentStateError(
                    "operations approver must differ from risk approver")
            assignments = ["state = ?", "version = version + 1",
                           "reason = ?", "updated_at = ?"]
            values: list[Any] = [target.value, reason, now]
            if target == DeploymentState.RISK_APPROVED:
                assignments.append("risk_approver = ?")
                values.append(actor)
            if target == DeploymentState.OPS_APPROVED:
                assignments.append("ops_approver = ?")
                values.append(actor)
            values.extend([digest, int(row["version"]), current.value])
            cursor = conn.execute(
                "UPDATE foundation_deployments SET " + ", ".join(assignments)
                + " WHERE manifest_hash = ? AND version = ? AND state = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise DeploymentStateError("concurrent deployment transition")
            self.audit._append_in_transaction(
                conn, actor, event_type, {
                    "manifest_hash": digest,
                    "from": current.value,
                    "to": target.value,
                    "reason": reason,
                }, timestamp=now)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self.get(digest)

    def approve_risk(self, manifest_hash: str, actor: str) -> dict[str, Any]:
        return self._transition(
            manifest_hash, allowed=[DeploymentState.STAGED],
            target=DeploymentState.RISK_APPROVED, actor=actor,
            event_type="deployment_risk_approved")

    def approve_operations(self, manifest_hash: str,
                           actor: str) -> dict[str, Any]:
        return self._transition(
            manifest_hash, allowed=[DeploymentState.RISK_APPROVED],
            target=DeploymentState.OPS_APPROVED, actor=actor,
            event_type="deployment_operations_approved",
            require_distinct_risk_actor=True)

    def activate(self, manifest_hash: str, actor: str) -> dict[str, Any]:
        return self._transition(
            manifest_hash, allowed=[DeploymentState.OPS_APPROVED],
            target=DeploymentState.ACTIVATED, actor=actor,
            event_type="deployment_preflight_passed")

    def arm(self, manifest_hash: str, actor: str) -> dict[str, Any]:
        return self._transition(
            manifest_hash, allowed=[DeploymentState.ACTIVATED],
            target=DeploymentState.ARMED, actor=actor,
            event_type="deployment_armed")

    def mark_running(self, manifest_hash: str, actor: str) -> dict[str, Any]:
        return self._transition(
            manifest_hash, allowed=[DeploymentState.ARMED],
            target=DeploymentState.RUNNING, actor=actor,
            event_type="deployment_running")

    def pause(self, manifest_hash: str, actor: str,
              reason: str) -> dict[str, Any]:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("pause reason is required")
        return self._transition(
            manifest_hash,
            allowed=[DeploymentState.ACTIVATED, DeploymentState.ARMED,
                     DeploymentState.RUNNING],
            target=DeploymentState.PAUSED, actor=actor,
            event_type="deployment_paused", reason=reason)

    def revoke(self, manifest_hash: str, actor: str,
               reason: str) -> dict[str, Any]:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("revocation reason is required")
        return self._transition(
            manifest_hash,
            allowed=[state for state in DeploymentState
                     if state != DeploymentState.REVOKED],
            target=DeploymentState.REVOKED, actor=actor,
            event_type="deployment_revoked", reason=reason)

    def authorize_order(
            self, manifest_hash: str, *, intent_id: str, side: str,
            symbol: str, quantity: int, reference_price: float,
            trading_date: str, opening: bool, actor: str = "deployment_guard"
            ) -> dict[str, Any]:
        """Atomically enforce and reserve manifest order/day limits.

        Exit orders are still universe and state checked but intentionally do
        not consume opening limits: safety controls must never trap a position.
        Repeating one intent id with byte-identical economics is idempotent.
        """
        digest = _hash(manifest_hash, "manifest_hash")
        intent_id = str(intent_id).strip()
        side = str(side).strip().upper()
        symbol = str(symbol).strip().upper()
        if not intent_id or side not in {"BUY", "SELL"} or not symbol:
            raise DeploymentStateError("invalid deployment order identity")
        if isinstance(quantity, bool) or int(quantity) < 1:
            raise DeploymentStateError("deployment quantity must be positive")
        price = _positive(reference_price, "reference_price")
        notional = int(quantity) * price
        now = self._clock().astimezone(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM foundation_deployments WHERE manifest_hash = ?",
                (digest,),
            ).fetchone()
            state = (DeploymentState(row["state"])
                     if row is not None else None)
            if state not in {DeploymentState.ARMED,
                             DeploymentState.RUNNING}:
                raise DeploymentStateError(
                    "deployment is not armed or running")
            manifest = DeploymentManifest.from_json(row["manifest_json"])
            if datetime.fromisoformat(manifest.expires_at) <= self._clock():
                raise DeploymentStateError("deployment manifest has expired")
            if symbol not in set(manifest.allowed_universe):
                raise DeploymentStateError(
                    f"symbol {symbol} is outside the approved universe")
            existing = conn.execute("""
                SELECT * FROM deployment_order_authorizations
                WHERE manifest_hash = ? AND intent_id = ?
            """, (digest, intent_id)).fetchone()
            economics = (str(trading_date), side, symbol, int(quantity),
                         price, notional, int(bool(opening)))
            if existing is not None:
                stored = (existing["trading_date"], existing["side"],
                          existing["symbol"], int(existing["quantity"]),
                          float(existing["reference_price"]),
                          float(existing["notional"]), int(existing["opening"]))
                if stored != economics:
                    raise DeploymentStateError(
                        "intent id was reused with different economics")
                conn.rollback()
                return {"authorized": True, "idempotent": True,
                        "notional": notional}
            if opening:
                if notional > float(manifest.max_order_notional):
                    raise DeploymentStateError(
                        "order exceeds manifest max_order_notional")
                totals = conn.execute("""
                    SELECT COUNT(*) AS n, COALESCE(SUM(notional), 0) AS total
                    FROM deployment_order_authorizations
                    WHERE manifest_hash = ? AND trading_date = ? AND opening = 1
                """, (digest, str(trading_date))).fetchone()
                if int(totals["n"]) >= int(manifest.max_daily_orders):
                    raise DeploymentStateError(
                        "deployment max_daily_orders is exhausted")
                if float(totals["total"]) + notional \
                        > float(manifest.max_daily_notional):
                    raise DeploymentStateError(
                        "deployment max_daily_notional is exhausted")
            conn.execute("""
                INSERT INTO deployment_order_authorizations
                    (manifest_hash, intent_id, trading_date, side, symbol,
                     quantity, reference_price, notional, opening,
                     authorized_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (digest, intent_id, str(trading_date), side, symbol,
                  int(quantity), price, notional, int(bool(opening)), now))
            self.audit._append_in_transaction(
                conn, actor, "deployment_order_authorized", {
                    "manifest_hash": digest,
                    "intent_id": intent_id,
                    "side": side,
                    "symbol": symbol,
                    "quantity": int(quantity),
                    "reference_price": price,
                    "notional": notional,
                    "opening": bool(opening),
                }, timestamp=now)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"authorized": True, "idempotent": False,
                "notional": notional}


__all__ = [
    "DeploymentManifest",
    "DeploymentState",
    "DeploymentStateError",
    "DeploymentStore",
]
