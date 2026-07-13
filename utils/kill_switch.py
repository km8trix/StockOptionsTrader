"""
Global live-trading kill switch (contract C17).

One persistent boolean that gates every new order. State lives in SQLite
(table kill_switch_state in TRADING_DB_PATH) so it survives process
restarts — a switch that forgets it was thrown is worse than none. Every
flip is appended to the audit log with the reason and the actor.

EtradeClient.preview_order/place_order check this FIRST and raise
KillSwitchEngaged when engaged; cancel, order status, and quotes remain
allowed so an operator can always flatten and observe.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Dict, Optional

from utils.audit import AuditLog

logger = logging.getLogger(__name__)


class KillSwitchEngaged(RuntimeError):
    """Raised when a blocked operation is attempted while engaged."""


class KillSwitch:
    """Persistent kill switch; every flip is audit-logged.

    Args:
        db_path: SQLite path; defaults to env TRADING_DB_PATH, falling
            back to ``trading_data.db``.
        audit: AuditLog to record flips into; defaults to one on the same
            database so a KillSwitch is never un-audited.
    """

    def __init__(self, db_path: Optional[str] = None,
                 audit: Optional[AuditLog] = None):
        self.db_path = (db_path or os.environ.get("TRADING_DB_PATH")
                        or "trading_data.db")
        if self.db_path == ":memory:" or self.db_path.startswith("file:"):
            raise ValueError(
                "kill switch requires a filesystem-backed SQLite path; "
                "in-memory and URI databases are not supported")
        self.audit = audit if audit is not None else AuditLog(self.db_path)
        if self._canonical_path(self.audit.db_path) != self._canonical_path(
                self.db_path):
            raise ValueError(
                "kill switch and audit log must use the same database")
        self._init_table()

    @staticmethod
    def _canonical_path(path: str) -> str:
        return os.path.realpath(os.path.abspath(path))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kill_switch_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    engaged INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    actor TEXT,
                    flipped_at TEXT
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO kill_switch_state (id, engaged) "
                "VALUES (1, 0)")
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def engaged(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT engaged FROM kill_switch_state WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        return bool(row["engaged"]) if row is not None else False

    def state(self) -> Dict:
        """Full state for dashboards: engaged/reason/actor/flipped_at."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT engaged, reason, actor, flipped_at "
                "FROM kill_switch_state WHERE id = 1").fetchone()
        finally:
            conn.close()
        if row is None:
            return {"engaged": False, "reason": None, "actor": None,
                    "flipped_at": None}
        return {"engaged": bool(row["engaged"]), "reason": row["reason"],
                "actor": row["actor"], "flipped_at": row["flipped_at"]}

    def engage(self, reason: str, actor: str) -> None:
        """Engage; an already-engaged switch is a no-op (no double audit)."""
        if not self._flip(True, reason, actor, "kill_switch_engaged",
                          {"reason": reason}):
            logger.info("Kill switch already engaged; engage() ignored")
            return
        logger.warning("KILL SWITCH ENGAGED by %s: %s", actor, reason)

    def disengage(self, actor: str) -> None:
        """Disengage; an already-clear switch is a no-op."""
        if not self._flip(False, None, actor, "kill_switch_disengaged", {}):
            logger.info("Kill switch already clear; disengage() ignored")
            return
        logger.warning("Kill switch disengaged by %s", actor)

    def _flip(self, engaged: bool, reason: Optional[str],
              actor: str, event_type: str, payload: Dict) -> bool:
        """Atomically flip and audit; True only when this call changed it.

        The update-where-changed and hash-chain append share one SQLite
        transaction.  An append failure therefore rolls the state back, and
        concurrent callers cannot observe or commit a state without its
        canonical audit event.
        """
        value = 1 if engaged else 0
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE kill_switch_state SET engaged = ?, reason = ?, "
                "actor = ?, flipped_at = NULL "
                "WHERE id = 1 AND engaged != ?",
                (value, reason, actor, value))
            changed = cursor.rowcount > 0
            if changed:
                timestamp = self.audit._clock().isoformat()
                conn.execute(
                    "UPDATE kill_switch_state SET flipped_at = ? "
                    "WHERE id = 1", (timestamp,))
                self.audit._append_in_transaction(
                    conn, actor, event_type, payload, timestamp=timestamp)
            conn.commit()
            return changed
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
