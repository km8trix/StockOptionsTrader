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
        self.audit = audit if audit is not None else AuditLog(self.db_path)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
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
        if not self._flip(True, reason, actor):
            logger.info("Kill switch already engaged; engage() ignored")
            return
        logger.warning("KILL SWITCH ENGAGED by %s: %s", actor, reason)
        self.audit.append(actor, "kill_switch_engaged", {"reason": reason})

    def disengage(self, actor: str) -> None:
        """Disengage; an already-clear switch is a no-op."""
        if not self._flip(False, None, actor):
            logger.info("Kill switch already clear; disengage() ignored")
            return
        logger.warning("Kill switch disengaged by %s", actor)
        self.audit.append(actor, "kill_switch_disengaged", {})

    def _flip(self, engaged: bool, reason: Optional[str],
              actor: str) -> bool:
        """Atomically flip the state; True only when THIS call changed it.

        The no-op check and the write are ONE update-where-changed inside
        BEGIN IMMEDIATE — never a read on one connection followed by a
        write on another. Under multi-worker concurrency (gunicorn) two
        simultaneous engage() calls therefore resolve to exactly one
        winner (rowcount 1) and one no-op (rowcount 0), so flips are
        audited exactly once and engage/disengage cannot interleave into
        duplicate audit rows.
        """
        value = 1 if engaged else 0
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE kill_switch_state SET engaged = ?, reason = ?, "
                "actor = ?, flipped_at = datetime('now') "
                "WHERE id = 1 AND engaged != ?",
                (value, reason, actor, value))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
