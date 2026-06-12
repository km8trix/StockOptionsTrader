"""
Append-only audit log with a sha256 hash chain (contract C18).

Every live-trading event (auth lifecycle, kill-switch flips, order
preview/place/cancel, reconciliation, session steps) lands here. The log
is APPEND-ONLY by API: there is no update or delete method, and any
tampering done behind the API's back (raw SQL) is detectable because each
row carries

    hash = sha256(prev_hash + canonical_json(row_core))

where row_core is {'seq','ts','env','actor','event_type','payload'} and
canonical_json is json.dumps(..., sort_keys=True, separators=(',',':')).
The first row chains off GENESIS_HASH. verify_chain() walks the whole
table and reports the first sequence number whose hash no longer matches.

SECRETS NEVER GO IN PAYLOADS: callers (auth manager, client) audit event
metadata only — consumer keys, access tokens, and verifier codes are
stored in their own table / env and are never logged here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: prev_hash of the very first row.
GENESIS_HASH = "0" * 64


def canonical_json(obj) -> str:
    """Deterministic JSON used for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _row_hash(prev_hash: str, row_core: Dict) -> str:
    return hashlib.sha256(
        (prev_hash + canonical_json(row_core)).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only, hash-chained audit log persisted in SQLite.

    Args:
        db_path: SQLite path; defaults to env TRADING_DB_PATH, falling back
            to ``trading_data.db`` (same convention as TradingDatabase).
        env: 'sandbox' | 'production' stamped on every row; defaults to env
            ETRADE_ENV, falling back to 'sandbox'.
        clock: injectable callable returning an aware datetime (tests freeze
            it); defaults to UTC now.
    """

    def __init__(self, db_path: Optional[str] = None,
                 env: Optional[str] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.db_path = (db_path or os.environ.get("TRADING_DB_PATH")
                        or "trading_data.db")
        self.env = env or os.environ.get("ETRADE_ENV", "sandbox")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    seq INTEGER PRIMARY KEY,
                    ts TEXT NOT NULL,
                    env TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Append (the ONLY mutation this class exposes)
    # ------------------------------------------------------------------
    def append(self, actor: str, event_type: str, payload: Dict) -> int:
        """Append one event; returns its sequence number.

        seq is computed inside a BEGIN IMMEDIATE transaction so concurrent
        appenders (gunicorn threads) serialize and the chain never forks.
        """
        payload = dict(payload or {})
        ts = self._clock().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT seq, hash FROM audit_log "
                "ORDER BY seq DESC LIMIT 1").fetchone()
            if row is None:
                seq, prev_hash = 1, GENESIS_HASH
            else:
                seq, prev_hash = row["seq"] + 1, row["hash"]
            row_core = {"seq": seq, "ts": ts, "env": self.env,
                        "actor": actor, "event_type": event_type,
                        "payload": payload}
            conn.execute(
                "INSERT INTO audit_log "
                "(seq, ts, env, actor, event_type, payload, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, ts, self.env, actor, event_type,
                 canonical_json(payload), prev_hash,
                 _row_hash(prev_hash, row_core)))
            conn.commit()
        finally:
            conn.close()
        logger.debug("audit #%d %s/%s", seq, actor, event_type)
        return seq

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def entries(self, limit: int = 100, offset: int = 0,
                event_type: Optional[str] = None,
                ascending: bool = False) -> List[Dict]:
        """Page through entries (newest-first by default, for dashboards).

        Tests asserting chronological ordering pass ascending=True.
        """
        query = ("SELECT seq, ts, env, actor, event_type, payload "
                 "FROM audit_log")
        params: list = []
        if event_type is not None:
            query += " WHERE event_type = ?"
            params.append(event_type)
        query += f" ORDER BY seq {'ASC' if ascending else 'DESC'}"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [{"seq": r["seq"], "ts": r["ts"], "env": r["env"],
                 "actor": r["actor"], "event_type": r["event_type"],
                 "payload": json.loads(r["payload"])} for r in rows]

    def verify_chain(self) -> Dict:
        """Recompute every hash; report the first broken sequence number.

        Detects both payload tampering (row hash mismatch) and splices
        (prev_hash not matching the prior row's hash).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT seq, ts, env, actor, event_type, payload, "
                "prev_hash, hash FROM audit_log ORDER BY seq ASC").fetchall()
        finally:
            conn.close()
        expected_prev = GENESIS_HASH
        for r in rows:
            row_core = {"seq": r["seq"], "ts": r["ts"], "env": r["env"],
                        "actor": r["actor"], "event_type": r["event_type"],
                        "payload": json.loads(r["payload"])}
            if (r["prev_hash"] != expected_prev
                    or _row_hash(r["prev_hash"], row_core) != r["hash"]):
                return {"ok": False, "first_bad_seq": r["seq"]}
            expected_prev = r["hash"]
        return {"ok": True, "first_bad_seq": None}
