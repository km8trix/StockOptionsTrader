"""Phase 4 ops: SQLite connections run in WAL with a busy timeout so the
threaded single-worker app does not hit 'database is locked'."""

from utils.audit import AuditLog
from utils.database import TradingDatabase


def _assert_hardened(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_trading_database_connection_is_hardened(tmp_path):
    db = TradingDatabase(db_path=str(tmp_path / "t.db"))
    conn = db._connect()
    try:
        _assert_hardened(conn)
    finally:
        conn.close()


def test_audit_log_connection_is_hardened(tmp_path):
    audit = AuditLog(db_path=str(tmp_path / "a.db"))
    conn = audit._connect()
    try:
        _assert_hardened(conn)
    finally:
        conn.close()
