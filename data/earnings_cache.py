"""SQLite-backed offline earnings calendar.

Implements the desks ``EarningsCalendar`` protocol
(``next_earnings(symbol, date) -> date | None``) by reading scheduled
earnings dates out of a local SQLite table — NEVER the network. This is
what makes Renaissance's earnings-entry gate safe to enable on the
registry path: ``create_desk`` drives Trading-Floor / fund BACKTESTS that
loop per symbol per day, so the calendar they consume must do zero I/O
beyond the cache.

Two halves, deliberately split:

* ``next_earnings`` / ``store`` — pure SQLite, no imports beyond stdlib +
  pandas. The only path a backtest ever touches. An empty (or absent)
  cache makes ``next_earnings`` return ``None`` for everything, so
  injecting an un-ingested cache is byte-identical to no calendar at all
  (the gate no-ops). [[stockoptionstrader-improvement-roadmap]]
* ``ingest`` — the OPERATOR-ONLY network step. Run once (``python -m
  data.earnings_cache AAPL MSFT ...``) to populate the cache from
  yfinance; backtests then read it offline. yfinance earnings history is
  shallow (~a couple of years past + the next quarter), so deep backtests
  see the gate bite only near their recent end — an honest data limit, not
  a code bug.

The earnings cache lives in its OWN file (default ``earnings_data.db``,
co-located beside ``TRADING_DB_PATH``), deliberately NOT the OHLCV
``trading_data.db``: that keeps an un-ingested cache a *missing file*, so
the registry's no-op read is the fast ``os.path.exists -> None`` path and
never opens — or couples to — the OHLCV db.

CONSTRUCTION IS SIDE-EFFECT-FREE for file paths: building an EarningsCache
neither creates a directory nor a db file, and a read against a missing
file/table just returns None. Only ``store``/``ingest`` (writes) create
the file — so the registry can inject one unconditionally without a stray
db appearing. Schema/style otherwise mirrors data/cache.py (CREATE TABLE
IF NOT EXISTS, one connection per call, dates as zero-padded ISO so
lexicographic order equals chronological order).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date as date_type
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CREATE_TABLE = '''
    CREATE TABLE IF NOT EXISTS earnings_dates (
        symbol TEXT,
        earnings_date TEXT,
        source TEXT,
        fetched_at TEXT,
        PRIMARY KEY(symbol, earnings_date)
    )
'''


def _default_db_path() -> str:
    """Dedicated earnings db path (NOT the OHLCV trading_data.db).

    Honors ``EARNINGS_DB_PATH``; else co-locates ``earnings_data.db`` beside
    ``TRADING_DB_PATH`` (so a Docker /data volume keeps both together); else
    the CWD. Separate file => an un-ingested cache is simply a missing file.
    """
    explicit = os.environ.get('EARNINGS_DB_PATH')
    if explicit:
        return explicit
    trading = os.environ.get('TRADING_DB_PATH')
    if trading and trading != ':memory:':
        return os.path.join(os.path.dirname(trading) or '.', 'earnings_data.db')
    return 'earnings_data.db'


def _yfinance_earnings_dates(symbol: str) -> List[date_type]:
    """Default ingestion fetcher: yfinance earnings dates (NETWORK).

    Lazy-imports yfinance so the module imports cleanly offline and so a
    backtest that never calls ``ingest`` never pulls the wheel in. Any
    failure degrades to ``[]`` with a warning — the operator re-runs.
    """
    try:  # pragma: no cover — network path, exercised only on operator ingest
        import yfinance as yf

        df = yf.Ticker(symbol).get_earnings_dates(limit=40)
        if df is None or df.empty:
            return []
        return sorted({pd.Timestamp(ts).date() for ts in df.index})
    except Exception as exc:  # pragma: no cover — degrade, don't crash ingest
        logger.warning("yfinance earnings unavailable for %s: %s", symbol, exc)
        return []


class EarningsCache:
    """Offline, SQLite-backed earnings calendar (EarningsCalendar protocol).

    Args:
        db_path: SQLite file path. Defaults to :func:`_default_db_path`
            (EARNINGS_DB_PATH, else 'earnings_data.db' beside TRADING_DB_PATH).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        # ':memory:' needs ONE persistent handle (a fresh connect() would be a
        # new empty database every call) and an eager table. File-backed paths
        # are created lazily by store() — construction touches no disk.
        self._mem_conn = None
        if self.db_path == ':memory:':
            self._mem_conn = sqlite3.connect(':memory:')
            self._mem_conn.execute(_CREATE_TABLE)
            self._mem_conn.commit()

    def store(self, symbol: str, dates, source: str = 'yfinance') -> int:
        """Upsert scheduled earnings ``dates`` for ``symbol``; return the count.

        ``dates`` may be strings, datetimes, or pd.Timestamps; each is
        normalized to a padded-ISO date. Unparseable entries are skipped
        loudly. Re-storing a known (symbol, date) just refreshes its
        fetched_at — idempotent. This is the FIRST thing that creates the db
        file for a file-backed cache.
        """
        rows = []
        fetched_at = datetime.now().isoformat()
        for d in dates:
            try:
                ts = pd.Timestamp(d)
            except (TypeError, ValueError):
                ts = pd.NaT
            # NaT is the trap: pd.Timestamp(pd.NaT).date().isoformat() returns
            # the STRING 'NaT' (no exception), which would sort high and later
            # crash fromisoformat. pd.isna catches NaT / None / unparseable.
            if pd.isna(ts):
                logger.warning("EarningsCache.store: bad date %r for %s; skipped",
                               d, symbol)
                continue
            rows.append((symbol, ts.date().isoformat(), source, fetched_at))
        if not rows:
            return 0
        conn = self._writable_conn()
        conn.execute(_CREATE_TABLE)
        conn.executemany('''
            INSERT OR REPLACE INTO earnings_dates
            (symbol, earnings_date, source, fetched_at) VALUES (?, ?, ?, ?)
        ''', rows)
        conn.commit()
        self._close(conn)
        return len(rows)

    def next_earnings(self, symbol: str, date) -> Optional[date_type]:
        """Earliest cached earnings date ON OR AFTER ``date``, else None.

        OFFLINE: a pure SQLite read, never the network — safe inside the
        per-symbol/per-day backtest loop. A missing db file or table is an
        empty cache (None), so an un-ingested cache is a clean no-op without
        creating anything. Mirrors SyntheticEarningsCalendar semantics so a
        cache and an injected dict gate identically.
        """
        try:
            current = pd.Timestamp(date).date().isoformat()
        except (TypeError, ValueError):
            return None
        if self._mem_conn is None and not os.path.exists(self.db_path):
            return None  # un-ingested file cache: no read, no file creation
        conn = self._mem_conn or sqlite3.connect(self.db_path)
        try:
            row = conn.execute('''
                SELECT earnings_date FROM earnings_dates
                WHERE symbol = ? AND earnings_date >= ?
                ORDER BY earnings_date LIMIT 1
            ''', (symbol, current)).fetchone()
        except sqlite3.OperationalError:
            row = None  # file exists but no earnings_dates table yet
        finally:
            self._close(conn)
        if not row:
            return None
        try:  # defensive: never let a corrupt stored value crash a backtest
            return date_type.fromisoformat(row[0])
        except ValueError:
            logger.warning("EarningsCache: corrupt earnings_date %r for %s; ignoring",
                           row[0], symbol)
            return None

    def ingest(self, symbols,
               fetcher: Optional[Callable[[str], List[date_type]]] = None
               ) -> Dict[str, int]:
        """OPERATOR-ONLY: populate the cache from a network source.

        Calls ``fetcher(symbol)`` (default: yfinance) for each symbol and
        stores the returned dates. NEVER call this from a backtest or the
        registry — it does network I/O. Returns {symbol: dates_stored}.
        """
        fetch = fetcher or _yfinance_earnings_dates
        counts: Dict[str, int] = {}
        for symbol in symbols:
            counts[symbol] = self.store(symbol, fetch(symbol))
            logger.info("Ingested %d earnings dates for %s", counts[symbol], symbol)
        return counts

    def _writable_conn(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _close(self, conn: sqlite3.Connection) -> None:
        if conn is not self._mem_conn:
            conn.close()


if __name__ == '__main__':  # operator CLI: python -m data.earnings_cache AAPL ...
    import sys

    syms = sys.argv[1:]
    if not syms:
        from data.universe import LARGE_CAP_100
        syms = LARGE_CAP_100
    logging.basicConfig(level=logging.INFO)
    result = EarningsCache().ingest(syms)
    print(f"Ingested earnings for {len(result)} symbols "
          f"({sum(result.values())} dates) into "
          f"{os.environ.get('TRADING_DB_PATH') or 'trading_data.db'}")
