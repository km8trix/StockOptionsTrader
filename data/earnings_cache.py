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


# ----------------------------------------------------------------------
# SEC EDGAR fetcher — free, authoritative, decades deep
# ----------------------------------------------------------------------
# Earnings date := the 8-K Item 2.02 ("Results of Operations") filing date,
# i.e. the actual press release. Reliable from ~Aug 2004, when the SEC
# restructured 8-K items; older filings tag 2.02 inconsistently (a documented
# depth caveat, not a bug). US filers only.
_SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
_SEC_SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'


def _sec_get(url: str, user_agent: str):  # pragma: no cover — network path
    """GET + parse a SEC JSON endpoint, honoring SEC fair-access policy."""
    import time

    import requests

    resp = requests.get(url, timeout=30, headers={
        'User-Agent': user_agent, 'Accept-Encoding': 'gzip, deflate'})
    resp.raise_for_status()
    time.sleep(0.11)  # SEC asks for <10 requests/second
    return resp.json()


def _edgar_cik_map(user_agent: str) -> Dict[str, str]:  # pragma: no cover
    """ticker (upper) -> zero-padded 10-digit CIK, from SEC's master list."""
    data = _sec_get(_SEC_TICKERS_URL, user_agent)
    return {row['ticker'].upper(): f"{int(row['cik_str']):010d}"
            for row in data.values()}


def _edgar_earnings_dates(cik: str, user_agent: str) -> List[date_type]:  # pragma: no cover
    """Every 8-K Item 2.02 filing date for a CIK, across ALL history.

    Walks filings.recent plus the older filings.files batches so depth goes
    back as far as EDGAR has the filer's 8-Ks, not just the recent ~1000.
    """
    main = _sec_get(_SEC_SUBMISSIONS_URL.format(cik=cik), user_agent)
    batches = [main['filings']['recent']]
    for older in main['filings'].get('files', []):
        batches.append(_sec_get(
            f"https://data.sec.gov/submissions/{older['name']}", user_agent))
    dates = []
    for batch in batches:
        forms = batch['form']
        items = batch.get('items') or [''] * len(forms)
        for form, item, filed in zip(forms, items, batch['filingDate']):
            if form == '8-K' and '2.02' in (item or ''):
                dates.append(pd.Timestamp(filed).date())
    return dates


def make_edgar_fetcher(user_agent: Optional[str] = None
                       ) -> Callable[[str], List[date_type]]:
    """Build an EDGAR earnings-date fetcher for EarningsCache.ingest().

    SEC fair-access REQUIRES a User-Agent naming a real contact; pass one or
    set SEC_USER_AGENT (e.g. 'Jane Doe jane@example.com'). The ticker->CIK map
    is fetched ONCE and shared across symbols. Unknown tickers (e.g. foreign
    6-K filers not in EDGAR's equity list) degrade to [].
    """
    ua = user_agent or os.environ.get('SEC_USER_AGENT')
    if not ua:
        raise ValueError(
            "EDGAR needs a contact User-Agent; set SEC_USER_AGENT="
            "'Your Name your@email.com' (SEC fair-access policy).")
    cik_map = _edgar_cik_map(ua)

    def fetch(symbol: str) -> List[date_type]:
        cik = cik_map.get(symbol.upper())
        if cik is None:
            logger.warning("EDGAR: no CIK for %s (US filers only); skipped", symbol)
            return []
        try:
            return sorted(set(_edgar_earnings_dates(cik, ua)))
        except Exception as exc:  # pragma: no cover — degrade per symbol
            logger.warning("EDGAR earnings unavailable for %s: %s", symbol, exc)
            return []

    return fetch


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


if __name__ == '__main__':  # operator CLI (networked, one-time):
    #   SEC_USER_AGENT='You you@email.com' python -m data.earnings_cache
    #   python -m data.earnings_cache --source yfinance AAPL MSFT
    import argparse

    parser = argparse.ArgumentParser(prog='python -m data.earnings_cache')
    parser.add_argument('symbols', nargs='*',
                        help='tickers (default: data.universe.LARGE_CAP_100)')
    parser.add_argument('--source', choices=['edgar', 'yfinance'], default='edgar',
                        help='earnings source (default: edgar — deepest, free; '
                             'needs SEC_USER_AGENT)')
    cli = parser.parse_args()

    syms = cli.symbols
    if not syms:
        from data.universe import LARGE_CAP_100
        syms = LARGE_CAP_100
    logging.basicConfig(level=logging.INFO)
    fetcher = make_edgar_fetcher() if cli.source == 'edgar' else None
    cache = EarningsCache()
    result = cache.ingest(syms, fetcher=fetcher)
    print(f"Ingested earnings ({cli.source}) for {len(result)} symbols "
          f"({sum(result.values())} dates) into {cache.db_path}")
