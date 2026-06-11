"""SQLite-backed OHLCV cache.

Persists daily bars plus fetch-coverage records so repeated backtests stop
refetching the same history from network providers. Schema style follows
utils/database.py (CREATE TABLE IF NOT EXISTS, one connection per call).

STALENESS POLICY (implemented in :meth:`OHLCVCache.get`): a coverage row
qualifies for a request (symbol, start, end) iff

    cov_start <= start AND cov_end >= end
    AND (
        requested end_date <= fetched_at_date - 2 calendar days
            -- fully historical data is immutable: never expires
        OR fetched_at is within CACHE_MAX_AGE_HOURS (default 12) of now
            -- recent data may gain or revise bars: expires
    )

If no coverage row qualifies, ``get`` returns ``None`` and the caller is
expected to refetch from a provider.

DATE CANONICALIZATION: stored dates are zero-padded ISO ('YYYY-MM-DD'), so
SQL string comparison equals chronological comparison ONLY for equally
padded input. Every public boundary (:meth:`get`, :meth:`store`) therefore
parses caller-supplied dates and re-emits them via ``date.isoformat()``
before any SQL comparison; raw caller strings are never compared against
stored dates. Unparseable dates are a cache miss in ``get`` (triggering a
provider refetch) and are logged loudly in ``store``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


def _canonical_date(value: str) -> Optional[date]:
    """Parse a caller-supplied 'YYYY-MM-DD' string; None if unparseable.

    ``strptime`` is lenient about zero padding ('2023-1-2' parses), but
    such strings compare WRONG lexicographically against stored padded ISO
    dates ('2023-01-02' <= '2023-1-2' is true as strings). Callers must
    re-emit the parsed date with ``.isoformat()`` before any SQL
    comparison so lexicographic order equals chronological order.
    """
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except (TypeError, ValueError):
        return None


class OHLCVCache:
    """SQLite cache of daily OHLCV bars with coverage-based staleness.

    Args:
        db_path: SQLite file path. Defaults to the TRADING_DB_PATH
            environment variable, falling back to 'trading_data.db'.
        now_fn: injectable clock returning a ``datetime`` (defaults to
            ``datetime.now``) so tests control time deterministically.
    """

    #: Maximum age (hours) before cached data covering recent dates expires.
    CACHE_MAX_AGE_HOURS = 12

    #: store() warns when the first stored bar is more than this many
    #: business days after the requested start (likely provider truncation).
    TRUNCATION_WARN_BDAYS = 5

    def __init__(self, db_path: Optional[str] = None,
                 now_fn: Optional[Callable[[], datetime]] = None):
        self.db_path = db_path or os.environ.get('TRADING_DB_PATH') or 'trading_data.db'
        self._now_fn = now_fn or datetime.now
        # Ensure parent directory exists for non-memory databases
        if self.db_path != ':memory:':
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize cache schema (idempotent)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                provider TEXT,
                fetched_at TEXT,
                PRIMARY KEY(symbol, date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fetch_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                start_date TEXT,
                end_date TEXT,
                provider TEXT,
                fetched_at TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def store(self, symbol: str, df: pd.DataFrame, provider: str,
              start_date: str, end_date: str) -> None:
        """Upsert daily bars and record a coverage row for the fetch.

        ``df`` must be shaped like MarketDataHandler output: DatetimeIndex,
        lowercase ohlcv columns. Empty frames are ignored (no coverage is
        claimed for ranges that produced no rows).

        ``start_date``/``end_date`` are canonicalized to padded ISO before
        the coverage row is written. If either is unparseable, coverage is
        claimed only for the bars actually stored (and logged loudly). If
        the first stored bar is more than TRUNCATION_WARN_BDAYS business
        days after the requested start, a WARNING is logged: the provider
        likely truncated deep history, and caching the requested range
        would otherwise serve the truncated frame forever.
        """
        if df is None or df.empty:
            logger.debug("OHLCVCache.store: empty frame for %s %s..%s; not cached",
                         symbol, start_date, end_date)
            return

        fetched_at = self._now_fn().isoformat()
        index = pd.to_datetime(df.index)
        first_bar = index.min().date()
        last_bar = index.max().date()

        start = _canonical_date(start_date)
        end = _canonical_date(end_date)
        if start is None or end is None:
            logger.warning(
                "OHLCVCache.store: unparseable requested range %r..%r for %s; "
                "claiming coverage only for stored bars %s..%s",
                start_date, end_date, symbol,
                first_bar.isoformat(), last_bar.isoformat(),
            )
            start, end = first_bar, last_bar
        else:
            gap_bdays = len(pd.bdate_range(start, first_bar, inclusive='left'))
            if gap_bdays > self.TRUNCATION_WARN_BDAYS:
                logger.warning(
                    "OHLCVCache.store: provider %s may have truncated history "
                    "for %s: requested start %s but first bar is %s "
                    "(%d business days later); caching as full coverage anyway",
                    provider, symbol, start.isoformat(),
                    first_bar.isoformat(), gap_bdays,
                )

        def _val(row, col):
            if col not in row.index:
                return None
            value = row[col]
            return None if pd.isna(value) else float(value)

        rows = []
        for i, ts in enumerate(index):
            row = df.iloc[i]
            rows.append((
                symbol,
                ts.strftime('%Y-%m-%d'),
                _val(row, 'open'),
                _val(row, 'high'),
                _val(row, 'low'),
                _val(row, 'close'),
                _val(row, 'volume'),
                provider,
                fetched_at,
            ))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO ohlcv_daily
            (symbol, date, open, high, low, close, volume, provider, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        cursor.execute('''
            INSERT INTO fetch_coverage (symbol, start_date, end_date, provider, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, start.isoformat(), end.isoformat(), provider, fetched_at))
        conn.commit()
        conn.close()

        logger.info("Cached %d rows of %s (%s..%s) from provider %s",
                    len(rows), symbol, start.isoformat(), end.isoformat(),
                    provider)

    def get(self, symbol: str, start_date: str,
            end_date: str) -> Optional[pd.DataFrame]:
        """Return cached bars for the range, or None on a cache miss.

        A hit requires a qualifying coverage row per the staleness policy in
        the module docstring. The returned frame matches MarketDataHandler
        output: DatetimeIndex named 'date', lowercase ohlcv columns.

        Dates are canonicalized to padded ISO before any SQL comparison
        (so '2023-1-2' selects the same bars as '2023-01-02'); unparseable
        dates are a clean cache miss, never a silently-empty hit.
        """
        start = _canonical_date(start_date)
        end = _canonical_date(end_date)
        if start is None or end is None:
            logger.warning(
                "OHLCVCache.get: unparseable date range %r..%r for %s; "
                "treating as cache miss", start_date, end_date, symbol)
            return None
        start_iso, end_iso = start.isoformat(), end.isoformat()

        if not self._has_valid_coverage(symbol, start_iso, end_iso):
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT date, open, high, low, close, volume FROM ohlcv_daily
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        ''', (symbol, start_iso, end_iso))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            # Covered, but no trading days in the sub-range (e.g. weekend).
            return pd.DataFrame(
                columns=OHLCV_COLUMNS,
                index=pd.DatetimeIndex([], name='date'),
            ).astype(float)

        df = pd.DataFrame(rows, columns=['date'] + OHLCV_COLUMNS)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # Canonical 'us' unit, matching MarketDataHandler provider output.
        df.index = df.index.as_unit('us')
        return df

    def _has_valid_coverage(self, symbol: str, start_date: str,
                            end_date: str) -> bool:
        """True iff some coverage row spans the range and passes staleness.

        Dates are canonicalized to padded ISO before the SQL comparison so
        the coverage check and the bar slice in :meth:`get` can never
        disagree about the same request; unparseable input never qualifies.
        """
        start = _canonical_date(start_date)
        requested_end = _canonical_date(end_date)
        if start is None or requested_end is None:
            logger.warning(
                "OHLCVCache: unparseable date range %r..%r for %s; "
                "coverage does not qualify", start_date, end_date, symbol)
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT end_date, fetched_at FROM fetch_coverage
            WHERE symbol = ? AND start_date <= ? AND end_date >= ?
            ORDER BY fetched_at DESC
        ''', (symbol, start.isoformat(), requested_end.isoformat()))
        candidates = cursor.fetchall()
        conn.close()

        if not candidates:
            return False

        now = self._now_fn()
        for _cov_end, fetched_at_str in candidates:
            try:
                fetched_at = datetime.fromisoformat(fetched_at_str)
            except (TypeError, ValueError):
                continue
            # Fully historical data is immutable: a range ending 2+ calendar
            # days before the fetch never goes stale.
            historical = requested_end <= fetched_at.date() - timedelta(days=2)
            fresh = (now - fetched_at) <= timedelta(hours=self.CACHE_MAX_AGE_HOURS)
            if historical or fresh:
                return True

        logger.info("OHLCVCache: coverage for %s %s..%s exists but is stale; refetch",
                    symbol, start_date, end_date)
        return False
