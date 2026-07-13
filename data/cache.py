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
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


class _USMarketHolidayCalendar(AbstractHolidayCalendar):
    """Regular full-day US equity-market holidays used at fetch boundaries.

    This deliberately models only whether a requested boundary could contain
    a daily bar. Early closes remain sessions. One-off exchange closures are
    uncommon and are conservatively treated as missing provider coverage.
    """

    rules = [
        Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            'Juneteenth', month=6, day=19,
            start_date=pd.Timestamp('2022-01-01'), observance=nearest_workday,
        ),
        Holiday('Independence Day', month=7, day=4,
                observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
    ]


def _market_sessions(start: date, end: date) -> pd.DatetimeIndex:
    if start > end:
        return pd.DatetimeIndex([])
    weekdays = pd.bdate_range(start, end).normalize()
    holidays = _USMarketHolidayCalendar().holidays(start=start, end=end)
    return weekdays.difference(holidays.normalize())


@dataclass(frozen=True)
class CoverageQuality:
    """Observed data and the conservative range one fetch may claim."""

    requested_start: Optional[date]
    requested_end: Optional[date]
    observed_start: date
    observed_end: date
    covered_start: date
    covered_end: date
    request_complete: bool
    missing_start_sessions: int
    missing_end_sessions: int
    missing_interior_sessions: int
    cache_eligible: bool
    reason: str


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
                fetched_at TEXT,
                requested_start_date TEXT,
                requested_end_date TEXT,
                observed_start_date TEXT,
                observed_end_date TEXT,
                request_complete INTEGER NOT NULL DEFAULT 0,
                quality_reason TEXT,
                missing_start_sessions INTEGER NOT NULL DEFAULT 0,
                missing_end_sessions INTEGER NOT NULL DEFAULT 0,
                missing_interior_sessions INTEGER NOT NULL DEFAULT 0,
                cache_eligible INTEGER NOT NULL DEFAULT 0,
                quality_version INTEGER NOT NULL DEFAULT 0
            )
        ''')

        # Existing databases predate observed/complete quality metadata. Their
        # old rows remain at quality_version=0 and intentionally miss once,
        # because their requested range may have been provider-truncated.
        existing_columns = {
            row[1] for row in cursor.execute('PRAGMA table_info(fetch_coverage)')
        }
        additions = {
            'requested_start_date': 'TEXT',
            'requested_end_date': 'TEXT',
            'observed_start_date': 'TEXT',
            'observed_end_date': 'TEXT',
            'request_complete': 'INTEGER NOT NULL DEFAULT 0',
            'quality_reason': 'TEXT',
            'missing_start_sessions': 'INTEGER NOT NULL DEFAULT 0',
            'missing_end_sessions': 'INTEGER NOT NULL DEFAULT 0',
            'missing_interior_sessions': 'INTEGER NOT NULL DEFAULT 0',
            'cache_eligible': 'INTEGER NOT NULL DEFAULT 0',
            'quality_version': 'INTEGER NOT NULL DEFAULT 0',
        }
        for column, declaration in additions.items():
            if column not in existing_columns:
                cursor.execute(
                    f'ALTER TABLE fetch_coverage ADD COLUMN {column} {declaration}'
                )

        conn.commit()
        conn.close()

    @staticmethod
    def assess_coverage(df: pd.DataFrame, start_date: str,
                        end_date: str) -> Optional[CoverageQuality]:
        """Assess requested versus observed coverage without writing data.

        Weekend and regular US market-holiday boundaries do not make a fetch
        incomplete. Missing expected sessions at either edge do. Coverage is
        then narrowed to the observed edge so the original request refetches,
        while a legitimately narrower request can reuse the stored bars.
        """
        if df is None or df.empty:
            return None
        index = pd.to_datetime(df.index, errors='coerce')
        valid_index = index[~index.isna()]
        if len(valid_index) == 0:
            return None
        observed_start = valid_index.min().date()
        observed_end = valid_index.max().date()
        requested_start = _canonical_date(start_date)
        requested_end = _canonical_date(end_date)
        if requested_start is None or requested_end is None \
                or requested_start > requested_end:
            return CoverageQuality(
                requested_start=requested_start,
                requested_end=requested_end,
                observed_start=observed_start,
                observed_end=observed_end,
                covered_start=observed_start,
                covered_end=observed_end,
                request_complete=False,
                missing_start_sessions=0,
                missing_end_sessions=0,
                missing_interior_sessions=0,
                cache_eligible=True,
                reason='invalid_request_observed_only',
            )

        sessions = _market_sessions(requested_start, requested_end)
        expected_dates = [ts.date() for ts in sessions]
        observed_dates = {
            ts.date() for ts in valid_index
            if requested_start <= ts.date() <= requested_end
        }
        matched_indices = [
            index for index, session in enumerate(expected_dates)
            if session in observed_dates
        ]
        if not expected_dates:
            missing_start = missing_end = missing_interior = 0
            covered_start, covered_end = requested_start, requested_end
            cache_eligible = True
        elif not matched_indices:
            missing_start = len(expected_dates)
            missing_end = missing_interior = 0
            covered_start, covered_end = observed_start, observed_end
            cache_eligible = False
        else:
            first_match, last_match = matched_indices[0], matched_indices[-1]
            missing_start = first_match
            missing_end = len(expected_dates) - last_match - 1
            missing_interior = sum(
                session not in observed_dates
                for session in expected_dates[first_match:last_match + 1]
            )
            covered_start = (
                requested_start if not missing_start
                else expected_dates[first_match]
            )
            covered_end = (
                requested_end if not missing_end
                else expected_dates[last_match]
            )
            # A single interval cannot truthfully represent an internal hole.
            # Store the quality record and bars, but expose no cache coverage.
            cache_eligible = missing_interior == 0
        complete = not (missing_start or missing_end or missing_interior)
        reasons = []
        if missing_start:
            reasons.append('missing_start')
        if missing_end:
            reasons.append('missing_end')
        if missing_interior:
            reasons.append('missing_interior')
        return CoverageQuality(
            requested_start=requested_start,
            requested_end=requested_end,
            observed_start=observed_start,
            observed_end=observed_end,
            covered_start=covered_start,
            covered_end=covered_end,
            request_complete=complete,
            missing_start_sessions=missing_start,
            missing_end_sessions=missing_end,
            missing_interior_sessions=missing_interior,
            cache_eligible=cache_eligible,
            reason='+'.join(reasons) if reasons else 'complete',
        )

    def store(self, symbol: str, df: pd.DataFrame, provider: str,
              start_date: str, end_date: str) -> Optional[CoverageQuality]:
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
            return None

        fetched_at = self._now_fn().isoformat()
        index = pd.to_datetime(df.index)
        quality = self.assess_coverage(df, start_date, end_date)
        if quality is None:
            logger.warning(
                "OHLCVCache.store: frame for %s has no valid dated bars; not cached",
                symbol,
            )
            return None

        start = quality.requested_start
        end = quality.requested_end
        if start is None or end is None or start > end:
            logger.warning(
                "OHLCVCache.store: unparseable requested range %r..%r for %s; "
                "claiming coverage only for stored bars %s..%s",
                start_date, end_date, symbol,
                quality.observed_start.isoformat(),
                quality.observed_end.isoformat(),
            )
        else:
            # Keep the long-standing diagnostic threshold in raw weekdays;
            # completeness itself uses the more accurate market calendar.
            gap_bdays = len(pd.bdate_range(
                start, quality.observed_start, inclusive='left'
            ))
            if gap_bdays > self.TRUNCATION_WARN_BDAYS:
                logger.warning(
                    "OHLCVCache.store: provider %s truncated history "
                    "for %s: requested start %s but first bar is %s "
                    "(%d market sessions later); full coverage is not claimed",
                    provider, symbol, start.isoformat(),
                    quality.observed_start.isoformat(), gap_bdays,
                )
            elif not quality.request_complete:
                logger.warning(
                    "OHLCVCache.store: incomplete provider response for %s from %s "
                    "(%s); requested %s..%s, observed %s..%s, claiming only %s..%s",
                    symbol, provider, quality.reason,
                    start.isoformat(), end.isoformat(),
                    quality.observed_start.isoformat(),
                    quality.observed_end.isoformat(),
                    quality.covered_start.isoformat(),
                    quality.covered_end.isoformat(),
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
            INSERT INTO fetch_coverage
            (symbol, start_date, end_date, provider, fetched_at,
             requested_start_date, requested_end_date,
             observed_start_date, observed_end_date, request_complete,
             quality_reason, missing_start_sessions, missing_end_sessions,
             missing_interior_sessions, cache_eligible, quality_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', (
            symbol,
            quality.covered_start.isoformat(),
            quality.covered_end.isoformat(),
            provider,
            fetched_at,
            start.isoformat() if start is not None else None,
            end.isoformat() if end is not None else None,
            quality.observed_start.isoformat(),
            quality.observed_end.isoformat(),
            int(quality.request_complete),
            quality.reason,
            quality.missing_start_sessions,
            quality.missing_end_sessions,
            quality.missing_interior_sessions,
            int(quality.cache_eligible),
        ))
        conn.commit()
        conn.close()

        logger.info(
            "Cached %d rows of %s; requested %s..%s, claimed %s..%s "
            "from provider %s (complete=%s)",
            len(rows), symbol, start_date, end_date,
            quality.covered_start.isoformat(), quality.covered_end.isoformat(),
            provider, quality.request_complete,
        )
        return quality

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
              AND quality_version = 1
              AND cache_eligible = 1
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
