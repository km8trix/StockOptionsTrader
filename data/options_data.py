"""Options chain data via OpenBB, with same-day SQLite snapshot caching
and an IV history table for the Phase 8 IV-rank scanner.

No network at import time: OpenBB is imported lazily (the _get_openbb
pattern from data/market_data.py), and every provider failure is logged
loudly instead of being swallowed.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Normalized chain columns returned by OptionsDataHandler.fetch_chain.
CHAIN_COLUMNS = [
    'expiration', 'strike', 'option_type', 'bid', 'ask', 'last', 'mid',
    'volume', 'open_interest', 'implied_volatility', 'dte', 'underlying_price',
]

#: Raw fields persisted in the same-day snapshot table (mid and dte are
#: derived at read time, so they are not stored).
_SNAPSHOT_FIELDS = [
    'expiration', 'strike', 'option_type', 'bid', 'ask', 'last',
    'volume', 'open_interest', 'implied_volatility', 'underlying_price',
]


class OptionsDataHandler:
    """Fetches, normalizes, and caches options chains using OpenBB.

    Args:
        db_path: SQLite file path. Defaults to the TRADING_DB_PATH
            environment variable, falling back to 'trading_data.db'.
        now_fn: injectable clock returning a ``datetime`` (defaults to
            ``datetime.now``) so tests control "today" deterministically.
    """

    #: OpenBB providers for options chains, tried in order.
    PROVIDERS = ['cboe', 'tradier', 'intrinio']

    #: hv_20 values above this bound (300% annualized) are a near-certain
    #: percent/decimal confusion (e.g. 18.0 passed instead of 0.18) and are
    #: not stored; see record_iv_snapshot.
    HV_20_SANITY_BOUND = 3.0

    def __init__(self, db_path: Optional[str] = None,
                 now_fn: Optional[Callable[[], datetime]] = None):
        self.db_path = db_path or os.environ.get('TRADING_DB_PATH') or 'trading_data.db'
        self._now_fn = now_fn or datetime.now
        self.providers = list(self.PROVIDERS)
        # Ensure parent directory exists for non-memory databases
        if self.db_path != ':memory:':
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self.init_database()

    def _get_openbb(self):
        """Import OpenBB only when it is needed; initialization can touch user-level files."""
        try:
            from openbb import obb
            return obb
        except Exception as e:
            logger.warning("OpenBB unavailable: %s", e)
            return None

    def init_database(self):
        """Initialize snapshot and IV-history schema (idempotent)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                symbol TEXT,
                snapshot_date TEXT,
                expiration TEXT,
                strike REAL,
                option_type TEXT,
                bid REAL,
                ask REAL,
                last REAL,
                volume REAL,
                open_interest REAL,
                implied_volatility REAL,
                underlying_price REAL,
                provider TEXT,
                fetched_at TEXT,
                PRIMARY KEY(symbol, snapshot_date, expiration, strike, option_type)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iv_history (
                symbol TEXT,
                date TEXT,
                atm_iv REAL,
                hv_20 REAL,
                underlying_price REAL,
                PRIMARY KEY(symbol, date)
            )
        ''')

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Chain fetching
    # ------------------------------------------------------------------

    def fetch_chain(self, symbol: str, min_dte: int = 0,
                    max_dte: Optional[int] = None,
                    moneyness_low: float = 0.8,
                    moneyness_high: float = 1.2) -> pd.DataFrame:
        """Fetch the options chain for symbol, normalized and filtered.

        Returns a DataFrame with CHAIN_COLUMNS. The full (unfiltered)
        normalized chain is snapshotted in SQLite per (symbol, day): a
        second call for the same symbol on the same date is served from
        SQLite without touching any provider. Filters:
            - dte in [min_dte, max_dte] (max_dte=None means unbounded),
            - strike/underlying in [moneyness_low, moneyness_high]
              (skipped, with a warning, if no underlying price is known).
        """
        today = self._now_fn().date()
        snapshot_date = today.isoformat()

        raw = self._load_snapshot(symbol, snapshot_date)
        if raw is None:
            raw, provider = self._fetch_from_providers(symbol)
            if raw is not None and not raw.empty:
                try:
                    self._store_snapshot(symbol, snapshot_date, raw, provider)
                except Exception as e:
                    logger.warning("Failed to snapshot options chain for %s: %s",
                                   symbol, e)
        else:
            logger.info("Served %s options chain from same-day snapshot (%s)",
                        symbol, snapshot_date)

        if raw is None or raw.empty:
            logger.warning("No options chain available for %s", symbol)
            return pd.DataFrame(columns=CHAIN_COLUMNS)

        return self._finalize(raw, today, min_dte, max_dte,
                              moneyness_low, moneyness_high)

    def _fetch_from_providers(
            self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Try each options provider in order; return (raw chain, provider)."""
        obb = self._get_openbb()
        if obb is None:
            return None, None

        for provider in self.providers:
            try:
                result = obb.derivatives.options.chains(
                    symbol=symbol, provider=provider
                )

                if result is None or not hasattr(result, 'results'):
                    logger.warning(
                        "OpenBB options provider %s returned no results for %s",
                        provider, symbol
                    )
                    continue

                records = []
                for item in result.results:
                    records.append({
                        'expiration': getattr(item, 'expiration', None),
                        'strike': getattr(item, 'strike', None),
                        'option_type': getattr(item, 'option_type', None),
                        'bid': getattr(item, 'bid', None),
                        'ask': getattr(item, 'ask', None),
                        'last': self._first_attr(
                            item, ('last', 'last_trade_price', 'close')),
                        'volume': getattr(item, 'volume', None),
                        'open_interest': getattr(item, 'open_interest', None),
                        'implied_volatility': getattr(
                            item, 'implied_volatility', None),
                        'underlying_price': getattr(
                            item, 'underlying_price', None),
                    })

                if records:
                    raw = self._normalize_raw(pd.DataFrame(records), symbol)
                    if not raw.empty:
                        logger.info(
                            "Fetched options chain for %s from OpenBB provider "
                            "%s (%d contracts)", symbol, provider, len(raw)
                        )
                        return raw, provider

                logger.warning(
                    "OpenBB options provider %s returned an empty chain for %s",
                    provider, symbol
                )

            except Exception as e:
                logger.warning(
                    "OpenBB options provider %s failed for %s: %s",
                    provider, symbol, e
                )
                continue

        return None, None

    @staticmethod
    def _first_attr(item, names):
        """First non-None attribute among names, else None (never KeyError)."""
        for name in names:
            value = getattr(item, name, None)
            if value is not None:
                return value
        return None

    def _normalize_raw(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Coerce raw chain fields defensively: missing -> NaN, never KeyError."""
        df = df.copy()
        for col in _SNAPSHOT_FIELDS:
            if col not in df.columns:
                df[col] = np.nan

        df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
        for col in ('strike', 'bid', 'ask', 'last', 'volume', 'open_interest',
                    'implied_volatility', 'underlying_price'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['option_type'] = (
            df['option_type'].astype(str).str.lower()
            .replace({'c': 'call', 'p': 'put'})
        )

        # Contracts without an expiration or strike cannot be filtered or
        # priced; drop them rather than poisoning dte/moneyness math.
        before = len(df)
        df = df[df['expiration'].notna() & df['strike'].notna()]
        dropped = before - len(df)
        if dropped:
            logger.warning(
                "Dropped %d %s contracts missing expiration/strike",
                dropped, symbol
            )
        df = df.copy()
        df['expiration'] = df['expiration'].map(lambda ts: ts.date())

        # A chain shares one underlying; backfill missing per-row values.
        underlying = df['underlying_price'].dropna()
        if not underlying.empty:
            df['underlying_price'] = df['underlying_price'].fillna(
                float(underlying.iloc[0])
            )

        return df[_SNAPSHOT_FIELDS].reset_index(drop=True)

    def _finalize(self, raw: pd.DataFrame, today: date, min_dte: int,
                  max_dte: Optional[int], moneyness_low: float,
                  moneyness_high: float) -> pd.DataFrame:
        """Derive mid/dte and apply dte + moneyness filters."""
        df = raw.copy()

        df['dte'] = df['expiration'].map(lambda d: (d - today).days)
        # Only trust a bid/ask midpoint when BOTH are positive AND the spread
        # is not inverted (bid <= ask). A provider glitch like bid=100/ask=50
        # would otherwise yield a nonsensical mid (75) that silently corrupts
        # moneyness filtering and IV selection; fall back to last in that case.
        inverted = (df['bid'] > 0) & (df['ask'] > 0) & (df['bid'] > df['ask'])
        n_inverted = int(inverted.sum())
        if n_inverted:
            logger.warning(
                "%d contracts have an inverted bid/ask spread (bid > ask); "
                "using last for mid", n_inverted)
        valid_quote = (df['bid'] > 0) & (df['ask'] > 0) & (df['bid'] <= df['ask'])
        df['mid'] = np.where(
            valid_quote,
            (df['bid'] + df['ask']) / 2.0,
            df['last'],
        )
        # A contract with no valid quote AND no last (illiquid/new strikes)
        # leaves mid as NaN. We keep the row (sparse contracts are retained by
        # contract — see _normalize_raw), but a NaN mid that flows silently
        # into moneyness/IV math is exactly the corruption to surface: warn so
        # downstream consumers know to guard rather than discovering it as a
        # mystery NaN.
        n_nan_mid = int(df['mid'].isna().sum())
        if n_nan_mid:
            logger.warning(
                "%d contracts have no usable mid (no valid quote and no last); "
                "mid is NaN — consumers must guard", n_nan_mid)

        df = df[df['dte'] >= min_dte]
        if max_dte is not None:
            df = df[df['dte'] <= max_dte]

        if df['underlying_price'].notna().any():
            moneyness = df['strike'] / df['underlying_price']
            df = df[(moneyness >= moneyness_low) & (moneyness <= moneyness_high)]
        elif not df.empty:
            logger.warning(
                "No underlying price in chain; skipping moneyness filter"
            )

        df = df.copy()
        df['dte'] = df['dte'].astype(int)
        df = df.sort_values(['expiration', 'strike', 'option_type'])
        return df[CHAIN_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Same-day snapshot persistence
    # ------------------------------------------------------------------

    def _store_snapshot(self, symbol: str, snapshot_date: str,
                        raw: pd.DataFrame, provider: Optional[str]) -> None:
        """Persist the full normalized chain for (symbol, snapshot_date)."""
        fetched_at = self._now_fn().isoformat()

        def _num(value):
            return None if pd.isna(value) else float(value)

        rows = []
        for i in range(len(raw)):
            row = raw.iloc[i]
            rows.append((
                symbol,
                snapshot_date,
                row['expiration'].isoformat(),
                _num(row['strike']),
                row['option_type'],
                _num(row['bid']),
                _num(row['ask']),
                _num(row['last']),
                _num(row['volume']),
                _num(row['open_interest']),
                _num(row['implied_volatility']),
                _num(row['underlying_price']),
                provider,
                fetched_at,
            ))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO option_chain_snapshots
            (symbol, snapshot_date, expiration, strike, option_type, bid, ask,
             last, volume, open_interest, implied_volatility, underlying_price,
             provider, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()
        conn.close()

        logger.info("Snapshotted %d %s contracts for %s",
                    len(rows), symbol, snapshot_date)

    def _load_snapshot(self, symbol: str,
                       snapshot_date: str) -> Optional[pd.DataFrame]:
        """Load the full normalized chain for (symbol, snapshot_date), or None."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT expiration, strike, option_type, bid, ask, last, volume,
                   open_interest, implied_volatility, underlying_price
            FROM option_chain_snapshots
            WHERE symbol = ? AND snapshot_date = ?
        ''', (symbol, snapshot_date))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=_SNAPSHOT_FIELDS)
        return self._normalize_raw(df, symbol)

    # ------------------------------------------------------------------
    # IV history (D4)
    # ------------------------------------------------------------------

    def record_iv_snapshot(self, symbol: str, chain_df: pd.DataFrame,
                           hv_20: Optional[float]) -> Optional[float]:
        """Record today's ATM IV for symbol into iv_history.

        ATM IV = mean of the call IV and the put IV at the strike nearest
        the underlying price, taken from the expiration nearest 30 DTE
        within [20, 45] DTE. If no expiration falls in that window, the
        nearest available DTE >= 7 is used and a warning is logged. Rows
        with NaN call/put IV are skipped (logged), nothing is recorded.

        Args:
            hv_20: annualized 20-day historical volatility as a DECIMAL
                (e.g. 0.18 for 18%) — i.e. the output of
                ``MarketDataHandler.calculate_volatility(data, window=20)``
                — in the SAME UNITS as atm_iv, so Phase 8 IV-vs-HV
                comparisons are like-for-like. A finite value above
                HV_20_SANITY_BOUND (3.0 = 300% annualized) is treated as
                percent/decimal confusion: a WARNING is logged and hv_20
                is stored as NULL instead of poisoning the table. None/NaN
                is stored as NULL.

        Returns the recorded ATM IV, or None when nothing was recorded.
        """
        if chain_df is None or chain_df.empty:
            logger.warning("record_iv_snapshot: empty chain for %s; skipping",
                           symbol)
            return None

        underlying = chain_df['underlying_price'].dropna()
        if underlying.empty:
            logger.warning(
                "record_iv_snapshot: no underlying price for %s; skipping",
                symbol
            )
            return None
        underlying_price = float(underlying.iloc[0])

        dtes = sorted(int(d) for d in chain_df['dte'].dropna().unique())
        in_window = [d for d in dtes if 20 <= d <= 45]
        if in_window:
            target_dte = min(in_window, key=lambda d: abs(d - 30))
        else:
            eligible = [d for d in dtes if d >= 7]
            if not eligible:
                logger.warning(
                    "record_iv_snapshot: no expiration with DTE >= 7 for %s; "
                    "skipping", symbol
                )
                return None
            target_dte = min(eligible, key=lambda d: abs(d - 30))
            logger.warning(
                "record_iv_snapshot: no expiration in [20, 45] DTE for %s; "
                "falling back to nearest available DTE %d", symbol, target_dte
            )

        at_expiry = chain_df[chain_df['dte'] == target_dte]
        strikes = at_expiry['strike'].dropna().unique()
        if len(strikes) == 0:
            logger.warning(
                "record_iv_snapshot: no strikes at DTE %d for %s; skipping",
                target_dte, symbol
            )
            return None
        atm_strike = float(min(strikes, key=lambda k: abs(k - underlying_price)))

        at_strike = at_expiry[at_expiry['strike'] == atm_strike]
        call_iv = self._single_iv(at_strike, 'call')
        put_iv = self._single_iv(at_strike, 'put')
        if np.isnan(call_iv) or np.isnan(put_iv):
            logger.warning(
                "record_iv_snapshot: NaN IV at ATM strike %.2f (DTE %d) for "
                "%s; skipping", atm_strike, target_dte, symbol
            )
            return None

        atm_iv = (call_iv + put_iv) / 2.0
        hv_value = None if hv_20 is None or pd.isna(hv_20) else float(hv_20)
        if hv_value is not None and (
                not np.isfinite(hv_value)
                or hv_value > self.HV_20_SANITY_BOUND):
            logger.warning(
                "record_iv_snapshot: hv_20=%r for %s fails the sanity bound "
                "%.1f (expected an annualized DECIMAL like 0.18, same units "
                "as atm_iv; likely percent/decimal confusion); storing NULL",
                hv_20, symbol, self.HV_20_SANITY_BOUND
            )
            hv_value = None
        date_str = self._now_fn().date().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO iv_history
            (symbol, date, atm_iv, hv_20, underlying_price)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, date_str, atm_iv, hv_value, underlying_price))
        conn.commit()
        conn.close()

        logger.info("Recorded ATM IV %.4f for %s on %s (DTE %d, strike %.2f)",
                    atm_iv, symbol, date_str, target_dte, atm_strike)
        return atm_iv

    @staticmethod
    def _single_iv(at_strike: pd.DataFrame, option_type: str) -> float:
        """IV of the given option type at one strike/expiry; NaN if absent."""
        values = at_strike.loc[
            at_strike['option_type'] == option_type, 'implied_volatility'
        ].dropna()
        if values.empty:
            return float('nan')
        return float(values.iloc[0])

    def get_iv_history(self, symbol: str, start: Optional[str] = None,
                       end: Optional[str] = None) -> pd.DataFrame:
        """IV history for symbol as a DataFrame indexed by date.

        Columns: atm_iv, hv_20, underlying_price. start/end are inclusive
        'YYYY-MM-DD' bounds; None means unbounded.
        """
        query = ('SELECT date, atm_iv, hv_20, underlying_price FROM iv_history '
                 'WHERE symbol = ?')
        params: list = [symbol]
        if start is not None:
            query += ' AND date >= ?'
            params.append(start)
        if end is not None:
            query += ' AND date <= ?'
            params.append(end)
        query += ' ORDER BY date'

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(rows, columns=['date', 'atm_iv', 'hv_20',
                                         'underlying_price'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
