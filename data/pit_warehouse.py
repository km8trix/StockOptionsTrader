"""Point-in-time Sharadar warehouse: DuckDB SQL over local Parquet.

Same PIT/survivorship-free read interface as ``data/pit_provider.PitCache``
(``universe_asof`` / ``fundamentals_asof`` / ``prices`` / ``insider_net_buys`` /
``institutional_asof`` / ``daily_metric``), but backed by columnar Parquet read
through DuckDB instead of row-store SQLite. The reason for the switch is SCALE:
the insider edge screen needs the FULL small/mid-cap universe (~10x the 250-name
subset), which is millions of SEP/SF2 rows — bulk-export → Parquet → DuckDB
ingests and scans that in one pass, where per-ticker SQLite ingest does not.

Why this is honest the same way PitCache is:
  * SURVIVORSHIP-FREE — ``universe_asof`` keeps every name live on the date,
    delisted ones included (firstpricedate <= date <= lastpricedate).
  * POINT-IN-TIME fundamentals — gated on ``datekey`` (SEC filing date), ARQ.
  * POINT-IN-TIME insiders — ``filingdate <= asof``; open-market P/S netted with
    the sign convention fixed in PR #67 (abs(magnitude) + direction-from-code).
  * POINT-IN-TIME institutional — 13F lagged ~45 days (visible at
    calendardate + lag), expressed as ``calendardate <= asof - lag``.

CONSTRUCTION IS SIDE-EFFECT-FREE (like PitCache): building a PitWarehouse creates
no directory; a read against a missing Parquet returns empty. Only ``ingest``
(operator-only, network) writes Parquet. The read path imports only duckdb +
pandas + stdlib — no requests, no sqlite.

Dep: ``duckdb`` (reads/writes Parquet natively — no pyarrow). Ingest also needs
``requests`` (imported lazily) and reuses ``PitCache._get_json`` retry/backoff.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import os
from typing import Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# logical name -> (Sharadar datatable, default export params). TICKERS is scoped
# to the SF1 universe (matches PitCache.ingest_tickers); SF1 to as-reported ARQ.
_TABLES = {
    'tickers': ('SHARADAR/TICKERS', {'table': 'SF1'}),
    'sep': ('SHARADAR/SEP', {}),
    'sf1': ('SHARADAR/SF1', {'dimension': 'ARQ'}),
    'sf2': ('SHARADAR/SF2', {}),
    'sf3': ('SHARADAR/SF3', {}),
    'daily': ('SHARADAR/DAILY', {}),
    'actions': ('SHARADAR/ACTIONS', {}),
    'events': ('SHARADAR/EVENTS', {}),
}

#: Intraday minute-bar tables — SIBLING registry to ``_TABLES``, deliberately
#: separate: these come from REST-paged minute-bar APIs (operator-run
#: ``scripts/ingest_alpaca_bars.py``), NOT the Sharadar bulk-export machinery,
#: so ``ingest_table`` must keep rejecting them. Layout differs too: one
#: Parquet PER SYMBOL under ``<warehouse>/<table>/<TICKER>.parquet`` so a
#: re-ingest of one name never rewrites the others. Same (source, params)
#: value shape as ``_TABLES`` for auditability. ``bars_1m`` is the Alpaca
#: IEX pilot table; ``bars_1m_sip`` is the Massive (ex-Polygon)
#: consolidated-tape sibling (``--provider massive --table bars_1m_sip``) —
#: same schema, kept separate so neither ingest overwrites the other.
_INTRADAY_TABLES = {
    'bars_1m': ('ALPACA/v2/stocks/{symbol}/bars',
                {'timeframe': '1Min', 'feed': 'iex'}),
    'bars_1m_sip': ('MASSIVE/v2/aggs/ticker/{symbol}/range/1/minute',
                    {'adjusted': 'false', 'limit': 50000}),
}

#: bars_1m column contract (writer validates, readers return the non-key
#: columns). ``ts`` is tz-NAIVE US/Eastern wall-clock — Alpaca serves UTC and
#: the ingest converts before writing; every reader/consumer assumes ET-naive.
_BARS_1M_COLUMNS = ('ticker', 'ts', 'open', 'high', 'low', 'close',
                    'volume', 'trade_count', 'vwap')

#: Option end-of-day bar tables — a THIRD sibling registry (added 2026-07-10
#: for the VRP existence screen), deliberately separate from both ``_TABLES``
#: (Sharadar bulk export; ``ingest_table`` keeps rejecting this name) and
#: ``_INTRADAY_TABLES`` (whose pinned contents stay byte-identical). Source is
#: the Massive (ex-Polygon) free tier: one reference-contracts call per
#: monthly selection date plus one per-contract daily-aggregates call
#: (operator-run ``scripts/ingest_massive_options.py``). Layout: one Parquet
#: PER UNDERLYING under ``<warehouse>/option_bars_eod/<UNDERLYING>.parquet``.
#: Same (source, params) value shape as the other registries for
#: auditability.
_OPTION_TABLES = {
    'option_bars_eod': ('MASSIVE/v3/reference/options/contracts '
                        '+ v2/aggs/ticker/{contract}/range/1/day',
                        {'adjusted': 'false', 'limit': 50000}),
}

#: option_bars_eod column contract (writer validates; reader returns all).
#: ``ts`` is the tz-naive ET SESSION DATE (daily bars, normalized at ingest);
#: ``expiry``/``selection_date`` are dates too. ``close`` is the day's last
#: TRADE print — the free tier carries NO quotes/NBBO, so close stands in for
#: mid everywhere downstream (documented + haircut in scripts/vrp_screen.py).
_OPTION_BARS_EOD_COLUMNS = ('underlying', 'contract', 'type', 'strike',
                            'expiry', 'selection_date', 'ts', 'open', 'high',
                            'low', 'close', 'volume')

#: SHARADAR/EVENTS code for an 8-K Item 2.02 "Results of Operations" filing —
#: the earnings press release. Announcement dates lead the SF1 10-Q/10-K
#: datekey by days (large caps) to weeks (small caps).
EARNINGS_EVENT_CODE = '22'

_PRICE_FIELDS = ('closeadj', 'close', 'open', 'high', 'low', 'volume')
_DAILY_FIELDS = ('marketcap', 'pe', 'pb', 'ps', 'ev', 'evebit', 'evebitda')


def _default_warehouse_dir() -> str:
    """Honor ``PIT_WAREHOUSE_DIR``; else co-locate ``pit_warehouse/`` beside
    ``TRADING_DB_PATH``; else the CWD."""
    explicit = os.environ.get('PIT_WAREHOUSE_DIR')
    if explicit:
        return explicit
    trading = os.environ.get('TRADING_DB_PATH')
    if trading and trading != ':memory:':
        return os.path.join(os.path.dirname(trading) or '.', 'pit_warehouse')
    return 'pit_warehouse'


class PitWarehouse:
    """Offline DuckDB-over-Parquet Sharadar warehouse (see module docstring)."""

    def __init__(self, warehouse_dir: Optional[str] = None):
        self.warehouse_dir = warehouse_dir or _default_warehouse_dir()
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pq(self, table: str) -> str:
        return os.path.join(self.warehouse_dir, f"{table}.parquet")

    def _have(self, table: str) -> bool:
        return os.path.exists(self._pq(table))

    def _con(self) -> duckdb.DuckDBPyConnection:
        # One reused in-memory connection; read_parquet re-reads the file each
        # query so newly-ingested Parquet is picked up without reconnecting.
        # ponytail: inline read_parquet, fine for the screen's per-ticker scans;
        # CREATE VIEW per table if repeated full scans ever dominate.
        if self._conn is None:
            self._conn = duckdb.connect(database=':memory:')
            # Cache Parquet metadata across queries — without this every
            # read_parquet re-parses the file's footer, which dominates
            # per-name point queries in the full-universe screens.
            self._conn.execute("SET enable_object_cache=true")
        return self._conn

    def _query(self, table: str, sql: str, params: list):
        """Run ``sql`` (which must reference ``src``) against a table's Parquet,
        or return None if the table was never ingested / the query errors."""
        if not self._have(table):
            return None
        full = sql.replace('src', f"read_parquet('{self._pq(table)}')")
        try:
            return self._con().execute(full, params)
        except duckdb.Error:
            logger.warning("warehouse query failed on %s", table, exc_info=True)
            return None

    def snapshot_version(
            self, tables: Optional[Sequence[str]] = None) -> Dict:
        """Content-address the exact local Parquet source snapshot.

        Hashing is explicit (never on the hot fetch path) and streams files so
        promotion artifacts can reference immutable source bytes rather than a
        mutable directory name or timestamp.
        """
        selected = sorted(set(tables or _TABLES))
        unknown = set(selected) - set(_TABLES)
        if unknown:
            raise ValueError(f"unknown warehouse tables: {sorted(unknown)}")
        digest = hashlib.sha256()
        manifest = []
        missing = []
        for table in selected:
            path = self._pq(table)
            if not os.path.isfile(path):
                missing.append(table)
                continue
            file_digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    file_digest.update(chunk)
            item = {
                'table': table,
                'sha256': file_digest.hexdigest(),
                'bytes': os.path.getsize(path),
            }
            manifest.append(item)
            digest.update(
                f"{table}:{item['sha256']}:{item['bytes']}\n".encode())
        return {
            'version': digest.hexdigest(),
            'tables': manifest,
            'complete': not missing,
            'quality_flags': ([f"missing_table:{table}" for table in missing]
                              if missing else []),
        }

    @staticmethod
    def _date(d):
        try:
            ts = pd.Timestamp(d)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(ts) else ts.date()

    # ------------------------------------------------------------------
    # Reads — offline, point-in-time (mirror PitCache semantics)
    # ------------------------------------------------------------------
    def universe_asof(self, date, *,
                      category: Optional[str] = 'Domestic Common Stock',
                      scalemarketcap: Optional[Sequence[str]] = None
                      ) -> List[str]:
        """Tickers LIVE on ``date`` (firstpricedate <= date <= lastpricedate),
        survivorship-free — delisted names kept for their live span."""
        d = self._date(date)
        if d is None:
            return []
        sql = ("SELECT ticker FROM src WHERE firstpricedate IS NOT NULL "
               "AND CAST(firstpricedate AS DATE) <= ? AND "
               "(CAST(lastpricedate AS DATE) >= ? OR lastpricedate IS NULL)")
        args: list = [d, d]
        if category is not None:
            sql += " AND category = ?"
            args.append(category)
        if scalemarketcap:
            sql += " AND scalemarketcap IN (%s)" % ",".join("?" * len(scalemarketcap))
            args.extend(scalemarketcap)
        res = self._query('tickers', sql + " ORDER BY ticker", args)
        return [r[0] for r in res.fetchall()] if res else []

    def delisting_date(self, ticker: str):
        """Return a ticker's final listed price date, or ``None`` when live.

        Sharadar leaves ``lastpricedate`` null for currently listed securities
        and records the final tradable session for delisted names.  Exposing the
        date through the feed lets the backtest engine liquidate a held delisted
        security at its final observable close instead of carrying a stale mark
        forever.
        """
        res = self._query(
            'tickers',
            "SELECT CAST(lastpricedate AS DATE) FROM src "
            "WHERE ticker = ? LIMIT 1",
            [ticker],
        )
        if not res:
            return None
        row = res.fetchone()
        return self._date(row[0]) if row and row[0] is not None else None

    def delisting_action(self, ticker: str) -> Optional[Dict]:
        """Best available corporate action around the final listed session.

        ACTIONS ``value`` is treated as a per-share cash term only for an
        acquisition/merger/tender event. Bankruptcy/liquidation without a
        positive value is an explicit zero recovery. An ordinary delisting
        with no economic term returns ``None`` so the engine can use the final
        tradable close and flag that fallback instead of inventing a payout.
        """
        final_date = self.delisting_date(ticker)
        if final_date is None:
            return None
        start = (pd.Timestamp(final_date) - pd.Timedelta(days=45)).date()
        end = (pd.Timestamp(final_date) + pd.Timedelta(days=45)).date()
        res = self._query(
            'actions',
            "SELECT CAST(date AS DATE), action, value, contraticker "
            "FROM src WHERE ticker = ? AND CAST(date AS DATE) >= ? "
            "AND CAST(date AS DATE) <= ? ORDER BY CAST(date AS DATE)",
            [ticker, start, end],
        )
        rows = res.fetchall() if res else []
        for action_date, raw_action, raw_value, contra in reversed(rows):
            action = str(raw_action or '').strip().lower()
            value = (float(raw_value)
                     if raw_value is not None and np.isfinite(float(raw_value))
                     else None)
            if any(token in action for token in (
                    'acquisition', 'merger', 'tender')):
                if value is None or value <= 0:
                    continue
                return {
                    'date': self._date(action_date),
                    'action': action,
                    'payout_per_share': value,
                    'contraticker': contra,
                    'quality_flags': ['corporate_action_cash_term'],
                }
            if any(token in action for token in ('bankrupt', 'liquidat')):
                return {
                    'date': self._date(action_date),
                    'action': action,
                    'payout_per_share': max(0.0, value or 0.0),
                    'contraticker': contra,
                    'quality_flags': ['corporate_action_zero_recovery'],
                }
        return None

    def fundamentals_asof(self, ticker: str, date, *,
                          dimension: str = 'ARQ') -> Optional[Dict]:
        """Latest SF1 row KNOWN on ``date`` — datekey <= date (point-in-time).
        Returns the full native record (all SF1 columns) or None."""
        d = self._date(date)
        if d is None:
            return None
        res = self._query(
            'sf1',
            "SELECT * FROM src WHERE ticker = ? AND dimension = ? "
            "AND CAST(datekey AS DATE) <= ? ORDER BY CAST(datekey AS DATE) DESC "
            "LIMIT 1", [ticker, dimension, d])
        if not res:
            return None
        cols = [c[0] for c in res.description]
        row = res.fetchone()
        return dict(zip(cols, row)) if row else None

    def fundamentals_asof_series(self, ticker: str, dates, *,
                                 dimension: str = 'ARQ'
                                 ) -> List[Optional[Dict]]:
        """Batched twin of ``fundamentals_asof``: ONE SF1 scan for the ticker,
        then per-date as-of resolution (latest datekey <= date) in memory.
        Returns a list aligned with ``dates``."""
        ds = [self._date(d) for d in dates]
        valid = [d for d in ds if d is not None]
        if not valid:
            return [None] * len(ds)
        res = self._query(
            'sf1',
            "SELECT * FROM src WHERE ticker = ? AND dimension = ? "
            "AND CAST(datekey AS DATE) <= ? ORDER BY CAST(datekey AS DATE)",
            [ticker, dimension, max(valid)])
        if not res:
            return [None] * len(ds)
        cols = [c[0] for c in res.description]
        rows = res.fetchall()
        if not rows:
            return [None] * len(ds)
        ki = cols.index('datekey')
        keys = [self._date(r[ki]) for r in rows]     # ascending (ORDER BY)
        out: List[Optional[Dict]] = []
        for d in ds:
            if d is None:
                out.append(None)
                continue
            i = bisect.bisect_right(keys, d)
            out.append(dict(zip(cols, rows[i - 1])) if i else None)
        return out

    def fundamentals_quarterly(self, tickers: Optional[Sequence[str]] = None,
                               *, fields: Sequence[str] = ('epsdil',),
                               asof=None) -> pd.DataFrame:
        """Quarterly as-reported SF1 field rows, PIT-shaped.

        One scan of SF1 (dimension=ARQ) returning columns
        ``ticker, reportperiod, datekey`` + ``fields`` — deduped to ONE row
        per (ticker, reportperiod) keeping the EARLIEST datekey (the
        first-known filing; amendments/restatements filed later are dropped,
        the PIT-cleanest choice). Rows where the FIRST field is NULL are
        excluded. Restricted to ``datekey <= asof`` when given. Empty
        DataFrame when sf1 was never ingested.
        """
        fields = list(fields)
        cols = ['ticker', 'reportperiod', 'datekey'] + fields
        field_sql = ", ".join(fields)
        sql = (f"SELECT ticker, CAST(reportperiod AS DATE) AS reportperiod, "
               f"CAST(datekey AS DATE) AS datekey, {field_sql} FROM src "
               f"WHERE dimension = 'ARQ' AND {fields[0]} IS NOT NULL")
        args: list = []
        if tickers is not None:
            if not len(tickers):
                return pd.DataFrame(columns=cols)
            sql += " AND ticker IN (%s)" % ",".join("?" * len(tickers))
            args.extend(tickers)
        if asof is not None:
            d = self._date(asof)
            if d is None:
                return pd.DataFrame(columns=cols)
            sql += " AND CAST(datekey AS DATE) <= ?"
            args.append(d)
        sql += (" QUALIFY row_number() OVER (PARTITION BY ticker, reportperiod"
                " ORDER BY CAST(datekey AS DATE)) = 1"
                " ORDER BY ticker, CAST(reportperiod AS DATE)")
        res = self._query('sf1', sql, args)
        if not res:
            return pd.DataFrame(columns=cols)
        df = res.fetchdf()
        df['reportperiod'] = pd.to_datetime(df['reportperiod'])
        df['datekey'] = pd.to_datetime(df['datekey'])
        for f in fields:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        return df

    def eps_quarterly(self, tickers: Optional[Sequence[str]] = None, *,
                      asof=None) -> pd.DataFrame:
        """Quarterly as-reported diluted EPS rows for SUE-style computations
        (see fundamentals_quarterly — this is the fields=('epsdil',) view)."""
        return self.fundamentals_quarterly(tickers, fields=('epsdil',),
                                           asof=asof)

    def daily_fields_bulk(self, tickers: Sequence[str], date, *,
                          fields: Sequence[str] = ('pb',)
                          ) -> Dict[str, Dict]:
        """Bulk one-scan DAILY metrics for many names on ONE date:
        {ticker: {field: float}}. Names with no row that day are omitted;
        NULL fields come back as None. PIT by nature (DAILY is a daily
        point-in-time table)."""
        d = self._date(date)
        if d is None or not len(tickers):
            return {}
        fields = [f for f in fields if f in _DAILY_FIELDS]
        if not fields:
            return {}
        sql = (f"SELECT ticker, {', '.join(fields)} FROM src "
               f"WHERE CAST(date AS DATE) = ? AND ticker IN "
               f"({','.join('?' * len(tickers))})")
        res = self._query('daily', sql, [d, *tickers])
        if not res:
            return {}
        out: Dict[str, Dict] = {}
        for row in res.fetchall():
            out[row[0]] = {f: (float(v) if v is not None else None)
                           for f, v in zip(fields, row[1:])}
        return out

    def earnings_events(self, tickers: Optional[Sequence[str]] = None
                        ) -> pd.DataFrame:
        """Earnings ANNOUNCEMENT dates (8-K Item 2.02 press releases) from
        SHARADAR/EVENTS: columns ``ticker, date``, one row per announcement,
        sorted by (ticker, date). Empty DataFrame when events was never
        ingested. eventcodes is a pipe-separated code string ('22|91')."""
        cols = ['ticker', 'date']
        sql = ("SELECT ticker, CAST(date AS DATE) AS date FROM src "
               "WHERE list_contains(string_split(eventcodes, '|'), ?)")
        args: list = [EARNINGS_EVENT_CODE]
        if tickers is not None:
            if not len(tickers):
                return pd.DataFrame(columns=cols)
            sql += " AND ticker IN (%s)" % ",".join("?" * len(tickers))
            args.extend(tickers)
        sql += " ORDER BY ticker, CAST(date AS DATE)"
        res = self._query('events', sql, args)
        if not res:
            return pd.DataFrame(columns=cols)
        df = res.fetchdf()
        df['date'] = pd.to_datetime(df['date'])
        return df

    def prices(self, ticker: str, start, end, *,
               field: str = 'closeadj') -> pd.Series:
        """Total-return-adjusted close series (default ``closeadj``), [start, end].
        Empty Series for an un-ingested warehouse / unknown ticker."""
        s, e = self._date(start), self._date(end)
        if s is None or e is None:
            return pd.Series(dtype=float, name=ticker)
        col = field if field in _PRICE_FIELDS else 'closeadj'
        res = self._query(
            'sep',
            f"SELECT CAST(date AS DATE), {col} FROM src WHERE ticker = ? "
            "AND CAST(date AS DATE) >= ? AND CAST(date AS DATE) <= ? "
            "ORDER BY CAST(date AS DATE)", [ticker, s, e])
        rows = res.fetchall() if res else []
        if not rows:
            return pd.Series(dtype=float, name=ticker)
        idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
        return pd.Series([r[1] for r in rows], index=idx, name=ticker)

    def ohlcv(self, ticker: str, start, end) -> pd.DataFrame:
        """Split/dividend-ADJUSTED OHLCV frame (DatetimeIndex, 'us' unit) over
        [start, end] — survivorship-free, so a backtest can hold delisted names.

        Sharadar SEP carries raw open/high/low/close plus the adjusted closeadj
        (total return). We adjust O/H/L by the daily closeadj/close factor and set
        close = closeadj, so every traded price is total-return-consistent (no
        spurious split jumps). Volume is adjusted by the inverse price factor,
        preserving shares-times-price participation across splits. Columns
        match MarketDataHandler.fetch_stock_data
        (lowercase open/high/low/close/volume); empty frame if un-ingested."""
        cols = ['open', 'high', 'low', 'close', 'volume']
        s, e = self._date(start), self._date(end)
        if s is None or e is None:
            return pd.DataFrame(columns=cols)
        res = self._query(
            'sep',
            "SELECT CAST(date AS DATE), open, high, low, close, volume, closeadj "
            "FROM src WHERE ticker = ? AND CAST(date AS DATE) >= ? "
            "AND CAST(date AS DATE) <= ? ORDER BY CAST(date AS DATE)",
            [ticker, s, e])
        rows = res.fetchall() if res else []
        if not rows:
            return pd.DataFrame(columns=cols)
        raw = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close',
                                          'volume', 'closeadj'])
        idx = pd.DatetimeIndex(pd.to_datetime(raw['date'])).as_unit('us')
        # numpy math + explicit arrays: a Series*Series here would align on the
        # default RangeIndex and then mismatch the DatetimeIndex -> all-NaN.
        close = raw['close'].to_numpy(dtype=float)
        closeadj = raw['closeadj'].to_numpy(dtype=float)
        factor = np.where(close > 0, closeadj / np.where(close > 0, close, 1.0), 1.0)
        # REBASE the adjusted series to the raw price at the window start: returns
        # are preserved (a constant rescale), but magnitudes stay realistic. Serial
        # reverse-split names have raw-consistent adjusted prices in the thousands,
        # which the engine's integer-share sizing silently drops to 0 shares.
        scale = (close[0] / closeadj[0]) if closeadj[0] > 0 else 1.0
        adj = factor * scale
        out = pd.DataFrame({
            'open': raw['open'].to_numpy(dtype=float) * adj,
            'high': raw['high'].to_numpy(dtype=float) * adj,
            'low': raw['low'].to_numpy(dtype=float) * adj,
            'close': closeadj * scale,
            'volume': np.divide(
                raw['volume'].to_numpy(dtype=float),
                adj,
                out=np.zeros_like(adj, dtype=float),
                where=adj > 0,
            ),
        }, index=idx)
        out.index.name = 'date'
        return out

    def insider_net_buys(self, ticker: str, asof, *,
                         lookback_days: int = 90) -> Dict:
        """Net OPEN-MARKET insider trades over [asof - lookback, asof], KNOWN by
        ``asof`` (filingdate <= asof). P = buy, S = sell; net = buys - sells.

        Sign convention is the PR #67 fix: Sharadar signs transactionshares
        negative for dispositions but leaves transactionvalue a positive
        magnitude, so take abs of BOTH and apply direction from the code — else
        a seller's already-negative shares get negated again into a buy.
        """
        a = self._date(asof)
        empty = {'net_value': 0.0, 'net_shares': 0.0, 'n_buys': 0,
                 'n_sells': 0, 'window_start': None}
        if a is None:
            return empty
        start = (pd.Timestamp(a) - pd.Timedelta(days=lookback_days)).date()
        res = self._query(
            'sf2',
            "SELECT "
            "COALESCE(SUM(CASE WHEN transactioncode='P' THEN abs(transactionvalue) "
            "  WHEN transactioncode='S' THEN -abs(transactionvalue) END), 0), "
            "COALESCE(SUM(CASE WHEN transactioncode='P' THEN abs(transactionshares) "
            "  WHEN transactioncode='S' THEN -abs(transactionshares) END), 0), "
            "COUNT(*) FILTER (WHERE transactioncode='P'), "
            "COUNT(*) FILTER (WHERE transactioncode='S') "
            "FROM src WHERE ticker = ? AND CAST(filingdate AS DATE) <= ? "
            "AND CAST(filingdate AS DATE) >= ?", [ticker, a, start])
        if not res:
            return empty
        nv, ns, nb, nsell = res.fetchone()
        return {'net_value': float(nv or 0.0), 'net_shares': float(ns or 0.0),
                'n_buys': int(nb or 0), 'n_sells': int(nsell or 0),
                'window_start': start.isoformat()}

    def insider_net_buys_bulk(self, tickers: Sequence[str], asof, *,
                              lookback_days: int = 90) -> Dict[str, Dict]:
        """Bulk twin of ``insider_net_buys`` for MANY tickers at ONE asof:
        one GROUP BY scan instead of a query per name (the insider desk's
        monthly rescore path). Same aggregate expressions as the per-call
        SQL, so values are identical; names with no rows get the same
        zero dict the per-call path returns."""
        a = self._date(asof)
        names = list(tickers)
        if a is None or not names:
            return {}
        start = (pd.Timestamp(a) - pd.Timedelta(days=lookback_days)).date()
        empty = {'net_value': 0.0, 'net_shares': 0.0, 'n_buys': 0,
                 'n_sells': 0, 'window_start': start.isoformat()}
        res = self._query(
            'sf2',
            "SELECT ticker, "
            "COALESCE(SUM(CASE WHEN transactioncode='P' THEN abs(transactionvalue) "
            "  WHEN transactioncode='S' THEN -abs(transactionvalue) END), 0), "
            "COALESCE(SUM(CASE WHEN transactioncode='P' THEN abs(transactionshares) "
            "  WHEN transactioncode='S' THEN -abs(transactionshares) END), 0), "
            "COUNT(*) FILTER (WHERE transactioncode='P'), "
            "COUNT(*) FILTER (WHERE transactioncode='S') "
            "FROM src WHERE CAST(filingdate AS DATE) <= ? "
            "AND CAST(filingdate AS DATE) >= ? AND ticker IN (%s) "
            "GROUP BY ticker" % ",".join("?" * len(names)),
            [a, start, *names])
        rows = res.fetchall() if res else []
        found = {r[0]: {'net_value': float(r[1] or 0.0),
                        'net_shares': float(r[2] or 0.0),
                        'n_buys': int(r[3] or 0), 'n_sells': int(r[4] or 0),
                        'window_start': start.isoformat()} for r in rows}
        return {t: found.get(t, dict(empty)) for t in names}

    def insider_net_buys_series(self, ticker: str, asofs, *,
                                lookback_days: int = 90) -> List[Dict]:
        """Batched twin of ``insider_net_buys``: ONE SF2 scan spanning all
        ``asofs`` windows, aggregated per window in memory (same PR #67 sign
        convention via the shared helper). Aligned with ``asofs``."""
        from data.pit_provider import _aggregate_insider_rows
        empty = {'net_value': 0.0, 'net_shares': 0.0, 'n_buys': 0,
                 'n_sells': 0, 'window_start': None}
        ds = [self._date(a) for a in asofs]
        windows = [
            (a, (pd.Timestamp(a) - pd.Timedelta(days=lookback_days)).date())
            if a is not None else None for a in ds]
        valid = [w for w in windows if w is not None]
        if not valid:
            return [dict(empty) for _ in windows]
        hi = max(a for a, _ in valid)
        lo = min(s for _, s in valid)
        res = self._query(
            'sf2',
            "SELECT CAST(filingdate AS DATE), transactioncode, "
            "transactionshares, transactionvalue FROM src WHERE ticker = ? "
            "AND CAST(filingdate AS DATE) <= ? AND CAST(filingdate AS DATE) >= ?",
            [ticker, hi, lo])
        rows = res.fetchall() if res else []
        out = []
        for w in windows:
            if w is None:
                out.append(dict(empty))
                continue
            a, start = w
            nv, ns, nb, nsell = _aggregate_insider_rows(rows, start, a)
            out.append({'net_value': nv, 'net_shares': ns, 'n_buys': nb,
                        'n_sells': nsell, 'window_start': start.isoformat()})
        return out

    def institutional_asof(self, ticker: str, asof, *,
                           lag_days: int = 45) -> Optional[Dict]:
        """Total 13F holding KNOWN by ``asof``. 13F is filed up to ``lag_days``
        after its calendardate, so a quarter is only public at
        calendardate + lag — equivalently calendardate <= asof - lag (the lag
        guard). Latest such quarter, summed across investors, or None."""
        a = self._date(asof)
        if a is None:
            return None
        cutoff = (pd.Timestamp(a) - pd.Timedelta(days=lag_days)).date()
        res = self._query(
            'sf3',
            "SELECT CAST(calendardate AS DATE), SUM(value), SUM(units), "
            "COUNT(DISTINCT investorname) FROM src WHERE ticker = ? "
            "AND CAST(calendardate AS DATE) <= ? AND CAST(calendardate AS DATE) = "
            "(SELECT MAX(CAST(calendardate AS DATE)) FROM src WHERE ticker = ? "
            "AND CAST(calendardate AS DATE) <= ?) GROUP BY CAST(calendardate AS DATE)",
            [ticker, cutoff, ticker, cutoff])
        row = res.fetchone() if res else None
        if not row or row[0] is None:
            return None
        cd = row[0]
        return {'calendardate': cd.isoformat() if hasattr(cd, 'isoformat') else cd,
                'total_value': row[1], 'total_units': row[2],
                'n_investors': row[3]}

    def daily_metric(self, ticker: str, date, field: Optional[str] = None):
        """DAILY valuation row for ``date`` (point-in-time by nature). Full dict
        {marketcap,pe,pb,ps,ev,evebit,evebitda}, a single field, or None."""
        d = self._date(date)
        if d is None:
            return None
        # ponytail: fetchone assumes (ticker,date) is unique, as Sharadar DAILY
        # is (SQLite enforces it via PK; Parquet does not). If a dup-row source
        # ever appears, add ORDER BY / dedup-on-ingest for a deterministic pick.
        res = self._query(
            'daily',
            "SELECT marketcap,pe,pb,ps,ev,evebit,evebitda FROM src "
            "WHERE ticker = ? AND CAST(date AS DATE) = ?", [ticker, d])
        row = res.fetchone() if res else None
        if not row:
            return None
        rec = dict(zip(_DAILY_FIELDS, row))
        return rec if field is None else rec.get(field)

    def daily_metrics(self, ticker: str, dates) -> List[Optional[Dict]]:
        """Batched twin of ``daily_metric``: ONE DAILY scan for all ``dates``
        instead of one per date. Returns full metric dicts (or None for dates
        with no row), aligned with ``dates``."""
        ds = [self._date(d) for d in dates]
        valid = sorted({d for d in ds if d is not None})
        if not valid:
            return [None] * len(ds)
        res = self._query(
            'daily',
            "SELECT CAST(date AS DATE), marketcap,pe,pb,ps,ev,evebit,evebitda "
            "FROM src WHERE ticker = ? AND CAST(date AS DATE) IN (%s)"
            % ",".join("?" * len(valid)), [ticker, *valid])
        rows = res.fetchall() if res else []
        by_date = {r[0]: dict(zip(_DAILY_FIELDS, r[1:])) for r in rows}
        return [by_date.get(d) if d is not None else None for d in ds]

    def daily_marketcaps(self, tickers: Sequence[str], date) -> Dict[str, float]:
        """Bulk twin of ``daily_metric(..., 'marketcap')``: ONE Parquet scan
        for all names on ``date`` instead of one scan per name. Names with no
        daily row (or NULL marketcap) are omitted."""
        d = self._date(date)
        if d is None or not tickers:
            return {}
        names = list(tickers)
        res = self._query(
            'daily',
            "SELECT ticker, marketcap FROM src WHERE CAST(date AS DATE) = ? "
            "AND ticker IN (%s)" % ",".join("?" * len(names)),
            [d, *names])
        if not res:
            return {}
        return {t: mc for t, mc in res.fetchall() if mc is not None}

    # ------------------------------------------------------------------
    # Intraday minute bars (bars_1m and siblings; see _INTRADAY_TABLES).
    # Every helper takes an additive ``table`` param defaulting to
    # 'bars_1m' — the default path is byte-identical to the pre-sibling
    # behavior (pinned in tests).
    # ------------------------------------------------------------------
    def _bars_pq(self, ticker: str, table: str = 'bars_1m') -> str:
        """Per-symbol Parquet path: ``<warehouse>/<table>/<TICKER>.parquet``."""
        return os.path.join(self.warehouse_dir, table, f"{ticker}.parquet")

    def write_bars_1m(self, ticker: str, df: pd.DataFrame, *,
                      table: str = 'bars_1m') -> int:
        """Write one symbol's minute bars (OVERWRITES that symbol's Parquet;
        idempotency policy — skip-existing vs re-download — belongs to the
        caller, scripts/ingest_alpaca_bars.py). ``df`` must carry the full
        ``_BARS_1M_COLUMNS`` contract with ``ts`` tz-naive US/Eastern; rows
        are sorted by ``ts`` and de-duplicated on it before the write. An
        empty frame writes NOTHING (so a failed fetch never masquerades as
        an ingested symbol) and returns 0. Returns the row count written.
        ``table`` picks the sibling intraday table (e.g. ``bars_1m_sip``)."""
        missing = [c for c in _BARS_1M_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"bars_1m frame missing columns {missing}")
        if df.empty:
            return 0
        out = (df.loc[:, list(_BARS_1M_COLUMNS)]
                 .sort_values('ts')
                 .drop_duplicates(subset='ts', keep='first'))
        path = self._bars_pq(ticker, table)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con = self._con()
        con.register('_bars_1m_df', out)
        try:
            con.execute(
                f"COPY (SELECT * FROM _bars_1m_df) TO '{path}' (FORMAT PARQUET)")
        finally:
            con.unregister('_bars_1m_df')
        return len(out)

    def _bars_query(self, ticker: str, where: str, params: list, *,
                    table: str = 'bars_1m') -> pd.DataFrame:
        """Shared bars_1m read: ts-indexed OHLCV+trade_count+vwap frame, empty
        (same columns) for a never-ingested symbol or a failed query."""
        cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name='ts'))
        path = self._bars_pq(ticker, table)
        if not os.path.exists(path):
            return empty
        sql = (f"SELECT ts, \"open\", high, low, \"close\", volume, "
               f"trade_count, vwap FROM read_parquet('{path}') "
               f"WHERE {where} ORDER BY ts")
        try:
            df = self._con().execute(sql, params).fetchdf()
        except duckdb.Error:
            logger.warning("bars_1m query failed on %s", ticker, exc_info=True)
            return empty
        df['ts'] = pd.to_datetime(df['ts'])
        return df.set_index('ts')

    def ohlcv_intraday(self, ticker: str, date, *,
                       table: str = 'bars_1m') -> pd.DataFrame:
        """ONE session's 1-minute bars for ``ticker``: DataFrame indexed by
        ``ts`` (tz-naive US/Eastern, ascending) with columns open/high/low/
        close/volume/trade_count/vwap. ALL bars of the calendar day are
        returned (extended hours included — the warehouse stores what the
        feed served; session filtering is the consumer's job). Empty frame
        for a bad date or a never-ingested symbol. ``table`` picks the
        sibling intraday table (e.g. ``bars_1m_sip``)."""
        d = self._date(date)
        if d is None:
            return self._bars_query(ticker, 'false', [], table=table)
        return self._bars_query(ticker, 'CAST(ts AS DATE) = ?', [d],
                                table=table)

    def ohlcv_intraday_range(self, ticker: str, start, end, *,
                             table: str = 'bars_1m') -> pd.DataFrame:
        """Bulk twin of ``ohlcv_intraday`` over [start, end] calendar days —
        ONE Parquet scan for a multi-year study instead of one per session."""
        s, e = self._date(start), self._date(end)
        if s is None or e is None:
            return self._bars_query(ticker, 'false', [], table=table)
        return self._bars_query(
            ticker, 'CAST(ts AS DATE) >= ? AND CAST(ts AS DATE) <= ?', [s, e],
            table=table)

    # ------------------------------------------------------------------
    # Option end-of-day bars (option_bars_eod; see _OPTION_TABLES).
    # Additive 2026-07-10 — nothing above this section changed.
    # ------------------------------------------------------------------
    def _option_bars_pq(self, underlying: str) -> str:
        """Per-underlying Parquet path:
        ``<warehouse>/option_bars_eod/<UNDERLYING>.parquet``."""
        return os.path.join(self.warehouse_dir, 'option_bars_eod',
                            f"{underlying}.parquet")

    def write_option_bars_eod(self, underlying: str, df: pd.DataFrame) -> int:
        """Write one underlying's option daily bars (OVERWRITES that
        underlying's Parquet; skip-existing idempotency belongs to the
        caller, scripts/ingest_massive_options.py). ``df`` must carry the
        full ``_OPTION_BARS_EOD_COLUMNS`` contract; rows are sorted by
        (selection_date, contract, ts) and de-duplicated on that key before
        the write — the key keeps ``selection_date`` so a contract selected
        in two months would keep both months' tags. An empty frame writes
        NOTHING (a failed fetch never masquerades as an ingested underlying)
        and returns 0. Returns the row count written."""
        missing = [c for c in _OPTION_BARS_EOD_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"option_bars_eod frame missing columns {missing}")
        if df.empty:
            return 0
        key = ['selection_date', 'contract', 'ts']
        out = (df.loc[:, list(_OPTION_BARS_EOD_COLUMNS)]
                 .sort_values(key)
                 .drop_duplicates(subset=key, keep='first'))
        path = self._option_bars_pq(underlying)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con = self._con()
        con.register('_option_bars_df', out)
        try:
            con.execute(f"COPY (SELECT * FROM _option_bars_df) TO '{path}' "
                        f"(FORMAT PARQUET)")
        finally:
            con.unregister('_option_bars_df')
        return len(out)

    def option_bars_eod(self, underlying: str) -> pd.DataFrame:
        """Full option_bars_eod frame for one underlying, sorted by
        (selection_date, contract, ts), with ``expiry``/``selection_date``/
        ``ts`` as datetime64. Empty frame (same columns) for a
        never-ingested underlying or a failed query."""
        cols = list(_OPTION_BARS_EOD_COLUMNS)
        path = self._option_bars_pq(underlying)
        if not os.path.exists(path):
            return pd.DataFrame(columns=cols)
        try:
            df = self._con().execute(
                f"SELECT * FROM read_parquet('{path}') "
                f"ORDER BY selection_date, contract, ts").fetchdf()
        except duckdb.Error:
            logger.warning("option_bars_eod query failed on %s", underlying,
                           exc_info=True)
            return pd.DataFrame(columns=cols)
        for c in ('expiry', 'selection_date', 'ts'):
            df[c] = pd.to_datetime(df[c])
        return df

    # ------------------------------------------------------------------
    # Ingest — OPERATOR-ONLY, network: bulk-export -> zip -> Parquet
    # ------------------------------------------------------------------
    def _export_link(self, table: str, params: Dict, *,
                     poll_attempts: int = 30, poll_wait: int = 10
                     ) -> str:  # pragma: no cover — network
        """Request a bulk export and poll until the snapshot is 'fresh', then
        return the signed S3 zip link. Reuses PitCache's retry/backoff."""
        import time

        from data.pit_provider import PitCache, _api_key
        p = dict(params)
        p['qopts.export'] = 'true'
        p['api_key'] = _api_key()
        for _ in range(poll_attempts):
            body = PitCache._get_json(table, p)
            f = body['datatable_bulk_download']['file']
            if f.get('status') == 'fresh':
                return f['link']
            logger.info("export %s: status=%s, waiting…", table, f.get('status'))
            time.sleep(poll_wait)
        raise RuntimeError(f"bulk export for {table} never became 'fresh'")

    def ingest_table(self, name: str) -> int:  # pragma: no cover — network
        """Bulk-export one Sharadar table to ``<warehouse>/<name>.parquet`` and
        return the row count. Streams the zip to disk and lets DuckDB do the
        CSV->Parquet conversion so multi-GB tables never load into memory."""
        import tempfile
        import zipfile

        import requests

        if name not in _TABLES:
            raise ValueError(f"unknown table {name!r}; one of {list(_TABLES)}")
        datatable, params = _TABLES[name]
        link = self._export_link(datatable, params)

        os.makedirs(self.warehouse_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, f"{name}.zip")
            with requests.get(link, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith('.csv'))
                zf.extract(csv_name, tmp)
            csv_path = os.path.join(tmp, csv_name)
            out = self._pq(name)
            con = self._con()
            # sample_size=-1: full scan for type inference — robust against
            # date/number columns whose first rows are null on big tables.
            con.execute(
                f"COPY (SELECT * FROM read_csv_auto('{csv_path}', sample_size=-1)) "
                f"TO '{out}' (FORMAT PARQUET)")
            row = con.execute(
                f"SELECT count(*) FROM read_parquet('{out}')").fetchone()
            # SELECT count(*) always returns exactly one aggregate row.
            assert row is not None
            return row[0]


if __name__ == '__main__':  # operator CLI (networked):
    #   set -a; source .env; set +a
    #   python -m data.pit_warehouse --tables tickers sep sf2
    #   python -m data.pit_warehouse --tables sf1 sf3 daily actions
    import argparse

    ap = argparse.ArgumentParser(prog='python -m data.pit_warehouse')
    ap.add_argument('--tables', nargs='+', required=True,
                    choices=list(_TABLES),
                    help='Sharadar tables to bulk-export to Parquet')
    ap.add_argument('--dir', default=None, help='warehouse dir (else PIT_WAREHOUSE_DIR)')
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    wh = PitWarehouse(cli.dir)
    for t in cli.tables:
        print(f"{t}: {wh.ingest_table(t)} rows -> {wh._pq(t)}")
