"""Point-in-time, survivorship-free market data via Sharadar (Nasdaq Data Link).

Mirrors data/earnings_cache.py: ``ingest`` (OPERATOR-ONLY, network) populates a
local SQLite cache from the Sharadar tables; every read is pure-SQLite, offline,
and POINT-IN-TIME by construction. This is the honest-data foundation the whole
"real Sharpe" effort rests on — the two invariants it exists to guarantee:

  * SURVIVORSHIP-FREE universe — ``universe_asof(date)`` returns the names that
    were ACTUALLY live on ``date`` (firstpricedate <= date <= lastpricedate),
    INCLUDING names later delisted. A universe of "today's survivors" overstates
    every backtest and hides the left tail; this fixes that at the source.
  * POINT-IN-TIME fundamentals — ``fundamentals_asof(ticker, date)`` returns the
    latest SF1 row whose DATEKEY (the SEC filing date, i.e. when the number
    became KNOWN) is <= ``date``, on the ARQ (as-first-reported) dimension.
    Filtering on calendardate/reportperiod, or using restated values, is
    lookahead — the classic way a backtest validates beautifully and fails live.

Prices use ``closeadj`` (split + dividend adjusted = total return) for returns.

CONSTRUCTION IS SIDE-EFFECT-FREE (like EarningsCache): building a PitCache
creates no file; a read against a missing db/table returns empty. Only ingest
(writes) creates the file. ':memory:' is supported for hermetic tests.

Key from ``NASDAQ_DATA_LINK_API_KEY`` (.env) — needed only for ``ingest``, never
for reads. Schema captured from the live API:
  TICKERS: permaticker,ticker,name,exchange,isdelisted,category,sector,industry,
           scalemarketcap,firstpricedate,lastpricedate,...
  SF1:     ticker,dimension,calendardate,datekey,reportperiod,eps,revenue,gp,
           assets,equity,marketcap,pb,roe,... (ARQ = as-reported quarterly)
  SEP:     ticker,date,open,high,low,close,volume,closeadj,closeunadj
  ACTIONS: date,action,ticker,value,contraticker
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_API_BASE = "https://data.nasdaq.com/api/v3/datatables/{}.json"

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS sep (
        ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
        volume REAL, closeadj REAL, PRIMARY KEY(ticker, date))""",
    """CREATE TABLE IF NOT EXISTS sf1 (
        ticker TEXT, dimension TEXT, datekey TEXT, calendardate TEXT,
        data_json TEXT, PRIMARY KEY(ticker, dimension, datekey))""",
    """CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT PRIMARY KEY, permaticker TEXT, name TEXT, exchange TEXT,
        isdelisted TEXT, category TEXT, sector TEXT, industry TEXT,
        scalemarketcap TEXT, firstpricedate TEXT, lastpricedate TEXT,
        data_json TEXT)""",
    """CREATE TABLE IF NOT EXISTS actions (
        date TEXT, action TEXT, ticker TEXT, value REAL, contraticker TEXT)""",
)


def _default_db_path() -> str:
    """Dedicated PIT db path (NOT trading_data.db / earnings_data.db).

    Honors ``PIT_DB_PATH``; else co-locates ``pit_data.db`` beside
    ``TRADING_DB_PATH``; else the CWD.
    """
    explicit = os.environ.get('PIT_DB_PATH')
    if explicit:
        return explicit
    trading = os.environ.get('TRADING_DB_PATH')
    if trading and trading != ':memory:':
        return os.path.join(os.path.dirname(trading) or '.', 'pit_data.db')
    return 'pit_data.db'


def _api_key() -> str:
    key = os.environ.get('NASDAQ_DATA_LINK_API_KEY')
    if not key:
        raise ValueError(
            "NASDAQ_DATA_LINK_API_KEY not set — ingest needs it. "
            "`set -a; source .env; set +a` first (reads never need a key).")
    return key


class PitCache:
    """Offline, SQLite-backed Sharadar cache (see module docstring)."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        self._mem_conn = None
        if self.db_path == ':memory:':
            self._mem_conn = sqlite3.connect(':memory:')
            for ddl in _SCHEMA:
                self._mem_conn.execute(ddl)
            self._mem_conn.commit()

    # ------------------------------------------------------------------
    # Connection helpers (mirror EarningsCache)
    # ------------------------------------------------------------------
    def _writable_conn(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        for ddl in _SCHEMA:
            conn.execute(ddl)
        return conn

    def _readable_conn(self) -> Optional[sqlite3.Connection]:
        """A connection for reads, or None for an un-ingested (missing) cache."""
        if self._mem_conn is not None:
            return self._mem_conn
        if not os.path.exists(self.db_path):
            return None
        return sqlite3.connect(self.db_path)

    def _close(self, conn: Optional[sqlite3.Connection]) -> None:
        if conn is not None and conn is not self._mem_conn:
            conn.close()

    @staticmethod
    def _iso(d) -> Optional[str]:
        try:
            ts = pd.Timestamp(d)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(ts) else ts.date().isoformat()

    # ------------------------------------------------------------------
    # Reads — offline, point-in-time
    # ------------------------------------------------------------------
    def universe_asof(self, date, *,
                      category: Optional[str] = 'Domestic Common Stock',
                      scalemarketcap: Optional[Sequence[str]] = None
                      ) -> List[str]:
        """Tickers that were LIVE on ``date`` (survivorship-free).

        Live := firstpricedate <= date <= lastpricedate. Includes names later
        delisted (for their live span) and excludes names not yet listed —
        the whole point. ``category`` filters to e.g. common stock; an
        optional ``scalemarketcap`` list (Sharadar buckets like '1 - Nano' ..
        '6 - Mega') restricts size. NOTE: scalemarketcap is the CURRENT bucket
        (not point-in-time), so size filtering is approximate — use it to scope
        a universe, not as a point-in-time signal.
        """
        d = self._iso(date)
        conn = self._readable_conn()
        if d is None or conn is None:
            return []
        sql = ("SELECT ticker FROM tickers WHERE firstpricedate IS NOT NULL "
               "AND firstpricedate <= ? AND (lastpricedate >= ? OR "
               "lastpricedate IS NULL)")
        args: list = [d, d]
        if category is not None:
            sql += " AND category = ?"
            args.append(category)
        if scalemarketcap:
            sql += " AND scalemarketcap IN (%s)" % ",".join("?" * len(scalemarketcap))
            args.extend(scalemarketcap)
        try:
            rows = conn.execute(sql + " ORDER BY ticker", args).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            self._close(conn)
        return [r[0] for r in rows]

    def fundamentals_asof(self, ticker: str, date, *,
                          dimension: str = 'ARQ') -> Optional[Dict]:
        """Latest SF1 row KNOWN on ``date`` — datekey <= date (point-in-time).

        Returns the full fundamental record (dict) or None. Gating on
        ``datekey`` (the filing date) — never calendardate/reportperiod — is
        what makes this lookahead-free: a number filed 2015-05-01 is simply not
        visible to a 2015-03-01 query.
        """
        d = self._iso(date)
        conn = self._readable_conn()
        if d is None or conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT data_json FROM sf1 WHERE ticker = ? AND dimension = ? "
                "AND datekey <= ? ORDER BY datekey DESC LIMIT 1",
                (ticker, dimension, d)).fetchone()
        except sqlite3.OperationalError:
            row = None
        finally:
            self._close(conn)
        return json.loads(row[0]) if row else None

    def prices(self, ticker: str, start, end, *,
               field: str = 'closeadj') -> pd.Series:
        """Total-return-adjusted close series (default ``closeadj``), [start, end].

        Empty Series for an un-ingested cache / unknown ticker. Use closeadj for
        returns (split + dividend adjusted); 'close' for raw.
        """
        s, e = self._iso(start), self._iso(end)
        conn = self._readable_conn()
        if s is None or e is None or conn is None:
            return pd.Series(dtype=float, name=ticker)
        col = field if field in ('closeadj', 'close', 'open', 'high', 'low',
                                  'volume') else 'closeadj'
        try:
            rows = conn.execute(
                f"SELECT date, {col} FROM sep WHERE ticker = ? AND date >= ? "
                "AND date <= ? ORDER BY date", (ticker, s, e)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            self._close(conn)
        if not rows:
            return pd.Series(dtype=float, name=ticker)
        idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
        return pd.Series([r[1] for r in rows], index=idx, name=ticker)

    # ------------------------------------------------------------------
    # Writes — store (used by ingest AND tests)
    # ------------------------------------------------------------------
    def store_tickers(self, rows: Sequence[Dict]) -> int:
        recs = [(r.get('ticker'), str(r.get('permaticker')), r.get('name'),
                 r.get('exchange'), r.get('isdelisted'), r.get('category'),
                 r.get('sector'), r.get('industry'), r.get('scalemarketcap'),
                 self._iso(r.get('firstpricedate')),
                 self._iso(r.get('lastpricedate')), json.dumps(r))
                for r in rows if r.get('ticker')]
        return self._executemany(
            "INSERT OR REPLACE INTO tickers (ticker,permaticker,name,exchange,"
            "isdelisted,category,sector,industry,scalemarketcap,firstpricedate,"
            "lastpricedate,data_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", recs)

    def store_sf1(self, rows: Sequence[Dict]) -> int:
        recs = [(r.get('ticker'), r.get('dimension'), self._iso(r.get('datekey')),
                 self._iso(r.get('calendardate')), json.dumps(r))
                for r in rows if r.get('ticker') and r.get('datekey')]
        return self._executemany(
            "INSERT OR REPLACE INTO sf1 (ticker,dimension,datekey,calendardate,"
            "data_json) VALUES (?,?,?,?,?)", recs)

    def store_sep(self, rows: Sequence[Dict]) -> int:
        recs = [(r.get('ticker'), self._iso(r.get('date')), r.get('open'),
                 r.get('high'), r.get('low'), r.get('close'), r.get('volume'),
                 r.get('closeadj'))
                for r in rows if r.get('ticker') and r.get('date')]
        return self._executemany(
            "INSERT OR REPLACE INTO sep (ticker,date,open,high,low,close,volume,"
            "closeadj) VALUES (?,?,?,?,?,?,?,?)", recs)

    def store_actions(self, rows: Sequence[Dict]) -> int:
        recs = [(self._iso(r.get('date')), r.get('action'), r.get('ticker'),
                 r.get('value'), r.get('contraticker')) for r in rows
                if r.get('ticker') and r.get('date')]
        return self._executemany(
            "INSERT INTO actions (date,action,ticker,value,contraticker) "
            "VALUES (?,?,?,?,?)", recs)

    def _executemany(self, sql: str, recs: list) -> int:
        if not recs:
            return 0
        conn = self._writable_conn()
        conn.executemany(sql, recs)
        conn.commit()
        self._close(conn)
        return len(recs)

    # ------------------------------------------------------------------
    # Ingest — OPERATOR-ONLY, network (requests; mirrors EDGAR fetcher)
    # ------------------------------------------------------------------
    def _get_table(self, table: str, params: Dict) -> List[Dict]:
        """All rows of a Sharadar datatable for ``params`` (cursor-paginated)."""
        import requests
        out: List[Dict] = []
        cursor = None
        while True:
            p = dict(params)
            p['api_key'] = _api_key()
            p['qopts.per_page'] = '10000'
            if cursor:
                p['qopts.cursor_id'] = cursor
            resp = requests.get(_API_BASE.format(table), params=p, timeout=60)
            resp.raise_for_status()
            body = resp.json()
            dt = body['datatable']
            cols = [c['name'] for c in dt['columns']]
            out.extend(dict(zip(cols, row)) for row in dt['data'])
            cursor = body.get('meta', {}).get('next_cursor_id')
            if not cursor:
                return out

    def ingest_tickers(self) -> int:  # pragma: no cover — network
        return self.store_tickers(self._get_table('SHARADAR/TICKERS', {'table': 'SF1'}))

    def ingest_prices(self, tickers: Sequence[str],
                      start: str, end: str) -> int:  # pragma: no cover — network
        n = 0
        for t in tickers:
            n += self.store_sep(self._get_table(
                'SHARADAR/SEP', {'ticker': t, 'date.gte': start, 'date.lte': end}))
        return n

    def ingest_fundamentals(self, tickers: Sequence[str],
                            dimension: str = 'ARQ') -> int:  # pragma: no cover
        n = 0
        for t in tickers:
            n += self.store_sf1(self._get_table(
                'SHARADAR/SF1', {'ticker': t, 'dimension': dimension}))
        return n


if __name__ == '__main__':  # operator CLI (networked):
    #   set -a; source .env; set +a
    #   python -m data.pit_provider --tickers
    #   python -m data.pit_provider --prices --start 2004-01-01 --end 2024-12-31 --symbols AAPL MSFT
    #   python -m data.pit_provider --fundamentals --symbols AAPL MSFT
    import argparse

    ap = argparse.ArgumentParser(prog='python -m data.pit_provider')
    ap.add_argument('--tickers', action='store_true', help='ingest the TICKERS universe table')
    ap.add_argument('--prices', action='store_true', help='ingest SEP prices for --symbols')
    ap.add_argument('--fundamentals', action='store_true', help='ingest SF1 (ARQ) for --symbols')
    ap.add_argument('--symbols', nargs='+', help='tickers for --prices/--fundamentals')
    ap.add_argument('--start', default='2004-01-01')
    ap.add_argument('--end', default='2024-12-31')
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cache = PitCache()
    if cli.tickers:
        print(f"tickers ingested: {cache.ingest_tickers()}")
    if cli.prices:
        if not cli.symbols:
            ap.error('--prices needs --symbols')
        print(f"price rows ingested: {cache.ingest_prices(cli.symbols, cli.start, cli.end)}")
    if cli.fundamentals:
        if not cli.symbols:
            ap.error('--fundamentals needs --symbols')
        print(f"fundamental rows ingested: {cache.ingest_fundamentals(cli.symbols)}")
    if not (cli.tickers or cli.prices or cli.fundamentals):
        ap.error('pass --tickers and/or --prices/--fundamentals --symbols ...')
