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

import logging
import os
from typing import Dict, List, Optional, Sequence

import duckdb
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
}

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
            return con.execute(
                f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]


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
