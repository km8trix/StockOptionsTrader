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
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

from data.corporate_action_evidence import (
    ACTIONS_RECEIPT_FILE,
    ActionsEvidenceError,
    archive_raw_zip,
    atomic_write_json,
    build_actions_acquisition_document,
    expected_datatable_metadata,
    inspect_actions_parquet,
    inspect_actions_zip,
    validate_actions_evidence,
)
from data.session_close_calendar import load_session_close_calendar_evidence
from data.sharadar_source_evidence import (
    CANDIDATE_TABLES,
    SharadarSourceEvidenceError,
    build_pead_sharadar_source_snapshot,
    build_sharadar_table_acquisition_document,
    convert_sharadar_zip_to_parquet,
    load_sharadar_table_acquisition,
    normalize_datatable_metadata,
    publish_pead_sharadar_source_snapshot,
    publish_sharadar_table_acquisition,
)

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

_PRICE_FIELDS = (
    'closeadj', 'closeunadj', 'close', 'open', 'high', 'low', 'volume',
)
_DAILY_FIELDS = ('marketcap', 'pe', 'pb', 'ps', 'ev', 'evebit', 'evebitda')


class WarehouseReadError(ValueError):
    """Base class for a strict local-warehouse read contract failure."""


class WarehouseTableMissingError(WarehouseReadError):
    """A strict read was requested from a table that is not present."""


class WarehouseQueryError(WarehouseReadError):
    """DuckDB could not read or query a required warehouse table."""


class WarehouseSchemaError(WarehouseReadError):
    """A required warehouse column or value violates its contract."""


class SecurityLifecycleError(WarehouseReadError):
    """TICKERS/SEP cannot prove one internally consistent lifecycle."""


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

    def __init__(
            self, warehouse_dir: Optional[str] = None,
            session_close_calendar_path: Optional[str] = None):
        self.warehouse_dir = warehouse_dir or _default_warehouse_dir()
        self.session_close_calendar_path = session_close_calendar_path
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

    def _query_strict(self, table: str, sql: str, params: list):
        """Run a warehouse query without collapsing evidence failures.

        The ordinary readers intentionally preserve their historical
        empty-on-missing behavior.  Research qualification cannot use that
        behavior because an absent Parquet file and a legitimately empty
        ticker/date result mean different things.  Strict readers route through
        this helper so callers can distinguish both cases.
        """
        if not self._have(table):
            raise WarehouseTableMissingError(
                f"required warehouse table is missing: {table}")
        escaped_path = self._pq(table).replace("'", "''")
        full = sql.replace('src', f"read_parquet('{escaped_path}')")
        try:
            return self._con().execute(full, params)
        except duckdb.Error as exc:
            raise WarehouseQueryError(
                f"warehouse query failed on {table}: {exc}") from exc

    def _strict_schema(self, table: str, required: Sequence[str]) -> Dict[str, str]:
        """Return an exact-case schema and require every named column."""
        result = self._query_strict(
            table, "DESCRIBE SELECT * FROM src", [])
        rows = result.fetchall()
        names = [str(row[0]) for row in rows]
        if len(names) != len(set(names)):
            raise WarehouseSchemaError(
                f"{table} contains duplicate column names")
        schema = {str(row[0]): str(row[1]).upper() for row in rows}
        missing = [column for column in required if column not in schema]
        if missing:
            raise WarehouseSchemaError(
                f"{table} is missing required columns: {missing}")
        return schema

    @staticmethod
    def _price_field(field: str) -> str:
        if field not in _PRICE_FIELDS:
            raise ValueError(
                f"unsupported price field {field!r}; expected one of "
                f"{list(_PRICE_FIELDS)}")
        return field

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

    def corporate_action_evidence(self, start, end) -> Dict:
        """Revalidate the active ACTIONS evidence for a required date window.

        Unlike the compatibility readers, this research boundary never treats
        a missing or malformed table as an empty result.  The receipt,
        immutable raw ZIP, converted Parquet, schema, hashes, and requested
        coverage must all validate or :class:`ActionsEvidenceError` is raised.
        """
        required_start = self._date(start)
        required_end = self._date(end)
        if required_start is None or required_end is None:
            raise ActionsEvidenceError(
                "required ACTIONS evidence dates must be valid")
        return validate_actions_evidence(
            self.warehouse_dir,
            required_start=required_start.isoformat(),
            required_end=required_end.isoformat(),
        )

    def corporate_actions_for_tickers(
            self, tickers: Sequence[str], start, end) -> List[Dict]:
        """Return an exact, strictly read corporate-action slice.

        This is the research-grade companion to
        :meth:`corporate_action_evidence`.  The evidence method proves the
        immutable source bytes; this method exposes the rows needed to account
        for distributions and to detect unsupported holder-affecting events.
        It deliberately returns the generic vendor ``value`` without assigning
        economic semantics to it.  In particular, callers must never use this
        reader to infer terminal merger or delisting proceeds.

        Missing tables, schema drift, malformed rows, or ambiguous input dates
        raise instead of becoming an empty action history.  A genuinely empty
        result for a valid ticker/date request remains an empty list.
        """
        required_start = self._date(start)
        required_end = self._date(end)
        if (
            required_start is None
            or required_end is None
            or required_start > required_end
        ):
            raise WarehouseSchemaError(
                "corporate-action slice requires an ordered date window")
        if isinstance(tickers, (str, bytes)):
            raise WarehouseSchemaError(
                "corporate-action tickers must be a sequence of symbols")
        normalized: List[str] = []
        for raw_ticker in tickers:
            if not isinstance(raw_ticker, str):
                raise WarehouseSchemaError(
                    "corporate-action ticker must be text")
            ticker = raw_ticker.strip().upper()
            if not ticker or ticker != raw_ticker:
                raise WarehouseSchemaError(
                    "corporate-action tickers must be canonical uppercase text")
            normalized.append(ticker)
        if len(normalized) != len(set(normalized)):
            raise WarehouseSchemaError(
                "corporate-action tickers must be unique")
        if not normalized:
            return []

        self._strict_schema(
            'actions',
            (
                'date', 'action', 'ticker', 'name', 'value',
                'contraticker', 'contraname',
            ),
        )
        placeholders = ','.join('?' for _ in normalized)
        result = self._query_strict(
            'actions',
            "SELECT CAST(date AS DATE), action, ticker, name, value, "
            "contraticker, contraname FROM src "
            f"WHERE ticker IN ({placeholders}) "
            "AND CAST(date AS DATE) >= ? AND CAST(date AS DATE) <= ? "
            "ORDER BY ticker, CAST(date AS DATE), action, name, "
            "contraticker, contraname",
            [*normalized, required_start, required_end],
        )
        requested = set(normalized)
        rows: List[Dict] = []
        identities = set()
        for raw_row in result.fetchall():
            (
                raw_date, raw_action, raw_ticker, raw_name, raw_value,
                raw_contraticker, raw_contraname,
            ) = raw_row
            action_date = self._date(raw_date)
            ticker = str(raw_ticker) if raw_ticker is not None else ''
            action = str(raw_action) if raw_action is not None else ''
            name = str(raw_name) if raw_name is not None else ''
            if (
                action_date is None
                or not required_start <= action_date <= required_end
                or ticker not in requested
                or not action.strip()
                or action != action.strip()
                or not name.strip()
                or name != name.strip()
            ):
                raise WarehouseSchemaError(
                    "corporate-action slice contains a malformed key")
            value = None
            if raw_value is not None:
                if isinstance(raw_value, bool):
                    raise WarehouseSchemaError(
                        "corporate-action value must be numeric or null")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise WarehouseSchemaError(
                        "corporate-action value must be numeric or null") from exc
                if not np.isfinite(value):
                    raise WarehouseSchemaError(
                        "corporate-action value must be finite")
                if value == 0.0:
                    value = 0.0
            contra_ticker = (
                None if raw_contraticker is None else str(raw_contraticker)
            )
            contra_name = None if raw_contraname is None else str(raw_contraname)
            for label, value_text in (
                ('contraticker', contra_ticker), ('contraname', contra_name)
            ):
                if value_text is not None and (
                    not value_text.strip() or value_text != value_text.strip()
                ):
                    raise WarehouseSchemaError(
                        f"corporate-action {label} must be canonical text or null")
            identity = (
                action_date, ticker, name, action, contra_name, contra_ticker,
            )
            if identity in identities:
                raise WarehouseSchemaError(
                    "corporate-action slice contains a duplicate primary key")
            identities.add(identity)
            rows.append({
                'date': action_date.isoformat(),
                'action': action,
                'ticker': ticker,
                'name': name,
                'value': value,
                'contraticker': contra_ticker,
                'contraname': contra_name,
            })
        return rows

    def market_sessions(self, start, end) -> pd.DatetimeIndex:
        """Observed US-equity sessions in SEP between ``start`` and ``end``.

        Research formation dates must come from the data actually being
        evaluated.  ``pandas.bdate_range`` knows weekdays but not exchange
        holidays (or one-off closures), which can silently assign a monthly
        rebalance to a day with no prices, volume, or dated market cap.  The
        union of SEP dates is the warehouse's authoritative session calendar.
        """
        lo, hi = self._date(start), self._date(end)
        if lo is None or hi is None or lo > hi:
            return pd.DatetimeIndex([], name='date')
        result = self._query(
            'sep',
            "SELECT DISTINCT CAST(date AS DATE) AS session FROM src "
            "WHERE CAST(date AS DATE) >= ? AND CAST(date AS DATE) <= ? "
            "ORDER BY session",
            [lo, hi],
        )
        rows = result.fetchall() if result else []
        return pd.DatetimeIndex([row[0] for row in rows], name='date')

    def market_session_close_calendar(self) -> Dict:
        """Return immutable NYSE close-time evidence for research validation.

        The provider deliberately does not interpret this document.  Primary
        and independent reference implementations each validate its content
        address, source coverage, early-close rows, and required date range
        before deriving any visibility cutoff.
        """
        return load_session_close_calendar_evidence(self.session_close_calendar_path)

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

    def security_lifecycle(self, ticker: str) -> Dict:
        """Return one strictly validated TICKERS/SEP lifecycle.

        A qualifying lifecycle needs exactly one TICKERS identity row, a
        positive permanent identifier, literal ``Y``/``N`` delisting evidence,
        and an explicit ``lastpricedate``.  For a delisted security the claimed
        final date must also be the final observed SEP session.  Any missing or
        contradictory evidence raises :class:`SecurityLifecycleError`; it is
        never converted into a live-security assumption here.
        """
        try:
            self._strict_schema(
                'tickers',
                ('ticker', 'permaticker', 'isdelisted', 'lastpricedate'),
            )
            result = self._query_strict(
                'tickers',
                "SELECT ticker, permaticker, "
                "CAST(isdelisted AS VARCHAR), lastpricedate FROM src "
                "WHERE ticker = ?",
                [ticker],
            )
            rows = result.fetchall()
        except WarehouseReadError as exc:
            raise SecurityLifecycleError(
                f"cannot validate lifecycle for {ticker!r}: {exc}") from exc

        if len(rows) != 1:
            raise SecurityLifecycleError(
                f"expected exactly one TICKERS row for {ticker!r}; "
                f"found {len(rows)}")
        row_ticker, raw_permaticker, raw_status, raw_last_price = rows[0]

        if isinstance(raw_permaticker, bool):
            raise SecurityLifecycleError(
                f"invalid permaticker for {ticker!r}: {raw_permaticker!r}")
        try:
            permaticker = int(raw_permaticker)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SecurityLifecycleError(
                f"invalid permaticker for {ticker!r}: "
                f"{raw_permaticker!r}") from exc
        try:
            exact_permaticker = float(raw_permaticker) == permaticker
        except (TypeError, ValueError, OverflowError):
            exact_permaticker = str(raw_permaticker) == str(permaticker)
        if permaticker <= 0 or not exact_permaticker:
            raise SecurityLifecycleError(
                f"invalid permaticker for {ticker!r}: {raw_permaticker!r}")

        status = str(raw_status) if raw_status is not None else None
        if status not in ('N', 'Y'):
            raise SecurityLifecycleError(
                f"isdelisted must be literal N or Y for {ticker!r}; "
                f"found {raw_status!r}")
        last_price_date = self._date(raw_last_price)
        if last_price_date is None:
            raise SecurityLifecycleError(
                f"lastpricedate is required for {ticker!r}")

        sep_last_price_date = None
        if status == 'Y':
            try:
                self._strict_schema('sep', ('ticker', 'date'))
                result = self._query_strict(
                    'sep',
                    "SELECT MAX(CAST(date AS DATE)) FROM src "
                    "WHERE ticker = ?",
                    [ticker],
                )
                row = result.fetchone()
            except WarehouseReadError as exc:
                raise SecurityLifecycleError(
                    f"cannot validate final SEP date for {ticker!r}: "
                    f"{exc}") from exc
            sep_last_price_date = self._date(row[0]) if row else None
            if sep_last_price_date != last_price_date:
                raise SecurityLifecycleError(
                    f"TICKERS lastpricedate {last_price_date} does not match "
                    f"SEP max date {sep_last_price_date} for {ticker!r}")

        return {
            'ticker': str(row_ticker),
            'permaticker': permaticker,
            'isdelisted': status,
            'lastpricedate': last_price_date,
            'sep_lastpricedate': sep_last_price_date,
        }

    def security_currency(self, ticker: str) -> Dict:
        """Return the exact listing currency for economic cash accounting.

        ACTIONS has no currency column.  A cash distribution therefore cannot
        safely be combined with a USD research book unless the corresponding
        TICKERS identity independently proves its currency.  This strict reader
        keeps that evidence separate from :meth:`security_lifecycle` so the
        established lifecycle contract remains backward compatible.
        """
        if not isinstance(ticker, str) or not ticker or ticker != ticker.strip():
            raise WarehouseSchemaError("security currency requires a canonical ticker")
        self._strict_schema('tickers', ('ticker', 'currency'))
        rows = self._query_strict(
            'tickers',
            "SELECT ticker, currency FROM src WHERE ticker = ?",
            [ticker],
        ).fetchall()
        if len(rows) != 1:
            raise WarehouseSchemaError(
                f"expected exactly one TICKERS currency row for {ticker!r}; "
                f"found {len(rows)}")
        row_ticker, raw_currency = rows[0]
        currency = str(raw_currency) if raw_currency is not None else ''
        if row_ticker != ticker or not currency or currency != currency.strip():
            raise WarehouseSchemaError(
                f"invalid TICKERS currency evidence for {ticker!r}")
        return {'ticker': ticker, 'currency': currency.upper()}

    def delisting_date(self, ticker: str):
        """Return the proven final listed date, or fail closed to ``None``.

        Call :meth:`security_lifecycle` when the caller must distinguish a live
        security from missing or contradictory lifecycle evidence.
        """
        try:
            lifecycle = self.security_lifecycle(ticker)
        except SecurityLifecycleError:
            logger.warning(
                "security lifecycle validation failed for %s",
                ticker,
                exc_info=True,
            )
            return None
        return (lifecycle['lastpricedate']
                if lifecycle['isdelisted'] == 'Y' else None)

    def delisting_action(self, ticker: str) -> Optional[Dict]:
        """Classify a corporate action near the final listed session.

        Sharadar ACTIONS ``value`` is reported event metadata, not a verified
        per-share settlement term (large acquisition rows can contain total
        deal values).  Preserve it for auditability but never expose it as a
        payout or infer zero recovery from it.  A trusted, independently sourced
        terminal-settlement ledger is required before execution may model the
        action's economics.
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
            try:
                value = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None and not np.isfinite(value):
                value = None
            if any(token in action for token in (
                    'acquisition', 'merger', 'tender', 'bankrupt',
                    'liquidat')):
                return {
                    'date': self._date(action_date),
                    'action': action,
                    'reported_value': value,
                    'value_semantics': 'vendor_reported_value_not_per_share',
                    'contraticker': contra,
                    'quality_flags': [
                        'corporate_action_value_not_per_share',
                        'delisting_terms_unavailable',
                    ],
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
        col = self._price_field(field)
        s, e = self._date(start), self._date(end)
        if s is None or e is None:
            return pd.Series(dtype=float, name=ticker)
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

    def prices_strict(self, ticker: str, start, end, field: str) -> pd.Series:
        """Read the exact requested SEP field under a fail-loud contract.

        An unknown ticker or a range with no observations is a valid empty
        result.  Missing/corrupt Parquet, missing columns, duplicate dates, and
        non-finite prices are evidence failures and raise a typed
        :class:`WarehouseReadError` subclass.  No as-of or nearest-session
        substitution is performed: both range endpoints are exact and
        inclusive.
        """
        col = self._price_field(field)
        start_date, end_date = self._date(start), self._date(end)
        if start_date is None or end_date is None:
            raise WarehouseSchemaError(
                f"invalid price range: {start!r} to {end!r}")
        if start_date > end_date:
            raise WarehouseSchemaError(
                f"price range starts after it ends: "
                f"{start_date} to {end_date}")

        schema = self._strict_schema('sep', ('ticker', 'date', col))
        numeric_prefixes = (
            'TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT',
            'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT', 'FLOAT',
            'REAL', 'DOUBLE', 'DECIMAL',
        )
        if not schema[col].startswith(numeric_prefixes):
            raise WarehouseSchemaError(
                f"sep.{col} must be numeric; found {schema[col]}")

        result = self._query_strict(
            'sep',
            f"SELECT CAST(date AS DATE), {col} FROM src WHERE ticker = ? "
            "AND CAST(date AS DATE) >= ? AND CAST(date AS DATE) <= ? "
            "ORDER BY CAST(date AS DATE)",
            [ticker, start_date, end_date],
        )
        rows = result.fetchall()
        dates = [self._date(row[0]) for row in rows]
        if any(date_value is None for date_value in dates):
            raise WarehouseSchemaError(
                f"sep contains an invalid date for {ticker!r}")
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise WarehouseSchemaError(
                f"sep dates must be sorted and unique for {ticker!r}")

        values = []
        for _, raw_value in rows:
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise WarehouseSchemaError(
                    f"sep.{col} contains a non-numeric value for "
                    f"{ticker!r}: {raw_value!r}") from exc
            if not np.isfinite(value):
                raise WarehouseSchemaError(
                    f"sep.{col} contains a non-finite value for "
                    f"{ticker!r}")
            values.append(value)

        index = pd.DatetimeIndex(
            [pd.Timestamp(date_value) for date_value in dates])
        return pd.Series(values, index=index, name=ticker, dtype=float)

    def prices_bulk(self, tickers: Sequence[str], start, end, *,
                    field: str = 'closeadj') -> Dict[str, pd.Series]:
        """Batch twin of :meth:`prices` for a bounded group of tickers.

        Full-universe research callers should invoke this in chunks. One
        Parquet scan per chunk is materially faster than thousands of
        per-ticker scans while keeping peak memory bounded.
        """
        col = self._price_field(field)
        names = sorted({str(ticker) for ticker in tickers if str(ticker)})
        s, e = self._date(start), self._date(end)
        if not names or s is None or e is None:
            return {}
        result = self._query(
            'sep',
            f"SELECT ticker, CAST(date AS DATE), {col} FROM src "
            f"WHERE ticker IN ({','.join('?' * len(names))}) "
            "AND CAST(date AS DATE) >= ? AND CAST(date AS DATE) <= ? "
            "ORDER BY ticker, CAST(date AS DATE)",
            [*names, s, e],
        )
        rows = result.fetchall() if result else []
        grouped: Dict[str, list[tuple]] = {}
        for ticker, date_value, price in rows:
            grouped.setdefault(ticker, []).append((date_value, price))
        return {
            ticker: pd.Series(
                [row[1] for row in values],
                index=pd.DatetimeIndex([pd.Timestamp(row[0]) for row in values]),
                name=ticker,
            )
            for ticker, values in grouped.items()
        }

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

    def daily_marketcaps_for_dates(
            self, tickers: Sequence[str], dates) -> Dict[tuple[str, object], float]:
        """Batch dated market caps keyed by ``(ticker, datetime.date)``.

        This is the bounded-panel twin of :meth:`daily_marketcaps`; callers
        should pass a ticker chunk and all required formation dates.
        """
        names = sorted({str(ticker) for ticker in tickers if str(ticker)})
        valid_dates = sorted({self._date(value) for value in dates
                              if self._date(value) is not None})
        if not names or not valid_dates:
            return {}
        result = self._query(
            'daily',
            "SELECT ticker, CAST(date AS DATE), marketcap FROM src "
            "WHERE ticker IN (%s) AND CAST(date AS DATE) IN (%s)"
            % (','.join('?' * len(names)),
               ','.join('?' * len(valid_dates))),
            [*names, *valid_dates],
        )
        rows = result.fetchall() if result else []
        return {(ticker, date_value): float(marketcap)
                for ticker, date_value, marketcap in rows
                if marketcap is not None}

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
    @staticmethod
    def _candidate_datatable_metadata(
            logical_name: str, table: str) -> Dict:  # pragma: no cover — network
        """Fetch only credential-free metadata used by candidate receipts."""
        from data.pit_provider import PitCache, _api_key

        if logical_name not in CANDIDATE_TABLES:
            raise ValueError(
                f"candidate metadata is limited to {list(CANDIDATE_TABLES)}")
        expected_table = _TABLES[logical_name][0]
        if table != expected_table:
            raise ValueError(
                f"candidate metadata table differs for {logical_name!r}")
        body = PitCache._get_json(
            f"{table}/metadata", {'api_key': _api_key()})
        raw = body.get('datatable') if isinstance(body, Mapping) else None
        if not isinstance(raw, Mapping):
            raise SharadarSourceEvidenceError(
                f"{logical_name} datatable metadata response is malformed")
        raw_columns = raw.get('columns')
        if not isinstance(raw_columns, list):
            raise SharadarSourceEvidenceError(
                f"{logical_name} datatable metadata columns are malformed")
        columns = []
        for column in raw_columns:
            if not isinstance(column, Mapping):
                raise SharadarSourceEvidenceError(
                    f"{logical_name} datatable metadata column is malformed")
            columns.append({
                'name': column.get('name'),
                'type': column.get('type'),
                'description': column.get('description'),
            })
        status = raw.get('status')
        if not isinstance(status, Mapping):
            raise SharadarSourceEvidenceError(
                f"{logical_name} datatable metadata status is missing")
        sanitized = {
            'vendor_code': raw.get('vendor_code'),
            'datatable_code': raw.get('datatable_code'),
            'name': raw.get('name'),
            'description': raw.get('description'),
            'columns': columns,
            'filters': raw.get('filters'),
            'primary_key': raw.get('primary_key'),
            'premium': raw.get('premium'),
            'status': {
                'expected_at': status.get('expected_at'),
                'refreshed_at': status.get('refreshed_at'),
                'status': status.get('status'),
                'update_frequency': status.get('update_frequency'),
            },
        }
        return normalize_datatable_metadata(logical_name, sanitized)

    @staticmethod
    def _actions_datatable_metadata(
            table: str) -> Dict:  # pragma: no cover — network
        """Fetch and sanitize the official ACTIONS metadata contract.

        The API credential is request-only.  Only the schema/status fields
        consumed by the evidence receipt are returned, so neither credentials,
        request URLs, nor unrelated response fields can enter persisted
        provenance.
        """
        from data.pit_provider import PitCache, _api_key

        if table != 'SHARADAR/ACTIONS':
            raise ValueError("datatable metadata evidence is ACTIONS-only")
        body = PitCache._get_json(
            f"{table}/metadata", {'api_key': _api_key()})
        raw = body.get('datatable') if isinstance(body, Mapping) else None
        if not isinstance(raw, Mapping):
            raise ActionsEvidenceError(
                "ACTIONS datatable metadata response is malformed")
        expected = expected_datatable_metadata()
        sanitized = {field: raw.get(field) for field in expected}
        status = raw.get('status')
        if not isinstance(status, Mapping):
            raise ActionsEvidenceError(
                "ACTIONS datatable metadata status is missing")
        sanitized['status'] = dict(status)
        return sanitized

    def _export_link(self, table: str, params: Dict, *,
                     poll_attempts: int = 30, poll_wait: int = 10,
                     include_metadata: bool = False,
                     ) -> str | tuple[str, Dict]:  # pragma: no cover — network
        """Request a bulk export and poll until the snapshot is 'fresh', then
        return the signed S3 zip link. Reuses PitCache's retry/backoff.

        ``include_metadata`` is used by the ACTIONS and candidate-grade
        SF1/SEP/TICKERS evidence paths.  The companion mapping intentionally
        excludes the signed link and contains only provider timestamps and the
        freshness status bound by the receipt.  The default remains the
        historical string return value for compatibility callers.
        """
        import time

        from data.pit_provider import PitCache, _api_key
        p = dict(params)
        p['qopts.export'] = 'true'
        p['api_key'] = _api_key()
        for _ in range(poll_attempts):
            body = PitCache._get_json(table, p)
            download = body['datatable_bulk_download']
            f = download['file']
            if f.get('status') == 'fresh':
                if include_metadata:
                    datatable = download.get('datatable') or {}
                    return f['link'], {
                        'last_refreshed_time': datatable.get(
                            'last_refreshed_time'),
                        'data_snapshot_time': f.get('data_snapshot_time'),
                        'status': f.get('status'),
                    }
                return f['link']
            logger.info("export %s: status=%s, waiting…", table, f.get('status'))
            time.sleep(poll_wait)
        raise RuntimeError(f"bulk export for {table} never became 'fresh'")

    @staticmethod
    def _activate_candidate_parquet(source: Path, destination: Path) -> None:
        """Atomically update the mutable cache after immutable publication."""
        temporary: str | None = None
        try:
            with source.open('rb') as incoming, tempfile.NamedTemporaryFile(
                    mode='wb', dir=destination.parent,
                    prefix=f'.{destination.name}.', delete=False) as outgoing:
                temporary = outgoing.name
                shutil.copyfileobj(incoming, outgoing, 1 << 20)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _ingest_candidate_table(
            self, logical_name: str, datatable: str, params: Dict) -> int:
        """Acquire SF1/SEP/TICKERS through immutable candidate evidence.

        Publication order is raw/converted archives, immutable receipt,
        mutable cache Parquet, then mutable active receipt.  A failure before
        the immutable receipt leaves at most unreferenced content-addressed
        files; a failure after it leaves a valid immutable acquisition even if
        the convenience cache has not advanced.
        """
        import requests

        if logical_name not in CANDIDATE_TABLES:
            raise ValueError(
                f"candidate ingest is limited to {list(CANDIDATE_TABLES)}")
        try:
            metadata = self._candidate_datatable_metadata(logical_name, datatable)
            export = self._export_link(
                datatable, params, include_metadata=True)
        except requests.exceptions.RequestException:
            raise SharadarSourceEvidenceError(
                f"{logical_name} candidate metadata/export request failed") from None
        if (
            not isinstance(export, tuple)
            or len(export) != 2
            or not isinstance(export[0], str)
            or not isinstance(export[1], Mapping)
        ):
            raise SharadarSourceEvidenceError(
                f"{logical_name} bulk export response is malformed")
        link, bulk_metadata = export
        expected_bulk_fields = {
            'last_refreshed_time', 'data_snapshot_time', 'status'
        }
        if (
            set(bulk_metadata) != expected_bulk_fields
            or bulk_metadata.get('status') != 'fresh'
        ):
            raise SharadarSourceEvidenceError(
                f"{logical_name} bulk export metadata is not fresh or complete")

        root = Path(self.warehouse_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                dir=root, prefix=f'.{logical_name}-candidate-ingest-') as temporary:
            staging = Path(temporary)
            raw_zip = staging / f'{logical_name}.zip'
            try:
                with requests.get(link, stream=True, timeout=600) as response:
                    response.raise_for_status()
                    with raw_zip.open('wb') as stream:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            if chunk:
                                stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
            except requests.exceptions.RequestException:
                raise SharadarSourceEvidenceError(
                    f"{logical_name} candidate bulk download failed") from None
            parquet = staging / f'{logical_name}.parquet'
            convert_sharadar_zip_to_parquet(
                raw_zip,
                parquet,
                logical_name=logical_name,
                datatable_metadata=metadata,
            )
            document = build_sharadar_table_acquisition_document(
                logical_name=logical_name,
                raw_zip_path=raw_zip,
                parquet_path=parquet,
                acquired_at_utc=(
                    datetime.now(timezone.utc).isoformat(timespec='seconds')
                    .replace('+00:00', 'Z')),
                last_refreshed_time=bulk_metadata['last_refreshed_time'],
                data_snapshot_time=bulk_metadata['data_snapshot_time'],
                datatable_metadata=metadata,
            )
            verified, _ = publish_sharadar_table_acquisition(
                root,
                raw_zip_path=raw_zip,
                parquet_path=parquet,
                document=document,
            )
            immutable_parquet = (
                root / verified['payload']['parquet']['relative_path'])
            self._activate_candidate_parquet(
                immutable_parquet, root / f'{logical_name}.parquet')
            atomic_write_json(
                root / f'{logical_name}.acquisition.json', verified)
            return int(
                verified['payload']['parquet']['statistics']['rows'])

    def candidate_source_snapshot(
            self, candidate_id: str, *,
            created_at_utc: str | None = None) -> Dict:
        """Publish an aggregate receipt from the three active acquisitions."""
        root = Path(self.warehouse_dir).resolve()
        acquisitions = {
            name: load_sharadar_table_acquisition(
                root / f'{name}.acquisition.json', warehouse_dir=root)
            for name in CANDIDATE_TABLES
        }
        created = created_at_utc or (
            datetime.now(timezone.utc).isoformat(timespec='seconds')
            .replace('+00:00', 'Z'))
        document = build_pead_sharadar_source_snapshot(
            warehouse_dir=root,
            candidate_id=candidate_id,
            created_at_utc=created,
            acquisitions=acquisitions,
        )
        verified, _ = publish_pead_sharadar_source_snapshot(root, document)
        atomic_write_json(root / 'sharadar_source_snapshot.json', verified)
        return verified

    def _ingest_actions(self, datatable: str, params: Dict) -> int:
        """Acquire ACTIONS with immutable raw and validated receipt evidence.

        All mutable active files remain untouched until the download,
        conversion, exact schema/statistics checks, raw archive, and receipt
        construction succeed.  Publication replaces the Parquet first and the
        receipt second; an interruption between those two steps is detectable
        and therefore fails closed at :meth:`corporate_action_evidence`.
        """
        import tempfile
        import zipfile

        import requests

        datatable_metadata = self._actions_datatable_metadata(datatable)
        export = self._export_link(
            datatable, params, include_metadata=True)
        if (
            not isinstance(export, tuple)
            or len(export) != 2
            or not isinstance(export[0], str)
            or not isinstance(export[1], Mapping)
        ):
            raise ActionsEvidenceError(
                "ACTIONS bulk export response is malformed")
        link, bulk_metadata = export
        expected_bulk_fields = {
            'last_refreshed_time', 'data_snapshot_time', 'status'
        }
        if (
            set(bulk_metadata) != expected_bulk_fields
            or bulk_metadata.get('status') != 'fresh'
        ):
            raise ActionsEvidenceError(
                "ACTIONS bulk export metadata is not fresh or complete")

        root = Path(self.warehouse_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                dir=root, prefix='.actions-ingest-') as temporary:
            staging = Path(temporary)
            zip_path = staging / 'actions.zip'
            with requests.get(link, stream=True, timeout=600) as response:
                response.raise_for_status()
                with zip_path.open('wb') as stream:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())

            zip_stats = inspect_actions_zip(zip_path)
            csv_path = staging / 'actions.csv'
            with zipfile.ZipFile(zip_path) as archive:
                with archive.open(zip_stats['member'], 'r') as incoming:
                    with csv_path.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 1 << 20)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())

            parquet_path = staging / 'actions.parquet'
            csv_sql = str(csv_path).replace("'", "''")
            parquet_sql = str(parquet_path).replace("'", "''")
            self._con().execute(
                f"COPY (SELECT * FROM read_csv_auto('{csv_sql}', "
                f"sample_size=-1)) TO '{parquet_sql}' (FORMAT PARQUET)",
            )
            # Validate before any active file is replaced.  The receipt builder
            # repeats the check while binding these exact bytes.
            statistics = inspect_actions_parquet(parquet_path)
            with parquet_path.open('rb') as stream:
                os.fsync(stream.fileno())

            archived_zip = archive_raw_zip(zip_path, root)
            document = build_actions_acquisition_document(
                parquet_path=parquet_path,
                raw_zip_path=archived_zip,
                raw_zip_relative_path=(
                    archived_zip.relative_to(root).as_posix()),
                acquired_at_utc=(
                    datetime.now(timezone.utc).isoformat(timespec='seconds')
                    .replace('+00:00', 'Z')),
                last_refreshed_time=bulk_metadata['last_refreshed_time'],
                data_snapshot_time=bulk_metadata['data_snapshot_time'],
                datatable_metadata=datatable_metadata,
            )
            receipt_rows = document['payload']['parquet']['statistics']['rows']
            if int(receipt_rows) != statistics['rows']:
                raise ActionsEvidenceError(
                    "ACTIONS receipt row count changed during acquisition")

            os.replace(parquet_path, root / 'actions.parquet')
            atomic_write_json(root / ACTIONS_RECEIPT_FILE, document)
            return int(receipt_rows)

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
        if name in CANDIDATE_TABLES:
            return self._ingest_candidate_table(name, datatable, params)
        if name == 'actions':
            return self._ingest_actions(datatable, params)
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
    ap.add_argument(
        '--candidate-id', default=None,
        help=(
            'after ingest, publish pead_sharadar_source_snapshot.v1 from the '
            'active candidate-grade SF1/SEP/TICKERS receipts'))
    cli = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    wh = PitWarehouse(cli.dir)
    for t in cli.tables:
        print(f"{t}: {wh.ingest_table(t)} rows -> {wh._pq(t)}")
    if cli.candidate_id:
        snapshot = wh.candidate_source_snapshot(cli.candidate_id)
        print(
            'source_snapshot: '
            f"{snapshot['artifact_hash']} -> "
            f"{Path(wh.warehouse_dir) / 'sharadar_source_snapshot.json'}")
