#!/usr/bin/env python
"""Capture independent Zacks earnings inputs as immutable PEAD evidence.

This is an operator-run network command.  It supports a deliberately
non-qualifying bounded historical sample, a licensed full-history acquisition,
and one append-only prospective capture.  All modes require explicit filters;
there is no Sharadar/SF1 fallback and no unbounded implicit download.

Examples::

    python -m scripts.ingest_earnings_announcements historical-sample \
      --start 2023-01-01 --end 2023-12-31 \
      --tables ES SS EEH SEH --ticker AAPL MSFT

    python -m scripts.ingest_earnings_announcements historical-full \
      --start 2015-01-01 --end 2024-12-31 --filters-json filters.json

    python -m scripts.ingest_earnings_announcements prospective \
      --start 2026-07-13 --end 2026-12-31 --ticker AAPL MSFT

The credential is read only from ``NASDAQ_DATA_LINK_API_KEY``.  Output includes
content hashes and local paths, never request credentials or response bodies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from data.earnings_announcements import (
    DEFAULT_CANDIDATE_ID,
    EarningsAnnouncementError,
    EarningsAnnouncementStore,
    SUPPORTED_MODES,
    SUPPORTED_TABLES,
    ZacksTablesClient,
)


DEFAULT_STORE = "research/pead_vq_locked_replication_v1/earnings_announcements"


def _strict_json_file(path: str) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate filters JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(token):
        raise ValueError(f"invalid filters JSON number: {token}")

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except OSError as exc:
        raise ValueError(f"cannot read filters JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("filters JSON is invalid") from exc


def load_filters(
        *, tables: Sequence[str], filters_json: str | None,
        tickers: Sequence[str] | None) -> dict[str, Mapping[str, Any]]:
    """Return an exact table->filters mapping from one explicit filter source."""
    if bool(filters_json) == bool(tickers):
        raise ValueError("choose exactly one of --filters-json or --ticker")
    selected = [str(table).strip().upper() for table in tables]
    if filters_json:
        value = _strict_json_file(filters_json)
        if not isinstance(value, Mapping) or set(value) != set(selected):
            raise ValueError("filters JSON must exactly cover selected tables")
        result: dict[str, Mapping[str, Any]] = {}
        for table in selected:
            filters = value[table]
            if not isinstance(filters, Mapping) or not filters:
                raise ValueError(f"filters for {table} must be a nonempty object")
            result[table] = dict(filters)
        return result
    assert tickers is not None
    if {"MT", "EA"}.intersection(selected):
        raise ValueError(
            "--ticker cannot safely query MT/EA stable identifiers; use --filters-json")
    normalized = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    if not normalized:
        raise ValueError("--ticker requires at least one nonempty ticker")
    value = ",".join(normalized)
    return {table: {"ticker": value} for table in selected}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ingest_earnings_announcements",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("mode", choices=SUPPORTED_MODES)
    parser.add_argument("--start", required=True, help="requested window start YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="requested window end YYYY-MM-DD")
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--tables", nargs="+", default=list(SUPPORTED_TABLES),
                        choices=list(SUPPORTED_TABLES))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--filters-json",
        help="JSON object mapping every selected table to explicit API filters")
    source.add_argument("--ticker", nargs="+", help="ticker filter applied to every table")
    parser.add_argument("--store-dir", default=DEFAULT_STORE)
    parser.add_argument("--per-page", type=int, default=10_000)
    parser.add_argument("--max-pages", type=int, default=1_000)
    parser.add_argument("--max-rows", type=int, default=10_000_000)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None, *, get=None, clock=None,
         environ=None, store_clock=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        filters = load_filters(
            tables=args.tables, filters_json=args.filters_json,
            tickers=args.ticker)
        if get is None:  # pragma: no cover - real network path
            import requests
            get = requests.get
        client = ZacksTablesClient(get=get, clock=clock, environ=environ)
        snapshot = client.capture(
            mode=args.mode,
            candidate_id=args.candidate_id,
            requested_start=args.start,
            requested_end=args.end,
            tables=args.tables,
            filters_by_table=filters,
            per_page=args.per_page,
            max_pages=args.max_pages,
            max_rows=args.max_rows,
            timeout=args.timeout,
        )
        stored = EarningsAnnouncementStore(
            args.store_dir, clock=store_clock).persist(snapshot)
    except (EarningsAnnouncementError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    coverage = snapshot.payload["coverage"]
    print(f"artifact_hash={stored.artifact_hash}")
    print(f"snapshot_path={stored.snapshot_path}")
    print(f"journal_event_hash={stored.journal_event_hash}")
    print(f"journal_event_path={stored.journal_event_path}")
    print(f"coverage_full_window={str(coverage['full_window']).lower()}")
    if coverage["blockers"]:
        print("coverage_blockers=" + ",".join(coverage["blockers"]))
    # Samples are expected to be incomplete and remain useful acquisition
    # evidence. Prospective/full modes fail closed when their declared window
    # is incomplete, while still persisting the evidence explaining why.
    if args.mode != "historical-sample" and not coverage["full_window"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
