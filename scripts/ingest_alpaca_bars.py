#!/usr/bin/env python
"""Ingest 1-minute IEX bars from the Alpaca Data API v2 into the PitWarehouse.

Operator-run, NETWORK. Pulls ``GET https://data.alpaca.markets/v2/stocks/
{symbol}/bars?timeframe=1Min&feed=iex`` in 10k-bar pages (``next_page_token``)
and writes one Parquet per symbol under ``<warehouse>/bars_1m/<TICKER>.parquet``
via ``PitWarehouse.write_bars_1m`` (registered as ``bars_1m`` in
``data.pit_warehouse._INTRADAY_TABLES``; read back with
``PitWarehouse.ohlcv_intraday`` / ``ohlcv_intraday_range``).

Pre-registered pilot universe (CLI-overridable): SPY QQQ IWM TQQQ,
2016-01-01 .. 2024-12-31 — the ORB pilot (scripts/orb_gate.py).

DATA CAVEAT — IEX-only feed: Alpaca's free feed carries IEX prints only
(~2-3% of consolidated tape). Volume is heavily understated and intraday
high/low can understate the consolidated range, so breakout levels trigger
later / less often than on SIP data. Documented, not corrected — the ORB
gate inherits this caveat.

CONVENTIONS (documented, consistent everywhere):
  * ``ts`` is tz-NAIVE US/Eastern wall-clock, converted from Alpaca's UTC
    bar timestamps at ingest (bar labels are bar START times).
  * ALL bars returned by the feed are stored (extended hours included);
    session filtering (09:30-16:00) is the consumer's job.
  * ``adjustment=raw`` — actual traded prices. ORB is flat by close, so
    cross-day split consistency is irrelevant; within a day raw is exact.
  * IDEMPOTENT re-runs: a symbol whose Parquet already exists is SKIPPED;
    pass ``--force`` to re-download and overwrite that symbol's file.
  * Rate limit (~200 req/min): simple sleep-and-retry on HTTP 429, honoring
    a NUMERIC Retry-After when present; the HTTP-date form (or junk) falls
    back to RETRY_429_DEFAULT and every retry sleep is floored at
    MIN_RETRY_SLEEP so ``Retry-After: 0`` cannot hot-loop (amended
    2026-07-10).

MASSIVE (ex-Polygon) SIP PROVIDER — added 2026-07-10: ``--provider massive``
pulls CONSOLIDATED-TAPE (SIP) minute aggregates from ``GET https://
api.polygon.io/v2/aggs/ticker/{SYM}/range/1/minute/{from}/{to}`` with
``limit=50000&adjusted=false`` and header ``Authorization: Bearer
$MASSIVE_API_KEY``. Response items carry ``t`` as epoch-ms UTC (converted to
the SAME tz-naive US/Eastern convention as the Alpaca path), plus o/h/l/c/v/
n/vw — the SAME bars_1m schema, written via the same write_bars_1m path.
Pagination follows ``next_url`` VERBATIM (it carries the cursor; ONLY the
Authorization header is re-attached, no query params re-sent). Free tier:
~5 requests/min (429s sleep-and-retry like the Alpaca path) and ~2 years of
history. ``--table`` (default ``bars_1m``) picks the warehouse subdir —
``--table bars_1m_sip`` keeps SIP bars a SIBLING of the IEX pilot data so
neither ingest ever overwrites the other.

SECRETS: keys are read from the environment by NAME only —
``ALPACA_PAPER_API_KEY`` / ``ALPACA_PAPER_API_SECRET`` (the data API accepts
paper keys) and ``MASSIVE_API_KEY`` for ``--provider massive``. They live in
.env like the E*TRADE/Nasdaq keys; never printed.

    set -a; source .env; set +a
    python scripts/ingest_alpaca_bars.py                       # pilot universe
    python scripts/ingest_alpaca_bars.py --symbols SPY --force # one name, redo
    python scripts/ingest_alpaca_bars.py --provider massive \\
        --table bars_1m_sip --symbols SPY QQQ IWM TQQQ \\
        --start 2024-07-15 --end 2026-07-09                    # SIP sibling
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pit_warehouse import _BARS_1M_COLUMNS, PitWarehouse  # noqa: E402

BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
MASSIVE_AGGS_URL = ("https://api.polygon.io/v2/aggs/ticker/{symbol}"
                    "/range/1/minute/{start}/{end}")
PILOT_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'TQQQ']
FULL_START, FULL_END = '2016-01-01', '2024-12-31'
PAGE_LIMIT = 10_000
MASSIVE_PAGE_LIMIT = 50_000
RETRY_429_DEFAULT = 10.0   # seconds, when Retry-After is absent/unparseable
MIN_RETRY_SLEEP = 1.0      # floor — "Retry-After: 0" must not hot-loop
MAX_429_RETRIES = 30


def _retry_after_seconds(header_value) -> float:
    """``Retry-After`` header value -> seconds to sleep (numeric form only).

    The HTTP-date form ("Wed, 21 Oct 2026 07:28:00 GMT"), a missing header
    (None) or any other junk falls back to RETRY_429_DEFAULT; the result is
    floored at MIN_RETRY_SLEEP so ``Retry-After: 0`` cannot spin a hot
    retry loop.
    """
    try:
        wait = float(header_value)
    except (TypeError, ValueError):
        wait = RETRY_429_DEFAULT
    return max(wait, MIN_RETRY_SLEEP)


def _headers() -> dict:
    """Auth headers from the environment — values are never logged/printed."""
    key = os.environ.get('ALPACA_PAPER_API_KEY')
    secret = os.environ.get('ALPACA_PAPER_API_SECRET')
    if not key or not secret:
        raise SystemExit(
            "ALPACA_PAPER_API_KEY / ALPACA_PAPER_API_SECRET not set — ingest "
            "needs them (data API accepts paper keys). Load the repo .env "
            "first:  set -a; source .env; set +a")
    return {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret}


def _massive_headers() -> dict:
    """Massive auth header from the environment — value never logged/printed."""
    key = os.environ.get('MASSIVE_API_KEY')
    if not key:
        raise SystemExit(
            "MASSIVE_API_KEY not set — the massive provider needs it. Load "
            "the repo .env first:  set -a; source .env; set +a")
    return {'Authorization': f"Bearer {key}"}


def _bars_frame(ts, items, ticker: str) -> pd.DataFrame:
    """Shared frame assembly: tz-naive-ET ``ts`` + bar dicts (Alpaca and
    Massive use the SAME o/h/l/c/v/n/vw field names) -> bars_1m frame,
    sorted by ts and de-duplicated on ts (keep first)."""
    df = pd.DataFrame({
        'ticker': ticker,
        'ts': ts,
        'open': [float(b['o']) for b in items],
        'high': [float(b['h']) for b in items],
        'low': [float(b['l']) for b in items],
        'close': [float(b['c']) for b in items],
        'volume': [int(b['v']) for b in items],
        'trade_count': [int(b.get('n', 0)) for b in items],
        'vwap': [float(b.get('vw', 0.0)) for b in items],
    })
    return (df.sort_values('ts')
              .drop_duplicates(subset='ts', keep='first')
              .reset_index(drop=True))


def bars_to_df(items, ticker: str) -> pd.DataFrame:
    """Alpaca bar dicts -> bars_1m frame (columns ``_BARS_1M_COLUMNS``).

    ``t`` is RFC-3339 UTC; converted to tz-NAIVE US/Eastern wall-clock (the
    warehouse convention). Sorted by ts, de-duplicated on ts (keep first).
    """
    if not items:
        return pd.DataFrame(columns=list(_BARS_1M_COLUMNS))
    ts = (pd.to_datetime([b['t'] for b in items], utc=True)
            .tz_convert('America/New_York').tz_localize(None))
    return _bars_frame(ts, items, ticker)


def massive_bars_to_df(items, ticker: str) -> pd.DataFrame:
    """Massive aggregate dicts -> bars_1m frame (columns ``_BARS_1M_COLUMNS``).

    ``t`` is epoch-MILLISECONDS UTC; converted to the SAME tz-NAIVE
    US/Eastern wall-clock convention as the Alpaca path (DST-correct via
    America/New_York). Sorted by ts, de-duplicated on ts (keep first).
    """
    if not items:
        return pd.DataFrame(columns=list(_BARS_1M_COLUMNS))
    ts = (pd.to_datetime([int(b['t']) for b in items], unit='ms', utc=True)
            .tz_convert('America/New_York').tz_localize(None))
    return _bars_frame(ts, items, ticker)


def fetch_symbol_bars(symbol: str, start: str, end: str, *,
                      get, headers: dict, sleep=time.sleep,
                      page_limit: int = PAGE_LIMIT) -> pd.DataFrame:
    """Page through the bars endpoint for one symbol -> bars_1m frame.

    ``get`` is an injectable requests.get-compatible callable (tests pass a
    fake transport; the suite hard-blocks sockets). 429s sleep-and-retry the
    SAME page; any other non-2xx raises via raise_for_status.
    """
    params = {
        'timeframe': '1Min', 'feed': 'iex', 'adjustment': 'raw',
        'start': f"{start}T00:00:00Z", 'end': f"{end}T23:59:59Z",
        'limit': page_limit,
    }
    items: list = []
    token = None
    page = 0
    retries_429 = 0
    while True:
        p = dict(params)
        if token:
            p['page_token'] = token
        resp = get(BARS_URL.format(symbol=symbol), params=p,
                   headers=headers, timeout=120)
        if resp.status_code == 429:
            retries_429 += 1
            if retries_429 > MAX_429_RETRIES:
                raise RuntimeError(f"{symbol}: gave up after "
                                   f"{MAX_429_RETRIES} consecutive 429s")
            wait = _retry_after_seconds(resp.headers.get('Retry-After'))
            print(f"  {symbol}: 429 rate-limited, sleeping {wait:.0f}s",
                  file=sys.stderr)
            sleep(wait)
            continue
        retries_429 = 0
        resp.raise_for_status()
        body = resp.json()
        chunk = body.get('bars') or []
        items.extend(chunk)
        page += 1
        print(f"  {symbol}: page {page} (+{len(chunk)} bars, "
              f"total {len(items)})", file=sys.stderr)
        token = body.get('next_page_token')
        if not token:
            break
    return bars_to_df(items, symbol)


def fetch_symbol_bars_massive(symbol: str, start: str, end: str, *,
                              get, headers: dict, sleep=time.sleep,
                              page_limit: int = MASSIVE_PAGE_LIMIT
                              ) -> pd.DataFrame:
    """Page through the Massive aggregates endpoint for one symbol -> frame.

    Same injectable-transport shape as ``fetch_symbol_bars``. Pagination
    follows ``next_url`` VERBATIM (it carries the cursor) with NO query
    params re-sent — ONLY the Authorization header rides along. 429s
    sleep-and-retry the SAME URL (free tier is ~5 req/min).
    """
    url = MASSIVE_AGGS_URL.format(symbol=symbol, start=start, end=end)
    params: dict | None = {'limit': page_limit, 'adjusted': 'false'}
    items: list = []
    page = 0
    retries_429 = 0
    while True:
        resp = get(url, params=params, headers=headers, timeout=120)
        if resp.status_code == 429:
            retries_429 += 1
            if retries_429 > MAX_429_RETRIES:
                raise RuntimeError(f"{symbol}: gave up after "
                                   f"{MAX_429_RETRIES} consecutive 429s")
            wait = _retry_after_seconds(resp.headers.get('Retry-After'))
            print(f"  {symbol}: 429 rate-limited, sleeping {wait:.0f}s",
                  file=sys.stderr)
            sleep(wait)
            continue
        retries_429 = 0
        resp.raise_for_status()
        body = resp.json()
        chunk = body.get('results') or []
        items.extend(chunk)
        page += 1
        print(f"  {symbol}: page {page} (+{len(chunk)} bars, "
              f"total {len(items)})", file=sys.stderr)
        nxt = body.get('next_url')
        if not nxt:
            break
        url, params = nxt, None          # next_url verbatim, auth header only
    return massive_bars_to_df(items, symbol)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser (module-level so tests can pin the defaults)."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--symbols', nargs='+', default=PILOT_SYMBOLS,
                    help='pilot universe override (default: SPY QQQ IWM TQQQ)')
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--dir', default=None,
                    help='warehouse dir (else PIT_WAREHOUSE_DIR resolution)')
    ap.add_argument('--force', action='store_true',
                    help='re-download symbols whose Parquet already exists')
    ap.add_argument('--provider', choices=('alpaca', 'massive'),
                    default='alpaca',
                    help='bar source: alpaca (IEX, default) or massive '
                         '(consolidated tape / SIP)')
    ap.add_argument('--table', default='bars_1m',
                    help='warehouse intraday table (subdir) to write — use '
                         'bars_1m_sip for the Massive SIP sibling')
    return ap


def main() -> None:  # pragma: no cover — network CLI (pieces unit-tested)
    import requests

    args = build_parser().parse_args()

    if args.provider == 'massive':
        headers = _massive_headers()
        fetch = fetch_symbol_bars_massive
    else:
        headers = _headers()
        fetch = fetch_symbol_bars
    wh = PitWarehouse(args.dir)
    for sym in args.symbols:
        sym = sym.upper()
        path = wh._bars_pq(sym, args.table)
        if os.path.exists(path) and not args.force:
            print(f"{sym}: skip — {path} exists (use --force to redo)",
                  file=sys.stderr)
            continue
        feed = 'SIP' if args.provider == 'massive' else 'IEX'
        print(f"{sym}: fetching 1Min {feed} bars {args.start}..{args.end}",
              file=sys.stderr)
        df = fetch(sym, args.start, args.end,
                   get=requests.get, headers=headers)
        n = wh.write_bars_1m(sym, df, table=args.table)
        if n == 0:
            print(f"{sym}: NO bars returned — nothing written", file=sys.stderr)
        else:
            print(f"{sym}: {n} bars -> {path}", file=sys.stderr)


if __name__ == '__main__':
    main()
