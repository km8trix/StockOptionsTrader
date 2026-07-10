#!/usr/bin/env python
"""Ingest monthly ATM-straddle option DAILY bars from Massive (ex-Polygon).

Operator-run, NETWORK. Feeds the pre-registered VRP existence screen
(scripts/vrp_screen.py). For each underlying (default SPY QQQ IWM) and each
MONTHLY SELECTION DATE — the first trading session of each month,
2024-08 .. 2026-06, derived from the warehouse's OWN SIP minute bars
(``bars_1m_sip``; last regular-hours bar close per session) — make exactly:

  1. ONE reference call — ``GET /v3/reference/options/contracts`` with
     ``underlying_ticker``, ``as_of=<selection date>`` (point-in-time
     listing), ``expiration_date.gte/.lte`` spanning [21, 45] calendar DTE,
     ``strike_price.gte/.lte`` a ±10% band around that session's underlying
     close, ``limit=1000``; ``next_url`` pagination followed VERBATIM (only
     the Authorization header rides along). From the result pick the target
     CALL and PUT: expiry = the listed expiration in [21, 45] DTE closest to
     30 (ties -> the EARLIER expiry); strike = nearest to the underlying
     close among strikes listing BOTH a call and a put at that expiry
     (exact ties -> the LOWER strike).
  2. ONE aggregates call PER CONTRACT — ``GET /v2/aggs/ticker/O:<OCC>/range/
     1/day/<from>/<to>`` pulling the contract's FULL daily-bar life inside
     the authorized window (from = --bars-start, default 2024-07-01, the
     free tier's ~2-year floor; to = the contract's expiry).

Storage: ONE Parquet per underlying at
``<warehouse>/option_bars_eod/<UNDERLYING>.parquet`` with columns
``underlying, contract (OCC ticker), type, strike, expiry, selection_date,
ts, open, high, low, close, volume`` (``_OPTION_BARS_EOD_COLUMNS``), written
via ``PitWarehouse.write_option_bars_eod``. Registered as its OWN registry
entry (``data.pit_warehouse._OPTION_TABLES``) — the existing ``_TABLES`` /
``_INTRADAY_TABLES`` registries are byte-identical (pinned in tests).

DATA CAVEATS (documented, inherited by the screen):
  * FREE TIER HAS NO QUOTES/NBBO — only per-day aggregates of TRADES. The
    stored ``close`` is the day's last print; downstream, close stands in
    for mid, which is an APPROXIMATION compensated by a pre-registered
    spread haircut in scripts/vrp_screen.py. Real spreads are UNKNOWN here.
  * ~2-year rolling window: only ~2024-07 onward is authorized.
  * A month is SKIPPED (logged) when no expiration lands in the [21, 45]
    DTE window or no strike lists both legs — no substitution, no widening.

CONVENTIONS:
  * ``ts`` is the tz-naive ET SESSION DATE (epoch-ms UTC -> America/New_York
    -> normalized), same DST-correct conversion as the minute-bar ingest.
  * Underlying closes come from the SIP minute table: the close of the LAST
    bar starting inside 09:30-15:59 ET per session (bars are labeled by
    START time, so the 15:59 bar's close IS the 16:00 session close).
  * IDEMPOTENT per underlying: an existing Parquet is SKIPPED; ``--force``
    re-downloads and overwrites that underlying's file.
  * Rate-limit safe: reuses the hardened 429 sleep-and-retry helper
    (numeric Retry-After, floored, HTTP-date/junk -> default) from
    scripts/ingest_alpaca_bars.py. ``--pace`` (default 13s, ~4.6 req/min)
    additionally sleeps between calls so the ~5 req/min free tier is rarely
    hit at all; --pace 0 falls back to purely reactive 429 handling.

SECRETS: ``MASSIVE_API_KEY`` read from the environment by NAME only (lives
in .env like the other keys; never printed).

    set -a; source .env; set +a
    python scripts/ingest_massive_options.py                    # SPY QQQ IWM
    python scripts/ingest_massive_options.py --underlyings SPY --force
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pit_warehouse import _OPTION_BARS_EOD_COLUMNS, PitWarehouse  # noqa: E402
from scripts.ingest_alpaca_bars import (  # noqa: E402
    MAX_429_RETRIES, _massive_headers, _retry_after_seconds)

CONTRACTS_URL = 'https://api.polygon.io/v3/reference/options/contracts'
OPTION_AGGS_URL = ('https://api.polygon.io/v2/aggs/ticker/{contract}'
                   '/range/1/day/{start}/{end}')

UNDERLYINGS = ['SPY', 'QQQ', 'IWM']
START_MONTH, END_MONTH = '2024-08', '2026-06'
BARS_START, BARS_END = '2024-07-01', '2026-07-31'   # authorized ~2y window
MIN_DTE, MAX_DTE, TARGET_DTE = 21, 45, 30
STRIKE_BAND = 0.10          # reference query band: ±10% of the session close
REF_PAGE_LIMIT = 1000
AGGS_PAGE_LIMIT = 50_000
DEFAULT_PACE = 13.0         # seconds between calls (~4.6 req/min free tier)
SIP_TABLE = 'bars_1m_sip'
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)


# ---------------------------------------------------------------------------
# Underlying session closes from the SIP minute table (offline, warehouse)
# ---------------------------------------------------------------------------
def sip_session_closes(wh: PitWarehouse, symbol: str, start: str, end: str, *,
                       table: str = SIP_TABLE) -> pd.Series:
    """Daily REGULAR-HOURS session closes derived from SIP minute bars:
    per session, the close of the LAST bar whose start time is inside
    [09:30, 16:00) ET (bars are labeled by START time, so the 15:59 bar's
    close is the 16:00 close; extended-hours bars are ignored). Returns a
    float Series indexed by the normalized session Timestamp, ascending;
    empty for a never-ingested symbol."""
    bars = wh.ohlcv_intraday_range(symbol, start, end, table=table)
    if bars.empty:
        return pd.Series(dtype=float, name=symbol)
    tod = bars.index.time
    rth = bars[(tod >= RTH_OPEN) & (tod < RTH_CLOSE)]
    if rth.empty:
        return pd.Series(dtype=float, name=symbol)
    closes = rth.groupby(rth.index.normalize())['close'].last()
    closes.name = symbol
    return closes.sort_index()


def monthly_selection_dates(sessions, start_month: str, end_month: str) -> list:
    """First trading session of each calendar month in [start_month,
    end_month] (inclusive 'YYYY-MM' strings), from an ascending
    DatetimeIndex/iterable of session dates. Months with no session in the
    data are simply absent."""
    idx = pd.DatetimeIndex(sessions).sort_values()
    lo, hi = pd.Period(start_month, 'M'), pd.Period(end_month, 'M')
    out, seen = [], set()
    for ts in idx:
        p = ts.to_period('M')
        if lo <= p <= hi and p not in seen:
            seen.add(p)
            out.append(ts)
    return out


# ---------------------------------------------------------------------------
# Contract selection (pure, hermetically tested)
# ---------------------------------------------------------------------------
def pick_expiry(expiries, selection_date, *, lo: int = MIN_DTE,
                hi: int = MAX_DTE, target: int = TARGET_DTE):
    """Nearest-to-``target`` DTE expiration within [lo, hi] calendar days of
    ``selection_date``; ties on |DTE - target| -> the EARLIER expiry.
    Returns a datetime.date, or None when nothing lands in the window."""
    sel = pd.Timestamp(selection_date).date()
    best = None
    for e in expiries:
        d = pd.Timestamp(e).date()
        dte = (d - sel).days
        if not (lo <= dte <= hi):
            continue
        key = (abs(dte - target), dte)
        if best is None or key < best[0]:
            best = (key, d)
    return best[1] if best else None


def pick_strike(strikes, spot: float):
    """Nearest-to-``spot`` strike; exact distance ties -> the LOWER strike.
    Returns a float, or None for an empty strike set."""
    ss = sorted({float(k) for k in strikes})
    if not ss:
        return None
    return min(ss, key=lambda k: (abs(k - spot), k))


def select_straddle(contracts, selection_date, spot: float):
    """From reference-contract dicts (``ticker``, ``contract_type``,
    ``strike_price``, ``expiration_date``) pick the target ATM straddle:
    expiry via ``pick_expiry`` first, then the nearest-ATM strike AMONG
    STRIKES LISTING BOTH a call and a put at that expiry. Returns
    (call_dict, put_dict) or None when no expiry/strike qualifies."""
    by_expiry: dict = {}
    for c in contracts:
        d = pd.Timestamp(c['expiration_date']).date()
        by_expiry.setdefault(d, []).append(c)
    expiry = pick_expiry(by_expiry.keys(), selection_date)
    if expiry is None:
        return None
    calls = {float(c['strike_price']): c for c in by_expiry[expiry]
             if c['contract_type'] == 'call'}
    puts = {float(c['strike_price']): c for c in by_expiry[expiry]
            if c['contract_type'] == 'put'}
    k = pick_strike(set(calls) & set(puts), spot)
    if k is None:
        return None
    return calls[k], puts[k]


# ---------------------------------------------------------------------------
# Massive transport (injectable ``get``; 429 helper reused verbatim)
# ---------------------------------------------------------------------------
def massive_paged(url: str, params, *, get, headers: dict, sleep=time.sleep,
                  label: str = '') -> list:
    """GET a Massive endpoint accumulating ``results`` across ``next_url``
    pages (followed VERBATIM with NO params re-sent — only the Authorization
    header rides along). 429s sleep-and-retry the SAME url via the hardened
    Retry-After helper; any other non-2xx raises via raise_for_status."""
    items: list = []
    retries_429 = 0
    while True:
        resp = get(url, params=params, headers=headers, timeout=120)
        if resp.status_code == 429:
            retries_429 += 1
            if retries_429 > MAX_429_RETRIES:
                raise RuntimeError(f"{label}: gave up after "
                                   f"{MAX_429_RETRIES} consecutive 429s")
            wait = _retry_after_seconds(resp.headers.get('Retry-After'))
            print(f"  {label}: 429 rate-limited, sleeping {wait:.0f}s",
                  file=sys.stderr)
            sleep(wait)
            continue
        retries_429 = 0
        resp.raise_for_status()
        body = resp.json()
        items.extend(body.get('results') or [])
        nxt = body.get('next_url')
        if not nxt:
            return items
        url, params = nxt, None      # next_url verbatim, auth header only


def fetch_contracts(underlying: str, selection_date, spot: float, *,
                    get, headers: dict, sleep=time.sleep) -> list:
    """ONE reference call (plus pagination) listing calls AND puts as of the
    selection date, expiring [MIN_DTE, MAX_DTE] days out, strikes within
    ±STRIKE_BAND of the session close."""
    sel = pd.Timestamp(selection_date).date()
    params = {
        'underlying_ticker': underlying,
        'as_of': sel.isoformat(),
        'expiration_date.gte': (sel + dt.timedelta(days=MIN_DTE)).isoformat(),
        'expiration_date.lte': (sel + dt.timedelta(days=MAX_DTE)).isoformat(),
        'strike_price.gte': round(spot * (1.0 - STRIKE_BAND), 2),
        'strike_price.lte': round(spot * (1.0 + STRIKE_BAND), 2),
        'limit': REF_PAGE_LIMIT,
    }
    return massive_paged(CONTRACTS_URL, params, get=get, headers=headers,
                         sleep=sleep, label=f"{underlying} {sel} contracts")


def fetch_contract_bars(contract: str, start: str, end, *, get, headers: dict,
                        sleep=time.sleep) -> list:
    """ONE aggregates call (plus pagination) for a contract's daily bars
    over [start, end] — its full life inside the authorized window."""
    e = pd.Timestamp(end).date().isoformat()
    url = OPTION_AGGS_URL.format(contract=contract, start=start, end=e)
    return massive_paged(url, {'adjusted': 'false', 'limit': AGGS_PAGE_LIMIT},
                         get=get, headers=headers, sleep=sleep, label=contract)


def option_bars_frame(items, *, underlying: str, contract: str, ctype: str,
                      strike: float, expiry, selection_date) -> pd.DataFrame:
    """Massive daily-aggregate dicts -> option_bars_eod frame. ``t`` is
    epoch-ms UTC; converted to tz-naive ET and NORMALIZED to the session
    date (daily bars). Sorted by ts, de-duplicated on ts (keep first)."""
    cols = list(_OPTION_BARS_EOD_COLUMNS)
    if not items:
        return pd.DataFrame(columns=cols)
    ts = (pd.to_datetime([int(b['t']) for b in items], unit='ms', utc=True)
            .tz_convert('America/New_York').tz_localize(None).normalize())
    df = pd.DataFrame({
        'underlying': underlying,
        'contract': contract,
        'type': ctype,
        'strike': float(strike),
        'expiry': pd.Timestamp(expiry),
        'selection_date': pd.Timestamp(selection_date),
        'ts': ts,
        'open': [float(b['o']) for b in items],
        'high': [float(b['h']) for b in items],
        'low': [float(b['l']) for b in items],
        'close': [float(b['c']) for b in items],
        'volume': [int(b['v']) for b in items],
    })
    return (df.sort_values('ts')
              .drop_duplicates(subset='ts', keep='first')
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """CLI parser (module-level so tests can pin the defaults)."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--underlyings', nargs='+', default=UNDERLYINGS,
                    help='underlyings (default: SPY QQQ IWM)')
    ap.add_argument('--start-month', default=START_MONTH)
    ap.add_argument('--end-month', default=END_MONTH)
    ap.add_argument('--bars-start', default=BARS_START,
                    help='daily-bar life floor (authorized window start)')
    ap.add_argument('--bars-end', default=BARS_END,
                    help='underlying session-close scan end')
    ap.add_argument('--dir', default=None,
                    help='warehouse dir (else PIT_WAREHOUSE_DIR resolution)')
    ap.add_argument('--force', action='store_true',
                    help='re-download underlyings whose Parquet exists')
    ap.add_argument('--pace', type=float, default=DEFAULT_PACE,
                    help='seconds to sleep between API calls (free tier is '
                         '~5 req/min); 0 = reactive 429 handling only')
    return ap


def main() -> None:  # pragma: no cover — network CLI (pieces unit-tested)
    import requests

    args = build_parser().parse_args()
    headers = _massive_headers()
    wh = PitWarehouse(args.dir)

    def paced() -> None:
        if args.pace > 0:
            time.sleep(args.pace)

    for sym in args.underlyings:
        sym = sym.upper()
        path = wh._option_bars_pq(sym)
        if os.path.exists(path) and not args.force:
            print(f"{sym}: skip — {path} exists (use --force to redo)",
                  file=sys.stderr)
            continue
        closes = sip_session_closes(wh, sym, args.bars_start, args.bars_end)
        if closes.empty:
            print(f"{sym}: no {SIP_TABLE} sessions in the warehouse — run "
                  f"scripts/ingest_alpaca_bars.py --provider massive --table "
                  f"{SIP_TABLE} first", file=sys.stderr)
            continue
        sels = monthly_selection_dates(closes.index, args.start_month,
                                       args.end_month)
        print(f"{sym}: {len(sels)} monthly selection dates "
              f"({args.start_month}..{args.end_month})", file=sys.stderr)
        frames = []
        for sel in sels:
            spot = float(closes[sel])
            contracts = fetch_contracts(sym, sel, spot, get=requests.get,
                                        headers=headers)
            paced()
            pick = select_straddle(contracts, sel, spot)
            if pick is None:
                print(f"  {sym} {sel.date()}: no straddle in "
                      f"[{MIN_DTE},{MAX_DTE}] DTE — month skipped",
                      file=sys.stderr)
                continue
            call, put = pick
            print(f"  {sym} {sel.date()}: K={float(call['strike_price']):g} "
                  f"exp={call['expiration_date']} spot={spot:.2f}",
                  file=sys.stderr)
            for leg in (call, put):
                items = fetch_contract_bars(
                    leg['ticker'], args.bars_start, leg['expiration_date'],
                    get=requests.get, headers=headers)
                paced()
                frames.append(option_bars_frame(
                    items, underlying=sym, contract=leg['ticker'],
                    ctype=leg['contract_type'],
                    strike=float(leg['strike_price']),
                    expiry=leg['expiration_date'], selection_date=sel))
        df = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=list(_OPTION_BARS_EOD_COLUMNS)))
        n = wh.write_option_bars_eod(sym, df)
        if n == 0:
            print(f"{sym}: NO option bars — nothing written", file=sys.stderr)
        else:
            print(f"{sym}: {n} option bars -> {path}", file=sys.stderr)


if __name__ == '__main__':
    main()
