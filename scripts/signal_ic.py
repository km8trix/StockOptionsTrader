#!/usr/bin/env python
"""Signal-edge (information-coefficient) diagnostics — measure alpha BEFORE a desk.

A desk's P&L blends signal quality with sizing, costs, turnover and netting, so a
losing desk can't tell you whether the SIGNAL had edge. This harness evaluates a
raw cross-sectional signal in isolation, leakage-free:

    IC_t(h) = rank-correlation across the universe between the signal on day t
              (computed from data <= t) and each name's forward return t -> t+h.

Reported per signal x horizon: mean IC, IC information-ratio (mean/std), a t-stat
(IC_IR * sqrt(n_days)), the hit rate (% of days IC > 0), the long-short quantile
spread (mean fwd return of the top minus the bottom quintile), and signal
turnover. A signal whose IC is ~0 / t-stat < ~2 has no edge — no portfolio
construction will save it, so don't build a desk around it.

NETWORK on first use (warms the shared OHLCV cache, like desk_backtest.py),
operator-run:

    python scripts/signal_ic.py                  # all built-in signals, full window
    python scripts/signal_ic.py --limit 30       # quick subset
    python scripts/signal_ic.py --selftest       # offline, asserts the IC math
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import MarketDataHandler  # noqa: E402
from data.universe import LARGE_CAP_100  # noqa: E402

FULL_START, FULL_END = '2015-01-01', '2024-12-31'
HORIZONS = (1, 5, 21)          # forward-return horizons in trading days
REBALANCE_STEP = 5             # evaluate IC every N trading days (cuts overlap)
QUANTILES = 5                  # quintiles for the long-short spread
MIN_NAMES = 10                 # need this many scored names for a cross-section
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'analysis', 'signal_ic')


# ----------------------------------------------------------------------
# Signals: each maps a wide close panel [dates x symbols] -> a same-shaped
# panel of scores using ONLY past data (no lookahead). These are the
# price-signal families the desks actually trade (momentum / reversal /
# low-vol), so their IC is a direct read on the desks' alpha DNA.
# ----------------------------------------------------------------------
def _signals():
    return {
        # 12-1 momentum (skip the most recent month) — AQR/foundation momentum.
        'mom_12_1': lambda c: c.shift(21) / c.shift(252) - 1.0,
        # 1-month momentum.
        'mom_21': lambda c: c / c.shift(21) - 1.0,
        # 5-day reversal — Renaissance short-horizon mean reversion.
        'reversal_5': lambda c: -(c / c.shift(5) - 1.0),
        # Low volatility (negated 21d realized vol) — AQR low-vol factor.
        'low_vol_21': lambda c: -c.pct_change().rolling(21).std(),
    }


def _forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    """Per-name return from t to t+h (future closes — the thing we predict)."""
    return close.shift(-h) / close - 1.0


def _rank_ic(sig_row: pd.Series, fwd_row: pd.Series):
    """Spearman IC for one cross-section (Pearson on ranks); None if degenerate."""
    df = pd.concat([sig_row, fwd_row], axis=1).dropna()
    if len(df) < MIN_NAMES:
        return None
    a = df.iloc[:, 0].rank()
    b = df.iloc[:, 1].rank()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _ic_stats(sig: pd.DataFrame, fwd: pd.DataFrame, dates) -> dict:
    ics = [ic for t in dates
           if (ic := _rank_ic(sig.loc[t], fwd.loc[t])) is not None]
    n = len(ics)
    if n < 2:
        return {'n_days': n, 'mean_ic': None, 'ic_ir': None,
                't_stat': None, 'hit_rate': None}
    arr = np.array(ics)
    mean_ic = float(arr.mean())
    sd = float(arr.std(ddof=1))
    ic_ir = mean_ic / sd if sd > 0 else 0.0
    return {
        'n_days': n,
        'mean_ic': mean_ic,
        'ic_ir': ic_ir,
        't_stat': ic_ir * math.sqrt(n),
        'hit_rate': float((arr > 0).mean()) * 100.0,
    }


def _quantile_spread(sig: pd.DataFrame, fwd: pd.DataFrame, dates):
    """Mean fwd return of the top quintile minus the bottom, averaged over days."""
    spreads = []
    for t in dates:
        df = pd.concat([sig.loc[t], fwd.loc[t]], axis=1).dropna()
        if len(df) < QUANTILES * 2:
            continue
        df.columns = ['s', 'r']
        buckets = pd.qcut(df['s'].rank(method='first'), QUANTILES, labels=False)
        top = df['r'][buckets == QUANTILES - 1].mean()
        bot = df['r'][buckets == 0].mean()
        spreads.append(top - bot)
    return float(np.mean(spreads)) * 100.0 if spreads else None


def _turnover(sig: pd.DataFrame, dates):
    """Mean fraction of the top quintile that rotates out each rebalance."""
    prev = None
    churn = []
    for t in dates:
        s = sig.loc[t].dropna()
        if len(s) < QUANTILES * 2:
            prev = None
            continue
        top = set(s.nlargest(max(1, len(s) // QUANTILES)).index)
        if prev is not None and prev:
            churn.append(1.0 - len(top & prev) / len(prev))
        prev = top
    return float(np.mean(churn)) * 100.0 if churn else None


def _fetch_close_panel(symbols, start, end) -> pd.DataFrame:
    handler = MarketDataHandler()
    cols = {}
    for sym in symbols:
        df = handler.fetch_stock_data(sym, start, end)
        if df is not None and not df.empty and 'close' in df.columns:
            cols[sym] = df['close']
    if not cols:
        raise SystemExit("No data fetched — check the universe / network / cache.")
    return pd.DataFrame(cols).sort_index()


def evaluate(close: pd.DataFrame) -> dict:
    dates = close.index[::REBALANCE_STEP]
    out = {}
    for name, fn in _signals().items():
        sig = fn(close)
        per_h = {}
        for h in HORIZONS:
            fwd = _forward_returns(close, h)
            common = [d for d in dates if d in sig.index and d in fwd.index]
            stats = _ic_stats(sig, fwd, common)
            stats['spread_pct'] = _quantile_spread(sig, fwd, common)
            per_h[h] = stats
        out[name] = {'horizons': per_h, 'turnover_pct': _turnover(sig, dates)}
    return out


def _fmt(x, nd=4):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else '  n/a'


def print_report(report, window, n_symbols):
    print(f"\n{'=' * 78}\nSignal IC  ({window[0]} .. {window[1]}, {n_symbols} names, "
          f"rebalance every {REBALANCE_STEP}d, quintile spreads)\n{'=' * 78}")
    print("  edge bar: |t-stat| >~ 2 and a monotone, same-signed quantile spread")
    for name, r in report.items():
        print(f"\n  {name}   (turnover {_fmt(r['turnover_pct'], 1)}%/rebalance)")
        print(f"    {'horizon':>8}{'mean_IC':>10}{'IC_IR':>9}{'t_stat':>9}"
              f"{'hit%':>8}{'LS_spread%':>12}")
        for h in HORIZONS:
            s = r['horizons'][h]
            print(f"    {str(h) + 'd':>8}{_fmt(s['mean_ic']):>10}{_fmt(s['ic_ir'], 2):>9}"
                  f"{_fmt(s['t_stat'], 2):>9}{_fmt(s['hit_rate'], 1):>8}"
                  f"{_fmt(s['spread_pct'], 3):>12}")


def save_report(report, window, n_symbols):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'signal_ic.json')
    with open(path, 'w') as fh:
        json.dump({'window': list(window), 'n_symbols': n_symbols,
                   'rebalance_step': REBALANCE_STEP, 'report': report},
                  fh, indent=2, default=str)
    print(f"\n  saved -> {os.path.relpath(path)}")


def _selftest():
    """Offline asserts: a perfectly predictive signal scores IC=+1, an
    inverted one IC=-1, and a constant one is degenerate (skipped)."""
    dates = pd.date_range('2020-01-01', periods=60, freq='B')
    syms = [f'S{i:02d}' for i in range(20)]
    rng = np.random.RandomState(0)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(syms))), 0)),
        index=dates, columns=syms)
    fwd = _forward_returns(close, 1)
    eval_dates = [d for d in close.index[:-1] if d in fwd.dropna(how='all').index]
    # Oracle signal == the forward return -> rank IC must be +1 each day.
    oracle = _ic_stats(fwd, fwd, eval_dates)
    assert oracle['mean_ic'] is not None and abs(oracle['mean_ic'] - 1.0) < 1e-9, oracle
    inverted = _ic_stats(-fwd, fwd, eval_dates)
    assert abs(inverted['mean_ic'] + 1.0) < 1e-9, inverted
    # A constant signal is degenerate -> every cross-section skipped.
    flat = pd.DataFrame(1.0, index=close.index, columns=syms)
    assert _ic_stats(flat, fwd, eval_dates)['n_days'] == 0
    # Quantile spread of the oracle is strongly positive.
    sp = _quantile_spread(fwd, fwd, eval_dates)
    assert sp is not None and sp > 0, sp
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--limit', type=int, default=None, help='subset the universe')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    symbols = LARGE_CAP_100[:args.limit] if args.limit else list(LARGE_CAP_100)
    close = _fetch_close_panel(symbols, args.start, args.end)
    report = evaluate(close)
    print_report(report, (args.start, args.end), close.shape[1])
    save_report(report, (args.start, args.end), close.shape[1])


if __name__ == '__main__':
    main()
