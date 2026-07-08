#!/usr/bin/env python
"""Repeatable desk backtest harness for the iterative quant loop.

Runs a ready desk over a regime-spanning window and prints the four headline
metrics (Total Return, Sharpe, Max Drawdown, Win Rate) plus the codebase's
research-integrity metrics (Deflated Sharpe, PSR) and a per-regime breakdown,
so successive tuning rounds are directly comparable. Each run is saved to
analysis/backtests/<desk>[_<model>].json.

Data flows through MarketDataHandler's OHLCVCache (trading_data.db): the first
run fetches from yfinance (a few minutes for ~100 names x 10y); every later run
reads the cache and is fast. NETWORK on first use, operator-run:

    python scripts/desk_backtest.py --desk aqr           # one desk, full window
    python scripts/desk_backtest.py --all                # leaderboard, all 6 desks
    python scripts/desk_backtest.py --desk twosigma --model ridge
    python scripts/desk_backtest.py --selftest           # offline, asserts helpers
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

# Run as a loose script (python scripts/desk_backtest.py) by putting the repo
# root on the path, mirroring how the package is imported elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from data.universe import LARGE_CAP_100  # noqa: E402
from desks.registry import create_desk, list_desks  # noqa: E402

# Full window spans diverse regimes; the slices below carve it into named
# regimes for a per-regime read (a desk profitable only in one regime is not
# robust). Dates are inclusive 'YYYY-MM-DD' string-comparable bounds.
FULL_START, FULL_END = '2015-01-01', '2024-12-31'
REGIMES = [
    ('2018 Q4 selloff', '2018-10-01', '2018-12-31'),
    ('2019 bull', '2019-01-01', '2019-12-31'),
    ('2020 COVID', '2020-02-01', '2020-12-31'),
    ('2021 mania', '2021-01-01', '2021-12-31'),
    ('2022 bear', '2022-01-01', '2022-12-31'),
    ('2023-24 AI bull', '2023-01-01', '2024-12-31'),
]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'analysis', 'backtests')


def _datestr(ts) -> str:
    """Snapshot timestamp -> 'YYYY-MM-DD' (mirrors manager.get_drawdown_series)."""
    return ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]


def _slice_metrics(history, start, end):
    """Return %-return and annualized Sharpe over a date slice of the run.

    history is portfolio_history: [{'timestamp', 'portfolio_value', ...}, ...].
    None when the slice has too few snapshots to be meaningful.
    """
    vals = [h['portfolio_value'] for h in history
            if start <= _datestr(h['timestamp']) <= end]
    if len(vals) < 2:
        return None
    rets = [vals[i] / vals[i - 1] - 1
            for i in range(1, len(vals)) if vals[i - 1] > 0]
    ret_pct = (vals[-1] / vals[0] - 1) * 100 if vals[0] else 0.0
    sd = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    sharpe = statistics.mean(rets) / sd * math.sqrt(252) if sd > 0 else 0.0
    return {'return_pct': ret_pct, 'sharpe': sharpe, 'days': len(vals)}


def _trades_in(closed, start, end):
    """Count, win-rate% and total PnL of trades that EXIT within a date slice."""
    sel = [t for t in closed if start <= t['exit_time'] <= end]
    wins = sum(1 for t in sel if t['pnl'] > 0)
    return {'n': len(sel),
            'win_rate': (wins / len(sel) * 100) if sel else 0.0,
            'pnl': sum(t['pnl'] for t in sel)}


def build_result_dict(desk_name, model_name, symbols, summary, closed,
                      regimes, oos_folds, start, end, t0):
    """The compact comparable result dict shared by run_desk and the
    proto_momentum / proto_hybrid research runners (which differ only in the
    hard-coded desk/model label). Single source of truth for the
    analysis/backtests JSON shape."""
    return {
        'desk': desk_name,
        'model': model_name,
        'window': [start, end],
        'n_symbols': len(symbols),
        'wall_seconds': round(time.time() - t0, 1),
        'trades': len(closed),
        'total_return': summary['total_return'],
        'total_return_pct': summary['total_return_pct'],
        'sharpe': summary['sharpe_ratio'],
        'deflated_sharpe': summary['deflated_sharpe'],
        'psr': summary['psr'],
        'max_drawdown_pct': summary['max_drawdown'],
        'win_rate': summary['win_rate'],
        'oos_folds': oos_folds,
        'regimes': regimes,
    }


def run_desk(key, symbols, start, end, model_key=None, capital=100_000.0):
    """Backtest one desk; return a compact comparable result dict."""
    desk = create_desk(key, model_key=model_key)
    engine = BacktestEngine(desk=desk, initial_capital=capital)
    t0 = time.time()
    report = engine.run(symbols, start, end)
    summary = report['summary']
    closed = report['closed_trades']
    regimes = {name: {**(_slice_metrics(report['portfolio_history'], s, e) or {}),
                      **_trades_in(closed, s, e)}
               for name, s, e in REGIMES}
    return build_result_dict(
        key, model_key, symbols, summary, closed, regimes,
        report.get('oos_folds', {'available': False}), start, end, t0)


def _fmt(x, nd=2):
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else str(x)


def _profitable(r) -> bool:
    """OOS-gated success rule: net-positive return, Sharpe>0.5, Deflated Sharpe>0."""
    ds = r['deflated_sharpe']
    return (r['total_return'] > 0 and r['sharpe'] > 0.5
            and ds is not None and ds > 0)


def print_result(r):
    print(f"\n{'=' * 70}\n{r['desk']}"
          + (f" [model={r['model']}]" if r['model'] else "")
          + f"  ({r['window'][0]} .. {r['window'][1]}, {r['n_symbols']} names, "
          f"{r['wall_seconds']}s)\n{'=' * 70}")
    print(f"  Trades         : {r['trades']}"
          + ("" if r['trades'] >= 100 else "  *** under 100 — widen universe/window ***"))
    print(f"  Total Return   : {_fmt(r['total_return_pct'])}%  (${_fmt(r['total_return'])})")
    print(f"  Sharpe         : {_fmt(r['sharpe'])}")
    print(f"  Deflated Sharpe: {_fmt(r['deflated_sharpe'])}   PSR: {_fmt(r['psr'])}")
    print(f"  Max Drawdown   : {_fmt(r['max_drawdown_pct'])}%")
    print(f"  Win Rate       : {_fmt(r['win_rate'])}%")
    print(f"  PROFITABLE (OOS-gated): {'YES' if _profitable(r) else 'no'}")
    print(f"  {'regime':<18}{'ret%':>9}{'sharpe':>9}{'trades':>8}{'win%':>8}")
    for name, _, _ in REGIMES:
        g = r['regimes'][name]
        print(f"  {name:<18}{_fmt(g.get('return_pct', 0)):>9}"
              f"{_fmt(g.get('sharpe', 0)):>9}{g.get('n', 0):>8}"
              f"{_fmt(g.get('win_rate', 0)):>8}")


def save_result(r):
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"{r['desk']}{'_' + r['model'] if r['model'] else ''}.json"
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w') as fh:
        json.dump(r, fh, indent=2, default=str)
    print(f"  saved -> {os.path.relpath(path)}")


def _selftest():
    """Offline asserts for the regime-math helpers (no network)."""
    hist = [{'timestamp': '2020-03-01', 'portfolio_value': 100.0},
            {'timestamp': '2020-03-02', 'portfolio_value': 110.0},
            {'timestamp': '2020-06-01', 'portfolio_value': 121.0}]
    m = _slice_metrics(hist, '2020-02-01', '2020-12-31')
    assert m and abs(m['return_pct'] - 21.0) < 1e-6, m
    assert _slice_metrics(hist, '2021-01-01', '2021-12-31') is None  # empty slice
    closed = [{'exit_time': '2020-04-01', 'pnl': 5.0},
              {'exit_time': '2020-04-02', 'pnl': -3.0},
              {'exit_time': '2021-01-01', 'pnl': 9.0}]
    t = _trades_in(closed, '2020-01-01', '2020-12-31')
    assert t == {'n': 2, 'win_rate': 50.0, 'pnl': 2.0}, t
    assert _profitable({'total_return': 1, 'sharpe': 0.6, 'deflated_sharpe': 0.1})
    assert not _profitable({'total_return': 1, 'sharpe': 0.6, 'deflated_sharpe': None})
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    keys = [d['key'] for d in list_desks()]
    ap.add_argument('--desk', choices=keys, help='single desk to backtest')
    ap.add_argument('--all', action='store_true', help='backtest every ready desk')
    ap.add_argument('--model', default=None, help='walk-forward model_key (foundation/twosigma)')
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--limit', type=int, default=None, help='subset the universe (quick smoke)')
    ap.add_argument('--selftest', action='store_true', help='offline helper asserts, then exit')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.all and not args.desk:
        ap.error('pass --desk <key> or --all (or --selftest)')

    symbols = LARGE_CAP_100[:args.limit] if args.limit else list(LARGE_CAP_100)
    targets = keys if args.all else [args.desk]
    results = []
    for key in targets:
        try:
            r = run_desk(key, symbols, args.start, args.end, model_key=args.model)
        except Exception as e:  # one desk failing must not sink the sweep
            print(f"\n!!! {key} FAILED: {type(e).__name__}: {e}")
            continue
        print_result(r)
        save_result(r)
        results.append(r)

    if len(results) > 1:
        print(f"\n{'=' * 70}\nLEADERBOARD (by Deflated Sharpe)\n{'=' * 70}")
        print(f"  {'desk':<12}{'trades':>8}{'ret%':>9}{'sharpe':>9}"
              f"{'dsharpe':>9}{'maxDD%':>9}{'win%':>8}  profit")
        for r in sorted(results, key=lambda x: (x['deflated_sharpe'] or -9), reverse=True):
            print(f"  {r['desk']:<12}{r['trades']:>8}{_fmt(r['total_return_pct']):>9}"
                  f"{_fmt(r['sharpe']):>9}{_fmt(r['deflated_sharpe']):>9}"
                  f"{_fmt(r['max_drawdown_pct']):>9}{_fmt(r['win_rate']):>8}"
                  f"  {'YES' if _profitable(r) else 'no'}")


if __name__ == '__main__':
    main()
