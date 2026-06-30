#!/usr/bin/env python
"""Universe/parameter sweep for the trend follower — an HONEST edge hunt.

Searching configurations IS multiple testing: the best of N configs must clear
a DEFLATED bar (deflated_sharpe with n_trials=N), not a single-trial PSR, or we
just overfit the search. This runs a grid of (universe x breakout_window x
atr_stop_mult), ranks by annualized Sharpe, then reports the BEST config's
deflated Sharpe — penalized for the whole search — plus its sub-period OOS
robustness (Benjamini-Hochberg across calendar years).

All Sharpe/PSR/DSR figures are on EXCESS-over-risk-free returns (RF=2%), matching
the production gate. NETWORK on first run (fetches each universe via the
OHLCVCache); cache-backed thereafter. Operator-run:

    python scripts/trend_follower_sweep.py
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.research_stats import (  # noqa: E402
    benjamini_hochberg, deflated_sharpe_ratio, fold_oos_pvalue,
    probabilistic_sharpe_ratio)
from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from desks.trend_follower import TrendFollowerDesk  # noqa: E402

RF = 0.02
PPY = 252
START, END = '2015-01-01', '2024-12-31'

UNIVERSES = {
    'indices': ['SPY', 'QQQ'],
    'sectors': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY'],
    'multiasset': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC'],
}
BREAKOUTS = [20, 50, 100]
ATRS = [2.5, 3.5]


def _datestr(ts):
    return ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]


def _excess_returns_years(history):
    """portfolio_history -> (excess daily returns, calendar-year labels)."""
    rf_pp = RF / PPY
    rows = [(_datestr(h['timestamp']), h['portfolio_value']) for h in history]
    ex, yr = [], []
    for i in range(1, len(rows)):
        pv = rows[i - 1][1]
        if pv and pv > 0:
            ex.append(rows[i][1] / pv - 1.0 - rf_pp)
            yr.append(rows[i][0][:4])
    return ex, yr


def _ann_sharpe(ex):
    if len(ex) < 2:
        return 0.0
    sd = statistics.pstdev(ex)
    return statistics.mean(ex) / sd * (PPY ** 0.5) if sd > 0 else 0.0


def run_one(symbols, breakout, atr):
    desk = TrendFollowerDesk(breakout_window=breakout, atr_stop_mult=atr)
    engine = BacktestEngine(desk=desk, initial_capital=100_000, seed=42)
    rep = engine.run(symbols, START, END)
    ex, yr = _excess_returns_years(rep['portfolio_history'])
    return rep['summary'], ex, yr, len(rep['closed_trades'])


def main():
    configs = [(u, b, a) for u in UNIVERSES for b in BREAKOUTS for a in ATRS]
    n_trials = len(configs)
    print(f"sweep: {n_trials} configs "
          f"({list(UNIVERSES)} x breakout {BREAKOUTS} x atr {ATRS})\n")

    results = []
    for (u, b, a) in configs:
        summary, ex, yr, ntr = run_one(UNIVERSES[u], b, a)
        psr = probabilistic_sharpe_ratio(ex, 0.0)
        sharpe = _ann_sharpe(ex)
        results.append({'u': u, 'breakout': b, 'atr': a, 'sharpe': sharpe,
                        'psr': psr, 'ret': summary['total_return_pct'],
                        'trades': ntr, 'ex': ex, 'yr': yr})
        print(f"  {u:11} b={b:>3} atr={a:>3}  sharpe={sharpe:+.2f} "
              f"psr={(psr if psr is not None else float('nan')):.3f} "
              f"ret={summary['total_return_pct']:+6.1f}%  trades={ntr}")

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    print("\nTOP 5 by annualized Sharpe (excess):")
    for r in results[:5]:
        print(f"  {r['u']:11} b={r['breakout']:>3} atr={r['atr']:>3}  "
              f"sharpe={r['sharpe']:+.2f} psr={r['psr']:.3f} ret={r['ret']:+.1f}%")

    best = results[0]
    dsr = deflated_sharpe_ratio(best['ex'], n_trials)  # penalize the search
    by_year = {}
    for r_, y_ in zip(best['ex'], best['yr']):
        by_year.setdefault(y_, []).append(r_)
    labels = sorted(by_year)
    pvals = [fold_oos_pvalue(by_year[y]) if len(by_year[y]) >= 20 else None
             for y in labels]
    bh = benjamini_hochberg(pvals)
    n_tested = sum(p is not None for p in pvals)
    passed = (dsr is not None and dsr >= 0.95) and bh['n_significant_bh'] >= 1

    print(f"\n{'=' * 60}\nHONEST DEFLATED VERDICT (best of {n_trials} configs)\n{'=' * 60}")
    print(f"  best config         : {best['u']} breakout={best['breakout']} "
          f"atr={best['atr']}  (Sharpe {best['sharpe']:+.2f}, {best['ret']:+.1f}%)")
    print(f"  raw PSR (vs rf, n=1): {best['psr']:.4f}")
    print(f"  DEFLATED Sharpe     : "
          f"{(dsr if dsr is not None else float('nan')):.4f}   (need >= 0.95)")
    print(f"  BH-significant years: {bh['n_significant_bh']} of {n_tested}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")


if __name__ == '__main__':
    main()
