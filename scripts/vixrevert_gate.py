#!/usr/bin/env python
"""Gate backtest for the VIX Reversion desk (docs/vix_pead_desks_spec.md).

Runs VixReversionDesk through the event-driven engine on SPY with ^VIX as the
auxiliary (never traded) signal series, then applies the single research gate
(analysis.research_stats.validate_strategy_oos): PSR(excess vs 2% risk-free)
>= 0.95 AND >= 1 Benjamini-Hochberg-significant out-of-sample calendar year.

NETWORK on first run (yfinance daily history via the OHLCVCache,
trading_data.db); cache-backed and fast thereafter. Operator-run:

    python scripts/vixrevert_gate.py                       # 2015-2024
    python scripts/vixrevert_gate.py --start 2005-01-01    # robustness window
    python scripts/vixrevert_gate.py --selftest            # offline asserts
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.research_stats import validate_strategy_oos  # noqa: E402
from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from desks.vix_revert import VixReversionDesk  # noqa: E402
from scripts.trend_follower_gate import (  # noqa: E402
    _daily_returns_with_years, print_report)

DEFAULT_SYMBOLS = ['SPY', '^VIX']
FULL_START, FULL_END = '2015-01-01', '2024-12-31'


def run_gate(symbols, start, end, *, capital=100_000.0, seed=42):
    """Backtest the desk and return (summary, gate_result, n_trades, report)."""
    desk = VixReversionDesk()
    engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed)
    report = engine.run(symbols, start, end)
    if 'error' in report:
        raise SystemExit(f"backtest failed: {report['error']}")
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = validate_strategy_oos(returns, years, psr_threshold=0.95)
    return report['summary'], gate, len(report['closed_trades']), report


def _selftest():
    """Offline asserts: the desk never trades the VIX symbol and enters only
    on a fresh crossing (pure-frame test, no engine/network)."""
    import numpy as np
    import pandas as pd
    from portfolio.manager import PortfolioManager

    idx = pd.bdate_range('2023-01-02', periods=80)
    vix_close = np.full(80, 15.0)
    vix_close[70:] = 25.0                      # spike on bar 70
    vix = pd.DataFrame({'close': vix_close}, index=idx)
    vix['sma_20'] = vix['close'].rolling(20).mean()
    spy = pd.DataFrame({'close': np.linspace(400, 420, 80)}, index=idx)

    desk = VixReversionDesk()
    pm = PortfolioManager(initial_capital=100_000.0)
    all_data = {'^VIX': vix.iloc[:71], 'SPY': spy.iloc[:71]}
    intents = desk.generate_intents(all_data, idx[70], pm)
    assert len(intents) == 1 and intents[0].action == 'BUY', intents
    assert intents[0].asset.symbol == 'SPY', intents
    # Next day the signal is still ON but there is no fresh crossing.
    all_data = {'^VIX': vix.iloc[:72], 'SPY': spy.iloc[:72]}
    assert desk.generate_intents(all_data, idx[71], pm) == []
    # A VIX-less universe degrades to a flat book with one info note.
    desk2 = VixReversionDesk()
    assert desk2.generate_intents({'SPY': spy.iloc[:71]}, idx[70], pm) == []
    assert desk2.notes and desk2.notes[-1].category == 'info'
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS)
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--selftest', action='store_true',
                    help='offline desk-behavior asserts, then exit')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    summary, gate, n_trades, _ = run_gate(
        args.symbols, args.start, args.end, seed=args.seed)
    print_report(summary, gate, n_trades, args.symbols, args.start, args.end,
                 title='VIX Reversion')


if __name__ == '__main__':
    main()
