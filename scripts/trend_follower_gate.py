#!/usr/bin/env python
"""Live gate backtest for the Trend Follower desk (PLAN.md step 4b).

Runs TrendFollowerDesk through the event-driven engine on the index ETFs,
extracts the realized daily return series + calendar-year labels, and applies
the single research gate (analysis.research_stats.validate_strategy_oos):
PSR(0) >= 0.95 AND >= 1 Benjamini-Hochberg-significant out-of-sample year.

Underlying-first (no options yet). NETWORK on first run (OpenBB/yfinance daily
history via the OHLCVCache, trading_data.db); cache-backed and fast thereafter.
Operator-run:

    python scripts/trend_follower_gate.py
    python scripts/trend_follower_gate.py --symbols SPY QQQ --start 2015-01-01 --end 2024-12-31
    python scripts/trend_follower_gate.py --selftest      # offline, asserts the return parser
"""

from __future__ import annotations

import argparse
import os
import sys

# Run as a loose script: put the repo root on the path (mirrors desk_backtest).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.research_stats import validate_strategy_oos  # noqa: E402
from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from desks.trend_follower import TrendFollowerDesk  # noqa: E402

DEFAULT_SYMBOLS = ['SPY', 'QQQ']
FULL_START, FULL_END = '2015-01-01', '2024-12-31'


def _datestr(ts) -> str:
    return ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]


def _daily_returns_with_years(history):
    """portfolio_history -> (daily returns, calendar-year label per return).

    history is [{'timestamp', 'portfolio_value', ...}, ...]; a return is the
    one-day fractional change in portfolio_value, labelled by the year of the
    LATER day. A non-positive prior value is skipped (no meaningful return).
    """
    rows = [(_datestr(h['timestamp']), h['portfolio_value']) for h in history]
    returns, years = [], []
    for i in range(1, len(rows)):
        prev_v = rows[i - 1][1]
        if prev_v and prev_v > 0:
            returns.append(rows[i][1] / prev_v - 1.0)
            years.append(rows[i][0][:4])
    return returns, years


def run_gate(symbols, start, end, *, capital=100_000.0, seed=42):
    """Backtest the desk and return (summary, gate_result, n_trades)."""
    desk = TrendFollowerDesk()
    engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed)
    report = engine.run(symbols, start, end)
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = validate_strategy_oos(returns, years, psr_threshold=0.95)
    return report['summary'], gate, len(report['closed_trades'])


def print_report(summary, gate, n_trades, symbols, start, end):
    print(f"\n{'=' * 64}\nTrend Follower — gate backtest "
          f"({start} .. {end}, {', '.join(symbols)})\n{'=' * 64}")
    print(f"  Trades         : {n_trades}"
          + ("" if n_trades >= 30 else "  *** thin book — few trades ***"))
    print(f"  Total Return   : {summary['total_return_pct']:.2f}%")
    print(f"  Sharpe         : {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown   : {summary['max_drawdown']:.2f}%")
    psr = gate['psr']
    psr_str = f"{psr:.4f}" if psr is not None else "n/a"
    rf = gate.get('risk_free_rate', 0.02)
    print(f"\n  GATE: PSR(excess vs {rf:.0%} risk-free) >= 0.95 "
          "AND >= 1 BH-significant OOS year")
    print(f"    PSR (vs risk-free)  : {psr_str}")
    print(f"    PSR pass (>=0.95)   : {gate['psr_pass']}")
    print(f"    OOS years tested    : {gate['n_periods_tested']}")
    print(f"    BH-significant years: {gate['bh']['n_significant_bh']}")
    print(f"    fold pass (>=1)     : {gate['fold_pass']}")
    print(f"\n  VERDICT: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"\n  {'year':<8}{'p-value':>10}{'reject':>9}")
    for label, p, rej in zip(gate['fold_labels'], gate['fold_pvalues'],
                             gate['bh']['rejected_bh']):
        ps = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {str(label):<8}{ps:>10}{str(rej):>9}")


def _selftest():
    """Offline assert for the return parser (no network)."""
    hist = [{'timestamp': '2020-12-30', 'portfolio_value': 100.0},
            {'timestamp': '2020-12-31', 'portfolio_value': 110.0},
            {'timestamp': '2021-01-04', 'portfolio_value': 121.0}]
    rets, years = _daily_returns_with_years(hist)
    assert years == ['2020', '2021'], years
    assert [round(r, 10) for r in rets] == [0.1, 0.1], rets
    # Non-positive prior value is skipped.
    guard = _daily_returns_with_years(
        [{'timestamp': '2020-01-01', 'portfolio_value': 0.0},
         {'timestamp': '2020-01-02', 'portfolio_value': 5.0}])
    assert guard == ([], []), guard
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
                    help='offline parser asserts, then exit')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    summary, gate, n_trades = run_gate(
        args.symbols, args.start, args.end, seed=args.seed)
    print_report(summary, gate, n_trades, args.symbols, args.start, args.end)


if __name__ == '__main__':
    main()
