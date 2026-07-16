#!/usr/bin/env python
"""Legacy diagnostic for the Insider Net-Buy desk.

Runs one PIT size sleeve (InsiderNetBuyDesk) through the event-driven engine on
the survivorship-free warehouse feed, extracts the realized daily return series +
calendar-year labels, and applies the single research gate
(analysis.research_stats.validate_strategy_oos): PSR(excess vs risk-free) >= 0.95
AND >= 1 Benjamini-Hochberg-significant out-of-sample year.

The book prices come from WarehouseMarketData (delisted names priced), and size
buckets use point-in-time daily_metric market cap.  However, this legacy driver
passes the *union* of dated listing membership to an engine whose universe is
static for the run.  Its results are therefore diagnostics only and are forced
to ``qualifying=False`` / ``passed=False`` even when the historical statistical
gate would pass. OPERATOR-run (reads the warehouse; requires
tickers/sep/sf2/daily ingested):

    python scripts/insider_desk_gate.py --band small --limit 300   # fast smoke
    python scripts/insider_desk_gate.py --band small               # full universe (slow)
    python scripts/insider_desk_gate.py --selftest                 # offline parser assert
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from analysis.research_stats import validate_strategy_oos  # noqa: E402
from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from data.pit_warehouse import PitWarehouse  # noqa: E402
from data.warehouse_feed import WarehouseMarketData  # noqa: E402
from desks.insider_netbuy import InsiderNetBuyDesk  # noqa: E402
from scripts.insider_screen import resolve_universe  # noqa: E402

FULL_START, FULL_END = '2015-01-01', '2024-12-31'
RETRIEVAL_UNION_NONQUALIFYING = (
    'static engine universe uses the union of dated membership; exact '
    'point-in-time eligibility is not enforced at every signal date'
)


def _datestr(ts) -> str:
    return ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]


def _daily_returns_with_years(history):
    """portfolio_history -> (daily returns, calendar-year label per return)."""
    rows = [(_datestr(h['timestamp']), h['portfolio_value']) for h in history]
    returns, years = [], []
    for i in range(1, len(rows)):
        prev_v = rows[i - 1][1]
        if prev_v and prev_v > 0:
            returns.append(rows[i][1] / prev_v - 1.0)
            years.append(rows[i][0][:4])
    return returns, years


def _mark_retrieval_union_nonqualifying(gate):
    """Fail closed when a legacy driver feeds a dated-universe union to the engine.

    Keep the statistical result as ``diagnostic_passed`` for reproducibility,
    but never expose it as promotion evidence through the conventional
    ``passed`` field.
    """
    result = dict(gate)
    result['diagnostic_passed'] = gate.get('passed')
    result['qualifying'] = False
    result['passed'] = False
    result['universe_membership_model'] = 'static_retrieval_union'
    result['nonqualifying_reasons'] = [RETRIEVAL_UNION_NONQUALIFYING]
    return result


def run_gate(band, start, end, *, limit=None, capital=100_000.0, seed=42,
             long_only=False):
    wh = PitWarehouse()
    rb = pd.bdate_range(start, end, freq='BMS')
    universe = resolve_universe(wh, rb)
    if limit and limit < len(universe):
        # seeded RANDOM sample (not the alphabetical head) so a subset is
        # representative of the full dated eligible universe, not a-names only.
        universe = sorted(random.Random(seed).sample(universe, limit))
    desk = InsiderNetBuyDesk(band, provider=wh, long_only=long_only)
    engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed,
                            market_data=WarehouseMarketData(wh))
    report = engine.run(universe, start, end)
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = _mark_retrieval_union_nonqualifying(
        validate_strategy_oos(returns, years, psr_threshold=0.95))
    return report['summary'], gate, len(report['closed_trades']), len(universe)


def print_report(summary, gate, n_trades, n_names, band, start, end):
    print(f"\n{'=' * 64}\nInsider Net-Buy ({band}-cap) — gate backtest "
          f"({start} .. {end}, {n_names} names)\n{'=' * 64}")
    print(f"  Trades         : {n_trades}"
          + ("" if n_trades >= 30 else "  *** thin book ***"))
    print(f"  Total Return   : {summary['total_return_pct']:.2f}%")
    print(f"  Sharpe         : {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown   : {summary['max_drawdown']:.2f}%")
    print("\n  NON-QUALIFYING DIAGNOSTIC: "
          + '; '.join(gate['nonqualifying_reasons']))
    psr = gate['psr']
    print(f"\n  GATE: PSR(excess vs {gate.get('risk_free_rate', 0.02):.0%} rf) "
          ">= 0.95 AND >= 1 BH-significant OOS year")
    print(f"    PSR (vs risk-free)  : {psr:.4f}" if psr is not None
          else "    PSR (vs risk-free)  : n/a")
    print(f"    PSR pass (>=0.95)   : {gate['psr_pass']}")
    print(f"    OOS years tested    : {gate['n_periods_tested']}")
    print(f"    BH-significant years: {gate['bh']['n_significant_bh']}")
    print("\n  VERDICT: NON-QUALIFYING (statistical diagnostic "
          f"would {'PASS' if gate['diagnostic_passed'] else 'FAIL'})")
    print(f"\n  {'year':<8}{'p-value':>10}{'reject':>9}")
    for label, p, rej in zip(gate['fold_labels'], gate['fold_pvalues'],
                             gate['bh']['rejected_bh']):
        ps = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {str(label):<8}{ps:>10}{str(rej):>9}")


def _selftest():
    hist = [{'timestamp': '2020-12-30', 'portfolio_value': 100.0},
            {'timestamp': '2020-12-31', 'portfolio_value': 110.0},
            {'timestamp': '2021-01-04', 'portfolio_value': 121.0}]
    rets, years = _daily_returns_with_years(hist)
    assert years == ['2020', '2021'], years
    assert [round(r, 10) for r in rets] == [0.1, 0.1], rets
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band', choices=['micro', 'small', 'mid'], default='small')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe (fast smoke; terciles then over the subset)')
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--long-only', action='store_true',
                    help='drop the short leg (long insider-buyers only)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    summary, gate, n_trades, n_names = run_gate(
        args.band, args.start, args.end, limit=args.limit, seed=args.seed,
        long_only=args.long_only)
    print_report(summary, gate, n_trades, n_names, args.band, args.start, args.end)


if __name__ == '__main__':
    main()
