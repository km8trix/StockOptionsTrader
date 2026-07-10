#!/usr/bin/env python
"""Gate backtest for the PEAD desk (see docs/vix_pead_desks_spec.md).

Runs PEADDesk (true-SUE post-earnings drift, optionally one PIT size sleeve)
through the event-driven engine on the survivorship-free warehouse feed,
extracts the realized daily return series + calendar-year labels, and applies
the single research gate (analysis.research_stats.validate_strategy_oos):
PSR(excess vs risk-free) >= 0.95 AND >= 1 Benjamini-Hochberg-significant
out-of-sample year.

Book prices come from WarehouseMarketData (delisted names priced) and the SUE
signal from PIT SF1 filings — honest by construction. OPERATOR-run (requires
tickers/sep/sf1/daily ingested):

    python scripts/pead_desk_gate.py --limit 300                # fast smoke
    python scripts/pead_desk_gate.py --limit 600 --long-only    # long book only
    python scripts/pead_desk_gate.py --band small --limit 600   # one size sleeve
    python scripts/pead_desk_gate.py --selftest                 # offline assert

PEAD VARIANTS (--variant, 2026-07-10): strengthen the PEAD signal with the
merged SURGE screen fact (revenue surprise is real in micro, 63d net +2.96%
t=+3.23, and moderately independent of SUE — Spearman +0.34 on 215k matched
filings; Jegadeesh-Livnat 2006). base = default, byte-identical desk;
surge_confirm = longs must also clear the median fresh SURGE (missing SURGE
never excludes); rank_combine = equal-weight mean of the SUE and SURGE
percentile ranks (missing SURGE ranks on SUE alone). Full pre-registered
rules and constants: desks/pead.py module docstring. HONESTY: the TWO
non-base variants are TWO pre-registered trials this round — declared in
code before any variant backtest ran; results land here unedited, and no
further variant joins this round after seeing them.
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
from desks.pead import PEADDesk  # noqa: E402
from scripts.insider_desk_gate import (  # noqa: E402
    _daily_returns_with_years)
from scripts.insider_screen import SCALE_SMALL_MID, resolve_universe  # noqa: E402

FULL_START, FULL_END = '2015-01-01', '2024-12-31'

#: --variant -> PEADDesk constructor kwargs. 'base' is empty by construction
#: (byte-identical default desk); the two non-base entries are the round's
#: two pre-registered trials (desks/pead.py module docstring): the SURGE
#: median confirm-filter and the SUE+SURGE equal-weight rank combine —
#: rules and constants declared in the desk before any variant backtest
#: ran. scripts/vq_fund_gate.py imports this registry for the fund's PEAD
#: leg (--pead-variant), so solo and fund runs share ONE mapping. Pinned in
#: tests/test_pead.py — adding or tuning an entry must fail a test and be
#: argued in review, not slipped in.
PEAD_VARIANT_KWARGS = {
    'base': {},
    'surge_confirm': {'surge_confirm': True},
    'rank_combine': {'rank_combine': True},
}


def run_gate(band, start, end, *, limit=None, capital=100_000.0, seed=42,
             long_only=False, commission_bps=10.0, slippage_bps=5.0,
             dating='filing', variant='base'):
    wh = PitWarehouse()
    rb = pd.bdate_range(start, end, freq='BMS')
    universe = resolve_universe(wh, rb, SCALE_SMALL_MID)
    if limit and limit < len(universe):
        # seeded RANDOM sample (not the alphabetical head) so a subset is
        # representative of the small/mid universe, not a-names only.
        universe = sorted(random.Random(seed).sample(universe, limit))
    desk = PEADDesk(band, provider=wh, long_only=long_only, dating=dating,
                    **PEAD_VARIANT_KWARGS[variant])
    engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed,
                            commission=commission_bps / 1e4,
                            slippage_bps=slippage_bps,
                            market_data=WarehouseMarketData(wh))
    report = engine.run(universe, start, end)
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = validate_strategy_oos(returns, years, psr_threshold=0.95)
    return report['summary'], gate, len(report['closed_trades']), len(universe)


def print_report(summary, gate, n_trades, n_names, band, start, end,
                 variant='base'):
    label = ((band or 'all small/mid')
             + ('' if variant == 'base' else f" [pead={variant}]"))
    print(f"\n{'=' * 64}\nPEAD ({label}) — gate backtest "
          f"({start} .. {end}, {n_names} names)\n{'=' * 64}")
    print(f"  Trades         : {n_trades}"
          + ("" if n_trades >= 30 else "  *** thin book ***"))
    print(f"  Total Return   : {summary['total_return_pct']:.2f}%")
    print(f"  Sharpe         : {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown   : {summary['max_drawdown']:.2f}%")
    psr = gate['psr']
    print(f"\n  GATE: PSR(excess vs {gate.get('risk_free_rate', 0.02):.0%} rf) "
          ">= 0.95 AND >= 1 BH-significant OOS year")
    print(f"    PSR (vs risk-free)  : {psr:.4f}" if psr is not None
          else "    PSR (vs risk-free)  : n/a")
    print(f"    PSR pass (>=0.95)   : {gate['psr_pass']}")
    print(f"    OOS years tested    : {gate['n_periods_tested']}")
    print(f"    BH-significant years: {gate['bh']['n_significant_bh']}")
    print(f"\n  VERDICT: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"\n  {'year':<8}{'p-value':>10}{'reject':>9}")
    for label_, p, rej in zip(gate['fold_labels'], gate['fold_pvalues'],
                              gate['bh']['rejected_bh']):
        ps = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {str(label_):<8}{ps:>10}{str(rej):>9}")


def _selftest():
    """Offline: a hermetic provider drives PEADDesk end-to-end (no warehouse)."""
    import numpy as np

    class FakeProvider:
        def __init__(self, eps):
            self._eps = eps

        def eps_quarterly(self, tickers=None, *, asof=None):
            df = self._eps
            if tickers is not None:
                df = df[df['ticker'].isin(tickers)]
            if asof is not None:
                df = df[df['datekey'] <= pd.Timestamp(asof)]
            return df.reset_index(drop=True)

    rows = []
    for name, jump in (('AAA', 0.0), ('BBB', 4.0), ('CCC', -4.0), ('DDD', 0.0)):
        for q in range(12):
            rp = pd.Timestamp('2020-03-31') + pd.DateOffset(months=3 * q)
            rows.append({'ticker': name, 'reportperiod': rp,
                         'datekey': rp + pd.Timedelta(days=40),
                         'epsdil': 1.0 + 0.05 * q
                         + (jump if q == 11 else 0.0)
                         + 0.03 * ((q * 7 + hash(name) % 5) % 3 - 1)})
    eps = pd.DataFrame(rows)
    desk = PEADDesk(provider=FakeProvider(eps), quantile=0.25)
    idx = pd.bdate_range('2023-02-01', periods=15)
    frames = {n: pd.DataFrame({'close': np.full(15, 50.0)}, index=idx)
              for n in ('AAA', 'BBB', 'CCC', 'DDD')}
    asof = idx[-1]                              # after q11 datekeys, fresh
    scores = desk._alpha_scores(frames, asof)
    assert scores is not None and scores['BBB'] > 2 and scores['CCC'] < -2, scores
    # PIT: before the surprise quarter was filed, no q11 SUE is visible.
    desk2 = PEADDesk(provider=FakeProvider(eps), quantile=0.25)
    early = pd.Timestamp('2022-11-15')          # q10 filed, still fresh
    frames_early = {n: pd.DataFrame({'close': np.full(15, 50.0)},
                                    index=pd.bdate_range('2022-10-25',
                                                         periods=15))
                    for n in frames}
    s_early = desk2._alpha_scores(frames_early, early)
    if s_early is not None:
        assert abs(s_early.get('BBB', 0.0)) < 2, s_early
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--band', choices=['micro', 'small', 'mid'], default=None,
                    help='restrict to one PIT size tercile (default: all)')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe (fast smoke)')
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--long-only', action='store_true',
                    help='drop the short leg (long positive surprises only)')
    ap.add_argument('--commission-bps', type=float, default=10.0,
                    help='per-side commission in bps (engine default 10; '
                         'IBKR-tiered small-cap is ~2-5)')
    ap.add_argument('--slippage-bps', type=float, default=5.0,
                    help='per-side adverse slippage in bps (engine default 5)')
    ap.add_argument('--dating', choices=['filing', 'announce'],
                    default='filing',
                    help="SUE event timestamp: SF1 filing datekey (strict "
                         "PIT) or 8-K press-release date (needs events "
                         "ingested)")
    ap.add_argument('--variant', choices=sorted(PEAD_VARIANT_KWARGS),
                    default='base',
                    help='PEAD desk construction: base = byte-identical '
                         'default desk (unlabelled in the report); '
                         'surge_confirm = longs must clear the median fresh '
                         'SURGE; rank_combine = SUE+SURGE equal-weight rank '
                         'mean (desks/pead.py pre-registered rules)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    summary, gate, n_trades, n_names = run_gate(
        args.band, args.start, args.end, limit=args.limit, seed=args.seed,
        long_only=args.long_only, commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps, dating=args.dating,
        variant=args.variant)
    print_report(summary, gate, n_trades, n_names, args.band, args.start,
                 args.end, variant=args.variant)


if __name__ == '__main__':
    main()
