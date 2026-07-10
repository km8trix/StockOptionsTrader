#!/usr/bin/env python
"""Gate backtests for the Value+Quality desk and the PEAD+VQ combined fund
(docs/vix_pead_desks_spec.md, unlock c: combine decorrelated survivors).

Modes:
  --mode vq        ValueQualityDesk solo through the engine (its own number).
  --mode issuance  IssuanceDesk solo through the engine (desk defaults:
                   long-only, ex-micro, 400d staleness) — the third-leg
                   candidate's own number.
  --mode fund      PEAD(micro, long-only) + ValueQuality(long-only) through
                   ReweightingFundBacktest (risk-parity, monthly rebalance,
                   shadow-solo-book) — N+1 backtests, the honest combine.
                   --weighting static instead runs ONE engine at fixed
                   construction weights (reweighter=None, no solo shadow
                   passes) — the reweighter-vs-cash-drag A/B arm. Fund mode
                   also prints post-hoc deployment fractions (1 - cash/NAV).
                   --issuance swaps in the pre-registered three-leg fund
                   (PEAD micro + VQ + issuance, equal 1/3 split — weights
                   fixed BEFORE the run, not tuned).

Both run on the survivorship-free warehouse feed with a seeded random
small/mid universe subset, and apply the single research gate
(analysis.research_stats.validate_strategy_oos). OPERATOR-run (requires
tickers/sep/sf1/daily ingested; events too when --dating announce):

    python scripts/vq_fund_gate.py --mode vq --limit 600
    python scripts/vq_fund_gate.py --mode fund --limit 600
    python scripts/vq_fund_gate.py --selftest

A/B RESULT (2026-07-10, --mode fund --limit 600 --dating announce, seed 42,
2015-2024, post key-fix): static 50/50 = 1382 trades, +96.9%, Sharpe 0.41,
maxDD -21.8%, PSR 0.9113 FAIL; TRUE risk parity (first ever run — the
pre-fix "risk-parity" numbers were this static fund via the silent
equal-weight fallback) = 1379 trades, +74.5%, Sharpe 0.35, maxDD -20.4%,
PSR 0.8704 FAIL. Inverse-vol tilting HURTS this two-leg fund: it overweights
the quieter leg into its flat stretches. Static weights are the better
configuration; both arms FAIL the 0.95/BH gate (0 BH years). Deployment
(mean ~0.62, min 0.39 in-run) bounds the cash-drag term: ~38% idle cash at
the gate's 2% rf costs ~0.76%/yr of excess return — the residual
engine-vs-paper Sharpe gap, now measured, not assumed.

3-LEG RESULT (2026-07-10, --issuance --weighting static --dating announce,
600 names seed 42): PEAD+VQ+issuance 1/3 each = 1732 trades, +73.1%, Sharpe
0.36, maxDD -20.8%, PSR 0.8784 FAIL (0 BH years) — WORSE than the two-leg
static 50/50. Solo issuance (--mode issuance): 489 trades, +45.4%, Sharpe
0.21, PSR 0.7415 FAIL — the screen's 8/8 BH edge attenuates hard long-only
ex-micro (the spread's short leg carried much of it). The weak leg dilutes
more than decorrelation adds; deployment mean 0.596. DO NOT mix the
issuance leg in this form — two-leg static stays the best honest fund.

VQ VARIANTS (--vq-variant, 2026-07-10): strengthen the VQ leg itself (the
PR #96 lesson: a weak third leg dilutes; the best honest fund is the
two-leg static 50/50, so the marginal PSR lives in the VQ leg's solo
Sharpe 0.35). base = default, byte-identical current desk; gpa = gp/assets
(Novy-Marx) rank added to the composite at equal rank weight; issfilter =
top-quintile YoY net issuers barred from VQ longs; both = the two composed.
Applied identically to --mode vq and to the fund's VQ leg. HONESTY: the
THREE non-base variants are THREE pre-registered trials this round —
declared (equal weights, top-quintile rank threshold, composition) before
any variant backtest ran; results land in the RESULT blocks below unedited,
and no further variant joins this round after seeing them.
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
from backtesting.reweighting_fund import ReweightingFundBacktest  # noqa: E402
from data.pit_warehouse import PitWarehouse  # noqa: E402
from data.warehouse_feed import WarehouseMarketData  # noqa: E402
from desks.citadel import CitadelDesk  # noqa: E402
from desks.foundation import FoundationDesk  # noqa: E402
from desks.issuance import IssuanceDesk  # noqa: E402
from desks.orchestrator import FundOrchestrator  # noqa: E402
from desks.pead import PEADDesk  # noqa: E402
from desks.renaissance import RenaissanceDesk  # noqa: E402
from desks.trend_follower import TrendFollowerDesk  # noqa: E402
from desks.value_quality import ValueQualityDesk  # noqa: E402
from portfolio.risk_manager import RiskManager  # noqa: E402
from scripts.insider_desk_gate import (  # noqa: E402
    _daily_returns_with_years)
from scripts.insider_screen import SCALE_SMALL_MID, resolve_universe  # noqa: E402

FULL_START, FULL_END = '2015-01-01', '2024-12-31'
#: Fund legs: the strongest realized PEAD config + the VQ composite, both
#: long-only (small-cap short legs bleed — insider/PEAD lesson), 50/50 start,
#: risk-parity reweighted in-run.
#:
#: Allocation keys MUST equal the constructed desks' .key (pinned in
#: tests/test_vq_fund_gate.py): ReweightingFundBacktest stores solo curves
#: under the allocation key while DynamicReweighter looks them up by
#: desk.key. Before 2026-07-10 this dict said 'pead' vs PEADDesk('micro').key
#: == 'pead_micro', so every rebalance silently degenerated to the
#: whole-fund equal-weight fallback — the recorded "risk-parity" PSR 0.9113
#: baseline is de facto a static 50/50 fund (reweight_log: fallback=True
#: throughout).
FUND_ALLOCATIONS = {'pead_micro': 0.5, 'value_quality': 0.5}
#: --issuance three-leg fund: the two incumbent legs + the issuance desk at a
#: PRE-REGISTERED equal split. The 1/3 weights are fixed here, before any
#: three-leg backtest ran — tuning them post hoc would turn the gate into an
#: in-sample fit. Keys MUST equal the constructed desks' .key (same contract
#: as FUND_ALLOCATIONS above; pinned in tests/test_vq_fund_gate.py).
THREE_LEG_ALLOCATIONS = {'pead_micro': 1 / 3, 'value_quality': 1 / 3,
                         'issuance': 1 / 3}
#: --legacy mix: the validated core keeps half the fund; the four stock-only
#: frozen legacy desks split the rest. Exercises the fund-wide ownership
#: scoping (Desk._owns_position) — legacy sweeps/exits must never close the
#: PEAD/VQ books. Jane Street is left out: its vol books need synthetic
#: options pricing (the documented premium-mispricing ruin) and its RV book
#: is inert without an ETF in the small/mid universe.
LEGACY_FUND_ALLOCATIONS = {'pead_micro': 0.25, 'value_quality': 0.25,
                           'foundation': 0.125, 'trend_follower': 0.125,
                           'renaissance': 0.125, 'citadel': 0.125}
#: --vq-variant -> ValueQualityDesk constructor kwargs. 'base' is empty by
#: construction (byte-identical default desk); the three non-base entries
#: are the round's three pre-registered trials (module docstring). Pinned in
#: tests/test_vq_fund_gate.py — adding or tuning an entry must fail a test
#: and be argued in review, not slipped in.
VQ_VARIANT_KWARGS = {
    'base': {},
    'gpa': {'include_gp_assets': True},
    'issfilter': {'issuance_filter': True},
    'both': {'include_gp_assets': True, 'issuance_filter': True},
}


def _make_desk(key, wh, *, capital_allocation=1.0, dating='filing',
               vq_variant='base'):
    if key == 'pead_micro':
        return PEADDesk('micro', provider=wh, long_only=True,
                        capital_allocation=capital_allocation, dating=dating)
    if key == 'value_quality':
        # ex-micro: the validated tradeable slice, AND disjoint from the
        # micro-band PEAD leg (overlapping books churn on a shared portfolio).
        return ValueQualityDesk(provider=wh, long_only=True,
                                capital_allocation=capital_allocation,
                                exclude_micro=True,
                                **VQ_VARIANT_KWARGS[vq_variant])
    if key == 'issuance':
        # Desk defaults ARE the fund slot: long_only=True, exclude_micro=True
        # (rides the same ex-micro slice as VQ — the screen's uniquely
        # surviving tradeable slice), stale_days=400.
        return IssuanceDesk(provider=wh,
                            capital_allocation=capital_allocation)
    if key == 'foundation':
        return FoundationDesk(capital_allocation=capital_allocation)
    if key == 'trend_follower':
        return TrendFollowerDesk(capital_allocation=capital_allocation)
    if key == 'renaissance':
        return RenaissanceDesk(capital_allocation=capital_allocation)
    if key == 'citadel':
        return CitadelDesk(capital_allocation=capital_allocation)
    raise ValueError(f"unknown fund leg {key!r}")


def _universe(wh, start, end, limit, seed):
    rb = pd.bdate_range(start, end, freq='BMS')
    universe = resolve_universe(wh, rb, SCALE_SMALL_MID)
    if limit and limit < len(universe):
        universe = sorted(random.Random(seed).sample(universe, limit))
    return universe


def run_vq(start, end, *, limit=None, capital=100_000.0, seed=42,
           long_only=False, vq_variant='base'):
    wh = PitWarehouse()
    universe = _universe(wh, start, end, limit, seed)
    desk = ValueQualityDesk(provider=wh, long_only=long_only,
                            **VQ_VARIANT_KWARGS[vq_variant])
    engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed,
                            market_data=WarehouseMarketData(wh))
    report = engine.run(universe, start, end)
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = validate_strategy_oos(returns, years, psr_threshold=0.95)
    return report['summary'], gate, len(report['closed_trades']), len(universe)


def run_issuance(start, end, *, limit=None, capital=100_000.0, seed=42):
    """Solo IssuanceDesk gate — run_vq's exact shape on the third-leg
    candidate at its desk defaults (long-only, ex-micro, 400d staleness)."""
    wh = PitWarehouse()
    universe = _universe(wh, start, end, limit, seed)
    desk = IssuanceDesk(provider=wh)
    engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed,
                            market_data=WarehouseMarketData(wh))
    report = engine.run(universe, start, end)
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = validate_strategy_oos(returns, years, psr_threshold=0.95)
    return report['summary'], gate, len(report['closed_trades']), len(universe)


def _deployment_stats(portfolio_history):
    """Mean/median/min deployed fraction (1 - cash/NAV) across the run's
    daily snapshots. Pure post-hoc reporting on fields the engine already
    records (portfolio/manager.record_snapshot) — the cash-drag diagnostic
    for the engine-vs-paper Sharpe gap. None when there are no snapshots.

    The day-1 snapshot is skipped: intents queue on the signal day and can
    only fill at the NEXT open, so cash == NAV on day one by construction
    and 'min' would always read 0.000 instead of measuring in-run drag.
    The truthiness guard drops NAV == 0 snapshots (ruin) rather than
    dividing by zero."""
    fracs = [1.0 - float(h['cash']) / float(h['portfolio_value'])
             for h in portfolio_history[1:] if float(h['portfolio_value'])]
    if not fracs:
        return None
    s = pd.Series(fracs)
    return {'mean': float(s.mean()), 'median': float(s.median()),
            'min': float(s.min())}


def run_fund(start, end, *, limit=None, capital=100_000.0, seed=42,
             dating='filing', legacy=False, weighting='risk_parity',
             issuance=False, vq_variant='base'):
    allocations = dict(THREE_LEG_ALLOCATIONS if issuance
                       else LEGACY_FUND_ALLOCATIONS if legacy
                       else FUND_ALLOCATIONS)
    wh = PitWarehouse()
    universe = _universe(wh, start, end, limit, seed)
    feed = WarehouseMarketData(wh)

    def solo_curve(key, symbols, s, e):
        desk = _make_desk(key, wh, dating=dating, vq_variant=vq_variant)
        engine = BacktestEngine(desk=desk, initial_capital=capital, seed=seed,
                                market_data=feed)
        report = engine.run(list(symbols), s, e, benchmark_symbol=None)
        if 'error' in report:
            return []
        return [(pd.Timestamp(h['timestamp']), float(h['portfolio_value']))
                for h in report.get('portfolio_history', [])]

    def orchestrator_factory(allocations):
        # Account-wide risk gate mirrors the legs' own config: both desks
        # run monthly signals under a wide 50% stop (their solo default);
        # the orchestrator's default RiskManager would 2%-stop the shared
        # book daily and churn both books to noise.
        return FundOrchestrator(
            [_make_desk(k, wh, capital_allocation=a, dating=dating,
                        vq_variant=vq_variant)
             for k, a in allocations.items()],
            risk_manager=RiskManager(position_stop_loss=0.50))

    if weighting == 'static':
        # A/B arm: reweighter=None makes the engine run the fund at its
        # construction-time weights unchanged (backtest_engine guard), so the
        # N expensive solo shadow passes are skipped. Same orchestrator
        # factory (wide 0.50 stop included), capital, seed, feed and default
        # cost model as the risk-parity arm — it differs ONLY in weighting.
        engine = BacktestEngine(
            orchestrator=orchestrator_factory(allocations), reweighter=None,
            initial_capital=capital, seed=seed, market_data=feed)
        report = engine.run(universe, start, end, benchmark_symbol=None)
    else:
        fund = ReweightingFundBacktest(
            allocations, initial_capital=capital, seed=seed,
            weighting='risk_parity', market_data=feed,
            solo_curve_provider=solo_curve,
            orchestrator_factory=orchestrator_factory)
        report = fund.run(universe, start, end, benchmark_symbol=None)
    if 'error' in report:
        raise SystemExit(f"fund backtest failed: {report['error']}")
    returns, years = _daily_returns_with_years(report['portfolio_history'])
    gate = validate_strategy_oos(returns, years, psr_threshold=0.95)
    n_trades = len(report.get('closed_trades', []))
    deploy = _deployment_stats(report.get('portfolio_history', []))
    return report['summary'], gate, n_trades, len(universe), deploy


def print_report(summary, gate, n_trades, n_names, label, start, end):
    print(f"\n{'=' * 64}\n{label} — gate backtest "
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
    for lab, p, rej in zip(gate['fold_labels'], gate['fold_pvalues'],
                           gate['bh']['rejected_bh']):
        ps = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {str(lab):<8}{ps:>10}{str(rej):>9}")


def _selftest():
    """Offline: ValueQualityDesk scores a synthetic cross-section correctly."""
    import numpy as np

    class FakeProvider:
        def fundamentals_quarterly(self, tickers=None, *, fields=('netmargin',),
                                   asof=None):
            rows = []
            for t, nm in (('CHEAPGOOD', 0.30), ('RICHBAD', -0.10),
                          ('MID1', 0.10), ('MID2', 0.05)):
                rows.append({'ticker': t,
                             'reportperiod': pd.Timestamp('2023-03-31'),
                             'datekey': pd.Timestamp('2023-05-10'),
                             'netmargin': nm})
            return pd.DataFrame(rows)

        def daily_fields_bulk(self, tickers, date, *, fields=('pb',)):
            pbs = {'CHEAPGOOD': 0.8, 'RICHBAD': 12.0, 'MID1': 3.0,
                   'MID2': 5.0}
            return {t: {'pb': pbs[t]} for t in tickers if t in pbs}

    desk = ValueQualityDesk(provider=FakeProvider(), quantile=0.25)
    idx = pd.bdate_range('2023-06-01', periods=5)
    frames = {n: pd.DataFrame({'close': np.full(5, 10.0)}, index=idx)
              for n in ('CHEAPGOOD', 'RICHBAD', 'MID1', 'MID2')}
    scores = desk._alpha_scores(frames, idx[-1])
    assert scores is not None
    assert scores['CHEAPGOOD'] == max(scores.values()), scores
    assert scores['RICHBAD'] == min(scores.values()), scores
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['vq', 'issuance', 'fund'],
                    default='fund')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--start', default=FULL_START)
    ap.add_argument('--end', default=FULL_END)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--long-only', action='store_true',
                    help='(vq mode) drop the short leg')
    ap.add_argument('--dating', choices=['filing', 'announce'],
                    default='filing', help='(fund mode) PEAD leg dating')
    ap.add_argument('--weighting', choices=['risk_parity', 'static'],
                    default='risk_parity',
                    help='(fund mode) risk_parity = in-run reweighting '
                         '(default, unchanged); static = fixed construction '
                         'weights, no reweighter, no solo shadow passes')
    ap.add_argument('--legacy', action='store_true',
                    help='(fund mode) mix the four stock-only frozen legacy '
                         'desks into the fund (ownership-scoping exercise)')
    ap.add_argument('--issuance', action='store_true',
                    help='(fund mode) run the pre-registered three-leg fund '
                         '(PEAD micro + VQ + issuance, equal 1/3 split)')
    ap.add_argument('--vq-variant', choices=sorted(VQ_VARIANT_KWARGS),
                    default='base',
                    help='VQ leg construction (vq AND fund modes): base = '
                         'byte-identical current desk (default, unlabelled '
                         'in the report); gpa = + gp/assets rank; issfilter '
                         '= top-quintile issuers barred from longs; both')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    vtag = '' if args.vq_variant == 'base' else f" [vq={args.vq_variant}]"
    if args.mode == 'vq':
        summary, gate, n_trades, n_names = run_vq(
            args.start, args.end, limit=args.limit, seed=args.seed,
            long_only=args.long_only, vq_variant=args.vq_variant)
        label = ('Value+Quality' + vtag
                 + (' (long-only)' if args.long_only else ''))
        deploy = None
    elif args.mode == 'issuance':
        if args.vq_variant != 'base':
            ap.error('--vq-variant has no effect in --mode issuance')
        summary, gate, n_trades, n_names = run_issuance(
            args.start, args.end, limit=args.limit, seed=args.seed)
        label = 'Net-Issuance (long-only, ex-micro)'
        deploy = None
    else:
        if args.issuance and args.legacy:
            ap.error('--issuance and --legacy select different allocation '
                     'mixes; pick one')
        summary, gate, n_trades, n_names, deploy = run_fund(
            args.start, args.end, limit=args.limit, seed=args.seed,
            dating=args.dating, legacy=args.legacy,
            weighting=args.weighting, issuance=args.issuance,
            vq_variant=args.vq_variant)
        core = ('PEAD+VQ+issuance' if args.issuance
                else 'PEAD+VQ+4 legacy desks' if args.legacy else 'PEAD+VQ')
        wlabel = ('static weights' if args.weighting == 'static'
                  else 'risk-parity')
        label = (f"{core} fund{vtag} ({wlabel}, long-only core legs, "
                 f"{args.dating} dating)")
    print_report(summary, gate, n_trades, n_names, label, args.start,
                 args.end)
    if deploy is not None:
        print(f"\n  Deployment (1 - cash/NAV): mean {deploy['mean']:.3f}"
              f"  median {deploy['median']:.3f}  min {deploy['min']:.3f}")


if __name__ == '__main__':
    main()
