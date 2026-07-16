"""Cross-sectional fundamental-factor screen — value factors on honest data.

Same apparatus as scripts/insider_screen.py (Fama-MacBeth per-date spread,
Newey-West/HAC t, Benjamini-Hochberg across the family, cost-netting), but the
signal is a continuous cross-sectional VALUE ratio from the PIT DAILY table
(point-in-time by nature) rather than the binary insider sign. At each monthly
rebalance, rank the survivorship-free dated eligible universe by each ratio,
long the CHEAP quantile and short the EXPENSIVE, and test the monthly
long-short spread.

Value ratios (all from daily_metric; lower = cheaper = the value bet):
  pb, pe, ps, evebit, evebitda   (non-positive values dropped — value traps /
  negative-earnings noise are not a clean "cheap").

Deterministic; reads the warehouse (tickers + sep + daily ingested). Quality
factors live in SF1 — see scripts/quality_screen.py.

Raw forward-return P&L is always retained. ``--winsor 0.01`` additionally builds
a per-date winsorized series for robust inference because this universe's raw
returns reach +2000x (micro-cap / split artifacts). Winsorized returns are NEVER
reported as executable economics: the report shows raw gross/net P&L separately
from robust net spread/t/p. Costs are deducted from EVERY date's spread before
the HAC test, and BH consumes one-sided NET p-values for H1: mean net spread > 0.
Earlier screen reports instead tested gross spreads while only shifting the
displayed mean; their t statistics and BH survivor counts are legacy diagnostics
and must be rerun with this code. They also deducted only two 30bp charges while
downstream CLIs described 30bp as one-way: the corrected fixed model charges
four trades (enter/exit long and short), so their net means are stale too.

CAVEAT — even robust inference can overstate the tradeable edge. The validation
battery in scripts/value_validate.py shows it is (1) OOS-only (a 2020+ regime
phenomenon, nothing 2015-2019), (2) micro-concentrated (the tradeable small/mid
ex-micro slice is only t~1.7), and (3) ~2 independent bets, not 5. Read that
verdict before graduating value to a desk.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.research_stats import benjamini_hochberg  # noqa: E402
from data.pit_warehouse import PitWarehouse  # noqa: E402
from scripts.insider_screen import (  # noqa: E402
    _eligible_on,
    resolve_universe_membership,
    universe_union,
)

VALUE_FACTORS = ['pb', 'pe', 'ps', 'evebit', 'evebitda']


def collect_factor_events(prov, names, rebal_dates, horizons, factors,
                          price_start, price_end, *, membership_by_date=None):
    """One row per (name, rebalance): the PIT factor values + forward returns."""
    maxh = max(horizons)
    recs = []
    n_names = 0
    batch = getattr(prov, 'daily_metrics', None)
    for name in names:
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        idx, vals = px.index, px.values
        # Cheap bounds/entry checks FIRST so out-of-range dates never pay a
        # DAILY query; then ONE batched fetch per name instead of one
        # point query per rebalance date.
        live = []
        for t in rebal_dates:
            if not _eligible_on(name, t, membership_by_date):
                continue
            pos = int(idx.searchsorted(pd.Timestamp(t)))
            if pos + maxh >= len(px) or pos >= len(px):
                continue
            entry = vals[pos]
            if not entry or entry <= 0:
                continue
            live.append((t, pos, entry))
        if not live:
            continue
        n_names += 1
        if batch is not None:
            rows = batch(name, [t for t, _, _ in live])
        else:
            rows = [prov.daily_metric(name, t) for t, _, _ in live]
        for (t, pos, entry), metrics in zip(live, rows):
            if not metrics:                          # PIT valuation row or None
                continue
            rec = {'date': pd.Timestamp(t), 'name': name}
            for f in factors:
                rec[f] = metrics.get(f)
            for h in horizons:
                rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
            recs.append(rec)
    cols = (['date', 'name'] + list(factors)
            + [f'fwd_{h}' for h in horizons])
    return pd.DataFrame(recs, columns=cols), n_names


def _hac_mean_test(values, lag):
    """Return mean, HAC t, one-sided p(+), and two-sided diagnostic p.

    The research hypothesis is directional: positive net spread. Statsmodels'
    robust-covariance p-value is symmetric/two-sided, so convert it to the
    corresponding upper-tail value. A large negative t therefore has p(+) near
    one and can never be mistaken for a BH-significant positive edge.
    """
    arr = np.asarray(values, dtype=float)
    fit = sm.OLS(arr, np.ones(len(arr))).fit(
        cov_type='HAC', cov_kwds={'maxlags': lag})
    t_value = float(fit.tvalues[0])
    p_two_sided = float(fit.pvalues[0])
    p_greater = (p_two_sided / 2.0 if t_value >= 0
                 else 1.0 - p_two_sided / 2.0)
    return float(arr.mean()), t_value, p_greater, p_two_sided


def _period_cost_drags(dates, cost_bps, spread_cost_bps_by_date):
    """Resolve total long-short return drag for every formation date.

    The scalar ``cost_bps`` is a one-way cost for one trade on one leg.  Each
    forward long-short cohort enters and exits both its long and short legs, so
    the full-turnover approximation charges four trades, ``4 * cost_bps``, per
    period. ``spread_cost_bps_by_date`` is an
    optional mapping/Series/callable of TOTAL spread-level basis-point drag and
    overrides that assumption. It lets callers supply realized long+short
    turnover, slippage, borrow, and financing costs without changing the legacy
    call signature. Missing, negative, or non-finite dated costs fail closed.
    """
    dates = pd.DatetimeIndex(dates)
    if spread_cost_bps_by_date is None:
        if not np.isfinite(cost_bps) or cost_bps < 0:
            raise ValueError('cost_bps must be finite and non-negative')
        bps = np.full(len(dates), 4.0 * float(cost_bps), dtype=float)
        return bps / 1e4, 'fixed_one_way_four_trade_round_trip'

    if callable(spread_cost_bps_by_date):
        bps = np.asarray([spread_cost_bps_by_date(t) for t in dates],
                         dtype=float)
    elif np.isscalar(spread_cost_bps_by_date):
        bps = np.full(len(dates), float(spread_cost_bps_by_date), dtype=float)
    else:
        costs = pd.Series(spread_cost_bps_by_date, dtype=float)
        try:
            costs.index = pd.DatetimeIndex(pd.to_datetime(costs.index))
        except (TypeError, ValueError) as exc:
            raise ValueError('dated spread costs require date-like keys') from exc
        if costs.index.has_duplicates:
            raise ValueError('dated spread costs contain duplicate dates')
        aligned = costs.reindex(dates)
        if aligned.isna().any():
            missing = ', '.join(t.date().isoformat()
                                for t in dates[aligned.isna()][:3])
            raise ValueError(f'missing spread cost for formation date(s): {missing}')
        bps = aligned.to_numpy(dtype=float)
    if not np.isfinite(bps).all() or (bps < 0).any():
        raise ValueError('spread costs must be finite and non-negative')
    return bps / 1e4, 'dated_total_spread_cost'


def factor_study(events, factor, horizons, cost_bps, *, quantile=0.2,
                 cheap_is_long=True, min_dates=12, drop_nonpositive=True,
                 winsor_returns=None, spread_cost_bps_by_date=None,
                 include_series=False):
    """PURE Fama-MacBeth long-short by one cross-sectional factor.

    Per rebalance date, rank names by ``factor`` (dropping non-positive values),
    long the cheap quantile / short the expensive, spread = mean(long fwd) -
    mean(short fwd); test the monthly spread series with a Newey-West (HAC) t.
    Returns a list of per-horizon dicts. (cheap_is_long: for value ratios the LOW
    end is the long. drop_nonpositive: True for value ratios where a non-positive
    value is a trap, not a "cheap"; set False for profitability factors where a
    negative value — an unprofitable firm — is a legitimate short-leg member.
    winsor_returns: if set (e.g. 0.01), clip EACH DATE's forward returns to the
    [w, 1-w] quantiles before averaging — useful robust inference on this
    survivorship-free micro-cap universe where raw forward returns reach +2000x.
    The raw spread is ALWAYS retained as the economic result; clipping never
    rewrites executable P&L. Default None = no robust series.

    ``spread_cost_bps_by_date`` optionally supplies TOTAL strategy-return cost
    in bps for each formation date (mapping/Series/callable). When absent, the
    fixed full-turnover assumption deducts ``4 * cost_bps`` each period: entry
    and exit on both the long and short legs.

    ``include_series`` retains the complete dated gross, cost, and net arrays
    used by inference. It is off by default to keep ordinary screen summaries
    compact; persisted evidence should enable it so every statistic can be
    independently recomputed.

    Compatibility aliases ``gross_spread``, ``net_spread``, ``t``, and ``p``
    always describe the unmodified executable series. Critically, ``t``/``p``
    test the per-date RAW NET spreads and ``p`` is the one-sided upper-tail
    value for a positive edge. ``p_two_sided`` is retained only as a diagnostic.
    Explicit ``robust_*`` keys contain winsorized sensitivity analysis; those
    diagnostics never drive BH or a promotion verdict.
    """
    if winsor_returns is not None:
        winsor_returns = float(winsor_returns)
        if winsor_returns == 0:
            winsor_returns = None
        elif not 0 < winsor_returns < 0.5:
            raise ValueError('winsor_returns must be between 0 and 0.5')
    out = []
    for h in horizons:
        col = f'fwd_{h}'
        df = events.dropna(subset=[factor, col])
        if drop_nonpositive:
            df = df[df[factor] > 0]                    # drop value traps / neg
        if len(df) < 20:
            out.append({'factor': factor, 'h': h, 'insufficient': True})
            continue
        spreads = []
        for date, g in df.groupby('date'):
            n = len(g)
            k = max(1, int(quantile * n))
            if n < 2 * k:
                continue                              # can't form both legs
            g = g.sort_values(factor)                 # ascending: cheap first
            raw = g[col]
            raw_cheap = raw.iloc[:k].mean()
            raw_rich = raw.iloc[-k:].mean()
            raw_spread = ((raw_cheap - raw_rich) if cheap_is_long
                          else (raw_rich - raw_cheap))
            robust_spread = raw_spread
            if winsor_returns:
                robust = raw.clip(raw.quantile(winsor_returns),
                                  raw.quantile(1.0 - winsor_returns))
                robust_cheap = robust.iloc[:k].mean()
                robust_rich = robust.iloc[-k:].mean()
                robust_spread = ((robust_cheap - robust_rich) if cheap_is_long
                                 else (robust_rich - robust_cheap))
            spreads.append((pd.Timestamp(date), raw_spread, robust_spread))
        n_dates = len(spreads)
        if n_dates < min_dates:
            out.append({'factor': factor, 'h': h, 'n_dates': n_dates,
                        'insufficient': True})
            continue
        dates = [x[0] for x in spreads]
        raw_gross = np.asarray([x[1] for x in spreads], dtype=float)
        robust_gross = np.asarray([x[2] for x in spreads], dtype=float)
        cost_drags, cost_model = _period_cost_drags(
            dates, cost_bps, spread_cost_bps_by_date)
        raw_net = raw_gross - cost_drags
        robust_net = robust_gross - cost_drags
        lag = max(1, int(np.ceil(h / 21)))
        raw_gross_stats = _hac_mean_test(raw_gross, lag)
        raw_net_stats = _hac_mean_test(raw_net, lag)
        result = {
            'factor': factor, 'h': h, 'n_dates': n_dates,
            'raw_gross_spread': raw_gross_stats[0],
            'raw_net_spread': raw_net_stats[0],
            'raw_gross_t': raw_gross_stats[1],
            'raw_gross_p': raw_gross_stats[2],
            'raw_gross_p_two_sided': raw_gross_stats[3],
            'raw_net_t': raw_net_stats[1],
            'raw_net_p': raw_net_stats[2],
            'raw_net_p_two_sided': raw_net_stats[3],
            'mean_total_cost_bps': float(cost_drags.mean() * 1e4),
            'cost_model': cost_model,
            'inference_basis': 'raw_net',
            'inference_is_executable': True,
        }
        if include_series:
            result['inference_series'] = {
                'formation_dates': [date.date().isoformat() for date in dates],
                'raw_gross_spreads': [float(value) for value in raw_gross],
                'total_cost_drags': [float(value) for value in cost_drags],
                'raw_net_spreads': [float(value) for value in raw_net],
            }
        if winsor_returns:
            robust_gross_stats = _hac_mean_test(robust_gross, lag)
            robust_net_stats = _hac_mean_test(robust_net, lag)
            result.update({
                'robust_gross_spread': robust_gross_stats[0],
                'robust_net_spread': robust_net_stats[0],
                'robust_gross_t': robust_gross_stats[1],
                'robust_gross_p': robust_gross_stats[2],
                'robust_gross_p_two_sided': robust_gross_stats[3],
                'robust_net_t': robust_net_stats[1],
                'robust_net_p': robust_net_stats[2],
                'robust_net_p_two_sided': robust_net_stats[3],
                'winsor_returns': winsor_returns,
                'robust_inference_basis': 'winsorized_net_diagnostic',
            })
            if include_series:
                result['inference_series'].update({
                    'robust_gross_spreads': [
                        float(value) for value in robust_gross],
                    'robust_net_spreads': [
                        float(value) for value in robust_net],
                })
        # Backward-compatible names now point at the raw executable series.
        # Winsorized diagnostics remain available only under robust_* keys.
        result.update({
            'gross_spread': raw_gross_stats[0],
            'net_spread': raw_net_stats[0],
            'gross_t': raw_gross_stats[1],
            'gross_p': raw_gross_stats[2],
            'gross_p_two_sided': raw_gross_stats[3],
            't': raw_net_stats[1],
            'p': raw_net_stats[2],
            'p_two_sided': raw_net_stats[3],
        })
        out.append(result)
    return out


def _print_report(all_stats, bh, cost_bps, n_names, n_events, label='value',
                  width=9):
    valid = [s for s in all_stats if not s.get('insufficient')]
    has_robust = any('robust_net_spread' in s for s in valid)
    has_dated_costs = any(s.get('cost_model') == 'dated_total_spread_cost'
                          for s in valid)
    cost_desc = ("dated total spread costs" if has_dated_costs
                 else (f"fixed {cost_bps:.0f}bp one-way/trade "
                       f"({4 * cost_bps:.0f}bp four-trade round trip)"))
    print(f"\n{label} factor screen: {n_names} names, {n_events} events, "
          f"{cost_desc} (Fama-MacBeth spread, Newey-West/HAC net test)\n")
    if has_robust:
        print(f"  {'factor':>{width}}{'h':>4}{'dates':>7}{'raw-g%':>9}"
              f"{'raw-n%':>9}{'rob-n%':>9}{'net-t':>8}{'p(+)':>9}{'BH*':>5}")
    else:
        print(f"  {'factor':>{width}}{'h':>4}{'dates':>7}{'raw-g%':>9}"
              f"{'raw-n%':>9}{'net-t':>8}{'p(+)':>9}{'BH*':>5}")
    rej = bh['rejected_bh'] if bh else []
    j = 0
    for s in all_stats:
        if s.get('insufficient'):
            print(f"  {s['factor']:>{width}}{s['h']:>4}   (insufficient)")
            continue
        star = '  *' if j < len(rej) and rej[j] else ''
        j += 1
        raw_gross = s.get('raw_gross_spread', s['gross_spread'])
        raw_net = s.get('raw_net_spread', s['net_spread'])
        row = (f"  {s['factor']:>{width}}{s['h']:>4}{s['n_dates']:>7}"
               f"{raw_gross*100:>+9.3f}{raw_net*100:>+9.3f}")
        if has_robust:
            robust_net = s.get('robust_net_spread', s['net_spread'])
            row += f"{robust_net*100:>+9.3f}"
        print(row + f"{s['t']:>+8.2f}{s['p']:>9.4f}{star:>5}")
    if has_robust:
        print("\nraw-g/raw-n/net-t/p(+) = unmodified executable inference; "
              "rob-n is a winsorized robustness diagnostic only and never "
              "drives BH or promotion.")
    else:
        print("\nraw-g/raw-n/net-t/net-p use unmodified economic returns.")
    print("Costs are deducted from each date's spread before HAC; BH uses the "
          "resulting one-sided H1 net-mean>0 p-values.")
    if bh:
        print(f"\nBH across {bh['m']} (factor x horizon) tests: "
              f"{bh['n_significant_bh']} significant @α={bh['alpha']}")
        survivors = [s for s, r in zip(valid, rej)
                     if r and s.get('raw_net_spread', s['net_spread']) > 0]
        if survivors:
            print("VERDICT: net-inference BH survivor(s) with positive raw-net "
                  "economics — "
                  + ", ".join(f"{s['factor']}@{s['h']}d (net "
                              f"{s.get('raw_net_spread', s['net_spread'])*100:+.2f}%)"
                              for s in survivors))
        else:
            print(f"VERDICT: no {label} factor clears net-inference BH + "
                  "positive raw-net economics.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.factor_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--factors', nargs='+', default=VALUE_FACTORS)
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-09-30')
    ap.add_argument('--horizons', nargs='+', type=int, default=[21, 63])
    ap.add_argument('--cost-bps', type=float, default=30.0,
                    help='one-way cost for one trade on one leg; the fixed '
                         'screen charges entry and exit on both long and short '
                         'legs (4x) per formation-date cohort')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe for a faster smoke')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date robust-inference winsorization; raw economic '
                         'P&L is always retained (0 = off)')
    cli = ap.parse_args(argv)

    import random
    wh = PitWarehouse()
    rebal = pd.bdate_range(cli.start, cli.end, freq='BMS')
    membership = resolve_universe_membership(wh, rebal)
    names = universe_union(membership)
    if cli.limit and cli.limit < len(names):
        names = sorted(random.Random(cli.seed).sample(names, cli.limit))
    pstart = (pd.Timestamp(cli.start) - pd.Timedelta(days=30)).date().isoformat()
    pend = (pd.Timestamp(cli.end) + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()
    print(f"names={len(names)} rebalances={len(rebal)} factors={cli.factors}",
          file=sys.stderr)

    events, n_names = collect_factor_events(wh, names, rebal, cli.horizons,
                                            cli.factors, pstart, pend,
                                            membership_by_date=membership)
    all_stats, pvals = [], []
    for f in cli.factors:
        stats = factor_study(events, f, cli.horizons, cli.cost_bps,
                             winsor_returns=cli.winsor or None)
        all_stats.extend(stats)
        pvals.extend(s['p'] for s in stats if not s.get('insufficient'))
    bh = benjamini_hochberg(pvals) if pvals else None
    _print_report(all_stats, bh, cli.cost_bps, n_names, len(events))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
