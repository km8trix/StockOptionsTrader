"""Cross-sectional fundamental-factor screen — value factors on honest data.

Same honest apparatus as scripts/insider_screen.py (Fama-MacBeth per-date spread,
Newey-West/HAC t, Benjamini-Hochberg across the family, cost-netting), but the
signal is a continuous cross-sectional VALUE ratio from the PIT DAILY table
(point-in-time by nature) rather than the binary insider sign. At each monthly
rebalance, rank the survivorship-free small/mid universe by each ratio, long the
CHEAP quantile and short the EXPENSIVE, and test the monthly long-short spread.

Value ratios (all from daily_metric; lower = cheaper = the value bet):
  pb, pe, ps, evebit, evebitda   (non-positive values dropped — value traps /
  negative-earnings noise are not a clean "cheap").

Deterministic; reads the warehouse (tickers + sep + daily ingested). Quality
factors live in SF1 — see scripts/quality_screen.py.

Forward returns are WINSORIZED per date (--winsor 0.01 default): this universe's
raw forward returns reach +2000x (micro-cap / split artifacts) and one such name
dominates a leg. The screen's ORIGINAL headline was computed WITHOUT this and is
inflated ~40% — winsorized, pb 63d net is ~+1.6% (t~2.2), not the +2.52% first
reported. Pass --winsor 0 to reproduce the old (contaminated) numbers.

CAVEAT — even winsorized, the screen OVERSTATES the tradeable edge. The validation
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
from scripts.insider_screen import SCALE_SMALL_MID, resolve_universe  # noqa: E402

VALUE_FACTORS = ['pb', 'pe', 'ps', 'evebit', 'evebitda']


def collect_factor_events(prov, names, rebal_dates, horizons, factors,
                          price_start, price_end):
    """One row per (name, rebalance): the PIT factor values + forward returns."""
    maxh = max(horizons)
    recs = []
    n_names = 0
    batch = getattr(prov, 'daily_metrics', None)
    for name in names:
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        n_names += 1
        idx, vals = px.index, px.values
        # Cheap bounds/entry checks FIRST so out-of-range dates never pay a
        # DAILY query; then ONE batched fetch per name instead of one
        # point query per rebalance date.
        live = []
        for t in rebal_dates:
            pos = int(idx.searchsorted(pd.Timestamp(t)))
            if pos + maxh >= len(px) or pos >= len(px):
                continue
            entry = vals[pos]
            if not entry or entry <= 0:
                continue
            live.append((t, pos, entry))
        if not live:
            continue
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


def factor_study(events, factor, horizons, cost_bps, *, quantile=0.2,
                 cheap_is_long=True, min_dates=12, drop_nonpositive=True,
                 winsor_returns=None):
    """PURE Fama-MacBeth long-short by one cross-sectional factor.

    Per rebalance date, rank names by ``factor`` (dropping non-positive values),
    long the cheap quantile / short the expensive, spread = mean(long fwd) -
    mean(short fwd); test the monthly spread series with a Newey-West (HAC) t.
    Returns a list of per-horizon dicts. (cheap_is_long: for value ratios the LOW
    end is the long. drop_nonpositive: True for value ratios where a non-positive
    value is a trap, not a "cheap"; set False for profitability factors where a
    negative value — an unprofitable firm — is a legitimate short-leg member.
    winsor_returns: if set (e.g. 0.01), clip EACH DATE's forward returns to the
    [w, 1-w] quantiles before averaging — essential on this survivorship-free
    micro-cap universe where raw forward returns reach +2000x and one name would
    otherwise dominate a leg. Default None = no clipping.)
    """
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
        for _, g in df.groupby('date'):
            n = len(g)
            k = max(1, int(quantile * n))
            if n < 2 * k:
                continue                              # can't form both legs
            g = g.sort_values(factor)                 # ascending: cheap first
            r = g[col]
            if winsor_returns:                        # tame micro-cap return tails
                r = r.clip(r.quantile(winsor_returns),
                           r.quantile(1.0 - winsor_returns))
            cheap = r.iloc[:k].mean()
            rich = r.iloc[-k:].mean()
            spreads.append((cheap - rich) if cheap_is_long else (rich - cheap))
        n_dates = len(spreads)
        if n_dates < min_dates:
            out.append({'factor': factor, 'h': h, 'n_dates': n_dates,
                        'insufficient': True})
            continue
        arr = np.asarray(spreads, float)
        gross = float(arr.mean())
        net = gross - 2.0 * cost_bps / 1e4
        lag = max(1, int(np.ceil(h / 21)))
        fit = sm.OLS(arr, np.ones(n_dates)).fit(cov_type='HAC',
                                                cov_kwds={'maxlags': lag})
        out.append({'factor': factor, 'h': h, 'n_dates': n_dates,
                    'gross_spread': gross, 'net_spread': net,
                    't': float(fit.tvalues[0]), 'p': float(fit.pvalues[0])})
    return out


def _print_report(all_stats, bh, cost_bps, n_names, n_events, label='value'):
    print(f"\n{label} factor screen: {n_names} names, {n_events} events, "
          f"cost {cost_bps:.0f}bp/leg (Fama-MacBeth spread, Newey-West t)\n")
    print(f"  {'factor':>9}{'h':>4}{'dates':>7}{'gross%':>9}{'net%':>9}"
          f"{'t':>7}{'p':>9}{'BH*':>5}")
    rej = bh['rejected_bh'] if bh else []
    j = 0
    for s in all_stats:
        if s.get('insufficient'):
            print(f"  {s['factor']:>9}{s['h']:>4}   (insufficient)")
            continue
        star = '  *' if j < len(rej) and rej[j] else ''
        j += 1
        print(f"  {s['factor']:>9}{s['h']:>4}{s['n_dates']:>7}"
              f"{s['gross_spread']*100:>+9.3f}{s['net_spread']*100:>+9.3f}"
              f"{s['t']:>+7.2f}{s['p']:>9.4f}{star:>5}")
    if bh:
        print(f"\nBH across {bh['m']} (factor x horizon) tests: "
              f"{bh['n_significant_bh']} significant @α={bh['alpha']}")
        survivors = [s for s, r in zip(
            [x for x in all_stats if not x.get('insufficient')], rej)
            if r and s['net_spread'] > 0]
        if survivors:
            print("VERDICT: survivor(s) — "
                  + ", ".join(f"{s['factor']}@{s['h']}d (net "
                              f"{s['net_spread']*100:+.2f}%)" for s in survivors))
        else:
            print(f"VERDICT: no {label} factor clears BH + cost (honest negative).")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.factor_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--factors', nargs='+', default=VALUE_FACTORS)
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-09-30')
    ap.add_argument('--horizons', nargs='+', type=int, default=[21, 63])
    ap.add_argument('--cost-bps', type=float, default=30.0)
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe for a faster smoke')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date forward-return winsorization (0 = off)')
    cli = ap.parse_args(argv)

    import random
    wh = PitWarehouse()
    rebal = pd.bdate_range(cli.start, cli.end, freq='BMS')
    names = resolve_universe(wh, rebal, SCALE_SMALL_MID)
    if cli.limit and cli.limit < len(names):
        names = sorted(random.Random(cli.seed).sample(names, cli.limit))
    pstart = (pd.Timestamp(cli.start) - pd.Timedelta(days=30)).date().isoformat()
    pend = (pd.Timestamp(cli.end) + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()
    print(f"names={len(names)} rebalances={len(rebal)} factors={cli.factors}",
          file=sys.stderr)

    events, n_names = collect_factor_events(wh, names, rebal, cli.horizons,
                                            cli.factors, pstart, pend)
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
