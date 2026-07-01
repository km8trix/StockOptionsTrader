"""Insider-buying edge screen — pooled buy-vs-sell event study, point-in-time.

Promoted from the data session's scratchpad screen (the most encouraging result:
on a 250-name small/mid subset, insider buyers beat sellers +1.19% @63d, t=1.77,
p=0.077 — right sign, grows with horizon, cost-positive, but UNDERPOWERED). This
exists to rerun that screen on the FULL small/mid universe via the DuckDB+Parquet
warehouse (~10x events) and resolve whether the edge survives a multiple-testing
+ transaction-cost bar, or dies.

Method (unchanged from the scratchpad, which was debugged in PR #67):
  * At each monthly rebalance t, score every name by ``insider_net_buys`` over the
    trailing ``lookback`` days. PIT by construction — the provider gates Form-4s
    on ``filingdate <= t``, so nothing filed after t is visible.
  * Direction = sign of net_shares (buy / sell). Forward total-return at each
    horizon h (closeadj).
  * Form the LONG-SHORT SPREAD per rebalance date (mean buy fwd - mean sell fwd,
    both legs required) — the common market factor differences out, so no SPY
    proxy and no separate demean is needed. Test the monthly spread SERIES with a
    Newey-West (HAC) t of the mean (lag = horizon overlap in months), which is
    honest against overlapping windows + cross-sectional clustering that a naive
    pool-all-events Welch t inflates through. Benjamini-Hochberg across horizons;
    net the spread for round-trip cost on BOTH legs (long + short).

Provider is selectable: ``--provider warehouse`` (default, DuckDB+Parquet, full
universe) or ``--provider sqlite`` (the per-ticker PitCache — used to REPRODUCE
the 250-name result against the existing pit_data.db before the warehouse is
fully ingested). The two share an identical read interface, so the screen is
backend-agnostic.

Deterministic: fixed rebalance grid + horizons, no randomness.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.research_stats import benjamini_hochberg  # noqa: E402

# Sharadar scalemarketcap buckets that make up "small/mid" — where insider
# signals are strongest and least arbitraged. NOTE: scalemarketcap is the
# CURRENT bucket, not point-in-time, so this scopes the universe approximately
# (a name's size band can have drifted); it is not used as a PIT signal.
SCALE_SMALL_MID = ['2 - Micro', '3 - Small', '4 - Mid']


def make_provider(kind: str, path: str = None):
    """Return a PIT reader exposing prices / insider_net_buys / universe_asof."""
    if kind == 'warehouse':
        from data.pit_warehouse import PitWarehouse
        return PitWarehouse(path)
    if kind == 'sqlite':
        from data.pit_provider import PitCache
        return PitCache(path)
    raise ValueError(f"unknown provider {kind!r}")


def resolve_universe(prov, rebal_dates, scale) -> list:
    """Survivorship-free union of the names live across the rebalance window —
    every name that was tradeable on ANY rebalance date (delisted names kept for
    their live span). This is the whole point of universe_asof."""
    names = set()
    for t in rebal_dates:
        names.update(prov.universe_asof(t, scalemarketcap=scale))
    return sorted(names)


def collect_events(prov, names, rebal_dates, horizons, lookback,
                   price_start, price_end):
    """One row per (name, rebalance) with non-zero insider activity: date, name,
    signed net_value, direction, and a forward total-return per horizon."""
    maxh = max(horizons)
    recs = []
    n_names = 0
    for name in names:
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        n_names += 1
        idx, vals = px.index, px.values
        for t in rebal_dates:
            nb = prov.insider_net_buys(name, t, lookback_days=lookback)
            nsh = nb['net_shares']
            if (nb['n_buys'] + nb['n_sells']) == 0 or nsh == 0:
                continue
            pos = int(idx.searchsorted(pd.Timestamp(t)))
            if pos + maxh >= len(px) or pos >= len(px):
                continue
            entry = vals[pos]
            if not entry or entry <= 0:
                continue
            # cohort key is the REBALANCE date t, not idx[pos] (the entry bar):
            # the Fama-MacBeth cross-section is "all names rebalanced this month",
            # and the market factor only differences out within a shared cohort.
            # Illiquid names whose first bar is after t must still land in t's
            # cohort, or the per-date spread fragments (n_dates > n_rebalances).
            rec = {'date': pd.Timestamp(t), 'name': name,
                   'nv': nb['net_value'],
                   'dir': 'buy' if nsh > 0 else 'sell'}
            for h in horizons:
                rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
            recs.append(rec)
    cols = ['date', 'name', 'nv', 'dir'] + [f'fwd_{h}' for h in horizons]
    return pd.DataFrame(recs, columns=cols), n_names


def event_study(events: pd.DataFrame, horizons, cost_bps: float, *,
                min_dates: int = 12):
    """PURE Fama-MacBeth buy-vs-sell event study (no I/O).

    A naive pool-all-events Welch t is optimistically biased here: same-name
    events at a 63d horizon on a monthly grid have OVERLAPPING forward windows
    (serially correlated) and same-date events CLUSTER on shared factors, so the
    effective sample is far below the event count and the t is inflated. The
    honest design tests the LONG-SHORT SPREAD as a time series instead:

      * Per rebalance date, form spread = mean(buy fwd) - mean(sell fwd) over the
        names with insider activity that date, keeping only dates with BOTH legs.
        The common date/market factor differences out of the spread, so no
        separate market-demean is needed; a one-sided date honestly cannot form a
        tradeable long-short and simply contributes no spread.
      * Test the resulting monthly spread series with a Newey-West (HAC) t of the
        mean, lag = ceil(horizon / 21) months to absorb the window overlap. One
        observation per date removes the cross-sectional clustering; statsmodels
        reports a proper t-distribution p (not a normal approximation).

    Net spread subtracts ``2 * cost_bps`` (round-trip on the long AND short leg).
    Returns (list of per-horizon dicts, BH dict across horizons or None).

    NOTE: this is deliberately MORE conservative than the scratchpad pool-Welch
    number (the 63d p=0.077 headline) — that inflation is what this fixes.
    """
    out = []
    pvals = []
    for h in horizons:
        col = f'fwd_{h}'
        df = events.dropna(subset=[col])
        if len(df) < 20:
            out.append({'h': h, 'events': int(len(df)), 'insufficient': True})
            continue
        # per-date long-short spread (both legs required); market factor cancels
        legs = df.groupby(['date', 'dir'])[col].mean().unstack('dir')
        if 'buy' not in legs.columns or 'sell' not in legs.columns:
            out.append({'h': h, 'events': int(len(df)), 'insufficient': True})
            continue
        spreads = (legs['buy'] - legs['sell']).dropna().to_numpy()
        n_dates = int(len(spreads))
        if n_dates < min_dates:
            out.append({'h': h, 'events': int(len(df)), 'n_dates': n_dates,
                        'insufficient': True})
            continue
        gross = float(spreads.mean())
        net = gross - 2.0 * cost_bps / 1e4
        lag = max(1, int(np.ceil(h / 21)))          # window overlap in ~months
        fit = sm.OLS(spreads, np.ones(n_dates)).fit(
            cov_type='HAC', cov_kwds={'maxlags': lag})
        t, p = float(fit.tvalues[0]), float(fit.pvalues[0])
        pvals.append(p)
        # descriptive pooled rank-IC: signed net_value vs date-demeaned return
        abn = (df[col] - df.groupby('date')[col].transform('mean')).to_numpy()
        ic = float(np.corrcoef(rankdata(df['nv'].to_numpy()), rankdata(abn))[0, 1])
        vc = df['dir'].value_counts()
        out.append({'h': h, 'events': int(len(df)), 'n_dates': n_dates,
                    'buys': int(vc.get('buy', 0)), 'sells': int(vc.get('sell', 0)),
                    'ic': ic, 'gross_spread': gross, 'net_spread': net,
                    't': t, 'p': p})
    bh = benjamini_hochberg(pvals) if pvals else None
    return out, bh


def _print_report(stats, bh, cost_bps, n_names, n_events):
    print(f"\ninsider screen: {n_names} names, {n_events} events, "
          f"cost {cost_bps:.0f}bp/leg round-trip "
          f"(Fama-MacBeth spread, Newey-West t)\n")
    print(f"  {'h':>4}{'dates':>7}{'buys':>7}{'sells':>7}{'IC':>9}"
          f"{'gross%':>9}{'net%':>9}{'t':>7}{'p':>9}{'BH*':>5}")
    rejected = bh['rejected_bh'] if bh else []
    j = 0
    for s in stats:
        if s.get('insufficient'):
            print(f"  {s['h']:>4}  (too few dates/events: "
                  f"{s.get('n_dates', 0)} dates)")
            continue
        star = '  *' if (j < len(rejected) and rejected[j]) else ''
        j += 1
        print(f"  {s['h']:>4}{s['n_dates']:>7}{s['buys']:>7}{s['sells']:>7}"
              f"{s['ic']:>+9.4f}{s['gross_spread']*100:>+9.3f}"
              f"{s['net_spread']*100:>+9.3f}{s['t']:>+7.2f}{s['p']:>9.4f}{star:>5}")
    if bh:
        nsig = bh['n_significant_bh']
        print(f"\nBH across {bh['m']} horizons: {nsig} significant @α={bh['alpha']}")
        survives = [s for s, r in zip(
            [x for x in stats if not x.get('insufficient')], rejected)
            if r and s['net_spread'] > 0]
        if survives:
            print(f"VERDICT: edge SURVIVES at horizon(s) "
                  f"{[s['h'] for s in survives]} (BH-significant AND net>0).")
        else:
            print("VERDICT: no edge clears BH + cost on this universe "
                  "(honest negative — the apparatus working, not a bug).")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.insider_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--provider', choices=['warehouse', 'sqlite'],
                    default='warehouse', help='PIT backend (default warehouse)')
    ap.add_argument('--db', default=None,
                    help='provider path (warehouse dir / sqlite db); else default')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--symbols-file', help='one ticker per line (reproduce a subset)')
    src.add_argument('--universe-asof', action='store_true',
                     help='derive the survivorship-free small/mid universe from the provider')
    ap.add_argument('--scale', nargs='+', default=SCALE_SMALL_MID,
                    help='scalemarketcap buckets for --universe-asof')
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-09-30')
    ap.add_argument('--horizons', nargs='+', type=int, default=[21, 63])
    ap.add_argument('--lookback', type=int, default=90, help='insider window (days)')
    ap.add_argument('--cost-bps', type=float, default=30.0,
                    help='round-trip cost per leg in bps (small/mid ~30)')
    cli = ap.parse_args(argv)

    prov = make_provider(cli.provider, cli.db)
    rebal = pd.bdate_range(cli.start, cli.end, freq='BMS')
    # price pad: lookback before start, max-horizon (in calendar days) after end
    pstart = (pd.Timestamp(cli.start) - pd.Timedelta(days=cli.lookback + 200)).date().isoformat()
    pend = (pd.Timestamp(cli.end) + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()

    if cli.symbols_file:
        names = [s.strip() for s in open(cli.symbols_file) if s.strip()]
    elif cli.universe_asof:
        names = resolve_universe(prov, rebal, cli.scale)
    else:
        ap.error('choose --symbols-file or --universe-asof')
    print(f"provider={cli.provider} names={len(names)} rebalances={len(rebal)}",
          file=sys.stderr)

    events, n_names = collect_events(prov, names, rebal, cli.horizons,
                                     cli.lookback, pstart, pend)
    stats, bh = event_study(events, cli.horizons, cli.cost_bps)
    _print_report(stats, bh, cli.cost_bps, n_names, len(events))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
