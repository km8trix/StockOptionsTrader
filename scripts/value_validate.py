"""Validation battery for the value-factor screen (scripts/factor_screen.py).

The raw screen looks great — 10/10 factor x horizon clear BH + 30bp cost, all
net-positive (63d: ps +3.25%, pb +2.52%, ...). But the insider lesson is that a
strong SCREEN can be a regime/scale artifact that dies as a tradeable desk. This
battery applies the same four cuts to value before anyone graduates it:

  OOS split · ex-2020 · scale (per-date PIT market-cap tercile) · correlation
  (how many INDEPENDENT bets the "10 significant" really are).

Reuses factor_study / collect_factor_events from factor_screen (already unit
tested), so the only new logic here is _date_spreads (per-date spread series for
the correlation cut) and the per-date size bucketing.

RECORDED VERDICT (2026-07-01, survivorship-free small/mid universe, pb/ps focus):
the headline is real but heavily QUALIFIED — the tradeable edge is MARGINAL and
regime-dependent, very likely to fail a desk gate the way insider did.

STATISTICAL STATUS (2026-07-13): the verdict numbers below were recorded before
both winsorization and the net-array inference repair. Their displayed means
were cost-net but their t/p values tested GROSS spreads. They are research
provenance, not current significance claims. The battery now defaults to robust
winsorized inference and ``factor_study`` deducts per-date costs before HAC;
pass ``--winsor 0`` only to reproduce the contaminated raw-return sensitivity.

  1. OOS — entirely a back-half phenomenon (the real "value comeback"):
       IS  2015-2019  pb 63d net +0.32%  t=0.87  p=0.39   (nothing)
       OOS 2020-2024  pb 63d net +4.75%  t=3.52  p=0.0004 (very strong)
       ex-2020         pb 63d net +2.37%  t=2.88          (holds, not just COVID)
  2. SCALE — micro-concentrated (the untradeable end), same trap as insider:
       micro  pb 63d net +3.82%  t=3.74   (strong; widest spreads, worst liq.)
       small  pb 63d net +2.06%  t=1.78   (MARGINAL)
       mid    pb 63d net +1.34%  t=1.69   (MARGINAL)
     -> the tradeable small/mid ex-micro slice is only t~1.7, p~0.08-0.09.
  3. CORRELATION — mean pairwise 0.61 -> ~2 independent bets, not 5
     (pb/pe/ps cluster; evebit/evebitda cluster).

Deterministic; reads the warehouse (tickers + sep + daily ingested). Pass
``--cache path.pkl`` to collect the events once and reuse them across runs.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.pit_warehouse import PitWarehouse  # noqa: E402
from data.size_buckets import size_buckets  # noqa: E402
from scripts.factor_screen import collect_factor_events, factor_study  # noqa: E402
from scripts.insider_screen import (  # noqa: E402
    resolve_universe_membership,
    universe_union,
)

FACTORS = ['pb', 'pe', 'ps', 'evebit', 'evebitda']
HZ = [21, 63]


def _date_spreads(df, factor, h=63, q=0.2, winsor=None):
    """Per-date cheap-minus-rich spread series for a factor — the raw material for
    the correlation cut (factor_study collapses this to one HAC t; here we keep the
    series so we can correlate factors against each other). ``winsor`` clips each
    date-group's forward returns to its [w, 1-w] quantiles, mirroring
    factor_study's winsor_returns."""
    col = f'fwd_{h}'
    d = df.dropna(subset=[factor, col])
    d = d[d[factor] > 0]
    out = {}
    for date, g in d.groupby('date'):
        n = len(g)
        k = max(1, int(q * n))
        if n < 2 * k:
            continue
        r = g.sort_values(factor)[col]
        if winsor:
            r = r.clip(r.quantile(winsor), r.quantile(1.0 - winsor))
        out[date] = r.iloc[:k].mean() - r.iloc[-k:].mean()
    return pd.Series(out)


def _rep(df, label, factors=FACTORS, winsor=None):
    print(f"\n=== {label}: {len(df)} events ===")
    print(f"  {'factor':>9}{'h':>4}{'raw-n%':>9}{'rob-n%':>9}"
          f"{'net-t':>7}{'p(+)':>9}")
    for f in factors:
        for s in factor_study(df, f, HZ, 30.0, winsor_returns=winsor):
            if s.get('insufficient'):
                continue
            raw_net = s.get('raw_net_spread', s['net_spread'])
            print(f"  {s['factor']:>9}{s['h']:>4}{raw_net*100:>+9.3f}"
                  f"{s['net_spread']*100:>+9.3f}{s['t']:>+7.2f}"
                  f"{s['p']:>9.4f}")


def load_events(cache):
    if cache and os.path.exists(cache):
        ev = pd.read_pickle(cache)
        print(f"loaded {len(ev)} cached events from {cache}", file=sys.stderr)
        return ev
    wh = PitWarehouse()
    rb = pd.bdate_range('2015-01-01', '2024-09-30', freq='BMS')
    membership = resolve_universe_membership(wh, rb)
    names = universe_union(membership)
    pstart = (pd.Timestamp('2015-01-01') - pd.Timedelta(days=30)).date().isoformat()
    pend = (pd.Timestamp('2024-09-30')
            + pd.Timedelta(days=max(HZ) * 2 + 30)).date().isoformat()
    print(f"collecting {len(names)} names...", file=sys.stderr)
    ev, n = collect_factor_events(wh, names, rb, HZ, FACTORS + ['marketcap'],
                                  pstart, pend,
                                  membership_by_date=membership)
    print(f"collected {len(ev)} events from {n} names", file=sys.stderr)
    if cache:
        ev.to_pickle(cache)
        print(f"cached -> {cache}", file=sys.stderr)
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.value_validate',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--cache', default=None,
                    help='pkl path to collect events once and reuse')
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date forward-return winsorization, matching '
                         'factor_screen (0 = off, reproducing the original '
                         'un-winsorized/contaminated recorded verdict)')
    cli = ap.parse_args(argv)
    w = cli.winsor or None

    ev = load_events(cli.cache)
    ev['yr'] = ev['date'].dt.year

    _rep(ev, "FULL 2015-2024", winsor=w)
    _rep(ev[ev['yr'] < 2020], "IN-SAMPLE 2015-2019", factors=['pb', 'ps'],
         winsor=w)
    _rep(ev[ev['yr'] >= 2020], "OUT-OF-SAMPLE 2020-2024", factors=['pb', 'ps'],
         winsor=w)
    _rep(ev[ev['yr'] != 2020], "EX-2020", factors=['pb', 'ps'], winsor=w)

    # scale breakdown: per-date PIT market-cap tercile (0=micro,1=small,2=mid)
    print("\n=== scale breakdown (pb, 63d) — per-date marketcap tercile ===")
    ev['bucket'] = -1
    for _, g in ev.groupby('date'):
        b = size_buckets(dict(zip(g['name'], g['marketcap'])), 3)
        ev.loc[g.index, 'bucket'] = g['name'].map(b).fillna(-1).astype(int).values
    for bk, lbl in [(0, 'micro'), (1, 'small'), (2, 'mid')]:
        sub = ev[ev['bucket'] == bk]
        s = [x for x in factor_study(sub, 'pb', [63], 30.0, winsor_returns=w)
             if not x.get('insufficient')]
        if s:
            s = s[0]
            raw_net = s.get('raw_net_spread', s['net_spread'])
            print(f"  {lbl:>6}: n={len(sub):>6} raw-net={raw_net*100:+.3f}% "
                  f"robust-net={s['net_spread']*100:+.3f}% "
                  f"t={s['t']:+.2f} p(+)={s['p']:.4f}")

    print("\n=== factor correlation (63d per-date spread series) ===")
    sp = pd.DataFrame({f: _date_spreads(ev, f, winsor=w) for f in FACTORS}).dropna()
    corr = sp.corr()
    print(corr.round(2).to_string())
    iu = np.triu_indices(len(FACTORS), k=1)
    print(f"\nmean pairwise corr: {corr.values[iu].mean():.2f} "
          f"(high -> the 5 factors are ~1-2 bets, not 5)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
