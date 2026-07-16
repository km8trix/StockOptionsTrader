"""Cross-sectional QUALITY / profitability factor screen — the FF5 RMW sibling
to the value screen (scripts/factor_screen.py), decorrelated from value + insider.

Same honest apparatus (Fama-MacBeth per-date long-short spread, Newey-West/HAC t,
Benjamini-Hochberg across the family, cost-netting), but the signal is a
profitability ratio from the PIT SF1 fundamentals table (point-in-time by
datekey) rather than a valuation ratio. At each monthly rebalance, rank the
survivorship-free dated eligible universe by each ratio, LONG the HIGH-quality
quantile and SHORT the LOW-quality (opposite direction to value's cheap-long),
and test the monthly long-short spread.

Two differences from the value screen, both economically motivated:
  - HIGH is the long (``cheap_is_long=False``): more profitable = the bet.
  - Non-positive values are KEPT (``drop_nonpositive=False``): an unprofitable firm
    (negative ROE / margin) is a legitimate short-leg member, not a dropped trap.

Quality ratios (from SF1 ARQ, higher = better = the long):
  netmargin, grossmargin              (native SF1 ratio columns)
  gp_assets = gp / assets             (COMPUTED — Novy-Marx 2013 gross profitability)
  roa       = netinc / assets         (COMPUTED)
NOTE: the roe / roa RATIO columns are 100% NULL in the SF1 ARQ dimension
(Sharadar only populates them in the trailing AR*T dimensions) — but the RAW
inputs (gp 3.4% null, netinc 3.5%, assets 0.05%) are near-complete, so the
COMPUTED_RATIOS above are rebuilt from raws per SF1 row (both inputs from the
same quarter by construction; assets guarded by a relative-epsilon positive
floor so a shell-company asset sliver cannot explode the ratio). grossmargin is
bounded <=1 by definition, so values outside [-1,1] (near-zero-revenue garbage)
are dropped. Forward returns are winsorized per date (--winsor 0.01) — without it
+2000x micro-cap return artifacts dominate a leg and flip the sign (the first,
un-winsorized run reported a spurious NEGATIVE spread).

STATISTICAL STATUS (2026-07-13): the recorded results below predate the net-array
inference repair in ``factor_study``. Their displayed means were cost-net, but
their t/p and BH decisions tested GROSS spreads. They are retained only as
research provenance and are not evidence of a net tradeable edge until rerun.
The current report separately shows raw economic P&L and winsorized robustness
inference, deducts cost per date before HAC, and sends one-sided net p-values to
BH. The legacy fixed-cost implementation also deducted two charges while the
CLI described a one-way cost. The corrected screen charges four trades per
cohort (enter/exit long and short), so the historical net means must be rerun.

LEGACY RESULT (2026-07-02, winsorized): netmargin is a profitability premium —
63d net +3.96% t=2.26 (full), +6.73% t=5.42 (ex-2020); grossmargin is weak
(t~1.1). But netmargin has value's exact caveats: micro-concentrated (small/mid
ex-micro only t~1.5) and regime-sensitive (strong in-sample, decays OOS 2020-24).
A second decorrelated-but-marginal signal, like value.
NOTE: the default factor family is now 4 (m=8 BH tests, stricter thresholds);
reproducing the numbers above needs ``--factors netmargin grossmargin``.

LEGACY RESULT (2026-07-10, computed ratios, assets>=$1M floor, --terciles, full
universe 4627 names / 330,411 events): 19/32 BH survivors — the strongest
quality screen in the program. gp_assets (Novy-Marx) pooled 63d net +3.60%
t=+4.44, micro 63d net +5.26% t=+4.98; roa survives in ALL 8 cells incl.
MID-CAP (63d net +2.04% t=+2.36) — the tradeable slice where the native
margins die. rank-corr(gp_assets, netmargin) per-date Spearman mean +0.067:
profitability-scaled-by-assets is a nearly orthogonal quality dimension, an
ADDITION to the VQ leg, not a replacement. The unguarded first run read the
same 19 survivors with slightly weaker micro gp_assets — the unit-error rows
were noise-dragging, not inflating. Screen-level only; no promotion claim.

Deterministic; reads the warehouse (tickers + sep + sf1 ingested).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.research_stats import benjamini_hochberg  # noqa: E402
from data.pit_warehouse import PitWarehouse  # noqa: E402
# Ratio math + guard constants live in the data seam (data/quality_ratios —
# the sue_table/issuance_table convention) so the Value+Quality desk shares
# ONE definition with this screen; re-exported here byte-identically (the
# IS-identity pin in tests/test_quality_screen.py).
from data.quality_ratios import (  # noqa: E402, F401
    _MIN_ASSETS, _REL_EPS, computed_ratio)
from scripts.factor_screen import _print_report, factor_study  # noqa: E402
from scripts.insider_screen import (  # noqa: E402
    _eligible_on,
    resolve_universe_membership,
    universe_union,
)

# netmargin/grossmargin are native SF1 columns; gp_assets/roa are COMPUTED from
# near-complete raws (the native roe/roa ratio columns are null in SF1 ARQ).
QUALITY_FACTORS = ['netmargin', 'grossmargin', 'gp_assets', 'roa']

COMPUTED_RATIOS = {                # factor -> (numerator, denominator) SF1 raws
    'gp_assets': ('gp', 'assets'),      # Novy-Marx (2013) gross profitability
    'roa': ('netinc', 'assets'),
}


def collect_quality_events(prov, names, rebal_dates, horizons, factors,
                           price_start, price_end, *, with_mcap=False,
                           membership_by_date=None):
    """One row per (name, rebalance): the PIT SF1 quality values + forward returns.

    Mirrors factor_screen.collect_factor_events but reads fundamentals_asof (SF1,
    datekey<=t) instead of daily_metric — profitability lives in the quarterly
    fundamentals, not the daily valuation table. Factors named in COMPUTED_RATIOS
    are rebuilt from the row's raw fields (see computed_ratio); with_mcap=True
    additionally records the row's SF1 marketcap as ``mcap`` (for size terciles;
    default False keeps the output byte-identical for existing callers).
    """
    maxh = max(horizons)
    recs = []
    n_names = 0
    batch = getattr(prov, 'fundamentals_asof_series', None)
    for name in names:
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        idx, vals = px.index, px.values
        # Cheap bounds/entry checks FIRST so out-of-range dates never pay an
        # SF1 query; then ONE batched as-of fetch per name instead of a
        # full-width sort-and-pick per rebalance date.
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
            funds = batch(name, [t for t, _, _ in live])
        else:
            funds = [prov.fundamentals_asof(name, t) for t, _, _ in live]
        for (t, pos, entry), fund in zip(live, funds):
            if not fund:                             # PIT SF1 row or None
                continue
            rec = {'date': pd.Timestamp(t), 'name': name}
            for f in factors:
                if f in COMPUTED_RATIOS:
                    rec[f] = computed_ratio(fund, *COMPUTED_RATIOS[f])
                else:
                    v = fund.get(f)
                    rec[f] = float(v) if v is not None else None  # DuckDB may hand Decimal
            if with_mcap:
                v = fund.get('marketcap')
                rec['mcap'] = float(v) if v is not None else None
            for h in horizons:
                rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
            recs.append(rec)
    cols = (['date', 'name'] + list(factors) + (['mcap'] if with_mcap else [])
            + [f'fwd_{h}' for h in horizons])
    return pd.DataFrame(recs, columns=cols), n_names


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.quality_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--factors', nargs='+', default=QUALITY_FACTORS)
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-09-30')
    ap.add_argument('--horizons', nargs='+', type=int, default=[21, 63])
    ap.add_argument('--cost-bps', type=float, default=30.0,
                    help='one-way cost for one trade on one leg; charges entry '
                         'and exit on both long and short legs (4x)')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe for a faster smoke')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date robust-inference winsorization; raw economic '
                         'P&L is always retained (0 = off)')
    ap.add_argument('--terciles', action='store_true',
                    help='also test per-date PIT size terciles (SF1 marketcap);'
                         ' enlarges the BH family to factors x'
                         ' (pooled+t0/t1/t2) x horizons')
    cli = ap.parse_args(argv)

    import random
    wh = PitWarehouse()
    rebal = pd.bdate_range(cli.start, cli.end, freq='BMS')
    membership = resolve_universe_membership(wh, rebal)
    names = universe_union(membership)
    if cli.limit and cli.limit < len(names):
        names = sorted(random.Random(cli.seed).sample(names, cli.limit))
    pstart = (pd.Timestamp(cli.start) - pd.Timedelta(days=30)).date().isoformat()
    pend = (pd.Timestamp(cli.end)
            + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()
    print(f"names={len(names)} rebalances={len(rebal)} factors={cli.factors}",
          file=sys.stderr)

    events, n_names = collect_quality_events(wh, names, rebal, cli.horizons,
                                             cli.factors, pstart, pend,
                                             with_mcap=cli.terciles,
                                             membership_by_date=membership)
    if 'grossmargin' in events:               # gross margin is <=1 by definition;
        events['grossmargin'] = events['grossmargin'].where(   # drop garbage tails
            events['grossmargin'].between(-1, 1))
    if cli.terciles:                          # per-date PIT size slices, pead-style
        from scripts.pead_screen import add_size_terciles
        events = add_size_terciles(events)
        slices = [('pooled', events)] + [
            (f't{b}', events[events['tercile'] == b]) for b in (0, 1, 2)]
    else:
        slices = [('pooled', events)]
    all_stats, pvals = [], []
    for f in cli.factors:
        for slabel, ev in slices:
            # HIGH quality = long; keep negatives (unprofitable = short-leg
            # member); winsorize returns (micro-cap +2000x artifacts otherwise
            # flip the sign)
            stats = factor_study(ev, f, cli.horizons, cli.cost_bps,
                                 cheap_is_long=False, drop_nonpositive=False,
                                 winsor_returns=cli.winsor or None)
            if cli.terciles:
                for s in stats:                # reuse the report printer
                    s['factor'] = f'{f}@{slabel}'
            all_stats.extend(stats)
            pvals.extend(s['p'] for s in stats if not s.get('insufficient'))
    bh = benjamini_hochberg(pvals) if pvals else None
    # Dynamic width only for the factor@slice labels; the default report keeps
    # the compact fixed-9 factor label column.
    width = (max(9, *(len(s['factor']) for s in all_stats))
             if cli.terciles and all_stats else 9)
    _print_report(all_stats, bh, cli.cost_bps, n_names, len(events),
                  label='quality', width=width)
    if cli.terciles:
        print("\n(t0=smallest, 1=middle, 2=largest — relative per-date "
              "cross-sections on PIT SF1 marketcap; high quality = long)")
    if {'gp_assets', 'netmargin'}.issubset(events.columns):
        both = events.dropna(subset=['gp_assets', 'netmargin'])
        rc = both.groupby('date').apply(
            lambda g: g['gp_assets'].corr(g['netmargin'], method='spearman'),
            include_groups=False)
        print(f"\nrank-corr(gp_assets, netmargin): per-date Spearman mean "
              f"{rc.mean():+.3f} median {rc.median():+.3f} ({len(rc)} dates)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
