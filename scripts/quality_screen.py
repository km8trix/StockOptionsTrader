"""Cross-sectional QUALITY / profitability factor screen — the FF5 RMW sibling
to the value screen (scripts/factor_screen.py), decorrelated from value + insider.

Same honest apparatus (Fama-MacBeth per-date long-short spread, Newey-West/HAC t,
Benjamini-Hochberg across the family, cost-netting), but the signal is a
profitability ratio from the PIT SF1 fundamentals table (point-in-time by
datekey) rather than a valuation ratio. At each monthly rebalance, rank the
survivorship-free small/mid universe by each ratio, LONG the HIGH-quality quantile
and SHORT the LOW-quality (opposite direction to value's cheap-long), and test the
monthly long-short spread.

Two differences from the value screen, both economically motivated:
  - HIGH is the long (``cheap_is_long=False``): more profitable = the bet.
  - Non-positive values are KEPT (``drop_nonpositive=False``): an unprofitable firm
    (negative ROE / margin) is a legitimate short-leg member, not a dropped trap.

Quality ratios (from SF1 ARQ, higher = better = the long):
  netmargin, grossmargin
NOTE: roe / roa are 100% NULL in the SF1 ARQ dimension (Sharadar only populates
them in the trailing AR*T dimensions) — do not use them here. grossmargin is
bounded <=1 by definition, so values outside [-1,1] (near-zero-revenue garbage)
are dropped. Forward returns are winsorized per date (--winsor 0.01) — without it
+2000x micro-cap return artifacts dominate a leg and flip the sign (the first,
un-winsorized run reported a spurious NEGATIVE spread).

RESULT (2026-07-02, winsorized): netmargin is a real profitability premium —
63d net +3.96% t=2.26 (full), +6.73% t=5.42 (ex-2020); grossmargin is weak
(t~1.1). But netmargin has value's exact caveats: micro-concentrated (small/mid
ex-micro only t~1.5) and regime-sensitive (strong in-sample, decays OOS 2020-24).
A second decorrelated-but-marginal signal, like value.

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
from scripts.factor_screen import _print_report, factor_study  # noqa: E402
from scripts.insider_screen import SCALE_SMALL_MID, resolve_universe  # noqa: E402

QUALITY_FACTORS = ['netmargin', 'grossmargin']   # roe/roa are null in SF1 ARQ


def collect_quality_events(prov, names, rebal_dates, horizons, factors,
                           price_start, price_end):
    """One row per (name, rebalance): the PIT SF1 quality values + forward returns.

    Mirrors factor_screen.collect_factor_events but reads fundamentals_asof (SF1,
    datekey<=t) instead of daily_metric — profitability lives in the quarterly
    fundamentals, not the daily valuation table.
    """
    maxh = max(horizons)
    recs = []
    n_names = 0
    batch = getattr(prov, 'fundamentals_asof_series', None)
    for name in names:
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        n_names += 1
        idx, vals = px.index, px.values
        # Cheap bounds/entry checks FIRST so out-of-range dates never pay an
        # SF1 query; then ONE batched as-of fetch per name instead of a
        # full-width sort-and-pick per rebalance date.
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
            funds = batch(name, [t for t, _, _ in live])
        else:
            funds = [prov.fundamentals_asof(name, t) for t, _, _ in live]
        for (t, pos, entry), fund in zip(live, funds):
            if not fund:                             # PIT SF1 row or None
                continue
            rec = {'date': pd.Timestamp(t), 'name': name}
            for f in factors:
                v = fund.get(f)
                rec[f] = float(v) if v is not None else None  # DuckDB may hand Decimal
            for h in horizons:
                rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
            recs.append(rec)
    cols = (['date', 'name'] + list(factors) + [f'fwd_{h}' for h in horizons])
    return pd.DataFrame(recs, columns=cols), n_names


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.quality_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--factors', nargs='+', default=QUALITY_FACTORS)
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
    pend = (pd.Timestamp(cli.end)
            + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()
    print(f"names={len(names)} rebalances={len(rebal)} factors={cli.factors}",
          file=sys.stderr)

    events, n_names = collect_quality_events(wh, names, rebal, cli.horizons,
                                             cli.factors, pstart, pend)
    if 'grossmargin' in events:               # gross margin is <=1 by definition;
        events['grossmargin'] = events['grossmargin'].where(   # drop garbage tails
            events['grossmargin'].between(-1, 1))
    all_stats, pvals = [], []
    for f in cli.factors:
        # HIGH quality = long; keep negatives (unprofitable = short-leg member);
        # winsorize returns (micro-cap +2000x artifacts otherwise flip the sign)
        stats = factor_study(events, f, cli.horizons, cli.cost_bps,
                             cheap_is_long=False, drop_nonpositive=False,
                             winsor_returns=cli.winsor or None)
        all_stats.extend(stats)
        pvals.extend(s['p'] for s in stats if not s.get('insufficient'))
    bh = benjamini_hochberg(pvals) if pvals else None
    _print_report(all_stats, bh, cli.cost_bps, n_names, len(events),
                  label='quality')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
