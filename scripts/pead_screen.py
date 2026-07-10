"""PEAD screen — true SUE (SF1 diluted EPS) on the survivorship-free
small/mid universe, through the honest apparatus.

Post-earnings-announcement drift with a REAL standardized earnings surprise
(data/earnings_surprise.sue_table over PitWarehouse.eps_quarterly), replacing
the dates-only proxy that was directionally-right-but-insignificant on large
caps. At each business-month-start rebalance t, each name's signal is the SUE
of its latest filing KNOWN at t (datekey <= t), kept only when FRESH (datekey
within --fresh-days of t). High SUE = long, low = short. The spread is tested
with the exact machinery that validated/killed insider, value and quality:
factor_screen.factor_study (per-date Fama-MacBeth spread, Newey-West/HAC t,
per-date winsorized forward returns), Benjamini-Hochberg across the whole
family (pooled + size terciles x horizons), 30bp/leg cost-netting.

PIT notes: datekey is the SEC FILING date (median ~41 days after quarter end),
not the press-release date — entries are conservative/late by construction,
and lookahead-clean. Size terciles are per-date event cross-sections on PIT
DAILY marketcap (the pooling-dilution lesson: report within-bucket).

Deterministic; warehouse-only (tickers + sep + sf1 + daily ingested).

    python -m scripts.pead_screen                    # full universe
    python -m scripts.pead_screen --limit 250        # seeded smoke subset
    python -m scripts.pead_screen --field revenue    # SURGE (revenue surprise)
    python -m scripts.pead_screen --selftest         # offline, no warehouse

SURGE RESULT (2026-07-10, --field revenue --dating announce, 4291 names /
181,461 events): 2/8 BH survivors, micro only — 21d net +0.55% t=+3.23,
63d net +2.96% t=+3.23. A real but weaker echo of EPS-SUE announce (4/8;
micro 63d t=+5.09): comparable micro-63d magnitude, lower t, nothing
outside micro. Independence on 215,296 matched filings: Spearman +0.34,
sign agreement 62.8% (Pearson ~0 is winsorization-free tail noise —
rank correlation is the honest number); degenerate-variance guard drops
3.09% of revenue candidates vs 0.05% of EPS. Moderately independent ->
a micro-band SUE x SURGE double-sort is the only combine worth testing;
SURGE alone adds nothing outside micro. Screen-level only; no promotion.
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

from analysis.research_stats import benjamini_hochberg  # noqa: E402
from data.earnings_surprise import (  # noqa: E402
    apply_announcement_dating, sue_table)
from data.pit_warehouse import PitWarehouse  # noqa: E402
from scripts.factor_screen import _print_report, factor_study  # noqa: E402
from scripts.insider_screen import SCALE_SMALL_MID, resolve_universe  # noqa: E402


def collect_sue_events(prov, names, rebal_dates, horizons, *,
                       fresh_days=63, price_start=None, price_end=None,
                       announcements=None, field='epsdil'):
    """One row per (name, rebalance) with a fresh PIT SUE: columns
    ['date','name','sue','mcap'] + fwd_h. Cohort key = the rebalance date.
    announcements (optional, PitWarehouse.earnings_events shape): re-dates
    each SUE from the SF1 filing to the 8-K press release (--dating announce),
    capturing the early-drift window the ~41-day filing lag forfeits.
    field: SF1 quarterly column to standardize — 'epsdil' = classic SUE,
    'revenue' = SURGE (Jegadeesh-Livnat 2006); same math either way."""
    maxh = max(horizons)
    eps = prov.fundamentals_quarterly(list(names), fields=(field,))
    sues = sue_table(eps, column=field)
    if announcements is not None and len(announcements):
        sues = apply_announcement_dating(sues, announcements)
    n_names = 0
    recs = []
    batch = getattr(prov, 'daily_metrics', None)
    for name, g in sues.groupby('ticker', sort=False):
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        n_names += 1
        idx, vals = px.index, px.values
        g = g.sort_values('datekey')
        keys = g['datekey'].to_numpy(dtype='datetime64[ns]')
        svals = g['sue'].to_numpy(dtype=float)
        live = []
        for t in rebal_dates:
            ts = pd.Timestamp(t)
            # side='left': a filing with datekey == t is NOT usable at t —
            # its datekey covers after-close EDGAR acceptances, so the day-t
            # close printed before the filing was public (and the tradeable
            # desk only fills T+1 anyway).
            i = int(np.searchsorted(keys, np.datetime64(ts), side='left'))
            if i == 0:
                continue                       # nothing filed yet (PIT)
            dk = pd.Timestamp(keys[i - 1])
            if (ts - dk).days > fresh_days:
                continue                       # stale filing — drift decayed
            pos = int(idx.searchsorted(ts))
            if pos + maxh >= len(px) or pos >= len(px):
                continue
            entry = vals[pos]
            if not entry or entry <= 0:
                continue
            live.append((ts, pos, entry, svals[i - 1]))
        if not live:
            continue
        caps = (batch(name, [t for t, _, _, _ in live]) if batch is not None
                else [prov.daily_metric(name, t) for t, _, _, _ in live])
        for (ts, pos, entry, sue), metrics in zip(live, caps):
            rec = {'date': ts, 'name': name, 'sue': sue,
                   'mcap': (metrics or {}).get('marketcap')}
            for h in horizons:
                rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
            recs.append(rec)
    cols = ['date', 'name', 'sue', 'mcap'] + [f'fwd_{h}' for h in horizons]
    return pd.DataFrame(recs, columns=cols), n_names


def add_size_terciles(events: pd.DataFrame) -> pd.DataFrame:
    """Per-date size tercile (0=micro..2=mid) over each event cross-section's
    PIT marketcap; NaN where the cap is missing or the date has < 3 names."""
    events = events.copy()

    def _rank(g):
        caps = pd.to_numeric(g['mcap'], errors='coerce')
        ok = caps.notna() & (caps > 0)
        if ok.sum() < 3:
            return pd.Series(np.nan, index=g.index)
        r = caps[ok].rank(method='first')
        terc = np.minimum(2, ((r - 1) * 3 / ok.sum()).astype(int))
        return terc.reindex(g.index)

    events['tercile'] = (events.groupby('date', group_keys=False)
                         .apply(_rank, include_groups=False))
    return events


def _selftest():
    """Hermetic: synthetic EPS + prices through the full stats path."""
    from data.earnings_surprise import latest_fresh_sue
    rng = np.random.default_rng(7)
    # 12 quarters of EPS for 2 names; B has a big positive surprise at the end.
    rows = []
    for name, jump in (('AAA', 0.0), ('BBB', 3.0)):
        for q in range(12):
            rp = pd.Timestamp('2020-03-31') + pd.DateOffset(months=3 * q)
            rows.append({'ticker': name, 'reportperiod': rp,
                         'datekey': rp + pd.Timedelta(days=40),
                         'epsdil': 1.0 + 0.1 * q
                         + (jump if q == 11 and name == 'BBB' else 0.0)
                         + rng.normal(0, 0.02)})
    sues = sue_table(pd.DataFrame(rows))
    assert set(sues['ticker']) == {'AAA', 'BBB'}, sues
    # earliest computable filing needs 4 seasonal diffs -> q8 at the earliest
    assert sues.groupby('ticker').size().min() >= 3
    asof = pd.Timestamp('2023-03-01')            # after q11's datekey, fresh
    s = latest_fresh_sue(sues, asof, fresh_days=63)
    # BBB's jump is a real surprise; AAA's steady growth is drift, NOT surprise.
    assert s['BBB'] > 5.0 and abs(s['AAA']) < 5.0, s
    # Stale filings are dropped.
    assert len(latest_fresh_sue(sues, asof + pd.Timedelta(days=200))) == 0
    # PIT: nothing visible before the first datekey.
    assert len(latest_fresh_sue(sues, pd.Timestamp('2019-01-01'))) == 0
    # factor_study wiring: high-SUE long must recover a planted spread.
    dates = pd.bdate_range('2021-01-01', '2022-12-01', freq='BMS')
    ev = []
    for t in dates:
        for i in range(20):
            sue = (i - 10) / 3.0
            ev.append({'date': t, 'name': f'N{i}', 'sue': sue,
                       'mcap': 1e8 * (1 + i),
                       'fwd_21': 0.002 * sue + rng.normal(0, 0.001)})
    stats = factor_study(pd.DataFrame(ev), 'sue', [21], 0.0,
                         cheap_is_long=False, drop_nonpositive=False,
                         winsor_returns=0.01)
    assert not stats[0].get('insufficient') and stats[0]['t'] > 3, stats
    terc = add_size_terciles(pd.DataFrame(ev))
    assert set(terc['tercile'].dropna().unique()) == {0, 1, 2}
    print('selftest OK')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.pead_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-09-30')
    ap.add_argument('--horizons', nargs='+', type=int, default=[21, 63])
    ap.add_argument('--cost-bps', type=float, default=30.0)
    ap.add_argument('--fresh-days', type=int, default=63)
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe for a faster smoke')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date forward-return winsorization (0 = off)')
    ap.add_argument('--dating', choices=['filing', 'announce'],
                    default='filing',
                    help="event timestamp: SF1 filing datekey (strict PIT) "
                         "or 8-K press-release date (needs events ingested; "
                         "documented 8-K-EPS==10-Q-EPS approximation)")
    ap.add_argument('--field', choices=['epsdil', 'revenue'],
                    default='epsdil',
                    help="SF1 quarterly column to standardize: 'epsdil' = "
                         "classic SUE, 'revenue' = SURGE (standardized "
                         "revenue-growth surprise, Jegadeesh-Livnat 2006). "
                         "With --dating announce the EPS approximation "
                         "extends to revenue: 8-K 2.02 press releases carry "
                         "revenue too, and we treat it as equal to the later "
                         "10-Q figure (same known-at-release caveat).")
    ap.add_argument('--selftest', action='store_true')
    cli = ap.parse_args(argv)
    if cli.selftest:
        _selftest()
        return 0

    import random
    wh = PitWarehouse()
    rebal = pd.bdate_range(cli.start, cli.end, freq='BMS')
    names = resolve_universe(wh, rebal, SCALE_SMALL_MID)
    if cli.limit and cli.limit < len(names):
        names = sorted(random.Random(cli.seed).sample(names, cli.limit))
    pstart = (pd.Timestamp(cli.start) - pd.Timedelta(days=30)).date().isoformat()
    pend = (pd.Timestamp(cli.end)
            + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()
    print(f"names={len(names)} rebalances={len(rebal)} "
          f"fresh_days={cli.fresh_days}", file=sys.stderr)

    ann = None
    if cli.dating == 'announce':
        ann = wh.earnings_events(list(names))
        if not len(ann):
            raise SystemExit("--dating announce needs the events table: "
                             "python -m data.pit_warehouse --tables events")
        print(f"announcements={len(ann)}", file=sys.stderr)
    events, n_names = collect_sue_events(
        wh, names, rebal, cli.horizons, fresh_days=cli.fresh_days,
        price_start=pstart, price_end=pend, announcements=ann,
        field=cli.field)
    events = add_size_terciles(events)

    slices = [('pooled', events)] + [
        (f'tercile{b}', events[events['tercile'] == b]) for b in (0, 1, 2)]
    all_stats, pvals = [], []
    for label, ev in slices:
        stats = factor_study(ev, 'sue', cli.horizons, cli.cost_bps,
                             cheap_is_long=False, drop_nonpositive=False,
                             winsor_returns=cli.winsor or None)
        for s in stats:
            s['factor'] = label                    # reuse the report printer
        all_stats.extend(stats)
        pvals.extend(s['p'] for s in stats if not s.get('insufficient'))
    bh = benjamini_hochberg(pvals) if pvals else None
    label = 'PEAD-SUE' if cli.field == 'epsdil' else 'SURGE-REV'
    _print_report(all_stats, bh, cli.cost_bps, n_names, len(events),
                  label=label)
    print("\n(tercile0=micro, 1=small, 2=mid — per-date event cross-sections "
          "on PIT marketcap; high SUE = long)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
