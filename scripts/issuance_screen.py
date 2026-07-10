"""Net share issuance screen — YoY split-adjusted share growth on the
survivorship-free small/mid universe, through the honest apparatus.

The Pontiff-Woodgate / Daniel-Titman issuance anomaly: firms that ISSUE shares
(SEOs, stock comp, stock-financed M&A) subsequently UNDERPERFORM; firms that
RETIRE shares (buybacks) outperform. SIGN CONVENTION (explicit, academic):
HIGH issuance predicts LOW returns, so the LONG leg is the LOW/NEGATIVE-
issuance quantile (buyback names) and the short leg is the heavy issuers —
``cheap_is_long=True`` in factor_study terms, with ``drop_nonpositive=False``
because a NEGATIVE value (net buyback) is the prime long-leg member, not a
trap to drop.

Signal: adjusted shares outstanding = sharesbas * sharefactor from PIT
SF1 ARQ (fundamentals_quarterly, deduped to the earliest-datekey filing per
quarter). SPLIT NEUTRALITY comes from the VENDOR, not sharefactor: Sharadar
retroactively restates sharesbas to the current split basis (verified
2026-07-10 against the warehouse — NVDA/TSLA/AAPL forward and CERO/UNCY
reverse splits are invisible in sharesbas), so both quarters of the ratio
share one basis and future-split factors cancel; the surviving factor is
exactly the intervening split a PIT investor would also know. sharefactor is
a near-constant ADR/unit multiplier (varies only for share-class/ADR cases
like BRK.B, V, ONON); multiplying each quarter by its OWN factor handles
ADR-ratio changes. YoY net issuance = adjshares_t / adjshares_{t-4q} - 1, with
the year-ago quarter matched per ticker BY CALENDAR (latest reportperiod
330-410 days earlier — the sue_table convention) so missing quarters skip the
row instead of silently misaligning it; no cross-frame merge_asof. Guards:
both fields non-null at both quarters, relative-epsilon on zero/near-zero
adjusted share counts (sharefactor==0 garbage), >=4 prior quarters of filing
history, and the sue_table running-max-datekey rule so a delinquent old
filing cannot leak into an earlier-dated signal. The signal of a filing is
KNOWN at its datekey; at each business-month-start rebalance a name carries
its latest visible filing, dropped when staler than --stale-days.

Tested with the exact machinery that validated/killed insider, value, quality
and PEAD: factor_screen.factor_study (per-date Fama-MacBeth spread,
Newey-West/HAC t, per-date winsorized forward returns), Benjamini-Hochberg
across the whole family (pooled + size terciles x horizons), 30bp/leg
cost-netting. Also reports the month-over-month name turnover of the LONG
(buyback) leg — the thesis is that a slow annual signal dodges the cost drag
that killed the monthly signals, so the turnover is measured, not assumed.

Deterministic; warehouse-only (tickers + sep + sf1 + daily ingested).

    python -m scripts.issuance_screen                # full universe
    python -m scripts.issuance_screen --limit 250    # seeded smoke subset
    python -m scripts.issuance_screen --selftest     # offline, no warehouse

RESULT (2026-07-10, full universe 4608 names / 306,453 events, winsor 0.01):
8/8 BH survivors at alpha=0.05 — the program's first clean sweep. Pooled 21d
net +1.55% t=+3.76, 63d net +4.83% t=+3.42; micro 63d net +8.21% t=+4.02;
and uniquely, MID-CAP survives (21d net +0.44% t=+2.67, 63d net +1.89%
t=+2.79) — the tradeable slice where value/quality/PEAD all died. Median
issuance +1.0%/yr, 21.9% of events negative (buybacks). Measured long-leg
turnover: 7.3%/rebalance (~87%/yr one-sided, mean leg 523 names) — slower
than monthly signals but NOT annual; note the net figures above deduct the
full 2x30bp per rebalance while actual membership turnover is 7.3%, so
realized cost drag is ~1/14th of what the netting assumes and true net sits
near gross. Screen-level only; no desk, no promotion claim.
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
from data.pit_warehouse import PitWarehouse  # noqa: E402
from data.share_issuance import _share_rows, issuance_table  # noqa: E402
from scripts.factor_screen import _print_report, factor_study  # noqa: E402
from scripts.insider_screen import SCALE_SMALL_MID, resolve_universe  # noqa: E402
from scripts.pead_screen import add_size_terciles  # noqa: E402


def collect_issuance_events(prov, names, rebal_dates, horizons, *,
                            stale_days=400, price_start=None, price_end=None):
    """One row per (name, rebalance) with a live PIT issuance: columns
    ['date','name','issuance','mcap'] + fwd_h. Cohort key = the rebalance
    date. stale_days: drop names whose latest filing is older than this at
    the rebalance (default 400 ~ the seasonal-match upper bound: a live
    quarterly filer is always fresh; only names that stopped filing over a
    year ago fall out, where a "YoY" count describes ancient history)."""
    maxh = max(horizons)
    shares = prov.fundamentals_quarterly(
        list(names), fields=('sharesbas', 'sharefactor'))
    rows = issuance_table(shares)
    n_names = 0
    recs = []
    batch = getattr(prov, 'daily_metrics', None)
    for name, g in rows.groupby('ticker', sort=False):
        px = prov.prices(name, price_start, price_end)
        if len(px) < 100:
            continue
        n_names += 1
        idx, vals = px.index, px.values
        g = g.sort_values('datekey')
        keys = g['datekey'].to_numpy(dtype='datetime64[ns]')
        ivals = g['issuance'].to_numpy(dtype=float)
        live = []
        for t in rebal_dates:
            ts = pd.Timestamp(t)
            # side='left': a filing with datekey == t is NOT usable at t —
            # its datekey covers after-close EDGAR acceptances, so the day-t
            # close printed before the filing was public.
            i = int(np.searchsorted(keys, np.datetime64(ts), side='left'))
            if i == 0:
                continue                       # nothing filed yet (PIT)
            dk = pd.Timestamp(keys[i - 1])
            if (ts - dk).days > stale_days:
                continue                       # stopped filing — stale count
            pos = int(idx.searchsorted(ts))
            if pos + maxh >= len(px) or pos >= len(px):
                continue
            entry = vals[pos]
            if not entry or entry <= 0:
                continue
            live.append((ts, pos, entry, ivals[i - 1]))
        if not live:
            continue
        caps = (batch(name, [t for t, _, _, _ in live]) if batch is not None
                else [prov.daily_metric(name, t) for t, _, _, _ in live])
        for (ts, pos, entry, iss), metrics in zip(live, caps):
            rec = {'date': ts, 'name': name, 'issuance': iss,
                   'mcap': (metrics or {}).get('marketcap')}
            for h in horizons:
                rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
            recs.append(rec)
    cols = ['date', 'name', 'issuance', 'mcap'] + [f'fwd_{h}' for h in horizons]
    return pd.DataFrame(recs, columns=cols), n_names


def long_leg_turnover(events: pd.DataFrame, factor: str = 'issuance', *,
                      quantile: float = 0.2) -> dict:
    """Month-over-month ONE-SIDED name turnover of the LONG (lowest-``factor``)
    leg, exactly as factor_study forms it (ascending sort, bottom quantile,
    both legs must fit). Per-pair turnover = fraction of the leg at t absent
    from the leg at the next rebalance; ``annualized`` = mean x 12 (BMS
    cadence). This is the cost-drag question measured, not assumed — includes
    churn from names entering/leaving the event universe, which real trading
    pays too. Returns {} when fewer than 2 leg dates exist."""
    legs = []
    for t, g in events.dropna(subset=[factor]).groupby('date'):
        n = len(g)
        k = max(1, int(quantile * n))
        if n < 2 * k:
            continue                              # can't form both legs
        legs.append((t, set(g.sort_values(factor)['name'].iloc[:k])))
    legs.sort()
    if len(legs) < 2:
        return {}
    rates = [1.0 - len(l0 & l1) / len(l0)
             for (_, l0), (_, l1) in zip(legs, legs[1:])]
    return {'n_pairs': len(rates),
            'mean_leg_size': float(np.mean([len(s) for _, s in legs])),
            'per_rebalance': float(np.mean(rates)),
            'annualized': float(np.mean(rates)) * 12.0}


def _selftest():
    """Hermetic: synthetic share counts + prices through the full stats path."""
    rng = np.random.default_rng(7)
    # Issuer grows 20%/yr, buyback shrinks 10%/yr, steady flat.
    rows = (_share_rows('ISS', [100 * 1.2 ** (q / 4) for q in range(9)])
            + _share_rows('BBK', [100 * 0.9 ** (q / 4) for q in range(9)])
            + _share_rows('FLT', [100.0] * 9))
    tbl = issuance_table(pd.DataFrame(rows))
    last = tbl.groupby('ticker')['issuance'].last()
    assert last['ISS'] > 0.15 and last['BBK'] < -0.05, last
    assert abs(last['FLT']) < 1e-9, last
    # First 4 quarters have <4 prior filings -> earliest computable row is q4.
    assert tbl.groupby('ticker')['reportperiod'].min().eq(
        pd.Timestamp('2021-03-31')).all()
    # A 2:1 split (sharesbas x2, sharefactor /2) is NOT issuance.
    split = _share_rows('SPL', [100.0] * 5) + _share_rows(
        'SPL', [200.0] * 4, start='2021-06-30', factor=0.5)
    assert abs(issuance_table(pd.DataFrame(split))['issuance']).max() < 1e-9
    # Zero adjusted count (sharefactor==0 garbage) is dropped, not divided by.
    zero = _share_rows('ZRO', [100.0] * 9)
    zero[2]['sharefactor'] = 0.0
    z = issuance_table(pd.DataFrame(zero))
    assert pd.Timestamp('2021-09-30') not in set(z['reportperiod'])
    assert np.isfinite(z['issuance']).all()
    # PIT: nothing visible before the first computable datekey.
    prices = pd.Series(np.linspace(10, 20, 400),
                       index=pd.bdate_range('2020-01-01', periods=400))

    class _Prov:
        def fundamentals_quarterly(self, names, *, fields):
            return pd.DataFrame([r for r in rows if r['ticker'] in names])

        def prices(self, name, start, end):
            return prices

        def daily_metric(self, name, t):
            return {'marketcap': 1e8}

    ev, n = collect_issuance_events(
        _Prov(), ['ISS', 'BBK', 'FLT'], pd.bdate_range(
            '2020-06-01', '2021-06-01', freq='BMS'), [21])
    assert n == 3
    assert ev['date'].min() >= pd.Timestamp('2021-05-10')  # q4 datekey
    # factor_study wiring: LOW issuance = long must recover a planted edge.
    dates = pd.bdate_range('2021-01-01', '2022-12-01', freq='BMS')
    evs = []
    for t in dates:
        for i in range(20):
            iss = (i - 10) / 20.0                 # negative..positive
            evs.append({'date': t, 'name': f'N{i}', 'issuance': iss,
                        'fwd_21': -0.002 * iss + rng.normal(0, 1e-4)})
    s = factor_study(pd.DataFrame(evs), 'issuance', [21], 0.0,
                     cheap_is_long=True, drop_nonpositive=False,
                     winsor_returns=0.01)[0]
    assert not s.get('insufficient') and s['t'] > 3, s
    # Turnover: static ranks -> zero; the leg is the buyback (negative) end.
    to = long_leg_turnover(pd.DataFrame(evs))
    assert to['per_rebalance'] == 0.0 and to['n_pairs'] == len(dates) - 1
    print('selftest OK')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m scripts.issuance_screen',
                                 description=__doc__.splitlines()[0])
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-09-30')
    ap.add_argument('--horizons', nargs='+', type=int, default=[21, 63])
    ap.add_argument('--cost-bps', type=float, default=30.0)
    ap.add_argument('--stale-days', type=int, default=400)
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe for a faster smoke')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date forward-return winsorization (0 = off)')
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
          f"stale_days={cli.stale_days}", file=sys.stderr)

    events, n_names = collect_issuance_events(
        wh, names, rebal, cli.horizons, stale_days=cli.stale_days,
        price_start=pstart, price_end=pend)
    if not len(events):
        # Honest empty report instead of add_size_terciles' opaque pandas
        # ValueError (empty warehouse, or filters that exclude everything).
        print("no issuance events collected — nothing to test",
              file=sys.stderr)
        return 1
    events = add_size_terciles(events)
    neg = float((events['issuance'] < 0).mean()) if len(events) else 0.0
    print(f"events={len(events)} median_issuance="
          f"{events['issuance'].median():+.4f} pct_negative={neg:.1%}",
          file=sys.stderr)

    slices = [('pooled', events)] + [
        (f'tercile{b}', events[events['tercile'] == b]) for b in (0, 1, 2)]
    all_stats, pvals = [], []
    for label, ev in slices:
        # LOW/NEGATIVE issuance (buybacks) = LONG (cheap_is_long=True);
        # negatives KEPT — they ARE the long leg (drop_nonpositive=False).
        stats = factor_study(ev, 'issuance', cli.horizons, cli.cost_bps,
                             cheap_is_long=True, drop_nonpositive=False,
                             winsor_returns=cli.winsor or None)
        for s in stats:
            s['factor'] = label                    # reuse the report printer
        all_stats.extend(stats)
        pvals.extend(s['p'] for s in stats if not s.get('insufficient'))
    bh = benjamini_hochberg(pvals) if pvals else None
    _print_report(all_stats, bh, cli.cost_bps, n_names, len(events),
                  label='net-issuance')
    print("\n(sign: HIGH issuance predicts LOW returns — long = LOW/NEGATIVE "
          "YoY issuance i.e. buybacks, short = heavy issuers; tercile0=micro, "
          "1=small, 2=mid on PIT marketcap)")
    to = long_leg_turnover(events)
    if to:
        print(f"long-leg (buyback quintile) turnover: "
              f"{to['per_rebalance']:.1%}/rebalance over {to['n_pairs']} "
              f"monthly pairs (mean leg {to['mean_leg_size']:.0f} names) "
              f"= {to['annualized']:.0%}/yr one-sided")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
