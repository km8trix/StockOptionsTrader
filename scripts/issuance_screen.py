"""Net share issuance screen — YoY split-adjusted share growth on the
survivorship-free dated eligible universe, through the research apparatus.

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
KNOWN at its datekey; at each first-observed-session monthly formation a name
carries its latest visible filing, dropped when staler than --stale-days. The
formation-session close supplies the dated size bucket, so execution is delayed
to the next observed session (T+1); a name without a formation-session bar is
dropped instead of silently entering later.

Tested with the exact machinery that validated/killed insider, value, quality
and PEAD: factor_screen.factor_study (per-date Fama-MacBeth spread,
Newey-West/HAC t, per-date winsorized forward returns), Benjamini-Hochberg
across the whole family (pooled + size terciles x horizons), using the
historical 30bp/leg fixed-cost convention corrected below. Also reports the
month-over-month name turnover of the LONG
(buyback) leg — the thesis is that a slow annual signal dodges the cost drag
that killed the monthly signals, so the turnover is measured, not assumed.

Deterministic; warehouse-only (tickers + sep + sf1 + daily ingested).

    python -m scripts.issuance_screen                # full universe
    python -m scripts.issuance_screen --limit 250    # seeded smoke subset
    python -m scripts.issuance_screen --selftest     # offline, no warehouse

STATISTICAL STATUS (2026-07-13): the recorded result below predates the net-array
inference repair in ``factor_study``. Its displayed means were cost-net, but its
t/p and BH decisions tested GROSS spreads. It is retained only as research
provenance and is not evidence of a net tradeable edge until rerun. The current
report separates raw economic P&L from winsorized robustness inference, deducts
cost per formation date before HAC, and sends one-sided net p-values to BH. The
legacy implementation deducted only two 30bp charges while the CLI described a
one-way cost. The corrected fixed model charges four trades (120bp total at the
default), making the historical net means stale as well.

LEGACY RESULT (2026-07-10, full universe 4608 names / 306,453 events, winsor 0.01):
8/8 BH survivors at alpha=0.05 — the program's first clean sweep. Pooled 21d
net +1.55% t=+3.76, 63d net +4.83% t=+3.42; micro 63d net +8.21% t=+4.02;
and uniquely, MID-CAP survives (21d net +0.44% t=+2.67, 63d net +1.89%
t=+2.79) — the tradeable slice where value/quality/PEAD all died. Median
issuance +1.0%/yr, 21.9% of events negative (buybacks). Measured long-leg
turnover: 7.3%/rebalance (~87%/yr one-sided, mean leg 523 names) — slower
than monthly signals but NOT annual. The net figures above used the erroneous
2x30bp fixed convention. A corrected fixed run charges 4x30bp; a lower
turnover-based claim requires complete dated long/short turnover and execution
cost inputs and cannot be inferred from long-leg membership turnover alone.
Screen-level only; no desk, no promotion claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
from scripts.insider_screen import (  # noqa: E402
    _eligible_on,
    resolve_universe_membership,
    universe_union,
)
from scripts.pead_screen import add_size_terciles  # noqa: E402
from utils.provenance import capture_run_provenance  # noqa: E402


def monthly_formation_dates(sessions, start, end) -> pd.DatetimeIndex:
    """First observed trading session of every calendar month in range.

    Missing months fail closed instead of being replaced with a weekday that
    has no market data.  The caller can compare the result with the expected
    month count before running a qualifying study.
    """
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if lo > hi:
        raise ValueError("start must be on or before end")
    idx = pd.DatetimeIndex(sessions).dropna().normalize().unique().sort_values()
    idx = idx[(idx >= lo) & (idx <= hi)]
    if idx.empty:
        return pd.DatetimeIndex([], name='date')
    frame = pd.DataFrame({'session': idx})
    first = frame.groupby(frame['session'].dt.to_period('M'), sort=True)[
        'session'].min()
    return pd.DatetimeIndex(first.to_numpy(), name='date')


def _canonical_utc_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.endswith('Z'):
        raise ValueError('recorded_at must be canonical UTC ending in Z')
    try:
        parsed = datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError as exc:
        raise ValueError('recorded_at must be a valid UTC timestamp') from exc
    parsed = parsed.astimezone(timezone.utc)
    timespec = 'microseconds' if parsed.microsecond else 'seconds'
    canonical = parsed.isoformat(timespec=timespec).replace('+00:00', 'Z')
    if canonical != value:
        raise ValueError(f'recorded_at must be canonical: {canonical}')
    return canonical


def _write_development_report(path_value, report) -> str:
    """Create one canonical report, allowing only byte-identical retries."""
    encoded = (json.dumps(
        report, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False) + '\n').encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(
                f'development report already exists with different bytes: {path}')
        return digest
    with path.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def collect_issuance_events(prov, names, rebal_dates, horizons, *,
                            stale_days=400, price_start=None, price_end=None,
                            membership_by_date=None, price_batch_size=250):
    """One row per (name, formation) with a live PIT issuance: columns
    ['date','entry_date','name','issuance','mcap'] + fwd_h. Cohort key is the
    formation date and returns enter on the next observed session. stale_days:
    drop names whose latest filing is older than this at the formation
    (default 400 ~ the seasonal-match upper bound: a live
    quarterly filer is always fresh; only names that stopped filing over a
    year ago fall out, where a "YoY" count describes ancient history)."""
    maxh = max(horizons)
    shares = prov.fundamentals_quarterly(
        list(names), fields=('sharesbas', 'sharefactor'))
    rows = issuance_table(shares)
    n_names = 0
    recs = []
    batch = getattr(prov, 'daily_metrics', None)
    price_bulk = getattr(prov, 'prices_bulk', None)
    cap_bulk = getattr(prov, 'daily_marketcaps_for_dates', None)
    groups = [(name, group) for name, group in rows.groupby(
        'ticker', sort=False)]
    if type(price_batch_size) is not int or price_batch_size < 1:
        raise ValueError('price_batch_size must be a positive integer')
    for offset in range(0, len(groups), price_batch_size):
        group_batch = groups[offset:offset + price_batch_size]
        batch_names = [name for name, _ in group_batch]
        prices = (price_bulk(batch_names, price_start, price_end)
                  if price_bulk is not None else {})
        marketcaps = (cap_bulk(batch_names, rebal_dates)
                      if cap_bulk is not None else {})
        for name, g in group_batch:
            px = (prices.get(name, pd.Series(dtype=float, name=name))
                  if price_bulk is not None
                  else prov.prices(name, price_start, price_end))
            if len(px) < 100:
                continue
            idx, vals = px.index, px.values
            g = g.sort_values('datekey')
            keys = g['datekey'].to_numpy(dtype='datetime64[ns]')
            ivals = g['issuance'].to_numpy(dtype=float)
            live = []
            for t in rebal_dates:
                ts = pd.Timestamp(t)
                if not _eligible_on(name, ts, membership_by_date):
                    continue
                # side='left': a filing with datekey == t is NOT usable at t —
                # its datekey covers after-close EDGAR acceptances, so the day-t
                # close printed before the filing was public.
                i = int(np.searchsorted(keys, np.datetime64(ts), side='left'))
                if i == 0:
                    continue                       # nothing filed yet (PIT)
                dk = pd.Timestamp(keys[i - 1])
                if (ts - dk).days > stale_days:
                    continue                       # stopped filing — stale count
                formation_pos = int(idx.searchsorted(ts))
                if formation_pos >= len(px):
                    continue
                # The dated market cap used for the size rank is known only at
                # this session's close. Require that exact formation bar, then
                # enter at the following observed close (T+1). A halted/missing
                # formation bar may not slide the signal into a later session.
                if pd.Timestamp(idx[formation_pos]).normalize() != ts.normalize():
                    continue
                entry_pos = formation_pos + 1
                if entry_pos + maxh >= len(px):
                    continue
                entry = vals[entry_pos]
                if not entry or entry <= 0:
                    continue
                live.append((ts, entry_pos, entry, ivals[i - 1]))
            if not live:
                continue
            n_names += 1
            if cap_bulk is not None:
                caps = [
                    ({'marketcap': marketcaps.get((name, ts.date()))}
                     if (name, ts.date()) in marketcaps else None)
                    for ts, _, _, _ in live
                ]
            else:
                caps = (batch(name, [t for t, _, _, _ in live])
                        if batch is not None else [
                            prov.daily_metric(name, t)
                            for t, _, _, _ in live])
            for (ts, pos, entry, iss), metrics in zip(live, caps):
                rec = {'date': ts, 'entry_date': pd.Timestamp(idx[pos]),
                       'name': name, 'issuance': iss,
                       'mcap': (metrics or {}).get('marketcap')}
                for h in horizons:
                    rec[f'fwd_{h}'] = vals[pos + h] / entry - 1.0
                recs.append(rec)
    cols = ['date', 'entry_date', 'name', 'issuance', 'mcap'] + [
        f'fwd_{h}' for h in horizons]
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
    ap.add_argument('--cost-bps', type=float, default=30.0,
                    help='one-way cost for one trade on one leg; charges entry '
                         'and exit on both long and short legs (4x)')
    ap.add_argument('--stale-days', type=int, default=400)
    ap.add_argument('--limit', type=int, default=None,
                    help='cap the universe for a faster smoke')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--winsor', type=float, default=0.01,
                    help='per-date robust-inference winsorization; raw economic '
                         'P&L is always retained (0 = off)')
    ap.add_argument(
        '--output-json', default=None,
        help='create a canonical development evidence report with dated arrays')
    ap.add_argument(
        '--recorded-at', default=None,
        help='explicit canonical UTC timestamp required with --output-json')
    ap.add_argument('--selftest', action='store_true')
    cli = ap.parse_args(argv)
    if (cli.output_json is None) != (cli.recorded_at is None):
        ap.error('--output-json and --recorded-at must be supplied together')
    recorded_at = (_canonical_utc_timestamp(cli.recorded_at)
                   if cli.recorded_at is not None else None)
    if cli.selftest:
        _selftest()
        return 0

    import random
    wh = PitWarehouse()
    sessions = wh.market_sessions(cli.start, cli.end)
    rebal = monthly_formation_dates(sessions, cli.start, cli.end)
    expected_months = len(pd.period_range(cli.start, cli.end, freq='M'))
    if len(rebal) != expected_months:
        print(
            f"incomplete SEP session calendar: expected {expected_months} "
            f"months, found {len(rebal)} — refusing to screen",
            file=sys.stderr,
        )
        return 1
    # Full live PIT universe.  Size slices are formed from dated market cap;
    # Sharadar's current ``scalemarketcap`` field is never a qualifying filter.
    membership = resolve_universe_membership(wh, rebal)
    names = universe_union(membership)
    if cli.limit and cli.limit < len(names):
        names = sorted(random.Random(cli.seed).sample(names, cli.limit))
    pstart = (pd.Timestamp(cli.start) - pd.Timedelta(days=30)).date().isoformat()
    pend = (pd.Timestamp(cli.end)
            + pd.Timedelta(days=max(cli.horizons) * 2 + 30)).date().isoformat()
    print(f"names={len(names)} rebalances={len(rebal)} "
          f"stale_days={cli.stale_days}", file=sys.stderr)

    events, n_names = collect_issuance_events(
        wh, names, rebal, cli.horizons, stale_days=cli.stale_days,
        price_start=pstart, price_end=pend,
        membership_by_date=membership)
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
                             winsor_returns=cli.winsor or None,
                             include_series=cli.output_json is not None)
        for s in stats:
            s['factor'] = label                    # reuse the report printer
        all_stats.extend(stats)
        pvals.extend(s['p'] for s in stats if not s.get('insufficient'))
    bh = benjamini_hochberg(pvals) if pvals else None
    _print_report(all_stats, bh, cli.cost_bps, n_names, len(events),
                  label='net-issuance')
    print("\n(sign: HIGH issuance predicts LOW returns — long = LOW/NEGATIVE "
          "YoY issuance i.e. buybacks, short = heavy issuers; "
          "tercile0=smallest, 1=middle, 2=largest relative to each dated "
          "eligible cross-section on PIT marketcap)")
    to = long_leg_turnover(events)
    if to:
        print(f"long-leg (buyback quintile) turnover: "
              f"{to['per_rebalance']:.1%}/rebalance over {to['n_pairs']} "
              f"monthly pairs (mean leg {to['mean_leg_size']:.0f} names) "
              f"= {to['annualized']:.0%}/yr one-sided")
    if cli.output_json is not None:
        valid = [item for item in all_stats if not item.get('insufficient')]
        rejected = bh['rejected_bh'] if bh else [False] * len(valid)
        survivors = [
            {'slice': item['factor'], 'horizon_days': item['h']}
            for item, reject in zip(valid, rejected)
            if reject and item['raw_net_spread'] > 0
        ]
        report = {
            'schema_version': 1,
            'report_type': 'development_factor_screen',
            'evidence_class': 'development',
            'candidate_id': 'issuance-midcap-ls-v1',
            'recorded_at': recorded_at,
            'disposition': (
                'development_screen_survived' if survivors
                else 'stop_candidate_no_corrected_net_edge'),
            'disposition_reason': (
                'No member of the declared eight-test screen family survived '
                'Benjamini-Hochberg on raw cost-net cohort returns.'
                if not survivors else
                'At least one development slice survived; this is hypothesis '
                'generation only and is not holdout evidence.'),
            'configuration': {
                'start': pd.Timestamp(cli.start).date().isoformat(),
                'end': pd.Timestamp(cli.end).date().isoformat(),
                'horizons_days': list(cli.horizons),
                'one_way_cost_bps_per_trade_per_leg': float(cli.cost_bps),
                'fixed_total_round_trip_bps': float(4 * cli.cost_bps),
                'stale_days': int(cli.stale_days),
                'quantile': 0.2,
                'winsor_fraction_diagnostic_only': float(cli.winsor),
                'formation_calendar': 'first_observed_sep_session_of_month',
                'entry_timing': 'next_observed_session_close_t_plus_1',
                'size_slices': 'relative_dated_market_cap_terciles',
                'long_leg': 'lowest_net_share_issuance_quintile',
                'short_leg': 'highest_net_share_issuance_quintile',
            },
            'counts': {
                'eligible_name_union': len(names),
                'names_with_events': int(n_names),
                'events': len(events),
                'formation_dates': len(rebal),
                'family_tests': len(valid),
            },
            'descriptives': {
                'median_issuance': float(events['issuance'].median()),
                'fraction_negative_issuance': neg,
                'long_leg_turnover': to,
            },
            'tests': all_stats,
            'multiple_testing': bh,
            'positive_net_bh_survivors': survivors,
            'warehouse_snapshot': wh.snapshot_version(
                ('daily', 'sep', 'sf1', 'tickers')),
            'provenance': capture_run_provenance(seed=cli.seed),
            'qualification': {
                'qualifying_evidence': False,
                'raw_event_rows_preserved': False,
                'blockers': [
                    'development window was previously inspected',
                    'historical borrow/locate/fee evidence is absent',
                    'independent filing-derived share reconstruction is absent',
                    'corporate-actions table is absent from the local warehouse',
                    'spread/impact costs are fixed rather than calibrated',
                    'working tree is not a sealed clean revision',
                ],
            },
        }
        digest = _write_development_report(cli.output_json, report)
        print(f"development report: {cli.output_json} sha256={digest}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
