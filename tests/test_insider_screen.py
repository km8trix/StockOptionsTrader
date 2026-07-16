"""Tests for the Fama-MacBeth event-study core of scripts/insider_screen.

The verdict rests on three things that can silently corrupt it: the per-date
long-short spread must difference out a common market shock, the Newey-West t
must run on the SERIES of monthly spreads (one obs per date, not pooled events),
and the cost-netting must charge both legs. Each has a test.
"""

import numpy as np
import pandas as pd

from scripts.insider_screen import (
    collect_events,
    event_study,
    resolve_universe,
    resolve_universe_membership,
)


class _UniverseProvider:
    def __init__(self):
        self.calls = []

    def universe_asof(self, date, **kwargs):
        self.calls.append((pd.Timestamp(date), kwargs))
        return ['OLD', 'LIVE'] if pd.Timestamp(date).month == 1 else ['LIVE', 'NEW']


def test_qualifying_universe_never_uses_current_size_bucket():
    provider = _UniverseProvider()
    names = resolve_universe(
        provider, [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-02-03')])
    assert names == ['LIVE', 'NEW', 'OLD']
    assert [kwargs for _, kwargs in provider.calls] == [{}, {}]


def test_legacy_current_size_filter_is_explicit_and_reproducible():
    provider = _UniverseProvider()
    scale = ['3 - Small', '4 - Mid']
    resolve_universe(provider, [pd.Timestamp('2020-01-02')], scale)
    assert provider.calls[0][1] == {'scalemarketcap': scale}


def test_collector_enforces_membership_on_each_rebalance_date():
    dates = [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-02-03')]
    provider = _UniverseProvider()
    membership = resolve_universe_membership(provider, dates)
    idx = pd.bdate_range('2019-10-01', periods=180)
    prices = pd.Series(np.linspace(10.0, 20.0, len(idx)), index=idx)

    events, _ = collect_events(
        _StubProvider(prices), ['OLD', 'LIVE', 'NEW'], dates, horizons=[2],
        lookback=90, price_start='2019-10-01', price_end='2020-06-30',
        membership_by_date=membership,
    )

    observed = set(zip(events['date'], events['name']))
    assert observed == {
        (dates[0], 'OLD'), (dates[0], 'LIVE'),
        (dates[1], 'LIVE'), (dates[1], 'NEW'),
    }


class _StubProvider:
    """Minimal PIT provider: a fixed price series + a constant insider buy."""

    def __init__(self, px):
        self._px = px

    def prices(self, name, start, end, field='closeadj'):
        return self._px

    def insider_net_buys(self, name, asof, lookback_days=90):
        return {'net_value': 100.0, 'net_shares': 100.0, 'n_buys': 1,
                'n_sells': 0, 'window_start': None}


def test_collect_events_cohort_key_is_rebalance_date():
    # Price series whose FIRST bar (2020-06-03) is after the rebalance date
    # (2020-06-01, no bar). The recorded cohort date must be the rebalance date,
    # not the later entry bar — otherwise Fama-MacBeth cohorts fragment.
    idx = pd.bdate_range('2020-06-03', periods=120)
    px = pd.Series(np.arange(10.0, 130.0), index=idx, name='X')
    t = pd.Timestamp('2020-06-01')
    ev, n = collect_events(_StubProvider(px), ['X'], [t], horizons=[2],
                           lookback=90, price_start='2020-01-01',
                           price_end='2020-12-31')
    assert n == 1 and len(ev) == 1
    assert ev.iloc[0]['date'] == t          # NOT idx[0] == 2020-06-03
    assert ev.iloc[0]['dir'] == 'buy'


def _events(n_dates=24, buy_ab=0.03, sell_ab=-0.03, n=30,
            shock=lambda k: 0.0, spread_noise=lambda k: 0.0, seed=0):
    """Synthetic events over `n_dates` monthly cohorts. Each date k gets `n`
    buyers and `n` sellers; a buyer's forward-21 return is
    (buy_ab + spread_noise(k)/2 + shock(k) + within-date jitter) and a seller's
    is (sell_ab - spread_noise(k)/2 + shock(k) + jitter). So the per-date spread
    is (buy_ab - sell_ab) + spread_noise(k) and the common shock(k) cancels."""
    rng = np.random.default_rng(seed)
    mid = (n - 1) / 2.0
    recs = []
    for k in range(n_dates):
        d = pd.Timestamp('2015-01-01') + pd.offsets.BMonthBegin(k)
        for i in range(n):
            jit = (i - mid) * 1e-4  # symmetric -> preserves the group mean
            recs.append({'date': d, 'name': f'B{i}', 'nv': 1000.0 + i, 'dir': 'buy',
                         'fwd_21': buy_ab + spread_noise(k) / 2 + shock(k) + jit
                                   + rng.normal(0, 1e-4)})
            recs.append({'date': d, 'name': f'S{i}', 'nv': -1000.0 - i, 'dir': 'sell',
                         'fwd_21': sell_ab - spread_noise(k) / 2 + shock(k) + jit
                                   + rng.normal(0, 1e-4)})
    return pd.DataFrame(recs)


def test_spread_recovers_edge_and_is_significant():
    # per-date spread ~ 0.06 with mild cross-date noise -> finite, large HAC t.
    ev = _events(spread_noise=lambda k: 0.005 * np.sin(k))
    stats, bh = event_study(ev, [21], cost_bps=0.0)
    s = stats[0]
    assert s['n_dates'] == 24
    assert abs(s['gross_spread'] - 0.06) < 0.01
    assert np.isfinite(s['t']) and s['t'] > 3   # clean edge -> significant
    assert s['ic'] > 0
    assert bh['m'] == 1


def test_common_market_shock_differences_out():
    # A huge per-date market shock (+/-50%) must NOT change the spread, because
    # buy and sell share it and the long-short cancels it.
    def noise(k):
        return 0.005 * np.sin(k)
    base = event_study(_events(spread_noise=noise), [21], cost_bps=0.0)[0][0]
    shocked = event_study(
        _events(spread_noise=noise, shock=lambda k: 0.5 * (-1) ** k),
        [21], cost_bps=0.0)[0][0]
    assert abs(base['gross_spread'] - shocked['gross_spread']) < 1e-9


def test_cost_nets_both_legs():
    events = _events(spread_noise=lambda k: 0.005 * np.sin(k))
    gross, _ = event_study(events, [21], cost_bps=0.0)
    stats, _ = event_study(events, [21], cost_bps=30.0)
    s = stats[0]
    # 30bp/leg round-trip on long AND short = 60bp off the gross spread
    assert abs(s['net_spread'] - (s['gross_spread'] - 0.0060)) < 1e-12
    assert s['inference_basis'] == 'net'
    assert s['t'] < gross[0]['t']


def test_cost_can_destroy_gross_significance():
    # A small, stable gross edge is statistically strong before costs but
    # negative after a deliberately larger executable cost.  BH and p-values
    # must describe the latter, not the attractive gross series.
    events = _events(buy_ab=0.004, sell_ab=0.0,
                     spread_noise=lambda k: 0.0005 * np.sin(k))
    gross, gross_bh = event_study(events, [21], cost_bps=0.0)
    net, net_bh = event_study(events, [21], cost_bps=30.0)
    assert gross[0]['t'] > 3 and gross_bh['rejected_bh'] == [True]
    assert net[0]['net_spread'] < 0 and net[0]['t'] < 0
    assert net_bh['rejected_bh'] == [False]


def test_too_few_dates_flagged():
    stats, bh = event_study(_events(n_dates=6), [21], cost_bps=0.0)  # < min_dates
    assert stats[0]['insufficient'] is True
    assert stats[0]['n_dates'] == 6
    assert bh is None


def test_no_edge_not_significant():
    # buyers and sellers same distribution -> spread ~ 0, not significant.
    stats, _ = event_study(
        _events(buy_ab=0.0, sell_ab=0.0, spread_noise=lambda k: 0.0, seed=1),
        [21], cost_bps=0.0)
    assert abs(stats[0]['gross_spread']) < 0.01
    assert stats[0]['p'] > 0.05
