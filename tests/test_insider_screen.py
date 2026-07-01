"""Tests for the Fama-MacBeth event-study core of scripts/insider_screen.

The verdict rests on three things that can silently corrupt it: the per-date
long-short spread must difference out a common market shock, the Newey-West t
must run on the SERIES of monthly spreads (one obs per date, not pooled events),
and the cost-netting must charge both legs. Each has a test.
"""

import numpy as np
import pandas as pd

from scripts.insider_screen import event_study


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
    noise = lambda k: 0.005 * np.sin(k)
    base = event_study(_events(spread_noise=noise), [21], cost_bps=0.0)[0][0]
    shocked = event_study(
        _events(spread_noise=noise, shock=lambda k: 0.5 * (-1) ** k),
        [21], cost_bps=0.0)[0][0]
    assert abs(base['gross_spread'] - shocked['gross_spread']) < 1e-9


def test_cost_nets_both_legs():
    stats, _ = event_study(_events(spread_noise=lambda k: 0.005 * np.sin(k)),
                           [21], cost_bps=30.0)
    s = stats[0]
    # 30bp/leg round-trip on long AND short = 60bp off the gross spread
    assert abs(s['net_spread'] - (s['gross_spread'] - 0.0060)) < 1e-12


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
