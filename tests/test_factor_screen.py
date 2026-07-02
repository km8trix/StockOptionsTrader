"""Tests for the pure factor_study core of scripts/factor_screen."""

import numpy as np
import pandas as pd

from scripts.factor_screen import factor_study


def _events(n_dates=24, k_per=20, edge=0.02, seed=0):
    """Each date: k_per names with a value ratio in [1, k_per]; the CHEAP names
    (low ratio) get a higher forward return by `edge`, so a cheap-is-long
    long-short should earn ~`edge` (+ a per-date market shock that cancels)."""
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        shock = rng.normal(0, 0.05)                 # common market move (cancels)
        date = pd.Timestamp('2015-01-01') + pd.offsets.BMonthBegin(d)
        for i in range(k_per):
            ratio = float(i + 1)                    # 1 = cheapest
            # cheaper (low ratio) -> higher return: linear tilt around the mean
            tilt = edge * (k_per / 2 - i) / k_per
            recs.append({'date': date, 'name': f'N{i}', 'pb': ratio,
                         'fwd_21': shock + tilt + rng.normal(0, 1e-4)})
    return pd.DataFrame(recs)


def test_cheap_long_earns_positive_spread():
    s = factor_study(_events(), 'pb', [21], cost_bps=0.0)[0]
    assert s['gross_spread'] > 0        # cheap outperforms rich
    assert s['t'] > 3
    assert s['n_dates'] == 24


def test_positive_spread_survives_market_shock():
    # the per-date common shock cancels in the long-short; spread stays positive
    base = factor_study(_events(seed=1), 'pb', [21], cost_bps=0.0)[0]
    assert base['gross_spread'] > 0


def test_nonpositive_factor_values_dropped():
    ev = _events()
    ev.loc[ev['name'] == 'N0', 'pb'] = -1.0     # a "cheap" trap (neg ratio)
    s = factor_study(ev, 'pb', [21], cost_bps=0.0)[0]
    # still forms a book from the remaining positive names; no crash
    assert not s.get('insufficient')


def test_cost_nets_both_legs():
    s = factor_study(_events(), 'pb', [21], cost_bps=25.0)[0]
    assert abs(s['net_spread'] - (s['gross_spread'] - 0.0050)) < 1e-12


def test_too_few_dates_flagged():
    s = factor_study(_events(n_dates=6), 'pb', [21], cost_bps=0.0)[0]
    assert s['insufficient'] is True
