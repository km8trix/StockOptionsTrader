"""Tests for the quality screen: SF1 collect + high-is-long / keep-negatives study."""

import numpy as np
import pandas as pd

from scripts.factor_screen import factor_study
from scripts.quality_screen import collect_quality_events


class _FakeProv:
    """Minimal PitWarehouse stand-in: rising prices + a per-name SF1 dict."""

    def __init__(self, funds):
        self._funds = funds
        self._idx = pd.bdate_range('2014-12-01', periods=400)

    def prices(self, name, start, end):
        return pd.Series(np.linspace(10, 20, len(self._idx)),
                         index=self._idx, name=name)

    def fundamentals_asof(self, name, t):
        return self._funds.get(name)


def test_collect_keeps_negatives_and_coerces_none():
    funds = {'A': {'roe': 0.20}, 'B': {'roe': -0.05}, 'C': {'roe': None}}
    prov = _FakeProv(funds)
    rebal = pd.bdate_range('2015-01-01', '2015-06-30', freq='BMS')
    ev, n = collect_quality_events(prov, ['A', 'B', 'C'], rebal, [21], ['roe'],
                                   '2014-12-01', '2016-06-30')
    assert n == 3                               # all have >=100 price points
    assert set(ev['name']) == {'A', 'B', 'C'}
    assert (ev.loc[ev['name'] == 'B', 'roe'] < 0).all()   # negative kept at collect
    assert ev.loc[ev['name'] == 'C', 'roe'].isna().all()  # None -> NaN
    assert 'fwd_21' in ev.columns


def _q_events(n_dates=24, k=20, edge=0.02, seed=0):
    """Each date: k names with roe spanning negative..positive; HIGH roe gets a
    higher forward return, so a high-is-long book should earn ~edge."""
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        shock = rng.normal(0, 0.05)                 # common move (cancels in L/S)
        date = pd.Timestamp('2015-01-01') + pd.offsets.BMonthBegin(d)
        for i in range(k):
            roe = (i - k / 2) / k                   # spans ~[-0.5, +0.45]
            tilt = edge * (i - k / 2) / k           # HIGH roe -> higher return
            recs.append({'date': date, 'name': f'N{i}', 'roe': roe,
                         'fwd_63': shock + tilt + rng.normal(0, 1e-4)})
    return pd.DataFrame(recs)


def test_high_quality_long_earns_positive_spread():
    s = factor_study(_q_events(), 'roe', [63], cost_bps=0.0,
                     cheap_is_long=False, drop_nonpositive=False)[0]
    assert s['gross_spread'] > 0        # high roe outperforms low
    assert s['t'] > 3


def test_drop_nonpositive_changes_the_book():
    # the quality screen must KEEP unprofitable (negative-roe) names as the short
    # leg; dropping them (value behavior) shrinks the book and changes the spread.
    ev = _q_events()
    kept = factor_study(ev, 'roe', [63], 0.0, cheap_is_long=False,
                        drop_nonpositive=False)[0]
    dropped = factor_study(ev, 'roe', [63], 0.0, cheap_is_long=False,
                           drop_nonpositive=True)[0]
    assert kept['gross_spread'] != dropped['gross_spread']
