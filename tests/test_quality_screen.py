"""Tests for the quality screen: SF1 collect + high-is-long / keep-negatives study."""

import numpy as np
import pandas as pd

from scripts.factor_screen import factor_study
from scripts.quality_screen import (COMPUTED_RATIOS, collect_quality_events,
                                    computed_ratio)


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


def test_computed_ratio_guards():
    from decimal import Decimal
    assert computed_ratio({'gp': 30e6, 'assets': 100e6},
                          'gp', 'assets') == 0.3
    # Decimal inputs (DuckDB) are coerced
    assert computed_ratio({'gp': Decimal('30000000'),
                           'assets': Decimal('100000000')},
                          'gp', 'assets') == 0.3
    # either input missing -> None (same-quarter completeness guard)
    assert computed_ratio({'gp': None, 'assets': 100e6},
                          'gp', 'assets') is None
    assert computed_ratio({'gp': 30e6, 'assets': None},
                          'gp', 'assets') is None
    assert computed_ratio({}, 'gp', 'assets') is None
    # non-positive / relative-epsilon-sliver denominator -> None
    assert computed_ratio({'gp': 30e6, 'assets': 0.0}, 'gp', 'assets') is None
    assert computed_ratio({'gp': 30e6, 'assets': -5.0}, 'gp', 'assets') is None
    assert computed_ratio({'gp': 1e21, 'assets': 100e6},
                          'gp', 'assets') is None
    # unit-inconsistency floor: the real VIVS row (gp $M-scale, assets
    # raw-dollar) must be rejected, as must any sub-$1M assets base
    assert computed_ratio({'gp': 1_677_000.0, 'assets': 3_628.0},
                          'gp', 'assets') is None
    assert computed_ratio({'gp': 400_000.0, 'assets': 999_999.0},
                          'gp', 'assets') is None
    # negative NUMERATOR is legitimate (unprofitable firm = short-leg member)
    assert computed_ratio({'netinc': -10e6, 'assets': 100e6},
                          'netinc', 'assets') == -0.1


def test_collect_computes_gp_assets_and_roa():
    funds = {'A': {'gp': 40e6, 'assets': 100e6, 'netinc': 10e6},
             'B': {'gp': 5e6, 'assets': 0.0, 'netinc': None},  # guarded -> NaN
             'C': {'gp': None, 'assets': 200e6, 'netinc': -20e6}}
    prov = _FakeProv(funds)
    rebal = pd.bdate_range('2015-01-01', '2015-06-30', freq='BMS')
    ev, _ = collect_quality_events(prov, ['A', 'B', 'C'], rebal, [21],
                                   ['gp_assets', 'roa'],
                                   '2014-12-01', '2016-06-30')
    a = ev[ev['name'] == 'A']
    assert (a['gp_assets'] == 0.4).all() and (a['roa'] == 0.1).all()
    b = ev[ev['name'] == 'B']
    assert b['gp_assets'].isna().all() and b['roa'].isna().all()
    c = ev[ev['name'] == 'C']
    assert c['gp_assets'].isna().all() and (c['roa'] == -0.1).all()


def test_mcap_column_is_opt_in():
    funds = {'A': {'netmargin': 0.1, 'marketcap': 5e8},
             'B': {'netmargin': 0.2, 'marketcap': None}}
    prov = _FakeProv(funds)
    rebal = pd.bdate_range('2015-01-01', '2015-06-30', freq='BMS')
    args = (prov, ['A', 'B'], rebal, [21], ['netmargin'],
            '2014-12-01', '2016-06-30')
    ev, _ = collect_quality_events(*args)
    assert 'mcap' not in ev.columns                       # default: unchanged
    ev2, _ = collect_quality_events(*args, with_mcap=True)
    assert (ev2.loc[ev2['name'] == 'A', 'mcap'] == 5e8).all()
    assert ev2.loc[ev2['name'] == 'B', 'mcap'].isna().all()


def test_native_factors_unchanged_by_computed_additions():
    # netmargin rows/values and study stats must be BYTE-IDENTICAL whether or
    # not the computed factors ride along (they are additive columns only).
    funds = {f'N{i}': {'netmargin': (i - 3) / 10.0, 'gp': 10.0 * i,
                       'assets': 100.0, 'netinc': 2.0 * i - 5}
             for i in range(8)}
    prov = _FakeProv(funds)
    rebal = pd.bdate_range('2015-01-01', '2016-06-30', freq='BMS')
    names = sorted(funds)
    old, _ = collect_quality_events(prov, names, rebal, [21], ['netmargin'],
                                    '2014-12-01', '2016-06-30')
    new, _ = collect_quality_events(prov, names, rebal, [21],
                                    ['netmargin', 'gp_assets', 'roa'],
                                    '2014-12-01', '2016-06-30')
    pd.testing.assert_frame_equal(old, new[old.columns])
    s_old = factor_study(old, 'netmargin', [21], 30.0, cheap_is_long=False,
                         drop_nonpositive=False, winsor_returns=0.01)
    s_new = factor_study(new, 'netmargin', [21], 30.0, cheap_is_long=False,
                         drop_nonpositive=False, winsor_returns=0.01)
    np.testing.assert_equal(s_old, s_new)     # NaN-tolerant (degenerate t)


def test_computed_ratios_registry():
    assert COMPUTED_RATIOS == {'gp_assets': ('gp', 'assets'),
                               'roa': ('netinc', 'assets')}


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
