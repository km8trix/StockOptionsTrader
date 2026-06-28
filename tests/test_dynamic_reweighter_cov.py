"""Tests for the OPT-IN full-covariance risk parity mode on DynamicReweighter.

DynamicReweighter's ``weighting`` param gains 'risk_parity_cov': the DEFAULT
'risk_parity' stays byte-identical to the inverse-vol path (proved by the
headline test), and 'risk_parity_cov' opts into full-covariance risk parity that
DOWN-WEIGHTS correlated desks. All modes share the SAME degenerate_desks audit
and the SAME walk-forward curve slicing.

Pure and offline — no engine, no market data. Mirrors
tests/test_dynamic_reweighter_performance.py.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from desks.capital_allocator import CrossDeskCapitalAllocator
from desks.dynamic_reweighter import DynamicReweighter


class StubDesk:
    """Minimal stand-in: the reweighter only reads .key and writes
    .capital_allocation."""

    def __init__(self, key: str, capital_allocation: float = 0.5):
        self.key = key
        self.capital_allocation = capital_allocation


def curve(values: Sequence[float],
          start: str = '2022-01-03') -> List[Tuple[pd.Timestamp, float]]:
    index = pd.bdate_range(start, periods=len(values))
    return list(zip(index, [float(v) for v in values]))


# Low-vol vs high-vol equity paths (reused for the byte-identical default test).
LOW_VOL = [100.0 + (0.5 if i % 2 else 0.0) for i in range(30)]
HIGH_VOL = [100.0 + (5.0 if i % 2 else 0.0) for i in range(30)]


def _curve_from_returns(returns, start: str = '2022-01-03'):
    """Compound a return series into an equity curve (timestamp, level)."""
    level = 100.0
    levels = []
    for r in returns:
        level *= (1.0 + r)
        levels.append(level)
    return curve(levels, start=start)


def correlated_curves(seed: int = 42, n: int = 80, vol: float = 0.01,
                      idio: float = 0.4):
    """Equity curves for three desks: A and B share a factor (correlated), C is
    independent. Deterministic via numpy's frozen PCG64 stream."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    g = rng.standard_normal(n)
    a = vol * (f + idio * rng.standard_normal(n))
    b = vol * (f + idio * rng.standard_normal(n))
    c = vol * (g + idio * rng.standard_normal(n))
    return {'A': _curve_from_returns(a), 'B': _curve_from_returns(b),
            'C': _curve_from_returns(c)}


def hedge_curves(seed: int = 7, n: int = 80, vol: float = 0.01,
                 idio: float = 0.4):
    """Like correlated_curves but C is a HEDGE: negatively correlated with the
    A/B common factor. Full-cov risk parity is ill-defined there (negative
    marginal contribution) and degrades to inverse-vol — a numerical degrade
    with NO degenerate desk."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    a = vol * (f + idio * rng.standard_normal(n))
    b = vol * (f + idio * rng.standard_normal(n))
    c = vol * (-0.8 * f + idio * rng.standard_normal(n))
    return {'A': _curve_from_returns(a), 'B': _curve_from_returns(b),
            'C': _curve_from_returns(c)}


# ----------------------------------------------------------------------
# Construction: the new weighting mode
# ----------------------------------------------------------------------
class TestConstruction:
    def test_default_weighting_is_risk_parity(self):
        assert DynamicReweighter().weighting == 'risk_parity'

    def test_cov_mode_accepted(self):
        assert DynamicReweighter(weighting='risk_parity_cov').weighting \
            == 'risk_parity_cov'

    def test_invalid_weighting_rejected(self):
        with pytest.raises(ValueError, match='weighting must be one of'):
            DynamicReweighter(weighting='bogus')

    def test_cov_mode_registered_in_weighting_modes(self):
        # The dispatch tuple must list the new mode alongside the originals.
        assert 'risk_parity_cov' in DynamicReweighter._WEIGHTING_MODES
        assert DynamicReweighter._WEIGHTING_MODES == (
            'risk_parity', 'performance', 'risk_parity_cov',
            'max_diversification', 'mean_variance')


# ----------------------------------------------------------------------
# The DEFAULT path is byte-identical to risk parity (additive mode)
# ----------------------------------------------------------------------
class TestDefaultByteIdentical:
    def test_default_weights_equal_risk_parity_on_sliced_returns(self):
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        date = curve(LOW_VOL)[21][0]

        weights = rw.on_day([a, b], date, day_number=21)

        as_of = pd.Timestamp(date)
        expected = alloc.risk_parity_weights({
            'a': alloc.returns_from_equity(
                [v for ts, v in curve(LOW_VOL) if ts <= as_of]),
            'b': alloc.returns_from_equity(
                [v for ts, v in curve(HIGH_VOL) if ts <= as_of]),
        })
        assert weights == expected
        assert a.capital_allocation == weights['a']
        assert b.capital_allocation == weights['b']

    def test_default_log_entry_records_risk_parity_mode(self):
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        rw.on_day([a, b], curve(LOW_VOL)[21][0], day_number=21)
        entry = rw.rebalance_log[-1]
        assert entry['mode'] == 'risk_parity'
        assert entry['fallback'] is False
        assert entry['degraded_desks'] == []


# ----------------------------------------------------------------------
# Cov mode: full-covariance reallocation that de-correlates desks
# ----------------------------------------------------------------------
class TestCovMode:
    def test_delegates_to_cov_weights_on_sliced_returns(self):
        # The reweighter calls allocator.risk_parity_cov_weights on the ts<=date
        # slice (NOT risk_parity_weights), and the two genuinely differ here.
        alloc = CrossDeskCapitalAllocator()
        curves = correlated_curves()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                               weighting='risk_parity_cov')
        rw.set_curves(curves)
        desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
        date = curves['A'][60][0]

        weights = rw.on_day(desks, date, day_number=60)

        as_of = pd.Timestamp(date)
        sliced = {k: alloc.returns_from_equity(
            [v for ts, v in curves[k] if ts <= as_of]) for k in curves}
        assert weights == alloc.risk_parity_cov_weights(sliced)
        assert weights != alloc.risk_parity_weights(sliced)
        for desk in desks:
            assert desk.capital_allocation == weights[desk.key]

    def test_correlated_pair_down_weighted_vs_inverse_vol(self):
        curves = correlated_curves()
        date = curves['A'][60][0]

        def run(mode):
            alloc = CrossDeskCapitalAllocator()
            rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                                   weighting=mode)
            rw.set_curves(curves)
            desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
            return rw.on_day(desks, date, day_number=60)

        cov = run('risk_parity_cov')
        inv = run('risk_parity')
        # The correlated A/B pair carries less combined weight under full-cov.
        assert cov['A'] + cov['B'] < inv['A'] + inv['B']
        assert sum(cov.values()) == pytest.approx(1.0)

    def test_mode_recorded_on_rebalance(self):
        curves = correlated_curves()
        rw = DynamicReweighter(rebalance_every=60, weighting='risk_parity_cov')
        rw.set_curves(curves)
        rw.on_day([StubDesk('A'), StubDesk('B'), StubDesk('C')],
                  curves['A'][60][0], day_number=60)
        entry = rw.rebalance_log[-1]
        assert entry['mode'] == 'risk_parity_cov'
        assert entry['fallback'] is False
        assert entry['degrade_reason'] is None  # genuine full-cov result

    def test_numerical_degrade_recorded_as_honest_fallback(self):
        # A negatively-correlated hedge desk makes full-cov ill-defined, so it
        # degrades to inverse-vol. NO desk is degenerate, so previously this
        # logged fallback=False — now the audit records fallback=True WITH a
        # reason and an empty degraded_desks list.
        curves = hedge_curves()
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                               weighting='risk_parity_cov')
        rw.set_curves(curves)
        desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
        date = curves['A'][60][0]
        weights = rw.on_day(desks, date, day_number=60)

        as_of = pd.Timestamp(date)
        sliced = {k: alloc.returns_from_equity(
            [v for ts, v in curves[k] if ts <= as_of]) for k in curves}
        assert weights == alloc.risk_parity_weights(sliced)  # inverse-vol
        entry = rw.rebalance_log[-1]
        assert entry['fallback'] is True
        assert entry['degrade_reason'] is not None
        assert entry['degraded_desks'] == []   # not a degenerate-desk fallback
        assert entry['mode'] == 'risk_parity_cov'

    def test_cov_mode_shares_degenerate_fallback(self):
        # 'C' has no curve -> degenerate -> whole-fund equal weight, recorded as
        # a fallback even in cov mode (same gate as risk parity).
        curves = correlated_curves()
        rw = DynamicReweighter(rebalance_every=60, weighting='risk_parity_cov')
        rw.set_curves({'A': curves['A'], 'B': curves['B']})
        desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
        weights = rw.on_day(desks, curves['A'][60][0], day_number=60)
        assert weights['A'] == pytest.approx(1.0 / 3.0)
        assert weights['B'] == pytest.approx(1.0 / 3.0)
        assert weights['C'] == pytest.approx(1.0 / 3.0)
        entry = rw.rebalance_log[-1]
        assert entry['fallback'] is True
        assert entry['degraded_desks'] == ['C']
        assert entry['degrade_reason'] is None  # degenerate gate, not numerical
        assert entry['mode'] == 'risk_parity_cov'

    def test_target_gross_below_one_respected(self):
        curves = correlated_curves()
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=0.6),
            rebalance_every=60, weighting='risk_parity_cov')
        rw.set_curves(curves)
        desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
        weights = rw.on_day(desks, curves['A'][60][0], day_number=60)
        assert sum(weights.values()) == pytest.approx(0.6)
        assert sum(weights.values()) <= 0.6 + 1e-9


# ----------------------------------------------------------------------
# Walk-forward honesty in COV mode
# ----------------------------------------------------------------------
class TestWalkForwardCov:
    def test_appending_future_points_does_not_change_a_past_weight(self):
        curves = correlated_curves(n=80)
        as_of = curves['A'][60][0]

        def weight_at_boundary(cs):
            alloc = CrossDeskCapitalAllocator()
            rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                                   weighting='risk_parity_cov')
            rw.set_curves(cs)
            desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
            return rw.on_day(desks, as_of, day_number=60)

        short = weight_at_boundary(curves)
        # Truncate each curve to the boundary, then append wild future points.
        extended = {}
        for k, points in curves.items():
            prefix = [(ts, v) for ts, v in points if ts <= pd.Timestamp(as_of)]
            future = curve([9.0 + (40.0 if i % 2 else 0.0) for i in range(20)],
                           start='2099-01-01')
            extended[k] = prefix + future
        ext = weight_at_boundary(extended)
        assert short == ext


# ----------------------------------------------------------------------
# Over-leverage guard (cov mode)
# ----------------------------------------------------------------------
class TestOverLeverageGuard:
    @pytest.mark.parametrize('target', [1.0, 0.6])
    def test_cov_weights_never_exceed_target_gross(self, target):
        curves = correlated_curves()
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=target),
            rebalance_every=20, weighting='risk_parity_cov')
        rw.set_curves(curves)
        desks = [StubDesk('A'), StubDesk('B'), StubDesk('C')]
        dates = [ts for ts, _ in curves['A']]
        for day in range(1, 61):
            w = rw.on_day(desks, dates[min(day - 1, len(dates) - 1)], day)
            if w is not None:
                assert sum(w.values()) <= target + 1e-9
