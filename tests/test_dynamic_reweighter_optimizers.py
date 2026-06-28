"""Tests for the OPT-IN portfolio-optimizer weighting modes on DynamicReweighter:
'max_diversification' and 'mean_variance'.

DynamicReweighter's ``weighting`` param gains two covariance-aware modes
alongside 'risk_parity_cov'. The DEFAULT 'risk_parity' stays byte-identical to
the inverse-vol path (proved in test_dynamic_reweighter.py); the new modes
dispatch to the allocator's optimizer methods on the SAME walk-forward curve
slice, and record an HONEST fallback in the rebalance log when they degrade.

Pure and offline — no engine, no market data. Mirrors
tests/test_dynamic_reweighter_cov.py.
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
    independent. Max diversification down-weights the redundant A/B pair."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    g = rng.standard_normal(n)
    a = vol * (f + idio * rng.standard_normal(n))
    b = vol * (f + idio * rng.standard_normal(n))
    c = vol * (g + idio * rng.standard_normal(n))
    return {'A': _curve_from_returns(a), 'B': _curve_from_returns(b),
            'C': _curve_from_returns(c)}


def sharpe_curves(seed: int = 11, n: int = 80, vol: float = 0.01):
    """Equity curves for three near-uncorrelated, equal-vol desks with DISTINCT
    means (Sharpe lo < mid < hi). Mean-variance tilts toward 'hi'."""
    rng = np.random.default_rng(seed)
    means = {'lo': 0.0002, 'mid': 0.0006, 'hi': 0.0012}
    return {k: _curve_from_returns(m + vol * rng.standard_normal(n))
            for k, m in means.items()}


def singular_curves(seed: int = 3, n: int = 80, vol: float = 0.01):
    """A and B are IDENTICAL (singular covariance), C distinct. NO desk is
    degenerate, so the optimizer modes degrade to inverse-vol via a NUMERICAL
    guard — an honest fallback with a reason and no degraded desks."""
    rng = np.random.default_rng(seed)
    shared = vol * rng.standard_normal(n)
    c = vol * rng.standard_normal(n)
    return {'A': _curve_from_returns(shared), 'B': _curve_from_returns(shared),
            'C': _curve_from_returns(c)}


#: mode -> (fixture, allocator method name) for the data-driven tests.
MODES = {
    'max_diversification': (correlated_curves, 'max_diversification_weights'),
    'mean_variance': (sharpe_curves, 'mean_variance_weights'),
}
PARAMS = list(MODES.items())


# ----------------------------------------------------------------------
# Construction: the new weighting modes
# ----------------------------------------------------------------------
class TestConstruction:
    @pytest.mark.parametrize('mode', list(MODES))
    def test_mode_accepted(self, mode):
        assert DynamicReweighter(weighting=mode).weighting == mode

    def test_invalid_weighting_rejected(self):
        with pytest.raises(ValueError, match='weighting must be one of'):
            DynamicReweighter(weighting='bogus')

    def test_new_modes_registered(self):
        assert DynamicReweighter._WEIGHTING_MODES == (
            'risk_parity', 'performance', 'risk_parity_cov',
            'max_diversification', 'mean_variance')


# ----------------------------------------------------------------------
# Each optimizer mode dispatches to the matching allocator method
# ----------------------------------------------------------------------
class TestDispatch:
    @pytest.mark.parametrize('mode,spec', PARAMS)
    def test_delegates_to_optimizer_on_sliced_returns(self, mode, spec):
        fixture, method = spec
        alloc = CrossDeskCapitalAllocator()
        curves = fixture()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                               weighting=mode)
        rw.set_curves(curves)
        desks = [StubDesk(k) for k in curves]
        date = curves[next(iter(curves))][60][0]

        weights = rw.on_day(desks, date, day_number=60)

        as_of = pd.Timestamp(date)
        sliced = {k: alloc.returns_from_equity(
            [v for ts, v in curves[k] if ts <= as_of]) for k in curves}
        assert weights == getattr(alloc, method)(sliced)
        # The optimizer genuinely differs from the inverse-vol default here.
        assert weights != alloc.risk_parity_weights(sliced)
        for desk in desks:
            assert desk.capital_allocation == weights[desk.key]

    @pytest.mark.parametrize('mode,spec', PARAMS)
    def test_mode_recorded_on_genuine_result(self, mode, spec):
        fixture, _ = spec
        curves = fixture()
        rw = DynamicReweighter(rebalance_every=60, weighting=mode)
        rw.set_curves(curves)
        rw.on_day([StubDesk(k) for k in curves],
                  curves[next(iter(curves))][60][0], day_number=60)
        entry = rw.rebalance_log[-1]
        assert entry['mode'] == mode
        assert entry['fallback'] is False
        assert entry['degrade_reason'] is None  # genuine optimizer result

    @pytest.mark.parametrize('mode,spec', PARAMS)
    def test_target_gross_below_one_respected(self, mode, spec):
        fixture, _ = spec
        curves = fixture()
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=0.6),
            rebalance_every=60, weighting=mode)
        rw.set_curves(curves)
        weights = rw.on_day([StubDesk(k) for k in curves],
                            curves[next(iter(curves))][60][0], day_number=60)
        assert sum(weights.values()) == pytest.approx(0.6)
        assert sum(weights.values()) <= 0.6 + 1e-9


# ----------------------------------------------------------------------
# Mode-specific headline behavior
# ----------------------------------------------------------------------
class TestModeBehavior:
    def test_max_div_down_weights_correlated_pair(self):
        curves = correlated_curves()
        date = curves['A'][60][0]

        def run(mode):
            alloc = CrossDeskCapitalAllocator()
            rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                                   weighting=mode)
            rw.set_curves(curves)
            return rw.on_day([StubDesk('A'), StubDesk('B'), StubDesk('C')],
                             date, day_number=60)

        md = run('max_diversification')
        inv = run('risk_parity')
        assert md['A'] + md['B'] < inv['A'] + inv['B']
        assert sum(md.values()) == pytest.approx(1.0)

    def test_mean_variance_favors_best_sharpe(self):
        curves = sharpe_curves()
        date = curves['lo'][60][0]
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                               weighting='mean_variance')
        rw.set_curves(curves)
        mv = rw.on_day([StubDesk(k) for k in curves], date, day_number=60)
        assert mv['hi'] == max(mv.values())
        assert sum(mv.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Honest fallback auditing (mirrors the cov mode)
# ----------------------------------------------------------------------
class TestHonestFallback:
    @pytest.mark.parametrize('mode,spec', PARAMS)
    def test_numerical_degrade_recorded_as_honest_fallback(self, mode, spec):
        # Identical A/B curves -> singular covariance -> degrade to inverse-vol.
        # NO desk is degenerate, so the audit must record fallback=True WITH a
        # reason and an EMPTY degraded_desks list (not a degenerate-desk fall).
        _, _ = spec
        curves = singular_curves()
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                               weighting=mode)
        rw.set_curves(curves)
        desks = [StubDesk(k) for k in curves]
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
        assert entry['mode'] == mode

    @pytest.mark.parametrize('mode,spec', PARAMS)
    def test_shares_degenerate_fallback(self, mode, spec):
        # 'C' has no curve -> degenerate -> whole-fund equal weight, recorded as
        # a fallback even in optimizer mode (same gate as risk parity).
        fixture, _ = spec
        curves = fixture()
        keys = list(curves)
        rw = DynamicReweighter(rebalance_every=60, weighting=mode)
        rw.set_curves({keys[0]: curves[keys[0]], keys[1]: curves[keys[1]]})
        desks = [StubDesk(k) for k in keys]
        weights = rw.on_day(desks, curves[keys[0]][60][0], day_number=60)
        for k in keys:
            assert weights[k] == pytest.approx(1.0 / 3.0)
        entry = rw.rebalance_log[-1]
        assert entry['fallback'] is True
        assert entry['degraded_desks'] == [keys[2]]
        assert entry['degrade_reason'] is None  # degenerate gate, not numerical
        assert entry['mode'] == mode


# ----------------------------------------------------------------------
# Walk-forward honesty
# ----------------------------------------------------------------------
class TestWalkForward:
    @pytest.mark.parametrize('mode,spec', PARAMS)
    def test_appending_future_points_does_not_change_a_past_weight(
            self, mode, spec):
        fixture, _ = spec
        curves = fixture(n=80)
        keys = list(curves)
        as_of = curves[keys[0]][60][0]

        def weight_at_boundary(cs):
            alloc = CrossDeskCapitalAllocator()
            rw = DynamicReweighter(allocator=alloc, rebalance_every=60,
                                   weighting=mode)
            rw.set_curves(cs)
            return rw.on_day([StubDesk(k) for k in keys], as_of, day_number=60)

        short = weight_at_boundary(curves)
        extended = {}
        for k, points in curves.items():
            prefix = [(ts, v) for ts, v in points if ts <= pd.Timestamp(as_of)]
            future = curve([9.0 + (40.0 if i % 2 else 0.0) for i in range(20)],
                           start='2099-01-01')
            extended[k] = prefix + future
        assert short == weight_at_boundary(extended)


# ----------------------------------------------------------------------
# Over-leverage guard
# ----------------------------------------------------------------------
class TestOverLeverageGuard:
    @pytest.mark.parametrize('mode,spec', PARAMS)
    @pytest.mark.parametrize('target', [1.0, 0.6])
    def test_weights_never_exceed_target_gross(self, mode, spec, target):
        fixture, _ = spec
        curves = fixture()
        keys = list(curves)
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=target),
            rebalance_every=20, weighting=mode)
        rw.set_curves(curves)
        desks = [StubDesk(k) for k in keys]
        dates = [ts for ts, _ in curves[keys[0]]]
        for day in range(1, 61):
            w = rw.on_day(desks, dates[min(day - 1, len(dates) - 1)], day)
            if w is not None:
                assert sum(w.values()) <= target + 1e-9
