"""Tests for the OPT-IN performance weighting mode on DynamicReweighter.

DynamicReweighter gains a ``weighting`` param: the DEFAULT 'risk_parity' must be
byte-identical to the pre-existing inverse-vol path (the headline test below
proves the default reweighter's weights AND every asserted log field match
risk_parity_weights on the sliced returns), and 'performance' opts into the
guarded risk-adjusted tilt. Both modes share the SAME degenerate_desks audit and
the SAME walk-forward curve slicing.

Pure and offline — no engine, no market data. Mirrors
tests/test_dynamic_reweighter.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

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


# Low-vol vs high-vol equity paths (distinct, finite, positive vol).
LOW_VOL = [100.0 + (0.5 if i % 2 else 0.0) for i in range(30)]
HIGH_VOL = [100.0 + (5.0 if i % 2 else 0.0) for i in range(30)]

# A high-Sharpe equity path (steady compounding positive drift + tiny
# alternating jitter for finite-but-low vol) vs a choppy near-zero-Sharpe path
# (~0 mean, larger vol). The high-Sharpe desk's realized Sharpe dominates at
# every slice boundary, so the performance tilt is unambiguous.
def _high_sharpe(n: int) -> List[float]:
    out: List[float] = []
    level = 100.0
    for i in range(n):
        level *= (1.0 + 0.003 + (0.0003 if i % 2 else -0.0003))
        out.append(level)
    return out


HIGH_SHARPE = _high_sharpe(30)
LOW_SHARPE = [100.0 + (3.0 if i % 2 else 0.0) for i in range(30)]


# ----------------------------------------------------------------------
# Construction: the new weighting param
# ----------------------------------------------------------------------
class TestConstruction:
    def test_default_weighting_is_risk_parity(self):
        rw = DynamicReweighter()
        assert rw.weighting == 'risk_parity'

    def test_performance_mode_accepted(self):
        rw = DynamicReweighter(weighting='performance')
        assert rw.weighting == 'performance'

    def test_invalid_weighting_rejected(self):
        with pytest.raises(ValueError, match='weighting must be one of'):
            DynamicReweighter(weighting='bogus')


# ----------------------------------------------------------------------
# HEADLINE: the DEFAULT path is byte-identical to risk parity
# ----------------------------------------------------------------------
class TestDefaultByteIdentical:
    def test_default_weights_equal_risk_parity_on_sliced_returns(self):
        # A default DynamicReweighter rebalances to EXACTLY the weights the
        # allocator's risk_parity_weights produces on the ts<=date slices —
        # the additive 'performance' mode does not perturb the default.
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
        # Every pre-existing log field is unchanged; 'mode' is additive.
        assert entry['mode'] == 'risk_parity'
        assert entry['fallback'] is False
        assert entry['degraded_desks'] == []


# ----------------------------------------------------------------------
# Performance mode: realized risk-adjusted reallocation
# ----------------------------------------------------------------------
class TestPerformanceMode:
    def test_higher_sharpe_desk_gets_more_and_mode_logged(self):
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=21,
                               weighting='performance')
        rw.set_curves({'a': curve(HIGH_SHARPE), 'b': curve(LOW_SHARPE)})
        a, b = StubDesk('a'), StubDesk('b')
        date = curve(HIGH_SHARPE)[21][0]

        weights = rw.on_day([a, b], date, day_number=21)

        assert set(weights) == {'a', 'b'}
        # Higher-Sharpe desk gets more, bounded by the default band.
        assert weights['a'] > weights['b']
        assert 0.05 - 1e-9 <= weights['b'] and weights['a'] <= 0.60 + 1e-9
        assert a.capital_allocation == weights['a']
        assert b.capital_allocation == weights['b']
        assert sum(weights.values()) == pytest.approx(1.0)
        entry = rw.rebalance_log[-1]
        assert entry['mode'] == 'performance'
        assert entry['fallback'] is False

    def test_performance_weights_match_allocator_on_sliced_returns(self):
        # The reweighter delegates to allocator.performance_weights on the
        # ts<=date slice (not risk_parity_weights).
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=21,
                               weighting='performance')
        rw.set_curves({'a': curve(HIGH_SHARPE), 'b': curve(LOW_SHARPE)})
        a, b = StubDesk('a'), StubDesk('b')
        date = curve(HIGH_SHARPE)[21][0]

        weights = rw.on_day([a, b], date, day_number=21)

        as_of = pd.Timestamp(date)
        expected = alloc.performance_weights({
            'a': alloc.returns_from_equity(
                [v for ts, v in curve(HIGH_SHARPE) if ts <= as_of]),
            'b': alloc.returns_from_equity(
                [v for ts, v in curve(LOW_SHARPE) if ts <= as_of]),
        })
        assert weights == expected
        # And NOT the risk-parity weights (the two modes genuinely differ here).
        rp = alloc.risk_parity_weights({
            'a': alloc.returns_from_equity(
                [v for ts, v in curve(HIGH_SHARPE) if ts <= as_of]),
            'b': alloc.returns_from_equity(
                [v for ts, v in curve(LOW_SHARPE) if ts <= as_of]),
        })
        assert weights != rp

    def test_performance_mode_shares_degenerate_fallback(self):
        # 'b' has no curve -> degenerate -> whole-fund equal weight, recorded
        # as a fallback even in performance mode (same gate as risk parity).
        rw = DynamicReweighter(rebalance_every=21, weighting='performance')
        rw.set_curves({'a': curve(HIGH_SHARPE)})
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], curve(HIGH_SHARPE)[21][0], day_number=21)
        assert weights['a'] == pytest.approx(0.5)
        assert weights['b'] == pytest.approx(0.5)
        entry = rw.rebalance_log[-1]
        assert entry['fallback'] is True
        assert entry['degraded_desks'] == ['b']
        assert entry['mode'] == 'performance'

    def test_performance_target_gross_below_one_respected(self):
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=0.6),
            rebalance_every=21, weighting='performance')
        rw.set_curves({'a': curve(HIGH_SHARPE), 'b': curve(LOW_SHARPE)})
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], curve(HIGH_SHARPE)[21][0], day_number=21)
        assert sum(weights.values()) == pytest.approx(0.6)
        assert sum(weights.values()) <= 0.6 + 1e-9


# ----------------------------------------------------------------------
# Walk-forward honesty in PERFORMANCE mode
# ----------------------------------------------------------------------
class TestWalkForwardPerformance:
    def test_only_past_points_drive_the_performance_weight(self):
        # First 15 points: a is the steady winner. Last 15: a collapses and b
        # surges. A performance weight at day 15 must reflect ONLY the first 15.
        a_vals = [100.0 * (1.0 + 0.004) ** i for i in range(15)] + \
                 [100.0 - (3.0 if i % 2 else 0.0) for i in range(15)]
        b_vals = [100.0 + (3.0 if i % 2 else 0.0) for i in range(15)] + \
                 [100.0 * (1.0 + 0.006) ** i for i in range(15)]
        # Inject finite vol into a's first-15 winning segment so it is not
        # degenerate (a smooth geometric series has zero vol).
        a_vals = [v + (0.2 if i % 2 else 0.0) for i, v in enumerate(a_vals)]
        b_vals = [v + (0.2 if i % 2 else 0.0) for i, v in enumerate(b_vals)]

        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=15,
                               weighting='performance')
        rw.set_curves({'a': curve(a_vals), 'b': curve(b_vals)})
        a, b = StubDesk('a'), StubDesk('b')

        as_of = curve(a_vals)[14][0]  # the 15th point
        weights = rw.on_day([a, b], as_of, day_number=15)

        # Equivalent to running performance_weights on ONLY the first-15 slices.
        expected = alloc.performance_weights({
            'a': alloc.returns_from_equity(
                [v for _, v in curve(a_vals)[:15]]),
            'b': alloc.returns_from_equity(
                [v for _, v in curve(b_vals)[:15]]),
        })
        assert weights == expected
        # And NOT equal to using the whole (future-contaminated) curve.
        full = alloc.performance_weights({
            'a': alloc.returns_from_equity(a_vals),
            'b': alloc.returns_from_equity(b_vals),
        })
        assert weights != full

    def test_appending_future_points_does_not_change_a_past_weight(self):
        # Compute a weight at day 15; then append future points and recompute
        # at the same boundary date — the weight is unchanged.
        a_vals = [100.0 + (0.5 if i % 2 else 0.0) + 0.05 * i for i in range(15)]
        b_vals = [100.0 + (3.0 if i % 2 else 0.0) for i in range(15)]
        as_of = curve(a_vals)[14][0]

        def weight_at_boundary(a_curve, b_curve):
            alloc = CrossDeskCapitalAllocator()
            rw = DynamicReweighter(allocator=alloc, rebalance_every=15,
                                   weighting='performance')
            rw.set_curves({'a': a_curve, 'b': b_curve})
            a, b = StubDesk('a'), StubDesk('b')
            return rw.on_day([a, b], as_of, day_number=15)

        short = weight_at_boundary(curve(a_vals), curve(b_vals))
        # Same prefix, but with wild future points appended after the boundary.
        a_ext = a_vals + [50.0 + (40.0 if i % 2 else 0.0) for i in range(15)]
        b_ext = b_vals + [200.0 * (1.0 + 0.02) ** i for i in range(15)]
        extended = weight_at_boundary(curve(a_ext), curve(b_ext))
        assert short == extended


# ----------------------------------------------------------------------
# Over-leverage guard (performance mode)
# ----------------------------------------------------------------------
class TestOverLeverageGuard:
    @pytest.mark.parametrize('target', [1.0, 0.6])
    def test_performance_weights_never_exceed_target_gross(self, target):
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=target),
            rebalance_every=10, weighting='performance')
        rw.set_curves({'a': curve(HIGH_SHARPE), 'b': curve(LOW_SHARPE)})
        a, b = StubDesk('a'), StubDesk('b')
        dates = [ts for ts, _ in curve(HIGH_SHARPE)]
        for day in range(1, 26):
            w = rw.on_day([a, b], dates[min(day - 1, len(dates) - 1)], day)
            if w is not None:
                assert sum(w.values()) <= target + 1e-9


# ----------------------------------------------------------------------
# Multi-rebalance run records the mode every time
# ----------------------------------------------------------------------
class TestMultipleRebalances:
    def test_performance_mode_logged_on_every_boundary(self):
        rw = DynamicReweighter(rebalance_every=10, weighting='performance')
        rw.set_curves({'a': curve(HIGH_SHARPE), 'b': curve(LOW_SHARPE)})
        a, b = StubDesk('a'), StubDesk('b')
        dates = [ts for ts, _ in curve(HIGH_SHARPE)]
        for day in range(1, 26):
            rw.on_day([a, b], dates[min(day - 1, len(dates) - 1)], day)
        assert [e['day_number'] for e in rw.rebalance_log] == [10, 20]
        assert all(e['mode'] == 'performance' for e in rw.rebalance_log)
        # The realized higher-Sharpe desk keeps getting more across boundaries.
        assert all(e['weights']['a'] > e['weights']['b']
                   for e in rw.rebalance_log)


# ----------------------------------------------------------------------
# REGRESSION: the on_day mutation-site guard renormalizes an over-target set
# DOWN to target_gross BEFORE writing it to desk.capital_allocation. Both
# built-in modes already sum to target_gross, so to exercise the guard we feed
# a CONTRIVED allocator whose weighting method deliberately returns an
# over-target set — exactly the allocator-contract-breach the guard defends.
# ----------------------------------------------------------------------
class _FakeAllocator:
    """A stand-in with the surface DynamicReweighter.on_day touches, whose
    performance_weights/risk_parity_weights return a FIXED over-target set."""

    def __init__(self, target_gross: float, over_set: Dict[str, float]):
        self.target_gross = float(target_gross)
        self._over_set = dict(over_set)

    def returns_from_equity(self, values):  # non-degenerate placeholder
        return [0.01, 0.02, 0.01]

    def degenerate_desks(self, returns_by_desk):  # never degenerate
        return {}

    def performance_weights(self, returns_by_desk):
        return dict(self._over_set)

    def risk_parity_weights(self, returns_by_desk):
        return dict(self._over_set)


class TestOnDayGuardRenormalizesDown:
    def test_over_one_set_is_scaled_down_to_target_before_write(self):
        # A breaching set summing to 1.7 with target_gross=1.0 must be scaled
        # back to sum EXACTLY 1.0, and that scaled-down value is what lands on
        # each desk.capital_allocation (never the over-leveraged value).
        fake = _FakeAllocator(target_gross=1.0, over_set={'a': 0.9, 'b': 0.8})
        rw = DynamicReweighter(allocator=fake, rebalance_every=1,
                               weighting='performance')
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], pd.Timestamp('2022-02-01'), day_number=1)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert sum(weights.values()) <= 1.0 + 1e-9
        # Proportions preserved by the down-scale (0.9/1.7, 0.8/1.7).
        assert weights['a'] == pytest.approx(0.9 / 1.7)
        assert weights['b'] == pytest.approx(0.8 / 1.7)
        # The mutated desks carry the renormalized (safe) weights, not 0.9/0.8.
        assert a.capital_allocation == pytest.approx(weights['a'])
        assert b.capital_allocation == pytest.approx(weights['b'])
        assert a.capital_allocation < 0.9

    def test_breach_of_sub_one_target_still_caught_even_below_absolute_one(
            self):
        # target_gross=0.6: a set summing to 0.9 is BELOW absolute 1.0 but
        # breaches the sub-1.0 target — the guard compares against the
        # allocator's OWN target_gross (not literal 1.0) and scales down to 0.6.
        fake = _FakeAllocator(target_gross=0.6, over_set={'a': 0.5, 'b': 0.4})
        rw = DynamicReweighter(allocator=fake, rebalance_every=1,
                               weighting='performance')
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], pd.Timestamp('2022-02-01'), day_number=1)
        assert sum(weights.values()) == pytest.approx(0.6)
        assert sum(weights.values()) <= 0.6 + 1e-9
        assert weights['a'] == pytest.approx(0.6 * 0.5 / 0.9)
        assert weights['b'] == pytest.approx(0.6 * 0.4 / 0.9)
        assert a.capital_allocation == pytest.approx(weights['a'])
        assert b.capital_allocation == pytest.approx(weights['b'])

    def test_guard_logs_an_error_on_breach(self, caplog):
        fake = _FakeAllocator(target_gross=1.0, over_set={'a': 0.9, 'b': 0.8})
        rw = DynamicReweighter(allocator=fake, rebalance_every=1,
                               weighting='performance')
        a, b = StubDesk('a'), StubDesk('b')
        with caplog.at_level(logging.ERROR, logger='desks.dynamic_reweighter'):
            rw.on_day([a, b], pd.Timestamp('2022-02-01'), day_number=1)
        assert any('target_gross' in rec.message and rec.levelno == logging.ERROR
                   for rec in caplog.records)
