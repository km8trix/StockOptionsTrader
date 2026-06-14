"""Tests for desks.dynamic_reweighter.DynamicReweighter.

Covers construction validation, the rebalance schedule (warmup + cadence),
walk-forward-honest curve slicing (no future point ever reaches a weight),
risk-parity application onto live desks (lower vol -> higher weight), the
equal-weight fallback when a desk has no/degenerate curve, and the rebalance
log. Pure and offline — no engine, no market data.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

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


# Low-vol vs high-vol equity paths (both with finite, positive, DISTINCT vol).
LOW_VOL = [100.0 + (0.5 if i % 2 else 0.0) for i in range(30)]
HIGH_VOL = [100.0 + (5.0 if i % 2 else 0.0) for i in range(30)]


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
class TestConstruction:
    def test_defaults(self):
        rw = DynamicReweighter()
        assert rw.rebalance_every == 21
        assert rw.warmup == 0
        assert isinstance(rw.allocator, CrossDeskCapitalAllocator)
        assert rw.rebalance_log == []

    def test_rejects_non_positive_cadence(self):
        with pytest.raises(ValueError, match='rebalance_every must be >= 1'):
            DynamicReweighter(rebalance_every=0)

    def test_rejects_negative_warmup(self):
        with pytest.raises(ValueError, match='warmup must be >= 0'):
            DynamicReweighter(warmup=-1)


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------
class TestSchedule:
    def test_monthly_cadence_no_warmup(self):
        rw = DynamicReweighter(rebalance_every=21, warmup=0)
        assert not any(rw.is_rebalance_day(d) for d in range(1, 21))
        assert rw.is_rebalance_day(21)
        assert not any(rw.is_rebalance_day(d) for d in range(22, 42))
        assert rw.is_rebalance_day(42)

    def test_warmup_delays_first_rebalance(self):
        rw = DynamicReweighter(rebalance_every=5, warmup=10)
        assert not any(rw.is_rebalance_day(d) for d in range(1, 11))
        assert rw.is_rebalance_day(15)
        assert rw.is_rebalance_day(20)
        assert not rw.is_rebalance_day(16)


# ----------------------------------------------------------------------
# on_day: application + no-op
# ----------------------------------------------------------------------
class TestOnDay:
    def test_non_boundary_is_a_noop(self):
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        result = rw.on_day([a, b], pd.Timestamp('2022-01-20'), day_number=10)
        assert result is None
        assert a.capital_allocation == 0.5 and b.capital_allocation == 0.5
        assert rw.rebalance_log == []

    def test_boundary_sets_capital_and_logs(self):
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        date = curve(LOW_VOL)[21][0]
        weights = rw.on_day([a, b], date, day_number=21)

        assert set(weights) == {'a', 'b'}
        # Lower-vol desk gets MORE capital under risk parity.
        assert weights['a'] > weights['b']
        assert a.capital_allocation == weights['a']
        assert b.capital_allocation == weights['b']
        # Weights sum to the allocator target_gross (default 1.0) <= 1.0.
        assert sum(weights.values()) == pytest.approx(1.0)
        assert len(rw.rebalance_log) == 1
        entry = rw.rebalance_log[0]
        assert entry['day_number'] == 21
        assert entry['date'] == pd.Timestamp(date)
        assert entry['weights'] == weights
        # Genuine risk-parity result -> not a fallback.
        assert entry['fallback'] is False
        assert entry['degraded_desks'] == []

    def test_target_gross_below_one_is_respected(self):
        rw = DynamicReweighter(
            allocator=CrossDeskCapitalAllocator(target_gross=0.6),
            rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], curve(LOW_VOL)[21][0], day_number=21)
        assert sum(weights.values()) == pytest.approx(0.6)


# ----------------------------------------------------------------------
# Walk-forward honesty
# ----------------------------------------------------------------------
class TestWalkForward:
    def test_only_past_points_drive_the_weight(self):
        # First 15 points: a is calmer than b. Last 15 points: REVERSED, wildly.
        # A weight computed at day 15 must reflect ONLY the first 15 points.
        a_vals = [100.0 + (0.5 if i % 2 else 0.0) for i in range(15)] + \
                 [100.0 + (20.0 if i % 2 else 0.0) for i in range(15)]
        b_vals = [100.0 + (4.0 if i % 2 else 0.0) for i in range(15)] + \
                 [100.0 + (0.1 if i % 2 else 0.0) for i in range(15)]
        alloc = CrossDeskCapitalAllocator()
        rw = DynamicReweighter(allocator=alloc, rebalance_every=15)
        rw.set_curves({'a': curve(a_vals), 'b': curve(b_vals)})
        a, b = StubDesk('a'), StubDesk('b')

        as_of = curve(a_vals)[14][0]  # the 15th point
        weights = rw.on_day([a, b], as_of, day_number=15)

        # Equivalent to running the allocator on ONLY the first-15 slices.
        expected = alloc.risk_parity_weights({
            'a': alloc.returns_from_equity([v for _, v in curve(a_vals)[:15]]),
            'b': alloc.returns_from_equity([v for _, v in curve(b_vals)[:15]]),
        })
        assert weights == expected
        # And NOT equal to using the whole (future-contaminated) curve.
        full = alloc.risk_parity_weights({
            'a': alloc.returns_from_equity(a_vals),
            'b': alloc.returns_from_equity(b_vals),
        })
        assert weights != full


# ----------------------------------------------------------------------
# Fallback behaviour
# ----------------------------------------------------------------------
class TestFallback:
    def test_missing_curve_falls_back_to_equal_weight(self):
        # 'b' has no curve at all -> degenerate -> whole-fund equal weight.
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], curve(LOW_VOL)[21][0], day_number=21)
        assert weights['a'] == pytest.approx(0.5)
        assert weights['b'] == pytest.approx(0.5)
        assert a.capital_allocation == pytest.approx(0.5)
        assert b.capital_allocation == pytest.approx(0.5)
        # The fallback is recorded (not mistaken for a real risk-parity 0.5/0.5).
        entry = rw.rebalance_log[-1]
        assert entry['fallback'] is True
        assert entry['degraded_desks'] == ['b']  # b had no curve

    def test_set_curves_replaces_previous(self):
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        rw.set_curves({'a': curve(HIGH_VOL), 'b': curve(LOW_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        weights = rw.on_day([a, b], curve(LOW_VOL)[21][0], day_number=21)
        # Now 'b' is the calm one -> 'b' gets more.
        assert weights['b'] > weights['a']

    def test_set_curves_accepts_unsorted_pairs(self):
        rw = DynamicReweighter(rebalance_every=21)
        pairs = curve(LOW_VOL)
        rw.set_curves({'a': list(reversed(pairs)), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        # Reversed input must still slice/return correctly (stored sorted).
        weights = rw.on_day([a, b], pairs[21][0], day_number=21)
        assert weights['a'] > weights['b']


# ----------------------------------------------------------------------
# Degeneracy helper (single source of the fallback rule)
# ----------------------------------------------------------------------
class TestDegenerateHelper:
    def test_flags_short_flat_and_clean_series(self):
        alloc = CrossDeskCapitalAllocator()
        good = alloc.returns_from_equity([v for _, v in curve(HIGH_VOL)])
        deg = alloc.degenerate_desks({
            'short': [0.01],          # < 2 returns
            'flat': [0.0, 0.0, 0.0],  # zero vol
            'good': good,             # usable
        })
        assert set(deg) == {'short', 'flat'}
        assert 'good' not in deg
        assert 'short' in deg['short'] or '< 2' in deg['short'] \
            or 'finite returns' in deg['short']


# ----------------------------------------------------------------------
# Multiple rebalances
# ----------------------------------------------------------------------
class TestMultipleRebalances:
    def test_log_accumulates_per_boundary(self):
        rw = DynamicReweighter(rebalance_every=10)
        rw.set_curves({'a': curve(LOW_VOL), 'b': curve(HIGH_VOL)})
        a, b = StubDesk('a'), StubDesk('b')
        dates = [ts for ts, _ in curve(LOW_VOL)]
        for day in range(1, 26):
            rw.on_day([a, b], dates[min(day - 1, len(dates) - 1)], day)
        assert [e['day_number'] for e in rw.rebalance_log] == [10, 20]
