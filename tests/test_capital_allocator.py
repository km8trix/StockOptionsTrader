"""Tests for desks.capital_allocator.CrossDeskCapitalAllocator.

Covers construction validation, risk-parity (inverse-vol) weighting and its
properties (lower vol -> more weight, equal vol -> equal weight, sum ==
target_gross), the equal-weight fallback on degenerate/short/non-finite vol,
allocate() mutating Desk.capital_allocation into a fund-valid set, and the
equity->returns helper. Offline, deterministic, no engine.
"""

from __future__ import annotations

import pytest

from desks.capital_allocator import CrossDeskCapitalAllocator
from desks.orchestrator import FundOrchestrator
from tests.test_fund_orchestrator import ScriptedDesk

LOW_VOL = [0.001, -0.001, 0.001, -0.001]
HIGH_VOL = [0.05, -0.05, 0.05, -0.05]


class TestConstruction:
    @pytest.mark.parametrize('bad', [0.0, -0.1, 1.5])
    def test_invalid_target_rejected(self, bad):
        with pytest.raises(ValueError, match='target_gross'):
            CrossDeskCapitalAllocator(target_gross=bad)


class TestRiskParityWeights:
    def test_lower_vol_gets_more_weight(self):
        w = CrossDeskCapitalAllocator().risk_parity_weights(
            {'lo': LOW_VOL, 'hi': HIGH_VOL})
        assert w['lo'] > w['hi']
        assert sum(w.values()) == pytest.approx(1.0)

    def test_equal_vol_equal_weight(self):
        s = [0.01, -0.01, 0.02, -0.02]
        w = CrossDeskCapitalAllocator().risk_parity_weights(
            {'x': s, 'y': list(s)})
        assert w['x'] == pytest.approx(w['y'])
        assert sum(w.values()) == pytest.approx(1.0)

    def test_target_gross_scales_the_sum(self):
        w = CrossDeskCapitalAllocator(target_gross=0.8).risk_parity_weights(
            {'x': [0.01, -0.01, 0.02], 'y': [0.02, -0.02, 0.01]})
        assert sum(w.values()) == pytest.approx(0.8)

    def test_zero_vol_desk_triggers_equal_weight_fallback(self):
        w = CrossDeskCapitalAllocator().risk_parity_weights(
            {'flat': [0.0, 0.0, 0.0], 'vary': [0.01, -0.02, 0.03]})
        assert w['flat'] == pytest.approx(w['vary'])  # equal weight
        assert sum(w.values()) == pytest.approx(1.0)

    def test_too_few_returns_triggers_fallback(self):
        w = CrossDeskCapitalAllocator().risk_parity_weights(
            {'x': [0.01], 'y': [0.01, -0.01, 0.02]})
        assert w['x'] == pytest.approx(w['y'])

    def test_nonfinite_leaving_too_few_triggers_fallback(self):
        w = CrossDeskCapitalAllocator().risk_parity_weights(
            {'x': [float('nan'), 0.01], 'y': [0.01, -0.01, 0.02]})
        assert w['x'] == pytest.approx(w['y'])

    def test_empty_is_empty(self):
        assert CrossDeskCapitalAllocator().risk_parity_weights({}) == {}


class TestAllocate:
    def test_sets_capital_allocation_and_sums_to_one(self):
        allocator = CrossDeskCapitalAllocator()
        lo, hi = ScriptedDesk('lo', 1.0), ScriptedDesk('hi', 1.0)
        weights = allocator.allocate([lo, hi], {'lo': LOW_VOL, 'hi': HIGH_VOL})
        assert lo.capital_allocation == pytest.approx(weights['lo'])
        assert hi.capital_allocation == pytest.approx(weights['hi'])
        assert lo.capital_allocation > hi.capital_allocation  # lower vol
        assert (lo.capital_allocation
                + hi.capital_allocation) == pytest.approx(1.0)

    def test_missing_returns_falls_back_to_equal(self):
        allocator = CrossDeskCapitalAllocator()
        x, y = ScriptedDesk('x', 1.0), ScriptedDesk('y', 1.0)
        weights = allocator.allocate([x, y], {'x': [0.01, -0.01, 0.02]})
        assert weights['x'] == pytest.approx(weights['y'])  # y missing

    def test_allocated_desks_build_a_valid_fund(self):
        allocator = CrossDeskCapitalAllocator()
        desks = [ScriptedDesk('lo', 1.0), ScriptedDesk('hi', 1.0)]
        allocator.allocate(desks, {'lo': LOW_VOL, 'hi': HIGH_VOL})
        # Weights sum to 1.0 <= 1.0, so the fund constructs without raising.
        orch = FundOrchestrator(desks)
        assert orch.active_capital == pytest.approx(1.0)

    def test_keeps_internal_risk_book_allocation_in_sync(self):
        class RiskBook:
            capital_allocation = 1.0

        lo, hi = ScriptedDesk('lo', 1.0), ScriptedDesk('hi', 1.0)
        lo.risk_book = RiskBook()
        hi.risk_book = RiskBook()

        weights = CrossDeskCapitalAllocator().allocate(
            [lo, hi], {'lo': LOW_VOL, 'hi': HIGH_VOL})

        assert lo.risk_book.capital_allocation == pytest.approx(weights['lo'])
        assert hi.risk_book.capital_allocation == pytest.approx(weights['hi'])


class TestReturnsFromEquity:
    def test_basic_returns(self):
        out = CrossDeskCapitalAllocator.returns_from_equity([100, 110, 99])
        assert out == pytest.approx([0.1, 99 / 110 - 1.0])

    def test_skips_gaps_but_keeps_valid_consecutive(self):
        out = CrossDeskCapitalAllocator.returns_from_equity(
            [100, 110, float('nan'), 120, 132])
        # 100->110 = .1 ; 110->nan skip ; nan->120 skip ; 120->132 = .1
        assert out == pytest.approx([0.1, 0.1])

    def test_nonpositive_breaks_the_segment(self):
        out = CrossDeskCapitalAllocator.returns_from_equity([100, 0, 120])
        assert out == []
