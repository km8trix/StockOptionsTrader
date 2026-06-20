"""Tests for the OPT-IN performance weighting on CrossDeskCapitalAllocator.

CrossDeskCapitalAllocator.performance_weights tilts capital toward desks with
better REALIZED RISK-ADJUSTED out-of-sample performance (a Sharpe-like
mean/vol score), with overfitting guards that are ON BY DEFAULT: a minimum
track record (the SAME degenerate_desks gate as risk parity), NON-NEGATIVE
score clipping, bounded per-desk weights [weight_min, weight_max] enforced by
clip-then-water-fill renormalization, and SHRINKAGE toward the equal-weight
prior. Weights are scaled to sum EXACTLY to target_gross <= 1.0 (no
over-leverage).

These tests cover ONLY the new performance path. The risk-parity default and
its byte-identity live in tests/test_capital_allocator.py (unchanged). Offline,
deterministic, no engine.
"""

from __future__ import annotations

import logging
import random

import pytest

from desks.capital_allocator import CrossDeskCapitalAllocator

# A high-Sharpe desk: positive mean, modest finite vol.
HIGH_SHARPE = [0.02, 0.01, 0.02, 0.01, 0.02]
# A near-zero-Sharpe desk: ~0 mean, finite vol.
ZERO_SHARPE = [0.001, -0.001, 0.001, -0.001, 0.001]
# A negative-Sharpe (drawing-down) desk: negative mean, finite vol.
NEG_SHARPE = [-0.02, 0.01, -0.02, 0.01, -0.02]


# ----------------------------------------------------------------------
# Construction validation (the new guard params)
# ----------------------------------------------------------------------
class TestConstruction:
    def test_default_guard_params(self):
        a = CrossDeskCapitalAllocator()
        assert a.weight_min == pytest.approx(0.05)
        assert a.weight_max == pytest.approx(0.60)
        assert a.shrinkage == pytest.approx(0.5)

    def test_weight_min_above_max_rejected(self):
        with pytest.raises(ValueError, match='weight_min'):
            CrossDeskCapitalAllocator(weight_min=0.6, weight_max=0.5)

    @pytest.mark.parametrize('bad', [-0.1, 1.5])
    def test_shrinkage_outside_unit_interval_rejected(self, bad):
        with pytest.raises(ValueError, match='shrinkage'):
            CrossDeskCapitalAllocator(shrinkage=bad)


# ----------------------------------------------------------------------
# Core tilt + invariants
# ----------------------------------------------------------------------
class TestPerformanceTilt:
    def test_higher_sharpe_gets_more_weight(self):
        w = CrossDeskCapitalAllocator().performance_weights(
            {'hi': HIGH_SHARPE, 'lo': ZERO_SHARPE})
        assert w['hi'] > w['lo']
        # Bounded by the per-desk band and summing to target_gross.
        assert 0.05 - 1e-9 <= w['lo'] and w['hi'] <= 0.60 + 1e-9
        assert sum(w.values()) == pytest.approx(1.0)

    def test_two_desk_capped_split_matches_design(self):
        # The documented two-desk case: higher-Sharpe desk reaches the cap.
        w = CrossDeskCapitalAllocator().performance_weights(
            {'hi': HIGH_SHARPE, 'lo': ZERO_SHARPE})
        assert w['hi'] == pytest.approx(0.6)
        assert w['lo'] == pytest.approx(0.4)

    def test_weights_sum_to_target_gross_one(self):
        w = CrossDeskCapitalAllocator(target_gross=1.0).performance_weights(
            {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE})
        assert sum(w.values()) == pytest.approx(1.0)

    def test_weights_sum_to_target_gross_below_one(self):
        w = CrossDeskCapitalAllocator(target_gross=0.6).performance_weights(
            {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE})
        assert sum(w.values()) == pytest.approx(0.6)

    def test_empty_is_empty(self):
        assert CrossDeskCapitalAllocator().performance_weights({}) == {}

    def test_determinism(self):
        a = CrossDeskCapitalAllocator()
        book = {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE}
        assert a.performance_weights(book) == a.performance_weights(book)


# ----------------------------------------------------------------------
# Bounds: cap and floor each bind, via water-filling redistribution
# ----------------------------------------------------------------------
class TestBounds:
    def test_dominant_desk_pinned_at_max_weak_floored_at_min(self):
        # Pure performance tilt (shrinkage=1) spreads enough to bind BOTH
        # bounds: the dominant desk hits weight_max, the negative-Sharpe desk
        # is floored at weight_min, never negative.
        dom = [0.05, 0.045, 0.05, 0.045, 0.05]
        mid = [0.01, 0.005, 0.01, 0.005, 0.01]
        weak = NEG_SHARPE
        a = CrossDeskCapitalAllocator(
            weight_min=0.05, weight_max=0.50, shrinkage=1.0)
        w = a.performance_weights({'dom': dom, 'mid': mid, 'weak': weak})
        assert w['dom'] == pytest.approx(0.50)   # pinned at weight_max
        assert w['weak'] == pytest.approx(0.05)  # floored at weight_min
        assert w['weak'] >= 0.0                  # never negative/short
        assert sum(w.values()) == pytest.approx(1.0)

    def test_no_desk_below_floor_or_above_cap(self):
        a = CrossDeskCapitalAllocator(
            weight_min=0.05, weight_max=0.50, shrinkage=1.0)
        w = a.performance_weights(
            {'dom': [0.05, 0.045, 0.05, 0.045],
             'mid': [0.01, 0.005, 0.01, 0.005],
             'weak': NEG_SHARPE})
        for frac in w.values():
            assert 0.05 - 1e-9 <= frac <= 0.50 + 1e-9

    def test_water_filling_redistributes_when_two_desks_bind_cap(self):
        # Two equally-strong desks both exceed weight_max -> both pinned at the
        # cap, the remaining mass flows to the weak desk (exercises the
        # multi-desk water-filling redistribution, not just a single cap).
        strong = [0.05, 0.045, 0.05, 0.045]
        weak = [0.001, -0.001, 0.001, -0.001]
        a = CrossDeskCapitalAllocator(
            weight_min=0.05, weight_max=0.40, shrinkage=1.0)
        w = a.performance_weights({'A': strong, 'B': list(strong), 'C': weak})
        assert w['A'] == pytest.approx(0.40)
        assert w['B'] == pytest.approx(0.40)
        assert w['C'] == pytest.approx(0.20)
        assert sum(w.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Shrinkage: tempers the tilt toward equal weight
# ----------------------------------------------------------------------
class TestShrinkage:
    def test_shrinkage_zero_is_pure_equal_weight(self):
        a = CrossDeskCapitalAllocator(shrinkage=0.0)
        w = a.performance_weights(
            {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE})
        for frac in w.values():
            assert frac == pytest.approx(1.0 / 3.0)

    def test_lower_shrinkage_moves_good_desk_toward_equal(self):
        # With the bounds slack (so neither binds), the dominant desk's weight
        # is STRICTLY monotone in shrinkage and equals the equal-weight prior
        # at shrinkage=0. Lowering shrinkage shrinks the good desk's tilt.
        book = {'dom': [0.05, 0.045, 0.05, 0.045, 0.05],
                'mid': [0.01, 0.005, 0.01, 0.005, 0.01],
                'weak': [0.001, -0.001, 0.001, -0.001, 0.001]}
        prev = None
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            a = CrossDeskCapitalAllocator(
                weight_min=0.0, weight_max=1.0, shrinkage=s)
            dom = a.performance_weights(book)['dom']
            if prev is not None:
                assert dom > prev  # strictly increases as shrinkage rises
            prev = dom
        # At shrinkage=0 the good desk is exactly the equal-weight prior.
        eq = CrossDeskCapitalAllocator(
            weight_min=0.0, weight_max=1.0, shrinkage=0.0)
        assert eq.performance_weights(book)['dom'] == pytest.approx(1.0 / 3.0)


# ----------------------------------------------------------------------
# Non-negative scores + all-fallback branches
# ----------------------------------------------------------------------
class TestNonNegativeAndFallback:
    def test_negative_sharpe_desk_floored_never_negative(self):
        # A drawing-down desk paired with a winner: the loser is floored at
        # weight_min, not shorted or zeroed.
        w = CrossDeskCapitalAllocator(weight_min=0.05).performance_weights(
            {'win': HIGH_SHARPE, 'lose': NEG_SHARPE})
        assert w['lose'] >= 0.05 - 1e-9
        assert w['lose'] > 0.0
        assert w['win'] > w['lose']

    def test_all_negative_sharpe_falls_back_to_equal_weight(self):
        w = CrossDeskCapitalAllocator().performance_weights(
            {'x': [-0.01, -0.02, -0.01], 'y': [-0.03, -0.01, -0.02]})
        assert w['x'] == pytest.approx(w['y'])
        assert sum(w.values()) == pytest.approx(1.0)

    def test_too_short_desk_falls_back_to_equal_weight(self):
        # Reuses the SAME degenerate_desks gate as risk_parity_weights.
        w = CrossDeskCapitalAllocator().performance_weights(
            {'short': [0.01], 'good': HIGH_SHARPE})
        assert w['short'] == pytest.approx(w['good'])

    def test_zero_vol_desk_falls_back_to_equal_weight(self):
        w = CrossDeskCapitalAllocator().performance_weights(
            {'flat': [0.0, 0.0, 0.0], 'good': HIGH_SHARPE})
        assert w['flat'] == pytest.approx(w['good'])

    def test_nonfinite_leaving_too_few_falls_back(self):
        w = CrossDeskCapitalAllocator().performance_weights(
            {'x': [float('nan'), 0.01], 'good': HIGH_SHARPE})
        assert w['x'] == pytest.approx(w['good'])

    def test_infeasible_min_bound_falls_back_to_equal_weight(self):
        # weight_min * n > 1 (0.4 * 3 = 1.2): no bounded set can sum to 1, so
        # the fund falls back to equal weight (the documented precedence:
        # exact sum over the per-desk bound).
        a = CrossDeskCapitalAllocator(
            weight_min=0.4, weight_max=0.6, shrinkage=1.0)
        w = a.performance_weights(
            {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE})
        for frac in w.values():
            assert frac == pytest.approx(1.0 / 3.0)

    def test_infeasible_max_bound_falls_back_to_equal_weight(self):
        # weight_max * n < 1 (0.3 * 3 = 0.9 < 1): infeasible -> equal weight.
        a = CrossDeskCapitalAllocator(
            weight_min=0.05, weight_max=0.30, shrinkage=1.0)
        w = a.performance_weights(
            {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE})
        for frac in w.values():
            assert frac == pytest.approx(1.0 / 3.0)


# ----------------------------------------------------------------------
# No over-leverage: the sum never exceeds target_gross
# ----------------------------------------------------------------------
class TestOverLeverageGuard:
    @pytest.mark.parametrize('target', [1.0, 0.8, 0.6, 0.3])
    def test_sum_never_exceeds_target_gross(self, target):
        a = CrossDeskCapitalAllocator(target_gross=target)
        # Across the tilting, fallback, and bound-binding regimes.
        for book in (
            {'hi': HIGH_SHARPE, 'lo': ZERO_SHARPE},
            {'a': HIGH_SHARPE, 'b': ZERO_SHARPE, 'c': NEG_SHARPE},
            {'x': [-0.01, -0.02, -0.01], 'y': [-0.03, -0.01, -0.02]},
            {'short': [0.01], 'good': HIGH_SHARPE},
        ):
            w = a.performance_weights(book)
            assert sum(w.values()) <= target + 1e-9


# ----------------------------------------------------------------------
# Risk-parity default is UNAFFECTED by the new guard params
# ----------------------------------------------------------------------
class TestGuardParamsDoNotTouchRiskParity:
    def test_risk_parity_ignores_guard_params(self):
        book = {'lo': [0.001, -0.001, 0.001, -0.001],
                'hi': [0.05, -0.05, 0.05, -0.05]}
        baseline = CrossDeskCapitalAllocator().risk_parity_weights(book)
        # Wildly different guard params must not change risk_parity_weights.
        tweaked = CrossDeskCapitalAllocator(
            weight_min=0.2, weight_max=0.45, shrinkage=0.1
        ).risk_parity_weights(book)
        assert tweaked == baseline


# ----------------------------------------------------------------------
# REGRESSION: the exact bounded-renormalize configs that previously
# over-leveraged. The old greedy clip-then-pin water-filling returned bounded
# fractions summing ABOVE 1.0 (e.g. weight_min=0.2, weight_max=0.9, n=5 -> 1.7
# = 170% gross). The fill-the-budget rewrite must now place EXACTLY the budget
# (sum == 1 for _bounded_renormalize, == target_gross end-to-end) with every
# weight inside [weight_min, weight_max].
# ----------------------------------------------------------------------
class TestBoundedRenormalizeOverLeverageRegression:
    # (weight_min, weight_max, n, old_buggy_sum) — the documented pre-fix sums.
    CASES = [
        (0.2, 0.9, 5, 1.7),
        (0.15, 0.4, 5, 1.25),
        (0.2, 0.324, 5, 1.124),
    ]

    @pytest.mark.parametrize('lo,hi,n,old_buggy_sum', CASES)
    def test_bounded_renormalize_fills_exactly_the_budget(
            self, lo, hi, n, old_buggy_sum):
        # A strongly DOMINANT score on one desk — the precise shape that drove
        # the old water-filling to pin one desk and floor the rest above budget.
        a = CrossDeskCapitalAllocator(weight_min=lo, weight_max=hi)
        keys = [f'd{i}' for i in range(n)]
        fractions = {k: (1.0 if i == 0 else 1e-6)
                     for i, k in enumerate(keys)}
        bounded = a._bounded_renormalize(fractions)
        # The whole budget is placed — no over-leverage, no under-deploy.
        assert sum(bounded.values()) == pytest.approx(1.0)
        # Every desk inside the band.
        for frac in bounded.values():
            assert lo - 1e-9 <= frac <= hi + 1e-9
        # Sanity: the budget is decisively below the buggy gross it used to hit.
        assert sum(bounded.values()) < old_buggy_sum - 0.05

    @pytest.mark.parametrize('lo,hi,n,old_buggy_sum', CASES)
    @pytest.mark.parametrize('target', [1.0, 0.6])
    def test_performance_weights_end_to_end_sums_to_budget(
            self, lo, hi, n, old_buggy_sum, target):
        # End-to-end via performance_weights: a dominant high-Sharpe desk plus
        # near-zero-Sharpe peers, the same configs that over-leveraged. The
        # bounded set must sum EXACTLY to target_gross with no out-of-bounds.
        a = CrossDeskCapitalAllocator(
            target_gross=target, weight_min=lo, weight_max=hi, shrinkage=1.0)
        book = {'dom': HIGH_SHARPE}
        for i in range(1, n):
            book[f'peer{i}'] = ZERO_SHARPE
        w = a.performance_weights(book)
        assert len(w) == n
        assert sum(w.values()) == pytest.approx(target)
        # No over-leverage: never above target.
        assert sum(w.values()) <= target + 1e-9
        # Each weight is a fraction of target_gross inside the band.
        for weight in w.values():
            frac = weight / target
            assert lo - 1e-9 <= frac <= hi + 1e-9
            assert weight >= 0.0


# ----------------------------------------------------------------------
# REGRESSION (property/fuzz): over randomized feasible configs, the bounded
# performance weights NEVER over-leverage and NEVER go negative. Seeded for
# determinism.
# ----------------------------------------------------------------------
class TestPerformanceWeightsFuzzNoOverLeverage:
    def _random_series(self, rng):
        length = rng.randint(3, 9)
        center = rng.choice([-0.015, -0.005, 0.0, 0.01, 0.02, 0.05])
        return [rng.gauss(center, rng.uniform(0.005, 0.03))
                for _ in range(length)]

    def test_fuzz_never_over_leverages_or_goes_negative(self, caplog):
        # Infeasible-bound configs legitimately log a warning + fall back to
        # equal weight; silence that expected noise so the run stays clean.
        caplog.set_level(logging.ERROR, logger='desks.capital_allocator')
        rng = random.Random(0xF00DCAFE)
        checked = 0
        for _ in range(4000):
            n = rng.randint(2, 12)
            target = rng.choice([1.0, 0.8, 0.6, 0.3])
            # Randomized feasible band: high floors and caps near 1/n included.
            lo = rng.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30])
            cap_floor = 1.0 / n
            hi = rng.uniform(max(lo, cap_floor * 0.4), 1.0)
            if hi < lo:
                hi = lo
            shrinkage = rng.random()
            try:
                a = CrossDeskCapitalAllocator(
                    target_gross=target, weight_min=lo, weight_max=hi,
                    shrinkage=shrinkage)
            except ValueError:
                continue
            # Skewed / concentrated scores: one strong desk, the rest noisy.
            book = {}
            for i in range(n):
                if i == 0 and rng.random() < 0.7:
                    book[f'd{i}'] = [rng.gauss(0.06, 0.002) for _ in range(7)]
                else:
                    book[f'd{i}'] = self._random_series(rng)
            w = a.performance_weights(book)
            checked += 1
            # No over-leverage beyond float slop.
            assert sum(w.values()) <= target + 1e-9
            # No negative / short weight.
            assert all(weight >= 0.0 for weight in w.values())
        assert checked > 3000  # the vast majority of configs were exercised


# ----------------------------------------------------------------------
# FILL SEMANTICS: a dominant desk caps at weight_max and the remainder is
# SHARED among the free desks; a zero-score desk under full shrinkage still
# absorbs leftover budget (it is not stuck at the floor) so the full target is
# deployed.
# ----------------------------------------------------------------------
class TestFillSemantics:
    def test_dominant_caps_and_remainder_shared_equally(self):
        # {A dominant, others tiny}, weight_min=0.05, weight_max=0.6, n=5:
        # A pins at 0.6, the remaining 0.4 budget spreads to the four free
        # desks as 0.1 each (their tiny equal fractions split it evenly).
        a = CrossDeskCapitalAllocator(weight_min=0.05, weight_max=0.6)
        fractions = {'A': 0.92, 'b': 0.02, 'c': 0.02, 'd': 0.02, 'e': 0.02}
        bounded = a._bounded_renormalize(fractions)
        assert bounded['A'] == pytest.approx(0.6)
        for other in ('b', 'c', 'd', 'e'):
            assert bounded[other] == pytest.approx(0.1)
        assert sum(bounded.values()) == pytest.approx(1.0)

    def test_all_zero_fractions_split_budget_not_stuck_at_floor(self):
        # With NO score signal (every fraction 0 — e.g. zero-score desks under
        # full shrinkage), the leftover budget is split EQUALLY rather than
        # leaving desks idle at weight_min. floor=0.1, n=3 -> 1/3 each (not
        # 0.1), proving the full target is deployed.
        a = CrossDeskCapitalAllocator(weight_min=0.1, weight_max=0.6)
        bounded = a._bounded_renormalize({'x': 0.0, 'y': 0.0, 'z': 0.0})
        for frac in bounded.values():
            assert frac == pytest.approx(1.0 / 3.0)
            assert frac > 0.1  # absorbed leftover, not stuck at the floor
        assert sum(bounded.values()) == pytest.approx(1.0)

    def test_zero_score_desk_absorbs_leftover_full_target_deployed(self):
        # End-to-end: a zero-score (negative-Sharpe, clipped) desk under
        # shrinkage=1.0 still takes a share of the budget so the FULL target is
        # deployed — not left idle at the floor.
        a = CrossDeskCapitalAllocator(
            weight_min=0.05, weight_max=0.6, shrinkage=1.0, target_gross=1.0)
        w = a.performance_weights(
            {'win': HIGH_SHARPE, 'zero': ZERO_SHARPE, 'neg': NEG_SHARPE})
        # The full target is deployed (no idle budget).
        assert sum(w.values()) == pytest.approx(1.0)
        # The zero/negative-edge desk sits strictly above the floor (absorbed
        # leftover the winner could not take past its cap).
        assert w['neg'] >= 0.05 - 1e-9
        assert w['zero'] > 0.05 + 1e-9
