"""Tests for the OPT-IN portfolio-optimizer weighting modes on
CrossDeskCapitalAllocator: MAX DIVERSIFICATION and MEAN-VARIANCE.

max_diversification_weights maximizes the diversification ratio (wᵀσ)/sqrt(wᵀΣw)
via the closed form w ∝ Σ⁻¹σ, so a REDUNDANT (highly correlated) desk earns LESS
weight than inverse-vol — which ignores correlation — would give it.
mean_variance_weights is the long-only max-Sharpe tangency portfolio w ∝ Σ⁻¹μ,
tilting toward desks with a better risk-adjusted MEAN. Both clip negatives to 0
(long-only) and scale to sum EXACTLY to target_gross <= 1.0.

Both degrade CONSERVATIVELY (same discipline as the rest of the allocator): a
too-short overlap, any degenerate-vol desk, a singular/non-finite covariance, or
a long-only solution that clips to nothing falls back to risk_parity_weights
(inverse-vol), which itself falls back to equal weight.

These tests cover ONLY the new optimizer paths. The risk-parity default and its
byte-identity live in tests/test_capital_allocator.py (unchanged); the
full-covariance mode in tests/test_capital_allocator_cov.py. Offline,
deterministic (fixed RNG seed), no engine. Mirrors test_capital_allocator_cov.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from desks.capital_allocator import CrossDeskCapitalAllocator


# ----------------------------------------------------------------------
# Deterministic fixtures
# ----------------------------------------------------------------------
def correlated_book(seed: int = 42, n: int = 260,
                    vol: float = 0.01, idio: float = 0.4):
    """Three desks: A and B share a common factor (positively correlated), C is
    driven by an independent factor. Equal vol scale, distinct idiosyncratic
    noise. Deterministic via numpy's frozen PCG64 stream."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)          # shared factor -> A,B correlated
    g = rng.standard_normal(n)          # independent factor -> C uncorrelated
    a = vol * (f + idio * rng.standard_normal(n))
    b = vol * (f + idio * rng.standard_normal(n))
    c = vol * (g + idio * rng.standard_normal(n))
    return {'A': a.tolist(), 'B': b.tolist(), 'C': c.tolist()}


def sharpe_book(seed: int = 11, n: int = 300, vol: float = 0.01):
    """Three near-uncorrelated, equal-vol desks with DISTINCT mean returns, so
    their risk-adjusted (Sharpe) ranking is lo < mid < hi. Mean-variance should
    tilt toward 'hi'; inverse-vol (equal vol) splits them ~evenly."""
    rng = np.random.default_rng(seed)
    means = {'lo': 0.0002, 'mid': 0.0006, 'hi': 0.0012}
    return {k: (m + vol * rng.standard_normal(n)).tolist()
            for k, m in means.items()}


def diversification_ratio(book, weights):
    """DR = (wᵀσ)/sqrt(wᵀΣw) on the SAME sample covariance the allocator builds."""
    keys = list(book)
    cov = np.cov(np.vstack([np.asarray(book[k], dtype=float) for k in keys]))
    sigma = np.sqrt(np.diag(cov))
    w = np.array([weights[k] for k in keys], dtype=float)
    return float(w @ sigma) / float(np.sqrt(w @ cov @ w))


# ======================================================================
# MAX DIVERSIFICATION
# ======================================================================
class TestMaxDiversification:
    def test_correlated_pair_combined_weight_below_inverse_vol(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        inv = a.risk_parity_weights(book)
        md = a.max_diversification_weights(book)
        arr = {k: np.asarray(v) for k, v in book.items()}
        assert np.corrcoef(arr['A'], arr['B'])[0, 1] > 0.5
        assert abs(np.corrcoef(arr['A'], arr['C'])[0, 1]) < 0.3
        # Headline: a redundant (correlated) pair gets LESS combined weight.
        assert md['A'] + md['B'] < inv['A'] + inv['B']
        # The freed capital flows to the uncorrelated desk.
        assert md['C'] > inv['C']

    def test_each_correlated_desk_individually_reduced(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        inv = a.risk_parity_weights(book)
        md = a.max_diversification_weights(book)
        assert md['A'] < inv['A']
        assert md['B'] < inv['B']

    def test_improves_diversification_ratio_over_inverse_vol(self):
        # The whole point of the mode: it should achieve a HIGHER diversification
        # ratio than inverse-vol on a correlated book.
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        dr_md = diversification_ratio(book, a.max_diversification_weights(book))
        dr_inv = diversification_ratio(book, a.risk_parity_weights(book))
        assert dr_md > dr_inv

    def test_determinism(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        assert a.max_diversification_weights(book) \
            == a.max_diversification_weights(book)


class TestMaxDiversificationSums:
    def test_sum_to_one(self):
        w = CrossDeskCapitalAllocator().max_diversification_weights(
            correlated_book())
        assert sum(w.values()) == pytest.approx(1.0)

    def test_sum_to_target_gross_below_one(self):
        w = CrossDeskCapitalAllocator(
            target_gross=0.6).max_diversification_weights(correlated_book())
        assert sum(w.values()) == pytest.approx(0.6)
        assert sum(w.values()) <= 0.6 + 1e-9

    def test_empty_is_empty(self):
        assert CrossDeskCapitalAllocator().max_diversification_weights({}) == {}

    def test_single_desk_is_full_target(self):
        a = CrossDeskCapitalAllocator(target_gross=0.8)
        w = a.max_diversification_weights({'solo': [0.01, -0.02, 0.015, -0.01]})
        assert w == {'solo': pytest.approx(0.8)}

    def test_no_over_leverage_across_targets(self):
        for target in (1.0, 0.8, 0.6, 0.3):
            a = CrossDeskCapitalAllocator(target_gross=target)
            w = a.max_diversification_weights(correlated_book())
            assert sum(w.values()) <= target + 1e-9
            assert all(v >= 0.0 for v in w.values())


# ======================================================================
# MEAN VARIANCE
# ======================================================================
class TestMeanVariance:
    def test_favors_higher_risk_adjusted_mean(self):
        a = CrossDeskCapitalAllocator()
        book = sharpe_book()
        inv = a.risk_parity_weights(book)
        mv = a.mean_variance_weights(book)
        sharpe = {k: float(np.mean(v)) / float(np.std(v))
                  for k, v in book.items()}
        # Sanity: the fixture really does rank lo < mid < hi by Sharpe.
        assert sharpe['lo'] < sharpe['mid'] < sharpe['hi']
        # Headline: mean-variance gives the best-Sharpe desk the MOST weight and
        # MORE than inverse-vol (equal-vol -> ~even) would.
        assert mv['hi'] == max(mv.values())
        assert mv['hi'] > inv['hi']
        assert mv['hi'] > mv['mid'] > mv['lo']

    def test_determinism(self):
        a = CrossDeskCapitalAllocator()
        book = sharpe_book()
        assert a.mean_variance_weights(book) == a.mean_variance_weights(book)


class TestMeanVarianceSums:
    def test_sum_to_one(self):
        w = CrossDeskCapitalAllocator().mean_variance_weights(sharpe_book())
        assert sum(w.values()) == pytest.approx(1.0)

    def test_sum_to_target_gross_below_one(self):
        w = CrossDeskCapitalAllocator(
            target_gross=0.6).mean_variance_weights(sharpe_book())
        assert sum(w.values()) == pytest.approx(0.6)
        assert sum(w.values()) <= 0.6 + 1e-9

    def test_empty_is_empty(self):
        assert CrossDeskCapitalAllocator().mean_variance_weights({}) == {}

    def test_single_desk_is_full_target(self):
        a = CrossDeskCapitalAllocator(target_gross=0.8)
        w = a.mean_variance_weights({'solo': [0.01, -0.02, 0.015, -0.01]})
        assert w == {'solo': pytest.approx(0.8)}

    def test_no_over_leverage_across_targets(self):
        for target in (1.0, 0.8, 0.6, 0.3):
            a = CrossDeskCapitalAllocator(target_gross=target)
            w = a.mean_variance_weights(sharpe_book())
            assert sum(w.values()) <= target + 1e-9
            assert all(v >= 0.0 for v in w.values())


# ======================================================================
# Conservative degrade paths (shared by both modes)
# ======================================================================
class TestDegradePaths:
    @pytest.mark.parametrize('method', ['max_diversification_weights',
                                        'mean_variance_weights'])
    def test_short_overlap_falls_back_to_inverse_vol(self, method):
        # Fewer than n+1 overlapping rows -> cannot estimate an n-desk covariance
        # -> inverse-vol (NOT equal weight: distinct vol -> UNEQUAL split).
        a = CrossDeskCapitalAllocator()
        book = {'A': [0.01, 0.03], 'B': [0.01, 0.005]}
        out = getattr(a, method)(book)
        assert out == a.risk_parity_weights(book)
        assert out['A'] != pytest.approx(out['B'])  # genuinely inverse-vol

    @pytest.mark.parametrize('method', ['max_diversification_weights',
                                        'mean_variance_weights'])
    def test_degenerate_vol_falls_back_to_inverse_vol_then_equal(self, method):
        a = CrossDeskCapitalAllocator()
        book = {'good': [0.01, -0.02, 0.015, -0.01, 0.02],
                'flat': [0.0, 0.0, 0.0, 0.0, 0.0]}
        out = getattr(a, method)(book)
        assert out['good'] == pytest.approx(0.5)
        assert out['flat'] == pytest.approx(0.5)
        assert out == a.risk_parity_weights(book)  # equal-weight fallback

    @pytest.mark.parametrize('method', ['max_diversification_weights',
                                        'mean_variance_weights'])
    def test_singular_covariance_falls_back_to_inverse_vol(self, method):
        # Two IDENTICAL desks make the covariance singular -> np.linalg.solve
        # degrades to inverse-vol rather than blowing up.
        a = CrossDeskCapitalAllocator()
        x = [0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.01, -0.015]
        y = [0.005, 0.01, -0.01, 0.02, -0.015, 0.01, -0.005, 0.012]
        book = {'A': list(x), 'B': list(x), 'C': y}
        out = getattr(a, method)(book)
        assert all(np.isfinite(v) for v in out.values())
        assert sum(out.values()) == pytest.approx(1.0)
        assert out == a.risk_parity_weights(book)  # graceful inverse-vol degrade

    @pytest.mark.parametrize('method', ['max_diversification_weights',
                                        'mean_variance_weights'])
    def test_numerical_degrade_logs_reason_not_silent(self, method, caplog):
        # A short-overlap book degrades via a NUMERICAL guard (not the
        # degenerate-desk gate); it must LOG why so the reweighter's audit isn't
        # mistaken for a genuine optimizer result.
        a = CrossDeskCapitalAllocator()
        with caplog.at_level(logging.INFO, logger='desks.capital_allocator'):
            getattr(a, method)({'A': [0.01, 0.03], 'B': [0.01, 0.005]})
        assert any('degraded to inverse-vol' in r.message
                   for r in caplog.records)

    def test_mean_variance_no_positive_mean_degrades(self):
        # Every desk loses money on average -> w ∝ Σ⁻¹μ clips to nothing ->
        # degrade to inverse-vol with a named reason (not a silent equal weight).
        a = CrossDeskCapitalAllocator()
        rng = np.random.default_rng(5)
        neg = {k: (-0.001 + 0.01 * rng.standard_normal(120)).tolist()
               for k in ('A', 'B', 'C')}
        w, reason = a.mean_variance_weights_with_status(neg)
        assert reason is not None
        assert w == a.risk_parity_weights(neg)

    @pytest.mark.parametrize('method', ['max_diversification_weights',
                                        'mean_variance_weights'])
    def test_never_raises_on_pathological_input(self, method):
        a = CrossDeskCapitalAllocator()
        for book in (
            {'A': [0.01, 0.02, 0.03], 'B': [float('nan'), 0.01, 0.02]},
            {'A': [0.0, 0.0, 0.0], 'B': [0.0, 0.0, 0.0]},
            {'A': [1e-300, -1e-300, 1e-300], 'B': [1e-300, 1e-300, -1e-300]},
        ):
            w = getattr(a, method)(book)
            assert sum(w.values()) == pytest.approx(a.target_gross)


# ======================================================================
# Status variants: (weights, reason) for the reweighter's honest audit
# ======================================================================
class TestStatusVariants:
    @pytest.mark.parametrize('method,book', [
        ('max_diversification_weights_with_status', correlated_book()),
        ('mean_variance_weights_with_status', sharpe_book()),
    ])
    def test_success_reports_no_reason(self, method, book):
        a = CrossDeskCapitalAllocator()
        w, reason = getattr(a, method)(book)
        assert reason is None
        plain = getattr(a, method.replace('_with_status', ''))(book)
        assert w == plain  # public method delegates to the status variant

    @pytest.mark.parametrize('method', [
        'max_diversification_weights_with_status',
        'mean_variance_weights_with_status'])
    def test_numerical_degrade_reports_a_reason(self, method):
        a = CrossDeskCapitalAllocator()
        book = {'A': [0.01, 0.03], 'B': [0.01, 0.005]}  # short overlap
        w, reason = getattr(a, method)(book)
        assert reason is not None
        assert w == a.risk_parity_weights(book)

    @pytest.mark.parametrize('method', [
        'max_diversification_weights_with_status',
        'mean_variance_weights_with_status'])
    def test_degenerate_gate_reports_no_reason(self, method):
        # The degenerate-desk gate is captured by the caller via
        # degenerate_desks(), so the status reason stays None (not double-named).
        a = CrossDeskCapitalAllocator()
        book = {'good': [0.01, -0.02, 0.015, -0.01, 0.02],
                'flat': [0.0, 0.0, 0.0, 0.0, 0.0]}
        w, reason = getattr(a, method)(book)
        assert reason is None
        assert w == a.risk_parity_weights(book)


# ======================================================================
# The default risk-parity path is UNAFFECTED by the new modes
# ======================================================================
class TestDefaultPathUnaffected:
    @pytest.mark.parametrize('method', ['max_diversification_weights',
                                        'mean_variance_weights'])
    def test_risk_parity_weights_unchanged_alongside_new_mode(self, method):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        before = a.risk_parity_weights(book)
        getattr(a, method)(book)
        after = a.risk_parity_weights(book)
        assert before == after
