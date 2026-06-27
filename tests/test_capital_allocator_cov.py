"""Tests for the OPT-IN full-covariance risk parity on CrossDeskCapitalAllocator.

CrossDeskCapitalAllocator.risk_parity_cov_weights equalizes each desk's RISK
CONTRIBUTION over the FULL covariance matrix, rather than the inverse-vol
DIAGONAL approximation of risk_parity_weights. The headline property: two
POSITIVELY-CORRELATED desks carry LESS combined weight than inverse-vol gives
them, because inverse-vol ignores the correlation that makes their risks add up.
The converged weights have ~EQUAL marginal risk contributions, scaled to sum
EXACTLY to target_gross <= 1.0.

It degrades CONSERVATIVELY (same discipline as the rest of the allocator): a
too-short overlap, any degenerate-vol desk, or a singular/non-finite covariance
falls back to risk_parity_weights (inverse-vol), which itself falls back to
equal weight.

These tests cover ONLY the new full-covariance path. The risk-parity default and
its byte-identity live in tests/test_capital_allocator.py (unchanged). Offline,
deterministic (fixed RNG seed), no engine.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from desks.capital_allocator import CrossDeskCapitalAllocator


# ----------------------------------------------------------------------
# Deterministic correlated/uncorrelated fixtures
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


def risk_contributions(book, weights):
    """Per-desk risk contribution rc_i = w_i * (Cov @ w)_i / port_vol on the
    SAME sample covariance the allocator builds."""
    keys = list(book)
    cov = np.cov(np.vstack([np.asarray(book[k], dtype=float) for k in keys]))
    w = np.array([weights[k] for k in keys], dtype=float)
    port_vol = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / port_vol


# ----------------------------------------------------------------------
# (a) The correlated pair is DOWN-WEIGHTED vs plain inverse-vol
# ----------------------------------------------------------------------
class TestCorrelatedPairDownWeighted:
    def test_correlated_pair_combined_weight_below_inverse_vol(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        inv = a.risk_parity_weights(book)
        cov = a.risk_parity_cov_weights(book)
        # Sanity: A,B really are positively correlated and C is not.
        arr = {k: np.asarray(v) for k, v in book.items()}
        assert np.corrcoef(arr['A'], arr['B'])[0, 1] > 0.5
        assert abs(np.corrcoef(arr['A'], arr['C'])[0, 1]) < 0.3
        # Headline: full-cov gives the correlated pair LESS combined weight.
        assert cov['A'] + cov['B'] < inv['A'] + inv['B']
        # The freed capital flows to the uncorrelated desk.
        assert cov['C'] > inv['C']

    def test_each_correlated_desk_individually_reduced(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        inv = a.risk_parity_weights(book)
        cov = a.risk_parity_cov_weights(book)
        assert cov['A'] < inv['A']
        assert cov['B'] < inv['B']


# ----------------------------------------------------------------------
# (b) Weights sum to target_gross
# ----------------------------------------------------------------------
class TestSumsToTargetGross:
    def test_sum_to_one(self):
        w = CrossDeskCapitalAllocator().risk_parity_cov_weights(
            correlated_book())
        assert sum(w.values()) == pytest.approx(1.0)

    def test_sum_to_target_gross_below_one(self):
        w = CrossDeskCapitalAllocator(target_gross=0.6).risk_parity_cov_weights(
            correlated_book())
        assert sum(w.values()) == pytest.approx(0.6)
        assert sum(w.values()) <= 0.6 + 1e-9

    def test_empty_is_empty(self):
        assert CrossDeskCapitalAllocator().risk_parity_cov_weights({}) == {}

    def test_single_desk_is_full_target(self):
        # One desk: nothing to de-correlate; it takes the whole target_gross.
        a = CrossDeskCapitalAllocator(target_gross=0.8)
        w = a.risk_parity_cov_weights({'solo': [0.01, -0.02, 0.015, -0.01]})
        assert w == {'solo': pytest.approx(0.8)}

    def test_determinism(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        assert a.risk_parity_cov_weights(book) == a.risk_parity_cov_weights(book)


# ----------------------------------------------------------------------
# (c) Marginal risk contributions are ~equal after convergence
# ----------------------------------------------------------------------
class TestEqualRiskContributions:
    def test_risk_contributions_approximately_equal(self):
        book = correlated_book()
        w = CrossDeskCapitalAllocator().risk_parity_cov_weights(book)
        rc = risk_contributions(book, w)
        # The damped iteration converges: every desk contributes ~the same
        # risk. (Far tighter than inverse-vol, which leaves the correlated pair
        # over-contributing — asserted below.)
        rel_spread = (rc.max() - rc.min()) / rc.mean()
        assert rel_spread < 0.05

    def test_more_equal_than_inverse_vol(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        rc_cov = risk_contributions(book, a.risk_parity_cov_weights(book))
        rc_inv = risk_contributions(book, a.risk_parity_weights(book))
        cov_spread = (rc_cov.max() - rc_cov.min()) / rc_cov.mean()
        inv_spread = (rc_inv.max() - rc_inv.min()) / rc_inv.mean()
        assert cov_spread < inv_spread


# ----------------------------------------------------------------------
# (d) Conservative degrade: short overlap / degenerate vol / singular cov
# ----------------------------------------------------------------------
class TestDegradePaths:
    def test_short_overlap_falls_back_to_inverse_vol(self):
        # Fewer than n+1 overlapping rows -> cannot estimate an n-desk
        # covariance -> fall back to inverse-vol (NOT equal weight: the two
        # desks have distinct vol, so inverse-vol gives an UNEQUAL split).
        a = CrossDeskCapitalAllocator()
        book = {'A': [0.01, 0.03], 'B': [0.01, 0.005]}
        cov = a.risk_parity_cov_weights(book)
        inv = a.risk_parity_weights(book)
        assert cov == inv
        assert cov['A'] != pytest.approx(cov['B'])  # genuinely inverse-vol

    def test_degenerate_vol_falls_back_to_inverse_vol_then_equal(self):
        # A flat (zero-vol) desk is degenerate -> inverse-vol, which itself
        # falls back to WHOLE-FUND equal weight.
        a = CrossDeskCapitalAllocator()
        book = {'good': [0.01, -0.02, 0.015, -0.01, 0.02],
                'flat': [0.0, 0.0, 0.0, 0.0, 0.0]}
        cov = a.risk_parity_cov_weights(book)
        assert cov['good'] == pytest.approx(0.5)
        assert cov['flat'] == pytest.approx(0.5)
        assert cov == a.risk_parity_weights(book)  # equal-weight fallback

    def test_singular_covariance_falls_back_to_inverse_vol(self):
        # Two IDENTICAL desks make the covariance rank-deficient (singular). The
        # iteration's port-variance guard catches the degeneracy and degrades to
        # inverse-vol rather than blowing up — finite weights summing to target.
        a = CrossDeskCapitalAllocator()
        x = [0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.01, -0.015]
        y = [0.005, 0.01, -0.01, 0.02, -0.015, 0.01, -0.005, 0.012]
        book = {'A': list(x), 'B': list(x), 'C': y}
        cov = a.risk_parity_cov_weights(book)
        assert all(np.isfinite(v) for v in cov.values())
        assert sum(cov.values()) == pytest.approx(1.0)
        assert cov == a.risk_parity_weights(book)  # graceful inverse-vol degrade

    def test_numerical_degrade_logs_reason_not_silent(self, caplog):
        # A short-overlap book degrades via a NUMERICAL guard (not the
        # degenerate-desk gate), the path that previously returned inverse-vol
        # weights silently. It must now log WHY, so the reweighter's
        # fallback=False audit entry isn't mistaken for a genuine cov result.
        a = CrossDeskCapitalAllocator()
        with caplog.at_level(logging.INFO, logger='desks.capital_allocator'):
            a.risk_parity_cov_weights({'A': [0.01, 0.03], 'B': [0.01, 0.005]})
        assert any('degraded to inverse-vol' in r.message
                   for r in caplog.records)

    def test_hedge_desk_degrade_logs_reason(self, caplog):
        # A negatively-correlated (hedge) desk makes the damped step non-finite
        # -> degrade. The reason names the hedge case so it is diagnosable.
        a = CrossDeskCapitalAllocator()
        rng = np.random.default_rng(1)
        f = rng.standard_normal(200)
        book = {
            'A': (0.01 * (f + 0.5 * rng.standard_normal(200))).tolist(),
            'B': (0.01 * (f + 0.5 * rng.standard_normal(200))).tolist(),
            'C': (0.01 * (-0.8 * f + 0.5 * rng.standard_normal(200))).tolist(),
        }
        with caplog.at_level(logging.INFO, logger='desks.capital_allocator'):
            cov = a.risk_parity_cov_weights(book)
        assert cov == a.risk_parity_weights(book)  # degraded to inverse-vol
        assert any('degraded to inverse-vol' in r.message
                   for r in caplog.records)

    def test_never_raises_on_pathological_input(self):
        # The try/except contract: any numerical failure degrades, never raises.
        a = CrossDeskCapitalAllocator()
        for book in (
            {'A': [0.01, 0.02, 0.03], 'B': [float('nan'), 0.01, 0.02]},
            {'A': [0.0, 0.0, 0.0], 'B': [0.0, 0.0, 0.0]},
            {'A': [1e-300, -1e-300, 1e-300], 'B': [1e-300, 1e-300, -1e-300]},
        ):
            w = a.risk_parity_cov_weights(book)
            assert sum(w.values()) == pytest.approx(a.target_gross)


# ----------------------------------------------------------------------
# Status variant: (weights, reason) for the reweighter's honest audit
# ----------------------------------------------------------------------
class TestStatusVariant:
    def test_success_reports_no_reason(self):
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        w, reason = a.risk_parity_cov_weights_with_status(book)
        assert reason is None
        assert w == a.risk_parity_cov_weights(book)  # public method delegates

    def test_numerical_degrade_reports_a_reason(self):
        # Short overlap degrades to inverse-vol via a numerical guard -> reason.
        a = CrossDeskCapitalAllocator()
        book = {'A': [0.01, 0.03], 'B': [0.01, 0.005]}
        w, reason = a.risk_parity_cov_weights_with_status(book)
        assert reason is not None
        assert w == a.risk_parity_weights(book)

    def test_degenerate_gate_reports_no_reason(self):
        # The degenerate-desk gate is captured by the caller via
        # degenerate_desks(), so the status reason stays None (not double-named).
        a = CrossDeskCapitalAllocator()
        book = {'good': [0.01, -0.02, 0.015, -0.01, 0.02],
                'flat': [0.0, 0.0, 0.0, 0.0, 0.0]}
        w, reason = a.risk_parity_cov_weights_with_status(book)
        assert reason is None
        assert w == a.risk_parity_weights(book)


# ----------------------------------------------------------------------
# (e) The default risk-parity path is UNAFFECTED by the new mode
# ----------------------------------------------------------------------
class TestDefaultPathUnaffected:
    def test_risk_parity_weights_unchanged_alongside_cov_mode(self):
        # Calling the new method must not perturb the inverse-vol default for
        # the same book (separate, additive code path).
        a = CrossDeskCapitalAllocator()
        book = correlated_book()
        before = a.risk_parity_weights(book)
        a.risk_parity_cov_weights(book)
        after = a.risk_parity_weights(book)
        assert before == after

    def test_no_over_leverage_across_targets(self):
        for target in (1.0, 0.8, 0.6, 0.3):
            a = CrossDeskCapitalAllocator(target_gross=target)
            w = a.risk_parity_cov_weights(correlated_book())
            assert sum(w.values()) <= target + 1e-9
            assert all(v >= 0.0 for v in w.values())
