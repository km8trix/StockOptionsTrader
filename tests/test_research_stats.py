"""Tests for analysis.research_stats (Probabilistic / Deflated Sharpe).
Offline, deterministic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import kurtosis, norm, skew
from scipy.stats import t as student_t

from analysis.research_stats import (benjamini_hochberg, bonferroni_alpha,
                                     deflated_sharpe_ratio,
                                     expected_max_sharpe_z, fold_oos_pvalue,
                                     fold_oos_tstat,
                                     probabilistic_sharpe_ratio)


def _normal_returns(n=252, mean=0.001, std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return (mean + std * rng.standard_normal(n)).tolist()


class TestProbabilisticSharpe:
    def test_in_unit_interval(self):
        psr = probabilistic_sharpe_ratio(_normal_returns(), 0.0)
        assert psr is not None and 0.0 <= psr <= 1.0

    def test_strong_positive_mean_gives_high_psr(self):
        assert probabilistic_sharpe_ratio(
            _normal_returns(mean=0.002, std=0.01), 0.0) > 0.9

    def test_zero_mean_near_half(self):
        psr = probabilistic_sharpe_ratio(
            _normal_returns(mean=0.0, std=0.01, seed=3), 0.0)
        assert 0.2 < psr < 0.8

    def test_higher_benchmark_lowers_psr(self):
        returns = _normal_returns(mean=0.002, seed=1)
        assert (probabilistic_sharpe_ratio(returns, 0.0)
                > probabilistic_sharpe_ratio(returns, 0.2))

    def test_too_few_returns_is_none(self):
        assert probabilistic_sharpe_ratio([0.01]) is None
        assert probabilistic_sharpe_ratio([]) is None

    def test_zero_variance_is_none(self):
        assert probabilistic_sharpe_ratio([0.01] * 10) is None

    def test_non_finite_filtered(self):
        # NaN/inf are dropped; the finite remainder still yields a value.
        clean = _normal_returns(seed=5)
        dirty = clean[:10] + [float('nan'), float('inf')] + clean[10:]
        assert probabilistic_sharpe_ratio(dirty, 0.0) is not None

    def test_matches_independent_formula(self):
        returns = _normal_returns(seed=7)
        arr = np.asarray(returns)
        sr = arr.mean() / arr.std(ddof=1)
        sk = skew(arr, bias=False)
        kt = kurtosis(arr, fisher=False, bias=False)
        var_term = 1.0 - sk * sr + ((kt - 1.0) / 4.0) * sr * sr
        z = sr * math.sqrt(len(arr) - 1) / math.sqrt(var_term)
        assert probabilistic_sharpe_ratio(returns, 0.0) == pytest.approx(
            float(norm.cdf(z)))


class TestExpectedMaxSharpe:
    def test_zero_for_single_or_no_trial(self):
        assert expected_max_sharpe_z(1) == 0.0
        assert expected_max_sharpe_z(0) == 0.0

    def test_monotonic_in_trials(self):
        assert (expected_max_sharpe_z(2) < expected_max_sharpe_z(10)
                < expected_max_sharpe_z(100))


class TestDeflatedSharpe:
    def test_equals_psr_when_single_trial(self):
        returns = _normal_returns(seed=2)
        psr = probabilistic_sharpe_ratio(returns, 0.0)
        assert deflated_sharpe_ratio(returns, 1) == psr
        assert deflated_sharpe_ratio(returns, 0) == psr  # clamped to no-deflation

    def test_decreases_with_more_trials(self):
        returns = _normal_returns(mean=0.0015, seed=2)
        d1 = deflated_sharpe_ratio(returns, 1)
        d10 = deflated_sharpe_ratio(returns, 10)
        d100 = deflated_sharpe_ratio(returns, 100)
        assert d1 >= d10 >= d100
        assert d100 < d1  # multiple testing strictly raises the bar

    def test_too_few_returns_is_none(self):
        assert deflated_sharpe_ratio([0.01], 5) is None

    def test_in_unit_interval(self):
        dsr = deflated_sharpe_ratio(_normal_returns(seed=4), 25)
        assert dsr is not None and 0.0 <= dsr <= 1.0


class TestFoldOOSTstat:
    def test_matches_one_sample_t_identity(self):
        # t = mean / (std_ddof1 / sqrt(N)) == sharpe * sqrt(N).
        returns = [0.02, 0.015, 0.025, 0.018, 0.021]
        arr = np.asarray(returns)
        expected = arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))
        assert fold_oos_tstat(returns) == pytest.approx(float(expected))

    def test_positive_mean_positive_tstat(self):
        assert fold_oos_tstat([0.02, 0.015, 0.025, 0.018]) > 0.0

    def test_negative_mean_negative_tstat(self):
        assert fold_oos_tstat([-0.02, -0.015, -0.025, -0.018]) < 0.0

    def test_too_few_returns_is_none(self):
        assert fold_oos_tstat([0.01]) is None
        assert fold_oos_tstat([]) is None

    def test_zero_variance_is_none(self):
        # Degenerate std -> None (same guard as PSR/DSR), never a giant t.
        assert fold_oos_tstat([0.01] * 6) is None

    def test_non_finite_filtered(self):
        clean = _normal_returns(mean=0.001, seed=11)
        dirty = clean[:5] + [float('nan'), float('inf')] + clean[5:]
        assert fold_oos_tstat(dirty) is not None


class TestFoldOOSPvalue:
    def test_matches_one_sided_upper_tail(self):
        returns = _normal_returns(mean=0.0008, seed=13)
        t_stat = fold_oos_tstat(returns)
        n = len(returns)
        expected = float(student_t.sf(t_stat, df=n - 1))
        assert fold_oos_pvalue(returns) == pytest.approx(expected)

    def test_strong_positive_mean_small_pvalue(self):
        # Consistent positive returns -> significant one-sided edge.
        assert fold_oos_pvalue([0.02, 0.018, 0.022, 0.019, 0.021]) < 0.05

    def test_negative_mean_pvalue_above_half(self):
        # One-sided (mean > 0): a losing fold is NOT significant (p > 0.5).
        assert fold_oos_pvalue([-0.02, -0.018, -0.022, -0.019]) > 0.5

    def test_in_unit_interval(self):
        p = fold_oos_pvalue(_normal_returns(seed=14))
        assert p is not None and 0.0 <= p <= 1.0

    def test_too_few_returns_is_none(self):
        assert fold_oos_pvalue([0.01]) is None
        assert fold_oos_pvalue([0.01] * 6) is None  # zero variance


class TestBonferroniAlpha:
    def test_divides_alpha_by_m(self):
        assert bonferroni_alpha(0.05, 5) == pytest.approx(0.01)
        assert bonferroni_alpha(0.05, 1) == pytest.approx(0.05)

    def test_non_positive_m_is_none(self):
        assert bonferroni_alpha(0.05, 0) is None
        assert bonferroni_alpha(0.05, -3) is None


class TestBenjaminiHochberg:
    def test_step_up_rejects_expected_set(self):
        # m=3, alpha=0.05. thresholds k/m*alpha = [.01667, .03333, .05].
        # sorted p [.01, .02, .5]: k=1 ok, k=2 ok, k=3 fail -> reject ranks 1,2.
        result = benjamini_hochberg([0.5, 0.01, 0.02], alpha=0.05)
        assert result['m'] == 3
        assert result['bh_threshold'] == pytest.approx(0.02)
        assert result['n_significant_bh'] == 2
        # rejected_bh aligned to INPUT order [0.5, 0.01, 0.02].
        assert result['rejected_bh'] == [False, True, True]
        assert result['alpha'] == pytest.approx(0.05)

    def test_none_entries_dropped_from_family(self):
        # The None fold is not counted in m and never rejected.
        result = benjamini_hochberg([0.01, None, 0.02, 0.5], alpha=0.05)
        assert result['m'] == 3
        assert result['bh_threshold'] == pytest.approx(0.02)
        assert result['rejected_bh'] == [True, False, True, False]

    def test_no_rejections_when_all_large(self):
        result = benjamini_hochberg([0.4, 0.6, 0.9], alpha=0.05)
        assert result['m'] == 3
        assert result['bh_threshold'] is None
        assert result['n_significant_bh'] == 0
        assert result['rejected_bh'] == [False, False, False]

    def test_empty_or_all_none_family(self):
        result = benjamini_hochberg([None, None], alpha=0.05)
        assert result['m'] == 0
        assert result['bh_threshold'] is None
        assert result['n_significant_bh'] == 0
        assert result['rejected_bh'] == [False, False]
        empty = benjamini_hochberg([], alpha=0.05)
        assert empty['m'] == 0 and empty['rejected_bh'] == []
