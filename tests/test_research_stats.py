"""Tests for analysis.research_stats (Probabilistic / Deflated Sharpe).
Offline, deterministic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import kurtosis, norm, skew

from analysis.research_stats import (deflated_sharpe_ratio,
                                     expected_max_sharpe_z,
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
