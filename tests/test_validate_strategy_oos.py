"""validate_strategy_oos — the single pass/fail research gate (PLAN.md step 4).

Pass iff PSR(0) >= threshold AND >= 1 BH-significant positive sub-period. The
two lenses are independent (overall edge vs sub-period robustness), so each
must be exercised in isolation as well as together. Deterministic: seeded
synthetic returns, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.research_stats import validate_strategy_oos


def _year_labels(n_per_year, years):
    labels = []
    for y in years:
        labels.extend([y] * n_per_year)
    return labels


def test_strong_positive_series_passes_both_lenses():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.005, 750)        # Sharpe ~0.2/day, 3 years
    labels = _year_labels(250, [2021, 2022, 2023])
    res = validate_strategy_oos(returns, labels)
    assert res['psr'] > 0.95
    assert res['psr_pass'] is True
    assert res['fold_pass'] is True
    assert res['passed'] is True
    assert res['n_trials'] == 1


def test_positive_but_below_risk_free_fails():
    """Reliably-positive raw return that is BELOW the 2% risk-free hurdle:
    it would pass a benchmark-0 PSR, but the gate benchmarks risk-free, so the
    excess edge is negative and it must FAIL (regression: a benchmark-0 gate
    passes any upward drift, buy-and-hold included)."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.00003, 0.0005, 750)     # ~0.76%/yr, below 2% rf
    labels = _year_labels(250, [2021, 2022, 2023])
    res = validate_strategy_oos(returns, labels)
    assert res['psr_pass'] is False
    assert res['passed'] is False


def test_pure_noise_fails_on_overall_edge():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, 750)           # Sharpe ~0
    labels = _year_labels(250, [2021, 2022, 2023])
    res = validate_strategy_oos(returns, labels)
    assert res['psr_pass'] is False                # PSR ~0.5 < 0.95
    assert res['passed'] is False


def test_strong_edge_but_no_testable_period_fails_folds():
    # AND-logic: overall edge passes, but every sub-period is a single point
    # (< min_period_obs) so no fold is testable -> fold_pass False -> not passed.
    rng = np.random.default_rng(2)
    returns = rng.normal(0.001, 0.005, 750)
    labels = list(range(750))                      # 750 folds of size 1
    res = validate_strategy_oos(returns, labels)
    assert res['psr_pass'] is True
    assert res['fold_pass'] is False
    assert res['n_periods_tested'] == 0
    assert res['passed'] is False


def test_min_period_obs_drops_short_periods():
    rng = np.random.default_rng(3)
    returns = np.concatenate([
        rng.normal(0.001, 0.005, 250),             # 2021: full year
        rng.normal(0.001, 0.005, 5),               # 2022: only 5 obs
    ])
    labels = [2021] * 250 + [2022] * 5
    res = validate_strategy_oos(returns, labels, min_period_obs=20)
    assert res['fold_pvalues'][1] is None          # 2022 untestable, dropped
    assert res['n_periods_tested'] == 1


def test_single_period_when_no_labels():
    rng = np.random.default_rng(4)
    res = validate_strategy_oos(rng.normal(0.001, 0.005, 500))
    assert res['fold_labels'] == [None]
    assert res['n_periods_tested'] == 1


def test_mismatched_labels_raise():
    with pytest.raises(ValueError):
        validate_strategy_oos([0.01, 0.02], [2021])


def test_deterministic():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.001, 0.005, 500)
    labels = _year_labels(250, [2021, 2022])
    assert validate_strategy_oos(returns, labels) == \
        validate_strategy_oos(returns, labels)
