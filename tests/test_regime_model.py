"""Tests for desks.regime.RegimeHMMModel.

Synthetic 3-regime market: a mean-reverting AR(-0.5) segment, a trending
drift segment, and a high-vol segment, all seeded. The deterministic
labeling rule (highest within-state std -> high_vol FIRST; then the most
negative within-state AR(1) clearing the negative margin ->
mean_reverting, with the |mean| tiebreak below the margin; last ->
trending) must assign the expected labels — identically across repeated
fresh fits, and across fresh data seeds (TestMultiSeedRobustness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desks.regime import REGIME_LABELS, RegimeHMMModel

SEGMENT_DAYS = 130


def three_regime_frame(seed: int = 42) -> pd.DataFrame:
    """One market symbol whose returns switch regime every SEGMENT_DAYS:
    mean-reverting AR(-0.5), then trending drift, then high-vol noise."""
    rng = np.random.default_rng(seed)
    returns = []

    prev = 0.0
    for _ in range(SEGMENT_DAYS):          # mean-reverting: AR(1) = -0.5
        prev = -0.5 * prev + rng.normal(0.0, 0.01)
        returns.append(prev)
    for _ in range(SEGMENT_DAYS):          # trending: drift + AR(1) = +0.3
        prev = 0.004 + 0.3 * prev + rng.normal(0.0, 0.002)
        returns.append(prev)
    for _ in range(SEGMENT_DAYS):          # high-vol: 4x the MR noise
        prev = rng.normal(0.0, 0.04)
        returns.append(prev)

    n_days = 3 * SEGMENT_DAYS
    close = 100.0 * np.cumprod(1.0 + np.array(returns))
    index = pd.bdate_range('2020-01-02', periods=n_days)
    # Volume jitters so the volume_ratio feature is not constant.
    volume = rng.integers(450_000, 550_000, n_days).astype(float)
    return pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': volume,
    }, index=index)


def segment_end(frame: pd.DataFrame, segment: int) -> pd.Timestamp:
    """Last date of segment 0, 1 or 2."""
    return frame.index[(segment + 1) * SEGMENT_DAYS - 1]


@pytest.fixture(scope='module')
def market() -> pd.DataFrame:
    return three_regime_frame()


@pytest.fixture(scope='module')
def fitted(market) -> RegimeHMMModel:
    model = RegimeHMMModel()
    model.fit({'MKT': market})
    return model


class TestLabelingRule:
    def test_each_segment_end_is_labeled_correctly(self, market, fitted):
        expected = ['mean_reverting', 'trending', 'high_vol']
        for segment, label in enumerate(expected):
            date = segment_end(market, segment)
            window = market[market.index <= date]
            result = fitted.predict({'MKT': window}, date)
            assert result['state'] == label, (
                f"segment {segment} expected {label}, got {result}")
            # The posterior backs the labeled state decisively.
            assert result['probs'][label] > 0.5

    def test_labels_are_deterministic_across_three_fresh_fits(self, market):
        date = segment_end(market, 2)
        results = []
        for _ in range(3):
            model = RegimeHMMModel()
            model.fit({'MKT': market})
            results.append(model.predict({'MKT': market}, date))
        assert results[0] == results[1] == results[2]
        assert results[0]['state'] == 'high_vol'

    def test_all_three_labels_are_assigned(self, fitted):
        assert sorted(fitted._state_labels.values()) == sorted(REGIME_LABELS)


class TestMultiSeedRobustness:
    def test_segment_labeling_holds_across_fresh_data_seeds(self):
        """Regression: the labeling rule must not be tuned to the
        canonical seed. Under the original rule ('mean_reverting'
        assigned FIRST by most-negative AR(1), same-state-pairs-only
        AR(1), no margin) only 5/10 fresh seeds of the canonical
        three-regime generator labeled their segments correctly — seed 2
        even labeled the high-vol segment 'mean_reverting', exactly the
        confusion that would activate the mean-reversion book during
        high volatility. The hardened rule (high_vol first by std,
        AR(1) margin, alternating-split-robust estimator, standardized
        log-vol design matrix, seeded EM restarts) must label at least
        8/10 seeds fully correctly; each fit stays deterministic via
        random_state=42."""
        expected = ['mean_reverting', 'trending', 'high_vol']
        failures = []
        for seed in range(10):
            frame = three_regime_frame(seed)
            model = RegimeHMMModel()
            model.fit({'MKT': frame})
            got = []
            for segment in range(3):
                date = segment_end(frame, segment)
                window = frame[frame.index <= date]
                result = model.predict({'MKT': window}, date)
                got.append(result.get('state'))
            if got != expected:
                failures.append((seed, got))
        assert len(failures) <= 2, (
            f"segment labeling wrong on {len(failures)}/10 seeds: "
            f"{failures}")


class TestPredictShape:
    def test_predict_shape_matches_contract(self, market, fitted):
        result = fitted.predict({'MKT': market}, market.index[-1])
        assert set(result) == {'state', 'probs'}
        assert result['state'] in REGIME_LABELS
        assert set(result['probs']) == set(REGIME_LABELS)
        for prob in result['probs'].values():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0
        assert sum(result['probs'].values()) == pytest.approx(1.0)
        # The reported state is the argmax of the posterior.
        assert result['state'] == max(result['probs'],
                                      key=result['probs'].get)

    def test_insufficient_data_returns_empty(self, market, fitted):
        # 10 rows cannot survive the 20-day rolling warm-up.
        short = market.iloc[:10]
        assert fitted.predict({'MKT': short}, short.index[-1]) == {}

    def test_unfitted_model_returns_empty(self, market):
        model = RegimeHMMModel()
        assert model.predict({'MKT': market}, market.index[-1]) == {}


class TestDegenerateData:
    def test_constant_data_does_not_crash_and_stays_unfitted(self):
        index = pd.bdate_range('2020-01-02', periods=200)
        constant = pd.DataFrame({
            'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0,
            'volume': 500_000.0,
        }, index=index)

        model = RegimeHMMModel()
        model.fit({'FLAT': constant})  # must not raise

        assert model._fitted is False
        assert model.predict({'FLAT': constant}, index[-1]) == {}

    def test_too_few_rows_stays_unfitted(self, market):
        model = RegimeHMMModel()
        model.fit({'MKT': market.iloc[:40]})  # ~20 feature rows < MIN_FIT_ROWS
        assert model._fitted is False

    def test_failed_refit_retains_the_previous_fit(self, market):
        # A transient failure at a REFIT must not kill the regime engine:
        # the previously fitted model keeps answering predict().
        index = pd.bdate_range('2020-01-02', periods=200)
        constant = pd.DataFrame({
            'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0,
            'volume': 500_000.0,
        }, index=index)
        model = RegimeHMMModel()
        model.fit({'MKT': market})
        assert model._fitted is True
        before = model.predict({'MKT': market}, market.index[-1])

        model.fit({'FLAT': constant})  # degenerate refit: abandoned

        assert model._fitted is True
        assert model.predict({'MKT': market}, market.index[-1]) == before
