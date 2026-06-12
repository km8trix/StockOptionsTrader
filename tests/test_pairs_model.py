"""Tests for desks.pairs.PairsCointegrationModel.

Two cointegrated series are built as log(A) = beta * log(B) + c +
stationary AR(0.5) noise over a shared random walk driver, plus one
independent walk: fit must find exactly the cointegrated pair, recover
beta near truth, and produce a z-score the test verifies by hand. All
seeded, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desks.pairs import PairsCointegrationModel

N_DAYS = 300
BETA_TRUE = 1.5


def _frame(log_close: np.ndarray, volume: float,
           index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.exp(log_close)
    return pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': np.full(len(close), volume),
    }, index=index)


def cointegrated_universe(seed: int = 42):
    """A/B cointegrated (beta=BETA_TRUE), C an independent walk.
    Volumes rank A > B > C so the liquidity ordering is deterministic."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2022-01-03', periods=N_DAYS)

    log_b = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, N_DAYS))
    noise = np.empty(N_DAYS)
    prev = 0.0
    for i in range(N_DAYS):                  # stationary AR(0.5) spread
        prev = 0.5 * prev + rng.normal(0.0, 0.01)
        noise[i] = prev
    log_a = BETA_TRUE * log_b + 0.5 + noise
    log_c = np.log(80.0) + np.cumsum(rng.normal(0.0, 0.01, N_DAYS))

    return {
        'AAA': _frame(log_a, 900_000.0, index),
        'BBB': _frame(log_b, 800_000.0, index),
        'CCC': _frame(log_c, 700_000.0, index),
    }


@pytest.fixture(scope='module')
def universe():
    return cointegrated_universe()


@pytest.fixture(scope='module')
def fitted(universe) -> PairsCointegrationModel:
    model = PairsCointegrationModel()
    model.fit(universe)
    return model


class TestFit:
    def test_finds_exactly_the_cointegrated_pair(self, fitted):
        assert len(fitted.pairs) == 1
        pair = fitted.pairs[0]
        # Liquidity ordering A > B makes the pair (AAA, BBB), in order.
        assert (pair['a'], pair['b']) == ('AAA', 'BBB')
        assert pair['p_value'] < 0.05

    def test_beta_is_recovered_near_truth(self, fitted):
        assert fitted.pairs[0]['beta'] == pytest.approx(BETA_TRUE, abs=0.1)

    def test_spread_parameters_match_hand_computation(self, universe, fitted):
        pair = fitted.pairs[0]
        log_a = np.log(universe['AAA']['close'].to_numpy())
        log_b = np.log(universe['BBB']['close'].to_numpy())
        spread = log_a - pair['beta'] * log_b
        assert pair['spread_mean'] == pytest.approx(float(np.mean(spread)))
        # Population std (ddof=0) by contract.
        assert pair['spread_std'] == pytest.approx(float(np.std(spread)))

    def test_liquidity_cap_limits_the_scan(self, universe):
        # With only the 2 most liquid symbols (AAA, BBB by volume), the
        # pair is still found; cap at 1 symbol and no scan is possible.
        capped = PairsCointegrationModel(max_symbols=2)
        capped.fit(universe)
        assert [(p['a'], p['b']) for p in capped.pairs] == [('AAA', 'BBB')]

        starved = PairsCointegrationModel(max_symbols=1)
        starved.fit(universe)
        assert starved.pairs == []


class TestPredict:
    def test_z_score_hand_verified_at_a_known_date(self, universe, fitted):
        date = universe['AAA'].index[-1]
        result = fitted.predict(universe, date)

        assert set(result) == {'pairs'}
        assert len(result['pairs']) == 1
        quote = result['pairs'][0]
        assert set(quote) == {'a', 'b', 'beta', 'z'}

        pair = fitted.pairs[0]
        spread_today = (np.log(float(universe['AAA']['close'].iloc[-1]))
                        - pair['beta']
                        * np.log(float(universe['BBB']['close'].iloc[-1])))
        expected_z = (spread_today - pair['spread_mean']) / pair['spread_std']
        assert quote['z'] == pytest.approx(expected_z)
        assert quote['beta'] == pytest.approx(pair['beta'])

    def test_missing_leg_is_skipped(self, universe, fitted):
        date = universe['AAA'].index[-1]
        without_b = {sym: df for sym, df in universe.items() if sym != 'BBB'}
        assert fitted.predict(without_b, date) == {'pairs': []}

    def test_stale_leg_is_skipped(self, universe, fitted):
        # BBB's latest bar lags AAA's by one day: the legs cannot be
        # priced on the same bar, so the pair is skipped for the day.
        stale = dict(universe)
        stale['BBB'] = universe['BBB'].iloc[:-1]
        date = universe['AAA'].index[-1]
        assert fitted.predict(stale, date) == {'pairs': []}

    def test_unfitted_model_returns_empty_pairs(self, universe):
        model = PairsCointegrationModel()
        assert model.predict(universe, universe['AAA'].index[-1]) == \
            {'pairs': []}


class TestNumericalGuards:
    def test_zero_spread_std_pair_is_skipped(self):
        # Y is an EXACT multiple of X in log space: the spread is
        # constant, so the pair must be dropped (never a div-by-zero z).
        rng = np.random.default_rng(7)
        index = pd.bdate_range('2022-01-03', periods=N_DAYS)
        log_x = np.log(40.0) + np.cumsum(rng.normal(0.0, 0.01, N_DAYS))
        data = {
            'XXX': _frame(2.0 * log_x, 900_000.0, index),
            'YYY': _frame(log_x, 800_000.0, index),
        }
        model = PairsCointegrationModel()
        model.fit(data)  # must not raise
        assert model.pairs == []

    def test_negative_beta_pair_is_skipped(self):
        # log(A) = -1.0 * log(B) + noise: cointegrated but inversely
        # hedged — skipped by the beta <= 0 guard.
        rng = np.random.default_rng(11)
        index = pd.bdate_range('2022-01-03', periods=N_DAYS)
        log_b = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, N_DAYS))
        noise = np.empty(N_DAYS)
        prev = 0.0
        for i in range(N_DAYS):
            prev = 0.5 * prev + rng.normal(0.0, 0.01)
            noise[i] = prev
        log_a = -1.0 * log_b + 9.0 + noise
        data = {
            'AAA': _frame(log_a, 900_000.0, index),
            'BBB': _frame(log_b, 800_000.0, index),
        }
        model = PairsCointegrationModel()
        model.fit(data)
        assert model.pairs == []

    def test_short_overlap_is_skipped(self, universe):
        model = PairsCointegrationModel(min_overlap=60)
        short = {sym: df.iloc[:40] for sym, df in universe.items()}
        model.fit(short)
        assert model.pairs == []
