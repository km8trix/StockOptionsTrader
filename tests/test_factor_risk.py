"""Tests for desks.factor_risk.FactorRiskModel — the Citadel desk's BOOK
factor risk model (Phase E).

Where desks.models.factor SCORES symbols cross-sectionally (a ranking signal
a pod trades ON), FactorRiskModel MEASURES the aggregate book's net loading
on those same four factors (momentum / reversal / low_vol / risk_adj_mom), so
the central risk book can keep the pods factor-neutral. The factor set and
the per-date cross-sectional z-score are REUSED verbatim from
desks.models.factor (one factor definition in the codebase).

These tests pin:
  * loadings are the standardized cross-sectional z-scores (one per factor,
    per symbol) of the AQR pipeline, as of `date`;
  * NO LEAKAGE — every loading at date D depends on rows index<=D only, so
    appending FUTURE bars and re-slicing yields byte-identical loadings at D
    (the defensive re-slice is what guarantees this on an unsliced panel);
  * DETERMINISM — two identical calls return byte-identical loadings AND
    exposures (pure pandas/numpy reductions, no RNG);
  * book_exposure math matches a hand example (signed dollar-weighted sum of
    loadings, normalized by desk capital), the candidate-book helper is the
    held book with one symbol overridden (non-mutating);
  * graceful DEGRADE — thin/empty/None data and a degenerate (single-symbol)
    cross-section yield {} or all-zero loadings; desk_capital<=0 and absent
    symbols/factors contribute 0 — never a crash, never a false reading.

Offline, seeded, deterministic — synthetic OHLCV frames only, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desks.factor_risk import FactorRiskModel
from desks.models.factor import FACTOR_COLUMNS


# ----------------------------------------------------------------------
# Synthetic OHLCV fixtures (seeded, deterministic) — same shape as the
# factor-model tests so the two modules share one fixture style.
# ----------------------------------------------------------------------
def frame(seed: int, drift: float = 0.0005, n: int = 160,
          start: str = '2022-06-01', vol: float = 0.012) -> pd.DataFrame:
    """Seeded synthetic OHLCV with a bar on every business day; a per-symbol
    drift gives the cross-section a real ranking to separate on."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        'open': close, 'high': close * 1.002, 'low': close * 0.998,
        'close': close, 'volume': np.full(n, 500_000.0),
    }, index=index)


def panel(n_symbols: int = 6, n: int = 160, drift_spread: float = 0.0010):
    """A factor-tilted universe: symbol i carries drift drift_spread*(i-mid),
    so the names genuinely separate by momentum (a real cross-section)."""
    mid = n_symbols / 2.0
    return {f'S{i}': frame(50 + i, drift_spread * (i - mid), n=n)
            for i in range(n_symbols)}


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------
class TestConstruction:
    def test_default_uses_full_four_factor_set(self):
        model = FactorRiskModel()
        assert tuple(model.factors) == tuple(FACTOR_COLUMNS)

    def test_factor_subset_is_respected(self):
        model = FactorRiskModel(factors=['momentum'])
        assert tuple(model.factors) == ('momentum',)

    def test_unknown_factor_rejected(self):
        with pytest.raises(ValueError, match='Unknown factor'):
            FactorRiskModel(factors=['not_a_factor'])

    def test_empty_factor_set_rejected(self):
        with pytest.raises(ValueError, match='at least one'):
            FactorRiskModel(factors=[])


# ----------------------------------------------------------------------
# loadings: shape + reuse of the AQR factor set
# ----------------------------------------------------------------------
class TestLoadingsShape:
    def test_loadings_return_each_symbol_each_factor(self):
        data = panel()
        date = data['S0'].index[-1]
        loadings = FactorRiskModel().loadings(data, date)
        assert set(loadings) <= set(data)
        assert loadings  # at least one symbol formed factors
        for symbol, per_factor in loadings.items():
            assert set(per_factor) == set(FACTOR_COLUMNS)
            for value in per_factor.values():
                assert isinstance(value, float)
                assert np.isfinite(value)

    def test_loadings_are_the_cross_sectional_zscore(self):
        # Each factor's loadings across symbols on the as-of date are the
        # cross-sectional z-score: they (very nearly) sum to 0 across the
        # formed symbols — the standardization the factor pipeline uses.
        data = panel()
        date = data['S0'].index[-1]
        loadings = FactorRiskModel().loadings(data, date)
        for factor in FACTOR_COLUMNS:
            column = [per[factor] for per in loadings.values()]
            assert sum(column) == pytest.approx(0.0, abs=1e-9)

    def test_subset_model_returns_only_requested_factor(self):
        data = panel()
        date = data['S0'].index[-1]
        loadings = FactorRiskModel(factors=['momentum']).loadings(data, date)
        for per_factor in loadings.values():
            assert set(per_factor) == {'momentum'}


# ----------------------------------------------------------------------
# NO LEAKAGE — the headline guarantee
# ----------------------------------------------------------------------
class TestNoLeakage:
    def test_future_bars_do_not_change_past_loadings(self):
        # Append FUTURE rows after `date`, re-slice via the model: the
        # loadings AT `date` must be byte-identical (every loading is a
        # function of rows index<=date only).
        data = panel()
        date = data['S0'].index[-1]
        model = FactorRiskModel()
        loadings_at_d = model.loadings(data, date)

        extended = {}
        for i, (symbol, df) in enumerate(data.items()):
            future = frame(900 + i, drift=0.003, n=30,
                           start='2023-02-01')  # strictly after `date`
            extended[symbol] = pd.concat([df, future])
        loadings_full_at_d = model.loadings(extended, date)

        assert loadings_full_at_d == loadings_at_d

    def test_reslice_matches_pre_sliced_panel(self):
        # loadings(full_panel, D) == loadings(panel_already_sliced_to_D, D):
        # the defensive re-slice is idempotent on a pre-sliced frame.
        data = panel()
        mid_date = data['S0'].index[120]
        model = FactorRiskModel()
        full = model.loadings(data, mid_date)
        sliced = {s: df[df.index <= mid_date] for s, df in data.items()}
        assert model.loadings(sliced, mid_date) == full


# ----------------------------------------------------------------------
# DETERMINISM
# ----------------------------------------------------------------------
class TestDeterminism:
    def test_two_identical_loadings_calls_are_byte_identical(self):
        data = panel()
        date = data['S0'].index[-1]
        model = FactorRiskModel()
        assert model.loadings(data, date) == model.loadings(data, date)

    def test_two_identical_book_exposures_are_byte_identical(self):
        data = panel()
        date = data['S0'].index[-1]
        model = FactorRiskModel()
        loadings = model.loadings(data, date)
        per_symbol = {s: (10_000.0 if i % 2 == 0 else -7_500.0)
                      for i, s in enumerate(sorted(loadings))}
        first = model.book_exposure(per_symbol, loadings, 100_000.0)
        second = model.book_exposure(per_symbol, loadings, 100_000.0)
        assert first == second


# ----------------------------------------------------------------------
# book_exposure — hand example + neutrality + normalization
# ----------------------------------------------------------------------
class TestBookExposure:
    def test_hand_example_single_factor(self):
        # net momentum = (30000*2.0 + (-10000)*(-1.0)) / 100000
        #              = (60000 + 10000) / 100000 = 0.70.
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}, 'BBB': {'momentum': -1.0}}
        per_symbol = {'AAA': 30_000.0, 'BBB': -10_000.0}
        exposure = model.book_exposure(per_symbol, loadings, 100_000.0)
        assert exposure == {'momentum': pytest.approx(0.70)}

    def test_always_returns_every_configured_factor(self):
        model = FactorRiskModel()
        exposure = model.book_exposure(
            {'AAA': 10_000.0}, {'AAA': {'momentum': 1.0}}, 100_000.0)
        assert set(exposure) == set(FACTOR_COLUMNS)
        # A factor absent from the symbol's loading dict reads 0 on that axis.
        assert exposure['reversal'] == 0.0

    def test_market_neutral_dollar_book_is_factor_neutral(self):
        # Equal-and-opposite dollar weights on equal-and-opposite loadings
        # net to zero exposure (the neutrality the band enforces).
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 1.5}, 'BBB': {'momentum': 1.5}}
        per_symbol = {'AAA': 20_000.0, 'BBB': -20_000.0}
        exposure = model.book_exposure(per_symbol, loadings, 100_000.0)
        assert exposure['momentum'] == pytest.approx(0.0)

    def test_absent_symbol_contributes_zero(self):
        # A symbol with no loading entry is neutral (0 on every factor).
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}}
        with_absent = model.book_exposure(
            {'AAA': 30_000.0, 'ZZZ': 99_999.0}, loadings, 100_000.0)
        without = model.book_exposure({'AAA': 30_000.0}, loadings, 100_000.0)
        assert with_absent == without == {'momentum': pytest.approx(0.6)}

    def test_zero_value_symbol_skipped(self):
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}, 'BBB': {'momentum': 5.0}}
        exposure = model.book_exposure(
            {'AAA': 30_000.0, 'BBB': 0.0}, loadings, 100_000.0)
        assert exposure == {'momentum': pytest.approx(0.6)}

    def test_normalized_by_desk_capital(self):
        # Halving desk capital doubles the fraction.
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 1.0}}
        big = model.book_exposure({'AAA': 10_000.0}, loadings, 100_000.0)
        small = model.book_exposure({'AAA': 10_000.0}, loadings, 50_000.0)
        assert small['momentum'] == pytest.approx(2.0 * big['momentum'])


# ----------------------------------------------------------------------
# candidate_exposure — the incremental helper the risk-book limit uses
# ----------------------------------------------------------------------
class TestCandidateExposure:
    def test_overrides_one_symbol_on_held_book(self):
        # Held: AAA +30000, BBB -10000. Override BBB to -40000:
        # net momentum = (30000*2 + (-40000)*(-1)) / 100000
        #              = (60000 + 40000) / 100000 = 1.0.
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}, 'BBB': {'momentum': -1.0}}
        held = {'AAA': 30_000.0, 'BBB': -10_000.0}
        candidate = model.candidate_exposure(
            held, 'BBB', -40_000.0, loadings, 100_000.0)
        assert candidate == {'momentum': pytest.approx(1.0)}

    def test_is_non_mutating(self):
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}, 'BBB': {'momentum': -1.0}}
        held = {'AAA': 30_000.0, 'BBB': -10_000.0}
        snapshot = dict(held)
        model.candidate_exposure(held, 'BBB', -40_000.0, loadings, 100_000.0)
        assert held == snapshot

    def test_new_candidate_symbol_added_to_book(self):
        # A candidate symbol not in the held book is simply included.
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}, 'CCC': {'momentum': 1.0}}
        candidate = model.candidate_exposure(
            {'AAA': 30_000.0}, 'CCC', 20_000.0, loadings, 100_000.0)
        # (30000*2 + 20000*1) / 100000 = 0.8.
        assert candidate == {'momentum': pytest.approx(0.8)}


# ----------------------------------------------------------------------
# Graceful degrade — never crash, never false-block
# ----------------------------------------------------------------------
class TestGracefulDegrade:
    def test_empty_data_yields_empty_loadings(self):
        assert FactorRiskModel().loadings({}, '2022-06-01') == {}

    def test_none_frame_skipped(self):
        assert FactorRiskModel().loadings({'X': None}, '2022-06-01') == {}

    def test_empty_frame_skipped(self):
        assert FactorRiskModel().loadings(
            {'X': pd.DataFrame()}, '2022-06-01') == {}

    def test_close_less_frame_skipped(self):
        bad = pd.DataFrame({'volume': [1.0, 2.0]},
                           index=pd.bdate_range('2022-06-01', periods=2))
        assert FactorRiskModel().loadings({'X': bad}, '2022-06-02') == {}

    def test_too_short_history_omitted(self):
        # Below the factor model's minimum-rows threshold -> no factors.
        short = frame(1, n=3)
        assert FactorRiskModel().loadings(
            {'X': short}, short.index[-1]) == {}

    def test_single_symbol_cross_section_collapses_to_zero(self):
        # A degenerate (single-symbol) cross-section has no spread to
        # standardize -> z-scores collapse to 0.0 (a valid neutral reading,
        # never a false exposure), and the symbol is still present.
        data = {'AAA': frame(1, drift=0.001)}
        date = data['AAA'].index[-1]
        loadings = FactorRiskModel().loadings(data, date)
        assert set(loadings) == {'AAA'}
        assert loadings['AAA'] == {f: 0.0 for f in FACTOR_COLUMNS}

    def test_date_before_any_data_yields_empty(self):
        data = panel()
        early = pd.Timestamp('2000-01-01')  # before every frame
        assert FactorRiskModel().loadings(data, early) == {}

    def test_book_exposure_desk_capital_non_positive_is_all_zero(self):
        model = FactorRiskModel(factors=['momentum'])
        loadings = {'AAA': {'momentum': 2.0}}
        per_symbol = {'AAA': 30_000.0}
        assert model.book_exposure(per_symbol, loadings, 0.0) \
            == {'momentum': 0.0}
        assert model.book_exposure(per_symbol, loadings, -5.0) \
            == {'momentum': 0.0}

    def test_book_exposure_empty_book_is_all_zero(self):
        model = FactorRiskModel()
        assert model.book_exposure({}, {}, 100_000.0) \
            == {f: 0.0 for f in FACTOR_COLUMNS}

    def test_book_exposure_no_loadings_is_neutral(self):
        # A held book with no loadings at all -> every symbol neutral -> 0.
        model = FactorRiskModel(factors=['momentum'])
        exposure = model.book_exposure(
            {'AAA': 30_000.0, 'BBB': -10_000.0}, {}, 100_000.0)
        assert exposure == {'momentum': 0.0}
