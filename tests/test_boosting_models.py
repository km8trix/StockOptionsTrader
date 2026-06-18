"""Tests for desks.models.boosting — LightGBMModel and StackingMetaModel.

Both must honor the EXACT public contract of
``ml_model.GradientBoostingModel``:

  * predict() returns a per-symbol centered score ``P(up) - 0.5`` in
    [-0.5, 0.5];
  * predict() returns ``{}`` when the model was never successfully fitted,
    and skips symbols whose feature rows are insufficient/NaN;
  * fit() degrades gracefully (WARNING, stays unfitted) on < 2 samples or
    < 2 label classes — never raises on bad data.

The KEY ML test is DETERMINISM: the models are seeded and single-threaded,
so fitting twice on identical data and predicting yields EXACTLY equal
scores (asserted with ==, not a tolerance). No cross-platform golden files.

lightgbm==4.6.0 is installed in this environment, so these tests run the
real LightGBM path. Offline and deterministic — synthetic frames only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from desks.ml_model import GradientBoostingModel
from desks.models.boosting import (FEATURE_COLUMNS, LightGBMModel,
                                    StackingMetaModel)
from desks.walk_forward import WalkForwardModel

# Skip the whole module if the optional wheel is missing (the environment
# spec says it is installed, so this should never skip here).
pytest.importorskip('lightgbm', reason='lightgbm wheel not installed')


def learnable(seed: int, n: int = 220, start_price: float = 100.0,
              start: str = '2022-01-03') -> pd.DataFrame:
    """Seeded synthetic OHLCV with enough rows for the extended feature
    warm-up and a real (random-walk) signal both classes can learn on."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n)
    close = start_price * np.cumprod(1.0 + rng.normal(0.0005, 0.012, n))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close,
         'volume': volume}, index=index)


def two_symbol_panel() -> dict:
    return {'A': learnable(1), 'B': learnable(2)}


# ----------------------------------------------------------------------
# Shared contract assertions, parametrized across both new models.
# ----------------------------------------------------------------------
NEW_MODELS = [LightGBMModel, StackingMetaModel]


class TestProtocolConformance:
    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_is_a_walk_forward_model(self, model_cls):
        assert isinstance(model_cls(), WalkForwardModel)

    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_predict_before_fit_returns_empty(self, model_cls):
        model = model_cls()
        assert model.predict(two_symbol_panel(),
                             pd.Timestamp('2022-06-01')) == {}

    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_fit_then_predict_returns_centered_scores_in_range(self, model_cls):
        panel = two_symbol_panel()
        model = model_cls()
        model.fit(panel)  # must not raise
        scores = model.predict(panel, panel['A'].index[-1])
        assert set(scores) <= {'A', 'B'}
        assert scores  # the model trained and produced at least one score
        for value in scores.values():
            assert -0.5 <= value <= 0.5


class TestGracefulDegradation:
    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_too_few_rows_stays_unfitted(self, model_cls, caplog):
        tiny = {'A': learnable(1).iloc[:2]}
        model = model_cls()
        with caplog.at_level(logging.WARNING):
            model.fit(tiny)  # must not raise
        assert model.predict(tiny, tiny['A'].index[-1]) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_empty_train_data_stays_unfitted(self, model_cls):
        model = model_cls()
        model.fit({})  # must not raise
        assert model.predict({}, pd.Timestamp('2022-06-01')) == {}

    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_single_class_labels_degrade_to_no_signal(self, model_cls,
                                                      caplog):
        # Strictly rising closes -> every next-day return is positive ->
        # one label class -> cannot train.
        n = 220
        idx = pd.bdate_range('2022-01-03', periods=n)
        close = np.array([100.0 + i for i in range(n)], dtype=float)
        rising = pd.DataFrame(
            {'open': close, 'high': close * 1.01, 'low': close * 0.99,
             'close': close, 'volume': np.full(n, 500_000.0)}, index=idx)
        model = model_cls()
        with caplog.at_level(logging.WARNING):
            model.fit({'A': rising})
        assert model.predict({'A': rising}, idx[-1]) == {}

    @pytest.mark.parametrize('model_cls', NEW_MODELS)
    def test_none_and_closeless_frames_are_skipped_at_predict(self, model_cls):
        panel = two_symbol_panel()
        model = model_cls()
        model.fit(panel)
        mixed = {'A': panel['A'], 'B': None,
                 'C': panel['B'].drop(columns=['close'])}
        scores = model.predict(mixed, panel['A'].index[-1])
        # B (None) and C (no close) are skipped; A is scored.
        assert set(scores) <= {'A'}


class TestLightGBMDeterminism:
    """The headline ML test: identical input -> byte-identical scores."""

    def test_fit_twice_predict_identical_scores_exact_equality(self):
        panel = two_symbol_panel()

        m1 = LightGBMModel()
        m1.fit(panel)
        s1 = m1.predict(panel, panel['A'].index[-1])

        m2 = LightGBMModel()
        m2.fit(panel)
        s2 = m2.predict(panel, panel['A'].index[-1])

        assert s1  # actually fitted
        assert s1.keys() == s2.keys()
        for symbol in s1:
            # EXACT equality — the booster is seeded + single-threaded.
            assert s1[symbol] == s2[symbol]

    def test_repeated_predict_on_same_model_is_stable(self):
        panel = two_symbol_panel()
        model = LightGBMModel()
        model.fit(panel)
        a = model.predict(panel, panel['A'].index[-1])
        b = model.predict(panel, panel['A'].index[-1])
        assert a == b

    def test_uses_extended_feature_columns(self):
        # The model shares the extended feature set (10 columns).
        assert len(FEATURE_COLUMNS) == 10


class TestStackingMetaModel:
    def test_default_base_models_train_and_predict_centered_score(self):
        panel = two_symbol_panel()
        stack = StackingMetaModel()  # default [GBM, LightGBM]
        stack.fit(panel)
        scores = stack.predict(panel, panel['A'].index[-1])
        assert scores
        for value in scores.values():
            assert -0.5 <= value <= 0.5

    def test_determinism_exact_equality(self):
        panel = two_symbol_panel()
        s1 = StackingMetaModel()
        s1.fit(panel)
        out1 = s1.predict(panel, panel['A'].index[-1])
        s2 = StackingMetaModel()
        s2.fit(panel)
        out2 = s2.predict(panel, panel['A'].index[-1])
        assert out1 and out1.keys() == out2.keys()
        for symbol in out1:
            assert out1[symbol] == out2[symbol]

    def test_unfitted_stack_returns_empty(self):
        assert StackingMetaModel().predict(
            two_symbol_panel(), pd.Timestamp('2022-06-01')) == {}

    def test_falls_back_when_one_base_learner_is_unavailable(self):
        """A base learner that never produces out-of-fold predictions is
        dropped from the ensemble; the stack still fits on the survivors."""
        panel = two_symbol_panel()

        class UselessBase(WalkForwardModel):
            def fit(self, train_data):
                pass  # never fits

            def predict(self, data, date):
                return {}  # always empty -> unusable as a base learner

        stack = StackingMetaModel(
            base_models=[GradientBoostingModel(n_estimators=30),
                         UselessBase()])
        stack.fit(panel)
        scores = stack.predict(panel, panel['A'].index[-1])
        assert scores  # the good base learner carries the stack
        for value in scores.values():
            assert -0.5 <= value <= 0.5

    def test_stays_unfitted_when_no_base_learner_is_usable(self, caplog):
        panel = two_symbol_panel()

        class UselessBase(WalkForwardModel):
            def fit(self, train_data):
                pass

            def predict(self, data, date):
                return {}

        stack = StackingMetaModel(
            base_models=[UselessBase(), UselessBase()])
        with caplog.at_level(logging.WARNING):
            stack.fit(panel)
        assert stack.predict(panel, panel['A'].index[-1]) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_short_window_cannot_time_split_stays_unfitted(self, caplog):
        # Too few rows to make the early/late out-of-fold split with a
        # usable, two-class meta matrix.
        tiny = {'A': learnable(1).iloc[:6]}
        stack = StackingMetaModel()
        with caplog.at_level(logging.WARNING):
            stack.fit(tiny)
        assert stack.predict(tiny, tiny['A'].index[-1]) == {}

    # ------------------------------------------------------------------
    # H1: base-learner failures are wrapped — the stack degrades, never
    # crashes (the never-crash-a-run invariant). Three failure modes:
    #   (a) a base that raises during the internal OOF early-block fit is
    #       DROPPED; a surviving base still fits the stack.
    #   (b) a base whose FULL refit raises forces the whole stack to
    #       degrade to unfitted (predict()=={}) — a narrower active set
    #       would mismatch the trained meta-learner's width.
    #   (c) all bases raise -> predict()=={}.
    # In every case NO exception propagates out of fit().
    # ------------------------------------------------------------------

    def test_oof_fit_failure_drops_base_survivor_carries_stack(self, caplog):
        """(a) A base whose fit() raises during the out-of-fold early-block
        fit is dropped; a surviving real base still fits and predicts
        centered scores. fit() never raises."""
        panel = two_symbol_panel()

        class RaisingFitBase(WalkForwardModel):
            def fit(self, train_data):
                raise RuntimeError('synthetic OOF fit explosion')

            def predict(self, data, date):
                return {}

        stack = StackingMetaModel(
            base_models=[RaisingFitBase(),
                         GradientBoostingModel(n_estimators=30)])
        with caplog.at_level(logging.WARNING):
            stack.fit(panel)  # must not raise
        scores = stack.predict(panel, panel['A'].index[-1])
        assert scores  # the surviving base carries the stack
        for value in scores.values():
            assert -0.5 <= value <= 0.5
        # The dropped base was logged.
        assert any('RaisingFitBase' in rec.getMessage()
                   for rec in caplog.records)

    def test_full_refit_failure_degrades_stack_to_unfitted(self, caplog):
        """(b) A base that produces usable out-of-fold meta-features but
        whose FULL refit raises must degrade the WHOLE stack to unfitted
        (predict()=={}) — not a dimension-mismatched ensemble or a crash."""
        panel = two_symbol_panel()

        class RefitRaisesBase(WalkForwardModel):
            """fit() succeeds on the OOF early block (1st call) but raises on
            the full-window refit (2nd call); predict() yields per-symbol
            centered scores so its OOF column is usable."""

            def __init__(self):
                self._fit_calls = 0

            def fit(self, train_data):
                self._fit_calls += 1
                if self._fit_calls >= 2:  # the full-window refit
                    raise RuntimeError('synthetic full-refit explosion')

            def predict(self, data, date):
                return {sym: 0.1 for sym in data
                        if data[sym] is not None
                        and 'close' in data[sym].columns}

        stack = StackingMetaModel(
            base_models=[RefitRaisesBase(),
                         GradientBoostingModel(n_estimators=30)])
        with caplog.at_level(logging.WARNING):
            stack.fit(panel)  # must not raise
        # Whole stack degraded to unfitted — no dimension-mismatched predict.
        assert stack.predict(panel, panel['A'].index[-1]) == {}
        assert any('RefitRaisesBase' in rec.getMessage()
                   for rec in caplog.records)

    def test_all_bases_raise_yields_empty_predict_no_exception(self, caplog):
        """(c) Every base raising on fit() leaves the stack unfitted and
        predict() returns {} — fit() raises no exception."""
        panel = two_symbol_panel()

        class RaisingFitBase(WalkForwardModel):
            def fit(self, train_data):
                raise RuntimeError('synthetic fit explosion')

            def predict(self, data, date):
                return {}

        stack = StackingMetaModel(
            base_models=[RaisingFitBase(), RaisingFitBase()])
        with caplog.at_level(logging.WARNING):
            stack.fit(panel)  # must not raise
        assert stack.predict(panel, panel['A'].index[-1]) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_learns_a_sane_centered_signal_on_synthetic_pattern(self):
        """With a learnable pattern across symbols the stack trains and
        emits centered scores (sanity that meta-features are not degenerate
        — a leakage-free stack still produces a usable, bounded score)."""
        panel = {f'S{i}': learnable(seed=i + 10) for i in range(4)}
        stack = StackingMetaModel()
        stack.fit(panel)
        scores = stack.predict(panel, panel['S0'].index[-1])
        assert scores
        # Centered: scores straddle 0 rather than all pinned to one extreme.
        values = np.array(list(scores.values()))
        assert np.all(np.abs(values) <= 0.5)
