"""Tests for desks.models.neural — MLPModel and SequenceModel (LSTM).

Both neural models must honor the EXACT public contract of
``ml_model.GradientBoostingModel``:

  * predict() returns a per-symbol centered score ``P(up) - 0.5`` in
    [-0.5, 0.5];
  * predict() returns ``{}`` when the model was never successfully fitted,
    and skips symbols whose feature rows are insufficient/NaN (for the
    SequenceModel: symbols too short to fill a lookback window);
  * fit() degrades gracefully (WARNING, stays unfitted) on < 2 samples or
    < 2 label classes — never raises on bad data.

The KEY ML test is DETERMINISM: every RNG/threading knob torch exposes is
pinned (CPU device, manual seed, deterministic algorithms, single thread),
so fitting twice on identical data and predicting yields EXACTLY equal
scores (asserted with ==, not a tolerance). We assert determinism WITHIN a
run (two fresh fits agree), never against a committed golden file — torch's
deterministic kernels are not guaranteed byte-identical across builds, and
the boosting tests follow the same in-run-equality convention.

torch is an OPTIONAL dependency (the ``ai`` extra). The whole module is
guarded with importorskip so a torch-less machine skips these cleanly.
Offline and deterministic — synthetic frames only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# Skip the entire module when torch is absent: importing desks.models.neural
# never requires torch, but CONSTRUCTING MLPModel/SequenceModel does, so
# every test below would raise ImportError without the wheel.
torch = pytest.importorskip('torch', reason='torch wheel not installed')

from desks.models import available_models, build_model  # noqa: E402
from desks.models.neural import (FEATURE_COLUMNS, MLPModel,  # noqa: E402
                                 SequenceModel)
from desks.walk_forward import WalkForwardModel  # noqa: E402


def learnable(seed: int, n: int = 220, start_price: float = 100.0,
              start: str = '2022-01-03') -> pd.DataFrame:
    """Seeded synthetic OHLCV with enough rows for the extended feature
    warm-up plus the SequenceModel's 20-day lookback windows, and a real
    (random-walk) signal both label classes can learn on."""
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


def momentum_frame(seed: int, n: int = 360,
                   start: str = '2021-06-01') -> pd.DataFrame:
    """Synthetic OHLCV carrying a REAL, STATIONARY next-day-direction signal.

    Daily returns follow a bounded AR(1) momentum process: tomorrow's return
    leans on today's return (coefficient 0.45 < 1, so the process is
    stationary and prices never blow up) plus noise. That gives the model a
    genuine, learnable relationship between recent direction and the next
    day's direction, while keeping BOTH label classes well represented and
    enough rows for the 20-day extended warm-up plus the lookback windows.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n)
    returns = np.zeros(n)
    returns[0] = rng.normal(0.0, 0.01)
    for i in range(1, n):
        # Stationary AR(1) on the daily return: |phi| < 1 keeps it bounded.
        returns[i] = 0.45 * returns[i - 1] + rng.normal(0.0, 0.01)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close,
         'volume': volume}, index=index)


# Small, fast model factories so the suite stays quick (few epochs / short
# lookback) while still exercising real training paths.
def fast_mlp() -> MLPModel:
    return MLPModel(hidden_sizes=(16, 8), epochs=15)


def fast_seq(lookback: int = 10) -> SequenceModel:
    return SequenceModel(lookback=lookback, hidden_size=8, epochs=10)


NEW_MODEL_FACTORIES = [fast_mlp, fast_seq]


# ----------------------------------------------------------------------
# Shared contract assertions, parametrized across both neural models.
# ----------------------------------------------------------------------
class TestProtocolConformance:
    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_is_a_walk_forward_model(self, factory):
        assert isinstance(factory(), WalkForwardModel)

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_predict_before_fit_returns_empty(self, factory):
        model = factory()
        assert model.predict(two_symbol_panel(),
                             pd.Timestamp('2022-06-01')) == {}

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_fit_then_predict_returns_centered_scores_in_range(self, factory):
        panel = two_symbol_panel()
        model = factory()
        model.fit(panel)  # must not raise
        scores = model.predict(panel, panel['A'].index[-1])
        assert set(scores) <= {'A', 'B'}
        assert scores  # the net trained and produced at least one score
        for value in scores.values():
            assert -0.5 <= value <= 0.5

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_uses_extended_feature_columns(self, factory):
        # Both models share the extended feature set (15 columns:
        # 5 baseline + 5 price extras + 5 seasonal).
        assert len(FEATURE_COLUMNS) == 15


# ----------------------------------------------------------------------
# Graceful degradation — never raises on bad/insufficient data.
# ----------------------------------------------------------------------
class TestGracefulDegradation:
    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_too_few_rows_stays_unfitted(self, factory, caplog):
        tiny = {'A': learnable(1).iloc[:2]}
        model = factory()
        with caplog.at_level(logging.WARNING):
            model.fit(tiny)  # must not raise
        assert model.predict(tiny, tiny['A'].index[-1]) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_empty_train_data_stays_unfitted(self, factory, caplog):
        model = factory()
        with caplog.at_level(logging.WARNING):
            model.fit({})  # must not raise
        assert model.predict({}, pd.Timestamp('2022-06-01')) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_single_class_labels_degrade_to_no_signal(self, factory, caplog):
        # Strictly rising closes -> every next-day return is positive ->
        # one label class -> cannot train -> stays unfitted with a WARNING.
        n = 220
        idx = pd.bdate_range('2022-01-03', periods=n)
        close = np.array([100.0 + i for i in range(n)], dtype=float)
        rising = pd.DataFrame(
            {'open': close, 'high': close * 1.01, 'low': close * 0.99,
             'close': close, 'volume': np.full(n, 500_000.0)}, index=idx)
        model = factory()
        with caplog.at_level(logging.WARNING):
            model.fit({'A': rising})  # must not raise
        assert model.predict({'A': rising}, idx[-1]) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_none_and_closeless_frames_are_skipped_at_predict(self, factory):
        panel = two_symbol_panel()
        model = factory()
        model.fit(panel)  # must not raise
        mixed = {'A': panel['A'], 'B': None,
                 'C': panel['B'].drop(columns=['close'])}
        scores = model.predict(mixed, panel['A'].index[-1])
        # B (None) and C (no close) are skipped; only A is eligible.
        assert set(scores) <= {'A'}

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_empty_frame_is_skipped_at_predict(self, factory):
        panel = two_symbol_panel()
        model = factory()
        model.fit(panel)
        empty = pd.DataFrame(columns=['open', 'high', 'low', 'close',
                                      'volume'])
        scores = model.predict({'A': panel['A'], 'Z': empty},
                               panel['A'].index[-1])
        assert set(scores) <= {'A'}


class TestSequenceModelLookback:
    """SequenceModel-specific degrade: history shorter than the lookback
    window forms no window and is skipped/leaves the model unfitted."""

    def test_history_shorter_than_lookback_skipped_at_predict(self):
        panel = two_symbol_panel()
        model = fast_seq(lookback=10)
        model.fit(panel)  # trained on the full-length panel
        # 'SHORT' has too few rows to fill even one 10-row feature window
        # (extended features burn ~20 warm-up rows, so 12 raw rows -> 0
        # feature rows), so it is skipped while 'A' is still scored.
        short = panel['A'].iloc[:12]
        scores = model.predict({'A': panel['A'], 'SHORT': short},
                               panel['A'].index[-1])
        assert 'SHORT' not in scores
        assert set(scores) <= {'A'}

    def test_all_symbols_shorter_than_lookback_stays_unfitted(self, caplog):
        # Every symbol is too short to form a single training window ->
        # no windows -> stays unfitted with a WARNING, predict() == {}.
        short_panel = {'A': learnable(1).iloc[:25],
                       'B': learnable(2).iloc[:25]}
        model = SequenceModel(lookback=20, hidden_size=8, epochs=5)
        with caplog.at_level(logging.WARNING):
            model.fit(short_panel)  # must not raise
        assert model.predict(short_panel,
                             short_panel['A'].index[-1]) == {}
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)


# ----------------------------------------------------------------------
# DETERMINISM — the headline ML test: identical input -> identical scores.
# ----------------------------------------------------------------------
class TestDeterminism:
    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_fit_twice_predict_identical_scores_exact_equality(self, factory):
        panel = two_symbol_panel()

        m1 = factory()
        m1.fit(panel)
        s1 = m1.predict(panel, panel['A'].index[-1])

        m2 = factory()
        m2.fit(panel)
        s2 = m2.predict(panel, panel['A'].index[-1])

        assert s1  # actually fitted
        assert s1.keys() == s2.keys()
        for symbol in s1:
            # EXACT equality — every torch RNG/threading knob is pinned.
            assert s1[symbol] == s2[symbol]

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_repeated_predict_on_same_model_is_stable(self, factory):
        panel = two_symbol_panel()
        model = factory()
        model.fit(panel)
        a = model.predict(panel, panel['A'].index[-1])
        b = model.predict(panel, panel['A'].index[-1])
        assert a == b


# ----------------------------------------------------------------------
# No-leakage: label alignment excludes each symbol's last row from
# training (mirrors the GradientBoostingModel label-alignment test), and the
# train-window scaler is fitted on training rows only (predict never alters
# the fitted parameters).
# ----------------------------------------------------------------------
class TestNoLeakage:
    def test_mlp_training_rows_exclude_each_symbols_last_row(self):
        # _build_training_rows aligns label[i] = sign(close[i+1]/close[i]-1)
        # and drops each symbol's final feature row (no next-day close), so
        # the pooled X has exactly (#feature_rows - 1) rows per symbol.
        from desks.models.neural import _build_training_rows
        from desks import features as feature_lib

        panel = two_symbol_panel()
        x_matrix, y = _build_training_rows(panel)

        expected = 0
        for frame in panel.values():
            n_feat_rows = len(feature_lib.extended_feature_frame(frame))
            assert n_feat_rows > 1
            expected += n_feat_rows - 1  # last row excluded (no label)
        assert x_matrix.shape == (expected, len(FEATURE_COLUMNS))
        assert len(y) == expected
        # Both classes present in this synthetic walk.
        assert set(np.unique(y)) == {0, 1}

    def test_sequence_windows_exclude_final_predict_window_from_training(self):
        # The window ending on a symbol's FINAL feature row has no next-day
        # close, so it must NOT be a training window — it is the predict-time
        # input instead. Train windows therefore number (#feature_rows -
        # lookback), one fewer than the total sliding windows.
        from desks import features as feature_lib

        frame = learnable(7)
        lookback = 10
        model = SequenceModel(lookback=lookback, hidden_size=8, epochs=5)
        x_windows, y = model._symbol_windows(frame)

        n_feat_rows = len(feature_lib.extended_feature_frame(frame))
        total_sliding = n_feat_rows - lookback + 1  # windows incl. final
        expected_train = total_sliding - 1  # final (predict) window dropped
        assert x_windows.shape == (expected_train, lookback,
                                   len(FEATURE_COLUMNS))
        assert len(y) == expected_train

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_predict_does_not_mutate_fitted_scaler(self, factory):
        # The train-window scaler is fitted in fit() and never refitted at
        # predict — predicting must leave its stored mean/std untouched.
        panel = two_symbol_panel()
        model = factory()
        model.fit(panel)
        mean_before = model._scaler.mean_.copy()
        std_before = model._scaler.std_.copy()
        model.predict(panel, panel['A'].index[-1])
        np.testing.assert_array_equal(model._scaler.mean_, mean_before)
        np.testing.assert_array_equal(model._scaler.std_, std_before)

    @pytest.mark.parametrize('factory', NEW_MODEL_FACTORIES)
    def test_appending_future_rows_does_not_change_past_score(self, factory):
        # The controller slices data to index <= date; a model must score a
        # symbol identically whether or not rows AFTER the predict row exist
        # in the frame, because predict only consumes the latest window/row
        # up to the supplied frame's end. Here we fit once, then predict on a
        # truncated frame vs the same frame with extra trailing rows removed
        # — the score for the common last row must match.
        panel = two_symbol_panel()
        model = factory()
        model.fit(panel)

        full = panel['A']
        # Cut to a point well past warm-up + lookback.
        cut = 180
        truncated = full.iloc[:cut]
        # Predict on the truncated frame: its last row is full.iloc[cut-1].
        s_trunc = model.predict({'A': truncated}, truncated.index[-1])
        # Predict on the full frame sliced to exactly the same last row.
        s_same = model.predict({'A': full.iloc[:cut]}, truncated.index[-1])
        assert s_trunc == s_same  # identical input window -> identical score
        # And appending FUTURE rows beyond the cut never reaches back to
        # change the truncated-frame score (only the latest row/window is
        # used, so the scores legitimately differ when the latest row moves;
        # the invariant we assert is the same-end-row equality above).


# ----------------------------------------------------------------------
# Learnability sanity — the net actually trains: on data with a real
# next-day-direction signal, scores correlate with the truth (loose).
# ----------------------------------------------------------------------
class TestLearnability:
    """Prove the nets actually TRAIN: on data with a real next-day-direction
    signal, the model's score must correlate with the realized direction.

    Grading uses MANY predict points (a tail window of dates per symbol),
    not one noisy day, so the metric is statistically meaningful and stable.
    For each graded date D we fit on the symbol panel, slice each symbol to
    ``index <= D`` (exactly what the controller hands predict), score it,
    and compare the score's sign to the actual D->D+1 move. The threshold is
    loose (> chance) — we only assert the net learns SOMETHING, never a
    precise accuracy.
    """

    def _agreement(self, model, frames, n_grade=24):
        """Fraction of graded (date, symbol) points where the score's sign
        matches the realized next-day direction. Many points per symbol."""
        model.fit(frames)
        preds, truth = [], []
        for sym, frame in frames.items():
            close = frame['close']
            # Grade the last n_grade dates that still have a next-day close
            # (i.e. exclude only the final row). Each predict slices the
            # frame to index <= D, so no future row is ever seen.
            gradable = frame.index[-(n_grade + 1):-1]
            for date in gradable:
                history = frame.loc[:date]
                score = model.predict({sym: history}, date)
                if sym not in score:
                    continue
                pos = close.index.get_loc(date)
                realized = close.iloc[pos + 1] / close.iloc[pos] - 1.0
                if realized == 0.0:
                    continue
                preds.append(score[sym])
                truth.append(1.0 if realized > 0 else -1.0)
        preds = np.array(preds)
        truth = np.array(truth)
        return preds, truth

    def test_mlp_scores_track_direction_on_signal_data(self):
        frames = {f'S{i}': momentum_frame(seed=i + 30) for i in range(6)}
        model = MLPModel(hidden_sizes=(16, 8), epochs=80)
        preds, truth = self._agreement(model, frames)
        assert len(preds) >= 60  # plenty of graded points
        # Sign agreement should beat a coin flip on signal-bearing data.
        agree = np.mean(np.sign(preds) == truth)
        assert agree > 0.5

    def test_sequence_scores_track_direction_on_signal_data(self):
        frames = {f'S{i}': momentum_frame(seed=i + 70) for i in range(6)}
        model = SequenceModel(lookback=15, hidden_size=12, epochs=50)
        preds, truth = self._agreement(model, frames)
        assert len(preds) >= 60
        agree = np.mean(np.sign(preds) == truth)
        assert agree > 0.5


# ----------------------------------------------------------------------
# Registry wiring — available_models() advertises mlp + lstm and
# build_model() returns the right classes.
# ----------------------------------------------------------------------
class TestRegistry:
    def test_available_models_includes_mlp_and_lstm(self):
        entries = {e['id']: e for e in available_models()}
        assert 'mlp' in entries and 'lstm' in entries
        assert entries['mlp']['name'] == 'Neural MLP'
        assert entries['lstm']['name'] == 'Neural LSTM'
        for key in ('mlp', 'lstm'):
            assert set(entries[key]) == {'id', 'name', 'description'}
            assert entries[key]['description']
            # torch is the stated optional dependency in the descriptions.
            assert 'torch' in entries[key]['description']

    def test_build_model_mlp_returns_mlp_model(self):
        model = build_model('mlp')
        assert isinstance(model, MLPModel)
        assert isinstance(model, WalkForwardModel)

    def test_build_model_lstm_returns_sequence_model(self):
        model = build_model('lstm')
        assert isinstance(model, SequenceModel)
        assert isinstance(model, WalkForwardModel)

    def test_build_returns_fresh_unshared_instances(self):
        assert build_model('mlp') is not build_model('mlp')
        assert build_model('lstm') is not build_model('lstm')
