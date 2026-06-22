"""Tests for desks.ml_model.GradientBoostingModel.

The load-bearing assertion is label alignment: the label for feature row i
is computed from closes i and i+1, so the LAST usable training row is row
n-2 — the final row has no next-day close and must be excluded from the
training matrix (off-by-one). A spy on the classifier's fit records the
exact matrix the sklearn model receives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from desks.ml_model import FEATURE_COLUMNS, GradientBoostingModel

#: Hand-picked closes: direction changes guarantee both label classes.
CLOSES = [100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 100.0, 105.0, 106.0]
#: label[i] = 1 if close[i+1] > close[i]; defined for i = 0..8.
#: Training rows are i = 1..8 (row 0 has no return; row 9 has no label).
EXPECTED_LABELS = [0, 1, 1, 0, 1, 0, 1, 1]


def hand_frame(closes=CLOSES) -> pd.DataFrame:
    """10-row frame with pre-filled indicator columns (no warmup NaNs).

    Indicators are constants/offsets so every feature except the row-0
    return is defined; the only rows dropped from training are the ones
    label alignment itself demands.
    """
    n = len(closes)
    index = pd.bdate_range('2023-01-02', periods=n)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({
        'open': closes,
        'high': closes * 1.01,
        'low': closes * 0.99,
        'close': closes,
        'volume': np.full(n, 500_000.0),
        'rsi': np.full(n, 50.0),
        'macd': np.full(n, 0.1),
        'bb_upper': closes + 5.0,
        'bb_lower': closes - 5.0,
        'volume_sma': np.full(n, 400_000.0),
    }, index=index)


class TestLabelAlignment:
    def test_training_matrix_stops_at_row_n_minus_2(self):
        df = hand_frame()
        model = GradientBoostingModel()
        x_matrix, y = model.build_training_set({'SYM': df})

        # 10 rows -> 8 training rows: row 0 (no prior close for the return
        # feature) and row 9 (no next close for the label) are excluded.
        assert x_matrix.shape == (len(df) - 2, len(FEATURE_COLUMNS))
        assert list(y) == EXPECTED_LABELS

    def test_features_align_with_their_own_row_not_the_label_row(self):
        df = hand_frame()
        model = GradientBoostingModel()
        x_matrix, _ = model.build_training_set({'SYM': df})

        # Column 0 is the 1-day return of the FEATURE row (i), while the
        # label comes from i+1 — verify rows 1..8 exactly.
        expected_returns = (df['close'].pct_change().iloc[1:-1]
                            .to_numpy(dtype=float))
        np.testing.assert_allclose(x_matrix[:, 0], expected_returns)

    def test_fit_never_errors_and_spy_proves_matrix_shape(self):
        df = hand_frame()
        model = GradientBoostingModel(n_estimators=5)

        recorded = {}
        original_fit = model._classifier.fit

        def spying_fit(x_matrix, y, **kwargs):
            recorded['shape'] = x_matrix.shape
            recorded['labels'] = list(y)
            return original_fit(x_matrix, y, **kwargs)

        model._classifier.fit = spying_fit
        model.fit({'SYM': df})  # must not raise

        assert recorded['shape'] == (len(df) - 2, len(FEATURE_COLUMNS))
        assert recorded['labels'] == EXPECTED_LABELS

        scores = model.predict({'SYM': df}, df.index[-1])
        assert set(scores) == {'SYM'}
        assert -0.5 <= scores['SYM'] <= 0.5

    def test_pooling_across_symbols_concatenates_training_rows(self):
        df_a = hand_frame()
        df_b = hand_frame(closes=[50.0, 51.0, 49.5, 52.0, 53.0,
                                  51.0, 54.0, 50.0, 55.0, 56.0])
        model = GradientBoostingModel()
        x_matrix, y = model.build_training_set({'A': df_a, 'B': df_b})
        assert x_matrix.shape == (2 * (len(df_a) - 2), len(FEATURE_COLUMNS))
        assert len(y) == len(x_matrix)


class TestInsufficientData:
    def test_too_few_rows_yields_empty_dict_not_exception(self):
        df = hand_frame().iloc[:2]
        model = GradientBoostingModel()
        model.fit({'SYM': df})  # must not raise
        assert model.predict({'SYM': df}, df.index[-1]) == {}

    def test_empty_train_data_is_handled(self):
        model = GradientBoostingModel()
        model.fit({})
        assert model.predict({}, pd.Timestamp('2023-01-02')) == {}
        x_matrix, y = model.build_training_set({})
        assert x_matrix.shape == (0, len(FEATURE_COLUMNS))
        assert len(y) == 0

    def test_single_class_labels_degrade_to_no_signal(self):
        # Strictly rising closes -> every label is 1 -> cannot train.
        rising = [100.0 + i for i in range(10)]
        df = hand_frame(closes=rising)
        model = GradientBoostingModel()
        model.fit({'SYM': df})
        assert model.predict({'SYM': df}, df.index[-1]) == {}

    def test_predict_before_any_fit_returns_empty(self):
        model = GradientBoostingModel()
        assert model.predict({'SYM': hand_frame()},
                             pd.Timestamp('2023-01-13')) == {}

    def test_raw_ohlcv_without_indicator_columns_uses_fallbacks(
            self, make_ohlcv):
        # 120 raw OHLCV rows: the model computes its own rsi/macd/bb/volume
        # features, drops the warmup NaNs, and still trains cleanly.
        df = make_ohlcv(n_days=120, seed=77)
        model = GradientBoostingModel(n_estimators=5)
        model.fit({'SYM': df})
        scores = model.predict({'SYM': df}, df.index[-1])
        assert set(scores) == {'SYM'}
        assert -0.5 <= scores['SYM'] <= 0.5
