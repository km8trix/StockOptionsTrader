"""
Gradient-boosting walk-forward model — the clean replacement for the
look-ahead-contaminated strategies.advanced.MachineLearningStrategy.

The old strategy trained once on the full expanding window inside
generate_signals, so early predictions were made by a model that had
already seen later prices. This implementation is a WalkForwardModel: it
only ever trains on what WalkForwardController.fit hands it (sliced to
index <= the simulation date and capped to the train window), and refits
on the controller's schedule.

Label alignment (off-by-one matters): the label for feature row i is the
sign of close[i+1] / close[i] - 1. The LAST usable training row is
therefore the second-to-last row of the train window — the final row has
no next-day close and is excluded from the training matrix (it is still
used as the feature row at predict time, where no label is needed).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

from desks.walk_forward import WalkForwardModel

logger = logging.getLogger(__name__)

#: Feature column order of the training/prediction matrix.
FEATURE_COLUMNS = ('ret_1', 'rsi', 'macd', 'bb_position', 'volume_ratio')


class GradientBoostingModel(WalkForwardModel):
    """Predicts next-day direction from indicator features.

    Features per row (reusing the indicator engineering already present in
    the enriched frames, with self-contained fallbacks when a column is
    missing): 1-day return, RSI, MACD, Bollinger-band position, and
    volume / 20-day average volume. fit() pools rows across symbols;
    predict() returns per-symbol probability-of-up minus 0.5 (a centered
    score: positive = long signal strength, per the WalkForwardModel
    protocol).
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 3,
                 random_state: int = 42):
        self._classifier = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state)
        self._fitted = False

    def checkpoint_spec(self) -> Dict[str, object]:
        """Identity required to deterministically refit a safe checkpoint."""
        return {
            'checkpoint_type': 'gradient-boosting-refit-v1',
            'feature_columns': list(FEATURE_COLUMNS),
            # Pin every sklearn hyperparameter, including defaults.  A caller
            # that mutates the estimator via set_params cannot accidentally
            # restore as though it were the constructor-default model.
            'parameters': self._classifier.get_params(deep=False),
            'sklearn_version': sklearn.__version__,
        }

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Per-row features (FEATURE_COLUMNS order); rows with any NaN or
        non-finite feature are dropped. Prefers indicator columns already
        on the frame (the engine enriches them); computes fallbacks from
        raw OHLCV otherwise."""
        close = data['close']
        features = pd.DataFrame(index=data.index)

        features['ret_1'] = close.pct_change()

        if 'rsi' in data.columns:
            features['rsi'] = data['rsi']
        else:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            features['rsi'] = 100 - (100 / (1 + gain / loss))

        if 'macd' in data.columns:
            features['macd'] = data['macd']
        else:
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            features['macd'] = ema_12 - ema_26

        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            bb_upper, bb_lower = data['bb_upper'], data['bb_lower']
        else:
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
        bb_width = bb_upper - bb_lower
        features['bb_position'] = np.where(
            bb_width > 0, (close - bb_lower) / bb_width, 0.5)

        if 'volume_sma' in data.columns:
            volume_avg = data['volume_sma']
        else:
            volume_avg = data['volume'].rolling(window=20).mean()
        features['volume_ratio'] = data['volume'] / volume_avg

        features = features.replace([np.inf, -np.inf], np.nan)
        return features.dropna()

    def build_training_set(
            self, train_data: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pooled (X, y) across symbols.

        For each symbol: feature rows are aligned with labels
        label[i] = 1 if close[i+1]/close[i] - 1 > 0 else 0, so the last
        row of each frame contributes features only at predict time and
        is EXCLUDED here (no next close exists for it). Exposed publicly
        so tests and Phase 6 implementers can verify the alignment.
        """
        x_parts, y_parts = [], []
        for symbol, data in train_data.items():
            if data is None or data.empty or 'close' not in data.columns:
                continue
            features = self._feature_frame(data)
            if features.empty:
                continue
            close = data['close']
            # next_return[i] = close[i+1]/close[i] - 1; NaN on the last row.
            next_return = close.shift(-1) / close - 1.0
            labels = (next_return > 0).astype(int)
            # Keep feature rows that have a defined next-day return: this
            # drops the final row — the off-by-one the label demands.
            usable = features.index.intersection(next_return.dropna().index)
            if usable.empty:
                continue
            x_parts.append(features.loc[usable].to_numpy(dtype=float))
            y_parts.append(labels.loc[usable].to_numpy(dtype=int))

        if not x_parts:
            return (np.empty((0, len(FEATURE_COLUMNS))), np.empty((0,)))
        return np.vstack(x_parts), np.concatenate(y_parts)

    # ------------------------------------------------------------------
    # WalkForwardModel protocol
    # ------------------------------------------------------------------
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Train on pooled rows across symbols; degrade gracefully.

        With fewer than 2 samples or a single label class the classifier
        cannot train — the model is marked unfitted and predict() returns
        {} until a later refit succeeds.
        """
        x_matrix, y = self.build_training_set(train_data)
        if len(x_matrix) < 2 or len(np.unique(y)) < 2:
            self._fitted = False
            logger.warning(
                "GradientBoostingModel.fit skipped: %d samples, %d classes "
                "(need >= 2 of each)", len(x_matrix), len(np.unique(y)))
            return
        self._classifier.fit(x_matrix, y)
        self._fitted = True
        logger.debug("GradientBoostingModel fitted on %d pooled samples",
                     len(x_matrix))

    def _fast_last_row(self, frame: pd.DataFrame) -> np.ndarray | None:
        """O(1) feature vector for the frame's FINAL row, or None when the
        full _feature_frame path must be used instead.

        predict() only ever consumes the last feature row, so recomputing
        _feature_frame over the whole history every simulated day is
        O(history) per symbol per day. When the engine-enriched indicator
        columns are already on the frame, the last row is plain column
        reads plus scalar arithmetic that is bit-identical to the
        vectorized ops (pct_change's last value is close[-1]/close[-2]-1).

        Returns None — demanding the full path — unless ALL of:
        - every fast-path column is present with a plain numpy float64 or
          integer dtype (float32 would round differently; pandas
          extension dtypes can hold pd.NA);
        - the frame has >= 2 rows (ret_1 needs a prior close);
        - close[-2] != 0 and volume_sma[-1] != 0 (the full path turns the
          resulting inf into NaN);
        - every computed feature value is finite. _feature_frame's
          dropna() makes predict() silently back off to the last VALID
          row (or skip the symbol) when the final row has a NaN/inf
          feature, and only the full path reproduces that back-off.
        The bb_position quirk is replicated: np.where(bb_width > 0, ...,
        0.5) yields 0.5 for NaN or non-positive width, so a NaN band does
        NOT invalidate the row.
        """
        if len(frame.index) < 2:
            return None
        for name in ('close', 'volume', 'rsi', 'macd',
                     'bb_upper', 'bb_lower', 'volume_sma'):
            if name not in frame.columns:
                return None
            dtype = frame.dtypes[name]
            if not isinstance(dtype, np.dtype):
                return None
            if dtype != np.float64 and dtype.kind not in 'iu':
                return None

        close_prev = frame['close'].iloc[-2]
        if close_prev == 0:
            return None
        close = frame['close'].iloc[-1]
        ret_1 = close / close_prev - 1

        bb_upper = frame['bb_upper'].iloc[-1]
        bb_lower = frame['bb_lower'].iloc[-1]
        bb_width = bb_upper - bb_lower
        if bb_width > 0:  # NaN width compares False -> 0.5, as np.where does
            bb_position = (close - bb_lower) / bb_width
        else:
            bb_position = 0.5

        volume_sma = frame['volume_sma'].iloc[-1]
        if volume_sma == 0:
            return None
        volume_ratio = frame['volume'].iloc[-1] / volume_sma

        vector = np.array(
            [ret_1, frame['rsi'].iloc[-1], frame['macd'].iloc[-1],
             bb_position, volume_ratio], dtype=float)
        if not np.isfinite(vector).all():
            return None
        return vector

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol centered score: P(up tomorrow) - 0.5.

        Symbols with insufficient/NaN feature rows are skipped; if the
        model never trained successfully, returns {}.

        Feature vectors come from the O(1) fast path when a frame
        qualifies (see _fast_last_row) and the full _feature_frame path
        otherwise; all rows are then scored with a SINGLE predict_proba
        call (GBM decision values are per-row independent, so batching is
        bit-identical to per-symbol calls).
        """
        if not self._fitted:
            return {}
        symbols = []
        rows = []
        for symbol, frame in data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            vector = self._fast_last_row(frame)
            if vector is None:
                features = self._feature_frame(frame)
                if features.empty:
                    continue
                vector = features.iloc[[-1]].to_numpy(dtype=float)[0]
            symbols.append(symbol)
            rows.append(vector)
        if not rows:
            return {}
        probs = self._classifier.predict_proba(np.vstack(rows))[:, 1]
        return {symbol: float(prob_up) - 0.5
                for symbol, prob_up in zip(symbols, probs)}
