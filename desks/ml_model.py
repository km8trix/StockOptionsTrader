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

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol centered score: P(up tomorrow) - 0.5.

        Symbols with insufficient/NaN feature rows are skipped; if the
        model never trained successfully, returns {}.
        """
        if not self._fitted:
            return {}
        scores: Dict[str, float] = {}
        for symbol, frame in data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            features = self._feature_frame(frame)
            if features.empty:
                continue
            latest = features.iloc[[-1]].to_numpy(dtype=float)
            prob_up = float(self._classifier.predict_proba(latest)[0, 1])
            scores[symbol] = prob_up - 0.5
        return scores
