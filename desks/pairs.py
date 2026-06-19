"""
Cointegration pairs model — Engle-Granger scan over the liquid universe.

PairsCointegrationModel is a WalkForwardModel managed by
WalkForwardController (the leakage chokepoint): fit() scans only the
controller-sliced train window; predict() prices spreads on frames sliced
to index <= the simulation date.

Fit procedure:
    1. Rank symbols by average volume over the train window; cap the scan
       at the `max_symbols` (default 40) most liquid (ties break by symbol
       for determinism).
    2. For each pair (itertools.combinations over that ranking): align
       LOG closes on common dates; require `min_overlap` rows; run
       statsmodels Engle-Granger coint and keep p < `p_threshold`.
    3. Hedge ratio beta = OLS slope of log(a) on log(b) (with intercept).
       Pairs with beta <= 0 are skipped (an inverse hedge is not a
       tradable long/short spread here).
    4. spread = log(a) - beta * log(b); its mean and SAMPLE std
       (ddof=1) are stored. Near-zero spread std is skipped.
    5. Keep at most `max_pairs` (default 5) by lowest p-value (ties break
       by symbol pair for determinism).

predict(data, date) -> {'pairs': [{'a', 'b', 'beta', 'z'}]} where z is
the current spread z-score. Numerical guards: a leg with no usable bar,
mismatched latest bar dates between the legs, or non-positive prices
cause that pair to be skipped for the day (logged at DEBUG).

Z-SCORE REFINEMENT (roadmap Phase 5 "pairs ddof + rolling mean"):
    - The stored spread std uses ddof=1 (sample std) rather than the old
      population std (ddof=0). For the train-window length this is a tiny
      sqrt(n/(n-1)) rescale, but it is the unbiased estimator and matches
      the Sharpe/Sortino ddof=1 convention adopted in Phase 3.
    - The z-score centers on a ROLLING mean of the most recent
      `z_mean_window` spread observations (computed live in predict() from
      the date-sliced leg histories) instead of the static fit-window
      `spread_mean`. A cointegrated spread's level drifts slowly between
      63-day refits; centering on the trailing window tracks that drift so
      |z| reflects the CURRENT deviation, not the deviation from a stale
      fit-time average. The rolling mean uses ONLY past spread values
      (index <= date), so it remains leakage-free. With
      `z_mean_window=None` the model recovers the old static-mean
      behavior exactly (`spread_mean` is still stored either way).
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from desks.walk_forward import WalkForwardModel

logger = logging.getLogger(__name__)


class PairsCointegrationModel(WalkForwardModel):
    """Engle-Granger pairs scanner with stored spread parameters."""

    def __init__(self, max_pairs: int = 5, p_threshold: float = 0.05,
                 max_symbols: int = 40, min_overlap: int = 60,
                 z_mean_window: Optional[int] = 60,
                 spread_std_ddof: int = 1):
        self.max_pairs = max_pairs
        self.p_threshold = p_threshold
        self.max_symbols = max_symbols
        self.min_overlap = min_overlap
        #: Delta-degrees-of-freedom for the spread std. Default 1 (sample/
        #: unbiased, matching the project-wide Sharpe/Sortino ddof=1
        #: convention); set 0 to recover the pre-Phase-E population std and
        #: thus the exact pre-Phase-E z-score (full opt-out).
        if spread_std_ddof not in (0, 1):
            raise ValueError(
                f"spread_std_ddof must be 0 or 1, got {spread_std_ddof}")
        self.spread_std_ddof = spread_std_ddof
        #: Trailing # of spread observations used to center the predict()
        #: z-score (rolling mean). None recovers the static fit-window
        #: `spread_mean` behavior. Must be >= 2 when set.
        if z_mean_window is not None and z_mean_window < 2:
            raise ValueError(
                f"z_mean_window must be >= 2 (or None), got {z_mean_window}")
        self.z_mean_window = z_mean_window
        #: Fitted pairs: {'a','b','beta','p_value','spread_mean','spread_std'}
        self.pairs: List[Dict] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _log_closes(frame: pd.DataFrame) -> Optional[pd.Series]:
        """Log closes with non-positive/NaN rows dropped; None if unusable."""
        if frame is None or frame.empty or 'close' not in frame.columns:
            return None
        close = frame['close'].dropna()
        close = close[close > 0]
        if close.empty:
            return None
        return np.log(close)

    def _liquid_symbols(self, train_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Symbols ranked by average volume (desc), capped at max_symbols.
        Ties break by symbol name for determinism."""
        ranked = []
        for symbol, frame in train_data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            if 'volume' in frame.columns:
                avg_volume = float(frame['volume'].mean())
                if np.isnan(avg_volume):
                    avg_volume = 0.0
            else:
                avg_volume = 0.0
            ranked.append((symbol, avg_volume))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [symbol for symbol, _ in ranked[:self.max_symbols]]

    # ------------------------------------------------------------------
    # WalkForwardModel protocol
    # ------------------------------------------------------------------
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Scan the train window for cointegrated pairs (module docstring)."""
        self.pairs = []
        self._fitted = False

        symbols = self._liquid_symbols(train_data)
        log_closes = {symbol: self._log_closes(train_data[symbol])
                      for symbol in symbols}

        candidates: List[Dict] = []
        for sym_a, sym_b in itertools.combinations(symbols, 2):
            log_a, log_b = log_closes[sym_a], log_closes[sym_b]
            if log_a is None or log_b is None:
                continue
            common = log_a.index.intersection(log_b.index)
            if len(common) < self.min_overlap:
                logger.debug("Pair %s/%s skipped: %d overlapping rows (< %d)",
                             sym_a, sym_b, len(common), self.min_overlap)
                continue
            series_a = log_a.loc[common].to_numpy(dtype=float)
            series_b = log_b.loc[common].to_numpy(dtype=float)

            try:
                _, p_value, _ = coint(series_a, series_b)
            except Exception as exc:
                logger.debug("Pair %s/%s skipped: coint failed (%s)",
                             sym_a, sym_b, exc)
                continue
            if not np.isfinite(p_value) or p_value >= self.p_threshold:
                continue

            # Hedge ratio: OLS slope of log(a) on log(b), with intercept.
            b_var = float(np.var(series_b))
            if b_var <= 0.0:
                logger.debug("Pair %s/%s skipped: zero-variance leg %s",
                             sym_a, sym_b, sym_b)
                continue
            beta = float(np.mean((series_b - series_b.mean())
                                 * (series_a - series_a.mean())) / b_var)
            if beta <= 0.0:
                logger.debug("Pair %s/%s skipped: non-positive hedge ratio "
                             "beta=%.4f", sym_a, sym_b, beta)
                continue

            spread = series_a - beta * series_b
            spread_std = float(np.std(spread, ddof=self.spread_std_ddof))
            if spread_std <= 1e-12:
                logger.debug("Pair %s/%s skipped: zero spread std",
                             sym_a, sym_b)
                continue

            candidates.append({
                'a': sym_a, 'b': sym_b, 'beta': beta,
                'p_value': float(p_value),
                'spread_mean': float(np.mean(spread)),
                'spread_std': spread_std,
            })

        candidates.sort(key=lambda c: (c['p_value'], c['a'], c['b']))
        self.pairs = candidates[:self.max_pairs]
        self._fitted = True
        logger.info("PairsCointegrationModel fitted: %d pair(s) kept of %d "
                    "symbols scanned", len(self.pairs), len(symbols))

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict:
        """Current spread z-scores for the fitted pairs.

        Returns {'pairs': [{'a','b','beta','z'}]}; pairs whose legs lack a
        shared latest bar (missing/stale leg) are skipped for the day.

        z = (today's spread - center) / spread_std, where `center` is the
        rolling mean of the most recent `z_mean_window` spread observations
        (over the legs' overlapping, date-sliced dates) when z_mean_window
        is set, else the static fit-window `spread_mean`. The rolling mean
        uses ONLY past spread values (the frames are already sliced to
        index <= date by the controller), so it is leakage-free. If too few
        overlapping rows exist to form the window, the static `spread_mean`
        is used as a graceful fallback.
        """
        if not self._fitted:
            return {'pairs': []}
        priced: List[Dict] = []
        for pair in self.pairs:
            frame_a = data.get(pair['a'])
            frame_b = data.get(pair['b'])
            log_a = self._log_closes(frame_a)
            log_b = self._log_closes(frame_b)
            if log_a is None or log_b is None:
                logger.debug("Pair %s/%s unpriced: missing leg data",
                             pair['a'], pair['b'])
                continue
            if log_a.index[-1] != log_b.index[-1]:
                logger.debug("Pair %s/%s unpriced: legs' latest bars differ "
                             "(%s vs %s)", pair['a'], pair['b'],
                             log_a.index[-1], log_b.index[-1])
                continue
            spread = float(log_a.iloc[-1]) - pair['beta'] * float(log_b.iloc[-1])
            center = self._z_center(pair, log_a, log_b)
            z_score = (spread - center) / pair['spread_std']
            priced.append({'a': pair['a'], 'b': pair['b'],
                           'beta': pair['beta'], 'z': float(z_score)})
        return {'pairs': priced}

    def _z_center(self, pair: Dict, log_a: pd.Series,
                  log_b: pd.Series) -> float:
        """Center for the predict() z-score.

        Rolling mean of the trailing `z_mean_window` spread observations on
        the legs' overlapping dates (today's bar included; only past data,
        as the frames are date-sliced upstream). Falls back to the static
        fit-window `spread_mean` when z_mean_window is None or there are
        fewer than `z_mean_window` overlapping rows."""
        window = self.z_mean_window
        if window is None:
            return pair['spread_mean']
        common = log_a.index.intersection(log_b.index)
        if len(common) < window:
            return pair['spread_mean']
        tail = common[-window:]
        spread = (log_a.loc[tail].to_numpy(dtype=float)
                  - pair['beta'] * log_b.loc[tail].to_numpy(dtype=float))
        return float(np.mean(spread))
