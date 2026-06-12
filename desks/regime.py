"""
HMM market-regime model — Gaussian HMM over market-level features.

RegimeHMMModel is a WalkForwardModel managed by WalkForwardController
(the leakage chokepoint): fit() trusts the controller's slicing/capping
and never re-implements it; predict() sees only frames sliced to
index <= the simulation date.

Features (market-level, computed from the symbol -> frame dict):
    mkt_ret      cross-sectional mean daily close-to-close return
    ret_std_20   20-day rolling std of mkt_ret
    volume_ratio cross-sectional mean of volume / 20-day average volume
NaN warm-up rows are dropped.

FITTING (determinism + robustness):
    The design matrix the HMM sees applies two transforms to the raw
    features, both replayed identically at predict time:
        1. ret_std_20 enters in LOG space. Volatility is approximately
           log-normal: in raw space the full-covariance EM gains more
           likelihood by splitting the fat upper tail of one high-vol
           regime across two states (merging the calm regimes into the
           third) than by separating the regimes; in log space the vol
           levels are similar-variance Gaussians and EM separates them.
        2. Column z-scores. The mean/std come from the TRAINING window
           only, are stored on the model, and are re-applied at predict
           — no leakage. Unstandardized, EM is dominated by the
           largest-scale column and routinely converges to degenerate
           decodes that merge mean-reverting and trending into shared
           states.
    EM is then run N_FIT_RESTARTS times with seeds random_state + k and
    the fit with the best training log-likelihood wins (strictly-greater
    comparison: the earliest seed wins ties) — a deterministic guard
    against local optima.

STATE LABELING (deterministic rule):
    After fitting the 3-state HMM, training rows are assigned to states
    via Viterbi (model.predict). For each state, over its assigned rows:
        (a) AR(1) coefficient of mkt_ret within the state — the OLS
            slope of ret[t] on ret[t-1] over consecutive row pairs that
            are both assigned to the state. When the state has fewer
            than MIN_AR1_SAME_STATE_PAIRS such pairs (the HMM can split
            one oscillating mean-reverting regime across two ALTERNATING
            states, leaving almost no same-state pairs and masking the
            negative autocorrelation behind a neutral 0.0), the estimate
            falls back to every pair whose CURRENT row t is assigned to
            the state, whatever the state at t-1. 0.0 when even the
            fallback has fewer than 2 pairs or a zero-variance
            regressor;
        (b) within-state std of mkt_ret (0.0 with fewer than 2 rows);
        (c) within-state |mean| of mkt_ret (0.0 with no rows).
    'high_vol'       = the state with the highest std — assigned FIRST,
                       so a noisy small-sample AR(1) estimate of an iid
                       high-vol state can never capture another label;
    'mean_reverting' = of the remaining two, the one with the most
                       negative AR(1), PROVIDED it clears the margin
                       AR1_MEAN_REVERTING_MARGIN (sampling noise around
                       zero must not claim the label). If neither
                       remaining state clears the margin, the documented
                       tiebreak applies: the state with the SMALLER
                       within-state |mean| is 'mean_reverting' (trending
                       regimes are characterized by persistent drift);
    'trending'       = the last remaining state.
    Ties break by state index (lower index wins).

Determinism: seeded restarts (GaussianHMM(random_state=42 + k)) plus the
rules above — refitting on identical data always yields identical labels
and posteriors.

Degenerate-data robustness: constant / zero-variance feature matrices are
skipped before hmmlearn ever sees them, and any hmmlearn failure is
caught — in both cases the model logs a WARNING and stays unfitted
(predict returns {} until a later refit succeeds).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from desks.walk_forward import WalkForwardModel

logger = logging.getLogger(__name__)

#: The three regime labels (contract C5).
REGIME_LABELS = ('mean_reverting', 'trending', 'high_vol')

#: Feature column order of the market-level feature matrix.
FEATURE_COLUMNS = ('mkt_ret', 'ret_std_20', 'volume_ratio')

#: Rolling window for the volatility / volume-average features.
ROLLING_WINDOW = 20

#: Minimum feature rows required to attempt a 3-state full-covariance fit.
MIN_FIT_ROWS = 30

#: A state's AR(1) must be below this margin before it may claim
#: 'mean_reverting' — small-sample noise around zero (an iid state's
#: AR(1) estimate can easily reach ~ -0.07 by chance alone over a short
#: window, but the margin plus the high-vol-first ordering keeps such a
#: state from activating the mean-reversion book) must not capture the
#: label. Below the margin: documented |mean| tiebreak (module docstring).
AR1_MEAN_REVERTING_MARGIN = -0.05

#: Same-state consecutive pairs needed before the within-state AR(1)
#: trusts them; below this the estimator falls back to pairs entering
#: the state (robust to alternating-state splits, module docstring).
MIN_AR1_SAME_STATE_PAIRS = 10

#: Seeded EM restarts per fit (random_state + k, best log-likelihood
#: wins); deterministic protection against local optima.
N_FIT_RESTARTS = 3


class RegimeHMMModel(WalkForwardModel):
    """3-state Gaussian HMM over market-level features, seeded."""

    def __init__(self, n_components: int = 3, n_iter: int = 100,
                 random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: Optional[GaussianHMM] = None
        # state index -> regime label, set by the labeling rule at fit.
        self._state_labels: Dict[int, str] = {}
        # (column means, column stds) of the TRAINING window, applied to
        # every matrix the fitted HMM sees (fit and predict alike).
        self._scaler: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _market_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Market-level feature frame (FEATURE_COLUMNS order), NaN warm-up
        rows dropped. Empty frame when no usable inputs exist."""
        returns: Dict[str, pd.Series] = {}
        volume_ratios: Dict[str, pd.Series] = {}
        for symbol, frame in data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            returns[symbol] = frame['close'].pct_change()
            if 'volume' in frame.columns:
                volume_avg = frame['volume'].rolling(ROLLING_WINDOW).mean()
                volume_ratios[symbol] = frame['volume'] / volume_avg

        if not returns:
            return pd.DataFrame(columns=list(FEATURE_COLUMNS))

        mkt_ret = pd.DataFrame(returns).mean(axis=1)
        features = pd.DataFrame({'mkt_ret': mkt_ret})
        features['ret_std_20'] = mkt_ret.rolling(ROLLING_WINDOW).std()
        if volume_ratios:
            features['volume_ratio'] = pd.DataFrame(volume_ratios).mean(axis=1)
        else:
            features['volume_ratio'] = np.nan

        features = features.replace([np.inf, -np.inf], np.nan)
        return features.dropna()

    @staticmethod
    def _design_matrix(features: pd.DataFrame) -> np.ndarray:
        """Raw (pre-standardization) matrix the HMM models: feature
        values with ret_std_20 in log space (module docstring, FITTING).
        The floor keeps a zero rolling std finite."""
        x_matrix = features.to_numpy(dtype=float, copy=True)
        vol_col = FEATURE_COLUMNS.index('ret_std_20')
        x_matrix[:, vol_col] = np.log(
            np.maximum(x_matrix[:, vol_col], 1e-12))
        return x_matrix

    # ------------------------------------------------------------------
    # State labeling
    # ------------------------------------------------------------------
    @staticmethod
    def _ar1_slope(returns: np.ndarray,
                   current: np.ndarray) -> Optional[float]:
        """OLS slope of ret[t] on ret[t-1] over the pair end-positions in
        `current`; None with fewer than 2 pairs or a zero-variance
        regressor."""
        if current.size < 2:
            return None
        lagged = returns[current - 1]
        now = returns[current]
        lag_var = float(np.var(lagged))
        if lag_var <= 0.0:
            return None
        covariance = float(np.mean((lagged - lagged.mean())
                                   * (now - now.mean())))
        return covariance / lag_var

    @classmethod
    def _within_state_ar1(cls, returns: np.ndarray,
                          mask: np.ndarray) -> float:
        """AR(1) coefficient of `returns` within a state (module
        docstring, labeling rule (a)).

        Prefers consecutive pairs assigned WHOLLY to the state. With
        fewer than MIN_AR1_SAME_STATE_PAIRS of those — e.g. the HMM
        decoded one oscillating mean-reverting regime as two ALTERNATING
        states, which leaves (almost) no same-state pairs and would mask
        strongly negative autocorrelation behind a neutral 0.0 — it
        falls back to every pair whose CURRENT row is in the state,
        whatever the state at t-1: ret[t] responds to ret[t-1]
        regardless of how t-1 was decoded. 0.0 when even the fallback
        is unusable."""
        indices = np.flatnonzero(mask)
        if indices.size >= 2:
            # Positions where both t-1 and t are assigned to the state.
            same_state = indices[1:][np.diff(indices) == 1]
            if same_state.size >= MIN_AR1_SAME_STATE_PAIRS:
                slope = cls._ar1_slope(returns, same_state)
                if slope is not None:
                    return slope
        # Alternating-split fallback: pairs entering the state.
        slope = cls._ar1_slope(returns, indices[indices >= 1])
        return slope if slope is not None else 0.0

    def _label_states(self, returns: np.ndarray,
                      states: np.ndarray) -> Dict[int, str]:
        """Apply the deterministic labeling rule (module docstring)."""
        stats: List[Dict] = []
        for state in range(self.n_components):
            mask = states == state
            in_state = returns[mask]
            std = float(np.std(in_state)) if in_state.size >= 2 else 0.0
            ar1 = self._within_state_ar1(returns, mask)
            abs_mean = (float(abs(np.mean(in_state)))
                        if in_state.size > 0 else 0.0)
            stats.append({'state': state, 'ar1': ar1, 'std': std,
                          'abs_mean': abs_mean})

        # 'high_vol' FIRST: highest std; ties break by lower state index.
        # Removing the high-vol state before AR(1) is consulted stops an
        # iid high-vol state's small-sample AR(1) noise from capturing
        # the mean_reverting label (and thereby activating the
        # mean-reversion book during high volatility).
        high_vol = max(stats, key=lambda s: (s['std'], -s['state']))
        remaining = [s for s in stats if s['state'] != high_vol['state']]
        # 'mean_reverting': most negative AR(1) of the remaining two,
        # required to clear the negative margin; ties by lower index.
        candidate = min(remaining, key=lambda s: (s['ar1'], s['state']))
        if candidate['ar1'] < AR1_MEAN_REVERTING_MARGIN:
            mean_reverting = candidate
        else:
            # Documented tiebreak: neither state shows convincingly
            # negative autocorrelation — the one WITHOUT persistent
            # drift (smaller within-state |mean|) is mean_reverting.
            mean_reverting = min(remaining,
                                 key=lambda s: (s['abs_mean'], s['state']))
        trending = next(s for s in remaining
                        if s['state'] != mean_reverting['state'])

        labels = {
            mean_reverting['state']: 'mean_reverting',
            high_vol['state']: 'high_vol',
            trending['state']: 'trending',
        }
        logger.debug("Regime state labels: %s (stats %s)", labels, stats)
        return labels

    # ------------------------------------------------------------------
    # WalkForwardModel protocol
    # ------------------------------------------------------------------
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Fit the HMM on the controller-sliced training window.

        Failure semantics (WARNING logged in every case): on insufficient
        rows, degenerate (all-constant) features, or any hmmlearn failure
        the NEW fit is abandoned — if the model was never successfully
        fitted it stays unfitted (predict returns {}), while a previously
        fitted model is RETAINED so a transient EM failure at a refit
        does not kill the regime engine mid-run.
        """
        features = self._market_features(train_data)
        if len(features) < MIN_FIT_ROWS:
            logger.warning(
                "RegimeHMMModel.fit skipped: %d feature rows (< %d); %s",
                len(features), MIN_FIT_ROWS,
                "previous fit retained" if self._fitted else "unfitted")
            return

        x_matrix = self._design_matrix(features)
        if bool(np.all(np.std(x_matrix, axis=0) < 1e-12)):
            logger.warning(
                "RegimeHMMModel.fit skipped: degenerate (constant) "
                "features; %s",
                "previous fit retained" if self._fitted else "unfitted")
            return

        # Standardize on TRAINING statistics (stored for predict; a
        # zero-variance column scales by 1.0 and just centers to 0).
        col_mean = x_matrix.mean(axis=0)
        col_std = x_matrix.std(axis=0)
        col_std = np.where(col_std < 1e-12, 1.0, col_std)
        z_matrix = (x_matrix - col_mean) / col_std

        # Seeded multi-restart EM: best training log-likelihood wins,
        # strictly-greater comparison so the earliest seed wins ties.
        model: Optional[GaussianHMM] = None
        best_score = -np.inf
        last_error: Optional[Exception] = None
        for restart in range(N_FIT_RESTARTS):
            candidate = GaussianHMM(n_components=self.n_components,
                                    covariance_type='full',
                                    n_iter=self.n_iter,
                                    random_state=self.random_state + restart)
            try:
                candidate.fit(z_matrix)
                score = float(candidate.score(z_matrix))
            except Exception as exc:  # this restart failed; try the next
                last_error = exc
                continue
            if score > best_score:
                model, best_score = candidate, score
        if model is None:
            logger.warning(
                "RegimeHMMModel.fit failed (%s); %s", last_error,
                "previous fit retained" if self._fitted else "unfitted")
            return
        try:
            states = model.predict(z_matrix)
        except Exception as exc:
            logger.warning(
                "RegimeHMMModel.fit failed (%s); %s", exc,
                "previous fit retained" if self._fitted else "unfitted")
            return

        self._state_labels = self._label_states(
            features['mkt_ret'].to_numpy(dtype=float), states)
        self._scaler = (col_mean, col_std)
        self._model = model
        self._fitted = True
        logger.debug("RegimeHMMModel fitted on %d rows (log-likelihood "
                     "%.2f over %d restarts)",
                     len(features), best_score, N_FIT_RESTARTS)

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict:
        """Posterior regime at `date` from the final feature row.

        Returns {'state': label, 'probs': {label: float}} or {} when the
        model is unfitted or the data cannot produce a feature row
        (insufficient history — the desk treats {} as no-regime).
        """
        if not self._fitted or self._model is None or self._scaler is None:
            return {}
        features = self._market_features(data)
        if features.empty:
            return {}
        col_mean, col_std = self._scaler
        z_matrix = (self._design_matrix(features) - col_mean) / col_std
        try:
            posteriors = self._model.predict_proba(z_matrix)
        except Exception as exc:
            logger.warning("RegimeHMMModel.predict failed (%s)", exc)
            return {}
        last = posteriors[-1]
        probs = {self._state_labels[state]: float(last[state])
                 for state in range(self.n_components)}
        state = self._state_labels[int(np.argmax(last))]
        return {'state': state, 'probs': probs}
