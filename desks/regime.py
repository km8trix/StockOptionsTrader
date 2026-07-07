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

try:  # private hmmlearn internals power the incremental predict fast
    # path; absent or renamed -> permanent verbatim-path fallback.
    from hmmlearn import _hmmc
    from hmmlearn.utils import log_normalize
except ImportError:  # pragma: no cover - depends on hmmlearn build
    _hmmc = None
    log_normalize = None

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
        # --- incremental-predict cache (bit-identical fast path) ------
        # Bumped ONLY when a NEW fit is installed; a failed refit that
        # retains the previous model keeps the cache valid.
        self._fit_generation = 0
        # Generation whose z-matrix contains a non-finite row: the
        # verbatim path raises/warns/returns {} on EVERY call for the
        # rest of that generation, so the fast path stands aside for it.
        self._fast_disabled_gen: Optional[int] = None
        # Derived-state cache validated as an exact extension of the
        # controller-handed data on every call (see predict docstring).
        self._predict_cache: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _market_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Market-level feature frame (FEATURE_COLUMNS order), NaN warm-up
        rows dropped. Empty frame when no usable inputs exist."""
        return self._market_features_parts(data)[0]

    def _market_features_parts(
            self, data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Tuple[str, ...],
               Tuple[str, ...]]:
        """The _market_features construction — ONE construction shared by
        both predict paths — additionally exposing the raw (PRE-dropna)
        cross-sectional mean-return series and the exact symbol
        classification, which the fast-path cache needs (the rolling-std
        kernel is path-dependent, so it must always be fed the same full
        raw series the verbatim path feeds it)."""
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
            return (pd.DataFrame(columns=list(FEATURE_COLUMNS)), None,
                    (), ())

        mkt_ret = pd.DataFrame(returns).mean(axis=1)
        features = pd.DataFrame({'mkt_ret': mkt_ret})
        features['ret_std_20'] = mkt_ret.rolling(ROLLING_WINDOW).std()
        if volume_ratios:
            features['volume_ratio'] = pd.DataFrame(volume_ratios).mean(axis=1)
        else:
            features['volume_ratio'] = np.nan

        features = features.replace([np.inf, -np.inf], np.nan)
        return (features.dropna(), mkt_ret, tuple(returns),
                tuple(volume_ratios))

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
        # New model/scaler: the predict cache belongs to the previous
        # generation. Failed refits above RETAIN model+scaler and must
        # not reach this line, so their cache stays valid.
        self._fit_generation += 1
        self._predict_cache = None
        logger.debug("RegimeHMMModel fitted on %d rows (log-likelihood "
                     "%.2f over %d restarts)",
                     len(features), best_score, N_FIT_RESTARTS)

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict:
        """Posterior regime at `date` from the final feature row.

        Returns {'state': label, 'probs': {label: float}} or {} when the
        model is unfitted or the data cannot produce a feature row
        (insufficient history — the desk treats {} as no-regime).

        Served through an incremental forward-recursion cache when the
        handed frames are validated as an exact extension of the frames
        already consumed (append-only, the engine's access pattern).
        BIT-IDENTICAL to the verbatim path by construction: the smoothed
        posterior of the LAST row equals the normalized forward (filter)
        posterior exactly (the backward lattice's final row is exactly
        zero), the C++ forward lattice is bitwise prefix-stable under
        append, and the single new alpha step reuses hmmlearn's own
        _hmmc.forward_log via a 2-row call with a ones startprob
        (log(1.0) == 0.0, so row 0 IS the cached alpha) — NEVER a
        hand-rolled logaddexp replica (measured ~1-ulp wrong) and NEVER
        a trailing-window rolling recompute (path-dependent Kahan
        state); every new rolling value comes from a full-series pandas
        pass. Any validation failure rebuilds from scratch (cheaper than
        the verbatim path — no backward pass); any surprise falls back
        to the verbatim path, which is preserved as _predict_full."""
        if not self._fitted or self._model is None or self._scaler is None:
            return {}
        if (_hmmc is None or log_normalize is None
                or getattr(self._model, 'implementation', None) != 'log'
                or self._fast_disabled_gen == self._fit_generation):
            return self._predict_full(data, date)
        try:
            result = self._predict_fast(data)
        except Exception:
            self._predict_cache = None
            result = None
        if result is None:  # refused ({} is a real answer, not a refusal)
            return self._predict_full(data, date)
        return result

    def _predict_full(self, data: Dict[str, pd.DataFrame], date) -> Dict:
        """The original predict body, preserved VERBATIM: the fallback
        for anything the fast path refuses and the oracle the
        equivalence tests compare against."""
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

    # ------------------------------------------------------------------
    # Incremental predict fast path
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(data: Dict[str, pd.DataFrame]):
        """Symbol classification exactly as the feature construction
        sees it: ordered tuples of close-bearing and volume-bearing
        symbols (dict iteration order is load-bearing — it fixes the
        cross-sectional column order and therefore the mean bits)."""
        close_syms = []
        vol_syms = []
        for symbol, frame in data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            close_syms.append(symbol)
            if 'volume' in frame.columns:
                vol_syms.append(symbol)
        return tuple(close_syms), tuple(vol_syms)

    @staticmethod
    def _same_value(a: float, b: float) -> bool:
        """NaN-aware scalar equality for the endpoint spot checks."""
        return a == b or (a != a and b != b)

    def _predict_fast(self, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Cache-validated incremental predict. Returns the result dict
        ({} included) or None to refuse to the verbatim path."""
        cache = self._predict_cache
        if cache is None or cache['gen'] != self._fit_generation:
            return self._rebuild_cache(data)
        close_syms, vol_syms = self._classify(data)
        if (close_syms != cache['close_syms']
                or vol_syms != cache['vol_syms']):
            return self._rebuild_cache(data)
        # Uniform-growth validation: every close symbol grew by the same
        # n_new rows on top of a verified endpoint, and for n_new == 1
        # all new tail dates are identical (one new union row). Arrays
        # are fetched ONCE per symbol here and reused below — the
        # accessors, not the checks, are the validation cost.
        vol_set = set(vol_syms)
        closes_np: Dict[str, np.ndarray] = {}
        volume_ser: Dict[str, pd.Series] = {}
        n_new = None
        new_tail = None
        for symbol in close_syms:
            frame = data[symbol]
            meta = cache['meta'][symbol]
            n_prev, first_ts, last_ts, last_close, last_vol = meta
            n = len(frame)
            if n < n_prev or frame.index[0] != first_ts:
                return self._rebuild_cache(data)
            grew = n - n_prev
            if n_new is None:
                n_new = grew
            elif grew != n_new:
                return self._rebuild_cache(data)
            if frame.index[n_prev - 1] != last_ts:
                return self._rebuild_cache(data)
            closes = frame['close'].to_numpy()
            closes_np[symbol] = closes
            if not self._same_value(float(closes[n_prev - 1]), last_close):
                return self._rebuild_cache(data)
            if symbol in vol_set:
                volume = frame['volume']
                volume_ser[symbol] = volume
                if last_vol is not None and not self._same_value(
                        float(volume.to_numpy()[n_prev - 1]), last_vol):
                    return self._rebuild_cache(data)
            if grew:
                if new_tail is None:
                    new_tail = frame.index[-1]
                elif frame.index[-1] != new_tail:
                    return self._rebuild_cache(data)
        if n_new == 0:
            if cache['n_feat'] == 0:
                return {}
            return self._posterior_from_alpha(cache['alpha'])
        if n_new != 1:
            return self._rebuild_cache(data)

        # ---- one new row per symbol: extend features incrementally ----
        col_mean, col_std = self._scaler
        ret_last: Dict[str, np.float64] = {}
        vr_last: Dict[str, np.float64] = {}
        vol_np: Dict[str, np.ndarray] = {}
        with np.errstate(all='ignore'):  # exactly pandas' wrap
            for symbol in close_syms:
                closes = closes_np[symbol]
                ret_last[symbol] = (
                    np.float64(closes[-1] / closes[-2] - 1.0)
                    if len(closes) >= 2 else np.float64(np.nan))
            for symbol in vol_syms:
                volume = volume_ser[symbol]
                # FULL-series pandas pass: the rolling kernel's Kahan
                # state is path-dependent, a trailing window is ~1 ulp
                # wrong — only the full pass is bit-identical.
                vol_avg = volume.rolling(ROLLING_WINDOW).mean().to_numpy()
                vals = volume.to_numpy()
                vol_np[symbol] = vals
                vr_last[symbol] = np.float64(vals[-1] / vol_avg[-1])
        mkt_new = np.float64(
            pd.DataFrame(ret_last, index=[0]).mean(axis=1).iloc[0])
        vr_new = (np.float64(
            pd.DataFrame(vr_last, index=[0]).mean(axis=1).iloc[0])
            if vol_syms else np.float64(np.nan))
        mkt_raw = np.concatenate([cache['mkt_raw'], [mkt_new]])
        ret_std_new = np.float64(pd.Series(mkt_raw)
                                 .rolling(ROLLING_WINDOW).std()
                                 .to_numpy()[-1])

        alpha = cache['alpha']
        n_feat = cache['n_feat']
        # replace(±inf -> NaN) + dropna: the row survives iff every
        # feature is finite; a dropped row leaves the design matrix —
        # and therefore alpha — untouched.
        if np.isfinite([mkt_new, ret_std_new, vr_new]).all():
            frow = pd.DataFrame({'mkt_ret': [mkt_new],
                                 'ret_std_20': [ret_std_new],
                                 'volume_ratio': [vr_new]})
            z_row = (self._design_matrix(frow) - col_mean) / col_std
            if not np.isfinite(z_row).all():
                # The verbatim path's full matrix now contains this row
                # and raises/warns/{} on every later call this
                # generation — stand aside for the whole generation.
                self._fast_disabled_gen = self._fit_generation
                self._predict_cache = None
                return None
            frame_logprob = self._model._compute_log_likelihood(z_row)
            if n_feat == 0:
                _, fwd = _hmmc.forward_log(
                    self._model.startprob_, self._model.transmat_,
                    frame_logprob)
                alpha = fwd[-1]
            else:
                # ones startprob: log(1.0) == 0.0 exactly, so row 0 of
                # this 2-row lattice IS the cached alpha and row 1 is
                # produced by the same C++ recursion as the full pass.
                _, fwd = _hmmc.forward_log(
                    np.ones(self._model.n_components),
                    self._model.transmat_,
                    np.vstack([alpha, frame_logprob[0]]))
                alpha = fwd[1]
            n_feat += 1

        # Commit only after every step succeeded.
        for symbol in close_syms:
            frame = data[symbol]
            closes = closes_np[symbol]
            vols = vol_np.get(symbol)
            cache['meta'][symbol] = (
                len(frame), frame.index[0], frame.index[-1],
                float(closes[-1]),
                float(vols[-1]) if vols is not None else None)
        cache['mkt_raw'] = mkt_raw
        cache['alpha'] = alpha
        cache['n_feat'] = n_feat
        if n_feat == 0:
            return {}
        return self._posterior_from_alpha(alpha)

    def _rebuild_cache(self, data: Dict[str, pd.DataFrame]
                       ) -> Optional[Dict]:
        """Cold start or any validation failure: rebuild from the full
        construction (one forward-only pass — cheaper than the verbatim
        path's forward+backward), then answer from the cache."""
        self._predict_cache = None
        features, mkt_raw, close_syms, vol_syms = \
            self._market_features_parts(data)
        col_mean, col_std = self._scaler
        alpha = None
        n_feat = len(features)
        if n_feat:
            z_matrix = (self._design_matrix(features) - col_mean) / col_std
            if not np.isfinite(z_matrix).all():
                # The verbatim path raises/warns/{} on this generation's
                # every call from now on — stand aside entirely.
                self._fast_disabled_gen = self._fit_generation
                return None
            frame_logprob = self._model._compute_log_likelihood(z_matrix)
            _, fwd = _hmmc.forward_log(
                self._model.startprob_, self._model.transmat_,
                frame_logprob)
            alpha = fwd[-1]
        meta = {}
        for symbol in close_syms:
            frame = data[symbol]
            closes = frame['close'].to_numpy()
            vols = (data[symbol]['volume'].to_numpy()
                    if symbol in set(vol_syms) else None)
            meta[symbol] = (
                len(frame), frame.index[0], frame.index[-1],
                float(closes[-1]),
                float(vols[-1]) if vols is not None else None)
        self._predict_cache = {
            'gen': self._fit_generation,
            'close_syms': close_syms,
            'vol_syms': vol_syms,
            'meta': meta,
            'mkt_raw': (mkt_raw.to_numpy(dtype=np.float64, copy=True)
                        if mkt_raw is not None
                        else np.empty(0, dtype=np.float64)),
            'alpha': alpha,
            'n_feat': n_feat,
        }
        if n_feat == 0:
            return {}
        return self._posterior_from_alpha(alpha)

    def _posterior_from_alpha(self, alpha: np.ndarray) -> Dict:
        """{'state','probs'} from the cached forward log-alphas — the
        backward lattice's final row is exactly zero, so this equals
        predict_proba's last smoothed row bitwise (hmmlearn's own
        log_normalize does the normalization)."""
        log_gamma = alpha[None, :] + 0.0  # + backward row of exact zeros
        log_normalize(log_gamma, axis=1)
        with np.errstate(under='ignore'):
            last = np.exp(log_gamma)[0]
        probs = {self._state_labels[state]: float(last[state])
                 for state in range(self.n_components)}
        state = self._state_labels[int(np.argmax(last))]
        return {'state': state, 'probs': probs}
