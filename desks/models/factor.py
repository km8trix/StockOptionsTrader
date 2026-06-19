"""
Factor model (AQR desk) — a TRANSPARENT, price-based cross-sectional factor
model, the classical-quant counterpart to the Two Sigma ML zoo.

WHERE THE ML MODELS ARE A BLACK BOX, THIS IS A GLASS BOX. Every score is a
small linear combination of a handful of named, economically-motivated
price factors, each standardized cross-sectionally per date so the
combination is a relative (market-neutral) ranking signal. The combination
weights are fitted by RIDGE regression of next-day return on the pooled
standardized factor exposures, and they are EXPOSED for attribution
(``factor_weights``) so the AQR desk can report which factors drive the
book — the AQR transparency signature.

FACTORS (OHLCV-only, all strictly BACKWARD-LOOKING; each row at timestamp t
uses only rows <= t, so a frame already sliced by WalkForwardController is
leakage-free):

    momentum      12-1 momentum: the return from ~t-252 to ~t-21, i.e. the
                  trailing ~12-month return SKIPPING the most recent ~21
                  days (one month). Economic sign POSITIVE: cross-sectional
                  winners keep winning (the classic UMD/momentum premium);
                  the one-month skip avoids contaminating it with the
                  short-term reversal below.
    reversal      short-term 1-month reversal: the NEGATIVE of the last ~21
                  trading days' return. Economic sign POSITIVE as an
                  exposure (we negate the raw return) because last month's
                  losers tend to bounce (short-horizon mean reversion), so a
                  recent loser scores HIGH.
    low_vol       low-volatility: the NEGATIVE of the trailing ~60-day
                  realized vol of daily returns. Economic sign POSITIVE as
                  an exposure (we negate vol) so LOW-volatility names score
                  HIGH — the low-volatility / betting-against-beta anomaly
                  (low-risk names earn higher risk-adjusted returns).
    risk_adj_mom  risk-adjusted momentum: the 12-1 momentum DIVIDED by the
                  trailing ~60-day vol. Economic sign POSITIVE: a steadier
                  uptrend (high return per unit of risk) is a stronger,
                  more persistent signal than raw momentum alone.

Each raw factor is standardized CROSS-SECTIONALLY per date (z-score across
symbols at that timestamp) with :func:`desks.features.cross_sectional_rank`
— a no-lookahead, row-wise-across-symbols transform — so the composite is a
relative ranking, comparable across dates and regimes.

COMBINATION (ridge): training pools, over the train window, each symbol's
standardized factor exposures at date t against its golden next-day-return
label (``close[t+1]/close[t] - 1``); the last row per symbol has no next
return and is excluded, identical to ``GradientBoostingModel``. A
scikit-learn ``Ridge`` (fixed ``alpha``, pinned deterministic
``solver='cholesky'``, ``fit_intercept=True``) learns one weight per factor;
the intercept is intentionally DROPPED at predict (only the per-factor
coefficients are exposed/used) because the desk ranks scores
cross-sectionally, so a constant offset is inert. ``predict`` returns, per symbol, the weighted
sum of its CURRENT standardized exposures — a composite score the desk
ranks cross-sectionally, so absolute scale is irrelevant.

DEGRADE (never crash a run; documented choice = EQUAL-WEIGHT FALLBACK):
insufficient history, a degenerate cross-section (no spread to standardize),
or a ridge solve failure leaves the model FITTED with an equal-weight
composite (every factor weight = 1.0). Rationale: the factors are already
economically signed (each is constructed so HIGHER = more attractive), so an
unweighted average is a sound, fully-transparent prior — the desk still gets
a usable ranking on day one and whenever data is too thin to estimate
weights, rather than going dark. A WARNING is logged each time the fallback
is taken. (Contrast the ML models, which stay UNFITTED -> ``{}`` on
degrade; the factor model's hand-signed factors make an equal-weight prior
meaningful, which the learned ML scores do not have.)

DETERMINISM: pure pandas/numpy reductions plus a seeded, single deterministic
ridge solve — identical input frames yield byte-identical weights and
scores across runs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from desks import features as feature_lib
from desks.walk_forward import WalkForwardModel

logger = logging.getLogger(__name__)

#: Factor column order (also the attribution order). Each factor is built so
#: that HIGHER raw value = MORE attractive (winners / recent losers / low
#: vol / steady trend), so an equal-weight average is a meaningful prior.
FACTOR_COLUMNS: Sequence[str] = (
    'momentum', 'reversal', 'low_vol', 'risk_adj_mom')

#: Lookback constants (trading days). Degraded gracefully on short history.
_MOM_LOOKBACK = 252   # ~12 months
_MOM_SKIP = 21        # skip the most recent ~1 month (12-1 momentum)
_REVERSAL_LOOKBACK = 21   # ~1 month short-term reversal
_VOL_LOOKBACK = 60        # ~3 months realized-vol window

#: Minimum trailing rows a frame needs before ANY factor can be formed (we
#: need at least a couple of returns to compute a vol / a short return). Far
#: below the full lookbacks — short history simply degrades each factor to
#: the longest window the frame can support (see ``_factor_exposures``).
_MIN_ROWS = 5


def _trailing_return(close: pd.Series, lookback: int, skip: int = 0) -> float:
    """Backward-looking simple return over ``[t-lookback-skip, t-skip]``.

    Degrades on short history: clamps ``lookback``/``skip`` to what the
    series can support. Returns ``np.nan`` when fewer than two usable
    points remain (no return is defined).
    """
    n = len(close)
    if n < 2:
        return float('nan')
    end_idx = n - 1 - skip
    if end_idx < 1:
        # Not enough room to skip a full month; fall back to no skip so a
        # short frame still yields a (shorter-horizon) momentum reading.
        end_idx = n - 1
    start_idx = end_idx - lookback
    if start_idx < 0:
        start_idx = 0
    if start_idx >= end_idx:
        return float('nan')
    start_px = close.iloc[start_idx]
    end_px = close.iloc[end_idx]
    if not np.isfinite(start_px) or start_px == 0 or not np.isfinite(end_px):
        return float('nan')
    return float(end_px / start_px - 1.0)


def _trailing_vol(close: pd.Series, lookback: int) -> float:
    """Std of daily returns over the trailing ``lookback`` rows.

    Degrades to whatever window the frame supports (down to ``_MIN_ROWS``
    returns). Returns ``np.nan`` on insufficient/degenerate data.
    """
    rets = close.pct_change().dropna()
    if len(rets) < 2:
        return float('nan')
    window = rets.tail(lookback)
    vol = float(window.std())
    if not np.isfinite(vol):
        return float('nan')
    return vol


def _factor_exposures(close: pd.Series) -> Optional[Dict[str, float]]:
    """The four raw (pre-standardization) factor readings at the LATEST row.

    Every factor is signed so HIGHER = MORE attractive. Returns ``None``
    when the frame is too short to form any usable factor, so the symbol is
    simply omitted from the cross-section that day.
    """
    if close is None or len(close) < _MIN_ROWS:
        return None
    close = close.astype(float)

    mom = _trailing_return(close, _MOM_LOOKBACK, skip=_MOM_SKIP)
    rev_raw = _trailing_return(close, _REVERSAL_LOOKBACK, skip=0)
    vol = _trailing_vol(close, _VOL_LOOKBACK)

    if not np.isfinite(mom) and not np.isfinite(rev_raw) \
            and not np.isfinite(vol):
        return None

    exposures: Dict[str, float] = {}
    # momentum: POSITIVE sign (winners keep winning).
    exposures['momentum'] = mom if np.isfinite(mom) else 0.0
    # reversal: NEGATIVE of last month's return (recent losers bounce).
    exposures['reversal'] = -rev_raw if np.isfinite(rev_raw) else 0.0
    # low_vol: NEGATIVE of realized vol (low-vol names score higher).
    exposures['low_vol'] = -vol if np.isfinite(vol) else 0.0
    # risk_adj_mom: momentum / vol (steady trend). Guard tiny/zero vol.
    if np.isfinite(mom) and np.isfinite(vol) and vol > 1e-9:
        exposures['risk_adj_mom'] = mom / vol
    else:
        exposures['risk_adj_mom'] = 0.0
    return exposures


def _raw_factor_panel(
        data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Per-symbol frame of RAW factor columns over ALL rows.

    Each factor at row t is computed from the prefix ``close[:t+1]`` only
    (backward-looking), so the panel is leakage-free and is the input to the
    per-date cross-sectional standardization. Symbols too short to form any
    factor are dropped.
    """
    panel: Dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        if frame is None or frame.empty or 'close' not in frame.columns:
            continue
        close = frame['close'].astype(float)
        n = len(close)
        if n < _MIN_ROWS:
            continue
        rows: Dict[str, List[float]] = {c: [] for c in FACTOR_COLUMNS}
        idx: List = []
        for i in range(n):
            exp = _factor_exposures(close.iloc[:i + 1])
            if exp is None:
                continue
            idx.append(close.index[i])
            for c in FACTOR_COLUMNS:
                rows[c].append(exp[c])
        if not idx:
            continue
        panel[symbol] = pd.DataFrame(rows, index=pd.Index(idx))
    return panel


def _standardized_panel(
        raw_panel: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Cross-sectionally z-score each factor per date across symbols.

    Uses :func:`desks.features.cross_sectional_rank` (no-lookahead, row-wise
    across symbols) once per factor, then re-assembles a per-symbol frame of
    standardized exposures aligned on each symbol's own dates.
    """
    if not raw_panel:
        return {}
    # column -> DataFrame(date x symbol) of the cross-sectional z-score.
    z_by_factor: Dict[str, pd.DataFrame] = {}
    for col in FACTOR_COLUMNS:
        z_by_factor[col] = feature_lib.cross_sectional_rank(
            raw_panel, column=col, method='zscore')

    standardized: Dict[str, pd.DataFrame] = {}
    for symbol, frame in raw_panel.items():
        cols: Dict[str, pd.Series] = {}
        for col in FACTOR_COLUMNS:
            zf = z_by_factor[col]
            if symbol in zf.columns:
                cols[col] = zf[symbol].reindex(frame.index)
            else:
                cols[col] = pd.Series(np.nan, index=frame.index)
        std_frame = pd.DataFrame(cols, index=frame.index)
        std_frame = std_frame.reindex(columns=list(FACTOR_COLUMNS))
        standardized[symbol] = std_frame
    return standardized


class FactorModel(WalkForwardModel):
    """Transparent price-based cross-sectional factor model (module docstring).

    Combines momentum / reversal / low-vol / risk-adjusted-momentum factors,
    each standardized cross-sectionally per date, by ridge regression of the
    next-day return on the pooled standardized exposures. Degrades to an
    equal-weight composite (never unfitted, never crashing). Exposes the
    fitted weights via :attr:`factor_weights` for AQR attribution.
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        #: factor name -> fitted weight. Empty until the first fit.
        self._weights: Dict[str, float] = {}
        self._fitted = False
        #: True when the last fit fell back to the equal-weight prior.
        self._degraded = False

    # ------------------------------------------------------------------
    # Attribution (AQR transparency)
    # ------------------------------------------------------------------
    @property
    def factor_weights(self) -> Dict[str, float]:
        """The fitted weight per factor (the AQR attribution surface).

        Returns a COPY in :data:`FACTOR_COLUMNS` order; empty before the
        first fit. The AQR desk reports these on each refit so the operator
        sees exactly which factors drive the book.
        """
        return {c: self._weights[c] for c in FACTOR_COLUMNS
                if c in self._weights}

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def is_degraded(self) -> bool:
        """True when the current fit is the equal-weight fallback."""
        return self._degraded

    def _equal_weight(self) -> None:
        """Adopt the transparent equal-weight prior (every factor = 1.0)."""
        self._weights = {c: 1.0 for c in FACTOR_COLUMNS}
        self._fitted = True
        self._degraded = True

    # ------------------------------------------------------------------
    # WalkForwardModel protocol
    # ------------------------------------------------------------------
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Fit ridge factor weights on the pooled, standardized exposures.

        Golden label alignment: ``label[i] = close[i+1]/close[i] - 1`` per
        symbol; the final row of each frame (no next return) is excluded.
        Degrades to the equal-weight prior (WARNING) on insufficient data, a
        degenerate cross-section, or a ridge failure — never crashes, never
        stays unfitted.
        """
        raw_panel = _raw_factor_panel(train_data)
        if not raw_panel:
            logger.warning(
                "FactorModel.fit: no symbol had enough history to form "
                "factors — falling back to the equal-weight composite")
            self._equal_weight()
            return

        standardized = _standardized_panel(raw_panel)

        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        for symbol, std_frame in standardized.items():
            frame = train_data.get(symbol)
            if frame is None or 'close' not in frame.columns:
                continue
            close = frame['close'].astype(float)
            next_return = close.shift(-1) / close - 1.0
            # Golden alignment: drop the last row (no next return); keep only
            # rows where every standardized factor is defined.
            exposures = std_frame.dropna()
            usable = exposures.index.intersection(next_return.dropna().index)
            if usable.empty:
                continue
            x_parts.append(exposures.loc[usable].to_numpy(dtype=float))
            y_parts.append(next_return.loc[usable].to_numpy(dtype=float))

        if not x_parts:
            logger.warning(
                "FactorModel.fit: no usable (exposure, next-return) rows "
                "after standardization — falling back to equal-weight")
            self._equal_weight()
            return

        x_matrix = np.vstack(x_parts)
        y = np.concatenate(y_parts)
        # A degenerate cross-section produces all-zero exposures (no spread
        # to standardize): ridge would learn nothing meaningful, so use the
        # transparent prior instead.
        if x_matrix.shape[0] < 2 or not np.any(np.abs(x_matrix) > 1e-12):
            logger.warning(
                "FactorModel.fit: degenerate exposures (%d rows, no spread) "
                "— falling back to equal-weight", x_matrix.shape[0])
            self._equal_weight()
            return

        # solver='cholesky' is pinned (not 'auto') so the determinism
        # guarantee is robust to any future change in how sklearn resolves
        # 'auto'; for these dense, full-rank inputs 'auto' already resolves to
        # cholesky, so this is behavior-identical. fit_intercept=True absorbs
        # the (non-zero) mean pooled exposure / mean forward return into the
        # intercept rather than the slopes; the intercept is intentionally
        # DROPPED at predict (only coef_ is used as factor_weights), because
        # the desk ranks scores cross-sectionally so a constant offset is inert.
        ridge = Ridge(alpha=self.alpha, fit_intercept=True,
                      solver='cholesky', random_state=self.random_state)
        try:
            ridge.fit(x_matrix, y)
        except Exception as exc:  # noqa: BLE001 - never crash a run
            logger.warning(
                "FactorModel.fit: ridge solve failed (%s) — falling back to "
                "equal-weight", exc)
            self._equal_weight()
            return

        coefs = np.asarray(ridge.coef_, dtype=float).ravel()
        if coefs.shape[0] != len(FACTOR_COLUMNS) \
                or not np.all(np.isfinite(coefs)):
            logger.warning(
                "FactorModel.fit: ridge produced non-finite/mismatched "
                "coefficients — falling back to equal-weight")
            self._equal_weight()
            return

        self._weights = {c: float(w)
                         for c, w in zip(FACTOR_COLUMNS, coefs)}
        self._fitted = True
        self._degraded = False
        logger.debug("FactorModel fitted: weights %s (n=%d)",
                     self._weights, x_matrix.shape[0])

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol composite = weighted sum of CURRENT standardized factors.

        Builds the raw factor panel, standardizes each factor cross-
        sectionally per date, then dots the LATEST row's standardized
        exposures with the fitted weights. Symbols too short to form factors
        are omitted; ``{}`` when somehow unfitted (the equal-weight fallback
        normally guarantees a usable model, so this only fires if ``fit`` was
        never called).
        """
        if not self._fitted or not self._weights:
            return {}
        raw_panel = _raw_factor_panel(data)
        if not raw_panel:
            return {}
        standardized = _standardized_panel(raw_panel)
        weights = np.array([self._weights[c] for c in FACTOR_COLUMNS],
                           dtype=float)
        scores: Dict[str, float] = {}
        for symbol, std_frame in standardized.items():
            if std_frame.empty:
                continue
            latest = std_frame.iloc[-1]
            # Treat an absent (still-warming) factor as 0 (neutral on that
            # axis) rather than dropping the whole name — keeps the universe
            # wide while a long lookback warms up.
            exposure = latest.reindex(FACTOR_COLUMNS).fillna(0.0).to_numpy(
                dtype=float)
            if not np.all(np.isfinite(exposure)):
                continue
            scores[symbol] = float(np.dot(weights, exposure))
        return scores


__all__ = ['FactorModel', 'FACTOR_COLUMNS']
