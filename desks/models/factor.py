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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
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


def _trailing_return_series(
        close_vals: np.ndarray, lookback: int, skip: int) -> np.ndarray:
    """Vectorized per-row equivalent of :func:`_trailing_return` over EVERY
    prefix ``close[:i+1]`` of ``close_vals``, in O(n) for the whole series.

    Row ``i`` reproduces ``_trailing_return(close[:i+1], lookback, skip)``
    bit-for-bit, including the documented short-history edge cases:

      * ``i == 0`` (prefix length < 2) -> NaN;
      * the skip->no-skip fallback when there isn't a full ``skip`` of room
        (the loop's ``end_idx = n-1-skip < 1`` branch, here ``i-skip < 1``,
        falls back to ``end = i``);
      * the start-index clamp to 0 on short history;
      * ``start >= end`` -> NaN;
      * the finite / non-zero start-price and finite end-price guards.

    The per-row arithmetic is the identical IEEE ``end_px/start_px - 1.0``,
    so the result is byte-identical to the prefix loop.
    """
    n = close_vals.shape[0]
    i = np.arange(n)
    end = i - skip
    end = np.where(end < 1, i, end)          # skip -> no-skip fallback
    start = np.maximum(end - lookback, 0)     # clamp start to 0
    valid = (i >= 1) & (start < end)
    start_px = close_vals[start]
    end_px = close_vals[end]
    ok = (valid & np.isfinite(start_px) & (start_px != 0.0)
          & np.isfinite(end_px))
    out = np.full(n, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = end_px / start_px - 1.0
    out[ok] = ratio[ok]
    return out


def _trailing_vol_series(close: pd.Series, lookback: int) -> np.ndarray:
    """Vectorized per-row equivalent of :func:`_trailing_vol`, in O(n).

    Row ``i`` reproduces ``_trailing_vol(close[:i+1], lookback)`` exactly:
    ``close[:i+1].pct_change().dropna().tail(lookback).std()`` (ddof=1) with
    the same NaN-drop, the same trailing-window-of-clean-returns selection,
    and the same ``<2 returns`` / non-finite -> NaN guards.

    Because ``pct_change`` only looks back one row, the prefix's clean
    returns are simply the global clean-return stream truncated to positions
    ``<= i``; ``tail(lookback)`` of that prefix is therefore a trailing
    window over the clean-return stream. Each distinct window's std is
    computed once (the long full-length tail is batched via a strided view),
    then mapped back to rows by the running count of clean returns. Both the
    per-window ``np.std(..., ddof=1)`` and the batched two-pass std are
    byte-identical to ``pandas.Series.std`` (see ``tests/test_factor_model``
    golden check).
    """
    n = len(close)
    rets = close.pct_change()
    notna = rets.notna().to_numpy()
    # cum[i] = number of clean (non-NaN) returns at positions <= i, i.e. the
    # ``len(rets)`` the loop's _trailing_vol sees for the prefix close[:i+1].
    cum = np.cumsum(notna)
    clean = rets.to_numpy()[notna]            # clean-return stream (order kept)
    k = clean.shape[0]

    # vol_j[j] = std of the trailing window of clean returns ENDING at clean
    # index j (the window is clean[max(0, j-lookback+1) : j+1]).
    vol_j = np.full(k, np.nan)
    # A non-finite return (e.g. an inf from a zero price) makes its windows'
    # std non-finite; that is squashed to NaN by the finite guard below, just
    # as the loop's ``np.isfinite(vol)`` check does, so the arithmetic warning
    # it raises along the way carries no information — silence it.
    with np.errstate(invalid='ignore'):
        if k >= lookback:
            # Warm-up windows (length 2 .. lookback-1): the whole clean prefix.
            for j in range(1, lookback - 1):
                vol_j[j] = np.std(clean[:j + 1], ddof=1)
            # Full-length windows (length == lookback): batched two-pass std,
            # byte-identical to per-window np.std on each contiguous row.
            sw = sliding_window_view(clean, lookback)   # row m -> clean[m:m+L]
            means = sw.mean(axis=1, keepdims=True)
            var = ((sw - means) ** 2).sum(axis=1) / (lookback - 1)
            vol_j[lookback - 1:] = np.sqrt(var)
        else:
            for j in range(1, k):
                vol_j[j] = np.std(clean[:j + 1], ddof=1)
    vol_j[~np.isfinite(vol_j)] = np.nan           # finite guard

    out = np.full(n, np.nan)
    jj = cum - 1                                   # clean index of latest return
    has = jj >= 1                                  # need >= 2 clean returns
    out[has] = vol_j[jj[has]]
    return out


def _raw_factor_panel(
        data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Per-symbol frame of RAW factor columns over ALL rows.

    Each factor at row t is computed from the prefix ``close[:t+1]`` only
    (backward-looking), so the panel is leakage-free and is the input to the
    per-date cross-sectional standardization. Symbols too short to form any
    factor are dropped.

    Vectorized: rather than re-running :func:`_factor_exposures` on a growing
    prefix for every row (O(n^2) per symbol), the four factors are computed
    over the whole series at once in O(n) via :func:`_trailing_return_series`
    and :func:`_trailing_vol_series`. The output is byte-identical to the
    former prefix loop (guarded by the golden test in
    ``tests/test_factor_model.py``).
    """
    panel: Dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        if frame is None or frame.empty or 'close' not in frame.columns:
            continue
        close = frame['close'].astype(float)
        n = len(close)
        if n < _MIN_ROWS:
            continue
        close_vals = close.to_numpy(dtype=float)

        mom = _trailing_return_series(close_vals, _MOM_LOOKBACK, _MOM_SKIP)
        rev_raw = _trailing_return_series(close_vals, _REVERSAL_LOOKBACK, 0)
        vol = _trailing_vol_series(close, _VOL_LOOKBACK)

        finite_mom = np.isfinite(mom)
        finite_rev = np.isfinite(rev_raw)
        finite_vol = np.isfinite(vol)

        # Same signing as _factor_exposures (HIGHER = more attractive), with
        # a non-finite factor treated as 0.0 on that axis.
        with np.errstate(divide='ignore', invalid='ignore'):
            risk = np.where(finite_mom & finite_vol & (vol > 1e-9),
                            mom / vol, 0.0)
        col_arrays = {
            'momentum': np.where(finite_mom, mom, 0.0),
            'reversal': np.where(finite_rev, -rev_raw, 0.0),
            'low_vol': np.where(finite_vol, -vol, 0.0),
            'risk_adj_mom': risk,
        }

        # _factor_exposures returns a row only when the prefix has >= _MIN_ROWS
        # rows (i >= _MIN_ROWS-1) AND at least one factor is finite.
        keep = ((np.arange(n) >= _MIN_ROWS - 1)
                & (finite_mom | finite_rev | finite_vol))
        if not keep.any():
            continue
        # Build the index exactly as the former loop did (``pd.Index`` over a
        # list of the kept timestamps) so a DatetimeIndex carries no ``freq``
        # — a boolean mask would otherwise preserve it and diverge.
        panel[symbol] = pd.DataFrame(
            {c: col_arrays[c][keep] for c in FACTOR_COLUMNS},
            index=pd.Index(close.index[keep].tolist()))
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


#: Safe tail length (rows) beyond each symbol's needed-date span for the
#: predict-site truncation fast path. Momentum's full positional reach at a
#: consumed row is ``lookback + skip`` = 273 rows; +2 covers the pad row and
#: one row of margin. A consumed row at truncated position >= _SAFE_TAIL
#: therefore reads every input it reads in the full frame at IDENTICAL
#: relative offsets (its ``start = end - lookback`` index never clamps to
#: the truncation boundary), and its 60-return vol window (61 rows,
#: far below _SAFE_TAIL) sits wholly inside the guarded finite tail.
_SAFE_TAIL = _MOM_LOOKBACK + _MOM_SKIP + 2


def _truncated_predict_data(
        data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, pd.DataFrame]]:
    """Tail-truncated copy of ``data`` for predict, or ``None`` = full build.

    ``FactorModel.predict`` consumes ONLY each symbol's rows at the needed
    dates (every symbol's last panel date, plus one pad date when they all
    coincide — see ``_score_sliced``), yet ``_raw_factor_panel``
    costs O(history) per symbol. Truncating each frame to its last
    ``k_sym = (rows back to the oldest needed date) + _SAFE_TAIL`` rows
    keeps every consumed row's factor inputs at identical relative offsets,
    so the sliced build is byte-identical to the full one — PROVIDED the
    guards below hold. Any guard failing anywhere returns ``None`` and
    predict runs the verbatim full-history build for the WHOLE call
    (cross-sectional coupling makes per-symbol mixing unsafe: one wrong
    mid-tail row would shift a needed date's z-scores for every symbol).

    Guards (all per call — the helper is stateless):

      * every eligible frame's index is monotonic increasing and unique
        (duplicate/unsorted timestamps -> full build);
      * every eligible frame's closes over its last ``min(n, k_sym)`` rows
        are finite and non-zero. This both keeps each consumed row's vol
        window 60 contiguous clean returns wholly inside the tail AND makes
        every frame's LAST rows formable, so the needed dates computed here
        from ``frame.index`` provably equal the panel's post-build last
        dates (a NaN tail would move a symbol's last FORMABLE date
        arbitrarily far back — the trap this guard exists for);
      * mixed index types, tz mismatches, or any other surprise raise
        inside and are swallowed into the full-build fallback.

    Ineligible frames (None/empty/no close/under ``_MIN_ROWS``) pass through
    untouched — ``_raw_factor_panel`` skips them identically either way.
    Returns ``None`` too when no frame is long enough to truncate (no win).
    """
    try:
        eligible: Dict[str, pd.DataFrame] = {}
        for symbol, frame in data.items():
            if frame is None or frame.empty \
                    or 'close' not in frame.columns \
                    or len(frame) < _MIN_ROWS:
                continue
            index = frame.index
            if not (index.is_monotonic_increasing and index.is_unique):
                return None
            eligible[symbol] = frame
        if not eligible:
            return None

        last_dates = [f.index[-1] for f in eligible.values()]
        first = last_dates[0]
        if all(d == first for d in last_dates[1:]):
            # One shared needed date: _score_sliced pads the slice
            # with one symbol's second-to-last panel row; the min over ALL
            # candidates over-covers whichever symbol supplies the pad.
            needed_oldest = min(f.index[-2] for f in eligible.values())
        else:
            needed_oldest = min(last_dates)

        out: Dict[str, pd.DataFrame] = {}
        truncated = False
        for symbol, frame in data.items():
            fr = eligible.get(symbol)
            if fr is None:
                out[symbol] = frame
                continue
            n = len(fr)
            pos = int(fr.index.searchsorted(needed_oldest, side='left'))
            k = (n - pos) + _SAFE_TAIL
            tail = fr['close'].iloc[-min(n, k):].to_numpy(dtype=float)
            if not np.isfinite(tail).all() or (tail == 0.0).any():
                return None
            if n > k:
                out[symbol] = fr.iloc[-k:]
                truncated = True
            else:
                out[symbol] = fr
        return out if truncated else None
    except Exception:  # any surprise -> verbatim full build, never a crash
        return None


# ----------------------------------------------------------------------
# Once-per-process platform calibration of the WHOLE predict fast path
# (tail truncation + needed-date slicing) against the full-history
# reference.
#
# Both optimizations change the SHAPE of arrays fed to numpy/SIMD
# reductions — truncation the clean-return stream height feeding the vol
# two-pass std, slicing the number of date-rows feeding the cross-sectional
# mean/std — and those kernels can round shape-equivalent reductions apart
# across platforms (x86 rounds the sliced standardization 1 ULP from the
# full one at long history; PR #82). So the fast path's consumed scores are
# probed against the FULL-HISTORY scorer's once per process on a diverse
# sweep of long, variously-stale panel shapes, and the whole fast path
# stands aside for the process on any bitwise mismatch — predict then
# computes the full-history value directly. On a sensitive platform this is
# a logged, perf-only regression; a silent number change is impossible, and
# predict's output is the full-history value on EVERY platform. (The two
# optimizations are calibrated together, not separately, because they are
# coupled: standardizing a truncated panel in full is not full-history
# either, so the fallback must drop both.)
# ----------------------------------------------------------------------
_FAST_PATH_OK: Optional[bool] = None


def _calibration_universe():
    """Deterministic panels exercising the engaged-truncation shapes:
    shared-last-date (pad path) and mixed-last-date (stale symbols) panels
    across a range of history lengths (up to real-backtest depth) and
    staleness patterns, at wide symbol counts. Each panel MUST engage
    truncation (n well over the 275-row tail) so the probe compares the fast
    path, not a no-op."""
    def _panel(n, n_sym, seed, stale=()):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range('2015-01-02', periods=n)
        data = {}
        for i in range(n_sym):
            close = 100.0 * np.exp(
                np.cumsum(rng.normal(0.0, 0.01 + 0.004 * i, n)))
            data[f'F{i}'] = pd.DataFrame(
                {'close': close, 'volume': np.full(n, 1e6)}, index=idx)
        for sym, drop in stale:
            data[sym] = data[sym].iloc[:n - drop]
        return data

    return [
        _panel(800, 8, 11),                                  # shared, medium
        _panel(1500, 12, 12),                                # shared, wide+long
        _panel(2200, 8, 13),                                 # shared, backtest-deep
        _panel(900, 10, 14, stale=[('F2', 5), ('F5', 35),    # stale, long
                                   ('F7', 100)]),
        _panel(1500, 10, 15, stale=[('F1', 2), ('F4', 60),   # stale, varied
                                    ('F8', 250)]),
    ]


def _run_calibration() -> bool:
    """Drive the fast path (truncate + slice, ``_score_sliced``) against the
    full-history scorer (``_score_full_history``) over every calibration
    panel, requiring truncation to actually engage and every per-factor
    CONSUMED score to match bit-for-bit.

    Compared at the CONSUMED-EXPOSURE level via four unit-vector weightings:
    scoring with weights ``e_k`` makes each score exactly the standardized
    exposure of factor ``k`` at the symbol's consumed row, so the four runs
    pin every factor axis independently at precisely the rows predict reads
    — nothing else. This is the right granularity:

      * A single hand-weighted score can MASK a 1-ULP factor drift (the
        weights round it away). Unit vectors cannot mask — each isolates
        one factor.
      * Comparing whole RAW PANELS is too STRICT: they can differ by a ULP
        at a NON-consumed tail row (observed on both macOS and Linux) that
        never reaches any score, which would needlessly disable a
        bit-identical platform. Only consumed rows feed scores, so only
        they must match."""
    model = FactorModel()
    model._fitted = True
    unit_weights = [dict(zip(FACTOR_COLUMNS, w)) for w in (
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))]
    for data in _calibration_universe():
        trunc = _truncated_predict_data(data)
        if trunc is None:
            return False
        if not any(len(trunc[s]) < len(f) for s, f in data.items()):
            return False
        raw_fast = _raw_factor_panel(trunc)      # truncated (fast path input)
        raw_full = _raw_factor_panel(data)       # full history (reference)
        for weights in unit_weights:
            model._weights = weights
            fast = model._score_sliced(raw_fast)          # truncate + slice
            full = model._score_full_history(raw_full)    # canonical
            if list(fast) != list(full):
                return False
            for symbol in full:
                if fast[symbol] != full[symbol]:  # 1-ULP drift -> not equal
                    return False
    return True


def _calibrated_fast_path() -> bool:
    """Lazy once-per-process probe; any failure or exception disables the
    predict fast path (truncation + slicing) for the process with a
    WARNING, so predict computes the full-history value directly."""
    global _FAST_PATH_OK
    if _FAST_PATH_OK is None:
        try:
            _FAST_PATH_OK = bool(_run_calibration())
        except Exception:  # calibration must never break predict
            _FAST_PATH_OK = False
        if not _FAST_PATH_OK:
            logger.warning(
                "FactorModel predict fast path disabled: platform "
                "calibration found a bitwise mismatch against the "
                "full-history reference; using full-history standardization "
                "(results unaffected)")
    return _FAST_PATH_OK


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

    @staticmethod
    def fast_path_calibrated() -> bool:
        """Whether this platform passed the once-per-process bitwise
        calibration of the predict fast path (truncation + needed-date
        slicing) against the full-history reference. Tests use this to
        decide whether fast-path-engagement assertions apply."""
        return _calibrated_fast_path()

    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Per-symbol composite = weighted sum of CURRENT standardized factors.

        Builds the raw factor panel, standardizes each factor cross-
        sectionally per date, then dots the LATEST row's standardized
        exposures with the fitted weights. Symbols too short to form factors
        are omitted; ``{}`` when somehow unfitted (the equal-weight fallback
        normally guarantees a usable model, so this only fires if ``fit`` was
        never called).

        CANONICAL SEMANTICS = full-history: standardize the WHOLE raw factor
        panel, then read each symbol's latest standardized row
        (:meth:`_score_full_history`). Two compounding optimizations make
        that cheaper — tail-truncating each frame before the O(history) raw
        build (:func:`_truncated_predict_data`) and restricting
        standardization to the consumed dates (:meth:`_score_sliced`) — but
        the sliced cross-sectional standardization is NOT bit-identical to
        the full one on every platform (x86 SIMD rounds the row-wise
        mean/std 1 ULP apart at long history; PR #82). So the WHOLE fast
        path is gated behind a once-per-process bitwise probe vs the
        full-history reference: it engages only where the two match to the
        bit, and falls back to full-history otherwise, so predict's output
        is the full-history value on every platform. ``fit`` and
        ``desks.factor_risk`` are independent of this path.
        """
        if not self._fitted or not self._weights:
            return {}
        if _calibrated_fast_path():
            trunc = _truncated_predict_data(data)
            raw_panel = _raw_factor_panel(trunc if trunc is not None else data)
            if not raw_panel:
                return {}
            return self._score_sliced(raw_panel)
        # Full-history fallback (canonical): no truncation, no slicing.
        raw_panel = _raw_factor_panel(data)
        if not raw_panel:
            return {}
        return self._score_full_history(raw_panel)

    def _score_from_standardized(
            self, standardized: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Weighted score of each symbol's LATEST standardized row (the tail
        shared verbatim by the sliced and full-history scorers)."""
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

    def _score_full_history(
            self, raw_panel: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """CANONICAL scorer: standardize the FULL raw panel (every date),
        then score each symbol's latest row. The reference the fast path is
        calibrated against and the fallback used wherever the fast path is
        not bit-identical to it."""
        return self._score_from_standardized(_standardized_panel(raw_panel))

    def _score_sliced(
            self, raw_panel: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """FAST scorer: standardize ONLY the consumed dates' cross-sections,
        then score. Bit-identical to :meth:`_score_full_history` on a
        calibrated platform; predict only reaches it behind that probe."""
        # PERF: only each symbol's LATEST standardized row is consumed below,
        # and the cross-sectional z-score is strictly PER-DATE (row-wise
        # across symbols — desks.features.cross_sectional_rank), so
        # standardizing the panel's FULL history every call is pure waste
        # (O(dates x symbols) per predict, O(dates^2 x symbols) over a
        # growing backtest). Restrict the raw panel to the union of needed
        # dates — each symbol's own latest formable date. Usually that is one
        # shared date; a stale symbol contributes its own last date, and
        # every OTHER symbol's row at that date is kept too, so each needed
        # date's cross-section is COMPLETE. On a CALIBRATED platform the
        # sliced standardization is byte-identical to the full one (probe +
        # golden test TestPredictLatestSliceGolden); the whole-path probe in
        # predict is what makes that safe on platforms where it is not.
        needed_dates = {frame.index[-1] for frame in raw_panel.values()}
        if len(needed_dates) == 1:
            # BYTE-IDENTITY GUARD: a lone needed date would make the
            # standardization panel ONE row, and numpy reduces a 1-row
            # (contiguous) cross-section with PAIRWISE summation while a
            # multi-row (strided) panel accumulates each row SEQUENTIALLY —
            # at >= 8 symbols the two orders can round 1 ULP apart from the
            # full-history panel. Pad with one earlier date (any symbol's
            # second-to-last) so the sliced panel keeps the multi-row
            # layout; the pad row's z-scores are computed and discarded
            # (each symbol's scored row is still its own last raw row —
            # boolean masking preserves position order, so ``iloc[-1]``
            # below is unchanged). If no symbol has a second date, the full
            # panel is single-row too and the slice is already identical.
            for frame in raw_panel.values():
                if len(frame.index) >= 2:
                    needed_dates.add(frame.index[-2])
                    if len(needed_dates) > 1:
                        break
        latest_panel = {
            symbol: frame.loc[frame.index.isin(needed_dates)]
            for symbol, frame in raw_panel.items()}
        return self._score_from_standardized(_standardized_panel(latest_panel))


__all__ = ['FactorModel', 'FACTOR_COLUMNS']
