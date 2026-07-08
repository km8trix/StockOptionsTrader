"""
Shared feature library — the canonical, reusable feature engineering the
new walk-forward ML models (and the upcoming Two Sigma / AQR
cross-sectional desks) build on.

WHY THIS MODULE EXISTS:
    ``ml_model.GradientBoostingModel._feature_frame`` is the golden,
    byte-identical baseline and must NOT change. Rather than copy its logic
    into every new model, this module re-derives the SAME per-symbol
    feature set (same columns, same fallbacks, same NaN-drop) so the new
    models share one source of truth, then layers additional, documented
    features on top. It is pure infrastructure: it imports NO model and
    mutates NO global state, so importing it can never perturb the golden
    paths.

GUARANTEES:
    * Pure functions, fully deterministic — identical input frames yield
      identical output frames (no RNG, no wall-clock, no global state).
    * No look-ahead. Every feature at row ``i`` is a function of rows
      ``<= i`` only (``pct_change``/rolling/EWMA all look backward), so a
      frame already sliced to ``index <= date`` by WalkForwardController
      stays leakage-free. The cross-sectional helper is row-wise across
      symbols at a single timestamp and never reaches into future rows.
    * Self-contained fallbacks. Indicator columns the engine already
      enriches (rsi, macd, bb_*, volume_sma) are preferred; when absent
      they are recomputed from raw OHLCV with the SAME formulas
      ml_model.py uses, so a frame missing enrichment still produces the
      baseline columns.

The label convention the models pair with these features is unchanged from
ml_model.py: ``label[i] = sign(close[i+1]/close[i] - 1)``; the last row of
each frame has no next-day close and is excluded from training (it is still
a valid predict-time feature row). This module produces only features — the
label alignment lives in the models, exactly as it does for
GradientBoostingModel.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Baseline per-symbol feature columns, byte-for-byte the set
#: ``ml_model.GradientBoostingModel`` trains on (same order). The new
#: models reuse this so the existing golden behavior has a faithful twin.
BASE_FEATURE_COLUMNS: Sequence[str] = (
    'ret_1', 'rsi', 'macd', 'bb_position', 'volume_ratio')

#: Additional features layered on top of the baseline (documented inline in
#: :func:`extended_feature_frame`). Kept separate from the baseline so a
#: caller can ask for just the golden-equivalent set when it wants parity.
EXTRA_FEATURE_COLUMNS: Sequence[str] = (
    'ret_5', 'ret_10', 'vol_20', 'momentum_10', 'zscore_20',
    'dollar_vol_ratio')

#: Calendar/seasonality features derived from the frame's DatetimeIndex
#: (documented in :func:`extended_feature_frame`). Cyclically encoded so the
#: models see no false ordinal jump at the wrap (Friday->Monday,
#: December->January).
SEASONAL_FEATURE_COLUMNS: Sequence[str] = (
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'turn_of_month')

#: Full extended column order: baseline, then extras, then seasonal.
FEATURE_COLUMNS: Sequence[str] = (
    tuple(BASE_FEATURE_COLUMNS) + tuple(EXTRA_FEATURE_COLUMNS)
    + tuple(SEASONAL_FEATURE_COLUMNS))

#: Columns :func:`enrich_extended` adds to a frame (extras + seasonal).
ENRICHED_EXTRA_COLUMNS: Sequence[str] = (
    tuple(EXTRA_FEATURE_COLUMNS) + tuple(SEASONAL_FEATURE_COLUMNS))

#: Raw/indicator columns the predict-time fast paths read. The engine's
#: calculate_indicators provides the first seven; enrich_extended the rest.
_FAST_PATH_COLUMNS: Sequence[str] = (
    ('close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower',
     'volume_sma') + tuple(ENRICHED_EXTRA_COLUMNS))


# ----------------------------------------------------------------------
# Per-symbol technical features
# ----------------------------------------------------------------------
def base_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Baseline per-row features in :data:`BASE_FEATURE_COLUMNS` order.

    Mirrors ``ml_model.GradientBoostingModel._feature_frame`` exactly —
    same columns, same indicator preference, same fallback formulas, same
    final NaN/non-finite drop — so models using this share the golden
    feature definition without importing the golden model.

    Features:
        ret_1         1-day simple return (close.pct_change()).
        rsi           14-period RSI; prefers the enriched ``rsi`` column,
                      else Wilder-style gain/loss rolling mean ratio.
        macd          12/26 EWMA MACD line; prefers enriched ``macd``.
        bb_position   position of close within its 20-day, 2-sigma
                      Bollinger band ((close-lower)/(upper-lower)); 0.5
                      when the band has zero width. Prefers enriched
                      ``bb_upper``/``bb_lower``.
        volume_ratio  volume / 20-day average volume; prefers enriched
                      ``volume_sma``.

    Rows with any NaN or non-finite feature are dropped. An empty or
    close-less frame yields an empty frame with the baseline columns.
    """
    if data is None or data.empty or 'close' not in data.columns:
        return pd.DataFrame(columns=list(BASE_FEATURE_COLUMNS))

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


def extended_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Baseline features plus several documented predictive additions.

    Columns are :data:`FEATURE_COLUMNS` (baseline then extras). Each extra
    is a backward-looking transform of ``close`` (no look-ahead) chosen to
    add information the baseline lacks:

        ret_5        5-day return (close/close.shift(5) - 1) — captures a
                     short multi-day trend the 1-day return misses.
        ret_10       10-day return — a longer momentum horizon, so the
                     model can separate fast reversals from slow drift.
        vol_20       20-day rolling std of the 1-day return — realized
                     volatility; direction predictability differs sharply
                     between calm and turbulent regimes.
        momentum_10  price distance from its own 10-day moving average
                     (close / close.rolling(10).mean() - 1). A trend /
                     stretch signal distinct from ret_10: ret_10 compares
                     close to its value 10 days ago (a point-to-point
                     return), whereas momentum_10 compares close to the
                     AVERAGE of the last 10 closes, so it measures how far
                     price has pulled away from its recent mean. Positive
                     when price sits above its 10-day average (up-trend /
                     overbought), negative below.
        zscore_20    z-score of close vs its 20-day mean/std
                     ((close-mean_20)/std_20) — a mean-reversion signal
                     measuring how stretched price is from its recent
                     average; 0.0 when the 20-day std is zero.
        dollar_vol_ratio
                     dollar (turnover) volume vs its 20-day average:
                     (close*volume) / mean_20(close*volume). volume_ratio
                     captures SHARE volume vs average; this captures DOLLAR
                     volume, so a $2 name and a $2,000 name with equal share
                     volume are no longer indistinguishable in liquidity —
                     cross-name turnover comparability the share ratio lacks.

    Plus seasonal columns (:data:`SEASONAL_FEATURE_COLUMNS`) derived from
    the DatetimeIndex — documented daily-equity seasonality the price
    features cannot express:

        dow_sin/cos     day-of-week (Mon=0..Fri=4) on a 5-day cycle,
                        sin/cos encoded (Monday/Friday effects).
        month_sin/cos   month-of-year on a 12-month cycle, sin/cos encoded
                        (January / month-of-year effect).
        turn_of_month   1.0 on the first/last few calendar days
                        (``day <= 3 or day >= 28``), else 0.0 — the
                        turn-of-month effect.

    The extras are computed from raw ``close`` (independent of indicator
    enrichment), then concatenated with :func:`base_feature_frame` and a
    single combined NaN/non-finite drop is applied so the warm-up rows of
    every column are removed together. Empty/close-less input yields an
    empty frame with the full column set.
    """
    if data is None or data.empty or 'close' not in data.columns:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))

    base = base_feature_frame(data)

    close = data['close']
    extras = pd.DataFrame(index=data.index)
    extras['ret_5'] = close / close.shift(5) - 1.0
    extras['ret_10'] = close / close.shift(10) - 1.0
    extras['vol_20'] = close.pct_change().rolling(window=20).std()
    extras['momentum_10'] = close / close.rolling(window=10).mean() - 1.0
    roll_mean_20 = close.rolling(window=20).mean()
    roll_std_20 = close.rolling(window=20).std()
    extras['zscore_20'] = np.where(
        roll_std_20 > 0, (close - roll_mean_20) / roll_std_20, 0.0)
    # Dollar (turnover) volume vs its 20-day average — backward-looking,
    # like volume_ratio but in dollar terms. A zero 20-day average (all-zero
    # volume window) yields inf, swept to NaN by the combined drop below.
    dollar_vol = close * data['volume']
    extras['dollar_vol_ratio'] = dollar_vol / dollar_vol.rolling(20).mean()

    # Calendar/seasonality features from the DatetimeIndex. Each is a
    # function of the row's OWN timestamp only (no look-ahead), defined on
    # every row (no warm-up). Day-of-week and month are cyclically encoded
    # (sin/cos) so the wrap carries no false ordinal gap; turn_of_month is a
    # binary flag for the first/last few calendar days (a documented equity
    # seasonal).
    idx = data.index
    dow = idx.dayofweek.to_numpy(dtype=float)   # Mon=0 .. Fri=4
    month = idx.month.to_numpy(dtype=float)     # 1 .. 12
    seasonal = pd.DataFrame(index=idx)
    seasonal['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)
    seasonal['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)
    seasonal['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    seasonal['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    seasonal['turn_of_month'] = (
        (idx.day <= 3) | (idx.day >= 28)).astype(float)

    # Both extras and seasonal share `data.index`; the combined dropna()
    # removes any row where the baseline or an extra is still warming up
    # (seasonal columns are defined on every row), so all columns are
    # defined on every surviving row.
    combined = base.join(extras, how='outer').join(seasonal, how='outer')
    combined = combined.reindex(columns=list(FEATURE_COLUMNS))
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined.dropna()


# ----------------------------------------------------------------------
# Predict-time fast path (enriched-column reads)
# ----------------------------------------------------------------------
def enrich_extended(data: pd.DataFrame) -> pd.DataFrame:
    """Precompute the extras + seasonal columns ONCE on a full frame.

    Adds :data:`ENRICHED_EXTRA_COLUMNS` in place (and returns the frame,
    matching ``calculate_indicators``'s style) with formulas byte-copied
    from :func:`extended_feature_frame` — including the ``zscore_20``
    ``np.where(std > 0, ..., 0.0)`` quirk, which makes that column 0.0
    (NOT NaN) through its warm-up. Because rolling/shift/pct_change are
    forward-only streaming passes, row ``t`` of a full-frame column is
    bit-identical to the value a prefix ``[:t+1]`` recompute produces —
    the same prefix-stability the engine's indicator precompute and the
    stacking OOF fast path already rely on. The predict-time fast paths
    below READ these columns; the full :func:`extended_feature_frame`
    path deliberately never prefers them, so fit-time numerics and every
    fallback recompute stay byte-for-byte unchanged.

    Empty/close-less frames are returned untouched.
    """
    if data is None or data.empty or 'close' not in data.columns:
        return data

    close = data['close']
    data['ret_5'] = close / close.shift(5) - 1.0
    data['ret_10'] = close / close.shift(10) - 1.0
    data['vol_20'] = close.pct_change().rolling(window=20).std()
    data['momentum_10'] = close / close.rolling(window=10).mean() - 1.0
    roll_mean_20 = close.rolling(window=20).mean()
    roll_std_20 = close.rolling(window=20).std()
    data['zscore_20'] = np.where(
        roll_std_20 > 0, (close - roll_mean_20) / roll_std_20, 0.0)
    dollar_vol = close * data['volume']
    data['dollar_vol_ratio'] = dollar_vol / dollar_vol.rolling(20).mean()

    idx = data.index
    dow = idx.dayofweek.to_numpy(dtype=float)   # Mon=0 .. Fri=4
    month = idx.month.to_numpy(dtype=float)     # 1 .. 12
    data['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)
    data['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)
    data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    data['turn_of_month'] = (
        (idx.day <= 3) | (idx.day >= 28)).astype(float)
    return data


def _fast_columns_ok(frame: pd.DataFrame) -> bool:
    """All fast-path columns present with plain float64/integer dtypes.

    Mirrors ml_model._fast_last_row's dtype discipline: float32 would
    round differently and pandas extension dtypes can hold pd.NA, so
    anything but a plain numpy float64/int dtype demands the full path.
    """
    for name in _FAST_PATH_COLUMNS:
        if name not in frame.columns:
            return False
        dtype = frame.dtypes[name]
        if not isinstance(dtype, np.dtype):
            return False
        if dtype != np.float64 and dtype.kind not in 'iu':
            return False
    return True


def fast_last_extended_row(frame: pd.DataFrame) -> Optional[np.ndarray]:
    """O(1) extended feature vector for the frame's FINAL row, or None.

    The 16-wide twin of ``ml_model._fast_last_row`` for
    :data:`FEATURE_COLUMNS`: when the engine has enriched the frame
    (indicators + :func:`enrich_extended`), the last row is plain column
    reads plus the three scalar computations (ret_1, bb_position,
    volume_ratio) whose formulas are bit-identical to the vectorized
    ops. Returns None — demanding the full
    :func:`extended_feature_frame` path — unless ALL of:

    - every fast-path column is present with a plain float64/int dtype;
    - the frame has >= 2 rows (ret_1 needs a prior close);
    - close[-2] != 0 and volume_sma[-1] != 0 (the full path sweeps the
      resulting inf to NaN and backs off to an earlier row);
    - every one of the 16 values is finite — a NaN/inf feature means the
      full path would drop this row, and only the full path reproduces
      that back-off.

    The bb_position quirk is replicated: ``np.where(width > 0, ..., 0.5)``
    yields 0.5 for a NaN or non-positive band width, so a NaN band does
    NOT invalidate the row.
    """
    if len(frame.index) < 2 or not _fast_columns_ok(frame):
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
         bb_position, volume_ratio]
        + [frame[name].iloc[-1] for name in ENRICHED_EXTRA_COLUMNS],
        dtype=float)
    if not np.isfinite(vector).all():
        return None
    return vector


def fast_tail_extended_window(frame: pd.DataFrame,
                              lookback: int) -> Optional[np.ndarray]:
    """The last ``lookback`` extended feature rows as a ``(lookback, 16)``
    matrix, or None when the full path must be used.

    For sequence models the predict-time window is the feature frame's
    last ``lookback`` rows. Those equal the DATA frame's last ``lookback``
    rows exactly when every one of them survives the full path's
    ``dropna()`` — which the all-finite gate below guarantees. Any
    non-finite cell in the candidate window means the full path would
    SPLICE earlier rows into the window (dropna removes the bad row), and
    only the full path reproduces that composition — so refuse.

    ret_1/bb_position/volume_ratio are recomputed per row with the same
    scalar-exact elementwise formulas as the vectorized ops (division and
    the np.where quirk are per-element, so a tail recompute is exact for
    THESE — unlike rolling ops, which is why the rolling-derived columns
    are read from the enriched frame instead). Requires ``lookback + 1``
    rows (the window's oldest ret_1 needs its prior close) and refuses on
    any zero close/volume_sma denominator inside the window.
    """
    if lookback < 1 or len(frame.index) < lookback + 1:
        return None
    if not _fast_columns_ok(frame):
        return None

    tail = frame.iloc[-(lookback + 1):]
    close = tail['close'].to_numpy(dtype=float)
    if (close[:-1] == 0).any():
        return None
    ret_1 = close[1:] / close[:-1] - 1

    bb_upper = tail['bb_upper'].to_numpy(dtype=float)[1:]
    bb_lower = tail['bb_lower'].to_numpy(dtype=float)[1:]
    bb_width = bb_upper - bb_lower
    bb_position = np.where(
        bb_width > 0, (close[1:] - bb_lower) / bb_width, 0.5)

    volume_sma = tail['volume_sma'].to_numpy(dtype=float)[1:]
    if (volume_sma == 0).any():
        return None
    volume_ratio = tail['volume'].to_numpy(dtype=float)[1:] / volume_sma

    columns = [ret_1, tail['rsi'].to_numpy(dtype=float)[1:],
               tail['macd'].to_numpy(dtype=float)[1:],
               bb_position, volume_ratio]
    columns += [tail[name].to_numpy(dtype=float)[1:]
                for name in ENRICHED_EXTRA_COLUMNS]
    window = np.column_stack(columns)
    if not np.isfinite(window).all():
        return None
    # LAYOUT IS PART OF THE CONTRACT, not an optimization: torch's LSTM
    # CPU kernels produce ULP-different logits for C- vs F-ordered inputs
    # of the SAME values (measured: one float32 ULP at 0.5 flips the
    # score). The full path's window is F-ordered because DataFrame
    # .to_numpy() yields column-major data, so the fast window must
    # reproduce that layout — value equality alone is NOT bit-identity
    # downstream. The fast-vs-full score-equality tests pin this.
    return np.asfortranarray(window)


# ----------------------------------------------------------------------
# Cross-sectional features (for the upcoming Two Sigma / AQR desks)
# ----------------------------------------------------------------------
def cross_sectional_rank(
        data: Dict[str, pd.DataFrame], column: str = 'ret_1',
        method: str = 'zscore') -> pd.DataFrame:
    """Cross-sectional rank/z-score of ``column`` across symbols per date.

    For each timestamp present in the union of the symbols' frames, the
    chosen column's values ACROSS symbols at that timestamp are ranked or
    standardized — a relative, market-neutral feature (e.g. "which name is
    cheap/expensive vs its peers today"). This is the cross-sectional view
    the Two Sigma / AQR desks need.

    NO LOOK-AHEAD: the transform is applied row-wise across symbols within
    a single timestamp. Every output cell at date D uses ONLY the symbols'
    values AT date D (other symbols' contemporaneous rows), never any
    symbol's future rows. The standardization/ranking statistics are
    computed per-row from that row's cross-section alone, so no past or
    future row leaks into another row.

    Args:
        data:   symbol -> frame. The frame must already carry ``column``
                (e.g. a feature frame from this module, or a raw OHLCV
                frame with a column like 'close'). Symbols lacking the
                column are skipped.
        column: the column to take cross-sectionally (default 'ret_1').
        method: 'zscore' (default) -> per-date (x - mean) / std across
                symbols (0.0 for a degenerate zero-std / single-symbol
                cross-section); 'rank' -> per-date average rank scaled to
                [-0.5, 0.5] (0.0 when only one symbol is present).

    Returns:
        A DataFrame indexed by date, columns = symbols, values = the
        cross-sectional feature (NaN where a symbol has no value on a
        given date). Empty DataFrame when no symbol supplies ``column``.

    Deterministic: pure pandas reductions, no RNG.
    """
    if method not in ('zscore', 'rank'):
        raise ValueError(f"Unknown method: {method} (expected 'zscore' or 'rank')")

    series_by_symbol: Dict[str, pd.Series] = {}
    for symbol, frame in data.items():
        if frame is None or frame.empty or column not in frame.columns:
            continue
        series_by_symbol[symbol] = frame[column]

    if not series_by_symbol:
        return pd.DataFrame()

    # Align all symbols onto a shared date index — each row is one date's
    # cross-section (symbols as columns). Outer join keeps every observed
    # date; missing cells stay NaN and are excluded from each row's stats.
    panel = pd.DataFrame(series_by_symbol).sort_index()

    if method == 'zscore':
        row_mean = panel.mean(axis=1)
        row_std = panel.std(axis=1)
        # Subtract this row's cross-sectional mean, divide by this row's
        # cross-sectional std. A zero/NaN std (single symbol or identical
        # values) collapses to 0.0 for the defined cells.
        centered = panel.sub(row_mean, axis=0)
        result = centered.div(row_std.replace(0.0, np.nan), axis=0)
        # Where std was 0 (defined values but no spread) -> 0.0, but keep
        # genuinely-absent cells (NaN in the input panel) as NaN.
        zero_std_rows = (row_std.fillna(0.0) == 0.0)
        if zero_std_rows.any():
            sub = panel.loc[zero_std_rows]
            # float64 fill: defined cells -> 0.0, absent cells stay NaN.
            # (A bare .replace({True: 0.0, False: np.nan}) yields an
            # object-dtype frame, which pandas 3.x refuses to assign into
            # the float64 `result` via .loc — build float64 directly.)
            fill = pd.DataFrame(
                np.where(sub.notna(), 0.0, np.nan),
                index=sub.index, columns=sub.columns)
            result.loc[zero_std_rows] = fill
        return result

    # method == 'rank': average rank per date, recentred to [-0.5, 0.5].
    ranks = panel.rank(axis=1, method='average')
    counts = panel.notna().sum(axis=1)
    # Normalize ranks 1..n to (0, 1], then shift so the cross-section is
    # centered on 0 (a market-neutral spread): (rank - (n+1)/2) / n.
    centered_ranks = ranks.sub((counts + 1) / 2.0, axis=0).div(
        counts.replace(0, np.nan), axis=0)
    # A single-symbol cross-section has no spread -> 0.0 for that cell.
    single = (counts == 1)
    if single.any():
        sub = panel.loc[single]
        # float64 fill: the single defined cell -> 0.0, absent cells stay
        # NaN. Built directly as float64 (an object-dtype .replace frame
        # cannot be assigned into the float64 result under pandas 3.x).
        fill = pd.DataFrame(
            np.where(sub.notna(), 0.0, np.nan),
            index=sub.index, columns=sub.columns)
        centered_ranks.loc[single] = fill
    return centered_ranks


def build_training_set(
        train_data: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled per-row (X, y) across symbols with the golden label alignment.

    For each symbol, feature rows (:func:`extended_feature_frame`) are aligned
    to ``label[i] = 1 if close[i+1]/close[i] - 1 > 0 else 0``; the final row of
    each frame has no next-day return and is dropped (the off-by-one the label
    demands — identical to ``GradientBoostingModel.build_training_set``). The
    single source of truth shared by the boosting (LightGBM/stacking) and
    neural (MLP/LSTM) models, whose training matrices were byte-for-byte the
    same construction.
    """
    x_parts, y_parts = [], []
    for _symbol, data in train_data.items():
        if data is None or data.empty or 'close' not in data.columns:
            continue
        frame = extended_feature_frame(data)
        if frame.empty:
            continue
        close = data['close']
        next_return = close.shift(-1) / close - 1.0
        labels = (next_return > 0).astype(int)
        usable = frame.index.intersection(next_return.dropna().index)
        if usable.empty:
            continue
        x_parts.append(frame.loc[usable].to_numpy(dtype=float))
        y_parts.append(labels.loc[usable].to_numpy(dtype=int))

    if not x_parts:
        return (np.empty((0, len(FEATURE_COLUMNS))), np.empty((0,)))
    return np.vstack(x_parts), np.concatenate(y_parts)
