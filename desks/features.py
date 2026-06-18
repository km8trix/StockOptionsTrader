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

from typing import Dict, Sequence

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
    'ret_5', 'ret_10', 'vol_20', 'momentum_10', 'zscore_20')

#: Full extended column order: baseline followed by extras.
FEATURE_COLUMNS: Sequence[str] = (
    tuple(BASE_FEATURE_COLUMNS) + tuple(EXTRA_FEATURE_COLUMNS))


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

    # Reindex extras onto the base frame's index first is unnecessary —
    # both share `data.index`; the combined dropna() removes any row where
    # EITHER the baseline or an extra is still warming up, so all columns
    # are defined on every surviving row.
    combined = base.join(extras, how='outer')
    combined = combined.reindex(columns=list(FEATURE_COLUMNS))
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined.dropna()


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
