"""Tests for desks.features — the shared feature library.

Three load-bearing properties are pinned here:

  * No NaN/inf survives in any output row (the combined dropna() removes
    every warm-up row across all columns together).
  * ``extended_feature_frame`` is a strict superset of
    ``base_feature_frame`` (same baseline columns, in order, plus the
    documented extras).
  * ``cross_sectional_rank`` has NO look-ahead: its output for the first
    N rows is byte-identical whether or not later rows are present in the
    input (prefix invariance) — the property the Two Sigma / AQR desks
    rely on.

Offline and deterministic: synthetic frames only, no network, no RNG in
the library itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desks.features import (BASE_FEATURE_COLUMNS, EXTRA_FEATURE_COLUMNS,
                            FEATURE_COLUMNS, SEASONAL_FEATURE_COLUMNS,
                            base_feature_frame, cross_sectional_rank,
                            extended_feature_frame)
from desks.ml_model import GradientBoostingModel

# C1 FIX (landed): the zero-spread fallbacks in
# desks.features.cross_sectional_rank now build their float64 fill DIRECTLY
# via ``np.where(sub.notna(), 0.0, np.nan)`` instead of the old
# object-dtype ``panel.notna().replace({True: 0.0, False: np.nan})``, so
# assigning it back into the float64 ``result`` no longer trips pandas
# 3.x's strict dtype-upcast guard. The two zero-spread tests below were
# xfail(raises=TypeError) under the bug; they are now normal passes
# asserting the documented contract: zero-spread / single-symbol /
# single-non-NaN-symbol cells collapse to 0.0 (float), absent cells stay
# NaN, the output stays float64, and no TypeError is raised.


def synth(seed: int = 0, n: int = 200, start_price: float = 100.0,
          start: str = '2022-01-03') -> pd.DataFrame:
    """Seeded synthetic OHLCV frame (no indicator columns -> fallbacks)."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n)
    close = start_price * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close,
         'volume': volume}, index=index)


class TestColumnContract:
    def test_base_columns_match_documented_set_and_order(self):
        frame = base_feature_frame(synth())
        assert list(frame.columns) == list(BASE_FEATURE_COLUMNS)
        assert list(BASE_FEATURE_COLUMNS) == [
            'ret_1', 'rsi', 'macd', 'bb_position', 'volume_ratio']

    def test_extended_columns_in_documented_order(self):
        frame = extended_feature_frame(synth())
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        # baseline (5) + extras (6) + seasonal (5) = 16.
        assert len(FEATURE_COLUMNS) == 16
        assert list(EXTRA_FEATURE_COLUMNS) == [
            'ret_5', 'ret_10', 'vol_20', 'momentum_10', 'zscore_20',
            'dollar_vol_ratio']
        assert list(SEASONAL_FEATURE_COLUMNS) == [
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'turn_of_month']

    def test_extended_is_superset_of_base(self):
        base = base_feature_frame(synth())
        ext = extended_feature_frame(synth())
        # Baseline columns appear in the extended frame, in the same order.
        assert list(ext.columns[:len(base.columns)]) == list(base.columns)
        assert set(base.columns) <= set(ext.columns)

    def test_momentum_10_is_price_distance_from_10day_ma(self):
        # M2: momentum_10 was redefined from a ret_10 duplicate to the
        # price distance from its own 10-day moving average:
        #   close / close.rolling(10).mean() - 1.0
        # The column NAME and 10-column count are unchanged (pinned in
        # test_extended_has_ten_columns_in_documented_order); here we pin
        # the VALUE and that it is now DISTINCT from ret_10.
        frame = synth(seed=3, n=80)
        ext = extended_feature_frame(frame)
        close = frame['close']

        # New formula, reconstructed on the surviving (post-warm-up) index.
        expected_mom = (close / close.rolling(window=10).mean() - 1.0)
        expected_mom = expected_mom.reindex(ext.index)
        pd.testing.assert_series_equal(
            ext['momentum_10'], expected_mom, check_names=False)

        # It is NOT the old ret_10 (close/close.shift(10)-1) duplicate.
        expected_ret10 = (close / close.shift(10) - 1.0).reindex(ext.index)
        pd.testing.assert_series_equal(
            ext['ret_10'], expected_ret10, check_names=False)
        # On non-trivial (random-walk) data the two columns are distinct.
        assert not np.allclose(
            ext['momentum_10'].to_numpy(), ext['ret_10'].to_numpy())


class TestSeasonalFeatures:
    def test_seasonal_columns_match_index_encoding(self):
        ext = extended_feature_frame(synth(seed=1, n=120))
        idx = ext.index
        dow = idx.dayofweek.to_numpy(dtype=float)
        month = idx.month.to_numpy(dtype=float)
        assert np.allclose(ext['dow_sin'].to_numpy(),
                           np.sin(2 * np.pi * dow / 5.0))
        assert np.allclose(ext['dow_cos'].to_numpy(),
                           np.cos(2 * np.pi * dow / 5.0))
        assert np.allclose(ext['month_sin'].to_numpy(),
                           np.sin(2 * np.pi * month / 12.0))
        assert np.allclose(ext['month_cos'].to_numpy(),
                           np.cos(2 * np.pi * month / 12.0))
        expected_tom = ((idx.day <= 3) | (idx.day >= 28)).astype(float)
        assert np.array_equal(ext['turn_of_month'].to_numpy(), expected_tom)
        # Cyclic encodings stay in [-1, 1]; the flag is binary.
        for col in ('dow_sin', 'dow_cos', 'month_sin', 'month_cos'):
            assert ext[col].abs().max() <= 1.0
        assert set(np.unique(ext['turn_of_month'].to_numpy())) <= {0.0, 1.0}

    def test_seasonal_has_no_lookahead(self):
        # Each seasonal value depends ONLY on its own date — truncating
        # future rows leaves earlier rows' seasonal columns byte-identical.
        full = extended_feature_frame(synth(seed=2, n=120))
        prefix = extended_feature_frame(synth(seed=2, n=120).iloc[:80])
        common = prefix.index.intersection(full.index)
        cols = list(SEASONAL_FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(
            full.loc[common, cols], prefix.loc[common, cols])


class TestDollarVolumeFeature:
    def test_dollar_vol_ratio_matches_close_times_volume(self):
        frame = synth(seed=4, n=120)
        ext = extended_feature_frame(frame)
        # Reconstruct on the surviving (post-warm-up) index.
        dollar_vol = frame['close'] * frame['volume']
        expected = (dollar_vol / dollar_vol.rolling(20).mean()).reindex(
            ext.index)
        pd.testing.assert_series_equal(
            ext['dollar_vol_ratio'], expected, check_names=False)

    def test_dollar_vol_ratio_distinct_from_share_volume_ratio(self):
        # The whole point: dollar terms carry info the share ratio lacks.
        ext = extended_feature_frame(synth(seed=5, n=120))
        assert not np.allclose(
            ext['dollar_vol_ratio'].to_numpy(),
            ext['volume_ratio'].to_numpy())

    def test_dollar_vol_ratio_is_finite(self):
        ext = extended_feature_frame(synth(seed=6, n=120))
        vals = ext['dollar_vol_ratio'].to_numpy(dtype=float)
        assert not np.isnan(vals).any()
        assert not np.isinf(vals).any()

    def test_dollar_vol_ratio_has_no_lookahead(self):
        # Backward-looking only — truncating future rows leaves earlier
        # rows' value byte-identical.
        full = extended_feature_frame(synth(seed=7, n=120))
        prefix = extended_feature_frame(synth(seed=7, n=120).iloc[:80])
        common = prefix.index.intersection(full.index)
        pd.testing.assert_series_equal(
            full.loc[common, 'dollar_vol_ratio'],
            prefix.loc[common, 'dollar_vol_ratio'])

    def test_dollar_vol_ratio_deterministic(self):
        a = extended_feature_frame(synth(seed=8, n=90))
        b = extended_feature_frame(synth(seed=8, n=90))
        pd.testing.assert_series_equal(
            a['dollar_vol_ratio'], b['dollar_vol_ratio'])


class TestNoNaNOrInf:
    @pytest.mark.parametrize('builder', [base_feature_frame,
                                         extended_feature_frame])
    @pytest.mark.parametrize('seed', [0, 7, 42])
    def test_no_nan_or_inf_in_output_rows(self, builder, seed):
        frame = builder(synth(seed=seed))
        assert not frame.empty
        values = frame.to_numpy(dtype=float)
        assert not np.isnan(values).any()
        assert not np.isinf(values).any()

    def test_extended_drops_warmup_rows_across_all_columns_together(self):
        # The longest warm-up is zscore_20 / vol_20 (20-day windows) and
        # the baseline bb/volume 20-day windows; the combined dropna() must
        # leave every surviving row fully defined.
        ext = extended_feature_frame(synth(n=60))
        assert len(ext) > 0
        assert ext.notna().all().all()

    def test_empty_input_yields_empty_frame_with_full_columns(self):
        empty = extended_feature_frame(pd.DataFrame())
        assert list(empty.columns) == list(FEATURE_COLUMNS)
        assert empty.empty
        base_empty = base_feature_frame(pd.DataFrame())
        assert list(base_empty.columns) == list(BASE_FEATURE_COLUMNS)
        assert base_empty.empty

    def test_close_less_frame_yields_empty(self):
        no_close = pd.DataFrame({'volume': [1.0, 2.0, 3.0]})
        assert extended_feature_frame(no_close).empty
        assert base_feature_frame(no_close).empty


class TestBaselineParityWithGoldenModel:
    """base_feature_frame must reproduce GradientBoostingModel._feature_frame
    byte-for-byte (it is the documented source-of-truth twin)."""

    @pytest.mark.parametrize('seed', [0, 11, 99])
    def test_base_frame_is_byte_identical_to_golden_model(self, seed):
        frame = synth(seed=seed)
        golden = GradientBoostingModel()._feature_frame(frame)
        lib = base_feature_frame(frame)
        pd.testing.assert_frame_equal(lib, golden)


class TestCrossSectionalRank:
    def _panel(self, n=120):
        return {'A': synth(seed=1, n=n), 'B': synth(seed=2, n=n),
                'C': synth(seed=3, n=n)}

    @pytest.mark.parametrize('method', ['zscore', 'rank'])
    def test_prefix_invariance_no_lookahead(self, method):
        """The first N output rows are identical whether or not later rows
        exist in the input — the defining no-look-ahead property."""
        full_input = self._panel(n=120)
        full = cross_sectional_rank(full_input, column='close', method=method)

        n_prefix = 30
        prefix_input = {sym: frame.iloc[:n_prefix]
                        for sym, frame in full_input.items()}
        prefix = cross_sectional_rank(prefix_input, column='close',
                                      method=method)

        assert len(prefix) == n_prefix
        np.testing.assert_array_equal(
            full.iloc[:n_prefix].to_numpy(),
            prefix.to_numpy())  # exact, including NaN placement

    def test_zscore_centers_each_row_cross_section(self):
        result = cross_sectional_rank(self._panel(n=60), column='close',
                                      method='zscore')
        # Each row is standardized across the (3) symbols: mean ~0, the
        # values are finite where all three symbols are present.
        row = result.iloc[-1]
        assert row.notna().all()
        assert row.mean() == pytest.approx(0.0, abs=1e-9)

    def test_rank_is_recentered_to_symmetric_range(self):
        result = cross_sectional_rank(self._panel(n=60), column='close',
                                      method='rank')
        row = result.iloc[-1].to_numpy(dtype=float)
        assert np.all(row >= -0.5) and np.all(row <= 0.5)
        # A 3-symbol cross-section recenters to {-1/3, 0, +1/3}.
        assert row.sum() == pytest.approx(0.0, abs=1e-9)

    def test_single_symbol_cross_section_collapses_to_zero(self):
        # C1: a single-symbol cross-section has no spread; every defined
        # cell collapses to 0.0 (float) with NO TypeError, output float64.
        result = cross_sectional_rank({'A': synth(n=30)}, column='close',
                                      method='zscore')
        assert (result['A'].dropna() == 0.0).all()
        # The fill stays float (0.0 is a float, not object/bool).
        assert result['A'].dtype == np.float64
        assert isinstance(result['A'].dropna().iloc[0], float)

    def test_zero_std_row_collapses_to_zero_not_nan(self):
        # C1: two symbols with identical closes -> zero cross-sectional
        # spread -> every cell collapses to 0.0 (not NaN), no TypeError.
        idx = pd.bdate_range('2023-01-02', periods=5)
        same = pd.DataFrame({'close': [10.0, 11.0, 12.0, 13.0, 14.0]},
                            index=idx)
        result = cross_sectional_rank({'A': same, 'B': same.copy()},
                                      column='close', method='zscore')
        values = result.to_numpy(dtype=float)
        assert (values == 0.0).all()
        assert not np.isnan(values).any()  # defined cells, none left NaN
        assert (result.dtypes == np.float64).all()  # output stays float64

    def test_rank_branch_single_non_nan_symbol_collapses_to_zero(self):
        # C1 (rank branch): on a date where exactly ONE symbol has a value,
        # the single defined cell collapses to 0.0 and the absent symbol's
        # cell stays NaN. Symbol B starts a day late, so the FIRST date has
        # only A defined (count == 1 -> the single-symbol rank fallback).
        idx = pd.bdate_range('2023-01-02', periods=4)
        a = pd.DataFrame({'close': [10.0, 11.0, 12.0, 13.0]}, index=idx)
        b = pd.DataFrame({'close': [21.0, 22.0, 23.0]}, index=idx[1:])
        result = cross_sectional_rank({'A': a, 'B': b}, column='close',
                                      method='rank')
        first = result.iloc[0]
        assert first['A'] == 0.0           # the lone defined cell -> 0.0
        assert pd.isna(first['B'])         # genuinely absent -> NaN, not 0.0
        assert (result.dtypes == np.float64).all()

    def test_missing_column_symbols_are_skipped(self):
        # Two symbols carry the column (so the surviving cross-section has
        # spread and does not trip the zero-spread fallback); one lacks it
        # and must be dropped from the output columns.
        good_a = synth(seed=1, n=30)
        good_b = synth(seed=2, n=30)
        bad = synth(seed=3, n=30).drop(columns=['close'])
        result = cross_sectional_rank(
            {'A': good_a, 'B': good_b, 'BAD': bad}, column='close')
        assert sorted(result.columns) == ['A', 'B']
        assert 'BAD' not in result.columns

    def test_no_symbol_supplies_column_yields_empty(self):
        result = cross_sectional_rank({'A': synth(n=10)}, column='not_here')
        assert result.empty

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match='Unknown method'):
            cross_sectional_rank(self._panel(n=10), method='median')

    def test_deterministic_repeat_calls_are_identical(self):
        a = cross_sectional_rank(self._panel(n=60), column='close')
        b = cross_sectional_rank(self._panel(n=60), column='close')
        pd.testing.assert_frame_equal(a, b)
