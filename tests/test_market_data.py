"""True range / ATR behavior of MarketDataHandler.calculate_indicators.

Pins the Wilder convention: the first bar has no prior close, so its true
range is high - low, and the 14-bar ATR is therefore valid at bar 14
(index 13), not bar 15. No network: indicators run on synthetic OHLCV.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.market_data import MarketDataHandler


def test_first_bar_tr_is_high_minus_low(make_ohlcv):
    data = make_ohlcv(n_days=30)
    result = MarketDataHandler().calculate_indicators(data.copy())

    first_tr = result['tr'].iloc[0]
    assert not np.isnan(first_tr)
    assert first_tr == (data['high'].iloc[0] - data['low'].iloc[0])


def test_later_tr_values_match_wilder_formula(make_ohlcv):
    """Seeding the first bar must leave every later TR value identical."""
    data = make_ohlcv(n_days=30)
    result = MarketDataHandler().calculate_indicators(data.copy())

    prev_close = data['close'].shift()
    expected = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            (data['high'] - prev_close).abs(),
            (data['low'] - prev_close).abs(),
        ),
    )
    np.testing.assert_allclose(
        result['tr'].to_numpy()[1:], expected.to_numpy()[1:]
    )


def test_atr_valid_at_bar_14(make_ohlcv):
    data = make_ohlcv(n_days=30)
    result = MarketDataHandler().calculate_indicators(data.copy())

    # 14-bar rolling mean: NaN through index 12, first valid value at
    # index 13 (the 14th bar) now that TR_1 is seeded.
    assert result['atr'].iloc[:13].isna().all()
    assert not np.isnan(result['atr'].iloc[13])
    assert result['atr'].iloc[13] == pytest.approx(result['tr'].iloc[:14].mean())


def test_provider_list_is_valid_for_equity_historical():
    """Regression: backtests returned 'No data available' because the
    provider list held names the installed OpenBB rejects for
    equity.price.historical. The free, keyless 'yfinance' provider must be
    present (and first), and the known-invalid names must be gone."""
    providers = MarketDataHandler().providers
    assert 'yfinance' in providers, 'free keyless provider must be present'
    assert providers[0] == 'yfinance', 'free provider should be tried first'
    invalid = {'cboe', 'tmx', 'polygon', 'alpha_vantage', 'tradier'}
    assert not (invalid & set(providers)), (
        f'invalid equity-historical providers still listed: '
        f'{invalid & set(providers)}')
