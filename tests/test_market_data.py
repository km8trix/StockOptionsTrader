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


_KEYED_ENV = ('FMP_API_KEY', 'TIINGO_TOKEN', 'INTRINIO_API_KEY')


def test_provider_list_falls_back_to_yfinance_with_no_keys(monkeypatch):
    """Regression: backtests returned 'No data available' because the provider
    list held names the installed OpenBB rejects for equity.price.historical.
    With no credentials configured, the only provider is free keyless
    yfinance, and the known-invalid names are gone."""
    for env in _KEYED_ENV:
        monkeypatch.delenv(env, raising=False)
    providers = MarketDataHandler().providers
    assert providers == ['yfinance']
    invalid = {'cboe', 'tmx', 'polygon', 'alpha_vantage', 'tradier'}
    assert not (invalid & set(providers))


def test_configured_key_is_tried_before_yfinance(monkeypatch):
    """A configured keyed provider must come BEFORE yfinance, so a paid key
    actually serves the data (higher rate limit / reliability) rather than
    yfinance staying primary."""
    for env in _KEYED_ENV:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv('FMP_API_KEY', 'test-key')
    providers = MarketDataHandler().providers
    assert providers == ['fmp', 'yfinance']


def test_multiple_keys_keep_preference_order_then_yfinance(monkeypatch):
    for env in _KEYED_ENV:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv('TIINGO_TOKEN', 't')
    monkeypatch.setenv('INTRINIO_API_KEY', 'i')
    providers = MarketDataHandler().providers
    # Preference order (fmp, tiingo, intrinio) filtered to configured keys,
    # then yfinance — fmp absent because its key is unset.
    assert providers == ['tiingo', 'intrinio', 'yfinance']


def test_ensure_ssl_certs_points_at_certifi_when_unset(monkeypatch):
    """aiohttp-based providers (Tiingo) fail with 'certificate verify failed'
    on python.org macOS builds; the handler points SSL_CERT_FILE at the
    certifi bundle when nothing is configured."""
    import os
    import certifi
    monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    monkeypatch.delenv('REQUESTS_CA_BUNDLE', raising=False)
    MarketDataHandler._ensure_ssl_certs()
    assert os.environ.get('SSL_CERT_FILE') == certifi.where()


def test_ensure_ssl_certs_respects_existing(monkeypatch):
    import os
    monkeypatch.setenv('SSL_CERT_FILE', '/custom/ca.pem')
    MarketDataHandler._ensure_ssl_certs()
    assert os.environ['SSL_CERT_FILE'] == '/custom/ca.pem'
