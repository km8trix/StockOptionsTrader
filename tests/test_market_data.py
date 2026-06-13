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


def test_index_symbol_detection():
    h = MarketDataHandler(cache=False)
    assert h._index_symbol('SPX') == 'SPX'
    assert h._index_symbol('^GSPC') == 'SPX'   # ^-prefix + alias
    assert h._index_symbol('spx') == 'SPX'     # case-insensitive
    assert h._index_symbol('COMP') == 'IXIC'   # alias -> Nasdaq Composite
    assert h._index_symbol('VIX') == 'VIX'
    assert h._index_symbol('AAPL') is None      # equity
    assert h._index_symbol('SPY') is None       # ETF, not an index
    assert h._index_symbol('') is None


def test_index_symbol_routes_to_index_endpoint(monkeypatch):
    """Index tickers must hit obb.index.price.historical (yfinance), and
    equities the equity endpoint — never the other way around."""
    import datetime as dt
    from types import SimpleNamespace
    calls = []

    def make_price(kind):
        def historical(symbol, start_date, end_date, provider):
            calls.append((kind, symbol, provider))
            item = SimpleNamespace(date=dt.date(2024, 1, 2), open=100.0,
                                   high=101.0, low=99.0, close=100.5,
                                   volume=1000.0)
            return SimpleNamespace(results=[item])
        return SimpleNamespace(historical=historical)

    fake_obb = SimpleNamespace(
        index=SimpleNamespace(price=make_price('index')),
        equity=SimpleNamespace(price=make_price('equity')),
    )
    h = MarketDataHandler(cache=False)
    monkeypatch.setattr(h, '_get_openbb', lambda: fake_obb)

    df = h.fetch_stock_data('SPX', '2024-01-01', '2024-02-01')
    assert not df.empty
    assert ('index', 'SPX', 'yfinance') in calls
    assert all(kind == 'index' for (kind, _s, _p) in calls)

    calls.clear()
    h.fetch_stock_data('AAPL', '2024-01-01', '2024-02-01')
    assert all(kind == 'equity' for (kind, _s, _p) in calls)
    assert any(s == 'AAPL' for (_k, s, _p) in calls)


# --- Phase 0: silent-corruption guards -------------------------------------

def test_calculate_volatility_returns_nan_on_short_window(make_ohlcv):
    """Too few closes for a full window must yield an explicit NaN, never an
    IndexError or a silently-undersampled value that flows into pricing."""
    h = MarketDataHandler(cache=False)
    short = make_ohlcv(n_days=5)          # < window + 1 (= 21)
    vol = h.calculate_volatility(short, window=20)
    assert np.isnan(vol)


def test_calculate_volatility_valid_when_enough_data(make_ohlcv):
    h = MarketDataHandler(cache=False)
    data = make_ohlcv(n_days=60)
    vol = h.calculate_volatility(data, window=20)
    assert np.isfinite(vol) and vol > 0


def test_estimate_option_price_guards_zero_volatility():
    """volatility <= 0 (or NaN) must NOT divide by zero in d1; the price falls
    back to discounted intrinsic, clamped >= 0."""
    h = MarketDataHandler(cache=False)
    # ITM call, zero vol -> ~ intrinsic S - K*e^{-rT}
    price = h.estimate_option_price(stock_price=110.0, strike=100.0,
                                    time_to_expiry=0.5, volatility=0.0,
                                    option_type='call', rate=0.05)
    assert np.isfinite(price)
    assert price == pytest.approx(110.0 - 100.0 * np.exp(-0.05 * 0.5))
    # OTM call, zero vol -> intrinsic clamped to 0 (never negative/NaN)
    otm = h.estimate_option_price(90.0, 100.0, 0.5, 0.0, 'call', 0.05)
    assert otm == 0.0


def test_estimate_option_price_guards_nan_volatility():
    h = MarketDataHandler(cache=False)
    price = h.estimate_option_price(100.0, 100.0, 0.5, float('nan'), 'put', 0.05)
    assert np.isfinite(price)


def test_empty_data_is_shaped_not_hollow():
    """The no-data failure frame must carry OHLCV columns so downstream
    data['close'] / .iloc[-1] cannot KeyError, while still being .empty."""
    h = MarketDataHandler(cache=False)
    df = h._empty_data('NOPE')
    assert df.empty
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert df.index.name == 'date'
    # column access on the failure frame is safe (no KeyError)
    assert len(df['close']) == 0
