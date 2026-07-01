"""Tests for the survivorship-free warehouse price feed.

Covers PitWarehouse.ohlcv (split/div adjustment + shape), the WarehouseMarketData
adapter (delegates to the warehouse, matches the MarketDataHandler contract), and
that BacktestEngine actually accepts an injected market_data source.
"""

import duckdb
import pandas as pd

from backtesting.backtest_engine import BacktestEngine
from data.market_data import MarketDataHandler
from data.pit_warehouse import PitWarehouse
from data.warehouse_feed import WarehouseMarketData

_SEP_COLS = "t(ticker,date,open,high,low,close,volume,closeadj)"


def _write_sep(path, rows):
    values = ",".join("(" + ",".join(r) + ")" for r in rows)
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS {_SEP_COLS}) "
        f"TO '{path}' (FORMAT PARQUET)")


def _fixture(tmp_path):
    _write_sep(tmp_path / "sep.parquet", [
        # day1 pre-split: close 100 but closeadj 50 -> adjust factor 0.5
        ("'AAPL'", "DATE '2020-01-02'", "100.0", "110.0", "90.0", "100.0",
         "1000.0", "50.0"),
        # day2: close == closeadj 60 -> factor 1.0, unchanged
        ("'AAPL'", "DATE '2020-01-03'", "60.0", "66.0", "54.0", "60.0",
         "2000.0", "60.0"),
    ])
    return PitWarehouse(str(tmp_path))


def test_ohlcv_is_adjusted_and_shaped(tmp_path):
    w = _fixture(tmp_path)
    df = w.ohlcv('AAPL', '2020-01-01', '2020-01-31')
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert isinstance(df.index, pd.DatetimeIndex)
    # REBASED to raw at the window start (close[0]=100): day1 == raw O/H/L/C,
    # volume untouched. (factor 0.5 * scale 2.0 = 1.0)
    r0 = df.iloc[0]
    assert (r0['open'], r0['high'], r0['low'], r0['close'], r0['volume']) == \
        (100.0, 110.0, 90.0, 100.0, 1000.0)
    # day2: +20% total return preserved (raw closeadj 50->60), rebased to 120
    r1 = df.iloc[1]
    assert (r1['open'], r1['close']) == (120.0, 120.0)


def test_ohlcv_empty_for_unknown_or_uningested(tmp_path):
    assert _fixture(tmp_path).ohlcv('ZZZZ', '2020-01-01', '2020-12-31').empty
    assert PitWarehouse(str(tmp_path / "nope")).ohlcv(
        'AAPL', '2020-01-01', '2020-12-31').empty


def test_adapter_delegates_to_warehouse(tmp_path):
    feed = WarehouseMarketData(warehouse=_fixture(tmp_path))
    df = feed.fetch_stock_data('AAPL', '2020-01-01', '2020-01-31')
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert df.iloc[0]['close'] == 100.0        # rebased to raw at window start
    assert feed.get_last_fetch_info('AAPL')['provider'] == 'pit_warehouse'
    # unknown name -> empty frame (matches MarketDataHandler's _empty_data)
    assert feed.fetch_stock_data('ZZZZ', '2020-01-01', '2020-01-31').empty


def test_engine_accepts_injected_market_data(tmp_path):
    feed = WarehouseMarketData(warehouse=_fixture(tmp_path))
    eng = BacktestEngine(strategy=object(), market_data=feed)
    assert eng.market_data is feed
    # default is still the live handler
    assert isinstance(BacktestEngine(strategy=object()).market_data,
                      MarketDataHandler)
