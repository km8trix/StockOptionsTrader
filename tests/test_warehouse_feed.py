"""Tests for the survivorship-free warehouse price feed.

Covers PitWarehouse.ohlcv (split/div adjustment + shape), the WarehouseMarketData
adapter (delegates to the warehouse, matches the MarketDataHandler contract), and
that BacktestEngine actually accepts an injected market_data source.
"""

import duckdb
import pandas as pd
import pytest

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


def _write_tickers(path, delisted_ticker='DEAD'):
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES "
        f"('{delisted_ticker}', 1001, DATE '2010-01-01', "
        "DATE '2020-01-06', 'Y'),"
        "('LIVE', 1002, DATE '2010-01-01', DATE '2026-07-10', 'N')) "
        "AS t(ticker,permaticker,firstpricedate,lastpricedate,isdelisted)) "
        f"TO '{path}' (FORMAT PARQUET)")


def _write_actions(path, rows):
    values = ",".join("(" + ",".join(row) + ")" for row in rows)
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS "
        "t(date,action,ticker,value,contraticker)) "
        f"TO '{path}' (FORMAT PARQUET)")


def _write_daily(path):
    duckdb.connect().execute(
        "COPY (SELECT * FROM (VALUES "
        "('AAPL', DATE '2020-01-02', 1000000000.0),"
        "('AAPL', DATE '2020-01-03', 1100000000.0),"
        "('OTHER', DATE '2020-01-02', 2000000000.0)) "
        "AS t(ticker,date,marketcap)) "
        f"TO '{path}' (FORMAT PARQUET)")


def test_ohlcv_is_adjusted_and_shaped(tmp_path):
    w = _fixture(tmp_path)
    df = w.ohlcv('AAPL', '2020-01-01', '2020-01-31')
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert isinstance(df.index, pd.DatetimeIndex)
    # REBASED to raw at the window start (close[0]=100): day1 == raw O/H/L/C,
    # volume stays 1000 because the rebased adjustment is 1.0.
    r0 = df.iloc[0]
    assert (r0['open'], r0['high'], r0['low'], r0['close'], r0['volume']) == \
        (100.0, 110.0, 90.0, 100.0, 1000.0)
    # day2: +20% total return preserved (raw closeadj 50->60), rebased to 120
    r1 = df.iloc[1]
    assert (r1['open'], r1['close'], r1['volume']) == (120.0, 120.0, 1000.0)
    # Adjusted-dollar participation is invariant: raw 2000*60 == 1000*120.
    assert r1['volume'] * r1['close'] == 2000.0 * 60.0


def test_ohlcv_empty_for_unknown_or_uningested(tmp_path):
    assert _fixture(tmp_path).ohlcv('ZZZZ', '2020-01-01', '2020-12-31').empty
    assert PitWarehouse(str(tmp_path / "nope")).ohlcv(
        'AAPL', '2020-01-01', '2020-12-31').empty


def test_bulk_prices_match_single_ticker_series(tmp_path):
    warehouse = _fixture(tmp_path)

    bulk = warehouse.prices_bulk(
        ['MISSING', 'AAPL'], '2020-01-01', '2020-01-31')

    assert set(bulk) == {'AAPL'}
    pd.testing.assert_series_equal(
        bulk['AAPL'], warehouse.prices('AAPL', '2020-01-01', '2020-01-31'))


def test_bulk_dated_marketcaps_are_exactly_keyed(tmp_path):
    _write_daily(tmp_path / 'daily.parquet')
    warehouse = PitWarehouse(str(tmp_path))

    panel = warehouse.daily_marketcaps_for_dates(
        ['AAPL'], ['2020-01-02', '2020-01-03', '2020-01-04'])

    assert panel == {
        ('AAPL', pd.Timestamp('2020-01-02').date()): 1_000_000_000.0,
        ('AAPL', pd.Timestamp('2020-01-03').date()): 1_100_000_000.0,
    }


def test_adapter_delegates_to_warehouse(tmp_path):
    feed = WarehouseMarketData(warehouse=_fixture(tmp_path))
    df = feed.fetch_stock_data('AAPL', '2020-01-01', '2020-01-31')
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert df.iloc[0]['close'] == 100.0        # rebased to raw at window start
    assert feed.get_last_fetch_info('AAPL')['provider'] == 'pit_warehouse'
    # unknown name -> empty frame (matches MarketDataHandler's _empty_data)
    assert feed.fetch_stock_data('ZZZZ', '2020-01-01', '2020-01-31').empty


def test_snapshot_version_is_content_addressed_and_flags_missing(tmp_path):
    warehouse = _fixture(tmp_path)
    first = warehouse.snapshot_version(['sep', 'actions'])
    second = warehouse.snapshot_version(['actions', 'sep'])
    assert first == second
    assert first['complete'] is False
    assert first['quality_flags'] == ['missing_table:actions']

    _write_actions(tmp_path / 'actions.parquet', [
        ("DATE '2020-01-03'", "'Split'", "'AAPL'", "2.0", "NULL"),
    ])
    updated = warehouse.snapshot_version(['sep', 'actions'])
    assert updated['complete'] is True
    assert updated['quality_flags'] == []
    assert updated['version'] != first['version']


def test_market_sessions_come_from_observed_sep_dates(tmp_path):
    warehouse = _fixture(tmp_path)

    sessions = warehouse.market_sessions('2020-01-01', '2020-01-31')

    assert sessions.equals(pd.DatetimeIndex(
        ['2020-01-02', '2020-01-03'], name='date'))
    assert warehouse.market_sessions('2020-02-01', '2020-01-01').empty
    assert PitWarehouse(str(tmp_path / 'missing')).market_sessions(
        '2020-01-01', '2020-01-31').empty


def test_engine_accepts_injected_market_data(tmp_path):
    feed = WarehouseMarketData(warehouse=_fixture(tmp_path))
    eng = BacktestEngine(strategy=object(), market_data=feed)
    assert eng.market_data is feed
    # default is still the live handler
    assert isinstance(BacktestEngine(strategy=object()).market_data,
                      MarketDataHandler)


def test_delisting_date_requires_explicit_delisted_status(tmp_path):
    _write_sep(tmp_path / 'sep.parquet', [
        ("'DEAD'", "DATE '2020-01-06'", "8.0", "8.0", "8.0", "8.0",
         "1000.0", "8.0"),
    ])
    _write_tickers(tmp_path / 'tickers.parquet')
    warehouse = PitWarehouse(str(tmp_path))

    assert warehouse.delisting_date('DEAD') == pd.Timestamp(
        '2020-01-06').date()
    # A live row may carry the vendor snapshot's latest price date.
    assert warehouse.delisting_date('LIVE') is None

    duckdb.connect().execute(
        "COPY (SELECT 'UNFLAGGED' AS ticker, "
        "DATE '2020-01-06' AS lastpricedate) "
        f"TO '{tmp_path / 'tickers.parquet'}' (FORMAT PARQUET)")
    # Missing status evidence fails closed instead of assuming a delisting.
    assert PitWarehouse(str(tmp_path)).delisting_date('UNFLAGGED') is None


def test_engine_liquidates_delisted_holding_at_final_close(tmp_path):
    _write_sep(tmp_path / "sep.parquet", [
        ("'DEAD'", "DATE '2020-01-02'", "10.0", "10.0", "10.0", "10.0",
         "1000.0", "10.0"),
        ("'DEAD'", "DATE '2020-01-03'", "10.0", "10.0", "10.0", "10.0",
         "1000.0", "10.0"),
        ("'DEAD'", "DATE '2020-01-06'", "8.0", "8.0", "8.0", "8.0",
         "1000.0", "8.0"),
        # Keeps the union calendar running after DEAD's final session.
        ("'LIVE'", "DATE '2020-01-02'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
        ("'LIVE'", "DATE '2020-01-03'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
        ("'LIVE'", "DATE '2020-01-06'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
        ("'LIVE'", "DATE '2020-01-07'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
    ])
    _write_tickers(tmp_path / "tickers.parquet")

    class BuyDead:
        name = 'buy-dead'

        def generate_signals(self, data, asset):
            return ('BUY' if asset.symbol == 'DEAD'
                    and data.index[-1] == pd.Timestamp('2020-01-02')
                    else 'HOLD')

    feed = WarehouseMarketData(PitWarehouse(str(tmp_path)))
    engine = BacktestEngine(
        strategy=BuyDead(), market_data=feed, commission=0.0,
        slippage_bps=0.0)
    report = engine.run(
        ['DEAD', 'LIVE'], '2020-01-01', '2020-01-31',
        position_size=0.1, benchmark_symbol=None)

    assert 'error' not in report
    assert all(asset.symbol != 'DEAD' for asset in engine.portfolio.positions)
    liquidation = [t for t in engine.trades_log
                   if t.get('reason') == 'delisting liquidation']
    assert len(liquidation) == 1
    assert liquidation[0]['date'] == pd.Timestamp('2020-01-06')
    assert liquidation[0]['price'] == 8.0
    assert liquidation[0]['data_quality_flags'] == [
        'delisting_terms_unavailable']

    strict = BacktestEngine(
        strategy=BuyDead(), market_data=feed, commission=0.0,
        slippage_bps=0.0, enable_realistic_fills=True,
        reject_fills_without_adv=True, participation_cap=0.01)
    with pytest.raises(RuntimeError, match="lacks delisting terms"):
        strict.run(
            ['DEAD', 'LIVE'], '2020-01-01', '2020-01-31',
            position_size=0.1, benchmark_symbol=None)


@pytest.mark.parametrize(('symbol', 'reported_value', 'buyer'), [
    ('TWTR', 41093.7, 'X'),
    ('ATVI', 74289.5, 'MSFT'),
])
def test_acquisition_deal_value_is_never_used_as_share_price(
        tmp_path, symbol, reported_value, buyer):
    _write_sep(tmp_path / "sep.parquet", [
        (f"'{symbol}'", "DATE '2020-01-02'", "10.0", "10.0", "10.0", "10.0",
         "1000.0", "10.0"),
        (f"'{symbol}'", "DATE '2020-01-06'", "8.0", "8.0", "8.0", "8.0",
         "1000.0", "8.0"),
        ("'LIVE'", "DATE '2020-01-02'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
        ("'LIVE'", "DATE '2020-01-06'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
        ("'LIVE'", "DATE '2020-01-07'", "20.0", "20.0", "20.0", "20.0",
         "1000.0", "20.0"),
    ])
    _write_tickers(tmp_path / "tickers.parquet", symbol)
    _write_actions(tmp_path / "actions.parquet", [
        ("DATE '2020-01-06'", "'Acquisition'", f"'{symbol}'",
         str(reported_value), f"'{buyer}'"),
    ])

    class BuyDead:
        name = 'buy-dead-action'

        def generate_signals(self, data, asset):
            return ('BUY' if asset.symbol == symbol
                    and data.index[-1] == pd.Timestamp('2020-01-02')
                    else 'HOLD')

    feed = WarehouseMarketData(PitWarehouse(str(tmp_path)))
    classification = feed._wh.delisting_action(symbol)
    assert classification['reported_value'] == reported_value
    assert classification['value_semantics'] == \
        'vendor_reported_value_not_per_share'
    assert 'payout_per_share' not in classification
    payout = feed.delisting_payout(symbol, 8.0)
    assert payout['price'] == 8.0
    assert payout['reported_value'] == reported_value
    assert payout['source'] == 'final_tradable_close'

    engine = BacktestEngine(
        strategy=BuyDead(), market_data=feed, commission=0.0,
        slippage_bps=0.0)
    engine.run([symbol, 'LIVE'], '2020-01-01', '2020-01-31',
               position_size=0.1, benchmark_symbol=None)

    liquidation = [trade for trade in engine.trades_log
                   if trade.get('reason') == 'delisting liquidation'][0]
    assert liquidation['price'] == 8.0
    assert liquidation['price'] != reported_value
    assert liquidation['payout_source'] == 'final_tradable_close'
    assert liquidation['data_quality_flags'] == [
        'delisting_terms_unavailable',
        'corporate_action_value_not_per_share',
    ]

    strict = BacktestEngine(
        strategy=BuyDead(), market_data=feed, commission=0.0,
        slippage_bps=0.0, enable_realistic_fills=True,
        reject_fills_without_adv=True, participation_cap=0.01)
    with pytest.raises(RuntimeError, match="lacks delisting terms"):
        strict.run([symbol, 'LIVE'], '2020-01-01', '2020-01-31',
                   position_size=0.1, benchmark_symbol=None)


def test_bankruptcy_action_does_not_imply_zero_recovery(tmp_path):
    _write_sep(tmp_path / 'sep.parquet', [
        ("'DEAD'", "DATE '2020-01-06'", "1.25", "1.25", "1.25", "1.25",
         "1000.0", "1.25"),
    ])
    _write_tickers(tmp_path / 'tickers.parquet')
    _write_actions(tmp_path / 'actions.parquet', [
        ("DATE '2020-01-06'", "'Bankruptcy'", "'DEAD'", "0.0", "NULL"),
    ])
    feed = WarehouseMarketData(PitWarehouse(str(tmp_path)))

    action = feed._wh.delisting_action('DEAD')
    assert action['reported_value'] == 0.0
    assert 'payout_per_share' not in action
    payout = feed.delisting_payout('DEAD', 1.25)
    assert payout['price'] == 1.25
    assert payout['source'] == 'final_tradable_close'
    assert 'delisting_terms_unavailable' in payout['quality_flags']
