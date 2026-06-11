"""Tests for backtesting.backtest_engine.BacktestEngine.

Offline and deterministic: MarketDataHandler.fetch_stock_data is
monkeypatched to return synthetic OHLCV frames; a scripted strategy emits
signals on known dates. The core property under test: a signal generated
from day T's data fills at day T+1's OPEN (with slippage and commission),
never at day T's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine, MAX_PENDING_DAYS
from core.models import Asset
from data.market_data import MarketDataHandler
from strategies.base import Strategy

COMMISSION = 0.001
DEFAULT_SLIPPAGE_BPS = 5.0


class ScriptedStrategy(Strategy):
    """Emits BUY/SELL when the last bar of the window hits a scripted date."""

    def __init__(self, buy_date=None, sell_date=None, symbol=None):
        super().__init__("Scripted Strategy")
        self.buy_date = pd.Timestamp(buy_date) if buy_date is not None else None
        self.sell_date = pd.Timestamp(sell_date) if sell_date is not None else None
        self.symbol = symbol  # None = act on every symbol

    def generate_signals(self, data: pd.DataFrame, asset: Asset) -> str:
        if self.symbol is not None and asset.symbol != self.symbol:
            return 'HOLD'
        current_date = data.index[-1]
        if self.buy_date is not None and current_date == self.buy_date:
            return 'BUY'
        if self.sell_date is not None and current_date == self.sell_date:
            return 'SELL'
        return 'HOLD'


@pytest.fixture
def patch_market_data(monkeypatch):
    """Patch MarketDataHandler.fetch_stock_data to serve canned frames."""

    def _patch(frames_by_symbol):
        def fake_fetch(self, symbol, start_date, end_date):
            return frames_by_symbol.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

    return _patch


class TestNextDayOpenExecution:
    def test_buy_fills_next_day_open_with_slippage_not_signal_day_close(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=20, seed=7)
        patch_market_data({'TEST': df})

        dates = df.index
        buy_date = dates[10]
        strategy = ScriptedStrategy(buy_date=buy_date)
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)

        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        assert len(engine.trades_log) == 1
        trade = engine.trades_log[0]

        next_day_open = float(df.loc[dates[11], 'open'])
        expected_fill = next_day_open * (1 + DEFAULT_SLIPPAGE_BPS / 10000.0)

        assert trade['action'] == 'BUY'
        assert trade['date'] == dates[11]            # fill date = T+1
        assert trade['signal_date'] == buy_date      # signal date = T
        assert trade['price'] == pytest.approx(expected_fill)
        # The regression this guards against: same-bar fills at T's close.
        assert trade['price'] != pytest.approx(float(df.loc[buy_date, 'close']))

        # Sizing at fill time: 10% of the (all-cash) portfolio.
        expected_qty = int(100000.0 * 0.1 / expected_fill)
        assert trade['quantity'] == expected_qty
        # Commission applied on cost.
        assert trade['cost'] == pytest.approx(
            expected_qty * expected_fill * (1 + COMMISSION))
        assert 'error' not in report

    def test_round_trip_cash_flow_is_consistent(self, make_ohlcv,
                                                patch_market_data):
        df = make_ohlcv(n_days=20, seed=11)
        patch_market_data({'TEST': df})

        dates = df.index
        strategy = ScriptedStrategy(buy_date=dates[5], sell_date=dates[12])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        assert [t['action'] for t in engine.trades_log] == ['BUY', 'SELL']
        buy, sell = engine.trades_log

        buy_fill = float(df.loc[dates[6], 'open']) * (1 + DEFAULT_SLIPPAGE_BPS / 10000.0)
        sell_fill = float(df.loc[dates[13], 'open']) * (1 - DEFAULT_SLIPPAGE_BPS / 10000.0)

        assert buy['date'] == dates[6]
        assert sell['date'] == dates[13]
        assert sell['signal_date'] == dates[12]
        assert sell['price'] == pytest.approx(sell_fill)
        assert sell['proceeds'] == pytest.approx(
            buy['quantity'] * sell_fill * (1 - COMMISSION))

        # Position is flat at the end; all value is cash.
        assert engine.portfolio.positions == {}
        expected_cash = 100000.0 - buy['cost'] + sell['proceeds']
        assert engine.portfolio.cash == pytest.approx(expected_cash)
        assert report['summary']['current_value'] == pytest.approx(expected_cash)

        # The closed trade records the slipped fill prices.
        assert len(report['closed_trades']) == 1
        closed = report['closed_trades'][0]
        assert closed['entry_price'] == pytest.approx(buy_fill)
        assert closed['exit_price'] == pytest.approx(sell_fill)

    def test_portfolio_history_marks_positions_at_daily_close(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=20, seed=13)
        patch_market_data({'TEST': df})

        dates = df.index
        strategy = ScriptedStrategy(buy_date=dates[3])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        history = report['portfolio_history']
        assert len(history) == len(dates)  # one snapshot per trading day

        buy = engine.trades_log[0]
        cash_after_buy = 100000.0 - buy['cost']
        # Snapshot on the fill day values the position at THAT day's close.
        snapshot = history[4]
        assert snapshot['timestamp'] == dates[4]
        expected_value = cash_after_buy + buy['quantity'] * float(df.loc[dates[4], 'close'])
        assert snapshot['portfolio_value'] == pytest.approx(expected_value)

    def test_custom_slippage_bps_is_applied(self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=20, seed=17)
        patch_market_data({'TEST': df})

        dates = df.index
        strategy = ScriptedStrategy(buy_date=dates[8])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION, slippage_bps=100.0)
        engine.run(['TEST'], '2023-01-01', '2023-12-31', position_size=0.1)

        trade = engine.trades_log[0]
        expected_fill = float(df.loc[dates[9], 'open']) * 1.01  # 100 bps
        assert trade['price'] == pytest.approx(expected_fill)


class TestMissingOpenFallback:
    def test_nan_open_falls_back_to_same_day_close(self, make_ohlcv,
                                                   patch_market_data):
        df = make_ohlcv(n_days=20, seed=19)
        dates = df.index
        df.loc[dates[11], 'open'] = np.nan  # fill day has no open print
        patch_market_data({'TEST': df})

        strategy = ScriptedStrategy(buy_date=dates[10])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        engine.run(['TEST'], '2023-01-01', '2023-12-31', position_size=0.1)

        assert len(engine.trades_log) == 1
        trade = engine.trades_log[0]
        expected_fill = float(df.loc[dates[11], 'close']) * (
            1 + DEFAULT_SLIPPAGE_BPS / 10000.0)
        assert trade['date'] == dates[11]
        assert trade['price'] == pytest.approx(expected_fill)


class TestLastDaySignals:
    def test_signal_on_final_day_never_fills_and_is_reported_pending(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=20, seed=23)
        patch_market_data({'TEST': df})

        last_day = df.index[-1]
        strategy = ScriptedStrategy(buy_date=last_day)
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        assert engine.trades_log == []
        assert engine.portfolio.positions == {}
        assert report['pending_signals'] == [{
            'symbol': 'TEST',
            'signal': 'BUY',
            'signal_date': last_day,
        }]
        # No trades: portfolio stayed all-cash.
        assert report['summary']['current_value'] == pytest.approx(100000.0)


class TestPendingIntentExpiry:
    def test_intent_dropped_after_max_pending_days_without_a_bar(
            self, make_ohlcv, patch_market_data):
        # Symbol AAA trades all 20 days and keeps the calendar going; BBB
        # stops trading after day 9, so a BUY signalled on BBB's final bar
        # can never fill and must be dropped after MAX_PENDING_DAYS.
        df_a = make_ohlcv(n_days=20, seed=29)
        df_b = make_ohlcv(n_days=10, seed=31)
        patch_market_data({'AAA': df_a, 'BBB': df_b})

        bbb_last_bar = df_b.index[-1]
        strategy = ScriptedStrategy(buy_date=bbb_last_bar, symbol='BBB')
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['AAA', 'BBB'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        # 10 trading days remain after the signal — more than MAX_PENDING_DAYS,
        # so the intent expired rather than lingering to the end of the run.
        assert MAX_PENDING_DAYS < 10
        assert engine.trades_log == []
        assert report['pending_signals'] == []
