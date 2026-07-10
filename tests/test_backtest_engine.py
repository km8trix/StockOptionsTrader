"""Tests for backtesting.backtest_engine.BacktestEngine.

Offline and deterministic: MarketDataHandler.fetch_stock_data is
monkeypatched to return synthetic OHLCV frames; a scripted strategy emits
signals on known dates. The core property under test: a signal generated
from day T's data fills at day T+1's OPEN (with slippage and commission),
never at day T's close.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine, MAX_PENDING_DAYS
from core.models import Asset
from data.market_data import MarketDataHandler
from portfolio.manager import PortfolioManager
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


class TestNaNCloseMarkGuard:
    """Phase 0: a NaN close on a mark-to-market day must NOT overwrite an open
    position's price. Before the guard, one NaN flowed through
    get_portfolio_value() and turned the WHOLE equity curve — and every
    metric — into NaN while the backtest still 'completed'."""

    def test_nan_close_keeps_prior_mark_and_metrics_stay_finite(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=20, seed=23)
        dates = df.index
        # Open a position early (buy at dates[3] -> fills dates[4]), then a
        # later mark day (dates[8]) reports a NaN close.
        df.loc[dates[8], 'close'] = np.nan
        patch_market_data({'TEST': df})

        strategy = ScriptedStrategy(buy_date=dates[3])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        # 1. No snapshot is poisoned to NaN.
        values = [h['portfolio_value'] for h in report['portfolio_history']]
        assert len(values) == len(dates)
        assert all(np.isfinite(v) for v in values)

        # 2. On the NaN-close day the position kept its PRIOR (dates[7]) mark.
        buy = engine.trades_log[0]
        cash_after_buy = 100000.0 - buy['cost']
        prior_close = float(df.loc[dates[7], 'close'])
        expected = cash_after_buy + buy['quantity'] * prior_close
        assert report['portfolio_history'][8]['portfolio_value'] == pytest.approx(
            expected)

        # 3. Downstream metrics are real numbers, not NaN.
        assert np.isfinite(engine.portfolio.get_sharpe_ratio())
        assert np.isfinite(report['summary']['current_value'])

    def test_inf_close_is_also_rejected(self, make_ohlcv, patch_market_data):
        # np.isnan would let +inf through; the guard uses np.isfinite, so an
        # inf close (provider parse glitch) cannot poison the equity curve.
        df = make_ohlcv(n_days=20, seed=29)
        dates = df.index
        df.loc[dates[8], 'close'] = np.inf
        patch_market_data({'TEST': df})

        strategy = ScriptedStrategy(buy_date=dates[3])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1)

        values = [h['portfolio_value'] for h in report['portfolio_history']]
        assert all(np.isfinite(v) for v in values)
        assert np.isfinite(engine.portfolio.get_sharpe_ratio())


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


class TestProgressCallback:
    def test_progress_monotonic_and_hits_100_exactly_once(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=20, seed=37)
        patch_market_data({'TEST': df})

        strategy = ScriptedStrategy(buy_date=df.index[5])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        calls: list = []
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1,
                            progress_callback=calls.append,
                            benchmark_symbol=None)

        assert 'error' not in report
        # One call per simulated trading day.
        assert len(calls) == len(df.index)
        # Monotonically nondecreasing, bounded by (0, 100].
        assert all(later >= earlier
                   for earlier, later in zip(calls, calls[1:]))
        assert all(0.0 < value <= 100.0 for value in calls)
        # 100.0 exactly once, and it is the final call.
        assert calls[-1] == 100.0
        assert calls.count(100.0) == 1

    def test_single_day_backtest_reports_100_once(self, make_ohlcv,
                                                  patch_market_data):
        df = make_ohlcv(n_days=1, seed=41)
        patch_market_data({'TEST': df})

        engine = BacktestEngine(ScriptedStrategy(), initial_capital=100000.0,
                                commission=COMMISSION)
        calls: list = []
        engine.run(['TEST'], '2023-01-01', '2023-12-31',
                   progress_callback=calls.append, benchmark_symbol=None)
        assert calls == [100.0]


class TestBenchmark:
    def test_benchmark_is_normalized_buy_and_hold_of_initial_capital(
            self, make_ohlcv, patch_market_data):
        initial_capital = 100000.0
        df = make_ohlcv(n_days=20, seed=43)
        spy = make_ohlcv(n_days=20, seed=99, start_price=400.0)
        patch_market_data({'TEST': df, 'SPY': spy})

        engine = BacktestEngine(ScriptedStrategy(),
                                initial_capital=initial_capital,
                                commission=COMMISSION)
        # benchmark_symbol defaults to 'SPY'.
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31')

        bench = report['benchmark']
        assert bench is not None
        assert bench['symbol'] == 'SPY'

        curve = bench['equity_curve']
        assert len(curve) == len(spy.index)
        # Normalized: the first marked value is exactly the initial capital.
        assert curve[0]['value'] == pytest.approx(initial_capital)

        # Each point is a buy-and-hold mark at that session's close.
        base_close = float(spy['close'].iloc[0])
        for point, (ts, close) in zip(curve, spy['close'].items()):
            assert point['date'] == ts.strftime('%Y-%m-%d')
            assert point['value'] == pytest.approx(
                float(close) / base_close * initial_capital)

        # All benchmark dates fall within the requested backtest range.
        dates = [point['date'] for point in curve]
        assert min(dates) >= '2023-01-01'
        assert max(dates) <= '2023-12-31'

    def test_benchmark_fetch_failure_does_not_break_run(
            self, make_ohlcv, monkeypatch, caplog):
        df = make_ohlcv(n_days=20, seed=47)

        def fake_fetch(self, symbol, start_date, end_date):
            if symbol == 'SPY':
                raise RuntimeError('provider down')
            return df

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        engine = BacktestEngine(ScriptedStrategy(buy_date=df.index[5]),
                                initial_capital=100000.0,
                                commission=COMMISSION)
        with caplog.at_level('WARNING', logger='backtesting.backtest_engine'):
            report = engine.run(['TEST'], '2023-01-01', '2023-12-31')

        assert 'error' not in report
        assert report['benchmark'] is None
        # The backtest itself still ran to completion.
        assert len(report['portfolio_history']) == len(df.index)
        assert len(engine.trades_log) == 1
        assert any('Benchmark' in record.message for record in caplog.records)

    def test_benchmark_empty_data_yields_none(self, make_ohlcv,
                                              patch_market_data):
        df = make_ohlcv(n_days=20, seed=53)
        # No SPY frame: the patched fetch serves an empty DataFrame for it.
        patch_market_data({'TEST': df})

        engine = BacktestEngine(ScriptedStrategy(), initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31')
        assert 'error' not in report
        assert report['benchmark'] is None

    def test_benchmark_none_disables_benchmark(self, make_ohlcv,
                                               patch_market_data):
        df = make_ohlcv(n_days=20, seed=59)
        patch_market_data({'TEST': df})

        engine = BacktestEngine(ScriptedStrategy(), initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)
        assert report['benchmark'] is None


class TestEngineModeSelection:
    """Exactly one of strategy=/desk= drives the engine (contract C2)."""

    def test_both_strategy_and_desk_raises(self):
        from desks.registry import create_desk
        with pytest.raises(ValueError, match='exactly one'):
            BacktestEngine(strategy=ScriptedStrategy(),
                           desk=create_desk('foundation'),
                           initial_capital=100000.0)

    def test_neither_strategy_nor_desk_raises(self):
        with pytest.raises(ValueError, match='exactly one'):
            BacktestEngine(initial_capital=100000.0)

    def test_strategy_mode_report_carries_no_desk_keys(self, make_ohlcv,
                                                       patch_market_data):
        df = make_ohlcv(n_days=20, seed=67)
        patch_market_data({'TEST': df})

        strategy = ScriptedStrategy(buy_date=df.index[5])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.1, benchmark_symbol=None)

        assert 'error' not in report
        # Backward compatibility: strategy-mode reports are unchanged.
        assert 'desk' not in report
        assert 'trader_notes' not in report
        assert 'walk_forward' not in report
        assert report['strategy'] == 'Scripted Strategy'


class TestDrawdownSeries:
    def test_hand_computed_sequence(self):
        # Values: 100 -> 110 -> 99 -> 104.5 -> 88.
        # Running max: 100, 110, 110, 110, 110.
        # Drawdowns:     0%,   0%, -10%,  -5%, -20%.
        pm = PortfolioManager(initial_capital=100.0)
        values = [100.0, 110.0, 99.0, 104.5, 88.0]
        dates = pd.bdate_range('2023-01-02', periods=len(values))
        for ts, value in zip(dates, values):
            pm.portfolio_history.append(
                {'timestamp': ts, 'portfolio_value': value})

        series = pm.get_drawdown_series()
        assert [point['date'] for point in series] == [
            '2023-01-02', '2023-01-03', '2023-01-04',
            '2023-01-05', '2023-01-06']
        expected = [0.0, 0.0, -10.0, -5.0, -20.0]
        assert [point['drawdown_pct'] for point in series] == pytest.approx(
            expected)
        # Same math as get_max_drawdown: the series minimum IS the max DD.
        assert min(point['drawdown_pct'] for point in series) == \
            pytest.approx(pm.get_max_drawdown())

    def test_empty_history_returns_empty_series(self):
        assert PortfolioManager(initial_capital=100.0).get_drawdown_series() == []

    def test_report_drawdown_series_matches_history_and_max_drawdown(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=30, seed=61)
        patch_market_data({'TEST': df})

        strategy = ScriptedStrategy(buy_date=df.index[2])
        engine = BacktestEngine(strategy, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            position_size=0.5, benchmark_symbol=None)

        series = report['drawdown_series']
        history = report['portfolio_history']
        assert len(series) == len(history) == len(df.index)

        # Hand-recompute the running-max drawdown with plain Python.
        running_max = float('-inf')
        for point, snapshot in zip(series, history):
            value = snapshot['portfolio_value']
            running_max = max(running_max, value)
            expected_dd = (value - running_max) / running_max * 100.0
            assert point['date'] == snapshot['timestamp'].strftime('%Y-%m-%d')
            assert point['drawdown_pct'] == pytest.approx(expected_dd)
            assert point['drawdown_pct'] <= 0.0

        assert min(point['drawdown_pct'] for point in series) == \
            pytest.approx(engine.portfolio.get_max_drawdown())
        assert min(point['drawdown_pct'] for point in series) == \
            pytest.approx(report['summary']['max_drawdown'])


def _vol_frame(dates, opens, volume):
    """OHLCV frame: scalar or per-date opens, constant volume."""
    n = len(dates)
    op = [float(opens)] * n if not isinstance(opens, (list, tuple)) else \
        [float(o) for o in opens]
    return pd.DataFrame({
        'open': op, 'high': [o * 1.01 for o in op],
        'low': [o * 0.99 for o in op], 'close': op,
        'volume': [float(volume)] * n,
    }, index=dates)


class TestRealisticFills:
    """Phase 3 Step 6 (opt-in): ADV market impact + cap-and-requeue partial
    fills. Default OFF is byte-identical (covered by the rest of this file +
    the greeks golden); these exercise the ON path directly."""

    def _engine(self, **kw):
        from strategies.base import MomentumStrategy
        return BacktestEngine(MomentumStrategy(), initial_capital=10_000_000.0,
                              commission=COMMISSION, **kw)

    def _asset(self):
        from core.models import Asset, AssetType
        return Asset('AAA', AssetType.STOCK)

    def test_realistic_buy_caps_impacts_and_requeues(self):
        dates = pd.bdate_range('2023-01-02', periods=6)
        frame = _vol_frame(dates, 100.0, 10000)
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True, impact_coef=0.1,
                              participation_cap=0.1, adv_window=5)
        engine.pending_intents[asset] = {
            'signal': 'BUY', 'signal_date': dates[4], 'days_waiting': 0,
            'quantity': 5000}
        engine._fill_pending_intents({'AAA': frame}, dates[5],
                                     position_size=0.1)

        # ADV = 10000 (5 prior days), cap = 1000, impact = 0.1*sqrt(1000/10000).
        impact = 0.1 * math.sqrt(1000 / 10000.0)
        expected_price = 100.0 * (1 + DEFAULT_SLIPPAGE_BPS / 10000.0 + impact)
        assert len(engine.trades_log) == 1
        trade = engine.trades_log[0]
        assert trade['quantity'] == 1000  # capped, not 5000
        assert trade['price'] == pytest.approx(expected_price)
        assert engine.portfolio.get_position(asset).quantity == 1000
        # Remainder re-queued to fill on later days.
        requeued = engine.pending_intents[asset]
        assert requeued['quantity'] == 4000
        assert requeued['accumulate'] is True

    def test_requeued_remainder_accumulates_into_position(self):
        dates = pd.bdate_range('2023-01-02', periods=8)
        # Different opens on the two fill days so the average entry is non-trivial.
        frame = _vol_frame(dates, [100.0] * 6 + [110.0, 110.0], 10000)
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True, impact_coef=0.1,
                              participation_cap=0.1, adv_window=5)
        engine.pending_intents[asset] = {
            'signal': 'BUY', 'signal_date': dates[5], 'days_waiting': 0,
            'quantity': 5000}

        engine._fill_pending_intents({'AAA': frame}, dates[6],
                                     position_size=0.1)
        p1 = engine.trades_log[0]['price']
        assert engine.portfolio.get_position(asset).quantity == 1000

        # Next day the re-queued 4000 fills another 1000 and accumulates.
        engine._fill_pending_intents({'AAA': frame}, dates[7],
                                     position_size=0.1)
        p2 = engine.trades_log[1]['price']
        pos = engine.portfolio.get_position(asset)
        assert pos.quantity == 2000
        assert pos.avg_entry_price == pytest.approx((1000 * p1 + 1000 * p2)
                                                    / 2000)
        assert engine.pending_intents[asset]['quantity'] == 3000

    def test_flag_off_fills_fully_no_cap(self):
        dates = pd.bdate_range('2023-01-02', periods=6)
        frame = _vol_frame(dates, 100.0, 10000)
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=False)
        engine.pending_intents[asset] = {
            'signal': 'BUY', 'signal_date': dates[4], 'days_waiting': 0,
            'quantity': 5000}
        engine._fill_pending_intents({'AAA': frame}, dates[5],
                                     position_size=0.1)

        trade = engine.trades_log[0]
        assert trade['quantity'] == 5000  # no cap when off
        assert trade['price'] == pytest.approx(
            100.0 * (1 + DEFAULT_SLIPPAGE_BPS / 10000.0))  # no impact
        assert asset not in engine.pending_intents  # nothing re-queued

    def test_thin_adv_cap_floors_at_one_share(self):
        # Review fix (HIGH): int(participation_cap*ADV) rounds to 0 for thin
        # names; the cap is floored at 1 so the order trickles, never dumps.
        dates = pd.bdate_range('2023-01-02', periods=6)
        frame = _vol_frame(dates, 100.0, 5)  # ADV = 5 shares
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True, impact_coef=0.1,
                              participation_cap=0.1, adv_window=5)
        engine.pending_intents[asset] = {
            'signal': 'BUY', 'signal_date': dates[4], 'days_waiting': 0,
            'quantity': 5000}
        engine._fill_pending_intents({'AAA': frame}, dates[5],
                                     position_size=0.1)
        assert engine.trades_log[0]['quantity'] == 1  # floored, not 5000
        assert engine.pending_intents[asset]['quantity'] == 4999

    def test_new_signal_clobbering_remainder_warns(self, caplog):
        # Review fix (HIGH): a fresh signal that supersedes an in-flight
        # remainder must surface the abandoned shares, not drop them silently.
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True)
        engine.pending_intents[asset] = {
            'signal': 'BUY', 'signal_date': None, 'days_waiting': 0,
            'quantity': 4000, 'accumulate': True}
        with caplog.at_level('WARNING'):
            engine._queue_pending_intent(asset, {
                'signal': 'SELL', 'signal_date': None, 'days_waiting': 0})
        assert engine.pending_intents[asset]['signal'] == 'SELL'
        assert any('Abandoning' in r.message for r in caplog.records)

    def test_orphaned_accumulate_remainder_dropped(self):
        # Review fix (MEDIUM): an accumulate remainder whose position is gone
        # (flat/flipped) is dropped deterministically, not re-opened/silently lost.
        dates = pd.bdate_range('2023-01-02', periods=6)
        frame = _vol_frame(dates, 100.0, 10000)
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True)
        # accumulate remainder but NO existing position (it was closed).
        engine.pending_intents[asset] = {
            'signal': 'BUY', 'signal_date': dates[4], 'days_waiting': 0,
            'quantity': 1000, 'accumulate': True}
        engine._fill_pending_intents({'AAA': frame}, dates[5],
                                     position_size=0.1)
        assert engine.trades_log == []  # not re-opened
        assert engine.portfolio.get_position(asset) is None
        assert asset not in engine.pending_intents  # consumed, not re-queued

    def test_impact_clamped_keeps_fill_price_positive(self):
        # Review fix (MEDIUM): a huge (uncapped) close into thin ADV must not
        # produce a negative fill price; impact is clamped (mirrors options 0.0).
        from core.models import Position
        dates = pd.bdate_range('2023-01-02', periods=6)
        frame = _vol_frame(dates, 100.0, 100)  # ADV = 100
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True, impact_coef=0.1,
                              participation_cap=0.1, adv_window=5)
        engine.portfolio.add_position(Position(
            asset=asset, quantity=1_000_000, avg_entry_price=90.0,
            current_price=100.0, timestamp=dates[0]))
        engine.pending_intents[asset] = {
            'signal': 'SELL', 'signal_date': dates[4], 'days_waiting': 0}
        engine._fill_pending_intents({'AAA': frame}, dates[5],
                                     position_size=0.1)
        # Uncapped impact would be 0.1*sqrt(1e6/100)=10; clamped to 0.5.
        expected = 100.0 * (1 - DEFAULT_SLIPPAGE_BPS / 10000.0 - 0.5)
        trade = engine.trades_log[0]
        assert trade['price'] > 0
        assert trade['price'] == pytest.approx(expected)

    def test_realistic_sell_pays_impact_no_cap(self):
        from core.models import Position
        dates = pd.bdate_range('2023-01-02', periods=6)
        frame = _vol_frame(dates, 100.0, 10000)
        asset = self._asset()
        engine = self._engine(enable_realistic_fills=True, impact_coef=0.1,
                              participation_cap=0.1, adv_window=5)
        engine.portfolio.add_position(Position(
            asset=asset, quantity=500, avg_entry_price=90.0,
            current_price=100.0, timestamp=dates[0]))
        engine.pending_intents[asset] = {
            'signal': 'SELL', 'signal_date': dates[4], 'days_waiting': 0}
        engine._fill_pending_intents({'AAA': frame}, dates[5],
                                     position_size=0.1)

        # Full exit (no cap on closes) but pays slippage + impact on 500 shares.
        impact = 0.1 * math.sqrt(500 / 10000.0)
        expected_price = 100.0 * (1 - DEFAULT_SLIPPAGE_BPS / 10000.0 - impact)
        trade = engine.trades_log[0]
        assert trade['action'] == 'SELL'
        assert trade['quantity'] == 500
        assert trade['price'] == pytest.approx(expected_price)
        assert engine.portfolio.get_position(asset) is None
        assert asset not in engine.pending_intents


class TestCashYield:
    """Opt-in dated idle-cash yield (cash_yield=). Default None is
    byte-identical: cash is never touched outside fills (pinned exactly,
    no approx). The ON path accrues cash * rate(date)/252 just before the
    daily snapshot, so the recorded equity curve carries the interest."""

    #: 3 trading days; a HOLD-only strategy leaves cash untouched except
    #: for the accrual, so snapshot cash is pure compounded interest.
    DATES = pd.bdate_range('2023-01-02', periods=3)
    #: Dated regime split: a ZIRP print on day 1, a 2023-style 5% after —
    #: the whole point of a dated series vs a flat retro assumption.
    RATES = {DATES[0]: 0.001, DATES[1]: 0.05, DATES[2]: 0.05}

    def _run(self, patch_market_data, make_ohlcv, **engine_kw):
        df = make_ohlcv(n_days=3)
        patch_market_data({'TEST': df})
        engine = BacktestEngine(ScriptedStrategy(),  # never signals
                                initial_capital=100000.0,
                                commission=COMMISSION, **engine_kw)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31')
        assert 'error' not in report
        return [h['cash'] for h in report['portfolio_history']]

    def test_accrual_math_three_days_hand_computed(self, patch_market_data,
                                                   make_ohlcv):
        cash = self._run(patch_market_data, make_ohlcv,
                         cash_yield=lambda d: self.RATES[d])
        # Hand-computed daily compounding on 100k (c += c*r/252 each day):
        #   day1 @0.1%: +100000*0.001/252       = +0.39682539...
        #   day2 @5%:   +100000.3968...*0.05/252 = +19.84134857...
        #   day3 @5%:   +100020.2381...*0.05/252 = +19.84528535...
        assert cash[0] == pytest.approx(100000.39682539682, rel=1e-12)
        assert cash[1] == pytest.approx(100020.23817397328, rel=1e-12)
        assert cash[2] == pytest.approx(100040.08345932527, rel=1e-12)
        # ZIRP-vs-5% is visible in the increments (dated, not flat).
        assert cash[1] - cash[0] > 40 * (cash[0] - 100000.0)

    def test_series_input_forward_fills_and_prehistory_is_zero(
            self, patch_market_data, make_ohlcv):
        # A pd.Series is accepted directly (wrapped in rate_asof): first
        # trading day predates the series -> 0.0 accrual; the single 5%
        # print then forward-fills across the remaining days.
        series = pd.Series([0.05], index=[self.DATES[1]])
        cash = self._run(patch_market_data, make_ohlcv, cash_yield=series)
        assert cash[0] == 100000.0                       # pre-history: zero
        assert cash[1] == pytest.approx(100000.0 * (1 + 0.05 / 252),
                                        rel=1e-12)
        assert cash[2] == pytest.approx(cash[1] * (1 + 0.05 / 252),
                                        rel=1e-12)       # ffilled carry

    def test_default_none_is_byte_identical(self, patch_market_data,
                                            make_ohlcv):
        # The default pin: with the param ABSENT, cash is EXACTLY the
        # initial capital on every snapshot of a no-trade run (the accrual
        # seam never touches it), and identical to an explicit
        # cash_yield=None run. Exact ==, not approx — byte identity.
        absent = self._run(patch_market_data, make_ohlcv)
        explicit_none = self._run(patch_market_data, make_ohlcv,
                                  cash_yield=None)
        assert absent == explicit_none
        assert absent == [100000.0] * 3
        # Not tautological: the ON path does change the same run.
        on = self._run(patch_market_data, make_ohlcv,
                       cash_yield=lambda d: 0.05)
        assert on != absent

    def test_negative_cash_accrues_nothing(self):
        # Margin-debit guard: no free leverage — a negative balance earns
        # zero (and is charged nothing; the docstring owns that asymmetry).
        engine = BacktestEngine(ScriptedStrategy(),
                                initial_capital=100000.0,
                                cash_yield=lambda d: 0.05)
        engine.portfolio.cash = -500.0
        engine._accrue_cash_yield(self.DATES[0])
        assert engine.portfolio.cash == -500.0
        # And zero cash accrues nothing (no 0*rate float dust).
        engine.portfolio.cash = 0.0
        engine._accrue_cash_yield(self.DATES[0])
        assert engine.portfolio.cash == 0.0


class TestEnrichedFeatureColumnsPlumbing:
    def test_engine_frames_carry_the_enriched_extras(
            self, make_ohlcv, patch_market_data):
        """run() enriches every symbol's frame with the extended-feature
        extras/seasonal columns so the ML models' predict fast paths can
        read them instead of rebuilding O(history) features daily."""
        from desks.features import ENRICHED_EXTRA_COLUMNS

        df = make_ohlcv(n_days=30, seed=11)
        patch_market_data({'TEST': df})
        engine = BacktestEngine(ScriptedStrategy(),  # never trades
                                initial_capital=100000.0,
                                commission=COMMISSION)
        engine.run(['TEST'], '2023-01-01', '2023-12-31', position_size=0.1)
        frame = engine._enriched_all['TEST']
        assert set(ENRICHED_EXTRA_COLUMNS) <= set(frame.columns)
        # Inert for existing consumers: raw OHLCV untouched.
        assert (frame['close'] == df['close']).all()
