"""Tests for desk-mode SHORT/COVER support (contract C4).

Fill arithmetic is hand-computed exactly — slippage direction (a short is
a SELL so adverse slippage LOWERS the fill; a cover is a BUY so it RAISES
it) and commission signs included. Frames are hand-built (linear prices)
so every number is controlled; no network, no randomness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position, Trade
from data.market_data import MarketDataHandler
from desks.base import Desk, DeskIntent
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager
from strategies.base import Strategy

COMMISSION = 0.001
SLIPPAGE = 5.0 / 10000.0  # engine default 5 bps


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def linear_frame(n: int = 20, start: float = 100.0,
                 step: float = -1.0) -> pd.DataFrame:
    """Deterministic frame with open == close == start + i*step."""
    index = pd.bdate_range('2023-01-02', periods=n)
    price = start + step * np.arange(n, dtype=float)
    return pd.DataFrame({
        'open': price, 'high': price * 1.001, 'low': price * 0.999,
        'close': price, 'volume': np.full(n, 500_000.0),
    }, index=index)


class ScriptedDesk(Desk):
    """Desk emitting scripted intents keyed by simulation date."""

    def __init__(self, script=None, **kwargs):
        kwargs.setdefault('risk_manager', RiskManager(
            max_position_size=1.0, max_daily_loss=0.99,
            position_stop_loss=0.90))  # wide limits: fills stay pure
        super().__init__(key='scripted', name='Scripted Desk',
                         description='test-only desk', accent='#123456',
                         **kwargs)
        self.script = {pd.Timestamp(d): intents
                       for d, intents in (script or {}).items()}

    def generate_intents(self, all_data, date, portfolio):
        return list(self.script.get(pd.Timestamp(date), []))


@pytest.fixture
def patch_market_data(monkeypatch):
    def _patch(frames_by_symbol):
        def fake_fetch(self, symbol, start_date, end_date):
            return frames_by_symbol.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

    return _patch


def run_scripted(df, script, initial_capital=100000.0):
    desk = ScriptedDesk(script=script)
    engine = BacktestEngine(desk=desk, initial_capital=initial_capital,
                            commission=COMMISSION)
    report = engine.run(['TST'], '2023-01-01', '2023-12-31',
                        benchmark_symbol=None)
    return engine, desk, report


class TestShortFillArithmetic:
    def test_short_opens_negative_position_with_exact_fill_math(
            self, patch_market_data):
        df = linear_frame(step=-1.0)  # falling prices: 100, 99, 98, ...
        patch_market_data({'TST': df})
        dates = df.index
        script = {dates[5]: [DeskIntent(asset=stock('TST'), action='SHORT',
                                        size_fraction=0.05, reason='t')]}

        engine, _, _ = run_scripted(df, script)

        assert len(engine.trades_log) == 1
        trade = engine.trades_log[0]
        assert trade['action'] == 'SHORT'
        assert trade['date'] == dates[6]          # next-day open fill
        assert trade['signal_date'] == dates[5]

        # A short is a SELL: adverse slippage LOWERS the fill.
        open_t1 = float(df.loc[dates[6], 'open'])  # 100 - 6 = 94.0
        expected_fill = open_t1 * (1 - SLIPPAGE)
        assert trade['price'] == pytest.approx(expected_fill)
        assert trade['price'] < open_t1  # direction of slippage

        expected_qty = int(100000.0 * 1.0 * 0.05 / expected_fill)
        assert trade['quantity'] == expected_qty
        expected_proceeds = expected_qty * expected_fill * (1 - COMMISSION)
        assert trade['proceeds'] == pytest.approx(expected_proceeds)

        # Position is negative, entered at the fill; proceeds sit in cash.
        position = engine.portfolio.get_position(stock('TST'))
        assert position.quantity == -expected_qty
        assert position.avg_entry_price == pytest.approx(expected_fill)
        # Final cash = initial + proceeds (no other trades).
        assert engine.portfolio.cash == pytest.approx(
            100000.0 + expected_proceeds)

    def test_profitable_short_round_trip_cash_and_pnl(self,
                                                      patch_market_data):
        df = linear_frame(step=-1.0)  # falling: short is profitable
        patch_market_data({'TST': df})
        dates = df.index
        asset = stock('TST')
        script = {
            dates[3]: [DeskIntent(asset=asset, action='SHORT',
                                  size_fraction=0.05, reason='t')],
            dates[8]: [DeskIntent(asset=asset, action='COVER',
                                  size_fraction=1.0, reason='t')],
        }

        engine, _, report = run_scripted(df, script)

        assert [t['action'] for t in engine.trades_log] == ['SHORT', 'COVER']
        short, cover = engine.trades_log

        short_fill = float(df.loc[dates[4], 'open']) * (1 - SLIPPAGE)
        cover_fill = float(df.loc[dates[9], 'open']) * (1 + SLIPPAGE)
        qty = int(100000.0 * 0.05 / short_fill)

        assert short['price'] == pytest.approx(short_fill)
        assert cover['date'] == dates[9]
        assert cover['price'] == pytest.approx(cover_fill)
        assert cover['quantity'] == qty
        assert cover['cost'] == pytest.approx(
            qty * cover_fill * (1 + COMMISSION))

        # --- cash conservation through the round trip ---------------
        expected_cash = (100000.0
                         + qty * short_fill * (1 - COMMISSION)
                         - qty * cover_fill * (1 + COMMISSION))
        assert engine.portfolio.cash == pytest.approx(expected_cash)
        # Flat book: portfolio value == cash.
        assert engine.portfolio.get_portfolio_value() == pytest.approx(
            expected_cash)

        # --- Trade.pnl sign-correct: profitable short (exit < entry) -
        assert len(engine.portfolio.closed_trades) == 1
        closed = engine.portfolio.closed_trades[0]
        assert closed.quantity == -qty
        assert closed.pnl == pytest.approx(-qty * (cover_fill - short_fill))
        assert closed.pnl > 0
        assert closed.pnl_pct > 0  # direction-aware percentage

        # Report serialization carries the closed short.
        assert report['closed_trades'][0]['quantity'] == -qty
        assert report['closed_trades'][0]['pnl'] == pytest.approx(closed.pnl)

    def test_losing_short_round_trip_pnl_negative(self, patch_market_data):
        df = linear_frame(step=+1.0)  # rising: short loses
        patch_market_data({'TST': df})
        dates = df.index
        asset = stock('TST')
        script = {
            dates[3]: [DeskIntent(asset=asset, action='SHORT',
                                  size_fraction=0.05, reason='t')],
            dates[8]: [DeskIntent(asset=asset, action='COVER',
                                  size_fraction=1.0, reason='t')],
        }

        engine, _, _ = run_scripted(df, script)

        assert [t['action'] for t in engine.trades_log] == ['SHORT', 'COVER']
        closed = engine.portfolio.closed_trades[0]
        short_fill = float(df.loc[dates[4], 'open']) * (1 - SLIPPAGE)
        cover_fill = float(df.loc[dates[9], 'open']) * (1 + SLIPPAGE)
        assert closed.pnl == pytest.approx(
            closed.quantity * (cover_fill - short_fill))
        assert closed.pnl < 0
        assert closed.pnl_pct < 0

    def test_cover_without_short_and_short_on_existing_are_engine_noops(
            self, patch_market_data):
        df = linear_frame()
        patch_market_data({'TST': df})
        dates = df.index
        asset = stock('TST')
        script = {
            dates[2]: [DeskIntent(asset=asset, action='COVER',
                                  size_fraction=1.0, reason='t')],
            dates[5]: [DeskIntent(asset=asset, action='BUY',
                                  size_fraction=0.05, reason='t')],
            # SHORT against the open long is blocked at apply_risk, so it
            # never even reaches the engine fill path.
            dates[8]: [DeskIntent(asset=asset, action='SHORT',
                                  size_fraction=0.05, reason='t')],
        }
        engine, desk, _ = run_scripted(df, script)
        actions = [t['action'] for t in engine.trades_log]
        assert actions == ['BUY']
        blocked = [n for n in desk.notes if n.category == 'risk'
                   and 'Blocked SHORT' in n.message]
        assert len(blocked) == 1


class TestPartialCover:
    def test_portfolio_manager_partial_cover_keeps_remainder_open(self):
        portfolio = PortfolioManager(initial_capital=100000.0)
        asset = stock('SH')
        portfolio.add_position(Position(
            asset=asset, quantity=-100, avg_entry_price=50.0,
            current_price=48.0, timestamp=pd.Timestamp('2023-01-02')))

        portfolio.close_position(asset, 48.0, -40,
                                 pd.Timestamp('2023-01-02'),
                                 pd.Timestamp('2023-01-10'))

        position = portfolio.get_position(asset)
        assert position.quantity == -60  # -100 - (-40)
        assert len(portfolio.closed_trades) == 1
        trade = portfolio.closed_trades[0]
        assert trade.quantity == -40
        # Covered 40 shares 2 below entry: pnl = -40 * (48 - 50) = +80.
        assert trade.pnl == pytest.approx(80.0)

        # Full cover of the remainder removes the position.
        portfolio.close_position(asset, 51.0, -60,
                                 pd.Timestamp('2023-01-02'),
                                 pd.Timestamp('2023-01-12'))
        assert portfolio.get_position(asset) is None
        assert portfolio.closed_trades[1].pnl == pytest.approx(
            -60 * (51.0 - 50.0))


class TestApplyRiskShortRules:
    def _desk(self, **kwargs):
        return ScriptedDesk(**kwargs)

    def test_buy_on_short_blocked_with_risk_note(self):
        desk = self._desk()
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = PortfolioManager(initial_capital=100000.0)
        asset = stock('SH')
        portfolio.add_position(Position(
            asset=asset, quantity=-50, avg_entry_price=100.0,
            current_price=100.0, timestamp=pd.Timestamp('2023-05-01')))

        buy = DeskIntent(asset=asset, action='BUY', size_fraction=0.05,
                         reason='t')
        approved = desk.apply_risk([buy], portfolio, {},
                                   pd.Timestamp('2023-06-01'))

        assert approved == []
        notes = [n for n in desk.notes if n.category == 'risk']
        assert len(notes) == 1
        assert 'Blocked BUY SH' in notes[0].message
        assert 'flip' in notes[0].message.lower()
        assert notes[0].data['position_quantity'] == -50

    def test_short_on_long_blocked_with_risk_note(self):
        desk = self._desk()
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = PortfolioManager(initial_capital=100000.0)
        asset = stock('LG')
        portfolio.add_position(Position(
            asset=asset, quantity=50, avg_entry_price=100.0,
            current_price=100.0, timestamp=pd.Timestamp('2023-05-01')))

        short = DeskIntent(asset=asset, action='SHORT', size_fraction=0.05,
                           reason='t')
        approved = desk.apply_risk([short], portfolio, {},
                                   pd.Timestamp('2023-06-01'))

        assert approved == []
        notes = [n for n in desk.notes if n.category == 'risk']
        assert len(notes) == 1
        assert 'Blocked SHORT LG' in notes[0].message

    def test_short_stop_loss_fires_cover_at_the_boundary(self):
        # Default 2% stop: short entered at 100 stops at exactly 102.0.
        desk = ScriptedDesk(risk_manager=RiskManager())
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = PortfolioManager(initial_capital=100000.0)
        asset = stock('SH')
        portfolio.add_position(Position(
            asset=asset, quantity=-10, avg_entry_price=100.0,
            current_price=102.0, timestamp=pd.Timestamp('2023-05-01')))

        approved = desk.apply_risk([], portfolio, {},
                                   pd.Timestamp('2023-06-01'))

        assert len(approved) == 1
        cover = approved[0]
        assert cover.action == 'COVER'
        assert cover.asset == asset
        assert cover.size_fraction == 1.0
        notes = [n for n in desk.notes if n.category == 'risk']
        assert len(notes) == 1
        assert notes[0].data['stop_price'] == pytest.approx(102.0)
        assert notes[0].data['current_price'] == pytest.approx(102.0)

    def test_short_below_stop_does_not_cover(self):
        desk = ScriptedDesk(risk_manager=RiskManager())
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=stock('SH'), quantity=-10, avg_entry_price=100.0,
            current_price=101.99, timestamp=pd.Timestamp('2023-05-01')))
        assert desk.apply_risk([], portfolio, {},
                               pd.Timestamp('2023-06-01')) == []

    def test_daily_loss_circuit_blocks_shorts_but_covers_pass(self):
        desk = ScriptedDesk(risk_manager=RiskManager())  # 5% daily limit
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.cash = 94000.0  # -6% day
        portfolio.portfolio_history.append({
            'timestamp': pd.Timestamp('2023-05-31'),
            'portfolio_value': 100000.0,
        })

        short = DeskIntent(asset=stock('NEW'), action='SHORT',
                           size_fraction=0.05, reason='t')
        cover = DeskIntent(asset=stock('OLD'), action='COVER',
                           size_fraction=1.0, reason='t')
        approved = desk.apply_risk([short, cover], portfolio, {},
                                   pd.Timestamp('2023-06-01'))
        assert approved == [cover]

    def test_oversized_short_blocked_by_position_size_check(self):
        desk = ScriptedDesk(risk_manager=RiskManager())  # 10% limit
        desk.set_clock(pd.Timestamp('2023-06-01'))
        portfolio = PortfolioManager(initial_capital=100000.0)
        short = DeskIntent(asset=stock('BIG'), action='SHORT',
                           size_fraction=0.5, reason='t')
        approved = desk.apply_risk([short], portfolio, {},
                                   pd.Timestamp('2023-06-01'))
        assert approved == []
        notes = [n for n in desk.notes if n.category == 'risk']
        assert len(notes) == 1
        assert 'Blocked SHORT BIG' in notes[0].message
        assert notes[0].data['trade_value'] == pytest.approx(50000.0)


class TestMixedBookPortfolioValue:
    def test_portfolio_value_with_long_and_short_positions(self):
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.cash = 60000.0
        portfolio.add_position(Position(
            asset=stock('LG'), quantity=100, avg_entry_price=95.0,
            current_price=110.0, timestamp=pd.Timestamp('2023-01-02')))
        portfolio.add_position(Position(
            asset=stock('SH'), quantity=-200, avg_entry_price=50.0,
            current_price=45.0, timestamp=pd.Timestamp('2023-01-02')))

        # 60000 + 100*110 + (-200)*45 = 60000 + 11000 - 9000 = 62000.
        assert portfolio.get_portfolio_value() == pytest.approx(62000.0)
        # Unrealized: long +15/share, short +5/share on 200.
        assert portfolio.get_portfolio_pnl() == pytest.approx(
            100 * 15.0 + (-200) * (45.0 - 50.0))

        short_pos = portfolio.get_position(stock('SH'))
        assert short_pos.pnl() == pytest.approx(1000.0)
        assert short_pos.pnl_pct() == pytest.approx(10.0)  # price fell 10%


class TestStrategyModeUntouched:
    def test_strategy_short_signal_is_never_queued_or_filled(
            self, patch_market_data):
        df = linear_frame()
        patch_market_data({'TST': df})
        dates = df.index

        class ShortingStrategy(Strategy):
            def __init__(self):
                super().__init__('Shorting Strategy')

            def generate_signals(self, data, asset):
                # Tries to short every single day.
                return 'SHORT'

        engine = BacktestEngine(ShortingStrategy(),
                                initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TST'], '2023-01-01', '2023-12-31',
                            position_size=0.1, benchmark_symbol=None)

        # No SHORT path is reachable in strategy mode: nothing queued,
        # nothing filled, cash untouched.
        assert engine.trades_log == []
        assert engine.pending_intents == {}
        assert engine.portfolio.positions == {}
        assert engine.portfolio.cash == pytest.approx(100000.0)
        assert report['pending_signals'] == []
