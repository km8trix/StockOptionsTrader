"""Tests for the BacktestEngine's Phase 8 options execution path.

A scripted stub desk pins every number: fills are hand-computed against
the options cost model (model price x (1 +/- spread, min $0.05 haircut)
x 100 x contracts +/- $0.65/contract commission), the x100 multiplier is
asserted on a mixed stock+option book, short round trips are sign-exact,
and expiry settlement is checked ITM and OTM. Stock-path regression
lives in the untouched pre-Phase-8 suites (engine/portfolio/models tests
run byte-identical semantics). Offline, deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.base import Desk, DeskIntent
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

START = '2023-01-02'
N_DAYS = 12


def flat_frame(price=100.0, n=N_DAYS):
    index = pd.bdate_range(START, periods=n)
    return pd.DataFrame({'open': np.full(n, price),
                         'high': np.full(n, price),
                         'low': np.full(n, price),
                         'close': np.full(n, price),
                         'volume': np.full(n, 1e6)}, index=index)


DATES = pd.bdate_range(START, periods=N_DAYS)


def call(strike=100.0, expiry='2023-03-17'):
    return Asset('UND', AssetType.CALL, strike_price=strike,
                 expiration_date=expiry)


def put(strike=100.0, expiry='2023-03-17'):
    return Asset('UND', AssetType.PUT, strike_price=strike,
                 expiration_date=expiry)


class ScriptedDesk(Desk):
    """Scripted intents by date + scripted option model prices.

    prices maps str(asset) -> price, or (str(asset), 'YYYY-MM-DD') ->
    price for date-specific marks (the dated key wins).
    """

    def __init__(self, script=None, prices=None):
        super().__init__(
            key='stub', name='Stub Desk', description='', accent='#ffffff',
            risk_manager=RiskManager(max_position_size=1.0,
                                     max_daily_loss=0.99,
                                     position_stop_loss=0.99))
        self.script = {pd.Timestamp(d): intents
                       for d, intents in (script or {}).items()}
        self.prices = prices or {}

    def generate_intents(self, all_data, date, portfolio):
        return list(self.script.get(pd.Timestamp(date), []))

    def price_option(self, asset, underlying_frame, date, spot):
        dated = (str(asset), pd.Timestamp(date).strftime('%Y-%m-%d'))
        if dated in self.prices:
            return self.prices[dated]
        return self.prices.get(str(asset))


@pytest.fixture
def engine_factory(monkeypatch):
    def fake_fetch(self, symbol, start_date, end_date):
        return flat_frame() if symbol == 'UND' else pd.DataFrame()

    monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

    def build(script, prices):
        desk = ScriptedDesk(script=script, prices=prices)
        return BacktestEngine(desk=desk, initial_capital=100_000.0)

    def run(script, prices):
        engine = build(script, prices)
        report = engine.run(['UND'], START, '2023-01-31',
                            benchmark_symbol=None)
        return engine, report

    return run


class TestOptionFillArithmetic:
    def test_buy_fill_hand_computed(self, engine_factory):
        # BUY 3 calls, model $5.00: haircut max(0.10, 0.05) = 0.10 ->
        # traded 5.10; cost = 3 * 5.10 * 100 + 3 * 0.65 = 1531.95.
        intent = DeskIntent(asset=call(), action='BUY', size_fraction=0.05,
                            reason='t', quantity=3)
        engine, report = engine_factory({DATES[0]: [intent]},
                                        {str(call()): 5.00})
        fills = [t for t in engine.trades_log if t['action'] == 'BUY']
        assert len(fills) == 1
        fill = fills[0]
        assert fill['date'] == DATES[1]  # next bar's open
        assert fill['quantity'] == 3
        assert fill['price'] == pytest.approx(5.10)
        assert fill['commission'] == pytest.approx(1.95)
        assert fill['cost'] == pytest.approx(1531.95)
        assert fill['instrument'] == str(call())
        position = engine.portfolio.get_position(call())
        assert position.quantity == 3
        assert position.avg_entry_price == pytest.approx(5.10)

    def test_minimum_absolute_haircut_applies_to_cheap_options(
            self, engine_factory):
        # Model $1.00: 2% spread is $0.02 < the $0.05 floor -> buy 1.05.
        intent = DeskIntent(asset=call(), action='BUY', size_fraction=0.05,
                            reason='t', quantity=1)
        engine, _ = engine_factory({DATES[0]: [intent]},
                                   {str(call()): 1.00})
        fill = [t for t in engine.trades_log if t['action'] == 'BUY'][0]
        assert fill['price'] == pytest.approx(1.05)
        assert fill['cost'] == pytest.approx(1.05 * 100 + 0.65)

    def test_short_option_round_trip_sign_exact(self, engine_factory):
        # SHORT 2 puts at model 5.00 -> sell-to-open at 4.90:
        # proceeds = 2*4.90*100 - 2*0.65 = 978.70 credited.
        # COVER at model 3.00 -> buy-to-close at 3.06:
        # cost = 2*3.06*100 + 2*0.65 = 613.30 debited.
        asset = put()
        script = {
            DATES[0]: [DeskIntent(asset=asset, action='SHORT',
                                  size_fraction=0.05, reason='t',
                                  quantity=2)],
            DATES[2]: [DeskIntent(asset=asset, action='COVER',
                                  size_fraction=1.0, reason='t')],
        }
        prices = {(str(asset), DATES[1].strftime('%Y-%m-%d')): 5.00,
                  (str(asset), DATES[2].strftime('%Y-%m-%d')): 5.00,
                  (str(asset), DATES[3].strftime('%Y-%m-%d')): 3.00}
        engine, report = engine_factory(script, prices)

        short_fill = [t for t in engine.trades_log
                      if t['action'] == 'SHORT'][0]
        assert short_fill['price'] == pytest.approx(4.90)
        assert short_fill['proceeds'] == pytest.approx(978.70)

        cover_fill = [t for t in engine.trades_log
                      if t['action'] == 'COVER'][0]
        assert cover_fill['price'] == pytest.approx(3.06)
        assert cover_fill['cost'] == pytest.approx(613.30)

        # Trade P&L: quantity -2, x100 multiplier, sign-correct credit.
        trade = engine.portfolio.closed_trades[0]
        assert trade.quantity == -2
        assert trade.pnl == pytest.approx(-2 * (3.06 - 4.90) * 100)  # +368
        assert trade.pnl == pytest.approx(368.0)
        # Cash round trip = +978.70 - 613.30 = pnl - total commissions.
        assert engine.portfolio.cash == pytest.approx(100_000 + 365.40)
        assert not engine.portfolio.positions
        closed = report['closed_trades'][0]
        assert closed['instrument'] == str(asset)
        assert closed['pnl'] == pytest.approx(368.0)

    def test_quantity_overrides_value_sizing(self, engine_factory):
        # size_fraction alone would buy int(5%*100k / 510) = 9 contracts;
        # quantity=2 must win.
        intent = DeskIntent(asset=call(), action='BUY', size_fraction=0.05,
                            reason='t', quantity=2)
        engine, _ = engine_factory({DATES[0]: [intent]},
                                   {str(call()): 5.00})
        assert engine.portfolio.get_position(call()).quantity == 2


class TestMultiplierInPortfolio:
    def test_mixed_stock_and_option_book_is_exact(self, engine_factory):
        stock = Asset('UND', AssetType.STOCK)
        script = {DATES[0]: [
            DeskIntent(asset=stock, action='BUY', size_fraction=0.05,
                       reason='t', quantity=10),
            DeskIntent(asset=call(), action='BUY', size_fraction=0.05,
                       reason='t', quantity=2),
        ]}
        # Mark the option at 6.00 every day after the 5.00 fill day.
        prices = {str(call()): 6.00,
                  (str(call()), DATES[1].strftime('%Y-%m-%d')): 5.00}
        engine, _ = engine_factory(script, prices)

        # Stock fill: 100 * (1 + 5bps) = 100.05; cost = 10*100.05*1.001.
        stock_cost = 10 * 100.05 * 1.001
        option_cost = 2 * 5.10 * 100 + 2 * 0.65
        expected_cash = 100_000 - stock_cost - option_cost
        assert engine.portfolio.cash == pytest.approx(expected_cash)

        # MTM: stock at close 100, option at the 6.00 model mark x100.
        assert engine.portfolio.get_position(call()).current_price == 6.00
        expected_value = expected_cash + 10 * 100.0 + 2 * 6.00 * 100
        assert engine.portfolio.get_portfolio_value() == pytest.approx(
            expected_value)

        # Position P&L carries the multiplier: 2 * (6.00 - 5.10) * 100.
        assert engine.portfolio.get_position(call()).pnl() == pytest.approx(
            180.0)
        assert engine.portfolio.get_portfolio_pnl() == pytest.approx(
            180.0 + 10 * (100.0 - 100.05))


class TestExpirySettlement:
    EXPIRY = '2023-01-06'  # Friday of week 1; settles Monday 01-09

    def test_short_call_itm_settles_at_intrinsic(self, engine_factory):
        asset = call(strike=95.0, expiry=self.EXPIRY)
        script = {DATES[0]: [DeskIntent(asset=asset, action='SHORT',
                                        size_fraction=0.05, reason='t',
                                        quantity=1)]}
        engine, _ = engine_factory(script, {str(asset): 5.50})

        settles = [t for t in engine.trades_log
                   if t.get('reason') == 'expiry settlement']
        assert len(settles) == 1
        settle = settles[0]
        # Underlying close on the last session <= expiry is 100 ->
        # intrinsic 5.00; the short PAYS 1 * 5.00 * 100, no spread, no
        # commission; settled the first session AFTER expiry.
        assert settle['action'] == 'COVER'
        assert settle['price'] == pytest.approx(5.00)
        assert settle['cost'] == pytest.approx(500.0)
        assert settle['date'] == DATES[5]  # Monday 2023-01-09
        assert 'commission' not in settle

        # Sold-to-open at 5.50*(1-0.02)=5.39: proceeds 539 - 0.65.
        trade = engine.portfolio.closed_trades[0]
        assert trade.pnl == pytest.approx(-1 * (5.00 - 5.39) * 100)  # +39
        assert engine.portfolio.cash == pytest.approx(
            100_000 + 539.0 - 0.65 - 500.0)
        assert not engine.portfolio.positions

    def test_long_put_otm_settles_worthless(self, engine_factory):
        asset = put(strike=95.0, expiry=self.EXPIRY)
        script = {DATES[0]: [DeskIntent(asset=asset, action='BUY',
                                        size_fraction=0.05, reason='t',
                                        quantity=1)]}
        engine, _ = engine_factory(script, {str(asset): 2.00})

        settle = [t for t in engine.trades_log
                  if t.get('reason') == 'expiry settlement'][0]
        # Spot 100 > strike 95: the put dies worthless.
        assert settle['action'] == 'SELL'
        assert settle['price'] == 0.0
        assert settle['proceeds'] == 0.0
        # Bought at 2.05 (min haircut $0.05): full debit is the loss.
        trade = engine.portfolio.closed_trades[0]
        assert trade.pnl == pytest.approx(1 * (0.0 - 2.05) * 100)
        assert engine.portfolio.cash == pytest.approx(
            100_000 - 205.0 - 0.65)


class TestOptionStopLossExemption:
    def test_apply_risk_never_stops_option_legs_but_stops_stock(self):
        desk = ScriptedDesk()
        desk.risk_manager = RiskManager(max_position_size=1.0,
                                        max_daily_loss=0.99,
                                        position_stop_loss=0.02)
        desk.set_clock('2023-02-01')
        portfolio = PortfolioManager(100_000.0)
        # Short option moving violently against us: NO per-leg stop —
        # structure-level exits own options risk (Phase 8).
        portfolio.add_position(Position(
            asset=put(), quantity=-2, avg_entry_price=2.00,
            current_price=8.00, timestamp=pd.Timestamp('2023-01-20')))
        # Stock long through its 2% stop: the stop must still fire.
        portfolio.add_position(Position(
            asset=Asset('UND', AssetType.STOCK), quantity=10,
            avg_entry_price=100.0, current_price=90.0,
            timestamp=pd.Timestamp('2023-01-20')))

        approved = desk.apply_risk([], portfolio, {}, '2023-02-01')
        assert [(i.action, i.asset.asset_type) for i in approved] == [
            ('SELL', AssetType.STOCK)]


class TestC13Instrument:
    def test_stock_trades_carry_bare_symbol_instrument(self, engine_factory):
        stock = Asset('UND', AssetType.STOCK)
        script = {
            DATES[0]: [DeskIntent(asset=stock, action='BUY',
                                  size_fraction=0.05, reason='t')],
            DATES[2]: [DeskIntent(asset=stock, action='SELL',
                                  size_fraction=1.0, reason='t')],
        }
        engine, report = engine_factory(script, {})
        for trade in report['trades']:
            assert trade['instrument'] == 'UND'
        for closed in report['closed_trades']:
            assert closed['instrument'] == 'UND'
