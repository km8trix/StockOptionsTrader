"""Tests for desks.orchestrator.FundOrchestrator (the fund seam).

Covers capital validation (sum <= 1.0, unique keys, non-empty), the netting
rules (algebraic exposure netting of opens incl. opposing desks offsetting,
drop-both only on open/close ambiguity, option leg pass-through, same-direction
close collapse), the one account-wide apply_risk gate, account-absolute sizing,
option-pricing routing, note aggregation, the
BacktestEngine fund-mode driver (exactly-one-of guard, end-to-end two-desk
run, additive 'orchestrator' report key), and single-desk byte-additivity.
Offline, deterministic, no network.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.base import Desk, DeskIntent
from desks.orchestrator import FundOrchestrator
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


def wide_risk() -> RiskManager:
    return RiskManager(max_position_size=1.0, max_daily_loss=0.99,
                       position_stop_loss=0.90)


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def call(symbol: str, strike: float) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.CALL,
                 strike_price=strike, expiration_date='2022-06-17')


def intent(asset: Asset, action: str, size_fraction: float = 0.1,
           quantity: Optional[int] = None) -> DeskIntent:
    return DeskIntent(asset=asset, action=action, size_fraction=size_fraction,
                      reason='test', quantity=quantity)


class ScriptedDesk(Desk):
    """A desk that emits a fixed list of intents (or per-date), ignoring
    market data — lets the netting/sizing logic be driven deterministically."""

    def __init__(self, key: str, capital_allocation: float,
                 intents: Optional[List[DeskIntent]] = None,
                 per_date: Optional[Dict] = None,
                 risk_manager: Optional[RiskManager] = None):
        super().__init__(key=key, name=f"{key} desk", description='',
                         accent='#888888',
                         capital_allocation=capital_allocation,
                         risk_manager=risk_manager or wide_risk())
        self._intents = intents or []
        self._per_date = {pd.Timestamp(d): v
                          for d, v in (per_date or {}).items()}

    def generate_intents(self, all_data, date, portfolio):
        if self._per_date:
            return list(self._per_date.get(pd.Timestamp(date), []))
        return list(self._intents)


class OptionScriptedDesk(ScriptedDesk):
    """ScriptedDesk that can also price options (stands in for Jane Street)."""

    def price_option(self, asset, underlying_frame, date, spot):
        return 1.23


def flat_frame(n: int = 8, start: str = '2022-01-03',
               price: float = 100.0) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n)
    close = np.full(n, price)
    return pd.DataFrame({'open': close, 'high': close * 1.001,
                         'low': close * 0.999, 'close': close,
                         'volume': np.full(n, 1e6)}, index=index)


DATE = pd.Timestamp('2022-01-10')


def step_once(orch: FundOrchestrator,
              portfolio: Optional[PortfolioManager] = None) -> List[DeskIntent]:
    portfolio = portfolio or PortfolioManager(100_000.0)
    orch.set_clock(DATE)
    return orch.step({}, DATE, portfolio)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------
class TestConstruction:
    def test_empty_desks_rejected(self):
        with pytest.raises(ValueError, match='at least one desk'):
            FundOrchestrator([])

    def test_duplicate_keys_rejected(self):
        with pytest.raises(ValueError, match='unique'):
            FundOrchestrator([ScriptedDesk('a', 0.4),
                              ScriptedDesk('a', 0.4)])

    def test_overallocation_rejected(self):
        with pytest.raises(ValueError, match='must be <= 1.0'):
            FundOrchestrator([ScriptedDesk('a', 0.6),
                              ScriptedDesk('b', 0.6)])

    def test_sum_exactly_one_is_allowed(self):
        orch = FundOrchestrator([ScriptedDesk('a', 0.5),
                                 ScriptedDesk('b', 0.5)])
        assert orch.active_capital == pytest.approx(1.0)
        assert orch.capital_allocation == 1.0

    def test_under_allocation_is_allowed_and_kept_raw(self):
        orch = FundOrchestrator([ScriptedDesk('a', 0.3),
                                 ScriptedDesk('b', 0.3)])
        # No up-normalization: active_capital reflects the raw 0.6.
        assert orch.active_capital == pytest.approx(0.6)


# ----------------------------------------------------------------------
# Netting
# ----------------------------------------------------------------------
class TestNetting:
    def test_single_desk_intent_rescaled_to_account_absolute(self):
        # cap 0.5 * size_fraction 0.2 -> account-absolute 0.10.
        orch = FundOrchestrator([ScriptedDesk(
            'a', 0.5, intents=[intent(stock('AAA'), 'BUY', 0.2)])])
        approved = step_once(orch)
        (i,) = [x for x in approved if x.asset == stock('AAA')]
        assert i.action == 'BUY'
        assert i.size_fraction == pytest.approx(0.10)

    def test_same_direction_opens_combine(self):
        # 0.5*0.1 + 0.5*0.1 = 0.10 combined into ONE intent.
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'BUY', 0.1)]),
            ScriptedDesk('b', 0.5, intents=[intent(stock('AAA'), 'BUY', 0.1)]),
        ])
        approved = step_once(orch)
        aaa = [x for x in approved if x.asset == stock('AAA')]
        assert len(aaa) == 1
        assert aaa[0].action == 'BUY'
        assert aaa[0].size_fraction == pytest.approx(0.10)

    def test_opposite_opens_net_to_flat_when_equal(self):
        # Q1: opposing desks BOTH trade; equal sizes offset -> the fund stays
        # flat (not a conflict-drop). An 'allocation' note records the offset.
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'BUY', 0.1)]),
            ScriptedDesk('b', 0.5,
                         intents=[intent(stock('AAA'), 'SHORT', 0.1)]),
        ])
        approved = step_once(orch)
        assert [x for x in approved if x.asset == stock('AAA')] == []
        assert orch.conflicts_resolved == 0  # netted, NOT a conflict
        assert any('netted opposing' in n.message.lower() for n in orch.notes)

    def test_opposite_opens_net_to_long_residual(self):
        # BUY 0.5*0.16=0.08 vs SHORT 0.5*0.06=0.03 -> net BUY 0.05.
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5,
                         intents=[intent(stock('AAA'), 'BUY', 0.16)]),
            ScriptedDesk('b', 0.5,
                         intents=[intent(stock('AAA'), 'SHORT', 0.06)]),
        ], risk_manager=wide_risk())
        approved = step_once(orch)
        aaa = [x for x in approved if x.asset == stock('AAA')]
        assert len(aaa) == 1
        assert aaa[0].action == 'BUY'
        assert aaa[0].size_fraction == pytest.approx(0.05)
        assert orch.conflicts_resolved == 0

    def test_opposite_opens_net_to_short_residual(self):
        # BUY 0.5*0.06=0.03 vs SHORT 0.5*0.16=0.08 -> net SHORT 0.05.
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5,
                         intents=[intent(stock('AAA'), 'BUY', 0.06)]),
            ScriptedDesk('b', 0.5,
                         intents=[intent(stock('AAA'), 'SHORT', 0.16)]),
        ], risk_manager=wide_risk())
        approved = step_once(orch)
        aaa = [x for x in approved if x.asset == stock('AAA')]
        assert len(aaa) == 1
        assert aaa[0].action == 'SHORT'
        assert aaa[0].size_fraction == pytest.approx(0.05)
        assert orch.conflicts_resolved == 0

    def test_open_and_close_on_same_asset_is_a_conflict(self):
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'BUY')]),
            ScriptedDesk('b', 0.5, intents=[intent(stock('AAA'), 'SELL')]),
        ])
        approved = step_once(orch)
        assert [x for x in approved if x.asset == stock('AAA')] == []
        assert orch.conflicts_resolved == 1

    def test_disagreeing_closes_are_a_conflict(self):
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'SELL')]),
            ScriptedDesk('b', 0.5, intents=[intent(stock('AAA'), 'COVER')]),
        ])
        approved = step_once(orch)
        assert [x for x in approved if x.asset == stock('AAA')] == []
        assert orch.conflicts_resolved == 1

    def test_same_direction_closes_collapse_to_one_full_close(self):
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'SELL')]),
            ScriptedDesk('b', 0.5, intents=[intent(stock('AAA'), 'SELL')]),
        ])
        approved = step_once(orch)
        aaa = [x for x in approved if x.asset == stock('AAA')]
        assert len(aaa) == 1
        assert aaa[0].action == 'SELL' and aaa[0].size_fraction == 1.0

    def test_different_symbols_do_not_net(self):
        orch = FundOrchestrator([
            ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'BUY')]),
            ScriptedDesk('b', 0.5, intents=[intent(stock('BBB'), 'BUY')]),
        ])
        approved = step_once(orch)
        symbols = {x.asset.symbol for x in approved}
        assert {'AAA', 'BBB'} <= symbols
        assert orch.conflicts_resolved == 0

    def test_option_legs_bypass_netting(self):
        # Two option legs that LOOK like a stock conflict never net: both
        # pass through untouched (dropping one leg => a naked wing).
        leg_buy = intent(call('AAA', 105.0), 'BUY', 0.05, quantity=2)
        leg_short = intent(call('AAA', 115.0), 'SHORT', 0.05, quantity=2)
        orch = FundOrchestrator([
            OptionScriptedDesk('js', 1.0, intents=[leg_buy, leg_short])])
        approved = step_once(orch)
        opts = [x for x in approved
                if x.asset.asset_type in (AssetType.CALL, AssetType.PUT)]
        assert len(opts) == 2
        assert {o.action for o in opts} == {'BUY', 'SHORT'}
        assert all(o.quantity == 2 for o in opts)  # contracts preserved
        assert orch.conflicts_resolved == 0


# ----------------------------------------------------------------------
# Unified account-wide apply_risk
# ----------------------------------------------------------------------
class TestUnifiedRisk:
    def test_combined_open_blocked_by_account_position_size(self):
        # Two desks each 0.5*0.1 = 0.05; combined 0.10 > the 8% account cap.
        orch = FundOrchestrator(
            [ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'BUY', 0.1)]),
             ScriptedDesk('b', 0.5, intents=[intent(stock('AAA'), 'BUY', 0.1)])],
            risk_manager=RiskManager(max_position_size=0.08))
        approved = step_once(orch)
        assert [x for x in approved if x.asset == stock('AAA')] == []

    def test_account_stop_loss_closes_unified_position(self):
        # A losing long on the unified book is stopped once, account-wide.
        orch = FundOrchestrator(
            [ScriptedDesk('a', 1.0, intents=[])],
            risk_manager=RiskManager(position_stop_loss=0.05))
        portfolio = PortfolioManager(100_000.0)
        portfolio.add_position(Position(
            asset=stock('AAA'), quantity=100, avg_entry_price=100.0,
            current_price=90.0, timestamp=datetime(2022, 1, 3)))
        approved = step_once(orch, portfolio)
        stops = [x for x in approved
                 if x.asset == stock('AAA') and x.action == 'SELL']
        assert len(stops) == 1 and stops[0].size_fraction == 1.0


# ----------------------------------------------------------------------
# Option pricing routing + read surface
# ----------------------------------------------------------------------
class TestSurface:
    def test_price_option_routes_to_option_desk(self):
        orch = FundOrchestrator([ScriptedDesk('a', 0.5),
                                 OptionScriptedDesk('js', 0.5)])
        assert orch.price_option(call('AAA', 100.0), pd.DataFrame(),
                                 DATE, 100.0) == 1.23

    def test_price_option_none_when_no_option_desk(self):
        orch = FundOrchestrator([ScriptedDesk('a', 1.0)])
        assert orch.price_option(call('AAA', 100.0), pd.DataFrame(),
                                 DATE, 100.0) is None

    def test_notes_aggregate_subdesks_and_conflicts(self):
        # Open+close on the same name is a genuine (dropped) conflict — used
        # here to exercise note aggregation (sub-desk note + orchestrator note).
        a = ScriptedDesk('a', 0.5, intents=[intent(stock('AAA'), 'BUY')])
        b = ScriptedDesk('b', 0.5, intents=[intent(stock('AAA'), 'SELL')])
        a.set_clock(DATE)
        a.note('signal', 'desk a says hi')
        orch = FundOrchestrator([a, b])
        step_once(orch)  # produces a conflict note on the orchestrator
        messages = [n.message for n in orch.notes]
        assert any('desk a says hi' in m for m in messages)
        assert any('conflict' in m.lower() for m in messages)


# ----------------------------------------------------------------------
# BacktestEngine fund mode
# ----------------------------------------------------------------------
class TestEngineGuard:
    def test_exactly_one_driver_required(self):
        orch = FundOrchestrator([ScriptedDesk('a', 1.0)])
        with pytest.raises(ValueError, match='exactly one'):
            BacktestEngine(desk=ScriptedDesk('x', 1.0), orchestrator=orch)
        with pytest.raises(ValueError, match='exactly one'):
            BacktestEngine()  # none

    def test_orchestrator_alone_is_accepted(self):
        orch = FundOrchestrator([ScriptedDesk('a', 1.0)])
        engine = BacktestEngine(orchestrator=orch, initial_capital=100_000.0)
        assert engine.orchestrator is orch
        assert engine._desk_mode and engine._driver is orch


class TestEngineEndToEnd:
    def _universe(self):
        return {'AAA': flat_frame(), 'BBB': flat_frame()}

    def test_two_desk_run_opens_both_and_reports_orchestrator(
            self, monkeypatch):
        universe = self._universe()
        dates = universe['AAA'].index

        def fake_fetch(self, symbol, start, end):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        # Each desk opens one name early; signal day -> fills next open.
        sig = dates[1]
        a = ScriptedDesk('asim', 0.5,
                         per_date={sig: [intent(stock('AAA'), 'BUY', 0.05)]})
        b = ScriptedDesk('bsim', 0.5,
                         per_date={sig: [intent(stock('BBB'), 'BUY', 0.05)]})
        orch = FundOrchestrator([a, b])
        engine = BacktestEngine(orchestrator=orch, initial_capital=100_000.0)
        report = engine.run(['AAA', 'BBB'], '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)

        assert 'error' not in report
        held = {asset.symbol for asset in engine.portfolio.positions}
        assert {'AAA', 'BBB'} <= held
        assert report['orchestrator']['active_capital'] == pytest.approx(1.0)
        assert {d['key'] for d in report['orchestrator']['desks']} == {
            'asim', 'bsim'}
        assert report['desk']['key'] == 'fund'

    def test_single_desk_report_has_no_orchestrator_key(self, monkeypatch):
        universe = self._universe()

        def fake_fetch(self, symbol, start, end):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = ScriptedDesk('solo', 1.0, intents=[])
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        report = engine.run(['AAA'], '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)
        assert 'orchestrator' not in report
        assert report['desk']['key'] == 'solo'

    def test_netted_short_opens_then_covers_in_fund(self, monkeypatch):
        # Regression for the fund-mode SHORT/COVER dead-code bug: a net-SHORT
        # residual must OPEN a negative position, and a fund COVER must CLOSE
        # it — both filled through _fill_intent under _desk_mode (previously
        # gated on self.desk is not None, so they silently no-op'd in a fund).
        universe = {'AAA': flat_frame(n=8)}
        dates = universe['AAA'].index

        def fake_fetch(self, symbol, start, end):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        sig_open, sig_cover = dates[1], dates[4]
        # net = 0.5*0.06 (BUY) - 0.5*0.16 (SHORT) = -0.05 -> SHORT 0.05.
        a = ScriptedDesk('asim', 0.5, per_date={
            sig_open: [intent(stock('AAA'), 'BUY', 0.06)]})
        b = ScriptedDesk('bsim', 0.5, per_date={
            sig_open: [intent(stock('AAA'), 'SHORT', 0.16)],
            sig_cover: [intent(stock('AAA'), 'COVER')]})
        orch = FundOrchestrator([a, b], risk_manager=wide_risk())
        engine = BacktestEngine(orchestrator=orch, initial_capital=100_000.0)
        report = engine.run(['AAA'], '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)

        assert 'error' not in report
        actions = [t['action'] for t in report['trades']]
        assert 'SHORT' in actions   # the netted short actually opened
        assert 'COVER' in actions   # and the fund cover actually closed it
        pos = engine.portfolio.get_position(stock('AAA'))
        assert pos is None or pos.quantity == 0  # net flat after the cover


class TestEngineFundWithOptions:
    """The riskiest integration: a REAL options desk (Jane Street) inside a
    fund. Exercises option fills/marks/settlement routing through
    orchestrator.price_option, plus option-leg netting bypass, end to end."""

    def test_options_desk_in_fund_fills(self, monkeypatch):
        from desks.options_pricing import SyntheticEarningsCalendar
        from tests.test_janestreet_desk import make_desk, vol_frame, SHIFT

        universe = {'AAA': vol_frame(n=160, shift_at=SHIFT),
                    'BBB': vol_frame(n=160, shift_at=SHIFT)}
        dates = universe['AAA'].index

        def fake_fetch(self, symbol, start, end):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        js = make_desk(capital_allocation=1.0,
                       earnings_calendar=SyntheticEarningsCalendar(
                           {'BBB': [dates[40].date()]}))
        orch = FundOrchestrator([js], risk_manager=wide_risk())
        engine = BacktestEngine(orchestrator=orch, initial_capital=100_000.0)
        report = engine.run(sorted(universe), '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)

        assert 'error' not in report
        # Options actually filled through the fund's price_option routing.
        option_trades = [t for t in report['trades']
                         if 'call' in t['instrument']
                         or 'put' in t['instrument']]
        assert option_trades
        # And the desk tracked the structures it opened inside the fund.
        assert js.tracker.to_report()
        assert report['desk']['key'] == 'fund'
