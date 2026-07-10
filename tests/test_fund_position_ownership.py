"""Desk-scoped position ownership (fund mode).

Pins the contract that lets one fund host several cross-sectional desks
whose universes overlap THROUGH TIME (band membership migrates monthly):

  * Position.owners records which desk(s)' netted opening intent created a
    position (engine stamps it from DeskIntent.desk_keys, set by the
    orchestrator's netting).
  * A cross-sectional desk's book logic (reconcile held-check, orphan
    sweep, turnover retention) treats a position owned by ANOTHER desk as
    invisible — it must never close it.
  * An UNTAGGED position (owners is None — always the case outside fund
    mode) is treated as owned: single-desk behavior is byte-identical.

Frames are hand-built; no network, no randomness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.base import Desk, DeskIntent
from desks.cross_sectional import (CLOSE_RETRY_DAYS,
                                   CrossSectionalLongShortDesk,
                                   RECONCILE_GRACE_DAYS)
from desks.orchestrator import FundOrchestrator, _Tagged
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def held(asset: Asset, quantity: int = 100, owners=None) -> Position:
    return Position(asset=asset, quantity=quantity, avg_entry_price=10.0,
                    current_price=10.0, timestamp=pd.Timestamp('2023-01-02'),
                    owners=owners)


class FixedBookDesk(CrossSectionalLongShortDesk):
    """Minimal concrete cross-sectional desk (no committee, no alpha)."""

    def __init__(self, key: str = 'alpha', **kwargs):
        super().__init__(key=key, name=key, description='test-only',
                         accent='#123456', note_label=key, reason_prefix=key,
                         committee=[], model_label='fixed', **kwargs)

    def _alpha_scores(self, all_data, date):
        return None


class ScriptedDesk(Desk):
    """Desk emitting scripted intents keyed by simulation date."""

    def __init__(self, key='scripted', capital_allocation=1.0, script=None):
        super().__init__(key=key, name=key, description='test-only',
                         accent='#123456',
                         capital_allocation=capital_allocation,
                         risk_manager=RiskManager(
                             max_position_size=1.0, max_daily_loss=0.99,
                             position_stop_loss=0.90))
        self.script = {pd.Timestamp(d): intents
                       for d, intents in (script or {}).items()}

    def generate_intents(self, all_data, date, portfolio):
        return list(self.script.get(pd.Timestamp(date), []))


class TestOrphanSweepScoping:
    def _swept_symbols(self, owners):
        desk = FixedBookDesk(key='a')
        desk._traded_symbols['XYZ'] = True
        portfolio = PortfolioManager(100_000.0)
        portfolio.add_position(held(stock('XYZ'), owners=owners))
        intents = desk._reconcile_book(portfolio, desired={})
        return [i.asset.symbol for i in intents]

    def test_sweep_skips_position_owned_by_another_desk(self):
        assert self._swept_symbols(owners=('b',)) == []

    def test_sweep_closes_own_orphan(self):
        assert self._swept_symbols(owners=('a',)) == ['XYZ']

    def test_sweep_closes_untagged_orphan_legacy(self):
        # owners is None == pre-ownership behavior (single-desk mode).
        assert self._swept_symbols(owners=None) == ['XYZ']

    def test_sweep_closes_co_owned_orphan(self):
        assert self._swept_symbols(owners=('a', 'b')) == ['XYZ']


class TestReconcileHeldScoping:
    def test_foreign_position_is_not_this_desks_fill(self):
        """A tracked leg whose symbol holds ANOTHER desk's position (this
        desk's entry was netted away or blocked while the other's filled)
        is treated as never-filled: no close is ever emitted for it and the
        bookkeeping drops after the reconcile grace."""
        desk = FixedBookDesk(key='a')
        asset = stock('XYZ')
        desk._track_entry(asset, direction='long')
        portfolio = PortfolioManager(100_000.0)
        portfolio.add_position(held(asset, owners=('b',)))

        # Inside the grace: leg kept quietly, and crucially NO close intent.
        assert desk._reconcile_book(portfolio, desired={}) == []
        assert asset in desk._book_positions

        # Past the grace: bookkeeping dropped, still no close intent.
        desk._day_index += RECONCILE_GRACE_DAYS
        assert desk._reconcile_book(portfolio, desired={}) == []
        assert asset not in desk._book_positions

    def test_own_position_still_closed_when_it_leaves_the_book(self):
        desk = FixedBookDesk(key='a')
        asset = stock('XYZ')
        desk._track_entry(asset, direction='long')
        portfolio = PortfolioManager(100_000.0)
        portfolio.add_position(held(asset, owners=('a',)))

        intents = desk._reconcile_book(portfolio, desired={})
        assert [(i.asset.symbol, i.action) for i in intents] == [('XYZ', 'SELL')]


class TestClosingRetry:
    """FUND MODE ONLY: a dropped in-flight close is re-emitted, because
    ownership scoping means no other desk can ever close the position for
    us. Single-desk (untagged) keeps the emit-once behavior."""

    def _latched_desk(self, owners):
        desk = FixedBookDesk(key='a')
        asset = stock('XYZ')
        desk._track_entry(asset, direction='long')
        portfolio = PortfolioManager(100_000.0)
        portfolio.add_position(held(asset, owners=owners))
        first = desk._reconcile_book(portfolio, desired={})
        assert [i.action for i in first] == ['SELL']  # close emitted, latched
        return desk, portfolio

    def test_fund_close_retries_after_pending_lifetime(self):
        desk, portfolio = self._latched_desk(owners=('a',))
        # Still in flight: no duplicate close.
        assert desk._reconcile_book(portfolio, desired={}) == []
        desk._day_index += CLOSE_RETRY_DAYS
        retried = desk._reconcile_book(portfolio, desired={})
        assert [i.action for i in retried] == ['SELL']
        # Cadence resets: no immediate third emission.
        assert desk._reconcile_book(portfolio, desired={}) == []

    def test_single_desk_close_never_retries(self):
        desk, portfolio = self._latched_desk(owners=None)
        desk._day_index += 3 * CLOSE_RETRY_DAYS
        assert desk._reconcile_book(portfolio, desired={}) == []


class TestNettingStampsOwnership:
    def _orch(self):
        return FundOrchestrator([ScriptedDesk('a', 0.5),
                                 ScriptedDesk('b', 0.5)])

    def _tag(self, orch, key, action, fraction):
        desk = next(d for d in orch.desks if d.key == key)
        intent = DeskIntent(asset=stock('AAA'), action=action,
                            size_fraction=fraction, reason='t')
        return _Tagged(intent, desk, desk.capital_allocation * fraction)

    def test_same_direction_opens_owned_by_both(self):
        orch = self._orch()
        netted = orch._net_stock_asset(stock('AAA'), [
            self._tag(orch, 'a', 'BUY', 0.1),
            self._tag(orch, 'b', 'BUY', 0.1)])
        assert netted[0].desk_keys == ('a', 'b')

    def test_opposing_residual_owned_by_winning_side_only(self):
        orch = self._orch()
        netted = orch._net_stock_asset(stock('AAA'), [
            self._tag(orch, 'a', 'BUY', 0.2),
            self._tag(orch, 'b', 'SHORT', 0.1)])
        assert netted[0].action == 'BUY'
        assert netted[0].desk_keys == ('a',)

    def test_merged_close_carries_no_ownership(self):
        # Closes never create positions; nothing to own.
        orch = self._orch()
        netted = orch._net_stock_asset(stock('AAA'), [
            self._tag(orch, 'a', 'SELL', 1.0),
            self._tag(orch, 'b', 'SELL', 1.0)])
        assert netted[0].desk_keys is None

    def test_option_passthrough_owned_by_its_desk(self):
        orch = self._orch()
        option = Asset(symbol='AAA', asset_type=AssetType.CALL,
                       strike_price=100.0, expiration_date='2023-06-16')
        item = _Tagged(DeskIntent(asset=option, action='BUY',
                                  size_fraction=0.1, reason='t'),
                       orch.desks[0], 0.05)
        assert orch._account_scaled(item).desk_keys == ('a',)


def flat_frame(n: int = 8) -> pd.DataFrame:
    index = pd.bdate_range('2023-01-02', periods=n)
    return pd.DataFrame({
        'open': np.full(n, 100.0), 'high': np.full(n, 100.1),
        'low': np.full(n, 99.9), 'close': np.full(n, 100.0),
        'volume': np.full(n, 500_000.0)}, index=index)


@pytest.fixture
def patch_market_data(monkeypatch):
    def _patch(frames_by_symbol):
        def fake_fetch(self, symbol, start_date, end_date):
            return frames_by_symbol.get(symbol, pd.DataFrame())
        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
    return _patch


class TestEngineStampsOwners:
    def test_fund_fill_carries_owning_desk(self, patch_market_data):
        df = flat_frame()
        patch_market_data({'TST': df})
        buy = DeskIntent(asset=stock('TST'), action='BUY',
                         size_fraction=0.1, reason='t')
        orch = FundOrchestrator([ScriptedDesk('a', 0.5,
                                              script={df.index[2]: [buy]}),
                                 ScriptedDesk('b', 0.5)])
        engine = BacktestEngine(orchestrator=orch, initial_capital=100_000.0)
        engine.run(['TST'], '2023-01-01', '2023-01-31', benchmark_symbol=None)
        position = engine.portfolio.get_position(stock('TST'))
        assert position is not None and position.quantity > 0
        assert position.owners == ('a',)

    def test_single_desk_fill_stays_untagged(self, patch_market_data):
        df = flat_frame()
        patch_market_data({'TST': df})
        buy = DeskIntent(asset=stock('TST'), action='BUY',
                         size_fraction=0.1, reason='t')
        desk = ScriptedDesk('a', 1.0, script={df.index[2]: [buy]})
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        engine.run(['TST'], '2023-01-01', '2023-01-31', benchmark_symbol=None)
        position = engine.portfolio.get_position(stock('TST'))
        assert position is not None and position.quantity > 0
        assert position.owners is None  # byte-identical single-desk path
