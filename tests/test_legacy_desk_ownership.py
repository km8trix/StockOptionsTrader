"""Desk-scoped position ownership — frozen legacy desk families.

Mirrors tests/test_fund_position_ownership.py for the legacy desks:
a position owner-tagged for ANOTHER desk (core.models.Position.owners)
is invisible to each desk's sweep/exit logic — Renaissance/Citadel
reconcile + orphan sweep, Foundation MACD/RSI exits, trend-follower
regime exits, Jane Street relative-value reconcile/exits. An UNTAGGED
position (owners is None — always the case outside fund mode) behaves
byte-identically to pre-ownership code.

Frames are hand-built; no network, no randomness.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.models import Asset, AssetType, Position
from desks.citadel import CitadelDesk
from desks.citadel import RECONCILE_GRACE_DAYS as CITADEL_GRACE
from desks.foundation import FoundationDesk
from desks.janestreet import JaneStreetDesk
from desks.janestreet import RECONCILE_GRACE_DAYS as JS_GRACE
from desks.renaissance import RECONCILE_GRACE_DAYS as RENTECH_GRACE
from desks.renaissance import RenaissanceDesk
from desks.trend_follower import TrendFollowerDesk
from portfolio.manager import PortfolioManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def held(asset: Asset, quantity: int = 100, owners=None) -> Position:
    return Position(asset=asset, quantity=quantity, avg_entry_price=10.0,
                    current_price=10.0, timestamp=pd.Timestamp('2023-01-02'),
                    owners=owners)


def book(portfolio_positions) -> PortfolioManager:
    portfolio = PortfolioManager(100_000.0)
    for position in portfolio_positions:
        portfolio.add_position(position)
    return portfolio


class TestRenaissanceScoping:
    def _swept(self, owners):
        desk = RenaissanceDesk()
        desk._traded_books['XYZ'] = 'stat_arb'
        portfolio = book([held(stock('XYZ'), owners=owners)])
        return [i.asset.symbol for i in desk._reconcile_books(portfolio)]

    def test_sweep_skips_position_owned_by_another_desk(self):
        assert self._swept(owners=('other',)) == []

    def test_sweep_closes_own_orphan(self):
        assert self._swept(owners=('renaissance',)) == ['XYZ']

    def test_sweep_closes_untagged_orphan_legacy(self):
        assert self._swept(owners=None) == ['XYZ']

    def test_foreign_fill_treated_as_never_filled(self):
        """A tracked leg whose position belongs to another desk is not
        this desk's fill: no close is emitted and the bookkeeping drops
        after the reconcile grace (then the sweep still skips it)."""
        desk = RenaissanceDesk()
        asset = stock('XYZ')
        desk._track_entry(asset, book='stat_arb', direction='long',
                          entry_day=0)
        portfolio = book([held(asset, owners=('other',))])

        assert desk._reconcile_books(portfolio) == []  # in grace: kept
        assert asset in desk._book_positions

        desk._day_index += RENTECH_GRACE
        assert desk._reconcile_books(portfolio) == []  # dropped, no close
        assert asset not in desk._book_positions


class TestCitadelScoping:
    def _swept(self, owners):
        desk = CitadelDesk()
        desk._traded_pods['XYZ'] = 'pod_a'
        portfolio = book([held(stock('XYZ'), owners=owners)])
        return [i.asset.symbol for _, i in desk._reconcile_pods(portfolio)]

    def test_sweep_skips_position_owned_by_another_desk(self):
        assert self._swept(owners=('other',)) == []

    def test_sweep_closes_own_orphan(self):
        assert self._swept(owners=('citadel',)) == ['XYZ']

    def test_sweep_closes_untagged_orphan_legacy(self):
        assert self._swept(owners=None) == ['XYZ']

    def _nav_warnings(self, owners, caplog):
        desk = CitadelDesk()
        portfolio = book([held(stock('XYZ'), owners=owners)])
        with caplog.at_level(logging.WARNING, logger='desks.citadel'):
            desk._update_pod_navs(portfolio)
        return [r for r in caplog.records if 'no owning pod' in r.message]

    def test_nav_attribution_skips_foreign_position_silently(self, caplog):
        assert self._nav_warnings(('other',), caplog) == []

    def test_nav_attribution_still_warns_on_untagged_orphan(self, caplog):
        assert len(self._nav_warnings(None, caplog)) == 1

    def test_nav_attribution_still_warns_on_own_untracked(self, caplog):
        assert len(self._nav_warnings(('citadel',), caplog)) == 1

    def test_foreign_fill_treated_as_never_filled(self):
        desk = CitadelDesk()
        asset = stock('XYZ')
        desk._pod_positions[asset] = {'pod': 'pod_a', 'direction': 'long',
                                      'entry_day': 0}
        desk._traded_pods['XYZ'] = 'pod_a'
        portfolio = book([held(asset, owners=('other',))])

        assert desk._reconcile_pods(portfolio) == []  # in grace: kept
        assert asset in desk._pod_positions

        desk._day_index += CITADEL_GRACE
        assert desk._reconcile_pods(portfolio) == []  # dropped, no close
        assert asset not in desk._pod_positions


def momentum_frame(n: int = 60, cross: str = 'down') -> pd.DataFrame:
    """Indicator-enriched frame with a MACD cross on the LAST bar; RSI
    and volume sit inside FoundationDesk's default entry band."""
    index = pd.bdate_range('2023-01-02', periods=n)
    close = np.full(n, 100.0)
    macd = np.full(n, 1.0 if cross == 'down' else -1.0)
    macd[-1] = -1.0 if cross == 'down' else 1.0
    return pd.DataFrame({
        'open': close, 'high': close * 1.01, 'low': close * 0.99,
        'close': close, 'volume': np.full(n, 600_000.0),
        'macd': macd, 'signal': np.zeros(n),
        'rsi': np.full(n, 55.0), 'volume_sma': np.full(n, 400_000.0),
    }, index=index)


class TestFoundationScoping:
    def _intents(self, owners, cross='down', quantity=100):
        desk = FoundationDesk()
        frame = momentum_frame(cross=cross)
        portfolio = book([held(stock('TST'), quantity=quantity,
                               owners=owners)])
        return desk.generate_intents({'TST': frame}, frame.index[-1],
                                     portfolio)

    def test_exit_skips_position_owned_by_another_desk(self):
        assert self._intents(owners=('other',)) == []

    def test_exit_closes_own_position(self):
        intents = self._intents(owners=('foundation',))
        assert [(i.asset.symbol, i.action) for i in intents] \
            == [('TST', 'SELL')]

    def test_exit_closes_untagged_position_legacy(self):
        intents = self._intents(owners=None)
        assert [i.action for i in intents] == ['SELL']

    def test_no_entry_into_symbol_another_desk_holds(self):
        # Invisible for exits, but still NOT enterable: a BUY would
        # co-mingle the two desks' books into one position.
        assert self._intents(owners=('other',), cross='up') == []


def trend_frame(closes, sma200: float, atr: float = 2.0) -> pd.DataFrame:
    n = len(closes)
    index = pd.bdate_range(end='2024-06-28', periods=n)
    return pd.DataFrame({'close': list(closes), 'sma_200': [sma200] * n,
                         'atr': [atr] * n}, index=index)


class TestTrendFollowerScoping:
    def _regime_exit(self, owners):
        desk = TrendFollowerDesk()
        frame = trend_frame([90.0] * 220, sma200=100.0)  # close < SMA200
        portfolio = book([held(stock('SPY'), quantity=50, owners=owners)])
        return desk.generate_intents({'SPY': frame}, frame.index[-1],
                                     portfolio)

    def test_regime_exit_skips_position_owned_by_another_desk(self):
        assert self._regime_exit(owners=('other',)) == []

    def test_regime_exit_closes_own_position(self):
        intents = self._regime_exit(owners=('trend_follower',))
        assert [i.action for i in intents] == ['SELL']

    def test_regime_exit_closes_untagged_position_legacy(self):
        intents = self._regime_exit(owners=None)
        assert [i.action for i in intents] == ['SELL']

    def test_no_breakout_entry_into_symbol_another_desk_holds(self):
        desk = TrendFollowerDesk()
        frame = trend_frame(range(100, 320), sma200=200.0)  # new high
        portfolio = book([held(stock('SPY'), quantity=50,
                               owners=('other',))])
        assert desk.generate_intents({'SPY': frame}, frame.index[-1],
                                     portfolio) == []


class TestJaneStreetRvScoping:
    def _broken_spread_closes(self, a_owners, b_owners):
        desk = JaneStreetDesk()
        desk._rv_position = {'legs': {'AAA': 'long', 'BBB': 'short'},
                             'entry_day': 0, 'z_entry': 2.5}
        desk._day_index = JS_GRACE
        positions = [held(stock('AAA'), quantity=100, owners=a_owners),
                     held(stock('BBB'), quantity=-100, owners=b_owners)]
        intents = desk._reconcile_rv(book(positions))
        return desk, [(i.asset.symbol, i.action) for i in intents]

    def test_foreign_leg_breaks_spread_but_is_never_closed(self):
        # BBB belongs to another desk (this desk's SHORT was netted away):
        # the spread is broken, so the desk closes ITS surviving leg only.
        desk, closes = self._broken_spread_closes(
            a_owners=('janestreet',), b_owners=('other',))
        assert closes == [('AAA', 'SELL')]
        assert desk._rv_position is None

    def test_untagged_legs_both_held_keeps_spread_legacy(self):
        desk, closes = self._broken_spread_closes(a_owners=None,
                                                  b_owners=None)
        assert closes == []
        assert desk._rv_position is not None

    def test_all_legs_foreign_drops_tracking_silently(self):
        desk, closes = self._broken_spread_closes(a_owners=('other',),
                                                  b_owners=('other',))
        assert closes == []
        assert desk._rv_position is None

    def test_rv_exit_skips_foreign_leg(self, monkeypatch):
        desk = JaneStreetDesk()
        desk._rv_position = {'legs': {'AAA': 'long', 'BBB': 'long'},
                             'entry_day': 0, 'z_entry': 2.5}
        monkeypatch.setattr(desk, '_regime_controller',
                            SimpleNamespace(is_fitted=True))
        monkeypatch.setattr(desk, '_rv_universe',
                            lambda all_data: ('AAA', ['BBB']))
        monkeypatch.setattr(desk, '_rv_zscore',
                            lambda all_data, date, etf, basket: 0.0)
        portfolio = book([
            held(stock('AAA'), quantity=100, owners=('janestreet',)),
            held(stock('BBB'), quantity=100, owners=('other',))])
        intents = desk._relative_value_intents(
            {}, pd.Timestamp('2023-06-01'), portfolio)
        assert [(i.asset.symbol, i.action) for i in intents] \
            == [('AAA', 'SELL')]
        assert desk._rv_position is None
