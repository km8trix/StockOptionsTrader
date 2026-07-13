"""Tests for portfolio.risk_aggregator.PortfolioRiskAggregator.

Covers construction validation, account gross-leverage enforcement (cumulative,
multiplier-aware, deterministic order, boundary, fail-closed), closes always
passing, correlation enforcement + unevaluated discipline (reusing the Phase 0
RiskManager primitives), the sector no-op-records-unevaluated path, and the
report shape. Offline, deterministic, no engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd
import pytest

from core.models import Asset, AssetType, Position
from desks.base import DeskIntent
from portfolio.manager import PortfolioManager
from portfolio.risk_aggregator import PortfolioRiskAggregator

DATE = pd.Timestamp('2022-01-12')


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def call(symbol: str, strike: float) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.CALL,
                 strike_price=strike, expiration_date='2022-06-17')


def intent(asset: Asset, action: str, size_fraction: float) -> DeskIntent:
    return DeskIntent(asset=asset, action=action, size_fraction=size_fraction,
                      reason='test')


def position(asset: Asset, quantity: int, price: float) -> Position:
    return Position(asset=asset, quantity=quantity, avg_entry_price=price,
                    current_price=price, timestamp=datetime(2022, 1, 3))


def portfolio_with(*positions: Position,
                   capital: float = 100_000.0) -> PortfolioManager:
    # Debit the cost (multiplier-aware) so portfolio value stays == capital:
    # longs debit cash, shorts credit it (matching the engine's bookkeeping).
    portfolio = PortfolioManager(capital)
    for pos in positions:
        portfolio.cash -= (pos.quantity * pos.avg_entry_price
                           * pos.asset.multiplier)
        portfolio.add_position(pos)
    return portfolio


def frame(closes: List[float], start: str = '2022-01-03') -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({'close': closes}, index=idx)


class TestConstruction:
    def test_non_positive_gross_rejected(self):
        with pytest.raises(ValueError, match='max_gross_leverage'):
            PortfolioRiskAggregator(max_gross_leverage=0.0)

    def test_tiny_correlation_window_rejected(self):
        with pytest.raises(ValueError, match='correlation_window'):
            PortfolioRiskAggregator(correlation_window=2)


class TestGrossLeverage:
    def test_within_limit_all_approved(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0)
        intents = [intent(stock('AAA'), 'BUY', 0.4),
                   intent(stock('BBB'), 'BUY', 0.4)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      {}, DATE)
        assert len(approved) == 2 and blocked == []

    def test_cumulative_breach_blocks_the_overflow_open(self):
        # 0.4 each on a 100k account at 1.0x: AAA(40k), BBB(80k) pass; CCC
        # (120k) breaches. Deterministic order is by (symbol, action).
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0)
        intents = [intent(stock('CCC'), 'BUY', 0.4),
                   intent(stock('AAA'), 'BUY', 0.4),
                   intent(stock('BBB'), 'BUY', 0.4)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      {}, DATE)
        assert {i.asset.symbol for i in approved} == {'AAA', 'BBB'}
        assert [i.asset.symbol for i, _ in blocked] == ['CCC']
        assert any('gross' in r for _, r in blocked)

    def test_exactly_at_limit_passes(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0)
        intents = [intent(stock('AAA'), 'BUY', 0.5),
                   intent(stock('BBB'), 'BUY', 0.5)]  # 50k + 50k == 100k
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      {}, DATE)
        assert len(approved) == 2 and blocked == []

    def test_existing_book_counts_toward_gross(self):
        # 700 sh @ 100 = 70k held; a 40k open -> 110k > 100k -> blocked.
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0)
        book = portfolio_with(position(stock('AAA'), 700, 100.0))
        approved, blocked = agg.check([intent(stock('BBB'), 'BUY', 0.4)],
                                      book, {}, DATE)
        assert approved == []
        assert len(blocked) == 1

    def test_current_gross_is_multiplier_aware(self):
        # stock 100 sh @ 100 = 10k; short call 2 ct @ 50 = 2*50*100 = 10k.
        book = portfolio_with(position(stock('AAA'), 100, 100.0),
                              position(call('AAA', 105.0), -2, 50.0))
        assert PortfolioRiskAggregator._current_gross(book) == pytest.approx(
            20_000.0)

    def test_closes_always_pass_even_over_limit(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=0.01)  # ~nothing fits
        book = portfolio_with(position(stock('AAA'), 100, 100.0))
        sell = intent(stock('AAA'), 'SELL', 1.0)
        buy = intent(stock('BBB'), 'BUY', 0.5)
        approved, blocked = agg.check([sell, buy], book, {}, DATE)
        assert approved == [sell]
        assert [i.asset.symbol for i, _ in blocked] == ['BBB']

    def test_non_positive_capital_fails_closed(self):
        agg = PortfolioRiskAggregator()
        approved, blocked = agg.check([intent(stock('AAA'), 'BUY', 0.1)],
                                      PortfolioManager(0.0), {}, DATE)
        assert approved == []
        assert 'failing closed' in blocked[0][1]

    def test_absolute_stock_quantity_uses_actual_notional_not_fraction_proxy(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=0.02,
                                      max_correlation=1.0)
        absolute = DeskIntent(
            asset=stock('AAA'), action='SHORT', size_fraction=0.01,
            quantity=50, reason='absolute override')
        data = {'AAA': frame([100.0, 100.0, 100.0])}

        approved, blocked = agg.check(
            [absolute], PortfolioManager(100_000.0), data, DATE)

        assert approved == []
        assert len(blocked) == 1
        assert 'gross' in blocked[0][1]

    def test_absolute_stock_quantity_without_price_fails_closed(self):
        agg = PortfolioRiskAggregator(max_correlation=1.0)
        absolute = DeskIntent(
            asset=stock('MISSING'), action='BUY', size_fraction=0.01,
            quantity=1, reason='absolute override')

        approved, blocked = agg.check(
            [absolute], PortfolioManager(100_000.0), {}, DATE)

        assert approved == []
        assert 'cannot be valued' in blocked[0][1]


class TestCorrelation:
    def _correlated(self):
        # AAA and BBB share identical returns -> Pearson 1.0.
        return {'AAA': frame([100, 101, 102, 103, 104]),
                'BBB': frame([50, 50.5, 51, 51.5, 52])}

    def test_overcorrelated_book_blocks_pending_opens(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0,
                                      max_correlation=0.8,
                                      correlation_window=10)
        intents = [intent(stock('AAA'), 'BUY', 0.1),
                   intent(stock('BBB'), 'BUY', 0.1)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      self._correlated(), DATE)
        assert approved == []  # gross fine, but correlation holds them
        assert len(blocked) == 2
        assert any('Correlation' in v for v in agg.violations)

    def test_uncorrelated_book_passes(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0,
                                      max_correlation=0.8,
                                      correlation_window=10)
        all_data = {'AAA': frame([100, 101, 100, 101, 100, 101]),
                    'BBB': frame([100, 100.5, 101, 101.5, 102, 102.5])}
        intents = [intent(stock('AAA'), 'BUY', 0.1),
                   intent(stock('BBB'), 'BUY', 0.1)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      all_data, DATE)
        assert len(approved) == 2 and blocked == []

    def test_returns_are_inner_aligned_by_session_date(self):
        agg = PortfolioRiskAggregator(correlation_window=10)
        a_dates = pd.to_datetime([
            '2022-01-03', '2022-01-04', '2022-01-05',
            '2022-01-06', '2022-01-07'])
        b_dates = pd.to_datetime([
            '2022-01-03', '2022-01-05', '2022-01-06',
            '2022-01-07', '2022-01-10'])
        all_data = {
            'AAA': pd.DataFrame(
                {'close': [100.0, 110.0, 99.0, 108.9, 98.01]},
                index=a_dates),
            'BBB': pd.DataFrame(
                {'close': [100.0, 120.0, 132.0, 118.8, 130.68]},
                index=b_dates),
        }

        returns = agg._returns(['AAA', 'BBB'], all_data, DATE)

        # Only Jan 5/6/7 are present in BOTH dated return series. The missing
        # Jan 4 BBB bar must never shift Jan 5 against AAA Jan 4 by ordinal.
        assert returns['AAA'] == pytest.approx([-0.10, 0.10, -0.10])
        assert returns['BBB'] == pytest.approx([0.20, 0.10, -0.10])

    def test_short_series_recorded_unevaluated(self):
        agg = PortfolioRiskAggregator(max_correlation=0.8)
        all_data = {'AAA': frame([100, 101]), 'BBB': frame([50, 51])}
        intents = [intent(stock('AAA'), 'BUY', 0.1),
                   intent(stock('BBB'), 'BUY', 0.1)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      all_data, DATE)
        assert len(approved) == 2 and blocked == []
        assert any('correlation' in u for u in agg.unevaluated)

    def test_flat_series_records_unevaluated_not_silent_pass(self):
        # HIGH regression: a zero-variance (flat/halted) series makes
        # np.corrcoef NaN; the cap must be RECORDED as unevaluated, never a
        # silent pass (the previous check_correlation returned True silently).
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0,
                                      max_correlation=0.5,
                                      correlation_window=10)
        all_data = {'FLAT': frame([100, 100, 100, 100, 100]),
                    'VARY': frame([100, 101, 102, 103, 104])}
        intents = [intent(stock('FLAT'), 'BUY', 0.1),
                   intent(stock('VARY'), 'BUY', 0.1)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      all_data, DATE)
        assert len(approved) == 2 and blocked == []   # not blocked,
        assert any('correlation' in u for u in agg.unevaluated)  # but recorded

    def test_different_history_lengths_align_on_common_sessions(self):
        agg = PortfolioRiskAggregator(max_correlation=1.0,
                                      correlation_window=50)
        all_data = {'AAA': frame([100, 101, 102, 103]),
                    'BBB': frame([100, 101, 102, 103, 104, 105, 106])}
        intents = [intent(stock('AAA'), 'BUY', 0.1),
                   intent(stock('BBB'), 'BUY', 0.1)]
        approved, blocked = agg.check(intents, PortfolioManager(100_000.0),
                                      all_data, DATE)
        assert len(approved) == 2 and blocked == []
        assert not any('ragged' in u for u in agg.unevaluated)


class TestSectorAndReport:
    def test_sectorless_book_records_unevaluated(self):
        agg = PortfolioRiskAggregator(max_sector_exposure=0.3)
        book = portfolio_with(position(stock('AAA'), 100, 100.0))
        approved, _ = agg.check([intent(stock('BBB'), 'BUY', 0.05)],
                                book, {}, DATE)
        assert len(approved) == 1
        assert any('sector' in u for u in agg.unevaluated)

    def test_report_shape(self):
        agg = PortfolioRiskAggregator(max_gross_leverage=1.0)
        agg.check([intent(stock('AAA'), 'BUY', 0.5)],
                  PortfolioManager(100_000.0), {}, DATE)
        report = agg.get_risk_report()
        assert set(report) == {'violations', 'unevaluated',
                               'max_gross_leverage', 'max_correlation',
                               'max_sector_exposure'}
        assert report['max_gross_leverage'] == 1.0
