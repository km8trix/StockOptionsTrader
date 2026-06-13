"""Tests for portfolio.greeks_aggregator.PortfolioGreeksAggregator.

The aggregator is the desk-agnostic C12 primitive: per-share greeks (from
an injected provider) x signed contract count x the 100 multiplier, summed
over option positions. These tests pin the share-equivalent convention,
the sign (short premium -> negative greeks), the skip rules (zero-quantity,
stock-only, provider-None), determinism, and a real Black-Scholes parity
check against the exact inline arithmetic JaneStreetDesk used to run.
Offline, deterministic, no engine.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.models import Asset, AssetType, Position
from desks.options_pricing import black_scholes_greeks
from portfolio.greeks_aggregator import (GREEK_KEYS,
                                         PortfolioGreeksAggregator)
from portfolio.manager import PortfolioManager

EXPIRY = '2022-03-18'


def call(strike: float) -> Asset:
    return Asset(symbol='AAA', asset_type=AssetType.CALL,
                 strike_price=strike, expiration_date=EXPIRY)


def put(strike: float) -> Asset:
    return Asset(symbol='AAA', asset_type=AssetType.PUT,
                 strike_price=strike, expiration_date=EXPIRY)


def stock(symbol: str = 'AAA') -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def portfolio_with(*positions: Position,
                   capital: float = 100_000.0) -> PortfolioManager:
    portfolio = PortfolioManager(capital)
    for pos in positions:
        portfolio.add_position(pos)
    return portfolio


def position(asset: Asset, quantity: int, price: float = 1.0) -> Position:
    return Position(asset=asset, quantity=quantity, avg_entry_price=price,
                    current_price=price, timestamp=datetime(2022, 1, 3))


def const_provider(**per_share):
    """A provider returning the SAME per-share greeks for every asset."""
    full = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    full.update(per_share)
    return lambda asset: dict(full)


class TestConstruction:
    def test_non_callable_provider_rejected(self):
        with pytest.raises(ValueError, match='callable'):
            PortfolioGreeksAggregator('not-a-function')

    def test_zero_is_fresh_each_call(self):
        first = PortfolioGreeksAggregator.zero()
        first['vega'] = 99.0
        assert PortfolioGreeksAggregator.zero() == {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        assert set(PortfolioGreeksAggregator.zero()) == set(GREEK_KEYS)


class TestAggregation:
    def test_empty_portfolio_is_all_zeros(self):
        agg = PortfolioGreeksAggregator(const_provider(vega=10.0))
        assert agg.aggregate(PortfolioManager(100_000.0)) == {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

    def test_stock_only_portfolio_is_all_zeros(self):
        # Stocks are not options: the provider is never consulted for them.
        agg = PortfolioGreeksAggregator(const_provider(vega=10.0, delta=1.0))
        book = portfolio_with(position(stock('AAA'), 500),
                              position(stock('BBB'), -200))
        assert agg.aggregate(book) == {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

    def test_single_short_call_is_share_equivalent_and_negative(self):
        # Short premium: a -2-contract short call with +10 per-share vega
        # reports 10 * -2 * 100 = -2000 (NEGATIVE -> short vega).
        agg = PortfolioGreeksAggregator(
            const_provider(delta=0.5, gamma=0.02, theta=-0.03, vega=10.0))
        book = portfolio_with(position(call(105.0), -2))
        totals = agg.aggregate(book)
        assert totals['vega'] == pytest.approx(-2000.0)
        assert totals['delta'] == pytest.approx(-100.0)
        assert totals['gamma'] == pytest.approx(-4.0)
        assert totals['theta'] == pytest.approx(6.0)

    def test_long_and_short_legs_sum(self):
        # +3 long call and -1 short put, per-share vega 4.0 each:
        # 4*3*100 + 4*(-1)*100 = 1200 - 400 = 800.
        agg = PortfolioGreeksAggregator(const_provider(vega=4.0))
        book = portfolio_with(position(call(110.0), 3),
                              position(put(90.0), -1))
        assert agg.aggregate(book)['vega'] == pytest.approx(800.0)

    def test_zero_quantity_positions_are_skipped(self):
        agg = PortfolioGreeksAggregator(const_provider(vega=7.0))
        book = portfolio_with(position(call(105.0), 0),
                              position(put(95.0), -2))
        assert agg.aggregate(book)['vega'] == pytest.approx(-1400.0)

    def test_provider_none_leg_contributes_nothing(self):
        # The provider can't price the put today (returns None); only the
        # call counts. call vega 5*1*100 = 500.
        def provider(asset):
            if asset.asset_type is AssetType.PUT:
                return None
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 5.0}

        agg = PortfolioGreeksAggregator(provider)
        book = portfolio_with(position(call(105.0), 1),
                              position(put(95.0), -3))
        assert agg.aggregate(book)['vega'] == pytest.approx(500.0)

    def test_extra_provider_keys_ignored(self):
        # black_scholes_greeks also returns 'price'; the aggregator must
        # only read the four greek keys.
        agg = PortfolioGreeksAggregator(
            lambda a: {'price': 12.3, 'delta': 1.0, 'gamma': 0.0,
                       'theta': 0.0, 'vega': 2.0})
        totals = agg.aggregate(portfolio_with(position(call(100.0), 1)))
        assert set(totals) == set(GREEK_KEYS)
        assert totals['vega'] == pytest.approx(200.0)

    def test_matches_inline_black_scholes_arithmetic(self):
        # Parity: the aggregator reproduces, bit-for-bit, the C12 inline
        # loop JaneStreetDesk._update_greeks used to run with real BS.
        spot, iv, t_years, rate = 100.0, 0.25, 30 / 365.0, 0.04
        legs = [(call(105.0), -2), (call(115.0), 1),
                (put(95.0), -2), (put(85.0), 1)]
        book = portfolio_with(*[position(a, q) for a, q in legs])

        def provider(asset):
            return black_scholes_greeks(spot, asset.strike_price, t_years,
                                        iv, rate, asset.asset_type.value)

        # Mirror the aggregator's sorted(str(asset)) visit order so the
        # float summation is bit-identical, not just close.
        expected = {k: 0.0 for k in GREEK_KEYS}
        for asset, quantity in sorted(legs, key=lambda lq: str(lq[0])):
            g = black_scholes_greeks(spot, asset.strike_price, t_years, iv,
                                     rate, asset.asset_type.value)
            scale = quantity * asset.multiplier
            for k in expected:
                expected[k] += g[k] * scale

        got = PortfolioGreeksAggregator(provider).aggregate(book)
        assert got == expected  # exact equality: same ops, same order
