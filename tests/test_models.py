"""Tests for core.models: Asset identity, Position P&L, Trade P&L."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.models import Asset, AssetType, Position, Trade


class TestAssetEqualityAndHash:
    def test_identical_stocks_are_equal_and_hash_equal(self):
        a = Asset('AAPL', AssetType.STOCK)
        b = Asset('AAPL', AssetType.STOCK)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_symbols_are_not_equal(self):
        assert Asset('AAPL', AssetType.STOCK) != Asset('MSFT', AssetType.STOCK)

    def test_stock_vs_option_same_symbol_not_equal(self):
        stock = Asset('AAPL', AssetType.STOCK)
        call = Asset('AAPL', AssetType.CALL, strike_price=150.0,
                     expiration_date='2023-06-16')
        assert stock != call

    def test_options_differ_by_strike(self):
        call_150 = Asset('AAPL', AssetType.CALL, strike_price=150.0,
                         expiration_date='2023-06-16')
        call_155 = Asset('AAPL', AssetType.CALL, strike_price=155.0,
                         expiration_date='2023-06-16')
        assert call_150 != call_155
        assert hash(call_150) != hash(call_155)

    def test_options_differ_by_expiration(self):
        near = Asset('AAPL', AssetType.CALL, strike_price=150.0,
                     expiration_date='2023-06-16')
        far = Asset('AAPL', AssetType.CALL, strike_price=150.0,
                    expiration_date='2023-09-15')
        assert near != far

    def test_identical_options_usable_as_dict_key(self):
        a = Asset('AAPL', AssetType.PUT, strike_price=140.0,
                  expiration_date='2023-06-16')
        b = Asset('AAPL', AssetType.PUT, strike_price=140.0,
                  expiration_date='2023-06-16')
        positions = {a: 'first'}
        positions[b] = 'second'
        assert len(positions) == 1
        assert positions[a] == 'second'

    def test_asset_not_equal_to_other_types(self):
        assert Asset('AAPL', AssetType.STOCK) != 'AAPL'


class TestPosition:
    def _position(self, quantity=100, entry=150.0, current=155.0):
        return Position(
            asset=Asset('AAPL', AssetType.STOCK),
            quantity=quantity,
            avg_entry_price=entry,
            current_price=current,
            timestamp=datetime(2023, 1, 2),
        )

    def test_pnl_gain(self):
        pos = self._position(quantity=100, entry=150.0, current=155.0)
        assert pos.pnl() == pytest.approx(500.0)

    def test_pnl_loss(self):
        pos = self._position(quantity=50, entry=200.0, current=190.0)
        assert pos.pnl() == pytest.approx(-500.0)

    def test_pnl_pct(self):
        pos = self._position(quantity=10, entry=100.0, current=103.0)
        assert pos.pnl_pct() == pytest.approx(3.0)

    def test_pnl_pct_zero_entry_price_returns_zero(self):
        pos = self._position(entry=0.0, current=10.0)
        assert pos.pnl_pct() == 0


class TestTrade:
    def _trade(self, entry=100.0, exit_=110.0, quantity=10):
        return Trade(
            asset=Asset('MSFT', AssetType.STOCK),
            entry_price=entry,
            exit_price=exit_,
            quantity=quantity,
            entry_time=datetime(2023, 1, 2),
            exit_time=datetime(2023, 1, 10),
        )

    def test_pnl_gain(self):
        trade = self._trade(entry=100.0, exit_=110.0, quantity=10)
        assert trade.pnl == pytest.approx(100.0)
        assert trade.pnl_pct == pytest.approx(10.0)

    def test_pnl_loss(self):
        trade = self._trade(entry=100.0, exit_=95.0, quantity=20)
        assert trade.pnl == pytest.approx(-100.0)
        assert trade.pnl_pct == pytest.approx(-5.0)

    def test_pnl_pct_zero_entry_price_returns_zero(self):
        trade = self._trade(entry=0.0, exit_=10.0, quantity=5)
        assert trade.pnl_pct == 0
