"""PaperTrader tests: market orders, limit semantics, cancellation, status,
and the get_current_price window/staleness behavior.

No network: PaperTrader.get_current_price is monkeypatched in the order
tests, MarketDataHandler.fetch_stock_data is stubbed to fail loudly if
reached, and the window tests re-stub fetch_stock_data with canned frames.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from brokers.paper_trader import PaperTrader
from core.models import Asset, AssetType, OrderType, Position
from data.market_data import MarketDataHandler

CURRENT_PRICE = 100.0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that accidentally reaches the market data layer."""

    def _boom(self, symbol, start_date, end_date):
        raise AssertionError("network path reached: fetch_stock_data called")

    monkeypatch.setattr(MarketDataHandler, "fetch_stock_data", _boom)


@pytest.fixture
def trader(monkeypatch):
    monkeypatch.setattr(
        PaperTrader, "get_current_price",
        lambda self, symbol: CURRENT_PRICE,
    )
    return PaperTrader(initial_capital=100_000)


def _stock(symbol: str = "AAPL") -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def _seed_position(trader: PaperTrader, asset: Asset, quantity: int = 10,
                   entry: float = 90.0) -> None:
    trader.portfolio.add_position(Position(
        asset=asset,
        quantity=quantity,
        avg_entry_price=entry,
        current_price=entry,
        timestamp=datetime.now(),
    ))


class TestMarketOrders:
    def test_market_buy_fills_at_current_price(self, trader):
        asset = _stock()
        trader.place_order(asset, OrderType.BUY, 10, limit_price=None)

        trader.process_orders()

        assert trader.pending_orders == []
        pos = trader.portfolio.get_position(asset)
        assert pos is not None
        assert pos.quantity == 10
        assert pos.avg_entry_price == pytest.approx(CURRENT_PRICE)
        # Cost includes the 0.1% slippage applied by _execute_order.
        expected_cash = 100_000 - 10 * CURRENT_PRICE * 1.001
        assert trader.portfolio.cash == pytest.approx(expected_cash)

    def test_market_sell_fills_at_current_price(self, trader):
        asset = _stock()
        _seed_position(trader, asset, quantity=10, entry=90.0)
        trader.place_order(asset, OrderType.SELL, 10, limit_price=None)

        trader.process_orders()

        assert trader.pending_orders == []
        assert trader.portfolio.get_position(asset) is None
        expected_cash = 100_000 + 10 * CURRENT_PRICE * 0.999
        assert trader.portfolio.cash == pytest.approx(expected_cash)

    def test_get_portfolio_status_survives_market_order(self, trader):
        """Regression: a market order (price=None) used to raise TypeError in
        process_orders and poison every subsequent get_portfolio_status call."""
        asset = _stock()
        trader.place_order(asset, OrderType.BUY, 5, limit_price=None)

        status = trader.get_portfolio_status()

        assert status['pending_orders'] == 0
        assert status['positions'][0]['symbol'] == "AAPL"
        assert status['positions'][0]['quantity'] == 5

        # A second call must also succeed (the trader is not poisoned).
        status_again = trader.get_portfolio_status()
        assert status_again['pending_orders'] == 0


class TestLimitOrders:
    def test_limit_buy_below_market_stays_pending(self, trader):
        asset = _stock()
        trader.place_order(asset, OrderType.BUY, 10, limit_price=90.0)

        trader.process_orders()

        assert len(trader.pending_orders) == 1
        assert trader.portfolio.get_position(asset) is None
        assert trader.portfolio.cash == pytest.approx(100_000)

    def test_limit_buy_at_or_above_market_fills(self, trader):
        asset = _stock()
        trader.place_order(asset, OrderType.BUY, 10, limit_price=105.0)

        trader.process_orders()

        assert trader.pending_orders == []
        pos = trader.portfolio.get_position(asset)
        assert pos is not None
        assert pos.avg_entry_price == pytest.approx(CURRENT_PRICE)

    def test_limit_sell_above_market_stays_pending(self, trader):
        asset = _stock()
        _seed_position(trader, asset)
        trader.place_order(asset, OrderType.SELL, 10, limit_price=110.0)

        trader.process_orders()

        assert len(trader.pending_orders) == 1
        assert trader.portfolio.get_position(asset) is not None

    def test_limit_sell_at_or_below_market_fills(self, trader):
        asset = _stock()
        _seed_position(trader, asset)
        trader.place_order(asset, OrderType.SELL, 10, limit_price=95.0)

        trader.process_orders()

        assert trader.pending_orders == []
        assert trader.portfolio.get_position(asset) is None


class TestCancelOrder:
    def test_cancel_pending_order_returns_true(self, trader):
        asset = _stock()
        order_id = trader.place_order(asset, OrderType.BUY, 10, limit_price=90.0)

        assert trader.cancel_order(order_id) is True
        assert trader.pending_orders == []

    def test_cancel_unknown_order_returns_false(self, trader):
        assert trader.cancel_order("ORD-999999") is False


def _stub_fetch(monkeypatch, frame, captured=None):
    """Replace fetch_stock_data with a canned frame (overrides no_network)."""

    def fake_fetch(self, symbol, start_date, end_date):
        if captured is not None:
            captured['args'] = (symbol, start_date, end_date)
        return frame

    monkeypatch.setattr(MarketDataHandler, "fetch_stock_data", fake_fetch)


def _close_frame(dates, closes):
    return pd.DataFrame(
        {"close": closes},
        index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"),
    )


class TestGetCurrentPriceWindow:
    """D6: weekend/holiday-safe window fetch with staleness warning."""

    def test_latest_close_from_10_day_window(self, monkeypatch):
        # Weekend scenario: today has no bar; the latest close in the
        # window (e.g. Friday's) must be returned.
        today = datetime.now()
        dates = [today - timedelta(days=4), today - timedelta(days=2)]
        captured = {}
        _stub_fetch(monkeypatch, _close_frame(dates, [100.0, 123.45]),
                    captured)
        trader = PaperTrader()

        price = trader.get_current_price("AAPL")

        assert price == pytest.approx(123.45)
        symbol, start_str, end_str = captured['args']
        assert symbol == "AAPL"
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        assert end_str == today.strftime("%Y-%m-%d")  # through today
        assert (end - start).days == 10  # last 10 calendar days

    def test_price_date_recorded(self, monkeypatch):
        today = datetime.now()
        last_bar = today - timedelta(days=1)
        _stub_fetch(monkeypatch, _close_frame([last_bar], [50.0]))
        trader = PaperTrader()

        trader.get_current_price("AAPL")

        assert trader.last_price_dates["AAPL"] == last_bar.date()

    def test_stale_price_logs_warning(self, monkeypatch, caplog):
        # A close 14 calendar days back is always > 3 business days old.
        today = datetime.now()
        stale_day = today - timedelta(days=14)
        _stub_fetch(monkeypatch, _close_frame([stale_day], [77.0]))
        trader = PaperTrader()

        with caplog.at_level(logging.WARNING, logger="brokers.paper_trader"):
            price = trader.get_current_price("AAPL")

        assert price == pytest.approx(77.0)  # stale, but still returned
        assert any("Stale price" in r.message and "business days old"
                   in r.message for r in caplog.records)

    def test_fresh_price_no_stale_warning(self, monkeypatch, caplog):
        today = datetime.now()
        _stub_fetch(monkeypatch, _close_frame([today], [88.0]))
        trader = PaperTrader()

        with caplog.at_level(logging.WARNING, logger="brokers.paper_trader"):
            price = trader.get_current_price("AAPL")

        assert price == pytest.approx(88.0)
        assert not any("Stale price" in r.message for r in caplog.records)

    def test_empty_window_returns_none(self, monkeypatch, caplog):
        _stub_fetch(monkeypatch, pd.DataFrame())
        trader = PaperTrader()

        with caplog.at_level(logging.WARNING, logger="brokers.paper_trader"):
            assert trader.get_current_price("AAPL") is None

        assert any("No price data" in r.message for r in caplog.records)
