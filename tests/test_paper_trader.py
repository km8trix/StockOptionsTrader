"""PaperTrader tests: market orders, limit semantics, cancellation, status,
and the get_current_price window/staleness behavior.

No network: PaperTrader.get_current_price is monkeypatched in the order
tests, MarketDataHandler.fetch_stock_data is stubbed to fail loudly if
reached, and the window tests re-stub fetch_stock_data with canned frames.
"""

from __future__ import annotations

import logging
import threading
import time
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


class TestConcurrency:
    """Regression for the double-fill race under gunicorn --threads.

    Without the per-instance RLock, two overlapping process_orders() calls
    could both pass the fill check for the same pending order: double cash
    deduction, doubled position, then ValueError from the second
    pending_orders.remove(order). place_order's counter increment was
    likewise an unlocked read-modify-write minting duplicate order ids.
    """

    N_THREADS = 8

    @staticmethod
    def _run_threads(n, target):
        """Start n threads on target(i), join them, return raised exceptions."""
        errors = []

        def wrapper(i):
            try:
                target(i)
            except Exception as e:  # noqa: BLE001 - record every failure
                errors.append(e)

        threads = [threading.Thread(target=wrapper, args=(i,))
                   for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "thread deadlocked"
        return errors

    def test_concurrent_process_orders_fills_exactly_once(self, monkeypatch):
        # A slow price fetch widens the check-then-remove window that the
        # unlocked code raced on (barrier alone often missed it).
        def slow_price(self, symbol):
            time.sleep(0.005)
            return CURRENT_PRICE

        monkeypatch.setattr(PaperTrader, "get_current_price", slow_price)
        trader = PaperTrader(initial_capital=100_000)

        asset = _stock()
        trader.place_order(asset, OrderType.BUY, 10, limit_price=None)

        barrier = threading.Barrier(self.N_THREADS)

        def poll(i):
            barrier.wait(timeout=10)
            # Mix the two entry points that both execute fills.
            if i % 2:
                trader.get_portfolio_status()
            else:
                trader.process_orders()

        errors = self._run_threads(self.N_THREADS, poll)

        assert errors == [], f"thread(s) raised: {errors!r}"
        assert trader.pending_orders == []
        pos = trader.portfolio.get_position(asset)
        assert pos is not None
        assert pos.quantity == 10  # exactly one fill, not doubled
        # Cash deducted exactly once (incl. the 0.1% slippage).
        expected_cash = 100_000 - 10 * CURRENT_PRICE * 1.001
        assert trader.portfolio.cash == pytest.approx(expected_cash)

    def test_concurrent_place_order_mints_unique_ids(self, trader):
        barrier = threading.Barrier(self.N_THREADS)
        order_ids = []
        ids_lock = threading.Lock()

        def place(i):
            barrier.wait(timeout=10)
            # Limit far below market so nothing ever fills here.
            oid = trader.place_order(_stock(), OrderType.BUY, 1,
                                     limit_price=1.0)
            with ids_lock:
                order_ids.append(oid)

        errors = self._run_threads(self.N_THREADS, place)

        assert errors == [], f"thread(s) raised: {errors!r}"
        assert len(order_ids) == self.N_THREADS
        assert len(set(order_ids)) == self.N_THREADS, (
            f"duplicate order ids minted: {sorted(order_ids)}")
        assert len(trader.pending_orders) == self.N_THREADS


class TestOrderStatusParity:
    """Phase 1: order_status() gives paper the same fill-polling contract as
    LiveEtradeBroker, so a PatientExecutor can drive paper rehearsal."""

    def test_pending_order_reports_open(self, trader):
        oid = trader.place_order(_stock(), OrderType.BUY, 10, limit_price=90.0)
        status = trader.order_status(oid)
        assert status == {"status": "OPEN", "filled_quantity": 0,
                          "avg_fill_price": None}

    def test_filled_order_reports_full_quantity_and_price(self, trader):
        oid = trader.place_order(_stock(), OrderType.BUY, 10, limit_price=None)
        trader.process_orders()  # market order fills at CURRENT_PRICE
        status = trader.order_status(oid)
        assert status["status"] == "FILLED"
        assert status["filled_quantity"] == 10
        assert status["avg_fill_price"] == pytest.approx(CURRENT_PRICE)

    def test_unknown_order_is_none(self, trader):
        assert trader.order_status("ORD-999999") is None


class TestCancelProcessesFillsFirst:
    """Phase 1: cancel_order() processes pending fills FIRST, so a fill racing
    the cancel is captured (and an already-filled order is uncancellable)."""

    def test_cancel_of_marketable_order_captures_the_fill(self, trader):
        # A market order is immediately marketable; cancel must fill it first
        # (recording the position) and then report it un-cancellable.
        oid = trader.place_order(_stock(), OrderType.BUY, 10, limit_price=None)
        result = trader.cancel_order(oid)
        assert result is False                      # already filled, not cancelled
        pos = trader.portfolio.get_position(_stock())
        assert pos is not None and pos.quantity == 10   # the fill was preserved
        assert trader.order_status(oid)["status"] == "FILLED"

    def test_cancel_of_resting_limit_still_cancels(self, trader):
        oid = trader.place_order(_stock(), OrderType.BUY, 10, limit_price=90.0)
        assert trader.cancel_order(oid) is True     # never marketable -> cancelled
        assert trader.portfolio.get_position(_stock()) is None


class TestPriceSanityGuard:
    """Phase 1: opt-in fat-finger guard — warn-only, disabled by default."""

    def test_warns_on_fat_finger_when_enabled(self, monkeypatch, caplog):
        monkeypatch.setattr(PaperTrader, "get_current_price",
                            lambda self, symbol: 100.0)
        t = PaperTrader(price_sanity_threshold=0.5)
        with caplog.at_level(logging.WARNING, logger="brokers.base"):
            t.place_order(_stock(), OrderType.BUY, 1, limit_price=0.01)
        assert any("fat-finger" in r.message for r in caplog.records)

    def test_disabled_by_default(self, trader, caplog):
        with caplog.at_level(logging.WARNING, logger="brokers.base"):
            trader.place_order(_stock(), OrderType.BUY, 1, limit_price=0.01)
        assert not any("fat-finger" in r.message for r in caplog.records)

    def test_within_threshold_no_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(PaperTrader, "get_current_price",
                            lambda self, symbol: 100.0)
        t = PaperTrader(price_sanity_threshold=0.5)
        with caplog.at_level(logging.WARNING, logger="brokers.base"):
            t.place_order(_stock(), OrderType.BUY, 1, limit_price=99.0)
        assert not any("fat-finger" in r.message for r in caplog.records)


class TestRejectedOrders:
    """Declined executions must report REJECTED, never a phantom FILLED."""

    def test_insufficient_cash_buy_is_rejected(self, trader):
        # 100k cash, 2000 shares @ 100 * 1.001 slippage > 100k
        oid = trader.place_order(_stock(), OrderType.BUY, 2000, limit_price=None)
        trader.process_orders()
        st = trader.order_status(oid)
        assert st["status"] == "REJECTED"
        assert st["filled_quantity"] == 0
        assert trader.portfolio.cash == 100_000
        assert trader.portfolio.get_position(_stock()) is None
        assert oid not in [o.order_id for o in trader.pending_orders]

    def test_sell_without_position_is_rejected(self, trader):
        oid = trader.place_order(_stock(), OrderType.SELL, 5, limit_price=None)
        trader.process_orders()
        st = trader.order_status(oid)
        assert st["status"] == "REJECTED"
        assert st["filled_quantity"] == 0
        assert trader.portfolio.cash == 100_000

    def test_option_order_rejected_upfront(self, trader):
        opt = Asset(symbol="AAPL", asset_type=AssetType.CALL,
                    strike_price=100.0, expiration_date="2030-01-17")
        oid = trader.place_order(opt, OrderType.BUY, 1, limit_price=2.5)
        st = trader.order_status(oid)
        assert st["status"] == "REJECTED"
        assert trader.pending_orders == []
