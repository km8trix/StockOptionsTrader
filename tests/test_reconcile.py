"""reconcile() (contract C19): clean match, quantity drift, missing
symbols both directions, cash tolerance, option contracts with the x100
multiplier, and the exact mismatch report shape."""

from __future__ import annotations

from datetime import datetime

import pytest

from brokers.reconcile import position_market_value, reconcile

OPT_KEY = "SPY 2026-07-17 $440.0 put"


class FakeBroker:
    def __init__(self, positions, cash):
        self._positions = positions
        self._cash = cash

    def get_portfolio_status(self):
        return {"cash": self._cash, "positions": self._positions,
                "portfolio_value": 0.0, "pending_orders": 0}


class TestReconcile:
    def test_clean_match(self):
        broker = FakeBroker(
            [{"symbol": "AAPL", "quantity": 100},
             {"symbol": OPT_KEY, "quantity": -2,
              "market_value": position_market_value(-2, 1.10, 100)}],
            cash=12_345.67)
        result = reconcile({"AAPL": 100, OPT_KEY: -2}, 12_345.67, broker)
        assert result["ok"] is True
        assert result["mismatches"] == []
        # checked_at is a parseable ISO timestamp.
        datetime.fromisoformat(result["checked_at"])

    def test_position_quantity_drift(self):
        broker = FakeBroker([{"symbol": "AAPL", "quantity": 90}], 0.0)
        result = reconcile({"AAPL": 100}, 0.0, broker)
        assert result["ok"] is False
        assert result["mismatches"] == [{
            "kind": "position", "symbol": "AAPL",
            "local": 100.0, "broker": 90.0}]

    def test_missing_symbol_both_directions(self):
        broker = FakeBroker([{"symbol": "MSFT", "quantity": 10}], 0.0)
        result = reconcile({"AAPL": 100}, 0.0, broker)
        assert result["ok"] is False
        assert {"kind": "position", "symbol": "AAPL",
                "local": 100.0, "broker": 0.0} in result["mismatches"]
        assert {"kind": "position", "symbol": "MSFT",
                "local": 0.0, "broker": 10.0} in result["mismatches"]

    def test_cash_off_by_more_than_a_cent(self):
        broker = FakeBroker([], cash=1000.02)
        result = reconcile({}, 1000.00, broker)
        assert result["ok"] is False
        assert result["mismatches"] == [{
            "kind": "cash", "symbol": None,
            "local": 1000.00, "broker": 1000.02}]

    def test_cash_within_a_cent_is_clean(self):
        broker = FakeBroker([], cash=1000.01)
        assert reconcile({}, 1000.00, broker)["ok"] is True

    def test_option_position_contracts_with_multiplier(self):
        """Option quantities reconcile in CONTRACTS; the x100 lives in
        market-value math (position_market_value), never in quantity."""
        assert position_market_value(3, 1.10, 100) == pytest.approx(330.0)
        assert position_market_value(100, 1.10, 1) == pytest.approx(110.0)
        broker = FakeBroker([{"symbol": OPT_KEY, "quantity": 3,
                              "market_value": 330.0}], 0.0)
        assert reconcile({OPT_KEY: 3}, 0.0, broker)["ok"] is True
        # A one-contract drift on the SAME canonical key is caught.
        result = reconcile({OPT_KEY: 2}, 0.0, broker)
        assert result["ok"] is False
        assert result["mismatches"] == [{
            "kind": "position", "symbol": OPT_KEY,
            "local": 2.0, "broker": 3.0}]

    def test_report_shape(self):
        broker = FakeBroker([{"symbol": "AAPL", "quantity": 1}], 5.0)
        result = reconcile({}, 0.0, broker)
        assert set(result) == {"ok", "mismatches", "checked_at"}
        for mismatch in result["mismatches"]:
            assert set(mismatch) == {"kind", "symbol", "local", "broker"}
            assert mismatch["kind"] in ("position", "cash")
