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

    def test_option_assignment_is_caught_by_share_and_cash_checks(self):
        # Phase 1: an overnight short-put ASSIGNMENT — the broker now shows the
        # option gone, 100 new shares of the underlying, and ~$44k less cash,
        # while the local book still believes it holds the short put with the
        # old cash. The dangerous divergence DOES surface (engaging the
        # kill-switch safety net) through the existing union + cash checks — no
        # special known-positions machinery is needed.
        broker = FakeBroker(
            [{"symbol": "SPY", "quantity": 100}],   # assigned shares appear
            cash=56_000.0)                          # ~$44k assignment cost out
        result = reconcile({OPT_KEY: -1}, 100_000.0, broker)
        assert result["ok"] is False
        kinds = {(m["kind"], m["symbol"]) for m in result["mismatches"]}
        assert ("position", "SPY") in kinds         # new shares vs local 0
        assert ("position", OPT_KEY) in kinds       # local short put vs broker 0
        assert ("cash", None) in kinds              # the cash delta

    def test_both_books_agree_position_gone_stays_clean(self):
        # A worthless expiry zeroes a position in BOTH books with no share/cash
        # effect — intentionally NOT a mismatch (flagging it would false-halt
        # trading on every routine expiry).
        broker = FakeBroker([], cash=100_000.0)
        assert reconcile({}, 100_000.0, broker)["ok"] is True

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

    def test_fee_aware_cash_tolerance_widens_cash_only(self):
        """cash_tolerance= absorbs unreported per-fill fee drift (the
        operational pattern in brokers.local_book.LocalBook): a $3 cash
        drift fails the strict default but passes at $5.00 — while a
        POSITION drift still fails at any cash tolerance."""
        broker = FakeBroker([], cash=997.00)
        assert reconcile({}, 1000.00, broker)["ok"] is False  # default $0.01
        assert reconcile({}, 1000.00, broker,
                         cash_tolerance=5.00)["ok"] is True
        # Just past the tolerance still fails.
        assert reconcile({}, 1002.50, broker,
                         cash_tolerance=5.00)["ok"] is False
        # Positions stay strict regardless of cash tolerance.
        drifted = FakeBroker([{"symbol": "AAPL", "quantity": 10}],
                             cash=1000.00)
        result = reconcile({"AAPL": 9}, 1000.00, drifted,
                           cash_tolerance=5.00)
        assert result["ok"] is False
        assert result["mismatches"] == [{
            "kind": "position", "symbol": "AAPL",
            "local": 9.0, "broker": 10.0}]

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
