"""LocalBook (Step 5): the restart-surviving ledger — fill math, upserts,
zero-row deletion, cash — plus the ExecutionBroker ABC sweep now that
order_status is part of the contract. All hermetic (tmp sqlite files)."""

from __future__ import annotations

import pytest

from brokers.base import ExecutionBroker
from brokers.local_book import LocalBook


@pytest.fixture
def book(tmp_path):
    return LocalBook(str(tmp_path / "book.db"), env="sandbox")


class TestFillMath:
    def test_buy_opens_position_and_spends_cash(self, book):
        book.set_cash(1000.0)
        book.record_fill("SPY", 10, 50.0)
        assert book.positions() == {"SPY": 10.0}
        assert book.cash() == pytest.approx(500.0)
        snap = book.snapshot()["positions"]["SPY"]
        assert snap["avg_price"] == pytest.approx(50.0)

    def test_averaging_up_weights_the_basis(self, book):
        book.record_fill("SPY", 10, 50.0)
        book.record_fill("SPY", 10, 60.0)
        snap = book.snapshot()["positions"]["SPY"]
        assert snap["quantity"] == 20.0
        assert snap["avg_price"] == pytest.approx(55.0)

    def test_partial_sell_reduces_qty_keeps_basis_raises_cash(self, book):
        book.set_cash(0.0)
        book.record_fill("SPY", 10, 50.0)     # cash -500
        book.record_fill("SPY", -4, 55.0)     # cash +220
        snap = book.snapshot()
        assert snap["positions"]["SPY"]["quantity"] == 6.0
        assert snap["positions"]["SPY"]["avg_price"] == pytest.approx(50.0)
        assert snap["cash"] == pytest.approx(-280.0)

    def test_full_close_deletes_the_zero_row(self, book):
        book.record_fill("SPY", 10, 50.0)
        book.record_fill("SPY", -10, 55.0)
        assert book.positions() == {}
        assert book.snapshot()["positions"] == {}

    def test_flip_through_zero_resets_basis_to_fill_price(self, book):
        book.record_fill("SPY", 10, 50.0)
        book.record_fill("SPY", -15, 60.0)    # long 10 -> short 5
        snap = book.snapshot()["positions"]["SPY"]
        assert snap["quantity"] == -5.0
        assert snap["avg_price"] == pytest.approx(60.0)

    def test_short_adds_weight_on_absolute_qty_and_raise_cash(self, book):
        book.record_fill("SPY", -10, 50.0)
        book.record_fill("SPY", -10, 40.0)
        snap = book.snapshot()["positions"]["SPY"]
        assert snap["quantity"] == -20.0
        assert snap["avg_price"] == pytest.approx(45.0)
        assert book.cash() == pytest.approx(900.0)   # shorts raise cash


class TestPersistenceAndScoping:
    def test_state_survives_a_new_instance_on_the_same_file(self, tmp_path):
        db = str(tmp_path / "book.db")
        LocalBook(db, env="sandbox").record_fill("QQQ", 3, 10.0)
        fresh = LocalBook(db, env="sandbox")     # the "restart"
        assert fresh.positions() == {"QQQ": 3.0}
        assert fresh.cash() == pytest.approx(-30.0)

    def test_envs_are_isolated(self, tmp_path):
        db = str(tmp_path / "book.db")
        sandbox = LocalBook(db, env="sandbox")
        prod = LocalBook(db, env="production")
        sandbox.record_fill("SPY", 1, 100.0)
        assert prod.positions() == {}
        assert prod.cash() == 0.0
        assert sandbox.positions() == {"SPY": 1.0}

    def test_set_cash_is_absolute_and_clear_wipes_the_env(self, book):
        book.record_fill("SPY", 1, 100.0)
        book.set_cash(42.0)
        assert book.cash() == 42.0
        book.clear()
        assert book.positions() == {}
        assert book.cash() == 0.0

    def test_cash_defaults_to_zero(self, book):
        assert book.cash() == 0.0


class TestExecutionBrokerABC:
    """order_status(order_id) joined the ABC (Step 5) — previously duck-
    typed. Both concrete brokers must stay concrete, and a subclass
    missing it must refuse to instantiate."""

    def test_order_status_is_abstract_on_the_interface(self):
        assert "order_status" in ExecutionBroker.__abstractmethods__

    def test_concrete_brokers_have_no_abstract_holes(self):
        from brokers.live_trader import LiveEtradeBroker
        from brokers.paper_trader import PaperTrader
        assert not PaperTrader.__abstractmethods__
        assert not LiveEtradeBroker.__abstractmethods__

    def test_subclass_missing_order_status_cannot_instantiate(self):
        class NoStatus(ExecutionBroker):
            def place_order(self, asset, order_type, quantity, limit_price):
                return "x"

            def cancel_order(self, order_id):
                return False

            def get_portfolio_status(self):
                return {}

            def get_current_price(self, symbol):
                return None

        with pytest.raises(TypeError):
            NoStatus()
