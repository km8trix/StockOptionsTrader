"""LocalBook (Step 5): the restart-surviving ledger — fill math, upserts,
zero-row deletion, cash — plus the ExecutionBroker ABC sweep now that
order_status is part of the contract. All hermetic (tmp sqlite files)."""

from __future__ import annotations

import sqlite3

import pytest

from brokers.base import ExecutionBroker
from brokers.etrade_client import (build_equity_order, build_option_order,
                                   build_spread_order)
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

    def test_fees_reduce_cash_on_both_sides_never_touch_basis(self, book):
        """Per-fill fees (brokers that report them): cash moves by
        -signed_qty*price - fees — a fee costs cash on a BUY and on a
        SELL — and the position basis stays fee-free."""
        book.set_cash(1000.0)
        book.record_fill("SPY", 10, 50.0, fees=1.25)   # -500 - 1.25
        assert book.cash() == pytest.approx(1000.0 - 500.0 - 1.25)
        assert (book.snapshot()["positions"]["SPY"]["avg_price"]
                == pytest.approx(50.0))                # basis fee-free
        book.record_fill("SPY", -10, 55.0, fees=1.25)  # +550 - 1.25
        assert book.cash() == pytest.approx(
            1000.0 - 500.0 - 1.25 + 550.0 - 1.25)
        assert book.positions() == {}                  # closed clean

    def test_fees_default_zero_is_byte_identical(self, book):
        book.set_cash(100.0)
        book.record_fill("SPY", 1, 50.0)
        assert book.cash() == pytest.approx(50.0)


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

    def test_accounts_are_isolated_within_one_environment(self, tmp_path):
        db = str(tmp_path / "book.db")
        first = LocalBook(db, env="sandbox", account_id_key="ACC-1")
        second = LocalBook(db, env="sandbox", account_id_key="ACC-2")
        first.set_cash(500.0)
        first.record_fill("SPY", 2, 50.0)
        assert second.reconciliation_snapshot() == {
            "positions": {}, "cash": 0.0, "initialized": False,
        }
        assert first.reconciliation_snapshot() == {
            "positions": {"SPY": 2.0}, "cash": 400.0,
            "initialized": True,
        }

    def test_env_only_schema_migrates_into_legacy_account_scope(self,
                                                                 tmp_path):
        db = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE local_book (
                env TEXT NOT NULL, symbol TEXT NOT NULL,
                quantity REAL NOT NULL, avg_price REAL NOT NULL,
                PRIMARY KEY (env, symbol))
        """)
        conn.execute("""
            CREATE TABLE local_cash (
                env TEXT PRIMARY KEY, cash REAL NOT NULL)
        """)
        conn.execute("INSERT INTO local_book VALUES ('sandbox','QQQ',3,10)")
        conn.execute("INSERT INTO local_cash VALUES ('sandbox',970)")
        conn.commit()
        conn.close()

        legacy = LocalBook(db, env="sandbox")
        assert legacy.reconciliation_snapshot() == {
            "positions": {"QQQ": 3.0}, "cash": 970.0,
            "initialized": True,
        }
        assert LocalBook(db, env="sandbox", account_id_key="ACC").positions() == {}


class TestSnapshotLifecycle:
    def test_empty_and_initialized_empty_are_distinguishable(self, book):
        assert book.initialized is False
        assert book.bootstrap_snapshot({"positions": {}, "cash": 123.0}) is True
        assert book.initialized is True
        assert book.metadata()["initialized_at"] is not None
        assert book.reconciliation_snapshot() == {
            "positions": {}, "cash": 123.0, "initialized": True,
        }

    def test_bootstrap_is_once_only_and_replace_is_authoritative(self, book):
        assert book.bootstrap({"SPY": {"quantity": 2, "avg_price": 50}},
                              900.0) is True
        assert book.bootstrap({"QQQ": 9}, 1.0) is False
        assert book.positions() == {"SPY": 2.0}

        book.replace_snapshot([
            {"symbol": "QQQ", "quantity": 4, "current_price": 25},
            {"symbol": "DUST", "quantity": 0},
        ], 700.0)
        assert book.snapshot() == {
            "positions": {"QQQ": {"quantity": 4.0, "avg_price": 25.0}},
            "cash": 700.0,
        }

    def test_clear_returns_scope_to_explicitly_uninitialized(self, book):
        book.bootstrap_snapshot({}, 0.0)
        book.track_order("1", build_equity_order("SPY", "BUY", 1))
        book.clear()
        assert book.reconciliation_snapshot()["initialized"] is False
        assert book.tracked_orders() == []


class TestDurableTrackedOrders:
    def test_order_survives_restart_and_reregistration_is_idempotent(self,
                                                                     tmp_path):
        db = str(tmp_path / "book.db")
        request = build_equity_order("SPY", "BUY", 10,
                                     client_order_id="CLIENT1")
        book = LocalBook(db, env="sandbox", account_id_key="ACC")
        first = book.track_order("123", request, status="OPEN")
        again = book.track_order("123", request, status="OPEN")
        assert again["created_at"] == first["created_at"]
        # Registration cannot regress a status already advanced by polling.
        book.apply_order_status("123", {
            "status": "CANCELLED", "filled_quantity": 0,
            "avg_fill_price": None,
        })
        book.track_order("123", request)
        assert book.tracked_order("123")["status"] == "CANCELLED"

        tracked = LocalBook(db, env="sandbox",
                            account_id_key="ACC").tracked_order("123")
        assert tracked is not None
        assert tracked["account_id_key"] == "ACC"
        assert tracked["request"] == request
        assert tracked["status"] == "CANCELLED"
        assert tracked["cumulative_booked_quantity"] == 0.0
        assert tracked["avg_price"] is None
        assert tracked["created_at"]
        assert tracked["updated_at"]

    def test_order_id_cannot_be_reused_for_a_different_request(self, book):
        book.track_order("123", build_equity_order("SPY", "BUY", 1))
        with pytest.raises(ValueError, match="different request"):
            book.track_order("123", build_equity_order("QQQ", "BUY", 1))

    def test_equity_cumulative_deltas_are_atomic_and_idempotent(self, book):
        book.set_cash(1_000.0)
        request = build_equity_order("SPY", "BUY", 10,
                                     client_order_id="CLIENT1")
        book.track_order("123", request)
        assert book.apply_cumulative_order_fill(
            "123", 4, 50.0, status="PARTIAL") == 4.0
        # Cumulative average 54 over 10 implies 56.6667 on the six new shares.
        assert book.apply_order_status("123", {
            "status": "FILLED", "filled_quantity": 10,
            "avg_fill_price": 54.0,
        }) == 6.0
        assert book.apply_order_status("123", {
            "status": "FILLED", "filled_quantity": 10,
            "avg_fill_price": 54.0,
        }) == 0.0
        assert book.positions() == {"SPY": 10.0}
        assert book.cash() == pytest.approx(460.0)
        assert book.snapshot()["positions"]["SPY"]["avg_price"] == pytest.approx(54)
        tracked = book.tracked_order("123")
        assert tracked is not None
        assert tracked["status"] == "FILLED"
        assert tracked["cumulative_booked_quantity"] == 10.0
        assert tracked["avg_price"] == 54.0

    def test_single_option_uses_canonical_key_and_contract_multiplier(self,
                                                                      book):
        book.set_cash(1_000.0)
        request = build_option_order(
            "SPY", "PUT", 440, "2026-07-17", "BUY_OPEN", 2,
            client_order_id="CLIENT2")
        assert book.apply_cumulative_order_fill(
            "OPT-1", 2, 1.25, status="FILLED",
            order_request=request) == 2.0
        assert book.positions() == {"SPY 2026-07-17 $440.0 put": 2.0}
        assert book.cash() == pytest.approx(750.0)

    def test_spread_books_each_leg_and_signed_net_cash_once(self, book):
        book.set_cash(1_000.0)
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 3},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 3},
        ], 1.15, client_order_id="CLIENT3")
        book.track_order("SPR-1", request)
        # EtradeClient's spread avg is signed by legs: credit == negative.
        assert book.apply_cumulative_order_fill(
            "SPR-1", 1, -1.10, status="PARTIAL") == 1.0
        assert book.apply_cumulative_order_fill(
            "SPR-1", 3, -1.15, status="FILLED") == 2.0
        assert book.apply_cumulative_order_fill(
            "SPR-1", 3, -1.15, status="FILLED") == 0.0
        assert book.positions() == {
            "IWM 2026-01-16 $195.0 put": -3.0,
            "IWM 2026-01-16 $190.0 put": 3.0,
        }
        assert book.cash() == pytest.approx(1_345.0)

    def test_spread_partial_restart_replay_has_no_duplicate_legs(self,
                                                                  tmp_path):
        db = str(tmp_path / "spread-restart.db")
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 3},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 3},
        ], 1.15, client_order_id="RESTARTSPREAD1")
        first = LocalBook(db, env="sandbox", account_id_key="ACC")
        first.set_cash(1_000.0)
        first.track_order("SPR-RESTART", request)
        assert first.apply_cumulative_order_fill(
            "SPR-RESTART", 1, -1.10, status="PARTIAL") == 1.0

        restarted = LocalBook(db, env="sandbox", account_id_key="ACC")
        assert restarted.apply_cumulative_order_fill(
            "SPR-RESTART", 3, -1.15, status="FILLED") == 2.0
        replayed = LocalBook(db, env="sandbox", account_id_key="ACC")
        assert replayed.apply_cumulative_order_fill(
            "SPR-RESTART", 3, -1.15, status="FILLED") == 0.0
        assert replayed.positions() == {
            "IWM 2026-01-16 $195.0 put": -3.0,
            "IWM 2026-01-16 $190.0 put": 3.0,
        }
        assert replayed.cash() == pytest.approx(1_345.0)
        assert replayed.tracked_order(
            "SPR-RESTART")["cumulative_booked_quantity"] == 3.0

    def test_ratio_spread_uses_gcd_package_units_across_restart(self,
                                                                 tmp_path):
        db = str(tmp_path / "ratio-spread-restart.db")
        # Six and nine total contracts are three packages of a 2:3 spread.
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 6},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 9},
        ], 1.10, client_order_id="RATIORESTART1")
        first = LocalBook(db, env="sandbox", account_id_key="ACC")
        first.set_cash(1_000.0)
        first.track_order("RATIO", request)
        assert first.apply_cumulative_order_fill(
            "RATIO", 1, -1.00, status="PARTIAL") == 1.0
        assert first.positions() == {
            "IWM 2026-01-16 $195.0 put": -2.0,
            "IWM 2026-01-16 $190.0 put": 3.0,
        }
        assert first.cash() == pytest.approx(1_100.0)

        restarted = LocalBook(db, env="sandbox", account_id_key="ACC")
        assert restarted.apply_cumulative_order_fill(
            "RATIO", 3, -1.10, status="FILLED") == 2.0
        replayed = LocalBook(db, env="sandbox", account_id_key="ACC")
        assert replayed.apply_cumulative_order_fill(
            "RATIO", 3, -1.10, status="FILLED") == 0.0
        assert replayed.positions() == {
            "IWM 2026-01-16 $195.0 put": -6.0,
            "IWM 2026-01-16 $190.0 put": 9.0,
        }
        assert replayed.cash() == pytest.approx(1_330.0)
        assert replayed.tracked_order(
            "RATIO")["cumulative_booked_quantity"] == 3.0

    def test_ratio_spread_overfill_uses_gcd_capacity(self, book):
        book.set_cash(1_000.0)
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 6},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 9},
        ], 1.10, client_order_id="RATIOOVERFILL1")
        book.track_order("RATIO-OVERFILL", request)
        with pytest.raises(ValueError, match="native quantity 3"):
            book.apply_cumulative_order_fill(
                "RATIO-OVERFILL", 4, -1.10, status="FILLED")
        assert book.positions() == {}
        assert book.cash() == 1_000.0
        assert book.tracked_order(
            "RATIO-OVERFILL")["cumulative_booked_quantity"] == 0.0

    def test_ratio_spread_rejects_non_integer_leg_quantities(self, book):
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 2},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 3},
        ], 1.10, client_order_id="RATIOINTEGER1")
        request["Order"][0]["Instrument"][1]["orderedQuantity"] = 3.5
        request["Order"][0]["Instrument"][1]["quantity"] = 3.5
        book.track_order("RATIO-NONINTEGER", request)
        with pytest.raises(ValueError, match="positive integer"):
            book.apply_cumulative_order_fill(
                "RATIO-NONINTEGER", 1, -1.10, status="PARTIAL")
        assert book.positions() == {}

    def test_spread_close_reverses_all_legs_and_net_cash_atomically(self,
                                                                     book):
        book.set_cash(1_000.0)
        entry = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 3},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 3},
        ], 1.15, client_order_id="ENTRYSPREAD1")
        closing = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "BUY_CLOSE", "quantity": 3},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "SELL_CLOSE", "quantity": 3},
        ], -0.50, client_order_id="CLOSESPREAD1")
        book.apply_cumulative_order_fill(
            "ENTRY", 3, -1.15, status="FILLED", order_request=entry)
        book.apply_cumulative_order_fill(
            "CLOSE", 3, 0.50, status="FILLED", order_request=closing)
        assert book.positions() == {}
        assert book.cash() == pytest.approx(1_195.0)

    def test_spread_overfill_is_rejected_without_phantom_legs(self, book):
        book.set_cash(1_000.0)
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 2},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 2},
        ], 1.15, client_order_id="OVERFILLSPREAD1")
        book.track_order("OVERFILL", request)
        with pytest.raises(ValueError, match="native quantity 2"):
            book.apply_cumulative_order_fill(
                "OVERFILL", 4, -1.15, status="FILLED")
        assert book.positions() == {}
        assert book.cash() == 1_000.0
        assert book.tracked_order(
            "OVERFILL")["cumulative_booked_quantity"] == 0.0

    def test_malformed_later_spread_leg_rolls_back_first_leg(self, book):
        book.set_cash(1_000.0)
        request = build_spread_order([
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 2},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 2},
        ], 1.15, client_order_id="BROKENSPREAD1")
        del request["Order"][0]["Instrument"][1]["Product"]["expiryDay"]
        book.track_order("BROKEN", request)
        with pytest.raises(KeyError, match="expiryDay"):
            book.apply_cumulative_order_fill(
                "BROKEN", 1, -1.15, status="PARTIAL")
        assert book.positions() == {}
        assert book.cash() == 1_000.0
        assert book.tracked_order(
            "BROKEN")["cumulative_booked_quantity"] == 0.0

    def test_failed_delta_rolls_back_order_positions_and_cash(self, book):
        book.set_cash(100.0)
        malformed = {"orderType": "FUTURE", "Order": [{"Instrument": [{}]}]}
        book.track_order("BAD", malformed)
        with pytest.raises(ValueError, match="unsupported tracked order type"):
            book.apply_cumulative_order_fill("BAD", 1, 10.0)
        assert book.cash() == 100.0
        assert book.positions() == {}
        tracked = book.tracked_order("BAD")
        assert tracked is not None
        assert tracked["cumulative_booked_quantity"] == 0.0

    @pytest.mark.parametrize("quantity,price,match", [
        (float("nan"), 10.0, "finite"),
        (1.0, float("inf"), "finite"),
    ])
    def test_non_finite_cumulative_status_is_rejected(self, book, quantity,
                                                       price, match):
        request = build_equity_order("SPY", "BUY", 1)
        with pytest.raises(ValueError, match=match):
            book.apply_cumulative_order_fill(
                "BAD-NUMBER", quantity, price, order_request=request)


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
