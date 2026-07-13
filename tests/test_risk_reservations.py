"""Pending-order risk reservations are durable, atomic, and idempotent."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from portfolio.risk_reservations import (
    RiskCapacityExceeded,
    RiskReservationLedger,
)


def _risk(gross=60.0):
    return {
        "gross": gross,
        "cash_debit": gross / 2,
        "per_name": {"SPY": gross},
        "sector": {"ETF": gross},
        "option": {"delta": gross / 4, "vega": gross / 10},
    }


def _capacity(gross=100.0):
    return {
        "gross": gross,
        "cash_debit": gross,
        "per_name": {"*": gross},
        "sector": {"ETF": gross},
        "option": {"delta": gross, "vega": gross},
    }


@pytest.fixture
def ledger(tmp_path):
    return RiskReservationLedger(
        str(tmp_path / "risk.db"), env="sandbox", account_id_key="ACC-1"
    )


def _reserve(ledger, reservation_id="intent-1", **overrides):
    arguments = {
        "risk": _risk(),
        "capacity": _capacity(),
        "units": 10,
        "instrument_type": "EQ",
        "order_ids": ["order-1"],
        "metadata": {"strategy": "test"},
    }
    arguments.update(overrides)
    return ledger.check_and_reserve(reservation_id, **arguments)


class TestAtomicCapacity:
    def test_reserve_is_idempotent_and_conflicting_intent_fails(self, ledger):
        first = _reserve(ledger)
        second = _reserve(ledger)

        assert second["reservation_id"] == first["reservation_id"]
        assert second["created_at"] == first["created_at"]
        assert ledger.active_totals() == _risk()

        with pytest.raises(ValueError, match="different intent"):
            _reserve(ledger, risk=_risk(50.0))

    @pytest.mark.parametrize(
        ("risk", "capacity", "dimension", "key"),
        [
            (_risk(60), _capacity(50), "gross", None),
            (
                {"per_name": {"SPY": 60}},
                {"per_name": {"*": 50}},
                "per_name",
                "SPY",
            ),
            (
                {"sector": {"ETF": 60}},
                {"sector": {"ETF": 50}},
                "sector",
                "ETF",
            ),
            (
                {"option": {"short_gamma": 60}},
                {"option": {"short_gamma": 50}},
                "option",
                "short_gamma",
            ),
        ],
    )
    def test_each_configured_dimension_can_block(
        self, ledger, risk, capacity, dimension, key
    ):
        with pytest.raises(RiskCapacityExceeded) as error:
            ledger.check_and_reserve(
                "blocked",
                risk,
                capacity,
                units=1,
                instrument_type="SPREADS",
            )
        assert error.value.breaches[0]["dimension"] == dimension
        assert error.value.breaches[0]["key"] == key
        assert ledger.reservation("blocked") is None

    def test_base_usage_and_existing_reservations_are_both_counted(self, ledger):
        _reserve(ledger, risk=_risk(40), order_ids=[])
        with pytest.raises(RiskCapacityExceeded) as error:
            ledger.check_and_reserve(
                "intent-2",
                _risk(20),
                _capacity(100),
                base_usage=_risk(50),
                units=1,
                instrument_type="OPTN",
            )
        gross = next(
            breach for breach in error.value.breaches if breach["dimension"] == "gross"
        )
        assert gross == {
            "dimension": "gross",
            "key": None,
            "used": 90.0,
            "requested": 20.0,
            "projected": 110.0,
            "capacity": 100.0,
        }

    def test_concurrent_submitters_cannot_both_claim_capacity(self, tmp_path):
        db = str(tmp_path / "concurrent.db")
        RiskReservationLedger(db, "sandbox", "ACC")

        def submit(intent_id):
            candidate = RiskReservationLedger(db, "sandbox", "ACC")
            try:
                candidate.check_and_reserve(
                    intent_id,
                    {"gross": 60},
                    {"gross": 100},
                    units=1,
                    instrument_type="EQ",
                )
            except RiskCapacityExceeded:
                return "blocked"
            return "reserved"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, ("first", "second")))

        assert sorted(results) == ["blocked", "reserved"]
        assert RiskReservationLedger(db, "sandbox", "ACC").active_totals()["gross"] == 60


class TestLifecycle:
    def test_reducing_intent_never_consumes_opening_capacity(self, ledger):
        result = _reserve(
            ledger,
            risk=_risk(1_000_000),
            capacity={"gross": 0, "cash_debit": 0},
            reducing=True,
        )
        assert result["reducing"] is True
        assert result["remaining_risk"] == {
            "gross": 0.0,
            "cash_debit": 0.0,
            "per_name": {},
            "sector": {},
            "option": {},
        }
        assert ledger.active_totals()["gross"] == 0

    def test_cumulative_partial_fills_decrement_once_and_survive_restart(
        self, ledger
    ):
        _reserve(ledger)
        first = ledger.record_order_update(
            "order-1", cumulative_filled_units=2, status="PARTIAL"
        )
        retried = ledger.record_order_update(
            "order-1", cumulative_filled_units=2, status="PARTIAL"
        )
        assert retried["remaining_risk"] == first["remaining_risk"]
        assert retried["remaining_risk"]["gross"] == pytest.approx(48)

        restarted = RiskReservationLedger(
            ledger.db_path, ledger.env, ledger.account_id_key
        )
        assert restarted.active_totals()["gross"] == pytest.approx(48)
        assert restarted.reservation_for_order("order-1")["orders"][0][
            "cumulative_filled_units"
        ] == 2

    def test_replacement_shares_reservation_and_old_cancel_does_not_release(
        self, ledger
    ):
        _reserve(ledger)
        ledger.record_order_update(
            "order-1", cumulative_filled_units=3, status="PARTIAL"
        )
        bound = ledger.bind_order(
            "intent-1", "order-2", replaces_order_id="order-1"
        )
        assert [order["status"] for order in bound["orders"]] == [
            "REPLACED",
            "PENDING",
        ]

        ledger.record_order_update(
            "order-1", cumulative_filled_units=3, status="CANCELLED"
        )
        after_successor_fill = ledger.record_order_update(
            "order-2", cumulative_filled_units=2, status="PARTIAL"
        )
        assert after_successor_fill["status"] == "ACTIVE"
        assert after_successor_fill["remaining_risk"]["gross"] == pytest.approx(30)

        cancelled = ledger.record_order_update(
            "order-2", cumulative_filled_units=2, status="CANCELED"
        )
        assert cancelled["status"] == "RELEASED"
        assert cancelled["remaining_risk"]["gross"] == 0
        assert ledger.active_totals()["gross"] == 0

    @pytest.mark.parametrize(
        "status", ["CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "EXECUTED"]
    )
    def test_terminal_nonfill_status_releases(self, ledger, status):
        _reserve(ledger)
        result = ledger.record_order_update(
            "order-1", cumulative_filled_units=0, status=status
        )
        assert result["status"] == "RELEASED"
        assert result["release_reason"] == f"order_{status.lower()}"

    def test_filled_status_releases_even_at_rounding_boundary(self, ledger):
        _reserve(ledger)
        result = ledger.record_order_update(
            "order-1", cumulative_filled_units=9.9999999995, status="FILLED"
        )
        assert result["status"] == "FILLED"
        assert result["remaining_risk"]["gross"] == 0

    def test_manual_release_is_idempotent(self, ledger):
        _reserve(ledger, order_ids=[])
        released = ledger.release("intent-1", "submit_failed")
        again = ledger.release("intent-1", "different_reason")
        assert released["release_reason"] == "submit_failed"
        assert again["release_reason"] == "submit_failed"
        assert ledger.active_reservations() == []

    def test_unknown_bindings_and_regressing_counters_fail_loudly(self, ledger):
        _reserve(ledger)
        with pytest.raises(KeyError, match="unknown order_id"):
            ledger.record_order_update(
                "missing", cumulative_filled_units=0, status="OPEN"
            )
        ledger.record_order_update(
            "order-1", cumulative_filled_units=2, status="PARTIAL"
        )
        with pytest.raises(ValueError, match="cannot decrease"):
            ledger.record_order_update(
                "order-1", cumulative_filled_units=1, status="PARTIAL"
            )


class TestRevaluation:
    def test_price_increase_breach_rolls_back_original_risk(self, ledger):
        _reserve(
            ledger,
            "first",
            risk=_risk(40),
            order_ids=["first-order"],
        )
        _reserve(
            ledger,
            "second",
            risk=_risk(40),
            order_ids=["second-order"],
        )
        before = ledger.reservation("first")

        with pytest.raises(RiskCapacityExceeded) as error:
            ledger.revalue("first", _risk(70), _capacity(100))

        assert any(
            breach["dimension"] == "gross" and breach["projected"] == 110
            for breach in error.value.breaches
        )
        after = ledger.reservation("first")
        assert after == before
        assert ledger.active_totals()["gross"] == 80

    def test_success_scales_by_all_child_fills_and_is_idempotent(self, ledger):
        _reserve(ledger)
        ledger.record_order_update(
            "order-1", cumulative_filled_units=3, status="PARTIAL"
        )
        ledger.bind_order("intent-1", "order-2", replaces_order_id="order-1")
        ledger.record_order_update(
            "order-1", cumulative_filled_units=3, status="CANCELLED"
        )
        ledger.record_order_update(
            "order-2", cumulative_filled_units=2, status="PARTIAL"
        )

        first = ledger.revalue("intent-1", _risk(100), _capacity(200))
        again = ledger.revalue("intent-1", _risk(100), _capacity(200))

        assert first["requested_risk"] == _risk(100)
        assert first["remaining_risk"]["gross"] == pytest.approx(50)
        assert first["remaining_risk"]["cash_debit"] == pytest.approx(25)
        assert again == first
        assert ledger.active_totals()["gross"] == pytest.approx(50)

        # Revaluation advances the intent fingerprint: retrying the same
        # current-price payload is idempotent and cannot add a second row.
        retried_submit = _reserve(ledger, risk=_risk(100))
        assert retried_submit == first
        assert len(ledger.active_reservations()) == 1


class TestScopingAndValidation:
    @pytest.mark.parametrize("instrument_type", ["EQ", "OPTN", "SPREADS"])
    def test_instrument_types_share_the_generic_contract(
        self, ledger, instrument_type
    ):
        result = _reserve(ledger, instrument_type=instrument_type)
        assert result["instrument_type"] == instrument_type

    def test_environment_and_account_are_isolated(self, tmp_path):
        db = str(tmp_path / "scoped.db")
        first = RiskReservationLedger(db, "sandbox", "ACC-1")
        other_account = RiskReservationLedger(db, "sandbox", "ACC-2")
        production = RiskReservationLedger(db, "production", "ACC-1")
        _reserve(first)
        assert other_account.active_totals()["gross"] == 0
        assert production.active_totals()["gross"] == 0

    @pytest.mark.parametrize(
        ("risk", "match"),
        [
            ({"gross": float("nan")}, "finite non-negative"),
            ({"cash_debit": -1}, "finite non-negative"),
            ({"per_name": {"SPY": float("inf")}}, "finite non-negative"),
            ({"unknown": 1}, "unknown dimensions"),
        ],
    )
    def test_invalid_risk_is_rejected(self, ledger, risk, match):
        with pytest.raises(ValueError, match=match):
            ledger.check_and_reserve(
                "bad", risk, {}, units=1, instrument_type="EQ"
            )

    def test_metadata_and_snapshots_are_json_safe(self, ledger):
        with pytest.raises(ValueError, match="JSON-safe"):
            _reserve(ledger, metadata={"bad": object()})

        _reserve(ledger)
        snapshot = ledger.snapshot()
        encoded = json.dumps(snapshot, allow_nan=False, sort_keys=True)
        assert '"account_id_key": "ACC-1"' in encoded
        assert snapshot["reservations"][0]["orders"][0]["order_id"] == "order-1"

    def test_order_id_cannot_bind_to_two_intents(self, ledger):
        _reserve(ledger)
        with pytest.raises(ValueError, match="different reservation"):
            _reserve(
                ledger,
                "intent-2",
                risk=_risk(1),
                order_ids=["order-1"],
            )
