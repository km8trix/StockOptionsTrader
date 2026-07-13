"""Live order risk is freshly valued and durably reserved."""

from __future__ import annotations

from copy import deepcopy

import pytest

from brokers.etrade_client import (
    build_equity_order,
    build_option_order,
    build_spread_order,
)
from brokers.local_book import LocalBook
from execution.live_risk_gate import (
    LiveRiskGate,
    LiveRiskPolicy,
    ReservationPersistenceError,
    RiskValuationUnavailable,
    UnsupportedRiskOrder,
)
from portfolio.risk_reservations import (
    RiskCapacityExceeded,
    RiskReservationLedger,
)


ACCOUNT = "ACC-1"


class FakeClient:
    def __init__(self):
        self.balances = {
            "Computed": {
                "cashAvailableForInvestment": 100_000.0,
                "RealTimeValues": {"totalAccountValue": 100_000.0},
            }
        }
        self.positions = []
        self.quotes = {
            "AAPL": {"bid": 99.0, "ask": 101.0, "last": 100.0},
            "MSFT": {"bid": 199.0, "ask": 201.0, "last": 200.0},
        }
        self.calls = []

    def get_balances(self, account_id_key):
        self.calls.append(("balances", account_id_key))
        return deepcopy(self.balances)

    def get_portfolio(self, account_id_key):
        self.calls.append(("portfolio", account_id_key))
        return deepcopy(self.positions)

    def get_quotes(self, symbols):
        self.calls.append(("quotes", tuple(symbols)))
        return {
            symbol: deepcopy(self.quotes[symbol]) for symbol in symbols if symbol in self.quotes
        }


class FakeKillSwitch:
    def __init__(self):
        self.calls = []

    def engage(self, reason, actor):
        self.calls.append({"reason": reason, "actor": actor})


class FakeAudit:
    def __init__(self):
        self.events = []

    def append(self, actor, event_type, payload):
        self.events.append((actor, event_type, deepcopy(payload)))
        return len(self.events)


@pytest.fixture
def components(tmp_path):
    client = FakeClient()
    ledger = RiskReservationLedger(str(tmp_path / "risk.db"), "sandbox", ACCOUNT)
    kill = FakeKillSwitch()
    audit = FakeAudit()
    gate = LiveRiskGate(client, ledger, kill, audit, ACCOUNT)
    return gate, client, ledger, kill, audit


def _long_equity(symbol="AAPL", quantity=10):
    return {
        "positionId": 1,
        "quantity": quantity,
        "positionType": "LONG",
        "marketValue": quantity * 100.0,
        "Product": {"symbol": symbol, "securityType": "EQ"},
    }


def _option_position(strike, quantity, *, position_type=None):
    position = {
        "quantity": quantity,
        "Product": {
            "symbol": "AAPL",
            "securityType": "OPTN",
            "callPut": "PUT",
            "strikePrice": strike,
            "expiryYear": 2026,
            "expiryMonth": 7,
            "expiryDay": 17,
        },
    }
    if position_type is not None:
        position["positionType"] = position_type
    return position


class TestEstimateAndValuation:
    def test_construction_is_network_free_and_estimate_does_not_reserve(self, components):
        gate, client, ledger, _, _ = components
        assert client.calls == []

        result = gate.estimate(build_equity_order("AAPL", "BUY", 10, 99.0, "estimate1"))

        assert result["status"] == "estimated"
        assert result["revalidated_on_place"] is True
        assert result["policy"]["market_buy_slippage_fraction"] == 0.01
        assert result["risk"]["cash_debit"] == 990.0
        # Exposure is conservative at the fresh mark when a buy limit is lower.
        assert result["risk"]["gross"] == 1_000.0
        assert result["valuation"]["fresh_mark"] == 100.0
        assert ledger.active_reservations() == []
        assert [call[0] for call in client.calls] == ["balances", "portfolio", "quotes"]

    def test_market_buy_uses_fresh_ask_plus_policy_slippage(self, components):
        gate, client, ledger, _, _ = components
        request = build_equity_order("AAPL", "BUY", 10, None, "market1")

        handle = gate.begin(ACCOUNT, request)
        assert ledger.reservation(handle.reservation_id)["requested_risk"][
            "cash_debit"
        ] == pytest.approx(1_020.10)

        client.quotes["AAPL"]["ask"] = 111.0
        client.quotes["AAPL"]["last"] = 110.0
        gate.before_attempt(handle, 1)

        assert ledger.reservation(handle.reservation_id)["requested_risk"][
            "gross"
        ] == pytest.approx(1_121.10)
        assert [call[0] for call in client.calls] == [
            "balances",
            "portfolio",
            "quotes",
            "balances",
            "portfolio",
            "quotes",
        ]

    def test_long_limit_option_uses_premium_and_full_underlying_notional(self, components):
        gate, client, _, _, _ = components
        client.balances["Computed"]["RealTimeValues"]["totalAccountValue"] = 1_000_000.0
        request = build_option_order(
            "AAPL", "CALL", 105.0, "2026-07-17", "BUY_OPEN", 2, 2.50, "option1"
        )

        result = gate.estimate(request)

        assert result["risk"] == {
            "gross": 20_000.0,
            "cash_debit": 500.0,
            "per_name": {"AAPL": 20_000.0},
            "sector": {},
            "option": {"premium": 500.0, "max_loss": 500.0},
        }
        assert result["capacity"]["option"] == {
            "premium": 100_000.0,
            "max_loss": 100_000.0,
        }

    def test_missing_fresh_data_and_optional_sector_fail_closed(self, components):
        gate, client, ledger, kill, audit = components
        client.quotes = {}
        with pytest.raises(RiskValuationUnavailable, match="quote unavailable"):
            gate.estimate(build_equity_order("AAPL", "BUY", 1, 100.0, "missing1"))

        sector_gate = LiveRiskGate(
            client,
            ledger,
            kill,
            audit,
            ACCOUNT,
            policy=LiveRiskPolicy(sector_nav_fraction=0.25),
            sector_resolver=lambda _symbol: None,
        )
        client.quotes["AAPL"] = {"bid": 99.0, "ask": 101.0, "last": 100.0}
        with pytest.raises(RiskValuationUnavailable, match="sector classification"):
            sector_gate.estimate(build_equity_order("AAPL", "BUY", 1, 100.0, "sector1"))


class TestCapacityAndSupportedOrders:
    def test_overlapping_pending_orders_share_atomic_capacity(self, components):
        gate, _, ledger, _, _ = components
        first = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 60, 100.0, "overlap1"))
        assert ledger.reservation(first.reservation_id)["status"] == "ACTIVE"

        with pytest.raises(RiskCapacityExceeded) as error:
            gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 50, 100.0, "overlap2"))

        assert any(breach["dimension"] == "per_name" for breach in error.value.breaches)
        assert ledger.reservation("overlap2") is None

    def test_verified_pure_equity_close_is_reducing(self, components):
        gate, client, ledger, _, _ = components
        client.positions = [_long_equity(quantity=10)]
        handle = gate.begin(ACCOUNT, build_equity_order("AAPL", "SELL", 7, 100.0, "close1"))

        reservation = ledger.reservation(handle.reservation_id)
        assert reservation["reducing"] is True
        assert reservation["remaining_risk"]["gross"] == 0

        with pytest.raises(UnsupportedRiskOrder, match="shorts"):
            gate.estimate(build_equity_order("AAPL", "SELL", 11, 100.0, "short1"))

        client.positions = [{**_long_equity(quantity=-4), "positionType": "SHORT"}]
        buy_to_cover = gate.estimate(build_equity_order("AAPL", "BUY", 4, 100.0, "cover1"))
        assert buy_to_cover["reducing"] is True
        assert buy_to_cover["risk"]["cash_debit"] == 0

    @pytest.mark.parametrize(
        "order_request, match",
        [
            (
                build_equity_order("AAPL", "SELL", 1, 100.0, "short2"),
                "shorts",
            ),
            (
                build_option_order(
                    "AAPL",
                    "CALL",
                    105,
                    "2026-07-17",
                    "BUY_OPEN",
                    1,
                    None,
                    "marketopt1",
                ),
                "market option",
            ),
            (
                build_option_order(
                    "AAPL",
                    "CALL",
                    105,
                    "2026-07-17",
                    "SELL_OPEN",
                    1,
                    2.0,
                    "naked1",
                ),
                "shorts and closes",
            ),
        ],
    )
    def test_unsupported_exposure_fails_closed(self, components, order_request, match):
        gate, _, ledger, _, _ = components
        with pytest.raises(UnsupportedRiskOrder, match=match):
            gate.estimate(order_request)
        assert ledger.active_reservations() == []


class TestAtomicSpreadRisk:
    @staticmethod
    def _vertical(
        *,
        quantity=1,
        net_price=1.0,
        short_strike=105,
        long_strike=100,
        right="PUT",
        client_order_id="spread1",
    ):
        return build_spread_order(
            [
                {
                    "symbol": "AAPL",
                    "call_put": right,
                    "strike": short_strike,
                    "expiry": "2026-07-17",
                    "action": "SELL_OPEN",
                    "quantity": quantity,
                },
                {
                    "symbol": "AAPL",
                    "call_put": right,
                    "strike": long_strike,
                    "expiry": "2026-07-17",
                    "action": "BUY_OPEN",
                    "quantity": quantity,
                },
            ],
            net_price,
            client_order_id,
        )

    def test_credit_vertical_uses_defined_max_loss_for_all_capacity_vectors(self, components):
        gate, _, _, _, _ = components

        result = gate.estimate(self._vertical(quantity=3, net_price=1.20))

        # ($5 width - $1.20 credit) * 100 * 3 packages.
        assert result["instrument_type"] == "SPREADS"
        assert result["units"] == 3
        assert result["risk"] == {
            "gross": pytest.approx(1_140.0),
            "cash_debit": 0.0,
            "per_name": {"AAPL": pytest.approx(1_140.0)},
            "sector": {},
            "option": {"premium": 0.0, "max_loss": pytest.approx(1_140.0)},
        }
        assert result["valuation"]["valuation_price"] == 1.20
        assert result["valuation"]["fresh_mark"] == 100.0
        assert result["valuation"]["action"] == "NET_CREDIT"

    def test_debit_vertical_reserves_debit_as_max_loss_and_cash(self, components):
        gate, _, _, _, _ = components
        request = build_spread_order(
            [
                {
                    "symbol": "AAPL",
                    "call_put": "CALL",
                    "strike": 105,
                    "expiry": "2026-07-17",
                    "action": "BUY_OPEN",
                    "quantity": 2,
                },
                {
                    "symbol": "AAPL",
                    "call_put": "CALL",
                    "strike": 110,
                    "expiry": "2026-07-17",
                    "action": "SELL_OPEN",
                    "quantity": 2,
                },
            ],
            -1.25,
            "debitspread1",
        )

        result = gate.estimate(request)

        assert result["risk"] == {
            "gross": 250.0,
            "cash_debit": 250.0,
            "per_name": {"AAPL": 250.0},
            "sector": {},
            "option": {"premium": 250.0, "max_loss": 250.0},
        }

    def test_iron_condor_is_valued_as_one_package_not_four_naked_legs(self, components):
        gate, _, _, _, _ = components
        request = build_spread_order(
            [
                {
                    "symbol": "AAPL",
                    "call_put": "PUT",
                    "strike": 95,
                    "expiry": "2026-07-17",
                    "action": "SELL_OPEN",
                    "quantity": 2,
                },
                {
                    "symbol": "AAPL",
                    "call_put": "PUT",
                    "strike": 90,
                    "expiry": "2026-07-17",
                    "action": "BUY_OPEN",
                    "quantity": 2,
                },
                {
                    "symbol": "AAPL",
                    "call_put": "CALL",
                    "strike": 105,
                    "expiry": "2026-07-17",
                    "action": "SELL_OPEN",
                    "quantity": 2,
                },
                {
                    "symbol": "AAPL",
                    "call_put": "CALL",
                    "strike": 110,
                    "expiry": "2026-07-17",
                    "action": "BUY_OPEN",
                    "quantity": 2,
                },
            ],
            1.50,
            "condor1",
        )

        result = gate.estimate(request)

        assert result["units"] == 2
        assert result["risk"]["gross"] == pytest.approx(700.0)
        assert result["risk"]["option"]["max_loss"] == pytest.approx(700.0)

    @pytest.mark.parametrize(
        "mutation, match",
        [
            (
                lambda request: request["Order"][0]["Instrument"][1].update({"quantity": 2}),
                "quantity and orderedQuantity must match",
            ),
            (
                lambda request: request["Order"][0]["Instrument"][1]["Product"].update(
                    {"symbol": "MSFT"}
                ),
                "one underlying symbol",
            ),
            (
                lambda request: request["Order"][0]["Instrument"][1]["Product"].update(
                    {"expiryDay": 18}
                ),
                "calendar and diagonal",
            ),
            (
                lambda request: request["Order"][0]["Instrument"][0].update(
                    {"orderAction": "BUY_CLOSE"}
                ),
                "mixed open/close",
            ),
            (
                lambda request: request["Order"][0]["Instrument"][1].update(
                    {"orderAction": "SELL_OPEN"}
                ),
                "naked short put",
            ),
        ],
    )
    def test_ambiguous_or_naked_packages_fail_closed(self, components, mutation, match):
        gate, _, ledger, _, _ = components
        request = self._vertical()
        mutation(request)

        with pytest.raises(UnsupportedRiskOrder, match=match):
            gate.begin(ACCOUNT, request)

        assert ledger.active_reservations() == []

    def test_fractional_leg_quantity_fails_closed(self, components):
        gate, _, ledger, _, _ = components
        request = self._vertical()
        request["Order"][0]["Instrument"][0].update({"quantity": 1.5, "orderedQuantity": 1.5})

        with pytest.raises(UnsupportedRiskOrder, match="whole number"):
            gate.begin(ACCOUNT, request)

        assert ledger.active_reservations() == []

    def test_package_capacity_is_enforced_atomically(self, components):
        gate, _, ledger, _, _ = components
        request = self._vertical(
            short_strike=305,
            long_strike=100,
            net_price=1.0,
            client_order_id="too-wide",
        )

        with pytest.raises(RiskCapacityExceeded) as error:
            gate.begin(ACCOUNT, request)

        assert {breach["dimension"] for breach in error.value.breaches} >= {"per_name", "option"}
        assert ledger.reservation("too-wide") is None

    def test_partial_fill_scales_single_package_reservation_and_local_book(self, components):
        _, client, ledger, kill, audit = components
        book = LocalBook(ledger.db_path, env=ledger.env, account_id_key=ledger.account_id_key)
        assert book.bootstrap({}, 100_000.0) is True
        gate = LiveRiskGate(client, ledger, kill, audit, ACCOUNT, local_book=book)
        request = self._vertical(quantity=4, net_price=1.0, client_order_id="package-partial")
        handle = gate.begin(ACCOUNT, request)
        gate.accepted(handle, {"order_id": "spread-order"})

        gate.order_update(
            "spread-order",
            {"status": "PARTIAL", "filled_quantity": 1, "avg_fill_price": -1.0},
        )

        reservation = ledger.reservation("package-partial")
        assert reservation["units"] == 4
        assert len(reservation["orders"]) == 1
        assert reservation["remaining_risk"]["gross"] == pytest.approx(1_200.0)
        assert ledger.active_totals()["gross"] == pytest.approx(1_200.0)
        assert book.positions() == {
            "AAPL 2026-07-17 $100.0 put": 1.0,
            "AAPL 2026-07-17 $105.0 put": -1.0,
        }
        assert book.cash() == pytest.approx(100_100.0)

    def test_ratio_package_uses_gcd_units_and_ratio_weighted_max_loss(self, components):
        gate, _, ledger, _, _ = components
        request = build_spread_order(
            [
                {
                    "symbol": "AAPL",
                    "call_put": "CALL",
                    "strike": 105,
                    "expiry": "2026-07-17",
                    "action": "SELL_OPEN",
                    "quantity": 4,
                },
                {
                    "symbol": "AAPL",
                    "call_put": "CALL",
                    "strike": 110,
                    "expiry": "2026-07-17",
                    "action": "BUY_OPEN",
                    "quantity": 6,
                },
            ],
            1.0,
            "ratio-package",
        )

        handle = gate.begin(ACCOUNT, request)
        gate.accepted(handle, {"order_id": "ratio-order"})
        gate.order_update("ratio-order", {"status": "PARTIAL", "filled_quantity": 1})

        # Quantities 4:6 are two packages of ratio 2:3.  At S=110 the
        # two short calls lose $10 less the $1 credit: $900 per package.
        reservation = ledger.reservation("ratio-package")
        assert reservation["units"] == 2
        assert reservation["requested_risk"]["gross"] == pytest.approx(1_800.0)
        assert reservation["remaining_risk"]["gross"] == pytest.approx(900.0)
        assert reservation["remaining_risk"]["option"]["max_loss"] == pytest.approx(900.0)

    def test_ratio_with_more_short_than_hedge_fails_closed(self, components):
        gate, _, ledger, _, _ = components
        request = self._vertical(quantity=1, client_order_id="naked-ratio")
        request["Order"][0]["Instrument"][0].update({"quantity": 2, "orderedQuantity": 2})

        with pytest.raises(UnsupportedRiskOrder, match="naked short put"):
            gate.begin(ACCOUNT, request)

        assert ledger.active_reservations() == []

    @staticmethod
    def _close_vertical(*, quantity=2, client_order_id="close-spread"):
        return build_spread_order(
            [
                {
                    "symbol": "AAPL",
                    "call_put": "PUT",
                    "strike": 105,
                    "expiry": "2026-07-17",
                    "action": "BUY_CLOSE",
                    "quantity": quantity,
                },
                {
                    "symbol": "AAPL",
                    "call_put": "PUT",
                    "strike": 100,
                    "expiry": "2026-07-17",
                    "action": "SELL_CLOSE",
                    "quantity": quantity,
                },
            ],
            -0.75,
            client_order_id,
        )

    def test_verified_pure_close_is_reducing_and_bypasses_opening_rails(self, components):
        gate, client, ledger, _, _ = components
        client.positions = [
            _option_position(105, 2, position_type="SHORT"),
            _option_position(100, 2, position_type="LONG"),
        ]
        # Neither opening-cap input is needed to prove a reduction.
        client.balances = {}
        client.quotes = {}

        estimate = gate.estimate(self._close_vertical())
        handle = gate.begin(ACCOUNT, self._close_vertical())

        assert estimate["reducing"] is True
        assert estimate["risk"] == {
            "gross": 0.0,
            "cash_debit": 0.0,
            "per_name": {},
            "sector": {},
            "option": {},
        }
        assert estimate["capacity"] == {}
        assert estimate["base_usage"] == {
            "gross": 0.0,
            "cash_debit": 0.0,
            "per_name": {},
            "sector": {},
            "option": {},
        }
        assert estimate["valuation"]["nav"] is None
        assert estimate["valuation"]["cash_available"] is None
        assert estimate["valuation"]["fresh_mark"] is None
        assert client.calls == [("portfolio", ACCOUNT), ("portfolio", ACCOUNT)]
        reservation = ledger.reservation(handle.reservation_id)
        assert reservation["instrument_type"] == "SPREADS"
        assert reservation["reducing"] is True
        assert reservation["remaining_risk"]["gross"] == 0.0

    def test_ratio_close_verifies_full_leg_ratios_and_reserves_gcd_units(self, components):
        gate, client, ledger, _, _ = components
        client.positions = [_option_position(105, -4), _option_position(100, 6)]
        request = self._close_vertical(quantity=1, client_order_id="ratio-close")
        request["Order"][0]["Instrument"][0].update({"quantity": 4, "orderedQuantity": 4})
        request["Order"][0]["Instrument"][1].update({"quantity": 6, "orderedQuantity": 6})

        handle = gate.begin(ACCOUNT, request)

        reservation = ledger.reservation(handle.reservation_id)
        assert reservation["units"] == 2
        assert reservation["reducing"] is True

    @pytest.mark.parametrize(
        "positions",
        [
            [
                _option_position(105, -2),
            ],
            [
                _option_position(105, -2),
                _option_position(100, 1),
            ],
            [
                _option_position(105, 2),
                _option_position(100, 2),
            ],
        ],
        ids=["missing-leg", "insufficient-leg", "wrong-direction"],
    )
    def test_close_requires_every_fresh_broker_leg_in_sufficient_direction(
        self, components, positions
    ):
        gate, client, ledger, _, _ = components
        client.positions = positions

        with pytest.raises(UnsupportedRiskOrder, match="verified broker position.*insufficient"):
            gate.begin(ACCOUNT, self._close_vertical())

        assert ledger.active_reservations() == []


class TestLifecycleHooks:
    def test_tracked_fill_books_before_reservation_release(self, components):
        _, client, ledger, kill, audit = components
        book = LocalBook(ledger.db_path, env=ledger.env, account_id_key=ledger.account_id_key)
        assert book.bootstrap({}, 100_000.0) is True
        gate = LiveRiskGate(client, ledger, kill, audit, ACCOUNT, local_book=book)
        request = build_equity_order("AAPL", "BUY", 2, 100.0, "booked1")
        handle = gate.begin(ACCOUNT, request)
        gate.accepted(handle, {"order_id": "book-order"})

        partial = {"status": "PARTIAL", "filled_quantity": 1, "avg_fill_price": 100.0}
        gate.order_update("book-order", partial)
        gate.order_update("book-order", partial)
        assert book.positions() == {"AAPL": 1.0}
        assert ledger.reservation("booked1")["remaining_risk"]["gross"] == pytest.approx(100.0)

        gate.order_update(
            "book-order",
            {
                "status": "EXECUTED",
                "filled_quantity": 2,
                "avg_fill_price": 100.0,
            },
        )
        assert book.positions() == {"AAPL": 2.0}
        assert book.cash() == pytest.approx(99_800.0)
        assert ledger.reservation("booked1")["status"] == "RELEASED"

    def test_accept_partial_fill_and_terminal_update(self, components):
        gate, _, ledger, _, _ = components
        handle = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 10, 100.0, "lifecycle1"))
        gate.accepted(handle, {"order_id": 700})

        gate.order_update("700", {"status": "PARTIAL", "filled_quantity": 4})
        after_partial = ledger.reservation(handle.reservation_id)
        assert after_partial["remaining_risk"]["gross"] == pytest.approx(600.0)

        gate.order_update("700", {"status": "EXECUTED", "filled_quantity": 10})
        terminal = ledger.reservation(handle.reservation_id)
        assert terminal["status"] == "RELEASED"
        assert terminal["remaining_risk"]["gross"] == 0

    def test_rejected_releases_but_unknown_retains_and_stops(self, components):
        gate, _, ledger, kill, _ = components
        rejected = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 1, 100.0, "rejected1"))
        gate.rejected(rejected, RuntimeError("broker said no"))
        assert ledger.reservation("rejected1")["status"] == "RELEASED"

        unknown = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 1, 100.0, "unknown1"))
        gate.unknown(unknown, TimeoutError("lookup failed"))
        assert ledger.reservation("unknown1")["status"] == "ACTIVE"
        assert kill.calls[-1]["actor"] == "live_risk_gate"
        assert "unknown1" in kill.calls[-1]["reason"]

        # begin failures have no handle and own their cleanup themselves.
        gate.rejected(None, RuntimeError("begin failed"))
        assert ledger.reservation("unknown1")["status"] == "ACTIVE"

    def test_accept_without_order_id_is_unknown_and_retains(self, components):
        gate, _, ledger, kill, _ = components
        handle = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 1, 100.0, "bindfail1"))

        with pytest.raises(ReservationPersistenceError, match="no broker order_id"):
            gate.accepted(handle, {})

        assert ledger.reservation(handle.reservation_id)["status"] == "ACTIVE"
        assert kill.calls

    def test_unbound_update_is_ignored_but_bound_invariant_errors_propagate(self, components):
        gate, _, _, _, _ = components
        gate.order_update("old-untracked", {"status": "FILLED", "filled_quantity": 1})

        handle = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 2, 100.0, "updates1"))
        gate.accepted(handle, {"order_id": "701"})
        gate.order_update("701", {"status": "PARTIAL", "filled_quantity": 1})
        with pytest.raises(ValueError, match="cannot decrease"):
            gate.order_update("701", {"status": "PARTIAL", "filled_quantity": 0})

    def test_snapshot_and_bounded_recovery(self, components):
        gate, _, ledger, _, _ = components
        handle = gate.begin(ACCOUNT, build_equity_order("AAPL", "BUY", 2, 100.0, "recover1"))
        gate.accepted(handle, {"order_id": "702"})

        assert (
            gate.recover_order_updates(
                [{"order_id": "702", "status": "PARTIAL", "filled_quantity": 1}]
            )
            == 1
        )
        assert gate.snapshot() == ledger.snapshot()
        with pytest.raises(ValueError, match="safety bound"):
            gate.recover_order_updates(
                [{"order_id": "old", "status": "OPEN", "filled_quantity": 0}],
                limit=0,
            )
