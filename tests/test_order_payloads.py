"""GOLDEN-FIXTURE payload tests: iron condor + put credit spread preview
payloads byte-compared (sorted-key JSON) against checked-in expectations.
Any drift in the order JSON we send a real broker breaks these loudly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brokers.etrade_client import (build_equity_order, build_option_order,
                                   build_spread_order)

FIXTURES = Path(__file__).parent / "fixtures"


def golden_bytes(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


class TestGoldenSpreads:
    def test_iron_condor_exact_payload(self):
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 2},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 2},
            {"symbol": "SPY", "call_put": "CALL", "strike": 480,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 2},
            {"symbol": "SPY", "call_put": "CALL", "strike": 490,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 2},
        ]
        payload = {"PreviewOrderRequest": build_spread_order(
            legs, net_price=2.40, client_order_id="GOLDENCONDOR0001")}
        expected = (FIXTURES / "iron_condor_preview.json").read_text()
        assert golden_bytes(payload) == expected
        # Belt and braces on the structural facts the fixture encodes:
        order = payload["PreviewOrderRequest"]["Order"][0]
        instruments = order["Instrument"]
        assert len(instruments) == 4
        assert all(i["Product"]["securityType"] == "OPTN"
                   for i in instruments)
        assert order["priceType"] == "NET_CREDIT"
        assert order["limitPrice"] == 2.4
        assert [i["orderAction"] for i in instruments] == [
            "SELL_OPEN", "BUY_OPEN", "SELL_OPEN", "BUY_OPEN"]
        # Each option Instrument carries BOTH 'orderedQuantity' AND
        # 'quantity' — E*TRADE's documented Preview Order example sends
        # the pair; 'quantity' may be the authoritative request field.
        assert all(i["orderedQuantity"] == 2 and i["quantity"] == 2
                   for i in instruments)
        assert instruments[0]["Product"]["expiryYear"] == 2026
        assert instruments[0]["Product"]["expiryMonth"] == 7
        assert instruments[0]["Product"]["expiryDay"] == 17

    def test_put_credit_spread_exact_payload(self):
        legs = [
            {"symbol": "IWM", "call_put": "PUT", "strike": 195,
             "expiry": "2026-01-16", "action": "SELL_OPEN", "quantity": 3},
            {"symbol": "IWM", "call_put": "PUT", "strike": 190,
             "expiry": "2026-01-16", "action": "BUY_OPEN", "quantity": 3},
        ]
        payload = {"PreviewOrderRequest": build_spread_order(
            legs, net_price=1.15, client_order_id="GOLDENPUTCS00001")}
        expected = (FIXTURES / "put_credit_spread_preview.json").read_text()
        assert golden_bytes(payload) == expected


class TestSignConventions:
    def test_negative_net_price_is_net_debit(self):
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "BUY_CLOSE", "quantity": 2},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "SELL_CLOSE", "quantity": 2},
        ]
        # Closing the credit spread PAYS: signed net is negative, payload
        # carries NET_DEBIT with the positive magnitude.
        order = build_spread_order(legs, net_price=-0.55)["Order"][0]
        assert order["priceType"] == "NET_DEBIT"
        assert order["limitPrice"] == 0.55

    def test_net_price_half_up_tick_rounding(self):
        # 2.675 must round HALF-UP to 2.68 per the system tick
        # convention (execution.patient_executor.round_tick); Python's
        # banker's round(2.675, 2) would give 2.67 and underprice the
        # credit by a cent per spread.
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 1},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 1},
        ]
        order = build_spread_order(legs, net_price=2.675)["Order"][0]
        assert order["priceType"] == "NET_CREDIT"
        assert order["limitPrice"] == 2.68

    def test_net_price_genuinely_below_half_rounds_down(self):
        # 10.0249 is GENUINELY below the half-cent: it sits 1e-4 under
        # 10.025, far outside round_tick's 1e-9 epsilon, so it rounds
        # DOWN to 10.02. Pins that the epsilon only rescues float
        # representation dust, never a real sub-half market price.
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 1},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 1},
        ]
        order = build_spread_order(legs, net_price=10.0249)["Order"][0]
        assert order["limitPrice"] == 10.02

    def test_net_price_one_ulp_below_half_rounds_up_by_design(self):
        # DESIGNED behavior, not a bug: 10.024999999999999 is one ulp
        # below 10.025 — float64 cannot distinguish an intended 10.025
        # from this value (both are stored as 0x1.40ccccccccccc...p+3
        # neighbors and scale to ~1002.49999999999986), so round_tick's
        # 1e-9 epsilon treats it as a half-cent and rounds UP to 10.03.
        # No real market price sits within 1e-9 of a half-cent
        # unintentionally.
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 1},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 1},
        ]
        order = build_spread_order(
            legs, net_price=10.024999999999999)["Order"][0]
        assert order["limitPrice"] == 10.03

    def test_zero_net_price_rejected_as_ambiguous(self):
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 1},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 1},
        ]
        with pytest.raises(ValueError, match="ambiguous"):
            build_spread_order(legs, net_price=0.0)

    def test_invalid_leg_action_rejected(self):
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL", "quantity": 1},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 1},
        ]
        with pytest.raises(ValueError, match="orderAction"):
            build_spread_order(legs, net_price=1.0)


class TestSingleInstrumentBuilders:
    def test_equity_limit_order_shape(self):
        request = build_equity_order("AAPL", "BUY", 10, limit_price=190.005,
                                     client_order_id="EQ1")
        order = request["Order"][0]
        assert request["orderType"] == "EQ"
        assert order["priceType"] == "LIMIT"
        # Half-up tick convention (round_tick): .005 rounds UP to the
        # cent — Python's banker's round() would have given 190.0.
        assert order["limitPrice"] == 190.01
        assert order["orderTerm"] == "GOOD_FOR_DAY"
        assert order["marketSession"] == "REGULAR"
        instrument = order["Instrument"][0]
        assert instrument["Product"] == {"securityType": "EQ",
                                         "symbol": "AAPL"}
        assert instrument["orderAction"] == "BUY"
        assert instrument["quantity"] == 10

    def test_equity_market_order(self):
        order = build_equity_order("AAPL", "SELL", 5)["Order"][0]
        assert order["priceType"] == "MARKET"
        assert "limitPrice" not in order

    def test_option_order_expiry_split(self):
        request = build_option_order("SPY", "put", 440, "2026-07-17",
                                     "SELL_CLOSE", 2, limit_price=1.10)
        product = request["Order"][0]["Instrument"][0]["Product"]
        assert request["orderType"] == "OPTN"
        assert product["callPut"] == "PUT"  # normalized upper
        assert (product["expiryYear"], product["expiryMonth"],
                product["expiryDay"]) == (2026, 7, 17)
        assert product["strikePrice"] == 440.0

    def test_client_order_id_generated_when_absent(self):
        a = build_equity_order("AAPL", "BUY", 1)
        b = build_equity_order("AAPL", "BUY", 1)
        for request in (a, b):
            cid = request["clientOrderId"]
            assert cid.isalnum() and len(cid) <= 20
        assert a["clientOrderId"] != b["clientOrderId"]  # per-intent ids
