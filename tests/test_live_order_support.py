"""Characterization tests for extracted live order-route support."""

from __future__ import annotations

import threading

import pytest

from gui.routes import live_order_support as support


def _builder(name, calls):
    def build(*args, **kwargs):
        calls.append((name, args, kwargs))
        return {"builder": name, "args": args, "kwargs": kwargs}

    return build


@pytest.mark.parametrize("value", [None, "", "nan", "inf", object()])
def test_numeric_parsers_reject_absent_or_non_finite_values(value):
    assert support.to_float(value) is None
    assert support.to_int(value) is None


def test_numeric_parsers_preserve_finite_and_exact_values():
    assert support.to_float("1.25") == 1.25
    assert support.to_int("2.0") == 2
    assert support.to_int("2.5") is None


def test_equity_ticket_uses_injected_builder_and_normalizes_fields():
    calls = []
    request, error = support.build_order_request(
        "equity",
        {"symbol": " spy ", "side": "buy", "quantity": "2",
         "limit_price": "101.25"},
        equity_builder=_builder("equity", calls),
        option_builder=_builder("option", calls),
        spread_builder=_builder("spread", calls),
    )

    assert error is None
    assert request["builder"] == "equity"
    assert calls == [("equity", ("SPY", "BUY", 2),
                      {"limit_price": 101.25})]


def test_spread_ticket_preserves_explicit_leg_lifecycle():
    calls = []
    request, error = support.build_order_request(
        "spread",
        {
            "net_price": "1.25",
            "legs": [
                {"symbol": "spy", "call_put": "put", "strike": 450,
                 "expiry": "2026-08-21", "action": "sell_open",
                 "quantity": 1},
                {"symbol": "spy", "call_put": "put", "strike": 445,
                 "expiry": "2026-08-21", "action": "buy_open",
                 "quantity": 1},
            ],
        },
        equity_builder=_builder("equity", calls),
        option_builder=_builder("option", calls),
        spread_builder=_builder("spread", calls),
    )

    assert error is None
    assert request["builder"] == "spread"
    legs, net_price = calls[0][1]
    assert net_price == 1.25
    assert [leg["action"] for leg in legs] == ["SELL_OPEN", "BUY_OPEN"]


def test_bad_explicit_limit_never_silently_becomes_market():
    request, error = support.build_order_request(
        "equity",
        {"symbol": "SPY", "side": "BUY", "quantity": 1,
         "limit_price": "nan"},
        equity_builder=lambda *_args, **_kwargs: pytest.fail("builder called"),
        option_builder=lambda *_args, **_kwargs: {},
        spread_builder=lambda *_args, **_kwargs: {},
    )
    assert request is None
    assert error == "'limit_price' must be a positive finite number or blank"


def test_preview_summary_is_stable_when_optional_blocks_are_missing():
    assert support.preview_summary({}) == {"previewIds": []}
    assert support.preview_summary({
        "PreviewIds": [{"previewId": 7}, {}, "bad"],
        "Order": [{"estimatedCommission": 1.25, "ignored": 9}],
        "totalOrderValue": 400.0,
    }) == {
        "previewIds": [7],
        "estimatedCommission": 1.25,
        "totalOrderValue": 400.0,
    }


def test_order_ref_cache_expires_evicts_and_consumes_once():
    cache = {
        "expired": {"created": 0.0},
        "old": {"created": 95.0},
        "new": {"created": 99.0},
    }
    support.prune_order_refs(cache, now=100.0, ttl_s=10.0, max_entries=1)
    assert list(cache) == ["new"]

    cache = {}
    times = iter([100.0, 101.0])
    ref = support.cache_order_ref(
        cache, threading.RLock(), "acct", {"order": 1}, [{"previewId": 2}],
        clock=lambda: next(times), ref_factory=lambda size: f"ref-{size}",
        ttl_s=300.0, max_entries=256)
    assert ref == "ref-18"
    assert cache[ref]["created"] == 101.0

    lock = threading.RLock()
    first = support.consume_order_ref(
        cache, lock, ref, clock=lambda: 102.0,
        ttl_s=300.0, max_entries=256)
    second = support.consume_order_ref(
        cache, lock, ref, clock=lambda: 102.0,
        ttl_s=300.0, max_entries=256)
    assert first == {
        "request": {"order": 1},
        "preview_ids": [{"previewId": 2}],
        "account_id_key": "acct",
        "created": 101.0,
    }
    assert second is None
