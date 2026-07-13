"""Pure support code for the live order preview/place routes.

This module deliberately has no Flask or broker imports.  The blueprint keeps
the guarded broker-import boundary and injects the available request builders;
tests and degraded deployments can therefore keep using the same route-level
compatibility seams without coupling ticket validation to Flask state.
"""

from __future__ import annotations

import math
from typing import Any, Callable, MutableMapping


ORDER_REF_TTL_S = 300
ORDER_REF_CACHE_MAX = 256

OrderRecord = dict[str, Any]
OrderCache = MutableMapping[str, OrderRecord]
OrderBuilder = Callable[..., dict[str, Any]]


def to_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for absent/invalid input."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def to_int(value: Any) -> int | None:
    """Return an exact finite integer, rejecting fractional values."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed != int(parsed):
        return None
    return int(parsed)


def optional_limit_price(
        params: dict[str, Any], key: str = "limit_price", *,
        parse_float: Callable[[Any], float | None] = to_float,
) -> tuple[float | None, str | None]:
    """Parse an optional positive limit without broadening bad input to MKT."""
    raw = params.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, None
    if isinstance(raw, bool):
        return None, f"'{key}' must be a positive finite number or blank"
    value = parse_float(raw)
    if value is None or value <= 0:
        return None, f"'{key}' must be a positive finite number or blank"
    return value, None


def build_order_request(
        kind: str,
        params: dict[str, Any],
        *,
        equity_builder: OrderBuilder,
        option_builder: OrderBuilder,
        spread_builder: OrderBuilder,
        parse_float: Callable[[Any], float | None] = to_float,
        parse_int: Callable[[Any], int | None] = to_int,
        parse_limit: Callable[
            [dict[str, Any], str], tuple[float | None, str | None]
        ] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate an order ticket and delegate to the injected broker builder."""
    if parse_limit is None:
        def parse_limit(values: dict[str, Any], key: str = "limit_price"):
            return optional_limit_price(
                values, key, parse_float=parse_float)

    try:
        if kind == "equity":
            symbol = str(params.get("symbol") or "").strip().upper()
            side = str(params.get("side") or "").strip().upper()
            quantity = parse_int(params.get("quantity"))
            limit, limit_err = parse_limit(params, "limit_price")
            if not symbol:
                return None, "'symbol' is required"
            if side not in ("BUY", "SELL"):
                return None, "'side' must be BUY or SELL"
            if quantity is None or quantity <= 0:
                return None, "'quantity' must be a positive integer"
            if limit_err is not None:
                return None, limit_err
            return equity_builder(
                symbol, side, quantity, limit_price=limit), None

        if kind == "option":
            symbol = str(params.get("symbol") or "").strip().upper()
            call_put = str(params.get("call_put") or "").strip().upper()
            strike = parse_float(params.get("strike"))
            expiry = str(params.get("expiry") or "").strip()
            action = str(params.get("action") or "").strip().upper()
            quantity = parse_int(params.get("quantity"))
            limit, limit_err = parse_limit(params, "limit_price")
            if not symbol:
                return None, "'symbol' is required"
            if call_put not in ("CALL", "PUT"):
                return None, "'call_put' must be CALL or PUT"
            if strike is None or strike <= 0:
                return None, "'strike' must be a positive number"
            if not expiry:
                return None, "'expiry' is required (YYYY-MM-DD)"
            if quantity is None or quantity <= 0:
                return None, "'quantity' must be a positive integer"
            if limit_err is not None:
                return None, limit_err
            return option_builder(
                symbol, call_put, strike, expiry, action, quantity,
                limit_price=limit), None

        if kind == "spread":
            legs_in = params.get("legs")
            net_price = parse_float(params.get("net_price"))
            if not isinstance(legs_in, list) or len(legs_in) < 2:
                return None, "'legs' must be a list of at least 2 legs"
            if net_price is None or net_price == 0:
                return None, ("'net_price' must be a non-zero number "
                              "(+credit / -debit)")
            legs = []
            for index, leg in enumerate(legs_in):
                if not isinstance(leg, dict):
                    return None, f"leg {index} must be an object"
                symbol = str(leg.get("symbol") or "").strip().upper()
                call_put = str(leg.get("call_put") or "").strip().upper()
                strike = parse_float(leg.get("strike"))
                expiry = str(leg.get("expiry") or "").strip()
                action = str(leg.get("action") or "").strip().upper()
                quantity = parse_int(leg.get("quantity"))
                if not symbol:
                    return None, f"leg {index}: symbol is required"
                if call_put not in ("CALL", "PUT"):
                    return None, f"leg {index}: call_put must be CALL or PUT"
                if strike is None or strike <= 0:
                    return None, f"leg {index}: strike must be a positive number"
                if not expiry:
                    return None, f"leg {index}: expiry is required (YYYY-MM-DD)"
                if quantity is None or quantity <= 0:
                    return None, f"leg {index}: quantity must be a positive integer"
                legs.append({
                    "symbol": symbol,
                    "call_put": call_put,
                    "strike": strike,
                    "expiry": expiry,
                    "action": action,
                    "quantity": quantity,
                })
            return spread_builder(legs, net_price), None

        return None, f"unknown kind {kind!r}"
    except ValueError as exc:
        return None, str(exc)


def preview_summary(preview: dict[str, Any]) -> dict[str, Any]:
    """Extract the stable operator-facing fields from a broker preview."""
    preview_ids = [
        item.get("previewId") for item in preview.get("PreviewIds", [])
        if isinstance(item, dict) and item.get("previewId") is not None
    ]
    summary: dict[str, Any] = {"previewIds": preview_ids}
    orders = preview.get("Order") or []
    if orders and isinstance(orders[0], dict):
        first = orders[0]
        for key in ("estimatedTotalAmount", "estimatedCommission",
                    "netPrice", "netbid", "netask"):
            if first.get(key) is not None:
                summary[key] = first[key]
    for key in ("marginLevelCd", "dstFlag", "placedTime",
                "totalOrderValue", "totalCommission"):
        if preview.get(key) is not None:
            summary[key] = preview[key]
    return summary


def prune_order_refs(
        cache: OrderCache, *, now: float, ttl_s: float, max_entries: int,
) -> None:
    """Drop expired cache records, then evict oldest records to the cap."""
    expired = [
        ref for ref, record in cache.items()
        if now - record["created"] > ttl_s
    ]
    for ref in expired:
        cache.pop(ref, None)
    if len(cache) > max_entries:
        for ref, _record in sorted(
                cache.items(), key=lambda item: item[1]["created"]):
            if len(cache) <= max_entries:
                break
            cache.pop(ref, None)


def cache_order_ref(
        cache: OrderCache,
        lock: Any,
        account_id_key: str,
        order_request: dict[str, Any],
        preview_ids: list[Any],
        *,
        clock: Callable[[], float],
        ref_factory: Callable[[int], str],
        ttl_s: float,
        max_entries: int,
) -> str:
    """Store one preview record behind an opaque reference."""
    ref = ref_factory(18)
    with lock:
        prune_order_refs(
            cache, now=clock(), ttl_s=ttl_s, max_entries=max_entries)
        cache[ref] = {
            "request": order_request,
            "preview_ids": preview_ids,
            "account_id_key": account_id_key,
            "created": clock(),
        }
    return ref


def consume_order_ref(
        cache: OrderCache,
        lock: Any,
        ref: str,
        *,
        clock: Callable[[], float],
        ttl_s: float,
        max_entries: int,
) -> OrderRecord | None:
    """Atomically consume a non-expired reference exactly once."""
    with lock:
        prune_order_refs(
            cache, now=clock(), ttl_s=ttl_s, max_entries=max_entries)
        return cache.pop(ref, None)


__all__ = [
    "ORDER_REF_CACHE_MAX",
    "ORDER_REF_TTL_S",
    "build_order_request",
    "cache_order_ref",
    "consume_order_ref",
    "optional_limit_price",
    "preview_summary",
    "prune_order_refs",
    "to_float",
    "to_int",
]
