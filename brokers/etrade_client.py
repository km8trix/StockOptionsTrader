"""
Typed E*TRADE v1 API client (Phase 9, contract E2).

THE PHASE-9 RULE: this client never creates a transport. It asks the
EtradeAuthManager for a signing session per request; tests inject fake
factories into the manager and the suite runs with sockets hard-blocked.
The base URL (https://apisb.etrade.com sandbox / https://api.etrade.com
production) comes from the auth manager's env.

ORDER FLOW CORRECTNESS (the part that pages you at 9:31am):
  * preview FIRST: place_order() requires the PreviewIds returned by
    preview_order(); submit_order() runs the whole preview->place dance.
  * one clientOrderId per INTENT (<= 20 alphanum chars), generated once
    and threaded through preview AND place unchanged.
  * IDEMPOTENCY on ambiguous place failures (timeout / 5xx after send):
    NEVER blind-retry. The client queries list_orders for the
    clientOrderId; found -> that order IS our order, return it; not
    found -> safe to retry exactly ONCE; a second ambiguous failure
    raises. Every branch of the dance is audited.
  * Kill switch (C17) and the daily-loss circuit breaker (E5) are checked
    FIRST in preview_order/place_order. Cancel, order status, and quotes
    stay allowed while the switch is engaged.

Retries: bounded, jittered backoff on IDEMPOTENT GETs only. POST/PUT are
never transport-retried (that is what the idempotency dance is for).
429 surfaces as EtradeRateLimited, never a silent retry.
"""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from datetime import date as date_type, datetime
from typing import Any, Callable, Dict, List, Optional

from brokers.etrade_auth import EtradeAuthExpired, EtradeAuthManager
# Safe (acyclic): execution imports core/utils only, never brokers.
from execution.patient_executor import round_tick
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch, KillSwitchEngaged

logger = logging.getLogger(__name__)

#: Transport-level exceptions treated as "the request may have been sent
#: but no answer came back" (requests' exceptions subclass OSError).
AMBIGUOUS_TRANSPORT_ERRORS = (TimeoutError, ConnectionError, OSError)

#: GET retry policy (idempotent endpoints only).
MAX_GET_ATTEMPTS = 3
BACKOFF_BASE_S = 0.5

#: Allowed multi-leg order actions (E*TRADE OPTN instruments).
OPTION_ACTIONS = ("BUY_OPEN", "SELL_OPEN", "BUY_CLOSE", "SELL_CLOSE")

#: Order-scan paging. E*TRADE's orders.json returns a bounded
#: most-recent-first page (default 25, max 100) plus a 'marker' for the
#: next page. The ambiguous-failure recovery MUST NOT miss an order that
#: WAS accepted just because the day was busy, so scans request the max
#: page size and follow markers — bounded so a pathological marker chain
#: can never loop forever (10 x 100 = the most recent 1000 orders, far
#: beyond any day this system will produce).
ORDERS_PAGE_SIZE = 100
ORDERS_MAX_PAGES = 10


class EtradeApiError(RuntimeError):
    """Base class for E*TRADE API failures."""


class EtradeOrderRejected(EtradeApiError):
    """The broker rejected the order; .reason carries its message."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class EtradeRateLimited(EtradeApiError):
    """HTTP 429 — back off; surfaced typed, never silently retried."""


class EtradeUnavailable(EtradeApiError):
    """HTTP 5xx or transport failure."""


def new_client_order_id() -> str:
    """Stable-per-intent order id: <= 20 alphanumeric chars (E*TRADE's
    clientOrderId limit). Generated ONCE per intent by the caller/builders
    and reused across preview, place, and any idempotent recovery."""
    return uuid.uuid4().hex[:20]


# ----------------------------------------------------------------------
# Payload builders (PreviewOrderRequest inner bodies)
# ----------------------------------------------------------------------
def _split_expiry(expiry) -> Dict[str, int]:
    """'YYYY-MM-DD' (or date/datetime) -> expiryYear/Month/Day ints —
    E*TRADE wants the date split, not a string."""
    if isinstance(expiry, str):
        expiry = date_type.fromisoformat(expiry)
    elif isinstance(expiry, datetime):
        expiry = expiry.date()
    return {"expiryYear": expiry.year, "expiryMonth": expiry.month,
            "expiryDay": expiry.day}


def _option_instrument(symbol: str, call_put: str, strike: float, expiry,
                       action: str, quantity: int) -> Dict:
    """One OPTN Instrument for a Preview/PlaceOrderRequest.

    QUANTITY FIELD PAIR: the instrument carries BOTH 'orderedQuantity'
    AND 'quantity', mirroring E*TRADE's documented Preview Order example
    (each option Instrument in the doc sends both, e.g. "orderedQuantity":
    "10", "quantity": "10", and the API reference describes 'quantity' as
    the order quantity on requests). Sending both is harmless if one is
    redundant and required if 'quantity' is the authoritative request
    field — TO BE CONFIRMED at the user-authorized first sandbox preview,
    alongside confirming that orderType SPREADS is accepted for 4-leg
    condors (vs the enum's IRON_CONDOR orderType).
    """
    if action not in OPTION_ACTIONS:
        raise ValueError(f"orderAction {action!r} must be one of "
                         f"{OPTION_ACTIONS}")
    product = {"securityType": "OPTN", "symbol": symbol,
               "callPut": call_put.upper(), "strikePrice": float(strike)}
    product.update(_split_expiry(expiry))
    return {"Product": product, "orderAction": action,
            "orderedQuantity": int(quantity),
            "quantity": int(quantity)}


def build_equity_order(symbol: str, action: str, quantity: int,
                       limit_price: Optional[float] = None,
                       client_order_id: Optional[str] = None) -> Dict:
    """PreviewOrderRequest body for a plain equity order (BUY/SELL).

    limitPrice rounds half-up to the $0.01 tick; values within 1e-9 of
    a half-cent are treated as halves and round up (see
    execution.patient_executor.round_tick for the rationale — float64
    cannot represent half-cent boundaries exactly).
    """
    if action not in ("BUY", "SELL"):
        raise ValueError(f"equity action {action!r} must be BUY or SELL")
    order: Dict[str, Any] = {
        "allOrNone": False,
        "priceType": "LIMIT" if limit_price is not None else "MARKET",
        "orderTerm": "GOOD_FOR_DAY",
        "marketSession": "REGULAR",
        "Instrument": [{
            "Product": {"securityType": "EQ", "symbol": symbol},
            "orderAction": action,
            "quantityType": "QUANTITY",
            "quantity": int(quantity),
        }],
    }
    if limit_price is not None:
        order["limitPrice"] = round_tick(float(limit_price))
    return {
        "orderType": "EQ",
        "clientOrderId": client_order_id or new_client_order_id(),
        "Order": [order],
    }


def build_option_order(symbol: str, call_put: str, strike: float, expiry,
                       action: str, quantity: int,
                       limit_price: Optional[float] = None,
                       client_order_id: Optional[str] = None) -> Dict:
    """PreviewOrderRequest body for a single-leg option order.

    limitPrice rounds half-up to the $0.01 tick; values within 1e-9 of
    a half-cent are treated as halves and round up (see
    execution.patient_executor.round_tick for the rationale — float64
    cannot represent half-cent boundaries exactly).
    """
    order: Dict[str, Any] = {
        "allOrNone": False,
        "priceType": "LIMIT" if limit_price is not None else "MARKET",
        "orderTerm": "GOOD_FOR_DAY",
        "marketSession": "REGULAR",
        "Instrument": [_option_instrument(symbol, call_put, strike, expiry,
                                          action, quantity)],
    }
    if limit_price is not None:
        order["limitPrice"] = round_tick(float(limit_price))
    return {
        "orderType": "OPTN",
        "clientOrderId": client_order_id or new_client_order_id(),
        "Order": [order],
    }


def build_spread_order(legs: List[Dict], net_price: float,
                       client_order_id: Optional[str] = None) -> Dict:
    """PreviewOrderRequest body for a multi-leg (Level 3) options order.

    legs: [{'symbol', 'call_put', 'strike', 'expiry', 'action', 'quantity'}]
    with action in BUY_OPEN/SELL_OPEN/BUY_CLOSE/SELL_CLOSE.

    SIGN CONVENTION (documented, golden-fixture pinned): net_price > 0 is
    a CREDIT the account RECEIVES -> priceType NET_CREDIT; net_price < 0
    is a DEBIT it PAYS -> priceType NET_DEBIT. limitPrice itself is always
    the positive magnitude — E*TRADE carries the sign in priceType.

    limitPrice rounds half-up to the $0.01 tick; values within 1e-9 of
    a half-cent are treated as halves and round up (see
    execution.patient_executor.round_tick for the rationale — float64
    cannot represent half-cent boundaries exactly).
    """
    if not legs or len(legs) < 2:
        raise ValueError("a spread order needs at least 2 legs")
    if net_price == 0:
        raise ValueError("net_price 0 is ambiguous: pass a signed net "
                         "(+credit / -debit)")
    instruments = [
        _option_instrument(leg["symbol"], leg["call_put"], leg["strike"],
                           leg["expiry"], leg["action"], leg["quantity"])
        for leg in legs
    ]
    order = {
        "allOrNone": False,
        "priceType": "NET_CREDIT" if net_price > 0 else "NET_DEBIT",
        "limitPrice": round_tick(abs(float(net_price))),
        "orderTerm": "GOOD_FOR_DAY",
        "marketSession": "REGULAR",
        "Instrument": instruments,
    }
    return {
        "orderType": "SPREADS",
        "clientOrderId": client_order_id or new_client_order_id(),
        "Order": [order],
    }


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------
class EtradeClient:
    """Typed client over the E*TRADE v1 REST API.

    Args:
        auth_manager: EtradeAuthManager — supplies the signing session and
            the env-selected base URL.
        kill_switch: optional KillSwitch checked FIRST in preview/place.
        circuit_breaker: optional injected callable evaluated in the
            preview/place path. It returns falsy when trading is allowed;
            truthy (True or a dict with 'breached': True) blocks the order
            with KillSwitchEngaged. The daily-loss breaker wiring engages
            the persistent kill switch itself before returning truthy.
        reservation_gate: optional duck-typed placement reservation owner.
            ``place_order`` calls ``begin(account_id_key, order_request)``,
            then ``before_attempt(handle, attempt_number)`` immediately
            before each possible POST.  A terminal broker outcome is reported
            through ``accepted(handle, result)``, ``rejected(handle, exc)``,
            or ``unknown(handle, exc)``.  The gate owns the meaning and
            persistence of the opaque handle; this client only guarantees the
            lifecycle ordering around its idempotent placement state machine.
            If present, ``order_update(str(order_id), status)`` also observes
            every status dict returned by ``order_status``; its exceptions
            propagate so callers cannot mistake stale reservation state for a
            successfully processed broker observation.
        audit: AuditLog for the order-flow trail.
        sleep_fn / rng: injected timing/jitter for the bounded GET retries
            (tests pass no-ops).
    """

    def __init__(self, auth_manager: EtradeAuthManager,
                 kill_switch: Optional[KillSwitch] = None,
                 circuit_breaker: Optional[Callable[[], object]] = None,
                 reservation_gate: Optional[Any] = None,
                 audit: Optional[AuditLog] = None,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None):
        self.auth = auth_manager
        self.kill_switch = kill_switch
        self.circuit_breaker = circuit_breaker
        self.reservation_gate = reservation_gate
        self.audit = audit if audit is not None else AuditLog(
            auth_manager.db_path, env=auth_manager.env)
        self._sleep = sleep_fn
        self._rng = rng or random.Random()
        self.base_url = auth_manager.api_base_url
        # The manager owns this RLock.  Sharing (rather than creating a
        # client-local lock) makes an API request mutually exclusive with
        # renew/disconnect and with requests issued by sibling clients over
        # the same OAuth manager/session.
        self._transport_lock = auth_manager.transport_lock

    # ------------------------------------------------------------------
    # Transport plumbing
    # ------------------------------------------------------------------
    def _handle_response(self, response, order_endpoint: bool = False):
        status = response.status_code
        if status in (200, 201):
            return response.json()
        if status == 204:
            return {}
        text = getattr(response, "text", "") or ""
        if status == 401 and "oauth_problem" in text:
            # Token went stale mid-session; flip the auth state so the
            # GUI shows reauth-required, then surface typed.
            self.auth.mark_expired("oauth_problem on API call")
            raise EtradeAuthExpired(
                "E*TRADE returned 401 oauth_problem; re-authorize")
        if status == 429:
            raise EtradeRateLimited(
                "E*TRADE rate limit hit (HTTP 429); back off")
        if status >= 500:
            raise EtradeUnavailable(f"E*TRADE HTTP {status}")
        # 4xx: pull E*TRADE's Error.message when present.
        reason = f"HTTP {status}"
        try:
            body = response.json()
            error = body.get("Error", {})
            if error.get("message"):
                reason = error["message"]
        except (ValueError, AttributeError):
            if text:
                reason = text
        if order_endpoint:
            raise EtradeOrderRejected(reason)
        raise EtradeApiError(reason)

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Idempotent GET with bounded, jittered backoff on transport
        failures and 5xx. 429 and auth failures surface immediately."""
        url = self.base_url + path
        last_error: Optional[Exception] = None
        for attempt in range(MAX_GET_ATTEMPTS):
            try:
                # Lock one wire attempt, not the complete retry loop.  Backoff
                # sleeps remain interruptible by renew/disconnect, and the next
                # attempt picks up the resulting fresh session/state.
                with self._transport_lock:
                    session = self.auth.get_session()
                    response = session.get(url, params=params)
                    return self._handle_response(response)
            except (EtradeUnavailable, *AMBIGUOUS_TRANSPORT_ERRORS) as e:
                last_error = e
                if attempt < MAX_GET_ATTEMPTS - 1:
                    delay = (BACKOFF_BASE_S * (2 ** attempt)
                             * (1.0 + self._rng.random()))
                    logger.warning(
                        "GET %s failed (%s); retry %d/%d in %.2fs",
                        path, e, attempt + 1, MAX_GET_ATTEMPTS - 1, delay)
                    self._sleep(delay)
        if isinstance(last_error, EtradeUnavailable):
            raise last_error
        raise EtradeUnavailable(f"GET {path} failed: {last_error}")

    def _send(self, method: str, path: str, json_body: Dict,
              order_endpoint: bool = False) -> Dict:
        """Non-idempotent send — exactly ONE transport attempt, ever.
        Ambiguous-failure recovery is the caller's (place_order's) job."""
        url = self.base_url + path
        try:
            with self._transport_lock:
                session = self.auth.get_session()
                response = getattr(session, method)(url, json=json_body)
                return self._handle_response(
                    response, order_endpoint=order_endpoint)
        except AMBIGUOUS_TRANSPORT_ERRORS as e:
            raise EtradeUnavailable(f"{method.upper()} {path} failed: {e}")

    # ------------------------------------------------------------------
    # Pre-trade gates (kill switch FIRST, then circuit breaker)
    # ------------------------------------------------------------------
    def _pre_trade_checks(self) -> None:
        if self.kill_switch is not None and self.kill_switch.engaged():
            raise KillSwitchEngaged(
                "Kill switch engaged — new orders are blocked "
                "(cancel/status/quotes remain available)")
        if self.circuit_breaker is not None:
            result = self.circuit_breaker()
            breached = (result.get("breached", False)
                        if isinstance(result, dict) else bool(result))
            if breached:
                raise KillSwitchEngaged(
                    "Daily-loss circuit breaker tripped — new orders are "
                    "blocked for the day")

    # ------------------------------------------------------------------
    # Accounts / market data (read-only; allowed under the kill switch)
    # ------------------------------------------------------------------
    def list_accounts(self) -> List[Dict]:
        body = self._get("/v1/accounts/list.json")
        return (body.get("AccountListResponse", {})
                .get("Accounts", {}).get("Account", []))

    def get_balances(self, account_id_key: str) -> Dict:
        body = self._get(
            f"/v1/accounts/{account_id_key}/balance.json",
            params={"instType": "BROKERAGE", "realTimeNAV": "true"})
        return body.get("BalanceResponse", {})

    def get_portfolio(self, account_id_key: str) -> List[Dict]:
        body = self._get(f"/v1/accounts/{account_id_key}/portfolio.json")
        portfolios = body.get("PortfolioResponse", {}).get(
            "AccountPortfolio", [])
        positions: List[Dict] = []
        for account in portfolios:
            positions.extend(account.get("Position", []))
        return positions

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Multi-symbol quotes -> {symbol: {'bid','ask','last'}}."""
        if not symbols:
            return {}
        joined = ",".join(symbols)
        body = self._get(f"/v1/market/quote/{joined}.json")
        quotes: Dict[str, Dict] = {}
        for quote_data in body.get("QuoteResponse", {}).get("QuoteData", []):
            symbol = quote_data.get("Product", {}).get("symbol")
            all_block = quote_data.get("All", {})
            if symbol:
                quotes[symbol] = {
                    "bid": all_block.get("bid"),
                    "ask": all_block.get("ask"),
                    "last": all_block.get("lastTrade"),
                }
        return quotes

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def _orders_page(self, account_id_key: str,
                     count: Optional[int] = None,
                     marker: Optional[str] = None):
        """One page of orders.json -> (orders, next_marker)."""
        params: Dict = {}
        if count is not None:
            params["count"] = count
        if marker is not None:
            params["marker"] = marker
        body = self._get(f"/v1/accounts/{account_id_key}/orders.json",
                         params=params or None)
        orders_response = body.get("OrdersResponse", {})
        return (orders_response.get("Order", []),
                orders_response.get("marker") or None)

    def list_orders(self, account_id_key: str,
                    count: Optional[int] = None,
                    marker: Optional[str] = None) -> List[Dict]:
        """ONE page of recent orders (most-recent-first). Pass count
        (max 100) and the previous page's marker to page explicitly;
        full scans go through _iter_orders."""
        orders, _next_marker = self._orders_page(account_id_key,
                                                 count=count, marker=marker)
        return orders

    def _iter_orders(self, account_id_key: str):
        """Iterate recent orders across pages, most-recent-first,
        following E*TRADE's marker pagination up to ORDERS_MAX_PAGES
        pages of ORDERS_PAGE_SIZE."""
        marker: Optional[str] = None
        for _page in range(ORDERS_MAX_PAGES):
            orders, marker = self._orders_page(
                account_id_key, count=ORDERS_PAGE_SIZE, marker=marker)
            yield from orders
            if not marker:
                return
        logger.warning(
            "Order scan stopped after %d pages (%d orders) with more "
            "pages remaining — broker has an unusually deep order book "
            "today", ORDERS_MAX_PAGES, ORDERS_MAX_PAGES * ORDERS_PAGE_SIZE)

    def find_order_by_client_id(self, account_id_key: str,
                                client_order_id: str) -> Optional[Dict]:
        """Locate our order by the clientOrderId we generated — the pivot
        of the ambiguous-failure recovery.

        Scans ALL recent pages (count=100, marker-paged), not just the
        broker's default first 25: on a busy order day a first-page-only
        lookup could miss an order that WAS accepted and send the state
        machine down the 'not found -> retry' branch — a duplicate live
        order, the exact failure this dance exists to prevent.
        """
        for order in self._iter_orders(account_id_key):
            if order.get("clientOrderId") == client_order_id:
                return order
            for detail in order.get("OrderDetail", []):
                if detail.get("clientOrderId") == client_order_id:
                    return order
        return None

    def preview_order(self, account_id_key: str,
                      order_request: Dict) -> Dict:
        """POST orders/preview.json. Kill switch + circuit breaker FIRST."""
        self._pre_trade_checks()
        response = self._send(
            "post", f"/v1/accounts/{account_id_key}/orders/preview.json",
            {"PreviewOrderRequest": order_request}, order_endpoint=True)
        preview = response.get("PreviewOrderResponse", {})
        self.audit.append("etrade_client", "order_previewed", {
            "account_id_key": account_id_key,
            "client_order_id": order_request.get("clientOrderId"),
            "order_type": order_request.get("orderType"),
            "preview_ids": preview.get("PreviewIds", []),
        })
        return preview

    def place_order(self, account_id_key: str, order_request: Dict,
                    preview_ids: List[Dict]) -> Dict:
        """POST orders/place.json carrying the previewIds AND the same
        clientOrderId the preview used.

        Returns {'order_id', 'response', 'recovered', 'retried'}.

        THE IDEMPOTENCY STATE MACHINE (every branch audited):
          send -> ok ............................. return (placed)
          send -> ambiguous (timeout/5xx) ........ look up clientOrderId
              found ............................. return it (recovered);
                                                  NO second place call
              not found ......................... retry exactly ONCE
                  retry ok ...................... return (retried)
                  retry ambiguous/failed ........ raise (operator decides)
        """
        self._pre_trade_checks()
        if not preview_ids:
            raise EtradeOrderRejected(
                "place_order requires PreviewIds — preview first")
        client_order_id = order_request.get("clientOrderId")
        if not isinstance(client_order_id, str):
            raise EtradeOrderRejected(
                "order_request missing clientOrderId — build via "
                "build_equity_order/build_option_order")
        payload = {"PlaceOrderRequest": dict(order_request)}
        payload["PlaceOrderRequest"]["PreviewIds"] = preview_ids
        path = f"/v1/accounts/{account_id_key}/orders/place.json"

        # Reservation begins only after all local validation has succeeded.
        # No broker POST has occurred at this point.
        reservation_handle = None
        if self.reservation_gate is not None:
            try:
                reservation_handle = self.reservation_gate.begin(
                    account_id_key, order_request)
            except Exception as begin_error:
                # No opaque handle was returned, but still give the gate a
                # best-effort rollback notification.  Gates should treat a
                # ``None`` handle as "begin did not complete".
                try:
                    self.reservation_gate.rejected(None, begin_error)
                except Exception:
                    logger.exception(
                        "Reservation gate rejected callback failed after "
                        "begin failure")
                raise

        def _notify_preserving(name: str, error: Exception) -> None:
            """Best-effort terminal callback that never masks *error*."""
            if self.reservation_gate is None:
                return
            try:
                getattr(self.reservation_gate, name)(reservation_handle,
                                                     error)
            except Exception:
                logger.exception(
                    "Reservation gate %s callback failed while handling %s",
                    name, error)

        def _accepted(result: Dict) -> None:
            """Commit a known acceptance; callback failure becomes unknown."""
            if self.reservation_gate is None:
                return
            try:
                self.reservation_gate.accepted(reservation_handle, result)
            except Exception as callback_error:
                # The broker outcome is known, but the reservation commit is
                # not.  Escalate rather than releasing capacity that may back
                # a live order, while preserving the callback's exception.
                _notify_preserving("unknown", callback_error)
                raise

        def _before_attempt(attempt_number: int) -> None:
            if self.reservation_gate is not None:
                self.reservation_gate.before_attempt(
                    reservation_handle, attempt_number)

        def _attempt() -> Dict:
            response = self._send("post", path, payload, order_endpoint=True)
            placed = response.get("PlaceOrderResponse", {})
            order_ids = placed.get("OrderIds", [])
            if not order_ids:
                # A 200 with no OrderIds means the order was NOT accepted (or
                # the response is malformed) — a successful place always carries
                # them. Returning order_id=None lets the caller stringify it to
                # "None", which then poisons cancel_order and the audit trail
                # (a ghost order working at the broker while the session thinks
                # it was never placed). Fail loudly so the operator decides.
                raise EtradeOrderRejected(
                    "E*TRADE place response carried no OrderIds — the order was "
                    "not accepted (or the response was malformed)")
            return {
                "order_id": order_ids[0].get("orderId"),
                "response": placed,
            }

        try:
            _before_attempt(1)
        except Exception as before_error:
            # The reservation/revalue hook failed before the first wire send:
            # broker non-acceptance is certain, so release the reservation.
            _notify_preserving("rejected", before_error)
            raise

        try:
            result = _attempt()
        except (EtradeUnavailable, EtradeRateLimited) as first_error:
            if isinstance(first_error, EtradeRateLimited):
                # 429 is NOT ambiguous: the broker answered "slow down"
                # and did not accept the order. Surface typed, no dance.
                _notify_preserving("rejected", first_error)
                raise
            self.audit.append("etrade_client", "order_place_ambiguous", {
                "account_id_key": account_id_key,
                "client_order_id": client_order_id,
                "error": str(first_error),
            })
            logger.warning(
                "Ambiguous place failure for clientOrderId %s: %s — "
                "querying order status before any retry",
                client_order_id, first_error)
            try:
                existing = self.find_order_by_client_id(
                    account_id_key, client_order_id)
            except Exception as lookup_error:
                # The first send may have succeeded and the recovery query
                # could not resolve it.  Capacity must remain reserved.
                _notify_preserving("unknown", lookup_error)
                raise
            if existing is not None:
                self.audit.append("etrade_client", "order_place_recovered", {
                    "account_id_key": account_id_key,
                    "client_order_id": client_order_id,
                    "order_id": existing.get("orderId"),
                })
                logger.info("clientOrderId %s found at broker as order %s; "
                            "NOT retrying", client_order_id,
                            existing.get("orderId"))
                recovered = {"order_id": existing.get("orderId"),
                             "response": existing,
                             "recovered": True, "retried": False}
                _accepted(recovered)
                return recovered
            self.audit.append("etrade_client", "order_place_retry", {
                "account_id_key": account_id_key,
                "client_order_id": client_order_id,
            })
            logger.info("clientOrderId %s not found at broker; retrying "
                        "exactly once", client_order_id)
            try:
                _before_attempt(2)
                result = _attempt()
            except Exception as second_error:
                self.audit.append("etrade_client", "order_place_failed", {
                    "account_id_key": account_id_key,
                    "client_order_id": client_order_id,
                    "error": str(second_error),
                })
                # Even a definitive second-attempt rejection cannot prove the
                # earlier ambiguous send was not accepted after our lookup.
                _notify_preserving("unknown", second_error)
                raise
            result.update({"recovered": False, "retried": True})
            self._audit_placed(account_id_key, client_order_id, result)
            _accepted(result)
            return result
        except (EtradeApiError, EtradeAuthExpired) as rejected_error:
            # All typed non-availability API outcomes here are definitive and
            # occurred without an earlier ambiguous place attempt.
            _notify_preserving("rejected", rejected_error)
            raise
        except Exception as unknown_error:
            # A malformed success response or other unexpected post-send
            # failure is not proof the broker declined the order.
            _notify_preserving("unknown", unknown_error)
            raise
        result.update({"recovered": False, "retried": False})
        self._audit_placed(account_id_key, client_order_id, result)
        _accepted(result)
        return result

    def _audit_placed(self, account_id_key: str,
                      client_order_id: Optional[str], result: Dict) -> None:
        self.audit.append("etrade_client", "order_placed", {
            "account_id_key": account_id_key,
            "client_order_id": client_order_id,
            "order_id": result.get("order_id"),
            "retried": result.get("retried", False),
        })

    def submit_order(self, account_id_key: str, order_request: Dict) -> Dict:
        """The full correct flow: preview, then place threading the
        returned previewId + the SAME clientOrderId."""
        preview = self.preview_order(account_id_key, order_request)
        preview_ids = preview.get("PreviewIds", [])
        result = self.place_order(account_id_key, order_request, preview_ids)
        result["preview"] = preview
        return result

    def cancel_order(self, account_id_key: str, order_id) -> bool:
        """PUT orders/cancel.json — allowed even under the kill switch
        (an operator must always be able to pull orders)."""
        response = self._send(
            "put", f"/v1/accounts/{account_id_key}/orders/cancel.json",
            {"CancelOrderRequest": {"orderId": order_id}},
            order_endpoint=True)
        cancelled = "CancelOrderResponse" in response
        self.audit.append("etrade_client", "order_cancelled", {
            "account_id_key": account_id_key,
            "order_id": order_id,
            "ok": cancelled,
        })
        return cancelled

    def order_status(self, account_id_key: str, order_id) -> Optional[Dict]:
        """{'status', 'filled_quantity', 'avg_fill_price'} for one order,
        or None when the broker does not know the id. Marker-paged like
        find_order_by_client_id (a worked order is nearly always on the
        first page, so this normally costs one GET)."""
        for order in self._iter_orders(account_id_key):
            if order.get("orderId") != order_id:
                continue
            details = order.get("OrderDetail", [])
            status = details[0].get("status") if details else None
            instruments = [inst for detail in details
                           for inst in detail.get("Instrument", [])]
            if len(instruments) <= 1:
                inst = instruments[0] if instruments else {}
                filled = float(inst.get("filledQuantity", 0) or 0)
                avg_price = (float(inst["averageExecutionPrice"])
                             if inst.get("averageExecutionPrice") is not None
                             else None)
                result = {"status": status, "filled_quantity": filled,
                          "avg_fill_price": avg_price}
                self._reservation_order_update(order_id, result)
                return result
            # Multi-leg (SPREADS): legs fill as a PACKAGE. Summing leg
            # quantities would report n*q for q package contracts (a partial
            # fill then looks complete to PatientExecutor), and one leg's
            # price is meaningless — report package contracts (min across
            # legs) and the signed NET price (+ for BUY_*, - for SELL_*,
            # matching build_spread_order's credit/debit convention).
            ordered_values = [
                inst.get("orderedQuantity", inst.get("quantity"))
                for inst in instruments
            ]
            if all(value is None for value in ordered_values):
                # Some list-order fixtures/providers omit ordered quantities
                # for equal-ratio packages. Preserve the historical 1:1
                # interpretation, but never partially guess a ratio set.
                ratios = [1] * len(instruments)
            else:
                if any(value is None for value in ordered_values):
                    raise ValueError(
                        "spread status has incomplete ordered quantities")
                ordered = []
                for value in ordered_values:
                    numeric = float(value)
                    if (not math.isfinite(numeric) or numeric <= 0
                            or not numeric.is_integer()):
                        raise ValueError(
                            "spread ordered quantities must be positive integers")
                    ordered.append(int(numeric))
                divisor = ordered[0]
                for quantity in ordered[1:]:
                    divisor = math.gcd(divisor, quantity)
                ratios = [quantity // divisor for quantity in ordered]
            filled = min(
                float(inst.get("filledQuantity", 0) or 0) / ratio
                for inst, ratio in zip(instruments, ratios)
            )
            net = 0.0
            have_price = False
            for inst, ratio in zip(instruments, ratios):
                px = inst.get("averageExecutionPrice")
                if px is None:
                    continue
                have_price = True
                sign = 1.0 if str(inst.get("orderAction", "")).startswith(
                    "BUY") else -1.0
                net += sign * float(px) * ratio
            result = {"status": status, "filled_quantity": filled,
                      "avg_fill_price": net if have_price else None}
            self._reservation_order_update(order_id, result)
            return result
        return None

    def _reservation_order_update(self, order_id, status: Dict) -> None:
        """Forward one known broker observation to an optional reservation.

        Gates may leave ``order_update`` undefined when they do not bind
        reservations to order IDs.  Callback failures intentionally propagate:
        the caller must not continue as though reservation capacity reflects
        the status it just observed.
        """
        callback = (getattr(self.reservation_gate, "order_update", None)
                    if self.reservation_gate is not None else None)
        if callback is not None:
            callback(str(order_id), status)
