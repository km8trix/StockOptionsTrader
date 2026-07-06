"""EtradeClient (E2): documented-shape endpoint fixtures, preview->place
threading, the ambiguous-failure idempotency state machine, typed errors,
kill-switch gating, and the daily-loss circuit-breaker boundary."""

from __future__ import annotations

import pytest

from brokers.circuit_breaker import DailyLossCircuitBreaker
from brokers.etrade_auth import EtradeAuthExpired
from brokers.etrade_client import (EtradeClient, EtradeOrderRejected,
                                   EtradeRateLimited, EtradeUnavailable,
                                   build_equity_order, build_spread_order,
                                   new_client_order_id)
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch, KillSwitchEngaged

ACCOUNT = "dBZOKt9xDrtRSAOl4MSiiA"


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text or ("" if json_data is None else str(json_data))

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class ScriptedTransport:
    """Routes (method, url-fragment) -> queue of responses/exceptions.

    A queue with more than one item pops per call; the last item is
    sticky. Exceptions in the queue are raised (transport failures)."""

    def __init__(self):
        self.handlers = {}
        self.calls = []

    def route(self, method, fragment, *items):
        self.handlers[(method, fragment)] = list(items)

    def _do(self, method, url, params=None, json=None):
        self.calls.append({"method": method, "url": url, "json": json,
                           "params": params})
        for (m, fragment), queue in self.handlers.items():
            if m == method and fragment in url:
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(f"unrouted {method} {url}")

    def get(self, url, params=None, json=None):
        return self._do("get", url, params=params)

    def post(self, url, params=None, json=None):
        return self._do("post", url, json=json)

    def put(self, url, params=None, json=None):
        return self._do("put", url, json=json)

    def count(self, method, fragment):
        return sum(1 for call in self.calls
                   if call["method"] == method and fragment in call["url"])


class FakeAuth:
    """Stands in for EtradeAuthManager: hands out the scripted transport."""

    def __init__(self, transport, db_path):
        self.transport = transport
        self.db_path = db_path
        self.env = "sandbox"
        self.api_base_url = "https://apisb.etrade.com"
        self.expired_reasons = []

    def get_session(self):
        return self.transport

    def mark_expired(self, reason):
        self.expired_reasons.append(reason)


@pytest.fixture
def harness(tmp_path):
    transport = ScriptedTransport()
    db_path = str(tmp_path / "client.db")
    audit = AuditLog(db_path, env="sandbox")
    client = EtradeClient(FakeAuth(transport, db_path), audit=audit,
                          sleep_fn=lambda _s: None)
    return client, transport, audit


# ----------------------------------------------------------------------
# Documented-shape endpoint fixtures
# ----------------------------------------------------------------------
class TestEndpoints:
    def test_list_accounts(self, harness):
        client, transport, _ = harness
        transport.route("get", "/v1/accounts/list.json", FakeResponse(
            200, {"AccountListResponse": {"Accounts": {"Account": [
                {"accountId": "82314598", "accountIdKey": ACCOUNT,
                 "accountDesc": "Brokerage", "accountStatus": "ACTIVE",
                 "institutionType": "BROKERAGE"}]}}}))
        accounts = client.list_accounts()
        assert accounts[0]["accountIdKey"] == ACCOUNT

    def test_get_balances(self, harness):
        client, transport, _ = harness
        transport.route("get", f"/v1/accounts/{ACCOUNT}/balance.json",
                        FakeResponse(200, {"BalanceResponse": {
                            "accountId": "82314598",
                            "Computed": {
                                "cashAvailableForInvestment": 12000.0,
                                "RealTimeValues": {
                                    "totalAccountValue": 99000.0}}}}))
        balances = client.get_balances(ACCOUNT)
        assert balances["Computed"]["RealTimeValues"][
            "totalAccountValue"] == 99000.0

    def test_get_portfolio(self, harness):
        client, transport, _ = harness
        transport.route("get", f"/v1/accounts/{ACCOUNT}/portfolio.json",
                        FakeResponse(200, {"PortfolioResponse": {
                            "AccountPortfolio": [{"accountId": "82314598",
                                "Position": [{
                                    "positionId": 1, "quantity": 100,
                                    "marketValue": 19000.0,
                                    "totalGain": 150.0,
                                    "Product": {"symbol": "AAPL",
                                                "securityType": "EQ"}}]}]}}))
        positions = client.get_portfolio(ACCOUNT)
        assert len(positions) == 1
        assert positions[0]["Product"]["symbol"] == "AAPL"

    def test_get_quotes_multi_symbol(self, harness):
        client, transport, _ = harness
        transport.route("get", "/v1/market/quote/AAPL,SPY.json",
                        FakeResponse(200, {"QuoteResponse": {"QuoteData": [
                            {"Product": {"symbol": "AAPL",
                                         "securityType": "EQ"},
                             "All": {"bid": 189.95, "ask": 190.05,
                                     "lastTrade": 190.0}},
                            {"Product": {"symbol": "SPY",
                                         "securityType": "EQ"},
                             "All": {"bid": 520.10, "ask": 520.20,
                                     "lastTrade": 520.15}}]}}))
        quotes = client.get_quotes(["AAPL", "SPY"])
        assert quotes["AAPL"] == {"bid": 189.95, "ask": 190.05,
                                  "last": 190.0}
        assert quotes["SPY"]["bid"] == 520.10

    def test_cancel_order(self, harness):
        client, transport, _ = harness
        transport.route("put", f"/v1/accounts/{ACCOUNT}/orders/cancel.json",
                        FakeResponse(200, {"CancelOrderResponse": {
                            "accountId": "82314598", "orderId": 529}}))
        assert client.cancel_order(ACCOUNT, 529) is True
        sent = transport.calls[-1]["json"]
        assert sent == {"CancelOrderRequest": {"orderId": 529}}

    def test_order_status_from_list_orders(self, harness):
        client, transport, _ = harness
        transport.route("get", f"/v1/accounts/{ACCOUNT}/orders.json",
                        FakeResponse(200, {"OrdersResponse": {"Order": [
                            {"orderId": 529, "orderType": "EQ",
                             "OrderDetail": [{"status": "EXECUTED",
                                "clientOrderId": "abc123",
                                "Instrument": [{
                                    "filledQuantity": 10,
                                    "averageExecutionPrice": 190.01}]}]}]}}))
        status = client.order_status(ACCOUNT, 529)
        assert status == {"status": "EXECUTED", "filled_quantity": 10.0,
                          "avg_fill_price": 190.01}
        assert client.order_status(ACCOUNT, 999) is None

    def test_order_status_spread_partial_package_fill(self, harness):
        """4-leg condor, contracts=2, only 1 package contract filled: the
        old per-leg SUM reported 4 (>= 2 -> phantom 'complete'); the fix
        reports min across legs (1) and the signed net package price."""
        client, transport, _ = harness
        legs = [
            {"filledQuantity": 1, "averageExecutionPrice": 1.20,
             "orderAction": "SELL_OPEN"},
            {"filledQuantity": 1, "averageExecutionPrice": 0.80,
             "orderAction": "BUY_OPEN"},
            {"filledQuantity": 1, "averageExecutionPrice": 1.10,
             "orderAction": "SELL_OPEN"},
            {"filledQuantity": 1, "averageExecutionPrice": 0.70,
             "orderAction": "BUY_OPEN"},
        ]
        transport.route("get", f"/v1/accounts/{ACCOUNT}/orders.json",
                        FakeResponse(200, {"OrdersResponse": {"Order": [
                            {"orderId": 640, "orderType": "SPREADS",
                             "OrderDetail": [{"status": "OPEN",
                                              "Instrument": legs}]}]}}))
        status = client.order_status(ACCOUNT, 640)
        assert status["filled_quantity"] == 1.0          # package contracts
        # net = -1.20 + 0.80 - 1.10 + 0.70 = -0.80 (credit received)
        assert abs(status["avg_fill_price"] - (-0.80)) < 1e-9

    def test_get_retry_backoff_on_5xx_then_success(self, harness):
        client, transport, _ = harness
        sleeps = []
        client._sleep = sleeps.append
        transport.route("get", "/v1/market/quote/SPY.json",
                        FakeResponse(503, text="maintenance"),
                        FakeResponse(503, text="maintenance"),
                        FakeResponse(200, {"QuoteResponse": {"QuoteData": [
                            {"Product": {"symbol": "SPY"},
                             "All": {"bid": 1.0, "ask": 1.1,
                                     "lastTrade": 1.05}}]}}))
        quotes = client.get_quotes(["SPY"])
        assert quotes["SPY"]["last"] == 1.05
        assert transport.count("get", "quote") == 3  # bounded retries
        assert len(sleeps) == 2 and all(s > 0 for s in sleeps)  # jittered

    def test_rate_limit_surfaces_typed_without_retry(self, harness):
        client, transport, _ = harness
        transport.route("get", "/v1/market/quote/SPY.json",
                        FakeResponse(429, text="rate limit"))
        with pytest.raises(EtradeRateLimited):
            client.get_quotes(["SPY"])
        assert transport.count("get", "quote") == 1

    def test_401_oauth_problem_maps_to_auth_expired(self, harness):
        client, transport, _ = harness
        transport.route("get", "/v1/accounts/list.json",
                        FakeResponse(401, text="oauth_problem=token_expired"))
        with pytest.raises(EtradeAuthExpired):
            client.list_accounts()
        assert client.auth.expired_reasons  # state flipped for the GUI


# ----------------------------------------------------------------------
# Preview -> place threading + rejection mapping
# ----------------------------------------------------------------------
class TestOrderFlow:
    def _route_preview_place(self, transport, preview_id=1627181131,
                             order_id=529):
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {
                "PreviewIds": [{"previewId": preview_id}]}}))
        transport.route("post", "orders/place.json", FakeResponse(
            200, {"PlaceOrderResponse": {
                "OrderIds": [{"orderId": order_id}]}}))

    def test_preview_then_place_threads_ids(self, harness):
        client, transport, _ = harness
        self._route_preview_place(transport)
        cid = new_client_order_id()
        assert len(cid) <= 20 and cid.isalnum()
        request = build_equity_order("AAPL", "BUY", 10, limit_price=190.0,
                                     client_order_id=cid)
        result = client.submit_order(ACCOUNT, request)
        assert result["order_id"] == 529
        assert result["recovered"] is False and result["retried"] is False
        preview_call = next(c for c in transport.calls
                            if "preview" in c["url"])
        place_call = next(c for c in transport.calls if "place" in c["url"])
        assert preview_call["json"]["PreviewOrderRequest"][
            "clientOrderId"] == cid
        placed = place_call["json"]["PlaceOrderRequest"]
        # The SAME clientOrderId and the preview's PreviewIds, verbatim.
        assert placed["clientOrderId"] == cid
        assert placed["PreviewIds"] == [{"previewId": 1627181131}]

    def test_place_without_preview_ids_refused(self, harness):
        client, _, _ = harness
        with pytest.raises(EtradeOrderRejected, match="preview"):
            client.place_order(ACCOUNT, build_equity_order("A", "BUY", 1),
                               [])

    def test_rejection_maps_reason(self, harness):
        client, transport, _ = harness
        transport.route("post", "orders/preview.json", FakeResponse(
            400, {"Error": {"code": 1033,
                            "message": "Insufficient option level"}}))
        with pytest.raises(EtradeOrderRejected,
                           match="Insufficient option level") as excinfo:
            client.preview_order(ACCOUNT,
                                 build_equity_order("AAPL", "BUY", 10))
        assert excinfo.value.reason == "Insufficient option level"

    def test_order_flow_audited(self, harness):
        client, transport, audit = harness
        self._route_preview_place(transport)
        client.submit_order(ACCOUNT, build_equity_order("AAPL", "BUY", 10))
        events = [e["event_type"] for e in audit.entries(ascending=True)]
        assert events == ["order_previewed", "order_placed"]

    def test_empty_order_ids_raises_rejected(self, harness):
        # Phase 1: a 200 place response carrying no OrderIds must RAISE, not
        # return a None order_id that stringifies to "None" and poisons later
        # cancels / the audit trail (a ghost order at the broker).
        client, transport, _ = harness
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [{"previewId": 9}]}}))
        transport.route("post", "orders/place.json", FakeResponse(
            200, {"PlaceOrderResponse": {"OrderIds": []}}))
        with pytest.raises(EtradeOrderRejected, match="OrderIds"):
            client.submit_order(ACCOUNT, build_equity_order("AAPL", "BUY", 10))

    def test_missing_order_ids_key_raises_rejected(self, harness):
        client, transport, _ = harness
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [{"previewId": 9}]}}))
        transport.route("post", "orders/place.json", FakeResponse(
            200, {"PlaceOrderResponse": {}}))
        with pytest.raises(EtradeOrderRejected, match="OrderIds"):
            client.submit_order(ACCOUNT, build_equity_order("AAPL", "BUY", 10))


# ----------------------------------------------------------------------
# THE IDEMPOTENCY STATE MACHINE
# ----------------------------------------------------------------------
class TestPlaceIdempotency:
    def _request(self):
        return build_equity_order("AAPL", "BUY", 10, limit_price=190.0,
                                  client_order_id="intent0001abcd")

    def _preview_ids(self):
        return [{"previewId": 11}]

    def test_ambiguous_then_found_never_replaces(self, harness):
        client, transport, audit = harness
        transport.route("post", "orders/place.json",
                        FakeResponse(503, text="gateway timeout"))
        transport.route("get", f"/v1/accounts/{ACCOUNT}/orders.json",
                        FakeResponse(200, {"OrdersResponse": {"Order": [
                            {"orderId": 777, "OrderDetail": [
                                {"status": "OPEN",
                                 "clientOrderId": "intent0001abcd"}]}]}}))
        result = client.place_order(ACCOUNT, self._request(),
                                    self._preview_ids())
        assert result["recovered"] is True and result["retried"] is False
        assert result["order_id"] == 777
        # THE assertion that matters: exactly ONE place call ever hit the
        # wire — the order was found, so no blind second send.
        assert transport.count("post", "place") == 1
        events = [e["event_type"] for e in audit.entries(ascending=True)]
        assert events == ["order_place_ambiguous", "order_place_recovered"]

    def test_ambiguous_then_not_found_retries_exactly_once(self, harness):
        client, transport, audit = harness
        transport.route("post", "orders/place.json",
                        FakeResponse(500, text="boom"),
                        FakeResponse(200, {"PlaceOrderResponse": {
                            "OrderIds": [{"orderId": 530}]}}))
        transport.route("get", f"/v1/accounts/{ACCOUNT}/orders.json",
                        FakeResponse(200, {"OrdersResponse": {"Order": []}}))
        result = client.place_order(ACCOUNT, self._request(),
                                    self._preview_ids())
        assert result["order_id"] == 530
        assert result["retried"] is True and result["recovered"] is False
        assert transport.count("post", "place") == 2  # one retry, no more
        events = [e["event_type"] for e in audit.entries(ascending=True)]
        assert events == ["order_place_ambiguous", "order_place_retry",
                          "order_placed"]

    def test_second_ambiguous_failure_raises(self, harness):
        client, transport, audit = harness
        transport.route("post", "orders/place.json",
                        FakeResponse(503, text="down"),
                        FakeResponse(503, text="still down"))
        transport.route("get", f"/v1/accounts/{ACCOUNT}/orders.json",
                        FakeResponse(200, {"OrdersResponse": {"Order": []}}))
        with pytest.raises(EtradeUnavailable):
            client.place_order(ACCOUNT, self._request(),
                               self._preview_ids())
        assert transport.count("post", "place") == 2  # never a third
        events = [e["event_type"] for e in audit.entries(ascending=True)]
        assert events == ["order_place_ambiguous", "order_place_retry",
                          "order_place_failed"]

    def test_transport_exception_is_ambiguous_too(self, harness):
        client, transport, _ = harness
        transport.route("post", "orders/place.json",
                        TimeoutError("read timed out"),
                        FakeResponse(200, {"PlaceOrderResponse": {
                            "OrderIds": [{"orderId": 531}]}}))
        transport.route("get", f"/v1/accounts/{ACCOUNT}/orders.json",
                        FakeResponse(200, {"OrdersResponse": {"Order": []}}))
        result = client.place_order(ACCOUNT, self._request(),
                                    self._preview_ids())
        assert result["order_id"] == 531 and result["retried"] is True

    def test_recovery_lookup_follows_pagination_markers(self, harness):
        """A busy order day pushes our order off the broker's first page;
        the recovery lookup MUST page through (count=100 + marker) and
        find it rather than blind-retry into a duplicate live order."""
        client, transport, _ = harness
        transport.route("post", "orders/place.json",
                        FakeResponse(503, text="gateway timeout"))
        page1_orders = [{"orderId": 1000 + i, "OrderDetail": [
            {"status": "OPEN", "clientOrderId": f"other{i:04d}"}]}
            for i in range(100)]
        transport.route(
            "get", f"/v1/accounts/{ACCOUNT}/orders.json",
            FakeResponse(200, {"OrdersResponse": {
                "Order": page1_orders, "marker": "page2marker"}}),
            FakeResponse(200, {"OrdersResponse": {"Order": [
                {"orderId": 777, "OrderDetail": [
                    {"status": "OPEN",
                     "clientOrderId": "intent0001abcd"}]}]}}))
        result = client.place_order(ACCOUNT, self._request(),
                                    self._preview_ids())
        assert result["recovered"] is True and result["order_id"] == 777
        assert transport.count("post", "place") == 1  # never a retry
        orders_gets = [c for c in transport.calls
                       if c["method"] == "get" and "orders.json" in c["url"]]
        assert len(orders_gets) == 2
        # Explicit max page size on every page; page 2 carries the marker.
        assert orders_gets[0]["params"] == {"count": 100}
        assert orders_gets[1]["params"] == {"count": 100,
                                            "marker": "page2marker"}

    def test_order_scan_is_bounded_against_endless_markers(self, harness):
        """A marker chain that never terminates stops at ORDERS_MAX_PAGES
        — the lookup can come up empty, but it can never spin forever."""
        from brokers.etrade_client import ORDERS_MAX_PAGES
        client, transport, _ = harness
        transport.route(
            "get", f"/v1/accounts/{ACCOUNT}/orders.json",
            FakeResponse(200, {"OrdersResponse": {
                "Order": [{"orderId": 1, "OrderDetail": [
                    {"status": "OPEN", "clientOrderId": "nope"}]}],
                "marker": "again"}}))  # sticky: every page points onward
        assert client.find_order_by_client_id(ACCOUNT, "missing") is None
        assert transport.count("get", "orders.json") == ORDERS_MAX_PAGES


# ----------------------------------------------------------------------
# Kill switch + circuit breaker gating
# ----------------------------------------------------------------------
class TestPreTradeGates:
    def _client_with_switch(self, tmp_path, circuit_breaker=None):
        transport = ScriptedTransport()
        db_path = str(tmp_path / "gates.db")
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        client = EtradeClient(FakeAuth(transport, db_path),
                              kill_switch=switch,
                              circuit_breaker=circuit_breaker,
                              audit=audit, sleep_fn=lambda _s: None)
        return client, transport, switch, audit

    def test_kill_switch_blocks_preview_and_place_not_cancel_quotes(
            self, tmp_path):
        client, transport, switch, _ = self._client_with_switch(tmp_path)
        switch.engage("test halt", actor="ops")
        request = build_equity_order("AAPL", "BUY", 10)
        with pytest.raises(KillSwitchEngaged):
            client.preview_order(ACCOUNT, request)
        with pytest.raises(KillSwitchEngaged):
            client.place_order(ACCOUNT, request, [{"previewId": 1}])
        # Cancel and quotes stay available so the operator can flatten.
        transport.route("put", "orders/cancel.json", FakeResponse(
            200, {"CancelOrderResponse": {"orderId": 9}}))
        transport.route("get", "/v1/market/quote/SPY.json", FakeResponse(
            200, {"QuoteResponse": {"QuoteData": [
                {"Product": {"symbol": "SPY"},
                 "All": {"bid": 1.0, "ask": 1.2, "lastTrade": 1.1}}]}}))
        assert client.cancel_order(ACCOUNT, 9) is True
        assert client.get_quotes(["SPY"])["SPY"]["ask"] == 1.2
        assert transport.count("post", "preview") == 0  # never sent

    def test_circuit_breaker_boundary_exactly_2pct_breaches(self, tmp_path):
        """PINNED SEMANTICS: loss >= limit breaches — EXACTLY -2.000%
        blocks; -1.999% trades."""
        db_path = str(tmp_path / "cb.db")
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        breaker = DailyLossCircuitBreaker(switch, audit=audit,
                                          limit_pct=0.02)
        current = {"value": 98_001.0}  # -1.999%: allowed
        transport = ScriptedTransport()
        client = EtradeClient(
            FakeAuth(transport, db_path), kill_switch=switch,
            circuit_breaker=lambda: breaker.check(current["value"],
                                                  100_000.0),
            audit=audit, sleep_fn=lambda _s: None)
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [
                {"previewId": 5}]}}))
        request = build_equity_order("AAPL", "BUY", 1)
        preview = client.preview_order(ACCOUNT, request)  # passes at -1.999%
        assert preview["PreviewIds"] == [{"previewId": 5}]
        assert switch.engaged() is False

        current["value"] = 98_000.0  # EXACTLY -2.000%: inclusive breach
        with pytest.raises(KillSwitchEngaged):
            client.preview_order(ACCOUNT, request)
        assert switch.engaged() is True  # breach engaged it automatically
        breach_rows = audit.entries(event_type="circuit_breaker_breach")
        assert len(breach_rows) == 1
        assert breach_rows[0]["payload"]["loss_pct"] == pytest.approx(0.02)
        # And once engaged, even cancel-of-breaker the place path stays
        # blocked by the persistent switch.
        with pytest.raises(KillSwitchEngaged):
            client.place_order(ACCOUNT, request, [{"previewId": 5}])

    def test_spread_order_via_submit(self, tmp_path):
        """Multi-leg request flows through the same preview->place path."""
        client, transport, _, _ = self._client_with_switch(tmp_path)
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [
                {"previewId": 8}]}}))
        transport.route("post", "orders/place.json", FakeResponse(
            200, {"PlaceOrderResponse": {"OrderIds": [
                {"orderId": 600}]}}))
        legs = [
            {"symbol": "SPY", "call_put": "PUT", "strike": 440,
             "expiry": "2026-07-17", "action": "SELL_OPEN", "quantity": 2},
            {"symbol": "SPY", "call_put": "PUT", "strike": 430,
             "expiry": "2026-07-17", "action": "BUY_OPEN", "quantity": 2},
        ]
        result = client.submit_order(
            ACCOUNT, build_spread_order(legs, net_price=1.25))
        assert result["order_id"] == 600
        placed = transport.calls[-1]["json"]["PlaceOrderRequest"]
        assert placed["orderType"] == "SPREADS"
        assert placed["Order"][0]["priceType"] == "NET_CREDIT"


# ----------------------------------------------------------------------
# LiveEtradeBroker on top of the client (E3 rewrite)
# ----------------------------------------------------------------------
class TestLiveEtradeBroker:
    def _broker(self, harness):
        from brokers.live_trader import LiveEtradeBroker
        client, transport, audit = harness
        return LiveEtradeBroker(client, ACCOUNT), transport

    def test_place_equity_order_returns_order_id(self, harness):
        from core.models import Asset, AssetType, OrderType
        broker, transport = self._broker(harness)
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [
                {"previewId": 3}]}}))
        transport.route("post", "orders/place.json", FakeResponse(
            200, {"PlaceOrderResponse": {"OrderIds": [
                {"orderId": 700}]}}))
        order_id = broker.place_order(Asset("AAPL", AssetType.STOCK),
                                      OrderType.BUY, 10, 190.0)
        assert order_id == "700"
        sent = transport.calls[-1]["json"]["PlaceOrderRequest"]
        assert sent["orderType"] == "EQ"
        assert sent["PreviewIds"] == [{"previewId": 3}]

    def test_place_order_blocks_fat_finger_limit(self, harness):
        # Gap 2: the live broker REFUSES a limit > 50% from the current price.
        from brokers.live_trader import PriceSanityError
        from core.models import Asset, AssetType, OrderType
        broker, transport = self._broker(harness)
        broker.get_current_price = lambda _s: 100.0  # current ~100
        with pytest.raises(PriceSanityError):
            broker.place_order(Asset("AAPL", AssetType.STOCK),
                               OrderType.BUY, 1, 1000.0)  # 10x typo
        # Refused before any order endpoint was hit.
        assert not any("orders/" in c["url"] for c in transport.calls)

    def test_place_structure_blocks_absurd_net_price(self, harness):
        # Gap 2: structure net price beyond the defined-risk strike ceiling.
        from brokers.live_trader import PriceSanityError
        from core.models import Asset, AssetType
        broker, transport = self._broker(harness)
        legs = [
            {"asset": Asset("SPY", AssetType.PUT, 440.0, "2026-07-17"),
             "action": "SHORT"},
            {"asset": Asset("SPY", AssetType.PUT, 430.0, "2026-07-17"),
             "action": "BUY"},
        ]
        # net 5000 is a gross typo — blocked regardless of bound choice.
        with pytest.raises(PriceSanityError):
            broker.place_structure(legs, net_price=5000.0, contracts=1)
        assert not any("orders/" in c["url"] for c in transport.calls)

    def test_place_structure_blocks_realistic_magnitude_typo(self, harness):
        # Gap 2 (review): a 430/440 spread (width 10, ceiling 15) priced 440.0
        # instead of 4.40 is a 100x typo FAR BELOW any strike — the WIDTH
        # ceiling catches it where a max-strike ceiling (660) would not.
        from brokers.live_trader import PriceSanityError
        from core.models import Asset, AssetType
        broker, transport = self._broker(harness)
        legs = [
            {"asset": Asset("SPY", AssetType.PUT, 440.0, "2026-07-17"),
             "action": "SHORT"},
            {"asset": Asset("SPY", AssetType.PUT, 430.0, "2026-07-17"),
             "action": "BUY"},
        ]
        with pytest.raises(PriceSanityError):
            broker.place_structure(legs, net_price=440.0, contracts=1)
        assert not any("orders/" in c["url"] for c in transport.calls)

    def test_kill_switch_blocks_place_order_and_structure(self, tmp_path):
        # Phase 1 INTEGRATION: an engaged kill switch blocks BOTH order-entry
        # points on LiveEtradeBroker (place_order + place_structure) before any
        # order HTTP call. Pins the safety contract end-to-end through the
        # broker wrapper, not just the client unit.
        from brokers.live_trader import LiveEtradeBroker
        from core.models import Asset, AssetType, OrderType
        from utils.kill_switch import KillSwitch, KillSwitchEngaged
        db_path = str(tmp_path / "ks.db")
        audit = AuditLog(db_path, env="sandbox")
        kill_switch = KillSwitch(db_path, audit=audit)
        transport = ScriptedTransport()
        client = EtradeClient(FakeAuth(transport, db_path),
                              kill_switch=kill_switch, audit=audit,
                              sleep_fn=lambda _s: None)
        broker = LiveEtradeBroker(client, ACCOUNT)
        kill_switch.engage("test halt", actor="ops")

        with pytest.raises(KillSwitchEngaged):
            broker.place_order(Asset("AAPL", AssetType.STOCK),
                               OrderType.BUY, 10, 190.0)
        legs = [
            {"asset": Asset("SPY", AssetType.PUT, 440.0, "2026-07-17"),
             "action": "SHORT"},
            {"asset": Asset("SPY", AssetType.PUT, 430.0, "2026-07-17"),
             "action": "BUY"},
        ]
        with pytest.raises(KillSwitchEngaged):
            broker.place_structure(legs, net_price=1.15, contracts=2)
        # No order endpoint was ever hit (gated before any HTTP call).
        assert not any("orders/place" in c["url"] or "orders/preview" in c["url"]
                       for c in transport.calls)

    def test_place_structure_maps_desk_legs(self, harness):
        from core.models import Asset, AssetType
        broker, transport = self._broker(harness)
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [
                {"previewId": 4}]}}))
        transport.route("post", "orders/place.json", FakeResponse(
            200, {"PlaceOrderResponse": {"OrderIds": [
                {"orderId": 701}]}}))
        legs = [
            {"asset": Asset("SPY", AssetType.PUT, 440.0, "2026-07-17"),
             "action": "SHORT"},
            {"asset": Asset("SPY", AssetType.PUT, 430.0, "2026-07-17"),
             "action": "BUY"},
        ]
        assert broker.place_structure(legs, net_price=1.15,
                                      contracts=2) == "701"
        sent = transport.calls[-1]["json"]["PlaceOrderRequest"]
        instruments = sent["Order"][0]["Instrument"]
        assert [i["orderAction"] for i in instruments] == [
            "SELL_OPEN", "BUY_OPEN"]
        # BOTH quantity fields, mirroring E*TRADE's documented example
        # (sent as a pair in case 'quantity' is the authoritative one).
        assert all(i["orderedQuantity"] == 2 and i["quantity"] == 2
                   for i in instruments)
        assert sent["Order"][0]["priceType"] == "NET_CREDIT"

    def test_portfolio_status_uses_canonical_option_keys(self, harness):
        broker, transport = self._broker(harness)
        transport.route("get", "balance.json", FakeResponse(
            200, {"BalanceResponse": {"Computed": {
                "cashAvailableForInvestment": 5000.0,
                "RealTimeValues": {"totalAccountValue": 60_000.0}}}}))
        transport.route("get", "portfolio.json", FakeResponse(
            200, {"PortfolioResponse": {"AccountPortfolio": [{
                "Position": [
                    {"quantity": 100, "marketValue": 19_000.0,
                     "totalGain": 1.0, "totalGainPct": 0.1,
                     "Product": {"symbol": "AAPL", "securityType": "EQ"}},
                    {"quantity": -2, "marketValue": -220.0,
                     "totalGain": 30.0, "totalGainPct": 12.0,
                     "Product": {"symbol": "SPY", "securityType": "OPTN",
                                 "callPut": "PUT", "strikePrice": 440,
                                 "expiryYear": 2026, "expiryMonth": 7,
                                 "expiryDay": 17}},
                ]}]}}))
        transport.route("get", "orders.json", FakeResponse(
            200, {"OrdersResponse": {"Order": []}}))
        status = broker.get_portfolio_status()
        assert status["cash"] == 5000.0
        assert status["portfolio_value"] == 60_000.0
        symbols = {p["symbol"] for p in status["positions"]}
        assert symbols == {"AAPL", "SPY 2026-07-17 $440.0 put"}

    def test_get_current_price_and_cancel(self, harness):
        broker, transport = self._broker(harness)
        transport.route("get", "/v1/market/quote/SPY.json", FakeResponse(
            200, {"QuoteResponse": {"QuoteData": [
                {"Product": {"symbol": "SPY"},
                 "All": {"bid": 1.0, "ask": 1.2, "lastTrade": 1.1}}]}}))
        assert broker.get_current_price("SPY") == 1.1
        transport.route("put", "orders/cancel.json", FakeResponse(
            200, {"CancelOrderResponse": {"orderId": 700}}))
        assert broker.cancel_order("700") is True
        assert transport.calls[-1]["json"] == {
            "CancelOrderRequest": {"orderId": 700}}


# ----------------------------------------------------------------------
# Daily-loss gate: start-of-day capture + ET date roll (E5 wiring)
# ----------------------------------------------------------------------
def _balances_body(total_value):
    return {"BalanceResponse": {"Computed": {
        "RealTimeValues": {"totalAccountValue": total_value}}}}


class TestDailyLossGate:
    def _gate(self, tmp_path, values, times):
        """Gate over scripted account values and a frozen UTC clock
        (both lists consumed per call, last item sticky)."""
        from datetime import datetime, timezone

        from brokers.circuit_breaker import DailyLossGate

        db_path = str(tmp_path / "gate.db")
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        value_queue = list(values)
        time_queue = [datetime(*t, tzinfo=timezone.utc) for t in times]

        def value_fn():
            return (value_queue.pop(0) if len(value_queue) > 1
                    else value_queue[0])

        def clock():
            return (time_queue.pop(0) if len(time_queue) > 1
                    else time_queue[0])

        breaker = DailyLossCircuitBreaker(switch, audit=audit, clock=clock)
        return DailyLossGate(breaker, value_fn, clock=clock), switch, audit

    def test_baseline_resets_at_et_midnight_during_dst(self, tmp_path):
        """July (EDT, UTC-4): 23:59 ET vs 00:01 ET straddle 04:00 UTC.
        The new ET day recaptures the baseline, so yesterday's drawdown
        does not bleed into today."""
        gate, switch, audit = self._gate(
            tmp_path,
            values=[100_000.0, 90_000.0, 88_100.0],
            times=[(2026, 7, 11, 3, 59),    # Jul 10 23:59 ET — day 1
                   (2026, 7, 11, 4, 1),     # Jul 11 00:01 ET — day 2
                   (2026, 7, 11, 14, 0)])   # Jul 11 10:00 ET — day 2
        assert gate()["breached"] is False  # day-1 baseline = 100k
        # New ET day: 90k becomes the NEW baseline (not a -10% breach).
        result = gate()
        assert result["breached"] is False
        assert result["start_of_day_value"] == 90_000.0
        assert switch.engaged() is False
        # Same ET day, -2.11% vs the 90k baseline: breach.
        assert gate()["breached"] is True
        assert switch.engaged() is True
        captures = audit.entries(event_type="start_of_day_captured",
                                 ascending=True)
        assert [c["payload"]["value"] for c in captures] == [100_000.0,
                                                             90_000.0]
        assert [c["payload"]["date"] for c in captures] == ["2026-07-10",
                                                            "2026-07-11"]

    def test_utc_date_roll_alone_does_not_reset_in_january(self, tmp_path):
        """January (EST, UTC-5): 03:00 UTC on the 16th is still 22:00 ET
        on the 15th — the UTC date roll must NOT reset the baseline, so
        the day's -2.1% loss breaches against the morning's value."""
        gate, switch, _ = self._gate(
            tmp_path,
            values=[100_000.0, 97_900.0],
            times=[(2026, 1, 15, 18, 0),    # Jan 15 13:00 ET
                   (2026, 1, 16, 3, 0)])    # Jan 15 22:00 ET — SAME ET day
        assert gate()["breached"] is False
        assert gate()["breached"] is True   # same baseline, -2.1%
        assert switch.engaged() is True

    def test_baseline_reused_across_restart(self, tmp_path):
        """Gap 4: a mid-day process RESTART reuses the morning baseline from the
        audit, not the drawn-down value — so a pre-restart drop still breaches."""
        from datetime import datetime, timezone

        from brokers.circuit_breaker import DailyLossGate

        db_path = str(tmp_path / "restart.db")
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        when = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)  # 13:00 ET
        clock = lambda: when  # noqa: E731

        # Process 1: capture the morning baseline at 100k.
        g1 = DailyLossGate(
            DailyLossCircuitBreaker(switch, audit=audit, clock=clock),
            value_fn=lambda: 100_000.0, clock=clock)
        assert g1()["breached"] is False
        assert g1.start_of_day_value == 100_000.0

        # Process 2 (restart): account already down to 97k. A FRESH gate must
        # reuse the 100k baseline -> -3% -> breach, not re-baseline to 97k.
        g2 = DailyLossGate(
            DailyLossCircuitBreaker(switch, audit=audit, clock=clock),
            value_fn=lambda: 97_000.0, clock=clock)
        result = g2()
        assert result["start_of_day_value"] == 100_000.0  # reused, not 97k
        assert result["breached"] is True
        assert switch.engaged() is True
        # No duplicate capture for the day.
        caps = audit.entries(event_type="start_of_day_captured")
        assert len(caps) == 1

    def test_baseline_persistence_is_env_scoped(self, tmp_path):
        """Gap 4 (review): a PRODUCTION gate must NOT reuse a SANDBOX baseline
        captured the same day on the shared db — env isolation."""
        from datetime import datetime, timezone

        from brokers.circuit_breaker import DailyLossGate

        db_path = str(tmp_path / "shared.db")
        when = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)
        clock = lambda: when  # noqa: E731

        sb_audit = AuditLog(db_path, env="sandbox")
        sb_switch = KillSwitch(db_path, audit=sb_audit)
        DailyLossGate(DailyLossCircuitBreaker(sb_switch, audit=sb_audit,
                                              clock=clock),
                      value_fn=lambda: 100_000.0, clock=clock)()  # capture sb

        # A production gate on the SAME db, same day: it must capture its OWN
        # baseline (90k), not reuse the sandbox 100k.
        prod_audit = AuditLog(db_path, env="production")
        prod_switch = KillSwitch(db_path, audit=prod_audit)
        prod_gate = DailyLossGate(
            DailyLossCircuitBreaker(prod_switch, audit=prod_audit, clock=clock),
            value_fn=lambda: 90_000.0, clock=clock)
        result = prod_gate()
        assert result["start_of_day_value"] == 90_000.0   # own, not 100k
        assert result["breached"] is False                # 90k baseline == 90k


# ----------------------------------------------------------------------
# LiveEtradeBroker auto-wires the daily-loss gate (the production path)
# ----------------------------------------------------------------------
class TestLiveBrokerDailyLossWiring:
    def _live_broker(self, tmp_path, **kwargs):
        from brokers.live_trader import LiveEtradeBroker
        transport = ScriptedTransport()
        db_path = str(tmp_path / "wiring.db")
        audit = AuditLog(db_path, env="sandbox")
        switch = KillSwitch(db_path, audit=audit)
        broker = LiveEtradeBroker(FakeAuth(transport, db_path), ACCOUNT,
                                  kill_switch=switch, audit=audit, **kwargs)
        return broker, transport, switch, audit

    def test_gate_wired_into_client_by_default(self, tmp_path):
        broker, _, _, _ = self._live_broker(tmp_path)
        assert broker.circuit_breaker is not None
        # The SAME gate is the client's injected pre-trade hook.
        assert broker.client.circuit_breaker is broker.circuit_breaker

    def test_daily_loss_breach_blocks_preview_and_engages_switch(
            self, tmp_path):
        """The end-to-end rail: first order of the day captures the
        baseline and passes; after a -3% drop the next preview is BLOCKED
        and the kill switch is engaged — no frontend wiring required."""
        broker, transport, switch, audit = self._live_broker(tmp_path)
        transport.route("get", "balance.json",
                        FakeResponse(200, _balances_body(100_000.0)),
                        FakeResponse(200, _balances_body(97_000.0)))
        transport.route("post", "orders/preview.json", FakeResponse(
            200, {"PreviewOrderResponse": {"PreviewIds": [
                {"previewId": 9}]}}))
        request = build_equity_order("AAPL", "BUY", 1)
        preview = broker.client.preview_order(ACCOUNT, request)
        assert preview["PreviewIds"] == [{"previewId": 9}]
        captures = audit.entries(event_type="start_of_day_captured")
        assert captures[0]["payload"]["value"] == 100_000.0

        with pytest.raises(KillSwitchEngaged):
            broker.client.preview_order(ACCOUNT, request)
        assert switch.engaged() is True
        assert transport.count("post", "preview") == 1  # second never sent
        breach = audit.entries(event_type="circuit_breaker_breach")[0]
        assert breach["payload"]["loss_pct"] == pytest.approx(0.03)

    def test_explicit_none_disables_the_gate(self, tmp_path):
        broker, _, _, _ = self._live_broker(tmp_path, circuit_breaker=None)
        assert broker.circuit_breaker is None
        assert broker.client.circuit_breaker is None

    def test_prebuilt_client_gate_is_surfaced_not_replaced(self, harness):
        from brokers.live_trader import LiveEtradeBroker
        client, _, _ = harness
        sentinel_gate = lambda: {"breached": False}  # noqa: E731
        client.circuit_breaker = sentinel_gate
        broker = LiveEtradeBroker(client, ACCOUNT)
        assert broker.circuit_breaker is sentinel_gate
        assert broker.client.circuit_breaker is sentinel_gate
