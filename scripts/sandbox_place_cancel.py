"""Sandbox place+cancel test — drives the live GUI routes, then GUARANTEES
the order is cancelled (far-from-market sandbox limit; fake money).

Flow:
  1. GET  /api/live/accounts                     (route)  -> accountIdKey
  2. POST /api/live/order/preview  equity         (route)  -> order_ref
  3. POST /api/live/order/place    {order_ref}    (route)  -> orderId
  4. cancel: try POST /orders/<id>/cancel (route), then ALWAYS ensure the
     order is cancelled via client.cancel_order directly, and verify via
     order_status. Cancel runs in a finally block so a placed order is
     never left working.

Run: ETRADE_ALLOW_NETWORK=1 ETRADE_ENV=sandbox PYTHONPATH=<root> \
     .venv/bin/python scripts/sandbox_place_cancel.py
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:5001"


def _call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    print("=== 1. ACCOUNTS (route) ===")
    st, body = _call("GET", "/api/live/accounts")
    accts = body.get("accounts", [])
    print(f"  HTTP {st} | {len(accts)} accounts")
    if st != 200 or not accts:
        print("  cannot proceed:", body)
        return 2
    key = accts[0]["accountIdKey"]
    print(f"  using accountIdKey={key} ({accts[0].get('accountType')} "
          f"{accts[0].get('accountId_masked')})")

    print("\n=== 2. PREVIEW (route) — BUY 1 AAPL @ $1.00 LIMIT (far from market) ===")
    st, body = _call("POST", "/api/live/order/preview", {
        "account_id_key": key, "kind": "equity",
        "symbol": "AAPL", "side": "BUY", "quantity": 1, "limit_price": 1.00,
    })
    print(f"  HTTP {st} | previewIds={body.get('previewIds')} "
          f"order_ref={'set' if body.get('order_ref') else None}")
    if st != 200 or not body.get("order_ref"):
        print("  preview failed:", body)
        return 3
    order_ref = body["order_ref"]

    order_id = None
    try:
        print("\n=== 3. PLACE (route) — {order_ref} ===")
        st, body = _call("POST", "/api/live/order/place",
                         {"order_ref": order_ref})
        print(f"  HTTP {st} | {json.dumps(body)[:300]}")
        if st == 200:
            order_id = body.get("order", {}).get("orderId")
            print(f"  PLACED orderId={order_id} "
                  f"status={body.get('order', {}).get('status')}")
        else:
            print("  place did not succeed (no order to cancel).")
            return 4
    finally:
        if order_id is not None:
            print("\n=== 4. CANCEL ===")
            st, body = _call("POST", f"/api/live/orders/{order_id}/cancel")
            print(f"  route cancel -> HTTP {st} | {json.dumps(body)[:200]}")
            # Guaranteed backstop via the client (route uses a separate
            # broker registry that this flow never populated).
            try:
                import os
                from brokers.etrade_auth import EtradeAuthManager
                from brokers.etrade_client import EtradeClient
                client = EtradeClient(EtradeAuthManager())
                ok = client.cancel_order(key, order_id)
                print(f"  client.cancel_order -> {ok}")
                status = client.order_status(key, order_id)
                print(f"  order_status -> {json.dumps(status)[:200] if status else status}")
            except Exception as e:  # noqa: BLE001
                print(f"  backstop cancel error: {type(e).__name__}: {e}")
                print("  !! VERIFY THIS ORDER IS CANCELLED IN THE E*TRADE UI !!")

    print("\n=== PLACE/CANCEL TEST COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
