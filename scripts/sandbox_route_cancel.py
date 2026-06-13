"""Gold-standard live confirmation: place AND cancel entirely through the
GUI routes — the cancel goes through the newly-fixed account_id_key path
(/orders/<id>/cancel with {account_id_key}), not the client backstop.

A client-side cancel still runs in finally as a SAFETY NET so a placed
order is never left working, but the test's pass/fail is the ROUTE cancel.

Run: ETRADE_ALLOW_NETWORK=1 ETRADE_ENV=sandbox PYTHONPATH=<root> \
     .venv/bin/python scripts/sandbox_route_cancel.py
"""
from __future__ import annotations

import json
import sys
import time
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
    st, body = _call("GET", "/api/live/accounts")
    accts = body.get("accounts", [])
    if st != 200 or not accts:
        print(f"accounts failed: HTTP {st} {body}")
        return 2
    key = accts[0]["accountIdKey"]
    print(f"1. accounts -> HTTP {st}, accountIdKey={key}")

    time.sleep(2)
    st, body = _call("POST", "/api/live/order/preview", {
        "account_id_key": key, "kind": "equity",
        "symbol": "AAPL", "side": "BUY", "quantity": 1, "limit_price": 1.00,
    })
    if st != 200 or not body.get("order_ref"):
        print(f"2. preview failed: HTTP {st} {body}")
        return 3
    order_ref = body["order_ref"]
    print(f"2. preview -> HTTP {st}, order_ref set")

    order_id = None
    route_cancel_ok = False
    try:
        time.sleep(2)
        st, body = _call("POST", "/api/live/order/place",
                         {"order_ref": order_ref})
        if st != 200:
            print(f"3. place failed: HTTP {st} {body}")
            return 4
        order_id = body.get("order", {}).get("orderId")
        print(f"3. place -> HTTP {st}, orderId={order_id}")

        # THE TEST: cancel through the ROUTE with account_id_key.
        time.sleep(2)
        st, body = _call("POST", f"/api/live/orders/{order_id}/cancel",
                         {"account_id_key": key})
        print(f"4. ROUTE cancel (account_id_key) -> HTTP {st} | "
              f"{json.dumps(body)[:200]}")
        route_cancel_ok = (st == 200)
    finally:
        if order_id is not None and not route_cancel_ok:
            # Safety net only if the route cancel did not succeed.
            print("   route cancel did not confirm — running client backstop")
            try:
                from brokers.etrade_auth import EtradeAuthManager
                from brokers.etrade_client import EtradeClient
                ok = EtradeClient(EtradeAuthManager()).cancel_order(key, order_id)
                print(f"   client backstop cancel -> {ok}")
            except Exception as e:  # noqa: BLE001
                print(f"   backstop error: {type(e).__name__}: {e}")
                print("   !! VERIFY ORDER IS CANCELLED IN THE E*TRADE UI !!")

    print("\nRESULT:", "ROUTE CANCEL CONFIRMED" if route_cancel_ok
          else "route cancel did NOT confirm — see above")
    return 0 if route_cancel_ok else 5


if __name__ == "__main__":
    sys.exit(main())
