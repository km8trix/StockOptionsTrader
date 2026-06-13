"""Sandbox soak — READ + PREVIEW only (places no orders).

Drives the already-connected E*TRADE sandbox token (persisted in the
SQLite token store by the GUI OAuth flow) through the read-only and
non-binding preview endpoints, to validate the live wiring end to end:
accounts -> balances -> quotes -> iron-condor preview. The preview step
settles the open question from the Phase 9 quant review: does E*TRADE
accept orderType 'SPREADS' for a 4-leg iron condor, or require its
'IRON_CONDOR' enum?

Run (sandbox only, network gate required):
    ETRADE_ALLOW_NETWORK=1 ETRADE_ENV=sandbox .venv/bin/python \
        scripts/sandbox_soak.py
"""
from __future__ import annotations

import datetime as dt
import sys

from brokers.etrade_auth import EtradeAuthManager
from brokers.etrade_client import EtradeClient, build_spread_order


def _redact_acct(num: str) -> str:
    s = str(num or "")
    return ("*" * max(0, len(s) - 4)) + s[-4:] if s else "(none)"


def main() -> int:
    mgr = EtradeAuthManager()
    status = mgr.status()
    print(f"auth state: {status['state']} | env: {status['env']}")
    if status["state"] != "connected":
        print("NOT CONNECTED — complete the OAuth flow first.")
        return 2
    client = EtradeClient(mgr)

    print("\n=== 1. ACCOUNTS ===")
    accounts = client.list_accounts()
    if not accounts:
        print("no accounts returned")
        return 3
    for a in accounts:
        print(f"  {a.get('accountType')} {a.get('accountDesc')} "
              f"acct={_redact_acct(a.get('accountId'))} "
              f"idKey={a.get('accountIdKey')} status={a.get('accountStatus')}")
    key = accounts[0]["accountIdKey"]
    print(f"  -> using accountIdKey: {key}")

    print("\n=== 2. BALANCES ===")
    bal = client.get_balances(key)
    computed = bal.get("Computed", {})
    print(f"  net account value: {computed.get('RealTimeValues', {}).get('totalAccountValue')}")
    print(f"  cash available for investment: {computed.get('cashAvailableForInvestment')}")

    print("\n=== 3. QUOTES (SPY, AAPL) ===")
    quotes = client.get_quotes(["SPY", "AAPL"])
    for sym, q in quotes.items():
        print(f"  {sym}: bid={q.get('bid')} ask={q.get('ask')} last={q.get('last')}")
    spy = quotes.get("SPY", {})
    spot = spy.get("last") or spy.get("ask") or 500.0

    print("\n=== 4. IRON CONDOR PREVIEW (no order placed) ===")
    # Strikes ~1 / ~1.5 std out, rounded to $5; July monthly expiry.
    base = round(spot / 5.0) * 5.0
    expiry = dt.date(2026, 7, 17)
    sp_short, sp_long = base - 15, base - 25
    sc_short, sc_long = base + 15, base + 25
    legs = [
        {"symbol": "SPY", "call_put": "PUT", "strike": sp_short,
         "expiry": expiry, "action": "SELL_OPEN", "quantity": 1},
        {"symbol": "SPY", "call_put": "PUT", "strike": sp_long,
         "expiry": expiry, "action": "BUY_OPEN", "quantity": 1},
        {"symbol": "SPY", "call_put": "CALL", "strike": sc_short,
         "expiry": expiry, "action": "SELL_OPEN", "quantity": 1},
        {"symbol": "SPY", "call_put": "CALL", "strike": sc_long,
         "expiry": expiry, "action": "BUY_OPEN", "quantity": 1},
    ]
    print(f"  spot~{spot} | condor {sp_long}/{sp_short}P  {sc_short}/{sc_long}C "
          f"exp {expiry} | net credit $1.00 | orderType=SPREADS")
    req = build_spread_order(legs, net_price=1.00)
    try:
        preview = client.preview_order(key, req)
        print("  PREVIEW ACCEPTED — SPREADS orderType works for a 4-leg condor.")
        pids = (preview.get("PreviewOrderResponse", {}).get("PreviewIds")
                or preview.get("PreviewIds") or preview)
        print(f"  previewIds: {pids}")
    except Exception as e:  # noqa: BLE001 — soak reports every failure verbatim
        print(f"  PREVIEW REJECTED: {type(e).__name__}: {e}")
        print("  -> if this names orderType/SPREADS, branch build_spread_order "
              "to IRON_CONDOR for 4-leg call+put condors and re-pin the golden "
              "fixture.")

    print("\n=== SOAK COMPLETE (read + preview only; no orders placed) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
