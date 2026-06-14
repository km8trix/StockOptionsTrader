"""
Live-vs-backtest execution parity (Phase 3 Step 5) — READ-ONLY.

Replays AUDITED live fills through the backtest cost model and reports the drift
between realized execution cost and what the backtest would have modeled. A
backtest is only trustworthy if its cost assumptions match live reality; this
closes that loop.

DATA SOURCE: the append-only AuditLog (utils.audit). Each live fill leaves an
``execution_report`` row. Step 5 additionally records arrival_mid + asset_type +
quantity on that row so the replay is exact and per-asset-correct. LEGACY rows
(pre-Step-5) lack those fields, so the harness degrades gracefully: arrival_mid
is RECOVERED from ``avg_fill -/+ shortfall_per_unit`` (the executor's signed
shortfall convention), asset_type defaults to 'stock', and quantity is
backfilled from the paired ``execution_start`` row.

WHAT IS / ISN'T MEASURABLE:
  - Slippage / spread drift: FULLY measured. Realized ``shortfall_per_unit`` vs
    the cost model's expected per-unit cost (stock: arrival_mid*slippage_bps/1e4;
    option: max(arrival_mid*spread_pct, MIN_OPTION_HAIRCUT)).
  - Commission: the live executor/broker does NOT surface per-fill commission
    today, so ACTUAL commission is unavailable. The harness reports the MODELED
    commission the backtest would charge (stock: notional*commission; option:
    per_contract_commission*contracts) and flags actual as N/A. When the broker
    later records commission into the execution_report payload, the harness
    picks it up and computes true commission drift automatically.

SIGN CONVENTION: a drift > 0 means LIVE was WORSE than the model assumed (the
backtest was too optimistic about costs).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

#: Intent actions that execute as a BUY (price paid) vs a SELL (price received).
_BUY_ACTIONS = {"BUY", "COVER"}
_SELL_ACTIONS = {"SELL", "SHORT"}
_OPTION_TYPES = {"call", "put"}

_CAVEAT = (
    "Drift > 0 means live execution was WORSE than the backtest cost model "
    "assumed. Slippage/spread drift is fully measured (realized shortfall vs "
    "the per-asset cost model). Commission is MODELED (the cost the backtest "
    "would charge); actual live commission is not surfaced by the executor/"
    "broker yet, so commission drift stays N/A until that is recorded into the "
    "audit payload. Legacy (pre-Step-5) rows have arrival_mid recovered from "
    "avg_fill +/- shortfall and asset_type defaulted to stock. Per-unit drift "
    "figures mix stock ($/share) and option ($/contract-share) scales; the "
    "dollar totals (which apply the x100 option multiplier) are the comparable "
    "figures."
)


def _side_of(action: Optional[str]) -> Optional[str]:
    """Map an intent action to BUY/SELL execution side, or None if unknown."""
    if action in _BUY_ACTIONS:
        return "BUY"
    if action in _SELL_ACTIONS:
        return "SELL"
    return None


def _price_fill(entry: Dict, payload: Dict, start_quantity: Optional[int],
                params: Dict) -> Optional[Dict]:
    """Price one execution_report against the cost model, or None to skip
    (unfilled / errored / non-recoverable reference price)."""
    avg_fill = payload.get("avg_fill")
    shortfall = payload.get("shortfall_per_unit")
    action = payload.get("action")
    if avg_fill is None or shortfall is None or action is None:
        return None
    avg_fill = float(avg_fill)
    shortfall = float(shortfall)
    # Never trust an audit payload to be finite (corrupt/hand-written row, or a
    # third-party executor): a non-finite value would poison the whole report
    # and break jsonify. Skip it, exactly like the non-recoverable-mid path.
    if not (math.isfinite(avg_fill) and math.isfinite(shortfall)):
        return None
    side = _side_of(action)

    # Reference price: recorded (Step 5) or recovered from the signed shortfall.
    arrival_mid = payload.get("arrival_mid")
    recovered = False
    if arrival_mid is None:
        if side is None:
            return None
        arrival_mid = (avg_fill - shortfall) if side == "BUY" \
            else (avg_fill + shortfall)
        recovered = True
    else:
        arrival_mid = float(arrival_mid)
    if not math.isfinite(arrival_mid) or arrival_mid <= 0:
        return None

    asset_type = payload.get("asset_type") or "stock"
    is_option = asset_type in _OPTION_TYPES
    multiplier = 100 if is_option else 1

    # Prefer the ACTUAL filled size (a partial fill banks less than the
    # intended quantity, so intended-size totals would overstate drift and
    # commission); fall back to intended quantity, then to the paired
    # execution_start (legacy rows).
    quantity = payload.get("filled_quantity")
    if quantity is None:
        quantity = payload.get("quantity")
    if quantity is None:
        quantity = start_quantity
    qty = int(quantity) if quantity is not None else None

    # Expected per-unit cost from the backtest cost model.
    if is_option:
        expected = max(arrival_mid * params["spread_pct"],
                       params["min_option_haircut"])
    else:
        expected = arrival_mid * params["slippage_bps"] / 1e4
    slippage_drift = shortfall - expected
    slippage_drift_total = (slippage_drift * qty * multiplier
                            if qty is not None else None)

    # Modeled commission (what the backtest would charge for this fill).
    if is_option:
        modeled_commission = (qty * params["per_contract_commission"]
                              if qty is not None else None)
    else:
        slip = params["slippage_bps"] / 1e4
        modeled_fill = (arrival_mid * (1 + slip) if side == "BUY"
                        else arrival_mid * (1 - slip))
        modeled_commission = (qty * modeled_fill * params["stock_commission"]
                              if qty is not None else None)

    actual = payload.get("commission")
    actual_commission = float(actual) if actual is not None else None
    commission_drift = (actual_commission - modeled_commission
                        if actual_commission is not None
                        and modeled_commission is not None else None)

    return {
        "seq": entry.get("seq"),
        "symbol": payload.get("symbol"),
        "action": action,
        "asset_type": asset_type,
        "quantity": qty,
        "arrival_mid": arrival_mid,
        "arrival_mid_recovered": recovered,
        "avg_fill": avg_fill,
        "realized_shortfall": shortfall,
        "expected_shortfall": expected,
        "slippage_drift": slippage_drift,
        "slippage_drift_total": slippage_drift_total,
        "modeled_commission": modeled_commission,
        "actual_commission": actual_commission,
        "commission_drift": commission_drift,
    }


def _summarize(fills: List[Dict]) -> Dict:
    """Aggregate per-fill parity into totals + a per-asset-type breakdown."""
    if not fills:
        return {
            "n_priced": 0,
            "total_slippage_drift": 0.0,
            "mean_realized_shortfall": None,
            "mean_expected_shortfall": None,
            "mean_slippage_drift_per_unit": None,
            "worst_slippage_drift_per_unit": None,
            "total_modeled_commission": 0.0,
            "total_actual_commission": None,
            "total_commission_drift": None,
            "by_asset_type": {},
        }

    drifts = [f["slippage_drift"] for f in fills]
    realized = [f["realized_shortfall"] for f in fills]
    expected = [f["expected_shortfall"] for f in fills]
    drift_totals = [f["slippage_drift_total"] for f in fills
                    if f["slippage_drift_total"] is not None]
    modeled_comms = [f["modeled_commission"] for f in fills
                     if f["modeled_commission"] is not None]
    actual_comms = [f["actual_commission"] for f in fills
                    if f["actual_commission"] is not None]
    comm_drifts = [f["commission_drift"] for f in fills
                   if f["commission_drift"] is not None]

    by: Dict[str, Dict] = {}
    for f in fills:
        bucket = by.setdefault(f["asset_type"], {
            "n_fills": 0, "total_slippage_drift": 0.0,
            "total_modeled_commission": 0.0})
        bucket["n_fills"] += 1
        if f["slippage_drift_total"] is not None:
            bucket["total_slippage_drift"] += f["slippage_drift_total"]
        if f["modeled_commission"] is not None:
            bucket["total_modeled_commission"] += f["modeled_commission"]

    return {
        # n_priced = fills carrying a quantity (the dollar totals cover only
        # these; n_fills can be larger when a legacy fill had no recoverable
        # quantity).
        "n_priced": len(drift_totals),
        "total_slippage_drift": sum(drift_totals),
        "mean_realized_shortfall": sum(realized) / len(realized),
        "mean_expected_shortfall": sum(expected) / len(expected),
        "mean_slippage_drift_per_unit": sum(drifts) / len(drifts),
        "worst_slippage_drift_per_unit": max(drifts),
        "total_modeled_commission": sum(modeled_comms),
        "total_actual_commission": sum(actual_comms) if actual_comms else None,
        "total_commission_drift": sum(comm_drifts) if comm_drifts else None,
        "by_asset_type": by,
    }


def compute_parity(entries: List[Dict], *, slippage_bps: float = 5.0,
                   spread_pct: float = 0.02, min_option_haircut: float = 0.05,
                   stock_commission: float = 0.001,
                   per_contract_commission: float = 0.65) -> Dict:
    """Replay audited live fills through the backtest cost model.

    ``entries`` is a list of AuditLog entry dicts in ASCENDING seq order
    (execution_start / execution_report rows are consumed; others ignored). The
    cost-model parameters default to the BacktestEngine defaults so the parity
    is against a vanilla backtest; pass the parameters of the specific backtest
    you are validating to compare against that run.

    Returns a report dict: per-fill rows plus a summary (see module docstring).
    """
    params = {
        "slippage_bps": slippage_bps,
        "spread_pct": spread_pct,
        "min_option_haircut": min_option_haircut,
        "stock_commission": stock_commission,
        "per_contract_commission": per_contract_commission,
    }
    # FIFO of pending execution_start quantities per (symbol, action), used to
    # backfill quantity for legacy report rows that do not carry it.
    pending_starts: Dict = {}
    fills: List[Dict] = []
    n_skipped = 0

    for entry in entries:
        event_type = entry.get("event_type")
        payload = entry.get("payload") or {}
        if event_type == "execution_start":
            key = (payload.get("symbol"), payload.get("action"))
            pending_starts.setdefault(key, []).append(payload.get("quantity"))
            continue
        if event_type != "execution_report":
            continue
        key = (payload.get("symbol"), payload.get("action"))
        start_quantity = None
        # Only consume a pending start when this report cannot supply its own
        # quantity (legacy rows) — so an enriched report does not steal a start
        # a later legacy report needs. FIFO pairing assumes the executor's 1:1
        # start->report invariant (utils/live_session._execute_intent).
        if (payload.get("filled_quantity") is None
                and payload.get("quantity") is None):
            queue = pending_starts.get(key)
            if queue:
                start_quantity = queue.pop(0)
        fill = _price_fill(entry, payload, start_quantity, params)
        if fill is None:
            n_skipped += 1
        else:
            fills.append(fill)

    return {
        "n_fills": len(fills),
        "n_skipped": n_skipped,
        "commission_available": any(f["actual_commission"] is not None
                                    for f in fills),
        "params": params,
        "fills": fills,
        "summary": _summarize(fills),
        "caveat": _CAVEAT,
    }


def parity_report(audit, *, limit: int = 10000, **params) -> Dict:
    """Convenience wrapper: read execution_start/report rows off an AuditLog and
    compute parity. ``params`` are forwarded to :func:`compute_parity`."""
    # Most-recent `limit` rows of each type (newest first) so a long-lived log
    # reports CURRENT execution quality, not its oldest fills; then re-sort
    # ascending so the FIFO start->report pairing runs in chronological order.
    starts = audit.entries(event_type="execution_start", limit=limit,
                           ascending=False)
    reports = audit.entries(event_type="execution_report", limit=limit,
                            ascending=False)
    entries = sorted([*starts, *reports], key=lambda e: e.get("seq", 0))
    return compute_parity(entries, **params)
