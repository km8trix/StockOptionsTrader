"""
Live E*TRADE execution broker (Phase 9 rewrite).

LiveEtradeBroker implements the full ExecutionBroker ABC on top of the
typed EtradeClient: preview->place equity and option orders, cancel,
portfolio status from live balances/positions, quotes, plus
place_structure() for multi-leg (Level 3) defined-risk spreads.

CONSTRUCTOR CHANGE (frontend coordination, contract C16/C20):

    LiveEtradeBroker(auth, account_id_key, kill_switch=None, audit=None,
                     circuit_breaker=_AUTO)

where ``auth`` is EITHER an EtradeAuthManager (the broker builds its own
EtradeClient over it, wiring the kill switch through) OR an already-built
EtradeClient (it is used as-is). The old Phase-1 signature
(consumer_key/consumer_secret/access_token/access_secret/account_id_key)
is GONE — token lifecycle now lives in EtradeAuthManager, never in env
vars. gui/routes/api_trading.py's construction site must pass
``auth=EtradeAuthManager(...)`` (or a shared client) + account_id_key.

DAILY-LOSS CIRCUIT BREAKER (E5) — wired HERE by default: when the broker
builds its own client from an auth manager AND a kill switch was given,
it constructs a DailyLossCircuitBreaker + DailyLossGate over the live
balance endpoint and injects the gate as the client's ``circuit_breaker``
hook, so every preview/place enforces the -2% daily-loss rail with NO
frontend change (api_trading already passes the shared kill switch).
Pass ``circuit_breaker=None`` to disable explicitly, or your own zero-arg
gate callable to override. The broker exposes the wired gate as
``self.circuit_breaker`` so LiveTradingSession can evaluate the same rail
per evaluation cycle. Each gate call costs one balance GET (two extra
GETs per preview+place pair) — chatty but cheap insurance.

THE PHASE-9 RULE: no transport is created here. Every HTTP interaction
flows through the EtradeClient -> EtradeAuthManager -> injected session
factory; tests inject fakes and never touch the network.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union, cast

from brokers.base import ExecutionBroker
from brokers.circuit_breaker import (DailyLossCircuitBreaker, DailyLossGate,
                                     extract_account_value)
from brokers.etrade_auth import EtradeAuthManager
from brokers.etrade_client import (EtradeClient, build_equity_order,
                                   build_option_order, build_spread_order)
from core.models import Asset, AssetType, OrderType
from utils import metrics
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

#: Sentinel: "auto-wire the daily-loss gate when a kill switch is given".
#: Distinct from None, which disables the rail explicitly.
_AUTO = object()

#: Default fat-finger threshold for the LIVE broker: a limit price more than
#: 50% from the current price (or a spread net price beyond the same fraction
#: of its defined-risk strike ceiling) is REFUSED. Generous so a legitimately
#: far-from-market limit still places; only egregious typos block. None on a
#: LiveEtradeBroker disables the rail.
DEFAULT_PRICE_SANITY_THRESHOLD = 0.5


class PriceSanityError(ValueError):
    """A limit / spread net price is implausibly far from the market — almost
    certainly a fat-finger. Raised by the LIVE broker to BLOCK the order before
    it reaches E*TRADE (the base ExecutionBroker._check_price_sanity stays
    warn-only for paper)."""

#: Desk structure-leg action -> E*TRADE orderAction, by lifecycle stage.
#: Desk legs say SHORT (sell-to-open) / BUY (buy-to-open); closing flips.
_ENTRY_ACTIONS = {"SHORT": "SELL_OPEN", "BUY": "BUY_OPEN"}
_CLOSE_ACTIONS = {"SHORT": "BUY_CLOSE", "BUY": "SELL_CLOSE"}


def _option_position_key(product: Dict) -> str:
    """Canonical key for an option position, mirroring str(Asset):
    'SPY 2026-07-17 $440.0 put' — so reconciliation compares the exact
    contract, never just the underlying."""
    expiry = (f"{product['expiryYear']:04d}-{product['expiryMonth']:02d}"
              f"-{product['expiryDay']:02d}")
    right = "call" if product.get("callPut", "").upper() == "CALL" else "put"
    return (f"{product['symbol']} {expiry} "
            f"${float(product['strikePrice'])} {right}")


class LiveEtradeBroker(ExecutionBroker):
    """Live execution against E*TRADE through the typed client.

    Args:
        auth: EtradeAuthManager OR EtradeClient (see module docstring).
        account_id_key: E*TRADE accountIdKey routing every call.
        kill_switch: optional KillSwitch passed into a client built from
            an auth manager (ignored when a prebuilt client is supplied —
            that client already carries its own gates).
        audit: optional AuditLog for the same case.
        circuit_breaker: the daily-loss rail for a client built from an
            auth manager. Default (_AUTO) wires a DailyLossCircuitBreaker
            + DailyLossGate over live balances whenever kill_switch is
            present; None disables; a zero-arg callable overrides.
            Ignored when a prebuilt client is supplied (its own
            ``circuit_breaker`` hook, if any, is surfaced instead).
    """

    def __init__(self, auth: Union[EtradeAuthManager, EtradeClient],
                 account_id_key: str,
                 kill_switch: Optional[KillSwitch] = None,
                 audit: Optional[AuditLog] = None,
                 circuit_breaker=_AUTO,
                 price_sanity_threshold: Optional[float] =
                 DEFAULT_PRICE_SANITY_THRESHOLD,
                 reference_price_fn: Optional[Callable[[str], Optional[float]]]
                 = None,
                 quote_divergence_threshold: float = 0.05):
        account_id_key = str(account_id_key or '').strip()
        if not account_id_key:
            raise ValueError(
                'account_id_key is required for live order routing')
        if isinstance(auth, EtradeClient) or hasattr(auth, "preview_order"):
            # prebuilt (or fake) client — narrowed by the isinstance/duck check
            self.client: EtradeClient = cast(EtradeClient, auth)
            # Surface the client's own gate (if any) for the session.
            self.circuit_breaker = getattr(self.client, "circuit_breaker",
                                           None)
        else:
            self.client = EtradeClient(auth, kill_switch=kill_switch,
                                       audit=audit)
            if circuit_breaker is _AUTO:
                circuit_breaker = (self._build_daily_loss_gate(kill_switch,
                                                               audit)
                                   if kill_switch is not None else None)
            self.circuit_breaker = circuit_breaker
            # Inject AFTER construction: the gate reads balances through
            # this very client (read-only GETs are never gated).
            self.client.circuit_breaker = circuit_breaker
        self.account_id_key = account_id_key
        # BLOCKING fat-finger guard, default 50% (None disables). Refuses a
        # single-instrument limit or a spread net price implausibly far from
        # the market before it reaches E*TRADE.
        self.price_sanity_threshold = price_sanity_threshold
        # Optional second live-quote source for a sanity cross-check. When set,
        # get_current_price() compares E*TRADE's quote against it and WARNS on
        # material divergence. FAIL-OPEN: a missing/flaky reference never blocks
        # trading. Default None => no cross-check (byte-identical behavior).
        self.reference_price_fn = reference_price_fn
        self.quote_divergence_threshold = quote_divergence_threshold

    def _build_daily_loss_gate(self, kill_switch: KillSwitch,
                               audit: Optional[AuditLog]) -> DailyLossGate:
        """Default E5 wiring: -2% daily-loss breaker over live balances,
        start-of-day captured per ET day by the gate (see
        brokers.circuit_breaker module docstring)."""
        breaker = DailyLossCircuitBreaker(kill_switch, audit=audit)
        return DailyLossGate(
            breaker,
            value_fn=lambda: extract_account_value(
                self.client.get_balances(self.account_id_key)))

    # ------------------------------------------------------------------
    # Fat-finger guards (BLOCKING on the live path)
    # ------------------------------------------------------------------
    def _enforce_price_sanity(self, symbol: str,
                              limit_price: Optional[float]) -> None:
        """BLOCK a single-instrument order whose limit price is more than
        price_sanity_threshold away from the current price. No-op when the
        threshold or limit is None, or the current price is unavailable (the
        base warn-only check still logs). Raises PriceSanityError to refuse."""
        threshold = self.price_sanity_threshold
        if threshold is None or limit_price is None:
            return
        try:
            current = self.get_current_price(symbol)
        except Exception:  # noqa: BLE001 - a sanity check must never crash place
            return
        if current is None or current <= 0:
            return
        distance = abs(limit_price - current) / current
        if distance > threshold:
            raise PriceSanityError(
                f"Limit {limit_price:.4f} for {symbol} is {distance * 100:.1f}% "
                f"from current {current:.4f} (threshold {threshold * 100:.0f}%)"
                f" — refusing as a likely fat-finger")

    def _enforce_option_price_sanity(
            self, symbol: str, call_put: str, strike: float,
            limit_price: Optional[float]) -> None:
        """Block a single-option premium beyond a generous theoretical cap.

        Comparing an option premium to its underlying with a symmetric
        percentage-distance check is not meaningful: a $2 premium on a $100
        stock is normal, not 98% stale.  For calls, the spot price is the
        natural no-arbitrage upper reference; for puts, the strike is.  The
        configured threshold is added as generous headroom so this remains a
        typo rail rather than an execution model.  When the quote is dark,
        strike is used as a conservative fallback for both rights.
        """
        threshold = self.price_sanity_threshold
        if threshold is None or limit_price is None:
            return
        try:
            current = self.get_current_price(symbol)
        except Exception:  # noqa: BLE001 - same fail-open quote policy as EQ
            current = None
        if call_put.upper() == 'PUT':
            basis = strike
        else:
            basis = current if current is not None and current > 0 else strike
        if basis <= 0:
            return
        ceiling = basis * (1.0 + threshold)
        if limit_price > ceiling:
            raise PriceSanityError(
                f"Option limit {limit_price:.4f} for {symbol} exceeds the "
                f"{ceiling:.2f} premium sanity ceiling (basis {basis:.2f}, "
                f"threshold {threshold * 100:.0f}%) — refusing as a likely "
                "fat-finger")

    def _enforce_strike_package_sanity(self, strikes: List[float],
                                       net_price: float) -> None:
        """Shared quote-free spread check over already-normalized strikes."""
        threshold = self.price_sanity_threshold
        if threshold is None:
            return
        if not math.isfinite(net_price):
            raise PriceSanityError(f"Spread net price {net_price} is not finite")
        if not strikes:
            return
        width = max(strikes) - min(strikes)
        basis = width if width > 0 else max(strikes)
        ceiling = basis * (1.0 + threshold)
        if abs(net_price) > ceiling:
            raise PriceSanityError(
                f"Spread net price {net_price:.4f} exceeds the {ceiling:.2f} "
                f"defined-risk ceiling (strike width {width:.2f}, threshold "
                f"{threshold * 100:.0f}%) — refusing as a likely fat-finger")

    def _enforce_structure_sanity(self, legs: List[Dict],
                                  net_price: float) -> None:
        """BLOCK a spread whose net price is non-finite or larger than the
        package's defined-risk ceiling. Quote-free.

        The legitimate per-share net of a defined-risk spread is bounded by the
        STRIKE WIDTH (max - min strike), NOT the strike level — so the ceiling
        is width*(1+threshold). A magnitude typo (e.g. 440.0 entered for 4.40
        on a 430/440 spread) is then caught even though it is far below any
        single strike. Single-strike packages (width 0, e.g. a straddle) have
        no tight quote-free bound, so they fall back to a looser max-strike
        ceiling that still catches gross 10x-100x typos without false-rejecting
        a legitimate debit."""
        strikes = [leg["asset"].strike_price for leg in legs
                   if getattr(leg.get("asset"), "strike_price", None) is not None]
        self._enforce_strike_package_sanity(strikes, net_price)

    def validate_order_request(self, order_request: Dict) -> None:
        """Apply the live broker's blocking price rails to a built GUI ticket.

        The GUI previews through :class:`EtradeClient` so it can cache the
        exact previewed request behind an ``order_ref``.  Calling this method
        first lets that workflow reuse the same blocking fat-finger policy as
        broker-driven orders without combining preview and place.
        """
        try:
            order_type = order_request['orderType']
            order = order_request['Order'][0]
            if order.get('priceType') == 'MARKET':
                return
            limit_price = float(order['limitPrice'])
            instruments = order['Instrument']
            if order_type == 'EQ':
                product = instruments[0]['Product']
                self._enforce_price_sanity(product['symbol'], limit_price)
                return
            if order_type == 'OPTN':
                product = instruments[0]['Product']
                self._enforce_option_price_sanity(
                    product['symbol'], product['callPut'],
                    float(product['strikePrice']), limit_price)
                return
            if order_type == 'SPREADS':
                strikes = [float(leg['Product']['strikePrice'])
                           for leg in instruments]
                self._enforce_strike_package_sanity(strikes, limit_price)
                return
        except PriceSanityError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # The route only passes requests built by our typed builders, but
            # fail closed if that contract ever drifts.
            raise PriceSanityError(
                'Unable to validate live order price sanity') from exc
        raise PriceSanityError(
            f"Unsupported live order type {order_type!r} for price sanity")

    # ------------------------------------------------------------------
    # ExecutionBroker ABC
    # ------------------------------------------------------------------
    def place_order(self, asset: Asset, order_type: OrderType, quantity: int,
                    limit_price: float | None) -> str:
        """Preview->place a single-instrument order; returns the broker
        order id as a string.

        Mapping (ExecutionBroker's two-action surface): equity BUY/SELL
        pass through; option BUY opens a long (BUY_OPEN) and option SELL
        closes it (SELL_CLOSE). Short option exposure is only ever opened
        as part of a defined-risk structure via place_structure().
        """
        return self.place_order_with_client_id(
            asset, order_type, quantity, limit_price, None)

    def place_order_with_client_id(
            self, asset: Asset, order_type: OrderType, quantity: int,
            limit_price: float | None,
            client_order_id: Optional[str]) -> str:
        """Preview->place using an explicit idempotency identity.

        ``None`` preserves :meth:`place_order`'s existing builder-generated
        identity. A supplied id is passed verbatim to both preview and place
        by :class:`EtradeClient`.
        """
        if (client_order_id is not None
                and re.fullmatch(r"[A-Za-z0-9]{1,20}", client_order_id)
                is None):
            raise ValueError(
                "client_order_id must be 1-20 alphanumeric characters")
        action = "BUY" if order_type == OrderType.BUY else "SELL"
        if asset.asset_type is AssetType.STOCK:
            self._enforce_price_sanity(asset.symbol, limit_price)
            request = build_equity_order(asset.symbol, action, quantity,
                                         limit_price=limit_price,
                                         client_order_id=client_order_id)
        else:
            assert asset.strike_price is not None  # option assets carry a strike
            self._enforce_option_price_sanity(
                asset.symbol,
                "CALL" if asset.asset_type is AssetType.CALL else "PUT",
                asset.strike_price, limit_price)
            option_action = ("BUY_OPEN" if order_type == OrderType.BUY
                             else "SELL_CLOSE")
            request = build_option_order(
                asset.symbol,
                "CALL" if asset.asset_type is AssetType.CALL else "PUT",
                asset.strike_price, asset.expiration_date, option_action,
                quantity, limit_price=limit_price,
                client_order_id=client_order_id)
        result = self.client.submit_order(self.account_id_key, request)
        logger.info("Live order placed: %s %d %s -> order %s",
                    action, quantity, asset.symbol, result["order_id"])
        return str(result["order_id"])

    def place_structure(self, legs: List[Dict], net_price: float,
                        contracts: int, closing: bool = False) -> str:
        """Preview->place a multi-leg spread (Level 3) in ONE order.

        Args:
            legs: tracker-shaped legs [{'asset': Asset, 'action':
                'SHORT'|'BUY'}, ...] (desks/structures builders feed this).
            net_price: SIGNED package price — positive = NET_CREDIT
                received, negative = NET_DEBIT paid (build_spread_order's
                documented convention).
            contracts: contracts per leg (structures are sized uniformly).
            closing: False maps SHORT->SELL_OPEN / BUY->BUY_OPEN; True
                flips to BUY_CLOSE / SELL_CLOSE to unwind.
        """
        return self.place_structure_with_client_id(
            legs, net_price, contracts, None, closing=closing)

    def place_structure_with_client_id(
            self, legs: List[Dict], net_price: float, contracts: int,
            client_order_id: Optional[str], closing: bool = False) -> str:
        """Place one atomic package with an explicit recovery identity.

        The same client id is threaded through preview and place by
        :class:`EtradeClient`. Tracker actions (SHORT/BUY) are mapped from
        ``closing``; already-explicit E*TRADE lifecycle actions are preserved.
        """
        if (client_order_id is not None
                and re.fullmatch(r"[A-Za-z0-9]{1,20}", client_order_id)
                is None):
            raise ValueError(
                "client_order_id must be 1-20 alphanumeric characters")
        if isinstance(contracts, bool) or not isinstance(contracts, int) \
                or contracts <= 0:
            raise ValueError("contracts must be a positive integer")
        self._enforce_structure_sanity(legs, net_price)
        action_map = _CLOSE_ACTIONS if closing else _ENTRY_ACTIONS
        spread_legs = []
        for leg in legs:
            asset: Asset = leg["asset"]
            raw_action = str(leg["action"]).upper()
            ratio = leg.get("ratio", 1)
            if isinstance(ratio, bool) or not isinstance(ratio, int) \
                    or ratio <= 0:
                raise ValueError(
                    "structure leg ratio must be a positive integer")
            broker_action = (raw_action if raw_action in {
                "BUY_OPEN", "SELL_OPEN", "BUY_CLOSE", "SELL_CLOSE"}
                else action_map[raw_action])
            spread_legs.append({
                "symbol": asset.symbol,
                "call_put": ("CALL" if asset.asset_type is AssetType.CALL
                             else "PUT"),
                "strike": asset.strike_price,
                "expiry": asset.expiration_date,
                "action": broker_action,
                "quantity": contracts * ratio,
            })
        request = build_spread_order(
            spread_legs, net_price, client_order_id=client_order_id)
        result = self.client.submit_order(self.account_id_key, request)
        logger.info("Live spread placed: %d legs x%d contracts, net %.2f "
                    "-> order %s", len(legs), contracts, net_price,
                    result["order_id"])
        return str(result["order_id"])

    def cancel_order(self, order_id: str) -> bool:
        # E*TRADE order ids are numeric; tolerate string ids from callers.
        try:
            order_id_value: Union[int, str] = int(order_id)
        except (TypeError, ValueError):
            order_id_value = order_id
        return self.client.cancel_order(self.account_id_key, order_id_value)

    def get_portfolio_status(self) -> Dict:
        """Live balances + positions in the PaperTrader-compatible shape.

        Option positions get canonical keys (see _option_position_key)
        and quantities in CONTRACTS, so reconcile() compares like with
        like.
        """
        balances = self.client.get_balances(self.account_id_key)
        computed = balances.get("Computed", {})
        cash = computed.get("cashAvailableForInvestment", 0.0)
        total = computed.get("RealTimeValues", {}).get(
            "totalAccountValue", 0.0)
        positions = []
        for position in self.client.get_portfolio(self.account_id_key):
            product = position.get("Product", {})
            if product.get("securityType") == "OPTN":
                symbol_key = _option_position_key(product)
            else:
                symbol_key = product.get("symbol")
            quantity = float(position.get("quantity", 0))
            market_value = float(position.get("marketValue", 0.0))
            positions.append({
                "symbol": symbol_key,
                "quantity": quantity,
                "current_price": (market_value / quantity
                                  if quantity else 0.0),
                "market_value": market_value,
                "pnl": position.get("totalGain", 0.0),
                "pnl_pct": position.get("totalGainPct", 0.0),
            })
        return {
            "timestamp": datetime.now().isoformat(),
            "cash": cash,
            "portfolio_value": total,
            "positions": positions,
            "pending_orders": sum(
                1 for order in self.client.list_orders(self.account_id_key)
                for detail in order.get("OrderDetail", [])
                if detail.get("status") == "OPEN"),
        }

    def get_current_price(self, symbol: str) -> float | None:
        """Last trade from a live quote; mid as fallback; None if dark."""
        quote = self.client.get_quotes([symbol]).get(symbol)
        if quote is None:
            logger.warning("No live quote for %s", symbol)
            return None
        price: float | None = None
        if quote.get("last") is not None:
            price = float(quote["last"])
        else:
            bid, ask = quote.get("bid"), quote.get("ask")
            if bid is not None and ask is not None:
                price = (float(bid) + float(ask)) / 2.0
        if price is None:
            return None
        self._cross_check_price(symbol, price)
        return price

    def _cross_check_price(self, symbol: str, primary: float) -> None:
        """Compare the primary (E*TRADE) price against the optional second
        live-quote source. FAIL-OPEN: a missing or failing reference NEVER
        blocks trading — it only warns and counts a metric on material
        divergence (the primary price is always returned unchanged)."""
        if self.reference_price_fn is None:
            return
        try:
            reference = self.reference_price_fn(symbol)
        except Exception:  # noqa: BLE001 - a flaky reference must not break pricing
            logger.warning("Reference quote source failed for %s", symbol,
                           exc_info=True)
            return
        if reference is None or reference <= 0:
            return
        divergence = abs(primary - reference) / reference
        if divergence > self.quote_divergence_threshold:
            metrics.inc("quote_divergence_total", {"symbol": symbol})
            logger.warning(
                "Quote divergence for %s: primary=%.4f reference=%.4f "
                "(%.1f%% > %.1f%% threshold)",
                symbol, primary, reference,
                divergence * 100, self.quote_divergence_threshold * 100)

    # ------------------------------------------------------------------
    # PatientExecutor protocol
    # ------------------------------------------------------------------
    def order_status(self, order_id: str) -> Optional[Dict]:
        """{'status','filled_quantity','avg_fill_price'} for the executor's
        fill polling."""
        try:
            order_id_value: Union[int, str] = int(order_id)
        except (TypeError, ValueError):
            order_id_value = order_id
        return self.client.order_status(self.account_id_key, order_id_value)
