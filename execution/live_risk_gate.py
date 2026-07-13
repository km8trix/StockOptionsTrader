"""Fail-closed pending-order risk policy for live E*TRADE orders.

The gate is deliberately separate from transport.  Construction performs no
network I/O; fresh balances, positions, and quotes are fetched only by
``estimate``/``begin`` and immediately before every possible place attempt.
Durable capacity is owned by :class:`RiskReservationLedger`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from portfolio.risk_reservations import RiskReservationLedger
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch


class UnsupportedRiskOrder(RuntimeError):
    """The order cannot yet be valued safely by the live risk policy."""


class RiskValuationUnavailable(RuntimeError):
    """Fresh account or market data required for a risk decision is missing."""


class ReservationStateUnknown(RuntimeError):
    """The reservation outcome cannot be proven; live trading is stopped."""


class ReservationPersistenceError(ReservationStateUnknown):
    """A known broker acceptance could not be bound durably."""


@dataclass(frozen=True)
class LiveRiskPolicy:
    """Capacity ratios applied to the latest broker NAV.

    ``sector_nav_fraction`` is opt-in.  When enabled, every valued symbol
    must resolve through ``sector_resolver``; missing classifications fail
    closed rather than silently bypassing the configured cap.
    """

    gross_nav_multiple: float = 1.0
    per_name_nav_fraction: float = 0.10
    option_nav_fraction: float = 0.10
    sector_nav_fraction: Optional[float] = None
    market_buy_slippage_fraction: float = 0.01

    def __post_init__(self) -> None:
        values = {
            "gross_nav_multiple": self.gross_nav_multiple,
            "per_name_nav_fraction": self.per_name_nav_fraction,
            "option_nav_fraction": self.option_nav_fraction,
        }
        if self.sector_nav_fraction is not None:
            values["sector_nav_fraction"] = self.sector_nav_fraction
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if (
            not math.isfinite(float(self.market_buy_slippage_fraction))
            or float(self.market_buy_slippage_fraction) < 0
        ):
            raise ValueError("market_buy_slippage_fraction must be finite and non-negative")


@dataclass(frozen=True)
class LiveRiskHandle:
    """Opaque handle threaded through the EtradeClient place lifecycle."""

    reservation_id: str
    account_id_key: str
    order_request: Dict[str, Any]


def _number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise RiskValuationUnavailable(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskValuationUnavailable(f"{label} must be numeric") from exc
    minimum_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise RiskValuationUnavailable(f"{label} must be finite and {qualifier}")
    return result


def _quantity(value: Any, label: str) -> float:
    result = _number(value, label)
    if not result.is_integer():
        raise UnsupportedRiskOrder(f"{label} must be a whole number")
    return result


def _empty_risk() -> Dict[str, Any]:
    return {
        "gross": 0.0,
        "cash_debit": 0.0,
        "per_name": {},
        "sector": {},
        "option": {},
    }


class LiveRiskGate:
    """Freshly value and durably reserve risk around live order placement."""

    def __init__(
        self,
        client: Any,
        ledger: RiskReservationLedger,
        kill_switch: KillSwitch,
        audit: AuditLog,
        account_id_key: str,
        *,
        policy: Optional[LiveRiskPolicy] = None,
        sector_resolver: Optional[Callable[[str], Optional[str]]] = None,
        local_book: Any = None,
    ) -> None:
        account = str(account_id_key or "").strip()
        if not account:
            raise ValueError("account_id_key cannot be empty")
        if ledger.account_id_key != account:
            raise ValueError("ledger and gate account_id_key must match")
        self.client = client
        self.ledger = ledger
        self.kill_switch = kill_switch
        self.audit = audit
        self.account_id_key = account
        self.policy = policy or LiveRiskPolicy()
        self.sector_resolver = sector_resolver
        self.local_book = local_book
        self.books_fills = local_book is not None
        if self.policy.sector_nav_fraction is not None and sector_resolver is None:
            raise ValueError("sector_resolver is required when the sector cap is enabled")

    # ------------------------------------------------------------------
    # Public client hook contract
    # ------------------------------------------------------------------
    def estimate(self, order_request: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a fresh JSON-safe valuation without consuming capacity."""
        valuation = self._value_order(order_request)
        return copy.deepcopy(valuation)

    def begin(self, account_id_key: str, order_request: Mapping[str, Any]) -> LiveRiskHandle:
        """Atomically check and reserve one ``clientOrderId`` intent."""
        if str(account_id_key) != self.account_id_key:
            raise UnsupportedRiskOrder("order account does not match the risk gate account")
        self._ensure_book_ready()
        valuation = self._value_order(order_request)
        reservation_id = self._client_order_id(order_request)
        request_copy = copy.deepcopy(dict(order_request))
        metadata = dict(valuation["valuation"])
        metadata["order_request"] = request_copy
        try:
            self.ledger.check_and_reserve(
                reservation_id,
                valuation["risk"],
                valuation["capacity"],
                units=valuation["units"],
                instrument_type=valuation["instrument_type"],
                reducing=valuation["reducing"],
                base_usage=valuation["base_usage"],
                metadata=metadata,
            )
            self.audit.append(
                "live_risk_gate",
                "risk_reserved",
                {
                    "account_id_key": self.account_id_key,
                    "reservation_id": reservation_id,
                    "instrument_type": valuation["instrument_type"],
                    "reducing": valuation["reducing"],
                    "risk": valuation["risk"],
                },
            )
        except Exception as exc:
            # The ledger write may have committed before an audit/persistence
            # dependency failed.  Retain any row and stop new placements.
            if self.ledger.reservation(reservation_id) is not None:
                self._engage_unknown(reservation_id, exc)
                raise ReservationStateUnknown(
                    f"reservation {reservation_id!r} may be active"
                ) from exc
            raise
        return LiveRiskHandle(reservation_id, self.account_id_key, request_copy)

    def before_attempt(self, handle: LiveRiskHandle, attempt_number: int) -> None:
        """Revalue an active reservation immediately before each broker POST."""
        checked = self._handle(handle)
        self._ensure_book_ready()
        valuation = self._value_order(checked.order_request)
        current = self.ledger.reservation(checked.reservation_id)
        if current is None or current["status"] != "ACTIVE":
            raise ReservationStateUnknown("reservation is not active before placement")
        if (
            valuation["instrument_type"] != current["instrument_type"]
            or valuation["units"] != current["units"]
            or valuation["reducing"] != current["reducing"]
        ):
            raise UnsupportedRiskOrder("fresh portfolio state changed the order's risk class")
        self.ledger.revalue(
            checked.reservation_id,
            valuation["risk"],
            valuation["capacity"],
            base_usage=valuation["base_usage"],
        )
        self.audit.append(
            "live_risk_gate",
            "risk_revalued",
            {
                "reservation_id": checked.reservation_id,
                "attempt_number": int(attempt_number),
                "risk": valuation["risk"],
            },
        )

    def accepted(self, handle: LiveRiskHandle, result: Mapping[str, Any]) -> None:
        """Bind a known broker order ID to the durable reservation."""
        checked = self._handle(handle)
        order_id = str(result.get("order_id") or "").strip()
        if not order_id:
            error = ReservationPersistenceError("accepted order has no broker order_id")
            self._engage_unknown(checked.reservation_id, error)
            raise error
        try:
            if self.local_book is not None:
                self.local_book.track_order(order_id, checked.order_request)
            self.ledger.bind_order(checked.reservation_id, order_id)
            self.audit.append(
                "live_risk_gate",
                "risk_reservation_bound",
                {
                    "reservation_id": checked.reservation_id,
                    "order_id": order_id,
                },
            )
        except Exception as exc:
            self._engage_unknown(checked.reservation_id, exc)
            raise ReservationPersistenceError(f"could not bind broker order {order_id!r}") from exc

    def rejected(self, handle: Optional[LiveRiskHandle], exc: Exception) -> None:
        """Release capacity after a definitive broker non-acceptance."""
        if handle is None:
            # ``EtradeClient`` uses this callback when begin itself raised;
            # begin owns rollback/unknown handling until a handle exists.
            return
        checked = self._handle(handle)
        self.ledger.release(checked.reservation_id, "broker_rejected")
        self.audit.append(
            "live_risk_gate",
            "risk_reservation_released",
            {
                "reservation_id": checked.reservation_id,
                "reason": "broker_rejected",
                "error": str(exc),
            },
        )

    def unknown(self, handle: LiveRiskHandle, exc: Exception) -> None:
        """Retain capacity and engage the kill switch for ambiguity."""
        checked = self._handle(handle)
        self._engage_unknown(checked.reservation_id, exc)

    def order_update(self, order_id: str, status: Mapping[str, Any]) -> None:
        """Apply one broker cumulative-fill observation idempotently.

        Unbound orders may predate this gate and are ignored.  Once an order
        is bound, ledger invariant errors propagate so callers cannot proceed
        on a stale capacity snapshot.
        """
        order_key = str(order_id or "").strip()
        reservation = self.ledger.reservation_for_order(order_key) if order_key else None
        tracked = (
            self.local_book.tracked_order(order_key)
            if self.local_book is not None and order_key
            else None
        )
        if not order_key or (reservation is None and tracked is None):
            return
        broker_status = str(status.get("status") or "").strip().upper()
        if not broker_status:
            raise RiskValuationUnavailable("broker order update is missing status")
        filled = _number(
            status.get("filled_quantity", 0),
            "broker cumulative filled quantity",
            allow_zero=True,
        )
        if tracked is not None:
            # Book the fill before releasing its reservation. If the second
            # SQLite transaction fails, risk is temporarily over-counted.
            self.local_book.apply_order_status(order_key, status)
        updated = (
            self.ledger.record_order_update(
                order_key,
                cumulative_filled_units=filled,
                status=broker_status,
            )
            if reservation is not None
            else None
        )
        self.audit.append(
            "live_risk_gate",
            "risk_order_updated",
            {
                "order_id": order_key,
                "status": broker_status,
                "filled_quantity": filled,
                "reservation_status": (updated["status"] if updated is not None else None),
            },
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return the durable account reservation snapshot."""
        return self.ledger.snapshot()

    def recover_order_updates(
        self, updates: Iterable[Mapping[str, Any]], *, limit: int = 1000
    ) -> int:
        """Bounded restart helper for replaying known cumulative statuses."""
        applied = 0
        for update in updates:
            if applied >= limit:
                raise ValueError(f"recovery exceeds the {limit}-order safety bound")
            self.order_update(str(update.get("order_id") or ""), update)
            applied += 1
        return applied

    # ------------------------------------------------------------------
    # Fresh valuation
    # ------------------------------------------------------------------
    def _value_order(self, order_request: Mapping[str, Any]) -> Dict[str, Any]:
        order_type, order, instruments = self._parse_order(order_request)
        instrument_symbols = {
            str(instrument["Product"].get("symbol") or "").strip().upper()
            for instrument in instruments
        }
        if "" in instrument_symbols:
            raise UnsupportedRiskOrder("every instrument requires a symbol")
        pure_spread_close = order_type == "SPREADS" and all(
            str(instrument.get("orderAction") or "").strip().upper() in {"BUY_CLOSE", "SELL_CLOSE"}
            for instrument in instruments
        )

        if pure_spread_close:
            # A verified close consumes no opening capacity and must remain
            # available even when balances, quotes, or classifications are
            # unavailable.  Fresh broker positions are the only required
            # input because they prove every package leg can be reduced.
            try:
                positions = self.client.get_portfolio(self.account_id_key)
            except Exception as exc:
                raise RiskValuationUnavailable(
                    "fresh portfolio fetch failed for spread close"
                ) from exc
            if not isinstance(positions, list):
                raise RiskValuationUnavailable("broker portfolio response must be a list")
            nav = None
            cash = None
            base_usage = _empty_risk()
            capacity: Dict[str, Any] = {}
            sector = None
            valued = self._value_spread(order, instruments, positions, {}, sector)
        else:
            symbols = set(instrument_symbols)
            try:
                balances = self.client.get_balances(self.account_id_key)
                positions = self.client.get_portfolio(self.account_id_key)
            except Exception as exc:
                raise RiskValuationUnavailable("fresh balance/portfolio fetch failed") from exc
            if not isinstance(positions, list):
                raise RiskValuationUnavailable("broker portfolio response must be a list")
            symbols.update(self._position_symbols(positions))
            try:
                quotes = self.client.get_quotes(sorted(symbols))
            except Exception as exc:
                raise RiskValuationUnavailable("fresh quote fetch failed") from exc
            if not isinstance(quotes, Mapping):
                raise RiskValuationUnavailable("broker quote response must be a mapping")

            nav, cash = self._account_capacity(balances)
            base_usage = self._portfolio_usage(positions, quotes)
            capacity = self._capacity(nav, cash)
            sector = self._sector(next(iter(instrument_symbols)))

            if order_type == "EQ":
                valued = self._value_equity(order, instruments, positions, quotes, sector)
            elif order_type == "OPTN":
                valued = self._value_option(order, instruments, quotes, sector)
            elif order_type == "SPREADS":
                valued = self._value_spread(order, instruments, positions, quotes, sector)
            else:
                raise UnsupportedRiskOrder(f"unsupported orderType {order_type!r}")

        if valued["reducing"]:
            # A pure close must never be blocked because the existing book is
            # already over an opening cap.
            base_usage = _empty_risk()
            capacity = {}
        return {
            "status": "estimated",
            "revalidated_on_place": True,
            "policy": {
                "gross_nav_multiple": self.policy.gross_nav_multiple,
                "per_name_nav_fraction": self.policy.per_name_nav_fraction,
                "option_nav_fraction": self.policy.option_nav_fraction,
                "sector_nav_fraction": self.policy.sector_nav_fraction,
                "market_buy_slippage_fraction": self.policy.market_buy_slippage_fraction,
            },
            "instrument_type": order_type,
            "units": valued["units"],
            "reducing": valued["reducing"],
            "risk": valued["risk"],
            "capacity": capacity,
            "base_usage": base_usage,
            "valuation": {
                "nav": nav,
                "cash_available": cash,
                "symbol": valued["symbol"],
                "valuation_price": valued["price"],
                "fresh_mark": valued["mark"],
                "action": valued["action"],
                "sector": sector,
            },
        }

    @staticmethod
    def _parse_order(
        order_request: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any], list[Mapping[str, Any]]]:
        if not isinstance(order_request, Mapping):
            raise UnsupportedRiskOrder("order request must be a mapping")
        order_type = str(order_request.get("orderType") or "").strip().upper()
        orders = order_request.get("Order")
        if not isinstance(orders, list) or len(orders) != 1 or not isinstance(orders[0], Mapping):
            raise UnsupportedRiskOrder("risk gate requires exactly one Order")
        order = orders[0]
        instruments = order.get("Instrument")
        if not isinstance(instruments, list) or not instruments:
            raise UnsupportedRiskOrder("order requires at least one Instrument")
        if not all(isinstance(instrument, Mapping) for instrument in instruments):
            raise UnsupportedRiskOrder("every Instrument must be a mapping")
        if not all(isinstance(instrument.get("Product"), Mapping) for instrument in instruments):
            raise UnsupportedRiskOrder("every Instrument requires a Product")
        return order_type, order, instruments

    def _value_equity(
        self,
        order: Mapping[str, Any],
        instruments: list[Mapping[str, Any]],
        positions: list[Mapping[str, Any]],
        quotes: Mapping[str, Any],
        sector: Optional[str],
    ) -> Dict[str, Any]:
        if len(instruments) != 1:
            raise UnsupportedRiskOrder("equity orders must have exactly one instrument")
        instrument = instruments[0]
        product = instrument["Product"]
        if str(product.get("securityType") or "").upper() != "EQ":
            raise UnsupportedRiskOrder("EQ order must contain an EQ product")
        symbol = str(product.get("symbol") or "").strip().upper()
        action = str(instrument.get("orderAction") or "").strip().upper()
        if action not in {"BUY", "SELL"}:
            raise UnsupportedRiskOrder(f"unsupported equity action {action!r}")
        units = _quantity(instrument.get("quantity"), "equity quantity")
        position = self._equity_quantity(symbol, positions)

        if action == "SELL":
            if position <= 0 or units > position:
                raise UnsupportedRiskOrder("opening or mixed equity shorts are not supported")
            return {
                "symbol": symbol,
                "action": action,
                "units": units,
                "price": self._quote_price(symbol, quotes),
                "mark": self._quote_price(symbol, quotes),
                "reducing": True,
                "risk": _empty_risk(),
            }
        if position < 0:
            if units > abs(position):
                raise UnsupportedRiskOrder("mixed short close and long open is not supported")
            return {
                "symbol": symbol,
                "action": action,
                "units": units,
                "price": self._quote_price(symbol, quotes),
                "mark": self._quote_price(symbol, quotes),
                "reducing": True,
                "risk": _empty_risk(),
            }

        price_type = str(order.get("priceType") or "").strip().upper()
        mark = self._quote_price(symbol, quotes)
        if price_type == "LIMIT":
            price = _number(order.get("limitPrice"), "equity limitPrice")
            exposure_price = max(mark, price)
        elif price_type == "MARKET":
            fresh_buy = self._quote_price(symbol, quotes, buy=True)
            price = fresh_buy * (1.0 + self.policy.market_buy_slippage_fraction)
            exposure_price = price
        else:
            raise UnsupportedRiskOrder(f"unsupported equity priceType {price_type!r}")
        cash_debit = price * units
        notional = exposure_price * units
        risk = _empty_risk()
        risk.update(
            {
                "gross": notional,
                "cash_debit": cash_debit,
                "per_name": {symbol: notional},
            }
        )
        if sector is not None:
            risk["sector"] = {sector: notional}
        return {
            "symbol": symbol,
            "action": action,
            "units": units,
            "price": price,
            "mark": mark,
            "reducing": False,
            "risk": risk,
        }

    def _value_option(
        self,
        order: Mapping[str, Any],
        instruments: list[Mapping[str, Any]],
        quotes: Mapping[str, Any],
        sector: Optional[str],
    ) -> Dict[str, Any]:
        if len(instruments) != 1:
            raise UnsupportedRiskOrder("single-option risk requires exactly one instrument")
        instrument = instruments[0]
        product = instrument["Product"]
        if str(product.get("securityType") or "").upper() != "OPTN":
            raise UnsupportedRiskOrder("OPTN order must contain an OPTN product")
        action = str(instrument.get("orderAction") or "").strip().upper()
        if action != "BUY_OPEN":
            raise UnsupportedRiskOrder(
                "option shorts and closes are blocked until verified option-close risk is enabled"
            )
        if str(order.get("priceType") or "").strip().upper() != "LIMIT":
            raise UnsupportedRiskOrder("market option orders are not supported")
        symbol = str(product.get("symbol") or "").strip().upper()
        units = _quantity(
            instrument.get("quantity", instrument.get("orderedQuantity")),
            "option quantity",
        )
        premium_price = _number(order.get("limitPrice"), "option limitPrice")
        underlying_price = self._quote_price(symbol, quotes)
        premium = premium_price * units * 100.0
        underlying_notional = underlying_price * units * 100.0
        risk = _empty_risk()
        risk.update(
            {
                "gross": underlying_notional,
                "cash_debit": premium,
                "per_name": {symbol: underlying_notional},
                "option": {"premium": premium, "max_loss": premium},
            }
        )
        if sector is not None:
            risk["sector"] = {sector: underlying_notional}
        return {
            "symbol": symbol,
            "action": action,
            "units": units,
            "price": premium_price,
            "mark": underlying_price,
            "reducing": False,
            "risk": risk,
        }

    def _value_spread(
        self,
        order: Mapping[str, Any],
        instruments: list[Mapping[str, Any]],
        positions: list[Mapping[str, Any]],
        quotes: Mapping[str, Any],
        sector: Optional[str],
    ) -> Dict[str, Any]:
        """Value one defined-risk option package as an atomic unit.

        E*TRADE reports spread fills in package contracts, so every leg must
        carry a positive whole order quantity.  Their GCD is the package
        count and each leg's normalised integer ratio contributes to the
        package payoff.  The package risk is its worst expiry P&L at the
        submitted net limit, multiplied by that package quantity.
        Common-symbol/common-expiry opening structures
        are the deliberately narrow supported set.  A pure close is admitted
        only after fresh broker positions prove every leg and quantity;
        calendars and mixed open/close packages remain unsupported.
        """
        if len(instruments) < 2:
            raise UnsupportedRiskOrder("spread orders require at least two legs")
        price_type = str(order.get("priceType") or "").strip().upper()
        if price_type not in {"NET_CREDIT", "NET_DEBIT"}:
            raise UnsupportedRiskOrder("spread orders require NET_CREDIT or NET_DEBIT pricing")
        net_price = _number(order.get("limitPrice"), "spread limitPrice")

        leg_terms: list[tuple[str, float, float]] = []
        contracts: list[tuple[tuple[str, str, float, tuple[int, int, int]], str]] = []
        symbols: set[str] = set()
        expiries: set[tuple[int, int, int]] = set()
        quantities: list[float] = []
        actions: list[str] = []

        for index, instrument in enumerate(instruments, start=1):
            product = instrument["Product"]
            if str(product.get("securityType") or "").strip().upper() != "OPTN":
                raise UnsupportedRiskOrder("every spread leg must contain an OPTN product")
            symbol = str(product.get("symbol") or "").strip().upper()
            if not symbol:
                raise UnsupportedRiskOrder("every spread leg requires a symbol")
            symbols.add(symbol)

            right = str(product.get("callPut") or "").strip().upper()
            if right not in {"CALL", "PUT"}:
                raise UnsupportedRiskOrder(f"spread leg {index} requires CALL or PUT")
            strike = _number(product.get("strikePrice"), f"spread leg {index} strikePrice")
            expiry = self._spread_expiry(product, index)
            expiries.add(expiry)

            action = str(instrument.get("orderAction") or "").strip().upper()
            if action not in {"BUY_OPEN", "SELL_OPEN", "BUY_CLOSE", "SELL_CLOSE"}:
                raise UnsupportedRiskOrder(f"unsupported spread action {action!r}")
            actions.append(action)
            direction = 1.0 if action.startswith("BUY") else -1.0

            quantity = _quantity(
                instrument.get("quantity", instrument.get("orderedQuantity")),
                f"spread leg {index} quantity",
            )
            if "orderedQuantity" in instrument:
                ordered = _quantity(
                    instrument.get("orderedQuantity"),
                    f"spread leg {index} orderedQuantity",
                )
                if ordered != quantity:
                    raise UnsupportedRiskOrder("spread leg quantity and orderedQuantity must match")
            quantities.append(quantity)
            leg_terms.append((right, strike, direction))
            contracts.append(((symbol, right, strike, expiry), action))

        if len(symbols) != 1:
            raise UnsupportedRiskOrder("spread legs must share one underlying symbol")
        if len(expiries) != 1:
            raise UnsupportedRiskOrder("calendar and diagonal spreads are not supported")
        integer_quantities = [int(quantity) for quantity in quantities]
        units = float(math.gcd(*integer_quantities))
        if units <= 0:
            raise UnsupportedRiskOrder("spread package quantity must be positive")
        ratios = [int(quantity / units) for quantity in integer_quantities]
        if any(
            ratio <= 0 or ratio * units != quantity
            for ratio, quantity in zip(ratios, integer_quantities)
        ):
            raise UnsupportedRiskOrder(
                "spread leg quantities must form positive whole-number ratios"
            )
        legs = [
            (right, strike, direction, ratio)
            for (right, strike, direction), ratio in zip(leg_terms, ratios)
        ]
        opening = all(action.endswith("OPEN") for action in actions)
        closing = all(action.endswith("CLOSE") for action in actions)
        if not opening and not closing:
            raise UnsupportedRiskOrder("mixed open/close spread packages are not supported")
        symbol = next(iter(symbols))
        if closing:
            close_contracts = [
                (contract, action, ratio) for (contract, action), ratio in zip(contracts, ratios)
            ]
            self._verify_spread_close(close_contracts, units, positions)
            return {
                "symbol": symbol,
                "action": price_type,
                "units": units,
                "price": net_price,
                "mark": None,
                "reducing": True,
                "risk": _empty_risk(),
            }

        for right in ("CALL", "PUT"):
            bought = sum(
                ratio
                for leg_right, _, direction, ratio in legs
                if leg_right == right and direction > 0
            )
            sold = sum(
                ratio
                for leg_right, _, direction, ratio in legs
                if leg_right == right and direction < 0
            )
            if sold > bought:
                raise UnsupportedRiskOrder(f"naked short {right.lower()} exposure is not supported")

        # Option expiry P&L is piecewise linear.  Its finite minimum occurs at
        # S=0 or a strike unless the high-S call slope is negative, in which
        # case loss is unbounded and the package must fail closed.
        high_slope = sum(
            direction * ratio for right, _, direction, ratio in legs if right == "CALL"
        )
        if high_slope < 0:
            raise UnsupportedRiskOrder("spread has unbounded upside loss")
        initial_cash = net_price if price_type == "NET_CREDIT" else -net_price
        breakpoints = {0.0, *(strike for _, strike, _, _ in legs)}

        def expiry_pnl(underlying: float) -> float:
            option_payoff = 0.0
            for right, strike, direction, ratio in legs:
                intrinsic = (
                    max(underlying - strike, 0.0)
                    if right == "CALL"
                    else max(strike - underlying, 0.0)
                )
                option_payoff += direction * ratio * intrinsic
            return initial_cash + option_payoff

        worst_pnl = min(expiry_pnl(point) for point in breakpoints)
        max_loss_per_package = max(0.0, -worst_pnl) * 100.0
        max_loss = max_loss_per_package * units
        premium_debit = net_price * units * 100.0 if price_type == "NET_DEBIT" else 0.0
        underlying_price = self._quote_price(symbol, quotes)
        risk = _empty_risk()
        risk.update(
            {
                "gross": max_loss,
                "cash_debit": premium_debit,
                "per_name": {symbol: max_loss},
                "option": {"premium": premium_debit, "max_loss": max_loss},
            }
        )
        if sector is not None:
            risk["sector"] = {sector: max_loss}
        return {
            "symbol": symbol,
            "action": price_type,
            "units": units,
            "price": net_price,
            "mark": underlying_price,
            "reducing": False,
            "risk": risk,
        }

    def _verify_spread_close(
        self,
        contracts: list[tuple[tuple[str, str, float, tuple[int, int, int]], str, int]],
        units: float,
        positions: list[Mapping[str, Any]],
    ) -> None:
        required: Dict[tuple[str, str, float, tuple[int, int, int]], tuple[float, float]] = {}
        for contract, action, ratio in contracts:
            held_sign = -1.0 if action == "BUY_CLOSE" else 1.0
            previous = required.get(contract)
            if previous is not None and previous[0] != held_sign:
                raise UnsupportedRiskOrder(
                    "one spread contract cannot close long and short positions together"
                )
            required[contract] = (
                held_sign,
                (previous[1] if previous is not None else 0.0) + units * ratio,
            )

        held: Dict[tuple[str, str, float, tuple[int, int, int]], float] = {}
        for index, position in enumerate(positions, start=1):
            product = position.get("Product")
            if not isinstance(product, Mapping):
                raise RiskValuationUnavailable("portfolio position is missing Product")
            if str(product.get("securityType") or "").strip().upper() != "OPTN":
                continue
            contract = self._position_option_contract(product, index)
            held[contract] = held.get(contract, 0.0) + self._signed_position_quantity(position)

        for contract, (held_sign, needed) in required.items():
            available = held.get(contract, 0.0)
            sufficient = available >= needed if held_sign > 0 else available <= -needed
            if not sufficient:
                symbol, right, strike, expiry = contract
                description = (
                    f"{symbol} {expiry[0]:04d}-{expiry[1]:02d}-{expiry[2]:02d} {strike:g} {right}"
                )
                raise UnsupportedRiskOrder(
                    "verified broker position is missing or insufficient for "
                    f"spread close leg {description}"
                )

    @staticmethod
    def _position_option_contract(
        product: Mapping[str, Any], position_index: int
    ) -> tuple[str, str, float, tuple[int, int, int]]:
        label = f"option position {position_index}"
        symbol = str(product.get("symbol") or "").strip().upper()
        right = str(product.get("callPut") or "").strip().upper()
        try:
            strike = _number(product.get("strikePrice"), f"{label} strikePrice")
            raw_expiry = tuple(
                _number(product.get(field), f"{label} {field}")
                for field in ("expiryYear", "expiryMonth", "expiryDay")
            )
        except RiskValuationUnavailable as exc:
            raise RiskValuationUnavailable(f"{label} contract identity is incomplete") from exc
        if not symbol or right not in {"CALL", "PUT"}:
            raise RiskValuationUnavailable(f"{label} contract identity is incomplete")
        if any(not value.is_integer() for value in raw_expiry):
            raise RiskValuationUnavailable(f"{label} expiry must use whole numbers")
        year, month, day = (int(value) for value in raw_expiry)
        try:
            date(year, month, day)
        except ValueError as exc:
            raise RiskValuationUnavailable(f"{label} expiry is invalid") from exc
        return symbol, right, strike, (year, month, day)

    @staticmethod
    def _spread_expiry(product: Mapping[str, Any], leg_index: int) -> tuple[int, int, int]:
        year = int(_quantity(product.get("expiryYear"), f"spread leg {leg_index} expiryYear"))
        month = int(_quantity(product.get("expiryMonth"), f"spread leg {leg_index} expiryMonth"))
        day = int(_quantity(product.get("expiryDay"), f"spread leg {leg_index} expiryDay"))
        values = (year, month, day)
        try:
            date(*values)
        except ValueError as exc:
            raise UnsupportedRiskOrder(f"spread leg {leg_index} has an invalid expiry") from exc
        return values

    def _portfolio_usage(
        self, positions: list[Mapping[str, Any]], quotes: Mapping[str, Any]
    ) -> Dict[str, Any]:
        usage = _empty_risk()
        for position in positions:
            product = position.get("Product")
            if not isinstance(product, Mapping):
                raise RiskValuationUnavailable("portfolio position is missing Product")
            symbol = str(product.get("symbol") or "").strip().upper()
            if not symbol:
                raise RiskValuationUnavailable("portfolio position is missing symbol")
            units = abs(self._signed_position_quantity(position))
            if units == 0:
                continue
            security_type = str(product.get("securityType") or "").strip().upper()
            multiplier = 100.0 if security_type == "OPTN" else 1.0
            notional = units * multiplier * self._quote_price(symbol, quotes)
            usage["gross"] += notional
            usage["per_name"][symbol] = usage["per_name"].get(symbol, 0.0) + notional
            sector = self._sector(symbol)
            if sector is not None:
                usage["sector"][sector] = usage["sector"].get(sector, 0.0) + notional
            if security_type == "OPTN":
                market_value = position.get("marketValue")
                option_value = (
                    abs(_number(market_value, "option position marketValue", allow_zero=True))
                    if market_value is not None
                    else 0.0
                )
                usage["option"]["premium"] = usage["option"].get("premium", 0.0) + option_value
                usage["option"]["max_loss"] = usage["option"].get("max_loss", 0.0) + option_value
        return usage

    def _capacity(self, nav: float, cash: float) -> Dict[str, Any]:
        capacity: Dict[str, Any] = {
            "gross": nav * self.policy.gross_nav_multiple,
            "cash_debit": cash,
            "per_name": {"*": nav * self.policy.per_name_nav_fraction},
            "option": {
                "premium": nav * self.policy.option_nav_fraction,
                "max_loss": nav * self.policy.option_nav_fraction,
            },
        }
        if self.policy.sector_nav_fraction is not None:
            capacity["sector"] = {"*": nav * self.policy.sector_nav_fraction}
        return capacity

    @staticmethod
    def _account_capacity(balances: Any) -> tuple[float, float]:
        if not isinstance(balances, Mapping):
            raise RiskValuationUnavailable("broker balance response must be a mapping")
        computed = balances.get("Computed", {})
        computed = computed if isinstance(computed, Mapping) else {}
        realtime = computed.get("RealTimeValues", {})
        realtime = realtime if isinstance(realtime, Mapping) else {}
        nav_candidates = (
            realtime.get("totalAccountValue"),
            realtime.get("netAccountValue"),
            computed.get("netAccountValue"),
            computed.get("accountBalance"),
            balances.get("totalAccountValue"),
        )
        cash_candidates = (
            computed.get("cashAvailableForInvestment"),
            realtime.get("cashAvailableForInvestment"),
            computed.get("cashAvailableForWithdrawal"),
            balances.get("cashAvailableForInvestment"),
        )
        nav_raw = next((value for value in nav_candidates if value is not None), None)
        cash_raw = next((value for value in cash_candidates if value is not None), None)
        return (
            _number(nav_raw, "real-time account NAV"),
            _number(cash_raw, "cash available", allow_zero=True),
        )

    @staticmethod
    def _position_symbols(positions: list[Mapping[str, Any]]) -> set[str]:
        symbols: set[str] = set()
        for position in positions:
            product = position.get("Product")
            if not isinstance(product, Mapping):
                raise RiskValuationUnavailable("portfolio position is missing Product")
            symbol = str(product.get("symbol") or "").strip().upper()
            if not symbol:
                raise RiskValuationUnavailable("portfolio position is missing symbol")
            symbols.add(symbol)
        return symbols

    @staticmethod
    def _signed_position_quantity(position: Mapping[str, Any]) -> float:
        raw_quantity = position.get("quantity", 0)
        if isinstance(raw_quantity, bool):
            raise RiskValuationUnavailable("portfolio quantity must be numeric")
        try:
            quantity = float(raw_quantity)
        except (TypeError, ValueError) as exc:
            raise RiskValuationUnavailable("portfolio quantity must be numeric") from exc
        if not math.isfinite(quantity):
            raise RiskValuationUnavailable("portfolio quantity must be finite")
        position_type = str(position.get("positionType") or "").strip().upper()
        return -abs(quantity) if position_type == "SHORT" else quantity

    def _equity_quantity(self, symbol: str, positions: list[Mapping[str, Any]]) -> float:
        total = 0.0
        for position in positions:
            product = position.get("Product")
            if not isinstance(product, Mapping):
                continue
            if (
                str(product.get("securityType") or "").upper() == "EQ"
                and str(product.get("symbol") or "").strip().upper() == symbol
            ):
                total += self._signed_position_quantity(position)
        return total

    @staticmethod
    def _quote_price(symbol: str, quotes: Mapping[str, Any], *, buy: bool = False) -> float:
        quote = quotes.get(symbol)
        if not isinstance(quote, Mapping):
            raise RiskValuationUnavailable(f"fresh quote unavailable for {symbol}")
        if buy:
            candidates = (quote.get("ask"), quote.get("last"), quote.get("bid"))
        else:
            candidates = (quote.get("last"), quote.get("ask"), quote.get("bid"))
        for candidate in candidates:
            try:
                return _number(candidate, f"{symbol} quote")
            except RiskValuationUnavailable:
                continue
        raise RiskValuationUnavailable(f"fresh positive quote unavailable for {symbol}")

    def _sector(self, symbol: str) -> Optional[str]:
        if self.policy.sector_nav_fraction is None:
            return None
        assert self.sector_resolver is not None
        sector = str(self.sector_resolver(symbol) or "").strip()
        if not sector:
            raise RiskValuationUnavailable(f"sector classification unavailable for {symbol}")
        return sector

    @staticmethod
    def _client_order_id(order_request: Mapping[str, Any]) -> str:
        reservation_id = str(order_request.get("clientOrderId") or "").strip()
        if not reservation_id:
            raise UnsupportedRiskOrder("order request requires clientOrderId")
        return reservation_id

    def _handle(self, handle: LiveRiskHandle) -> LiveRiskHandle:
        if not isinstance(handle, LiveRiskHandle):
            raise ReservationStateUnknown("invalid live risk reservation handle")
        if handle.account_id_key != self.account_id_key:
            raise ReservationStateUnknown("reservation handle belongs to another account")
        return handle

    def _ensure_book_ready(self) -> None:
        if self.local_book is None:
            return
        initialized = getattr(self.local_book, "is_initialized", None)
        try:
            ready = bool(initialized()) if callable(initialized) else False
        except Exception as exc:
            raise RiskValuationUnavailable(
                "persistent local book readiness could not be verified"
            ) from exc
        if not ready:
            raise RiskValuationUnavailable(
                "persistent local book is uninitialized; adopt and reconcile "
                "the broker snapshot before placing orders"
            )

    def _engage_unknown(self, reservation_id: str, exc: Exception) -> None:
        reason = f"risk reservation state unknown for {reservation_id}: {exc}"
        self.kill_switch.engage(reason=reason, actor="live_risk_gate")
        self.audit.append(
            "live_risk_gate",
            "risk_reservation_unknown",
            {
                "reservation_id": reservation_id,
                "error": str(exc),
            },
        )
