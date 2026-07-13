"""Canonical target-position contract and deterministic order deltas.

Strategies describe the position they want, while execution operates on the
difference between that target and the account's filled *and still-working*
quantity.  Keeping this arithmetic in one small module prevents retries,
partial fills, and restarts from producing duplicate orders.

Quantities are always native instrument units: shares for equities and
contracts for options.  Contract multipliers belong in valuation and risk,
never in position-target arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from core.models import Asset, AssetType


SnapshotVersion = Union[int, str]
_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})


class ConflictingTargetError(ValueError):
    """The same asset was assigned two non-identical targets."""


class PositionFlipNotAllowed(ValueError):
    """A target would cross through zero without explicit authorization."""


class DeltaPhase(str, Enum):
    """Execution phase for an order delta."""

    OPEN = "open"
    CLOSE = "close"
    REBALANCE = "rebalance"


def _whole_quantity(value: Any, label: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be a finite integer")
    result = int(numeric)
    if not allow_zero and result == 0:
        raise ValueError(f"{label} cannot be zero")
    return result


def _optional_text(value: Optional[str], label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string when provided")
    return value.strip()


def _freeze_json(value: Any, path: str = "metadata") -> Any:
    """Return a deeply immutable, JSON-safe copy."""
    if isinstance(value, Mapping):
        frozen = {}
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} keys must be strings")
        for key in sorted(value):
            frozen[key] = _freeze_json(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return value
    raise ValueError(f"{path} must be JSON-safe")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_asset(asset: Asset) -> Asset:
    if not isinstance(asset, Asset):
        raise ValueError("asset must be an Asset")
    symbol = str(asset.symbol or "").strip().upper()
    if not symbol:
        raise ValueError("asset symbol cannot be empty")
    if not isinstance(asset.asset_type, AssetType):
        raise ValueError("asset_type must be an AssetType")
    if asset.asset_type is AssetType.STOCK:
        if asset.strike_price is not None or asset.expiration_date is not None:
            raise ValueError("stock assets cannot have strike or expiration")
        return Asset(symbol, AssetType.STOCK)

    try:
        strike = float(asset.strike_price)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("option strike must be finite and positive") from exc
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("option strike must be finite and positive")
    expiration = str(asset.expiration_date or "").strip()
    if not expiration:
        raise ValueError("option expiration cannot be empty")
    try:
        expiration = date.fromisoformat(expiration).isoformat()
    except ValueError as exc:
        raise ValueError("option expiration must be YYYY-MM-DD") from exc
    return Asset(symbol, asset.asset_type, strike, expiration)


def _asset_key(asset: Asset) -> tuple[str, str, str, str]:
    canonical = _canonical_asset(asset)
    strike = "" if canonical.strike_price is None else format(
        canonical.strike_price, ".17g"
    )
    return (
        canonical.symbol,
        canonical.asset_type.value,
        strike,
        canonical.expiration_date or "",
    )


def _asset_payload(asset: Asset) -> Mapping[str, Any]:
    canonical = _canonical_asset(asset)
    return {
        "symbol": canonical.symbol,
        "asset_type": canonical.asset_type.value,
        "strike_price": canonical.strike_price,
        "expiration_date": canonical.expiration_date,
    }


@dataclass(frozen=True)
class TargetPosition:
    """An immutable desired signed position in native instrument units."""

    asset: Asset
    target_quantity: int
    owner: Optional[str] = None
    strategy: Optional[str] = None
    reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_MAPPING,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _canonical_asset(self.asset))
        object.__setattr__(
            self,
            "target_quantity",
            _whole_quantity(self.target_quantity, "target_quantity"),
        )
        object.__setattr__(self, "owner", _optional_text(self.owner, "owner"))
        object.__setattr__(
            self, "strategy", _optional_text(self.strategy, "strategy")
        )
        object.__setattr__(self, "reason", _optional_text(self.reason, "reason"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))


def _normalise_quantity_map(
    values: Mapping[Asset, Any], label: str
) -> Mapping[Asset, int]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalised: dict[Asset, int] = {}
    for raw_asset, raw_quantity in values.items():
        asset = _canonical_asset(raw_asset)
        quantity = _whole_quantity(raw_quantity, f"{label}[{asset}]")
        normalised[asset] = normalised.get(asset, 0) + quantity
    return MappingProxyType({
        asset: quantity
        for asset, quantity in normalised.items()
        if quantity != 0
    })


@dataclass(frozen=True)
class PortfolioSnapshot:
    """One immutable generation of filled and still-working quantities."""

    filled_quantities: Mapping[Asset, int]
    reserved_deltas: Mapping[Asset, int]
    version: SnapshotVersion
    as_of: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "filled_quantities",
            _normalise_quantity_map(self.filled_quantities, "filled_quantities"),
        )
        object.__setattr__(
            self,
            "reserved_deltas",
            _normalise_quantity_map(self.reserved_deltas, "reserved_deltas"),
        )
        if isinstance(self.version, bool) or not isinstance(self.version, (int, str)):
            raise ValueError("version must be an integer or non-empty string")
        if isinstance(self.version, int) and self.version < 0:
            raise ValueError("integer version cannot be negative")
        if isinstance(self.version, str):
            version = self.version.strip()
            if not version:
                raise ValueError("string version cannot be empty")
            object.__setattr__(self, "version", version)
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None:
            raise ValueError("as_of must be a timezone-aware datetime")
        if self.as_of.utcoffset() is None:
            raise ValueError("as_of must be a timezone-aware datetime")

    def effective_quantity(self, asset: Asset) -> int:
        """Filled quantity plus every non-terminal reserved signed delta."""
        canonical = _canonical_asset(asset)
        return (
            self.filled_quantities.get(canonical, 0)
            + self.reserved_deltas.get(canonical, 0)
        )

    @classmethod
    def from_portfolio_manager(
        cls,
        portfolio: Any,
        *,
        version: SnapshotVersion,
        as_of: datetime,
        reservation_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> "PortfolioSnapshot":
        """Adapt the existing PortfolioManager and optional risk-ledger snapshot."""
        return cls(
            filled_quantities=filled_quantities_from_portfolio(portfolio),
            reserved_deltas=(
                reserved_deltas_from_risk_snapshot(reservation_snapshot)
                if reservation_snapshot is not None
                else {}
            ),
            version=version,
            as_of=as_of,
        )


@dataclass(frozen=True)
class OrderDelta:
    """One deterministic native-unit order intent toward a target."""

    asset: Asset
    signed_quantity: int
    target_quantity: int
    effective_quantity: int
    intent_id: str
    phase: DeltaPhase
    owner: Optional[str] = None
    strategy: Optional[str] = None
    reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: _EMPTY_MAPPING,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _canonical_asset(self.asset))
        object.__setattr__(
            self,
            "signed_quantity",
            _whole_quantity(
                self.signed_quantity, "signed_quantity", allow_zero=False
            ),
        )
        object.__setattr__(
            self,
            "target_quantity",
            _whole_quantity(self.target_quantity, "target_quantity"),
        )
        object.__setattr__(
            self,
            "effective_quantity",
            _whole_quantity(self.effective_quantity, "effective_quantity"),
        )
        if not isinstance(self.intent_id, str) or not self.intent_id.strip():
            raise ValueError("intent_id must be a non-empty string")
        object.__setattr__(self, "intent_id", self.intent_id.strip())
        if not isinstance(self.phase, DeltaPhase):
            try:
                object.__setattr__(self, "phase", DeltaPhase(self.phase))
            except (TypeError, ValueError) as exc:
                raise ValueError("phase must be a DeltaPhase") from exc
        object.__setattr__(self, "owner", _optional_text(self.owner, "owner"))
        object.__setattr__(
            self, "strategy", _optional_text(self.strategy, "strategy")
        )
        object.__setattr__(self, "reason", _optional_text(self.reason, "reason"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @property
    def side(self) -> str:
        return "BUY" if self.signed_quantity > 0 else "SELL"

    @property
    def quantity(self) -> int:
        return abs(self.signed_quantity)


def _target_payload(target: TargetPosition) -> Mapping[str, Any]:
    return {
        "asset": _asset_payload(target.asset),
        "target_quantity": target.target_quantity,
        "owner": target.owner,
        "strategy": target.strategy,
        "reason": target.reason,
        "metadata": _thaw_json(target.metadata),
    }


def _intent_id(
    target: TargetPosition,
    snapshot: PortfolioSnapshot,
    *,
    phase: DeltaPhase,
    signed_quantity: int,
) -> str:
    payload = {
        "target": _target_payload(target),
        "snapshot": {
            "version": snapshot.version,
            "as_of": snapshot.as_of.astimezone(timezone.utc).isoformat(
                timespec="microseconds"
            ),
        },
        "phase": phase.value,
        "signed_quantity": signed_quantity,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    # Twenty characters fit the current E*TRADE clientOrderId limit.
    return "tp" + hashlib.sha256(encoded).hexdigest()[:18]


def _delta(
    target: TargetPosition,
    snapshot: PortfolioSnapshot,
    *,
    signed_quantity: int,
    effective_quantity: int,
    phase: DeltaPhase,
) -> OrderDelta:
    return OrderDelta(
        asset=target.asset,
        signed_quantity=signed_quantity,
        target_quantity=target.target_quantity,
        effective_quantity=effective_quantity,
        intent_id=_intent_id(
            target,
            snapshot,
            phase=phase,
            signed_quantity=signed_quantity,
        ),
        phase=phase,
        owner=target.owner,
        strategy=target.strategy,
        reason=target.reason,
        metadata=target.metadata,
    )


def build_order_deltas(
    targets: Iterable[TargetPosition],
    snapshot: PortfolioSnapshot,
    *,
    allow_flip: bool = False,
) -> tuple[OrderDelta, ...]:
    """Deterministically diff desired targets against filled + reserved units.

    Exact duplicate targets are coalesced.  Non-identical targets for the same
    asset fail closed.  Position reversals also fail closed unless ``allow_flip``
    is explicitly true; when enabled they are emitted close-first, then open.
    """
    if not isinstance(snapshot, PortfolioSnapshot):
        raise ValueError("snapshot must be a PortfolioSnapshot")
    if not isinstance(allow_flip, bool):
        raise ValueError("allow_flip must be a boolean")

    by_asset: dict[Asset, TargetPosition] = {}
    for target in targets:
        if not isinstance(target, TargetPosition):
            raise ValueError("targets must contain only TargetPosition values")
        existing = by_asset.get(target.asset)
        if existing is not None and existing != target:
            raise ConflictingTargetError(
                f"conflicting targets for {target.asset}: "
                f"{existing.target_quantity} and {target.target_quantity}"
            )
        by_asset[target.asset] = target

    result: list[OrderDelta] = []
    for asset in sorted(by_asset, key=_asset_key):
        target = by_asset[asset]
        effective = snapshot.effective_quantity(asset)
        desired = target.target_quantity
        if effective == desired:
            continue
        if effective != 0 and desired != 0 and (effective > 0) != (desired > 0):
            if not allow_flip:
                raise PositionFlipNotAllowed(
                    f"target for {asset} flips effective quantity "
                    f"from {effective} to {desired}"
                )
            result.append(
                _delta(
                    target,
                    snapshot,
                    signed_quantity=-effective,
                    effective_quantity=effective,
                    phase=DeltaPhase.CLOSE,
                )
            )
            result.append(
                _delta(
                    target,
                    snapshot,
                    signed_quantity=desired,
                    effective_quantity=0,
                    phase=DeltaPhase.OPEN,
                )
            )
            continue

        signed_quantity = desired - effective
        if desired == 0:
            phase = DeltaPhase.CLOSE
        elif effective == 0:
            phase = DeltaPhase.OPEN
        else:
            phase = DeltaPhase.REBALANCE
        result.append(
            _delta(
                target,
                snapshot,
                signed_quantity=signed_quantity,
                effective_quantity=effective,
                phase=phase,
            )
        )
    return tuple(result)


def filled_quantities_from_portfolio(portfolio: Any) -> Mapping[Asset, int]:
    """Read signed native quantities from the existing PortfolioManager API."""
    positions = getattr(portfolio, "positions", None)
    if not isinstance(positions, Mapping):
        raise ValueError("portfolio.positions must be a mapping")
    quantities: dict[Asset, int] = {}
    for position in positions.values():
        raw_asset = getattr(position, "asset", None)
        if not isinstance(raw_asset, Asset):
            raise ValueError("every portfolio position must carry an Asset")
        asset = _canonical_asset(raw_asset)
        quantity = _whole_quantity(
            getattr(position, "quantity", None), f"position quantity for {asset}"
        )
        quantities[asset] = quantities.get(asset, 0) + quantity
    return _normalise_quantity_map(quantities, "portfolio positions")


def _reservation_product_asset(product: Mapping[str, Any]) -> Asset:
    security_type = str(product.get("securityType") or "").strip().upper()
    symbol = str(product.get("symbol") or "").strip().upper()
    if security_type == "EQ":
        return _canonical_asset(Asset(symbol, AssetType.STOCK))
    if security_type != "OPTN":
        raise ValueError(f"unsupported reserved securityType {security_type!r}")
    call_put = str(product.get("callPut") or "").strip().upper()
    if call_put not in {"CALL", "PUT"}:
        raise ValueError("reserved option callPut must be CALL or PUT")
    try:
        expiration = date(
            int(product["expiryYear"]),
            int(product["expiryMonth"]),
            int(product["expiryDay"]),
        ).isoformat()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reserved option has invalid expiry") from exc
    return _canonical_asset(
        Asset(
            symbol,
            AssetType.CALL if call_put == "CALL" else AssetType.PUT,
            product.get("strikePrice"),
            expiration,
        )
    )


def _single_reserved_instrument(
    reservation: Mapping[str, Any],
) -> tuple[Asset, int, int]:
    metadata = reservation.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("active reservation is missing metadata")
    request = metadata.get("order_request")
    if not isinstance(request, Mapping):
        raise ValueError("active reservation is missing metadata.order_request")
    orders = request.get("Order")
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        raise ValueError("reserved order request has invalid Order")
    if len(orders) != 1 or not isinstance(orders[0], Mapping):
        raise ValueError("reserved order request must contain exactly one Order")
    instruments = orders[0].get("Instrument")
    if not isinstance(instruments, Sequence) or isinstance(
        instruments, (str, bytes)
    ):
        raise ValueError("reserved order request has invalid Instrument")
    if len(instruments) != 1 or not isinstance(instruments[0], Mapping):
        raise ValueError(
            "multi-leg reservation deltas require the atomic structure contract"
        )
    instrument = instruments[0]
    product = instrument.get("Product")
    if not isinstance(product, Mapping):
        raise ValueError("reserved instrument is missing Product")
    asset = _reservation_product_asset(product)
    request_quantity = _whole_quantity(
        instrument.get("quantity", instrument.get("orderedQuantity")),
        "reserved order quantity",
        allow_zero=False,
    )
    if request_quantity < 0:
        raise ValueError("reserved order quantity must be positive")
    action = str(instrument.get("orderAction") or "").strip().upper()
    if action in {"BUY", "BUY_OPEN", "BUY_CLOSE"}:
        sign = 1
    elif action in {"SELL", "SELL_OPEN", "SELL_CLOSE"}:
        sign = -1
    else:
        raise ValueError(f"unsupported reserved orderAction {action!r}")
    return asset, request_quantity, sign


def reserved_deltas_from_risk_snapshot(
    snapshot: Mapping[str, Any],
) -> Mapping[Asset, int]:
    """Adapt active RiskReservationLedger rows to working signed quantities.

    Current risk reservations persist the complete broker order request in
    ``metadata.order_request``.  The remaining native units are the reserved
    units less cumulative fills across original and replacement broker orders.
    Unsupported active structures fail closed instead of being omitted.
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("reservation snapshot must be a mapping")
    reservations = snapshot.get("reservations", [])
    if not isinstance(reservations, Sequence) or isinstance(
        reservations, (str, bytes)
    ):
        raise ValueError("reservation snapshot reservations must be a sequence")

    quantities: dict[Asset, int] = {}
    for reservation in reservations:
        if not isinstance(reservation, Mapping):
            raise ValueError("reservation rows must be mappings")
        status = str(reservation.get("status") or "").strip().upper()
        if not status:
            raise ValueError("reservation status cannot be empty")
        if status != "ACTIVE":
            continue
        asset, requested_quantity, sign = _single_reserved_instrument(reservation)
        units = _whole_quantity(
            reservation.get("units"), "active reservation units", allow_zero=False
        )
        if units <= 0 or units != requested_quantity:
            raise ValueError("active reservation units do not match order quantity")
        orders = reservation.get("orders", [])
        if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
            raise ValueError("active reservation orders must be a sequence")
        filled = 0
        for order in orders:
            if not isinstance(order, Mapping):
                raise ValueError("reservation order rows must be mappings")
            cumulative = _whole_quantity(
                order.get("cumulative_filled_units", 0),
                "reservation cumulative filled units",
            )
            if cumulative < 0:
                raise ValueError("reservation cumulative filled units cannot be negative")
            filled += cumulative
        if filled > units:
            raise ValueError("reservation cumulative fills exceed reserved units")
        remaining = units - filled
        if remaining:
            quantities[asset] = quantities.get(asset, 0) + sign * remaining
    return _normalise_quantity_map(quantities, "reservation deltas")


__all__ = [
    "ConflictingTargetError",
    "DeltaPhase",
    "OrderDelta",
    "PortfolioSnapshot",
    "PositionFlipNotAllowed",
    "TargetPosition",
    "build_order_deltas",
    "filled_quantities_from_portfolio",
    "reserved_deltas_from_risk_snapshot",
]
