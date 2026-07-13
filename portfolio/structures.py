"""Immutable atomic multi-leg option order contract.

One :class:`StructureIntent` is one broker package, one risk reservation and
one accounting mutation.  Leg lifecycle is explicit; an unwind can therefore
never be inferred as another opening trade.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from core.models import Asset, AssetType


class LegAction(str, Enum):
    BUY_OPEN = "BUY_OPEN"
    SELL_OPEN = "SELL_OPEN"
    BUY_CLOSE = "BUY_CLOSE"
    SELL_CLOSE = "SELL_CLOSE"

    @property
    def is_open(self) -> bool:
        return self in {self.BUY_OPEN, self.SELL_OPEN}

    @property
    def sign(self) -> int:
        return 1 if self in {self.BUY_OPEN, self.BUY_CLOSE} else -1


def _freeze(value: Any, path: str = "metadata") -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} keys must be strings")
        return MappingProxyType({
            key: _freeze(value[key], f"{path}.{key}") for key in sorted(value)
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} must be finite and JSON-safe")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _text(value: Optional[str], label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty when provided")
    return value.strip()


@dataclass(frozen=True)
class StructureLeg:
    """One option leg and its positive package ratio."""

    asset: Asset
    action: LegAction
    ratio: int = 1

    def __post_init__(self) -> None:
        asset = self.asset
        if not isinstance(asset, Asset) or asset.asset_type not in {
                AssetType.CALL, AssetType.PUT}:
            raise ValueError("structure legs must be CALL or PUT Assets")
        if (not asset.symbol or asset.strike_price is None
                or not math.isfinite(float(asset.strike_price))
                or float(asset.strike_price) <= 0
                or not asset.expiration_date):
            raise ValueError("structure leg requires symbol, expiry and strike")
        object.__setattr__(self, "asset", Asset(
            str(asset.symbol).strip().upper(), asset.asset_type,
            float(asset.strike_price), str(asset.expiration_date).strip()))
        if not isinstance(self.action, LegAction):
            try:
                object.__setattr__(self, "action", LegAction(self.action))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid structure leg action") from exc
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, int) \
                or self.ratio <= 0:
            raise ValueError("structure leg ratio must be a positive integer")

    def execution_dict(self) -> dict[str, Any]:
        """PatientExecutor/LiveEtradeBroker adapter with explicit lifecycle."""
        return {"asset": self.asset, "action": self.action.value,
                "ratio": self.ratio}


def _expiry_payoff(legs: Sequence[StructureLeg], spot: float) -> float:
    payoff = 0.0
    for leg in legs:
        strike_price = leg.asset.strike_price
        # StructureLeg.__post_init__ rejects option legs without a strike.
        assert strike_price is not None
        strike = float(strike_price)
        intrinsic = (max(0.0, spot - strike)
                     if leg.asset.asset_type is AssetType.CALL
                     else max(0.0, strike - spot))
        payoff += leg.action.sign * leg.ratio * intrinsic
    return payoff


def theoretical_max_loss_per_package(
        legs: Sequence[StructureLeg], net_price: float) -> float:
    """Expiry max loss per one ratio package, including its net credit/debit.

    ``net_price`` is per-share and signed: positive credit, negative debit.
    A negative aggregate call slope is unbounded and rejected.
    """
    call_slope = sum(
        leg.action.sign * leg.ratio for leg in legs
        if leg.asset.asset_type is AssetType.CALL)
    if call_slope < 0:
        raise ValueError("structure has unbounded upside loss")
    strikes = []
    for leg in legs:
        strike_price = leg.asset.strike_price
        # StructureLeg.__post_init__ rejects option legs without a strike.
        assert strike_price is not None
        strikes.append(float(strike_price))
    points = {0.0, *strikes}
    pnl_values = [_expiry_payoff(legs, spot) + net_price for spot in points]
    # With positive terminal slope, infinity is profitable; with zero slope,
    # the last strike already has the terminal constant payoff.
    return max(0.0, -min(pnl_values)) * 100.0


@dataclass(frozen=True)
class StructureIntent:
    """One atomic desired opening or closing option package."""

    legs: tuple[StructureLeg, ...]
    quantity: int
    net_price: float
    max_loss: float
    greeks: Mapping[str, float]
    reason: str
    owner: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False)
    intent_id: Optional[str] = None

    def __post_init__(self) -> None:
        legs = tuple(self.legs)
        if len(legs) < 2 or not all(isinstance(leg, StructureLeg) for leg in legs):
            raise ValueError("an atomic structure requires at least two legs")
        object.__setattr__(self, "legs", legs)
        symbols = {leg.asset.symbol for leg in legs}
        expiries = {leg.asset.expiration_date for leg in legs}
        if len(symbols) != 1:
            raise ValueError("structure legs must share one underlying")
        if len(expiries) != 1:
            raise ValueError("calendar/diagonal packages are not supported")
        lifecycle = {leg.action.is_open for leg in legs}
        if len(lifecycle) != 1:
            raise ValueError("a package cannot mix opening and closing legs")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) \
                or self.quantity <= 0:
            raise ValueError("quantity must be a positive package count")
        net = float(self.net_price)
        if not math.isfinite(net) or net == 0:
            raise ValueError("net_price must be finite and signed non-zero")
        object.__setattr__(self, "net_price", net)
        loss = float(self.max_loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("max_loss must be finite and non-negative")
        if self.opening:
            if not ({leg.action.sign for leg in legs} == {-1, 1}):
                raise ValueError("opening structures require long and short legs")
            minimum_loss = theoretical_max_loss_per_package(legs, net)
            if loss + 1e-9 < minimum_loss * self.quantity:
                raise ValueError(
                    "max_loss understates the package's expiry loss")
        object.__setattr__(self, "max_loss", loss)
        if not isinstance(self.greeks, Mapping):
            raise ValueError("greeks must be a mapping")
        normalised_greeks = {}
        for key, raw in self.greeks.items():
            value = float(raw)
            if not str(key).strip() or not math.isfinite(value):
                raise ValueError("greeks require named finite values")
            normalised_greeks[str(key).strip()] = value
        object.__setattr__(self, "greeks", MappingProxyType(normalised_greeks))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "owner", _text(self.owner, "owner"))
        object.__setattr__(self, "strategy", _text(self.strategy, "strategy"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        supplied_id = _text(self.intent_id, "intent_id")
        if supplied_id is None:
            payload = {
                "legs": [{"asset": str(leg.asset), "action": leg.action.value,
                          "ratio": leg.ratio} for leg in legs],
                "quantity": self.quantity,
                "net_price": net,
                "max_loss": loss,
                "greeks": dict(normalised_greeks),
                "owner": self.owner,
                "strategy": self.strategy,
                "reason": self.reason,
                "metadata": _thaw(self.metadata),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()
            supplied_id = "st" + hashlib.sha256(encoded).hexdigest()[:18]
        if len(supplied_id) > 64:
            raise ValueError("intent_id cannot exceed 64 characters")
        object.__setattr__(self, "intent_id", supplied_id)

    @property
    def opening(self) -> bool:
        return self.legs[0].action.is_open

    @property
    def closing(self) -> bool:
        return not self.opening

    @property
    def side(self) -> str:
        return "SELL" if self.net_price > 0 else "BUY"

    @property
    def execution_legs(self) -> list[dict[str, Any]]:
        return [leg.execution_dict() for leg in self.legs]


__all__ = [
    "LegAction",
    "StructureIntent",
    "StructureLeg",
    "theoretical_max_loss_per_package",
]
