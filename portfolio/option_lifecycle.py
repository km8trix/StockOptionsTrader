"""Physical exercise and assignment planning for American equity options."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

from core.models import Asset, AssetType


class LifecycleReason(str, Enum):
    EXPIRED_WORTHLESS = "expired_worthless"
    EXPIRATION_EXERCISE = "expiration_exercise"
    EXPIRATION_ASSIGNMENT = "expiration_assignment"
    EARLY_EXERCISE = "early_exercise"
    EARLY_ASSIGNMENT = "early_assignment"


@dataclass(frozen=True)
class EarlyExerciseDecision:
    exercise: bool
    intrinsic: float
    extrinsic: float
    economic_benefit: float
    rationale: str


@dataclass(frozen=True)
class OptionLifecycleEvent:
    """One indivisible plan that removes an option and creates stock/cash."""

    option: Asset
    signed_contracts: int
    stock_delta: int
    cash_delta: float
    settlement_price: float
    reason: LifecycleReason
    effective_date: date

    def __post_init__(self) -> None:
        if self.option.asset_type not in {AssetType.CALL, AssetType.PUT}:
            raise ValueError("lifecycle event requires an option Asset")
        if (isinstance(self.signed_contracts, bool)
                or not isinstance(self.signed_contracts, int)
                or self.signed_contracts == 0):
            raise ValueError("signed_contracts must be a non-zero integer")
        if isinstance(self.stock_delta, bool) \
                or not isinstance(self.stock_delta, int):
            raise ValueError("stock_delta must be an integer")
        if not math.isfinite(float(self.cash_delta)):
            raise ValueError("cash_delta must be finite")
        if (not math.isfinite(float(self.settlement_price))
                or float(self.settlement_price) < 0):
            raise ValueError("settlement_price must be finite and non-negative")


@dataclass(frozen=True)
class OptionLifecyclePolicy:
    """Automatic exercise/assignment and conservative early-exercise policy."""

    auto_exercise_threshold: float = 0.01
    enable_early_exercise: bool = True

    def __post_init__(self) -> None:
        if (not math.isfinite(float(self.auto_exercise_threshold))
                or float(self.auto_exercise_threshold) < 0):
            raise ValueError(
                "auto_exercise_threshold must be finite and non-negative")

    @staticmethod
    def _option(asset: Asset) -> Asset:
        if not isinstance(asset, Asset) or asset.asset_type not in {
                AssetType.CALL, AssetType.PUT}:
            raise ValueError("asset must be a call or put")
        if (asset.strike_price is None
                or not math.isfinite(float(asset.strike_price))
                or float(asset.strike_price) <= 0
                or not asset.expiration_date):
            raise ValueError("option requires a positive strike and expiry")
        return asset

    @staticmethod
    def intrinsic(asset: Asset, spot: float) -> float:
        asset = OptionLifecyclePolicy._option(asset)
        spot = float(spot)
        if not math.isfinite(spot) or spot < 0:
            raise ValueError("spot must be finite and non-negative")
        strike = float(asset.strike_price)
        if asset.asset_type is AssetType.CALL:
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    def early_exercise_decision(
            self, asset: Asset, *, spot: float, option_mark: float,
            days_to_expiry: int, annual_rate: float = 0.0,
            dividend: float = 0.0) -> EarlyExerciseDecision:
        """Holder economics; a short has symmetric early-assignment risk."""
        asset = self._option(asset)
        if isinstance(days_to_expiry, bool) or not isinstance(days_to_expiry, int) \
                or days_to_expiry <= 0:
            raise ValueError("days_to_expiry must be a positive integer")
        values = (float(option_mark), float(annual_rate), float(dividend))
        if (not all(math.isfinite(value) for value in values)
                or values[0] < 0 or values[2] < 0):
            raise ValueError("mark, rate and dividend inputs must be finite")
        intrinsic = self.intrinsic(asset, spot)
        extrinsic = max(0.0, values[0] - intrinsic)
        if not self.enable_early_exercise or intrinsic <= 0:
            return EarlyExerciseDecision(
                False, intrinsic, extrinsic, 0.0,
                "early exercise disabled or option is not in the money")
        strike = float(asset.strike_price)
        carry = max(0.0, strike * values[1] * days_to_expiry / 365.0)
        if asset.asset_type is AssetType.CALL:
            benefit = values[2] - extrinsic - carry
            rationale = "dividend captured exceeds forfeited extrinsic and carry"
        else:
            benefit = carry - extrinsic
            rationale = "strike carry exceeds forfeited extrinsic"
        return EarlyExerciseDecision(
            benefit > 0, intrinsic, extrinsic, benefit, rationale)

    def plan(
            self, asset: Asset, signed_contracts: int, *, spot: float,
            effective_date: date | datetime, early: bool = False,
            option_mark: Optional[float] = None, days_to_expiry: int = 0,
            annual_rate: float = 0.0, dividend: float = 0.0,
            force: bool = False) -> OptionLifecycleEvent:
        """Plan physical settlement without mutating a portfolio."""
        asset = self._option(asset)
        if (isinstance(signed_contracts, bool)
                or not isinstance(signed_contracts, int)
                or signed_contracts == 0):
            raise ValueError("signed_contracts must be a non-zero integer")
        day = (effective_date.date()
               if isinstance(effective_date, datetime) else effective_date)
        if not isinstance(day, date):
            raise ValueError("effective_date must be a date or datetime")
        intrinsic = self.intrinsic(asset, spot)
        exercise = force or intrinsic >= self.auto_exercise_threshold
        if early:
            if option_mark is None:
                raise ValueError("option_mark is required for early exercise")
            decision = self.early_exercise_decision(
                asset, spot=spot, option_mark=option_mark,
                days_to_expiry=days_to_expiry, annual_rate=annual_rate,
                dividend=dividend)
            exercise = force or decision.exercise
        if not exercise:
            return OptionLifecycleEvent(
                asset, signed_contracts, 0, 0.0, intrinsic,
                LifecycleReason.EXPIRED_WORTHLESS, day)

        direction = 1 if asset.asset_type is AssetType.CALL else -1
        stock_delta = signed_contracts * asset.multiplier * direction
        cash_delta = -stock_delta * float(asset.strike_price)
        if early:
            reason = (LifecycleReason.EARLY_EXERCISE
                      if signed_contracts > 0
                      else LifecycleReason.EARLY_ASSIGNMENT)
        else:
            reason = (LifecycleReason.EXPIRATION_EXERCISE
                      if signed_contracts > 0
                      else LifecycleReason.EXPIRATION_ASSIGNMENT)
        return OptionLifecycleEvent(
            asset, signed_contracts, stock_delta, cash_delta, intrinsic,
            reason, day)


__all__ = [
    "EarlyExerciseDecision",
    "LifecycleReason",
    "OptionLifecycleEvent",
    "OptionLifecyclePolicy",
]
