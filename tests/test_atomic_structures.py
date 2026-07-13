"""Canonical atomic option-package contract."""

from dataclasses import FrozenInstanceError

import pytest

from core.models import Asset, AssetType
from portfolio.structures import (
    LegAction,
    StructureIntent,
    StructureLeg,
    theoretical_max_loss_per_package,
)


def option(right, strike, expiry="2026-08-21"):
    return Asset("SPY", right, strike, expiry)


def vertical(actions=(LegAction.SELL_OPEN, LegAction.BUY_OPEN)):
    return (
        StructureLeg(option(AssetType.CALL, 450), actions[0]),
        StructureLeg(option(AssetType.CALL, 455), actions[1]),
    )


def intent(legs=None, **kwargs):
    values = {
        "legs": legs or vertical(), "quantity": 2, "net_price": 1.0,
        "max_loss": 800.0, "greeks": {"delta": -2.0, "vega": -15.0},
        "reason": "defined-risk call spread", "metadata": {"book": ["vrp"]},
    }
    values.update(kwargs)
    return StructureIntent(**values)


def test_opening_contract_is_immutable_stable_and_execution_ready():
    package = intent()
    assert package.opening and not package.closing
    assert package.side == "SELL"
    assert len(package.intent_id) == 20
    assert [leg["action"] for leg in package.execution_legs] == [
        "SELL_OPEN", "BUY_OPEN"]
    assert package.metadata["book"] == ("vrp",)
    with pytest.raises(FrozenInstanceError):
        package.quantity = 3
    with pytest.raises(TypeError):
        package.metadata["book"] = "other"
    assert intent().intent_id == package.intent_id


def test_closing_lifecycle_is_preserved_and_can_be_a_debit():
    package = intent(
        legs=vertical((LegAction.BUY_CLOSE, LegAction.SELL_CLOSE)),
        net_price=-0.4, max_loss=0.0)
    assert package.closing and package.side == "BUY"
    assert [leg["action"] for leg in package.execution_legs] == [
        "BUY_CLOSE", "SELL_CLOSE"]


def test_vertical_theoretical_loss_includes_credit():
    assert theoretical_max_loss_per_package(vertical(), 1.0) == 400.0


def test_unbounded_short_call_and_understated_loss_fail_closed():
    naked_ratio = (
        StructureLeg(option(AssetType.CALL, 450), LegAction.SELL_OPEN, 2),
        StructureLeg(option(AssetType.CALL, 455), LegAction.BUY_OPEN, 1),
    )
    with pytest.raises(ValueError, match="unbounded"):
        intent(legs=naked_ratio)
    with pytest.raises(ValueError, match="understates"):
        StructureIntent(
            legs=vertical(), quantity=2, net_price=1.0, max_loss=799.0,
            greeks={}, reason="too optimistic")


def test_mixed_lifecycle_calendar_and_stock_leg_fail_closed():
    mixed = (
        StructureLeg(option(AssetType.CALL, 450), LegAction.SELL_OPEN),
        StructureLeg(option(AssetType.CALL, 455), LegAction.BUY_CLOSE),
    )
    with pytest.raises(ValueError, match="mix"):
        intent(legs=mixed)
    calendar = (
        StructureLeg(option(AssetType.CALL, 450), LegAction.SELL_OPEN),
        StructureLeg(option(AssetType.CALL, 455, "2026-09-18"),
                     LegAction.BUY_OPEN),
    )
    with pytest.raises(ValueError, match="calendar"):
        intent(legs=calendar)
    with pytest.raises(ValueError, match="CALL or PUT"):
        StructureLeg(Asset("SPY", AssetType.STOCK), LegAction.BUY_OPEN)
