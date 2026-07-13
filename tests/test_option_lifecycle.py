from datetime import date

import pytest

from core.models import Asset, AssetType
from portfolio.option_lifecycle import LifecycleReason, OptionLifecyclePolicy


EXPIRY = "2026-08-21"


def option(right, strike=100.0):
    return Asset("SPY", right, strike, EXPIRY)


@pytest.mark.parametrize(
    "right,contracts,spot,stock,cash,reason",
    [
        (AssetType.CALL, 2, 110.0, 200, -20_000.0,
         LifecycleReason.EXPIRATION_EXERCISE),
        (AssetType.CALL, -2, 110.0, -200, 20_000.0,
         LifecycleReason.EXPIRATION_ASSIGNMENT),
        (AssetType.PUT, 2, 90.0, -200, 20_000.0,
         LifecycleReason.EXPIRATION_EXERCISE),
        (AssetType.PUT, -2, 90.0, 200, -20_000.0,
         LifecycleReason.EXPIRATION_ASSIGNMENT),
    ],
)
def test_physical_expiration_stock_and_strike_cash(
        right, contracts, spot, stock, cash, reason):
    event = OptionLifecyclePolicy().plan(
        option(right), contracts, spot=spot,
        effective_date=date(2026, 8, 24))
    assert event.stock_delta == stock
    assert event.cash_delta == cash
    assert event.settlement_price == 10.0
    assert event.reason is reason


def test_out_of_money_expiry_removes_option_without_stock_or_cash():
    event = OptionLifecyclePolicy().plan(
        option(AssetType.CALL), 3, spot=99.0,
        effective_date=date(2026, 8, 24))
    assert event.reason is LifecycleReason.EXPIRED_WORTHLESS
    assert event.stock_delta == 0 and event.cash_delta == 0


def test_ex_dividend_call_and_deep_itm_put_early_exercise_economics():
    policy = OptionLifecyclePolicy()
    call = policy.early_exercise_decision(
        option(AssetType.CALL), spot=110.0, option_mark=10.20,
        days_to_expiry=2, dividend=0.50, annual_rate=0.01)
    assert call.exercise and call.economic_benefit > 0
    put = policy.early_exercise_decision(
        option(AssetType.PUT), spot=50.0, option_mark=50.05,
        days_to_expiry=30, annual_rate=0.05)
    assert put.exercise and put.economic_benefit > 0


def test_extrinsic_value_blocks_early_exercise_and_force_is_explicit():
    policy = OptionLifecyclePolicy()
    decision = policy.early_exercise_decision(
        option(AssetType.CALL), spot=110.0, option_mark=12.0,
        days_to_expiry=2, dividend=0.50)
    assert not decision.exercise
    event = policy.plan(
        option(AssetType.CALL), 1, spot=110.0,
        effective_date=date(2026, 8, 20), early=True,
        option_mark=12.0, days_to_expiry=2, dividend=0.50, force=True)
    assert event.reason is LifecycleReason.EARLY_EXERCISE


def test_invalid_native_contract_and_market_inputs_fail_closed():
    policy = OptionLifecyclePolicy()
    with pytest.raises(ValueError, match="non-zero integer"):
        policy.plan(option(AssetType.CALL), 0, spot=110,
                    effective_date=date(2026, 8, 24))
    with pytest.raises(ValueError, match="spot"):
        policy.plan(option(AssetType.CALL), 1, spot=float("nan"),
                    effective_date=date(2026, 8, 24))
