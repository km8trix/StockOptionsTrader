"""Focused contracts for opt-in portfolio financing mechanics."""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from portfolio.mechanics import (
    AccountMode,
    AuthorizationDecision,
    BorrowInventory,
    BorrowQuote,
    ExposureKind,
    ExposureRequest,
    MarginPolicy,
    MissingBorrowQuoteError,
    PortfolioMechanics,
    ShortStockExposure,
)


DAY = date(2026, 7, 13)


def request(kind, *, quantity=10, price=100.0, request_id="req-1"):
    kwargs = {}
    if kind is ExposureKind.DEFINED_RISK_PACKAGE:
        kwargs["max_loss_per_package"] = price
    else:
        kwargs["unit_price"] = price
    return ExposureRequest(
        request_id=request_id,
        symbol="spy",
        kind=kind,
        quantity=quantity,
        **kwargs,
    )


class TestRequirements:
    @pytest.mark.parametrize(
        "mode,expected",
        [
            (AccountMode.CASH, (1000.0, 1000.0)),
            (AccountMode.REG_T, (500.0, 250.0)),
            (AccountMode.PORTFOLIO_MARGIN, (150.0, 150.0)),
        ],
    )
    def test_long_stock_defaults_by_account_mode(self, mode, expected):
        mechanics = PortfolioMechanics(MarginPolicy(mode))
        actual = mechanics.requirements(request(ExposureKind.LONG_STOCK))
        assert (actual.initial_requirement,
                actual.maintenance_requirement) == expected
        assert actual.permitted

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (AccountMode.REG_T, (500.0, 300.0)),
            (AccountMode.PORTFOLIO_MARGIN, (200.0, 200.0)),
        ],
    )
    def test_short_stock_equity_and_restricted_proceeds(self, mode, expected):
        mechanics = PortfolioMechanics(MarginPolicy(mode))
        actual = mechanics.requirements(request(ExposureKind.SHORT_STOCK))
        assert (actual.initial_requirement,
                actual.maintenance_requirement) == expected
        assert actual.restricted_short_proceeds == 1000.0
        assert actual.total_initial_hold == expected[0] + 1000.0

    def test_cash_mode_short_is_explicitly_prohibited(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.CASH))
        result = mechanics.authorize(
            request(ExposureKind.SHORT_STOCK), 100_000.0, as_of=DAY)
        assert not result.approved
        assert result.decision is AuthorizationDecision.ACCOUNT_MODE_PROHIBITED
        assert not result.requirement.permitted
        assert result.borrow is None

    def test_long_option_uses_contract_multiplier(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.REG_T))
        option = request(ExposureKind.LONG_OPTION, quantity=3, price=2.50)
        actual = mechanics.requirements(option)
        assert option.contract_multiplier == 100
        assert actual.exposure_value == 750.0
        assert actual.initial_requirement == 750.0

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (AccountMode.CASH, 2000.0),
            (AccountMode.REG_T, 2000.0),
            (AccountMode.PORTFOLIO_MARGIN, 1000.0),
        ],
    )
    def test_defined_risk_package_uses_declared_max_loss(self, mode, expected):
        mechanics = PortfolioMechanics(MarginPolicy(mode))
        package = request(
            ExposureKind.DEFINED_RISK_PACKAGE, quantity=4, price=500.0)
        assert mechanics.requirements(package).initial_requirement == expected

    def test_rates_are_configurable(self):
        policy = MarginPolicy(
            AccountMode.REG_T,
            long_stock_initial=0.60,
            long_stock_maintenance=0.40,
        )
        actual = PortfolioMechanics(policy).requirements(
            request(ExposureKind.LONG_STOCK))
        assert actual.initial_requirement == 600.0
        assert actual.maintenance_requirement == 400.0


class TestAuthorization:
    def test_insufficient_buying_power_returns_record(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.REG_T))
        result = mechanics.authorize(
            request(ExposureKind.LONG_STOCK), available_buying_power=499.99)
        assert not result.approved
        assert result.decision is AuthorizationDecision.INSUFFICIENT_BUYING_POWER
        assert result.requirement.initial_requirement == 500.0

    def test_short_fails_closed_without_borrow(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.REG_T))
        result = mechanics.authorize(
            request(ExposureKind.SHORT_STOCK), 1000.0, as_of=DAY)
        assert not result.approved
        assert result.decision is AuthorizationDecision.BORROW_UNAVAILABLE
        assert result.borrow is not None
        assert result.borrow.quote is None

    def test_short_requires_enough_incremental_borrow(self):
        inventory = BorrowInventory([
            BorrowQuote("SPY", DAY, 15, 0.12),
        ])
        mechanics = PortfolioMechanics(
            MarginPolicy(AccountMode.REG_T), borrow_inventory=inventory)
        result = mechanics.authorize(
            request(ExposureKind.SHORT_STOCK),
            1000.0,
            as_of=DAY,
            already_borrowed_quantity=6,
        )
        assert not result.approved
        assert result.decision is AuthorizationDecision.BORROW_INSUFFICIENT
        assert result.borrow is not None
        assert result.borrow.remaining_quantity == 9

    def test_short_authorizes_with_locate_and_buying_power(self):
        inventory = BorrowInventory([
            BorrowQuote("SPY", DAY, 20, 0.12),
        ])
        mechanics = PortfolioMechanics(
            MarginPolicy(AccountMode.REG_T), borrow_inventory=inventory)
        result = mechanics.authorize(
            request(ExposureKind.SHORT_STOCK), 500.0, as_of=DAY)
        assert result.approved
        assert result.decision is AuthorizationDecision.AUTHORIZED
        assert result.borrow is not None and result.borrow.approved
        assert result.borrow.remaining_quantity == 10

    def test_future_and_expired_quotes_are_never_used(self):
        prior = date(2026, 7, 12)
        future = date(2026, 7, 14)
        inventory = BorrowInventory([
            BorrowQuote("SPY", prior, 100, 0.10),
            BorrowQuote("SPY", future, 100, 0.20),
        ])
        mechanics = PortfolioMechanics(
            MarginPolicy(AccountMode.REG_T), borrow_inventory=inventory)
        assert mechanics.borrow_inventory.quote("SPY", DAY) is None

        valid_inventory = BorrowInventory([
            BorrowQuote("SPY", prior, 100, 0.10, valid_through=DAY),
        ])
        assert valid_inventory.quote("SPY", DAY).annual_fee_rate == 0.10

    def test_records_are_immutable(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.CASH))
        result = mechanics.authorize(
            request(ExposureKind.LONG_STOCK), 1000.0)
        with pytest.raises(FrozenInstanceError):
            result.approved = False


class TestFinancingAccrual:
    def test_debit_interest_and_borrow_fee_are_charged_daily(self):
        inventory = BorrowInventory([
            BorrowQuote("ABC", DAY, 1000, 0.36),
        ])
        mechanics = PortfolioMechanics(
            MarginPolicy(AccountMode.REG_T),
            borrow_inventory=inventory,
            annual_debit_rate=0.18,
            day_count_basis=360,
        )
        actual = mechanics.accrue(
            DAY,
            -10_000.0,
            [ShortStockExposure("ABC", 100, 20.0)],
        )
        assert actual.debit_interest == pytest.approx(5.0)
        assert actual.borrow_fees[0].market_value == 2000.0
        assert actual.borrow_fees[0].fee == pytest.approx(2.0)
        assert actual.total_charge == pytest.approx(7.0)
        assert actual.cash_delta == pytest.approx(-7.0)
        assert actual.compliance_flags == ()

    def test_positive_cash_has_no_debit_interest(self):
        mechanics = PortfolioMechanics(
            MarginPolicy(AccountMode.REG_T), annual_debit_rate=0.18)
        actual = mechanics.accrue(DAY, 10_000.0)
        assert actual.debit_principal == 0.0
        assert actual.debit_interest == 0.0
        assert actual.total_charge == 0.0

    def test_missing_borrow_fee_quote_fails_closed(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.REG_T))
        with pytest.raises(MissingBorrowQuoteError, match="ABC"):
            mechanics.accrue(
                DAY, 0.0, [ShortStockExposure("ABC", 5, 20.0)])

    def test_duplicate_or_backdated_accrual_is_rejected(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.REG_T))
        mechanics.accrue(DAY, 0.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            mechanics.accrue(DAY, 0.0)

    def test_non_finite_cash_and_derived_exposure_fail_closed(self):
        mechanics = PortfolioMechanics(MarginPolicy(AccountMode.REG_T))
        with pytest.raises(ValueError, match="cash_balance"):
            mechanics.accrue(DAY, float("nan"))
        huge = request(ExposureKind.LONG_STOCK, price=1e308)
        with pytest.raises(ValueError, match="exposure value"):
            mechanics.requirements(huge)

    def test_cash_account_violations_are_recorded_but_still_charged(self):
        inventory = BorrowInventory([
            BorrowQuote("ABC", DAY, 10, 0.36),
        ])
        mechanics = PortfolioMechanics(
            MarginPolicy(AccountMode.CASH),
            borrow_inventory=inventory,
            annual_debit_rate=0.18,
        )
        actual = mechanics.accrue(
            DAY, -1000.0, [ShortStockExposure("ABC", 1, 100.0)])
        assert actual.total_charge > 0
        assert actual.compliance_flags == (
            "NEGATIVE_CASH_IN_CASH_ACCOUNT",
            "SHORT_STOCK_IN_CASH_ACCOUNT",
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExposureRequest("x", "SPY", ExposureKind.LONG_STOCK, 1,
                                unit_price=float("nan")),
        lambda: ExposureRequest("x", "SPY", ExposureKind.LONG_OPTION, 1,
                                unit_price=float("inf")),
        lambda: ExposureRequest("x", "SPY", ExposureKind.DEFINED_RISK_PACKAGE,
                                1, max_loss_per_package=-1.0),
        lambda: BorrowQuote("SPY", DAY, 1, float("nan")),
        lambda: MarginPolicy(AccountMode.REG_T, long_stock_initial=-0.1),
        lambda: ShortStockExposure("SPY", 1, float("inf")),
    ],
)
def test_non_finite_or_negative_economic_inputs_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
