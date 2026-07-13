"""Opt-in margin, stock-borrow, and financing mechanics.

The legacy backtest remains a cash-flow simulation until a caller explicitly
constructs this module.  :class:`PortfolioMechanics` provides the two seams an
execution engine needs:

* :meth:`PortfolioMechanics.authorize` returns an immutable pre-trade record;
* :meth:`PortfolioMechanics.accrue` returns an immutable daily financing record.

All amounts are USD.  Stock quantities are shares, option quantities are
contracts, option prices are per share, and defined-risk package losses are
already expressed in dollars per package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Optional, Sequence


_EPSILON = 1e-9


class AccountMode(str, Enum):
    """Supported account buying-power regimes."""

    CASH = "CASH"
    REG_T = "REG_T"
    PORTFOLIO_MARGIN = "PORTFOLIO_MARGIN"


class ExposureKind(str, Enum):
    """Exposure classes understood by the conservative margin model."""

    LONG_STOCK = "LONG_STOCK"
    SHORT_STOCK = "SHORT_STOCK"
    LONG_OPTION = "LONG_OPTION"
    DEFINED_RISK_PACKAGE = "DEFINED_RISK_PACKAGE"


class AuthorizationDecision(str, Enum):
    """Machine-readable pre-trade decision."""

    AUTHORIZED = "AUTHORIZED"
    ACCOUNT_MODE_PROHIBITED = "ACCOUNT_MODE_PROHIBITED"
    INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"
    BORROW_UNAVAILABLE = "BORROW_UNAVAILABLE"
    BORROW_INSUFFICIENT = "BORROW_INSUFFICIENT"


class MissingBorrowQuoteError(RuntimeError):
    """A borrow fee cannot be priced without a valid, causal quote."""


def _finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a finite number >= {minimum}"
        ) from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return result


def _signed_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _finite_product(*values: float, label: str) -> float:
    try:
        result = math.prod(values)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite numeric range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} exceeds finite numeric range")
    return result


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _day(value: object, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, date):
        raise ValueError(f"{label} must be a date")
    return value


@dataclass(frozen=True)
class MarginPolicy:
    """Configurable initial and maintenance requirement rates.

    Short-stock rates are the trader's equity deposit.  Short sale proceeds
    are reported separately as restricted cash, rather than being counted a
    second time as buying power.  This makes the REG-T default the familiar
    50% initial equity requirement while still exposing the full 150% hold.
    """

    mode: AccountMode
    long_stock_initial: Optional[float] = None
    long_stock_maintenance: Optional[float] = None
    short_stock_initial: Optional[float] = None
    short_stock_maintenance: Optional[float] = None
    long_option_initial: Optional[float] = None
    long_option_maintenance: Optional[float] = None
    defined_risk_initial: Optional[float] = None
    defined_risk_maintenance: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AccountMode):
            try:
                object.__setattr__(self, "mode", AccountMode(self.mode))
            except (TypeError, ValueError) as exc:
                raise ValueError("mode must be a supported AccountMode") from exc

        defaults = {
            AccountMode.CASH: (1.0, 1.0, None, None, 1.0, 1.0, 1.0, 1.0),
            AccountMode.REG_T: (0.50, 0.25, 0.50, 0.30, 1.0, 1.0, 1.0, 1.0),
            AccountMode.PORTFOLIO_MARGIN: (
                0.15, 0.15, 0.20, 0.20, 1.0, 1.0, 0.50, 0.50
            ),
        }[self.mode]
        fields = (
            "long_stock_initial",
            "long_stock_maintenance",
            "short_stock_initial",
            "short_stock_maintenance",
            "long_option_initial",
            "long_option_maintenance",
            "defined_risk_initial",
            "defined_risk_maintenance",
        )
        for field_name, default in zip(fields, defaults):
            supplied = getattr(self, field_name)
            if supplied is None and default is None:
                continue
            value = default if supplied is None else supplied
            object.__setattr__(
                self, field_name, _finite(value, f"MarginPolicy.{field_name}")
            )


@dataclass(frozen=True)
class ExposureRequest:
    """One proposed exposure in native instrument units."""

    request_id: str
    symbol: str
    kind: ExposureKind
    quantity: int
    unit_price: Optional[float] = None
    contract_multiplier: Optional[int] = None
    max_loss_per_package: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol").upper())
        if not isinstance(self.kind, ExposureKind):
            try:
                object.__setattr__(self, "kind", ExposureKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("kind must be a supported ExposureKind") from exc
        object.__setattr__(
            self, "quantity", _positive_int(self.quantity, "quantity")
        )

        if self.kind is ExposureKind.DEFINED_RISK_PACKAGE:
            if self.unit_price is not None or self.contract_multiplier is not None:
                raise ValueError(
                    "defined-risk packages use max_loss_per_package only"
                )
            loss = _finite(
                self.max_loss_per_package,
                "max_loss_per_package",
                minimum=0.0,
            )
            if loss <= 0:
                raise ValueError("max_loss_per_package must be greater than zero")
            object.__setattr__(self, "max_loss_per_package", loss)
            return

        if self.max_loss_per_package is not None:
            raise ValueError("max_loss_per_package is only valid for packages")
        price = _finite(self.unit_price, "unit_price", minimum=0.0)
        if price <= 0:
            raise ValueError("unit_price must be greater than zero")
        object.__setattr__(self, "unit_price", price)
        expected_multiplier = (
            100 if self.kind is ExposureKind.LONG_OPTION else 1
        )
        multiplier = (
            expected_multiplier
            if self.contract_multiplier is None
            else _positive_int(self.contract_multiplier, "contract_multiplier")
        )
        if self.kind in {ExposureKind.LONG_STOCK, ExposureKind.SHORT_STOCK} \
                and multiplier != 1:
            raise ValueError("stock contract_multiplier must be 1")
        object.__setattr__(self, "contract_multiplier", multiplier)

    @property
    def exposure_value(self) -> float:
        """Stock notional, option premium, or package maximum loss."""
        if self.kind is ExposureKind.DEFINED_RISK_PACKAGE:
            assert self.max_loss_per_package is not None
            return _finite_product(
                self.quantity,
                self.max_loss_per_package,
                label="package maximum loss",
            )
        assert self.unit_price is not None
        assert self.contract_multiplier is not None
        return _finite_product(
            self.quantity,
            self.unit_price,
            self.contract_multiplier,
            label="exposure value",
        )


@dataclass(frozen=True)
class RequirementRecord:
    """Initial and maintenance buying-power amounts for a request."""

    request_id: str
    mode: AccountMode
    kind: ExposureKind
    exposure_value: float
    initial_requirement: float
    maintenance_requirement: float
    restricted_short_proceeds: float
    permitted: bool

    @property
    def total_initial_hold(self) -> float:
        """Initial equity requirement plus restricted short-sale proceeds."""
        return self.initial_requirement + self.restricted_short_proceeds


@dataclass(frozen=True)
class BorrowQuote:
    """A validity-dated locate and annualized borrow fee quote."""

    symbol: str
    as_of: date
    available_quantity: int
    annual_fee_rate: float
    valid_through: Optional[date] = None
    source: str = "configured"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol").upper())
        object.__setattr__(self, "as_of", _day(self.as_of, "as_of"))
        object.__setattr__(
            self,
            "available_quantity",
            _positive_int(
                self.available_quantity, "available_quantity", allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "annual_fee_rate",
            _finite(self.annual_fee_rate, "annual_fee_rate"),
        )
        valid = self.as_of if self.valid_through is None else _day(
            self.valid_through, "valid_through"
        )
        if valid < self.as_of:
            raise ValueError("valid_through cannot precede as_of")
        object.__setattr__(self, "valid_through", valid)
        object.__setattr__(self, "source", _text(self.source, "source"))


class BorrowInventory:
    """Immutable causal lookup over stock-borrow quotes."""

    def __init__(self, quotes: Iterable[BorrowQuote] = ()) -> None:
        by_key: dict[tuple[str, date], BorrowQuote] = {}
        for quote in quotes:
            if not isinstance(quote, BorrowQuote):
                raise ValueError("quotes must contain BorrowQuote records")
            key = (quote.symbol, quote.as_of)
            if key in by_key:
                raise ValueError(f"duplicate borrow quote for {quote.symbol} {quote.as_of}")
            by_key[key] = quote
        self._quotes = tuple(sorted(
            by_key.values(), key=lambda item: (item.symbol, item.as_of)
        ))

    def quote(self, symbol: str, as_of: date) -> Optional[BorrowQuote]:
        """Return the latest non-future, non-expired quote, if one exists."""
        normalised_symbol = _text(symbol, "symbol").upper()
        requested_day = _day(as_of, "as_of")
        candidates = (
            quote for quote in self._quotes
            if quote.symbol == normalised_symbol
            and quote.as_of <= requested_day
            and quote.valid_through is not None
            and requested_day <= quote.valid_through
        )
        return max(candidates, key=lambda item: item.as_of, default=None)


@dataclass(frozen=True)
class BorrowAuthorization:
    """Explicit stock-locate decision."""

    symbol: str
    as_of: date
    requested_quantity: int
    already_borrowed_quantity: int
    quote: Optional[BorrowQuote]
    approved: bool
    decision: AuthorizationDecision
    reason: str

    @property
    def remaining_quantity(self) -> int:
        if self.quote is None:
            return 0
        return max(
            0,
            self.quote.available_quantity
            - self.already_borrowed_quantity
            - (self.requested_quantity if self.approved else 0),
        )


@dataclass(frozen=True)
class BuyingPowerAuthorization:
    """Complete, immutable pre-trade authorization record."""

    request: ExposureRequest
    requirement: RequirementRecord
    available_buying_power: float
    approved: bool
    decision: AuthorizationDecision
    reason: str
    borrow: Optional[BorrowAuthorization] = None


@dataclass(frozen=True)
class ShortStockExposure:
    """One open short used to calculate a daily borrow charge."""

    symbol: str
    quantity: int
    mark_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol").upper())
        object.__setattr__(
            self, "quantity", _positive_int(self.quantity, "quantity")
        )
        price = _finite(self.mark_price, "mark_price")
        if price <= 0:
            raise ValueError("mark_price must be greater than zero")
        object.__setattr__(self, "mark_price", price)


@dataclass(frozen=True)
class BorrowFeeAccrual:
    """Daily fee line for one short stock."""

    symbol: str
    quantity: int
    mark_price: float
    market_value: float
    annual_fee_rate: float
    quote_as_of: date
    fee: float


@dataclass(frozen=True)
class FinancingAccrual:
    """Daily debit-interest and borrow-fee accounting record."""

    accrual_date: date
    mode: AccountMode
    cash_balance: float
    debit_principal: float
    annual_debit_rate: float
    day_count_basis: int
    debit_interest: float
    borrow_fees: tuple[BorrowFeeAccrual, ...]
    total_charge: float
    cash_delta: float
    compliance_flags: tuple[str, ...]


class PortfolioMechanics:
    """Facade for opt-in requirements, authorization, and daily accrual."""

    def __init__(
        self,
        margin_policy: MarginPolicy,
        *,
        borrow_inventory: Optional[BorrowInventory] = None,
        annual_debit_rate: float = 0.0,
        day_count_basis: int = 360,
    ) -> None:
        if not isinstance(margin_policy, MarginPolicy):
            raise ValueError("margin_policy must be a MarginPolicy")
        if borrow_inventory is not None and not isinstance(
                borrow_inventory, BorrowInventory):
            raise ValueError("borrow_inventory must be a BorrowInventory")
        self.margin_policy = margin_policy
        self.borrow_inventory = borrow_inventory or BorrowInventory()
        self.annual_debit_rate = _finite(
            annual_debit_rate, "annual_debit_rate"
        )
        self.day_count_basis = _positive_int(
            day_count_basis, "day_count_basis"
        )
        self._last_accrual_date: Optional[date] = None

    def requirements(self, request: ExposureRequest) -> RequirementRecord:
        """Calculate initial and maintenance requirements without authorizing."""
        if not isinstance(request, ExposureRequest):
            raise ValueError("request must be an ExposureRequest")
        policy = self.margin_policy
        rates = {
            ExposureKind.LONG_STOCK: (
                policy.long_stock_initial, policy.long_stock_maintenance
            ),
            ExposureKind.SHORT_STOCK: (
                policy.short_stock_initial, policy.short_stock_maintenance
            ),
            ExposureKind.LONG_OPTION: (
                policy.long_option_initial, policy.long_option_maintenance
            ),
            ExposureKind.DEFINED_RISK_PACKAGE: (
                policy.defined_risk_initial, policy.defined_risk_maintenance
            ),
        }[request.kind]
        permitted = rates[0] is not None and rates[1] is not None
        initial_rate = 0.0 if rates[0] is None else rates[0]
        maintenance_rate = 0.0 if rates[1] is None else rates[1]
        restricted = (
            request.exposure_value
            if request.kind is ExposureKind.SHORT_STOCK and permitted
            else 0.0
        )
        return RequirementRecord(
            request_id=request.request_id,
            mode=policy.mode,
            kind=request.kind,
            exposure_value=request.exposure_value,
            initial_requirement=request.exposure_value * initial_rate,
            maintenance_requirement=request.exposure_value * maintenance_rate,
            restricted_short_proceeds=restricted,
            permitted=permitted,
        )

    def borrow_authorization(
        self,
        symbol: str,
        quantity: int,
        as_of: date,
        *,
        already_borrowed_quantity: int = 0,
    ) -> BorrowAuthorization:
        """Check a short request against a validity-dated locate."""
        normalised_symbol = _text(symbol, "symbol").upper()
        requested = _positive_int(quantity, "quantity")
        borrowed = _positive_int(
            already_borrowed_quantity,
            "already_borrowed_quantity",
            allow_zero=True,
        )
        requested_day = _day(as_of, "as_of")
        quote = self.borrow_inventory.quote(normalised_symbol, requested_day)
        if quote is None:
            return BorrowAuthorization(
                normalised_symbol,
                requested_day,
                requested,
                borrowed,
                None,
                False,
                AuthorizationDecision.BORROW_UNAVAILABLE,
                "no valid borrow quote is available as of the request date",
            )
        remaining = max(0, quote.available_quantity - borrowed)
        if requested > remaining:
            return BorrowAuthorization(
                normalised_symbol,
                requested_day,
                requested,
                borrowed,
                quote,
                False,
                AuthorizationDecision.BORROW_INSUFFICIENT,
                f"borrow availability is {remaining} shares",
            )
        return BorrowAuthorization(
            normalised_symbol,
            requested_day,
            requested,
            borrowed,
            quote,
            True,
            AuthorizationDecision.AUTHORIZED,
            "borrow locate and fee quote are valid",
        )

    def authorize(
        self,
        request: ExposureRequest,
        available_buying_power: float,
        *,
        as_of: Optional[date] = None,
        already_borrowed_quantity: int = 0,
    ) -> BuyingPowerAuthorization:
        """Return an explicit authorization; invalid economic data raises."""
        available = _finite(
            available_buying_power, "available_buying_power"
        )
        requirement = self.requirements(request)
        if not requirement.permitted:
            return BuyingPowerAuthorization(
                request,
                requirement,
                available,
                False,
                AuthorizationDecision.ACCOUNT_MODE_PROHIBITED,
                f"{request.kind.value} is prohibited in {requirement.mode.value}",
            )

        borrow: Optional[BorrowAuthorization] = None
        if request.kind is ExposureKind.SHORT_STOCK:
            if as_of is None:
                borrow = BorrowAuthorization(
                    request.symbol,
                    date.min,
                    request.quantity,
                    already_borrowed_quantity,
                    None,
                    False,
                    AuthorizationDecision.BORROW_UNAVAILABLE,
                    "short authorization requires an as_of date",
                )
            else:
                borrow = self.borrow_authorization(
                    request.symbol,
                    request.quantity,
                    as_of,
                    already_borrowed_quantity=already_borrowed_quantity,
                )
            if not borrow.approved:
                return BuyingPowerAuthorization(
                    request,
                    requirement,
                    available,
                    False,
                    borrow.decision,
                    borrow.reason,
                    borrow,
                )

        if requirement.initial_requirement > available + _EPSILON:
            return BuyingPowerAuthorization(
                request,
                requirement,
                available,
                False,
                AuthorizationDecision.INSUFFICIENT_BUYING_POWER,
                (
                    f"requires {requirement.initial_requirement:.2f} buying "
                    f"power; {available:.2f} is available"
                ),
                borrow,
            )
        return BuyingPowerAuthorization(
            request,
            requirement,
            available,
            True,
            AuthorizationDecision.AUTHORIZED,
            "account, buying power, and borrow checks passed",
            borrow,
        )

    def accrue(
        self,
        accrual_date: date,
        cash_balance: float,
        short_stock: Sequence[ShortStockExposure] = (),
    ) -> FinancingAccrual:
        """Price one causal financing day without mutating portfolio cash.

        Calls must be strictly chronological, preventing accidental duplicate
        accrual.  Every short requires a quote valid on ``accrual_date``;
        missing fee data raises rather than granting free borrow.
        """
        on_date = _day(accrual_date, "accrual_date")
        if self._last_accrual_date is not None \
                and on_date <= self._last_accrual_date:
            raise ValueError("accrual dates must be strictly increasing")
        cash = _signed_finite(cash_balance, "cash_balance")
        debit_principal = max(0.0, -cash)
        debit_interest = (
            debit_principal * self.annual_debit_rate / self.day_count_basis
        )
        borrow_lines: list[BorrowFeeAccrual] = []
        seen_symbols: set[str] = set()
        exposures = tuple(short_stock)
        if not all(isinstance(item, ShortStockExposure) for item in exposures):
            raise ValueError("short_stock must contain ShortStockExposure records")
        for exposure in sorted(exposures, key=lambda item: item.symbol):
            if exposure.symbol in seen_symbols:
                raise ValueError("short_stock symbols must be unique")
            seen_symbols.add(exposure.symbol)
            quote = self.borrow_inventory.quote(exposure.symbol, on_date)
            if quote is None:
                raise MissingBorrowQuoteError(
                    f"no valid borrow fee quote for {exposure.symbol} on {on_date}"
                )
            market_value = _finite_product(
                exposure.quantity,
                exposure.mark_price,
                label=f"{exposure.symbol} short market value",
            )
            fee = _finite_product(
                market_value,
                quote.annual_fee_rate,
                label=f"{exposure.symbol} borrow fee numerator",
            ) / self.day_count_basis
            borrow_lines.append(BorrowFeeAccrual(
                symbol=exposure.symbol,
                quantity=exposure.quantity,
                mark_price=exposure.mark_price,
                market_value=market_value,
                annual_fee_rate=quote.annual_fee_rate,
                quote_as_of=quote.as_of,
                fee=fee,
            ))
        total = debit_interest + sum(line.fee for line in borrow_lines)
        flags = []
        if self.margin_policy.mode is AccountMode.CASH and debit_principal > 0:
            flags.append("NEGATIVE_CASH_IN_CASH_ACCOUNT")
        if self.margin_policy.mode is AccountMode.CASH and borrow_lines:
            flags.append("SHORT_STOCK_IN_CASH_ACCOUNT")
        record = FinancingAccrual(
            accrual_date=on_date,
            mode=self.margin_policy.mode,
            cash_balance=cash,
            debit_principal=debit_principal,
            annual_debit_rate=self.annual_debit_rate,
            day_count_basis=self.day_count_basis,
            debit_interest=debit_interest,
            borrow_fees=tuple(borrow_lines),
            total_charge=total,
            cash_delta=-total,
            compliance_flags=tuple(flags),
        )
        self._last_accrual_date = on_date
        return record


__all__ = [
    "AccountMode",
    "AuthorizationDecision",
    "BorrowAuthorization",
    "BorrowFeeAccrual",
    "BorrowInventory",
    "BorrowQuote",
    "BuyingPowerAuthorization",
    "ExposureKind",
    "ExposureRequest",
    "FinancingAccrual",
    "MarginPolicy",
    "MissingBorrowQuoteError",
    "PortfolioMechanics",
    "RequirementRecord",
    "ShortStockExposure",
]
