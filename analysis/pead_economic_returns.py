"""Fail-closed PEAD cash-return reconstruction.

This module is intentionally narrower than a backtest.  It reconstructs the
economic value of one already-frozen holding path from split-normalized SEP
closes and explicitly enumerated cash distributions, without reinvestment.  It
also audits every corporate action in the holding interval.  Generic ACTIONS
``value`` data is accepted only for an ordinary ``dividend`` row and is never a
terminal settlement fallback.

The caller remains responsible for proving the provider semantics of the
dividend fields.  Until that separate evidence exists, a resolved result is
labelled ``mechanically_reconstructed_nonqualifying`` rather than executable or
qualifying research evidence.
"""

from __future__ import annotations

from datetime import date
import math
from numbers import Real
from typing import Any, Mapping, Sequence


ACTION_FIELDS = frozenset(
    {
        "date",
        "action",
        "ticker",
        "name",
        "value",
        "contraticker",
        "contraname",
    }
)
IGNORED_ISSUER_EXTERNAL_ACTIONS = frozenset({"acquisitionof"})
TERMINAL_ACTIONS = frozenset(
    {
        "acquisitionby",
        "bankruptcyliquidation",
        "delisted",
        "mergerfrom",
        "mergerto",
        "regulatorydelisting",
        "voluntarydelisting",
    }
)
UNSUPPORTED_HOLDER_ACTIONS = frozenset(
    {
        "adrratiosplit",
        "relation",
        "spinoff",
        "spinoffdividend",
        "spunofffrom",
        "split",
        "tickerchangefrom",
        "tickerchangeto",
    }
)


class EconomicReturnError(ValueError):
    """An input violates the exact mechanical-return contract."""


def _iso_date(value: Any, field: str) -> date:
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise EconomicReturnError(f"{field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EconomicReturnError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise EconomicReturnError(f"{field} must be a canonical ISO date")
    return parsed


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EconomicReturnError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise EconomicReturnError(f"{field} must be {qualifier}")
    return 0.0 if number == 0.0 else number


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise EconomicReturnError(f"{field} must be canonical text or null")
    return value


def validate_action_rows(
    rows: Any,
    *,
    requested_tickers: Sequence[str],
    start: str | date,
    end: str | date,
) -> list[dict[str, Any]]:
    """Validate and canonically order one bounded ACTIONS slice."""
    if not isinstance(rows, list):
        raise EconomicReturnError("corporate-action slice must be an array")
    tickers = tuple(requested_tickers)
    if any(
        not isinstance(ticker, str)
        or not ticker
        or ticker != ticker.strip().upper()
        for ticker in tickers
    ):
        raise EconomicReturnError("requested tickers must be canonical uppercase text")
    if len(tickers) != len(set(tickers)):
        raise EconomicReturnError("requested tickers must be unique")
    allowed = set(tickers)
    lower = _iso_date(start, "action slice start")
    upper = _iso_date(end, "action slice end")
    if lower > upper:
        raise EconomicReturnError("action slice window is reversed")

    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != ACTION_FIELDS:
            raise EconomicReturnError(
                f"corporate-action row {index} fields differ from the contract"
            )
        action_date = _iso_date(raw["date"], f"corporate-action row {index}.date")
        ticker = raw["ticker"]
        action = raw["action"]
        name = raw["name"]
        if (
            not lower <= action_date <= upper
            or ticker not in allowed
            or not isinstance(action, str)
            or not action
            or action != action.strip().lower()
            or not isinstance(name, str)
            or not name
            or name != name.strip()
        ):
            raise EconomicReturnError(
                f"corporate-action row {index} has a malformed key"
            )
        raw_value = raw["value"]
        value = None if raw_value is None else _finite(
            raw_value, f"corporate-action row {index}.value"
        )
        contra_ticker = _optional_text(
            raw["contraticker"], f"corporate-action row {index}.contraticker"
        )
        contra_name = _optional_text(
            raw["contraname"], f"corporate-action row {index}.contraname"
        )
        identity = (
            action_date,
            ticker,
            name,
            action,
            contra_name,
            contra_ticker,
        )
        if identity in identities:
            raise EconomicReturnError("corporate-action slice has a duplicate key")
        identities.add(identity)
        normalized.append(
            {
                "date": action_date.isoformat(),
                "action": action,
                "ticker": ticker,
                "name": name,
                "value": value,
                "contraticker": contra_ticker,
                "contraname": contra_name,
            }
        )
    normalized.sort(
        key=lambda item: (
            item["ticker"],
            item["date"],
            item["action"],
            item["name"],
            item["contraticker"] or "",
            item["contraname"] or "",
        )
    )
    return normalized


def _price_map(value: Mapping[Any, Any], field: str) -> dict[date, float]:
    if not isinstance(value, Mapping):
        raise EconomicReturnError(f"{field} must be a date-to-price mapping")
    result: dict[date, float] = {}
    for raw_day, raw_price in value.items():
        day = _iso_date(raw_day, f"{field} date")
        if day in result:
            raise EconomicReturnError(f"{field} contains a duplicate date")
        result[day] = _finite(raw_price, f"{field}[{day.isoformat()}]", positive=True)
    return result


def _unresolved(
    *,
    reason: str,
    entry_price: float,
    exit_price: float | None,
    closeadj_return: float | None,
    distributions: list[dict[str, Any]],
    ignored_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "unresolved",
        "reason": reason,
        "pricing_path": "SEP.close_plus_explicit_cash_no_reinvestment_candidate",
        "entry_price_split_normalized": entry_price,
        "exit_price_split_normalized": exit_price,
        "cash_distributions": distributions,
        "cash_total": float(sum(item["amount"] for item in distributions)),
        "terminal_settlement_id": None,
        "gross_terminal_value": None,
        "gross_economic_return": None,
        "closeadj_diagnostic_return": closeadj_return,
        "ignored_actions": ignored_actions,
    }


def reconstruct_cash_return(
    *,
    ticker: str,
    entry_date: str | date,
    exit_date: str | date,
    split_normalized_prices: Mapping[Any, Any],
    adjusted_prices: Mapping[Any, Any],
    action_rows: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    currency: str,
    terminal_settlements: Sequence[Mapping[str, Any]],
    adjustment_absolute_tolerance: float,
    adjustment_relative_tolerance: float,
) -> dict[str, Any]:
    """Reconstruct one exact holding path, or return an explicit exclusion.

    Dividends are included when ``entry_date < ex_date <= exit_date``.  Their
    candidate amounts must independently agree with the contemporaneous change
    in ``SEP.closeadj`` versus ``SEP.close`` within the frozen tolerance.  That
    price-adjustment check detects unit/split-basis drift but does not itself
    prove the vendor field semantics.
    """
    if (
        not isinstance(ticker, str)
        or not ticker
        or ticker != ticker.strip().upper()
    ):
        raise EconomicReturnError("ticker must be canonical uppercase text")
    entry_day = _iso_date(entry_date, "entry_date")
    exit_day = _iso_date(exit_date, "exit_date")
    if entry_day >= exit_day:
        raise EconomicReturnError("economic return requires entry_date < exit_date")
    absolute = _finite(
        adjustment_absolute_tolerance, "adjustment absolute tolerance"
    )
    relative = _finite(
        adjustment_relative_tolerance, "adjustment relative tolerance"
    )
    if absolute < 0 or relative < 0:
        raise EconomicReturnError("adjustment tolerances must be non-negative")
    if currency != "USD":
        raise EconomicReturnError("cash-return reconstruction requires USD currency")

    close = _price_map(split_normalized_prices, "split-normalized prices")
    adjusted = _price_map(adjusted_prices, "adjusted prices")
    if entry_day not in close:
        raise EconomicReturnError("missing exact split-normalized entry price")
    entry_price = close[entry_day]
    closeadj_return = None
    if entry_day in adjusted and exit_day in adjusted:
        closeadj_return = adjusted[exit_day] / adjusted[entry_day] - 1.0

    if not isinstance(lifecycle, Mapping) or lifecycle.get("status") != "validated":
        return _unresolved(
            reason="security_lifecycle_unresolved",
            entry_price=entry_price,
            exit_price=close.get(exit_day),
            closeadj_return=closeadj_return,
            distributions=[],
            ignored_actions=[],
        )
    if lifecycle.get("isdelisted") not in {"N", "Y"}:
        raise EconomicReturnError("security lifecycle has an invalid delisting status")
    raw_permaticker = lifecycle.get("permaticker")
    if type(raw_permaticker) is not int or raw_permaticker <= 0:
        raise EconomicReturnError("security lifecycle has an invalid permaticker")
    final_day = _iso_date(lifecycle.get("lastpricedate"), "lifecycle lastpricedate")

    held_actions = [
        dict(item)
        for item in action_rows
        if item.get("ticker") == ticker
        and entry_day < _iso_date(item.get("date"), "action date") <= exit_day
    ]
    held_actions.sort(
        key=lambda item: (
            item["date"], item["action"], item["name"],
            item.get("contraticker") or "", item.get("contraname") or "",
        )
    )
    distributions: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    terminal_action_rows: list[dict[str, Any]] = []
    dividend_action_rows: dict[date, list[dict[str, Any]]] = {}
    for action in held_actions:
        kind = action["action"]
        if kind in IGNORED_ISSUER_EXTERNAL_ACTIONS:
            ignored.append(
                {
                    "date": action["date"],
                    "action": kind,
                    "contraticker": action.get("contraticker"),
                    "reason": "issuer_external_acquisition_no_direct_holder_cash_flow",
                }
            )
            continue
        if kind in TERMINAL_ACTIONS:
            terminal_action_rows.append(action)
            continue
        if kind in UNSUPPORTED_HOLDER_ACTIONS or kind != "dividend":
            return _unresolved(
                reason=f"held_corporate_action_terms_unresolved:{kind}",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                closeadj_return=closeadj_return,
                distributions=distributions,
                ignored_actions=ignored,
            )
        action_day = _iso_date(action["date"], "dividend date")
        dividend_action_rows.setdefault(action_day, []).append(action)

    # A regular and special dividend can share one ex-date.  SEP.closeadj
    # exposes one aggregate adjustment for that date, so validate the sum once
    # while retaining each provider action and action key in the evidence.
    for action_day in sorted(dividend_action_rows):
        actions_on_day = dividend_action_rows[action_day]
        amounts = [
            _finite(action.get("value"), "dividend value", positive=True)
            for action in actions_on_day
        ]
        aggregate_amount = sum(amounts)
        if action_day not in close or action_day not in adjusted:
            return _unresolved(
                reason="dividend_date_missing_exact_price_adjustment_evidence",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                closeadj_return=closeadj_return,
                distributions=distributions,
                ignored_actions=ignored,
            )
        prior_days = sorted(
            day for day in set(close).intersection(adjusted) if day < action_day
        )
        if not prior_days:
            return _unresolved(
                reason="dividend_missing_prior_price_adjustment_evidence",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                closeadj_return=closeadj_return,
                distributions=distributions,
                ignored_actions=ignored,
            )
        previous_day = prior_days[-1]
        implied = (
            close[previous_day]
            * adjusted[action_day]
            / adjusted[previous_day]
            - close[action_day]
        )
        if not math.isfinite(implied):
            raise EconomicReturnError("dividend adjustment check is non-finite")
        error = abs(implied - aggregate_amount)
        allowed = absolute + relative * max(abs(implied), abs(aggregate_amount))
        if error > allowed:
            return _unresolved(
                reason="cash_distribution_split_basis_or_amount_mismatch",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                closeadj_return=closeadj_return,
                distributions=distributions,
                ignored_actions=ignored,
            )
        for action, amount in zip(actions_on_day, amounts, strict=True):
            distributions.append(
                {
                    "action_key": {
                        "date": action["date"],
                        "ticker": ticker,
                        "name": action["name"],
                        "action": action["action"],
                        "contraname": action.get("contraname"),
                        "contraticker": action.get("contraticker"),
                    },
                    "date": action["date"],
                    "amount": amount,
                    "adjustment_previous_session": previous_day.isoformat(),
                    "adjustment_implied_amount": implied,
                    "adjustment_absolute_error": error,
                    "adjustment_allowed_error": allowed,
                }
            )

    delisted_before_exit = lifecycle["isdelisted"] == "Y" and final_day < exit_day
    terminal_id = None
    terminal_value = None
    exit_price = close.get(exit_day)
    if delisted_before_exit or terminal_action_rows:
        matches = []
        for settlement in terminal_settlements:
            if not isinstance(settlement, Mapping):
                raise EconomicReturnError("terminal settlement must be an object")
            if (
                settlement.get("ticker") == ticker
                and settlement.get("permaticker") == raw_permaticker
                and _iso_date(
                    settlement.get("last_price_date"),
                    "terminal settlement last_price_date",
                )
                == final_day
            ):
                matches.append(settlement)
        if len(matches) != 1:
            return _unresolved(
                reason="held_terminal_settlement_missing_or_ambiguous",
                entry_price=entry_price,
                exit_price=None,
                closeadj_return=closeadj_return,
                distributions=distributions,
                ignored_actions=ignored,
            )
        settlement = matches[0]
        settlement_day = _iso_date(
            settlement.get("settlement_date"), "terminal settlement date"
        )
        if settlement_day > exit_day:
            return _unresolved(
                reason="terminal_cash_not_received_by_exact_horizon",
                entry_price=entry_price,
                exit_price=None,
                closeadj_return=closeadj_return,
                distributions=distributions,
                ignored_actions=ignored,
            )
        terminal_value = _finite(
            settlement.get("cash_per_terminal_share"),
            "terminal cash per share",
            positive=True,
        )
        terminal_id = (
            f"{raw_permaticker}:{final_day.isoformat()}:"
            f"{settlement_day.isoformat()}"
        )
        exit_price = None
    elif exit_price is None:
        return _unresolved(
            reason="missing_exact_split_normalized_exit_price",
            entry_price=entry_price,
            exit_price=None,
            closeadj_return=closeadj_return,
            distributions=distributions,
            ignored_actions=ignored,
        )

    cash_total = float(sum(item["amount"] for item in distributions))
    gross_terminal_value = (
        terminal_value if terminal_value is not None else float(exit_price)
    ) + cash_total
    gross_return = gross_terminal_value / entry_price - 1.0
    if not math.isfinite(gross_return):
        raise EconomicReturnError("reconstructed economic return is non-finite")
    return {
        "status": "mechanically_reconstructed_nonqualifying",
        "reason": None,
        "pricing_path": "SEP.close_plus_explicit_cash_no_reinvestment_candidate",
        "entry_price_split_normalized": entry_price,
        "exit_price_split_normalized": exit_price,
        "cash_distributions": distributions,
        "cash_total": cash_total,
        "terminal_settlement_id": terminal_id,
        "gross_terminal_value": gross_terminal_value,
        "gross_economic_return": gross_return,
        "closeadj_diagnostic_return": closeadj_return,
        "ignored_actions": ignored,
    }


__all__ = [
    "ACTION_FIELDS",
    "EconomicReturnError",
    "IGNORED_ISSUER_EXTERNAL_ACTIONS",
    "TERMINAL_ACTIONS",
    "UNSUPPORTED_HOLDER_ACTIONS",
    "reconstruct_cash_return",
    "validate_action_rows",
]
