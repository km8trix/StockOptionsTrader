from __future__ import annotations

from copy import deepcopy

import pytest

from analysis.pead_economic_returns import (
    EconomicReturnError,
    reconstruct_cash_return,
    validate_action_rows,
)


def _action(day, action="dividend", value=0.25, *, contra=None):
    return {
        "date": day,
        "action": action,
        "ticker": "ABC",
        "name": "ABC Corp",
        "value": value,
        "contraticker": contra,
        "contraname": f"{contra} Corp" if contra else None,
    }


def _lifecycle(*, delisted=False, last="2030-01-01"):
    return {
        "status": "validated",
        "isdelisted": "Y" if delisted else "N",
        "permaticker": 123,
        "lastpricedate": last,
        "sep_lastpricedate": last if delisted else None,
    }


def _prices():
    close = {
        "2020-01-02": 10.0,
        "2020-01-04": 10.5,
        "2020-01-05": 11.0,
    }
    # Each adjustment transition exactly implies the declared cash amount.
    adjusted = {
        "2020-01-02": 10.0,
        "2020-01-04": 10.75,
        "2020-01-05": 11.5 * 10.75 / 10.5,
    }
    return close, adjusted


def _run(actions, **overrides):
    close, adjusted = _prices()
    values = {
        "ticker": "ABC",
        "entry_date": "2020-01-02",
        "exit_date": "2020-01-05",
        "split_normalized_prices": close,
        "adjusted_prices": adjusted,
        "action_rows": validate_action_rows(
            actions,
            requested_tickers=["ABC"],
            start="2020-01-01",
            end="2020-01-31",
        ),
        "lifecycle": _lifecycle(),
        "currency": "USD",
        "terminal_settlements": [],
        "adjustment_absolute_tolerance": 1e-12,
        "adjustment_relative_tolerance": 1e-12,
    }
    values.update(overrides)
    return reconstruct_cash_return(**values)


def test_adds_cash_without_reinvestment_and_uses_exact_boundaries():
    result = _run(
        [
            _action("2020-01-02", value=99.0),  # entry-date: not owned
            _action("2020-01-04", value=0.25),
            _action("2020-01-05", value=0.50),  # exit-date: still owned
        ]
    )

    assert result["status"] == "mechanically_reconstructed_nonqualifying"
    assert [item["amount"] for item in result["cash_distributions"]] == [0.25, 0.5]
    assert result["cash_total"] == pytest.approx(0.75)
    assert result["gross_terminal_value"] == pytest.approx(11.75)
    assert result["gross_economic_return"] == pytest.approx(0.175)
    assert result["gross_economic_return"] != pytest.approx(
        result["closeadj_diagnostic_return"]
    )


def test_no_distribution_is_exact_split_normalized_price_return():
    result = _run([])
    assert result["gross_economic_return"] == pytest.approx(0.1)
    assert result["cash_total"] == 0.0


def test_same_date_regular_and_special_dividends_use_one_aggregate_adjustment():
    regular = _action("2020-01-04", value=0.10)
    special = _action("2020-01-04", value=0.15)
    special["name"] = "ABC Corp special distribution"

    result = _run([regular, special])

    assert result["status"] == "mechanically_reconstructed_nonqualifying"
    assert [row["amount"] for row in result["cash_distributions"]] == [0.10, 0.15]
    assert result["cash_total"] == pytest.approx(0.25)
    assert {
        row["action_key"]["name"] for row in result["cash_distributions"]
    } == {"ABC Corp", "ABC Corp special distribution"}
    assert all(
        row["adjustment_implied_amount"] == pytest.approx(0.25)
        and row["adjustment_absolute_error"] == pytest.approx(0.0)
        for row in result["cash_distributions"]
    )


def test_issuer_acquisition_is_audited_but_not_shareholder_cash():
    result = _run([_action("2020-01-04", "acquisitionof", 9999.0, contra="XYZ")])
    assert result["gross_economic_return"] == pytest.approx(0.1)
    assert result["cash_total"] == 0.0
    assert result["ignored_actions"] == [
        {
            "date": "2020-01-04",
            "action": "acquisitionof",
            "contraticker": "XYZ",
            "reason": "issuer_external_acquisition_no_direct_holder_cash_flow",
        }
    ]


@pytest.mark.parametrize("action", ["spinoff", "split", "relation"])
def test_holder_transform_or_unknown_action_fails_closed(action):
    result = _run([_action("2020-01-04", action, 12345.0)])
    assert result["status"] == "unresolved"
    assert result["reason"] == f"held_corporate_action_terms_unresolved:{action}"
    assert result["gross_economic_return"] is None


def test_delisting_never_uses_actions_value_or_last_close_as_payout():
    result = _run(
        [_action("2020-01-04", "acquisitionby", 50_000.0, contra="XYZ")],
        lifecycle=_lifecycle(delisted=True, last="2020-01-04"),
    )
    assert result["status"] == "unresolved"
    assert result["reason"] == "held_terminal_settlement_missing_or_ambiguous"
    assert result["gross_terminal_value"] is None


def test_separately_sourced_cash_terminal_record_can_resolve():
    result = _run(
        [_action("2020-01-04", "acquisitionby", 50_000.0, contra="XYZ")],
        lifecycle=_lifecycle(delisted=True, last="2020-01-04"),
        terminal_settlements=[
            {
                "ticker": "ABC",
                "permaticker": 123,
                "last_price_date": "2020-01-04",
                "settlement_date": "2020-01-05",
                "cash_per_terminal_share": 12.0,
                "source_receipts": [{"independent": True}],
            }
        ],
    )
    assert result["gross_terminal_value"] == 12.0
    assert result["gross_economic_return"] == pytest.approx(0.2)
    assert result["exit_price_split_normalized"] is None


def test_dividend_basis_mismatch_is_not_rescaled_or_silently_accepted():
    result = _run([_action("2020-01-04", value=1.0)])
    assert result["status"] == "unresolved"
    assert result["reason"] == "cash_distribution_split_basis_or_amount_mismatch"


def test_action_schema_duplicates_nonfinite_and_currency_fail_closed():
    row = _action("2020-01-04")
    with pytest.raises(EconomicReturnError, match="duplicate"):
        validate_action_rows(
            [row, deepcopy(row)],
            requested_tickers=["ABC"],
            start="2020-01-01",
            end="2020-01-31",
        )
    bad = deepcopy(row)
    bad["value"] = float("nan")
    with pytest.raises(EconomicReturnError, match="finite"):
        validate_action_rows(
            [bad],
            requested_tickers=["ABC"],
            start="2020-01-01",
            end="2020-01-31",
        )
    with pytest.raises(EconomicReturnError, match="USD"):
        _run([], currency="CAD")
