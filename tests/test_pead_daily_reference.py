from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal, getcontext
import inspect

import pytest

import analysis.pead_daily_reference as daily_reference
from analysis.pead_daily_reference import (
    PeadIndependentDailyLedgerError,
    build_independent_daily_ledger,
    replication_observations,
    validate_independent_daily_ledger,
)
from data.pead_economic_evidence import content_hash


def _wrapper(payload):
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _date_series(count: int) -> list[str]:
    start = date(2020, 1, 2)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _fixture_documents(
    *,
    exit_distribution: bool = False,
    same_date_distribution_parts: bool = False,
    name_count: int = 10,
):
    sessions = _date_series(64)
    formation = "2020-01-01"
    observations = []
    formation_rows = []
    selected_paths = []
    path_coverage = []
    prices = []
    currencies = []

    distribution_dates = [sessions[5]]
    if exit_distribution:
        distribution_dates.append(sessions[21])
    action_specs = [
        (sessions[5], Decimal("0.4"), "T09 INC A"),
        (sessions[5], Decimal("0.6"), "T09 INC B"),
    ] if same_date_distribution_parts else [
        (sessions[5], Decimal("1"), "T09 INC"),
    ]
    if exit_distribution:
        action_specs.append((sessions[21], Decimal("1"), "T09 INC"))
    action_rows = []
    for distribution_date, amount, action_name in action_specs:
        action_rows.append(
            {
                "date": distribution_date,
                "action": "dividend",
                "ticker": "T09",
                "name": action_name,
                "value": format(amount, "f"),
                "contraticker": None,
                "contraname": None,
            }
        )

    names_per_leg = int(Decimal("0.2") * name_count)
    for index in range(name_count):
        ticker = f"T{index:02d}"
        currencies.append({"ticker": ticker, "currency": "USD"})
        adjusted = Decimal("100")
        for day in sessions:
            if ticker == "T09" and day in distribution_dates:
                adjusted *= Decimal("1.01")
            prices.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "close": "100.000000000000000000",
                    "closeadj": format(adjusted, "f"),
                }
            )
        source_key = {
            "m_ticker": ticker,
            "per_end_date": "2019-12-31",
            "per_type": "Q",
        }
        observation = {
            "formation_date": formation,
            "entry_date": sessions[0],
            "ticker": ticker,
            "m_ticker": ticker,
            "signal": index / 100,
            "source_event_key": source_key,
        }
        for horizon in (21, 63):
            held_distributions = (
                [spec for spec in action_specs if spec[0] <= sessions[horizon]]
                if ticker == "T09"
                else []
            )
            distributions = [
                {
                    "date": day,
                    "amount": float(amount),
                    "action_key": {
                        "date": day,
                        "ticker": ticker,
                        "name": action_name,
                        "action": "dividend",
                        "contraname": None,
                        "contraticker": None,
                    },
                    "adjustment_previous_session": sessions[sessions.index(day) - 1],
                    "adjustment_implied_amount": 1.0,
                    "adjustment_absolute_error": 0.0,
                    "adjustment_allowed_error": 0.00000101,
                }
                for day, amount, action_name in held_distributions
            ]
            cash_total = sum(
                (amount for _, amount, _ in held_distributions), Decimal("0")
            )
            observation[f"target_exit_date_{horizon}"] = sessions[horizon]
            observation[f"economic_forward_return_candidate_{horizon}"] = float(
                cash_total / 100
            )
            observation[f"economic_return_resolution_{horizon}"] = {
                "status": "mechanically_reconstructed_nonqualifying",
                "reason": None,
                "terminal_settlement_id": None,
                "entry_price_split_normalized": 100.0,
                "exit_price_split_normalized": 100.0,
                "cash_total": float(cash_total),
                "cash_distributions": distributions,
            }
        observations.append(observation)

        for horizon in (21, 63):
            leg = (
                "short"
                if index < names_per_leg
                else "long"
                if index >= name_count - names_per_leg
                else None
            )
            cohort_id = f"pooled:{formation}:{horizon}"
            formation_row = {
                "cohort_id": cohort_id,
                "formation_date": formation,
                "entry_date": sessions[0],
                "exit_date": sessions[horizon],
                "horizon_sessions": horizon,
                "ticker": ticker,
                "m_ticker": ticker,
                "permaticker": 1000 + index,
                "rank": index + 1,
                "selected_leg": leg,
                "cohort_status": "admitted",
                "cohort_reason": None,
                "source_event_key": source_key,
                "signal": f"{index / 100:.18f}",
            }
            formation_rows.append(formation_row)
            if leg is not None:
                selected_paths.append(
                    {
                        key: formation_row[key]
                        for key in (
                            "cohort_id",
                            "formation_date",
                            "entry_date",
                            "exit_date",
                            "horizon_sessions",
                            "ticker",
                            "m_ticker",
                            "permaticker",
                            "rank",
                            "source_event_key",
                        )
                    }
                    | {"leg": leg, "signal": formation_row["signal"]}
                )
                path_coverage.append(
                    {
                        "cohort_id": cohort_id,
                        "m_ticker": ticker,
                        "session_dates": sessions[: horizon + 1],
                    }
                )

    reference = _wrapper({"outputs": {"reference": {"portfolio_observations": observations}}})
    daily_inputs = _wrapper(
        {
            "schema_version": "pead_daily_input_snapshot.v1",
            "candidate_id": "pead-vq-locked-replication-v1",
            "bindings": {"combined_data_snapshot_hash": "a" * 64},
            "distribution_semantics": {
                "adjustment_check_absolute_tolerance": "0.000000010000000000",
                "adjustment_check_relative_tolerance": "0.000001000000000000",
            },
            "formation_observations": formation_rows,
            "selected_paths": selected_paths,
            "path_coverage": path_coverage,
            "sessions": sessions,
            "prices": prices,
            "actions": action_rows,
            "currencies": currencies,
        }
    )
    return reference, daily_inputs


@pytest.fixture(autouse=True)
def _neutral_upstream_verifiers(monkeypatch):
    monkeypatch.setattr(
        daily_reference,
        "verify_reference_artifact",
        lambda document: document["payload"],
    )
    monkeypatch.setattr(
        daily_reference,
        "verify_pead_daily_input_snapshot",
        lambda document: document,
    )


def test_independent_event_loop_is_exhaustive_and_keeps_receivables_out_of_cash():
    reference, daily_inputs = _fixture_documents()
    artifact = build_independent_daily_ledger(reference, daily_inputs)
    payload = artifact["payload"]

    assert payload["schema_version"] == "pead_independent_daily_ledger.v2"
    assert payload["coverage"] == {
        "formation_horizon_cells": 2,
        "formation_states": 20,
        "admitted_cohorts": 2,
        "selected_constituent_paths": 8,
        "daily_constituent_states": 344,
        "cohort_daily_states": 86,
        "distribution_applications": 2,
        "replication_projection_observations": 364,
    }
    cohort = [row for row in payload["cohort_daily_states"] if row["horizon_sessions"] == 21]
    assert cohort[0]["settled_cash"] == "0.994000000000000000"
    dividend_state = cohort[5]
    assert dividend_state["settled_cash"] == "0.994000000000000000"
    assert dividend_state["distribution_receivable"] == "0.005000000000000000"
    assert cohort[-1]["settled_cash"] == "0.988000000000000000"
    assert cohort[-1]["nav"] == "0.993000000000000000"
    assert cohort[-1]["pnl"] == "-0.007000000000000000"
    assert cohort[-1]["open_position_count"] == 0
    dividend_name_state = next(
        row
        for row in payload["daily_constituent_states"]
        if row["horizon_sessions"] == 21
        and row["ticker"] == "T09"
        and row["sequence"] == 5
    )
    assert dividend_name_state["price_pnl"] == "0.000000000000000000"
    assert dividend_name_state["net_pnl_contribution"] == "0.005000000000000000"

    assert (
        validate_independent_daily_ledger(
            artifact, reference_artifact=reference, daily_inputs=daily_inputs
        )
        == artifact
    )


def test_projection_contains_every_formation_and_selected_daily_key():
    reference, daily_inputs = _fixture_documents()
    artifact = build_independent_daily_ledger(reference, daily_inputs)
    projected = replication_observations(
        artifact, reference_artifact=reference, daily_inputs=daily_inputs
    )

    assert len(projected) == 364
    assert len({content_hash(row["key"]) for row in projected}) == 364
    assert {row["key"]["checkpoint"] for row in projected} == {
        "formation_target",
        "entry_close",
        "mark_close",
        "exit_close",
    }
    assert all(row["eligibility"] is True for row in projected)
    assert all("source_event_key" in row["key"] for row in projected)


def test_exit_date_distribution_is_accrued_before_liquidation():
    reference, daily_inputs = _fixture_documents(exit_distribution=True)
    artifact = build_independent_daily_ledger(reference, daily_inputs)
    cohort = [
        row for row in artifact["payload"]["cohort_daily_states"] if row["horizon_sessions"] == 21
    ]

    assert cohort[-1]["checkpoint"] == "exit_close"
    assert cohort[-1]["distribution_receivable"] == "0.010000000000000000"
    assert cohort[-1]["settled_cash"] == "0.988000000000000000"
    assert cohort[-1]["pnl"] == "-0.002000000000000000"


def test_same_date_distribution_parts_use_aggregate_adjustment_and_preserve_keys():
    reference, daily_inputs = _fixture_documents(same_date_distribution_parts=True)
    artifact = build_independent_daily_ledger(reference, daily_inputs)
    payload = artifact["payload"]

    assert payload["coverage"]["distribution_applications"] == 4
    state = next(
        row
        for row in payload["daily_constituent_states"]
        if row["horizon_sessions"] == 21
        and row["ticker"] == "T09"
        and row["sequence"] == 5
    )
    assert state["distribution_accrual_today"] == "0.005000000000000000"
    assert [key["name"] for key in state["applied_distribution_action_keys"]] == [
        "T09 INC A",
        "T09 INC B",
    ]


def test_nonterminating_target_fraction_stays_exact_until_serialization():
    reference, daily_inputs = _fixture_documents(name_count=15)
    artifact = build_independent_daily_ledger(reference, daily_inputs)
    entry = next(
        row
        for row in artifact["payload"]["daily_constituent_states"]
        if row["horizon_sessions"] == 21
        and row["ticker"] == "T14"
        and row["sequence"] == 0
    )

    assert entry["target"] == "0.333333333333333333"
    assert entry["order"] == "0.003333333333333333333333"
    assert entry["net_pnl_contribution"] == "-0.001000000000000000"


def test_missing_internal_price_bar_fails_closed():
    reference, daily_inputs = _fixture_documents()
    daily_inputs["payload"]["prices"] = [
        row
        for row in daily_inputs["payload"]["prices"]
        if not (row["ticker"] == "T09" and row["date"] == _date_series(64)[10])
    ]
    daily_inputs["artifact_hash"] = content_hash(daily_inputs["payload"])

    with pytest.raises(PeadIndependentDailyLedgerError, match="missing price bar"):
        build_independent_daily_ledger(reference, daily_inputs)


def test_non_usd_selected_name_fails_closed():
    reference, daily_inputs = _fixture_documents()
    next(row for row in daily_inputs["payload"]["currencies"] if row["ticker"] == "T09")[
        "currency"
    ] = "CAD"
    daily_inputs["artifact_hash"] = content_hash(daily_inputs["payload"])

    with pytest.raises(PeadIndependentDailyLedgerError, match="requires USD"):
        build_independent_daily_ledger(reference, daily_inputs)


def test_self_rehashed_state_tamper_is_rejected_by_exhaustive_rebuild():
    reference, daily_inputs = _fixture_documents()
    artifact = build_independent_daily_ledger(reference, daily_inputs)
    tampered = deepcopy(artifact)
    tampered["payload"]["cohort_daily_states"][0]["pnl"] = "9.000000000000000000"
    tampered["artifact_hash"] = content_hash(tampered["payload"])

    with pytest.raises(
        PeadIndependentDailyLedgerError,
        match="differs from its exhaustive input rebuild",
    ):
        validate_independent_daily_ledger(
            tampered, reference_artifact=reference, daily_inputs=daily_inputs
        )


def test_decimal_context_and_primary_implementation_cannot_change_results():
    reference, daily_inputs = _fixture_documents()
    before = getcontext().prec
    try:
        getcontext().prec = 6
        first = build_independent_daily_ledger(reference, daily_inputs)
        getcontext().prec = 38
        second = build_independent_daily_ledger(reference, daily_inputs)
    finally:
        getcontext().prec = before
    assert first == second

    source = inspect.getsource(daily_reference)
    forbidden_imports = (
        "import analysis.pead_execution_ledger",
        "from analysis.pead_execution_ledger",
        "import analysis.pead_replication",
        "from analysis.pead_replication",
        "import analysis.pead_daily_ledger",
        "from analysis.pead_daily_ledger",
    )
    assert all(value not in source for value in forbidden_imports)
