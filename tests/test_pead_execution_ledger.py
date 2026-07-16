from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import json
import math

import pytest

from analysis.pead_execution_ledger import (
    PeadExecutionLedgerError,
    build_pead_execution_ledger,
    replication_observations,
    source_report_file_sha256,
    validate_pead_execution_ledger,
)
from data.pead_economic_evidence import (
    build_current_unproven_cash_distribution_semantics,
    build_empty_terminal_settlement_ledger,
    canonical_json,
    content_hash,
)
from scripts.pead_execution_ledger import main


def _wrap(payload: dict) -> dict:
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _resolution(
    *, ticker: str, entry: float = 10.0,
    exit_price: float = 10.0, cash: float = 0.0,
) -> tuple[float, dict]:
    distributions = []
    if cash:
        distributions.append(
            {
                "date": "2020-01-15",
                "amount": cash,
                "action_key": {
                    "action": "dividend",
                    "date": "2020-01-15",
                    "ticker": ticker,
                    "name": f"{ticker} INC",
                    "contraticker": "N/A",
                    "contraname": "N/A",
                },
            }
        )
    gross_terminal = exit_price + cash
    gross_return = gross_terminal / entry - 1.0
    return gross_return, {
        "cash_distributions": distributions,
        "cash_total": cash,
        "closeadj_diagnostic_return": gross_return,
        "entry_price_split_normalized": entry,
        "exit_price_split_normalized": exit_price,
        "gross_economic_return": gross_return,
        "gross_terminal_value": gross_terminal,
        "ignored_actions": [],
        "pricing_path": "SEP.close_plus_explicit_cash_no_reinvestment_candidate",
        "reason": None,
        "status": "mechanically_reconstructed_nonqualifying",
        "terminal_settlement_id": None,
    }


def _observation(index: int) -> dict:
    ticker = f"T{index:02d}"
    signal = float(index - 5)
    exits = {0: (9.0, 0.2), 1: (9.0, 0.0), 8: (12.0, 0.0), 9: (11.5, 0.5)}
    exit_price, cash = exits.get(index, (10.0, 0.0))
    item = {
        "formation_date": "2020-01-02",
        "entry_date": "2020-01-03",
        "ticker": ticker,
        "m_ticker": f"M{index:02d}",
        "signal": signal,
        "entry_close_split_normalized": 10.0,
        "source_event_key": {
            "m_ticker": f"M{index:02d}",
            "per_end_date": "2019-12-31",
            "per_type": "Q",
        },
    }
    for horizon in (21, 63):
        gross, resolution = _resolution(
            ticker=ticker,
            exit_price=exit_price,
            cash=cash,
        )
        item[f"target_exit_date_{horizon}"] = (
            "2020-02-03" if horizon == 21 else "2020-04-01"
        )
        item[f"economic_forward_return_candidate_{horizon}"] = gross
        item[f"economic_return_resolution_{horizon}"] = resolution
    return item


def _frozen_selection(count: int) -> list[dict]:
    if count < 10:
        return []
    k = max(1, int(0.2 * count))
    names = [f"M{index:02d}" for index in range(count)]
    return [
        {
            "date": "2020-01-02",
            "eligible_names": count,
            "names_per_leg": k,
            "short_m_tickers": names[:k],
            "long_m_tickers": names[-k:],
        }
    ]


def _report(count: int = 10) -> dict:
    combined = _wrap({"schema_version": "fixture_combined_snapshot.v1"})
    semantics = build_current_unproven_cash_distribution_semantics()
    terminal = build_empty_terminal_settlement_ledger()
    economic = _wrap(
        {
            "schema_version": "pead_economic_return_inputs.v1",
            "candidate_id": "pead-vq-locked-replication-v1",
            "combined_data_snapshot_hash": combined["artifact_hash"],
            "cash_distribution_semantics": semantics,
            "terminal_settlement_ledger": terminal,
        }
    )
    manifest = _wrap(
        {
            "schema_version": "pead_research_manifest_binding.v1",
            "candidate_id": "pead-vq-locked-replication-v1",
            "fixture": True,
        }
    )
    observations = [_observation(index) for index in range(count)]
    lifecycle = {
        item["ticker"]: {
            "status": "validated",
            "permaticker": 1000 + index,
        }
        for index, item in enumerate(observations)
    }
    selection = _frozen_selection(count)
    return {
        "schema_version": "pead_replication_report.v6",
        "candidate_id": "pead-vq-locked-replication-v1",
        "combined_data_snapshot": combined,
        "economic_return_inputs": economic,
        "research_manifest_binding": manifest,
        "configuration": {
            "horizons_sessions": [21, 63],
            "minimum_names_per_formation_per_slice": 10,
            "quantile": 0.2,
            "signal_tie_break": "ascending stable m_ticker",
            "one_way_cost_bps_per_trade_per_leg": 30,
            "fixed_total_round_trip_bps": 120,
            "cash_distribution_holding_interval": (
                "entry_date_exclusive_exit_date_inclusive"
            ),
            "cash_distribution_reinvestment": False,
            "terminal_payout_from_actions_value_allowed": False,
        },
        "coverage": {"security_lifecycle_diagnostics": lifecycle},
        "slice_coverage": {
            "pooled": {
                "horizons": {
                    "21": {"frozen_selections": copy.deepcopy(selection)},
                    "63": {"frozen_selections": copy.deepcopy(selection)},
                }
            }
        },
        "raw_portfolio_observations": observations,
        "blockers": [
            "cash_distribution_semantics_source_missing",
            "historical_sample_is_not_full_window_evidence",
        ],
    }


def _resign(document: dict) -> None:
    document["artifact_hash"] = content_hash(document["payload"])


def test_builds_exact_modeled_cohort_money_equations_and_nonclaims():
    report = _report()
    artifact = build_pead_execution_ledger(
        report, report_sha256=source_report_file_sha256(report)
    )
    payload = validate_pead_execution_ledger(artifact, source_report=report)[
        "payload"
    ]

    assert payload["qualifying_evidence"] is False
    assert payload["replication_evidence_eligible"] is False
    assert payload["paper_execution_evidence"] is False
    assert payload["modeled_execution_claim"]["broker_fill_ids"] is None
    assert payload["coverage"] == {
        "formation_horizon_cells": 2,
        "below_floor_cells": 0,
        "admitted_cohorts": 2,
        "excluded_selected_path_cohorts": 0,
        "modeled_constituent_paths": 8,
        "atomic_checkpoints": 6,
        "replication_projection_observations": 24,
    }
    assert len(payload["selection_manifest"][0]["ranked_constituents"]) == 10
    assert payload["selection_manifest"][0]["short_m_tickers"] == ["M00", "M01"]
    assert payload["selection_manifest"][0]["long_m_tickers"] == ["M08", "M09"]
    summary = payload["cohort_summaries"][0]
    assert Decimal(summary["gross_factor_return"]) == Decimal("0.290000000000000000")
    assert Decimal(summary["net_factor_return"]) == Decimal("0.278000000000000000")
    assert Decimal(summary["terminal_nav"]) == Decimal("1.278000000000000000")
    short_dividend = next(
        row for row in payload["constituent_ledger"]
        if row["cohort_id"].endswith(":21") and row["m_ticker"] == "M00"
    )
    assert Decimal(short_dividend["distribution_pnl"]) < 0


def test_nine_name_formation_is_exhaustively_manifested_but_not_modeled():
    artifact = build_pead_execution_ledger(_report(9))
    payload = artifact["payload"]
    assert [item["status"] for item in payload["selection_manifest"]] == [
        "below_minimum_names",
        "below_minimum_names",
    ]
    assert all(
        len(item["ranked_constituents"]) == 9
        for item in payload["selection_manifest"]
    )
    assert payload["constituent_ledger"] == []
    assert payload["cohort_summaries"] == []
    assert payload["atomic_checkpoints"] == []


def test_one_selected_unresolved_path_excludes_the_whole_horizon_cohort():
    report = _report()
    selected = report["raw_portfolio_observations"][0]
    selected["economic_forward_return_candidate_21"] = None
    selected["economic_return_resolution_21"]["status"] = "unresolved"
    selected["economic_return_resolution_21"]["reason"] = "missing_exit"

    payload = build_pead_execution_ledger(report)["payload"]
    manifests = {item["horizon_sessions"]: item for item in payload["selection_manifest"]}
    assert manifests[21]["status"] == "excluded_selected_path"
    assert manifests[63]["status"] == "admitted"
    assert {row["horizon_sessions"] for row in payload["constituent_ledger"]} == {63}
    assert payload["coverage"]["excluded_selected_path_cohorts"] == 1


def test_terminal_path_is_never_backfilled_from_actions_or_silently_modeled():
    report = _report()
    resolution = report["raw_portfolio_observations"][9][
        "economic_return_resolution_21"
    ]
    resolution["terminal_settlement_id"] = "separately-sourced-record"

    payload = build_pead_execution_ledger(report)["payload"]
    manifest = next(
        item for item in payload["selection_manifest"]
        if item["horizon_sessions"] == 21
    )
    assert manifest["status"] == "excluded_selected_path"
    assert manifest["reason"] == "selected_terminal_settlement_path_not_supported"
    assert all(
        row["horizon_sessions"] != 21 for row in payload["constituent_ledger"]
    )


def test_selection_and_money_outputs_are_stable_under_raw_input_permutation():
    report = _report()
    first = build_pead_execution_ledger(report)["payload"]
    permuted = copy.deepcopy(report)
    permuted["raw_portfolio_observations"].reverse()
    second = build_pead_execution_ledger(permuted)["payload"]

    assert first["bindings"]["source_report_file_sha256"] != second["bindings"][
        "source_report_file_sha256"
    ]
    for field in (
        "selection_manifest",
        "constituent_ledger",
        "cohort_summaries",
        "atomic_checkpoints",
        "coverage",
    ):
        assert first[field] == second[field]


def test_hash_formula_and_source_bound_tampering_fail_closed():
    report = _report()
    artifact = build_pead_execution_ledger(report)

    unhashed = copy.deepcopy(artifact)
    unhashed["payload"]["constituent_ledger"][0]["net_pnl"] = (
        "9.000000000000000000"
    )
    with pytest.raises(PeadExecutionLedgerError, match="hash mismatch"):
        validate_pead_execution_ledger(unhashed, source_report=report)

    self_consistent = copy.deepcopy(unhashed)
    _resign(self_consistent)
    with pytest.raises(PeadExecutionLedgerError, match="money equation"):
        validate_pead_execution_ledger(self_consistent, source_report=report)

    source_tamper = copy.deepcopy(artifact)
    source_tamper["payload"]["constituent_ledger"][0]["source_event_key"][
        "per_type"
    ] = "A"
    _resign(source_tamper)
    with pytest.raises(PeadExecutionLedgerError, match="provenance"):
        validate_pead_execution_ledger(source_tamper, source_report=report)

    # Even a self-consistent identity rewrite cannot be projected without the
    # exact bound source report.
    identity_rewrite = copy.deepcopy(artifact)
    constituent = identity_rewrite["payload"]["constituent_ledger"][0]
    manifest = next(
        item for item in identity_rewrite["payload"]["selection_manifest"]
        if item["cohort_id"] == constituent["cohort_id"]
    )
    ranked = next(
        item for item in manifest["ranked_constituents"]
        if item["m_ticker"] == constituent["m_ticker"]
    )
    constituent["source_event_key"]["per_type"] = "A"
    ranked["source_event_key"]["per_type"] = "A"
    _resign(identity_rewrite)
    with pytest.raises(PeadExecutionLedgerError, match="source-report rebuild"):
        validate_pead_execution_ledger(identity_rewrite, source_report=report)


def test_generic_projection_has_exact_finite_numeric_contract_fields():
    report = _report()
    observations = replication_observations(
        build_pead_execution_ledger(report), source_report=report
    )
    assert len(observations) == 24
    assert len({canonical_json(item["key"]) for item in observations}) == 24
    for item in observations:
        assert set(item) == {
            "key", "eligibility", "signal", "rank", "target", "order",
            "position", "cash", "fees", "pnl",
        }
        assert item["eligibility"] is True
        assert all(
            isinstance(item[field], float) and math.isfinite(item[field])
            for field in (
                "signal", "rank", "target", "order", "position", "cash",
                "fees", "pnl",
            )
        )
    exit_rows = [
        item for item in observations if item["key"]["checkpoint"] == "modeled_exit"
    ]
    assert all(item["target"] == 0 and item["position"] == 0 for item in exit_rows)


def test_self_rehashed_admitted_cohort_cannot_be_erased():
    report = _report()
    artifact = build_pead_execution_ledger(report)
    tampered = copy.deepcopy(artifact)
    manifest = tampered["payload"]["selection_manifest"][0]
    cohort_id = manifest["cohort_id"]
    manifest["status"] = "excluded_selected_path"
    manifest["reason"] = "selected_economic_return_path_unresolved"
    tampered["payload"]["constituent_ledger"] = [
        row for row in tampered["payload"]["constituent_ledger"]
        if row["cohort_id"] != cohort_id
    ]
    tampered["payload"]["cohort_summaries"] = [
        row for row in tampered["payload"]["cohort_summaries"]
        if row["cohort_id"] != cohort_id
    ]
    tampered["payload"]["atomic_checkpoints"] = [
        row for row in tampered["payload"]["atomic_checkpoints"]
        if row["cohort_id"] != cohort_id
    ]
    tampered["payload"]["coverage"]["admitted_cohorts"] -= 1
    tampered["payload"]["coverage"]["excluded_selected_path_cohorts"] += 1
    tampered["payload"]["coverage"]["modeled_constituent_paths"] -= 4
    tampered["payload"]["coverage"]["atomic_checkpoints"] -= 3
    tampered["payload"]["coverage"]["replication_projection_observations"] -= 12
    _resign(tampered)

    with pytest.raises(PeadExecutionLedgerError, match="admission differs"):
        validate_pead_execution_ledger(tampered, source_report=report)


def test_decimal_context_is_pinned_and_internal_drift_fails_closed():
    report = _report()
    expected = build_pead_execution_ledger(report)
    with localcontext() as context:
        context.prec = 12
        assert build_pead_execution_ledger(report) == expected
        validate_pead_execution_ledger(expected, source_report=report)

    drift = copy.deepcopy(expected)
    original = Decimal(drift["payload"]["constituent_ledger"][0]["price_pnl"])
    drift["payload"]["constituent_ledger"][0]["price_pnl"] = format(
        original + Decimal("0.0000000000001"), "f"
    )
    _resign(drift)
    with pytest.raises(PeadExecutionLedgerError, match="money equation"):
        validate_pead_execution_ledger(drift, source_report=report)


def test_cli_requires_canonical_source_bytes_and_creates_immutable_artifact(
    tmp_path, capsys,
):
    report = _report()
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "ledger.json"
    report_path.write_text(canonical_json(report) + "\n", encoding="utf-8")

    assert main(["--report", str(report_path), "--output", str(output_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["paper_execution_evidence"] is False
    document = json.loads(output_path.read_text(encoding="utf-8"))
    validate_pead_execution_ledger(document, source_report=report)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert main(["--report", str(report_path), "--output", str(output_path)]) == 2
    assert "not the canonical" in json.loads(capsys.readouterr().err)["error"]
