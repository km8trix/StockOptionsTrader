from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from analysis.pead_daily_acceptance import (
    ValidatedPeadDailyReconciliation,
    replay_and_validate_pead_daily_reconciliation,
)
from analysis.pead_daily_reconciliation import (
    current_implementation_manifests,
    pead_reconciliation_input,
)
from data.pead_economic_evidence import content_hash


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "research"
    / "pead_vq_locked_replication_v1"
    / "development_sample_report_v6.json"
)
PACKAGE = REPORT.parent


def _load(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text())


def test_report_core_excludes_receipt_derived_status_but_not_research_values():
    report = json.loads(REPORT.read_text())
    original = pead_reconciliation_input(report)

    status_only = copy.deepcopy(report)
    status_only["status"] = "completed"
    status_only["blockers"] = []
    status_only["completed_full_replication"] = True
    status_only["independent_reconciliation_hash"] = "f" * 64
    assert pead_reconciliation_input(status_only) == original

    changed_research = copy.deepcopy(report)
    changed_research["raw_portfolio_observations"][0]["signal"] += 0.0001
    assert pead_reconciliation_input(changed_research) != original


def test_money_calculation_code_manifests_are_explicit_and_disjoint():
    manifests = current_implementation_manifests(ROOT)
    primary = {row["path"] for row in manifests["primary"]["files"]}
    reference = {row["path"] for row in manifests["reference"]["files"]}

    assert primary == {
        "analysis/pead_daily_ledger.py",
        "analysis/pead_execution_ledger.py",
    }
    assert reference == {"analysis/pead_daily_reference.py"}
    assert primary.isdisjoint(reference)
    assert manifests["primary"]["code_hash"] != manifests["reference"]["code_hash"]
    assert manifests["shared"]["implementation_id"] == (
        "pead-daily-shared-verification-v2"
    )
    assert {row["path"] for row in manifests["shared"]["files"]} == {
        "analysis/independent_replication.py",
        "analysis/pead_daily_acceptance.py",
        "analysis/pead_daily_inputs.py",
        "analysis/pead_daily_reconciliation.py",
        "analysis/pead_economic_returns.py",
        "analysis/pead_reference_replication.py",
        "analysis/pead_replication.py",
        "data/pead_economic_evidence.py",
    }


def test_published_receipt_replays_and_matches_the_exact_report_core():
    report = _load("development_sample_report_v6.json")
    receipt = _load("daily_money_path_reconciliation_v3.json")

    token = replay_and_validate_pead_daily_reconciliation(
        receipt,
        source_report=report,
        modeled_ledger=_load("modeled_execution_ledger_v1.json"),
        independent_reference=_load("independent_reference_comparison_v5.json"),
        daily_inputs=_load("daily_input_snapshot_v1.json"),
        protocol=_load("daily_money_path_protocol_v2.json"),
        primary_daily_ledger=_load("primary_daily_ledger_v2.json"),
        independent_daily_ledger=_load("independent_daily_ledger_v2.json"),
        repository_root=ROOT,
    )
    assert type(token) is ValidatedPeadDailyReconciliation
    assert token.document == receipt
    assert token.artifact_hash == receipt["artifact_hash"]
    assert token.source_report_core_hash == pead_reconciliation_input(report)[
        "artifact_hash"
    ]
    payload = token.document["payload"]
    assert payload["bounded_modeled_daily_money_path_reconciliation_passed"] is True
    assert payload["comparison"]["discrepancy_count"] == 0
    assert payload["comparison"]["expected_observations"] == 4114
    assert payload["component_reconciliation"]["passed"] is True
    assert payload["component_reconciliation"]["discrepancy_count"] == 0
    assert payload["component_reconciliation"]["coverage"]["expected"] == {
        "cohort_daily_states": 688,
        "daily_constituent_states": 3784,
        "distribution_action_applications": 61,
    }


def test_validated_acceptance_token_is_sealed_after_replay():
    token = replay_and_validate_pead_daily_reconciliation(
        _load("daily_money_path_reconciliation_v3.json"),
        source_report=_load("development_sample_report_v6.json"),
        modeled_ledger=_load("modeled_execution_ledger_v1.json"),
        independent_reference=_load("independent_reference_comparison_v5.json"),
        daily_inputs=_load("daily_input_snapshot_v1.json"),
        protocol=_load("daily_money_path_protocol_v2.json"),
        primary_daily_ledger=_load("primary_daily_ledger_v2.json"),
        independent_daily_ledger=_load("independent_daily_ledger_v2.json"),
        repository_root=ROOT,
    )

    with pytest.raises(TypeError, match="immutable"):
        token._source_report_core_hash = "f" * 64
    with pytest.raises(TypeError, match="immutable"):
        token._document_json = "{}"


def test_replay_rejects_daily_input_with_mutated_upstream_binding():
    daily_inputs = copy.deepcopy(_load("daily_input_snapshot_v1.json"))
    daily_inputs["payload"]["bindings"]["source_report_file_sha256"] = "f" * 64
    daily_inputs["artifact_hash"] = content_hash(daily_inputs["payload"])

    with pytest.raises(ValueError, match="upstream bindings|frozen development sample"):
        replay_and_validate_pead_daily_reconciliation(
            _load("daily_money_path_reconciliation_v3.json"),
            source_report=_load("development_sample_report_v6.json"),
            modeled_ledger=_load("modeled_execution_ledger_v1.json"),
            independent_reference=_load("independent_reference_comparison_v5.json"),
            daily_inputs=daily_inputs,
            protocol=_load("daily_money_path_protocol_v2.json"),
            primary_daily_ledger=_load("primary_daily_ledger_v2.json"),
            independent_daily_ledger=_load("independent_daily_ledger_v2.json"),
            repository_root=ROOT,
        )


def test_validated_acceptance_token_cannot_be_constructed_from_json_alone():
    with pytest.raises(TypeError, match="full replay"):
        ValidatedPeadDailyReconciliation(
            object(),
            _load("daily_money_path_reconciliation_v3.json"),
            "f" * 64,
        )


def test_validated_acceptance_token_cannot_be_subclassed():
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class ForgedPeadReplay(ValidatedPeadDailyReconciliation):
            pass
