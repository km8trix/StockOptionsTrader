from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "research" / "pead_vq_source_qualification_v3"
CANDIDATE_SHA256 = "5f17bde9c923bef237f099508c40299a815026088471de11c424b5370fbe14c6"


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AssertionError(f"non-JSON numeric constant: {value}")


def _strict_json(name: str) -> dict:
    raw = (PACKAGE / name).read_bytes()
    assert raw.startswith(b"{")
    assert raw.endswith(b"\n")
    document = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_no_duplicates,
        parse_constant=_reject_constant,
    )
    assert isinstance(document, dict)
    return document


def test_v3_package_json_is_strict_and_frozen_candidate_is_unchanged():
    candidate = _strict_json("candidate_specification.json")
    architecture = _strict_json("source_architecture.json")
    status = _strict_json("qualification_status.json")

    assert candidate["schema_version"] == "pead_candidate_specification.v3"
    assert architecture["schema_version"] == "pead_source_architecture.v2"
    assert status["schema_version"] == "pead_source_qualification_status.v2"
    assert {candidate["candidate_id"], architecture["candidate_id"], status["candidate_id"]} == {
        "pead-vq-source-qualification-v3"
    }
    assert (
        hashlib.sha256((PACKAGE / "candidate_specification.json").read_bytes()).hexdigest()
        == CANDIDATE_SHA256
    )


def test_architecture_names_every_implemented_authority_boundary():
    contracts = _strict_json("source_architecture.json")["implemented_contracts"]
    expected = {
        "sharadar_acquisition_and_identity": (
            "../../data/sharadar_source_evidence.py",
            "pead_sharadar_source_snapshot.v1",
        ),
        "sharadar_official_semantics": (
            "../../data/sharadar_semantics_evidence.py",
            "sharadar_semantics_source_receipt.v1",
        ),
        "event_census_and_child_universe": (
            "../../data/pead_event_universe.py",
            "pead_event_universe.v2",
        ),
        "sharadar_event_universe_replay": (
            "../../data/pead_sharadar_event_universe_replay.py",
            "pead_sharadar_event_universe_replay.v1",
        ),
        "annual_event_universe_index": (
            "../../data/pead_event_universe_index.py",
            "pead_event_universe_index.v1",
        ),
        "normalized_consensus": (
            "../../data/pead_consensus_evidence.py",
            "pead_consensus_evidence.v1",
        ),
        "consensus_normalization_replay": (
            "../../data/pead_consensus_replay.py",
            "pead_consensus_normalization_replay.v1",
        ),
        "announcement_actual_evidence": (
            "../../data/pead_announcement_evidence.py",
            "pead_announcement_evidence.v1",
        ),
        "announcement_availability": (
            "../../data/pead_announcement_availability.py",
            "pead_announcement_availability.v1",
        ),
        "event_source_reconciliation": (
            "../../analysis/pead_source_reconciliation_v2.py",
            "pead_source_reconciliation.v2",
        ),
        "market_accounting_receipt": (
            "../../data/pead_market_accounting_evidence.py",
            "pead_market_accounting_evidence.v1",
        ),
        "final_signal_input_reconciliation": (
            "../../analysis/pead_signal_input_reconciliation.py",
            "pead_signal_input_reconciliation.v1",
        ),
        "cross_partition_signal_input_index": (
            "../../analysis/pead_signal_input_index.py",
            "pead_signal_input_index.v1",
        ),
    }
    assert set(contracts) == set(expected)
    for name, (relative_module, schema) in expected.items():
        contract = contracts[name]
        assert contract["implemented"] is True
        assert contract["module"] == relative_module
        assert (PACKAGE / relative_module).resolve().is_file()
        assert schema in contract["schemas"]


def test_trust_boundaries_are_external_and_closed():
    trust = _strict_json("source_architecture.json")["trust_boundaries"]
    assert trust["external_allowlists_are_verifier_inputs"] is True
    assert trust["embedded_self_asserted_trust_qualifies"] is False
    assert trust["local_operator_admission_is_independent_review"] is False
    assert trust["consensus_replay_allowlists"] == [
        "trusted_event_universe_sha256s",
        "trusted_identity_snapshot_sha256s",
        "trusted_source_manifest_sha256s",
        "trusted_metric_profile_sha256s",
        "trusted_raw_artifact_sha256s",
    ]
    assert len(trust["market_accounting_allowlists"]) == 9
    assert len(trust["final_signal_allowlists"]) == 5
    assert len(trust["cross_partition_allowlists"]) == 4


def test_annual_partition_plan_is_exact_and_names_final_index():
    plan = _strict_json("source_architecture.json")["annual_event_partition_plan"]
    partitions = plan["partitions"]
    assert plan["child_schema_version"] == "pead_event_universe.v2"
    assert plan["target_window"] == {"start": "2015-01-01", "end": "2024-09-30"}
    assert [row["partition_id"] for row in partitions] == [
        str(year) for year in range(2015, 2025)
    ]
    assert partitions[0] == {
        "partition_id": "2015",
        "start": "2015-01-01",
        "end": "2015-12-31",
    }
    assert partitions[-1] == {
        "partition_id": "2024",
        "start": "2024-01-01",
        "end": "2024-09-30",
    }
    assert plan["final_cross_partition_receipt"] == "pead_signal_input_index.v1"
    assert "ARQ is the sole quarterly source scope" in plan["child_rules"][0]


def test_status_records_real_local_source_evidence_without_external_admission():
    status = _strict_json("qualification_status.json")
    bindings = status["artifact_bindings"]
    for name in (
        "sharadar_sf1_acquisition",
        "sharadar_sep_acquisition",
        "sharadar_tickers_acquisition",
        "sharadar_source_snapshot",
        "security_identity_snapshot",
        "annual_event_universe_partitions",
        "sharadar_event_universe_replay",
        "event_universe_index",
        "sharadar_semantics_source_receipt",
        "sep_semantic_profile",
        "nyse_session_close_calendar",
        "nyse_session_close_source_receipt",
    ):
        binding = bindings[name]
        assert binding["status"] in {
            "locally_sealed_not_independently_admitted",
            "locally_replayed_not_independently_admitted",
            "repository_preserved_not_independently_admitted",
        }
        assert binding.get("artifact_hash") or binding.get("artifact_hashes")

    for name in (
        "consensus_raw_artifacts",
        "consensus_replay_partitions",
        "announcement_evidence_partitions",
        "announcement_availability_partitions",
        "event_source_reconciliation_partitions",
        "market_accounting_receipt_partitions",
        "final_signal_input_reconciliation_partitions",
        "final_signal_input_index",
    ):
        assert bindings[name]["artifact_hash"] is None

    trust = status["external_trust_registry_bindings"]
    assert trust.pop("status") == "no_independent_registry_entries_bound"
    assert trust and all(value is None for value in trust.values())


def test_status_has_no_pending_engineering_contract_and_keeps_external_blockers():
    status = _strict_json("qualification_status.json")
    engineering = status["engineering_status"]
    resolved_codes = {row["code"] for row in engineering["resolved"]}
    assert len(resolved_codes) == 12
    assert {
        "E07_AUTHORITATIVE_SHARADAR_EVENT_REPLAY_IMPLEMENTED",
        "E08_OFFICIAL_SHARADAR_SEMANTICS_RECEIPT_IMPLEMENTED",
        "E09_MARKET_ACCOUNTING_RECEIPT_IMPLEMENTED",
        "E10_FINAL_SIGNAL_INPUT_RECONCILIATION_IMPLEMENTED",
        "E11_CROSS_PARTITION_SIGNAL_INPUT_INDEX_IMPLEMENTED",
    } <= resolved_codes
    assert engineering["pending"] == []
    assert _strict_json("source_architecture.json")["remaining_engineering_contracts"] == {}
    blocker_codes = {row["code"] for row in status["external_data_and_evidence_blockers"]}
    assert blocker_codes == {
        "Q02_LICENSED_POINT_IN_TIME_CONSENSUS_INCOMPLETE",
        "Q03_INDEPENDENT_SEC_ACTUAL_CORPUS_ABSENT",
        "Q04_HISTORICAL_DISTRIBUTION_EVIDENCE_ABSENT",
        "Q05_PROSPECTIVE_CHECKPOINT_INFRASTRUCTURE_NOT_STARTED",
        "Q06_INDEPENDENT_TRUST_REGISTRY_ADMISSION_ABSENT",
    }


def test_no_research_edge_or_deployment_permission_is_open():
    architecture = _strict_json("source_architecture.json")
    status = _strict_json("qualification_status.json")

    claim_boundary = architecture["claim_boundary"]
    assert claim_boundary["architecture_documentation_allowed"] is True
    assert claim_boundary["local_source_artifact_recording_allowed"] is True
    assert all(
        value is False
        for key, value in claim_boundary.items()
        if key
        not in {"architecture_documentation_allowed", "local_source_artifact_recording_allowed"}
    )
    qualification = status["qualification"]
    assert qualification["local_sharadar_source_structurally_qualified"] is True
    assert qualification["local_identity_structurally_qualified"] is True
    assert qualification["local_event_universe_structurally_replayed"] is True
    assert all(
        value is False
        for key, value in qualification.items()
        if not key.startswith("local_")
    )

    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "not evidence of an edge" in readme
    assert "exact global first-public" in readme
    assert "pead_event_universe.v2" in readme
    assert "pead_signal_input_index.v1" in readme
    assert "2018 development slice" in readme
