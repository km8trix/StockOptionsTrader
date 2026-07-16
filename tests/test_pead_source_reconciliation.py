from __future__ import annotations

import copy
import json

import pytest

from analysis.pead_source_reconciliation import (
    PeadSourceReconciliationError,
    _event_result,
    build_pead_source_reconciliation,
    load_pead_source_reconciliation,
    select_consensus_vintage,
    verify_pead_source_reconciliation,
)
from data.pead_announcement_evidence import (
    build_missing_outcome,
    build_pead_announcement_evidence,
)
from data.pead_consensus_evidence import (
    build_pead_consensus_evidence,
    canonical_json,
    content_hash,
)
from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
)
from scripts.pead_source_reconciliation import main


KEY = {
    "cik": "0000320193",
    "fiscal_period_end": "2020-03-28",
    "fiscal_period_type": "Q",
}
EVENT_ID = canonical_event_id(KEY)
HASHES = {character: character * 64 for character in "abcdef987654321"}


def _universe():
    receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256=HASHES["f"],
        canonical_query_sha256=HASHES["e"],
        source_record_ids=[HASHES["1"]],
    )
    return build_pead_event_universe(
        candidate_id="pead-source-reconciliation-test-v1",
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start="2020-01-01",
        event_end="2020-12-31",
        bindings={
            "market_snapshot_sha256": HASHES["a"],
            "identity_snapshot_sha256": HASHES["b"],
            "candidate_specification_sha256": HASHES["c"],
            "construction_code_sha256": HASHES["d"],
            "canonical_query_sha256": HASHES["e"],
        },
        census_receipt=receipt,
        census_dispositions=[
            {
                "source_record_id": HASHES["1"],
                "disposition": "expected_event",
                "event_id": EVENT_ID,
                "event_key": KEY,
                "reason": None,
            }
        ],
    )


def _metric():
    return {
        "metric_id": "earnings_per_share",
        "accounting_basis": "non_gaap",
        "per_share_basis": "diluted",
        "scope": "total_company",
        "canonical_share_basis": "split_restated",
        "currency_code": "USD",
        "unit": "currency_per_share",
        "metric_definition_sha256": HASHES["4"],
    }


def _vintage(
    *,
    as_of="2020-04-29",
    available=None,
    analyst_count=4,
    raw_hash=None,
    metric=None,
):
    return {
        "provider_as_of_date": as_of,
        "trusted_available_at_utc": available,
        "availability_precision": (
            "microsecond" if available and "." in available
            else "second" if available
            else "date"
        ),
        "consensus_value": "1",
        "analyst_count": analyst_count,
        "raw_record_sha256": raw_hash or HASHES["2"],
        "acquisition_receipt_sha256": HASHES["3"],
        "metric": metric or _metric(),
    }


def _consensus(universe, *, evidence_class="historical_reconstruction"):
    receipt_body = {
        "source_captured_at_utc": "2026-07-14T14:00:00Z",
        "query_scope": {
            "scope_kind": "full_export",
            "canonical_query_sha256": HASHES["7"],
            "expected_event_ids": [EVENT_ID],
        },
        "pagination": {
            "mode": "bulk_file",
            "terminal_page_observed": True,
            "page_count": 1,
            "pages": [
                {
                    "sequence": 1,
                    "request_sha256": HASHES["6"],
                    "raw_response_sha256": HASHES["5"],
                    "raw_response_bytes": 100,
                    "continuation_token_sha256": None,
                }
            ],
        },
    }
    receipt = {"receipt_sha256": content_hash(receipt_body), **receipt_body}
    vintage = _vintage()
    vintage["acquisition_receipt_sha256"] = receipt["receipt_sha256"]
    return build_pead_consensus_evidence(
        candidate_id=universe["payload"]["candidate_id"],
        evidence_class=evidence_class,
        event_universe=universe,
        source={
            "provider_id": "licensed-provider",
            "dataset_id": "point-in-time-consensus",
            "source_manifest_sha256": HASHES["8"],
            "captured_at_utc": "2026-07-14T15:00:00Z",
            "provider_snapshot_at_utc": "2026-07-14T13:00:00Z",
        },
        acquisition_receipts=[receipt],
        event_records=[
            {
                "event_id": EVENT_ID,
                "disposition": "available",
                "missing_reason": None,
                "vintages": [vintage],
            }
        ],
    )


def _missing_announcement(universe):
    return build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T16:00:00Z",
        outcomes=[build_missing_outcome(KEY, reason="independent_source_absent")],
    )


def _synthetic_announcement_outcome(*, accounting_basis="non_gaap"):
    return {
        "event_id": EVENT_ID,
        "event_key": KEY,
        "disposition": "available",
        "missing_reason": None,
        "available_record": {
            "source_kind": "independent-test-source",
            "accession_number": "0000320193-20-000001",
            "exhibit_document": {"raw_document": {"sha256": HASHES["9"]}},
            "edgar_acceptance_at_utc": "2020-05-01T19:59:30Z",
            "first_public_at_utc": "2020-05-01T20:00:00Z",
            "first_public_basis": "authoritative-test-proof",
            "observed_public_by_at_utc": "2020-05-01T20:00:01Z",
            "canonical_actual": {
                "canonical_value": "1.25",
                "metric": "earnings_per_share",
                "accounting_basis": accounting_basis,
                "per_share_basis": "diluted",
                "scope": "total_company",
                "canonical_share_basis": "split_restated",
                "currency": "USD",
                "unit": "currency_per_share",
                "metric_definition_sha256": HASHES["4"],
                "normalization_evidence_sha256": HASHES["a"],
            },
        },
    }


def _synthetic_consensus_record(vintages):
    return {
        "event_id": EVENT_ID,
        "disposition": "available",
        "missing_reason": None,
        "vintages": vintages,
    }


def _bindings():
    return {
        "event_universe_sha256": HASHES["a"],
        "consensus_evidence_sha256": HASHES["b"],
        "announcement_evidence_sha256": HASHES["c"],
    }


def _event_context(*, frozen="2020-01-01T00:00:00Z", captured="2020-04-30T12:00:00Z"):
    return {
        "universe_frozen_at_utc": frozen,
        "receipt_captured_at_by_hash": {HASHES["3"]: captured},
    }


def test_current_announcement_contract_reconciles_exhaustively_but_fails_closed():
    universe = _universe()
    consensus = _consensus(universe)
    announcement = _missing_announcement(universe)

    receipt = build_pead_source_reconciliation(
        consensus, announcement, reconciled_at_utc="2026-07-14T17:00:00Z"
    )

    assert receipt["payload"]["coverage"] == {
        "expected_event_count": 1,
        "reconciled_event_count": 0,
        "excluded_event_count": 1,
        "reconciliation_complete": True,
        "systemic_blockers": [],
        "event_blocker_counts": {
            "announcement_event_missing": 1,
        },
    }
    assert receipt["payload"]["qualification"] == {
        "has_reconciled_event_inputs": False,
        "all_expected_events_reconciled": False,
        "source_qualified_event_inputs_allowed": False,
        "raw_consensus_normalization_replay_required": True,
        "external_binding_replay_required": True,
        "market_accounting_join_required": True,
        "historical_replication_allowed": False,
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    assert verify_pead_source_reconciliation(
        receipt,
        consensus_evidence=consensus,
        announcement_evidence=announcement,
    ) == receipt


def test_reconciliation_rebuild_rejects_a_self_consistently_rehashed_edit():
    universe = _universe()
    consensus = _consensus(universe)
    announcement = _missing_announcement(universe)
    receipt = build_pead_source_reconciliation(
        consensus, announcement, reconciled_at_utc="2026-07-14T17:00:00Z"
    )
    tampered = copy.deepcopy(receipt)
    tampered["payload"]["event_results"][0]["blockers"] = []
    tampered["artifact_hash"] = content_hash(tampered["payload"])

    with pytest.raises(PeadSourceReconciliationError, match="does not replay"):
        verify_pead_source_reconciliation(
            tampered,
            consensus_evidence=consensus,
            announcement_evidence=announcement,
        )


def test_vintage_selection_is_strict_at_both_supported_precisions():
    same_day = _vintage(as_of="2020-05-01")
    selected, blockers, _, _ = select_consensus_vintage(
        [same_day], first_public_at_utc="2020-05-01T20:00:00Z"
    )
    assert selected is None
    assert blockers == ["consensus_no_eligible_preannouncement_vintage"]

    prior_day = _vintage(as_of="2020-04-30")
    selected, blockers, _, precision_rule = select_consensus_vintage(
        [prior_day], first_public_at_utc="2020-05-01T20:00:00Z"
    )
    assert selected == prior_day
    assert blockers == []
    assert precision_rule == "strict_prior_eastern_calendar_date"

    equal_instant = _vintage(
        as_of="2020-05-01", available="2020-05-01T20:00:00Z"
    )
    selected, blockers, _, _ = select_consensus_vintage(
        [equal_instant], first_public_at_utc="2020-05-01T20:00:00Z"
    )
    assert selected is None
    assert blockers == ["consensus_no_eligible_preannouncement_vintage"]


def test_latest_tie_and_low_analyst_count_never_fall_back():
    first = _vintage(
        as_of="2020-04-30",
        available="2020-04-30T19:00:00Z",
        raw_hash=HASHES["1"],
    )
    tied = _vintage(
        as_of="2020-04-30",
        available="2020-04-30T19:00:00Z",
        raw_hash=HASHES["2"],
    )
    selected, blockers, hashes, _ = select_consensus_vintage(
        [first, tied], first_public_at_utc="2020-05-01T20:00:00Z"
    )
    assert selected is None
    assert blockers == ["consensus_latest_vintage_ambiguous"]
    assert hashes == [HASHES["1"], HASHES["2"]]

    earlier = _vintage(
        as_of="2020-04-29", available="2020-04-29T19:00:00Z", analyst_count=5
    )
    latest = _vintage(
        as_of="2020-04-30",
        available="2020-04-30T19:00:00Z",
        analyst_count=1,
    )
    selected, blockers, _, _ = select_consensus_vintage(
        [earlier, latest], first_public_at_utc="2020-05-01T20:00:00Z"
    )
    assert selected == latest
    assert blockers == ["consensus_analyst_count_below_minimum"]


def test_exact_vintage_order_uses_instants_not_mixed_precision_strings():
    earlier = _vintage(
        as_of="2020-04-30",
        available="2020-04-30T19:00:00Z",
        raw_hash=HASHES["1"],
    )
    later = _vintage(
        as_of="2020-04-30",
        available="2020-04-30T19:00:00.500000Z",
        raw_hash=HASHES["2"],
    )
    selected, blockers, _, precision_rule = select_consensus_vintage(
        [earlier, later], first_public_at_utc="2020-05-01T20:00:00Z"
    )
    assert selected == later
    assert blockers == []
    assert precision_rule == "strict_prior_utc_instant"


def test_metric_history_and_cross_source_mismatches_exclude_without_coercion():
    changed_metric = _metric()
    changed_metric["accounting_basis"] = "gaap"
    selected, blockers, _, _ = select_consensus_vintage(
        [
            _vintage(as_of="2020-04-29", raw_hash=HASHES["1"]),
            _vintage(
                as_of="2020-04-30",
                raw_hash=HASHES["2"],
                metric=changed_metric,
            ),
        ],
        first_public_at_utc="2020-05-01T20:00:00Z",
    )
    assert selected is None
    assert blockers == ["consensus_metric_history_inconsistent"]

    result = _event_result(
        event={"event_id": EVENT_ID, "event_key": KEY},
        consensus_record=_synthetic_consensus_record([_vintage()]),
        announcement_outcome=_synthetic_announcement_outcome(
            accounting_basis="gaap"
        ),
        systemic_blockers=[],
        bindings=_bindings(),
        prospective=False,
        **_event_context(),
    )
    assert result["disposition"] == "excluded"
    assert result["blockers"] == ["accounting_basis_mismatch"]
    comparison = next(
        row for row in result["comparisons"] if row["field"] == "accounting_basis"
    )
    assert comparison == {
        "field": "accounting_basis",
        "consensus_value": "non_gaap",
        "announcement_value": "gaap",
        "decision": "mismatch",
    }


def test_pure_event_reconciliation_computes_decimal_surprise_and_preserves_sources():
    result = _event_result(
        event={"event_id": EVENT_ID, "event_key": KEY},
        consensus_record=_synthetic_consensus_record([_vintage()]),
        announcement_outcome=_synthetic_announcement_outcome(),
        systemic_blockers=[],
        bindings=_bindings(),
        prospective=False,
        **_event_context(),
    )

    assert result["disposition"] == "reconciled"
    assert result["blockers"] == []
    assert result["reconciled_event_input"]["raw_surprise"] == "0.25"
    assert result["reconciled_event_input"]["surprise_direction"] == "positive"
    assert result["reconciled_event_input"][
        "consensus_receipt_captured_at_utc"
    ] == "2020-04-30T12:00:00Z"
    assert result["source_values"]["announcement"]["canonical_actual"][
        "canonical_value"
    ] == "1.25"
    assert result["source_values"]["consensus"]["selected_vintage"][
        "consensus_value"
    ] == "1"


def test_prospective_event_requires_intraday_consensus_availability():
    result = _event_result(
        event={"event_id": EVENT_ID, "event_key": KEY},
        consensus_record=_synthetic_consensus_record([_vintage()]),
        announcement_outcome=_synthetic_announcement_outcome(),
        systemic_blockers=[],
        bindings=_bindings(),
        prospective=True,
        **_event_context(),
    )
    assert result["disposition"] == "excluded"
    assert result["blockers"] == [
        "prospective_consensus_intraday_timestamp_missing"
    ]


def test_prospective_event_rejects_post_event_freeze_and_acquisition():
    exact = _vintage(
        as_of="2020-04-30", available="2020-04-30T19:00:00Z"
    )
    result = _event_result(
        event={"event_id": EVENT_ID, "event_key": KEY},
        consensus_record=_synthetic_consensus_record([exact]),
        announcement_outcome=_synthetic_announcement_outcome(),
        systemic_blockers=[],
        bindings=_bindings(),
        prospective=True,
        **_event_context(
            frozen="2020-05-02T00:00:00Z",
            captured="2020-05-01T20:00:01Z",
        ),
    )
    assert result["disposition"] == "excluded"
    assert result["blockers"] == [
        "prospective_consensus_acquired_after_announcement",
        "prospective_universe_frozen_after_announcement",
    ]


def test_strict_loader_and_cli_write_a_create_only_blocked_receipt(tmp_path, capsys):
    universe = _universe()
    consensus = _consensus(universe)
    announcement = _missing_announcement(universe)
    consensus_path = tmp_path / "consensus.json"
    announcement_path = tmp_path / "announcement.json"
    output = tmp_path / "receipt.json"
    consensus_path.write_text(canonical_json(consensus), encoding="utf-8")
    announcement_path.write_text(canonical_json(announcement), encoding="utf-8")

    args = [
        "--consensus",
        str(consensus_path),
        "--announcement",
        str(announcement_path),
        "--reconciled-at-utc",
        "2026-07-14T17:00:00Z",
        "--output",
        str(output),
    ]
    assert main(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["has_reconciled_event_inputs"] is False
    assert summary["source_qualified_event_inputs_allowed"] is False
    loaded = load_pead_source_reconciliation(
        output,
        consensus_evidence=consensus,
        announcement_evidence=announcement,
    )
    assert loaded["artifact_hash"] == summary["artifact_hash"]

    changed = args.copy()
    changed[changed.index("2026-07-14T17:00:00Z")] = "2026-07-14T17:00:01Z"
    assert main(changed) == 2
    assert "refusing to overwrite" in capsys.readouterr().err

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"artifact_hash":"' + HASHES["a"] + '","artifact_hash":"'
        + HASHES["b"] + '","payload":{}}',
        encoding="utf-8",
    )
    with pytest.raises(PeadSourceReconciliationError, match="duplicate key"):
        load_pead_source_reconciliation(
            duplicate,
            consensus_evidence=consensus,
            announcement_evidence=announcement,
        )

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(PeadSourceReconciliationError, match="not canonical"):
        load_pead_source_reconciliation(
            pretty,
            consensus_evidence=consensus,
            announcement_evidence=announcement,
        )
