from __future__ import annotations

import copy
import hashlib

import pytest

import analysis.pead_source_reconciliation_v2 as subject
from analysis.pead_known_by_policy import KNOWN_BY_POLICY_SHA256
from analysis.pead_source_reconciliation_v2 import (
    PeadSourceReconciliationV2Error,
    build_pead_source_reconciliation_v2,
    verify_pead_source_reconciliation_v2,
)
from data.pead_event_universe import canonical_event_id, content_hash


KEY = {
    "cik": "0000320193",
    "fiscal_period_end": "2020-03-28",
    "fiscal_period_type": "Q",
}
EVENT_ID = canonical_event_id(KEY)
RAW = b'exact licensed consensus artifact bytes\n{"records":[]}'
RAW_SHA = hashlib.sha256(RAW).hexdigest()
HASHES = {character: character * 64 for character in "abcdef9876543210"}


def _metric() -> dict:
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


def _actual(*, accounting_basis="non_gaap") -> dict:
    return {
        "announced_value": "1.25",
        "canonical_value": "1.25",
        "normalization_factor": "1",
        "metric": "earnings_per_share",
        "source_metric_label": "Adjusted diluted earnings per share",
        "metric_definition_sha256": HASHES["4"],
        "accounting_basis": accounting_basis,
        "per_share_basis": "diluted",
        "scope": "total_company",
        "currency": "USD",
        "unit": "currency_per_share",
        "announced_share_basis": "issuer_as_reported_at_publication",
        "canonical_share_basis": "split_restated",
        "fiscal_period_end": KEY["fiscal_period_end"],
        "fiscal_period_type": "Q",
        "normalization_evidence_sha256": HASHES["5"],
    }


def _vintage(*, as_of="2020-04-30", analyst_count=4, metric=None) -> dict:
    return {
        "provider_as_of_date": as_of,
        "trusted_available_at_utc": "2020-04-30T19:00:00Z",
        "availability_precision": "second",
        "consensus_value": "1",
        "analyst_count": analyst_count,
        "raw_record_sha256": HASHES["6"],
        "acquisition_receipt_sha256": HASHES["7"],
        "metric": metric or _metric(),
    }


def _inputs(
    *,
    vintage: dict | None = None,
    actual: dict | None = None,
    consensus_disposition="available",
    availability_disposition="available",
    consensus_evidence_class="historical_reconstruction",
    availability_evidence_class="historical_reconstruction",
    replay_qualified=True,
) -> dict:
    universe = {
        "artifact_hash": HASHES["a"],
        "payload": {
            "candidate_id": "pead-source-v2-test",
            "frozen_at_utc": "2020-01-01T00:00:00Z",
            "bindings": {
                "market_snapshot_sha256": HASHES["b"],
                "identity_snapshot_sha256": HASHES["c"],
                "candidate_specification_sha256": HASHES["d"],
                "construction_code_sha256": HASHES["e"],
            },
            "expected_event_ids": [EVENT_ID],
            "expected_events": [{"event_id": EVENT_ID, "event_key": KEY}],
            "qualification_allowed": True,
        },
    }
    consensus_record = {
        "event_id": EVENT_ID,
        "disposition": consensus_disposition,
        "missing_reason": (None if consensus_disposition == "available" else "provider_row_absent"),
        "vintages": [vintage or _vintage()] if consensus_disposition == "available" else [],
    }
    consensus = {
        "artifact_hash": HASHES["f"],
        "payload": {
            "candidate_id": "pead-source-v2-test",
            "evidence_class": consensus_evidence_class,
            "event_universe": universe,
            "source": {"captured_at_utc": "2026-07-14T13:00:00Z"},
            "acquisition_receipts": [
                {
                    "receipt_sha256": HASHES["7"],
                    "source_captured_at_utc": "2026-07-14T12:59:00Z",
                }
            ],
            "event_records": [consensus_record],
        },
    }
    replay = {
        "artifact_hash": HASHES["8"],
        "payload": {
            "qualification_allowed": replay_qualified,
            "bindings": {
                "event_universe_sha256": HASHES["a"],
                "identity_snapshot_sha256": HASHES["c"],
                "source_manifest_sha256": HASHES["9"],
                "metric_profile_sha256": HASHES["1"],
                "raw_artifact_bytes_sha256": RAW_SHA,
                "consensus_evidence_sha256": HASHES["f"],
            },
            "consensus_evidence": consensus,
        },
    }
    canonical_actual = actual or _actual()
    announcement_outcome = {
        "event_id": EVENT_ID,
        "event_key": KEY,
        "disposition": "available",
        "missing_reason": None,
        "available_record": {
            "source_kind": "sec_edgar_item_2_02_exhibit",
            "accession_number": "0000320193-20-000001",
            "exhibit_document": {"raw_document": {"sha256": HASHES["2"]}},
            "edgar_acceptance_at_utc": "2020-05-01T19:59:00Z",
            "canonical_actual": canonical_actual,
        },
    }
    announcement = {
        "artifact_hash": HASHES["3"],
        "payload": {
            "candidate_id": "pead-source-v2-test",
            "created_at_utc": "2026-07-14T14:00:00Z",
            "outcomes": [announcement_outcome],
        },
    }
    eligibility = {
        "claim_semantics": "conservative_known_public_by_upper_bound",
        "first_public_proven": False,
        "eligible_for_declared_evidence_class": True,
        "historical_reconstruction_allowed": (
            availability_evidence_class == "historical_reconstruction"
        ),
        "prospective_observation_allowed": (availability_evidence_class == "prospective"),
        "consensus_cutoff_rule": "strict_prior_eastern_calendar_date",
        "market_cutoff_rule": "strict_prior_nyse_session",
        "same_day_consensus_allowed": False,
        "same_day_market_close_allowed": False,
    }
    if availability_disposition == "available":
        claim = {
            "claim_kind": "known_public_by",
            "known_public_by_at_utc": "2020-05-01T20:00:00Z",
            "adapter_id": "licensed_release_distribution.v1",
            "evidence": {"raw_provider_record_sha256": HASHES["0"]},
            "eligibility": eligibility,
        }
        missing_reason = None
    else:
        claim = None
        missing_reason = "independent_distribution_archive_absent"
    availability = {
        "artifact_hash": "0" * 64,
        "payload": {
            "candidate_id": "pead-source-v2-test",
            "evidence_class": availability_evidence_class,
            "created_at_utc": "2026-07-14T15:00:00Z",
            "outcomes": [
                {
                    "event_id": EVENT_ID,
                    "event_key": KEY,
                    "disposition": availability_disposition,
                    "missing_reason": missing_reason,
                    "claim": claim,
                }
            ],
            "coverage": {
                "blockers": (
                    []
                    if availability_disposition == "available"
                    else ["expected_events_missing_availability"]
                )
            },
        },
    }
    return {
        "universe": universe,
        "consensus": consensus,
        "replay": replay,
        "announcement": announcement,
        "availability": availability,
    }


def _install_authoritative_stubs(monkeypatch, inputs: dict) -> dict:
    calls: dict[str, dict] = {}

    def verify_replay(document, **kwargs):
        assert document is inputs["replay"]
        assert kwargs["raw_artifact"] == RAW
        calls["consensus"] = kwargs
        return inputs["replay"]

    def validate_announcement(document, *, expected_event_manifest):
        assert document is inputs["announcement"]
        assert expected_event_manifest is inputs["universe"]
        calls["announcement"] = {"universe": expected_event_manifest}
        return inputs["announcement"]

    def validate_availability(
        document,
        *,
        expected_event_manifest,
        announcement_evidence,
        trusted_provider_manifest_sha256s,
        trusted_provider_record_sha256s,
        trusted_checkpoint_sha256s,
    ):
        assert document is inputs["availability"]
        assert expected_event_manifest is inputs["universe"]
        assert announcement_evidence is inputs["announcement"]
        calls["availability"] = {
            "providers": trusted_provider_manifest_sha256s,
            "provider_records": trusted_provider_record_sha256s,
            "checkpoints": trusted_checkpoint_sha256s,
        }
        return inputs["availability"]

    monkeypatch.setattr(subject, "verify_pead_consensus_replay", verify_replay)
    monkeypatch.setattr(subject, "validate_pead_announcement_evidence", validate_announcement)
    monkeypatch.setattr(
        subject,
        "validate_pead_announcement_availability",
        validate_availability,
    )
    return calls


def _kwargs() -> dict:
    return {
        "consensus_raw_artifact": RAW,
        "reconciled_at_utc": "2026-07-14T16:00:00Z",
        "trusted_consensus_source_manifest_sha256s": [HASHES["9"]],
        "trusted_consensus_metric_profile_sha256s": [HASHES["1"]],
        "trusted_consensus_identity_snapshot_sha256s": [HASHES["c"]],
        "trusted_consensus_event_universe_sha256s": [HASHES["a"]],
        "trusted_consensus_raw_artifact_sha256s": [RAW_SHA],
        "trusted_announcement_evidence_sha256s": [HASHES["3"]],
        "trusted_announcement_provider_manifest_sha256s": [HASHES["2"]],
        "trusted_announcement_provider_record_sha256s": [HASHES["0"]],
        "trusted_announcement_checkpoint_sha256s": [HASHES["5"]],
    }


def _build(inputs: dict, **overrides) -> dict:
    kwargs = _kwargs()
    kwargs.update(overrides)
    return build_pead_source_reconciliation_v2(
        inputs["replay"],
        inputs["announcement"],
        inputs["availability"],
        **kwargs,
    )


def test_authoritative_sources_reconcile_one_exact_event_but_remain_nonresearch(
    monkeypatch,
):
    inputs = _inputs()
    calls = _install_authoritative_stubs(monkeypatch, inputs)

    receipt = _build(inputs)

    result = receipt["payload"]["event_results"][0]
    assert result["disposition"] == "event_source_reconciled"
    assert result["blockers"] == []
    assert result["event_source_input"]["raw_surprise"] == "0.25"
    assert result["event_source_input"]["surprise_direction"] == "positive"
    assert result["event_source_input"]["known_public_by_at_utc"] == ("2020-05-01T20:00:00Z")
    assert receipt["payload"]["coverage"] == {
        "expected_event_count": 1,
        "event_source_reconciled_count": 1,
        "excluded_event_count": 0,
        "exhaustive_event_accounting": True,
        "systemic_blockers": [],
        "event_blocker_counts": {},
    }
    assert receipt["payload"]["qualification"] == {
        "has_event_source_reconciled_inputs": True,
        "all_expected_events_source_reconciled": True,
        "event_source_reconciliation_allowed": True,
        "research_consumable": False,
        "market_accounting_join_required": True,
        "prospective_consensus_freeze_check_pending_market_evidence": False,
        "historical_replication_allowed": False,
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    assert receipt["payload"]["bindings"]["known_by_policy_sha256"] == (KNOWN_BY_POLICY_SHA256)
    assert calls["consensus"]["trusted_event_universe_sha256s"] == [HASHES["a"]]
    assert calls["consensus"]["trusted_identity_snapshot_sha256s"] == [HASHES["c"]]
    assert calls["consensus"]["trusted_raw_artifact_sha256s"] == [RAW_SHA]
    assert calls["availability"]["providers"] == [HASHES["2"]]
    assert calls["availability"]["provider_records"] == [HASHES["0"]]
    assert calls["availability"]["checkpoints"] == [HASHES["5"]]


def test_known_by_policy_rejects_same_eastern_date_consensus_even_with_exact_time(
    monkeypatch,
):
    inputs = _inputs(vintage=_vintage(as_of="2020-05-01"))
    _install_authoritative_stubs(monkeypatch, inputs)

    result = _build(inputs)["payload"]["event_results"][0]

    assert result["disposition"] == "excluded"
    assert result["blockers"] == ["consensus_no_eligible_prior_day_vintage"]
    assert result["event_source_input"] is None


def test_latest_undercovered_consensus_excludes_without_fallback(monkeypatch):
    inputs = _inputs(vintage=_vintage(analyst_count=1))
    _install_authoritative_stubs(monkeypatch, inputs)

    result = _build(inputs)["payload"]["event_results"][0]

    assert result["disposition"] == "excluded"
    assert result["blockers"] == ["consensus_analyst_count_below_minimum"]
    assert result["source_values"]["consensus"]["selected_vintage"] is not None


def test_metric_currency_and_share_semantics_are_exact_not_coerced(monkeypatch):
    inputs = _inputs(actual=_actual(accounting_basis="gaap"))
    _install_authoritative_stubs(monkeypatch, inputs)

    result = _build(inputs)["payload"]["event_results"][0]

    assert result["disposition"] == "excluded"
    assert result["blockers"] == ["accounting_basis_mismatch"]
    comparison = next(item for item in result["comparisons"] if item["field"] == "accounting_basis")
    assert comparison["consensus_value"] == "non_gaap"
    assert comparison["announcement_value"] == "gaap"
    assert comparison["decision"] == "mismatch"


def test_missing_availability_is_accounted_without_becoming_systemic(monkeypatch):
    inputs = _inputs(availability_disposition="missing")
    _install_authoritative_stubs(monkeypatch, inputs)

    receipt = _build(inputs)
    result = receipt["payload"]["event_results"][0]

    assert result["disposition"] == "excluded"
    assert result["blockers"] == ["announcement_availability_missing"]
    assert receipt["payload"]["coverage"]["exhaustive_event_accounting"] is True
    assert receipt["payload"]["coverage"]["systemic_blockers"] == []


def test_consensus_replay_must_be_qualified_after_authoritative_raw_rebuild(monkeypatch):
    inputs = _inputs(replay_qualified=False)
    _install_authoritative_stubs(monkeypatch, inputs)

    with pytest.raises(PeadSourceReconciliationV2Error, match="not source-qualified"):
        _build(inputs)


def test_consensus_and_availability_evidence_classes_must_match(monkeypatch):
    inputs = _inputs(availability_evidence_class="prospective")
    _install_authoritative_stubs(monkeypatch, inputs)

    with pytest.raises(PeadSourceReconciliationV2Error, match="classes differ"):
        _build(inputs)


def test_licensed_archive_cannot_cross_the_prospective_boundary(monkeypatch):
    inputs = _inputs(
        consensus_evidence_class="prospective_signal",
        availability_evidence_class="prospective",
    )
    _install_authoritative_stubs(monkeypatch, inputs)

    with pytest.raises(PeadSourceReconciliationV2Error, match="adapter is ineligible"):
        _build(inputs)


def test_announcement_actual_artifact_requires_external_trust_anchor(monkeypatch):
    inputs = _inputs()
    _install_authoritative_stubs(monkeypatch, inputs)

    with pytest.raises(PeadSourceReconciliationV2Error, match="announcement evidence"):
        _build(inputs, trusted_announcement_evidence_sha256s=[])


def test_consensus_raw_artifact_requires_external_trust_anchor(monkeypatch):
    inputs = _inputs()
    _install_authoritative_stubs(monkeypatch, inputs)

    with pytest.raises(PeadSourceReconciliationV2Error, match="raw artifact"):
        _build(inputs, trusted_consensus_raw_artifact_sha256s=[])


def test_verifier_rebuild_rejects_self_consistently_rehashed_edit(monkeypatch):
    inputs = _inputs()
    _install_authoritative_stubs(monkeypatch, inputs)
    receipt = _build(inputs)
    tampered = copy.deepcopy(receipt)
    tampered["payload"]["event_results"][0]["event_source_input"]["raw_surprise"] = "99"
    tampered["artifact_hash"] = content_hash(tampered["payload"])

    kwargs = _kwargs()
    kwargs.pop("reconciled_at_utc")
    with pytest.raises(PeadSourceReconciliationV2Error, match="does not replay"):
        verify_pead_source_reconciliation_v2(
            tampered,
            consensus_replay=inputs["replay"],
            announcement_evidence=inputs["announcement"],
            announcement_availability=inputs["availability"],
            **kwargs,
        )


def test_every_external_trust_set_is_content_bound(monkeypatch):
    inputs = _inputs()
    _install_authoritative_stubs(monkeypatch, inputs)
    receipt = _build(inputs)
    bindings = receipt["payload"]["bindings"]

    fields = [
        "trusted_consensus_source_manifest_set_sha256",
        "trusted_consensus_metric_profile_set_sha256",
        "trusted_consensus_identity_snapshot_set_sha256",
        "trusted_consensus_event_universe_set_sha256",
        "trusted_consensus_raw_artifact_set_sha256",
        "trusted_announcement_evidence_set_sha256",
        "trusted_announcement_provider_manifest_set_sha256",
        "trusted_announcement_provider_record_set_sha256",
        "trusted_announcement_checkpoint_set_sha256",
    ]
    assert all(len(bindings[field]) == 64 for field in fields)
    assert len({bindings[field] for field in fields}) == len(fields)
