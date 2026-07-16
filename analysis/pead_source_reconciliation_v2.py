"""Trust-rooted PEAD event-source reconciliation using conservative availability.

This additive v2 boundary preserves :mod:`analysis.pead_source_reconciliation`
unchanged.  It consumes one verified consensus-replay partition, independent
announcement actuals, and conservative ``known_public_by`` availability.  A
row may become ``event_source_reconciled`` only when those lanes agree exactly.

Event-source reconciliation is deliberately not a research-input boundary.  A
later market/accounting receipt must still prove the prior NYSE session,
Sharadar denominator, security identity, share normalization, and (for
prospective evidence) the consensus acquisition freeze against that session.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.pead_known_by_policy import (
    KNOWN_BY_POLICY,
    KNOWN_BY_POLICY_SHA256,
    PeadKnownByPolicyError,
    select_conservative_consensus_vintage,
)
from data.pead_announcement_availability import (
    LICENSED_RELEASE_ADAPTER,
    SEC_HTTPS_OBSERVATION_ADAPTER,
    PeadAnnouncementAvailabilityError,
    validate_pead_announcement_availability,
)
from data.pead_announcement_evidence import (
    PeadAnnouncementEvidenceError,
    validate_pead_announcement_evidence,
)
from data.pead_consensus_replay import (
    PeadConsensusReplayError,
    verify_pead_consensus_replay,
)
from data.pead_event_universe import canonical_json, content_hash


SOURCE_RECONCILIATION_V2_SCHEMA_VERSION = "pead_source_reconciliation.v2"
SOURCE_RECONCILIATION_V2_POLICY_SCHEMA_VERSION = "pead_source_reconciliation_policy.v2"
TRUST_ROOT_SET_SCHEMA_VERSION = "pead_sha256_trust_root_set.v1"
MAX_SOURCE_RECONCILIATION_V2_BYTES = 512 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "reconciled_at_utc",
    "policy",
    "bindings",
    "event_results",
    "coverage",
    "qualification",
}
_BINDING_FIELDS = {
    "event_universe_sha256",
    "consensus_replay_sha256",
    "consensus_raw_artifact_sha256",
    "consensus_evidence_sha256",
    "announcement_evidence_sha256",
    "announcement_availability_sha256",
    "market_snapshot_sha256",
    "identity_snapshot_sha256",
    "candidate_specification_sha256",
    "construction_code_sha256",
    "consensus_source_manifest_sha256",
    "consensus_metric_profile_sha256",
    "known_by_policy_sha256",
    "reconciliation_policy_sha256",
    "trusted_consensus_source_manifest_set_sha256",
    "trusted_consensus_metric_profile_set_sha256",
    "trusted_consensus_identity_snapshot_set_sha256",
    "trusted_consensus_event_universe_set_sha256",
    "trusted_consensus_raw_artifact_set_sha256",
    "trusted_announcement_evidence_set_sha256",
    "trusted_announcement_provider_manifest_set_sha256",
    "trusted_announcement_provider_record_set_sha256",
    "trusted_announcement_checkpoint_set_sha256",
}
_COVERAGE_FIELDS = {
    "expected_event_count",
    "event_source_reconciled_count",
    "excluded_event_count",
    "exhaustive_event_accounting",
    "systemic_blockers",
    "event_blocker_counts",
}
_QUALIFICATION_FIELDS = {
    "has_event_source_reconciled_inputs",
    "all_expected_events_source_reconciled",
    "event_source_reconciliation_allowed",
    "research_consumable",
    "market_accounting_join_required",
    "prospective_consensus_freeze_check_pending_market_evidence",
    "historical_replication_allowed",
    "edge_claim_allowed",
    "paper_execution_allowed",
    "live_deployment_allowed",
}
_POLICY = {
    "schema_version": SOURCE_RECONCILIATION_V2_POLICY_SCHEMA_VERSION,
    "availability_claim": "known_public_by",
    "known_by_policy": KNOWN_BY_POLICY,
    "known_by_policy_sha256": KNOWN_BY_POLICY_SHA256,
    "consensus_source_rule": "verified_raw_consensus_replay_only",
    "actual_source_rule": "independent_announcement_evidence_only",
    "availability_source_rule": "verified_announcement_availability_only",
    "metric_rule": "exact_canonical_semantics_no_implicit_mapping",
    "surprise_rule": "canonical_actual_minus_selected_consensus_decimal",
    "missing_value_rule": "exclude_event_without_changing_frozen_partition",
    "partition_rule": "one_embedded_child_event_universe_per_receipt",
    "market_join_rule": "separate_sharadar_market_accounting_receipt_required",
    "prospective_freeze_rule": ("deferred_until_selected_prior_nyse_session_is_replayed"),
}
_METRIC_COMPARISONS = (
    ("metric", "metric_id", "metric"),
    ("accounting_basis", "accounting_basis", "accounting_basis"),
    ("per_share_basis", "per_share_basis", "per_share_basis"),
    ("scope", "scope", "scope"),
    ("canonical_share_basis", "canonical_share_basis", "canonical_share_basis"),
    ("currency", "currency_code", "currency"),
    ("unit", "unit", "unit"),
    (
        "metric_definition_sha256",
        "metric_definition_sha256",
        "metric_definition_sha256",
    ),
)
_AVAILABILITY_EVENT_BLOCKERS = frozenset(
    {
        "announcement_actuals_incomplete",
        "expected_events_missing_availability",
    }
)


class PeadSourceReconciliationV2Error(ValueError):
    """A v2 source reconciliation is malformed or fails authoritative replay."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadSourceReconciliationV2Error(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadSourceReconciliationV2Error(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadSourceReconciliationV2Error(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadSourceReconciliationV2Error(f"{label} must be canonical UTC with Z") from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadSourceReconciliationV2Error(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _trust_roots(values: Collection[str], label: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise PeadSourceReconciliationV2Error(f"{label} must be a collection")
    return sorted({_sha(value, f"{label} entry") for value in values})


def _trust_root_set_hash(values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema_version": TRUST_ROOT_SET_SCHEMA_VERSION,
            "members": list(values),
        }
    )


def _decimal(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise PeadSourceReconciliationV2Error(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise PeadSourceReconciliationV2Error(f"{label} is not a decimal") from exc
    if not parsed.is_finite():
        raise PeadSourceReconciliationV2Error(f"{label} must be finite")
    return parsed


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _comparison(
    field: str,
    consensus_value: Any,
    announcement_value: Any,
    decision: str,
) -> dict[str, Any]:
    return {
        "field": field,
        "consensus_value": _plain(consensus_value),
        "announcement_value": _plain(announcement_value),
        "decision": decision,
    }


def _announcement_source(outcome: Mapping[str, Any]) -> dict[str, Any]:
    record = outcome["available_record"]
    if record is None:
        return {
            "disposition": outcome["disposition"],
            "missing_reason": outcome["missing_reason"],
            "source_kind": None,
            "accession_number": None,
            "exhibit_document_sha256": None,
            "edgar_acceptance_at_utc": None,
            "canonical_actual": None,
        }
    return {
        "disposition": "available",
        "missing_reason": None,
        "source_kind": record["source_kind"],
        "accession_number": record["accession_number"],
        "exhibit_document_sha256": record["exhibit_document"]["raw_document"]["sha256"],
        "edgar_acceptance_at_utc": record["edgar_acceptance_at_utc"],
        "canonical_actual": record["canonical_actual"],
    }


def _availability_source(outcome: Mapping[str, Any]) -> dict[str, Any]:
    claim = outcome["claim"]
    if claim is None:
        return {
            "disposition": outcome["disposition"],
            "missing_reason": outcome["missing_reason"],
            "claim_kind": None,
            "known_public_by_at_utc": None,
            "adapter_id": None,
            "claim_evidence_sha256": None,
            "eligibility": None,
        }
    return {
        "disposition": "available",
        "missing_reason": None,
        "claim_kind": claim["claim_kind"],
        "known_public_by_at_utc": claim["known_public_by_at_utc"],
        "adapter_id": claim["adapter_id"],
        "claim_evidence_sha256": content_hash(claim["evidence"]),
        "eligibility": claim["eligibility"],
    }


def _systemic_blockers(
    *,
    universe: Mapping[str, Any],
    availability: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not universe["payload"]["qualification_allowed"]:
        blockers.append("event_universe_not_qualified")
    blockers.extend(
        f"availability_{code}"
        for code in availability["payload"]["coverage"]["blockers"]
        if code not in _AVAILABILITY_EVENT_BLOCKERS
    )
    return sorted(set(blockers))


def _event_result(
    *,
    event: Mapping[str, Any],
    consensus_record: Mapping[str, Any],
    announcement_outcome: Mapping[str, Any],
    availability_outcome: Mapping[str, Any],
    systemic_blockers: Sequence[str],
    bindings: Mapping[str, str],
    receipt_captured_at_by_hash: Mapping[str, str],
) -> dict[str, Any]:
    blockers = list(systemic_blockers)
    announcement_source = _announcement_source(announcement_outcome)
    availability_source = _availability_source(availability_outcome)
    comparisons = [
        _comparison(
            "event_id_consensus_announcement",
            consensus_record["event_id"],
            announcement_outcome["event_id"],
            "match",
        ),
        _comparison(
            "event_id_consensus_availability",
            consensus_record["event_id"],
            availability_outcome["event_id"],
            "match",
        ),
    ]
    if consensus_record["disposition"] != "available":
        blockers.append("consensus_event_missing")
    if announcement_outcome["disposition"] != "available":
        blockers.append("announcement_event_missing")
    if availability_outcome["disposition"] != "available":
        blockers.append("announcement_availability_missing")

    selected: dict[str, Any] | None = None
    eligible_hashes: list[str] = []
    known_public_by = availability_source["known_public_by_at_utc"]
    if consensus_record["disposition"] == "available":
        vintages = consensus_record["vintages"]
        if len({content_hash(vintage["metric"]) for vintage in vintages}) != 1:
            blockers.append("consensus_metric_history_inconsistent")
        elif known_public_by is not None:
            try:
                selected, selection_blockers, eligible_hashes = (
                    select_conservative_consensus_vintage(
                        vintages,
                        known_public_by_at_utc=known_public_by,
                    )
                )
            except PeadKnownByPolicyError as exc:
                raise PeadSourceReconciliationV2Error(
                    "known-by consensus selection failed"
                ) from exc
            blockers.extend(selection_blockers)

    selected_receipt_captured_at: str | None = None
    if selected is not None:
        selected_receipt_captured_at = receipt_captured_at_by_hash.get(
            selected["acquisition_receipt_sha256"]
        )
        if selected_receipt_captured_at is None:
            raise PeadSourceReconciliationV2Error(
                "selected consensus vintage has no replayed acquisition receipt"
            )

    actual = announcement_source["canonical_actual"]
    if selected is not None and actual is not None:
        for field, consensus_field, announcement_field in _METRIC_COMPARISONS:
            left = selected["metric"][consensus_field]
            right = actual[announcement_field]
            matched = left == right
            comparisons.append(_comparison(field, left, right, "match" if matched else "mismatch"))
            if not matched:
                blockers.append(f"{field}_mismatch")
        comparisons.append(
            _comparison(
                "announcement_availability",
                selected["provider_as_of_date"],
                known_public_by,
                "strict_prior_eastern_calendar_date",
            )
        )
    else:
        for field, _, announcement_field in _METRIC_COMPARISONS:
            right = actual[announcement_field] if actual is not None else None
            comparisons.append(_comparison(field, None, right, "not_compared"))
        comparisons.append(
            _comparison("announcement_availability", None, known_public_by, "not_compared")
        )

    blockers = sorted(set(blockers))
    reconciled_input: dict[str, Any] | None = None
    if not blockers and selected is not None and actual is not None and known_public_by is not None:
        actual_value = _decimal(actual["canonical_value"], "canonical actual")
        consensus_value = _decimal(selected["consensus_value"], "consensus value")
        surprise = actual_value - consensus_value
        direction = "positive" if surprise > 0 else "negative" if surprise < 0 else "zero"
        reconciled_input = {
            "event_id": event["event_id"],
            "event_key": event["event_key"],
            "actual_value": actual["canonical_value"],
            "consensus_value": selected["consensus_value"],
            "raw_surprise": _canonical_decimal(surprise),
            "surprise_direction": direction,
            "analyst_count": selected["analyst_count"],
            "known_public_by_at_utc": known_public_by,
            "availability_adapter_id": availability_source["adapter_id"],
            "consensus_provider_as_of_date": selected["provider_as_of_date"],
            "consensus_available_at_utc": selected["trusted_available_at_utc"],
            "consensus_availability_precision": selected["availability_precision"],
            "consensus_receipt_captured_at_utc": selected_receipt_captured_at,
            "consensus_cutoff_rule": availability_source["eligibility"]["consensus_cutoff_rule"],
            "market_cutoff_rule": availability_source["eligibility"]["market_cutoff_rule"],
            "metric": selected["metric"],
            "provenance": {
                "event_universe_sha256": bindings["event_universe_sha256"],
                "consensus_replay_sha256": bindings["consensus_replay_sha256"],
                "consensus_raw_artifact_sha256": bindings["consensus_raw_artifact_sha256"],
                "consensus_evidence_sha256": bindings["consensus_evidence_sha256"],
                "announcement_evidence_sha256": bindings["announcement_evidence_sha256"],
                "announcement_availability_sha256": bindings["announcement_availability_sha256"],
                "consensus_raw_record_sha256": selected["raw_record_sha256"],
                "consensus_acquisition_receipt_sha256": selected["acquisition_receipt_sha256"],
                "announcement_exhibit_document_sha256": announcement_source[
                    "exhibit_document_sha256"
                ],
                "announcement_normalization_evidence_sha256": actual[
                    "normalization_evidence_sha256"
                ],
                "announcement_availability_claim_evidence_sha256": (
                    availability_source["claim_evidence_sha256"]
                ),
            },
        }

    consensus_source = {
        "disposition": consensus_record["disposition"],
        "missing_reason": consensus_record["missing_reason"],
        "vintages": consensus_record["vintages"],
        "eligible_raw_record_sha256s": eligible_hashes,
        "selected_vintage": selected,
        "selected_receipt_captured_at_utc": selected_receipt_captured_at,
        "selection_rule": "strict_prior_eastern_calendar_date",
    }
    return {
        "event_id": event["event_id"],
        "event_key": event["event_key"],
        "disposition": ("event_source_reconciled" if reconciled_input is not None else "excluded"),
        "blockers": blockers,
        "source_values": {
            "consensus": consensus_source,
            "announcement": announcement_source,
            "availability": availability_source,
        },
        "comparisons": sorted(comparisons, key=lambda row: row["field"]),
        "event_source_input": reconciled_input,
    }


def build_pead_source_reconciliation_v2(
    consensus_replay: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
    announcement_availability: Mapping[str, Any],
    *,
    consensus_raw_artifact: bytes,
    reconciled_at_utc: str,
    trusted_consensus_source_manifest_sha256s: Collection[str],
    trusted_consensus_metric_profile_sha256s: Collection[str],
    trusted_consensus_identity_snapshot_sha256s: Collection[str],
    trusted_consensus_event_universe_sha256s: Collection[str],
    trusted_consensus_raw_artifact_sha256s: Collection[str],
    trusted_announcement_evidence_sha256s: Collection[str],
    trusted_announcement_provider_manifest_sha256s: Collection[str] = (),
    trusted_announcement_provider_record_sha256s: Collection[str] = (),
    trusted_announcement_checkpoint_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    """Build one authoritative child-universe event-source receipt."""
    if not isinstance(consensus_raw_artifact, bytes):
        raise PeadSourceReconciliationV2Error("consensus_raw_artifact must be exact bytes")
    trust_sets = {
        "consensus_source_manifest": _trust_roots(
            trusted_consensus_source_manifest_sha256s,
            "trusted_consensus_source_manifest_sha256s",
        ),
        "consensus_metric_profile": _trust_roots(
            trusted_consensus_metric_profile_sha256s,
            "trusted_consensus_metric_profile_sha256s",
        ),
        "consensus_identity_snapshot": _trust_roots(
            trusted_consensus_identity_snapshot_sha256s,
            "trusted_consensus_identity_snapshot_sha256s",
        ),
        "consensus_event_universe": _trust_roots(
            trusted_consensus_event_universe_sha256s,
            "trusted_consensus_event_universe_sha256s",
        ),
        "consensus_raw_artifact": _trust_roots(
            trusted_consensus_raw_artifact_sha256s,
            "trusted_consensus_raw_artifact_sha256s",
        ),
        "announcement_evidence": _trust_roots(
            trusted_announcement_evidence_sha256s,
            "trusted_announcement_evidence_sha256s",
        ),
        "announcement_provider_manifest": _trust_roots(
            trusted_announcement_provider_manifest_sha256s,
            "trusted_announcement_provider_manifest_sha256s",
        ),
        "announcement_provider_record": _trust_roots(
            trusted_announcement_provider_record_sha256s,
            "trusted_announcement_provider_record_sha256s",
        ),
        "announcement_checkpoint": _trust_roots(
            trusted_announcement_checkpoint_sha256s,
            "trusted_announcement_checkpoint_sha256s",
        ),
    }
    raw_sha256 = hashlib.sha256(consensus_raw_artifact).hexdigest()
    if raw_sha256 not in trust_sets["consensus_raw_artifact"]:
        raise PeadSourceReconciliationV2Error(
            "consensus raw artifact is not in the external trust registry"
        )
    try:
        replay = verify_pead_consensus_replay(
            consensus_replay,
            raw_artifact=consensus_raw_artifact,
            trusted_source_manifest_sha256s=trust_sets["consensus_source_manifest"],
            trusted_metric_profile_sha256s=trust_sets["consensus_metric_profile"],
            trusted_identity_snapshot_sha256s=trust_sets["consensus_identity_snapshot"],
            trusted_event_universe_sha256s=trust_sets["consensus_event_universe"],
            trusted_raw_artifact_sha256s=trust_sets["consensus_raw_artifact"],
        )
    except PeadConsensusReplayError as exc:
        raise PeadSourceReconciliationV2Error(
            "consensus replay or raw artifact is invalid"
        ) from exc
    if not replay["payload"]["qualification_allowed"]:
        raise PeadSourceReconciliationV2Error("consensus replay is valid but not source-qualified")
    consensus = replay["payload"]["consensus_evidence"]
    universe = consensus["payload"]["event_universe"]
    try:
        announcement = validate_pead_announcement_evidence(
            announcement_evidence,
            expected_event_manifest=universe,
        )
        if announcement["artifact_hash"] not in trust_sets["announcement_evidence"]:
            raise PeadSourceReconciliationV2Error(
                "announcement evidence is not in the external trust registry"
            )
        availability = validate_pead_announcement_availability(
            announcement_availability,
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            trusted_provider_manifest_sha256s=trust_sets["announcement_provider_manifest"],
            trusted_provider_record_sha256s=trust_sets["announcement_provider_record"],
            trusted_checkpoint_sha256s=trust_sets["announcement_checkpoint"],
        )
    except PeadSourceReconciliationV2Error:
        raise
    except (PeadAnnouncementEvidenceError, PeadAnnouncementAvailabilityError) as exc:
        raise PeadSourceReconciliationV2Error(
            "announcement actual or availability evidence is invalid"
        ) from exc
    candidate = consensus["payload"]["candidate_id"]
    if (
        candidate != announcement["payload"]["candidate_id"]
        or candidate != availability["payload"]["candidate_id"]
    ):
        raise PeadSourceReconciliationV2Error("source evidence belongs to different candidates")
    evidence_class = consensus["payload"]["evidence_class"]
    expected_availability_class = {
        "historical_reconstruction": "historical_reconstruction",
        "prospective_signal": "prospective",
    }.get(evidence_class)
    if expected_availability_class is None:
        raise PeadSourceReconciliationV2Error(
            "consensus evidence class is not eligible for v2 reconciliation"
        )
    if availability["payload"]["evidence_class"] != expected_availability_class:
        raise PeadSourceReconciliationV2Error(
            "consensus and announcement availability evidence classes differ"
        )
    required_adapter, required_eligibility_flag = {
        "historical_reconstruction": (
            LICENSED_RELEASE_ADAPTER,
            "historical_reconstruction_allowed",
        ),
        "prospective": (
            SEC_HTTPS_OBSERVATION_ADAPTER,
            "prospective_observation_allowed",
        ),
    }[expected_availability_class]
    for outcome in availability["payload"]["outcomes"]:
        if outcome["disposition"] != "available":
            continue
        claim = outcome["claim"]
        eligibility = claim["eligibility"]
        if (
            claim["adapter_id"] != required_adapter
            or eligibility["eligible_for_declared_evidence_class"] is not True
            or eligibility[required_eligibility_flag] is not True
        ):
            raise PeadSourceReconciliationV2Error(
                "availability adapter is ineligible for declared evidence class"
            )

    reconciled_text, reconciled_at = _utc(reconciled_at_utc, "reconciled_at_utc")
    latest_input_time = max(
        _utc(
            consensus["payload"]["source"]["captured_at_utc"],
            "consensus source capture",
        )[1],
        _utc(
            announcement["payload"]["created_at_utc"],
            "announcement evidence creation",
        )[1],
        _utc(
            availability["payload"]["created_at_utc"],
            "announcement availability creation",
        )[1],
    )
    if reconciled_at < latest_input_time:
        raise PeadSourceReconciliationV2Error("reconciliation timestamp precedes a source artifact")

    replay_bindings = replay["payload"]["bindings"]
    universe_bindings = universe["payload"]["bindings"]
    if raw_sha256 != replay_bindings["raw_artifact_bytes_sha256"]:
        raise PeadSourceReconciliationV2Error("verified replay raw-artifact binding differs")
    bindings = {
        "event_universe_sha256": universe["artifact_hash"],
        "consensus_replay_sha256": replay["artifact_hash"],
        "consensus_raw_artifact_sha256": raw_sha256,
        "consensus_evidence_sha256": consensus["artifact_hash"],
        "announcement_evidence_sha256": announcement["artifact_hash"],
        "announcement_availability_sha256": availability["artifact_hash"],
        "market_snapshot_sha256": universe_bindings["market_snapshot_sha256"],
        "identity_snapshot_sha256": replay_bindings["identity_snapshot_sha256"],
        "candidate_specification_sha256": universe_bindings["candidate_specification_sha256"],
        "construction_code_sha256": universe_bindings["construction_code_sha256"],
        "consensus_source_manifest_sha256": replay_bindings["source_manifest_sha256"],
        "consensus_metric_profile_sha256": replay_bindings["metric_profile_sha256"],
        "known_by_policy_sha256": KNOWN_BY_POLICY_SHA256,
        "reconciliation_policy_sha256": content_hash(_POLICY),
        "trusted_consensus_source_manifest_set_sha256": _trust_root_set_hash(
            trust_sets["consensus_source_manifest"]
        ),
        "trusted_consensus_metric_profile_set_sha256": _trust_root_set_hash(
            trust_sets["consensus_metric_profile"]
        ),
        "trusted_consensus_identity_snapshot_set_sha256": _trust_root_set_hash(
            trust_sets["consensus_identity_snapshot"]
        ),
        "trusted_consensus_event_universe_set_sha256": _trust_root_set_hash(
            trust_sets["consensus_event_universe"]
        ),
        "trusted_consensus_raw_artifact_set_sha256": _trust_root_set_hash(
            trust_sets["consensus_raw_artifact"]
        ),
        "trusted_announcement_evidence_set_sha256": _trust_root_set_hash(
            trust_sets["announcement_evidence"]
        ),
        "trusted_announcement_provider_manifest_set_sha256": _trust_root_set_hash(
            trust_sets["announcement_provider_manifest"]
        ),
        "trusted_announcement_provider_record_set_sha256": _trust_root_set_hash(
            trust_sets["announcement_provider_record"]
        ),
        "trusted_announcement_checkpoint_set_sha256": _trust_root_set_hash(
            trust_sets["announcement_checkpoint"]
        ),
    }
    if set(bindings) != _BINDING_FIELDS:  # pragma: no cover - developer guard
        raise AssertionError("v2 binding field set drifted")

    systemic = _systemic_blockers(universe=universe, availability=availability)
    expected_events = universe["payload"]["expected_events"]
    consensus_by_id = {row["event_id"]: row for row in consensus["payload"]["event_records"]}
    announcement_by_id = {row["event_id"]: row for row in announcement["payload"]["outcomes"]}
    availability_by_id = {row["event_id"]: row for row in availability["payload"]["outcomes"]}
    receipt_captured_at_by_hash = {
        row["receipt_sha256"]: row["source_captured_at_utc"]
        for row in consensus["payload"]["acquisition_receipts"]
    }
    results = [
        _event_result(
            event=event,
            consensus_record=consensus_by_id[event["event_id"]],
            announcement_outcome=announcement_by_id[event["event_id"]],
            availability_outcome=availability_by_id[event["event_id"]],
            systemic_blockers=systemic,
            bindings=bindings,
            receipt_captured_at_by_hash=receipt_captured_at_by_hash,
        )
        for event in expected_events
    ]
    reconciled_count = sum(result["disposition"] == "event_source_reconciled" for result in results)
    blocker_counts: dict[str, int] = {}
    for result in results:
        for blocker in result["blockers"]:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    has_reconciled = reconciled_count > 0 and not systemic
    payload = {
        "schema_version": SOURCE_RECONCILIATION_V2_SCHEMA_VERSION,
        "candidate_id": candidate,
        "evidence_class": evidence_class,
        "reconciled_at_utc": reconciled_text,
        "policy": _POLICY,
        "bindings": bindings,
        "event_results": results,
        "coverage": {
            "expected_event_count": len(expected_events),
            "event_source_reconciled_count": reconciled_count,
            "excluded_event_count": len(expected_events) - reconciled_count,
            "exhaustive_event_accounting": (
                [result["event_id"] for result in results]
                == universe["payload"]["expected_event_ids"]
            ),
            "systemic_blockers": systemic,
            "event_blocker_counts": {key: blocker_counts[key] for key in sorted(blocker_counts)},
        },
        "qualification": {
            "has_event_source_reconciled_inputs": has_reconciled,
            "all_expected_events_source_reconciled": bool(
                results and reconciled_count == len(results) and not systemic
            ),
            "event_source_reconciliation_allowed": has_reconciled,
            "research_consumable": False,
            "market_accounting_join_required": True,
            "prospective_consensus_freeze_check_pending_market_evidence": (
                evidence_class == "prospective_signal"
            ),
            "historical_replication_allowed": False,
            "edge_claim_allowed": False,
            "paper_execution_allowed": False,
            "live_deployment_allowed": False,
        },
    }
    return _plain({"artifact_hash": content_hash(payload), "payload": payload})


def validate_pead_source_reconciliation_v2(
    document: Mapping[str, Any],
    *,
    consensus_replay: Mapping[str, Any],
    consensus_raw_artifact: bytes,
    announcement_evidence: Mapping[str, Any],
    announcement_availability: Mapping[str, Any],
    trusted_consensus_source_manifest_sha256s: Collection[str],
    trusted_consensus_metric_profile_sha256s: Collection[str],
    trusted_consensus_identity_snapshot_sha256s: Collection[str],
    trusted_consensus_event_universe_sha256s: Collection[str],
    trusted_consensus_raw_artifact_sha256s: Collection[str],
    trusted_announcement_evidence_sha256s: Collection[str],
    trusted_announcement_provider_manifest_sha256s: Collection[str] = (),
    trusted_announcement_provider_record_sha256s: Collection[str] = (),
    trusted_announcement_checkpoint_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    """Rebuild the receipt from every original artifact and external trust set."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "source reconciliation v2")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "source reconciliation v2.payload")
    claimed = _sha(wrapper["artifact_hash"], "source reconciliation v2.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadSourceReconciliationV2Error("source reconciliation v2 artifact hash mismatch")
    if payload["schema_version"] != SOURCE_RECONCILIATION_V2_SCHEMA_VERSION:
        raise PeadSourceReconciliationV2Error("unsupported source reconciliation v2 schema")
    _exact(payload["bindings"], _BINDING_FIELDS, "source reconciliation v2.bindings")
    _exact(payload["coverage"], _COVERAGE_FIELDS, "source reconciliation v2.coverage")
    _exact(
        payload["qualification"],
        _QUALIFICATION_FIELDS,
        "source reconciliation v2.qualification",
    )
    reconciled_at = _utc(payload["reconciled_at_utc"], "reconciled_at_utc")[0]
    expected = build_pead_source_reconciliation_v2(
        consensus_replay,
        announcement_evidence,
        announcement_availability,
        consensus_raw_artifact=consensus_raw_artifact,
        reconciled_at_utc=reconciled_at,
        trusted_consensus_source_manifest_sha256s=(trusted_consensus_source_manifest_sha256s),
        trusted_consensus_metric_profile_sha256s=(trusted_consensus_metric_profile_sha256s),
        trusted_consensus_identity_snapshot_sha256s=(trusted_consensus_identity_snapshot_sha256s),
        trusted_consensus_event_universe_sha256s=(trusted_consensus_event_universe_sha256s),
        trusted_consensus_raw_artifact_sha256s=(trusted_consensus_raw_artifact_sha256s),
        trusted_announcement_evidence_sha256s=(trusted_announcement_evidence_sha256s),
        trusted_announcement_provider_manifest_sha256s=(
            trusted_announcement_provider_manifest_sha256s
        ),
        trusted_announcement_provider_record_sha256s=(trusted_announcement_provider_record_sha256s),
        trusted_announcement_checkpoint_sha256s=(trusted_announcement_checkpoint_sha256s),
    )
    if _plain(document) != expected:
        raise PeadSourceReconciliationV2Error(
            "source reconciliation v2 does not replay from its original evidence"
        )
    return expected


def verify_pead_source_reconciliation_v2(
    document: Mapping[str, Any],
    *,
    consensus_replay: Mapping[str, Any],
    consensus_raw_artifact: bytes,
    announcement_evidence: Mapping[str, Any],
    announcement_availability: Mapping[str, Any],
    trusted_consensus_source_manifest_sha256s: Collection[str],
    trusted_consensus_metric_profile_sha256s: Collection[str],
    trusted_consensus_identity_snapshot_sha256s: Collection[str],
    trusted_consensus_event_universe_sha256s: Collection[str],
    trusted_consensus_raw_artifact_sha256s: Collection[str],
    trusted_announcement_evidence_sha256s: Collection[str],
    trusted_announcement_provider_manifest_sha256s: Collection[str] = (),
    trusted_announcement_provider_record_sha256s: Collection[str] = (),
    trusted_announcement_checkpoint_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    """Explicit authoritative verifier for downstream market evidence."""
    return validate_pead_source_reconciliation_v2(
        document,
        consensus_replay=consensus_replay,
        consensus_raw_artifact=consensus_raw_artifact,
        announcement_evidence=announcement_evidence,
        announcement_availability=announcement_availability,
        trusted_consensus_source_manifest_sha256s=(trusted_consensus_source_manifest_sha256s),
        trusted_consensus_metric_profile_sha256s=(trusted_consensus_metric_profile_sha256s),
        trusted_consensus_identity_snapshot_sha256s=(trusted_consensus_identity_snapshot_sha256s),
        trusted_consensus_event_universe_sha256s=(trusted_consensus_event_universe_sha256s),
        trusted_consensus_raw_artifact_sha256s=(trusted_consensus_raw_artifact_sha256s),
        trusted_announcement_evidence_sha256s=(trusted_announcement_evidence_sha256s),
        trusted_announcement_provider_manifest_sha256s=(
            trusted_announcement_provider_manifest_sha256s
        ),
        trusted_announcement_provider_record_sha256s=(trusted_announcement_provider_record_sha256s),
        trusted_announcement_checkpoint_sha256s=(trusted_announcement_checkpoint_sha256s),
    )


def _strict_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadSourceReconciliationV2Error(
            f"source reconciliation v2 is not a regular file: {path}"
        )
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_SOURCE_RECONCILIATION_V2_BYTES:
        raise PeadSourceReconciliationV2Error("source reconciliation v2 file size is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadSourceReconciliationV2Error("source reconciliation v2 is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadSourceReconciliationV2Error(
                    f"source reconciliation v2 contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadSourceReconciliationV2Error(
            f"source reconciliation v2 contains invalid number {token}"
        )

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadSourceReconciliationV2Error("invalid source reconciliation v2 JSON") from exc
    if not isinstance(value, dict):
        raise PeadSourceReconciliationV2Error("source reconciliation v2 root must be an object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadSourceReconciliationV2Error(
            "source reconciliation v2 bytes are not canonical JSON plus one newline"
        )
    return value


def load_pead_source_reconciliation_v2(
    path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load canonical bytes and authoritatively replay all original inputs."""
    return validate_pead_source_reconciliation_v2(_strict_json_file(Path(path)), **kwargs)


__all__ = [
    "MAX_SOURCE_RECONCILIATION_V2_BYTES",
    "PeadSourceReconciliationV2Error",
    "SOURCE_RECONCILIATION_V2_POLICY_SCHEMA_VERSION",
    "SOURCE_RECONCILIATION_V2_SCHEMA_VERSION",
    "build_pead_source_reconciliation_v2",
    "load_pead_source_reconciliation_v2",
    "validate_pead_source_reconciliation_v2",
    "verify_pead_source_reconciliation_v2",
]
