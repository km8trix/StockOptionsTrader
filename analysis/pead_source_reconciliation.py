"""Fail-closed reconciliation of independent PEAD actual and consensus evidence.

This is an upstream evidence boundary, not a tradable-signal artifact.  It
reconciles the frozen event universe, provider-neutral consensus vintages, and
independent announcement actuals.  A later, separately verified market lane
must still select the preannouncement price denominator and bind security
identity before research may consume the result.

The public verifier rebuilds the complete receipt from the original evidence.
Consequently, a content-addressed but self-consistent edited receipt is not
accepted without the same source artifacts that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from data.pead_announcement_evidence import (
    PeadAnnouncementEvidenceError,
    validate_pead_announcement_evidence,
)
from data.pead_consensus_evidence import (
    PeadConsensusEvidenceError,
    validate_pead_consensus_evidence,
)
from data.pead_event_universe import canonical_json, content_hash


SOURCE_RECONCILIATION_SCHEMA_VERSION = "pead_source_reconciliation.v1"
SOURCE_RECONCILIATION_POLICY_SCHEMA_VERSION = (
    "pead_source_reconciliation_policy.v1"
)
MAX_SOURCE_RECONCILIATION_BYTES = 512 * 1024 * 1024
MINIMUM_ANALYST_COUNT = 2

_EASTERN = ZoneInfo("America/New_York")
_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "reconciled_at_utc",
    "policy",
    "bindings",
    "event_results",
    "coverage",
    "qualification",
}
_BINDING_FIELDS = {
    "event_universe_sha256",
    "consensus_evidence_sha256",
    "announcement_evidence_sha256",
    "market_snapshot_sha256",
    "identity_snapshot_sha256",
    "candidate_specification_sha256",
    "construction_code_sha256",
    "consensus_source_manifest_sha256",
    "reconciliation_policy_sha256",
}
_POLICY = {
    "schema_version": SOURCE_RECONCILIATION_POLICY_SCHEMA_VERSION,
    "minimum_analyst_count": MINIMUM_ANALYST_COUNT,
    "actual_source_rule": "independent_announcement_artifact_only",
    "consensus_selection_rule": "latest_unambiguous_eligible_provider_vintage",
    "timestamp_rule": "trusted_consensus_availability_strictly_before_first_public",
    "date_only_rule": (
        "provider_as_of_date_strictly_before_first_public_america_new_york_date"
    ),
    "same_date_without_intraday_time_allowed": False,
    "metric_rule": "exact_canonical_semantics_no_implicit_mapping",
    "missing_value_rule": "exclude_event_without_changing_event_universe",
    "raw_consensus_replay_rule": (
        "required_at_final_signal_input_boundary_via_registered_provider_adapter"
    ),
    "external_binding_replay_rule": (
        "required_at_final_signal_input_boundary_from_exact_bound_bytes"
    ),
    "market_join_rule": "separate_pead_signal_input_reconciliation_required",
}
_COVERAGE_FIELDS = {
    "expected_event_count",
    "reconciled_event_count",
    "excluded_event_count",
    "reconciliation_complete",
    "systemic_blockers",
    "event_blocker_counts",
}
_QUALIFICATION_FIELDS = {
    "has_reconciled_event_inputs",
    "all_expected_events_reconciled",
    "source_qualified_event_inputs_allowed",
    "raw_consensus_normalization_replay_required",
    "external_binding_replay_required",
    "market_accounting_join_required",
    "historical_replication_allowed",
    "edge_claim_allowed",
    "paper_execution_allowed",
    "live_deployment_allowed",
}
_SYSTEMIC_CONSENSUS_EVENT_BLOCKERS = frozenset(
    {"expected_events_missing_consensus"}
)
_SYSTEMIC_ANNOUNCEMENT_EVENT_BLOCKERS = frozenset(
    {"expected_events_missing_announcement", "first_public_timestamps_unproven"}
)
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


class PeadSourceReconciliationError(ValueError):
    """The PEAD source reconciliation is malformed or cannot be replayed."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadSourceReconciliationError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadSourceReconciliationError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadSourceReconciliationError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadSourceReconciliationError(
            f"{label} must be canonical UTC with Z"
        ) from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadSourceReconciliationError(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _decimal(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise PeadSourceReconciliationError(f"{label} is not a decimal") from exc
    if not parsed.is_finite():
        raise PeadSourceReconciliationError(f"{label} is not finite")
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


def _metric_signature(vintage: Mapping[str, Any]) -> str:
    return content_hash(vintage["metric"])


def select_consensus_vintage(
    vintages: Sequence[Mapping[str, Any]],
    *,
    first_public_at_utc: str,
) -> tuple[dict[str, Any] | None, list[str], list[str], str | None]:
    """Select the latest unambiguous preannouncement vintage.

    Callers normally pass vintages from a validated consensus artifact.  This
    helper is public so the frozen selection policy can be independently tested.
    It never falls back from a tied or low-analyst latest vintage.
    """
    _, first_public = _utc(first_public_at_utc, "first_public_at_utc")
    if isinstance(vintages, (str, bytes)) or not isinstance(vintages, Sequence):
        raise PeadSourceReconciliationError("vintages must be a sequence")
    rows = [dict(item) for item in vintages]
    if not rows:
        return None, ["consensus_event_missing"], [], None

    if len({_metric_signature(item) for item in rows}) != 1:
        return None, ["consensus_metric_history_inconsistent"], [], None

    eligible: list[dict[str, Any]] = []
    first_public_local_day = first_public.astimezone(_EASTERN).date()
    for row in rows:
        precision = row.get("availability_precision")
        as_of_text = row.get("provider_as_of_date")
        try:
            as_of = date.fromisoformat(as_of_text)
        except (TypeError, ValueError) as exc:
            raise PeadSourceReconciliationError(
                "consensus provider_as_of_date is invalid"
            ) from exc
        trusted = row.get("trusted_available_at_utc")
        if precision == "date":
            if trusted is not None:
                raise PeadSourceReconciliationError(
                    "date-precision consensus cannot carry an exact timestamp"
                )
            if as_of < first_public_local_day:
                eligible.append(row)
        elif precision in {"second", "microsecond"}:
            if trusted is None:
                raise PeadSourceReconciliationError(
                    "exact-precision consensus requires an availability timestamp"
                )
            _, trusted_dt = _utc(trusted, "trusted_available_at_utc")
            if trusted_dt < first_public:
                eligible.append(row)
        else:
            raise PeadSourceReconciliationError(
                "consensus availability precision is unsupported"
            )

    eligible_hashes = sorted(row["raw_record_sha256"] for row in eligible)
    if not eligible:
        return (
            None,
            ["consensus_no_eligible_preannouncement_vintage"],
            eligible_hashes,
            None,
        )

    latest_day = max(row["provider_as_of_date"] for row in eligible)
    finalists = [row for row in eligible if row["provider_as_of_date"] == latest_day]
    date_only = [row for row in finalists if row["availability_precision"] == "date"]
    exact = [row for row in finalists if row["availability_precision"] != "date"]
    selected: dict[str, Any] | None = None
    precision_rule: str | None = None
    if date_only:
        if len(finalists) == 1:
            selected = date_only[0]
            precision_rule = "strict_prior_eastern_calendar_date"
    else:
        exact_with_times = [
            (
                _utc(
                    row["trusted_available_at_utc"],
                    "trusted_available_at_utc",
                )[1],
                row,
            )
            for row in exact
        ]
        latest_time = max(timestamp for timestamp, _ in exact_with_times)
        timed_finalists = [
            row for timestamp, row in exact_with_times if timestamp == latest_time
        ]
        if len(timed_finalists) == 1:
            selected = timed_finalists[0]
            precision_rule = "strict_prior_utc_instant"
    if selected is None:
        return (
            None,
            ["consensus_latest_vintage_ambiguous"],
            eligible_hashes,
            None,
        )
    if selected["analyst_count"] < MINIMUM_ANALYST_COUNT:
        return (
            selected,
            ["consensus_analyst_count_below_minimum"],
            eligible_hashes,
            precision_rule,
        )
    return selected, [], eligible_hashes, precision_rule


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
            "first_public_at_utc": None,
            "first_public_basis": None,
            "observed_public_by_at_utc": None,
            "canonical_actual": None,
        }
    return {
        "disposition": outcome["disposition"],
        "missing_reason": None,
        "source_kind": record["source_kind"],
        "accession_number": record["accession_number"],
        "exhibit_document_sha256": record["exhibit_document"]["raw_document"][
            "sha256"
        ],
        "edgar_acceptance_at_utc": record["edgar_acceptance_at_utc"],
        "first_public_at_utc": record["first_public_at_utc"],
        "first_public_basis": record["first_public_basis"],
        "observed_public_by_at_utc": record["observed_public_by_at_utc"],
        "canonical_actual": record["canonical_actual"],
    }


def _systemic_blockers(
    universe: Mapping[str, Any], consensus: Mapping[str, Any], announcement: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if not universe["payload"]["qualification_allowed"]:
        blockers.append("event_universe_not_qualified")
    blockers.extend(
        f"consensus_{code}"
        for code in consensus["payload"]["coverage"]["blockers"]
        if code not in _SYSTEMIC_CONSENSUS_EVENT_BLOCKERS
    )
    blockers.extend(
        f"announcement_{code}"
        for code in announcement["payload"]["coverage"]["blockers"]
        if code not in _SYSTEMIC_ANNOUNCEMENT_EVENT_BLOCKERS
    )
    if consensus["payload"]["evidence_class"] == "prospective_signal":
        frozen = _utc(universe["payload"]["frozen_at_utc"], "universe frozen_at_utc")[1]
        receipts = consensus["payload"]["acquisition_receipts"]
        if not receipts or any(
            _utc(row["source_captured_at_utc"], "receipt capture")[1] <= frozen
            for row in receipts
        ):
            blockers.append("prospective_universe_not_frozen_before_acquisition")
    return sorted(set(blockers))


def _event_result(
    *,
    event: Mapping[str, Any],
    consensus_record: Mapping[str, Any],
    announcement_outcome: Mapping[str, Any],
    systemic_blockers: Sequence[str],
    bindings: Mapping[str, str],
    prospective: bool,
    universe_frozen_at_utc: str,
    receipt_captured_at_by_hash: Mapping[str, str],
) -> dict[str, Any]:
    blockers = list(systemic_blockers)
    comparisons: list[dict[str, Any]] = [
        _comparison(
            "event_id",
            consensus_record["event_id"],
            announcement_outcome["event_id"],
            "match",
        )
    ]
    announcement_source = _announcement_source(announcement_outcome)
    selected: dict[str, Any] | None = None
    eligible_hashes: list[str] = []
    precision_rule: str | None = None

    if consensus_record["disposition"] != "available":
        blockers.append("consensus_event_missing")
    if announcement_outcome["disposition"] != "available":
        blockers.append("announcement_event_missing")
    first_public = announcement_source["first_public_at_utc"]
    if announcement_outcome["disposition"] == "available" and first_public is None:
        blockers.append("announcement_first_public_time_missing")

    if consensus_record["disposition"] == "available" and first_public is not None:
        selected, selection_blockers, eligible_hashes, precision_rule = (
            select_consensus_vintage(
                consensus_record["vintages"], first_public_at_utc=first_public
            )
        )
        blockers.extend(selection_blockers)
    if prospective and selected is not None and selected["availability_precision"] == "date":
        blockers.append("prospective_consensus_intraday_timestamp_missing")
    selected_receipt_captured_at: str | None = None
    if selected is not None:
        selected_receipt_captured_at = receipt_captured_at_by_hash.get(
            selected["acquisition_receipt_sha256"]
        )
        if selected_receipt_captured_at is None:
            blockers.append("consensus_receipt_capture_missing")
    if prospective and first_public is not None:
        first_public_dt = _utc(first_public, "first_public_at_utc")[1]
        frozen_dt = _utc(universe_frozen_at_utc, "universe frozen_at_utc")[1]
        if frozen_dt >= first_public_dt:
            blockers.append("prospective_universe_frozen_after_announcement")
        if selected is not None and selected_receipt_captured_at is not None:
            receipt_dt = _utc(
                selected_receipt_captured_at, "selected receipt capture"
            )[1]
            if receipt_dt >= first_public_dt:
                blockers.append("prospective_consensus_acquired_after_announcement")

    actual = announcement_source["canonical_actual"]
    if selected is not None and actual is not None:
        for field, consensus_field, announcement_field in _METRIC_COMPARISONS:
            left = selected["metric"][consensus_field]
            right = actual[announcement_field]
            matched = left == right
            comparisons.append(
                _comparison(field, left, right, "match" if matched else "mismatch")
            )
            if not matched:
                blockers.append(f"{field}_mismatch")
        comparisons.append(
            _comparison(
                "announcement_timing",
                selected["trusted_available_at_utc"]
                or selected["provider_as_of_date"],
                first_public,
                "strictly_before",
            )
        )
    else:
        for field, _, announcement_field in _METRIC_COMPARISONS:
            right = actual[announcement_field] if actual is not None else None
            comparisons.append(_comparison(field, None, right, "not_compared"))
        comparisons.append(
            _comparison("announcement_timing", None, first_public, "not_compared")
        )

    blockers = sorted(set(blockers))
    reconciled: dict[str, Any] | None = None
    if not blockers and selected is not None and actual is not None and first_public is not None:
        actual_value = _decimal(actual["canonical_value"], "canonical actual")
        consensus_value = _decimal(selected["consensus_value"], "consensus value")
        surprise = actual_value - consensus_value
        direction = "positive" if surprise > 0 else "negative" if surprise < 0 else "zero"
        reconciled = {
            "event_id": event["event_id"],
            "event_key": event["event_key"],
            "actual_value": actual["canonical_value"],
            "consensus_value": selected["consensus_value"],
            "raw_surprise": _canonical_decimal(surprise),
            "surprise_direction": direction,
            "analyst_count": selected["analyst_count"],
            "first_public_at_utc": first_public,
            "consensus_provider_as_of_date": selected["provider_as_of_date"],
            "consensus_available_at_utc": selected["trusted_available_at_utc"],
            "consensus_availability_precision": selected["availability_precision"],
            "consensus_receipt_captured_at_utc": selected_receipt_captured_at,
            "universe_frozen_at_utc": universe_frozen_at_utc,
            "metric": selected["metric"],
            "provenance": {
                "event_universe_sha256": bindings["event_universe_sha256"],
                "consensus_evidence_sha256": bindings["consensus_evidence_sha256"],
                "announcement_evidence_sha256": bindings[
                    "announcement_evidence_sha256"
                ],
                "consensus_raw_record_sha256": selected["raw_record_sha256"],
                "consensus_acquisition_receipt_sha256": selected[
                    "acquisition_receipt_sha256"
                ],
                "announcement_exhibit_document_sha256": announcement_source[
                    "exhibit_document_sha256"
                ],
                "announcement_normalization_evidence_sha256": actual[
                    "normalization_evidence_sha256"
                ],
            },
        }

    consensus_source = {
        "disposition": consensus_record["disposition"],
        "missing_reason": consensus_record["missing_reason"],
        "vintages": consensus_record["vintages"],
        "eligible_raw_record_sha256s": eligible_hashes,
        "selected_vintage": selected,
        "selected_receipt_captured_at_utc": selected_receipt_captured_at,
        "selection_precision_rule": precision_rule,
    }
    return {
        "event_id": event["event_id"],
        "event_key": event["event_key"],
        "disposition": "reconciled" if reconciled is not None else "excluded",
        "blockers": blockers,
        "source_values": {
            "announcement": announcement_source,
            "consensus": consensus_source,
        },
        "comparisons": sorted(comparisons, key=lambda item: item["field"]),
        "reconciled_event_input": reconciled,
    }


def build_pead_source_reconciliation(
    consensus_evidence: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
    *,
    reconciled_at_utc: str,
) -> dict[str, Any]:
    """Build a deterministic receipt from independently validated source lanes."""
    try:
        consensus = validate_pead_consensus_evidence(consensus_evidence)
    except PeadConsensusEvidenceError as exc:
        raise PeadSourceReconciliationError("consensus evidence is invalid") from exc
    universe = consensus["payload"]["event_universe"]
    try:
        announcement = validate_pead_announcement_evidence(
            announcement_evidence, expected_event_manifest=universe
        )
    except PeadAnnouncementEvidenceError as exc:
        raise PeadSourceReconciliationError("announcement evidence is invalid") from exc
    if consensus["payload"]["candidate_id"] != announcement["payload"]["candidate_id"]:
        raise PeadSourceReconciliationError("source evidence belongs to different candidates")

    reconciled_text, reconciled_at = _utc(reconciled_at_utc, "reconciled_at_utc")
    latest_input_time = max(
        _utc(consensus["payload"]["source"]["captured_at_utc"], "consensus capture")[1],
        _utc(announcement["payload"]["created_at_utc"], "announcement creation")[1],
    )
    if reconciled_at < latest_input_time:
        raise PeadSourceReconciliationError(
            "reconciliation timestamp precedes a source artifact"
        )

    universe_payload = universe["payload"]
    universe_bindings = universe_payload["bindings"]
    bindings = {
        "event_universe_sha256": universe["artifact_hash"],
        "consensus_evidence_sha256": consensus["artifact_hash"],
        "announcement_evidence_sha256": announcement["artifact_hash"],
        "market_snapshot_sha256": universe_bindings["market_snapshot_sha256"],
        "identity_snapshot_sha256": universe_bindings["identity_snapshot_sha256"],
        "candidate_specification_sha256": universe_bindings[
            "candidate_specification_sha256"
        ],
        "construction_code_sha256": universe_bindings["construction_code_sha256"],
        "consensus_source_manifest_sha256": consensus["payload"]["source"][
            "source_manifest_sha256"
        ],
        "reconciliation_policy_sha256": content_hash(_POLICY),
    }
    systemic = _systemic_blockers(universe, consensus, announcement)
    consensus_by_id = {
        row["event_id"]: row for row in consensus["payload"]["event_records"]
    }
    announcement_by_id = {
        row["event_id"]: row for row in announcement["payload"]["outcomes"]
    }
    receipt_captured_at_by_hash = {
        row["receipt_sha256"]: row["source_captured_at_utc"]
        for row in consensus["payload"]["acquisition_receipts"]
    }
    prospective = consensus["payload"]["evidence_class"] == "prospective_signal"
    results = [
        _event_result(
            event=event,
            consensus_record=consensus_by_id[event["event_id"]],
            announcement_outcome=announcement_by_id[event["event_id"]],
            systemic_blockers=systemic,
            bindings=bindings,
            prospective=prospective,
            universe_frozen_at_utc=universe_payload["frozen_at_utc"],
            receipt_captured_at_by_hash=receipt_captured_at_by_hash,
        )
        for event in universe_payload["expected_events"]
    ]
    reconciled_count = sum(row["disposition"] == "reconciled" for row in results)
    blocker_counts: dict[str, int] = {}
    for result in results:
        for blocker in result["blockers"]:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    has_reconciled = bool(reconciled_count and not systemic)
    payload = {
        "schema_version": SOURCE_RECONCILIATION_SCHEMA_VERSION,
        "candidate_id": consensus["payload"]["candidate_id"],
        "reconciled_at_utc": reconciled_text,
        "policy": _POLICY,
        "bindings": bindings,
        "event_results": results,
        "coverage": {
            "expected_event_count": len(results),
            "reconciled_event_count": reconciled_count,
            "excluded_event_count": len(results) - reconciled_count,
            "reconciliation_complete": len(results) == len(
                universe_payload["expected_event_ids"]
            ),
            "systemic_blockers": systemic,
            "event_blocker_counts": {
                key: blocker_counts[key] for key in sorted(blocker_counts)
            },
        },
        "qualification": {
            "has_reconciled_event_inputs": has_reconciled,
            "all_expected_events_reconciled": bool(
                results and reconciled_count == len(results) and not systemic
            ),
            "source_qualified_event_inputs_allowed": False,
            "raw_consensus_normalization_replay_required": True,
            "external_binding_replay_required": True,
            "market_accounting_join_required": True,
            "historical_replication_allowed": False,
            "edge_claim_allowed": False,
            "paper_execution_allowed": False,
            "live_deployment_allowed": False,
        },
    }
    document = {"artifact_hash": content_hash(payload), "payload": payload}
    return _plain(document)


def validate_pead_source_reconciliation(
    document: Mapping[str, Any],
    *,
    consensus_evidence: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the reconciliation and require exact equality with the receipt."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "source reconciliation")
    payload = _exact(
        wrapper["payload"], _PAYLOAD_FIELDS, "source reconciliation.payload"
    )
    claimed = _sha(wrapper["artifact_hash"], "source reconciliation.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadSourceReconciliationError(
            "source reconciliation artifact hash mismatch"
        )
    if payload["schema_version"] != SOURCE_RECONCILIATION_SCHEMA_VERSION:
        raise PeadSourceReconciliationError("unsupported source reconciliation schema")
    reconciled_at = _utc(payload["reconciled_at_utc"], "reconciled_at_utc")[0]
    expected = build_pead_source_reconciliation(
        consensus_evidence,
        announcement_evidence,
        reconciled_at_utc=reconciled_at,
    )
    if _plain(document) != expected:
        raise PeadSourceReconciliationError(
            "source reconciliation does not replay from its bound evidence"
        )
    return expected


def verify_pead_source_reconciliation(
    document: Mapping[str, Any],
    *,
    consensus_evidence: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicit verifier alias used by downstream evidence boundaries."""
    return validate_pead_source_reconciliation(
        document,
        consensus_evidence=consensus_evidence,
        announcement_evidence=announcement_evidence,
    )


def _strict_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadSourceReconciliationError(
            f"source reconciliation is not a regular file: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_SOURCE_RECONCILIATION_BYTES:
        raise PeadSourceReconciliationError(
            "source reconciliation exceeds its size limit"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadSourceReconciliationError(
            "source reconciliation is not UTF-8"
        ) from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadSourceReconciliationError(
                    f"source reconciliation contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadSourceReconciliationError(
            f"source reconciliation contains invalid number {token}"
        )

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadSourceReconciliationError(
            f"invalid source reconciliation JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadSourceReconciliationError(
            "source reconciliation root must be an object"
        )
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadSourceReconciliationError(
            "source reconciliation bytes are not canonical JSON plus one newline"
        )
    return value


def load_pead_source_reconciliation(
    path: str | Path,
    *,
    consensus_evidence: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Load strict JSON and replay it from the original source artifacts."""
    return validate_pead_source_reconciliation(
        _strict_json_file(Path(path)),
        consensus_evidence=consensus_evidence,
        announcement_evidence=announcement_evidence,
    )


__all__ = [
    "MAX_SOURCE_RECONCILIATION_BYTES",
    "MINIMUM_ANALYST_COUNT",
    "PeadSourceReconciliationError",
    "SOURCE_RECONCILIATION_POLICY_SCHEMA_VERSION",
    "SOURCE_RECONCILIATION_SCHEMA_VERSION",
    "build_pead_source_reconciliation",
    "load_pead_source_reconciliation",
    "select_consensus_vintage",
    "validate_pead_source_reconciliation",
    "verify_pead_source_reconciliation",
]
