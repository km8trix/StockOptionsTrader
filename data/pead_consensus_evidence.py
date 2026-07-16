"""Provider-neutral point-in-time consensus-vintage evidence for PEAD.

This contract preserves every acquired vintage and deliberately performs no
latest-vintage selection.  Exhaustiveness is measured against one separately
frozen :mod:`data.pead_event_universe` wrapper, never against a date envelope or
a provider completeness flag.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from data.pead_event_universe import (
    PeadEventUniverseError,
    validate_pead_event_universe,
)


CONSENSUS_EVIDENCE_SCHEMA_VERSION = "pead_consensus_evidence.v1"
MAX_CONSENSUS_EVIDENCE_BYTES = 512 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_CANONICAL_DECIMAL_PATTERN = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$"
)
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version", "candidate_id", "evidence_class", "event_universe",
    "source", "acquisition_receipts", "event_records", "coverage",
}
_SOURCE_FIELDS = {
    "provider_id", "dataset_id", "source_manifest_sha256", "captured_at_utc",
    "provider_snapshot_at_utc",
}
_RECEIPT_FIELDS = {
    "receipt_sha256", "source_captured_at_utc", "query_scope", "pagination",
}
_QUERY_FIELDS = {"scope_kind", "canonical_query_sha256", "expected_event_ids"}
_PAGINATION_FIELDS = {"mode", "terminal_page_observed", "page_count", "pages"}
_PAGE_FIELDS = {
    "sequence", "request_sha256", "raw_response_sha256", "raw_response_bytes",
    "continuation_token_sha256",
}
_EVENT_FIELDS = {"event_id", "disposition", "missing_reason", "vintages"}
_VINTAGE_FIELDS = {
    "provider_as_of_date", "trusted_available_at_utc", "availability_precision",
    "consensus_value", "analyst_count", "raw_record_sha256",
    "acquisition_receipt_sha256", "metric",
}
_METRIC_FIELDS = {
    "metric_id", "accounting_basis", "per_share_basis", "scope",
    "canonical_share_basis", "currency_code", "unit", "metric_definition_sha256",
}
_COVERAGE_FIELDS = {
    "expected_event_count", "available_event_count", "missing_event_count",
    "query_scoped_event_count", "vintage_count", "pagination_complete",
    "blockers", "qualification_allowed",
}
_EVIDENCE_CLASSES = {
    "development_sample", "historical_reconstruction", "prospective_signal",
}
_QUALIFYING_CLASSES = {"historical_reconstruction", "prospective_signal"}
_SCOPE_KINDS = {"full_export", "expected_event_partition"}
_PAGINATION_MODES = {"single_response", "cursor", "page_number", "bulk_file"}
_AVAILABILITY_PRECISIONS = {"date", "second", "microsecond"}


class PeadConsensusEvidenceError(ValueError):
    """Consensus evidence is malformed, incomplete, or self-inconsistent."""


def canonical_json(value: Any) -> str:
    """Return deterministic finite JSON for consensus evidence identities."""
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PeadConsensusEvidenceError("consensus evidence is non-finite")
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PeadConsensusEvidenceError("consensus evidence keys must be strings")
                result[key] = normalize(child)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise PeadConsensusEvidenceError(
            f"unsupported consensus evidence value: {type(item).__name__}"
        )
    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadConsensusEvidenceError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadConsensusEvidenceError(f"{label} must be nonempty canonical text")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadConsensusEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _day(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PeadConsensusEvidenceError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadConsensusEvidenceError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadConsensusEvidenceError(f"{label} must be canonical YYYY-MM-DD")
    return value


def _canonical_decimal(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or _CANONICAL_DECIMAL_PATTERN.fullmatch(value) is None
        or value == "-0"
    ):
        raise PeadConsensusEvidenceError(
            f"{label} must be a canonical finite decimal string"
        )
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadConsensusEvidenceError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadConsensusEvidenceError(f"{label} must be canonical UTC with Z") from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if value != canonical:
        raise PeadConsensusEvidenceError(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _normalize_source(value: Any) -> tuple[dict[str, Any], datetime]:
    source = _exact(value, _SOURCE_FIELDS, "source")
    captured, captured_dt = _utc(source["captured_at_utc"], "source.captured_at_utc")
    snapshot = source["provider_snapshot_at_utc"]
    if snapshot is not None:
        snapshot, snapshot_dt = _utc(snapshot, "source.provider_snapshot_at_utc")
        if snapshot_dt > captured_dt:
            raise PeadConsensusEvidenceError("provider snapshot follows source capture")
    return {
        "provider_id": _text(source["provider_id"], "source.provider_id"),
        "dataset_id": _text(source["dataset_id"], "source.dataset_id"),
        "source_manifest_sha256": _sha(
            source["source_manifest_sha256"], "source.source_manifest_sha256"
        ),
        "captured_at_utc": captured,
        "provider_snapshot_at_utc": snapshot,
    }, captured_dt


def _normalize_receipts(
    value: Any, *, expected_ids: set[str], source_captured: datetime,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PeadConsensusEvidenceError("acquisition_receipts must be an array")
    receipts: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        receipt = _exact(raw, _RECEIPT_FIELDS, f"acquisition_receipts[{index}]")
        receipt_hash = _sha(receipt["receipt_sha256"], f"receipt[{index}].receipt_sha256")
        captured, captured_dt = _utc(
            receipt["source_captured_at_utc"], f"receipt[{index}].source_captured_at_utc"
        )
        if captured_dt > source_captured:
            raise PeadConsensusEvidenceError("receipt capture follows source capture")
        scope = _exact(receipt["query_scope"], _QUERY_FIELDS, f"receipt[{index}].query_scope")
        if scope["scope_kind"] not in _SCOPE_KINDS:
            raise PeadConsensusEvidenceError("query scope kind is unsupported")
        scope_ids = scope["expected_event_ids"]
        if not isinstance(scope_ids, list):
            raise PeadConsensusEvidenceError("query expected_event_ids must be an array")
        normalized_scope_ids = [
            _sha(item, f"receipt[{index}].query_scope.expected_event_id")
            for item in scope_ids
        ]
        if normalized_scope_ids != sorted(set(normalized_scope_ids)):
            raise PeadConsensusEvidenceError("query expected_event_ids must be sorted and unique")
        if any(item not in expected_ids for item in normalized_scope_ids):
            raise PeadConsensusEvidenceError("query scope references an event outside the universe")
        pagination = _exact(
            receipt["pagination"], _PAGINATION_FIELDS, f"receipt[{index}].pagination"
        )
        if pagination["mode"] not in _PAGINATION_MODES:
            raise PeadConsensusEvidenceError("pagination mode is unsupported")
        if type(pagination["terminal_page_observed"]) is not bool:
            raise PeadConsensusEvidenceError("terminal_page_observed must be boolean")
        pages = pagination["pages"]
        if not isinstance(pages, list) or not pages:
            raise PeadConsensusEvidenceError("pagination pages must be nonempty")
        normalized_pages: list[dict[str, Any]] = []
        for page_index, raw_page in enumerate(pages, start=1):
            page = _exact(raw_page, _PAGE_FIELDS, f"receipt[{index}].pages[{page_index}]")
            if page["sequence"] != page_index:
                raise PeadConsensusEvidenceError("pagination page sequence is invalid")
            byte_count = page["raw_response_bytes"]
            if type(byte_count) is not int or byte_count < 0:
                raise PeadConsensusEvidenceError("raw_response_bytes must be nonnegative integer")
            token = page["continuation_token_sha256"]
            if token is not None:
                token = _sha(token, "continuation_token_sha256")
            normalized_pages.append({
                "sequence": page_index,
                "request_sha256": _sha(page["request_sha256"], "request_sha256"),
                "raw_response_sha256": _sha(
                    page["raw_response_sha256"], "raw_response_sha256"
                ),
                "raw_response_bytes": byte_count,
                "continuation_token_sha256": token,
            })
        if type(pagination["page_count"]) is not int or pagination["page_count"] != len(pages):
            raise PeadConsensusEvidenceError("pagination page_count mismatch")
        if pagination["terminal_page_observed"] and normalized_pages[-1][
            "continuation_token_sha256"
        ] is not None:
            raise PeadConsensusEvidenceError("terminal page cannot carry a continuation token")
        if pagination["mode"] in {"single_response", "bulk_file"} and len(pages) != 1:
            raise PeadConsensusEvidenceError("single/bulk acquisition must contain one page")
        receipt_body = {
            "source_captured_at_utc": captured,
            "query_scope": {
                "scope_kind": scope["scope_kind"],
                "canonical_query_sha256": _sha(
                    scope["canonical_query_sha256"], "canonical_query_sha256"
                ),
                "expected_event_ids": normalized_scope_ids,
            },
            "pagination": {
                "mode": pagination["mode"],
                "terminal_page_observed": pagination["terminal_page_observed"],
                "page_count": len(normalized_pages),
                "pages": normalized_pages,
            },
        }
        if content_hash(receipt_body) != receipt_hash:
            raise PeadConsensusEvidenceError(
                "acquisition receipt hash does not match its canonical body"
            )
        receipts.append({"receipt_sha256": receipt_hash, **receipt_body})
    receipts.sort(key=lambda row: row["receipt_sha256"])
    hashes = [row["receipt_sha256"] for row in receipts]
    if len(hashes) != len(set(hashes)):
        raise PeadConsensusEvidenceError("acquisition receipt hashes must be unique")
    return receipts


def _normalize_metric(value: Any, label: str) -> dict[str, str]:
    metric = _exact(value, _METRIC_FIELDS, label)
    currency = metric["currency_code"]
    if not isinstance(currency, str) or _CURRENCY_PATTERN.fullmatch(currency) is None:
        raise PeadConsensusEvidenceError(f"{label}.currency_code must be ISO-style uppercase")
    return {
        "metric_id": _text(metric["metric_id"], f"{label}.metric_id"),
        "accounting_basis": _text(metric["accounting_basis"], f"{label}.accounting_basis"),
        "per_share_basis": _text(metric["per_share_basis"], f"{label}.per_share_basis"),
        "scope": _text(metric["scope"], f"{label}.scope"),
        "canonical_share_basis": _text(
            metric["canonical_share_basis"], f"{label}.canonical_share_basis"
        ),
        "currency_code": currency,
        "unit": _text(metric["unit"], f"{label}.unit"),
        "metric_definition_sha256": _sha(
            metric["metric_definition_sha256"],
            f"{label}.metric_definition_sha256",
        ),
    }


def _normalize_vintage(
    value: Any, label: str, receipt_captures: Mapping[str, datetime]
) -> dict[str, Any]:
    row = _exact(value, _VINTAGE_FIELDS, label)
    as_of = _day(row["provider_as_of_date"], f"{label}.provider_as_of_date")
    precision = row["availability_precision"]
    if precision not in _AVAILABILITY_PRECISIONS:
        raise PeadConsensusEvidenceError(f"{label}.availability_precision is unsupported")
    available = row["trusted_available_at_utc"]
    available_dt: datetime | None = None
    if available is None:
        if precision != "date":
            raise PeadConsensusEvidenceError("missing availability timestamp requires date precision")
    else:
        available, available_dt = _utc(available, f"{label}.trusted_available_at_utc")
        actual_precision = "microsecond" if available_dt.microsecond else "second"
        if precision != actual_precision:
            raise PeadConsensusEvidenceError("availability precision contradicts timestamp")
    consensus = _canonical_decimal(
        row["consensus_value"], f"{label}.consensus_value"
    )
    analyst_count = row["analyst_count"]
    if type(analyst_count) is not int or analyst_count < 1:
        raise PeadConsensusEvidenceError(f"{label}.analyst_count must be positive integer")
    receipt_hash = _sha(
        row["acquisition_receipt_sha256"], f"{label}.acquisition_receipt_sha256"
    )
    if receipt_hash not in receipt_captures:
        raise PeadConsensusEvidenceError("vintage references an unknown acquisition receipt")
    if available_dt is not None and available_dt > receipt_captures[receipt_hash]:
        raise PeadConsensusEvidenceError(
            "trusted availability timestamp follows its acquisition receipt capture"
        )
    return {
        "provider_as_of_date": as_of,
        "trusted_available_at_utc": available,
        "availability_precision": precision,
        "consensus_value": consensus,
        "analyst_count": analyst_count,
        "raw_record_sha256": _sha(row["raw_record_sha256"], f"{label}.raw_record_sha256"),
        "acquisition_receipt_sha256": receipt_hash,
        "metric": _normalize_metric(row["metric"], f"{label}.metric"),
    }


def _normalize_records(
    value: Any, *, expected_ids: list[str],
    receipt_captures: Mapping[str, datetime],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PeadConsensusEvidenceError("event_records must be an array")
    records: list[dict[str, Any]] = []
    raw_hashes: set[str] = set()
    for index, raw in enumerate(value):
        row = _exact(raw, _EVENT_FIELDS, f"event_records[{index}]")
        event_id = _sha(row["event_id"], f"event_records[{index}].event_id")
        disposition = row["disposition"]
        if disposition not in {"available", "missing"}:
            raise PeadConsensusEvidenceError("event disposition must be available or missing")
        vintages_raw = row["vintages"]
        if not isinstance(vintages_raw, list):
            raise PeadConsensusEvidenceError("event vintages must be an array")
        vintages = [
            _normalize_vintage(
                item,
                f"event_records[{index}].vintages[{vintage_index}]",
                receipt_captures,
            )
            for vintage_index, item in enumerate(vintages_raw)
        ]
        vintages.sort(key=lambda item: (
            item["provider_as_of_date"], item["trusted_available_at_utc"] or "",
            item["raw_record_sha256"],
        ))
        for vintage in vintages:
            raw_hash = vintage["raw_record_sha256"]
            if raw_hash in raw_hashes:
                raise PeadConsensusEvidenceError("raw consensus records must be unique")
            raw_hashes.add(raw_hash)
        missing_reason = row["missing_reason"]
        if disposition == "available":
            if not vintages or missing_reason is not None:
                raise PeadConsensusEvidenceError("available events require vintages and no missing reason")
        else:
            if vintages:
                raise PeadConsensusEvidenceError("missing events cannot contain vintages")
            missing_reason = _text(missing_reason, f"event_records[{index}].missing_reason")
        records.append({
            "event_id": event_id, "disposition": disposition,
            "missing_reason": missing_reason, "vintages": vintages,
        })
    records.sort(key=lambda row: row["event_id"])
    record_ids = [row["event_id"] for row in records]
    if record_ids != expected_ids:
        raise PeadConsensusEvidenceError(
            "event_records must account for every frozen expected event exactly once"
        )
    return records


def _coverage(
    *, evidence_class: str, universe: Mapping[str, Any], source: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = universe["payload"]["expected_event_ids"]
    scoped = Counter(
        event_id
        for receipt in receipts
        for event_id in receipt["query_scope"]["expected_event_ids"]
    )
    exactly_scoped = sum(scoped[event_id] == 1 for event_id in expected_ids)
    available = sum(row["disposition"] == "available" for row in records)
    missing = len(records) - available
    pagination_complete = bool(receipts) and all(
        receipt["pagination"]["terminal_page_observed"] for receipt in receipts
    )
    blockers: list[str] = []
    if universe["payload"]["qualification_allowed"] is not True:
        blockers.append("event_universe_not_qualified")
    if evidence_class not in _QUALIFYING_CLASSES:
        blockers.append("evidence_class_not_qualifying")
    if source["provider_snapshot_at_utc"] is None:
        blockers.append("provider_snapshot_timestamp_missing")
    if not receipts:
        blockers.append("acquisition_receipts_missing")
    if exactly_scoped != len(expected_ids) or len(scoped) != len(expected_ids):
        blockers.append("query_scope_not_exhaustive_once")
    if not pagination_complete:
        blockers.append("pagination_not_complete")
    if missing:
        blockers.append("expected_events_missing_consensus")
    blockers = sorted(set(blockers))
    return {
        "expected_event_count": len(expected_ids),
        "available_event_count": available,
        "missing_event_count": missing,
        "query_scoped_event_count": exactly_scoped,
        "vintage_count": sum(len(row["vintages"]) for row in records),
        "pagination_complete": pagination_complete,
        "blockers": blockers,
        "qualification_allowed": not blockers,
    }


def build_pead_consensus_evidence(
    *, candidate_id: str, evidence_class: str, event_universe: Mapping[str, Any],
    source: Mapping[str, Any], acquisition_receipts: Sequence[Mapping[str, Any]],
    event_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an exact consensus artifact without selecting a vintage."""
    try:
        universe = validate_pead_event_universe(event_universe)
    except PeadEventUniverseError as exc:
        raise PeadConsensusEvidenceError("event universe is invalid") from exc
    candidate = _text(candidate_id, "candidate_id")
    if candidate != universe["payload"]["candidate_id"]:
        raise PeadConsensusEvidenceError("consensus evidence belongs to another candidate")
    if evidence_class not in _EVIDENCE_CLASSES:
        raise PeadConsensusEvidenceError("evidence_class is unsupported")
    normalized_source, captured = _normalize_source(source)
    expected_ids = universe["payload"]["expected_event_ids"]
    receipts = _normalize_receipts(
        list(acquisition_receipts), expected_ids=set(expected_ids), source_captured=captured
    )
    records = _normalize_records(
        list(event_records), expected_ids=expected_ids,
        receipt_captures={
            row["receipt_sha256"]: _utc(
                row["source_captured_at_utc"], "receipt capture"
            )[1]
            for row in receipts
        },
    )
    coverage = _coverage(
        evidence_class=evidence_class, universe=universe, source=normalized_source,
        receipts=receipts, records=records,
    )
    payload = {
        "schema_version": CONSENSUS_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": candidate, "evidence_class": evidence_class,
        "event_universe": universe, "source": normalized_source,
        "acquisition_receipts": receipts, "event_records": records,
        "coverage": coverage,
    }
    return validate_pead_consensus_evidence(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_consensus_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate content identity, exhaustive accounting, provenance, and status."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "consensus evidence")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "consensus evidence.payload")
    claimed = _sha(wrapper["artifact_hash"], "consensus evidence.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadConsensusEvidenceError("consensus evidence artifact hash mismatch")
    if payload["schema_version"] != CONSENSUS_EVIDENCE_SCHEMA_VERSION:
        raise PeadConsensusEvidenceError("unsupported consensus evidence schema")
    try:
        universe = validate_pead_event_universe(payload["event_universe"])
    except PeadEventUniverseError as exc:
        raise PeadConsensusEvidenceError("event universe is invalid") from exc
    candidate = _text(payload["candidate_id"], "candidate_id")
    if candidate != universe["payload"]["candidate_id"]:
        raise PeadConsensusEvidenceError("consensus evidence belongs to another candidate")
    evidence_class = payload["evidence_class"]
    if evidence_class not in _EVIDENCE_CLASSES:
        raise PeadConsensusEvidenceError("evidence_class is unsupported")
    source, captured = _normalize_source(payload["source"])
    expected_ids = universe["payload"]["expected_event_ids"]
    receipts = _normalize_receipts(
        payload["acquisition_receipts"], expected_ids=set(expected_ids),
        source_captured=captured,
    )
    if payload["acquisition_receipts"] != receipts:
        raise PeadConsensusEvidenceError("acquisition_receipts must be canonically sorted")
    records = _normalize_records(
        payload["event_records"], expected_ids=expected_ids,
        receipt_captures={
            row["receipt_sha256"]: _utc(
                row["source_captured_at_utc"], "receipt capture"
            )[1]
            for row in receipts
        },
    )
    if payload["event_records"] != records:
        raise PeadConsensusEvidenceError("event_records or vintages are not canonically sorted")
    coverage = _coverage(
        evidence_class=evidence_class, universe=universe, source=source,
        receipts=receipts, records=records,
    )
    _exact(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    if payload["coverage"] != coverage:
        raise PeadConsensusEvidenceError("coverage and qualification must be derived exactly")
    normalized = {
        "schema_version": CONSENSUS_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": candidate, "evidence_class": evidence_class,
        "event_universe": universe, "source": source,
        "acquisition_receipts": receipts, "event_records": records,
        "coverage": coverage,
    }
    if content_hash(normalized) != claimed:
        raise PeadConsensusEvidenceError("consensus evidence is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized)}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadConsensusEvidenceError(f"consensus evidence is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_CONSENSUS_EVIDENCE_BYTES:
        raise PeadConsensusEvidenceError("consensus evidence exceeds its size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadConsensusEvidenceError("consensus evidence is not UTF-8") from exc
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadConsensusEvidenceError(
                    f"consensus evidence contains duplicate key {key!r}"
                )
            result[key] = value
        return result
    def reject(token: str) -> None:
        raise PeadConsensusEvidenceError(f"consensus evidence contains invalid number {token}")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadConsensusEvidenceError(
            f"invalid consensus evidence JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadConsensusEvidenceError("consensus evidence root must be an object")
    return value


def load_pead_consensus_evidence(path: str | Path) -> dict[str, Any]:
    return validate_pead_consensus_evidence(_read(Path(path)))


__all__ = [
    "CONSENSUS_EVIDENCE_SCHEMA_VERSION", "MAX_CONSENSUS_EVIDENCE_BYTES",
    "PeadConsensusEvidenceError", "build_pead_consensus_evidence", "canonical_json",
    "content_hash", "load_pead_consensus_evidence", "validate_pead_consensus_evidence",
]
