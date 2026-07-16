"""Provider-independent frozen event census for PEAD evidence.

Announcement and consensus evidence bind this exact wrapper.  The expected set
is derived from an exhaustive disposition of a content-addressed source census;
neither evidence lane may redefine it or infer completeness from date bounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


EVENT_CENSUS_RECEIPT_SCHEMA_VERSION = "pead_event_census_receipt.v1"
EVENT_UNIVERSE_SCHEMA_VERSION = "pead_event_universe.v1"
EVENT_UNIVERSE_V2_SCHEMA_VERSION = "pead_event_universe.v2"
MAX_EVENT_UNIVERSE_BYTES = 64 * 1024 * 1024
EVENT_KEY_FIELDS = frozenset({"cik", "fiscal_period_end", "fiscal_period_type"})

_HEX = frozenset("0123456789abcdef")
_CIK_PATTERN = re.compile(r"^[0-9]{10}$")
_MACHINE_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_CENSUS_FIELDS = {
    "schema_version",
    "raw_census_artifact_sha256",
    "canonical_query_sha256",
    "source_record_ids",
    "source_record_count",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "frozen_at_utc",
    "event_window",
    "bindings",
    "census_receipt",
    "census_dispositions",
    "census_counts",
    "expected_events",
    "expected_event_ids",
    "identity_gaps",
    "blockers",
    "qualification_allowed",
}
_WINDOW_FIELDS = {"start", "end"}
_BINDING_FIELDS = {
    "market_snapshot_sha256",
    "identity_snapshot_sha256",
    "candidate_specification_sha256",
    "construction_code_sha256",
    "canonical_query_sha256",
}
_DISPOSITION_FIELDS = {
    "source_record_id", "disposition", "event_id", "event_key", "reason"
}
_EXPECTED_EVENT_FIELDS = {"event_id", "event_key"}
_IDENTITY_GAP_FIELDS = {"source_record_id", "reason"}
_COUNT_FIELDS = {
    "source_record_count",
    "disposition_count",
    "expected_event_count",
    "identity_gap_count",
    "out_of_scope_count",
    "census_complete",
}
_DISPOSITIONS = {"expected_event", "identity_gap", "out_of_scope"}


class PeadEventUniverseError(ValueError):
    """The frozen event universe is malformed or self-inconsistent."""


def canonical_json(value: Any) -> str:
    """Return deterministic finite JSON for all event-universe hashes."""
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PeadEventUniverseError("event-universe evidence is non-finite")
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PeadEventUniverseError("event-universe keys must be strings")
                result[key] = normalize(child)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise PeadEventUniverseError(
            f"unsupported event-universe value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _plain_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str] | frozenset[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadEventUniverseError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadEventUniverseError(f"{label} must be nonempty canonical text")
    return value


def _machine_reason(value: Any, label: str) -> str:
    reason = _text(value, label)
    if _MACHINE_REASON_PATTERN.fullmatch(reason) is None:
        raise PeadEventUniverseError(f"{label} must be a lowercase machine reason")
    return reason


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadEventUniverseError(f"{label} must be a lowercase SHA-256")
    return value


def _day(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PeadEventUniverseError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadEventUniverseError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadEventUniverseError(f"{label} must be canonical YYYY-MM-DD")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadEventUniverseError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeadEventUniverseError(f"{label} must be canonical UTC with Z") from exc
    utc = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if utc.microsecond else "seconds"
    if value != utc.isoformat(timespec=timespec).replace("+00:00", "Z"):
        raise PeadEventUniverseError(f"{label} must be canonical UTC with Z")
    return value


def validate_event_key(value: Any, *, label: str = "event_key") -> dict[str, str]:
    key = _exact(value, EVENT_KEY_FIELDS, label)
    cik = key["cik"]
    if (
        not isinstance(cik, str) or _CIK_PATTERN.fullmatch(cik) is None
        or cik == "0000000000"
    ):
        raise PeadEventUniverseError(f"{label}.cik must be a positive 10-digit CIK")
    period = _day(key["fiscal_period_end"], f"{label}.fiscal_period_end")
    if key["fiscal_period_type"] != "Q":
        raise PeadEventUniverseError(f"{label}.fiscal_period_type must be 'Q'")
    return {"cik": cik, "fiscal_period_end": period, "fiscal_period_type": "Q"}


def canonical_event_id(event_key: Mapping[str, Any]) -> str:
    return content_hash(validate_event_key(event_key))


def build_pead_event_census_receipt(
    *, raw_census_artifact_sha256: str, canonical_query_sha256: str,
    source_record_ids: Sequence[str],
) -> dict[str, Any]:
    """Bind the exact raw census and its exhaustive row-identity list."""
    if isinstance(source_record_ids, (str, bytes)) or not isinstance(
        source_record_ids, Sequence
    ):
        raise PeadEventUniverseError("source_record_ids must be a sequence")
    ids = sorted(_sha(item, "source_record_id") for item in source_record_ids)
    if len(ids) != len(set(ids)):
        raise PeadEventUniverseError("source_record_ids must be unique")
    payload = {
        "schema_version": EVENT_CENSUS_RECEIPT_SCHEMA_VERSION,
        "raw_census_artifact_sha256": _sha(
            raw_census_artifact_sha256, "raw_census_artifact_sha256"
        ),
        "canonical_query_sha256": _sha(
            canonical_query_sha256, "canonical_query_sha256"
        ),
        "source_record_ids": ids,
        "source_record_count": len(ids),
    }
    return validate_pead_event_census_receipt(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_event_census_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "event census receipt")
    payload = _exact(wrapper["payload"], _CENSUS_FIELDS, "event census receipt.payload")
    claimed = _sha(wrapper["artifact_hash"], "event census receipt.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadEventUniverseError("event census receipt artifact hash mismatch")
    if payload["schema_version"] != EVENT_CENSUS_RECEIPT_SCHEMA_VERSION:
        raise PeadEventUniverseError("unsupported event census receipt schema")
    ids = payload["source_record_ids"]
    if not isinstance(ids, list):
        raise PeadEventUniverseError("source_record_ids must be an array")
    normalized_ids = [_sha(item, "source_record_id") for item in ids]
    if normalized_ids != sorted(set(normalized_ids)):
        raise PeadEventUniverseError("source_record_ids must be sorted and unique")
    if type(payload["source_record_count"]) is not int or (
        payload["source_record_count"] != len(normalized_ids)
    ):
        raise PeadEventUniverseError("source_record_count does not match the census")
    normalized = {
        "schema_version": EVENT_CENSUS_RECEIPT_SCHEMA_VERSION,
        "raw_census_artifact_sha256": _sha(
            payload["raw_census_artifact_sha256"], "raw_census_artifact_sha256"
        ),
        "canonical_query_sha256": _sha(
            payload["canonical_query_sha256"], "canonical_query_sha256"
        ),
        "source_record_ids": normalized_ids,
        "source_record_count": len(normalized_ids),
    }
    if content_hash(normalized) != claimed:
        raise PeadEventUniverseError("event census receipt is not canonical")
    return {"artifact_hash": claimed, "payload": normalized}


def _normalize_dispositions(
    raw_rows: Sequence[Mapping[str, Any]], *, census_ids: list[str],
    start: str, end: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raise PeadEventUniverseError("census_dispositions must be a sequence")
    rows: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    gaps: list[dict[str, str]] = []
    for index, raw in enumerate(raw_rows):
        row = _exact(raw, _DISPOSITION_FIELDS, f"census_dispositions[{index}]")
        source_id = _sha(row["source_record_id"], f"census_dispositions[{index}].source_record_id")
        disposition = row["disposition"]
        if disposition not in _DISPOSITIONS:
            raise PeadEventUniverseError("census disposition is unsupported")
        if disposition == "expected_event":
            event_key = validate_event_key(
                row["event_key"], label=f"census_dispositions[{index}].event_key"
            )
            if not start <= event_key["fiscal_period_end"] <= end:
                raise PeadEventUniverseError("expected event period falls outside event_window")
            event_id = _sha(row["event_id"], f"census_dispositions[{index}].event_id")
            if event_id != canonical_event_id(event_key):
                raise PeadEventUniverseError("census event_id does not match event_key")
            if row["reason"] is not None:
                raise PeadEventUniverseError("expected_event disposition cannot have a reason")
            normalized = {
                "source_record_id": source_id, "disposition": disposition,
                "event_id": event_id, "event_key": event_key, "reason": None,
            }
            expected.append({"event_id": event_id, "event_key": event_key})
        else:
            if row["event_id"] is not None or row["event_key"] is not None:
                raise PeadEventUniverseError(
                    "only expected_event dispositions may carry event identity"
                )
            reason = _machine_reason(
                row["reason"], f"census_dispositions[{index}].reason"
            )
            normalized = {
                "source_record_id": source_id, "disposition": disposition,
                "event_id": None, "event_key": None, "reason": reason,
            }
            if disposition == "identity_gap":
                gaps.append({"source_record_id": source_id, "reason": reason})
        rows.append(normalized)
    rows.sort(key=lambda row: row["source_record_id"])
    row_ids = [row["source_record_id"] for row in rows]
    if row_ids != census_ids:
        raise PeadEventUniverseError(
            "census dispositions must account for every receipt record exactly once"
        )
    expected.sort(key=lambda row: row["event_id"])
    if len({row["event_id"] for row in expected}) != len(expected):
        raise PeadEventUniverseError("expected event identities must be unique")
    gaps.sort(key=lambda row: (row["source_record_id"], row["reason"]))
    return rows, expected, gaps


def _status(
    *, source_count: int, dispositions: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, str]],
    identity_gaps_block: bool = True,
) -> tuple[dict[str, Any], list[str], bool]:
    out_count = sum(row["disposition"] == "out_of_scope" for row in dispositions)
    complete = len(dispositions) == source_count
    counts = {
        "source_record_count": source_count,
        "disposition_count": len(dispositions),
        "expected_event_count": len(expected),
        "identity_gap_count": len(gaps),
        "out_of_scope_count": out_count,
        "census_complete": complete,
    }
    blockers: list[str] = []
    if source_count == 0:
        blockers.append("source_census_empty")
    if not complete:
        blockers.append("source_census_disposition_incomplete")
    if not expected:
        blockers.append("expected_event_manifest_empty")
    if gaps and identity_gaps_block:
        blockers.append("identity_gaps_present")
    return counts, blockers, not blockers


def _build_pead_event_universe(
    *, candidate_id: str, frozen_at_utc: str, event_start: str, event_end: str,
    bindings: Mapping[str, Any], census_receipt: Mapping[str, Any],
    census_dispositions: Sequence[Mapping[str, Any]],
    schema_version: str,
) -> dict[str, Any]:
    """Build the shared universe from one exhaustive source-census disposition."""
    receipt = validate_pead_event_census_receipt(census_receipt)
    start, end = _day(event_start, "event_window.start"), _day(event_end, "event_window.end")
    if start > end:
        raise PeadEventUniverseError("event window start follows end")
    binding = _exact(bindings, _BINDING_FIELDS, "bindings")
    normalized_bindings = {key: _sha(binding[key], f"bindings.{key}") for key in sorted(_BINDING_FIELDS)}
    if normalized_bindings["canonical_query_sha256"] != receipt["payload"]["canonical_query_sha256"]:
        raise PeadEventUniverseError("universe and census query hashes differ")
    dispositions, expected, gaps = _normalize_dispositions(
        census_dispositions, census_ids=receipt["payload"]["source_record_ids"],
        start=start, end=end,
    )
    counts, blockers, allowed = _status(
        source_count=receipt["payload"]["source_record_count"],
        dispositions=dispositions, expected=expected, gaps=gaps,
        identity_gaps_block=schema_version == EVENT_UNIVERSE_SCHEMA_VERSION,
    )
    payload = {
        "schema_version": schema_version,
        "candidate_id": _text(candidate_id, "candidate_id"),
        "frozen_at_utc": _utc(frozen_at_utc, "frozen_at_utc"),
        "event_window": {"start": start, "end": end},
        "bindings": normalized_bindings,
        "census_receipt": receipt,
        "census_dispositions": dispositions,
        "census_counts": counts,
        "expected_events": expected,
        "expected_event_ids": [row["event_id"] for row in expected],
        "identity_gaps": gaps,
        "blockers": blockers,
        "qualification_allowed": allowed,
    }
    return validate_pead_event_universe(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def build_pead_event_universe(
    *, candidate_id: str, frozen_at_utc: str, event_start: str, event_end: str,
    bindings: Mapping[str, Any], census_receipt: Mapping[str, Any],
    census_dispositions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the v1 universe, where any identity gap blocks the child."""
    return _build_pead_event_universe(
        candidate_id=candidate_id,
        frozen_at_utc=frozen_at_utc,
        event_start=event_start,
        event_end=event_end,
        bindings=bindings,
        census_receipt=census_receipt,
        census_dispositions=census_dispositions,
        schema_version=EVENT_UNIVERSE_SCHEMA_VERSION,
    )


def build_pead_event_universe_v2(
    *, candidate_id: str, frozen_at_utc: str, event_start: str, event_end: str,
    bindings: Mapping[str, Any], census_receipt: Mapping[str, Any],
    census_dispositions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build v2, retaining identity gaps as affected-row exclusions.

    Completeness, a nonempty census, and at least one expected event remain
    global requirements.  Explicit identity gaps stay in the immutable census
    accounting but no longer disqualify unrelated expected events.
    """
    return _build_pead_event_universe(
        candidate_id=candidate_id,
        frozen_at_utc=frozen_at_utc,
        event_start=event_start,
        event_end=event_end,
        bindings=bindings,
        census_receipt=census_receipt,
        census_dispositions=census_dispositions,
        schema_version=EVENT_UNIVERSE_V2_SCHEMA_VERSION,
    )


def validate_pead_event_universe(document: Mapping[str, Any]) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "event universe")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "event universe.payload")
    claimed = _sha(wrapper["artifact_hash"], "event universe.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadEventUniverseError("event universe artifact hash mismatch")
    schema_version = payload["schema_version"]
    if schema_version not in {
        EVENT_UNIVERSE_SCHEMA_VERSION,
        EVENT_UNIVERSE_V2_SCHEMA_VERSION,
    }:
        raise PeadEventUniverseError("unsupported event universe schema")
    candidate = _text(payload["candidate_id"], "candidate_id")
    frozen = _utc(payload["frozen_at_utc"], "frozen_at_utc")
    window = _exact(payload["event_window"], _WINDOW_FIELDS, "event_window")
    start, end = _day(window["start"], "event_window.start"), _day(window["end"], "event_window.end")
    if start > end:
        raise PeadEventUniverseError("event window start follows end")
    binding = _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    bindings = {key: _sha(binding[key], f"bindings.{key}") for key in sorted(_BINDING_FIELDS)}
    receipt = validate_pead_event_census_receipt(payload["census_receipt"])
    if bindings["canonical_query_sha256"] != receipt["payload"]["canonical_query_sha256"]:
        raise PeadEventUniverseError("universe and census query hashes differ")
    dispositions, expected, gaps = _normalize_dispositions(
        payload["census_dispositions"],
        census_ids=receipt["payload"]["source_record_ids"], start=start, end=end,
    )
    if payload["census_dispositions"] != dispositions:
        raise PeadEventUniverseError("census_dispositions must be canonically sorted")
    if payload["expected_events"] != expected:
        raise PeadEventUniverseError("expected_events must derive from census dispositions")
    ids = [row["event_id"] for row in expected]
    if payload["expected_event_ids"] != ids:
        raise PeadEventUniverseError("expected_event_ids must derive from census dispositions")
    if payload["identity_gaps"] != gaps:
        raise PeadEventUniverseError("identity_gaps must derive from census dispositions")
    counts, blockers, allowed = _status(
        source_count=receipt["payload"]["source_record_count"],
        dispositions=dispositions, expected=expected, gaps=gaps,
        identity_gaps_block=schema_version == EVENT_UNIVERSE_SCHEMA_VERSION,
    )
    _exact(payload["census_counts"], _COUNT_FIELDS, "census_counts")
    if payload["census_counts"] != counts:
        raise PeadEventUniverseError("census_counts are not derived exactly")
    if payload["blockers"] != blockers:
        raise PeadEventUniverseError("event-universe blockers are not derived exactly")
    if type(payload["qualification_allowed"]) is not bool or payload["qualification_allowed"] is not allowed:
        raise PeadEventUniverseError("event-universe qualification claim is inconsistent")
    normalized = {
        "schema_version": schema_version,
        "candidate_id": candidate, "frozen_at_utc": frozen,
        "event_window": {"start": start, "end": end}, "bindings": bindings,
        "census_receipt": receipt, "census_dispositions": dispositions,
        "census_counts": counts, "expected_events": expected,
        "expected_event_ids": ids, "identity_gaps": gaps,
        "blockers": blockers, "qualification_allowed": allowed,
    }
    if content_hash(normalized) != claimed:
        raise PeadEventUniverseError("event universe is not canonical")
    return {"artifact_hash": claimed, "payload": _plain_json(normalized)}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadEventUniverseError(f"event universe is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_EVENT_UNIVERSE_BYTES:
        raise PeadEventUniverseError("event universe exceeds its size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadEventUniverseError("event universe is not UTF-8") from exc
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadEventUniverseError(f"event universe contains duplicate key {key!r}")
            result[key] = value
        return result
    def reject(token: str) -> None:
        raise PeadEventUniverseError(f"event universe contains invalid number {token}")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadEventUniverseError(
            f"invalid event universe JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadEventUniverseError("event universe root must be an object")
    return value


def load_pead_event_census_receipt(path: str | Path) -> dict[str, Any]:
    return validate_pead_event_census_receipt(_read(Path(path)))


def load_pead_event_universe(path: str | Path) -> dict[str, Any]:
    return validate_pead_event_universe(_read(Path(path)))


__all__ = [
    "EVENT_CENSUS_RECEIPT_SCHEMA_VERSION", "EVENT_KEY_FIELDS",
    "EVENT_UNIVERSE_SCHEMA_VERSION", "EVENT_UNIVERSE_V2_SCHEMA_VERSION",
    "MAX_EVENT_UNIVERSE_BYTES",
    "PeadEventUniverseError", "build_pead_event_census_receipt",
    "build_pead_event_universe", "build_pead_event_universe_v2",
    "canonical_event_id", "canonical_json",
    "content_hash", "load_pead_event_census_receipt", "load_pead_event_universe",
    "validate_event_key", "validate_pead_event_census_receipt",
    "validate_pead_event_universe",
]
