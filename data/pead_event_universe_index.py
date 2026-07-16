"""Scalable index over calendar-year PEAD event-universe partitions.

The historical target contains far more events than should be copied into one
monolithic JSON document and then embedded repeatedly in downstream evidence.
Each calendar-year partition remains a complete ``pead_event_universe.v1`` or
``pead_event_universe.v2`` artifact.  This index binds their exact identities,
requires contiguous target-window coverage, and proves that event IDs do not
cross partitions.  Authoritative source replay still has to trust or rebuild
the child universes; this structural index does not turn caller-supplied census
hashes into source provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

from data.pead_event_universe import (
    PeadEventUniverseError,
    canonical_json,
    content_hash,
    validate_pead_event_universe,
)


EVENT_UNIVERSE_INDEX_SCHEMA_VERSION = "pead_event_universe_index.v1"
EVENT_UNIVERSE_PARTITION_POLICY_SCHEMA_VERSION = "pead_event_universe_partition_policy.v1"
MAX_EVENT_UNIVERSE_INDEX_BYTES = 8 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "indexed_at_utc",
    "target_window",
    "partition_policy",
    "bindings",
    "partitions",
    "counts",
    "qualification",
}
_WINDOW_FIELDS = {"start", "end"}
_BINDING_FIELDS = {
    "market_snapshot_sha256",
    "identity_snapshot_sha256",
    "candidate_specification_sha256",
    "construction_code_sha256",
    "partition_policy_sha256",
}
_PARTITION_FIELDS = {
    "partition_id",
    "event_window",
    "event_universe_sha256",
    "canonical_query_sha256",
    "expected_event_ids_sha256",
    "event_count",
    "source_record_count",
    "identity_gap_count",
    "qualification_allowed",
}
_COUNT_FIELDS = {
    "partition_count",
    "event_count",
    "source_record_count",
    "identity_gap_count",
}
_QUALIFICATION_FIELDS = {
    "partitions_contiguous",
    "event_ids_unique",
    "all_partitions_qualified",
    "qualification_allowed",
}

PARTITION_POLICY = {
    "schema_version": EVENT_UNIVERSE_PARTITION_POLICY_SCHEMA_VERSION,
    "partition_key": "fiscal_period_end_calendar_year",
    "partition_id_format": "YYYY",
    "window_rule": "contiguous_nonoverlapping_calendar_year_intersections",
    "event_membership_rule": "event_key_fiscal_period_end_within_partition_window",
    "cross_partition_event_rule": "event_id_must_appear_exactly_once",
    "empty_partition_rule": "forbidden_within_declared_target_window",
}


class PeadEventUniverseIndexError(ValueError):
    """The event-universe partition index is malformed or cannot replay."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadEventUniverseIndexError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadEventUniverseIndexError(f"{label} must be a lowercase SHA-256")
    return value


def _day(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PeadEventUniverseIndexError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadEventUniverseIndexError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadEventUniverseIndexError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadEventUniverseIndexError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadEventUniverseIndexError(f"{label} must be canonical UTC with Z") from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadEventUniverseIndexError(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _expected_windows(start: date, end: date) -> list[tuple[str, date, date]]:
    result: list[tuple[str, date, date]] = []
    for year in range(start.year, end.year + 1):
        partition_start = max(start, date(year, 1, 1))
        partition_end = min(end, date(year, 12, 31))
        result.append((str(year), partition_start, partition_end))
    return result


def build_pead_event_universe_index(
    *,
    partitions: Sequence[Mapping[str, Any]],
    target_start: str,
    target_end: str,
    indexed_at_utc: str,
) -> dict[str, Any]:
    """Build an exact root index from all nonempty calendar-year children."""
    if isinstance(partitions, (str, bytes)) or not isinstance(partitions, Sequence):
        raise PeadEventUniverseIndexError("partitions must be a sequence")
    if not partitions:
        raise PeadEventUniverseIndexError("partitions must be nonempty")
    start = _day(target_start, "target_start")
    end = _day(target_end, "target_end")
    if start > end:
        raise PeadEventUniverseIndexError("target window is reversed")
    indexed_text, indexed_at = _utc(indexed_at_utc, "indexed_at_utc")

    children: list[dict[str, Any]] = []
    for index, raw in enumerate(partitions):
        try:
            children.append(validate_pead_event_universe(raw))
        except PeadEventUniverseError as exc:
            raise PeadEventUniverseIndexError(
                f"partition {index} is not a valid event universe"
            ) from exc
    children.sort(key=lambda child: child["payload"]["event_window"]["start"])
    expected_windows = _expected_windows(start, end)
    if len(children) != len(expected_windows):
        raise PeadEventUniverseIndexError(
            "partitions must cover every target calendar year exactly once"
        )

    first = children[0]["payload"]
    candidate_id = first["candidate_id"]
    common_binding_names = (
        "market_snapshot_sha256",
        "identity_snapshot_sha256",
        "candidate_specification_sha256",
        "construction_code_sha256",
    )
    common_bindings = {
        field: _sha(first["bindings"][field], f"bindings.{field}") for field in common_binding_names
    }
    frozen_at = first["frozen_at_utc"]
    _, frozen_dt = _utc(frozen_at, "partition frozen_at_utc")
    if indexed_at < frozen_dt:
        raise PeadEventUniverseIndexError("index predates its event universes")

    descriptors: list[dict[str, Any]] = []
    all_event_ids: list[str] = []
    for child, (partition_id, expected_start, expected_end) in zip(
        children, expected_windows, strict=True
    ):
        payload = child["payload"]
        if payload["candidate_id"] != candidate_id:
            raise PeadEventUniverseIndexError("partition candidates differ")
        if payload["frozen_at_utc"] != frozen_at:
            raise PeadEventUniverseIndexError("partition freeze timestamps differ")
        for field, expected_value in common_bindings.items():
            if payload["bindings"][field] != expected_value:
                raise PeadEventUniverseIndexError(f"partition binding differs: {field}")
        window = payload["event_window"]
        actual_start = _day(window["start"], "partition window start")
        actual_end = _day(window["end"], "partition window end")
        if (actual_start, actual_end) != (expected_start, expected_end):
            raise PeadEventUniverseIndexError(
                "partition windows are not exact calendar-year intersections"
            )
        ids = payload["expected_event_ids"]
        if not ids:
            raise PeadEventUniverseIndexError("empty calendar-year partitions are not qualifying")
        for event in payload["expected_events"]:
            period = _day(
                event["event_key"]["fiscal_period_end"],
                "partition event fiscal_period_end",
            )
            if not expected_start <= period <= expected_end:
                raise PeadEventUniverseIndexError("event is outside its calendar-year partition")
        all_event_ids.extend(ids)
        counts = payload["census_counts"]
        descriptors.append(
            {
                "partition_id": partition_id,
                "event_window": {
                    "start": expected_start.isoformat(),
                    "end": expected_end.isoformat(),
                },
                "event_universe_sha256": child["artifact_hash"],
                "canonical_query_sha256": payload["bindings"]["canonical_query_sha256"],
                "expected_event_ids_sha256": content_hash(ids),
                "event_count": counts["expected_event_count"],
                "source_record_count": counts["source_record_count"],
                "identity_gap_count": counts["identity_gap_count"],
                "qualification_allowed": payload["qualification_allowed"],
            }
        )

    event_ids_unique = len(all_event_ids) == len(set(all_event_ids))
    if not event_ids_unique:
        raise PeadEventUniverseIndexError("event IDs cross calendar-year partitions")
    all_qualified = all(row["qualification_allowed"] for row in descriptors)
    counts = {
        "partition_count": len(descriptors),
        "event_count": sum(row["event_count"] for row in descriptors),
        "source_record_count": sum(row["source_record_count"] for row in descriptors),
        "identity_gap_count": sum(row["identity_gap_count"] for row in descriptors),
    }
    payload = {
        "schema_version": EVENT_UNIVERSE_INDEX_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "indexed_at_utc": indexed_text,
        "target_window": {"start": start.isoformat(), "end": end.isoformat()},
        "partition_policy": PARTITION_POLICY,
        "bindings": {
            **common_bindings,
            "partition_policy_sha256": content_hash(PARTITION_POLICY),
        },
        "partitions": descriptors,
        "counts": counts,
        "qualification": {
            "partitions_contiguous": True,
            "event_ids_unique": True,
            "all_partitions_qualified": all_qualified,
            "qualification_allowed": all_qualified and counts["event_count"] > 0,
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": _plain(payload)}


def validate_pead_event_universe_index(
    document: Mapping[str, Any], *, partitions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Rebuild the root from every child and require exact equality."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "event universe index")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "event universe index.payload")
    claimed = _sha(wrapper["artifact_hash"], "event universe index.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadEventUniverseIndexError("event universe index artifact hash mismatch")
    if payload["schema_version"] != EVENT_UNIVERSE_INDEX_SCHEMA_VERSION:
        raise PeadEventUniverseIndexError("unsupported event universe index schema")
    _exact(payload["target_window"], _WINDOW_FIELDS, "target_window")
    _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    if payload["partition_policy"] != PARTITION_POLICY:
        raise PeadEventUniverseIndexError("event universe partition policy changed")
    raw_descriptors = payload["partitions"]
    if not isinstance(raw_descriptors, list):
        raise PeadEventUniverseIndexError("partitions must be an array")
    for index, row in enumerate(raw_descriptors):
        _exact(row, _PARTITION_FIELDS, f"partitions[{index}]")
        _exact(row["event_window"], _WINDOW_FIELDS, f"partitions[{index}].event_window")
    _exact(payload["counts"], _COUNT_FIELDS, "counts")
    _exact(payload["qualification"], _QUALIFICATION_FIELDS, "qualification")
    expected = build_pead_event_universe_index(
        partitions=partitions,
        target_start=payload["target_window"]["start"],
        target_end=payload["target_window"]["end"],
        indexed_at_utc=payload["indexed_at_utc"],
    )
    if _plain(document) != expected:
        raise PeadEventUniverseIndexError(
            "event universe index does not replay from its child partitions"
        )
    return expected


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadEventUniverseIndexError(f"event universe index is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_EVENT_UNIVERSE_INDEX_BYTES:
        raise PeadEventUniverseIndexError("event universe index exceeds its size limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PeadEventUniverseIndexError("event universe index is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PeadEventUniverseIndexError("event universe index root must be an object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadEventUniverseIndexError(
            "event universe index bytes are not canonical JSON plus one newline"
        )
    return value


def load_pead_event_universe_index(
    path: str | Path, *, partitions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return validate_pead_event_universe_index(_strict_json(Path(path)), partitions=partitions)


__all__ = [
    "EVENT_UNIVERSE_INDEX_SCHEMA_VERSION",
    "EVENT_UNIVERSE_PARTITION_POLICY_SCHEMA_VERSION",
    "MAX_EVENT_UNIVERSE_INDEX_BYTES",
    "PARTITION_POLICY",
    "PeadEventUniverseIndexError",
    "build_pead_event_universe_index",
    "load_pead_event_universe_index",
    "validate_pead_event_universe_index",
]
