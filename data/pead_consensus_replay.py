"""Byte-replayed, trust-rooted consensus normalization for PEAD.

``pead_consensus_evidence.v1`` is the normalized interchange format.  It is
not, by itself, proof that the normalized values came from preserved provider
bytes.  This module closes that boundary: a replay receipt binds the exact raw
artifact, frozen event universe, dated security identities, reviewed source
manifest, reviewed metric profile, an exhaustive raw-row ledger, and the
derived consensus evidence.

Raw bytes are deliberately external to the receipt so licensed or very large
provider files need not be duplicated.  :func:`verify_pead_consensus_replay`
is therefore the authoritative operation: it reopens the supplied bytes and
rebuilds the complete receipt.  Structural validation alone never establishes
that those external bytes were present.
"""

from __future__ import annotations

import base64
import binascii
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any

from data.earnings_announcements import (
    EarningsAnnouncementSnapshot,
    SnapshotIntegrityError,
)
from data.pead_consensus_evidence import (
    PeadConsensusEvidenceError,
    build_pead_consensus_evidence,
    validate_pead_consensus_evidence,
)
from data.pead_event_universe import (
    PeadEventUniverseError,
    canonical_event_id,
    validate_pead_event_universe,
)


CONSENSUS_REPLAY_SCHEMA_VERSION = "pead_consensus_normalization_replay.v1"
CONSENSUS_SOURCE_MANIFEST_SCHEMA_VERSION = "pead_consensus_source_manifest.v1"
CONSENSUS_METRIC_PROFILE_SCHEMA_VERSION = "pead_consensus_metric_profile.v1"

ZACKS_EEH_ADAPTER = "nasdaq_data_link_zacks_eeh.v1"
PROVIDER_NEUTRAL_JSON_ADAPTER = "provider_neutral_json.v1"
PROVIDER_NEUTRAL_CSV_ADAPTER = "provider_neutral_csv.v1"
SUPPORTED_ADAPTERS = (
    ZACKS_EEH_ADAPTER,
    PROVIDER_NEUTRAL_JSON_ADAPTER,
    PROVIDER_NEUTRAL_CSV_ADAPTER,
)

MAX_RAW_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_REPLAY_RECEIPT_BYTES = 512 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_CIK = re.compile(r"^[0-9]{10}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_MACHINE_REASON = re.compile(r"^[a-z][a-z0-9_]*$")
_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")

_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_SOURCE_FIELDS = {
    "schema_version",
    "candidate_id",
    "adapter_id",
    "provider_id",
    "dataset_id",
    "evidence_class",
    "source_captured_at_utc",
    "provider_snapshot_at_utc",
    "canonical_query_sha256",
    "records_path",
    "field_map",
}
_GENERIC_FIELD_KEYS = {
    "provider_record_id",
    "provider_security_id",
    "ticker",
    "cik",
    "fiscal_period_end",
    "fiscal_period_type",
    "provider_as_of_date",
    "trusted_available_at_utc",
    "availability_precision",
    "consensus_value",
    "analyst_count",
    "currency_code",
}
_PROFILE_FIELDS = {
    "schema_version",
    "candidate_id",
    "profile_id",
    "adapter_id",
    "provider_id",
    "dataset_id",
    "value_field",
    "analyst_count_field",
    "metric",
}
_PROFILE_METRIC_FIELDS = {
    "metric_id",
    "accounting_basis",
    "per_share_basis",
    "scope",
    "canonical_share_basis",
    "unit",
    "metric_definition_sha256",
}
_REPLAY_FIELDS = {
    "schema_version",
    "candidate_id",
    "adapter_id",
    "bindings",
    "trust_policy",
    "raw_artifact",
    "source_manifest",
    "metric_profile",
    "identity_snapshot",
    "raw_record_ledger",
    "counts",
    "blockers",
    "qualification_allowed",
    "consensus_evidence",
}
_BINDING_FIELDS = {
    "event_universe_sha256",
    "identity_snapshot_sha256",
    "source_manifest_sha256",
    "metric_profile_sha256",
    "raw_artifact_bytes_sha256",
    "consensus_evidence_sha256",
}
_TRUST_FIELDS = {
    "event_universe_allowlist_sha256",
    "source_manifest_allowlist_sha256",
    "metric_profile_allowlist_sha256",
    "identity_snapshot_allowlist_sha256",
    "raw_artifact_allowlist_sha256",
    "event_universe_trusted",
    "source_manifest_trusted",
    "metric_profile_trusted",
    "identity_snapshot_trusted",
    "raw_artifact_trusted",
}
_RAW_ARTIFACT_FIELDS = {
    "format",
    "bytes_sha256",
    "byte_count",
    "content_artifact_sha256",
    "selected_record_set",
    "source_blockers",
}
_LEDGER_FIELDS = {
    "raw_record_sha256",
    "source_locator",
    "provider_record_id",
    "provider_security_id",
    "event_id",
    "disposition",
    "reasons",
    "normalized_vintage_sha256",
}
_LOCATOR_FIELDS = {
    "raw_artifact_bytes_sha256",
    "container_sha256",
    "member_name",
    "page_sequence",
    "row_index",
}
_COUNT_FIELDS = {
    "raw_record_count",
    "ledger_record_count",
    "matched_record_count",
    "outside_universe_record_count",
    "identity_gap_record_count",
    "invalid_record_count",
    "duplicate_record_count",
    "expected_event_count",
    "available_event_count",
    "missing_event_count",
    "normalized_vintage_count",
}
_DISPOSITIONS = {
    "matched_expected_event",
    "outside_event_universe",
    "identity_gap",
    "invalid_record",
    "duplicate_natural_vintage",
}
_EVIDENCE_CLASSES = {
    "development_sample",
    "historical_reconstruction",
    "prospective_signal",
}
_QUALIFYING_EVIDENCE_CLASSES = {
    "historical_reconstruction",
    "prospective_signal",
}
_AVAILABILITY_PRECISIONS = {"date", "second", "microsecond"}


class PeadConsensusReplayError(ValueError):
    """Raw consensus evidence or its deterministic replay is invalid."""


@dataclass(frozen=True)
class _RawNumber:
    token: str


class _RowError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def canonical_json(value: Any) -> str:
    """Return deterministic finite JSON for replay identities."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PeadConsensusReplayError("replay evidence is non-finite")
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PeadConsensusReplayError("replay evidence keys must be strings")
                result[key] = normalize(child)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise PeadConsensusReplayError(f"unsupported replay evidence value: {type(item).__name__}")

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadConsensusReplayError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadConsensusReplayError(f"{label} must be nonempty canonical text")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadConsensusReplayError(f"{label} must be a lowercase SHA-256")
    return value


def _day(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PeadConsensusReplayError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadConsensusReplayError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadConsensusReplayError(f"{label} must be canonical YYYY-MM-DD")
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadConsensusReplayError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadConsensusReplayError(f"{label} must be canonical UTC with Z") from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if value != canonical:
        raise PeadConsensusReplayError(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _optional_utc(value: Any, label: str) -> str | None:
    return None if value is None else _utc(value, label)[0]


def _machine_reasons(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise PeadConsensusReplayError(f"{label} must be an array")
    reasons = []
    for item in value:
        reason = _text(item, label)
        if _MACHINE_REASON.fullmatch(reason) is None:
            raise PeadConsensusReplayError(f"{label} must contain machine reasons")
        reasons.append(reason)
    if reasons != sorted(set(reasons)):
        raise PeadConsensusReplayError(f"{label} must be sorted and unique")
    return reasons


def _sorted_hashes(values: Sequence[str] | set[str] | frozenset[str], label: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, set, frozenset)):
        raise PeadConsensusReplayError(f"{label} must be a collection of SHA-256 values")
    hashes = sorted(_sha(item, label) for item in values)
    if len(hashes) != len(set(hashes)):
        raise PeadConsensusReplayError(f"{label} must not contain duplicates")
    return hashes


def _strict_raw_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadConsensusReplayError(f"{label} must be UTF-8 JSON") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadConsensusReplayError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadConsensusReplayError(f"{label} contains invalid number {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=unique,
            parse_int=_RawNumber,
            parse_float=_RawNumber,
            parse_constant=reject,
        )
    except PeadConsensusReplayError:
        raise
    except json.JSONDecodeError as exc:
        raise PeadConsensusReplayError(
            f"{label} is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _decimal(value: Any) -> str:
    if isinstance(value, _RawNumber):
        token = value.token
    elif isinstance(value, str) and value == value.strip() and value:
        token = value
    else:
        raise _RowError("consensus_value_invalid")
    if len(token) > 128:
        raise _RowError("consensus_value_invalid")
    try:
        parsed = Decimal(token)
    except InvalidOperation as exc:
        raise _RowError("consensus_value_invalid") from exc
    if not parsed.is_finite():
        raise _RowError("consensus_value_invalid")
    decimal_tuple = parsed.as_tuple()
    digits = len(decimal_tuple.digits)
    exponent = decimal_tuple.exponent
    estimated_plain_length = (
        digits + exponent + int(decimal_tuple.sign)
        if exponent >= 0
        else max(digits, -exponent + 1) + int(decimal_tuple.sign) + 1
    )
    if estimated_plain_length > 128:
        raise _RowError("consensus_value_invalid")
    try:
        normalized = format(parsed, "f")
    except (InvalidOperation, ValueError) as exc:
        raise _RowError("consensus_value_invalid") from exc
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if not normalized or Decimal(normalized) == 0:
        normalized = "0"
    if len(normalized) > 128:
        raise _RowError("consensus_value_invalid")
    return normalized


def _positive_integer(value: Any) -> int:
    token = value.token if isinstance(value, _RawNumber) else value
    if not isinstance(token, str) or _INTEGER.fullmatch(token) is None:
        raise _RowError("analyst_count_invalid")
    number = int(token)
    if number < 1:
        raise _RowError("analyst_count_invalid")
    return number


def _row_text(value: Any, reason: str, *, upper: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _RowError(reason)
    return value.upper() if upper else value


def _row_day(value: Any, reason: str) -> str:
    try:
        return _day(value, reason)
    except PeadConsensusReplayError as exc:
        raise _RowError(reason) from exc


def _row_cik(value: Any) -> str:
    cik = _row_text(value, "cik_invalid")
    if _CIK.fullmatch(cik) is None or cik == "0000000000":
        raise _RowError("cik_invalid")
    return cik


def _row_currency(value: Any) -> str:
    currency = _row_text(value, "currency_invalid", upper=True)
    if _CURRENCY.fullmatch(currency) is None:
        raise _RowError("currency_invalid")
    return currency


def _validate_identity_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from data.sharadar_source_evidence import (  # local import avoids a cycle
            SharadarSourceEvidenceError,
            validate_pead_security_identity_snapshot_structure,
        )
    except (ImportError, AttributeError) as exc:  # pragma: no cover - integration guard
        raise PeadConsensusReplayError(
            "pead_security_identity_snapshot.v1 validator is unavailable"
        ) from exc
    try:
        return validate_pead_security_identity_snapshot_structure(document)
    except SharadarSourceEvidenceError as exc:
        raise PeadConsensusReplayError("security identity snapshot is invalid") from exc


def build_pead_consensus_source_manifest(
    *,
    candidate_id: str,
    adapter_id: str,
    provider_id: str,
    dataset_id: str,
    evidence_class: str | None,
    source_captured_at_utc: str | None,
    provider_snapshot_at_utc: str | None,
    canonical_query_sha256: str | None,
    records_path: str | None,
    field_map: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Build the reviewed, provider-specific interpretation manifest."""
    payload = {
        "schema_version": CONSENSUS_SOURCE_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "adapter_id": adapter_id,
        "provider_id": provider_id,
        "dataset_id": dataset_id,
        "evidence_class": evidence_class,
        "source_captured_at_utc": source_captured_at_utc,
        "provider_snapshot_at_utc": provider_snapshot_at_utc,
        "canonical_query_sha256": canonical_query_sha256,
        "records_path": records_path,
        "field_map": None if field_map is None else dict(field_map),
    }
    return validate_pead_consensus_source_manifest(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_consensus_source_manifest(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "consensus source manifest")
    payload = _exact(wrapper["payload"], _SOURCE_FIELDS, "source manifest.payload")
    claimed = _sha(wrapper["artifact_hash"], "source manifest.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadConsensusReplayError("source manifest artifact hash mismatch")
    if payload["schema_version"] != CONSENSUS_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise PeadConsensusReplayError("unsupported consensus source manifest schema")
    candidate = _text(payload["candidate_id"], "source manifest.candidate_id")
    adapter = payload["adapter_id"]
    if adapter not in SUPPORTED_ADAPTERS:
        raise PeadConsensusReplayError("source manifest adapter is not registered")
    provider = _text(payload["provider_id"], "source manifest.provider_id")
    dataset = _text(payload["dataset_id"], "source manifest.dataset_id")
    snapshot = _optional_utc(
        payload["provider_snapshot_at_utc"], "source manifest.provider_snapshot_at_utc"
    )
    if adapter == ZACKS_EEH_ADAPTER:
        if provider != "nasdaq-data-link-zacks" or dataset != "ZACKS/EEH":
            raise PeadConsensusReplayError("Zacks adapter provider/dataset identity changed")
        for field in (
            "evidence_class",
            "source_captured_at_utc",
            "canonical_query_sha256",
            "records_path",
            "field_map",
        ):
            if payload[field] is not None:
                raise PeadConsensusReplayError(
                    f"Zacks adapter derives source manifest.{field} from raw bytes"
                )
        normalized = {
            "schema_version": CONSENSUS_SOURCE_MANIFEST_SCHEMA_VERSION,
            "candidate_id": candidate,
            "adapter_id": adapter,
            "provider_id": provider,
            "dataset_id": dataset,
            "evidence_class": None,
            "source_captured_at_utc": None,
            "provider_snapshot_at_utc": snapshot,
            "canonical_query_sha256": None,
            "records_path": None,
            "field_map": None,
        }
    else:
        evidence_class = payload["evidence_class"]
        if evidence_class not in _EVIDENCE_CLASSES:
            raise PeadConsensusReplayError("generic source evidence_class is unsupported")
        captured, captured_dt = _utc(
            payload["source_captured_at_utc"], "source manifest.source_captured_at_utc"
        )
        if snapshot is not None:
            _, snapshot_dt = _utc(snapshot, "source manifest.provider_snapshot_at_utc")
            if snapshot_dt > captured_dt:
                raise PeadConsensusReplayError("provider snapshot follows source capture")
        query = _sha(
            payload["canonical_query_sha256"],
            "source manifest.canonical_query_sha256",
        )
        path = payload["records_path"]
        allowed_paths = {"$", "$.records"} if adapter == PROVIDER_NEUTRAL_JSON_ADAPTER else {"$"}
        if path not in allowed_paths:
            raise PeadConsensusReplayError("generic source records_path is unsupported")
        field_map = _exact(payload["field_map"], _GENERIC_FIELD_KEYS, "source field_map")
        fields = {
            key: _text(field_map[key], f"source field_map.{key}")
            for key in sorted(_GENERIC_FIELD_KEYS)
        }
        if len(set(fields.values())) != len(fields):
            raise PeadConsensusReplayError("source field_map columns must be unique")
        normalized = {
            "schema_version": CONSENSUS_SOURCE_MANIFEST_SCHEMA_VERSION,
            "candidate_id": candidate,
            "adapter_id": adapter,
            "provider_id": provider,
            "dataset_id": dataset,
            "evidence_class": evidence_class,
            "source_captured_at_utc": captured,
            "provider_snapshot_at_utc": snapshot,
            "canonical_query_sha256": query,
            "records_path": path,
            "field_map": fields,
        }
    if content_hash(normalized) != claimed:
        raise PeadConsensusReplayError("source manifest is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized)}


def build_pead_consensus_metric_profile(
    *,
    candidate_id: str,
    profile_id: str,
    adapter_id: str,
    provider_id: str,
    dataset_id: str,
    value_field: str,
    analyst_count_field: str,
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a reviewed semantic profile for one raw consensus measure."""
    payload = {
        "schema_version": CONSENSUS_METRIC_PROFILE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "profile_id": profile_id,
        "adapter_id": adapter_id,
        "provider_id": provider_id,
        "dataset_id": dataset_id,
        "value_field": value_field,
        "analyst_count_field": analyst_count_field,
        "metric": dict(metric),
    }
    return validate_pead_consensus_metric_profile(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_consensus_metric_profile(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "consensus metric profile")
    payload = _exact(wrapper["payload"], _PROFILE_FIELDS, "metric profile.payload")
    claimed = _sha(wrapper["artifact_hash"], "metric profile.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadConsensusReplayError("metric profile artifact hash mismatch")
    if payload["schema_version"] != CONSENSUS_METRIC_PROFILE_SCHEMA_VERSION:
        raise PeadConsensusReplayError("unsupported consensus metric profile schema")
    adapter = payload["adapter_id"]
    if adapter not in SUPPORTED_ADAPTERS:
        raise PeadConsensusReplayError("metric profile adapter is not registered")
    metric = _exact(payload["metric"], _PROFILE_METRIC_FIELDS, "metric profile.metric")
    normalized_metric = {
        "metric_id": _text(metric["metric_id"], "metric.metric_id"),
        "accounting_basis": _text(metric["accounting_basis"], "metric.accounting_basis"),
        "per_share_basis": _text(metric["per_share_basis"], "metric.per_share_basis"),
        "scope": _text(metric["scope"], "metric.scope"),
        "canonical_share_basis": _text(
            metric["canonical_share_basis"], "metric.canonical_share_basis"
        ),
        "unit": _text(metric["unit"], "metric.unit"),
        "metric_definition_sha256": _sha(
            metric["metric_definition_sha256"], "metric.metric_definition_sha256"
        ),
    }
    normalized = {
        "schema_version": CONSENSUS_METRIC_PROFILE_SCHEMA_VERSION,
        "candidate_id": _text(payload["candidate_id"], "metric profile.candidate_id"),
        "profile_id": _text(payload["profile_id"], "metric profile.profile_id"),
        "adapter_id": adapter,
        "provider_id": _text(payload["provider_id"], "metric profile.provider_id"),
        "dataset_id": _text(payload["dataset_id"], "metric profile.dataset_id"),
        "value_field": _text(payload["value_field"], "metric profile.value_field"),
        "analyst_count_field": _text(
            payload["analyst_count_field"], "metric profile.analyst_count_field"
        ),
        "metric": normalized_metric,
    }
    if adapter == ZACKS_EEH_ADAPTER and (
        normalized["provider_id"] != "nasdaq-data-link-zacks"
        or normalized["dataset_id"] != "ZACKS/EEH"
        or normalized["value_field"] != "eps_mean_est"
        or normalized["analyst_count_field"] != "eps_cnt_est"
    ):
        raise PeadConsensusReplayError("Zacks EEH metric profile changed raw fields")
    if content_hash(normalized) != claimed:
        raise PeadConsensusReplayError("metric profile is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized)}


def _raw_artifact_bytes(raw_artifact: bytes) -> tuple[bytes, str]:
    if not isinstance(raw_artifact, bytes):
        raise PeadConsensusReplayError("raw_artifact must be exact bytes")
    if not raw_artifact:
        raise PeadConsensusReplayError("raw_artifact must not be empty")
    if len(raw_artifact) > MAX_RAW_ARTIFACT_BYTES:
        raise PeadConsensusReplayError("raw_artifact exceeds its size limit")
    return raw_artifact, hashlib.sha256(raw_artifact).hexdigest()


def _locator(
    *,
    artifact_hash: str,
    container_hash: str,
    member_name: str,
    page_sequence: int,
    row_index: int,
) -> dict[str, Any]:
    return {
        "raw_artifact_bytes_sha256": artifact_hash,
        "container_sha256": container_hash,
        "member_name": member_name,
        "page_sequence": page_sequence,
        "row_index": row_index,
    }


def _strict_page_rows(
    raw_body: bytes,
    *,
    label: str,
    expected_columns: Sequence[str],
) -> list[list[Any]]:
    document = _strict_raw_json(raw_body, label=label)
    outer = _exact(document, {"datatable", "meta"}, label)
    table = _exact(outer["datatable"], {"data", "columns"}, f"{label}.datatable")
    _exact(outer["meta"], {"next_cursor_id"}, f"{label}.meta")
    columns = table["columns"]
    if not isinstance(columns, list):
        raise PeadConsensusReplayError(f"{label}.columns must be an array")
    names: list[str] = []
    for index, raw_column in enumerate(columns):
        column = _exact(raw_column, {"name", "type"}, f"{label}.columns[{index}]")
        names.append(_text(column["name"], f"{label}.columns[{index}].name"))
        _text(column["type"], f"{label}.columns[{index}].type")
    if names != list(expected_columns):
        raise PeadConsensusReplayError(f"{label} raw columns changed")
    rows = table["data"]
    if not isinstance(rows, list):
        raise PeadConsensusReplayError(f"{label}.data must be an array")
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(names):
            raise PeadConsensusReplayError(f"{label}.data[{index}] width changed")
    return rows


def _zacks_table_rows(
    *,
    snapshot_payload: Mapping[str, Any],
    table_code: str,
    artifact_bytes_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tables = snapshot_payload["tables"]
    if table_code not in tables:
        raise PeadConsensusReplayError(f"Zacks snapshot is missing {table_code}")
    table = tables[table_code]
    column_names = [column["name"] for column in table["columns"]]
    pages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for page in table["provider_metadata"]["pages"]:
        encoded = page["response_body_base64"]
        try:
            body = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:  # already checked by snapshot validator
            raise PeadConsensusReplayError("Zacks raw page is not canonical base64") from exc
        body_hash = hashlib.sha256(body).hexdigest()
        rows = _strict_page_rows(
            body,
            label=f"{table_code}.page[{page['page_number']}]",
            expected_columns=column_names,
        )
        pages.append(
            {
                "sequence": page["page_number"],
                "request_sha256": page["request_sha256"],
                "raw_response_sha256": body_hash,
                "raw_response_bytes": len(body),
                "continuation_token_sha256": (
                    None
                    if page["next_cursor_id"] is None
                    else hashlib.sha256(page["next_cursor_id"].encode("utf-8")).hexdigest()
                ),
                "captured_at_utc": page["captured_at"],
            }
        )
        for row_index, raw_row in enumerate(rows):
            records.append(
                {
                    "raw": dict(zip(column_names, raw_row, strict=True)),
                    "locator": _locator(
                        artifact_hash=artifact_bytes_sha256,
                        container_hash=body_hash,
                        member_name=table_code,
                        page_sequence=page["page_number"],
                        row_index=row_index,
                    ),
                }
            )
    return records, pages


def _identity_is_valid(identity: Mapping[str, Any], period: str) -> bool:
    valid_from = identity.get("valid_from")
    valid_through = identity.get("valid_through")
    return (
        isinstance(valid_from, str)
        and valid_from <= period
        and (valid_through is None or (isinstance(valid_through, str) and period <= valid_through))
    )


def _matching_identities(
    identities: Sequence[Mapping[str, Any]], *, cik: str, ticker: str, period: str
) -> list[Mapping[str, Any]]:
    return [
        identity
        for identity in identities
        if identity["cik"] == cik
        and identity["ticker"].upper() == ticker.upper()
        and _identity_is_valid(identity, period)
    ]


def _query_values(value: Any) -> set[str] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return set()
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _zacks_scoped_event_ids(
    *,
    universe: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    mt_rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> list[str]:
    ticker_filter = _query_values(params.get("ticker"))
    master_filter = _query_values(params.get("m_ticker"))
    period_type = params.get("per_type")
    start = params.get("per_end_date.gte")
    end = params.get("per_end_date.lte")
    scoped: list[str] = []
    for expected in universe["payload"]["expected_events"]:
        event = expected["event_key"]
        period = event["fiscal_period_end"]
        if period_type is not None and period_type != "Q":
            continue
        if start is not None and (not isinstance(start, str) or start > period):
            continue
        if end is not None and (not isinstance(end, str) or end < period):
            continue
        dated = [
            row
            for row in identities
            if row["cik"] == event["cik"] and _identity_is_valid(row, period)
        ]
        provider_rows = [
            row
            for row in mt_rows
            if row.get("comp_cik") == event["cik"]
            and any(
                isinstance(row.get("ticker"), str)
                and row["ticker"].upper() == identity["ticker"].upper()
                for identity in dated
            )
        ]
        if not provider_rows:
            continue
        if ticker_filter is not None and not any(
            str(row.get("ticker", "")).upper() in ticker_filter for row in provider_rows
        ):
            continue
        if master_filter is not None and not any(
            str(row.get("m_ticker", "")).upper() in master_filter for row in provider_rows
        ):
            continue
        scoped.append(expected["event_id"])
    return sorted(scoped)


def _receipt(
    *,
    captured_at: str,
    scope_kind: str,
    query_sha256: str,
    expected_event_ids: Sequence[str],
    mode: str,
    pages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_pages = [
        {
            "sequence": page["sequence"],
            "request_sha256": page["request_sha256"],
            "raw_response_sha256": page["raw_response_sha256"],
            "raw_response_bytes": page["raw_response_bytes"],
            "continuation_token_sha256": page["continuation_token_sha256"],
        }
        for page in pages
    ]
    body = {
        "source_captured_at_utc": captured_at,
        "query_scope": {
            "scope_kind": scope_kind,
            "canonical_query_sha256": query_sha256,
            "expected_event_ids": sorted(expected_event_ids),
        },
        "pagination": {
            "mode": mode,
            "terminal_page_observed": bool(normalized_pages)
            and normalized_pages[-1]["continuation_token_sha256"] is None,
            "page_count": len(normalized_pages),
            "pages": normalized_pages,
        },
    }
    return {"receipt_sha256": content_hash(body), **body}


def _extract_zacks(
    *,
    raw: bytes,
    bytes_sha256: str,
    universe: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        snapshot = EarningsAnnouncementSnapshot.from_json(text)
    except (UnicodeDecodeError, SnapshotIntegrityError) as exc:
        raise PeadConsensusReplayError("Zacks source snapshot is invalid") from exc
    payload = snapshot.payload
    if payload["candidate_id"] != universe["payload"]["candidate_id"]:
        raise PeadConsensusReplayError("Zacks snapshot belongs to another candidate")
    eeh, eeh_pages = _zacks_table_rows(
        snapshot_payload=payload,
        table_code="ZACKS/EEH",
        artifact_bytes_sha256=bytes_sha256,
    )
    mt, mt_pages = _zacks_table_rows(
        snapshot_payload=payload,
        table_code="ZACKS/MT",
        artifact_bytes_sha256=bytes_sha256,
    )
    mt_rows = [item["raw"] for item in mt]
    mt_by_master: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in mt_rows:
        master = row.get("m_ticker")
        if isinstance(master, str) and master:
            mt_by_master[master.upper()].append(row)
    records: list[dict[str, Any]] = []
    for item in eeh:
        row = item["raw"]
        master = row.get("m_ticker")
        matches = mt_by_master.get(master.upper(), []) if isinstance(master, str) else []
        pre_reasons: list[str] = []
        source_cik = None
        if len(matches) != 1:
            pre_reasons.append("zacks_master_identity_ambiguous")
        else:
            mt_row = matches[0]
            source_cik = mt_row.get("comp_cik")
            if mt_row.get("ticker") != row.get("ticker"):
                pre_reasons.append("zacks_master_ticker_mismatch")
            if mt_row.get("currency_code") != row.get("currency_code"):
                pre_reasons.append("zacks_master_currency_mismatch")
        records.append(
            {
                "locator": item["locator"],
                "provider_record_id": None,
                "provider_security_id": row.get("m_ticker"),
                "ticker": row.get("ticker"),
                "cik": source_cik,
                "fiscal_period_end": row.get("per_end_date"),
                "fiscal_period_type": row.get("per_type"),
                "provider_as_of_date": row.get("obs_date"),
                "trusted_available_at_utc": None,
                "availability_precision": "date",
                "consensus_value": row.get("eps_mean_est"),
                "analyst_count": row.get("eps_cnt_est"),
                "currency_code": row.get("currency_code"),
                "pre_reasons": pre_reasons,
            }
        )
    all_pages = eeh_pages + mt_pages
    captured = max(page["captured_at_utc"] for page in all_pages)
    query = payload["tables"]["ZACKS/EEH"]["canonical_request"]
    scoped_ids = _zacks_scoped_event_ids(
        universe=universe,
        identities=identity_snapshot["payload"]["identities"],
        mt_rows=mt_rows,
        params=query["params"],
    )
    acquisition = _receipt(
        captured_at=captured,
        scope_kind="expected_event_partition",
        query_sha256=content_hash(query),
        expected_event_ids=scoped_ids,
        mode="cursor",
        pages=eeh_pages,
    )
    evidence_class = {
        "development_sample": "development_sample",
        "historical_replication": "historical_reconstruction",
        "prospective_signal": "prospective_signal",
    }[payload["evidence_class"]]
    return {
        "descriptor": {
            "format": "zacks_pead_snapshot_json",
            "bytes_sha256": bytes_sha256,
            "byte_count": len(raw),
            "content_artifact_sha256": snapshot.artifact_hash,
            "selected_record_set": "ZACKS/EEH",
            "source_blockers": sorted(set(payload["coverage"]["blockers"])),
        },
        "source": {
            "provider_id": manifest["payload"]["provider_id"],
            "dataset_id": manifest["payload"]["dataset_id"],
            "source_manifest_sha256": manifest["artifact_hash"],
            "captured_at_utc": captured,
            "provider_snapshot_at_utc": manifest["payload"]["provider_snapshot_at_utc"],
        },
        "evidence_class": evidence_class,
        "receipt": acquisition,
        "records": records,
    }


def _generic_json_records(raw: bytes, *, records_path: str) -> list[Mapping[str, Any]]:
    document = _strict_raw_json(raw, label="provider-neutral raw JSON")
    if records_path == "$":
        records = document
    else:
        if not isinstance(document, Mapping) or "records" not in document:
            raise PeadConsensusReplayError("provider-neutral JSON lacks $.records")
        records = document["records"]
    if not isinstance(records, list):
        raise PeadConsensusReplayError("provider-neutral JSON records must be an array")
    if any(not isinstance(record, Mapping) for record in records):
        raise PeadConsensusReplayError("provider-neutral JSON records must be objects")
    return records


def _generic_csv_records(raw: bytes) -> list[Mapping[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadConsensusReplayError("provider-neutral CSV must be UTF-8") from exc
    if text.startswith("\ufeff") or "\x00" in text:
        raise PeadConsensusReplayError("provider-neutral CSV contains a BOM or NUL")
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise PeadConsensusReplayError("provider-neutral CSV is malformed") from exc
    if not rows:
        raise PeadConsensusReplayError("provider-neutral CSV is empty")
    header = rows[0]
    if (
        not header
        or any(not name or name != name.strip() for name in header)
        or len(header) != len(set(header))
    ):
        raise PeadConsensusReplayError("provider-neutral CSV header is invalid")
    records: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows[1:]):
        if len(row) != len(header):
            raise PeadConsensusReplayError(f"provider-neutral CSV row {index} has the wrong width")
        records.append(dict(zip(header, row, strict=True)))
    return records


def _extract_generic(
    *,
    raw: bytes,
    bytes_sha256: str,
    universe: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = manifest["payload"]
    adapter = payload["adapter_id"]
    records = (
        _generic_json_records(raw, records_path=payload["records_path"])
        if adapter == PROVIDER_NEUTRAL_JSON_ADAPTER
        else _generic_csv_records(raw)
    )
    fields = payload["field_map"]
    required_columns = set(fields.values())
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        missing = required_columns - set(row)
        if missing:
            raise PeadConsensusReplayError(
                f"provider-neutral record {index} lacks mapped columns {sorted(missing)}"
            )
        normalized.append(
            {
                "locator": _locator(
                    artifact_hash=bytes_sha256,
                    container_hash=bytes_sha256,
                    member_name=payload["records_path"],
                    page_sequence=1,
                    row_index=index,
                ),
                **{canonical: row[column] for canonical, column in fields.items()},
                "pre_reasons": [],
            }
        )
    page = {
        "sequence": 1,
        "request_sha256": payload["canonical_query_sha256"],
        "raw_response_sha256": bytes_sha256,
        "raw_response_bytes": len(raw),
        "continuation_token_sha256": None,
        "captured_at_utc": payload["source_captured_at_utc"],
    }
    acquisition = _receipt(
        captured_at=payload["source_captured_at_utc"],
        scope_kind="full_export",
        query_sha256=payload["canonical_query_sha256"],
        expected_event_ids=universe["payload"]["expected_event_ids"],
        mode="bulk_file",
        pages=[page],
    )
    return {
        "descriptor": {
            "format": "json" if adapter == PROVIDER_NEUTRAL_JSON_ADAPTER else "csv",
            "bytes_sha256": bytes_sha256,
            "byte_count": len(raw),
            "content_artifact_sha256": bytes_sha256,
            "selected_record_set": payload["records_path"],
            "source_blockers": [],
        },
        "source": {
            "provider_id": payload["provider_id"],
            "dataset_id": payload["dataset_id"],
            "source_manifest_sha256": manifest["artifact_hash"],
            "captured_at_utc": payload["source_captured_at_utc"],
            "provider_snapshot_at_utc": payload["provider_snapshot_at_utc"],
        },
        "evidence_class": payload["evidence_class"],
        "receipt": acquisition,
        "records": normalized,
    }


def _metric_from_profile(profile: Mapping[str, Any], currency: str) -> dict[str, str]:
    metric = profile["payload"]["metric"]
    return {
        "metric_id": metric["metric_id"],
        "accounting_basis": metric["accounting_basis"],
        "per_share_basis": metric["per_share_basis"],
        "scope": metric["scope"],
        "canonical_share_basis": metric["canonical_share_basis"],
        "currency_code": currency,
        "unit": metric["unit"],
        "metric_definition_sha256": metric["metric_definition_sha256"],
    }


def _normalized_availability(row: Mapping[str, Any]) -> tuple[str | None, str]:
    precision = _row_text(row["availability_precision"], "availability_precision_invalid")
    if precision not in _AVAILABILITY_PRECISIONS:
        raise _RowError("availability_precision_invalid")
    raw_timestamp = row["trusted_available_at_utc"]
    if precision == "date":
        if raw_timestamp not in {None, ""}:
            raise _RowError("availability_timestamp_contradicts_precision")
        return None, precision
    try:
        timestamp, parsed = _utc(raw_timestamp, "trusted_available_at_utc")
    except PeadConsensusReplayError as exc:
        raise _RowError("availability_timestamp_invalid") from exc
    actual = "microsecond" if parsed.microsecond else "second"
    if actual != precision:
        raise _RowError("availability_timestamp_contradicts_precision")
    return timestamp, precision


def _ledger_base(row: Mapping[str, Any]) -> dict[str, Any]:
    locator = row["locator"]
    provider_record = row.get("provider_record_id")
    provider_security = row.get("provider_security_id")
    return {
        "raw_record_sha256": content_hash(locator),
        "source_locator": locator,
        "provider_record_id": (
            provider_record
            if isinstance(provider_record, str)
            and provider_record
            and provider_record == provider_record.strip()
            else None
        ),
        "provider_security_id": (
            provider_security
            if isinstance(provider_security, str)
            and provider_security
            and provider_security == provider_security.strip()
            else None
        ),
        "event_id": None,
        "disposition": "invalid_record",
        "reasons": [],
        "normalized_vintage_sha256": None,
    }


def _normalize_records(
    *,
    records: Sequence[Mapping[str, Any]],
    universe: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    metric_profile: Mapping[str, Any],
    receipt_hash: str,
    receipt_captured_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_ids = set(universe["payload"]["expected_event_ids"])
    identities = identity_snapshot["payload"]["identities"]
    _, receipt_capture = _utc(receipt_captured_at, "receipt_captured_at")
    ledger: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for raw_row in records:
        entry = _ledger_base(raw_row)
        ledger.append(entry)
        event_id: str | None = None
        try:
            provider_record_id = raw_row.get("provider_record_id")
            if provider_record_id is not None:
                provider_record_id = _row_text(provider_record_id, "provider_record_id_invalid")
                entry["provider_record_id"] = provider_record_id
            provider_security_id = _row_text(
                raw_row.get("provider_security_id"), "provider_security_id_invalid"
            )
            ticker = _row_text(raw_row.get("ticker"), "ticker_invalid", upper=True)
            cik = _row_cik(raw_row.get("cik"))
            period = _row_day(raw_row.get("fiscal_period_end"), "fiscal_period_end_invalid")
            if raw_row.get("fiscal_period_type") != "Q":
                raise _RowError("fiscal_period_type_invalid")
            event_id = canonical_event_id(
                {"cik": cik, "fiscal_period_end": period, "fiscal_period_type": "Q"}
            )
            entry["provider_security_id"] = provider_security_id
            if event_id not in expected_ids:
                entry["disposition"] = "outside_event_universe"
                entry["reasons"] = []
                continue
            entry["event_id"] = event_id
            pre_reasons = raw_row.get("pre_reasons")
            if pre_reasons:
                entry["reasons"] = sorted(set(pre_reasons))
                continue
            matches = _matching_identities(identities, cik=cik, ticker=ticker, period=period)
            if len(matches) != 1:
                entry["disposition"] = "identity_gap"
                entry["reasons"] = [
                    "dated_identity_missing" if not matches else "dated_identity_ambiguous"
                ]
                continue
            currency = _row_currency(raw_row.get("currency_code"))
            identity_currency = matches[0]["currency"].upper()
            if currency != identity_currency:
                raise _RowError("identity_currency_mismatch")
            as_of = _row_day(raw_row.get("provider_as_of_date"), "provider_as_of_date_invalid")
            available, precision = _normalized_availability(raw_row)
            if (
                available is not None
                and _utc(available, "trusted_available_at_utc")[1] > receipt_capture
            ):
                raise _RowError("availability_after_acquisition")
            vintage = {
                "provider_as_of_date": as_of,
                "trusted_available_at_utc": available,
                "availability_precision": precision,
                "consensus_value": _decimal(raw_row.get("consensus_value")),
                "analyst_count": _positive_integer(raw_row.get("analyst_count")),
                "raw_record_sha256": entry["raw_record_sha256"],
                "acquisition_receipt_sha256": receipt_hash,
                "metric": _metric_from_profile(metric_profile, currency),
            }
            entry["disposition"] = "matched_expected_event"
            entry["reasons"] = []
            entry["normalized_vintage_sha256"] = content_hash(vintage)
            candidates.append(
                {
                    "event_id": event_id,
                    "provider_record_id": provider_record_id,
                    "vintage": vintage,
                    "ledger": entry,
                }
            )
        except _RowError as exc:
            entry["event_id"] = event_id
            entry["disposition"] = "invalid_record"
            entry["reasons"] = [exc.reason]

    duplicate_candidates: set[int] = set()
    by_provider_record: dict[str, list[int]] = defaultdict(list)
    by_natural_vintage: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        provider_record_id = candidate["provider_record_id"]
        if provider_record_id is not None:
            by_provider_record[provider_record_id].append(index)
        vintage = candidate["vintage"]
        by_natural_vintage[
            (
                candidate["event_id"],
                vintage["provider_as_of_date"],
                vintage["trusted_available_at_utc"],
                vintage["availability_precision"],
                vintage["metric"]["metric_definition_sha256"],
            )
        ].append(index)
    duplicate_record_ids = {
        index for indices in by_provider_record.values() if len(indices) > 1 for index in indices
    }
    duplicate_natural_ids = {
        index for indices in by_natural_vintage.values() if len(indices) > 1 for index in indices
    }
    duplicate_candidates.update(duplicate_record_ids)
    duplicate_candidates.update(duplicate_natural_ids)
    accepted: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if index not in duplicate_candidates:
            accepted.append(candidate)
            continue
        reasons: list[str] = []
        if index in duplicate_record_ids:
            reasons.append("provider_record_id_duplicate")
        if index in duplicate_natural_ids:
            reasons.append("natural_vintage_duplicate")
        candidate["ledger"]["disposition"] = "duplicate_natural_vintage"
        candidate["ledger"]["reasons"] = sorted(reasons)

    vintages_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in accepted:
        vintages_by_event[candidate["event_id"]].append(candidate["vintage"])
    event_records: list[dict[str, Any]] = []
    for event_id in universe["payload"]["expected_event_ids"]:
        vintages = vintages_by_event.get(event_id, [])
        if vintages:
            event_records.append(
                {
                    "event_id": event_id,
                    "disposition": "available",
                    "missing_reason": None,
                    "vintages": vintages,
                }
            )
            continue
        related = [entry for entry in ledger if entry["event_id"] == event_id]
        dispositions = {entry["disposition"] for entry in related}
        if "identity_gap" in dispositions:
            reason = "identity_binding_unresolved"
        elif "invalid_record" in dispositions:
            reason = "raw_consensus_record_invalid"
        elif "duplicate_natural_vintage" in dispositions:
            reason = "duplicate_raw_consensus_vintage"
        else:
            reason = "provider_record_absent"
        event_records.append(
            {
                "event_id": event_id,
                "disposition": "missing",
                "missing_reason": reason,
                "vintages": [],
            }
        )
    ledger.sort(key=lambda entry: entry["raw_record_sha256"])
    return ledger, event_records


def _counts(ledger: Sequence[Mapping[str, Any]], consensus: Mapping[str, Any]) -> dict[str, int]:
    dispositions = Counter(entry["disposition"] for entry in ledger)
    events = consensus["payload"]["event_records"]
    available = sum(event["disposition"] == "available" for event in events)
    return {
        "raw_record_count": len(ledger),
        "ledger_record_count": len(ledger),
        "matched_record_count": dispositions["matched_expected_event"],
        "outside_universe_record_count": dispositions["outside_event_universe"],
        "identity_gap_record_count": dispositions["identity_gap"],
        "invalid_record_count": dispositions["invalid_record"],
        "duplicate_record_count": dispositions["duplicate_natural_vintage"],
        "expected_event_count": len(events),
        "available_event_count": available,
        "missing_event_count": len(events) - available,
        "normalized_vintage_count": sum(len(event["vintages"]) for event in events),
    }


def _blockers(
    *,
    trust_policy: Mapping[str, Any],
    raw_descriptor: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    consensus: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    trust_names = {
        "event_universe_trusted": "event_universe_not_trusted",
        "identity_snapshot_trusted": "identity_snapshot_not_trusted",
        "source_manifest_trusted": "source_manifest_not_trusted",
        "metric_profile_trusted": "metric_profile_not_trusted",
        "raw_artifact_trusted": "raw_artifact_not_trusted",
    }
    blockers.extend(
        blocker for field, blocker in trust_names.items() if trust_policy[field] is not True
    )
    if identity_snapshot["payload"]["qualification_allowed"] is not True:
        blockers.append("identity_snapshot_not_qualified")
    dispositions = Counter(entry["disposition"] for entry in ledger)
    if dispositions["identity_gap"]:
        blockers.append("raw_identity_gaps_present")
    if dispositions["invalid_record"]:
        blockers.append("raw_invalid_records_present")
    if dispositions["duplicate_natural_vintage"]:
        blockers.append("raw_duplicate_vintages_present")
    blockers.extend(f"raw_source:{blocker}" for blocker in raw_descriptor["source_blockers"])
    blockers.extend(
        f"consensus:{blocker}" for blocker in consensus["payload"]["coverage"]["blockers"]
    )
    return sorted(set(blockers))


def _trust_policy(
    *,
    universe_hash: str,
    identity_hash: str,
    source_hash: str,
    metric_hash: str,
    raw_artifact_hash: str,
    trusted_event_universe_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_identity_snapshot_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_source_manifest_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_metric_profile_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_raw_artifact_sha256s: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, Any]:
    universe_allowlist = _sorted_hashes(
        trusted_event_universe_sha256s, "trusted_event_universe_sha256s"
    )
    identity_allowlist = _sorted_hashes(
        trusted_identity_snapshot_sha256s, "trusted_identity_snapshot_sha256s"
    )
    source_allowlist = _sorted_hashes(
        trusted_source_manifest_sha256s, "trusted_source_manifest_sha256s"
    )
    metric_allowlist = _sorted_hashes(
        trusted_metric_profile_sha256s, "trusted_metric_profile_sha256s"
    )
    raw_allowlist = _sorted_hashes(trusted_raw_artifact_sha256s, "trusted_raw_artifact_sha256s")
    return {
        "event_universe_allowlist_sha256": content_hash(universe_allowlist),
        "identity_snapshot_allowlist_sha256": content_hash(identity_allowlist),
        "source_manifest_allowlist_sha256": content_hash(source_allowlist),
        "metric_profile_allowlist_sha256": content_hash(metric_allowlist),
        "raw_artifact_allowlist_sha256": content_hash(raw_allowlist),
        "event_universe_trusted": universe_hash in universe_allowlist,
        "identity_snapshot_trusted": identity_hash in identity_allowlist,
        "source_manifest_trusted": source_hash in source_allowlist,
        "metric_profile_trusted": metric_hash in metric_allowlist,
        "raw_artifact_trusted": raw_artifact_hash in raw_allowlist,
    }


def build_pead_consensus_replay(
    *,
    raw_artifact: bytes,
    event_universe: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    metric_profile: Mapping[str, Any],
    trusted_event_universe_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_identity_snapshot_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_source_manifest_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_metric_profile_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_raw_artifact_sha256s: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, Any]:
    """Replay exact provider bytes into one content-addressed consensus receipt."""
    raw, bytes_sha256 = _raw_artifact_bytes(raw_artifact)
    try:
        universe = validate_pead_event_universe(event_universe)
    except PeadEventUniverseError as exc:
        raise PeadConsensusReplayError("event universe is invalid") from exc
    identity = _validate_identity_snapshot(identity_snapshot)
    manifest = validate_pead_consensus_source_manifest(source_manifest)
    profile = validate_pead_consensus_metric_profile(metric_profile)
    candidate = universe["payload"]["candidate_id"]
    for label, document in (
        ("identity snapshot", identity),
        ("source manifest", manifest),
        ("metric profile", profile),
    ):
        if document["payload"]["candidate_id"] != candidate:
            raise PeadConsensusReplayError(f"{label} belongs to another candidate")
    if identity["artifact_hash"] != universe["payload"]["bindings"]["identity_snapshot_sha256"]:
        raise PeadConsensusReplayError("identity snapshot differs from universe binding")
    for field in ("adapter_id", "provider_id", "dataset_id"):
        if profile["payload"][field] != manifest["payload"][field]:
            raise PeadConsensusReplayError(f"metric profile and source manifest {field} differ")
    adapter = manifest["payload"]["adapter_id"]
    if adapter != ZACKS_EEH_ADAPTER:
        field_map = manifest["payload"]["field_map"]
        if (
            profile["payload"]["value_field"] != field_map["consensus_value"]
            or profile["payload"]["analyst_count_field"] != field_map["analyst_count"]
        ):
            raise PeadConsensusReplayError(
                "generic metric profile does not bind the mapped raw value/count fields"
            )
    extracted = (
        _extract_zacks(
            raw=raw,
            bytes_sha256=bytes_sha256,
            universe=universe,
            identity_snapshot=identity,
            manifest=manifest,
        )
        if adapter == ZACKS_EEH_ADAPTER
        else _extract_generic(
            raw=raw,
            bytes_sha256=bytes_sha256,
            universe=universe,
            manifest=manifest,
        )
    )
    ledger, event_records = _normalize_records(
        records=extracted["records"],
        universe=universe,
        identity_snapshot=identity,
        metric_profile=profile,
        receipt_hash=extracted["receipt"]["receipt_sha256"],
        receipt_captured_at=extracted["receipt"]["source_captured_at_utc"],
    )
    try:
        consensus = build_pead_consensus_evidence(
            candidate_id=candidate,
            evidence_class=extracted["evidence_class"],
            event_universe=universe,
            source=extracted["source"],
            acquisition_receipts=[extracted["receipt"]],
            event_records=event_records,
        )
    except PeadConsensusEvidenceError as exc:
        raise PeadConsensusReplayError("derived consensus evidence is invalid") from exc
    trust = _trust_policy(
        universe_hash=universe["artifact_hash"],
        identity_hash=identity["artifact_hash"],
        source_hash=manifest["artifact_hash"],
        metric_hash=profile["artifact_hash"],
        raw_artifact_hash=bytes_sha256,
        trusted_event_universe_sha256s=trusted_event_universe_sha256s,
        trusted_identity_snapshot_sha256s=trusted_identity_snapshot_sha256s,
        trusted_source_manifest_sha256s=trusted_source_manifest_sha256s,
        trusted_metric_profile_sha256s=trusted_metric_profile_sha256s,
        trusted_raw_artifact_sha256s=trusted_raw_artifact_sha256s,
    )
    blockers = _blockers(
        trust_policy=trust,
        raw_descriptor=extracted["descriptor"],
        identity_snapshot=identity,
        ledger=ledger,
        consensus=consensus,
    )
    payload = {
        "schema_version": CONSENSUS_REPLAY_SCHEMA_VERSION,
        "candidate_id": candidate,
        "adapter_id": adapter,
        "bindings": {
            "event_universe_sha256": universe["artifact_hash"],
            "identity_snapshot_sha256": identity["artifact_hash"],
            "source_manifest_sha256": manifest["artifact_hash"],
            "metric_profile_sha256": profile["artifact_hash"],
            "raw_artifact_bytes_sha256": bytes_sha256,
            "consensus_evidence_sha256": consensus["artifact_hash"],
        },
        "trust_policy": trust,
        "raw_artifact": extracted["descriptor"],
        "source_manifest": manifest,
        "metric_profile": profile,
        "identity_snapshot": identity,
        "raw_record_ledger": ledger,
        "counts": _counts(ledger, consensus),
        "blockers": blockers,
        "qualification_allowed": not blockers,
        "consensus_evidence": consensus,
    }
    return validate_pead_consensus_replay(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def _normalized_trust_policy(value: Any) -> dict[str, Any]:
    policy = _exact(value, _TRUST_FIELDS, "trust_policy")
    normalized: dict[str, Any] = {}
    for field in (
        "event_universe_allowlist_sha256",
        "identity_snapshot_allowlist_sha256",
        "source_manifest_allowlist_sha256",
        "metric_profile_allowlist_sha256",
        "raw_artifact_allowlist_sha256",
    ):
        normalized[field] = _sha(policy[field], f"trust_policy.{field}")
    for field in (
        "event_universe_trusted",
        "identity_snapshot_trusted",
        "source_manifest_trusted",
        "metric_profile_trusted",
        "raw_artifact_trusted",
    ):
        if type(policy[field]) is not bool:
            raise PeadConsensusReplayError(f"trust_policy.{field} must be boolean")
        normalized[field] = policy[field]
    return normalized


def _normalized_raw_descriptor(value: Any) -> dict[str, Any]:
    descriptor = _exact(value, _RAW_ARTIFACT_FIELDS, "raw_artifact")
    raw_format = descriptor["format"]
    if raw_format not in {"zacks_pead_snapshot_json", "json", "csv"}:
        raise PeadConsensusReplayError("raw_artifact.format is unsupported")
    byte_count = descriptor["byte_count"]
    if type(byte_count) is not int or byte_count < 1 or byte_count > MAX_RAW_ARTIFACT_BYTES:
        raise PeadConsensusReplayError("raw_artifact.byte_count is invalid")
    source_blockers = descriptor["source_blockers"]
    if not isinstance(source_blockers, list) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in source_blockers
    ):
        raise PeadConsensusReplayError("raw_artifact.source_blockers are invalid")
    if source_blockers != sorted(set(source_blockers)):
        raise PeadConsensusReplayError("raw_artifact.source_blockers must be sorted and unique")
    return {
        "format": raw_format,
        "bytes_sha256": _sha(descriptor["bytes_sha256"], "raw_artifact.bytes_sha256"),
        "byte_count": byte_count,
        "content_artifact_sha256": _sha(
            descriptor["content_artifact_sha256"],
            "raw_artifact.content_artifact_sha256",
        ),
        "selected_record_set": _text(
            descriptor["selected_record_set"], "raw_artifact.selected_record_set"
        ),
        "source_blockers": list(source_blockers),
    }


def _normalized_locator(value: Any, *, raw_artifact_sha256: str, label: str) -> dict[str, Any]:
    locator = _exact(value, _LOCATOR_FIELDS, label)
    artifact = _sha(locator["raw_artifact_bytes_sha256"], f"{label}.raw_artifact")
    if artifact != raw_artifact_sha256:
        raise PeadConsensusReplayError("raw locator references another artifact")
    page = locator["page_sequence"]
    row = locator["row_index"]
    if type(page) is not int or page < 1:
        raise PeadConsensusReplayError(f"{label}.page_sequence is invalid")
    if type(row) is not int or row < 0:
        raise PeadConsensusReplayError(f"{label}.row_index is invalid")
    return {
        "raw_artifact_bytes_sha256": artifact,
        "container_sha256": _sha(locator["container_sha256"], f"{label}.container"),
        "member_name": _text(locator["member_name"], f"{label}.member_name"),
        "page_sequence": page,
        "row_index": row,
    }


def _normalized_ledger(
    value: Any,
    *,
    raw_artifact_sha256: str,
    expected_ids: set[str],
    consensus: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PeadConsensusReplayError("raw_record_ledger must be an array")
    vintage_by_raw: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for event in consensus["payload"]["event_records"]:
        for vintage in event["vintages"]:
            vintage_by_raw[vintage["raw_record_sha256"]] = (event["event_id"], vintage)
    ledger: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(value):
        entry = _exact(raw_entry, _LEDGER_FIELDS, f"raw_record_ledger[{index}]")
        locator = _normalized_locator(
            entry["source_locator"],
            raw_artifact_sha256=raw_artifact_sha256,
            label=f"raw_record_ledger[{index}].source_locator",
        )
        raw_hash = _sha(entry["raw_record_sha256"], f"raw_record_ledger[{index}].raw_record_sha256")
        if raw_hash != content_hash(locator):
            raise PeadConsensusReplayError("raw_record_sha256 does not match its locator")
        provider_record = _optional_text(
            entry["provider_record_id"], f"raw_record_ledger[{index}].provider_record_id"
        )
        provider_security = _optional_text(
            entry["provider_security_id"],
            f"raw_record_ledger[{index}].provider_security_id",
        )
        event_id = entry["event_id"]
        if event_id is not None:
            event_id = _sha(event_id, f"raw_record_ledger[{index}].event_id")
            if event_id not in expected_ids:
                raise PeadConsensusReplayError("raw ledger event is outside frozen universe")
        disposition = entry["disposition"]
        if disposition not in _DISPOSITIONS:
            raise PeadConsensusReplayError("raw ledger disposition is unsupported")
        reasons = _machine_reasons(entry["reasons"], f"raw_record_ledger[{index}].reasons")
        normalized_hash = entry["normalized_vintage_sha256"]
        if normalized_hash is not None:
            normalized_hash = _sha(
                normalized_hash,
                f"raw_record_ledger[{index}].normalized_vintage_sha256",
            )
        if disposition == "matched_expected_event":
            if event_id is None or reasons or normalized_hash is None:
                raise PeadConsensusReplayError("matched raw ledger record is incomplete")
            match = vintage_by_raw.get(raw_hash)
            if match is None or match[0] != event_id:
                raise PeadConsensusReplayError(
                    "matched raw ledger record has no derived consensus vintage"
                )
            if content_hash(match[1]) != normalized_hash:
                raise PeadConsensusReplayError("normalized vintage hash mismatch")
        elif disposition == "outside_event_universe":
            if event_id is not None or reasons or normalized_hash is not None:
                raise PeadConsensusReplayError("outside-universe ledger record is inconsistent")
        elif disposition in {"identity_gap", "invalid_record"}:
            if not reasons or normalized_hash is not None:
                raise PeadConsensusReplayError("failed raw ledger record is inconsistent")
        elif disposition == "duplicate_natural_vintage":
            if event_id is None or not reasons or normalized_hash is None:
                raise PeadConsensusReplayError("duplicate raw ledger record is inconsistent")
        ledger.append(
            {
                "raw_record_sha256": raw_hash,
                "source_locator": locator,
                "provider_record_id": provider_record,
                "provider_security_id": provider_security,
                "event_id": event_id,
                "disposition": disposition,
                "reasons": reasons,
                "normalized_vintage_sha256": normalized_hash,
            }
        )
    hashes = [entry["raw_record_sha256"] for entry in ledger]
    if hashes != sorted(set(hashes)):
        raise PeadConsensusReplayError("raw_record_ledger must be sorted and unique")
    matched_hashes = {
        entry["raw_record_sha256"]
        for entry in ledger
        if entry["disposition"] == "matched_expected_event"
    }
    if matched_hashes != set(vintage_by_raw):
        raise PeadConsensusReplayError(
            "raw ledger and normalized consensus vintages do not account exactly"
        )
    return ledger


def validate_pead_consensus_replay(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate receipt structure; external raw bytes still require ``verify``."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "consensus replay")
    payload = _exact(wrapper["payload"], _REPLAY_FIELDS, "consensus replay.payload")
    claimed = _sha(wrapper["artifact_hash"], "consensus replay.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadConsensusReplayError("consensus replay artifact hash mismatch")
    if payload["schema_version"] != CONSENSUS_REPLAY_SCHEMA_VERSION:
        raise PeadConsensusReplayError("unsupported consensus replay schema")
    manifest = validate_pead_consensus_source_manifest(payload["source_manifest"])
    profile = validate_pead_consensus_metric_profile(payload["metric_profile"])
    identity = _validate_identity_snapshot(payload["identity_snapshot"])
    try:
        consensus = validate_pead_consensus_evidence(payload["consensus_evidence"])
    except PeadConsensusEvidenceError as exc:
        raise PeadConsensusReplayError("embedded consensus evidence is invalid") from exc
    universe = consensus["payload"]["event_universe"]
    candidate = _text(payload["candidate_id"], "consensus replay.candidate_id")
    adapter = payload["adapter_id"]
    if adapter not in SUPPORTED_ADAPTERS:
        raise PeadConsensusReplayError("consensus replay adapter is not registered")
    for label, value in (
        ("consensus evidence", consensus["payload"]["candidate_id"]),
        ("source manifest", manifest["payload"]["candidate_id"]),
        ("metric profile", profile["payload"]["candidate_id"]),
        ("identity snapshot", identity["payload"]["candidate_id"]),
    ):
        if value != candidate:
            raise PeadConsensusReplayError(f"{label} belongs to another candidate")
    if manifest["payload"]["adapter_id"] != adapter or profile["payload"]["adapter_id"] != adapter:
        raise PeadConsensusReplayError("embedded adapter identities differ")
    for field in ("provider_id", "dataset_id"):
        if profile["payload"][field] != manifest["payload"][field]:
            raise PeadConsensusReplayError(
                f"embedded metric profile and source manifest {field} differ"
            )
        if consensus["payload"]["source"][field] != manifest["payload"][field]:
            raise PeadConsensusReplayError(f"embedded consensus and source manifest {field} differ")
    if consensus["payload"]["source"]["source_manifest_sha256"] != manifest["artifact_hash"]:
        raise PeadConsensusReplayError("embedded consensus source manifest binding differs")
    if adapter != ZACKS_EEH_ADAPTER:
        field_map = manifest["payload"]["field_map"]
        if (
            profile["payload"]["value_field"] != field_map["consensus_value"]
            or profile["payload"]["analyst_count_field"] != field_map["analyst_count"]
            or consensus["payload"]["evidence_class"] != manifest["payload"]["evidence_class"]
        ):
            raise PeadConsensusReplayError("generic source/profile/evidence interpretation differs")
        for field in ("captured_at_utc", "provider_snapshot_at_utc"):
            manifest_field = "source_captured_at_utc" if field == "captured_at_utc" else field
            if consensus["payload"]["source"][field] != manifest["payload"][manifest_field]:
                raise PeadConsensusReplayError(
                    f"generic consensus source {field} differs from manifest"
                )
    profile_metric = profile["payload"]["metric"]
    for event in consensus["payload"]["event_records"]:
        for vintage in event["vintages"]:
            for field in _PROFILE_METRIC_FIELDS:
                if vintage["metric"][field] != profile_metric[field]:
                    raise PeadConsensusReplayError(
                        f"consensus vintage metric {field} differs from profile"
                    )
    bindings_raw = _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    bindings = {field: _sha(bindings_raw[field], f"bindings.{field}") for field in _BINDING_FIELDS}
    expected_bindings = {
        "event_universe_sha256": universe["artifact_hash"],
        "identity_snapshot_sha256": identity["artifact_hash"],
        "source_manifest_sha256": manifest["artifact_hash"],
        "metric_profile_sha256": profile["artifact_hash"],
        "raw_artifact_bytes_sha256": bindings["raw_artifact_bytes_sha256"],
        "consensus_evidence_sha256": consensus["artifact_hash"],
    }
    if bindings != expected_bindings:
        raise PeadConsensusReplayError("consensus replay bindings are inconsistent")
    if identity["artifact_hash"] != universe["payload"]["bindings"]["identity_snapshot_sha256"]:
        raise PeadConsensusReplayError("identity and universe binding differ")
    trust = _normalized_trust_policy(payload["trust_policy"])
    raw_descriptor = _normalized_raw_descriptor(payload["raw_artifact"])
    if raw_descriptor["bytes_sha256"] != bindings["raw_artifact_bytes_sha256"]:
        raise PeadConsensusReplayError("raw artifact descriptor and binding differ")
    expected_format = {
        ZACKS_EEH_ADAPTER: "zacks_pead_snapshot_json",
        PROVIDER_NEUTRAL_JSON_ADAPTER: "json",
        PROVIDER_NEUTRAL_CSV_ADAPTER: "csv",
    }[adapter]
    if raw_descriptor["format"] != expected_format:
        raise PeadConsensusReplayError("raw artifact format contradicts adapter")
    expected_record_set = (
        "ZACKS/EEH" if adapter == ZACKS_EEH_ADAPTER else manifest["payload"]["records_path"]
    )
    if raw_descriptor["selected_record_set"] != expected_record_set:
        raise PeadConsensusReplayError("raw selected record set contradicts adapter")
    ledger = _normalized_ledger(
        payload["raw_record_ledger"],
        raw_artifact_sha256=raw_descriptor["bytes_sha256"],
        expected_ids=set(universe["payload"]["expected_event_ids"]),
        consensus=consensus,
    )
    counts = _counts(ledger, consensus)
    _exact(payload["counts"], _COUNT_FIELDS, "counts")
    if payload["counts"] != counts:
        raise PeadConsensusReplayError("consensus replay counts are not derived exactly")
    blockers = _blockers(
        trust_policy=trust,
        raw_descriptor=raw_descriptor,
        identity_snapshot=identity,
        ledger=ledger,
        consensus=consensus,
    )
    if payload["blockers"] != blockers:
        raise PeadConsensusReplayError("consensus replay blockers are not derived exactly")
    if type(payload["qualification_allowed"]) is not bool or payload[
        "qualification_allowed"
    ] is not (not blockers):
        raise PeadConsensusReplayError("consensus replay qualification claim is inconsistent")
    normalized = {
        "schema_version": CONSENSUS_REPLAY_SCHEMA_VERSION,
        "candidate_id": candidate,
        "adapter_id": adapter,
        "bindings": bindings,
        "trust_policy": trust,
        "raw_artifact": raw_descriptor,
        "source_manifest": manifest,
        "metric_profile": profile,
        "identity_snapshot": identity,
        "raw_record_ledger": ledger,
        "counts": counts,
        "blockers": blockers,
        "qualification_allowed": not blockers,
        "consensus_evidence": consensus,
    }
    if content_hash(normalized) != claimed:
        raise PeadConsensusReplayError("consensus replay is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized)}


def verify_pead_consensus_replay(
    document: Mapping[str, Any],
    *,
    raw_artifact: bytes,
    trusted_event_universe_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_identity_snapshot_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_source_manifest_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_metric_profile_sha256s: Sequence[str] | set[str] | frozenset[str],
    trusted_raw_artifact_sha256s: Sequence[str] | set[str] | frozenset[str],
) -> dict[str, Any]:
    """Authoritatively reopen raw bytes and reproduce the complete receipt."""
    validated = validate_pead_consensus_replay(document)
    payload = validated["payload"]
    rebuilt = build_pead_consensus_replay(
        raw_artifact=raw_artifact,
        event_universe=payload["consensus_evidence"]["payload"]["event_universe"],
        identity_snapshot=payload["identity_snapshot"],
        source_manifest=payload["source_manifest"],
        metric_profile=payload["metric_profile"],
        trusted_event_universe_sha256s=trusted_event_universe_sha256s,
        trusted_identity_snapshot_sha256s=trusted_identity_snapshot_sha256s,
        trusted_source_manifest_sha256s=trusted_source_manifest_sha256s,
        trusted_metric_profile_sha256s=trusted_metric_profile_sha256s,
        trusted_raw_artifact_sha256s=trusted_raw_artifact_sha256s,
    )
    if canonical_json(rebuilt) != canonical_json(validated):
        raise PeadConsensusReplayError(
            "consensus replay does not reproduce from trusted raw bytes and allowlists"
        )
    return validated


def _read_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadConsensusReplayError(f"consensus replay is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_REPLAY_RECEIPT_BYTES:
        raise PeadConsensusReplayError("consensus replay exceeds its size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadConsensusReplayError("consensus replay is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadConsensusReplayError(f"consensus replay contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadConsensusReplayError(f"consensus replay contains invalid number {token}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except PeadConsensusReplayError:
        raise
    except json.JSONDecodeError as exc:
        raise PeadConsensusReplayError(
            f"invalid consensus replay JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadConsensusReplayError("consensus replay root must be an object")
    return value


def load_pead_consensus_replay(path: str | Path) -> dict[str, Any]:
    """Load structural receipt evidence; call ``verify`` before consumption."""
    return validate_pead_consensus_replay(_read_receipt(Path(path)))


__all__ = [
    "CONSENSUS_METRIC_PROFILE_SCHEMA_VERSION",
    "CONSENSUS_REPLAY_SCHEMA_VERSION",
    "CONSENSUS_SOURCE_MANIFEST_SCHEMA_VERSION",
    "MAX_RAW_ARTIFACT_BYTES",
    "MAX_REPLAY_RECEIPT_BYTES",
    "PROVIDER_NEUTRAL_CSV_ADAPTER",
    "PROVIDER_NEUTRAL_JSON_ADAPTER",
    "PeadConsensusReplayError",
    "SUPPORTED_ADAPTERS",
    "ZACKS_EEH_ADAPTER",
    "build_pead_consensus_metric_profile",
    "build_pead_consensus_replay",
    "build_pead_consensus_source_manifest",
    "canonical_json",
    "content_hash",
    "load_pead_consensus_replay",
    "validate_pead_consensus_metric_profile",
    "validate_pead_consensus_replay",
    "validate_pead_consensus_source_manifest",
    "verify_pead_consensus_replay",
]
