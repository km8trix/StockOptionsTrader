"""Immutable, point-in-time Zacks earnings-announcement acquisitions.

This module is deliberately independent from the Sharadar/SF1 warehouse.  It
talks only to Nasdaq Data Link's ZACKS Tables API, validates every JSON page,
and produces a content-addressed snapshot suitable for the locked PEAD
replication.  Credentials are read only from ``NASDAQ_DATA_LINK_API_KEY`` and
are never included in canonical requests, artifacts, journal entries, or
errors.

The store is create-only.  Snapshot files are named by the SHA-256 of their
canonical payload, while an append-only journal binds snapshots into a
sequence-numbered hash chain.  Re-persisting the exact artifact is idempotent;
no existing evidence file is overwritten.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence


SUPPORTED_TABLES = ("ES", "SS", "EEH", "SEH", "MT", "EA")
CANONICAL_TABLES = tuple(f"ZACKS/{table}" for table in SUPPORTED_TABLES)
SUPPORTED_MODES = ("historical-sample", "historical-full", "prospective")
SNAPSHOT_SCHEMA_VERSION = "zacks_pead_snapshot.v1"
JOURNAL_SCHEMA_VERSION = "zacks_pead_journal.v1"
SOURCE_ID = "nasdaq-data-link-zacks"
DEFAULT_CANDIDATE_ID = "pead-vq-locked-replication-v1"
NASDAQ_API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
TABLE_URL = "https://data.nasdaq.com/api/v3/datatables/ZACKS/{table}.json"
_HEX = frozenset("0123456789abcdef")
_PAYLOAD_FIELDS = {
    "schema_version", "candidate_id", "source_id", "evidence_class",
    "captured_at", "requested_window", "coverage", "tables",
}
_TABLE_FIELDS = {
    "columns", "rows", "canonical_request", "response_sha256",
    "provider_metadata",
}
_COVERAGE_DATE_COLUMNS = {
    "ES": "act_rpt_date",
    "SS": "act_rpt_date",
    "EEH": "obs_date",
    "SEH": "obs_date",
}


class EarningsAnnouncementError(RuntimeError):
    """Base error for the independent Zacks acquisition path."""


class CredentialError(EarningsAnnouncementError):
    """The required environment-only credential is absent."""


class ProviderResponseError(EarningsAnnouncementError):
    """The provider failed or changed its declared response schema."""


class SnapshotIntegrityError(EarningsAnnouncementError):
    """A snapshot or journal record is malformed, conflicting, or tampered."""


def _strict_json_loads(value: str | bytes, *, context: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise SnapshotIntegrityError(f"duplicate {context} JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(token: str) -> None:
        raise SnapshotIntegrityError(f"invalid {context} JSON number: {token}")

    try:
        return json.loads(
            value, object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except SnapshotIntegrityError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(f"invalid {context} JSON") from exc


def _normalize_json(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetimes in earnings evidence must be timezone-aware")
        return canonical_utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("earnings evidence cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("earnings evidence mapping keys must be strings")
            normalized[key] = _normalize_json(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported earnings evidence value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Canonical JSON used for all snapshot, request, and journal hashes."""
    return json.dumps(
        _normalize_json(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SnapshotIntegrityError(f"{field_name} must be a SHA-256 digest")
    candidate = value.lower()
    if len(candidate) != 64 or any(ch not in _HEX for ch in candidate):
        raise SnapshotIntegrityError(f"{field_name} must be a SHA-256 digest")
    if candidate != value:
        raise SnapshotIntegrityError(f"{field_name} must be lowercase")
    return candidate


def canonical_utc_timestamp(value: datetime | str) -> str:
    """Normalize an aware timestamp to canonical RFC-3339 UTC with ``Z``."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("capture timestamp must be valid ISO-8601") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("capture timestamp must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("capture timestamp must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if utc.microsecond else "seconds"
    return utc.isoformat(timespec=timespec).replace("+00:00", "Z")


def _validate_canonical_timestamp(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SnapshotIntegrityError(f"{field_name} must be a canonical UTC timestamp")
    try:
        canonical = canonical_utc_timestamp(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotIntegrityError(
            f"{field_name} must be a canonical UTC timestamp") from exc
    if value != canonical or not value.endswith("Z"):
        raise SnapshotIntegrityError(f"{field_name} is not canonical UTC")
    return value


def _timestamp_value(value: str) -> datetime:
    """Return a comparable aware datetime for an already-canonical timestamp."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotIntegrityError(f"{field_name} is required")
    if value != value.strip():
        raise SnapshotIntegrityError(f"{field_name} must not contain edge whitespace")
    return value


def _iso_date(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise SnapshotIntegrityError(f"{field_name} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SnapshotIntegrityError(f"{field_name} must be YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise SnapshotIntegrityError(f"{field_name} must be canonical YYYY-MM-DD")
    return value


def _normalize_tables(tables: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tables, (str, bytes)) or not isinstance(tables, Sequence):
        raise TypeError("tables must be a sequence")
    normalized = tuple(_short_table_code(table) for table in tables)
    if not normalized:
        raise ValueError("at least one Zacks table is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Zacks tables must be unique")
    return tuple(table for table in SUPPORTED_TABLES if table in normalized)


def _short_table_code(table: str, *, require_canonical: bool = False) -> str:
    if not isinstance(table, str) or table != table.strip():
        raise ValueError("Zacks table code must be a canonical string")
    if require_canonical:
        if not table.startswith("ZACKS/"):
            raise ValueError("snapshot table keys must use canonical ZACKS/ codes")
        candidate = table.removeprefix("ZACKS/")
    else:
        candidate = table.removeprefix("ZACKS/").upper()
    if candidate not in SUPPORTED_TABLES:
        raise ValueError(f"unsupported Zacks table: {table}")
    canonical = f"ZACKS/{candidate}"
    if require_canonical and table != canonical:
        raise ValueError("snapshot table keys must use canonical ZACKS/ codes")
    return candidate


def _canonical_table_code(table: str) -> str:
    return f"ZACKS/{_short_table_code(table)}"


def _request_value(value: Any, *, field_name: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return repr(value)
    if isinstance(value, str) and value and value == value.strip():
        return value
    raise ValueError(f"{field_name} must be a nonempty scalar")


def _canonical_query_params(
        filters: Mapping[str, Any] | None, *,
        allow_pagination: bool) -> dict[str, str]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise TypeError("table filters must be a mapping")
    result: dict[str, str] = {}
    forbidden = {"api_key", "apikey", "token", "authorization"}
    if not allow_pagination:
        forbidden |= {"qopts.cursor_id", "qopts.per_page"}
    for key, value in filters.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("query parameter names must be nonempty strings")
        if key.lower() in forbidden:
            raise ValueError(f"query parameter {key!r} is controlled by the client")
        result[key] = _request_value(value, field_name=f"query parameter {key}")
    return {key: result[key] for key in sorted(result)}


def canonical_query_params(filters: Mapping[str, Any] | None) -> dict[str, str]:
    """Validate filters and reject credentials plus client-owned pagination."""
    return _canonical_query_params(filters, allow_pagination=False)


def canonical_request(table: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical credential-free representation of one provider request."""
    normalized_table = _short_table_code(table)
    clean = _canonical_query_params(params, allow_pagination=True)
    return {
        "method": "GET",
        "url": TABLE_URL.format(table=normalized_table),
        "params": clean,
    }


def _validate_columns(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ProviderResponseError("provider columns must be a nonempty list")
    columns: list[dict[str, str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"name", "type"}:
            raise ProviderResponseError(
                "provider column descriptors require exactly name and type")
        name = item["name"]
        type_name = item["type"]
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(type_name, str) or not type_name.strip()):
            raise ProviderResponseError("provider column name/type must be nonempty")
        if name in names:
            raise ProviderResponseError(f"provider returned duplicate column: {name}")
        names.add(name)
        columns.append({"name": name, "type": type_name})
    return columns


def _validate_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return 0.0 if value == 0.0 else value
    raise ProviderResponseError("provider rows must contain finite JSON scalar values")


def _validate_provider_page(value: Any) -> tuple[list[dict[str, str]], list[list[Any]], Any]:
    if not isinstance(value, Mapping) or set(value) != {"datatable", "meta"}:
        raise ProviderResponseError(
            "provider response requires exactly datatable and meta")
    datatable = value["datatable"]
    meta = value["meta"]
    if not isinstance(datatable, Mapping) or set(datatable) != {"data", "columns"}:
        raise ProviderResponseError(
            "provider datatable requires exactly data and columns")
    if not isinstance(meta, Mapping) or set(meta) != {"next_cursor_id"}:
        raise ProviderResponseError(
            "provider meta requires exactly next_cursor_id")
    columns = _validate_columns(datatable["columns"])
    raw_rows = datatable["data"]
    if not isinstance(raw_rows, list):
        raise ProviderResponseError("provider data must be a list")
    rows: list[list[Any]] = []
    for row in raw_rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ProviderResponseError(
                "provider row width does not match declared columns")
        rows.append([_validate_cell(cell) for cell in row])
    cursor = meta["next_cursor_id"]
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise ProviderResponseError("provider cursor must be null or a nonempty string")
    return columns, rows, cursor


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    raise ProviderResponseError("HTTP transport response lacks raw content")


def _safe_header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    value = headers.get(name) or headers.get(name.lower())
    return str(value) if value is not None else None


def _aggregate_response_hash(raw_pages: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for body in raw_pages:
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _table_range(
        table: str, columns: Sequence[Mapping[str, str]],
        rows: Sequence[Mapping[str, Any]]) -> dict:
    date_column = _COVERAGE_DATE_COLUMNS.get(table)
    available = {column["name"] for column in columns}
    observed: list[str] = []
    if date_column in available:
        for row in rows:
            value = row[date_column]
            if value is None:
                continue
            try:
                observed.append(date.fromisoformat(str(value)[:10]).isoformat())
            except (TypeError, ValueError):
                continue
    return {
        "date_columns": [date_column] if date_column in available else [],
        "min_date": min(observed) if observed else None,
        "max_date": max(observed) if observed else None,
        "row_count": len(rows),
    }


@dataclass(frozen=True)
class EarningsAnnouncementSnapshot:
    """Exact ``{artifact_hash,payload}`` PEAD input artifact."""

    artifact_hash: str
    payload_json: str

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "EarningsAnnouncementSnapshot":
        normalized = _normalize_json(payload)
        if not isinstance(normalized, dict):
            raise TypeError("snapshot payload must be a mapping")
        payload_json = canonical_json(normalized)
        artifact = cls(
            hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            payload_json,
        )
        artifact._validate_payload(artifact.payload)
        return artifact

    @property
    def payload(self) -> dict[str, Any]:
        value = _strict_json_loads(self.payload_json, context="snapshot payload")
        if not isinstance(value, dict):
            raise SnapshotIntegrityError("snapshot payload must be an object")
        return value

    def to_json(self) -> str:
        return canonical_json({
            "artifact_hash": self.artifact_hash,
            "payload": self.payload,
        }) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "EarningsAnnouncementSnapshot":
        document = _strict_json_loads(value, context="snapshot")
        if not isinstance(document, Mapping) or set(document) != {
                "artifact_hash", "payload"}:
            raise SnapshotIntegrityError(
                "snapshot wrapper requires exactly artifact_hash and payload")
        claimed = _sha256(document["artifact_hash"], field_name="artifact_hash")
        payload = document["payload"]
        if not isinstance(payload, Mapping):
            raise SnapshotIntegrityError("snapshot payload must be an object")
        payload_json = canonical_json(payload)
        actual = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if claimed != actual:
            raise SnapshotIntegrityError("snapshot artifact hash mismatch")
        artifact = cls(actual, payload_json)
        artifact._validate_payload(artifact.payload)
        return artifact

    @staticmethod
    def _validate_payload(payload: Mapping[str, Any]) -> None:
        if set(payload) != _PAYLOAD_FIELDS:
            raise SnapshotIntegrityError("snapshot payload fields are not exact")
        if payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotIntegrityError("unsupported snapshot schema")
        _required_text(payload["candidate_id"], field_name="candidate_id")
        if payload["source_id"] != SOURCE_ID:
            raise SnapshotIntegrityError("invalid snapshot source_id")
        if payload["evidence_class"] not in {
                "development_sample", "historical_replication",
                "prospective_signal"}:
            raise SnapshotIntegrityError("invalid snapshot evidence_class")
        _validate_canonical_timestamp(payload["captured_at"], field_name="captured_at")
        window = payload["requested_window"]
        if not isinstance(window, Mapping) or set(window) != {"start", "end"}:
            raise SnapshotIntegrityError("requested_window fields are not exact")
        start = _iso_date(window["start"], field_name="requested_window.start")
        end = _iso_date(window["end"], field_name="requested_window.end")
        if start > end:
            raise SnapshotIntegrityError("requested window start follows end")
        coverage = payload["coverage"]
        if not isinstance(coverage, Mapping) or set(coverage) != {
                "full_window", "table_ranges", "blockers"}:
            raise SnapshotIntegrityError("coverage fields are not exact")
        if type(coverage["full_window"]) is not bool:
            raise SnapshotIntegrityError("coverage.full_window must be boolean")
        blockers = coverage["blockers"]
        if (not isinstance(blockers, list)
                or any(not isinstance(item, str) or not item for item in blockers)
                or blockers != sorted(set(blockers))):
            raise SnapshotIntegrityError("coverage blockers must be sorted and unique")
        tables = payload["tables"]
        if not isinstance(tables, Mapping) or not tables:
            raise SnapshotIntegrityError("snapshot tables must be nonempty")
        try:
            for table in tables:
                _short_table_code(table, require_canonical=True)
        except ValueError as exc:
            raise SnapshotIntegrityError("snapshot table keys are invalid") from exc
        ranges = coverage["table_ranges"]
        if not isinstance(ranges, Mapping) or set(ranges) != set(tables):
            raise SnapshotIntegrityError("table_ranges must exactly cover tables")
        for table, entry in tables.items():
            EarningsAnnouncementSnapshot._validate_table(
                table, entry, snapshot_captured_at=payload["captured_at"])
            table_range = ranges[table]
            if not isinstance(table_range, Mapping) or set(table_range) != {
                    "date_columns", "min_date", "max_date", "row_count"}:
                raise SnapshotIntegrityError("table range fields are not exact")
            if not isinstance(table_range["date_columns"], list):
                raise SnapshotIntegrityError("table range date_columns must be a list")
            if type(table_range["row_count"]) is not int or table_range["row_count"] < 0:
                raise SnapshotIntegrityError("table range row_count is invalid")
            if table_range["row_count"] != len(entry["rows"]):
                raise SnapshotIntegrityError("table range row_count mismatch")
            for key in ("min_date", "max_date"):
                if table_range[key] is not None:
                    _iso_date(table_range[key], field_name=f"table_ranges.{table}.{key}")
            short_table = _short_table_code(table, require_canonical=True)
            expected_range = _table_range(
                short_table, entry["columns"], entry["rows"])
            if dict(table_range) != expected_range:
                raise SnapshotIntegrityError(
                    f"table range for {table} does not match preserved rows")
        previous_capture = _timestamp_value(payload["captured_at"])
        for short_table in SUPPORTED_TABLES:
            table = _canonical_table_code(short_table)
            if table not in tables:
                continue
            for page in tables[table]["provider_metadata"]["pages"]:
                capture_value = _timestamp_value(page["captured_at"])
                if capture_value < previous_capture:
                    raise SnapshotIntegrityError(
                        "provider page capture timestamps moved backwards")
                previous_capture = capture_value
        if coverage["full_window"] and blockers:
            raise SnapshotIntegrityError("full-window coverage cannot have blockers")
        if (payload["evidence_class"] == "development_sample"
                and coverage["full_window"]):
            raise SnapshotIntegrityError("historical samples cannot claim full coverage")
        if payload["evidence_class"] == "development_sample" and (
                "historical_sample_is_not_full_window_evidence" not in blockers):
            raise SnapshotIntegrityError("historical samples require their coverage blocker")
        if payload["evidence_class"] == "prospective_signal":
            if coverage["full_window"]:
                raise SnapshotIntegrityError("prospective captures cannot claim full coverage")
            if "prospective_window_not_complete" not in blockers:
                raise SnapshotIntegrityError("prospective captures require their coverage blocker")
        if coverage["full_window"]:
            if payload["evidence_class"] != "historical_replication":
                raise SnapshotIntegrityError(
                    "only historical reconstruction may claim full coverage")
            for short_table in SUPPORTED_TABLES:
                table = _canonical_table_code(short_table)
                if table not in tables:
                    raise SnapshotIntegrityError(
                        f"full coverage requires table {table}")
            for short_table, date_column in _COVERAGE_DATE_COLUMNS.items():
                table = _canonical_table_code(short_table)
                table_range = ranges[table]
                if (table_range["date_columns"] != [date_column]
                        or table_range["min_date"] is None
                        or table_range["max_date"] is None
                        or table_range["min_date"] > start
                        or table_range["max_date"] < end):
                    raise SnapshotIntegrityError(
                        f"full coverage is not supported by {table}.{date_column}")

    @staticmethod
    def _validate_table(
            table: str, entry: Any, *, snapshot_captured_at: str) -> None:
        try:
            short_table = _short_table_code(table, require_canonical=True)
        except ValueError as exc:
            raise SnapshotIntegrityError(f"invalid snapshot table key: {table}") from exc
        if not isinstance(entry, Mapping) or set(entry) != _TABLE_FIELDS:
            raise SnapshotIntegrityError(f"table {table} fields are not exact")
        try:
            columns = _validate_columns(entry["columns"])
        except ProviderResponseError as exc:
            raise SnapshotIntegrityError(f"table {table} columns are invalid") from exc
        rows = entry["rows"]
        if not isinstance(rows, list):
            raise SnapshotIntegrityError(f"table {table} rows must be a list")
        column_names = {column["name"] for column in columns}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != column_names:
                raise SnapshotIntegrityError(f"table {table} row fields mismatch")
            try:
                for cell in row.values():
                    _validate_cell(cell)
            except ProviderResponseError as exc:
                raise SnapshotIntegrityError(f"table {table} row is invalid") from exc
        request = entry["canonical_request"]
        if not isinstance(request, Mapping) or set(request) != {"method", "url", "params"}:
            raise SnapshotIntegrityError(f"table {table} canonical request is invalid")
        if (request["method"] != "GET"
                or request["url"] != TABLE_URL.format(table=short_table)):
            raise SnapshotIntegrityError(f"table {table} canonical request target is invalid")
        params = _canonical_query_params(
            request["params"], allow_pagination=True)
        if request["params"] != params:
            raise SnapshotIntegrityError(f"table {table} request params are not canonical")
        if "qopts.cursor_id" in params:
            raise SnapshotIntegrityError(
                f"table {table} aggregate request cannot contain a cursor")
        _sha256(entry["response_sha256"], field_name=f"tables.{table}.response_sha256")
        metadata = entry["provider_metadata"]
        expected_metadata = {
            "api_version", "dataset_code", "table_code", "page_count",
            "row_count", "response_hash_basis", "pages",
        }
        if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata:
            raise SnapshotIntegrityError(f"table {table} provider metadata is invalid")
        if (metadata["api_version"] != "v3" or metadata["dataset_code"] != "ZACKS"
                or metadata["table_code"] != table
                or metadata["response_hash_basis"]
                != "sha256_length_prefixed_raw_page_bodies"):
            raise SnapshotIntegrityError(f"table {table} provider identity is invalid")
        pages = metadata["pages"]
        if (type(metadata["page_count"]) is not int
                or metadata["page_count"] != len(pages) or not pages
                or type(metadata["row_count"]) is not int
                or metadata["row_count"] != len(rows)):
            raise SnapshotIntegrityError(f"table {table} provider counts mismatch")
        raw_pages: list[bytes] = []
        provider_rows: list[list[Any]] = []
        previous_cursor: str | None = None
        observed_cursors: set[str] = set()
        previous_capture = _timestamp_value(snapshot_captured_at)
        for number, page in enumerate(pages, start=1):
            expected_page = {
                "page_number", "captured_at", "canonical_request",
                "request_sha256", "response_body_sha256", "http_status",
                "response_body_base64", "provider_request_id", "next_cursor_id",
            }
            if not isinstance(page, Mapping) or set(page) != expected_page:
                raise SnapshotIntegrityError(f"table {table} page metadata is invalid")
            if page["page_number"] != number or page["http_status"] != 200:
                raise SnapshotIntegrityError(f"table {table} page sequence/status invalid")
            page_captured_at = _validate_canonical_timestamp(
                page["captured_at"], field_name=f"tables.{table}.page.captured_at")
            capture_value = _timestamp_value(page_captured_at)
            if capture_value < previous_capture:
                raise SnapshotIntegrityError(
                    f"table {table} page capture timestamps moved backwards")
            previous_capture = capture_value
            _sha256(page["request_sha256"], field_name="request_sha256")
            body_hash = _sha256(
                page["response_body_sha256"], field_name="response_body_sha256")
            encoded_body = page["response_body_base64"]
            if not isinstance(encoded_body, str):
                raise SnapshotIntegrityError(
                    f"table {table} response body must be canonical base64")
            try:
                raw_body = base64.b64decode(encoded_body, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise SnapshotIntegrityError(
                    f"table {table} response body must be canonical base64") from exc
            if base64.b64encode(raw_body).decode("ascii") != encoded_body:
                raise SnapshotIntegrityError(
                    f"table {table} response body base64 is not canonical")
            if hashlib.sha256(raw_body).hexdigest() != body_hash:
                raise SnapshotIntegrityError(f"table {table} response body hash mismatch")
            raw_pages.append(raw_body)
            try:
                raw_document = _strict_json_loads(
                    raw_body, context=f"tables.{table}.raw_response")
                raw_columns, raw_rows, raw_cursor = _validate_provider_page(raw_document)
            except (ProviderResponseError, SnapshotIntegrityError) as exc:
                raise SnapshotIntegrityError(
                    f"table {table} preserved response is invalid") from exc
            if raw_columns != columns:
                raise SnapshotIntegrityError(
                    f"table {table} preserved response columns mismatch")
            if raw_cursor != page["next_cursor_id"]:
                raise SnapshotIntegrityError(
                    f"table {table} preserved response cursor mismatch")
            provider_rows.extend(raw_rows)
            if page["request_sha256"] != _content_hash(page["canonical_request"]):
                raise SnapshotIntegrityError(f"table {table} page request hash mismatch")
            page_request = page["canonical_request"]
            if not isinstance(page_request, Mapping) or "params" not in page_request:
                raise SnapshotIntegrityError(f"table {table} page request is invalid")
            if page_request != canonical_request(short_table, page_request["params"]):
                raise SnapshotIntegrityError(
                    f"table {table} page request is not canonical")
            expected_params = dict(params)
            if previous_cursor is not None:
                expected_params["qopts.cursor_id"] = previous_cursor
            if page_request["params"] != {
                    key: expected_params[key] for key in sorted(expected_params)}:
                raise SnapshotIntegrityError(
                    f"table {table} page cursor chain is invalid")
            if page["provider_request_id"] is not None and not isinstance(
                    page["provider_request_id"], str):
                raise SnapshotIntegrityError(f"table {table} request ID is invalid")
            if (page["next_cursor_id"] is not None
                    and (not isinstance(page["next_cursor_id"], str)
                         or not page["next_cursor_id"])):
                raise SnapshotIntegrityError(f"table {table} cursor is invalid")
            if page["next_cursor_id"] in observed_cursors:
                raise SnapshotIntegrityError(f"table {table} repeated a cursor")
            if page["next_cursor_id"] is not None:
                observed_cursors.add(page["next_cursor_id"])
            previous_cursor = page["next_cursor_id"]
        if previous_cursor is not None:
            raise SnapshotIntegrityError(f"table {table} final page must end pagination")
        if entry["response_sha256"] != _aggregate_response_hash(raw_pages):
            raise SnapshotIntegrityError(f"table {table} aggregate response hash mismatch")
        column_names = [column["name"] for column in columns]
        preserved_rows = [dict(zip(column_names, row)) for row in provider_rows]
        preserved_rows.sort(key=canonical_json)
        if rows != preserved_rows:
            raise SnapshotIntegrityError(
                f"table {table} rows do not match preserved provider responses")


class ZacksTablesClient:
    """Injectable-transport Nasdaq Tables client with strict cursor pagination."""

    def __init__(
            self, *, get: Callable[..., Any],
            clock: Callable[[], datetime] | None = None,
            environ: Mapping[str, str] | None = None) -> None:
        if not callable(get):
            raise TypeError("get transport must be callable")
        self._get = get
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._environ = os.environ if environ is None else environ

    def _api_key(self) -> str:
        key = self._environ.get(NASDAQ_API_KEY_ENV)
        if not isinstance(key, str) or not key.strip():
            raise CredentialError(
                f"{NASDAQ_API_KEY_ENV} is required for Zacks acquisition")
        return key.strip()

    def capture(
            self, *, mode: str, candidate_id: str,
            requested_start: str, requested_end: str,
            tables: Sequence[str], filters_by_table: Mapping[str, Mapping[str, Any]],
            per_page: int = 10_000, max_pages: int = 1_000,
            max_rows: int = 10_000_000, timeout: float = 120.0,
            ) -> EarningsAnnouncementSnapshot:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id is required")
        try:
            start = date.fromisoformat(requested_start).isoformat()
            end = date.fromisoformat(requested_end).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("requested window must use YYYY-MM-DD") from exc
        if start != requested_start or end != requested_end or start > end:
            raise ValueError("requested window is invalid or noncanonical")
        selected = _normalize_tables(tables)
        if not isinstance(filters_by_table, Mapping) or set(filters_by_table) != set(selected):
            raise ValueError("filters_by_table must exactly cover selected tables")
        if type(per_page) is not int or not 1 <= per_page <= 10_000:
            raise ValueError("per_page must be an integer from 1 to 10000")
        if type(max_pages) is not int or max_pages < 1:
            raise ValueError("max_pages must be a positive integer")
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")

        captured_at = canonical_utc_timestamp(self._clock())
        api_key = self._api_key()
        table_payloads: dict[str, Any] = {}
        table_ranges: dict[str, Any] = {}
        blockers: list[str] = []
        last_observed_at = captured_at
        for table in selected:
            filters = canonical_query_params(filters_by_table[table])
            entry, last_observed_at = self._fetch_table(
                table, filters, api_key=api_key,
                per_page=per_page, max_pages=max_pages, max_rows=max_rows,
                timeout=float(timeout), not_before=last_observed_at,
            )
            canonical_table = _canonical_table_code(table)
            table_payloads[canonical_table] = entry
            table_range = _table_range(table, entry["columns"], entry["rows"])
            table_ranges[canonical_table] = table_range
            if table in _COVERAGE_DATE_COLUMNS:
                date_column = _COVERAGE_DATE_COLUMNS[table]
                if table_range["date_columns"] != [date_column]:
                    blockers.append(
                        f"missing_coverage_column:{canonical_table}:{date_column}")
                elif table_range["row_count"] == 0:
                    blockers.append(f"empty_coverage_table:{canonical_table}")
                elif table_range["min_date"] is None:
                    blockers.append(
                        f"no_parseable_coverage_date:{canonical_table}:{date_column}")
                else:
                    if table_range["min_date"] > start:
                        blockers.append(
                            f"range_starts_after_requested:{canonical_table}")
                    if table_range["max_date"] < end:
                        blockers.append(
                            f"range_ends_before_requested:{canonical_table}")
        for table in SUPPORTED_TABLES:
            if table not in selected:
                blockers.append(
                    f"missing_required_table:{_canonical_table_code(table)}")
        if mode == "historical-sample":
            blockers.append("historical_sample_is_not_full_window_evidence")
        elif mode == "prospective":
            blockers.append("prospective_window_not_complete")
        blockers = sorted(set(blockers))
        full_window = mode == "historical-full" and not blockers
        evidence_class = {
            "historical-sample": "development_sample",
            "historical-full": "historical_replication",
            "prospective": "prospective_signal",
        }[mode]
        return EarningsAnnouncementSnapshot.create({
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "candidate_id": candidate_id.strip(),
            "source_id": SOURCE_ID,
            "evidence_class": evidence_class,
            "captured_at": captured_at,
            "requested_window": {"start": start, "end": end},
            "coverage": {
                "full_window": full_window,
                "table_ranges": table_ranges,
                "blockers": blockers,
            },
            "tables": table_payloads,
        })

    def _fetch_table(
            self, table: str, filters: Mapping[str, str], *, api_key: str,
            per_page: int, max_pages: int, max_rows: int,
            timeout: float, not_before: str) -> tuple[dict[str, Any], str]:
        base_params = dict(filters)
        base_params["qopts.per_page"] = str(per_page)
        provider_rows: list[list[Any]] = []
        columns: list[dict[str, str]] | None = None
        pages: list[dict[str, Any]] = []
        raw_pages: list[bytes] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        last_observed_at = not_before
        while True:
            if len(pages) >= max_pages:
                raise ProviderResponseError(
                    f"{table}: pagination exceeded max_pages={max_pages}")
            params = dict(base_params)
            if cursor is not None:
                params["qopts.cursor_id"] = cursor
            request = canonical_request(table, params)
            transport_params = dict(request["params"])
            transport_params["api_key"] = api_key
            try:
                response = self._get(
                    request["url"], params=transport_params, timeout=timeout)
            except Exception as exc:  # transport-specific exceptions stay private
                raise ProviderResponseError(f"{table}: provider request failed") from exc
            status = getattr(response, "status_code", None)
            if status != 200:
                try:
                    response.raise_for_status()
                except Exception as exc:
                    raise ProviderResponseError(
                        f"{table}: provider returned HTTP {status}") from exc
                raise ProviderResponseError(f"{table}: provider returned HTTP {status}")
            body = _response_bytes(response)
            request_id = _safe_header(response, "X-Request-Id")
            if api_key.encode("utf-8") in body or (
                    request_id is not None and api_key in request_id):
                raise ProviderResponseError(
                    f"{table}: provider response exposed request credentials")
            page_captured_at = canonical_utc_timestamp(self._clock())
            if _timestamp_value(page_captured_at) < _timestamp_value(last_observed_at):
                raise ProviderResponseError(
                    f"{table}: trusted capture clock moved backwards")
            last_observed_at = page_captured_at
            try:
                document = _strict_json_loads(body, context=f"{table} provider response")
            except SnapshotIntegrityError as exc:
                raise ProviderResponseError(f"{table}: provider returned invalid JSON") from exc
            page_columns, page_rows, next_cursor = _validate_provider_page(document)
            if columns is None:
                columns = page_columns
            elif columns != page_columns:
                raise ProviderResponseError(f"{table}: columns changed across pages")
            provider_rows.extend(page_rows)
            if len(provider_rows) > max_rows:
                raise ProviderResponseError(
                    f"{table}: result exceeded max_rows={max_rows}")
            body_hash = hashlib.sha256(body).hexdigest()
            pages.append({
                "page_number": len(pages) + 1,
                "captured_at": page_captured_at,
                "canonical_request": request,
                "request_sha256": _content_hash(request),
                "response_body_sha256": body_hash,
                "response_body_base64": base64.b64encode(body).decode("ascii"),
                "http_status": 200,
                "provider_request_id": request_id,
                "next_cursor_id": next_cursor,
            })
            raw_pages.append(body)
            if next_cursor is None:
                break
            if next_cursor in seen_cursors:
                raise ProviderResponseError(f"{table}: provider repeated a cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        assert columns is not None
        column_names = [column["name"] for column in columns]
        rows = [dict(zip(column_names, row)) for row in provider_rows]
        rows.sort(key=canonical_json)
        return {
            "columns": columns,
            "rows": rows,
            "canonical_request": canonical_request(table, base_params),
            "response_sha256": _aggregate_response_hash(raw_pages),
            "provider_metadata": {
                "api_version": "v3",
                "dataset_code": "ZACKS",
                "table_code": _canonical_table_code(table),
                "page_count": len(pages),
                "row_count": len(rows),
                "response_hash_basis": "sha256_length_prefixed_raw_page_bodies",
                "pages": pages,
            },
        }, last_observed_at


@dataclass(frozen=True)
class PersistedSnapshot:
    artifact_hash: str
    snapshot_path: Path
    journal_event_hash: str
    journal_event_path: Path


def _atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != encoded:
                raise SnapshotIntegrityError(
                    f"refusing to overwrite immutable evidence: {path}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class EarningsAnnouncementStore:
    """Create-only snapshot store with an append-only hash-chained journal."""

    def __init__(
            self, root: str | Path, *,
            clock: Callable[[], datetime] | None = None) -> None:
        self.root = Path(root)
        self.snapshots_directory = self.root / "snapshots"
        self.events_directory = self.root / "journal"
        self.lock_path = self.root / ".journal.lock"
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def snapshot_path(self, artifact_hash: str) -> Path:
        digest = _sha256(artifact_hash, field_name="artifact_hash")
        return self.snapshots_directory / f"{digest}.json"

    def persist(self, snapshot: EarningsAnnouncementSnapshot) -> PersistedSnapshot:
        verified = EarningsAnnouncementSnapshot.from_json(snapshot.to_json())
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_directory.mkdir(exist_ok=True)
        self.events_directory.mkdir(exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                path = self.snapshot_path(verified.artifact_hash)
                _atomic_create(path, verified.to_json())
                events = self._load_journal_unlocked()
                for event_hash, event_path, payload in events:
                    if payload["artifact_hash"] == verified.artifact_hash:
                        return PersistedSnapshot(
                            verified.artifact_hash, path, event_hash, event_path)
                previous = events[-1][0] if events else None
                event_payload = {
                    "schema_version": JOURNAL_SCHEMA_VERSION,
                    "sequence": len(events),
                    "previous_event_hash": previous,
                    "artifact_hash": verified.artifact_hash,
                    "candidate_id": verified.payload["candidate_id"],
                    "captured_at": verified.payload["captured_at"],
                    "recorded_at": canonical_utc_timestamp(self._clock()),
                }
                event_hash = _content_hash(event_payload)
                document = canonical_json({
                    "event_hash": event_hash, "payload": event_payload}) + "\n"
                event_path = self.events_directory / (
                    f"{len(events):020d}-{event_hash}.json")
                _atomic_create(event_path, document)
                return PersistedSnapshot(
                    verified.artifact_hash, path, event_hash, event_path)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load(self, artifact_hash: str) -> EarningsAnnouncementSnapshot:
        digest = _sha256(artifact_hash, field_name="artifact_hash")
        try:
            snapshot = EarningsAnnouncementSnapshot.from_json(
                self.snapshot_path(digest).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"earnings snapshot does not exist: {digest}") from exc
        if snapshot.artifact_hash != digest:
            raise SnapshotIntegrityError("snapshot filename and content hash differ")
        return snapshot

    def verify_journal(self) -> list[dict[str, Any]]:
        if not self.events_directory.exists():
            return []
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                return [payload for _, _, payload in self._load_journal_unlocked()]
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_journal_unlocked(self) -> list[tuple[str, Path, dict[str, Any]]]:
        events: list[tuple[str, Path, dict[str, Any]]] = []
        previous: str | None = None
        for sequence, path in enumerate(sorted(self.events_directory.glob("*.json"))):
            document = _strict_json_loads(
                path.read_text(encoding="utf-8"), context="journal event")
            if not isinstance(document, Mapping) or set(document) != {
                    "event_hash", "payload"}:
                raise SnapshotIntegrityError("journal event wrapper fields are not exact")
            event_hash = _sha256(document["event_hash"], field_name="event_hash")
            payload = document["payload"]
            expected = {
                "schema_version", "sequence", "previous_event_hash",
                "artifact_hash", "candidate_id", "captured_at", "recorded_at",
            }
            if not isinstance(payload, Mapping) or set(payload) != expected:
                raise SnapshotIntegrityError("journal event payload fields are not exact")
            if payload["schema_version"] != JOURNAL_SCHEMA_VERSION:
                raise SnapshotIntegrityError("unsupported journal schema")
            if payload["sequence"] != sequence or payload["previous_event_hash"] != previous:
                raise SnapshotIntegrityError("journal sequence or hash chain is broken")
            if event_hash != _content_hash(payload):
                raise SnapshotIntegrityError("journal event hash mismatch")
            expected_name = f"{sequence:020d}-{event_hash}.json"
            if path.name != expected_name:
                raise SnapshotIntegrityError("journal event filename is invalid")
            artifact_hash = _sha256(payload["artifact_hash"], field_name="artifact_hash")
            _required_text(payload["candidate_id"], field_name="candidate_id")
            _validate_canonical_timestamp(payload["captured_at"], field_name="captured_at")
            _validate_canonical_timestamp(payload["recorded_at"], field_name="recorded_at")
            snapshot = self.load(artifact_hash)
            if (snapshot.payload["candidate_id"] != payload["candidate_id"]
                    or snapshot.payload["captured_at"] != payload["captured_at"]):
                raise SnapshotIntegrityError("journal event does not match snapshot")
            events.append((event_hash, path, dict(payload)))
            previous = event_hash
        return events


__all__ = [
    "CANONICAL_TABLES",
    "DEFAULT_CANDIDATE_ID",
    "EarningsAnnouncementError",
    "EarningsAnnouncementSnapshot",
    "EarningsAnnouncementStore",
    "PersistedSnapshot",
    "ProviderResponseError",
    "SNAPSHOT_SCHEMA_VERSION",
    "SOURCE_ID",
    "SUPPORTED_MODES",
    "SUPPORTED_TABLES",
    "SnapshotIntegrityError",
    "ZacksTablesClient",
    "canonical_json",
    "canonical_query_params",
    "canonical_request",
    "canonical_utc_timestamp",
]
