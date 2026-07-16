"""Candidate-grade acquisition evidence for Sharadar SF1, SEP, and TICKERS.

The ordinary :mod:`data.pit_warehouse` bulk ingest is a useful cache builder,
but a mutable Parquet file is not acquisition evidence.  This module defines a
separate, fail-closed boundary that preserves the provider ZIP, the exact CSV
member, an explicitly typed Parquet conversion, provider metadata/timestamps,
and a bidirectional ``EXCEPT ALL`` replay proving CSV/Parquet row equivalence.

The immutable table receipts are then collected into
``pead_sharadar_source_snapshot.v1``.  A second artifact derives the dated
CIK/permaticker/ticker bridge from the exact TICKERS bytes.  The authoritative
identity validator always reopens those bytes and rebuilds the artifact; the
one-argument structural validator is intentionally not a trust boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import zipfile

import duckdb


SHARADAR_TABLE_ACQUISITION_SCHEMA_VERSION = "sharadar_table_acquisition.v1"
SHARADAR_ROW_EQUIVALENCE_SCHEMA_VERSION = "sharadar_row_equivalence.v1"
SHARADAR_SOURCE_RECORD_SCHEMA_VERSION = "sharadar_source_record.v1"
PEAD_SHARADAR_SOURCE_SNAPSHOT_SCHEMA_VERSION = "pead_sharadar_source_snapshot.v1"
PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION = "pead_security_identity_snapshot.v1"
PEAD_SECURITY_IDENTITY_SCHEMA_VERSION = "pead_security_identity.v1"

CANDIDATE_TABLES = ("sf1", "sep", "tickers")
SHARADAR_RECEIPT_ROOT = "source_snapshots/sharadar"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_IDENTITY_BYTES = 256 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CIK_URL = re.compile(
    r"^https://www\.sec\.gov/cgi-bin/browse-edgar\?"
    r"action=getcompany&CIK=([0-9]{10})$"
)

_TABLES: dict[str, dict[str, Any]] = {
    "sf1": {
        "dataset": "SHARADAR/SF1",
        "parameters": {"dimension": "ARQ", "qopts.export": "true"},
        "primary_key": ("ticker", "dimension", "datekey", "reportperiod"),
        "required_columns": frozenset(
            {
                "ticker",
                "dimension",
                "calendardate",
                "datekey",
                "reportperiod",
                "fiscalperiod",
                "eps",
                "epsdil",
                "sharesbas",
                "sharefactor",
            }
        ),
        "date_columns": ("calendardate", "datekey", "reportperiod"),
    },
    "sep": {
        "dataset": "SHARADAR/SEP",
        "parameters": {"qopts.export": "true"},
        "primary_key": ("ticker", "date"),
        "required_columns": frozenset(
            {"ticker", "date", "close", "closeadj", "closeunadj", "lastupdated"}
        ),
        "date_columns": ("date", "lastupdated"),
    },
    "tickers": {
        "dataset": "SHARADAR/TICKERS",
        "parameters": {"qopts.export": "true", "table": "SF1"},
        "primary_key": ("table", "permaticker", "ticker"),
        "required_columns": frozenset(
            {
                "table",
                "permaticker",
                "ticker",
                "exchange",
                "isdelisted",
                "category",
                "currency",
                "firstpricedate",
                "lastpricedate",
                "secfilings",
            }
        ),
        "date_columns": (
            "lastupdated",
            "firstadded",
            "firstpricedate",
            "lastpricedate",
            "firstquarter",
            "lastquarter",
        ),
    },
}

_PROVIDER_TYPE_MAP = {
    "date": "DATE",
    "text": "VARCHAR",
    "string": "VARCHAR",
    "varchar": "VARCHAR",
    "bigdecimal": "DOUBLE",
    "double": "DOUBLE",
    "float": "DOUBLE",
    "integer": "BIGINT",
    "bigint": "BIGINT",
    "boolean": "BOOLEAN",
}

_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_TABLE_PAYLOAD_FIELDS = {
    "schema_version",
    "logical_name",
    "source",
    "acquired_at_utc",
    "raw_zip",
    "csv",
    "parquet",
    "row_equivalence",
}
_SOURCE_FIELDS = {
    "vendor_code",
    "datatable_code",
    "canonical_request",
    "last_refreshed_time",
    "data_snapshot_time",
    "bulk_status",
    "datatable_metadata",
}
_REQUEST_FIELDS = {"method", "dataset", "parameters"}
_METADATA_FIELDS = {
    "vendor_code",
    "datatable_code",
    "name",
    "description",
    "columns",
    "filters",
    "primary_key",
    "premium",
    "status",
}
_METADATA_COLUMN_FIELDS = {"name", "type", "description"}
_METADATA_STATUS_FIELDS = {
    "expected_at",
    "refreshed_at",
    "status",
    "update_frequency",
}
_RAW_FIELDS = {"relative_path", "sha256", "bytes"}
_CSV_FIELDS = {"member", "sha256", "bytes", "compressed_bytes", "header"}
_PARQUET_FIELDS = {"relative_path", "sha256", "bytes", "schema", "statistics"}
_SCHEMA_FIELD_FIELDS = {"name", "logical_type"}
_STATISTICS_FIELDS = {
    "rows",
    "distinct_tickers",
    "primary_key",
    "missing_primary_key_values",
    "duplicate_primary_keys",
    "scope_violations",
    "date_ranges",
}
_DATE_RANGE_FIELDS = {"column", "min", "max"}
_ROW_EQUIVALENCE_FIELDS = {
    "schema_version",
    "method",
    "rows",
    "csv_minus_parquet_rows",
    "parquet_minus_csv_rows",
    "equivalent",
}

_SOURCE_SNAPSHOT_FIELDS = {
    "schema_version",
    "candidate_id",
    "created_at_utc",
    "tables",
    "coverage",
    "blockers",
    "qualification_allowed",
}
_SOURCE_SNAPSHOT_TABLE_FIELDS = {
    "logical_name",
    "datatable_code",
    "acquisition_artifact_hash",
    "acquisition_receipt_relative_path",
    "data_snapshot_time",
    "raw_zip_sha256",
    "parquet_sha256",
    "row_count",
}
_SOURCE_SNAPSHOT_COVERAGE_FIELDS = {
    "required_tables",
    "present_tables",
    "complete",
}

_IDENTITY_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "created_at_utc",
    "bindings",
    "source_dispositions",
    "identities",
    "coverage",
    "blockers",
    "qualification_allowed",
}
_IDENTITY_BINDING_FIELDS = {
    "sharadar_source_snapshot_sha256",
    "tickers_acquisition_sha256",
    "tickers_parquet_sha256",
}
_IDENTITY_DISPOSITION_FIELDS = {
    "source_record_sha256",
    "disposition",
    "identity_id",
    "reason",
}
_IDENTITY_FIELDS = {
    "identity_id",
    "cik",
    "permaticker",
    "ticker",
    "valid_from",
    "valid_through",
    "is_delisted",
    "category",
    "exchange",
    "currency",
    "source_record_sha256",
}
_IDENTITY_COVERAGE_FIELDS = {
    "source_row_count",
    "disposition_count",
    "identity_count",
    "identity_gap_count",
    "complete",
}


class SharadarSourceEvidenceError(ValueError):
    """Sharadar acquisition or identity evidence is not trustworthy."""


def canonical_json(value: Any) -> str:
    """Return deterministic finite JSON used by every artifact identity."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise SharadarSourceEvidenceError("evidence contains a non-finite number")
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise SharadarSourceEvidenceError("evidence keys must be strings")
                result[key] = normalize(child)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise SharadarSourceEvidenceError(f"unsupported evidence value: {type(item).__name__}")

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SharadarSourceEvidenceError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SharadarSourceEvidenceError(f"{label} must be nonempty canonical text")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SharadarSourceEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SharadarSourceEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _utc(value: Any, label: str, *, provider_format: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SharadarSourceEvidenceError(f"{label} must be a UTC timestamp")
    original = value
    text = value.strip().replace(" UTC", "+00:00")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SharadarSourceEvidenceError(f"{label} must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SharadarSourceEvidenceError(f"{label} must identify UTC")
    if provider_format:
        return original
    canonical = (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds" if parsed.microsecond else "seconds")
        .replace("+00:00", "Z")
    )
    if canonical != original:
        raise SharadarSourceEvidenceError(f"{label} must be canonical UTC with Z")
    return canonical


def _day(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SharadarSourceEvidenceError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SharadarSourceEvidenceError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise SharadarSourceEvidenceError(f"{label} must be canonical YYYY-MM-DD")
    return value


def _logical_name(value: Any) -> str:
    name = _text(value, "logical_name").lower()
    if name not in _TABLES:
        raise SharadarSourceEvidenceError(
            f"unsupported candidate table {name!r}; expected {list(CANDIDATE_TABLES)}"
        )
    return name


def _identifier(value: str) -> str:
    if _NAME.fullmatch(value) is None:
        raise SharadarSourceEvidenceError(f"unsafe provider column name {value!r}")
    return f'"{value}"'


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _canonical_request(logical_name: str) -> dict[str, Any]:
    spec = _TABLES[logical_name]
    return {
        "method": "GET",
        "dataset": spec["dataset"],
        "parameters": dict(sorted(spec["parameters"].items())),
    }


def _provider_logical_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("bigdecimal") or normalized.startswith("decimal"):
        return "DOUBLE"
    try:
        return _PROVIDER_TYPE_MAP[normalized]
    except KeyError as exc:
        raise SharadarSourceEvidenceError(f"unsupported provider column type {value!r}") from exc


def normalize_datatable_metadata(logical_name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize and validate the provider metadata persisted in a receipt.

    Column descriptions are retained verbatim when the provider supplies them.
    A missing description is represented as ``null``; consumers must never
    infer SEP field semantics merely from a column name.
    """
    name = _logical_name(logical_name)
    metadata = _exact(value, _METADATA_FIELDS, "datatable_metadata")
    dataset = _TABLES[name]["dataset"]
    vendor_code, datatable_code = dataset.split("/", 1)
    if metadata["vendor_code"] != vendor_code:
        raise SharadarSourceEvidenceError("datatable metadata vendor differs")
    if metadata["datatable_code"] != datatable_code:
        raise SharadarSourceEvidenceError("datatable metadata code differs")
    display_name = _text(metadata["name"], "datatable metadata name")
    description = metadata["description"]
    if description is not None and not isinstance(description, str):
        raise SharadarSourceEvidenceError("datatable metadata description is invalid")

    raw_columns = metadata["columns"]
    if not isinstance(raw_columns, list) or not raw_columns:
        raise SharadarSourceEvidenceError("datatable metadata columns are missing")
    columns: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_columns):
        column = _exact(raw, _METADATA_COLUMN_FIELDS, f"datatable metadata column {index}")
        column_name = _text(column["name"], f"metadata column {index} name")
        _identifier(column_name)
        if column_name in seen:
            raise SharadarSourceEvidenceError("datatable metadata columns are duplicated")
        seen.add(column_name)
        provider_type = _text(column["type"], f"metadata column {index} type")
        _provider_logical_type(provider_type)
        column_description = column["description"]
        if column_description is not None and not isinstance(column_description, str):
            raise SharadarSourceEvidenceError("metadata column description is invalid")
        columns.append(
            {
                "name": column_name,
                "type": provider_type,
                "description": column_description,
            }
        )
    required = _TABLES[name]["required_columns"]
    missing = sorted(required - seen)
    if missing:
        raise SharadarSourceEvidenceError(f"{name} metadata is missing required columns: {missing}")

    filters = metadata["filters"]
    if (
        not isinstance(filters, list)
        or any(not isinstance(item, str) or not item for item in filters)
        or len(filters) != len(set(filters))
    ):
        raise SharadarSourceEvidenceError("datatable metadata filters are invalid")
    primary_key = metadata["primary_key"]
    expected_primary_key = list(_TABLES[name]["primary_key"])
    if primary_key != expected_primary_key:
        raise SharadarSourceEvidenceError(
            f"{name} primary key differs: expected {expected_primary_key}, got {primary_key}"
        )
    if type(metadata["premium"]) is not bool:
        raise SharadarSourceEvidenceError("datatable metadata premium must be boolean")
    status = _exact(metadata["status"], _METADATA_STATUS_FIELDS, "metadata status")
    expected_at = status["expected_at"]
    if expected_at is not None and (not isinstance(expected_at, str) or not expected_at):
        raise SharadarSourceEvidenceError("metadata expected_at is invalid")
    refreshed_at = _utc(status["refreshed_at"], "metadata refreshed_at", provider_format=True)
    status_text = status["status"]
    if status_text is not None and (not isinstance(status_text, str) or not status_text):
        raise SharadarSourceEvidenceError("metadata status is invalid")
    frequency = status["update_frequency"]
    if frequency is not None and (not isinstance(frequency, str) or not frequency):
        raise SharadarSourceEvidenceError("metadata update_frequency is invalid")
    return {
        "vendor_code": vendor_code,
        "datatable_code": datatable_code,
        "name": display_name,
        "description": description,
        "columns": columns,
        "filters": list(filters),
        "primary_key": expected_primary_key,
        "premium": metadata["premium"],
        "status": {
            "expected_at": expected_at,
            "refreshed_at": refreshed_at,
            "status": status_text,
            "update_frequency": frequency,
        },
    }


def _typed_schema(metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "name": column["name"],
            "logical_type": _provider_logical_type(column["type"]),
        }
        for column in metadata["columns"]
    ]


def _normalize_schema(value: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise SharadarSourceEvidenceError(f"{label} must be a nonempty array")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        field = _exact(raw, _SCHEMA_FIELD_FIELDS, f"{label}[{index}]")
        name = _text(field["name"], f"{label}[{index}].name")
        _identifier(name)
        logical_type = _text(field["logical_type"], f"{label}[{index}].logical_type").upper()
        if logical_type not in set(_PROVIDER_TYPE_MAP.values()):
            raise SharadarSourceEvidenceError(
                f"{label}[{index}] has unsupported type {logical_type!r}"
            )
        if name in seen:
            raise SharadarSourceEvidenceError(f"{label} has duplicate columns")
        seen.add(name)
        result.append({"name": name, "logical_type": logical_type})
    return result


def _typed_value(value: Any, logical_type: str, label: str) -> Any:
    if value is None:
        return None
    if logical_type == "VARCHAR":
        if not isinstance(value, str):
            raise SharadarSourceEvidenceError(f"{label} must be text")
        return value
    if logical_type == "DATE":
        if isinstance(value, datetime) or not isinstance(value, (date, str)):
            raise SharadarSourceEvidenceError(f"{label} must be a date")
        rendered = value.isoformat() if isinstance(value, date) else value
        return _day(rendered, label)
    if logical_type == "BIGINT":
        if type(value) is not int:
            raise SharadarSourceEvidenceError(f"{label} must be an exact integer")
        return value
    if logical_type == "DOUBLE":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SharadarSourceEvidenceError(f"{label} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise SharadarSourceEvidenceError(f"{label} must be finite")
        return (0.0 if number == 0.0 else number).hex()
    if logical_type == "BOOLEAN":
        if type(value) is not bool:
            raise SharadarSourceEvidenceError(f"{label} must be boolean")
        return value
    raise SharadarSourceEvidenceError(f"{label} has unsupported type {logical_type}")


def sharadar_source_record_sha256(
    logical_name: str,
    schema: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any] | Sequence[Any],
) -> str:
    """Return the shared typed identity for one SF1, SEP, or TICKERS row.

    Event-census and market-accounting code must use this helper instead of
    reimplementing value normalization.  ``schema`` is the exact ordered schema
    stored in the acquisition receipt.
    """
    name = _logical_name(logical_name)
    normalized_schema = _normalize_schema(list(schema), label="source schema")
    if isinstance(row, Mapping):
        if set(row) != {field["name"] for field in normalized_schema}:
            raise SharadarSourceEvidenceError("source row fields differ from its schema")
        values = [row[field["name"]] for field in normalized_schema]
    elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        values = list(row)
        if len(values) != len(normalized_schema):
            raise SharadarSourceEvidenceError("source row width differs from its schema")
    else:
        raise SharadarSourceEvidenceError("source row must be a mapping or sequence")
    fields = [
        {
            "name": field["name"],
            "logical_type": field["logical_type"],
            "value": _typed_value(
                value,
                field["logical_type"],
                f"source row {field['name']}",
            ),
        }
        for field, value in zip(normalized_schema, values, strict=True)
    ]
    return content_hash(
        {
            "schema_version": SHARADAR_SOURCE_RECORD_SCHEMA_VERSION,
            "logical_name": name,
            "fields": fields,
        }
    )


def inspect_sharadar_zip(
    path: str | os.PathLike[str], *, expected_header: Sequence[str]
) -> dict[str, Any]:
    """Hash the sole CSV member and validate its exact ordered header."""
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise SharadarSourceEvidenceError(f"Sharadar ZIP is not a regular file: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            members = [entry for entry in archive.infolist() if not entry.is_dir()]
            csv_members = [entry for entry in members if entry.filename.lower().endswith(".csv")]
            if len(members) != 1 or len(csv_members) != 1:
                raise SharadarSourceEvidenceError(
                    "Sharadar ZIP must contain exactly one CSV member"
                )
            info = csv_members[0]
            if Path(info.filename).name != info.filename:
                raise SharadarSourceEvidenceError("Sharadar ZIP member path is unsafe")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, "r") as raw_stream:
                buffered = io.BufferedReader(raw_stream)
                header_bytes = buffered.readline()
                digest.update(header_bytes)
                size += len(header_bytes)
                try:
                    header_text = header_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise SharadarSourceEvidenceError("Sharadar CSV header is not UTF-8") from exc
                header = next(csv.reader([header_text]))
                for chunk in iter(lambda: buffered.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SharadarSourceEvidenceError("Sharadar ZIP is unreadable") from exc
    expected = list(expected_header)
    if header != expected or len(header) != len(set(header)):
        raise SharadarSourceEvidenceError(
            f"Sharadar CSV header differs: expected {expected}, got {header}"
        )
    return {
        "member": info.filename,
        "sha256": digest.hexdigest(),
        "bytes": size,
        "compressed_bytes": int(info.compress_size),
        "header": header,
    }


def _extract_csv(raw_zip_path: Path, destination: Path, *, member: str) -> None:
    try:
        with zipfile.ZipFile(raw_zip_path) as archive:
            info = archive.getinfo(member)
            if Path(info.filename).name != info.filename:
                raise SharadarSourceEvidenceError("Sharadar ZIP member path is unsafe")
            with archive.open(info, "r") as incoming, destination.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SharadarSourceEvidenceError("Sharadar CSV extraction failed") from exc


def _csv_columns_literal(schema: Sequence[Mapping[str, str]]) -> str:
    return (
        "{"
        + ",".join(
            f"{_sql_string(field['name'])}:{_sql_string(field['logical_type'])}" for field in schema
        )
        + "}"
    )


def _csv_relation(path: Path, schema: Sequence[Mapping[str, str]]) -> str:
    return (
        f"read_csv({_sql_string(str(path))}, header=true, "
        f"columns={_csv_columns_literal(schema)}, sample_size=-1, "
        "strict_mode=true, ignore_errors=false, null_padding=false)"
    )


def convert_sharadar_zip_to_parquet(
    raw_zip_path: str | os.PathLike[str],
    parquet_path: str | os.PathLike[str],
    *,
    logical_name: str,
    datatable_metadata: Mapping[str, Any],
) -> Path:
    """Convert a provider ZIP with metadata-pinned types, never inference."""
    name = _logical_name(logical_name)
    metadata = normalize_datatable_metadata(name, datatable_metadata)
    schema = _typed_schema(metadata)
    raw_zip = Path(raw_zip_path)
    target = Path(parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_receipt = inspect_sharadar_zip(raw_zip, expected_header=[field["name"] for field in schema])
    temporary_csv: str | None = None
    temporary_parquet: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{name}.", suffix=".csv", delete=False
        ) as stream:
            temporary_csv = stream.name
        _extract_csv(raw_zip, Path(temporary_csv), member=csv_receipt["member"])
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{name}.", suffix=".parquet", delete=False
        ) as stream:
            temporary_parquet = stream.name
        os.unlink(temporary_parquet)
        connection = duckdb.connect(database=":memory:")
        try:
            relation = _csv_relation(Path(temporary_csv), schema)
            connection.execute(
                f"COPY (SELECT * FROM {relation}) TO "
                f"{_sql_string(temporary_parquet)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        except duckdb.Error as exc:
            raise SharadarSourceEvidenceError(f"{name} CSV-to-Parquet conversion failed") from exc
        finally:
            connection.close()
        with open(temporary_parquet, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_parquet, target)
        temporary_parquet = None
    finally:
        for temporary in (temporary_csv, temporary_parquet):
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
    return target


def _duckdb_type(value: str) -> str:
    upper = value.upper()
    if upper.startswith("VARCHAR"):
        return "VARCHAR"
    if upper == "DATE":
        return "DATE"
    if upper in {"DOUBLE", "FLOAT", "REAL"}:
        return "DOUBLE"
    if upper in {
        "BIGINT",
        "INTEGER",
        "SMALLINT",
        "TINYINT",
        "UBIGINT",
        "UINTEGER",
        "USMALLINT",
        "UTINYINT",
    }:
        return "BIGINT"
    if upper == "BOOLEAN":
        return "BOOLEAN"
    return upper


def _date_range_endpoint(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return _day(value, label)
    raise SharadarSourceEvidenceError(
        f"{label} must be a native date, canonical date text, or null"
    )


def inspect_sharadar_parquet(
    path: str | os.PathLike[str],
    *,
    logical_name: str,
    datatable_metadata: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate exact typed schema, primary-key quality, scope, and ranges."""
    name = _logical_name(logical_name)
    metadata = normalize_datatable_metadata(name, datatable_metadata)
    expected_schema = _typed_schema(metadata)
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise SharadarSourceEvidenceError(f"{name} Parquet is not a regular file: {source}")
    escaped = _sql_string(str(source))
    relation = f"read_parquet({escaped})"
    connection = duckdb.connect(database=":memory:")
    try:
        described = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        actual_schema = [
            {"name": str(row[0]), "logical_type": _duckdb_type(str(row[1]))} for row in described
        ]
        if actual_schema != expected_schema:
            raise SharadarSourceEvidenceError(
                f"{name} Parquet schema differs: expected {expected_schema}, got {actual_schema}"
            )
        primary_key = list(_TABLES[name]["primary_key"])
        key_null = " OR ".join(f"{_identifier(column)} IS NULL" for column in primary_key)
        row = connection.execute(
            f"SELECT count(*), count(DISTINCT {_identifier('ticker')}), "
            f"sum(CASE WHEN {key_null} THEN 1 ELSE 0 END) FROM {relation}"
        ).fetchone()
        duplicate_row = connection.execute(
            "SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) AS n FROM "
            f"{relation} GROUP BY "
            f"{', '.join(_identifier(column) for column in primary_key)} "
            "HAVING n > 1)"
        ).fetchone()
        if name == "sf1":
            scope_sql = "sum(CASE WHEN dimension IS DISTINCT FROM 'ARQ' THEN 1 ELSE 0 END)"
        elif name == "tickers":
            scope_sql = "sum(CASE WHEN \"table\" IS DISTINCT FROM 'SF1' THEN 1 ELSE 0 END)"
        else:
            scope_sql = "0"
        scope_row = connection.execute(f"SELECT {scope_sql} FROM {relation}").fetchone()
        date_ranges: list[dict[str, Any]] = []
        schema_names = {field["name"] for field in expected_schema}
        for column in _TABLES[name]["date_columns"]:
            if column not in schema_names:
                continue
            range_row = connection.execute(
                f"SELECT min({_identifier(column)}), max({_identifier(column)}) FROM {relation}"
            ).fetchone()
            assert range_row is not None
            date_ranges.append(
                {
                    "column": column,
                    "min": _date_range_endpoint(range_row[0], label=f"{name} {column} minimum"),
                    "max": _date_range_endpoint(range_row[1], label=f"{name} {column} maximum"),
                }
            )
    except duckdb.Error as exc:
        raise SharadarSourceEvidenceError(f"{name} Parquet is unreadable") from exc
    finally:
        connection.close()
    assert row is not None and duplicate_row is not None and scope_row is not None
    statistics = {
        "rows": int(row[0]),
        "distinct_tickers": int(row[1]),
        "primary_key": primary_key,
        "missing_primary_key_values": int(row[2] or 0),
        "duplicate_primary_keys": int(duplicate_row[0] or 0),
        "scope_violations": int(scope_row[0] or 0),
        "date_ranges": date_ranges,
    }
    if statistics["rows"] <= 0:
        raise SharadarSourceEvidenceError(f"{name} Parquet is empty")
    if statistics["missing_primary_key_values"]:
        raise SharadarSourceEvidenceError(f"{name} primary key contains nulls")
    if statistics["duplicate_primary_keys"]:
        raise SharadarSourceEvidenceError(f"{name} primary key contains duplicates")
    if statistics["scope_violations"]:
        raise SharadarSourceEvidenceError(f"{name} rows violate the canonical export scope")
    return actual_schema, statistics


def _normalize_statistics(value: Any, *, logical_name: str) -> dict[str, Any]:
    name = _logical_name(logical_name)
    statistics = _exact(value, _STATISTICS_FIELDS, f"{name} statistics")
    rows = _integer(statistics["rows"], f"{name} row count", minimum=1)
    distinct = _integer(statistics["distinct_tickers"], f"{name} distinct tickers", minimum=1)
    expected_primary_key = list(_TABLES[name]["primary_key"])
    if statistics["primary_key"] != expected_primary_key:
        raise SharadarSourceEvidenceError(f"{name} statistics primary key differs")
    missing = _integer(
        statistics["missing_primary_key_values"],
        f"{name} missing primary keys",
    )
    duplicates = _integer(statistics["duplicate_primary_keys"], f"{name} duplicate primary keys")
    violations = _integer(statistics["scope_violations"], f"{name} scope violations")
    if missing or duplicates or violations:
        raise SharadarSourceEvidenceError(f"{name} statistics are not qualifying")
    raw_ranges = statistics["date_ranges"]
    if not isinstance(raw_ranges, list):
        raise SharadarSourceEvidenceError(f"{name} date ranges must be an array")
    ranges: list[dict[str, Any]] = []
    expected_columns = [
        column
        for column in _TABLES[name]["date_columns"]
        if column in {field["name"] for field in _typed_schema_from_name(name)}
    ]
    for index, raw in enumerate(raw_ranges):
        item = _exact(raw, _DATE_RANGE_FIELDS, f"{name} date range {index}")
        column = _text(item["column"], f"{name} date range column")
        minimum = item["min"]
        maximum = item["max"]
        if (minimum is None) != (maximum is None):
            raise SharadarSourceEvidenceError(f"{name} date range is half-null")
        if minimum is not None:
            minimum = _day(minimum, f"{name} {column} minimum")
            maximum = _day(maximum, f"{name} {column} maximum")
            if minimum > maximum:
                raise SharadarSourceEvidenceError(f"{name} date range is reversed")
        ranges.append({"column": column, "min": minimum, "max": maximum})
    if [item["column"] for item in ranges] != expected_columns:
        raise SharadarSourceEvidenceError(f"{name} date-range columns differ")
    return {
        "rows": rows,
        "distinct_tickers": distinct,
        "primary_key": expected_primary_key,
        "missing_primary_key_values": 0,
        "duplicate_primary_keys": 0,
        "scope_violations": 0,
        "date_ranges": ranges,
    }


def _typed_schema_from_name(logical_name: str) -> list[dict[str, str]]:
    """Only used for date-range ordering after metadata was validated elsewhere."""
    # The caller replaces this projection with the receipt schema when exact
    # metadata is available.  Returning required date names here keeps the
    # structural statistics validator independent of provider type spelling.
    return [
        {"name": name, "logical_type": "DATE"} for name in _TABLES[logical_name]["date_columns"]
    ]


def prove_sharadar_row_equivalence(
    raw_zip_path: str | os.PathLike[str],
    parquet_path: str | os.PathLike[str],
    *,
    logical_name: str,
    datatable_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the raw CSV and compare row multisets in DuckDB.

    ``EXCEPT ALL`` preserves duplicate multiplicity.  DuckDB performs the
    comparison internally and may spill to disk; no Python list scales with the
    46-million-row SEP table.
    """
    name = _logical_name(logical_name)
    metadata = normalize_datatable_metadata(name, datatable_metadata)
    schema = _typed_schema(metadata)
    raw_zip = Path(raw_zip_path)
    parquet = Path(parquet_path)
    csv_receipt = inspect_sharadar_zip(raw_zip, expected_header=[field["name"] for field in schema])
    with tempfile.TemporaryDirectory(prefix=f"sharadar-{name}-proof-") as temporary:
        csv_path = Path(temporary) / f"{name}.csv"
        _extract_csv(raw_zip, csv_path, member=csv_receipt["member"])
        csv_relation = _csv_relation(csv_path, schema)
        parquet_relation = f"read_parquet({_sql_string(str(parquet))})"
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(f"SET temp_directory={_sql_string(str(Path(temporary) / 'spill'))}")
            csv_count_row = connection.execute(f"SELECT count(*) FROM {csv_relation}").fetchone()
            parquet_count_row = connection.execute(
                f"SELECT count(*) FROM {parquet_relation}"
            ).fetchone()
            csv_minus_row = connection.execute(
                "SELECT count(*) FROM ("
                f"SELECT * FROM {csv_relation} EXCEPT ALL "
                f"SELECT * FROM {parquet_relation})"
            ).fetchone()
            parquet_minus_row = connection.execute(
                "SELECT count(*) FROM ("
                f"SELECT * FROM {parquet_relation} EXCEPT ALL "
                f"SELECT * FROM {csv_relation})"
            ).fetchone()
        except duckdb.Error as exc:
            raise SharadarSourceEvidenceError(
                f"{name} CSV/Parquet equivalence replay failed"
            ) from exc
        finally:
            connection.close()
    assert (
        csv_count_row is not None
        and parquet_count_row is not None
        and csv_minus_row is not None
        and parquet_minus_row is not None
    )
    csv_rows = int(csv_count_row[0])
    parquet_rows = int(parquet_count_row[0])
    csv_minus = int(csv_minus_row[0])
    parquet_minus = int(parquet_minus_row[0])
    equivalent = csv_rows == parquet_rows and csv_minus == 0 and parquet_minus == 0
    if not equivalent:
        raise SharadarSourceEvidenceError(f"{name} CSV and Parquet row multisets differ")
    return {
        "schema_version": SHARADAR_ROW_EQUIVALENCE_SCHEMA_VERSION,
        "method": "duckdb_bidirectional_except_all.v1",
        "rows": csv_rows,
        "csv_minus_parquet_rows": csv_minus,
        "parquet_minus_csv_rows": parquet_minus,
        "equivalent": True,
    }


def _normalize_row_equivalence(value: Any, *, logical_name: str) -> dict[str, Any]:
    name = _logical_name(logical_name)
    proof = _exact(value, _ROW_EQUIVALENCE_FIELDS, f"{name} row equivalence")
    if proof["schema_version"] != SHARADAR_ROW_EQUIVALENCE_SCHEMA_VERSION:
        raise SharadarSourceEvidenceError(f"{name} row-equivalence schema differs")
    if proof["method"] != "duckdb_bidirectional_except_all.v1":
        raise SharadarSourceEvidenceError(f"{name} row-equivalence method differs")
    rows = _integer(proof["rows"], f"{name} row-equivalence rows", minimum=1)
    csv_minus = _integer(proof["csv_minus_parquet_rows"], f"{name} CSV-minus-Parquet rows")
    parquet_minus = _integer(proof["parquet_minus_csv_rows"], f"{name} Parquet-minus-CSV rows")
    if proof["equivalent"] is not True or csv_minus or parquet_minus:
        raise SharadarSourceEvidenceError(f"{name} row equivalence is not qualifying")
    return {
        "schema_version": SHARADAR_ROW_EQUIVALENCE_SCHEMA_VERSION,
        "method": "duckdb_bidirectional_except_all.v1",
        "rows": rows,
        "csv_minus_parquet_rows": 0,
        "parquet_minus_csv_rows": 0,
        "equivalent": True,
    }


def _normalize_source(logical_name: str, value: Any) -> dict[str, Any]:
    name = _logical_name(logical_name)
    source = _exact(value, _SOURCE_FIELDS, f"{name} source")
    vendor_code, datatable_code = _TABLES[name]["dataset"].split("/", 1)
    if source["vendor_code"] != vendor_code or source["datatable_code"] != datatable_code:
        raise SharadarSourceEvidenceError(f"{name} source identity differs")
    request = _exact(source["canonical_request"], _REQUEST_FIELDS, f"{name} request")
    expected_request = _canonical_request(name)
    if _plain(request) != expected_request:
        raise SharadarSourceEvidenceError(f"{name} canonical request differs")
    refreshed = _utc(
        source["last_refreshed_time"], f"{name} last_refreshed_time", provider_format=True
    )
    snapshot_time = _utc(
        source["data_snapshot_time"], f"{name} data_snapshot_time", provider_format=True
    )
    if source["bulk_status"] != "fresh":
        raise SharadarSourceEvidenceError(f"{name} bulk export was not fresh")
    metadata = normalize_datatable_metadata(name, source["datatable_metadata"])
    return {
        "vendor_code": vendor_code,
        "datatable_code": datatable_code,
        "canonical_request": expected_request,
        "last_refreshed_time": refreshed,
        "data_snapshot_time": snapshot_time,
        "bulk_status": "fresh",
        "datatable_metadata": metadata,
    }


def build_sharadar_table_acquisition_document(
    *,
    logical_name: str,
    raw_zip_path: str | os.PathLike[str],
    parquet_path: str | os.PathLike[str],
    acquired_at_utc: str,
    last_refreshed_time: str,
    data_snapshot_time: str,
    datatable_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect staged bytes and build their immutable acquisition receipt."""
    name = _logical_name(logical_name)
    acquired = _utc(acquired_at_utc, "acquired_at_utc")
    metadata = normalize_datatable_metadata(name, datatable_metadata)
    refreshed = _utc(last_refreshed_time, f"{name} last_refreshed_time", provider_format=True)
    snapshot_time = _utc(data_snapshot_time, f"{name} data_snapshot_time", provider_format=True)
    raw_zip = Path(raw_zip_path)
    parquet = Path(parquet_path)
    schema = _typed_schema(metadata)
    csv_receipt = inspect_sharadar_zip(raw_zip, expected_header=[field["name"] for field in schema])
    actual_schema, statistics = inspect_sharadar_parquet(
        parquet, logical_name=name, datatable_metadata=metadata
    )
    proof = prove_sharadar_row_equivalence(
        raw_zip,
        parquet,
        logical_name=name,
        datatable_metadata=metadata,
    )
    if proof["rows"] != statistics["rows"]:
        raise SharadarSourceEvidenceError(f"{name} row-equivalence count differs from statistics")
    raw_hash = file_sha256(raw_zip)
    parquet_hash = file_sha256(parquet)
    payload = {
        "schema_version": SHARADAR_TABLE_ACQUISITION_SCHEMA_VERSION,
        "logical_name": name,
        "source": {
            "vendor_code": "SHARADAR",
            "datatable_code": _TABLES[name]["dataset"].split("/", 1)[1],
            "canonical_request": _canonical_request(name),
            "last_refreshed_time": refreshed,
            "data_snapshot_time": snapshot_time,
            "bulk_status": "fresh",
            "datatable_metadata": metadata,
        },
        "acquired_at_utc": acquired,
        "raw_zip": {
            "relative_path": (f"{SHARADAR_RECEIPT_ROOT}/{name}/raw/{raw_hash}.zip"),
            "sha256": raw_hash,
            "bytes": raw_zip.stat().st_size,
        },
        "csv": csv_receipt,
        "parquet": {
            "relative_path": (f"{SHARADAR_RECEIPT_ROOT}/{name}/parquet/{parquet_hash}.parquet"),
            "sha256": parquet_hash,
            "bytes": parquet.stat().st_size,
            "schema": actual_schema,
            "statistics": statistics,
        },
        "row_equivalence": proof,
    }
    return {
        "artifact_hash": content_hash(payload),
        "payload": _plain(payload),
    }


def _provider_datetime(value: str) -> datetime:
    text = value.strip().replace(" UTC", "+00:00")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _normalize_table_document(document: Mapping[str, Any]) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "Sharadar acquisition receipt")
    payload = _exact(wrapper["payload"], _TABLE_PAYLOAD_FIELDS, "Sharadar acquisition payload")
    claimed = _sha(wrapper["artifact_hash"], "Sharadar acquisition artifact_hash")
    if content_hash(payload) != claimed:
        raise SharadarSourceEvidenceError("Sharadar acquisition artifact hash mismatch")
    if payload["schema_version"] != SHARADAR_TABLE_ACQUISITION_SCHEMA_VERSION:
        raise SharadarSourceEvidenceError("unsupported Sharadar acquisition schema")
    name = _logical_name(payload["logical_name"])
    acquired = _utc(payload["acquired_at_utc"], "acquired_at_utc")
    source = _normalize_source(name, payload["source"])
    if _provider_datetime(source["data_snapshot_time"]) > _provider_datetime(acquired):
        raise SharadarSourceEvidenceError(f"{name} data snapshot timestamp follows acquisition")
    if _provider_datetime(source["last_refreshed_time"]) > _provider_datetime(acquired):
        raise SharadarSourceEvidenceError(f"{name} last-refreshed timestamp follows acquisition")

    raw = _exact(payload["raw_zip"], _RAW_FIELDS, f"{name} raw ZIP")
    raw_hash = _sha(raw["sha256"], f"{name} raw ZIP sha256")
    expected_raw_path = f"{SHARADAR_RECEIPT_ROOT}/{name}/raw/{raw_hash}.zip"
    if raw["relative_path"] != expected_raw_path:
        raise SharadarSourceEvidenceError(f"{name} raw ZIP path is not content-addressed")
    raw_bytes = _integer(raw["bytes"], f"{name} raw ZIP bytes", minimum=1)

    expected_schema = _typed_schema(source["datatable_metadata"])
    csv_value = _exact(payload["csv"], _CSV_FIELDS, f"{name} CSV")
    member = _text(csv_value["member"], f"{name} CSV member")
    if Path(member).name != member or not member.lower().endswith(".csv"):
        raise SharadarSourceEvidenceError(f"{name} CSV member is unsafe")
    csv_receipt = {
        "member": member,
        "sha256": _sha(csv_value["sha256"], f"{name} CSV sha256"),
        "bytes": _integer(csv_value["bytes"], f"{name} CSV bytes", minimum=1),
        "compressed_bytes": _integer(
            csv_value["compressed_bytes"], f"{name} compressed CSV bytes", minimum=1
        ),
        "header": csv_value["header"],
    }
    expected_header = [field["name"] for field in expected_schema]
    if csv_receipt["header"] != expected_header:
        raise SharadarSourceEvidenceError(f"{name} CSV header differs from metadata")

    parquet_value = _exact(payload["parquet"], _PARQUET_FIELDS, f"{name} Parquet")
    parquet_hash = _sha(parquet_value["sha256"], f"{name} Parquet sha256")
    expected_parquet_path = f"{SHARADAR_RECEIPT_ROOT}/{name}/parquet/{parquet_hash}.parquet"
    if parquet_value["relative_path"] != expected_parquet_path:
        raise SharadarSourceEvidenceError(f"{name} Parquet path is not content-addressed")
    parquet_schema = _normalize_schema(parquet_value["schema"], label=f"{name} Parquet schema")
    if parquet_schema != expected_schema:
        raise SharadarSourceEvidenceError(f"{name} Parquet schema differs from metadata")
    statistics = _normalize_statistics(parquet_value["statistics"], logical_name=name)
    parquet = {
        "relative_path": expected_parquet_path,
        "sha256": parquet_hash,
        "bytes": _integer(parquet_value["bytes"], f"{name} Parquet bytes", minimum=1),
        "schema": parquet_schema,
        "statistics": statistics,
    }
    proof = _normalize_row_equivalence(payload["row_equivalence"], logical_name=name)
    if proof["rows"] != statistics["rows"]:
        raise SharadarSourceEvidenceError(f"{name} row-equivalence count differs from statistics")
    normalized_payload = {
        "schema_version": SHARADAR_TABLE_ACQUISITION_SCHEMA_VERSION,
        "logical_name": name,
        "source": source,
        "acquired_at_utc": acquired,
        "raw_zip": {
            "relative_path": expected_raw_path,
            "sha256": raw_hash,
            "bytes": raw_bytes,
        },
        "csv": csv_receipt,
        "parquet": parquet,
        "row_equivalence": proof,
    }
    if content_hash(normalized_payload) != claimed:
        raise SharadarSourceEvidenceError("Sharadar acquisition receipt is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized_payload)}


def _resolve_under(root: Path, relative_path: str, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise SharadarSourceEvidenceError(f"{label} relative path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SharadarSourceEvidenceError(f"{label} relative path escapes the warehouse")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SharadarSourceEvidenceError(f"{label} is missing or escapes the warehouse") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise SharadarSourceEvidenceError(f"{label} is not a regular file")
    return resolved


def validate_sharadar_table_acquisition(
    warehouse_dir: str | os.PathLike[str], document: Mapping[str, Any]
) -> dict[str, Any]:
    """Authoritatively reopen raw/converted bytes and replay the table receipt."""
    root = Path(warehouse_dir).resolve()
    normalized = _normalize_table_document(document)
    payload = normalized["payload"]
    name = payload["logical_name"]
    raw_path = _resolve_under(root, payload["raw_zip"]["relative_path"], label=f"{name} raw ZIP")
    parquet_path = _resolve_under(
        root, payload["parquet"]["relative_path"], label=f"{name} Parquet"
    )
    if (
        raw_path.stat().st_size != payload["raw_zip"]["bytes"]
        or file_sha256(raw_path) != payload["raw_zip"]["sha256"]
    ):
        raise SharadarSourceEvidenceError(f"{name} raw ZIP identity mismatch")
    if (
        parquet_path.stat().st_size != payload["parquet"]["bytes"]
        or file_sha256(parquet_path) != payload["parquet"]["sha256"]
    ):
        raise SharadarSourceEvidenceError(f"{name} Parquet identity mismatch")
    metadata = payload["source"]["datatable_metadata"]
    actual_csv = inspect_sharadar_zip(
        raw_path,
        expected_header=[field["name"] for field in payload["parquet"]["schema"]],
    )
    if actual_csv != payload["csv"]:
        raise SharadarSourceEvidenceError(f"{name} CSV receipt mismatch")
    actual_schema, actual_statistics = inspect_sharadar_parquet(
        parquet_path,
        logical_name=name,
        datatable_metadata=metadata,
    )
    if (
        actual_schema != payload["parquet"]["schema"]
        or actual_statistics != payload["parquet"]["statistics"]
    ):
        raise SharadarSourceEvidenceError(f"{name} Parquet inspection differs")
    actual_proof = prove_sharadar_row_equivalence(
        raw_path,
        parquet_path,
        logical_name=name,
        datatable_metadata=metadata,
    )
    if actual_proof != payload["row_equivalence"]:
        raise SharadarSourceEvidenceError(f"{name} row-equivalence replay differs")
    return normalized


def load_sharadar_table_acquisition(
    path: str | os.PathLike[str], *, warehouse_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    return validate_sharadar_table_acquisition(
        warehouse_dir, _strict_json(Path(path), max_bytes=MAX_RECEIPT_BYTES)
    )


def _copy_create_only(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = file_sha256(source)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise SharadarSourceEvidenceError(
                f"immutable Sharadar destination is unsafe: {destination}"
            )
        if file_sha256(destination) != expected_hash:
            raise SharadarSourceEvidenceError(
                f"content-addressed Sharadar file collision: {destination}"
            )
        return destination
    temporary: str | None = None
    try:
        with (
            source.open("rb") as incoming,
            tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as outgoing,
        ):
            temporary = outgoing.name
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if file_sha256(destination) != expected_hash:
                raise SharadarSourceEvidenceError(
                    f"content-addressed Sharadar file collision: {destination}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def _publish_json_create_only(path: Path, document: Mapping[str, Any]) -> Path:
    encoded = (canonical_json(document) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            raise SharadarSourceEvidenceError(f"content-addressed Sharadar JSON collision: {path}")
        return path
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise SharadarSourceEvidenceError(
                    f"content-addressed Sharadar JSON collision: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


def publish_sharadar_table_acquisition(
    warehouse_dir: str | os.PathLike[str],
    *,
    raw_zip_path: str | os.PathLike[str],
    parquet_path: str | os.PathLike[str],
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Create immutable raw/Parquet/receipt files, then revalidate all three."""
    root = Path(warehouse_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_table_document(document)
    payload = normalized["payload"]
    archived_raw = root / payload["raw_zip"]["relative_path"]
    archived_parquet = root / payload["parquet"]["relative_path"]
    _copy_create_only(Path(raw_zip_path), archived_raw)
    _copy_create_only(Path(parquet_path), archived_parquet)
    receipt_path = (
        root
        / SHARADAR_RECEIPT_ROOT
        / payload["logical_name"]
        / "receipts"
        / f"{normalized['artifact_hash']}.json"
    )
    _publish_json_create_only(receipt_path, normalized)
    verified = validate_sharadar_table_acquisition(root, normalized)
    if _strict_json(receipt_path, max_bytes=MAX_RECEIPT_BYTES) != verified:
        raise SharadarSourceEvidenceError("published Sharadar receipt changed")
    return verified, receipt_path


def _identity_core(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    identity = _exact(value, _IDENTITY_FIELDS, label)
    cik = identity["cik"]
    if not isinstance(cik, str) or len(cik) != 10 or not cik.isdigit() or cik == "0000000000":
        raise SharadarSourceEvidenceError(f"{label}.cik must be a positive 10-digit CIK")
    permaticker = _integer(identity["permaticker"], f"{label}.permaticker", minimum=1)
    ticker = _text(identity["ticker"], f"{label}.ticker")
    if ticker != ticker.upper():
        raise SharadarSourceEvidenceError(f"{label}.ticker must be uppercase")
    valid_from = _day(identity["valid_from"], f"{label}.valid_from")
    valid_through = _day(identity["valid_through"], f"{label}.valid_through")
    if valid_from > valid_through:
        raise SharadarSourceEvidenceError(f"{label} validity interval is reversed")
    is_delisted = identity["is_delisted"]
    if is_delisted not in {"Y", "N"}:
        raise SharadarSourceEvidenceError(f"{label}.is_delisted must be Y or N")
    core = {
        "cik": cik,
        "permaticker": permaticker,
        "ticker": ticker,
        "valid_from": valid_from,
        "valid_through": valid_through,
        "is_delisted": is_delisted,
        "category": _text(identity["category"], f"{label}.category"),
        "exchange": _text(identity["exchange"], f"{label}.exchange"),
        "currency": _text(identity["currency"], f"{label}.currency").upper(),
        "source_record_sha256": _sha(
            identity["source_record_sha256"], f"{label}.source_record_sha256"
        ),
    }
    identity_id = _sha(identity["identity_id"], f"{label}.identity_id")
    expected_id = content_hash({"schema_version": PEAD_SECURITY_IDENTITY_SCHEMA_VERSION, **core})
    if identity_id != expected_id:
        raise SharadarSourceEvidenceError(f"{label}.identity_id differs from its fields")
    return {"identity_id": identity_id, **core}


def _intervals_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["valid_from"] <= right["valid_through"]
        and right["valid_from"] <= left["valid_through"]
    )


def _accepted_identity_conflicts(identities: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return source hashes involved in contradictions; multi-class CIKs are valid."""
    conflicts: set[str] = set()
    by_permaticker: dict[int, list[Mapping[str, Any]]] = {}
    by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    for identity in identities:
        by_permaticker.setdefault(identity["permaticker"], []).append(identity)
        by_ticker.setdefault(identity["ticker"], []).append(identity)
    for rows in by_permaticker.values():
        if len({row["cik"] for row in rows}) > 1:
            conflicts.update(row["source_record_sha256"] for row in rows)
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if _intervals_overlap(left, right):
                    conflicts.add(left["source_record_sha256"])
                    conflicts.add(right["source_record_sha256"])
    for rows in by_ticker.values():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if left["permaticker"] != right["permaticker"] and _intervals_overlap(left, right):
                    conflicts.add(left["source_record_sha256"])
                    conflicts.add(right["source_record_sha256"])
    return conflicts


def validate_pead_security_identity_snapshot_structure(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate internal identity structure only; this is non-authoritative.

    Qualification code must additionally require a trusted artifact hash or use
    :func:`validate_pead_security_identity_snapshot`, which reopens the bound
    Sharadar acquisition and deterministically rebuilds every row.
    """
    wrapper = _exact(document, _WRAPPER_FIELDS, "security identity snapshot")
    payload = _exact(wrapper["payload"], _IDENTITY_PAYLOAD_FIELDS, "security identity payload")
    claimed = _sha(wrapper["artifact_hash"], "identity artifact_hash")
    if content_hash(payload) != claimed:
        raise SharadarSourceEvidenceError("security identity artifact hash mismatch")
    if payload["schema_version"] != PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION:
        raise SharadarSourceEvidenceError("unsupported security identity schema")
    candidate_id = _text(payload["candidate_id"], "identity candidate_id")
    created = _utc(payload["created_at_utc"], "identity created_at_utc")
    raw_bindings = _exact(payload["bindings"], _IDENTITY_BINDING_FIELDS, "identity bindings")
    bindings = {
        key: _sha(raw_bindings[key], f"identity bindings.{key}")
        for key in sorted(_IDENTITY_BINDING_FIELDS)
    }

    raw_identities = payload["identities"]
    if not isinstance(raw_identities, list):
        raise SharadarSourceEvidenceError("identities must be an array")
    identities = [
        _identity_core(raw, label=f"identities[{index}]")
        for index, raw in enumerate(raw_identities)
    ]
    identities.sort(key=lambda item: item["identity_id"])
    if payload["identities"] != identities:
        raise SharadarSourceEvidenceError("identities must be canonically sorted")
    identity_ids = [item["identity_id"] for item in identities]
    source_hashes = [item["source_record_sha256"] for item in identities]
    if len(identity_ids) != len(set(identity_ids)) or len(source_hashes) != len(set(source_hashes)):
        raise SharadarSourceEvidenceError("accepted identities are duplicated")
    if _accepted_identity_conflicts(identities):
        raise SharadarSourceEvidenceError("accepted identities contain interval conflicts")

    raw_dispositions = payload["source_dispositions"]
    if not isinstance(raw_dispositions, list):
        raise SharadarSourceEvidenceError("identity dispositions must be an array")
    dispositions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_dispositions):
        item = _exact(raw, _IDENTITY_DISPOSITION_FIELDS, f"source_dispositions[{index}]")
        source_hash = _sha(
            item["source_record_sha256"],
            f"source_dispositions[{index}].source_record_sha256",
        )
        disposition = item["disposition"]
        if disposition == "identity":
            identity_id = _sha(item["identity_id"], f"source_dispositions[{index}].identity_id")
            if item["reason"] is not None:
                raise SharadarSourceEvidenceError(
                    "accepted identity disposition cannot have a reason"
                )
        elif disposition == "identity_gap":
            if item["identity_id"] is not None:
                raise SharadarSourceEvidenceError("identity gap cannot reference an identity_id")
            identity_id = None
            reason = _text(item["reason"], f"source_dispositions[{index}].reason")
            if re.fullmatch(r"[a-z][a-z0-9_]*", reason) is None:
                raise SharadarSourceEvidenceError(
                    "identity gap reason must be a lowercase machine reason"
                )
        else:
            raise SharadarSourceEvidenceError("unsupported identity disposition")
        dispositions.append(
            {
                "source_record_sha256": source_hash,
                "disposition": disposition,
                "identity_id": identity_id,
                "reason": item["reason"],
            }
        )
    dispositions.sort(key=lambda item: item["source_record_sha256"])
    if payload["source_dispositions"] != dispositions:
        raise SharadarSourceEvidenceError("identity dispositions must be canonically sorted")
    disposition_hashes = [item["source_record_sha256"] for item in dispositions]
    if len(disposition_hashes) != len(set(disposition_hashes)):
        raise SharadarSourceEvidenceError("identity source rows are duplicated")
    accepted = {
        item["source_record_sha256"]: item["identity_id"]
        for item in dispositions
        if item["disposition"] == "identity"
    }
    expected_accepted = {item["source_record_sha256"]: item["identity_id"] for item in identities}
    if accepted != expected_accepted:
        raise SharadarSourceEvidenceError("identity dispositions do not match accepted identities")
    gap_count = sum(item["disposition"] == "identity_gap" for item in dispositions)
    counts = {
        "source_row_count": len(dispositions),
        "disposition_count": len(dispositions),
        "identity_count": len(identities),
        "identity_gap_count": gap_count,
        "complete": bool(dispositions),
    }
    _exact(payload["coverage"], _IDENTITY_COVERAGE_FIELDS, "identity coverage")
    if payload["coverage"] != counts:
        raise SharadarSourceEvidenceError("identity coverage is not derived exactly")
    blockers: list[str] = []
    if not dispositions:
        blockers.append("identity_source_empty")
    if not identities:
        blockers.append("no_usable_identities")
    allowed = bool(dispositions and identities and not blockers)
    if payload["blockers"] != blockers:
        raise SharadarSourceEvidenceError("identity blockers are not derived exactly")
    if payload["qualification_allowed"] is not allowed:
        raise SharadarSourceEvidenceError("identity qualification claim is inconsistent")
    normalized_payload = {
        "schema_version": PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "created_at_utc": created,
        "bindings": bindings,
        "source_dispositions": dispositions,
        "identities": identities,
        "coverage": counts,
        "blockers": blockers,
        "qualification_allowed": allowed,
    }
    if content_hash(normalized_payload) != claimed:
        raise SharadarSourceEvidenceError("security identity snapshot is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized_payload)}


def _strict_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SharadarSourceEvidenceError(f"evidence is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise SharadarSourceEvidenceError(f"evidence exceeds its size limit: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SharadarSourceEvidenceError(f"evidence is not UTF-8: {path}") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SharadarSourceEvidenceError(
                    f"evidence contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise SharadarSourceEvidenceError(f"evidence contains invalid number {token}: {path}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise SharadarSourceEvidenceError(
            f"invalid evidence JSON at line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SharadarSourceEvidenceError(f"evidence root must be an object: {path}")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise SharadarSourceEvidenceError(
            f"evidence bytes are not canonical JSON plus one newline: {path}"
        )
    return value


def _source_snapshot_receipt_relative(name: str, artifact_hash: str) -> str:
    return f"{SHARADAR_RECEIPT_ROOT}/{name}/receipts/{artifact_hash}.json"


def _normalize_source_snapshot_structure(document: Mapping[str, Any]) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "Sharadar source snapshot")
    payload = _exact(
        wrapper["payload"], _SOURCE_SNAPSHOT_FIELDS, "Sharadar source snapshot payload"
    )
    claimed = _sha(wrapper["artifact_hash"], "Sharadar source snapshot artifact_hash")
    if content_hash(payload) != claimed:
        raise SharadarSourceEvidenceError("Sharadar source snapshot artifact hash mismatch")
    if payload["schema_version"] != PEAD_SHARADAR_SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise SharadarSourceEvidenceError("unsupported Sharadar source snapshot schema")
    candidate_id = _text(payload["candidate_id"], "source snapshot candidate_id")
    created = _utc(payload["created_at_utc"], "source snapshot created_at_utc")
    raw_tables = payload["tables"]
    if not isinstance(raw_tables, list):
        raise SharadarSourceEvidenceError("source snapshot tables must be an array")
    tables: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_tables):
        item = _exact(raw, _SOURCE_SNAPSHOT_TABLE_FIELDS, f"snapshot table {index}")
        name = _logical_name(item["logical_name"])
        artifact_hash = _sha(
            item["acquisition_artifact_hash"], f"snapshot table {name} acquisition hash"
        )
        expected_receipt = _source_snapshot_receipt_relative(name, artifact_hash)
        if item["acquisition_receipt_relative_path"] != expected_receipt:
            raise SharadarSourceEvidenceError(
                f"snapshot table {name} receipt path is not content-addressed"
            )
        expected_code = _TABLES[name]["dataset"].split("/", 1)[1]
        if item["datatable_code"] != expected_code:
            raise SharadarSourceEvidenceError(f"snapshot table {name} code differs")
        tables.append(
            {
                "logical_name": name,
                "datatable_code": expected_code,
                "acquisition_artifact_hash": artifact_hash,
                "acquisition_receipt_relative_path": expected_receipt,
                "data_snapshot_time": _utc(
                    item["data_snapshot_time"],
                    f"snapshot table {name} data_snapshot_time",
                    provider_format=True,
                ),
                "raw_zip_sha256": _sha(item["raw_zip_sha256"], f"snapshot table {name} raw ZIP"),
                "parquet_sha256": _sha(item["parquet_sha256"], f"snapshot table {name} Parquet"),
                "row_count": _integer(
                    item["row_count"], f"snapshot table {name} row count", minimum=1
                ),
            }
        )
    order = {name: index for index, name in enumerate(CANDIDATE_TABLES)}
    tables.sort(key=lambda item: order[item["logical_name"]])
    if payload["tables"] != tables:
        raise SharadarSourceEvidenceError("source snapshot tables are not canonically sorted")
    present = [item["logical_name"] for item in tables]
    if len(present) != len(set(present)):
        raise SharadarSourceEvidenceError("source snapshot contains duplicate tables")
    coverage = {
        "required_tables": list(CANDIDATE_TABLES),
        "present_tables": present,
        "complete": present == list(CANDIDATE_TABLES),
    }
    _exact(payload["coverage"], _SOURCE_SNAPSHOT_COVERAGE_FIELDS, "snapshot coverage")
    if payload["coverage"] != coverage:
        raise SharadarSourceEvidenceError("source snapshot coverage is not derived exactly")
    blockers = [f"missing_table_{name}" for name in CANDIDATE_TABLES if name not in set(present)]
    allowed = not blockers
    if payload["blockers"] != blockers:
        raise SharadarSourceEvidenceError("source snapshot blockers are not derived exactly")
    if payload["qualification_allowed"] is not allowed:
        raise SharadarSourceEvidenceError("source snapshot qualification claim is inconsistent")
    normalized_payload = {
        "schema_version": PEAD_SHARADAR_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "created_at_utc": created,
        "tables": tables,
        "coverage": coverage,
        "blockers": blockers,
        "qualification_allowed": allowed,
    }
    if content_hash(normalized_payload) != claimed:
        raise SharadarSourceEvidenceError("Sharadar source snapshot is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized_payload)}


def build_pead_sharadar_source_snapshot(
    *,
    warehouse_dir: str | os.PathLike[str],
    candidate_id: str,
    created_at_utc: str,
    acquisitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact three-table source snapshot from immutable receipts."""
    if not isinstance(acquisitions, Mapping):
        raise SharadarSourceEvidenceError("acquisitions must be a mapping")
    unknown = sorted(set(acquisitions) - set(CANDIDATE_TABLES))
    if unknown:
        raise SharadarSourceEvidenceError(f"source snapshot has unknown tables: {unknown}")
    created = _utc(created_at_utc, "source snapshot created_at_utc")
    root = Path(warehouse_dir).resolve()
    tables: list[dict[str, Any]] = []
    for name in CANDIDATE_TABLES:
        if name not in acquisitions:
            continue
        acquisition = validate_sharadar_table_acquisition(root, acquisitions[name])
        payload = acquisition["payload"]
        if payload["logical_name"] != name:
            raise SharadarSourceEvidenceError(f"source snapshot mapping key differs for {name}")
        if _provider_datetime(payload["acquired_at_utc"]) > _provider_datetime(created):
            raise SharadarSourceEvidenceError(
                f"source snapshot timestamp precedes {name} acquisition"
            )
        receipt_relative = _source_snapshot_receipt_relative(name, acquisition["artifact_hash"])
        receipt_path = _resolve_under(root, receipt_relative, label=f"{name} receipt")
        if _strict_json(receipt_path, max_bytes=MAX_RECEIPT_BYTES) != acquisition:
            raise SharadarSourceEvidenceError(f"{name} immutable receipt differs")
        tables.append(
            {
                "logical_name": name,
                "datatable_code": payload["source"]["datatable_code"],
                "acquisition_artifact_hash": acquisition["artifact_hash"],
                "acquisition_receipt_relative_path": receipt_relative,
                "data_snapshot_time": payload["source"]["data_snapshot_time"],
                "raw_zip_sha256": payload["raw_zip"]["sha256"],
                "parquet_sha256": payload["parquet"]["sha256"],
                "row_count": payload["parquet"]["statistics"]["rows"],
            }
        )
    present = [item["logical_name"] for item in tables]
    blockers = [name for name in CANDIDATE_TABLES if name not in set(present)]
    payload = {
        "schema_version": PEAD_SHARADAR_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "candidate_id": _text(candidate_id, "source snapshot candidate_id"),
        "created_at_utc": created,
        "tables": tables,
        "coverage": {
            "required_tables": list(CANDIDATE_TABLES),
            "present_tables": present,
            "complete": not blockers,
        },
        "blockers": [f"missing_table_{name}" for name in blockers],
        "qualification_allowed": not blockers,
    }
    return _normalize_source_snapshot_structure(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_sharadar_source_snapshot(
    document: Mapping[str, Any], *, warehouse_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Reopen every immutable acquisition receipt and all source bytes."""
    root = Path(warehouse_dir).resolve()
    normalized = _normalize_source_snapshot_structure(document)
    for table in normalized["payload"]["tables"]:
        receipt_path = _resolve_under(
            root,
            table["acquisition_receipt_relative_path"],
            label=f"{table['logical_name']} receipt",
        )
        acquisition = validate_sharadar_table_acquisition(
            root, _strict_json(receipt_path, max_bytes=MAX_RECEIPT_BYTES)
        )
        payload = acquisition["payload"]
        expected = {
            "logical_name": payload["logical_name"],
            "datatable_code": payload["source"]["datatable_code"],
            "acquisition_artifact_hash": acquisition["artifact_hash"],
            "acquisition_receipt_relative_path": table["acquisition_receipt_relative_path"],
            "data_snapshot_time": payload["source"]["data_snapshot_time"],
            "raw_zip_sha256": payload["raw_zip"]["sha256"],
            "parquet_sha256": payload["parquet"]["sha256"],
            "row_count": payload["parquet"]["statistics"]["rows"],
        }
        if table != expected:
            raise SharadarSourceEvidenceError(
                f"source snapshot table {table['logical_name']} differs from receipt"
            )
    return normalized


def publish_pead_sharadar_source_snapshot(
    warehouse_dir: str | os.PathLike[str], document: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    root = Path(warehouse_dir).resolve()
    normalized = validate_pead_sharadar_source_snapshot(document, warehouse_dir=root)
    path = root / SHARADAR_RECEIPT_ROOT / "snapshots" / f"{normalized['artifact_hash']}.json"
    _publish_json_create_only(path, normalized)
    if _strict_json(path, max_bytes=MAX_RECEIPT_BYTES) != normalized:
        raise SharadarSourceEvidenceError("published source snapshot changed")
    return normalized, path


def load_pead_sharadar_source_snapshot(
    path: str | os.PathLike[str], *, warehouse_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    return validate_pead_sharadar_source_snapshot(
        _strict_json(Path(path), max_bytes=MAX_RECEIPT_BYTES),
        warehouse_dir=warehouse_dir,
    )


def _tickers_acquisition(source_snapshot: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    table = next(
        (
            item
            for item in source_snapshot["payload"]["tables"]
            if item["logical_name"] == "tickers"
        ),
        None,
    )
    if table is None:
        raise SharadarSourceEvidenceError("source snapshot has no TICKERS acquisition")
    receipt_path = _resolve_under(
        root, table["acquisition_receipt_relative_path"], label="TICKERS receipt"
    )
    return validate_sharadar_table_acquisition(
        root, _strict_json(receipt_path, max_bytes=MAX_RECEIPT_BYTES)
    )


def _parse_tickers_identity(
    row: Mapping[str, Any], *, source_record_sha256: str
) -> tuple[dict[str, Any] | None, str | None]:
    raw_url = row.get("secfilings")
    match = _CIK_URL.fullmatch(raw_url) if isinstance(raw_url, str) else None
    if match is None or match.group(1) == "0000000000":
        return None, "invalid_cik_source_url"
    raw_permaticker = row.get("permaticker")
    if type(raw_permaticker) is not int or raw_permaticker <= 0:
        return None, "invalid_permaticker"
    raw_ticker = row.get("ticker")
    if (
        not isinstance(raw_ticker, str)
        or not raw_ticker
        or raw_ticker != raw_ticker.strip().upper()
    ):
        return None, "invalid_ticker"
    try:
        valid_from = (
            row["firstpricedate"].isoformat()
            if isinstance(row.get("firstpricedate"), date)
            and not isinstance(row.get("firstpricedate"), datetime)
            else _day(row.get("firstpricedate"), "TICKERS firstpricedate")
        )
        valid_through = (
            row["lastpricedate"].isoformat()
            if isinstance(row.get("lastpricedate"), date)
            and not isinstance(row.get("lastpricedate"), datetime)
            else _day(row.get("lastpricedate"), "TICKERS lastpricedate")
        )
    except SharadarSourceEvidenceError:
        return None, "invalid_ticker_validity_interval"
    if valid_from > valid_through:
        return None, "reversed_ticker_validity_interval"
    is_delisted = row.get("isdelisted")
    if is_delisted not in {"Y", "N"}:
        return None, "invalid_delisting_status"
    text_fields: dict[str, str] = {}
    for field in ("category", "exchange", "currency"):
        raw = row.get(field)
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            return None, f"invalid_{field}"
        text_fields[field] = raw.upper() if field == "currency" else raw
    core = {
        "cik": match.group(1),
        "permaticker": raw_permaticker,
        "ticker": raw_ticker,
        "valid_from": valid_from,
        "valid_through": valid_through,
        "is_delisted": is_delisted,
        "category": text_fields["category"],
        "exchange": text_fields["exchange"],
        "currency": text_fields["currency"],
        "source_record_sha256": source_record_sha256,
    }
    return {
        "identity_id": content_hash(
            {"schema_version": PEAD_SECURITY_IDENTITY_SCHEMA_VERSION, **core}
        ),
        **core,
    }, None


def _identity_conflict_reasons(
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    reasons: dict[str, set[str]] = {}

    def add(row: Mapping[str, Any], reason: str) -> None:
        reasons.setdefault(row["source_record_sha256"], set()).add(reason)

    by_permaticker: dict[int, list[Mapping[str, Any]]] = {}
    by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    by_identity: dict[str, list[Mapping[str, Any]]] = {}
    for identity in identities:
        by_permaticker.setdefault(identity["permaticker"], []).append(identity)
        by_ticker.setdefault(identity["ticker"], []).append(identity)
        by_identity.setdefault(identity["identity_id"], []).append(identity)
    for rows in by_identity.values():
        if len(rows) > 1:
            for row in rows:
                add(row, "duplicate_identity")
    for rows in by_permaticker.values():
        if len({row["cik"] for row in rows}) > 1:
            for row in rows:
                add(row, "contradictory_permaticker_cik")
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if _intervals_overlap(left, right):
                    add(left, "overlapping_permaticker_intervals")
                    add(right, "overlapping_permaticker_intervals")
    for rows in by_ticker.values():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if left["permaticker"] != right["permaticker"] and _intervals_overlap(left, right):
                    add(left, "overlapping_ticker_intervals")
                    add(right, "overlapping_ticker_intervals")
    return reasons


def _derive_identity_payload(
    *,
    root: Path,
    candidate_id: str,
    created_at_utc: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = validate_pead_sharadar_source_snapshot(source_snapshot, warehouse_dir=root)
    if snapshot["payload"]["qualification_allowed"] is not True:
        raise SharadarSourceEvidenceError(
            "a complete qualifying Sharadar source snapshot is required"
        )
    if snapshot["payload"]["candidate_id"] != candidate_id:
        raise SharadarSourceEvidenceError(
            "identity candidate differs from Sharadar source snapshot"
        )
    acquisition = _tickers_acquisition(snapshot, root=root)
    acquisition_payload = acquisition["payload"]
    parquet_path = _resolve_under(
        root, acquisition_payload["parquet"]["relative_path"], label="TICKERS Parquet"
    )
    schema = acquisition_payload["parquet"]["schema"]
    columns = [field["name"] for field in schema]
    relation = f"read_parquet({_sql_string(str(parquet_path))})"
    connection = duckdb.connect(database=":memory:")
    try:
        rows = connection.execute(
            f"SELECT {', '.join(_identifier(column) for column in columns)} "
            f'FROM {relation} ORDER BY "table", permaticker, ticker'
        ).fetchall()
    except duckdb.Error as exc:
        raise SharadarSourceEvidenceError("TICKERS identity rows are unreadable") from exc
    finally:
        connection.close()
    if len(rows) != acquisition_payload["parquet"]["statistics"]["rows"]:
        raise SharadarSourceEvidenceError("TICKERS identity row count changed")

    parsed_by_source: dict[str, dict[str, Any] | None] = {}
    reasons_by_source: dict[str, set[str]] = {}
    source_order: list[str] = []
    for raw_values in rows:
        row = dict(zip(columns, raw_values, strict=True))
        source_hash = sharadar_source_record_sha256("tickers", schema, row)
        source_order.append(source_hash)
        if source_hash in parsed_by_source:
            reasons_by_source.setdefault(source_hash, set()).add("duplicate_source_record_hash")
            continue
        identity, reason = _parse_tickers_identity(row, source_record_sha256=source_hash)
        parsed_by_source[source_hash] = identity
        if reason is not None:
            reasons_by_source.setdefault(source_hash, set()).add(reason)
    if len(source_order) != len(set(source_order)):
        # A source-hash collision or exact duplicate cannot be represented as
        # one-disposition-per-row, so fail rather than silently collapsing it.
        raise SharadarSourceEvidenceError("TICKERS source row identities are duplicated")
    candidates = [identity for identity in parsed_by_source.values() if identity is not None]
    for source_hash, conflict_reasons in _identity_conflict_reasons(candidates).items():
        reasons_by_source.setdefault(source_hash, set()).update(conflict_reasons)

    identities: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    for source_hash in sorted(source_order):
        identity = parsed_by_source[source_hash]
        reasons = sorted(reasons_by_source.get(source_hash, set()))
        if identity is None or reasons:
            dispositions.append(
                {
                    "source_record_sha256": source_hash,
                    "disposition": "identity_gap",
                    "identity_id": None,
                    "reason": reasons[0] if reasons else "unresolved_identity",
                }
            )
        else:
            identities.append(identity)
            dispositions.append(
                {
                    "source_record_sha256": source_hash,
                    "disposition": "identity",
                    "identity_id": identity["identity_id"],
                    "reason": None,
                }
            )
    identities.sort(key=lambda item: item["identity_id"])
    gap_count = sum(item["disposition"] == "identity_gap" for item in dispositions)
    blockers: list[str] = []
    if not dispositions:
        blockers.append("identity_source_empty")
    if not identities:
        blockers.append("no_usable_identities")
    return {
        "schema_version": PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "created_at_utc": _utc(created_at_utc, "identity created_at_utc"),
        "bindings": {
            "sharadar_source_snapshot_sha256": snapshot["artifact_hash"],
            "tickers_acquisition_sha256": acquisition["artifact_hash"],
            "tickers_parquet_sha256": acquisition_payload["parquet"]["sha256"],
        },
        "source_dispositions": dispositions,
        "identities": identities,
        "coverage": {
            "source_row_count": len(dispositions),
            "disposition_count": len(dispositions),
            "identity_count": len(identities),
            "identity_gap_count": gap_count,
            "complete": bool(dispositions),
        },
        "blockers": blockers,
        "qualification_allowed": bool(dispositions and identities and not blockers),
    }


def build_pead_security_identity_snapshot(
    *,
    warehouse_dir: str | os.PathLike[str],
    candidate_id: str,
    created_at_utc: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one dated identity disposition from every exact TICKERS row."""
    candidate = _text(candidate_id, "identity candidate_id")
    payload = _derive_identity_payload(
        root=Path(warehouse_dir).resolve(),
        candidate_id=candidate,
        created_at_utc=created_at_utc,
        source_snapshot=source_snapshot,
    )
    return validate_pead_security_identity_snapshot_structure(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_security_identity_snapshot(
    document: Mapping[str, Any],
    *,
    warehouse_dir: str | os.PathLike[str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Authoritatively rederive identity rows from the bound immutable TICKERS."""
    normalized = validate_pead_security_identity_snapshot_structure(document)
    expected = build_pead_security_identity_snapshot(
        warehouse_dir=warehouse_dir,
        candidate_id=normalized["payload"]["candidate_id"],
        created_at_utc=normalized["payload"]["created_at_utc"],
        source_snapshot=source_snapshot,
    )
    if normalized != expected:
        raise SharadarSourceEvidenceError(
            "security identity snapshot does not replay from Sharadar evidence"
        )
    return normalized


def publish_pead_security_identity_snapshot(
    warehouse_dir: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    root = Path(warehouse_dir).resolve()
    normalized = validate_pead_security_identity_snapshot(
        document, warehouse_dir=root, source_snapshot=source_snapshot
    )
    path = root / SHARADAR_RECEIPT_ROOT / "identities" / f"{normalized['artifact_hash']}.json"
    _publish_json_create_only(path, normalized)
    if _strict_json(path, max_bytes=MAX_IDENTITY_BYTES) != normalized:
        raise SharadarSourceEvidenceError("published identity snapshot changed")
    return normalized, path


def load_pead_security_identity_snapshot(
    path: str | os.PathLike[str],
    *,
    warehouse_dir: str | os.PathLike[str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_pead_security_identity_snapshot(
        _strict_json(Path(path), max_bytes=MAX_IDENTITY_BYTES),
        warehouse_dir=warehouse_dir,
        source_snapshot=source_snapshot,
    )


__all__ = [
    "CANDIDATE_TABLES",
    "PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION",
    "PEAD_SHARADAR_SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "SHARADAR_ROW_EQUIVALENCE_SCHEMA_VERSION",
    "SHARADAR_SOURCE_RECORD_SCHEMA_VERSION",
    "SHARADAR_TABLE_ACQUISITION_SCHEMA_VERSION",
    "SharadarSourceEvidenceError",
    "build_pead_security_identity_snapshot",
    "build_pead_sharadar_source_snapshot",
    "build_sharadar_table_acquisition_document",
    "canonical_json",
    "content_hash",
    "convert_sharadar_zip_to_parquet",
    "file_sha256",
    "inspect_sharadar_parquet",
    "inspect_sharadar_zip",
    "normalize_datatable_metadata",
    "load_pead_security_identity_snapshot",
    "load_pead_sharadar_source_snapshot",
    "load_sharadar_table_acquisition",
    "prove_sharadar_row_equivalence",
    "publish_pead_security_identity_snapshot",
    "publish_pead_sharadar_source_snapshot",
    "publish_sharadar_table_acquisition",
    "sharadar_source_record_sha256",
    "validate_pead_security_identity_snapshot",
    "validate_pead_security_identity_snapshot_structure",
    "validate_pead_sharadar_source_snapshot",
    "validate_sharadar_table_acquisition",
]
