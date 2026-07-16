"""Verifiable acquisition evidence for the Sharadar ACTIONS snapshot.

The ACTIONS table is useful evidence that a return path was checked for
corporate events, but its generic ``value`` column is *not* a terminal cash
price.  This module therefore does only two things:

* bind the exact vendor bulk snapshot, raw ZIP, converted Parquet, schema, and
  row-quality statistics in a content-addressed receipt; and
* verify that receipt again at the research boundary.

Terminal merger, acquisition, bankruptcy, and liquidation economics require a
separate authoritative settlement record.  Nothing in this module converts an
``ACTIONS.value`` into proceeds per share.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping
import zipfile

import duckdb


ACTIONS_ACQUISITION_SCHEMA_VERSION = "sharadar_actions_acquisition.v2"
LEGACY_ACTIONS_ACQUISITION_SCHEMA_VERSION = "sharadar_actions_acquisition.v1"
ACTIONS_EVIDENCE_SCHEMA_VERSION = "sharadar_actions_evidence.v1"
ACTIONS_ROW_MULTISET_SCHEMA_VERSION = "sharadar_actions_row_multiset.v1"
ACTIONS_RECEIPT_FILE = "actions.acquisition.json"
ACTIONS_RECEIPT_ARCHIVE_DIR = "source_snapshots/actions/receipts"
ACTIONS_COLUMNS = (
    ("date", "DATE"),
    ("action", "VARCHAR"),
    ("ticker", "VARCHAR"),
    ("name", "VARCHAR"),
    ("value", "DOUBLE"),
    ("contraticker", "VARCHAR"),
    ("contraname", "VARCHAR"),
)
ACTIONS_PRIMARY_KEY = ("date", "ticker", "name", "action", "contraname", "contraticker")
_HEX = frozenset("0123456789abcdef")


class ActionsEvidenceError(ValueError):
    """The ACTIONS dataset or acquisition receipt is not trustworthy."""


def canonical_json(value: Any) -> str:
    """Return deterministic, finite JSON for evidence identities."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ActionsEvidenceError("evidence JSON contains a non-finite number")
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ActionsEvidenceError("evidence JSON keys must be strings")
                result[key] = normalize(child)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise ActionsEvidenceError(f"unsupported evidence value: {type(item).__name__}")

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


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ActionsEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ActionsEvidenceError(f"{label} must be a non-empty timestamp")
    text = value.replace(" UTC", "+00:00")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ActionsEvidenceError(f"{label} is not an ISO/UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ActionsEvidenceError(f"{label} must identify UTC")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ActionsEvidenceError(f"ACTIONS receipt is missing: {path}")
    raw = path.read_text(encoding="utf-8")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ActionsEvidenceError(f"ACTIONS receipt has duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ActionsEvidenceError(f"ACTIONS receipt has invalid number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActionsEvidenceError("ACTIONS receipt is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ActionsEvidenceError("ACTIONS receipt root must be an object")
    return value


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ActionsEvidenceError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _logical_type(raw: str) -> str:
    value = str(raw).upper()
    if value.startswith("VARCHAR"):
        return "VARCHAR"
    if value in {"DOUBLE", "FLOAT", "REAL"} or value.startswith("DECIMAL"):
        return "DOUBLE"
    if value == "DATE":
        return "DATE"
    return value


def inspect_actions_parquet(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the exact ACTIONS schema and return deterministic statistics."""
    source = Path(path)
    if not source.is_file():
        raise ActionsEvidenceError(f"ACTIONS Parquet is missing: {source}")
    con = duckdb.connect(database=":memory:")
    escaped = str(source).replace("'", "''")
    relation = f"read_parquet('{escaped}')"
    try:
        described = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        columns = tuple((row[0], _logical_type(row[1])) for row in described)
        if columns != ACTIONS_COLUMNS:
            raise ActionsEvidenceError(f"ACTIONS Parquet schema mismatch: {columns!r}")
        row = con.execute(
            f"SELECT count(*), min(date), max(date), "
            "sum(CASE WHEN date IS NULL OR ticker IS NULL OR "
            "trim(ticker) = '' OR action IS NULL OR trim(action) = '' "
            f"THEN 1 ELSE 0 END), "
            "sum(CASE WHEN value IS NOT NULL AND NOT isfinite(value) "
            f"THEN 1 ELSE 0 END), count(DISTINCT ticker) FROM {relation}"
        ).fetchone()
        duplicates = con.execute(
            "SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) AS n FROM "
            f"{relation} GROUP BY {', '.join(ACTIONS_PRIMARY_KEY)} HAVING n > 1)"
        ).fetchone()
    except duckdb.Error as exc:
        raise ActionsEvidenceError("ACTIONS Parquet is unreadable") from exc
    finally:
        con.close()
    assert row is not None and duplicates is not None
    result = {
        "rows": int(row[0]),
        "min_date": row[1].isoformat() if row[1] is not None else None,
        "max_date": row[2].isoformat() if row[2] is not None else None,
        "missing_required_values": int(row[3] or 0),
        "nonfinite_values": int(row[4] or 0),
        "distinct_tickers": int(row[5]),
        "duplicate_primary_keys": int(duplicates[0] or 0),
        "columns": [{"name": name, "logical_type": logical_type} for name, logical_type in columns],
    }
    if result["rows"] <= 0 or result["min_date"] is None or result["max_date"] is None:
        raise ActionsEvidenceError("ACTIONS Parquet contains no dated rows")
    if result["missing_required_values"]:
        raise ActionsEvidenceError("ACTIONS Parquet has missing required values")
    if result["nonfinite_values"]:
        raise ActionsEvidenceError("ACTIONS Parquet has non-finite values")
    if result["duplicate_primary_keys"]:
        raise ActionsEvidenceError("ACTIONS Parquet has duplicate primary keys")
    return result


def inspect_actions_zip(
    path: str | os.PathLike[str], *, expected_member: str | None = None
) -> dict[str, Any]:
    """Hash the sole CSV member without extracting untrusted paths."""
    source = Path(path)
    if not source.is_file():
        raise ActionsEvidenceError(f"ACTIONS raw ZIP is missing: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            csv_members = [item for item in members if item.filename.lower().endswith(".csv")]
            if len(members) != 1 or len(csv_members) != 1:
                raise ActionsEvidenceError("ACTIONS ZIP must contain exactly one CSV")
            info = csv_members[0]
            if expected_member is not None and info.filename != expected_member:
                raise ActionsEvidenceError("ACTIONS ZIP CSV member changed")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, "r") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ActionsEvidenceError("ACTIONS raw ZIP is unreadable") from exc
    return {
        "member": info.filename,
        "sha256": digest.hexdigest(),
        "bytes": size,
        "compressed_bytes": int(info.compress_size),
    }


def _encode_normalized_action_row(row: tuple[Any, ...]) -> bytes:
    """Encode one typed ACTIONS row without lossy string coercion."""
    if len(row) != len(ACTIONS_COLUMNS):
        raise ActionsEvidenceError("ACTIONS row width changed")
    encoded = bytearray()
    for value, (_, logical_type) in zip(row, ACTIONS_COLUMNS, strict=True):
        if value is None:
            encoded.extend(b"\x00")
            continue
        encoded.extend(b"\x01")
        if logical_type == "DATE":
            # ``datetime`` is a ``date`` subclass, but accepting it would
            # silently discard intraday source information and could make a
            # timestamp-valued CSV falsely equal a DATE-valued Parquet row.
            if type(value) is not date:
                raise ActionsEvidenceError("ACTIONS date has an unexpected type")
            payload = value.isoformat().encode("ascii")
        elif logical_type == "VARCHAR":
            if not isinstance(value, str):
                raise ActionsEvidenceError("ACTIONS text has an unexpected type")
            payload = value.encode("utf-8")
        elif logical_type == "DOUBLE":
            if not isinstance(value, (int, float, Decimal)) or isinstance(value, bool):
                raise ActionsEvidenceError("ACTIONS value has an unexpected type")
            number = float(value)
            if not math.isfinite(number):
                raise ActionsEvidenceError("ACTIONS value is non-finite")
            # Treat the two IEEE encodings of zero as the same economic value.
            payload = struct.pack(">d", 0.0 if number == 0.0 else number)
        else:  # pragma: no cover - guarded by the fixed schema above
            raise ActionsEvidenceError(f"unsupported ACTIONS logical type {logical_type!r}")
        encoded.extend(struct.pack(">Q", len(payload)))
        encoded.extend(payload)
    return bytes(encoded)


def _relation_row_multiset(connection: duckdb.DuckDBPyConnection, relation: str) -> dict[str, Any]:
    """Commit to normalized rows as an order-independent multiset.

    Each duplicate contributes another identical row digest.  Sorting the
    fixed-width digests makes the final identity independent of CSV/Parquet
    physical order without collapsing duplicate rows into a set.
    """
    try:
        described = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        columns = tuple((row[0], _logical_type(row[1])) for row in described)
        if columns != ACTIONS_COLUMNS:
            raise ActionsEvidenceError(f"ACTIONS normalized-row schema mismatch: {columns!r}")
        cursor = connection.execute(
            f"SELECT {', '.join(name for name, _ in ACTIONS_COLUMNS)} FROM {relation}"
        )
        row_digests: list[bytes] = []
        while batch := cursor.fetchmany(10_000):
            for row in batch:
                row_digests.append(hashlib.sha256(_encode_normalized_action_row(row)).digest())
    except duckdb.Error as exc:
        raise ActionsEvidenceError("ACTIONS normalized rows are unreadable") from exc

    row_digests.sort()
    digest = hashlib.sha256()
    digest.update(ACTIONS_ROW_MULTISET_SCHEMA_VERSION.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack(">Q", len(row_digests)))
    for row_digest in row_digests:
        digest.update(row_digest)
    return {
        "schema_version": ACTIONS_ROW_MULTISET_SCHEMA_VERSION,
        "sha256": digest.hexdigest(),
        "rows": len(row_digests),
    }


def _parquet_row_multiset(path: Path) -> dict[str, Any]:
    connection = duckdb.connect(database=":memory:")
    escaped = str(path).replace("'", "''")
    try:
        return _relation_row_multiset(connection, f"read_parquet('{escaped}')")
    finally:
        connection.close()


def _zip_csv_row_multiset(path: Path, *, expected_member: str | None = None) -> dict[str, Any]:
    """Parse the archived CSV with the same full-scan inference as ingest."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            csv_members = [item for item in members if item.filename.lower().endswith(".csv")]
            if len(members) != 1 or len(csv_members) != 1:
                raise ActionsEvidenceError("ACTIONS ZIP must contain exactly one CSV")
            info = csv_members[0]
            if expected_member is not None and info.filename != expected_member:
                raise ActionsEvidenceError("ACTIONS ZIP CSV member changed")
            with tempfile.TemporaryDirectory(prefix="actions-evidence-") as temporary:
                csv_path = Path(temporary) / "actions.csv"
                with archive.open(info, "r") as incoming, csv_path.open("wb") as output:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                        output.write(chunk)
                connection = duckdb.connect(database=":memory:")
                escaped = str(csv_path).replace("'", "''")
                try:
                    return _relation_row_multiset(
                        connection,
                        f"read_csv_auto('{escaped}', sample_size=-1)",
                    )
                finally:
                    connection.close()
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ActionsEvidenceError("ACTIONS raw ZIP is unreadable") from exc


def _verified_row_equivalence(
    *, parquet_path: Path, raw_zip_path: Path, csv_member: str
) -> dict[str, Any]:
    csv_rows = _zip_csv_row_multiset(raw_zip_path, expected_member=csv_member)
    parquet_rows = _parquet_row_multiset(parquet_path)
    if csv_rows != parquet_rows:
        raise ActionsEvidenceError("ACTIONS CSV and Parquet normalized row multisets differ")
    return parquet_rows


def expected_datatable_metadata() -> dict[str, Any]:
    """The provider schema contract observed from ACTIONS metadata."""
    return {
        "vendor_code": "SHARADAR",
        "datatable_code": "ACTIONS",
        "name": "Corporate Actions",
        "description": None,
        "columns": [
            {"name": "date", "type": "Date"},
            {"name": "action", "type": "text"},
            {"name": "ticker", "type": "text"},
            {"name": "name", "type": "text"},
            {"name": "value", "type": "double"},
            {"name": "contraticker", "type": "text"},
            {"name": "contraname", "type": "text"},
        ],
        "filters": ["action", "contraticker", "date", "ticker"],
        "primary_key": list(ACTIONS_PRIMARY_KEY),
        "premium": True,
    }


def build_actions_acquisition_document(
    *,
    parquet_path: str | os.PathLike[str],
    raw_zip_path: str | os.PathLike[str],
    raw_zip_relative_path: str,
    acquired_at_utc: str,
    last_refreshed_time: str,
    data_snapshot_time: str,
    datatable_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a receipt after conversion; publishing is handled separately."""
    acquired = _utc_timestamp(acquired_at_utc, "acquired_at_utc")
    refreshed = _utc_timestamp(last_refreshed_time, "last_refreshed_time")
    snapshot_time = _utc_timestamp(data_snapshot_time, "data_snapshot_time")
    expected = expected_datatable_metadata()
    supplied = dict(datatable_metadata)
    status = supplied.pop("status", None)
    if supplied != expected:
        raise ActionsEvidenceError("ACTIONS datatable metadata contract changed")
    if not isinstance(status, Mapping):
        raise ActionsEvidenceError("ACTIONS metadata status is missing")
    status_copy = dict(status)
    if set(status_copy) != {"expected_at", "refreshed_at", "status", "update_frequency"}:
        raise ActionsEvidenceError("ACTIONS metadata status fields changed")
    _utc_timestamp(status_copy["refreshed_at"], "metadata refreshed_at")
    if status_copy["status"] != "ON TIME" or status_copy["update_frequency"] != "CONTINUOUS":
        raise ActionsEvidenceError("ACTIONS metadata is not continuously on time")

    parquet = Path(parquet_path)
    raw_zip = Path(raw_zip_path)
    stats = inspect_actions_parquet(parquet)
    csv = inspect_actions_zip(raw_zip)
    zip_hash = file_sha256(raw_zip)
    expected_relative = f"source_snapshots/actions/{zip_hash}.zip"
    if raw_zip_relative_path != expected_relative:
        raise ActionsEvidenceError("raw ACTIONS ZIP path is not content-addressed")
    row_equivalence = _verified_row_equivalence(
        parquet_path=parquet,
        raw_zip_path=raw_zip,
        csv_member=csv["member"],
    )
    if row_equivalence["rows"] != stats["rows"]:
        raise ActionsEvidenceError("ACTIONS normalized row count differs from Parquet statistics")
    payload = {
        "schema_version": ACTIONS_ACQUISITION_SCHEMA_VERSION,
        "source": {
            "vendor_code": "SHARADAR",
            "datatable_code": "ACTIONS",
            "canonical_request": {
                "method": "GET",
                "dataset": "SHARADAR/ACTIONS",
                "parameters": {"qopts.export": "true"},
            },
            "last_refreshed_time": refreshed,
            "data_snapshot_time": snapshot_time,
            "bulk_status": "fresh",
            "datatable_metadata": {**expected, "status": status_copy},
        },
        "acquired_at_utc": acquired,
        "raw_zip": {
            "relative_path": raw_zip_relative_path,
            "sha256": zip_hash,
            "bytes": raw_zip.stat().st_size,
        },
        "csv": csv,
        "parquet": {
            "relative_path": "actions.parquet",
            "sha256": file_sha256(parquet),
            "bytes": parquet.stat().st_size,
            "statistics": stats,
        },
        "row_equivalence": row_equivalence,
        "value_semantics": ("type-dependent vendor metadata; never terminal proceeds per share"),
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def atomic_write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> None:
    """Atomically replace the active receipt with canonical JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(document) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _create_only_json(
    path: str | os.PathLike[str], document: Mapping[str, Any]
) -> Path:
    """Publish canonical JSON once without replacing existing bytes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(document) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise ActionsEvidenceError(
                    "content-addressed ACTIONS receipt collision"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def archive_raw_zip(source: str | os.PathLike[str], warehouse_dir: str | os.PathLike[str]) -> Path:
    """Publish the vendor ZIP under its hash without overwriting prior bytes."""
    source_path = Path(source)
    digest = file_sha256(source_path)
    destination = Path(warehouse_dir) / "source_snapshots" / "actions" / f"{digest}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ActionsEvidenceError("content-addressed ACTIONS ZIP was mutated")
        return destination
    temporary: str | None = None
    try:
        with (
            source_path.open("rb") as incoming,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                delete=False,
            ) as outgoing,
        ):
            temporary = outgoing.name
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if file_sha256(destination) != digest:
                raise ActionsEvidenceError("content-addressed ACTIONS ZIP collision")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def _validate_actions_evidence_document(
    root: Path,
    document: Mapping[str, Any],
    *,
    required_start: str,
    required_end: str,
    allow_legacy: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one already-loaded receipt and return evidence plus row proof."""
    if type(allow_legacy) is not bool:
        raise ActionsEvidenceError("allow_legacy must be boolean")
    root = root.resolve()
    receipt = _exact_fields(document, {"artifact_hash", "payload"}, "receipt")
    payload = receipt["payload"]
    if not isinstance(payload, Mapping):
        raise ActionsEvidenceError("ACTIONS receipt payload must be an object")
    artifact_hash = _sha256(receipt["artifact_hash"], "receipt artifact_hash")
    if content_hash(payload) != artifact_hash:
        raise ActionsEvidenceError("ACTIONS receipt artifact hash mismatch")
    acquisition_schema = payload.get("schema_version")
    if acquisition_schema not in {
        ACTIONS_ACQUISITION_SCHEMA_VERSION,
        LEGACY_ACTIONS_ACQUISITION_SCHEMA_VERSION,
    }:
        raise ActionsEvidenceError("unsupported ACTIONS acquisition schema")
    if (
        acquisition_schema == LEGACY_ACTIONS_ACQUISITION_SCHEMA_VERSION
        and not allow_legacy
    ):
        raise ActionsEvidenceError(
            "legacy ACTIONS acquisition receipt is not candidate-grade; "
            "run the offline v2 receipt upgrade"
        )
    if payload.get("value_semantics") != (
        "type-dependent vendor metadata; never terminal proceeds per share"
    ):
        raise ActionsEvidenceError("ACTIONS value semantics are unsafe or missing")

    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ActionsEvidenceError("ACTIONS receipt source is missing")
    if source.get("vendor_code") != "SHARADAR" or source.get("datatable_code") != "ACTIONS":
        raise ActionsEvidenceError("ACTIONS receipt identifies another source")
    if source.get("bulk_status") != "fresh":
        raise ActionsEvidenceError("ACTIONS bulk export was not fresh")
    snapshot_time = _utc_timestamp(source.get("data_snapshot_time"), "data_snapshot_time")
    _utc_timestamp(source.get("last_refreshed_time"), "last_refreshed_time")
    metadata = source.get("datatable_metadata")
    if not isinstance(metadata, Mapping):
        raise ActionsEvidenceError("ACTIONS datatable metadata is missing")
    status = metadata.get("status")
    metadata_without_status = dict(metadata)
    metadata_without_status.pop("status", None)
    if metadata_without_status != expected_datatable_metadata():
        raise ActionsEvidenceError("ACTIONS datatable metadata no longer matches")
    if (
        not isinstance(status, Mapping)
        or status.get("status") != "ON TIME"
        or status.get("update_frequency") != "CONTINUOUS"
    ):
        raise ActionsEvidenceError("ACTIONS datatable status is not qualifying")

    raw = payload.get("raw_zip")
    parquet = payload.get("parquet")
    csv = payload.get("csv")
    if not all(isinstance(item, Mapping) for item in (raw, parquet, csv)):
        raise ActionsEvidenceError("ACTIONS file receipts are missing")
    zip_hash = _sha256(raw.get("sha256"), "raw ZIP sha256")
    expected_zip_relative = f"source_snapshots/actions/{zip_hash}.zip"
    if raw.get("relative_path") != expected_zip_relative:
        raise ActionsEvidenceError("raw ACTIONS ZIP path is not content-addressed")
    zip_path = (root / expected_zip_relative).resolve()
    if root not in zip_path.parents:
        raise ActionsEvidenceError("raw ACTIONS ZIP path escapes warehouse")
    if file_sha256(zip_path) != zip_hash or zip_path.stat().st_size != raw.get("bytes"):
        raise ActionsEvidenceError("raw ACTIONS ZIP identity mismatch")
    actual_csv = inspect_actions_zip(zip_path, expected_member=csv.get("member"))
    if actual_csv != dict(csv):
        raise ActionsEvidenceError("ACTIONS CSV receipt mismatch")

    if parquet.get("relative_path") != "actions.parquet":
        raise ActionsEvidenceError("ACTIONS Parquet path changed")
    parquet_path = root / "actions.parquet"
    parquet_hash = _sha256(parquet.get("sha256"), "ACTIONS Parquet sha256")
    if file_sha256(parquet_path) != parquet_hash or parquet_path.stat().st_size != parquet.get(
        "bytes"
    ):
        raise ActionsEvidenceError("ACTIONS Parquet identity mismatch")
    actual_stats = inspect_actions_parquet(parquet_path)
    if actual_stats != parquet.get("statistics"):
        raise ActionsEvidenceError("ACTIONS Parquet statistics mismatch")
    row_equivalence = _verified_row_equivalence(
        parquet_path=parquet_path,
        raw_zip_path=zip_path,
        csv_member=actual_csv["member"],
    )
    if row_equivalence["rows"] != actual_stats["rows"]:
        raise ActionsEvidenceError("ACTIONS normalized row count differs from Parquet statistics")
    if acquisition_schema == ACTIONS_ACQUISITION_SCHEMA_VERSION:
        receipt_equivalence = _exact_fields(
            payload.get("row_equivalence"),
            {"schema_version", "sha256", "rows"},
            "row_equivalence",
        )
        _sha256(receipt_equivalence.get("sha256"), "row_equivalence sha256")
        if dict(receipt_equivalence) != row_equivalence:
            raise ActionsEvidenceError("ACTIONS normalized row-equivalence receipt mismatch")

    try:
        start_date = date.fromisoformat(required_start)
        end_date = date.fromisoformat(required_end)
        min_date = date.fromisoformat(actual_stats["min_date"])
        max_date = date.fromisoformat(actual_stats["max_date"])
    except (TypeError, ValueError) as exc:
        raise ActionsEvidenceError("invalid ACTIONS coverage date") from exc
    if start_date > end_date:
        raise ActionsEvidenceError("required ACTIONS window is reversed")
    blockers: list[str] = []
    if min_date > start_date:
        blockers.append("actions_range_starts_after_required_window")
    if max_date < end_date:
        blockers.append("actions_range_ends_before_required_window")
    evidence_payload = {
        "schema_version": ACTIONS_EVIDENCE_SCHEMA_VERSION,
        "acquisition_artifact_hash": artifact_hash,
        "source_snapshot_time": snapshot_time,
        "parquet_sha256": parquet_hash,
        "raw_zip_sha256": zip_hash,
        "row_count": actual_stats["rows"],
        "min_date": actual_stats["min_date"],
        "max_date": actual_stats["max_date"],
        "required_window": {"start": required_start, "end": required_end},
        "complete": not blockers,
        "blockers": blockers,
        "value_is_terminal_payout_per_share": False,
    }
    evidence = {
        "artifact_hash": content_hash(evidence_payload),
        "payload": evidence_payload,
    }
    return evidence, row_equivalence


def validate_actions_evidence(
    warehouse_dir: str | os.PathLike[str],
    *,
    required_start: str,
    required_end: str,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Revalidate the active receipt and requested research-window coverage.

    Candidate-grade validation is the default and requires a v2 acquisition
    receipt that commits the independently recomputed CSV/Parquet row proof.
    Legacy v1 receipts can be inspected only through the explicit
    ``allow_legacy=True`` compatibility path; compatibility still recomputes
    every row and never upgrades the receipt implicitly.
    """
    root = Path(warehouse_dir).resolve()
    document = _strict_json(root / ACTIONS_RECEIPT_FILE)
    evidence, _ = _validate_actions_evidence_document(
        root,
        document,
        required_start=required_start,
        required_end=required_end,
        allow_legacy=allow_legacy,
    )
    return evidence


def _receipt_coverage_window(document: Mapping[str, Any]) -> tuple[str, str]:
    """Read the receipt's own observed range for whole-snapshot validation."""
    try:
        payload = document["payload"]
        parquet = payload["parquet"]
        statistics = parquet["statistics"]
        start = statistics["min_date"]
        end = statistics["max_date"]
    except (KeyError, TypeError) as exc:
        raise ActionsEvidenceError(
            "ACTIONS receipt omits its observed coverage window"
        ) from exc
    if not isinstance(start, str) or not isinstance(end, str):
        raise ActionsEvidenceError("ACTIONS receipt coverage window is malformed")
    return start, end


def upgrade_actions_acquisition_receipt(
    warehouse_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Upgrade the active v1 receipt to a locally proven v2 receipt.

    This operator path performs no network access.  It reopens the immutable
    raw ZIP and active Parquet, validates the legacy receipt in explicit
    compatibility mode, independently proves normalized row-multiset
    equivalence, and preserves all acquisition/source metadata byte-for-byte
    at the JSON-value level.  The v2 receipt is first published create-only
    under its artifact hash and fully validated there.  Only then is the active
    receipt atomically promoted.  Calling the function on a valid v2 receipt is
    idempotent and ensures its immutable archived copy exists.
    """
    root = Path(warehouse_dir).resolve()
    active_path = root / ACTIONS_RECEIPT_FILE
    original = _strict_json(active_path)
    payload = original.get("payload")
    if not isinstance(payload, Mapping):
        raise ActionsEvidenceError("ACTIONS receipt payload must be an object")
    schema = payload.get("schema_version")
    start, end = _receipt_coverage_window(original)

    if schema == ACTIONS_ACQUISITION_SCHEMA_VERSION:
        _validate_actions_evidence_document(
            root,
            original,
            required_start=start,
            required_end=end,
            allow_legacy=False,
        )
        artifact_hash = _sha256(
            original.get("artifact_hash"), "receipt artifact_hash"
        )
        archive_path = (
            root / ACTIONS_RECEIPT_ARCHIVE_DIR / f"{artifact_hash}.json"
        )
        _create_only_json(archive_path, original)
        if _strict_json(archive_path) != original:
            raise ActionsEvidenceError("archived ACTIONS v2 receipt changed")
        return original
    if schema != LEGACY_ACTIONS_ACQUISITION_SCHEMA_VERSION:
        raise ActionsEvidenceError("only ACTIONS v1 receipts can be upgraded")
    if "row_equivalence" in payload:
        raise ActionsEvidenceError(
            "legacy ACTIONS receipt unexpectedly contains a row proof"
        )

    _, row_equivalence = _validate_actions_evidence_document(
        root,
        original,
        required_start=start,
        required_end=end,
        allow_legacy=True,
    )
    upgraded_payload = json.loads(canonical_json(payload))
    upgraded_payload["schema_version"] = ACTIONS_ACQUISITION_SCHEMA_VERSION
    upgraded_payload["row_equivalence"] = row_equivalence
    upgraded = {
        "artifact_hash": content_hash(upgraded_payload),
        "payload": upgraded_payload,
    }

    archive_path = (
        root
        / ACTIONS_RECEIPT_ARCHIVE_DIR
        / f"{upgraded['artifact_hash']}.json"
    )
    _create_only_json(archive_path, upgraded)
    staged = _strict_json(archive_path)
    staged_evidence, staged_equivalence = _validate_actions_evidence_document(
        root,
        staged,
        required_start=start,
        required_end=end,
        allow_legacy=False,
    )
    if staged != upgraded or staged_equivalence != row_equivalence:
        raise ActionsEvidenceError("staged ACTIONS v2 receipt did not revalidate")
    if staged_evidence["payload"]["acquisition_artifact_hash"] != upgraded[
        "artifact_hash"
    ]:
        raise ActionsEvidenceError("staged ACTIONS v2 identity changed")

    # Do not overwrite a receipt that another operator or ingest replaced while
    # this potentially long full-row proof was running.
    if _strict_json(active_path) != original:
        raise ActionsEvidenceError("active ACTIONS receipt changed during upgrade")
    atomic_write_json(active_path, staged)
    if _strict_json(active_path) != staged:
        raise ActionsEvidenceError("active ACTIONS v2 promotion did not persist")
    return staged


__all__ = [
    "ACTIONS_ACQUISITION_SCHEMA_VERSION",
    "ACTIONS_COLUMNS",
    "ACTIONS_EVIDENCE_SCHEMA_VERSION",
    "ACTIONS_RECEIPT_ARCHIVE_DIR",
    "ACTIONS_RECEIPT_FILE",
    "ActionsEvidenceError",
    "archive_raw_zip",
    "atomic_write_json",
    "build_actions_acquisition_document",
    "canonical_json",
    "content_hash",
    "expected_datatable_metadata",
    "file_sha256",
    "inspect_actions_parquet",
    "inspect_actions_zip",
    "upgrade_actions_acquisition_receipt",
    "validate_actions_evidence",
]


def main(argv: list[str] | None = None) -> int:
    """Offline operator CLI for the create-only v1-to-v2 upgrade."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m data.corporate_action_evidence",
        description="Offline Sharadar ACTIONS receipt operations (no network).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    upgrade = subcommands.add_parser(
        "upgrade",
        help="prove raw-ZIP/Parquet equivalence and promote a v1 receipt to v2",
    )
    upgrade.add_argument("--warehouse-dir", default="pit_warehouse")
    args = parser.parse_args(argv)
    if args.command != "upgrade":  # pragma: no cover - argparse guards this
        raise ActionsEvidenceError(f"unsupported command {args.command!r}")
    document = upgrade_actions_acquisition_receipt(args.warehouse_dir)
    artifact_hash = document["artifact_hash"]
    output = {
        "artifact_hash": artifact_hash,
        "archive_relative_path": (
            f"{ACTIONS_RECEIPT_ARCHIVE_DIR}/{artifact_hash}.json"
        ),
        "row_equivalence": document["payload"]["row_equivalence"],
        "schema_version": document["payload"]["schema_version"],
    }
    print(canonical_json(output))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as an operator command
    raise SystemExit(main())
