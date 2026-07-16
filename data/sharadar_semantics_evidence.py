"""Immutable official Sharadar field-semantics evidence.

The market-accounting boundary must not infer what ``SEP.close`` and
``SEP.closeunadj`` mean from their names.  This module seals the exact response
from the licensed ``SHARADAR/INDICATORS`` table, validates the provider's closed
SEP definitions, and publishes both raw bytes and a canonical receipt with
create-only semantics.  Credentials are transport-only and never enter a
request identity, artifact, path, or error.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from data.pead_event_universe import canonical_json, content_hash


SHARADAR_SEMANTICS_RECEIPT_SCHEMA_VERSION = "sharadar_semantics_source_receipt.v1"
INDICATORS_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/INDICATORS.json"
CANONICAL_REQUEST = {
    "url": INDICATORS_URL,
    "params": {"qopts.per_page": "10000", "table": "SEP"},
}
MAX_RAW_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_COLUMNS = (
    "table",
    "indicator",
    "isfilter",
    "isprimarykey",
    "title",
    "description",
    "unittype",
)
_COLUMN_DECLARATIONS = tuple(
    {"name": name, "type": "text"} for name in _COLUMNS
)
_REQUIRED_SEMANTICS = {
    "ticker": {
        "title": "Ticker Symbol",
        "description": (
            "The ticker is a unique identifier for a security in the database. "
            "Where a company is delisted and the ticker subsequently recycled for use "
            "by a different company; we utilise that ticker for the currently active "
            "company and append a number to the ticker of the delisted company. The "
            "ACTIONS table provides a record of historical ticker changes."
        ),
        "unittype": "text",
        "isfilter": "Y",
        "isprimarykey": "Y",
    },
    "date": {
        "title": "Price Date",
        "description": "The trade date of the price observations.",
        "unittype": "date (YYYY-MM-DD)",
        "isfilter": "Y",
        "isprimarykey": "Y",
    },
    "close": {
        "title": "Close Price - Split Adjusted",
        "description": (
            "The official exchange close price; adjusted for stock splits and stock "
            "dividends. Not adjusted for cash dividends or spinoffs."
        ),
        "unittype": "USD/share",
        "isfilter": "N",
        "isprimarykey": "N",
    },
    "closeunadj": {
        "title": "Close Price - Unadjusted",
        "description": (
            "The official exchange close price; not adjusted for stock splits; stock "
            "dividends; cash dividends or spinoffs."
        ),
        "unittype": "USD/share",
        "isfilter": "N",
        "isprimarykey": "N",
    },
}


class SharadarSemanticsEvidenceError(ValueError):
    """Official semantics evidence is missing, malformed, or mutable."""


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SharadarSemanticsEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SharadarSemanticsEvidenceError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise SharadarSemanticsEvidenceError(f"{label} must be canonical UTC with Z") from exc
    rendered = parsed.isoformat(timespec="microseconds" if parsed.microsecond else "seconds")
    rendered = rendered.replace("+00:00", "Z")
    if rendered != value:
        raise SharadarSemanticsEvidenceError(f"{label} is not canonical UTC")
    return rendered


def _strict_json_bytes(raw: bytes, *, label: str, max_bytes: int) -> dict[str, Any]:
    if not raw or len(raw) > max_bytes:
        raise SharadarSemanticsEvidenceError(f"{label} size is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SharadarSemanticsEvidenceError(f"{label} is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SharadarSemanticsEvidenceError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise SharadarSemanticsEvidenceError(f"{label} contains invalid number {token}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except SharadarSemanticsEvidenceError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SharadarSemanticsEvidenceError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SharadarSemanticsEvidenceError(f"{label} root must be an object")
    return value


def _provider_rows(raw: bytes) -> list[dict[str, str]]:
    document = _strict_json_bytes(
        raw, label="SHARADAR/INDICATORS response", max_bytes=MAX_RAW_RESPONSE_BYTES
    )
    if set(document) != {"datatable", "meta"}:
        raise SharadarSemanticsEvidenceError(
            "INDICATORS response requires exactly datatable and meta"
        )
    datatable = document["datatable"]
    meta = document["meta"]
    if not isinstance(datatable, Mapping) or set(datatable) != {"columns", "data"}:
        raise SharadarSemanticsEvidenceError(
            "INDICATORS datatable requires exactly columns and data"
        )
    if not isinstance(meta, Mapping) or set(meta) != {"next_cursor_id"}:
        raise SharadarSemanticsEvidenceError(
            "INDICATORS meta requires exactly next_cursor_id"
        )
    if meta["next_cursor_id"] is not None:
        raise SharadarSemanticsEvidenceError("INDICATORS response is not the final page")
    columns = datatable["columns"]
    if not isinstance(columns, list):
        raise SharadarSemanticsEvidenceError("INDICATORS columns must be an array")
    declarations: list[dict[str, str]] = []
    for column in columns:
        if (
            not isinstance(column, Mapping)
            or set(column) != {"name", "type"}
            or not isinstance(column["name"], str)
            or not isinstance(column["type"], str)
        ):
            raise SharadarSemanticsEvidenceError("INDICATORS column declaration is invalid")
        declarations.append({"name": column["name"], "type": column["type"]})
    if tuple(declarations) != _COLUMN_DECLARATIONS:
        raise SharadarSemanticsEvidenceError(
            "INDICATORS column order, names, or types changed"
        )
    data = datatable["data"]
    if not isinstance(data, list):
        raise SharadarSemanticsEvidenceError("INDICATORS data must be an array")
    rows: list[dict[str, str]] = []
    for raw_row in data:
        if not isinstance(raw_row, list) or len(raw_row) != len(_COLUMNS):
            raise SharadarSemanticsEvidenceError("INDICATORS row width changed")
        if any(not isinstance(value, str) for value in raw_row):
            raise SharadarSemanticsEvidenceError("INDICATORS rows must contain text")
        row = dict(zip(_COLUMNS, raw_row, strict=True))
        if row["table"] != "SEP":
            raise SharadarSemanticsEvidenceError("INDICATORS response escaped the SEP filter")
        rows.append(row)
    indicators = [row["indicator"] for row in rows]
    if not rows or len(indicators) != len(set(indicators)):
        raise SharadarSemanticsEvidenceError("INDICATORS rows are empty or duplicated")
    return rows


def _selected_semantics(raw: bytes) -> list[dict[str, str]]:
    by_indicator = {row["indicator"]: row for row in _provider_rows(raw)}
    selected: list[dict[str, str]] = []
    for indicator in sorted(_REQUIRED_SEMANTICS):
        row = by_indicator.get(indicator)
        if row is None:
            raise SharadarSemanticsEvidenceError(
                f"INDICATORS response omits required SEP.{indicator}"
            )
        expected = {
            "table": "SEP",
            "indicator": indicator,
            **_REQUIRED_SEMANTICS[indicator],
        }
        if row != expected:
            raise SharadarSemanticsEvidenceError(
                f"official SEP.{indicator} semantics changed"
            )
        selected.append(expected)
    return selected


def _relative_raw_path(raw_sha256: str) -> str:
    return f"source_snapshots/sharadar/semantics/raw/{raw_sha256}.json"


def _relative_receipt_path(artifact_hash: str) -> str:
    return f"source_snapshots/sharadar/semantics/receipts/{artifact_hash}.json"


def build_sharadar_semantics_receipt(
    raw_response: bytes,
    *,
    candidate_id: str,
    captured_at_utc: str,
) -> dict[str, Any]:
    """Build a receipt from exact provider response bytes without credentials."""
    if not isinstance(raw_response, bytes):
        raise SharadarSemanticsEvidenceError("raw response must be bytes")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or candidate_id.strip() != candidate_id
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in candidate_id)
    ):
        raise SharadarSemanticsEvidenceError("candidate_id must be canonical text")
    raw_sha = hashlib.sha256(raw_response).hexdigest()
    selected = _selected_semantics(raw_response)
    payload = {
        "schema_version": SHARADAR_SEMANTICS_RECEIPT_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "captured_at_utc": _utc(captured_at_utc, "captured_at_utc"),
        "source": {
            "provider": "Nasdaq Data Link / Sharadar",
            "dataset": "SHARADAR/INDICATORS",
            # Copy the exported informational constant.  A caller mutating the
            # public dictionary must not be able to alter this trust boundary.
            "canonical_request": {
                "url": INDICATORS_URL,
                "params": {"qopts.per_page": "10000", "table": "SEP"},
            },
        },
        "raw_artifact": {
            "relative_path": _relative_raw_path(raw_sha),
            "sha256": raw_sha,
            "byte_count": len(raw_response),
        },
        "selected_semantics": selected,
        "coverage": {
            "required_indicators": sorted(_REQUIRED_SEMANTICS),
            "present_indicators": [row["indicator"] for row in selected],
            "complete": True,
        },
        "qualification_allowed": True,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _resolve_under(root: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise SharadarSemanticsEvidenceError(f"{label} path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SharadarSemanticsEvidenceError(f"{label} path escapes the warehouse")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise SharadarSemanticsEvidenceError(f"{label} is missing or unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SharadarSemanticsEvidenceError(f"{label} is missing or unsafe") from exc
    if not resolved.is_file():
        raise SharadarSemanticsEvidenceError(f"{label} is not a regular file")
    return resolved


def validate_sharadar_semantics_receipt(
    document: Mapping[str, Any], *, warehouse_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Reopen the exact provider bytes and rebuild the claimed receipt."""
    if not isinstance(document, Mapping) or set(document) != {"artifact_hash", "payload"}:
        raise SharadarSemanticsEvidenceError("semantics receipt wrapper fields differ")
    claimed = _sha(document["artifact_hash"], "receipt artifact_hash")
    payload = document["payload"]
    if not isinstance(payload, Mapping):
        raise SharadarSemanticsEvidenceError("receipt payload must be an object")
    required_fields = {
        "schema_version",
        "candidate_id",
        "captured_at_utc",
        "source",
        "raw_artifact",
        "selected_semantics",
        "coverage",
        "qualification_allowed",
    }
    if set(payload) != required_fields:
        raise SharadarSemanticsEvidenceError("semantics receipt payload fields differ")
    if payload["schema_version"] != SHARADAR_SEMANTICS_RECEIPT_SCHEMA_VERSION:
        raise SharadarSemanticsEvidenceError("unsupported semantics receipt schema")
    raw_artifact = payload["raw_artifact"]
    if not isinstance(raw_artifact, Mapping) or set(raw_artifact) != {
        "relative_path",
        "sha256",
        "byte_count",
    }:
        raise SharadarSemanticsEvidenceError("raw artifact fields differ")
    raw_path = _resolve_under(
        Path(warehouse_dir).resolve(), raw_artifact["relative_path"], label="raw response"
    )
    raw_size = raw_path.stat().st_size
    if raw_size <= 0 or raw_size > MAX_RAW_RESPONSE_BYTES:
        raise SharadarSemanticsEvidenceError("raw response size is invalid")
    raw = raw_path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    if (
        raw_sha != _sha(raw_artifact["sha256"], "raw response sha256")
        or raw_artifact["relative_path"] != _relative_raw_path(raw_sha)
        or type(raw_artifact["byte_count"]) is not int
        or raw_artifact["byte_count"] != len(raw)
    ):
        raise SharadarSemanticsEvidenceError("raw response identity differs")
    expected = build_sharadar_semantics_receipt(
        raw,
        candidate_id=payload["candidate_id"],
        captured_at_utc=payload["captured_at_utc"],
    )
    if dict(document) != expected or expected["artifact_hash"] != claimed:
        raise SharadarSemanticsEvidenceError(
            "semantics receipt does not replay from exact provider bytes"
        )
    return expected


def _create_only(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != len(raw)
            or path.read_bytes() != raw
        ):
            raise SharadarSemanticsEvidenceError(
                f"refusing to overwrite immutable semantics evidence: {path}"
            )
        return
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != len(raw)
                or path.read_bytes() != raw
            ):
                raise SharadarSemanticsEvidenceError(
                    f"refusing to overwrite immutable semantics evidence: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _publish_destination(root: Path, relative_path: str, *, label: str) -> Path:
    """Create ordinary parent directories without traversing nested symlinks."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SharadarSemanticsEvidenceError(f"{label} path escapes the warehouse")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise SharadarSemanticsEvidenceError("warehouse root is not a directory")
    parent = root
    for part in relative.parts[:-1]:
        parent /= part
        if parent.is_symlink():
            raise SharadarSemanticsEvidenceError(f"{label} parent is unsafe")
        try:
            parent.mkdir()
        except FileExistsError:
            pass
        if not parent.is_dir() or parent.is_symlink():
            raise SharadarSemanticsEvidenceError(f"{label} parent is unsafe")
    destination = parent / relative.parts[-1]
    if destination.is_symlink():
        raise SharadarSemanticsEvidenceError(f"{label} destination is unsafe")
    return destination


def publish_sharadar_semantics_receipt(
    warehouse_dir: str | os.PathLike[str],
    raw_response: bytes,
    *,
    candidate_id: str,
    captured_at_utc: str,
) -> tuple[dict[str, Any], Path]:
    """Create-only publish raw INDICATORS bytes and their canonical receipt."""
    root = Path(warehouse_dir).resolve()
    document = build_sharadar_semantics_receipt(
        raw_response,
        candidate_id=candidate_id,
        captured_at_utc=captured_at_utc,
    )
    raw_path = _publish_destination(
        root,
        document["payload"]["raw_artifact"]["relative_path"],
        label="raw response",
    )
    _create_only(raw_path, raw_response)
    receipt_path = _publish_destination(
        root,
        _relative_receipt_path(document["artifact_hash"]),
        label="semantics receipt",
    )
    receipt_raw = (canonical_json(document) + "\n").encode("utf-8")
    _create_only(receipt_path, receipt_raw)
    loaded = _strict_json_bytes(
        receipt_path.read_bytes(),
        label="semantics receipt",
        max_bytes=MAX_RECEIPT_BYTES,
    )
    verified = validate_sharadar_semantics_receipt(loaded, warehouse_dir=root)
    if verified != document:
        raise SharadarSemanticsEvidenceError("published semantics receipt changed")
    return verified, receipt_path


def load_sharadar_semantics_receipt(
    path: str | os.PathLike[str], *, warehouse_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise SharadarSemanticsEvidenceError("semantics receipt is not a regular file")
    size = source.stat().st_size
    if size <= 0 or size > MAX_RECEIPT_BYTES:
        raise SharadarSemanticsEvidenceError("semantics receipt size is invalid")
    raw = source.read_bytes()
    parsed = _strict_json_bytes(
        raw,
        label="semantics receipt",
        max_bytes=MAX_RECEIPT_BYTES,
    )
    if raw != (canonical_json(parsed) + "\n").encode("utf-8"):
        raise SharadarSemanticsEvidenceError(
            "semantics receipt bytes are not canonical JSON plus one newline"
        )
    return validate_sharadar_semantics_receipt(
        parsed,
        warehouse_dir=warehouse_dir,
    )


__all__ = [
    "CANONICAL_REQUEST",
    "INDICATORS_URL",
    "MAX_RAW_RESPONSE_BYTES",
    "MAX_RECEIPT_BYTES",
    "SHARADAR_SEMANTICS_RECEIPT_SCHEMA_VERSION",
    "SharadarSemanticsEvidenceError",
    "build_sharadar_semantics_receipt",
    "load_sharadar_semantics_receipt",
    "publish_sharadar_semantics_receipt",
    "validate_sharadar_semantics_receipt",
]
