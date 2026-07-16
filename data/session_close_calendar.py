"""Immutable NYSE session-close evidence used by the PEAD research boundary.

The calendar artifact contains official ICE/NYSE source URLs and the published
13:00 ET early-close sessions.  Trading-day membership still comes from the
observed SEP warehouse; this artifact supplies only the close timestamp for
those observed sessions.  The research implementations independently validate
the artifact and derive timestamps from it.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from collections.abc import Mapping
from typing import Any


DEFAULT_SESSION_CLOSE_CALENDAR = (
    Path(__file__).resolve().parent.parent
    / "research"
    / "pead_vq_locked_replication_v1"
    / "nyse_session_close_calendar.json"
)
MAX_CALENDAR_BYTES = 1024 * 1024
SOURCE_RECEIPT_SCHEMA_VERSION = "nyse_session_close_source_receipt.v1"
EXTRACTION_METHOD = "nyse_early_close_html_text.v1"
MAX_SOURCE_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_CLOCK_SKEW_SECONDS = 10 * 60
MAX_SOURCE_RECEIPT_DURATION_SECONDS = 60 * 60
DEFAULT_SOURCE_RECEIPT = (
    DEFAULT_SESSION_CLOSE_CALENDAR.parent
    / "nyse_session_close_sources"
    / "receipt.json"
)
_HEX = frozenset("0123456789abcdef")
_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}
_DATE_PATTERN = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s*,?\s*"
    r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\s*,\s*(\d{4})",
    re.IGNORECASE,
)
_EARLY_BLOCK_PATTERN = re.compile(
    r"\*{0,4}\s*Each market will close early at 1:00 p\.m\."
    r"(.*?)"
    r"(?=\*{1,4}\s*Each market will close early at 1:00 p\.m\."
    r"|NYSE Group Markets holidays|Link to (?:NYSE|Holidays)"
    r"|About (?:NYSE|Intercontinental)|SOURCE:|$)",
    re.IGNORECASE,
)


class SessionCloseCalendarError(ValueError):
    """The configured session-close calendar cannot be read exactly."""


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0 and data.strip():
            self.parts.append(data)


def canonical_json(value: Any) -> str:
    """Canonical finite JSON for source receipts."""
    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise SessionCloseCalendarError("receipt keys must be strings")
            return {key: normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise SessionCloseCalendarError(
            f"unsupported receipt value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SessionCloseCalendarError(f"{label} must be lowercase SHA-256")
    return value


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SessionCloseCalendarError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SessionCloseCalendarError(f"{label} must be canonical UTC") from exc
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if canonical != value:
        raise SessionCloseCalendarError(f"{label} must be canonical UTC")
    return value


def normalized_html_text(raw: bytes) -> str:
    """Return deterministic visible text from an archived official HTML page."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SessionCloseCalendarError("official calendar source is not UTF-8") from exc
    parser = _VisibleTextParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        raise SessionCloseCalendarError("official calendar HTML is malformed") from exc
    return " ".join(" ".join(parser.parts).split())


def extract_early_close_dates(raw: bytes) -> list[str]:
    """Extract dates only from official 1:00 p.m. early-close statements."""
    text = normalized_html_text(raw)
    dates: set[str] = set()
    for block in _EARLY_BLOCK_PATTERN.findall(text):
        for month_name, day_text, year_text in _DATE_PATTERN.findall(block):
            month = next(
                number for name, number in _MONTHS.items()
                if name.lower() == month_name.lower()
            )
            try:
                parsed = datetime(
                    int(year_text), month, int(day_text), tzinfo=timezone.utc
                ).date()
            except ValueError as exc:
                raise SessionCloseCalendarError(
                    "official early-close statement contains an invalid date"
                ) from exc
            dates.add(parsed.isoformat())
    return sorted(dates)


def _strict_json(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SessionCloseCalendarError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise SessionCloseCalendarError(f"{label} exceeds its size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SessionCloseCalendarError(f"{label} is not UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SessionCloseCalendarError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise SessionCloseCalendarError(f"{label} contains invalid number {token}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise SessionCloseCalendarError(
            f"invalid {label} JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise SessionCloseCalendarError(f"{label} root must be an object")
    return value


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SessionCloseCalendarError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _calendar_sources(calendar: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    wrapper = _exact_fields(calendar, {"artifact_hash", "payload"}, "calendar")
    payload = wrapper["payload"]
    if not isinstance(payload, Mapping):
        raise SessionCloseCalendarError("calendar payload must be an object")
    artifact_hash = _sha256(wrapper["artifact_hash"], "calendar artifact_hash")
    if content_hash(payload) != artifact_hash:
        raise SessionCloseCalendarError("calendar artifact hash mismatch")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SessionCloseCalendarError("calendar sources must be a non-empty list")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = _exact_fields(
            raw_source,
            {"source_id", "publisher", "url", "covered_years"},
            f"calendar source {index}",
        )
        source_id = source["source_id"]
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in seen
        ):
            raise SessionCloseCalendarError("calendar source IDs must be unique strings")
        if not isinstance(source["publisher"], str) or not source["publisher"]:
            raise SessionCloseCalendarError("calendar source publisher is invalid")
        if not isinstance(source["url"], str) or not source["url"].startswith("https://"):
            raise SessionCloseCalendarError("calendar source URL must use HTTPS")
        years = source["covered_years"]
        if (
            not isinstance(years, list)
            or not years
            or any(type(year) is not int or not 1900 <= year <= 2200 for year in years)
            or years != sorted(set(years))
        ):
            raise SessionCloseCalendarError(
                "calendar source covered_years must be sorted unique years"
            )
        seen.add(source_id)
        result.append(source)
    return result


def _optional_utc_timestamp(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _utc_timestamp(value, label)


def _utc_datetime(value: Any, label: str) -> datetime:
    canonical = _utc_timestamp(value, label)
    return datetime.fromisoformat(canonical[:-1] + "+00:00")


def _validate_source_chronology(
    http: Mapping[str, Any],
    *,
    retrieved_at_utc: Any,
    created_at_utc: Any,
    label: str,
) -> None:
    """Require a plausible acquisition chronology, not just parseable clocks."""
    retrieved = _utc_datetime(retrieved_at_utc, f"{label} retrieved_at_utc")
    created = _utc_datetime(created_at_utc, "receipt created_at_utc")
    if retrieved > created:
        raise SessionCloseCalendarError(f"{label} was retrieved after receipt creation")
    if (created - retrieved).total_seconds() > MAX_SOURCE_RECEIPT_DURATION_SECONDS:
        raise SessionCloseCalendarError(f"{label} receipt duration is implausible")
    if http["date_utc"] is None:
        raise SessionCloseCalendarError(f"{label} HTTP Date is required")
    server_date = _utc_datetime(http["date_utc"], f"{label} HTTP date")
    if abs((server_date - retrieved).total_seconds()) > MAX_SOURCE_CLOCK_SKEW_SECONDS:
        raise SessionCloseCalendarError(f"{label} HTTP Date clock skew is implausible")
    last_modified = http["last_modified_utc"]
    if last_modified is not None and _utc_datetime(
        last_modified, f"{label} HTTP last-modified"
    ) > server_date:
        raise SessionCloseCalendarError(
            f"{label} HTTP Last-Modified occurs after the response Date"
        )


def build_session_close_source_receipt(
    calendar_path: str | Path,
    source_documents: Mapping[str, bytes],
    source_metadata: Mapping[str, Mapping[str, Any]],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build a content-addressed receipt for exact official source bytes.

    The caller owns network acquisition and publication.  This pure builder
    derives all calendar-facing fields and extraction results itself so the
    acquisition script cannot silently transcribe them.
    """
    calendar = load_session_close_calendar(calendar_path)
    sources = _calendar_sources(calendar)
    expected_ids = {str(source["source_id"]) for source in sources}
    if set(source_documents) != expected_ids or set(source_metadata) != expected_ids:
        raise SessionCloseCalendarError(
            "source documents and metadata must exactly cover calendar sources"
        )
    created = _utc_timestamp(created_at_utc, "receipt created_at_utc")
    receipt_sources: list[dict[str, Any]] = []
    for source in sources:
        source_id = str(source["source_id"])
        raw = source_documents[source_id]
        if not isinstance(raw, bytes) or not raw or len(raw) > MAX_SOURCE_DOCUMENT_BYTES:
            raise SessionCloseCalendarError(
                f"source document {source_id!r} must be non-empty bounded bytes"
            )
        metadata = _exact_fields(
            source_metadata[source_id],
            {"retrieved_at_utc", "http"},
            f"source metadata {source_id}",
        )
        http = _exact_fields(
            metadata["http"],
            {"status_code", "date_utc", "content_type", "etag", "last_modified_utc"},
            f"source HTTP metadata {source_id}",
        )
        if http["status_code"] != 200:
            raise SessionCloseCalendarError(f"source {source_id!r} HTTP status is not 200")
        content_type = http["content_type"]
        if not isinstance(content_type, str) or "html" not in content_type.lower():
            raise SessionCloseCalendarError(f"source {source_id!r} is not HTML")
        etag = http["etag"]
        if etag is not None and (not isinstance(etag, str) or not etag):
            raise SessionCloseCalendarError(f"source {source_id!r} ETag is invalid")
        _validate_source_chronology(
            http,
            retrieved_at_utc=metadata["retrieved_at_utc"],
            created_at_utc=created,
            label=f"source {source_id}",
        )
        normalized = normalized_html_text(raw)
        if not normalized:
            raise SessionCloseCalendarError(f"source {source_id!r} has no visible text")
        sha256 = _sha256_bytes(raw)
        receipt_sources.append(
            {
                "source_id": source_id,
                "publisher": source["publisher"],
                "url": source["url"],
                "covered_years": list(source["covered_years"]),
                "retrieved_at_utc": _utc_timestamp(
                    metadata["retrieved_at_utc"],
                    f"source {source_id} retrieved_at_utc",
                ),
                "http": {
                    "status_code": 200,
                    "date_utc": _utc_timestamp(
                        http["date_utc"], f"source {source_id} HTTP date"
                    ),
                    "content_type": content_type,
                    "etag": etag,
                    "last_modified_utc": _optional_utc_timestamp(
                        http["last_modified_utc"],
                        f"source {source_id} HTTP last-modified",
                    ),
                },
                "raw_document": {
                    "relative_path": f"raw/{sha256}.html",
                    "sha256": sha256,
                    "bytes": len(raw),
                },
                "extraction": {
                    "method": EXTRACTION_METHOD,
                    "normalized_text_sha256": _sha256_bytes(normalized.encode("utf-8")),
                    "early_close_dates": extract_early_close_dates(raw),
                },
            }
        )
    payload = {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "calendar_artifact_hash": calendar["artifact_hash"],
        "created_at_utc": created,
        "sources": receipt_sources,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _validate_source_receipt(
    receipt: Mapping[str, Any],
    *,
    calendar: Mapping[str, Any],
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    wrapper = _exact_fields(receipt, {"artifact_hash", "payload"}, "source receipt")
    payload = _exact_fields(
        wrapper["payload"],
        {"schema_version", "calendar_artifact_hash", "created_at_utc", "sources"},
        "source receipt payload",
    )
    receipt_hash = _sha256(wrapper["artifact_hash"], "source receipt artifact_hash")
    if content_hash(payload) != receipt_hash:
        raise SessionCloseCalendarError("source receipt artifact hash mismatch")
    if payload["schema_version"] != SOURCE_RECEIPT_SCHEMA_VERSION:
        raise SessionCloseCalendarError("unsupported source receipt schema")
    if payload["calendar_artifact_hash"] != calendar["artifact_hash"]:
        raise SessionCloseCalendarError("source receipt binds a different calendar")
    _utc_timestamp(payload["created_at_utc"], "source receipt created_at_utc")
    calendar_sources = _calendar_sources(calendar)
    receipt_sources = payload["sources"]
    if not isinstance(receipt_sources, list) or len(receipt_sources) != len(calendar_sources):
        raise SessionCloseCalendarError("source receipt does not cover calendar sources")

    document_root = receipt_path.parent.resolve()
    encoded_documents: dict[str, str] = {}
    for index, (calendar_source, raw_entry) in enumerate(
        zip(calendar_sources, receipt_sources, strict=True)
    ):
        entry = _exact_fields(
            raw_entry,
            {
                "source_id", "publisher", "url", "covered_years",
                "retrieved_at_utc", "http", "raw_document", "extraction",
            },
            f"source receipt entry {index}",
        )
        for field in ("source_id", "publisher", "url", "covered_years"):
            if entry[field] != calendar_source[field]:
                raise SessionCloseCalendarError(
                    f"source receipt entry {index} differs from calendar {field}"
                )
        source_id = str(entry["source_id"])
        _utc_timestamp(entry["retrieved_at_utc"], f"source {source_id} retrieved_at_utc")
        http = _exact_fields(
            entry["http"],
            {"status_code", "date_utc", "content_type", "etag", "last_modified_utc"},
            f"source {source_id} HTTP metadata",
        )
        if http["status_code"] != 200:
            raise SessionCloseCalendarError(f"source {source_id} HTTP status is not 200")
        if not isinstance(http["content_type"], str) or "html" not in http["content_type"].lower():
            raise SessionCloseCalendarError(f"source {source_id} is not HTML")
        if http["etag"] is not None and (
            not isinstance(http["etag"], str) or not http["etag"]
        ):
            raise SessionCloseCalendarError(f"source {source_id} ETag is invalid")
        _validate_source_chronology(
            http,
            retrieved_at_utc=entry["retrieved_at_utc"],
            created_at_utc=payload["created_at_utc"],
            label=f"source {source_id}",
        )
        raw_document = _exact_fields(
            entry["raw_document"],
            {"relative_path", "sha256", "bytes"},
            f"source {source_id} raw document",
        )
        raw_hash = _sha256(raw_document["sha256"], f"source {source_id} raw SHA-256")
        expected_path = f"raw/{raw_hash}.html"
        if raw_document["relative_path"] != expected_path:
            raise SessionCloseCalendarError(
                f"source {source_id} raw document path is not content-addressed"
            )
        if type(raw_document["bytes"]) is not int or not 0 < raw_document["bytes"] <= MAX_SOURCE_DOCUMENT_BYTES:
            raise SessionCloseCalendarError(f"source {source_id} byte count is invalid")
        raw_path = receipt_path.parent / expected_path
        if not raw_path.is_file() or raw_path.is_symlink():
            raise SessionCloseCalendarError(f"source {source_id} raw document is missing")
        try:
            resolved = raw_path.resolve(strict=True)
            resolved.relative_to(document_root)
        except (OSError, ValueError) as exc:
            raise SessionCloseCalendarError(
                f"source {source_id} raw document escapes the evidence directory"
            ) from exc
        raw = raw_path.read_bytes()
        if len(raw) != raw_document["bytes"] or _sha256_bytes(raw) != raw_hash:
            raise SessionCloseCalendarError(f"source {source_id} raw document mismatch")
        normalized = normalized_html_text(raw)
        extraction = _exact_fields(
            entry["extraction"],
            {"method", "normalized_text_sha256", "early_close_dates"},
            f"source {source_id} extraction",
        )
        if extraction["method"] != EXTRACTION_METHOD:
            raise SessionCloseCalendarError(f"source {source_id} extraction method differs")
        normalized_hash = _sha256(
            extraction["normalized_text_sha256"],
            f"source {source_id} normalized text SHA-256",
        )
        if _sha256_bytes(normalized.encode("utf-8")) != normalized_hash:
            raise SessionCloseCalendarError(f"source {source_id} normalized text mismatch")
        dates = extraction["early_close_dates"]
        if (
            not isinstance(dates, list)
            or any(not isinstance(value, str) for value in dates)
            or dates != sorted(set(dates))
            or dates != extract_early_close_dates(raw)
        ):
            raise SessionCloseCalendarError(f"source {source_id} extraction differs")
        encoded_documents[source_id] = base64.b64encode(raw).decode("ascii")

    return {"artifact_hash": receipt_hash, "payload": dict(payload)}, encoded_documents


def load_session_close_calendar_evidence(
    calendar_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and byte-verify the calendar, receipt, and archived source pages."""
    resolved_calendar = (
        Path(calendar_path) if calendar_path is not None else DEFAULT_SESSION_CLOSE_CALENDAR
    )
    resolved_receipt = (
        Path(receipt_path)
        if receipt_path is not None
        else (
            DEFAULT_SOURCE_RECEIPT
            if calendar_path is None
            else resolved_calendar.parent / "nyse_session_close_sources" / "receipt.json"
        )
    )
    calendar = load_session_close_calendar(resolved_calendar)
    _calendar_sources(calendar)
    receipt = _strict_json(
        resolved_receipt,
        label="session-close source receipt",
        max_bytes=MAX_CALENDAR_BYTES,
    )
    verified_receipt, documents = _validate_source_receipt(
        receipt,
        calendar=calendar,
        receipt_path=resolved_receipt,
    )
    return {
        "calendar": calendar,
        "source_receipt": verified_receipt,
        "source_documents": documents,
    }


def load_session_close_calendar(path: str | Path | None = None) -> dict[str, Any]:
    """Read a strict, duplicate-key-free JSON calendar without changing it."""
    calendar_path = Path(path) if path is not None else DEFAULT_SESSION_CLOSE_CALENDAR
    return _strict_json(
        calendar_path,
        label="session-close calendar",
        max_bytes=MAX_CALENDAR_BYTES,
    )


__all__ = [
    "DEFAULT_SESSION_CLOSE_CALENDAR",
    "DEFAULT_SOURCE_RECEIPT",
    "EXTRACTION_METHOD",
    "MAX_SOURCE_CLOCK_SKEW_SECONDS",
    "MAX_SOURCE_DOCUMENT_BYTES",
    "MAX_SOURCE_RECEIPT_DURATION_SECONDS",
    "SOURCE_RECEIPT_SCHEMA_VERSION",
    "SessionCloseCalendarError",
    "build_session_close_source_receipt",
    "content_hash",
    "extract_early_close_dates",
    "load_session_close_calendar",
    "load_session_close_calendar_evidence",
    "normalized_html_text",
]
