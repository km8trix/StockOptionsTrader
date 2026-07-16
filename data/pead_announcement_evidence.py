"""Strict independent announcement evidence for the locked PEAD candidate.

This module is deliberately independent of the PEAD normalizers.  It records
issuer-period outcomes against a frozen :mod:`data.pead_event_universe`
manifest and, for available outcomes, preserves exact SEC filing metadata and
earnings-exhibit bytes.  Vendor actuals and vendor report times are not inputs
to this artifact.

An EDGAR acceptance time, a contemporaneous observation that a filing was
public, and the first-public time are three different facts.  The schema keeps
them separate.  In particular, a later retrieval can verify archived bytes but
cannot prove when a historical announcement first became public.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import binascii
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from data.pead_event_universe import (
    PeadEventUniverseError,
    canonical_event_id,
    canonical_json,
    content_hash,
    validate_event_key,
    validate_pead_event_universe,
)


ANNOUNCEMENT_EVIDENCE_SCHEMA_VERSION = "pead_announcement_evidence.v1"
SEC_SOURCE_KIND = "sec_edgar_item_2_02_exhibit"
MAX_ANNOUNCEMENT_EVIDENCE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_HTTP_CLOCK_SKEW_SECONDS = 10 * 60
MAX_CONTEMPORANEOUS_OBSERVATION_DELAY_SECONDS = 60 * 60

_HEX = frozenset("0123456789abcdef")
_ACCESSION_PATTERN = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_DECIMAL_PATTERN = re.compile(
    r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$"
)
_SEC_HOSTS = frozenset({"sec.gov", "www.sec.gov", "data.sec.gov"})
_EASTERN = ZoneInfo("America/New_York")

_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "created_at_utc",
    "expected_event_manifest_hash",
    "expected_event_ids",
    "outcomes",
    "coverage",
}
_OUTCOME_FIELDS = {
    "event_id",
    "event_key",
    "disposition",
    "missing_reason",
    "available_record",
}
_AVAILABLE_FIELDS = {
    "source_kind",
    "accession_number",
    "form",
    "item",
    "exhibit",
    "metadata_document",
    "exhibit_document",
    "edgar_acceptance_at_utc",
    "first_public_at_utc",
    "first_public_basis",
    "first_public_proof",
    "observed_public_by_at_utc",
    "observed_public_by_basis",
    "observed_public_document",
    "canonical_actual",
    "extraction",
}
_DOCUMENT_FIELDS = {"role", "url", "retrieved_at_utc", "http", "raw_document"}
_HTTP_FIELDS = {
    "status_code",
    "date_utc",
    "content_type",
    "etag",
    "last_modified_at_utc",
}
_RAW_DOCUMENT_FIELDS = {"sha256", "bytes", "base64"}
_ACTUAL_FIELDS = {
    "announced_value",
    "canonical_value",
    "normalization_factor",
    "metric",
    "source_metric_label",
    "metric_definition_sha256",
    "accounting_basis",
    "per_share_basis",
    "scope",
    "currency",
    "unit",
    "announced_share_basis",
    "canonical_share_basis",
    "fiscal_period_end",
    "fiscal_period_type",
    "normalization_evidence_sha256",
}
_EXTRACTION_FIELDS = {
    "method",
    "code_hash",
    "reviewer",
    "locator",
    "source_document_sha256",
}
_COVERAGE_FIELDS = {
    "expected_events",
    "available_events",
    "timing_eligible_events",
    "missing_events",
    "event_universe_qualified",
    "blockers",
    "complete",
}


class PeadAnnouncementEvidenceError(ValueError):
    """Announcement evidence is malformed, incomplete, or self-inconsistent."""


class _VisibleTextParser(HTMLParser):
    """Minimal deterministic visible-text extractor for archived SEC exhibits."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0 and data.strip():
            self.parts.append(data)


def _plain_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadAnnouncementEvidenceError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadAnnouncementEvidenceError(
            f"{label} must be nonempty canonical text"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadAnnouncementEvidenceError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadAnnouncementEvidenceError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeadAnnouncementEvidenceError(
            f"{label} must be canonical UTC with Z"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PeadAnnouncementEvidenceError(f"{label} must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if utc.microsecond else "seconds"
    canonical = utc.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadAnnouncementEvidenceError(f"{label} must be canonical UTC with Z")
    return value


def _utc_value(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _optional_utc(value: Any, label: str) -> str | None:
    return None if value is None else _utc_timestamp(value, label)


def _canonical_decimal(value: Any, label: str, *, positive: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 128
        or _DECIMAL_PATTERN.fullmatch(value) is None
    ):
        raise PeadAnnouncementEvidenceError(f"{label} must be a canonical decimal")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex already guards it
        raise PeadAnnouncementEvidenceError(f"{label} must be a decimal") from exc
    if not number.is_finite() or (positive and number <= 0):
        qualifier = "positive " if positive else ""
        raise PeadAnnouncementEvidenceError(
            f"{label} must be a finite {qualifier}decimal"
        )
    if number == 0 and value.startswith("-"):
        raise PeadAnnouncementEvidenceError(f"{label} cannot be negative zero")
    return value


def _https_url(value: Any, label: str, *, sec_only: bool) -> str:
    text = _required_text(value, label)
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise PeadAnnouncementEvidenceError(f"{label} must be HTTPS") from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise PeadAnnouncementEvidenceError(f"{label} must be a canonical HTTPS URL")
    if sec_only and parsed.hostname.lower() not in _SEC_HOSTS:
        raise PeadAnnouncementEvidenceError(f"{label} must be an SEC URL")
    return text


def _canonical_base64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise PeadAnnouncementEvidenceError(f"{label} must be canonical base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PeadAnnouncementEvidenceError(
            f"{label} must be canonical base64"
        ) from exc
    if base64.b64encode(raw).decode("ascii") != value:
        raise PeadAnnouncementEvidenceError(f"{label} must be canonical base64")
    if not raw or len(raw) > MAX_SOURCE_DOCUMENT_BYTES:
        raise PeadAnnouncementEvidenceError(f"{label} decoded size is invalid")
    return raw


def _document_bytes(document: Mapping[str, Any], label: str) -> bytes:
    return _canonical_base64(
        document["raw_document"]["base64"], f"{label}.raw_document.base64"
    )


def _unique_sgml_value(text: str, tag: str, label: str) -> str:
    values = [
        value.strip()
        for value in re.findall(
            rf"<{re.escape(tag)}>\s*([^\r\n<]+)", text, flags=re.IGNORECASE
        )
    ]
    if len(values) != 1 or not values[0]:
        raise PeadAnnouncementEvidenceError(
            f"{label} must contain exactly one SGML {tag} value"
        )
    return values[0]


def _replay_complete_submission_metadata(raw: bytes, label: str) -> dict[str, Any]:
    """Replay the narrow SEC complete-submission metadata contract."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadAnnouncementEvidenceError(
            f"{label} complete submission must be UTF-8"
        ) from exc
    accession = _unique_sgml_value(text, "ACCESSION-NUMBER", label)
    form = _unique_sgml_value(text, "CONFORMED-SUBMISSION-TYPE", label).upper()
    acceptance_text = _unique_sgml_value(text, "ACCEPTANCE-DATETIME", label)
    cik = _unique_sgml_value(text, "CENTRAL-INDEX-KEY", label).zfill(10)
    if not re.fullmatch(r"[0-9]{14}", acceptance_text):
        raise PeadAnnouncementEvidenceError(
            f"{label} ACCEPTANCE-DATETIME must be YYYYMMDDhhmmss"
        )
    try:
        local = datetime.strptime(acceptance_text, "%Y%m%d%H%M%S").replace(
            tzinfo=_EASTERN
        )
    except ValueError as exc:
        raise PeadAnnouncementEvidenceError(
            f"{label} ACCEPTANCE-DATETIME is invalid"
        ) from exc
    acceptance = local.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    items = {
        value.strip()
        for value in re.findall(r"<ITEMS>\s*([^\r\n<]+)", text, re.IGNORECASE)
    }
    exhibits: dict[str, str] = {}
    for block in re.findall(
        r"<DOCUMENT>(.*?)(?:</DOCUMENT>|\Z)", text, flags=re.IGNORECASE | re.DOTALL
    ):
        types = re.findall(r"<TYPE>\s*([^\r\n<]+)", block, re.IGNORECASE)
        filenames = re.findall(r"<FILENAME>\s*([^\r\n<]+)", block, re.IGNORECASE)
        if len(types) == 1 and types[0].strip().upper().startswith("EX-99"):
            exhibit_type = types[0].strip().upper()
            if len(filenames) != 1 or not filenames[0].strip():
                raise PeadAnnouncementEvidenceError(
                    f"{label} EX-99 document must identify exactly one filename"
                )
            if exhibit_type in exhibits:
                raise PeadAnnouncementEvidenceError(
                    f"{label} repeats an EX-99 exhibit type"
                )
            exhibits[exhibit_type] = filenames[0].strip()
    return {
        "accession_number": accession,
        "form": form,
        "items": items,
        "exhibits": exhibits,
        "edgar_acceptance_at_utc": acceptance,
        "cik": cik,
    }


def _visible_text(raw: bytes, label: str) -> str:
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadAnnouncementEvidenceError(f"{label} must be UTF-8") from exc
    parser = _VisibleTextParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as exc:
        raise PeadAnnouncementEvidenceError(f"{label} cannot be parsed") from exc
    text = " ".join(" ".join(parser.parts).split())
    if not text:
        raise PeadAnnouncementEvidenceError(f"{label} has no visible text")
    return text


def _replay_exhibit_metric(
    raw: bytes, *, source_metric_label: str, locator: str, label: str
) -> str:
    """Extract one explicitly labelled EPS value without any vendor actual."""
    expected_locator = f"visible_text:{source_metric_label}"
    if locator != expected_locator:
        raise PeadAnnouncementEvidenceError(
            f"{label} locator must bind the exact source metric label"
        )
    text = _visible_text(raw, label)
    escaped = re.escape(source_metric_label)
    pattern = re.compile(
        rf"(?<!\w){escaped}\s*(?::|was|were|of)?\s*"
        r"(?:\$|USD\s*)?\s*(\(?-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\)?)",
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise PeadAnnouncementEvidenceError(
            f"{label} must contain exactly one replayable labelled EPS value"
        )
    value = matches[0]
    negative_parentheses = value.startswith("(") and value.endswith(")")
    if negative_parentheses:
        value = "-" + value[1:-1]
    return _canonical_decimal(value, f"{label} extracted value")


def _validate_http(value: Any, *, label: str, retrieved_at: str) -> dict[str, Any]:
    http = _exact_fields(value, _HTTP_FIELDS, label)
    if type(http["status_code"]) is not int or http["status_code"] != 200:
        raise PeadAnnouncementEvidenceError(f"{label}.status_code must be 200")
    server_date = _utc_timestamp(http["date_utc"], f"{label}.date_utc")
    retrieved = _utc_value(retrieved_at)
    server = _utc_value(server_date)
    skew = (retrieved - server).total_seconds()
    if not 0 <= skew <= MAX_HTTP_CLOCK_SKEW_SECONDS:
        raise PeadAnnouncementEvidenceError(
            f"{label} HTTP Date must not follow retrieval and must have bounded skew"
        )
    content_type = _required_text(http["content_type"], f"{label}.content_type")
    etag = http["etag"]
    if etag is not None:
        etag = _required_text(etag, f"{label}.etag")
    modified = _optional_utc(
        http["last_modified_at_utc"], f"{label}.last_modified_at_utc"
    )
    if modified is not None and _utc_value(modified) > server:
        raise PeadAnnouncementEvidenceError(
            f"{label}.last_modified_at_utc follows HTTP Date"
        )
    return {
        "status_code": 200,
        "date_utc": server_date,
        "content_type": content_type,
        "etag": etag,
        "last_modified_at_utc": modified,
    }


def _validate_document(
    value: Any,
    *,
    expected_role: str,
    label: str,
    sec_only: bool = True,
) -> dict[str, Any]:
    document = _exact_fields(value, _DOCUMENT_FIELDS, label)
    if document["role"] != expected_role:
        raise PeadAnnouncementEvidenceError(f"{label}.role is invalid")
    url = _https_url(document["url"], f"{label}.url", sec_only=sec_only)
    retrieved = _utc_timestamp(
        document["retrieved_at_utc"], f"{label}.retrieved_at_utc"
    )
    http = _validate_http(document["http"], label=f"{label}.http", retrieved_at=retrieved)
    raw_document = _exact_fields(
        document["raw_document"], _RAW_DOCUMENT_FIELDS, f"{label}.raw_document"
    )
    claimed = _sha256(raw_document["sha256"], f"{label}.raw_document.sha256")
    count = raw_document["bytes"]
    if type(count) is not int or not 0 < count <= MAX_SOURCE_DOCUMENT_BYTES:
        raise PeadAnnouncementEvidenceError(
            f"{label}.raw_document.bytes must be a positive bounded integer"
        )
    raw = _canonical_base64(raw_document["base64"], f"{label}.raw_document.base64")
    if len(raw) != count or hashlib.sha256(raw).hexdigest() != claimed:
        raise PeadAnnouncementEvidenceError(f"{label} archived bytes/hash mismatch")
    return {
        "role": expected_role,
        "url": url,
        "retrieved_at_utc": retrieved,
        "http": http,
        "raw_document": {
            "sha256": claimed,
            "bytes": count,
            "base64": raw_document["base64"],
        },
    }


def _build_document(
    *, role: str, url: str, retrieved_at_utc: str, http: Mapping[str, Any], raw: bytes
) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise PeadAnnouncementEvidenceError(f"{role} source must be exact bytes")
    document = {
        "role": role,
        "url": url,
        "retrieved_at_utc": retrieved_at_utc,
        "http": dict(http),
        "raw_document": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        },
    }
    return _validate_document(
        document,
        expected_role=role,
        label=f"{role} document",
        sec_only=True,
    )


def _validate_actual(
    value: Any, *, event_key: Mapping[str, str], label: str
) -> dict[str, str]:
    actual = _exact_fields(value, _ACTUAL_FIELDS, label)
    announced = _canonical_decimal(actual["announced_value"], f"{label}.announced_value")
    canonical = _canonical_decimal(actual["canonical_value"], f"{label}.canonical_value")
    factor = _canonical_decimal(
        actual["normalization_factor"],
        f"{label}.normalization_factor",
        positive=True,
    )
    if Decimal(announced) * Decimal(factor) != Decimal(canonical):
        raise PeadAnnouncementEvidenceError(
            f"{label} canonical value does not equal announced value times factor"
        )
    if actual["metric"] != "earnings_per_share":
        raise PeadAnnouncementEvidenceError(f"{label}.metric is unsupported")
    source_metric_label = _required_text(
        actual["source_metric_label"], f"{label}.source_metric_label"
    )
    metric_definition_hash = _sha256(
        actual["metric_definition_sha256"], f"{label}.metric_definition_sha256"
    )
    if actual["accounting_basis"] not in {"gaap", "non_gaap"}:
        raise PeadAnnouncementEvidenceError(f"{label}.accounting_basis is invalid")
    if actual["per_share_basis"] not in {"basic", "diluted"}:
        raise PeadAnnouncementEvidenceError(f"{label}.per_share_basis is invalid")
    if actual["scope"] not in {"continuing_operations", "total_company"}:
        raise PeadAnnouncementEvidenceError(f"{label}.scope is invalid")
    currency = actual["currency"]
    if not isinstance(currency, str) or _CURRENCY_PATTERN.fullmatch(currency) is None:
        raise PeadAnnouncementEvidenceError(f"{label}.currency must be ISO-style uppercase")
    if actual["unit"] != "currency_per_share":
        raise PeadAnnouncementEvidenceError(f"{label}.unit is unsupported")
    announced_basis = _required_text(
        actual["announced_share_basis"], f"{label}.announced_share_basis"
    )
    canonical_basis = _required_text(
        actual["canonical_share_basis"], f"{label}.canonical_share_basis"
    )
    if actual["fiscal_period_end"] != event_key["fiscal_period_end"]:
        raise PeadAnnouncementEvidenceError(
            f"{label}.fiscal_period_end differs from event key"
        )
    if actual["fiscal_period_type"] != event_key["fiscal_period_type"]:
        raise PeadAnnouncementEvidenceError(
            f"{label}.fiscal_period_type differs from event key"
        )
    normalization_hash = _sha256(
        actual["normalization_evidence_sha256"],
        f"{label}.normalization_evidence_sha256",
    )
    return {
        "announced_value": announced,
        "canonical_value": canonical,
        "normalization_factor": factor,
        "metric": "earnings_per_share",
        "source_metric_label": source_metric_label,
        "metric_definition_sha256": metric_definition_hash,
        "accounting_basis": actual["accounting_basis"],
        "per_share_basis": actual["per_share_basis"],
        "scope": actual["scope"],
        "currency": currency,
        "unit": "currency_per_share",
        "announced_share_basis": announced_basis,
        "canonical_share_basis": canonical_basis,
        "fiscal_period_end": event_key["fiscal_period_end"],
        "fiscal_period_type": event_key["fiscal_period_type"],
        "normalization_evidence_sha256": normalization_hash,
    }


def _validate_extraction(
    value: Any,
    *,
    exhibit_document: Mapping[str, Any],
    actual: Mapping[str, str],
    label: str,
) -> dict[str, str]:
    extraction = _exact_fields(value, _EXTRACTION_FIELDS, label)
    source_hash = _sha256(
        extraction["source_document_sha256"], f"{label}.source_document_sha256"
    )
    exhibit_hash = exhibit_document["raw_document"]["sha256"]
    if source_hash != exhibit_hash:
        raise PeadAnnouncementEvidenceError(
            f"{label} does not bind the archived SEC exhibit"
        )
    if extraction["method"] != "sec_exhibit_label_value_visible_text.v1":
        raise PeadAnnouncementEvidenceError(
            f"{label}.method is not the supported replayable SEC exhibit parser"
        )
    locator = _required_text(extraction["locator"], f"{label}.locator")
    replayed_value = _replay_exhibit_metric(
        _document_bytes(exhibit_document, "SEC exhibit"),
        source_metric_label=actual["source_metric_label"],
        locator=locator,
        label="SEC exhibit",
    )
    if replayed_value != actual["announced_value"]:
        raise PeadAnnouncementEvidenceError(
            f"{label} announced value differs from archived SEC exhibit"
        )
    return {
        "method": "sec_exhibit_label_value_visible_text.v1",
        "code_hash": _sha256(extraction["code_hash"], f"{label}.code_hash"),
        "reviewer": _required_text(extraction["reviewer"], f"{label}.reviewer"),
        "locator": locator,
        "source_document_sha256": source_hash,
    }


def _validate_available(
    value: Any,
    *,
    event_key: Mapping[str, str],
    created_at: str,
    label: str,
) -> dict[str, Any]:
    record = _exact_fields(value, _AVAILABLE_FIELDS, label)
    if record["source_kind"] != SEC_SOURCE_KIND:
        raise PeadAnnouncementEvidenceError(
            f"{label}.source_kind must identify independent SEC evidence"
        )
    accession = record["accession_number"]
    if not isinstance(accession, str) or _ACCESSION_PATTERN.fullmatch(accession) is None:
        raise PeadAnnouncementEvidenceError(f"{label}.accession_number is invalid")
    if accession[:10] != event_key["cik"]:
        raise PeadAnnouncementEvidenceError(f"{label} accession CIK differs from event")
    if record["form"] != "8-K" or record["item"] != "2.02":
        raise PeadAnnouncementEvidenceError(
            f"{label} must be an SEC 8-K Item 2.02 filing"
        )
    exhibit = _required_text(record["exhibit"], f"{label}.exhibit")
    if not exhibit.upper().startswith("EX-99"):
        raise PeadAnnouncementEvidenceError(f"{label}.exhibit must be an EX-99 exhibit")

    metadata = _validate_document(
        record["metadata_document"],
        expected_role="filing_metadata",
        label=f"{label}.metadata_document",
    )
    exhibit_document = _validate_document(
        record["exhibit_document"],
        expected_role="earnings_exhibit",
        label=f"{label}.exhibit_document",
    )
    replayed_metadata = _replay_complete_submission_metadata(
        _document_bytes(metadata, "SEC filing metadata"),
        "SEC filing metadata",
    )
    if replayed_metadata["accession_number"] != accession:
        raise PeadAnnouncementEvidenceError(
            f"{label} accession differs from archived SEC metadata"
        )
    if replayed_metadata["cik"] != event_key["cik"]:
        raise PeadAnnouncementEvidenceError(
            f"{label} event CIK differs from archived SEC metadata"
        )
    if replayed_metadata["form"] != "8-K" or "2.02" not in replayed_metadata["items"]:
        raise PeadAnnouncementEvidenceError(
            f"{label} archived metadata does not prove 8-K Item 2.02"
        )
    if exhibit.upper() not in replayed_metadata["exhibits"]:
        raise PeadAnnouncementEvidenceError(
            f"{label} archived metadata does not bind the named EX-99 exhibit"
        )
    compact_accession = accession.replace("-", "")
    for document_name, document in (
        ("metadata", metadata),
        ("exhibit", exhibit_document),
    ):
        if f"/{compact_accession}/" not in urlsplit(document["url"]).path:
            raise PeadAnnouncementEvidenceError(
                f"{label} {document_name} URL does not bind the accession"
            )
    expected_exhibit_filename = replayed_metadata["exhibits"][exhibit.upper()]
    if Path(urlsplit(exhibit_document["url"]).path).name != expected_exhibit_filename:
        raise PeadAnnouncementEvidenceError(
            f"{label} exhibit URL differs from archived SEC metadata"
        )
    acceptance = _utc_timestamp(
        record["edgar_acceptance_at_utc"], f"{label}.edgar_acceptance_at_utc"
    )
    if acceptance != replayed_metadata["edgar_acceptance_at_utc"]:
        raise PeadAnnouncementEvidenceError(
            f"{label} acceptance time differs from archived SEC metadata"
        )
    accepted_dt = _utc_value(acceptance)
    created_dt = _utc_value(created_at)
    source_retrievals = (
        _utc_value(metadata["retrieved_at_utc"]),
        _utc_value(exhibit_document["retrieved_at_utc"]),
    )
    if any(retrieved > created_dt for retrieved in source_retrievals):
        raise PeadAnnouncementEvidenceError(
            f"{label} source retrieval follows receipt creation"
        )
    if accepted_dt > min(
        *source_retrievals,
        created_dt,
    ):
        raise PeadAnnouncementEvidenceError(
            f"{label} EDGAR acceptance follows retrieval or receipt creation"
        )

    first_public = _optional_utc(
        record["first_public_at_utc"], f"{label}.first_public_at_utc"
    )
    first_basis = record["first_public_basis"]
    first_proof = record["first_public_proof"]
    if first_public is not None or first_basis != "not_proven" or first_proof is not None:
        raise PeadAnnouncementEvidenceError(
            f"{label} v1 has no replayable authoritative first-public adapter"
        )
    normalized_first_proof = None

    observed = _optional_utc(
        record["observed_public_by_at_utc"],
        f"{label}.observed_public_by_at_utc",
    )
    observed_basis = record["observed_public_by_basis"]
    observed_document = record["observed_public_document"]
    if observed is None:
        if observed_basis != "not_proven" or observed_document is not None:
            raise PeadAnnouncementEvidenceError(
                f"{label} unproved public observation must have null document"
            )
        normalized_observed_document = None
    else:
        if observed_basis != "contemporaneous_sec_http_observation":
            raise PeadAnnouncementEvidenceError(
                f"{label} public observation basis is unsupported"
            )
        if observed_document is None:
            raise PeadAnnouncementEvidenceError(
                f"{label} public observation requires preserved bytes"
            )
        normalized_observed_document = _validate_document(
            observed_document,
            expected_role="availability_observation",
            label=f"{label}.observed_public_document",
            sec_only=True,
        )
        if observed != normalized_observed_document["http"]["date_utc"]:
            raise PeadAnnouncementEvidenceError(
                f"{label} observed-public bound must equal SEC HTTP Date"
            )
        if (
            _utc_value(normalized_observed_document["retrieved_at_utc"])
            > created_dt
        ):
            raise PeadAnnouncementEvidenceError(
                f"{label} public-observation retrieval follows receipt creation"
            )
        delay = (_utc_value(observed) - accepted_dt).total_seconds()
        if not 0 <= delay <= MAX_CONTEMPORANEOUS_OBSERVATION_DELAY_SECONDS:
            raise PeadAnnouncementEvidenceError(
                f"{label} public observation is not contemporaneous with acceptance"
            )
        if _utc_value(observed) > created_dt:
            raise PeadAnnouncementEvidenceError(
                f"{label} public observation follows receipt creation"
            )
    if first_public is not None and observed is not None:
        if _utc_value(first_public) > _utc_value(observed):
            raise PeadAnnouncementEvidenceError(
                f"{label} first-public time follows observed-public bound"
            )

    actual = _validate_actual(
        record["canonical_actual"],
        event_key=event_key,
        label=f"{label}.canonical_actual",
    )
    extraction = _validate_extraction(
        record["extraction"],
        exhibit_document=exhibit_document,
        actual=actual,
        label=f"{label}.extraction",
    )
    return {
        "source_kind": SEC_SOURCE_KIND,
        "accession_number": accession,
        "form": "8-K",
        "item": "2.02",
        "exhibit": exhibit,
        "metadata_document": metadata,
        "exhibit_document": exhibit_document,
        "edgar_acceptance_at_utc": acceptance,
        "first_public_at_utc": first_public,
        "first_public_basis": first_basis,
        "first_public_proof": normalized_first_proof,
        "observed_public_by_at_utc": observed,
        "observed_public_by_basis": observed_basis,
        "observed_public_document": normalized_observed_document,
        "canonical_actual": actual,
        "extraction": extraction,
    }


def _validate_outcome(
    value: Any,
    *,
    expected_event: Mapping[str, Any],
    created_at: str,
    label: str,
) -> dict[str, Any]:
    outcome = _exact_fields(value, _OUTCOME_FIELDS, label)
    event_id = _sha256(outcome["event_id"], f"{label}.event_id")
    event_key = validate_event_key(outcome["event_key"], label=f"{label}.event_key")
    if event_id != canonical_event_id(event_key):
        raise PeadAnnouncementEvidenceError(f"{label} event_id differs from event_key")
    if event_id != expected_event["event_id"] or event_key != expected_event["event_key"]:
        raise PeadAnnouncementEvidenceError(f"{label} differs from frozen event universe")
    disposition = outcome["disposition"]
    if disposition == "missing":
        reason = _required_text(outcome["missing_reason"], f"{label}.missing_reason")
        if outcome["available_record"] is not None:
            raise PeadAnnouncementEvidenceError(
                f"{label} missing outcome cannot carry available evidence"
            )
        available = None
    elif disposition == "available":
        if outcome["missing_reason"] is not None:
            raise PeadAnnouncementEvidenceError(
                f"{label} available outcome cannot carry a missing reason"
            )
        reason = None
        available = _validate_available(
            outcome["available_record"],
            event_key=event_key,
            created_at=created_at,
            label=f"{label}.available_record",
        )
    else:
        raise PeadAnnouncementEvidenceError(f"{label}.disposition is invalid")
    return {
        "event_id": event_id,
        "event_key": event_key,
        "disposition": disposition,
        "missing_reason": reason,
        "available_record": available,
    }


def _coverage(
    outcomes: Sequence[Mapping[str, Any]], *, event_universe_qualified: bool
) -> dict[str, Any]:
    expected = len(outcomes)
    available = sum(item["disposition"] == "available" for item in outcomes)
    missing = expected - available
    timing = sum(
        item["disposition"] == "available"
        and item["available_record"]["first_public_at_utc"] is not None
        for item in outcomes
    )
    blockers: list[str] = []
    if not event_universe_qualified:
        blockers.append("event_universe_not_qualified")
    if missing:
        blockers.append("expected_events_missing_announcement")
    if timing != expected:
        blockers.append("first_public_timestamps_unproven")
    return {
        "expected_events": expected,
        "available_events": available,
        "timing_eligible_events": timing,
        "missing_events": missing,
        "event_universe_qualified": event_universe_qualified,
        "blockers": blockers,
        "complete": expected > 0 and not blockers,
    }


def build_missing_outcome(
    event_key: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Build one explicit non-available outcome without inventing source facts."""
    try:
        key = validate_event_key(event_key)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementEvidenceError("invalid missing event key") from exc
    return {
        "event_id": canonical_event_id(key),
        "event_key": key,
        "disposition": "missing",
        "missing_reason": _required_text(reason, "missing reason"),
        "available_record": None,
    }


def build_sec_available_outcome(
    *,
    event_key: Mapping[str, Any],
    accession_number: str,
    exhibit: str,
    metadata_url: str,
    metadata_retrieved_at_utc: str,
    metadata_http: Mapping[str, Any],
    metadata_bytes: bytes,
    exhibit_url: str,
    exhibit_retrieved_at_utc: str,
    exhibit_http: Mapping[str, Any],
    exhibit_bytes: bytes,
    edgar_acceptance_at_utc: str,
    canonical_actual: Mapping[str, Any],
    extraction: Mapping[str, Any],
    first_public_at_utc: str | None = None,
    first_public_proof: Mapping[str, Any] | None = None,
    observed_public: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one SEC actual receipt; no vendor value or time is accepted.

    ``observed_public`` may contain ``url``, ``retrieved_at_utc``, ``http``, and
    exact ``bytes``.  Its HTTP Date is only an observed-public upper bound.
    V1 reserves ``first_public_at_utc`` and ``first_public_proof`` but rejects
    every non-null value until an authoritative replay adapter exists.  A later
    retrieval is never promoted to first-public evidence.
    """
    try:
        key = validate_event_key(event_key)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementEvidenceError("invalid available event key") from exc
    metadata = _build_document(
        role="filing_metadata",
        url=metadata_url,
        retrieved_at_utc=metadata_retrieved_at_utc,
        http=metadata_http,
        raw=metadata_bytes,
    )
    exhibit_document = _build_document(
        role="earnings_exhibit",
        url=exhibit_url,
        retrieved_at_utc=exhibit_retrieved_at_utc,
        http=exhibit_http,
        raw=exhibit_bytes,
    )
    extraction_payload = dict(extraction)
    extraction_payload.setdefault(
        "source_document_sha256", exhibit_document["raw_document"]["sha256"]
    )
    if observed_public is None:
        observed_at = None
        observed_basis = "not_proven"
        observed_document = None
    else:
        observation = _exact_fields(
            observed_public,
            {"url", "retrieved_at_utc", "http", "bytes"},
            "observed_public",
        )
        observed_document = _build_document(
            role="availability_observation",
            url=observation["url"],
            retrieved_at_utc=observation["retrieved_at_utc"],
            http=observation["http"],
            raw=observation["bytes"],
        )
        observed_at = observed_document["http"]["date_utc"]
        observed_basis = "contemporaneous_sec_http_observation"
    if first_public_at_utc is None:
        first_basis = "not_proven"
        first_proof = None
    else:
        first_basis = "authoritative_sec_dissemination_timestamp"
        first_proof = first_public_proof
    available = {
        "source_kind": SEC_SOURCE_KIND,
        "accession_number": accession_number,
        "form": "8-K",
        "item": "2.02",
        "exhibit": exhibit,
        "metadata_document": metadata,
        "exhibit_document": exhibit_document,
        "edgar_acceptance_at_utc": edgar_acceptance_at_utc,
        "first_public_at_utc": first_public_at_utc,
        "first_public_basis": first_basis,
        "first_public_proof": first_proof,
        "observed_public_by_at_utc": observed_at,
        "observed_public_by_basis": observed_basis,
        "observed_public_document": observed_document,
        "canonical_actual": dict(canonical_actual),
        "extraction": extraction_payload,
    }
    return {
        "event_id": canonical_event_id(key),
        "event_key": key,
        "disposition": "available",
        "missing_reason": None,
        "available_record": available,
    }


def build_pead_announcement_evidence(
    *,
    expected_event_manifest: Mapping[str, Any],
    created_at_utc: str,
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an exhaustive announcement artifact against one frozen universe."""
    try:
        universe = validate_pead_event_universe(expected_event_manifest)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementEvidenceError("expected event manifest is invalid") from exc
    created = _utc_timestamp(created_at_utc, "created_at_utc")
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise PeadAnnouncementEvidenceError("outcomes must be a sequence")
    expected_by_id = {
        item["event_id"]: item for item in universe["payload"]["expected_events"]
    }
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise PeadAnnouncementEvidenceError(f"outcomes[{index}] must be an object")
        event_id = outcome.get("event_id")
        if not isinstance(event_id, str) or event_id not in expected_by_id:
            raise PeadAnnouncementEvidenceError(
                f"outcomes[{index}] references an unexpected event"
            )
        if event_id in raw_by_id:
            raise PeadAnnouncementEvidenceError("outcomes contains a duplicate event")
        raw_by_id[event_id] = outcome
    if set(raw_by_id) != set(expected_by_id):
        raise PeadAnnouncementEvidenceError(
            "outcomes must account for every frozen expected event exactly once"
        )
    normalized = [
        _validate_outcome(
            raw_by_id[event_id],
            expected_event=expected_by_id[event_id],
            created_at=created,
            label=f"outcomes[{index}]",
        )
        for index, event_id in enumerate(sorted(expected_by_id))
    ]
    payload = {
        "schema_version": ANNOUNCEMENT_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": universe["payload"]["candidate_id"],
        "created_at_utc": created,
        "expected_event_manifest_hash": universe["artifact_hash"],
        "expected_event_ids": list(universe["payload"]["expected_event_ids"]),
        "outcomes": normalized,
        "coverage": _coverage(
            normalized,
            event_universe_qualified=universe["payload"]["qualification_allowed"],
        ),
    }
    document = {"artifact_hash": content_hash(payload), "payload": payload}
    return validate_pead_announcement_evidence(
        document, expected_event_manifest=universe
    )


def validate_pead_announcement_evidence(
    document: Mapping[str, Any], *, expected_event_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate hashes, source bytes, chronology, and exhaustive event coverage."""
    try:
        universe = validate_pead_event_universe(expected_event_manifest)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementEvidenceError("expected event manifest is invalid") from exc
    wrapper = _exact_fields(document, _WRAPPER_FIELDS, "announcement evidence")
    payload = _exact_fields(
        wrapper["payload"], _PAYLOAD_FIELDS, "announcement evidence.payload"
    )
    claimed = _sha256(wrapper["artifact_hash"], "announcement evidence.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadAnnouncementEvidenceError("announcement evidence artifact hash mismatch")
    if payload["schema_version"] != ANNOUNCEMENT_EVIDENCE_SCHEMA_VERSION:
        raise PeadAnnouncementEvidenceError("unsupported announcement evidence schema")
    if payload["candidate_id"] != universe["payload"]["candidate_id"]:
        raise PeadAnnouncementEvidenceError("announcement evidence candidate differs")
    created = _utc_timestamp(payload["created_at_utc"], "created_at_utc")
    if _utc_value(created) < _utc_value(universe["payload"]["frozen_at_utc"]):
        raise PeadAnnouncementEvidenceError(
            "announcement evidence predates its frozen event universe"
        )
    if payload["expected_event_manifest_hash"] != universe["artifact_hash"]:
        raise PeadAnnouncementEvidenceError("announcement evidence binds another universe")
    if payload["expected_event_ids"] != universe["payload"]["expected_event_ids"]:
        raise PeadAnnouncementEvidenceError("expected event IDs differ from universe")
    raw_outcomes = payload["outcomes"]
    if not isinstance(raw_outcomes, list):
        raise PeadAnnouncementEvidenceError("outcomes must be an array")
    expected = universe["payload"]["expected_events"]
    if len(raw_outcomes) != len(expected):
        raise PeadAnnouncementEvidenceError(
            "outcomes must account for every expected event"
        )
    outcomes = [
        _validate_outcome(
            raw,
            expected_event=expected[index],
            created_at=created,
            label=f"outcomes[{index}]",
        )
        for index, raw in enumerate(raw_outcomes)
    ]
    ids = [item["event_id"] for item in outcomes]
    if ids != universe["payload"]["expected_event_ids"]:
        raise PeadAnnouncementEvidenceError(
            "outcomes are missing, duplicated, unexpected, or not canonically ordered"
        )
    coverage = _exact_fields(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    derived_coverage = _coverage(
        outcomes,
        event_universe_qualified=universe["payload"]["qualification_allowed"],
    )
    if dict(coverage) != derived_coverage:
        raise PeadAnnouncementEvidenceError("announcement coverage is not derived exactly")
    normalized_payload = {
        "schema_version": ANNOUNCEMENT_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": universe["payload"]["candidate_id"],
        "created_at_utc": created,
        "expected_event_manifest_hash": universe["artifact_hash"],
        "expected_event_ids": list(universe["payload"]["expected_event_ids"]),
        "outcomes": outcomes,
        "coverage": derived_coverage,
    }
    if content_hash(normalized_payload) != claimed:
        raise PeadAnnouncementEvidenceError("announcement evidence is not canonical")
    return {"artifact_hash": claimed, "payload": _plain_json(normalized_payload)}


def timing_eligible_announcements(
    document: Mapping[str, Any], *, expected_event_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return only source-derived actuals with proved first-public timestamps."""
    verified = validate_pead_announcement_evidence(
        document, expected_event_manifest=expected_event_manifest
    )
    result: list[dict[str, Any]] = []
    for outcome in verified["payload"]["outcomes"]:
        record = outcome["available_record"]
        if outcome["disposition"] != "available" or record["first_public_at_utc"] is None:
            continue
        result.append(
            {
                "event_id": outcome["event_id"],
                "event_key": outcome["event_key"],
                "canonical_actual": record["canonical_actual"],
                "first_public_at_utc": record["first_public_at_utc"],
                "announcement_evidence_artifact_hash": verified["artifact_hash"],
            }
        )
    return result


def _strict_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadAnnouncementEvidenceError(
            f"announcement evidence is not a regular file: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_ANNOUNCEMENT_EVIDENCE_BYTES:
        raise PeadAnnouncementEvidenceError("announcement evidence exceeds size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadAnnouncementEvidenceError("announcement evidence is not UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadAnnouncementEvidenceError(
                    f"announcement evidence contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise PeadAnnouncementEvidenceError(
            f"announcement evidence contains invalid number {token}"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise PeadAnnouncementEvidenceError("invalid announcement evidence JSON") from exc
    if not isinstance(value, dict):
        raise PeadAnnouncementEvidenceError("announcement evidence root must be an object")
    return value


def load_pead_announcement_evidence(
    path: str | Path, *, expected_event_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Load strict JSON and revalidate it against the same frozen universe."""
    return validate_pead_announcement_evidence(
        _strict_json_file(Path(path)),
        expected_event_manifest=expected_event_manifest,
    )


__all__ = [
    "ANNOUNCEMENT_EVIDENCE_SCHEMA_VERSION",
    "MAX_ANNOUNCEMENT_EVIDENCE_BYTES",
    "MAX_SOURCE_DOCUMENT_BYTES",
    "PeadAnnouncementEvidenceError",
    "SEC_SOURCE_KIND",
    "build_missing_outcome",
    "build_pead_announcement_evidence",
    "build_sec_available_outcome",
    "load_pead_announcement_evidence",
    "timing_eligible_announcements",
    "validate_pead_announcement_evidence",
]
