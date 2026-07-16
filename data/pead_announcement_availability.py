"""Replayable, conservative announcement-availability evidence for PEAD.

This boundary never claims to know when an announcement was first public.  It
proves only a conservative upper bound: by ``known_public_by_at_utc`` the exact
announcement bytes bound by :mod:`data.pead_announcement_evidence` were public
through one supported channel.

Two adapters are closed in this version:

* ``licensed_release_distribution.v1`` replays a provider-attested historical
  distribution record under separately trusted provider-manifest and exact
  provider-record hashes; and
* ``sec_https_positive_observation.v1`` replays contemporaneous positive SEC
  responses and an externally trusted append-only checkpoint.  It is
  prospective-only and cannot reconstruct historical availability.

The trusted manifest, record, and checkpoint hashes are verifier inputs, not
assertions inside the artifact.  Embedding self-declared trust material is
therefore not enough to make a claim eligible.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import base64
import binascii
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

from data.pead_announcement_evidence import (
    PeadAnnouncementEvidenceError,
    validate_pead_announcement_evidence,
)
from data.pead_event_universe import (
    PeadEventUniverseError,
    canonical_event_id,
    canonical_json,
    content_hash,
    validate_event_key,
    validate_pead_event_universe,
)


ANNOUNCEMENT_AVAILABILITY_SCHEMA_VERSION = "pead_announcement_availability.v1"
LICENSED_RELEASE_ADAPTER = "licensed_release_distribution.v1"
SEC_HTTPS_OBSERVATION_ADAPTER = "sec_https_positive_observation.v1"
LICENSED_MANIFEST_SCHEMA_VERSION = "licensed_release_distribution_manifest.v1"
LICENSED_RECORD_SCHEMA_VERSION = "licensed_release_distribution_record.v1"
CHECKPOINT_SCHEMA_VERSION = "external_append_only_checkpoint_receipt.v1"
TRUST_ROOT_SET_SCHEMA_VERSION = "pead_sha256_trust_root_set.v1"

MAX_AVAILABILITY_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_RAW_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SEC_HTTP_CLOCK_SKEW_SECONDS = 10 * 60
MAX_CHECKPOINT_DELAY_SECONDS = 10 * 60
MAX_SEC_OBSERVATION_DELAY_SECONDS = 60 * 60

_HEX = frozenset("0123456789abcdef")
_ACCESSION = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
_SEC_HOSTS = frozenset({"sec.gov", "www.sec.gov", "data.sec.gov"})
_EVIDENCE_CLASSES = frozenset({"historical_reconstruction", "prospective"})

_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "created_at_utc",
    "expected_event_manifest_sha256",
    "announcement_evidence_sha256",
    "trust_policy",
    "expected_event_ids",
    "outcomes",
    "coverage",
}
_TRUST_POLICY_FIELDS = {
    "provider_manifest_allowlist_sha256",
    "provider_record_allowlist_sha256",
    "checkpoint_allowlist_sha256",
}
_OUTCOME_FIELDS = {
    "event_id",
    "event_key",
    "disposition",
    "missing_reason",
    "claim",
}
_CLAIM_FIELDS = {
    "claim_kind",
    "known_public_by_at_utc",
    "adapter_id",
    "evidence",
    "eligibility",
}
_ELIGIBILITY_FIELDS = {
    "claim_semantics",
    "first_public_proven",
    "eligible_for_declared_evidence_class",
    "historical_reconstruction_allowed",
    "prospective_observation_allowed",
    "consensus_cutoff_rule",
    "market_cutoff_rule",
    "same_day_consensus_allowed",
    "same_day_market_close_allowed",
}
_COVERAGE_FIELDS = {
    "expected_events",
    "available_claims",
    "eligible_claims",
    "missing_claims",
    "event_universe_qualified",
    "announcement_actuals_complete",
    "blockers",
    "complete",
}
_RAW_FIELDS = {"sha256", "bytes", "base64"}
_LICENSED_EVIDENCE_FIELDS = {
    "provider_url",
    "retrieved_at_utc",
    "provider_manifest",
    "provider_record",
}
_LICENSED_MANIFEST_FIELDS = {
    "schema_version",
    "provider_id",
    "dataset_id",
    "license_evidence_sha256",
    "record_schema_version",
    "distribution_timestamp_field",
    "distribution_timestamp_semantics",
    "timestamp_precision",
    "event_identity_fields",
    "actual_binding_fields",
    "source_document_binding",
}
_LICENSED_RECORD_FIELDS = {
    "schema_version",
    "provider_id",
    "dataset_id",
    "provider_record_id",
    "cik",
    "accession_number",
    "event_id",
    "event_key",
    "distribution_at_utc",
    "source_document_sha256",
    "canonical_actual_sha256",
    "canonical_actual",
}
_EVENT_IDENTITY_FIELDS = ["cik", "accession_number", "event_id", "event_key"]
_ACTUAL_BINDING_FIELDS = [
    "canonical_value",
    "metric",
    "metric_definition_sha256",
    "accounting_basis",
    "per_share_basis",
    "scope",
    "currency",
    "unit",
    "canonical_share_basis",
    "fiscal_period_end",
    "fiscal_period_type",
    "normalization_evidence_sha256",
]
_SEC_EVIDENCE_FIELDS = {
    "metadata_observation",
    "exhibit_observation",
    "checkpoint_receipt",
}
_HTTP_OBSERVATION_FIELDS = {
    "role",
    "url",
    "received_at_utc",
    "raw_status_and_headers",
    "raw_body",
}
_CHECKPOINT_FIELDS = {
    "schema_version",
    "authority_id",
    "authority_manifest_sha256",
    "log_id",
    "sequence",
    "previous_checkpoint_sha256",
    "observed_bundle_sha256",
    "recorded_at_utc",
}


class PeadAnnouncementAvailabilityError(ValueError):
    """Announcement availability is malformed, untrusted, or inconsistent."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadAnnouncementAvailabilityError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadAnnouncementAvailabilityError(f"{label} must be nonempty canonical text")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadAnnouncementAvailabilityError(f"{label} must be a lowercase SHA-256")
    return value


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadAnnouncementAvailabilityError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} must be canonical UTC with Z") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PeadAnnouncementAvailabilityError(f"{label} must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if utc.microsecond else "seconds"
    canonical = utc.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadAnnouncementAvailabilityError(f"{label} must be canonical UTC with Z")
    return value


def _utc_value(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _event_key(value: Any, label: str) -> dict[str, str]:
    try:
        return validate_event_key(value, label=label)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} is invalid") from exc


def _https_url(value: Any, label: str, *, sec_only: bool) -> str:
    text = _required_text(value, label)
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} must be HTTPS") from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise PeadAnnouncementAvailabilityError(f"{label} must be a canonical HTTPS URL")
    if sec_only and parsed.hostname.lower() not in _SEC_HOSTS:
        raise PeadAnnouncementAvailabilityError(f"{label} must be an SEC URL")
    return text


def _canonical_base64(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise PeadAnnouncementAvailabilityError(f"{label} must be canonical base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} must be canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != value:
        raise PeadAnnouncementAvailabilityError(f"{label} must be canonical base64")
    if not raw or len(raw) > MAX_RAW_SOURCE_BYTES:
        raise PeadAnnouncementAvailabilityError(f"{label} decoded size is invalid")
    return raw


def _raw_wrapper(raw: bytes, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_RAW_SOURCE_BYTES:
        raise PeadAnnouncementAvailabilityError(f"{label} must be nonempty bounded bytes")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _validate_raw(value: Any, label: str) -> tuple[dict[str, Any], bytes]:
    wrapper = _exact(value, _RAW_FIELDS, label)
    claimed = _sha256(wrapper["sha256"], f"{label}.sha256")
    count = wrapper["bytes"]
    if type(count) is not int or not 0 < count <= MAX_RAW_SOURCE_BYTES:
        raise PeadAnnouncementAvailabilityError(f"{label}.bytes must be a positive bounded integer")
    raw = _canonical_base64(wrapper["base64"], f"{label}.base64")
    if len(raw) != count or hashlib.sha256(raw).hexdigest() != claimed:
        raise PeadAnnouncementAvailabilityError(f"{label} bytes/hash mismatch")
    return {
        "sha256": claimed,
        "bytes": count,
        "base64": wrapper["base64"],
    }, raw


def _strict_json_bytes(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} must be UTF-8 JSON") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PeadAnnouncementAvailabilityError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    def invalid(token: str) -> None:
        raise PeadAnnouncementAvailabilityError(f"{label} contains invalid number {token}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    except PeadAnnouncementAvailabilityError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise PeadAnnouncementAvailabilityError(f"{label} root must be an object")
    return value


def _trusted_hashes(values: Collection[str], label: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise PeadAnnouncementAvailabilityError(f"{label} must be a collection")
    return frozenset(_sha256(value, f"{label} entry") for value in values)


def _trust_set_hash(values: frozenset[str]) -> str:
    return content_hash(
        {
            "schema_version": TRUST_ROOT_SET_SCHEMA_VERSION,
            "members": sorted(values),
        }
    )


def _trust_policy(
    *,
    provider_manifests: frozenset[str],
    provider_records: frozenset[str],
    checkpoints: frozenset[str],
) -> dict[str, str]:
    return {
        "provider_manifest_allowlist_sha256": _trust_set_hash(provider_manifests),
        "provider_record_allowlist_sha256": _trust_set_hash(provider_records),
        "checkpoint_allowlist_sha256": _trust_set_hash(checkpoints),
    }


def _announcement_context(
    *,
    expected_event: Mapping[str, Any],
    announcement_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    if announcement_outcome["event_id"] != expected_event["event_id"]:
        raise PeadAnnouncementAvailabilityError("announcement outcome differs from expected event")
    if announcement_outcome["disposition"] != "available":
        raise PeadAnnouncementAvailabilityError(
            "availability claim requires an available independent announcement actual"
        )
    record = announcement_outcome["available_record"]
    return {
        "event_id": expected_event["event_id"],
        "event_key": expected_event["event_key"],
        "cik": expected_event["event_key"]["cik"],
        "accession_number": record["accession_number"],
        "edgar_acceptance_at_utc": record["edgar_acceptance_at_utc"],
        "metadata_url": record["metadata_document"]["url"],
        "metadata_sha256": record["metadata_document"]["raw_document"]["sha256"],
        "exhibit_url": record["exhibit_document"]["url"],
        "exhibit_sha256": record["exhibit_document"]["raw_document"]["sha256"],
        "canonical_actual": record["canonical_actual"],
        "canonical_actual_sha256": content_hash(record["canonical_actual"]),
    }


def _eligibility(adapter_id: str, evidence_class: str) -> dict[str, Any]:
    historical = (
        adapter_id == LICENSED_RELEASE_ADAPTER and evidence_class == "historical_reconstruction"
    )
    prospective = adapter_id == SEC_HTTPS_OBSERVATION_ADAPTER and evidence_class == "prospective"
    return {
        "claim_semantics": "conservative_known_public_by_upper_bound",
        "first_public_proven": False,
        "eligible_for_declared_evidence_class": historical or prospective,
        "historical_reconstruction_allowed": historical,
        "prospective_observation_allowed": prospective,
        "consensus_cutoff_rule": "strict_prior_eastern_calendar_date",
        "market_cutoff_rule": "strict_prior_nyse_session",
        "same_day_consensus_allowed": False,
        "same_day_market_close_allowed": False,
    }


def _timestamp_precision(value: str) -> str:
    return "microsecond" if "." in value else "second"


def _validate_licensed_manifest(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    manifest = _exact(value, _LICENSED_MANIFEST_FIELDS, label)
    if manifest["schema_version"] != LICENSED_MANIFEST_SCHEMA_VERSION:
        raise PeadAnnouncementAvailabilityError(f"{label}.schema_version is unsupported")
    provider_id = _required_text(manifest["provider_id"], f"{label}.provider_id")
    dataset_id = _required_text(manifest["dataset_id"], f"{label}.dataset_id")
    license_hash = _sha256(manifest["license_evidence_sha256"], f"{label}.license_evidence_sha256")
    if manifest["record_schema_version"] != LICENSED_RECORD_SCHEMA_VERSION:
        raise PeadAnnouncementAvailabilityError(
            f"{label} does not declare the supported raw record schema"
        )
    if manifest["distribution_timestamp_field"] != "distribution_at_utc":
        raise PeadAnnouncementAvailabilityError(
            f"{label} distribution timestamp field is not exact"
        )
    if manifest["distribution_timestamp_semantics"] != "provider_attested_public_distribution_time":
        raise PeadAnnouncementAvailabilityError(f"{label} does not attest public distribution time")
    if manifest["timestamp_precision"] not in {"second", "microsecond"}:
        raise PeadAnnouncementAvailabilityError(f"{label}.timestamp_precision is unsupported")
    if manifest["event_identity_fields"] != _EVENT_IDENTITY_FIELDS:
        raise PeadAnnouncementAvailabilityError(f"{label} event identity semantics differ")
    if manifest["actual_binding_fields"] != _ACTUAL_BINDING_FIELDS:
        raise PeadAnnouncementAvailabilityError(f"{label} metric/actual semantics differ")
    if manifest["source_document_binding"] != "sha256_exact_bytes":
        raise PeadAnnouncementAvailabilityError(
            f"{label} source document binding is not exact bytes"
        )
    return {
        "schema_version": LICENSED_MANIFEST_SCHEMA_VERSION,
        "provider_id": provider_id,
        "dataset_id": dataset_id,
        "license_evidence_sha256": license_hash,
        "record_schema_version": LICENSED_RECORD_SCHEMA_VERSION,
        "distribution_timestamp_field": "distribution_at_utc",
        "distribution_timestamp_semantics": ("provider_attested_public_distribution_time"),
        "timestamp_precision": manifest["timestamp_precision"],
        "event_identity_fields": list(_EVENT_IDENTITY_FIELDS),
        "actual_binding_fields": list(_ACTUAL_BINDING_FIELDS),
        "source_document_binding": "sha256_exact_bytes",
    }


def _validate_licensed_evidence(
    value: Any,
    *,
    context: Mapping[str, Any],
    evidence_class: str,
    created_at: str,
    trusted_manifest_hashes: frozenset[str],
    trusted_record_hashes: frozenset[str],
    label: str,
) -> tuple[dict[str, Any], str]:
    evidence = _exact(value, _LICENSED_EVIDENCE_FIELDS, label)
    provider_url = _https_url(evidence["provider_url"], f"{label}.provider_url", sec_only=False)
    retrieved = _utc_timestamp(evidence["retrieved_at_utc"], f"{label}.retrieved_at_utc")
    if _utc_value(retrieved) > _utc_value(created_at):
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider retrieval follows artifact creation"
        )
    manifest_wrapper, manifest_bytes = _validate_raw(
        evidence["provider_manifest"], f"{label}.provider_manifest"
    )
    if manifest_wrapper["sha256"] not in trusted_manifest_hashes:
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider manifest is not in the external trust registry"
        )
    manifest = _validate_licensed_manifest(
        _strict_json_bytes(manifest_bytes, f"{label}.provider_manifest raw bytes"),
        label=f"{label}.provider_manifest payload",
    )
    record_wrapper, record_bytes = _validate_raw(
        evidence["provider_record"], f"{label}.provider_record"
    )
    if record_wrapper["sha256"] not in trusted_record_hashes:
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider record is not in the external trust registry"
        )
    record = _exact(
        _strict_json_bytes(record_bytes, f"{label}.provider_record raw bytes"),
        _LICENSED_RECORD_FIELDS,
        f"{label}.provider_record payload",
    )
    if record["schema_version"] != LICENSED_RECORD_SCHEMA_VERSION:
        raise PeadAnnouncementAvailabilityError(f"{label} provider record schema is unsupported")
    for field in ("provider_id", "dataset_id"):
        if record[field] != manifest[field]:
            raise PeadAnnouncementAvailabilityError(
                f"{label} provider record {field} differs from trusted manifest"
            )
    _required_text(record["provider_record_id"], f"{label}.provider_record_id")
    if record["cik"] != context["cik"]:
        raise PeadAnnouncementAvailabilityError(f"{label} provider record CIK differs from event")
    accession = record["accession_number"]
    if (
        not isinstance(accession, str)
        or _ACCESSION.fullmatch(accession) is None
        or accession != context["accession_number"]
    ):
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider record accession differs from announcement"
        )
    event_key = _event_key(record["event_key"], f"{label}.provider_record.event_key")
    if record["event_id"] != context["event_id"] or event_key != context["event_key"]:
        raise PeadAnnouncementAvailabilityError(f"{label} provider record event identity differs")
    distributed = _utc_timestamp(record["distribution_at_utc"], f"{label}.distribution_at_utc")
    if _timestamp_precision(distributed) != manifest["timestamp_precision"]:
        raise PeadAnnouncementAvailabilityError(
            f"{label} distribution timestamp precision differs from manifest"
        )
    if _utc_value(distributed) > _utc_value(retrieved):
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider distribution follows archive retrieval"
        )
    if record["source_document_sha256"] != context["exhibit_sha256"]:
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider record does not bind exact announcement bytes"
        )
    if record["canonical_actual_sha256"] != context["canonical_actual_sha256"]:
        raise PeadAnnouncementAvailabilityError(
            f"{label} provider record canonical metric/actual digest differs"
        )
    if record["canonical_actual"] != context["canonical_actual"]:
        raise PeadAnnouncementAvailabilityError(f"{label} provider metric/actual semantics differ")
    normalized = {
        "provider_url": provider_url,
        "retrieved_at_utc": retrieved,
        "provider_manifest": manifest_wrapper,
        "provider_record": record_wrapper,
    }
    if evidence_class not in _EVIDENCE_CLASSES:  # defensive; checked at payload
        raise PeadAnnouncementAvailabilityError("unsupported evidence class")
    return normalized, distributed


def _parse_http_headers(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PeadAnnouncementAvailabilityError(
            f"{label} must be exact ASCII HTTP status/header bytes"
        ) from exc
    if not text.endswith("\r\n\r\n") or "\n" in text.replace("\r\n", ""):
        raise PeadAnnouncementAvailabilityError(f"{label} must use exact CRLF HTTP framing")
    lines = text[:-4].split("\r\n")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        raise PeadAnnouncementAvailabilityError(f"{label} must prove an HTTP/1.1 200 response")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line or line.startswith((" ", "\t")):
            raise PeadAnnouncementAvailabilityError(f"{label} header framing is invalid")
        name, value = line.split(":", 1)
        lowered = name.lower()
        if not name or not value.startswith(" ") or lowered in headers:
            raise PeadAnnouncementAvailabilityError(
                f"{label} contains an invalid or duplicate header"
            )
        headers[lowered] = value[1:]
    if "date" not in headers or "content-type" not in headers:
        raise PeadAnnouncementAvailabilityError(f"{label} requires Date and Content-Type headers")
    try:
        parsed_date = parsedate_to_datetime(headers["date"])
    except (TypeError, ValueError) as exc:
        raise PeadAnnouncementAvailabilityError(f"{label} Date header is invalid") from exc
    if parsed_date.tzinfo is None or parsed_date.utcoffset() is None:
        raise PeadAnnouncementAvailabilityError(f"{label} Date header must be timezone-aware")
    utc = parsed_date.astimezone(timezone.utc)
    if headers["date"] != format_datetime(utc, usegmt=True):
        raise PeadAnnouncementAvailabilityError(
            f"{label} Date header must be canonical IMF-fixdate"
        )
    date_utc = utc.isoformat(timespec="seconds").replace("+00:00", "Z")
    content_length: int | None = None
    if "content-length" in headers:
        if not headers["content-length"].isdigit():
            raise PeadAnnouncementAvailabilityError(f"{label} Content-Length is invalid")
        content_length = int(headers["content-length"])
    return {
        "date_utc": date_utc,
        "content_type": _required_text(headers["content-type"], f"{label} Content-Type"),
        "content_length": content_length,
    }


def _validate_http_observation(
    value: Any,
    *,
    expected_role: str,
    expected_url: str,
    expected_body_sha256: str,
    created_at: str,
    label: str,
) -> tuple[dict[str, Any], datetime]:
    observation = _exact(value, _HTTP_OBSERVATION_FIELDS, label)
    if observation["role"] != expected_role:
        raise PeadAnnouncementAvailabilityError(f"{label}.role is invalid")
    url = _https_url(observation["url"], f"{label}.url", sec_only=True)
    if url != expected_url:
        raise PeadAnnouncementAvailabilityError(
            f"{label} URL differs from independently replayed announcement"
        )
    received = _utc_timestamp(observation["received_at_utc"], f"{label}.received_at_utc")
    received_dt = _utc_value(received)
    if received_dt > _utc_value(created_at):
        raise PeadAnnouncementAvailabilityError(f"{label} receipt follows artifact creation")
    headers_wrapper, headers_bytes = _validate_raw(
        observation["raw_status_and_headers"], f"{label}.raw_status_and_headers"
    )
    parsed_headers = _parse_http_headers(headers_bytes, f"{label} raw headers")
    server_dt = _utc_value(parsed_headers["date_utc"])
    skew = (received_dt - server_dt).total_seconds()
    if not 0 <= skew <= MAX_SEC_HTTP_CLOCK_SKEW_SECONDS:
        raise PeadAnnouncementAvailabilityError(
            f"{label} SEC HTTP Date/receipt chronology is invalid"
        )
    body_wrapper, body_bytes = _validate_raw(observation["raw_body"], f"{label}.raw_body")
    if body_wrapper["sha256"] != expected_body_sha256:
        raise PeadAnnouncementAvailabilityError(
            f"{label} body differs from independently replayed announcement bytes"
        )
    if parsed_headers["content_length"] is not None and parsed_headers["content_length"] != len(
        body_bytes
    ):
        raise PeadAnnouncementAvailabilityError(
            f"{label} Content-Length differs from preserved body"
        )
    return {
        "role": expected_role,
        "url": url,
        "received_at_utc": received,
        "raw_status_and_headers": headers_wrapper,
        "raw_body": body_wrapper,
    }, received_dt


def _validate_checkpoint(
    value: Any,
    *,
    observed_bundle_sha256: str,
    observation_received_at: datetime,
    created_at: str,
    trusted_checkpoint_hashes: frozenset[str],
    label: str,
) -> tuple[dict[str, Any], str]:
    wrapper, raw = _validate_raw(value, label)
    if wrapper["sha256"] not in trusted_checkpoint_hashes:
        raise PeadAnnouncementAvailabilityError(
            f"{label} is not in the external checkpoint trust registry"
        )
    receipt = _exact(
        _strict_json_bytes(raw, f"{label} raw bytes"),
        _CHECKPOINT_FIELDS,
        f"{label} payload",
    )
    if receipt["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise PeadAnnouncementAvailabilityError(f"{label} schema version is unsupported")
    _required_text(receipt["authority_id"], f"{label}.authority_id")
    _sha256(
        receipt["authority_manifest_sha256"],
        f"{label}.authority_manifest_sha256",
    )
    _required_text(receipt["log_id"], f"{label}.log_id")
    sequence = receipt["sequence"]
    if type(sequence) is not int or sequence < 0:
        raise PeadAnnouncementAvailabilityError(f"{label}.sequence must be a nonnegative integer")
    previous = receipt["previous_checkpoint_sha256"]
    if previous is not None:
        _sha256(previous, f"{label}.previous_checkpoint_sha256")
    if receipt["observed_bundle_sha256"] != observed_bundle_sha256:
        raise PeadAnnouncementAvailabilityError(
            f"{label} does not bind the exact positive SEC observations"
        )
    recorded = _utc_timestamp(receipt["recorded_at_utc"], f"{label}.recorded_at_utc")
    recorded_dt = _utc_value(recorded)
    delay = (recorded_dt - observation_received_at).total_seconds()
    if not 0 <= delay <= MAX_CHECKPOINT_DELAY_SECONDS:
        raise PeadAnnouncementAvailabilityError(
            f"{label} was not recorded promptly after positive observation"
        )
    if recorded_dt > _utc_value(created_at):
        raise PeadAnnouncementAvailabilityError(f"{label} follows artifact creation")
    return wrapper, recorded


def _validate_sec_evidence(
    value: Any,
    *,
    context: Mapping[str, Any],
    evidence_class: str,
    created_at: str,
    trusted_checkpoint_hashes: frozenset[str],
    label: str,
) -> tuple[dict[str, Any], str]:
    if evidence_class != "prospective":
        raise PeadAnnouncementAvailabilityError(
            f"{label} SEC HTTPS observation is prospective-only and cannot reconstruct historical timing"
        )
    evidence = _exact(value, _SEC_EVIDENCE_FIELDS, label)
    metadata, metadata_received = _validate_http_observation(
        evidence["metadata_observation"],
        expected_role="filing_metadata_positive_observation",
        expected_url=context["metadata_url"],
        expected_body_sha256=context["metadata_sha256"],
        created_at=created_at,
        label=f"{label}.metadata_observation",
    )
    exhibit, exhibit_received = _validate_http_observation(
        evidence["exhibit_observation"],
        expected_role="earnings_exhibit_positive_observation",
        expected_url=context["exhibit_url"],
        expected_body_sha256=context["exhibit_sha256"],
        created_at=created_at,
        label=f"{label}.exhibit_observation",
    )
    accepted = _utc_value(context["edgar_acceptance_at_utc"])
    for received in (metadata_received, exhibit_received):
        delay = (received - accepted).total_seconds()
        if not 0 <= delay <= MAX_SEC_OBSERVATION_DELAY_SECONDS:
            raise PeadAnnouncementAvailabilityError(
                f"{label} SEC positive observation is not contemporaneous with EDGAR acceptance"
            )
    compact_accession = context["accession_number"].replace("-", "")
    for name, observation in (("metadata", metadata), ("exhibit", exhibit)):
        if f"/{compact_accession}/" not in urlsplit(observation["url"]).path:
            raise PeadAnnouncementAvailabilityError(
                f"{label} {name} observation URL does not bind the accession"
            )
    bundle_hash = content_hash({"metadata_observation": metadata, "exhibit_observation": exhibit})
    checkpoint, known_public_by = _validate_checkpoint(
        evidence["checkpoint_receipt"],
        observed_bundle_sha256=bundle_hash,
        observation_received_at=max(metadata_received, exhibit_received),
        created_at=created_at,
        trusted_checkpoint_hashes=trusted_checkpoint_hashes,
        label=f"{label}.checkpoint_receipt",
    )
    return {
        "metadata_observation": metadata,
        "exhibit_observation": exhibit,
        "checkpoint_receipt": checkpoint,
    }, known_public_by


def _validate_claim(
    value: Any,
    *,
    expected_event: Mapping[str, Any],
    announcement_outcome: Mapping[str, Any],
    evidence_class: str,
    created_at: str,
    trusted_provider_manifest_hashes: frozenset[str],
    trusted_provider_record_hashes: frozenset[str],
    trusted_checkpoint_hashes: frozenset[str],
    label: str,
) -> dict[str, Any]:
    claim = _exact(value, _CLAIM_FIELDS, label)
    if claim["claim_kind"] != "known_public_by":
        raise PeadAnnouncementAvailabilityError(
            f"{label} may claim only known_public_by, never first_public"
        )
    context = _announcement_context(
        expected_event=expected_event,
        announcement_outcome=announcement_outcome,
    )
    adapter_id = claim["adapter_id"]
    if adapter_id == LICENSED_RELEASE_ADAPTER:
        evidence, derived_known_by = _validate_licensed_evidence(
            claim["evidence"],
            context=context,
            evidence_class=evidence_class,
            created_at=created_at,
            trusted_manifest_hashes=trusted_provider_manifest_hashes,
            trusted_record_hashes=trusted_provider_record_hashes,
            label=f"{label}.evidence",
        )
    elif adapter_id == SEC_HTTPS_OBSERVATION_ADAPTER:
        evidence, derived_known_by = _validate_sec_evidence(
            claim["evidence"],
            context=context,
            evidence_class=evidence_class,
            created_at=created_at,
            trusted_checkpoint_hashes=trusted_checkpoint_hashes,
            label=f"{label}.evidence",
        )
    else:
        raise PeadAnnouncementAvailabilityError(
            f"{label}.adapter_id is not a closed supported adapter"
        )
    claimed_known_by = _utc_timestamp(
        claim["known_public_by_at_utc"], f"{label}.known_public_by_at_utc"
    )
    if claimed_known_by != derived_known_by:
        raise PeadAnnouncementAvailabilityError(
            f"{label} known-public bound differs from replayed source evidence"
        )
    derived_eligibility = _eligibility(adapter_id, evidence_class)
    eligibility = _exact(claim["eligibility"], _ELIGIBILITY_FIELDS, f"{label}.eligibility")
    if dict(eligibility) != derived_eligibility:
        raise PeadAnnouncementAvailabilityError(
            f"{label} eligibility metadata is not derived exactly"
        )
    if not derived_eligibility["eligible_for_declared_evidence_class"]:
        raise PeadAnnouncementAvailabilityError(
            f"{label} adapter is ineligible for declared evidence class"
        )
    return {
        "claim_kind": "known_public_by",
        "known_public_by_at_utc": derived_known_by,
        "adapter_id": adapter_id,
        "evidence": evidence,
        "eligibility": derived_eligibility,
    }


def _validate_outcome(
    value: Any,
    *,
    expected_event: Mapping[str, Any],
    announcement_outcome: Mapping[str, Any],
    evidence_class: str,
    created_at: str,
    trusted_provider_manifest_hashes: frozenset[str],
    trusted_provider_record_hashes: frozenset[str],
    trusted_checkpoint_hashes: frozenset[str],
    label: str,
) -> dict[str, Any]:
    outcome = _exact(value, _OUTCOME_FIELDS, label)
    event_key = _event_key(outcome["event_key"], f"{label}.event_key")
    if (
        outcome["event_id"] != expected_event["event_id"]
        or event_key != expected_event["event_key"]
        or outcome["event_id"] != canonical_event_id(event_key)
    ):
        raise PeadAnnouncementAvailabilityError(f"{label} differs from frozen expected event")
    if outcome["disposition"] == "missing":
        reason = _required_text(outcome["missing_reason"], f"{label}.missing_reason")
        if outcome["claim"] is not None:
            raise PeadAnnouncementAvailabilityError(f"{label} missing outcome cannot carry a claim")
        claim = None
    elif outcome["disposition"] == "available":
        if outcome["missing_reason"] is not None:
            raise PeadAnnouncementAvailabilityError(
                f"{label} available outcome cannot carry a missing reason"
            )
        reason = None
        claim = _validate_claim(
            outcome["claim"],
            expected_event=expected_event,
            announcement_outcome=announcement_outcome,
            evidence_class=evidence_class,
            created_at=created_at,
            trusted_provider_manifest_hashes=trusted_provider_manifest_hashes,
            trusted_provider_record_hashes=trusted_provider_record_hashes,
            trusted_checkpoint_hashes=trusted_checkpoint_hashes,
            label=f"{label}.claim",
        )
    else:
        raise PeadAnnouncementAvailabilityError(f"{label}.disposition is invalid")
    return {
        "event_id": expected_event["event_id"],
        "event_key": expected_event["event_key"],
        "disposition": outcome["disposition"],
        "missing_reason": reason,
        "claim": claim,
    }


def _coverage(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    event_universe_qualified: bool,
    announcement_actuals_complete: bool,
) -> dict[str, Any]:
    expected = len(outcomes)
    available = sum(outcome["disposition"] == "available" for outcome in outcomes)
    eligible = sum(
        outcome["disposition"] == "available"
        and outcome["claim"]["eligibility"]["eligible_for_declared_evidence_class"]
        for outcome in outcomes
    )
    missing = expected - available
    blockers: list[str] = []
    if not event_universe_qualified:
        blockers.append("event_universe_not_qualified")
    if not announcement_actuals_complete:
        blockers.append("announcement_actuals_incomplete")
    if missing:
        blockers.append("expected_events_missing_availability")
    return {
        "expected_events": expected,
        "available_claims": available,
        "eligible_claims": eligible,
        "missing_claims": missing,
        "event_universe_qualified": event_universe_qualified,
        "announcement_actuals_complete": announcement_actuals_complete,
        "blockers": blockers,
        "complete": expected > 0 and eligible == expected and not blockers,
    }


def build_missing_availability_outcome(
    event_key: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    """Build one explicit missing availability disposition."""
    try:
        key = validate_event_key(event_key)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementAvailabilityError("invalid missing event key") from exc
    return {
        "event_id": canonical_event_id(key),
        "event_key": key,
        "disposition": "missing",
        "missing_reason": _required_text(reason, "missing reason"),
        "claim": None,
    }


def build_licensed_distribution_outcome(
    *,
    event_key: Mapping[str, Any],
    provider_url: str,
    retrieved_at_utc: str,
    provider_manifest_bytes: bytes,
    provider_record_bytes: bytes,
) -> dict[str, Any]:
    """Build a raw licensed-distribution outcome for later source replay."""
    try:
        key = validate_event_key(event_key)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementAvailabilityError("invalid licensed event key") from exc
    record = _exact(
        _strict_json_bytes(provider_record_bytes, "provider record bytes"),
        _LICENSED_RECORD_FIELDS,
        "provider record",
    )
    distributed = _utc_timestamp(record["distribution_at_utc"], "distribution_at_utc")
    return {
        "event_id": canonical_event_id(key),
        "event_key": key,
        "disposition": "available",
        "missing_reason": None,
        "claim": {
            "claim_kind": "known_public_by",
            "known_public_by_at_utc": distributed,
            "adapter_id": LICENSED_RELEASE_ADAPTER,
            "evidence": {
                "provider_url": provider_url,
                "retrieved_at_utc": retrieved_at_utc,
                "provider_manifest": _raw_wrapper(
                    provider_manifest_bytes, "provider manifest bytes"
                ),
                "provider_record": _raw_wrapper(provider_record_bytes, "provider record bytes"),
            },
            "eligibility": {},
        },
    }


def _http_observation(
    *,
    role: str,
    url: str,
    received_at_utc: str,
    headers_bytes: bytes,
    body_bytes: bytes,
) -> dict[str, Any]:
    return {
        "role": role,
        "url": url,
        "received_at_utc": received_at_utc,
        "raw_status_and_headers": _raw_wrapper(headers_bytes, f"{role} headers"),
        "raw_body": _raw_wrapper(body_bytes, f"{role} body"),
    }


def build_sec_https_observation_outcome(
    *,
    event_key: Mapping[str, Any],
    metadata_url: str,
    metadata_received_at_utc: str,
    metadata_headers_bytes: bytes,
    metadata_body_bytes: bytes,
    exhibit_url: str,
    exhibit_received_at_utc: str,
    exhibit_headers_bytes: bytes,
    exhibit_body_bytes: bytes,
    checkpoint_receipt_bytes: bytes,
) -> dict[str, Any]:
    """Build a prospective SEC observation bound by an external checkpoint."""
    try:
        key = validate_event_key(event_key)
    except PeadEventUniverseError as exc:
        raise PeadAnnouncementAvailabilityError("invalid SEC observation key") from exc
    checkpoint = _exact(
        _strict_json_bytes(checkpoint_receipt_bytes, "checkpoint receipt bytes"),
        _CHECKPOINT_FIELDS,
        "checkpoint receipt",
    )
    known_by = _utc_timestamp(checkpoint["recorded_at_utc"], "recorded_at_utc")
    return {
        "event_id": canonical_event_id(key),
        "event_key": key,
        "disposition": "available",
        "missing_reason": None,
        "claim": {
            "claim_kind": "known_public_by",
            "known_public_by_at_utc": known_by,
            "adapter_id": SEC_HTTPS_OBSERVATION_ADAPTER,
            "evidence": {
                "metadata_observation": _http_observation(
                    role="filing_metadata_positive_observation",
                    url=metadata_url,
                    received_at_utc=metadata_received_at_utc,
                    headers_bytes=metadata_headers_bytes,
                    body_bytes=metadata_body_bytes,
                ),
                "exhibit_observation": _http_observation(
                    role="earnings_exhibit_positive_observation",
                    url=exhibit_url,
                    received_at_utc=exhibit_received_at_utc,
                    headers_bytes=exhibit_headers_bytes,
                    body_bytes=exhibit_body_bytes,
                ),
                "checkpoint_receipt": _raw_wrapper(
                    checkpoint_receipt_bytes, "checkpoint receipt bytes"
                ),
            },
            "eligibility": {},
        },
    }


def _prepare_outcomes_for_build(
    outcomes: Sequence[Mapping[str, Any]], *, evidence_class: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for outcome in outcomes:
        copied = _plain(outcome)
        claim = copied.get("claim")
        if copied.get("disposition") == "available" and isinstance(claim, dict):
            adapter = claim.get("adapter_id")
            if isinstance(adapter, str):
                claim["eligibility"] = _eligibility(adapter, evidence_class)
        result.append(copied)
    return result


def build_pead_announcement_availability(
    *,
    expected_event_manifest: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
    evidence_class: str,
    created_at_utc: str,
    outcomes: Sequence[Mapping[str, Any]],
    trusted_provider_manifest_sha256s: Collection[str] = (),
    trusted_provider_record_sha256s: Collection[str] = (),
    trusted_checkpoint_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    """Build an exhaustive availability artifact against source announcement facts."""
    try:
        universe = validate_pead_event_universe(expected_event_manifest)
        announcement = validate_pead_announcement_evidence(
            announcement_evidence, expected_event_manifest=universe
        )
    except (PeadEventUniverseError, PeadAnnouncementEvidenceError) as exc:
        raise PeadAnnouncementAvailabilityError(
            "source event/announcement evidence is invalid"
        ) from exc
    if evidence_class not in _EVIDENCE_CLASSES:
        raise PeadAnnouncementAvailabilityError("unsupported availability evidence class")
    created = _utc_timestamp(created_at_utc, "created_at_utc")
    if _utc_value(created) < _utc_value(announcement["payload"]["created_at_utc"]):
        raise PeadAnnouncementAvailabilityError(
            "availability artifact predates announcement evidence"
        )
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise PeadAnnouncementAvailabilityError("outcomes must be a sequence")
    expected_ids = universe["payload"]["expected_event_ids"]
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for index, outcome in enumerate(
        _prepare_outcomes_for_build(outcomes, evidence_class=evidence_class)
    ):
        if not isinstance(outcome, Mapping):
            raise PeadAnnouncementAvailabilityError(f"outcomes[{index}] must be an object")
        event_id = outcome.get("event_id")
        if not isinstance(event_id, str) or event_id not in expected_ids:
            raise PeadAnnouncementAvailabilityError(
                f"outcomes[{index}] references an unexpected event"
            )
        if event_id in raw_by_id:
            raise PeadAnnouncementAvailabilityError("outcomes contains duplicate event")
        raw_by_id[event_id] = outcome
    if set(raw_by_id) != set(expected_ids):
        raise PeadAnnouncementAvailabilityError(
            "outcomes must account for every expected event exactly once"
        )
    manifest_roots = _trusted_hashes(
        trusted_provider_manifest_sha256s,
        "trusted_provider_manifest_sha256s",
    )
    record_roots = _trusted_hashes(
        trusted_provider_record_sha256s,
        "trusted_provider_record_sha256s",
    )
    checkpoint_roots = _trusted_hashes(trusted_checkpoint_sha256s, "trusted_checkpoint_sha256s")
    payload = {
        "schema_version": ANNOUNCEMENT_AVAILABILITY_SCHEMA_VERSION,
        "candidate_id": universe["payload"]["candidate_id"],
        "evidence_class": evidence_class,
        "created_at_utc": created,
        "expected_event_manifest_sha256": universe["artifact_hash"],
        "announcement_evidence_sha256": announcement["artifact_hash"],
        "trust_policy": _trust_policy(
            provider_manifests=manifest_roots,
            provider_records=record_roots,
            checkpoints=checkpoint_roots,
        ),
        "expected_event_ids": list(expected_ids),
        "outcomes": [raw_by_id[event_id] for event_id in expected_ids],
        "coverage": {
            "expected_events": 0,
            "available_claims": 0,
            "eligible_claims": 0,
            "missing_claims": 0,
            "event_universe_qualified": False,
            "announcement_actuals_complete": False,
            "blockers": [],
            "complete": False,
        },
    }
    provisional = {"artifact_hash": content_hash(payload), "payload": payload}
    # Validate once to derive normalized claims and coverage, then seal that result.
    normalized = _normalize_document(
        provisional,
        universe=universe,
        announcement=announcement,
        trusted_provider_manifest_hashes=manifest_roots,
        trusted_provider_record_hashes=record_roots,
        trusted_checkpoint_hashes=checkpoint_roots,
        require_claimed_hash=False,
        require_claimed_coverage=False,
    )
    document = {
        "artifact_hash": content_hash(normalized["payload"]),
        "payload": normalized["payload"],
    }
    return validate_pead_announcement_availability(
        document,
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        trusted_provider_manifest_sha256s=trusted_provider_manifest_sha256s,
        trusted_provider_record_sha256s=trusted_provider_record_sha256s,
        trusted_checkpoint_sha256s=trusted_checkpoint_sha256s,
    )


def _normalize_document(
    document: Mapping[str, Any],
    *,
    universe: Mapping[str, Any],
    announcement: Mapping[str, Any],
    trusted_provider_manifest_hashes: frozenset[str],
    trusted_provider_record_hashes: frozenset[str],
    trusted_checkpoint_hashes: frozenset[str],
    require_claimed_hash: bool,
    require_claimed_coverage: bool,
) -> dict[str, Any]:
    wrapper = _exact(document, _WRAPPER_FIELDS, "announcement availability")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "announcement availability.payload")
    claimed = _sha256(wrapper["artifact_hash"], "announcement availability.artifact_hash")
    if require_claimed_hash and content_hash(payload) != claimed:
        raise PeadAnnouncementAvailabilityError("announcement availability artifact hash mismatch")
    if payload["schema_version"] != ANNOUNCEMENT_AVAILABILITY_SCHEMA_VERSION:
        raise PeadAnnouncementAvailabilityError("unsupported announcement availability schema")
    if payload["candidate_id"] != universe["payload"]["candidate_id"]:
        raise PeadAnnouncementAvailabilityError("announcement availability candidate differs")
    evidence_class = payload["evidence_class"]
    if evidence_class not in _EVIDENCE_CLASSES:
        raise PeadAnnouncementAvailabilityError("unsupported availability evidence class")
    created = _utc_timestamp(payload["created_at_utc"], "created_at_utc")
    if _utc_value(created) < _utc_value(announcement["payload"]["created_at_utc"]):
        raise PeadAnnouncementAvailabilityError(
            "availability artifact predates announcement evidence"
        )
    if payload["expected_event_manifest_sha256"] != universe["artifact_hash"]:
        raise PeadAnnouncementAvailabilityError(
            "announcement availability binds another event universe"
        )
    if payload["announcement_evidence_sha256"] != announcement["artifact_hash"]:
        raise PeadAnnouncementAvailabilityError(
            "announcement availability binds another announcement artifact"
        )
    trust_policy = _exact(payload["trust_policy"], _TRUST_POLICY_FIELDS, "trust_policy")
    derived_trust_policy = _trust_policy(
        provider_manifests=trusted_provider_manifest_hashes,
        provider_records=trusted_provider_record_hashes,
        checkpoints=trusted_checkpoint_hashes,
    )
    if dict(trust_policy) != derived_trust_policy:
        raise PeadAnnouncementAvailabilityError(
            "announcement availability trust policy differs from external allowlists"
        )
    expected_ids = universe["payload"]["expected_event_ids"]
    if payload["expected_event_ids"] != expected_ids:
        raise PeadAnnouncementAvailabilityError(
            "announcement availability expected event IDs differ"
        )
    raw_outcomes = payload["outcomes"]
    expected_events = universe["payload"]["expected_events"]
    announcement_outcomes = announcement["payload"]["outcomes"]
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(expected_events):
        raise PeadAnnouncementAvailabilityError(
            "availability outcomes must account for every expected event"
        )
    outcomes = [
        _validate_outcome(
            raw_outcomes[index],
            expected_event=expected_events[index],
            announcement_outcome=announcement_outcomes[index],
            evidence_class=evidence_class,
            created_at=created,
            trusted_provider_manifest_hashes=trusted_provider_manifest_hashes,
            trusted_provider_record_hashes=trusted_provider_record_hashes,
            trusted_checkpoint_hashes=trusted_checkpoint_hashes,
            label=f"outcomes[{index}]",
        )
        for index in range(len(expected_events))
    ]
    if [outcome["event_id"] for outcome in outcomes] != expected_ids:
        raise PeadAnnouncementAvailabilityError(
            "availability outcomes are not in canonical event order"
        )
    if evidence_class == "prospective":
        frozen_at = _utc_value(universe["payload"]["frozen_at_utc"])
        for outcome in outcomes:
            if outcome["disposition"] != "available":
                continue
            known_public_by = _utc_value(outcome["claim"]["known_public_by_at_utc"])
            if frozen_at >= known_public_by:
                raise PeadAnnouncementAvailabilityError(
                    "prospective event universe was not frozen before announcement observation"
                )
    announcement_actuals_complete = all(
        outcome["disposition"] == "available" for outcome in announcement_outcomes
    )
    derived_coverage = _coverage(
        outcomes,
        event_universe_qualified=universe["payload"]["qualification_allowed"],
        announcement_actuals_complete=announcement_actuals_complete,
    )
    if require_claimed_coverage:
        coverage = _exact(payload["coverage"], _COVERAGE_FIELDS, "coverage")
        if dict(coverage) != derived_coverage:
            raise PeadAnnouncementAvailabilityError(
                "announcement availability coverage is not derived exactly"
            )
    normalized_payload = {
        "schema_version": ANNOUNCEMENT_AVAILABILITY_SCHEMA_VERSION,
        "candidate_id": universe["payload"]["candidate_id"],
        "evidence_class": evidence_class,
        "created_at_utc": created,
        "expected_event_manifest_sha256": universe["artifact_hash"],
        "announcement_evidence_sha256": announcement["artifact_hash"],
        "trust_policy": derived_trust_policy,
        "expected_event_ids": list(expected_ids),
        "outcomes": outcomes,
        "coverage": derived_coverage,
    }
    if require_claimed_hash and content_hash(normalized_payload) != claimed:
        raise PeadAnnouncementAvailabilityError("announcement availability is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized_payload)}


def validate_pead_announcement_availability(
    document: Mapping[str, Any],
    *,
    expected_event_manifest: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
    trusted_provider_manifest_sha256s: Collection[str] = (),
    trusted_provider_record_sha256s: Collection[str] = (),
    trusted_checkpoint_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    """Replay all source bytes, trust anchors, event facts, and eligibility."""
    try:
        universe = validate_pead_event_universe(expected_event_manifest)
        announcement = validate_pead_announcement_evidence(
            announcement_evidence, expected_event_manifest=universe
        )
    except (PeadEventUniverseError, PeadAnnouncementEvidenceError) as exc:
        raise PeadAnnouncementAvailabilityError(
            "source event/announcement evidence is invalid"
        ) from exc
    return _normalize_document(
        document,
        universe=universe,
        announcement=announcement,
        trusted_provider_manifest_hashes=_trusted_hashes(
            trusted_provider_manifest_sha256s,
            "trusted_provider_manifest_sha256s",
        ),
        trusted_provider_record_hashes=_trusted_hashes(
            trusted_provider_record_sha256s,
            "trusted_provider_record_sha256s",
        ),
        trusted_checkpoint_hashes=_trusted_hashes(
            trusted_checkpoint_sha256s, "trusted_checkpoint_sha256s"
        ),
        require_claimed_hash=True,
        require_claimed_coverage=True,
    )


def eligible_announcement_activations(
    document: Mapping[str, Any],
    *,
    expected_event_manifest: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
    trusted_provider_manifest_sha256s: Collection[str] = (),
    trusted_provider_record_sha256s: Collection[str] = (),
    trusted_checkpoint_sha256s: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Return conservative activations from a fully replayed availability artifact."""
    verified = validate_pead_announcement_availability(
        document,
        expected_event_manifest=expected_event_manifest,
        announcement_evidence=announcement_evidence,
        trusted_provider_manifest_sha256s=trusted_provider_manifest_sha256s,
        trusted_provider_record_sha256s=trusted_provider_record_sha256s,
        trusted_checkpoint_sha256s=trusted_checkpoint_sha256s,
    )
    result: list[dict[str, Any]] = []
    for outcome in verified["payload"]["outcomes"]:
        if outcome["disposition"] != "available":
            continue
        claim = outcome["claim"]
        result.append(
            {
                "event_id": outcome["event_id"],
                "event_key": outcome["event_key"],
                "claim_kind": "known_public_by",
                "signal_activation_at_utc": claim["known_public_by_at_utc"],
                "adapter_id": claim["adapter_id"],
                "consensus_cutoff_rule": claim["eligibility"]["consensus_cutoff_rule"],
                "market_cutoff_rule": claim["eligibility"]["market_cutoff_rule"],
                "announcement_availability_artifact_hash": verified["artifact_hash"],
                "announcement_evidence_artifact_hash": verified["payload"][
                    "announcement_evidence_sha256"
                ],
            }
        )
    return result


def _strict_json_file(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PeadAnnouncementAvailabilityError(
            f"cannot stat announcement availability file: {path}"
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise PeadAnnouncementAvailabilityError(
            f"announcement availability is not a regular file: {path}"
        )
    if size <= 0 or size > MAX_AVAILABILITY_ARTIFACT_BYTES:
        raise PeadAnnouncementAvailabilityError("announcement availability file size is invalid")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PeadAnnouncementAvailabilityError("cannot read announcement availability") from exc
    value = _strict_json_bytes(raw, "announcement availability")
    if canonical_json(value).encode("utf-8") != raw:
        raise PeadAnnouncementAvailabilityError(
            "announcement availability file is not canonical JSON"
        )
    return dict(value)


def load_pead_announcement_availability(
    path: str | Path,
    *,
    expected_event_manifest: Mapping[str, Any],
    announcement_evidence: Mapping[str, Any],
    trusted_provider_manifest_sha256s: Collection[str] = (),
    trusted_provider_record_sha256s: Collection[str] = (),
    trusted_checkpoint_sha256s: Collection[str] = (),
) -> dict[str, Any]:
    """Load canonical JSON and replay it against original evidence and trust roots."""
    return validate_pead_announcement_availability(
        _strict_json_file(Path(path)),
        expected_event_manifest=expected_event_manifest,
        announcement_evidence=announcement_evidence,
        trusted_provider_manifest_sha256s=trusted_provider_manifest_sha256s,
        trusted_provider_record_sha256s=trusted_provider_record_sha256s,
        trusted_checkpoint_sha256s=trusted_checkpoint_sha256s,
    )
