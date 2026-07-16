"""Strict evidence contracts for unresolved PEAD economic-return inputs.

This module records two deliberately narrow facts without importing or
depending on a PEAD research implementation:

* the current interpretation of Sharadar ``ACTIONS`` dividend rows is an
  empirically reconciled hypothesis, not qualifying source documentation; and
* terminal cash settlements must come from separately archived source
  receipts.  ``ACTIONS.value`` is never an allowed terminal payout.

Both documents are content-addressed wrappers.  Loaders reject duplicate JSON
keys, non-finite numbers, schema extensions, hash mismatches, and malformed
source receipts.  Terminal-settlement receipt bytes are verified relative to
the ledger file when loaded from disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION = (
    "pead_cash_distribution_semantics.v1"
)
TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION = (
    "pead_terminal_settlement_ledger.v1"
)

ACTIONS_DIVIDEND_DATE_ROLE = "candidate_ex_date_unproven"
ACTIONS_DIVIDEND_VALUE_ROLE = (
    "candidate_split_normalized_cash_per_share_unproven"
)
UNPROVEN_SEMANTICS_EVIDENCE_STATUS = (
    "empirically_reconciled_not_authoritatively_documented"
)

DEFAULT_ADJUSTMENT_CHECK_ABSOLUTE_TOLERANCE = 1e-8
DEFAULT_ADJUSTMENT_CHECK_RELATIVE_TOLERANCE = 1e-6
MAX_EVIDENCE_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_RECEIPT_BYTES = 64 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_CASH_SEMANTICS_FIELDS = {
    "schema_version",
    "ACTIONS",
    "evidence_status",
    "qualification_allowed",
    "adjustment_check_tolerance",
}
_CASH_ACTIONS_FIELDS = {"action", "date_role", "value_role"}
_TOLERANCE_FIELDS = {"absolute", "relative"}
_TERMINAL_LEDGER_FIELDS = {
    "schema_version",
    "ACTIONS",
    "cash_only_records",
}
_TERMINAL_ACTIONS_FIELDS = {"value_allowed"}
_TERMINAL_RECORD_FIELDS = {
    "ticker",
    "permaticker",
    "last_price_date",
    "settlement_date",
    "cash_per_terminal_share",
    "source_receipts",
}
_SOURCE_RECEIPT_FIELDS = {
    "url",
    "retrieved_at_utc",
    "local_path",
    "sha256",
    "bytes",
}


class PeadEconomicEvidenceError(ValueError):
    """A PEAD economic-evidence document is not exact or trustworthy."""


def canonical_json(value: Any) -> str:
    """Return deterministic JSON while rejecting non-JSON and non-finite data."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PeadEconomicEvidenceError(
                    "economic evidence contains a non-finite number"
                )
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PeadEconomicEvidenceError(
                        "economic evidence keys must be strings"
                    )
                result[key] = normalize(child)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise PeadEconomicEvidenceError(
            f"unsupported economic evidence value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Return the SHA-256 identity of a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadEconomicEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_fields(
    value: Any, fields: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadEconomicEvidenceError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadEconomicEvidenceError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_EVIDENCE_DOCUMENT_BYTES:
        raise PeadEconomicEvidenceError(f"{label} exceeds its size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadEconomicEvidenceError(f"{label} is not UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadEconomicEvidenceError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise PeadEconomicEvidenceError(
            f"{label} contains invalid number {token}"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise PeadEconomicEvidenceError(
            f"invalid {label} JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadEconomicEvidenceError(f"{label} root must be an object")
    return value


def _verified_wrapper(
    document: Mapping[str, Any], *, label: str
) -> tuple[str, Mapping[str, Any]]:
    wrapper = _exact_fields(document, _WRAPPER_FIELDS, label)
    payload = wrapper["payload"]
    if not isinstance(payload, Mapping):
        raise PeadEconomicEvidenceError(f"{label} payload must be an object")
    artifact_hash = _sha256(wrapper["artifact_hash"], f"{label} artifact_hash")
    if content_hash(payload) != artifact_hash:
        raise PeadEconomicEvidenceError(f"{label} artifact hash mismatch")
    return artifact_hash, payload


def _plain_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _finite_nonnegative(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PeadEconomicEvidenceError(f"{label} must be a finite non-negative number")
    if isinstance(value, float) and not math.isfinite(value):
        raise PeadEconomicEvidenceError(f"{label} must be a finite non-negative number")
    if value < 0:
        raise PeadEconomicEvidenceError(f"{label} must be a finite non-negative number")
    return value


def _finite_positive(value: Any, label: str) -> int | float:
    parsed = _finite_nonnegative(value, label)
    if parsed <= 0:
        raise PeadEconomicEvidenceError(f"{label} must be positive")
    return parsed


def _canonical_date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PeadEconomicEvidenceError(f"{label} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadEconomicEvidenceError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise PeadEconomicEvidenceError(f"{label} must be a canonical ISO date")
    return value


def _canonical_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadEconomicEvidenceError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeadEconomicEvidenceError(f"{label} must be canonical UTC") from exc
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if canonical != value:
        raise PeadEconomicEvidenceError(f"{label} must be canonical UTC")
    return value


def _https_url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadEconomicEvidenceError(f"{label} must be a non-empty HTTPS URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise PeadEconomicEvidenceError(
            f"{label} must be a non-empty HTTPS URL"
        ) from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise PeadEconomicEvidenceError(f"{label} must be a non-empty HTTPS URL")
    return value


def _relative_local_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
    ):
        raise PeadEconomicEvidenceError(f"{label} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PeadEconomicEvidenceError(f"{label} must be a canonical relative path")
    return value


def validate_cash_distribution_semantics(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact, explicitly nonqualifying dividend interpretation."""
    artifact_hash, raw_payload = _verified_wrapper(
        document, label="cash distribution semantics"
    )
    payload = _exact_fields(
        raw_payload, _CASH_SEMANTICS_FIELDS, "cash distribution semantics payload"
    )
    if payload["schema_version"] != CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION:
        raise PeadEconomicEvidenceError(
            "unsupported cash distribution semantics schema"
        )
    actions = _exact_fields(
        payload["ACTIONS"], _CASH_ACTIONS_FIELDS, "cash distribution ACTIONS"
    )
    expected_actions = {
        "action": "dividend",
        "date_role": ACTIONS_DIVIDEND_DATE_ROLE,
        "value_role": ACTIONS_DIVIDEND_VALUE_ROLE,
    }
    if dict(actions) != expected_actions:
        raise PeadEconomicEvidenceError(
            "cash distribution ACTIONS semantics differ from the unproven contract"
        )
    if payload["evidence_status"] != UNPROVEN_SEMANTICS_EVIDENCE_STATUS:
        raise PeadEconomicEvidenceError(
            "cash distribution evidence status must remain explicitly unproven"
        )
    if payload["qualification_allowed"] is not False:
        raise PeadEconomicEvidenceError(
            "unproven cash distribution semantics cannot allow qualification"
        )
    tolerance = _exact_fields(
        payload["adjustment_check_tolerance"],
        _TOLERANCE_FIELDS,
        "adjustment-check tolerance",
    )
    _finite_nonnegative(tolerance["absolute"], "absolute adjustment-check tolerance")
    _finite_nonnegative(tolerance["relative"], "relative adjustment-check tolerance")
    return {"artifact_hash": artifact_hash, "payload": _plain_json(payload)}


def build_current_unproven_cash_distribution_semantics(
    *,
    absolute_tolerance: float = DEFAULT_ADJUSTMENT_CHECK_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ADJUSTMENT_CHECK_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Build the current empirical, explicitly nonqualifying semantics artifact."""
    payload = {
        "schema_version": CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION,
        "ACTIONS": {
            "action": "dividend",
            "date_role": ACTIONS_DIVIDEND_DATE_ROLE,
            "value_role": ACTIONS_DIVIDEND_VALUE_ROLE,
        },
        "evidence_status": UNPROVEN_SEMANTICS_EVIDENCE_STATUS,
        "qualification_allowed": False,
        "adjustment_check_tolerance": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
        },
    }
    document = {"artifact_hash": content_hash(payload), "payload": payload}
    return validate_cash_distribution_semantics(document)


def build_unproven_cash_distribution_semantics(
    *,
    absolute_tolerance: float = DEFAULT_ADJUSTMENT_CHECK_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_ADJUSTMENT_CHECK_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Alias with a shorter name for the current unproven semantics builder."""
    return build_current_unproven_cash_distribution_semantics(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def _validate_source_receipt(
    value: Any, *, label: str
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    receipt = _exact_fields(value, _SOURCE_RECEIPT_FIELDS, label)
    url = _https_url(receipt["url"], f"{label} URL")
    retrieved_at = _canonical_utc(
        receipt["retrieved_at_utc"], f"{label} retrieval UTC"
    )
    local_path = _relative_local_path(
        receipt["local_path"], f"{label} local path"
    )
    sha256 = _sha256(receipt["sha256"], f"{label} sha256")
    byte_count = receipt["bytes"]
    if (
        type(byte_count) is not int
        or not 0 < byte_count <= MAX_SOURCE_RECEIPT_BYTES
    ):
        raise PeadEconomicEvidenceError(f"{label} bytes must be a positive integer")
    normalized = {
        "url": url,
        "retrieved_at_utc": retrieved_at,
        "local_path": local_path,
        "sha256": sha256,
        "bytes": byte_count,
    }
    identity = (url, retrieved_at, local_path, sha256, byte_count)
    return normalized, identity


def _verify_source_bytes(
    receipt: Mapping[str, Any], *, source_root: Path, label: str
) -> None:
    root = source_root.resolve()
    candidate = source_root / PurePosixPath(receipt["local_path"])
    if not candidate.is_file() or candidate.is_symlink():
        raise PeadEconomicEvidenceError(f"{label} source file is missing")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PeadEconomicEvidenceError(
            f"{label} source file escapes the ledger directory"
        ) from exc
    raw = resolved.read_bytes()
    if len(raw) != receipt["bytes"]:
        raise PeadEconomicEvidenceError(f"{label} source byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise PeadEconomicEvidenceError(f"{label} source SHA-256 mismatch")


def validate_terminal_settlement_ledger(
    document: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate exact cash-only terminal records and optional archived bytes."""
    artifact_hash, raw_payload = _verified_wrapper(
        document, label="terminal settlement ledger"
    )
    payload = _exact_fields(
        raw_payload, _TERMINAL_LEDGER_FIELDS, "terminal settlement ledger payload"
    )
    if payload["schema_version"] != TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION:
        raise PeadEconomicEvidenceError("unsupported terminal settlement ledger schema")
    actions = _exact_fields(
        payload["ACTIONS"], _TERMINAL_ACTIONS_FIELDS, "terminal settlement ACTIONS"
    )
    if actions["value_allowed"] is not False:
        raise PeadEconomicEvidenceError(
            "ACTIONS.value cannot be used as a terminal settlement"
        )
    records = payload["cash_only_records"]
    if not isinstance(records, list):
        raise PeadEconomicEvidenceError("cash_only_records must be a list")

    resolved_root = Path(source_root) if source_root is not None else None
    normalized_records: list[dict[str, Any]] = []
    record_keys: set[tuple[Any, ...]] = set()
    for index, raw_record in enumerate(records):
        label = f"terminal settlement record {index}"
        record = _exact_fields(raw_record, _TERMINAL_RECORD_FIELDS, label)
        ticker = record["ticker"]
        if (
            not isinstance(ticker, str)
            or not ticker
            or ticker != ticker.strip()
        ):
            raise PeadEconomicEvidenceError(f"{label} ticker must be exact and non-empty")
        permaticker = record["permaticker"]
        if type(permaticker) is not int or permaticker <= 0:
            raise PeadEconomicEvidenceError(
                f"{label} permaticker must be a positive integer"
            )
        last_price_date = _canonical_date(
            record["last_price_date"], f"{label} last_price_date"
        )
        settlement_date = _canonical_date(
            record["settlement_date"], f"{label} settlement_date"
        )
        if settlement_date < last_price_date:
            raise PeadEconomicEvidenceError(
                f"{label} settlement_date precedes last_price_date"
            )
        cash = _finite_positive(
            record["cash_per_terminal_share"],
            f"{label} cash_per_terminal_share",
        )
        receipts = record["source_receipts"]
        if not isinstance(receipts, list) or not receipts:
            raise PeadEconomicEvidenceError(
                f"{label} requires one or more source receipts"
            )
        normalized_receipts: list[dict[str, Any]] = []
        receipt_keys: set[tuple[Any, ...]] = set()
        for receipt_index, raw_receipt in enumerate(receipts):
            receipt_label = f"{label} source receipt {receipt_index}"
            normalized_receipt, receipt_key = _validate_source_receipt(
                raw_receipt, label=receipt_label
            )
            if receipt_key in receipt_keys:
                raise PeadEconomicEvidenceError(
                    f"{label} contains a duplicate source receipt"
                )
            receipt_keys.add(receipt_key)
            if resolved_root is not None:
                _verify_source_bytes(
                    normalized_receipt,
                    source_root=resolved_root,
                    label=receipt_label,
                )
            normalized_receipts.append(normalized_receipt)

        record_key = (ticker, permaticker, last_price_date, settlement_date)
        if record_key in record_keys:
            raise PeadEconomicEvidenceError(
                "terminal settlement ledger contains a duplicate record key"
            )
        record_keys.add(record_key)
        normalized_records.append(
            {
                "ticker": ticker,
                "permaticker": permaticker,
                "last_price_date": last_price_date,
                "settlement_date": settlement_date,
                "cash_per_terminal_share": cash,
                "source_receipts": normalized_receipts,
            }
        )

    normalized_payload = {
        "schema_version": TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION,
        "ACTIONS": {"value_allowed": False},
        "cash_only_records": normalized_records,
    }
    return {"artifact_hash": artifact_hash, "payload": normalized_payload}


def build_empty_terminal_settlement_ledger() -> dict[str, Any]:
    """Build an exact ledger that makes no terminal-settlement claim."""
    payload = {
        "schema_version": TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION,
        "ACTIONS": {"value_allowed": False},
        "cash_only_records": [],
    }
    document = {"artifact_hash": content_hash(payload), "payload": payload}
    return validate_terminal_settlement_ledger(document)


def load_cash_distribution_semantics(path: str | Path) -> dict[str, Any]:
    """Load and validate a duplicate-key-free semantics document."""
    document = _strict_json(Path(path), label="cash distribution semantics")
    return validate_cash_distribution_semantics(document)


def load_terminal_settlement_ledger(path: str | Path) -> dict[str, Any]:
    """Load a ledger and verify every archived source receipt's exact bytes."""
    resolved = Path(path)
    document = _strict_json(resolved, label="terminal settlement ledger")
    return validate_terminal_settlement_ledger(
        document, source_root=resolved.parent
    )


def load_pead_economic_evidence(path: str | Path) -> dict[str, Any]:
    """Load either supported economic-evidence document by its exact schema."""
    resolved = Path(path)
    document = _strict_json(resolved, label="PEAD economic evidence")
    wrapper = _exact_fields(document, _WRAPPER_FIELDS, "PEAD economic evidence")
    payload = wrapper["payload"]
    if not isinstance(payload, Mapping):
        raise PeadEconomicEvidenceError("PEAD economic evidence payload must be an object")
    schema_version = payload.get("schema_version")
    if schema_version == CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION:
        return validate_cash_distribution_semantics(document)
    if schema_version == TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION:
        return validate_terminal_settlement_ledger(
            document, source_root=resolved.parent
        )
    raise PeadEconomicEvidenceError("unsupported PEAD economic evidence schema")


__all__: Sequence[str] = (
    "ACTIONS_DIVIDEND_DATE_ROLE",
    "ACTIONS_DIVIDEND_VALUE_ROLE",
    "CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION",
    "DEFAULT_ADJUSTMENT_CHECK_ABSOLUTE_TOLERANCE",
    "DEFAULT_ADJUSTMENT_CHECK_RELATIVE_TOLERANCE",
    "MAX_EVIDENCE_DOCUMENT_BYTES",
    "MAX_SOURCE_RECEIPT_BYTES",
    "PeadEconomicEvidenceError",
    "TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION",
    "UNPROVEN_SEMANTICS_EVIDENCE_STATUS",
    "build_current_unproven_cash_distribution_semantics",
    "build_empty_terminal_settlement_ledger",
    "build_unproven_cash_distribution_semantics",
    "canonical_json",
    "content_hash",
    "load_cash_distribution_semantics",
    "load_pead_economic_evidence",
    "load_terminal_settlement_ledger",
    "validate_cash_distribution_semantics",
    "validate_terminal_settlement_ledger",
)
