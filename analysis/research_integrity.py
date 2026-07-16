"""Immutable preregistration, trial accounting, and one-shot holdouts.

This module is deliberately separate from the backtest and promotion code.  It
does not decide whether a strategy is good; it makes the evidence trail needed
to answer that question auditable:

* a research protocol is frozen as a canonical, content-addressed artifact;
* every attempted trial is registered in one append-only, hash-chained journal;
* trial counts are derived from that journal rather than supplied by a caller;
* a named holdout can be opened only after its protocol is frozen, and only
  once; and
* prospective holdouts bind an acquisition commitment before observations and
  bind realized bytes only at permanent finalization; and
* the holdout's result and decision are subsequently recorded exactly once.

The filesystem store is tamper-evident, not magically tamper-proof.  Every read
recomputes record hashes, verifies the complete event chain, and validates all
cross-record references.  Stronger deletion resistance requires anchoring the
latest event hash outside this process (for example in object-lock storage or a
signed release), which callers can do via :attr:`head_hash`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar


class ResearchIntegrityError(RuntimeError):
    """Base class for research-integrity storage and workflow failures."""


class IntegrityViolation(ResearchIntegrityError):
    """Persisted evidence is malformed, non-canonical, or has been changed."""


class DuplicateRecordError(ResearchIntegrityError):
    """A stable identifier is already bound to different immutable evidence."""


class ProtocolNotFrozen(ResearchIntegrityError):
    """The requested protocol is not frozen in this program ledger."""


class UnknownRecordError(ResearchIntegrityError):
    """A referenced protocol, trial, opening, or result does not exist."""


class HoldoutAlreadyOpened(ResearchIntegrityError):
    """A holdout has already been consumed by an opening event."""


class HoldoutAlreadyDecided(ResearchIntegrityError):
    """A holdout opening already has its permanent decision."""


_HEX = frozenset("0123456789abcdef")
_RECORD_SCHEMA = 1
_EVENT_KINDS = {
    "protocol_frozen": "research_protocol",
    "trial_registered": "research_trial",
    "trial_outcome_recorded": "trial_outcome",
    "holdout_opened": "holdout_opening",
    "holdout_decided": "holdout_decision",
}
_RECORD_DIRECTORIES = {
    "research_protocol": "protocols",
    "research_trial": "trials",
    "trial_outcome": "trial_outcomes",
    "holdout_opening": "holdout_openings",
    "holdout_decision": "holdout_decisions",
}

_PROSPECTIVE_HOLDOUT_MODE = "prospective"
_PROSPECTIVE_BINDING_MODE = "prospective_commitment"
_FIXED_HOLDOUT_MODE = "fixed_snapshot"
_PROSPECTIVE_HOLDOUT_FIELDS = {
    "mode", "start", "end", "source_manifest_hash", "query_hash",
    "acquisition_plan_hash", "append_only", "sealed",
}
_FIXED_HOLDOUT_FIELDS = {
    "mode", "start", "end", "data_artifact_hash", "sealed",
}
_PROSPECTIVE_ONLY_FIELDS = {
    "source_manifest_hash", "query_hash", "acquisition_plan_hash",
    "append_only",
}


def _canonical_value(value: Any) -> Any:
    """Convert *value* to deterministic JSON, rejecting ambiguous inputs."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("research records cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("research record mapping keys must be strings")
            normalized[key] = _canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported research record value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical UTF-8 JSON representation used for every hash."""
    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """SHA-256 of a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_loads(value: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise IntegrityViolation(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(token: str) -> None:
        raise IntegrityViolation(f"invalid JSON number: {token}")

    try:
        return json.loads(
            value, object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except ResearchIntegrityError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntegrityViolation("invalid research record JSON") from exc


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _sha256(value: Any, *, field_name: str) -> str:
    candidate = _required_text(value, field_name=field_name).lower()
    if len(candidate) != 64 or any(character not in _HEX for character in candidate):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return candidate


def _timestamp(value: str | datetime | None, *, field_name: str) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        parsed = value.astimezone(timezone.utc)
    elif isinstance(value, str):
        if not value.endswith("Z"):
            raise ValueError(f"{field_name} must be canonical UTC ending in Z")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid UTC timestamp") from exc
    else:
        raise TypeError(f"{field_name} must be a datetime or UTC timestamp")
    parsed = parsed.astimezone(timezone.utc)
    # A single representation prevents the same instant from acquiring two
    # hashes.  Microseconds are retained when present and omitted when zero.
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if isinstance(value, str) and value != canonical:
        raise ValueError(f"{field_name} must be canonical UTC: {canonical}")
    return canonical


def _calendar_date(value: Any, *, field_name: str) -> date:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be a canonical ISO date")
    return parsed


def _timestamp_value(value: str, *, field_name: str) -> datetime:
    canonical = _timestamp(value, field_name=field_name)
    return datetime.fromisoformat(canonical[:-1] + "+00:00")


def _mapping(value: Any, *, field_name: str, nonempty: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = _canonical_value(value)
    if nonempty and not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _mapping_sequence(
        value: Any, *, field_name: str, identity_field: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    for item in value:
        entry = _mapping(item, field_name=field_name)
        identity = _required_text(entry.get(identity_field), field_name=identity_field)
        if identity in identities:
            raise ValueError(f"duplicate {identity_field}: {identity}")
        entry[identity_field] = identity
        identities.add(identity)
        normalized.append(entry)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    # Hypothesis/candidate declarations are identity-keyed sets, not ordered
    # execution steps.  Sorting removes insertion order as a source of a new
    # protocol hash while leaving genuinely ordered lists inside each entry
    # untouched.
    return sorted(normalized, key=lambda entry: entry[identity_field])


def _exact_fields(payload: Mapping[str, Any], expected: set[str], *, record: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise IntegrityViolation(
            f"invalid {record} fields (missing={missing}, extra={extra})")


def _validate_holdout_specification(specification: Mapping[str, Any], *,
                                    field_name: str) -> None:
    """Validate an explicitly typed holdout while retaining legacy fixed specs."""
    mode = specification.get("mode")
    if mode is None:
        if set(specification) & _PROSPECTIVE_ONLY_FIELDS:
            raise ValueError(
                f"{field_name} uses prospective fields without mode=prospective")
        if "data_artifact_hash" not in specification:
            raise ValueError(
                f"{field_name} legacy fixed snapshot requires data_artifact_hash")
        normalized_hash = _sha256(
            specification["data_artifact_hash"],
            field_name=f"{field_name}.data_artifact_hash",
        )
        if specification["data_artifact_hash"] != normalized_hash:
            raise ValueError(
                f"{field_name}.data_artifact_hash must use lowercase hex")
        has_start = "start" in specification
        has_end = "end" in specification
        if has_start != has_end:
            raise ValueError(f"{field_name} must declare both start and end")
        if has_start:
            start = _calendar_date(
                specification["start"], field_name=f"{field_name}.start")
            end = _calendar_date(
                specification["end"], field_name=f"{field_name}.end")
            if start > end:
                raise ValueError(f"{field_name}.start cannot be after end")
        if "sealed" in specification and specification["sealed"] is not True:
            raise ValueError(f"{field_name}.sealed must be true")
        return
    if mode == _FIXED_HOLDOUT_MODE:
        actual = set(specification)
        if actual != _FIXED_HOLDOUT_FIELDS:
            missing = sorted(_FIXED_HOLDOUT_FIELDS - actual)
            extra = sorted(actual - _FIXED_HOLDOUT_FIELDS)
            raise ValueError(
                f"invalid {field_name} fields (missing={missing}, extra={extra})")
        start = _calendar_date(
            specification["start"], field_name=f"{field_name}.start")
        end = _calendar_date(
            specification["end"], field_name=f"{field_name}.end")
        if start > end:
            raise ValueError(f"{field_name}.start cannot be after end")
        normalized_hash = _sha256(
            specification["data_artifact_hash"],
            field_name=f"{field_name}.data_artifact_hash",
        )
        if specification["data_artifact_hash"] != normalized_hash:
            raise ValueError(
                f"{field_name}.data_artifact_hash must use lowercase hex")
        if specification["sealed"] is not True:
            raise ValueError(f"{field_name}.sealed must be true")
        return
    if mode != _PROSPECTIVE_HOLDOUT_MODE:
        raise ValueError(f"{field_name}.mode is unsupported")
    actual = set(specification)
    if actual != _PROSPECTIVE_HOLDOUT_FIELDS:
        missing = sorted(_PROSPECTIVE_HOLDOUT_FIELDS - actual)
        extra = sorted(actual - _PROSPECTIVE_HOLDOUT_FIELDS)
        raise ValueError(
            f"invalid {field_name} fields (missing={missing}, extra={extra})")
    start = _calendar_date(specification["start"], field_name=f"{field_name}.start")
    end = _calendar_date(specification["end"], field_name=f"{field_name}.end")
    if start > end:
        raise ValueError(f"{field_name}.start cannot be after end")
    for name in ("source_manifest_hash", "query_hash", "acquisition_plan_hash"):
        normalized_hash = _sha256(
            specification[name], field_name=f"{field_name}.{name}")
        if specification[name] != normalized_hash:
            raise ValueError(f"{field_name}.{name} must use lowercase hex")
    if specification["append_only"] is not True:
        raise ValueError(f"{field_name}.append_only must be true")
    if specification["sealed"] is not True:
        raise ValueError(f"{field_name}.sealed must be true")


def prospective_data_commitment_hash(specification: Mapping[str, Any]) -> str:
    """Hash the acquisition commitment for one prospective holdout spec."""
    normalized = _mapping(
        specification, field_name="prospective holdout specification")
    _validate_holdout_specification(
        normalized, field_name="prospective holdout specification")
    if normalized.get("mode") != _PROSPECTIVE_HOLDOUT_MODE:
        raise ValueError("holdout specification is not prospective")
    return content_hash({
        "schema_version": _RECORD_SCHEMA,
        "record_type": "prospective_data_commitment",
        "start": normalized["start"],
        "end": normalized["end"],
        "source_manifest_hash": normalized["source_manifest_hash"],
        "query_hash": normalized["query_hash"],
        "acquisition_plan_hash": normalized["acquisition_plan_hash"],
        "append_only": True,
    })


def _holdout_binding_key(specification: Mapping[str, Any]) -> tuple[str, str]:
    """Return the program-wide identity consumed when a holdout opens."""
    if specification.get("mode") == _PROSPECTIVE_HOLDOUT_MODE:
        return (_PROSPECTIVE_BINDING_MODE,
                prospective_data_commitment_hash(specification))
    return (
        _FIXED_HOLDOUT_MODE,
        _sha256(
            specification["data_artifact_hash"],
            field_name="holdout data_artifact_hash",
        ),
    )


def _validate_unique_holdout_bindings(
        holdouts: Mapping[str, Mapping[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for holdout_id, specification in holdouts.items():
        key = _holdout_binding_key(specification)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "duplicate holdout data commitment: "
                f"{previous!r} and {holdout_id!r}")
        seen[key] = holdout_id


TRecord = TypeVar("TRecord", bound="ContentRecord")


@dataclass(frozen=True)
class ContentRecord:
    """A canonical payload whose stable identity is its SHA-256 digest."""

    record_hash: str
    payload_json: str = field(repr=False)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def record_type(self) -> str:
        return str(self.payload["record_type"])

    def to_json(self) -> str:
        return canonical_json({
            "record_hash": self.record_hash,
            "payload": self.payload,
        }) + "\n"

    @classmethod
    def _from_payload(cls: type[TRecord], payload: Mapping[str, Any]) -> TRecord:
        normalized = _canonical_value(payload)
        payload_json = canonical_json(normalized)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        record = cls(digest, payload_json)
        record._validate_payload(normalized)
        return record

    @classmethod
    def from_json(cls: type[TRecord], value: str) -> TRecord:
        document = _strict_json_loads(value)
        if not isinstance(document, Mapping):
            raise IntegrityViolation("research record document must be a mapping")
        _exact_fields(
            document, {"record_hash", "payload"}, record="content record document")
        claimed = _sha256(document["record_hash"], field_name="record_hash")
        payload = document["payload"]
        if not isinstance(payload, Mapping):
            raise IntegrityViolation("research record payload must be a mapping")
        payload_json = canonical_json(payload)
        actual = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if claimed != actual:
            raise IntegrityViolation("research record hash mismatch")
        record = cls(actual, payload_json)
        record._validate_payload(record.payload)
        return record

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != _RECORD_SCHEMA:
            raise IntegrityViolation("unsupported research record schema")


@dataclass(frozen=True)
class ResearchProtocol(ContentRecord):
    """An immutable, fully specified research preregistration."""

    @classmethod
    def create(
            cls, *, program_id: str, protocol_id: str, version: str,
            objective: str, hypotheses: Sequence[Mapping[str, Any]],
            candidate_specifications: Sequence[Mapping[str, Any]],
            data_plan: Mapping[str, Any], evaluation_plan: Mapping[str, Any],
            decision_rules: Mapping[str, Any],
            holdouts: Mapping[str, Mapping[str, Any]], code_version: str,
            frozen_at: str | datetime | None = None) -> ResearchProtocol:
        normalized_holdouts: dict[str, Any] = {}
        if not isinstance(holdouts, Mapping) or not holdouts:
            raise ValueError("holdouts must declare at least one sealed holdout")
        for raw_id, specification in holdouts.items():
            holdout_id = _required_text(raw_id, field_name="holdout_id")
            if holdout_id in normalized_holdouts:
                raise ValueError(f"duplicate holdout_id: {holdout_id}")
            normalized_specification = _mapping(
                specification, field_name=f"holdouts[{holdout_id}]")
            _validate_holdout_specification(
                normalized_specification,
                field_name=f"holdouts[{holdout_id}]",
            )
            normalized_holdouts[holdout_id] = normalized_specification
        _validate_unique_holdout_bindings(normalized_holdouts)
        payload = {
            "schema_version": _RECORD_SCHEMA,
            "record_type": "research_protocol",
            "program_id": _required_text(program_id, field_name="program_id"),
            "protocol_id": _required_text(protocol_id, field_name="protocol_id"),
            "version": _required_text(version, field_name="version"),
            "status": "frozen",
            "objective": _required_text(objective, field_name="objective"),
            "hypotheses": _mapping_sequence(
                hypotheses, field_name="hypotheses", identity_field="hypothesis_id"),
            "candidate_specifications": _mapping_sequence(
                candidate_specifications, field_name="candidate_specifications",
                identity_field="candidate_id"),
            "data_plan": _mapping(data_plan, field_name="data_plan"),
            "evaluation_plan": _mapping(
                evaluation_plan, field_name="evaluation_plan"),
            "decision_rules": _mapping(decision_rules, field_name="decision_rules"),
            "holdouts": _canonical_value(normalized_holdouts),
            "code_version": _required_text(code_version, field_name="code_version"),
            "frozen_at": _timestamp(frozen_at, field_name="frozen_at"),
        }
        return cls._from_payload(payload)

    @property
    def protocol_hash(self) -> str:
        return self.record_hash

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        super()._validate_payload(payload)
        expected = {
            "schema_version", "record_type", "program_id", "protocol_id",
            "version", "status", "objective", "hypotheses",
            "candidate_specifications", "data_plan", "evaluation_plan",
            "decision_rules", "holdouts", "code_version", "frozen_at",
        }
        _exact_fields(payload, expected, record="research protocol")
        if payload["record_type"] != "research_protocol" or payload["status"] != "frozen":
            raise IntegrityViolation("invalid research protocol type or status")
        for name in ("program_id", "protocol_id", "version", "objective", "code_version"):
            _required_text(payload[name], field_name=name)
        _timestamp(payload["frozen_at"], field_name="frozen_at")
        _mapping_sequence(
            payload["hypotheses"], field_name="hypotheses",
            identity_field="hypothesis_id")
        _mapping_sequence(
            payload["candidate_specifications"],
            field_name="candidate_specifications", identity_field="candidate_id")
        for name in ("data_plan", "evaluation_plan", "decision_rules"):
            _mapping(payload[name], field_name=name)
        holdouts = _mapping(payload["holdouts"], field_name="holdouts")
        for holdout_id, specification in holdouts.items():
            _required_text(holdout_id, field_name="holdout_id")
            normalized_specification = _mapping(
                specification, field_name=f"holdouts[{holdout_id}]")
            _validate_holdout_specification(
                normalized_specification,
                field_name=f"holdouts[{holdout_id}]",
            )
        _validate_unique_holdout_bindings(holdouts)


@dataclass(frozen=True)
class TrialRegistration(ContentRecord):
    """One attempted configuration; its existence contributes to trial count."""

    @classmethod
    def create(
            cls, *, protocol_hash: str, trial_id: str, candidate_id: str,
            family: str, inputs: Mapping[str, Any], data_version: str,
            code_version: str, seed: int | None = None,
            registered_at: str | datetime | None = None) -> TrialRegistration:
        if seed is not None and (type(seed) is not int):
            raise TypeError("seed must be an integer or None")
        payload = {
            "schema_version": _RECORD_SCHEMA,
            "record_type": "research_trial",
            "protocol_hash": _sha256(protocol_hash, field_name="protocol_hash"),
            "trial_id": _required_text(trial_id, field_name="trial_id"),
            "candidate_id": _required_text(candidate_id, field_name="candidate_id"),
            "family": _required_text(family, field_name="family"),
            "inputs": _mapping(inputs, field_name="inputs"),
            "data_version": _required_text(data_version, field_name="data_version"),
            "code_version": _required_text(code_version, field_name="code_version"),
            "seed": seed,
            "registered_at": _timestamp(registered_at, field_name="registered_at"),
        }
        return cls._from_payload(payload)

    @property
    def trial_hash(self) -> str:
        return self.record_hash

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        super()._validate_payload(payload)
        expected = {
            "schema_version", "record_type", "protocol_hash", "trial_id",
            "candidate_id", "family", "inputs", "data_version", "code_version",
            "seed", "registered_at",
        }
        _exact_fields(payload, expected, record="research trial")
        if payload["record_type"] != "research_trial":
            raise IntegrityViolation("invalid research trial type")
        _sha256(payload["protocol_hash"], field_name="protocol_hash")
        for name in ("trial_id", "candidate_id", "family", "data_version", "code_version"):
            _required_text(payload[name], field_name=name)
        _mapping(payload["inputs"], field_name="inputs")
        if payload["seed"] is not None and type(payload["seed"]) is not int:
            raise IntegrityViolation("trial seed must be an integer or null")
        _timestamp(payload["registered_at"], field_name="registered_at")


@dataclass(frozen=True)
class TrialOutcome(ContentRecord):
    """The terminal, one-shot outcome of a registered trial."""

    @classmethod
    def create(
            cls, *, trial_hash: str, status: str,
            result_summary: Mapping[str, Any], evidence_hash: str,
            recorded_at: str | datetime | None = None) -> TrialOutcome:
        normalized_status = _required_text(status, field_name="status").lower()
        if normalized_status not in {"completed", "failed", "abandoned"}:
            raise ValueError("trial status must be completed, failed, or abandoned")
        payload = {
            "schema_version": _RECORD_SCHEMA,
            "record_type": "trial_outcome",
            "trial_hash": _sha256(trial_hash, field_name="trial_hash"),
            "status": normalized_status,
            "result_summary": _mapping(
                result_summary, field_name="result_summary", nonempty=False),
            "evidence_hash": _sha256(evidence_hash, field_name="evidence_hash"),
            "recorded_at": _timestamp(recorded_at, field_name="recorded_at"),
        }
        return cls._from_payload(payload)

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        super()._validate_payload(payload)
        expected = {
            "schema_version", "record_type", "trial_hash", "status",
            "result_summary", "evidence_hash", "recorded_at",
        }
        _exact_fields(payload, expected, record="trial outcome")
        if payload["record_type"] != "trial_outcome":
            raise IntegrityViolation("invalid trial outcome type")
        _sha256(payload["trial_hash"], field_name="trial_hash")
        if payload["status"] not in {"completed", "failed", "abandoned"}:
            raise IntegrityViolation("invalid trial outcome status")
        _mapping(payload["result_summary"], field_name="result_summary", nonempty=False)
        _sha256(payload["evidence_hash"], field_name="evidence_hash")
        _timestamp(payload["recorded_at"], field_name="recorded_at")


@dataclass(frozen=True)
class HoldoutOpening(ContentRecord):
    """Permanent evidence that one named holdout was exposed."""

    @property
    def opening_hash(self) -> str:
        return self.record_hash

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        super()._validate_payload(payload)
        common = {
            "schema_version", "record_type", "protocol_hash", "holdout_id",
            "holdout_spec_hash", "trial_hash", "actor",
            "program_trial_count_at_open", "protocol_trial_count_at_open",
            "opened_at",
        }
        prospective = payload.get("data_binding_mode") is not None
        expected = (common | {"data_binding_mode", "data_commitment_hash"}
                    if prospective else common | {"data_artifact_hash"})
        _exact_fields(payload, expected, record="holdout opening")
        if payload["record_type"] != "holdout_opening":
            raise IntegrityViolation("invalid holdout opening type")
        for name in ("protocol_hash", "holdout_spec_hash", "trial_hash"):
            _sha256(payload[name], field_name=name)
        if prospective:
            if payload["data_binding_mode"] != _PROSPECTIVE_BINDING_MODE:
                raise IntegrityViolation("invalid prospective data binding mode")
            _sha256(payload["data_commitment_hash"],
                    field_name="data_commitment_hash")
        else:
            _sha256(payload["data_artifact_hash"], field_name="data_artifact_hash")
        for name in ("holdout_id", "actor"):
            _required_text(payload[name], field_name=name)
        for name in ("program_trial_count_at_open", "protocol_trial_count_at_open"):
            if type(payload[name]) is not int or payload[name] < 1:
                raise IntegrityViolation(f"{name} must be a positive derived count")
        _timestamp(payload["opened_at"], field_name="opened_at")


@dataclass(frozen=True)
class HoldoutDecision(ContentRecord):
    """The permanent pass/fail/invalid result for one holdout opening."""

    @property
    def decision_hash(self) -> str:
        return self.record_hash

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        super()._validate_payload(payload)
        common = {
            "schema_version", "record_type", "opening_hash", "decision",
            "result_summary", "result_artifact_hash", "actor", "decided_at",
        }
        prospective = payload.get("data_binding_mode") is not None
        expected = (common | {
            "data_binding_mode", "realized_data_artifact_hash",
        } if prospective else common)
        _exact_fields(payload, expected, record="holdout decision")
        if payload["record_type"] != "holdout_decision":
            raise IntegrityViolation("invalid holdout decision type")
        _sha256(payload["opening_hash"], field_name="opening_hash")
        if payload["decision"] not in {"pass", "fail", "invalid"}:
            raise IntegrityViolation("invalid holdout decision")
        _mapping(payload["result_summary"], field_name="result_summary")
        _sha256(payload["result_artifact_hash"], field_name="result_artifact_hash")
        if prospective:
            if payload["data_binding_mode"] != _PROSPECTIVE_BINDING_MODE:
                raise IntegrityViolation("invalid prospective data binding mode")
            _sha256(payload["realized_data_artifact_hash"],
                    field_name="realized_data_artifact_hash")
        _required_text(payload["actor"], field_name="actor")
        _timestamp(payload["decided_at"], field_name="decided_at")


@dataclass(frozen=True)
class _LedgerEvent(ContentRecord):
    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        super()._validate_payload(payload)
        legacy_fields = {
            "schema_version", "record_type", "sequence", "previous_event_hash",
            "event_kind", "target_type", "target_hash",
        }
        actual = set(payload)
        if actual not in (legacy_fields, legacy_fields | {"recorded_at"}):
            _exact_fields(
                payload, legacy_fields | {"recorded_at"}, record="ledger event")
        if payload["record_type"] != "ledger_event":
            raise IntegrityViolation("invalid ledger event type")
        sequence = payload["sequence"]
        if type(sequence) is not int or sequence < 0:
            raise IntegrityViolation("event sequence must be a non-negative integer")
        previous = payload["previous_event_hash"]
        if previous is not None:
            _sha256(previous, field_name="previous_event_hash")
        kind = payload["event_kind"]
        if kind not in _EVENT_KINDS:
            raise IntegrityViolation(f"unsupported ledger event kind: {kind}")
        if payload["target_type"] != _EVENT_KINDS[kind]:
            raise IntegrityViolation("ledger event target type does not match kind")
        _sha256(payload["target_hash"], field_name="target_hash")
        if "recorded_at" in payload:
            _timestamp(payload["recorded_at"], field_name="recorded_at")


@dataclass
class _LedgerState:
    events: list[_LedgerEvent] = field(default_factory=list)
    protocols: dict[str, ResearchProtocol] = field(default_factory=dict)
    protocol_keys: dict[tuple[str, str], str] = field(default_factory=dict)
    trials: dict[str, TrialRegistration] = field(default_factory=dict)
    trial_ids: dict[str, str] = field(default_factory=dict)
    outcomes: dict[str, TrialOutcome] = field(default_factory=dict)
    openings: dict[str, HoldoutOpening] = field(default_factory=dict)
    opening_keys: dict[tuple[str, str], str] = field(default_factory=dict)
    opening_bindings: dict[tuple[str, str], str] = field(default_factory=dict)
    realized_data_hashes: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, HoldoutDecision] = field(default_factory=dict)
    mutation_times: dict[str, str | None] = field(default_factory=dict)


class ResearchIntegrityLedger:
    """Filesystem-backed append-only evidence ledger for one research program."""

    def __init__(
            self, root: str | Path, *, program_id: str,
            _clock: Callable[[], datetime] | None = None) -> None:
        self.root = Path(root)
        self.program_id = _required_text(program_id, field_name="program_id")
        self._clock = _clock or (lambda: datetime.now(timezone.utc))
        self.events_directory = self.root / "events"
        self.records_directory = self.root / "records"
        self.lock_path = self.root / ".ledger.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_directory.mkdir(exist_ok=True)
        self.records_directory.mkdir(exist_ok=True)
        for directory in _RECORD_DIRECTORIES.values():
            (self.records_directory / directory).mkdir(exist_ok=True)
        with self._locked():
            self._ensure_program_identity_unlocked()
            self._load_state_unlocked()

    def _trusted_timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("research-integrity clock must return a datetime")
        return _timestamp(value, field_name="trusted ledger mutation time")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _ensure_program_identity_unlocked(self) -> None:
        path = self.root / "program.json"
        payload = {
            "schema_version": _RECORD_SCHEMA,
            "record_type": "research_program",
            "program_id": self.program_id,
        }
        record = ContentRecord._from_payload(payload)
        if path.exists():
            existing = ContentRecord.from_json(path.read_text(encoding="utf-8"))
            if existing.record_hash != record.record_hash:
                raise IntegrityViolation(
                    "ledger root is already bound to a different research program")
            return
        self._publish_file(path, record.to_json())

    def _record_path(self, record_type: str, record_hash: str) -> Path:
        try:
            directory = _RECORD_DIRECTORIES[record_type]
        except KeyError as exc:
            raise IntegrityViolation(f"unsupported record type: {record_type}") from exc
        return self.records_directory / directory / f"{record_hash}.json"

    def _publish_record_unlocked(self, record: ContentRecord) -> None:
        path = self._record_path(record.record_type, record.record_hash)
        if path.exists():
            expected_type = type(record)
            existing = expected_type.from_json(path.read_text(encoding="utf-8"))
            if existing != record:
                raise IntegrityViolation("content-addressed record path has conflicting data")
            return
        self._publish_file(path, record.to_json())

    @staticmethod
    def _publish_file(path: Path, value: str) -> None:
        """Atomically publish a never-overwritten, read-only file."""
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent,
                    prefix=f".{path.name}.", delete=False) as handle:
                temporary_name = handle.name
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, path)
            except FileExistsError as exc:
                raise IntegrityViolation(f"immutable file already exists: {path}") from exc
            path.chmod(0o444)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def _load_target_unlocked(
            self, record_type: str, record_hash: str) -> ContentRecord:
        classes: dict[str, type[ContentRecord]] = {
            "research_protocol": ResearchProtocol,
            "research_trial": TrialRegistration,
            "trial_outcome": TrialOutcome,
            "holdout_opening": HoldoutOpening,
            "holdout_decision": HoldoutDecision,
        }
        path = self._record_path(record_type, record_hash)
        if not path.is_file():
            raise IntegrityViolation(f"ledger event references missing record: {record_hash}")
        record = classes[record_type].from_json(path.read_text(encoding="utf-8"))
        if record.record_hash != record_hash or record.record_type != record_type:
            raise IntegrityViolation("ledger target identity mismatch")
        return record

    def _event_files_unlocked(self) -> list[Path]:
        return sorted(self.events_directory.glob("*.json"))

    def _load_state_unlocked(self) -> _LedgerState:
        state = _LedgerState()
        previous_hash: str | None = None
        for expected_sequence, path in enumerate(self._event_files_unlocked()):
            event = _LedgerEvent.from_json(path.read_text(encoding="utf-8"))
            payload = event.payload
            sequence = payload["sequence"]
            expected_name = f"{sequence:020d}-{event.record_hash}.json"
            if sequence != expected_sequence or path.name != expected_name:
                raise IntegrityViolation("ledger event sequence is missing, duplicated, or renamed")
            if payload["previous_event_hash"] != previous_hash:
                raise IntegrityViolation("ledger event hash chain is broken")
            mutation_time = payload.get("recorded_at")
            if mutation_time is not None and state.events:
                prior_time = state.events[-1].payload.get("recorded_at")
                if (prior_time is not None
                        and _timestamp_value(
                            mutation_time, field_name="event recorded_at")
                        < _timestamp_value(
                            prior_time, field_name="previous event recorded_at")):
                    raise IntegrityViolation(
                        "trusted ledger mutation timestamps moved backwards")
            target = self._load_target_unlocked(
                payload["target_type"], payload["target_hash"])
            self._apply_event(
                state, payload["event_kind"], target,
                mutation_time=mutation_time,
            )
            state.mutation_times[target.record_hash] = mutation_time
            state.events.append(event)
            previous_hash = event.record_hash
        return state

    @staticmethod
    def _required_mutation_datetime(
            mutation_time: str | None, *, context: str) -> datetime:
        if mutation_time is None:
            raise IntegrityViolation(
                f"{context} lacks a trusted ledger mutation timestamp")
        return _timestamp_value(
            mutation_time, field_name=f"{context} mutation timestamp")

    @staticmethod
    def _prospective_trial_holdout(
            protocol: ResearchProtocol,
            trial: TrialRegistration) -> tuple[str, Mapping[str, Any]] | None:
        prefix = "prospective-commitment:"
        data_version = trial.payload["data_version"]
        if not data_version.startswith(prefix):
            return None
        digest = _sha256(
            data_version[len(prefix):],
            field_name="prospective trial data commitment",
        )
        matches = [
            (holdout_id, specification)
            for holdout_id, specification in protocol.payload["holdouts"].items()
            if (specification.get("mode") == _PROSPECTIVE_HOLDOUT_MODE
                and prospective_data_commitment_hash(specification) == digest)
        ]
        if len(matches) != 1:
            raise IntegrityViolation(
                "prospective trial data commitment must match exactly one holdout")
        return matches[0]

    @staticmethod
    def _trial_openings(
            state: _LedgerState, trial_hash: str) -> list[HoldoutOpening]:
        return [
            opening for opening in state.openings.values()
            if opening.payload["trial_hash"] == trial_hash
        ]

    def _apply_event(
            self, state: _LedgerState, kind: str, target: ContentRecord, *,
            mutation_time: str | None) -> None:
        if kind == "protocol_frozen":
            protocol = target
            assert isinstance(protocol, ResearchProtocol)
            payload = protocol.payload
            if payload["program_id"] != self.program_id:
                raise IntegrityViolation("protocol belongs to a different research program")
            key = (payload["protocol_id"], payload["version"])
            if protocol.record_hash in state.protocols or key in state.protocol_keys:
                raise IntegrityViolation("duplicate or conflicting frozen protocol event")
            prospective_holdouts = [
                specification for specification in payload["holdouts"].values()
                if specification.get("mode") == _PROSPECTIVE_HOLDOUT_MODE
            ]
            if prospective_holdouts:
                event_time = self._required_mutation_datetime(
                    mutation_time, context="prospective protocol freeze")
                frozen_time = _timestamp_value(
                    payload["frozen_at"], field_name="frozen_at")
                if frozen_time > event_time:
                    raise IntegrityViolation(
                        "prospective protocol frozen_at postdates its mutation")
                for specification in prospective_holdouts:
                    start = _calendar_date(
                        specification["start"], field_name="holdout start")
                    start_boundary = datetime(
                        start.year, start.month, start.day, tzinfo=timezone.utc)
                    if event_time >= start_boundary:
                        raise IntegrityViolation(
                            "prospective protocol was frozen after observations began")
            state.protocols[protocol.record_hash] = protocol
            state.protocol_keys[key] = protocol.record_hash
            return

        if kind == "trial_registered":
            trial = target
            assert isinstance(trial, TrialRegistration)
            payload = trial.payload
            if payload["protocol_hash"] not in state.protocols:
                raise IntegrityViolation("trial references a protocol not yet frozen")
            if any(protocol_hash == payload["protocol_hash"]
                   for protocol_hash, _holdout_id in state.opening_keys):
                raise IntegrityViolation(
                    "trial was registered after a holdout opened for its protocol")
            protocol = state.protocols[payload["protocol_hash"]]
            if payload["code_version"] != protocol.payload["code_version"]:
                raise IntegrityViolation(
                    "trial code_version differs from its frozen protocol")
            prospective_holdout = self._prospective_trial_holdout(protocol, trial)
            if prospective_holdout is not None:
                _holdout_id, specification = prospective_holdout
                event_time = self._required_mutation_datetime(
                    mutation_time, context="prospective trial registration")
                protocol_event_time = self._required_mutation_datetime(
                    state.mutation_times.get(protocol.record_hash),
                    context="prospective protocol freeze",
                )
                registered_time = _timestamp_value(
                    payload["registered_at"], field_name="registered_at")
                frozen_time = _timestamp_value(
                    protocol.payload["frozen_at"], field_name="frozen_at")
                start = _calendar_date(
                    specification["start"], field_name="holdout start")
                start_boundary = datetime(
                    start.year, start.month, start.day, tzinfo=timezone.utc)
                if not frozen_time <= registered_time <= event_time:
                    raise IntegrityViolation(
                        "prospective trial timestamps are out of order")
                if not protocol_event_time <= event_time < start_boundary:
                    raise IntegrityViolation(
                        "prospective trial was not registered before observations")
            candidate_ids = {
                item["candidate_id"]
                for item in protocol.payload["candidate_specifications"]
            }
            if payload["candidate_id"] not in candidate_ids:
                raise IntegrityViolation("trial candidate is not declared by its protocol")
            if trial.record_hash in state.trials or payload["trial_id"] in state.trial_ids:
                raise IntegrityViolation("duplicate or conflicting trial registration event")
            state.trials[trial.record_hash] = trial
            state.trial_ids[payload["trial_id"]] = trial.record_hash
            return

        if kind == "trial_outcome_recorded":
            outcome = target
            assert isinstance(outcome, TrialOutcome)
            trial_hash = outcome.payload["trial_hash"]
            if trial_hash not in state.trials:
                raise IntegrityViolation("trial outcome references an unknown trial")
            if trial_hash in state.outcomes:
                raise IntegrityViolation("trial has more than one terminal outcome")
            openings = self._trial_openings(state, trial_hash)
            if any(opening.opening_hash in state.decisions for opening in openings):
                raise IntegrityViolation(
                    "trial outcome was recorded after its holdout decision")
            trial = state.trials[trial_hash]
            protocol = state.protocols[trial.payload["protocol_hash"]]
            prospective_holdout = self._prospective_trial_holdout(protocol, trial)
            if prospective_holdout is not None:
                if len(openings) != 1 or openings[0].payload.get(
                        "data_binding_mode") != _PROSPECTIVE_BINDING_MODE:
                    raise IntegrityViolation(
                        "prospective trial outcome requires exactly one opening")
                opening = openings[0]
                event_time = self._required_mutation_datetime(
                    mutation_time, context="prospective trial outcome")
                opening_event_time = self._required_mutation_datetime(
                    state.mutation_times.get(opening.opening_hash),
                    context="prospective holdout opening",
                )
                recorded_time = _timestamp_value(
                    outcome.payload["recorded_at"], field_name="recorded_at")
                opened_time = _timestamp_value(
                    opening.payload["opened_at"], field_name="opened_at")
                if not opened_time <= recorded_time <= event_time:
                    raise IntegrityViolation(
                        "prospective outcome timestamps are out of order")
                if opening_event_time > event_time:
                    raise IntegrityViolation(
                        "prospective outcome mutation predates its opening")
                if outcome.payload["status"] == "completed":
                    _holdout_id, specification = prospective_holdout
                    end = _calendar_date(
                        specification["end"], field_name="holdout end")
                    end_boundary = datetime(
                        end.year, end.month, end.day, tzinfo=timezone.utc,
                    ) + timedelta(days=1)
                    if recorded_time < end_boundary or event_time < end_boundary:
                        raise IntegrityViolation(
                            "completed prospective outcome predates holdout end")
            state.outcomes[trial_hash] = outcome
            return

        if kind == "holdout_opened":
            opening = target
            assert isinstance(opening, HoldoutOpening)
            payload = opening.payload
            protocol_hash = payload["protocol_hash"]
            trial_hash = payload["trial_hash"]
            if protocol_hash not in state.protocols or trial_hash not in state.trials:
                raise IntegrityViolation("holdout opening references unknown evidence")
            trial = state.trials[trial_hash]
            if trial.payload["protocol_hash"] != protocol_hash:
                raise IntegrityViolation("holdout trial belongs to a different protocol")
            protocol = state.protocols[protocol_hash]
            holdout_id = payload["holdout_id"]
            if holdout_id not in protocol.payload["holdouts"]:
                raise IntegrityViolation("holdout was not declared in the frozen protocol")
            expected_spec_hash = content_hash(protocol.payload["holdouts"][holdout_id])
            if payload["holdout_spec_hash"] != expected_spec_hash:
                raise IntegrityViolation("holdout specification hash mismatch")
            holdout_specification = protocol.payload["holdouts"][holdout_id]
            prospective = (
                holdout_specification.get("mode")
                == _PROSPECTIVE_HOLDOUT_MODE
            )
            opening_is_prospective = (
                payload.get("data_binding_mode")
                == _PROSPECTIVE_BINDING_MODE
            )
            if prospective != opening_is_prospective:
                raise IntegrityViolation(
                    "holdout opening data binding differs from its protocol")
            if prospective:
                expected_commitment = prospective_data_commitment_hash(
                    holdout_specification)
                if payload["data_commitment_hash"] != expected_commitment:
                    raise IntegrityViolation(
                        "prospective opening commitment hash mismatch")
                if trial.payload["data_version"] != (
                        f"prospective-commitment:{expected_commitment}"):
                    raise IntegrityViolation(
                        "prospective trial is not bound to its data commitment")
                if trial_hash in state.outcomes:
                    raise IntegrityViolation(
                        "prospective trial had an outcome before its opening")
                protocol_time = _timestamp_value(
                    protocol.payload["frozen_at"], field_name="frozen_at")
                trial_time = _timestamp_value(
                    trial.payload["registered_at"], field_name="registered_at")
                opening_time = _timestamp_value(
                    payload["opened_at"], field_name="opened_at")
                opening_event_time = self._required_mutation_datetime(
                    mutation_time, context="prospective holdout opening")
                protocol_event_time = self._required_mutation_datetime(
                    state.mutation_times.get(protocol.record_hash),
                    context="prospective protocol freeze",
                )
                trial_event_time = self._required_mutation_datetime(
                    state.mutation_times.get(trial.record_hash),
                    context="prospective trial registration",
                )
                start = _calendar_date(
                    holdout_specification["start"], field_name="holdout start")
                start_boundary = datetime(
                    start.year, start.month, start.day, tzinfo=timezone.utc)
                if not protocol_time <= trial_time <= opening_time:
                    raise IntegrityViolation(
                        "prospective protocol, trial, and opening timestamps "
                        "are out of order")
                if not (protocol_event_time <= trial_event_time
                        <= opening_event_time):
                    raise IntegrityViolation(
                        "trusted prospective mutation timestamps are out of order")
                if opening_time > opening_event_time:
                    raise IntegrityViolation(
                        "prospective opened_at postdates its ledger mutation")
                if (opening_time >= start_boundary
                        or opening_event_time >= start_boundary):
                    raise IntegrityViolation(
                        "prospective holdout was not opened before its first "
                        "observation")
            else:
                declared_hash = holdout_specification.get("data_artifact_hash")
                if declared_hash is not None and _sha256(
                        declared_hash, field_name="declared data_artifact_hash") \
                        != payload["data_artifact_hash"]:
                    raise IntegrityViolation(
                        "fixed holdout artifact differs from its protocol")
            expected_program_count = len(state.trials)
            expected_protocol_count = sum(
                trial_record.payload["protocol_hash"] == protocol_hash
                for trial_record in state.trials.values())
            if (payload["program_trial_count_at_open"] != expected_program_count
                    or payload["protocol_trial_count_at_open"]
                    != expected_protocol_count):
                raise IntegrityViolation("holdout trial counts were not ledger-derived")
            key = (protocol_hash, holdout_id)
            if key in state.opening_keys or opening.record_hash in state.openings:
                raise IntegrityViolation("holdout has more than one opening event")
            binding_key = (
                (_PROSPECTIVE_BINDING_MODE, payload["data_commitment_hash"])
                if prospective else
                (_FIXED_HOLDOUT_MODE, payload["data_artifact_hash"])
            )
            if binding_key in state.opening_bindings:
                raise IntegrityViolation(
                    "holdout data binding was already consumed by another opening")
            state.openings[opening.record_hash] = opening
            state.opening_keys[key] = opening.record_hash
            state.opening_bindings[binding_key] = opening.record_hash
            return

        if kind == "holdout_decided":
            decision = target
            assert isinstance(decision, HoldoutDecision)
            opening_hash = decision.payload["opening_hash"]
            if opening_hash not in state.openings:
                raise IntegrityViolation("holdout decision references an unknown opening")
            if opening_hash in state.decisions:
                raise IntegrityViolation("holdout has more than one decision")
            opening = state.openings[opening_hash]
            opening_is_prospective = (
                opening.payload.get("data_binding_mode")
                == _PROSPECTIVE_BINDING_MODE
            )
            decision_is_prospective = (
                decision.payload.get("data_binding_mode")
                == _PROSPECTIVE_BINDING_MODE
            )
            if opening_is_prospective != decision_is_prospective:
                raise IntegrityViolation(
                    "holdout decision data binding differs from its opening")
            if opening_is_prospective:
                opening_time = _timestamp_value(
                    opening.payload["opened_at"], field_name="opened_at")
                decision_time = _timestamp_value(
                    decision.payload["decided_at"], field_name="decided_at")
                decision_event_time = self._required_mutation_datetime(
                    mutation_time, context="prospective holdout decision")
                opening_event_time = self._required_mutation_datetime(
                    state.mutation_times.get(opening.opening_hash),
                    context="prospective holdout opening",
                )
                if decision_time < opening_time:
                    raise IntegrityViolation(
                        "prospective decision predates its opening")
                if (decision_time > decision_event_time
                        or opening_event_time > decision_event_time):
                    raise IntegrityViolation(
                        "prospective decision timestamps are out of order")
                realized_hash = decision.payload["realized_data_artifact_hash"]
                if realized_hash in state.realized_data_hashes:
                    raise IntegrityViolation(
                        "realized prospective data was already finalized")
                if decision.payload["decision"] in {"pass", "fail"}:
                    protocol = state.protocols[opening.payload["protocol_hash"]]
                    holdout = protocol.payload["holdouts"][
                        opening.payload["holdout_id"]]
                    end = _calendar_date(holdout["end"], field_name="holdout end")
                    end_boundary = datetime(
                        end.year, end.month, end.day, tzinfo=timezone.utc,
                    ) + timedelta(days=1)
                    if (decision_time < end_boundary
                            or decision_event_time < end_boundary):
                        raise IntegrityViolation(
                            "prospective pass/fail decision predates holdout end")
                    trial_hash = opening.payload["trial_hash"]
                    outcome = state.outcomes.get(trial_hash)
                    if (outcome is None
                            or outcome.payload["status"] != "completed"):
                        raise IntegrityViolation(
                            "prospective pass/fail requires a completed trial outcome")
                    if (outcome.payload["evidence_hash"]
                            != decision.payload["result_artifact_hash"]):
                        raise IntegrityViolation(
                            "prospective decision evidence differs from its outcome")
                    if canonical_json(outcome.payload["result_summary"]) != \
                            canonical_json(decision.payload["result_summary"]):
                        raise IntegrityViolation(
                            "prospective decision summary differs from its outcome")
                    outcome_time = _timestamp_value(
                        outcome.payload["recorded_at"], field_name="recorded_at")
                    outcome_event_time = self._required_mutation_datetime(
                        state.mutation_times.get(outcome.record_hash),
                        context="prospective trial outcome",
                    )
                    if not (end_boundary <= outcome_time <= decision_time
                            and outcome_event_time <= decision_event_time):
                        raise IntegrityViolation(
                            "prospective outcome/decision timestamps are out of order")
                state.realized_data_hashes[realized_hash] = decision.record_hash
            state.decisions[opening_hash] = decision
            return

        raise IntegrityViolation(f"unsupported ledger event kind: {kind}")

    def _append_event_unlocked(
            self, state: _LedgerState, kind: str,
            target: ContentRecord, *, mutation_time: str) -> _LedgerEvent:
        sequence = len(state.events)
        payload = {
            "schema_version": _RECORD_SCHEMA,
            "record_type": "ledger_event",
            "sequence": sequence,
            "previous_event_hash": (
                state.events[-1].record_hash if state.events else None),
            "event_kind": kind,
            "target_type": target.record_type,
            "target_hash": target.record_hash,
            "recorded_at": mutation_time,
        }
        event = _LedgerEvent._from_payload(payload)
        if state.events:
            prior_time = state.events[-1].payload.get("recorded_at")
            if (prior_time is not None
                    and _timestamp_value(
                        mutation_time, field_name="event recorded_at")
                    < _timestamp_value(
                        prior_time, field_name="previous event recorded_at")):
                raise IntegrityViolation(
                    "trusted ledger mutation timestamps moved backwards")
        self._apply_event(
            state, kind, target, mutation_time=mutation_time)
        state.mutation_times[target.record_hash] = mutation_time
        self._publish_record_unlocked(target)
        path = self.events_directory / (
            f"{sequence:020d}-{event.record_hash}.json")
        self._publish_file(path, event.to_json())
        state.events.append(event)
        return event

    def freeze_protocol(self, protocol: ResearchProtocol) -> ResearchProtocol:
        """Persist *protocol* without allowing its ID/version to be rebound."""
        if not isinstance(protocol, ResearchProtocol):
            raise TypeError("protocol must be a ResearchProtocol")
        with self._locked():
            state = self._load_state_unlocked()
            payload = protocol.payload
            if payload["program_id"] != self.program_id:
                raise ValueError("protocol program_id does not match this ledger")
            key = (payload["protocol_id"], payload["version"])
            existing_hash = state.protocol_keys.get(key)
            if existing_hash is not None:
                if existing_hash == protocol.record_hash:
                    return state.protocols[existing_hash]
                raise DuplicateRecordError(
                    "protocol ID/version is already frozen with different content")
            self._append_event_unlocked(
                state, "protocol_frozen", protocol,
                mutation_time=self._trusted_timestamp(),
            )
            return protocol

    def register_trial(self, trial: TrialRegistration) -> TrialRegistration:
        """Append one attempted trial; exact retries are idempotent."""
        if not isinstance(trial, TrialRegistration):
            raise TypeError("trial must be a TrialRegistration")
        with self._locked():
            state = self._load_state_unlocked()
            protocol_hash = trial.payload["protocol_hash"]
            if protocol_hash not in state.protocols:
                raise ProtocolNotFrozen(
                    "trial cannot be registered before its protocol is frozen")
            existing_hash = state.trial_ids.get(trial.payload["trial_id"])
            if existing_hash is not None:
                if existing_hash == trial.record_hash:
                    return state.trials[existing_hash]
                raise DuplicateRecordError(
                    "trial_id is already registered with different content")
            if any(opened_protocol == protocol_hash
                   for opened_protocol, _holdout_id in state.opening_keys):
                raise HoldoutAlreadyOpened(
                    "cannot register another trial after this protocol's "
                    "holdout has opened")
            candidate_ids = {
                item["candidate_id"] for item in
                state.protocols[protocol_hash].payload["candidate_specifications"]
            }
            if trial.payload["candidate_id"] not in candidate_ids:
                raise ValueError("trial candidate_id is not declared by its protocol")
            protocol = state.protocols[protocol_hash]
            if trial.payload["code_version"] != protocol.payload["code_version"]:
                raise ValueError(
                    "trial code_version differs from its frozen protocol")
            self._append_event_unlocked(
                state, "trial_registered", trial,
                mutation_time=self._trusted_timestamp(),
            )
            return trial

    def record_trial_outcome(self, outcome: TrialOutcome) -> TrialOutcome:
        """Append the only terminal outcome permitted for one trial."""
        if not isinstance(outcome, TrialOutcome):
            raise TypeError("outcome must be a TrialOutcome")
        with self._locked():
            state = self._load_state_unlocked()
            trial_hash = outcome.payload["trial_hash"]
            if trial_hash not in state.trials:
                raise UnknownRecordError("cannot record an outcome for an unknown trial")
            existing = state.outcomes.get(trial_hash)
            if existing is not None:
                if existing.record_hash == outcome.record_hash:
                    return existing
                raise DuplicateRecordError(
                    "trial already has a different terminal outcome")
            openings = self._trial_openings(state, trial_hash)
            if any(opening.opening_hash in state.decisions for opening in openings):
                raise HoldoutAlreadyDecided(
                    "cannot record a trial outcome after its holdout decision")
            trial = state.trials[trial_hash]
            protocol = state.protocols[trial.payload["protocol_hash"]]
            prospective_holdout = self._prospective_trial_holdout(protocol, trial)
            mutation_time = self._trusted_timestamp()
            if prospective_holdout is not None:
                if len(openings) != 1 or openings[0].payload.get(
                        "data_binding_mode") != _PROSPECTIVE_BINDING_MODE:
                    raise ValueError(
                        "prospective trial outcome requires exactly one opening")
                opening = openings[0]
                event_time = _timestamp_value(
                    mutation_time, field_name="trusted outcome time")
                recorded_time = _timestamp_value(
                    outcome.payload["recorded_at"], field_name="recorded_at")
                opened_time = _timestamp_value(
                    opening.payload["opened_at"], field_name="opened_at")
                if not opened_time <= recorded_time <= event_time:
                    raise ValueError(
                        "prospective outcome timestamps must follow the opening "
                        "and not postdate the mutation")
                if outcome.payload["status"] == "completed":
                    _holdout_id, specification = prospective_holdout
                    end = _calendar_date(
                        specification["end"], field_name="holdout end")
                    end_boundary = datetime(
                        end.year, end.month, end.day, tzinfo=timezone.utc,
                    ) + timedelta(days=1)
                    if recorded_time < end_boundary or event_time < end_boundary:
                        raise ValueError(
                            "completed prospective outcome cannot predate "
                            "holdout end")
            self._append_event_unlocked(
                state, "trial_outcome_recorded", outcome,
                mutation_time=mutation_time,
            )
            return outcome

    def get_protocol(self, protocol_hash: str) -> ResearchProtocol:
        """Return one verified frozen protocol by content hash."""
        digest = _sha256(protocol_hash, field_name="protocol_hash")
        with self._locked():
            state = self._load_state_unlocked()
            try:
                return state.protocols[digest]
            except KeyError as exc:
                raise ProtocolNotFrozen("unknown frozen protocol") from exc

    def get_trial(self, trial_hash: str) -> TrialRegistration:
        """Return one verified registered trial by content hash."""
        digest = _sha256(trial_hash, field_name="trial_hash")
        with self._locked():
            state = self._load_state_unlocked()
            try:
                return state.trials[digest]
            except KeyError as exc:
                raise UnknownRecordError("unknown research trial") from exc

    def get_trial_outcome(self, trial_hash: str) -> TrialOutcome | None:
        """Return the terminal outcome, or ``None`` for an unfinished trial."""
        digest = _sha256(trial_hash, field_name="trial_hash")
        with self._locked():
            state = self._load_state_unlocked()
            if digest not in state.trials:
                raise UnknownRecordError("unknown research trial")
            return state.outcomes.get(digest)

    @property
    def trial_count(self) -> int:
        """Program-wide number of attempted trials, derived from the journal."""
        with self._locked():
            return len(self._load_state_unlocked().trials)

    def protocol_trial_count(self, protocol_hash: str) -> int:
        """Derived number of attempts registered under one frozen protocol."""
        digest = _sha256(protocol_hash, field_name="protocol_hash")
        with self._locked():
            state = self._load_state_unlocked()
            if digest not in state.protocols:
                raise ProtocolNotFrozen("unknown frozen protocol")
            return sum(
                trial.payload["protocol_hash"] == digest
                for trial in state.trials.values())

    def open_holdout(
            self, *, protocol_hash: str, holdout_id: str, trial_hash: str,
            actor: str, data_artifact_hash: str | None = None,
            data_commitment_hash: str | None = None,
            opened_at: str | datetime | None = None) -> HoldoutOpening:
        """Consume one declared holdout and permanently record the exposure.

        Trial counts are captured from the verified journal while its exclusive
        lock is held; callers cannot provide or override them. Fixed snapshots
        bind their realized bytes at opening. Prospective holdouts instead bind
        only the frozen acquisition commitment and reject realized data until
        the permanent decision.
        """
        protocol_digest = _sha256(protocol_hash, field_name="protocol_hash")
        trial_digest = _sha256(trial_hash, field_name="trial_hash")
        holdout_name = _required_text(holdout_id, field_name="holdout_id")
        data_digest = (_sha256(data_artifact_hash, field_name="data_artifact_hash")
                       if data_artifact_hash is not None else None)
        commitment_digest = (
            _sha256(data_commitment_hash, field_name="data_commitment_hash")
            if data_commitment_hash is not None else None)
        normalized_actor = _required_text(actor, field_name="actor")
        normalized_time = _timestamp(opened_at, field_name="opened_at")
        with self._locked():
            state = self._load_state_unlocked()
            if protocol_digest not in state.protocols:
                raise ProtocolNotFrozen(
                    "holdout cannot be opened before its protocol is frozen")
            protocol = state.protocols[protocol_digest]
            if holdout_name not in protocol.payload["holdouts"]:
                raise ValueError("holdout_id is not declared by the frozen protocol")
            key = (protocol_digest, holdout_name)
            if key in state.opening_keys:
                raise HoldoutAlreadyOpened(
                    "this holdout was already opened and cannot be reopened")
            if trial_digest not in state.trials:
                raise UnknownRecordError("holdout references an unknown trial")
            trial = state.trials[trial_digest]
            if trial.payload["protocol_hash"] != protocol_digest:
                raise ValueError("holdout trial belongs to a different protocol")
            holdout_specification = protocol.payload["holdouts"][holdout_name]
            prospective = (
                holdout_specification.get("mode")
                == _PROSPECTIVE_HOLDOUT_MODE
            )
            mutation_time = self._trusted_timestamp()
            mutation_datetime = _timestamp_value(
                mutation_time, field_name="trusted opening time")
            if prospective:
                if data_digest is not None:
                    raise ValueError(
                        "prospective holdout opening cannot receive realized data")
                if trial_digest in state.outcomes:
                    raise ValueError(
                        "prospective trial cannot have an outcome before opening")
                expected_commitment = prospective_data_commitment_hash(
                    holdout_specification)
                if commitment_digest is None:
                    raise ValueError(
                        "prospective holdout requires its data commitment hash")
                if commitment_digest != expected_commitment:
                    raise ValueError(
                        "opened holdout does not match the preregistered "
                        "data commitment")
                if trial.payload["data_version"] != (
                        f"prospective-commitment:{expected_commitment}"):
                    raise ValueError(
                        "prospective trial is not bound to its data commitment")
                protocol_time = _timestamp_value(
                    protocol.payload["frozen_at"], field_name="frozen_at")
                trial_time = _timestamp_value(
                    trial.payload["registered_at"], field_name="registered_at")
                opening_time = _timestamp_value(
                    normalized_time, field_name="opened_at")
                start = _calendar_date(
                    holdout_specification["start"], field_name="holdout start")
                start_boundary = datetime(
                    start.year, start.month, start.day, tzinfo=timezone.utc)
                if not protocol_time <= trial_time <= opening_time:
                    raise ValueError(
                        "prospective protocol, trial, and opening timestamps "
                        "must be ordered")
                if (opening_time >= start_boundary
                        or mutation_datetime >= start_boundary):
                    raise ValueError(
                        "prospective holdout must open before its first observation")
                if opening_time > mutation_datetime:
                    raise ValueError(
                        "prospective opened_at cannot postdate its ledger mutation")
            else:
                if commitment_digest is not None:
                    raise ValueError(
                        "fixed holdout opening cannot use a data commitment")
                if data_digest is None:
                    raise ValueError(
                        "fixed holdout requires its realized data artifact hash")
                declared_hash = holdout_specification.get("data_artifact_hash")
                if declared_hash is not None and _sha256(
                        declared_hash, field_name="declared data_artifact_hash") \
                        != data_digest:
                    raise ValueError(
                        "opened holdout data does not match the preregistered digest")
            binding_key = (
                (_PROSPECTIVE_BINDING_MODE, commitment_digest)
                if prospective else (_FIXED_HOLDOUT_MODE, data_digest)
            )
            if binding_key in state.opening_bindings:
                raise HoldoutAlreadyOpened(
                    "this holdout data binding was already consumed")
            protocol_count = sum(
                candidate.payload["protocol_hash"] == protocol_digest
                for candidate in state.trials.values())
            opening_payload = {
                "schema_version": _RECORD_SCHEMA,
                "record_type": "holdout_opening",
                "protocol_hash": protocol_digest,
                "holdout_id": holdout_name,
                "holdout_spec_hash": content_hash(
                    protocol.payload["holdouts"][holdout_name]),
                "trial_hash": trial_digest,
                "actor": normalized_actor,
                "program_trial_count_at_open": len(state.trials),
                "protocol_trial_count_at_open": protocol_count,
                "opened_at": normalized_time,
            }
            if prospective:
                opening_payload.update({
                    "data_binding_mode": _PROSPECTIVE_BINDING_MODE,
                    "data_commitment_hash": commitment_digest,
                })
            else:
                opening_payload["data_artifact_hash"] = data_digest
            opening = HoldoutOpening._from_payload(opening_payload)
            self._append_event_unlocked(
                state, "holdout_opened", opening,
                mutation_time=mutation_time,
            )
            return opening

    def record_holdout_decision(
            self, *, opening_hash: str, decision: str,
            result_summary: Mapping[str, Any], result_artifact_hash: str,
            actor: str, decided_at: str | datetime | None = None,
            realized_data_artifact_hash: str | None = None,
            ) -> HoldoutDecision:
        """Append the sole permanent decision/result for one opening."""
        opening_digest = _sha256(opening_hash, field_name="opening_hash")
        normalized_decision = _required_text(
            decision, field_name="decision").lower()
        if normalized_decision not in {"pass", "fail", "invalid"}:
            raise ValueError("decision must be pass, fail, or invalid")
        normalized_result = _mapping(result_summary, field_name="result_summary")
        result_digest = _sha256(
            result_artifact_hash, field_name="result_artifact_hash")
        normalized_actor = _required_text(actor, field_name="actor")
        normalized_time = _timestamp(decided_at, field_name="decided_at")
        realized_digest = (
            _sha256(realized_data_artifact_hash,
                    field_name="realized_data_artifact_hash")
            if realized_data_artifact_hash is not None else None)
        with self._locked():
            state = self._load_state_unlocked()
            if opening_digest not in state.openings:
                raise UnknownRecordError("cannot decide an unknown holdout opening")
            if opening_digest in state.decisions:
                raise HoldoutAlreadyDecided(
                    "this holdout already has a permanent decision")
            opening = state.openings[opening_digest]
            prospective = (
                opening.payload.get("data_binding_mode")
                == _PROSPECTIVE_BINDING_MODE
            )
            mutation_time = self._trusted_timestamp()
            mutation_datetime = _timestamp_value(
                mutation_time, field_name="trusted decision time")
            if prospective:
                if realized_digest is None:
                    raise ValueError(
                        "prospective decision requires the realized data "
                        "artifact hash")
                opening_time = _timestamp_value(
                    opening.payload["opened_at"], field_name="opened_at")
                decision_time = _timestamp_value(
                    normalized_time, field_name="decided_at")
                if decision_time < opening_time:
                    raise ValueError("prospective decision cannot predate its opening")
                if decision_time > mutation_datetime:
                    raise ValueError(
                        "prospective decided_at cannot postdate its ledger mutation")
                if realized_digest in state.realized_data_hashes:
                    raise ValueError(
                        "realized prospective data was already finalized")
                if normalized_decision in {"pass", "fail"}:
                    protocol = state.protocols[opening.payload["protocol_hash"]]
                    holdout = protocol.payload["holdouts"][
                        opening.payload["holdout_id"]]
                    end = _calendar_date(holdout["end"], field_name="holdout end")
                    end_boundary = datetime(
                        end.year, end.month, end.day, tzinfo=timezone.utc,
                    ) + timedelta(days=1)
                    if (decision_time < end_boundary
                            or mutation_datetime < end_boundary):
                        raise ValueError(
                            "prospective pass/fail decision cannot predate "
                            "holdout end")
                    trial_hash = opening.payload["trial_hash"]
                    outcome = state.outcomes.get(trial_hash)
                    if (outcome is None
                            or outcome.payload["status"] != "completed"):
                        raise ValueError(
                            "prospective pass/fail requires a completed "
                            "trial outcome")
                    if outcome.payload["evidence_hash"] != result_digest:
                        raise ValueError(
                            "prospective decision evidence differs from its outcome")
                    if canonical_json(outcome.payload["result_summary"]) != \
                            canonical_json(normalized_result):
                        raise ValueError(
                            "prospective decision summary differs from its outcome")
                    outcome_time = _timestamp_value(
                        outcome.payload["recorded_at"], field_name="recorded_at")
                    if not end_boundary <= outcome_time <= decision_time:
                        raise ValueError(
                            "prospective outcome/decision timestamps are out of order")
            elif realized_digest is not None:
                raise ValueError(
                    "fixed holdout decision cannot attach prospective data")
            decision_payload = {
                "schema_version": _RECORD_SCHEMA,
                "record_type": "holdout_decision",
                "opening_hash": opening_digest,
                "decision": normalized_decision,
                "result_summary": normalized_result,
                "result_artifact_hash": result_digest,
                "actor": normalized_actor,
                "decided_at": normalized_time,
            }
            if prospective:
                decision_payload.update({
                    "data_binding_mode": _PROSPECTIVE_BINDING_MODE,
                    "realized_data_artifact_hash": realized_digest,
                })
            result = HoldoutDecision._from_payload(decision_payload)
            self._append_event_unlocked(
                state, "holdout_decided", result,
                mutation_time=mutation_time,
            )
            return result

    def get_holdout_decision(
            self, *, protocol_hash: str,
            holdout_id: str) -> HoldoutDecision | None:
        """Return the permanent decision, or ``None`` while still pending."""
        protocol_digest = _sha256(protocol_hash, field_name="protocol_hash")
        holdout_name = _required_text(holdout_id, field_name="holdout_id")
        with self._locked():
            state = self._load_state_unlocked()
            opening_hash = state.opening_keys.get((protocol_digest, holdout_name))
            if opening_hash is None:
                return None
            return state.decisions.get(opening_hash)

    def get_holdout_opening(
            self, *, protocol_hash: str,
            holdout_id: str) -> HoldoutOpening | None:
        """Return an existing opening so an interrupted evaluation can resume."""
        protocol_digest = _sha256(protocol_hash, field_name="protocol_hash")
        holdout_name = _required_text(holdout_id, field_name="holdout_id")
        with self._locked():
            state = self._load_state_unlocked()
            opening_hash = state.opening_keys.get((protocol_digest, holdout_name))
            return (state.openings[opening_hash]
                    if opening_hash is not None else None)

    @property
    def head_hash(self) -> str | None:
        """Latest verified event hash, suitable for an external immutable anchor."""
        with self._locked():
            state = self._load_state_unlocked()
            return state.events[-1].record_hash if state.events else None

    def event_hash_for(self, *, event_kind: str, target_hash: str) -> str:
        """Return the chain checkpoint that appended one target record."""
        if event_kind not in _EVENT_KINDS:
            raise ValueError("unsupported research-integrity event kind")
        digest = _sha256(target_hash, field_name="target_hash")
        with self._locked():
            state = self._load_state_unlocked()
            for event in state.events:
                payload = event.payload
                if (payload["event_kind"] == event_kind
                        and payload["target_hash"] == digest):
                    return event.record_hash
        raise UnknownRecordError("target has no matching ledger event")

    def verify(self, *, expected_head_hash: str | None = None) -> dict[str, Any]:
        """Verify every event and reference and return an auditable summary."""
        expected = (
            _sha256(expected_head_hash, field_name="expected_head_hash")
            if expected_head_hash is not None else None)
        with self._locked():
            state = self._load_state_unlocked()
            head = state.events[-1].record_hash if state.events else None
            if expected is not None and head != expected:
                raise IntegrityViolation("ledger head does not match external anchor")
            return {
                "program_id": self.program_id,
                "event_count": len(state.events),
                "protocol_count": len(state.protocols),
                "trial_count": len(state.trials),
                "trial_outcome_count": len(state.outcomes),
                "holdout_opening_count": len(state.openings),
                "holdout_decision_count": len(state.decisions),
                "head_hash": head,
            }


__all__ = [
    "ContentRecord",
    "DuplicateRecordError",
    "HoldoutAlreadyDecided",
    "HoldoutAlreadyOpened",
    "HoldoutDecision",
    "HoldoutOpening",
    "IntegrityViolation",
    "ProtocolNotFrozen",
    "ResearchIntegrityError",
    "ResearchIntegrityLedger",
    "ResearchProtocol",
    "TrialOutcome",
    "TrialRegistration",
    "UnknownRecordError",
    "canonical_json",
    "content_hash",
    "prospective_data_commitment_hash",
]
