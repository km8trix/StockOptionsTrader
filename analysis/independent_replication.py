"""Independent implementation reconciliation with content-addressed evidence.

The research protocol and data snapshot say *what* was tested and against
which immutable inputs.  This module supplies a separate contract for proving
that two independently identified implementations produced the same keyed
result.  It deliberately has no dependency on the backtest, promotion, or
research-ledger modules.

Each observation represents one canonical portfolio checkpoint (normally a
date/symbol pair) and contains the full money-path progression::

    signal -> eligibility -> rank -> target -> order -> position -> cash -> fees -> pnl

Eligibility is compared exactly.  Every numeric field has a mandatory,
field-specific absolute and relative tolerance.  Reconciliation records all
schema, duplicate, missing, non-finite, and value discrepancies instead of
stopping at the first error.  The complete normalized outputs and discrepancy
records are included in an immutable, content-addressed evidence artifact.
Both implementations must exactly cover a nonempty observation-key manifest
frozen into the contract, so agreeing empty or incomplete outputs cannot pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
NUMERIC_FIELDS = (
    "signal", "rank", "target", "order", "position", "cash", "fees", "pnl",
)
OBSERVATION_FIELDS = frozenset(("key", "eligibility", *NUMERIC_FIELDS))
_NONFINITE_MARKER = "__replication_nonfinite__"
_HEX = frozenset("0123456789abcdef")


class ReplicationError(RuntimeError):
    """Base class for independent-replication failures."""


class ReplicationIntegrityError(ReplicationError):
    """Stored evidence is malformed, tampered with, or internally inconsistent."""


def _strict_json_loads(value: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ReplicationIntegrityError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(token: str) -> None:
        raise ReplicationIntegrityError(f"invalid JSON number: {token}")

    try:
        return json.loads(
            value,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except ReplicationIntegrityError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReplicationIntegrityError("invalid replication evidence JSON") from exc


def _nonfinite(value: Real) -> dict[str, str]:
    number = float(value)
    if math.isnan(number):
        label = "NaN"
    elif number > 0:
        label = "Infinity"
    else:
        label = "-Infinity"
    return {_NONFINITE_MARKER: label}


def _is_nonfinite_marker(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {_NONFINITE_MARKER}
        and value[_NONFINITE_MARKER] in {"NaN", "Infinity", "-Infinity"}
    )


def _evidence_value(value: Any) -> Any:
    """Normalize arbitrary evidence, tagging non-finite numbers explicitly.

    A failed replication still needs an artifact.  JSON cannot represent NaN
    or infinity canonically, so those invalid values are retained as explicit
    tags and are also emitted as discrepancies.
    """
    if isinstance(value, Enum):
        return _evidence_value(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return _nonfinite(number)
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("replication evidence mapping keys must be strings")
            normalized[key] = _evidence_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_evidence_value(item) for item in value]
    raise TypeError(f"unsupported replication evidence value: {type(value).__name__}")


def canonical_replication_json(value: Any) -> str:
    """Canonical JSON used for every contract and evidence digest."""
    return json.dumps(
        _evidence_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_replication_json(value).encode("utf-8")).hexdigest()


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _sha256(value: Any, *, field_name: str) -> str:
    candidate = _required_text(value, field_name=field_name).lower()
    if len(candidate) != 64 or any(character not in _HEX for character in candidate):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return candidate


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ReplicationIntegrityError(
            f"invalid {name} fields "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )


def _key_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if any(not isinstance(key, str) or not key.strip() for key in value):
        return False

    def valid_item(item: Any) -> bool:
        if _is_nonfinite_marker(item):
            return False
        if item is None or isinstance(item, (str, bool, int, float)):
            return not isinstance(item, float) or math.isfinite(item)
        if isinstance(item, Mapping):
            return all(
                isinstance(key, str) and key.strip() and valid_item(nested)
                for key, nested in item.items()
            )
        if isinstance(item, (list, tuple)):
            return all(valid_item(nested) for nested in item)
        return False

    return all(valid_item(item) for item in value.values())


def _freeze_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_evidence(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_evidence(item) for item in value)
    return value


def _normalize_expected_keys(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("expected_observation_keys must be a sequence")
    if not value:
        raise ValueError("expected_observation_keys must be nonempty")
    normalized: dict[str, dict[str, Any]] = {}
    for key in value:
        canonical_key = _evidence_value(key)
        if not _key_is_valid(canonical_key):
            raise ValueError("every expected observation key must be nonempty and valid")
        token = canonical_replication_json(canonical_key)
        if token in normalized:
            raise ValueError("expected observation keys must be unique")
        normalized[token] = canonical_key
    return tuple(
        _freeze_evidence(normalized[token]) for token in sorted(normalized)
    )


@dataclass(frozen=True)
class ImplementationIdentity:
    """Immutable identity of one independently written implementation."""

    implementation_id: str
    code_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implementation_id",
            _required_text(self.implementation_id, field_name="implementation_id"),
        )
        object.__setattr__(
            self,
            "code_hash",
            _sha256(self.code_hash, field_name="code_hash"),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "implementation_id": self.implementation_id,
            "code_hash": self.code_hash,
        }

    @classmethod
    def from_payload(cls, value: Any) -> "ImplementationIdentity":
        if not isinstance(value, Mapping):
            raise ReplicationIntegrityError("implementation identity must be an object")
        _exact_fields(
            value,
            {"implementation_id", "code_hash"},
            name="implementation identity",
        )
        try:
            return cls(
                implementation_id=value["implementation_id"],
                code_hash=value["code_hash"],
            )
        except (TypeError, ValueError) as exc:
            raise ReplicationIntegrityError("invalid implementation identity") from exc


@dataclass(frozen=True)
class NumericTolerance:
    """Explicit closeness rule: ``|a-b| <= absolute + relative*max(|a|,|b|)``."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        for field_name in ("absolute", "relative"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} tolerance must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{field_name} tolerance must be finite and non-negative")
            object.__setattr__(self, field_name, 0.0 if number == 0.0 else number)

    def to_payload(self) -> dict[str, float]:
        return {"absolute": self.absolute, "relative": self.relative}

    @classmethod
    def from_payload(cls, value: Any) -> "NumericTolerance":
        if not isinstance(value, Mapping):
            raise ReplicationIntegrityError("numeric tolerance must be an object")
        _exact_fields(value, {"absolute", "relative"}, name="numeric tolerance")
        try:
            return cls(absolute=value["absolute"], relative=value["relative"])
        except (TypeError, ValueError) as exc:
            raise ReplicationIntegrityError("invalid numeric tolerance") from exc


@dataclass(frozen=True)
class IndependentReplicationContract:
    """Frozen comparison contract bound to protocol, data, and code hashes."""

    protocol_hash: str
    data_snapshot_hash: str
    primary: ImplementationIdentity
    replication: ImplementationIdentity
    expected_observation_keys: Sequence[Mapping[str, Any]]
    tolerances: Mapping[str, NumericTolerance]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_hash",
            _sha256(self.protocol_hash, field_name="protocol_hash"),
        )
        object.__setattr__(
            self,
            "data_snapshot_hash",
            _sha256(self.data_snapshot_hash, field_name="data_snapshot_hash"),
        )
        if not isinstance(self.primary, ImplementationIdentity):
            raise TypeError("primary must be an ImplementationIdentity")
        if not isinstance(self.replication, ImplementationIdentity):
            raise TypeError("replication must be an ImplementationIdentity")
        if self.primary.implementation_id == self.replication.implementation_id:
            raise ValueError("independent implementations must have distinct IDs")
        if self.primary.code_hash == self.replication.code_hash:
            raise ValueError("independent implementations must have distinct code hashes")
        object.__setattr__(
            self,
            "expected_observation_keys",
            _normalize_expected_keys(self.expected_observation_keys),
        )
        if not isinstance(self.tolerances, Mapping):
            raise TypeError("tolerances must be a mapping")
        actual = set(self.tolerances)
        expected = set(NUMERIC_FIELDS)
        if actual != expected:
            raise ValueError(
                "tolerances must explicitly cover every numeric field "
                f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
            )
        normalized: dict[str, NumericTolerance] = {}
        for field_name in NUMERIC_FIELDS:
            tolerance = self.tolerances[field_name]
            if isinstance(tolerance, Mapping):
                tolerance = NumericTolerance.from_payload(tolerance)
            if not isinstance(tolerance, NumericTolerance):
                raise TypeError(f"tolerance for {field_name} must be NumericTolerance")
            normalized[field_name] = tolerance
        object.__setattr__(self, "tolerances", MappingProxyType(normalized))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "independent_replication_contract",
            "protocol_hash": self.protocol_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "primary": self.primary.to_payload(),
            "replication": self.replication.to_payload(),
            "expected_observation_keys": [
                _evidence_value(key) for key in self.expected_observation_keys
            ],
            "tolerances": {
                field_name: self.tolerances[field_name].to_payload()
                for field_name in NUMERIC_FIELDS
            },
        }

    @property
    def contract_hash(self) -> str:
        return _content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, value: Any) -> "IndependentReplicationContract":
        if not isinstance(value, Mapping):
            raise ReplicationIntegrityError("replication contract must be an object")
        _exact_fields(
            value,
            {
                "schema_version",
                "record_type",
                "protocol_hash",
                "data_snapshot_hash",
                "primary",
                "replication",
                "expected_observation_keys",
                "tolerances",
            },
            name="replication contract",
        )
        if value["schema_version"] != SCHEMA_VERSION:
            raise ReplicationIntegrityError("unsupported replication contract schema")
        if value["record_type"] != "independent_replication_contract":
            raise ReplicationIntegrityError("invalid replication contract record type")
        tolerances = value["tolerances"]
        if not isinstance(tolerances, Mapping):
            raise ReplicationIntegrityError("contract tolerances must be an object")
        try:
            return cls(
                protocol_hash=value["protocol_hash"],
                data_snapshot_hash=value["data_snapshot_hash"],
                primary=ImplementationIdentity.from_payload(value["primary"]),
                replication=ImplementationIdentity.from_payload(value["replication"]),
                expected_observation_keys=value["expected_observation_keys"],
                tolerances={
                    field_name: NumericTolerance.from_payload(tolerance)
                    for field_name, tolerance in tolerances.items()
                },
            )
        except ReplicationIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise ReplicationIntegrityError("invalid replication contract") from exc


@dataclass
class _ParsedObservation:
    key: dict[str, Any]
    key_token: str
    raw: dict[str, Any]
    occurrence: int
    valid_fields: dict[str, Any]


@dataclass
class _ParsedOutput:
    identity: ImplementationIdentity
    observations: list[dict[str, Any]]
    groups: dict[str, list[_ParsedObservation]]
    discrepancies: list[dict[str, Any]]


def _normalize_observations(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("implementation observations must be a sequence")
    normalized: list[dict[str, Any]] = []
    for observation in value:
        if not isinstance(observation, Mapping):
            raise TypeError("each implementation observation must be a mapping")
        item = _evidence_value(observation)
        # Equal numeric evidence should canonicalize identically whether an
        # implementation emitted 1 or 1.0.
        for field_name in NUMERIC_FIELDS:
            field_value = item.get(field_name)
            if isinstance(field_value, (int, float)) and not isinstance(field_value, bool):
                number = float(field_value)
                item[field_name] = 0.0 if number == 0.0 else number
        normalized.append(item)
    return sorted(normalized, key=canonical_replication_json)


def _parse_output(
    identity: ImplementationIdentity,
    observations: list[dict[str, Any]],
) -> _ParsedOutput:
    discrepancies: list[dict[str, Any]] = []
    groups: dict[str, list[_ParsedObservation]] = {}

    for occurrence, raw in enumerate(observations):
        actual_fields = set(raw)
        missing_fields = sorted(OBSERVATION_FIELDS - actual_fields)
        extra_fields = sorted(actual_fields - OBSERVATION_FIELDS)
        if missing_fields or extra_fields:
            discrepancies.append(
                {
                    "kind": "invalid_observation_fields",
                    "implementation_id": identity.implementation_id,
                    "occurrence": occurrence,
                    "missing_fields": missing_fields,
                    "extra_fields": extra_fields,
                    "observation": raw,
                }
            )

        key = raw.get("key")
        if not _key_is_valid(key):
            discrepancies.append(
                {
                    "kind": "invalid_observation_key",
                    "implementation_id": identity.implementation_id,
                    "occurrence": occurrence,
                    "key": key,
                }
            )
            continue
        normalized_key = _evidence_value(key)
        key_token = canonical_replication_json(normalized_key)
        valid_fields: dict[str, Any] = {}

        if "eligibility" in raw:
            if isinstance(raw["eligibility"], bool):
                valid_fields["eligibility"] = raw["eligibility"]
            else:
                discrepancies.append(
                    {
                        "kind": "invalid_value_type",
                        "implementation_id": identity.implementation_id,
                        "occurrence": occurrence,
                        "key": normalized_key,
                        "field": "eligibility",
                        "expected": "boolean",
                        "value": raw["eligibility"],
                    }
                )

        for field_name in NUMERIC_FIELDS:
            if field_name not in raw:
                continue
            field_value = raw[field_name]
            if _is_nonfinite_marker(field_value):
                discrepancies.append(
                    {
                        "kind": "nonfinite_value",
                        "implementation_id": identity.implementation_id,
                        "occurrence": occurrence,
                        "key": normalized_key,
                        "field": field_name,
                        "value": field_value,
                    }
                )
            elif isinstance(field_value, bool) or not isinstance(field_value, Real):
                discrepancies.append(
                    {
                        "kind": "invalid_value_type",
                        "implementation_id": identity.implementation_id,
                        "occurrence": occurrence,
                        "key": normalized_key,
                        "field": field_name,
                        "expected": "finite number",
                        "value": field_value,
                    }
                )
            else:
                number = float(field_value)
                # This normally arrives tagged by _normalize_observations, but
                # retains fail-closed behavior when validating stored payloads.
                if not math.isfinite(number):
                    discrepancies.append(
                        {
                            "kind": "nonfinite_value",
                            "implementation_id": identity.implementation_id,
                            "occurrence": occurrence,
                            "key": normalized_key,
                            "field": field_name,
                            "value": _nonfinite(number),
                        }
                    )
                else:
                    valid_fields[field_name] = 0.0 if number == 0.0 else number

        parsed = _ParsedObservation(
            key=normalized_key,
            key_token=key_token,
            raw=raw,
            occurrence=occurrence,
            valid_fields=valid_fields,
        )
        groups.setdefault(key_token, []).append(parsed)

    for group in groups.values():
        if len(group) > 1:
            discrepancies.append(
                {
                    "kind": "duplicate_observation",
                    "implementation_id": identity.implementation_id,
                    "key": group[0].key,
                    "count": len(group),
                    "occurrences": [item.occurrence for item in group],
                    "observations": [item.raw for item in group],
                }
            )

    return _ParsedOutput(identity, observations, groups, discrepancies)


def _expected_coverage_discrepancies(
    contract: IndependentReplicationContract,
    output: _ParsedOutput,
) -> list[dict[str, Any]]:
    expected = {
        canonical_replication_json(key): _evidence_value(key)
        for key in contract.expected_observation_keys
    }
    actual_tokens = set(output.groups)
    discrepancies: list[dict[str, Any]] = []
    for token in sorted(set(expected) - actual_tokens):
        discrepancies.append({
            "kind": "missing_expected_observation",
            "implementation_id": output.identity.implementation_id,
            "key": expected[token],
        })
    for token in sorted(actual_tokens - set(expected)):
        group = output.groups[token]
        discrepancies.append({
            "kind": "unexpected_observation",
            "implementation_id": output.identity.implementation_id,
            "key": group[0].key,
            "observations": [item.raw for item in group],
        })
    return discrepancies


def _safe_difference(left: float, right: float) -> float | dict[str, str]:
    difference = abs(left - right)
    return difference if math.isfinite(difference) else _nonfinite(difference)


def _allowed_error(
    tolerance: NumericTolerance,
    left: float,
    right: float,
) -> float:
    return tolerance.absolute + tolerance.relative * max(abs(left), abs(right))


def _compare_outputs(
    contract: IndependentReplicationContract,
    primary: _ParsedOutput,
    replication: _ParsedOutput,
) -> list[dict[str, Any]]:
    discrepancies = [
        *primary.discrepancies,
        *replication.discrepancies,
        *_expected_coverage_discrepancies(contract, primary),
        *_expected_coverage_discrepancies(contract, replication),
    ]
    primary_keys = set(primary.groups)
    replication_keys = set(replication.groups)

    for key_token in sorted(primary_keys - replication_keys):
        group = primary.groups[key_token]
        discrepancies.append(
            {
                "kind": "missing_observation",
                "key": group[0].key,
                "missing_from": replication.identity.implementation_id,
                "present_in": primary.identity.implementation_id,
                "present_observations": [item.raw for item in group],
            }
        )
    for key_token in sorted(replication_keys - primary_keys):
        group = replication.groups[key_token]
        discrepancies.append(
            {
                "kind": "missing_observation",
                "key": group[0].key,
                "missing_from": primary.identity.implementation_id,
                "present_in": replication.identity.implementation_id,
                "present_observations": [item.raw for item in group],
            }
        )

    for key_token in sorted(primary_keys & replication_keys):
        primary_group = primary.groups[key_token]
        replication_group = replication.groups[key_token]
        # A duplicate makes the keyed value ambiguous; the full duplicate
        # records above are the discrepancy.  Never select one occurrence and
        # accidentally report a successful field comparison.
        if len(primary_group) != 1 or len(replication_group) != 1:
            continue
        left = primary_group[0]
        right = replication_group[0]

        if "eligibility" in left.valid_fields and "eligibility" in right.valid_fields:
            if left.valid_fields["eligibility"] != right.valid_fields["eligibility"]:
                discrepancies.append(
                    {
                        "kind": "exact_mismatch",
                        "key": left.key,
                        "field": "eligibility",
                        "primary": {
                            "implementation_id": primary.identity.implementation_id,
                            "value": left.valid_fields["eligibility"],
                        },
                        "replication": {
                            "implementation_id": replication.identity.implementation_id,
                            "value": right.valid_fields["eligibility"],
                        },
                    }
                )

        for field_name in NUMERIC_FIELDS:
            if field_name not in left.valid_fields or field_name not in right.valid_fields:
                continue
            left_value = left.valid_fields[field_name]
            right_value = right.valid_fields[field_name]
            tolerance = contract.tolerances[field_name]
            allowed = _allowed_error(tolerance, left_value, right_value)
            difference = abs(left_value - right_value)
            if difference > allowed:
                discrepancies.append(
                    {
                        "kind": "numeric_mismatch",
                        "key": left.key,
                        "field": field_name,
                        "primary": {
                            "implementation_id": primary.identity.implementation_id,
                            "value": left_value,
                        },
                        "replication": {
                            "implementation_id": replication.identity.implementation_id,
                            "value": right_value,
                        },
                        "absolute_error": _safe_difference(left_value, right_value),
                        "allowed_error": (
                            allowed if math.isfinite(allowed) else _nonfinite(allowed)
                        ),
                        "tolerance": tolerance.to_payload(),
                    }
                )

    return sorted(discrepancies, key=canonical_replication_json)


def _build_evidence_payload(
    contract: IndependentReplicationContract,
    primary_observations: Any,
    replication_observations: Any,
) -> dict[str, Any]:
    normalized_primary = _normalize_observations(primary_observations)
    normalized_replication = _normalize_observations(replication_observations)
    primary = _parse_output(contract.primary, normalized_primary)
    replication = _parse_output(contract.replication, normalized_replication)
    discrepancies = _compare_outputs(contract, primary, replication)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "independent_replication_evidence",
        "protocol_hash": contract.protocol_hash,
        "data_snapshot_hash": contract.data_snapshot_hash,
        "contract_hash": contract.contract_hash,
        "contract": contract.to_payload(),
        "outputs": [
            {
                **contract.primary.to_payload(),
                "observations": normalized_primary,
            },
            {
                **contract.replication.to_payload(),
                "observations": normalized_replication,
            },
        ],
        "discrepancies": discrepancies,
        "passed": not discrepancies,
    }


@dataclass(frozen=True)
class ReplicationEvidence:
    """Complete reconciliation evidence identified by its canonical SHA-256."""

    evidence_hash: str
    payload_json: str

    @classmethod
    def create(
        cls,
        contract: IndependentReplicationContract,
        *,
        primary_observations: Sequence[Mapping[str, Any]],
        replication_observations: Sequence[Mapping[str, Any]],
    ) -> "ReplicationEvidence":
        if not isinstance(contract, IndependentReplicationContract):
            raise TypeError("contract must be an IndependentReplicationContract")
        payload = _build_evidence_payload(
            contract,
            primary_observations,
            replication_observations,
        )
        payload_json = canonical_replication_json(payload)
        return cls(
            hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            payload_json,
        )

    @property
    def payload(self) -> dict[str, Any]:
        value = _strict_json_loads(self.payload_json)
        if not isinstance(value, dict):
            raise ReplicationIntegrityError("replication evidence payload is not an object")
        return value

    @property
    def passed(self) -> bool:
        return bool(self.payload["passed"])

    @property
    def discrepancies(self) -> list[dict[str, Any]]:
        return self.payload["discrepancies"]

    def to_json(self) -> str:
        return canonical_replication_json(
            {"evidence_hash": self.evidence_hash, "evidence": self.payload}
        ) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "ReplicationEvidence":
        document = _strict_json_loads(value)
        if not isinstance(document, Mapping):
            raise ReplicationIntegrityError("replication evidence document must be an object")
        _exact_fields(document, {"evidence_hash", "evidence"}, name="evidence document")
        try:
            claimed_hash = _sha256(document["evidence_hash"], field_name="evidence_hash")
        except (TypeError, ValueError) as exc:
            raise ReplicationIntegrityError("invalid replication evidence hash") from exc
        payload = document["evidence"]
        if not isinstance(payload, Mapping):
            raise ReplicationIntegrityError("replication evidence must be an object")
        payload_json = canonical_replication_json(payload)
        actual_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if claimed_hash != actual_hash:
            raise ReplicationIntegrityError("replication evidence hash mismatch")
        _exact_fields(
            payload,
            {
                "schema_version",
                "record_type",
                "protocol_hash",
                "data_snapshot_hash",
                "contract_hash",
                "contract",
                "outputs",
                "discrepancies",
                "passed",
            },
            name="replication evidence",
        )
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ReplicationIntegrityError("unsupported replication evidence schema")
        if payload["record_type"] != "independent_replication_evidence":
            raise ReplicationIntegrityError("invalid replication evidence record type")
        contract = IndependentReplicationContract.from_payload(payload["contract"])
        if payload["protocol_hash"] != contract.protocol_hash:
            raise ReplicationIntegrityError("evidence protocol hash differs from contract")
        if payload["data_snapshot_hash"] != contract.data_snapshot_hash:
            raise ReplicationIntegrityError("evidence data snapshot hash differs from contract")
        if payload["contract_hash"] != contract.contract_hash:
            raise ReplicationIntegrityError("evidence contract hash mismatch")

        outputs = payload["outputs"]
        if not isinstance(outputs, list) or len(outputs) != 2:
            raise ReplicationIntegrityError("evidence must contain exactly two outputs")
        identities = (contract.primary, contract.replication)
        observations: list[Any] = []
        for output, identity in zip(outputs, identities):
            if not isinstance(output, Mapping):
                raise ReplicationIntegrityError("implementation output must be an object")
            _exact_fields(
                output,
                {"implementation_id", "code_hash", "observations"},
                name="implementation output",
            )
            if output["implementation_id"] != identity.implementation_id:
                raise ReplicationIntegrityError("implementation output ID differs from contract")
            if output["code_hash"] != identity.code_hash:
                raise ReplicationIntegrityError("implementation output code hash differs from contract")
            if not isinstance(output["observations"], list):
                raise ReplicationIntegrityError("implementation observations must be an array")
            observations.append(output["observations"])

        # Do not merely trust a self-consistent hash.  Re-run reconciliation
        # from the stored raw outputs and require every derived field and every
        # discrepancy record to match exactly.
        rebuilt = _build_evidence_payload(contract, observations[0], observations[1])
        if canonical_replication_json(rebuilt) != payload_json:
            raise ReplicationIntegrityError(
                "stored evidence does not match complete reconciliation"
            )
        return cls(actual_hash, payload_json)


def reconcile_implementations(
    contract: IndependentReplicationContract,
    *,
    primary_observations: Sequence[Mapping[str, Any]],
    replication_observations: Sequence[Mapping[str, Any]],
) -> ReplicationEvidence:
    """Reconcile two outputs and return a complete immutable evidence artifact."""
    return ReplicationEvidence.create(
        contract,
        primary_observations=primary_observations,
        replication_observations=replication_observations,
    )


def _atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ReplicationIntegrityError(
                    f"refusing to overwrite immutable replication evidence: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class ReplicationEvidenceStore:
    """Create-only store for independently verified replication artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.evidence_dir = self.root / "independent-replications"

    @staticmethod
    def _validate_hash(evidence_hash: str) -> str:
        return _sha256(evidence_hash, field_name="evidence_hash")

    def path_for(self, evidence_hash: str) -> Path:
        return self.evidence_dir / f"{self._validate_hash(evidence_hash)}.json"

    def persist(self, evidence: ReplicationEvidence) -> Path:
        if not isinstance(evidence, ReplicationEvidence):
            raise TypeError("evidence must be ReplicationEvidence")
        verified = ReplicationEvidence.from_json(evidence.to_json())
        if verified.evidence_hash != evidence.evidence_hash:
            raise ReplicationIntegrityError("replication evidence failed verification")
        path = self.path_for(verified.evidence_hash)
        _atomic_create(path, verified.to_json())
        return path

    def load(self, evidence_hash: str) -> ReplicationEvidence:
        digest = self._validate_hash(evidence_hash)
        try:
            evidence = ReplicationEvidence.from_json(
                self.path_for(digest).read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"replication evidence does not exist: {digest}") from exc
        if evidence.evidence_hash != digest:
            raise ReplicationIntegrityError(
                "replication evidence filename and content hash differ"
            )
        return evidence


__all__ = [
    "ImplementationIdentity",
    "IndependentReplicationContract",
    "NUMERIC_FIELDS",
    "NumericTolerance",
    "ReplicationError",
    "ReplicationEvidence",
    "ReplicationEvidenceStore",
    "ReplicationIntegrityError",
    "canonical_replication_json",
    "reconcile_implementations",
]
