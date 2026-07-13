"""Immutable, artifact-bound deployment configuration for Foundation.

Research construction remains intentionally flexible.  Paper/live construction
does not: every value that can change a :class:`FoundationDesk` is carried in
one canonical mapping and reconstructed from the approved promotion artifact.
This module is deliberately independent of the promotion store so it can be
used by artifact generators as well as the runtime factory.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from desks.foundation import FoundationDesk


def _finite_number(name: str, value: Any) -> float:
    """Return ``value`` as a finite float, rejecting bools and coercion.

    JSON numbers arrive as ``int`` or ``float``.  Strings such as ``"0.5"``
    are intentionally not accepted at the deployment boundary: accepting
    coercible values would make the artifact mapping less exact than the
    object it claims to describe.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class FoundationDeploymentConfig:
    """Complete, immutable construction contract for a deployed Foundation.

    The dataclass offers safe defaults for explicit Python construction, while
    :meth:`from_mapping` requires *every* field.  Promotion artifacts therefore
    cannot silently inherit a new application default after approval.
    """

    strategy_version: str
    capital_allocation: float = 1.0
    model_key: str = "gbm"
    rsi_entry_low: float = 40.0
    rsi_entry_high: float = 70.0
    rsi_exit: float = 70.0
    volume_confirmation_mult: float = 1.2
    gate_threshold: float = 0.0
    target_mode: bool = True

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "strategy_version",
        "capital_allocation",
        "model_key",
        "rsi_entry_low",
        "rsi_entry_high",
        "rsi_exit",
        "volume_confirmation_mult",
        "gate_threshold",
        "target_mode",
    )

    def __post_init__(self) -> None:
        if (not isinstance(self.strategy_version, str)
                or not self.strategy_version
                or self.strategy_version != self.strategy_version.strip()):
            raise ValueError("strategy_version must be a non-empty string")
        if (not isinstance(self.model_key, str) or not self.model_key
                or self.model_key != self.model_key.strip()):
            raise ValueError("model_key must be a non-empty string")

        # Reject unknown model ids at configuration time, before an approved
        # artifact can reach an optional dependency import in build().
        from desks.models import available_models
        known_models = {entry["id"] for entry in available_models()}
        if self.model_key not in known_models:
            raise ValueError(f"unknown Foundation model_key: {self.model_key}")

        if type(self.target_mode) is not bool:  # bool only; not truthy values
            raise ValueError("target_mode must be a boolean")
        if not self.target_mode:
            raise ValueError(
                "target_mode must be true for paper/live deployment")

        numeric_names = (
            "capital_allocation",
            "rsi_entry_low",
            "rsi_entry_high",
            "rsi_exit",
            "volume_confirmation_mult",
            "gate_threshold",
        )
        for name in numeric_names:
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))

        if not 0.0 < self.capital_allocation <= 1.0:
            raise ValueError("capital_allocation must be in (0, 1]")
        for name in ("rsi_entry_low", "rsi_entry_high", "rsi_exit"):
            if not 0.0 <= getattr(self, name) <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")
        if self.rsi_entry_low > self.rsi_entry_high:
            raise ValueError("rsi_entry_low cannot exceed rsi_entry_high")
        if self.volume_confirmation_mult <= 0.0:
            raise ValueError("volume_confirmation_mult must be positive")
        # Foundation's model score is P(up) - 0.5.
        if not -0.5 <= self.gate_threshold <= 0.5:
            raise ValueError("gate_threshold must be in [-0.5, 0.5]")

    @classmethod
    def from_mapping(
            cls, values: Mapping[str, Any]) -> FoundationDeploymentConfig:
        """Parse an exact canonical artifact mapping.

        Missing and unknown fields fail closed.  Defaults are never supplied
        here because doing so would let an old artifact change behavior when a
        constructor default changes in a later release.
        """
        if not isinstance(values, Mapping):
            raise ValueError("Foundation deployment parameters must be a mapping")
        provided = set(values)
        expected = set(cls.FIELD_NAMES)
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            raise ValueError("invalid Foundation deployment parameters ("
                             + "; ".join(details) + ")")
        return cls(**{name: values[name] for name in cls.FIELD_NAMES})

    # Artifact-generation code can use the more domain-specific name without
    # maintaining a second parser.
    from_parameters = from_mapping

    def to_mapping(self) -> dict[str, Any]:
        """Return the complete JSON-native mapping embedded in an artifact."""
        return {name: getattr(self, name) for name in self.FIELD_NAMES}

    to_parameters = to_mapping

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical construction mapping."""
        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def build(self) -> FoundationDesk:
        """Construct Foundation using every artifact-bound runtime value."""
        desk = FoundationDesk(
            capital_allocation=self.capital_allocation,
            model_key=self.model_key,
            rsi_entry_low=self.rsi_entry_low,
            rsi_entry_high=self.rsi_entry_high,
            rsi_exit=self.rsi_exit,
            volume_confirmation_mult=self.volume_confirmation_mult,
            gate_threshold=self.gate_threshold,
            target_mode=self.target_mode,
            deployment_version=self.strategy_version,
        )
        # The exact frozen source remains inspectable even before a promotion
        # identity is attached by the registry factory.
        setattr(desk, "deployment_config", self)
        return desk


@dataclass(frozen=True, slots=True)
class FoundationDeploymentIdentity:
    """Immutable provenance attached to a deployed Foundation instance."""

    artifact_hash: str
    strategy_id: str
    strategy_version: str
    required_level: str
    code_sha: str
    config_hash: str


def foundation_target_v1_config() -> FoundationDeploymentConfig:
    """The single intentionally small configuration for the first lane."""
    return FoundationDeploymentConfig(
        strategy_version="foundation-target-v1",
        capital_allocation=0.10,
        model_key="gbm",
        rsi_entry_low=40.0,
        rsi_entry_high=70.0,
        rsi_exit=70.0,
        volume_confirmation_mult=1.2,
        gate_threshold=-0.05,
        target_mode=True,
    )


__all__ = [
    "FoundationDeploymentConfig",
    "FoundationDeploymentIdentity",
    "foundation_target_v1_config",
]
