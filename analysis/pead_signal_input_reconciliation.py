"""Authoritative final PEAD signal-input reconciliation.

This is the only boundary that turns v3 event-source evidence into research
inputs.  It first authoritatively replays both the source reconciliation and
the Sharadar market-accounting receipt from their original evidence.  It then
joins them by the frozen event identity, proves the common dated security and
split-restated share basis, and evaluates the frozen signal formula with exact
decimal inputs.

The one-argument structural validator proves content identity and internal
math only.  The authoritative verifier always rebuilds the receipt from both
upstream artifacts, their original evidence, exact specification/code bytes,
and caller-supplied external trust registries.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from analysis.pead_known_by_policy import KNOWN_BY_POLICY_SHA256, MINIMUM_ANALYST_COUNT
from analysis.pead_source_reconciliation_v2 import (
    PeadSourceReconciliationV2Error,
    verify_pead_source_reconciliation_v2,
)
from data.pead_event_universe import (
    PeadEventUniverseError,
    canonical_event_id,
    canonical_json,
    content_hash,
    validate_event_key,
)
from data.pead_market_accounting_evidence import (
    PeadMarketAccountingEvidenceError,
    verify_pead_market_accounting_evidence,
)


SIGNAL_INPUT_RECONCILIATION_SCHEMA_VERSION = "pead_signal_input_reconciliation.v1"
SIGNAL_INPUT_RECONCILIATION_POLICY_SCHEMA_VERSION = "pead_signal_input_reconciliation_policy.v1"
TRUST_ROOT_SET_SCHEMA_VERSION = "pead_sha256_trust_root_set.v1"
MAX_SIGNAL_INPUT_RECONCILIATION_BYTES = 512 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_MACHINE_REASON = re.compile(r"^[a-z][a-z0-9_]*$")
_DECIMAL_TEXT = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "created_at_utc",
    "policy",
    "trust_policy",
    "bindings",
    "event_results",
    "coverage",
    "qualification",
}
_TRUST_POLICY_FIELDS = {
    "candidate_specification_set_sha256",
    "construction_code_set_sha256",
    "signal_reconciliation_code_set_sha256",
    "source_reconciliation_set_sha256",
    "market_accounting_evidence_set_sha256",
}
_BINDING_FIELDS = {
    "candidate_specification_sha256",
    "construction_code_sha256",
    "signal_reconciliation_code_sha256",
    "source_reconciliation_sha256",
    "market_accounting_evidence_sha256",
    "event_universe_sha256",
    "source_reconciliation_bindings_sha256",
    "market_accounting_bindings_sha256",
    "market_accounting_trust_policy_sha256",
    "known_by_policy_sha256",
    "source_reconciliation_policy_sha256",
    "market_accounting_policy_sha256",
    "signal_input_reconciliation_policy_sha256",
}
_EVENT_RESULT_FIELDS = {
    "event_id",
    "event_key",
    "source_disposition",
    "market_disposition",
    "disposition",
    "source_blockers",
    "market_blockers",
    "reconciliation_blockers",
    "identity",
    "source_input",
    "market_denominator",
    "signal",
}
_IDENTITY_FIELDS = {"ticker", "permaticker", "identity_id"}
_SOURCE_INPUT_FIELDS = {
    "event_id",
    "event_key",
    "actual_value",
    "consensus_value",
    "raw_surprise",
    "surprise_direction",
    "analyst_count",
    "known_public_by_at_utc",
    "availability_adapter_id",
    "consensus_provider_as_of_date",
    "consensus_available_at_utc",
    "consensus_availability_precision",
    "consensus_receipt_captured_at_utc",
    "consensus_cutoff_rule",
    "market_cutoff_rule",
    "metric",
    "provenance",
}
_METRIC_FIELDS = {
    "metric_id",
    "accounting_basis",
    "per_share_basis",
    "scope",
    "canonical_share_basis",
    "currency_code",
    "unit",
    "metric_definition_sha256",
}
_SIGNAL_FIELDS = {
    "formula",
    "actual_value",
    "consensus_value",
    "raw_surprise",
    "strictly_prior_split_normalized_close",
    "exact_ratio",
    "value_decimal_34",
    "direction",
}
_RATIO_FIELDS = {"numerator", "denominator"}
_COVERAGE_FIELDS = {
    "expected_event_count",
    "source_reconciled_event_count",
    "market_accounting_evidenced_count",
    "signal_input_accepted_count",
    "signal_input_excluded_count",
    "exhaustive_event_accounting",
    "partial_coverage",
    "blocker_counts",
}
_QUALIFICATION_FIELDS = {
    "has_research_consumable_signal_inputs",
    "all_expected_events_signal_accepted",
    "signal_input_reconciliation_allowed",
    "research_consumable",
    "historical_replication_allowed",
    "prospective_accumulation_allowed",
    "edge_claim_allowed",
    "paper_execution_allowed",
    "live_deployment_allowed",
}

_FORMULA = (
    "(independent_canonical_actual_eps - selected_point_in_time_consensus_eps) / "
    "strictly_prior_split_normalized_SEP_close"
)
_POLICY = {
    "schema_version": SIGNAL_INPUT_RECONCILIATION_POLICY_SCHEMA_VERSION,
    "signal_formula": _FORMULA,
    "source_rule": "authoritatively_replayed_pead_source_reconciliation_v2_only",
    "market_rule": "authoritatively_replayed_pead_market_accounting_evidence_v1_only",
    "join_rule": "exact_ordered_event_id_and_event_key_match",
    "identity_rule": "market_lineage_identity_retained_without_ticker_fallback",
    "share_basis_rule": "split_restated_eps_over_split_normalized_sep_close",
    "unit_rule": "usd_currency_per_share_over_usd_sep_close",
    "math_rule": "exact_decimal_inputs_reduced_rational_and_decimal_34_round_half_even",
    "minimum_analyst_count": MINIMUM_ANALYST_COUNT,
    "missing_rule": "exclude_event_without_imputation_or_fallback",
    "accounting_rule": "preserve_every_frozen_event_exactly_once",
    "research_rule": "only_signal_input_accepted_rows_are_research_consumable",
    "return_accounting_allowed": False,
    "edge_claim_allowed": False,
    "paper_execution_allowed": False,
    "live_deployment_allowed": False,
}


class PeadSignalInputReconciliationError(ValueError):
    """The final signal-input receipt is malformed or cannot replay."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadSignalInputReconciliationError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadSignalInputReconciliationError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PeadSignalInputReconciliationError(f"{label} must be nonempty canonical text")
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadSignalInputReconciliationError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadSignalInputReconciliationError(f"{label} must be canonical UTC with Z") from exc
    rendered = parsed.isoformat().replace("+00:00", "Z")
    if rendered != value:
        raise PeadSignalInputReconciliationError(f"{label} is not canonical UTC")
    return rendered, parsed


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise PeadSignalInputReconciliationError("decimal result must be finite")
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or len(value) > 256 or _DECIMAL_TEXT.fullmatch(value) is None:
        raise PeadSignalInputReconciliationError(
            f"{label} must be a finite canonical decimal string"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - guarded by regex
        raise PeadSignalInputReconciliationError(f"{label} is not decimal") from exc
    if not parsed.is_finite() or _canonical_decimal(parsed) != value:
        raise PeadSignalInputReconciliationError(f"{label} is not canonical decimal")
    if positive and parsed <= 0:
        raise PeadSignalInputReconciliationError(f"{label} must be positive")
    return parsed


def _trust_roots(values: Collection[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PeadSignalInputReconciliationError(f"{label} must be a hash collection")
    try:
        roots = tuple(sorted({_sha(value, label) for value in values}))
    except TypeError as exc:
        raise PeadSignalInputReconciliationError(f"{label} must be a hash collection") from exc
    if not roots:
        raise PeadSignalInputReconciliationError(f"{label} external trust registry is empty")
    return roots


def _trust_set_hash(values: Collection[str]) -> str:
    return content_hash(
        {
            "schema_version": TRUST_ROOT_SET_SCHEMA_VERSION,
            "members": sorted(values),
        }
    )


def _require_trusted(claimed: str, trusted: Collection[str], label: str) -> None:
    if claimed not in trusted:
        raise PeadSignalInputReconciliationError(
            f"{label} is absent from its external trust registry"
        )


def _file_bytes(path_value: str | Path, label: str, *, max_bytes: int) -> tuple[Path, bytes, str]:
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        raise PeadSignalInputReconciliationError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    if not raw or len(raw) > max_bytes:
        raise PeadSignalInputReconciliationError(f"{label} file size is invalid")
    return path, raw, hashlib.sha256(raw).hexdigest()


def _candidate_specification(raw: bytes) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadSignalInputReconciliationError("candidate specification is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadSignalInputReconciliationError(
                    f"candidate specification contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadSignalInputReconciliationError(
            f"candidate specification contains invalid number {token}"
        )

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadSignalInputReconciliationError("candidate specification is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise PeadSignalInputReconciliationError("candidate specification root must be an object")
    signal_rule = value.get("signal_rule")
    if not isinstance(signal_rule, Mapping) or signal_rule.get("formula") != _FORMULA:
        raise PeadSignalInputReconciliationError(
            "candidate specification signal formula differs from the frozen policy"
        )
    if signal_rule.get("minimum_analyst_count") != MINIMUM_ANALYST_COUNT:
        raise PeadSignalInputReconciliationError(
            "candidate specification analyst-count floor differs from the frozen policy"
        )
    return value


def _machine_reasons(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or value != sorted(set(value)):
        raise PeadSignalInputReconciliationError(f"{label} must be sorted and unique")
    for reason in value:
        if not isinstance(reason, str) or _MACHINE_REASON.fullmatch(reason) is None:
            raise PeadSignalInputReconciliationError(f"{label} has an invalid reason")
    return list(value)


def _signal(source_input: Mapping[str, Any], denominator: Mapping[str, Any]) -> dict[str, Any]:
    actual_text = source_input["actual_value"]
    consensus_text = source_input["consensus_value"]
    surprise_text = source_input["raw_surprise"]
    close_text = denominator["close_split_normalized"]
    actual = _decimal(actual_text, "actual_value")
    consensus = _decimal(consensus_text, "consensus_value")
    surprise = _decimal(surprise_text, "raw_surprise")
    close = _decimal(close_text, "close_split_normalized", positive=True)
    if Fraction(actual) - Fraction(consensus) != Fraction(surprise):
        raise PeadSignalInputReconciliationError("raw surprise is not exact actual minus consensus")
    expected_direction = "positive" if surprise > 0 else "negative" if surprise < 0 else "zero"
    if source_input["surprise_direction"] != expected_direction:
        raise PeadSignalInputReconciliationError("source surprise direction is not derived")
    exact = Fraction(surprise) / Fraction(close)
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        decimal_34 = _canonical_decimal(surprise / close)
    return {
        "formula": _FORMULA,
        "actual_value": actual_text,
        "consensus_value": consensus_text,
        "raw_surprise": surprise_text,
        "strictly_prior_split_normalized_close": close_text,
        "exact_ratio": {
            "numerator": exact.numerator,
            "denominator": exact.denominator,
        },
        "value_decimal_34": decimal_34,
        "direction": expected_direction,
    }


def _cross_lane_blockers(source_input: Mapping[str, Any]) -> list[str]:
    metric = source_input.get("metric")
    if not isinstance(metric, Mapping):
        raise PeadSignalInputReconciliationError("source input metric is missing")
    blockers: list[str] = []
    if metric.get("metric_id") != "earnings_per_share":
        blockers.append("metric_not_earnings_per_share")
    if metric.get("canonical_share_basis") != "split_restated":
        blockers.append("canonical_share_basis_not_split_restated")
    if metric.get("currency_code") != "USD":
        blockers.append("currency_not_usd")
    if metric.get("unit") != "currency_per_share":
        blockers.append("unit_not_currency_per_share")
    analyst_count = source_input.get("analyst_count")
    if isinstance(analyst_count, bool) or not isinstance(analyst_count, int) or analyst_count < 1:
        raise PeadSignalInputReconciliationError(
            "source input analyst count must be a positive integer"
        )
    if analyst_count < MINIMUM_ANALYST_COUNT:
        blockers.append("analyst_count_below_minimum")
    return sorted(blockers)


def build_pead_signal_input_reconciliation(
    source_reconciliation: Mapping[str, Any],
    market_accounting_evidence: Mapping[str, Any],
    *,
    candidate_specification_path: str | Path,
    construction_code_path: str | Path,
    signal_reconciliation_code_path: str | Path,
    created_at_utc: str,
    market_accounting_verification_kwargs: Mapping[str, Any],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
    trusted_signal_reconciliation_code_sha256s: Collection[str],
    trusted_source_reconciliation_sha256s: Collection[str],
    trusted_market_accounting_evidence_sha256s: Collection[str],
) -> dict[str, Any]:
    """Build the sole source-qualified research input receipt."""
    if not isinstance(market_accounting_verification_kwargs, Mapping):
        raise PeadSignalInputReconciliationError(
            "market_accounting_verification_kwargs must be a mapping"
        )
    trust_sets = {
        "candidate": _trust_roots(
            trusted_candidate_specification_sha256s,
            "trusted_candidate_specification_sha256s",
        ),
        "construction_code": _trust_roots(
            trusted_construction_code_sha256s,
            "trusted_construction_code_sha256s",
        ),
        "signal_code": _trust_roots(
            trusted_signal_reconciliation_code_sha256s,
            "trusted_signal_reconciliation_code_sha256s",
        ),
        "source": _trust_roots(
            trusted_source_reconciliation_sha256s,
            "trusted_source_reconciliation_sha256s",
        ),
        "market": _trust_roots(
            trusted_market_accounting_evidence_sha256s,
            "trusted_market_accounting_evidence_sha256s",
        ),
    }
    source_claim = _sha(
        source_reconciliation.get("artifact_hash"), "source reconciliation artifact_hash"
    )
    market_claim = _sha(
        market_accounting_evidence.get("artifact_hash"), "market evidence artifact_hash"
    )
    _require_trusted(source_claim, trust_sets["source"], "source reconciliation")
    _require_trusted(market_claim, trust_sets["market"], "market accounting evidence")

    candidate_path, candidate_raw, candidate_hash = _file_bytes(
        candidate_specification_path, "candidate specification", max_bytes=2 * 1024 * 1024
    )
    construction_path, construction_raw, construction_hash = _file_bytes(
        construction_code_path, "construction code", max_bytes=8 * 1024 * 1024
    )
    signal_path, signal_raw, signal_code_hash = _file_bytes(
        signal_reconciliation_code_path,
        "signal reconciliation code",
        max_bytes=8 * 1024 * 1024,
    )
    implementation_path, implementation_raw, implementation_hash = _file_bytes(
        Path(__file__),
        "executing signal reconciliation implementation",
        max_bytes=8 * 1024 * 1024,
    )
    if signal_path.resolve() != implementation_path.resolve() or signal_raw != implementation_raw:
        raise PeadSignalInputReconciliationError(
            "signal reconciliation code path is not the executing implementation"
        )
    if signal_code_hash != implementation_hash:  # pragma: no cover - equality follows bytes
        raise PeadSignalInputReconciliationError(
            "signal reconciliation code identity differs from the executing implementation"
        )
    _require_trusted(candidate_hash, trust_sets["candidate"], "candidate specification")
    _require_trusted(construction_hash, trust_sets["construction_code"], "construction code")
    _require_trusted(signal_code_hash, trust_sets["signal_code"], "signal reconciliation code")
    candidate_specification = _candidate_specification(candidate_raw)

    verification_kwargs = dict(market_accounting_verification_kwargs)
    for forbidden in ("document", "source_reconciliation"):
        if forbidden in verification_kwargs:
            raise PeadSignalInputReconciliationError(
                f"market_accounting_verification_kwargs may not contain {forbidden}"
            )
    source_kwargs = verification_kwargs.get("source_reconciliation_verification_kwargs")
    if not isinstance(source_kwargs, Mapping):
        raise PeadSignalInputReconciliationError(
            "market verification must contain source reconciliation replay kwargs"
        )
    try:
        source = verify_pead_source_reconciliation_v2(source_reconciliation, **dict(source_kwargs))
    except (PeadSourceReconciliationV2Error, TypeError, ValueError) as exc:
        raise PeadSignalInputReconciliationError(
            "source reconciliation does not replay authoritatively"
        ) from exc
    if source.get("artifact_hash") != source_claim:
        raise PeadSignalInputReconciliationError(
            "source verifier returned a different artifact identity"
        )

    verification_kwargs.update(
        {
            "candidate_specification_path": candidate_path,
            "construction_code_path": construction_path,
            "trusted_candidate_specification_sha256s": trust_sets["candidate"],
            "trusted_construction_code_sha256s": trust_sets["construction_code"],
            "trusted_source_reconciliation_sha256s": trust_sets["source"],
        }
    )
    try:
        market = verify_pead_market_accounting_evidence(
            market_accounting_evidence,
            source,
            **verification_kwargs,
        )
    except (PeadMarketAccountingEvidenceError, TypeError, ValueError) as exc:
        raise PeadSignalInputReconciliationError(
            "market accounting evidence does not replay authoritatively"
        ) from exc
    if market.get("artifact_hash") != market_claim:
        raise PeadSignalInputReconciliationError(
            "market verifier returned a different artifact identity"
        )

    source_payload = source["payload"]
    market_payload = market["payload"]
    candidate_id = source_payload["candidate_id"]
    if candidate_specification.get("candidate_id") != candidate_id:
        raise PeadSignalInputReconciliationError(
            "candidate specification belongs to another candidate"
        )
    if market_payload["candidate_id"] != candidate_id:
        raise PeadSignalInputReconciliationError("upstream artifacts have different candidates")
    if market_payload["evidence_class"] != source_payload["evidence_class"]:
        raise PeadSignalInputReconciliationError(
            "upstream artifacts have different evidence classes"
        )
    source_bindings = source_payload["bindings"]
    market_bindings = market_payload["bindings"]
    market_trust = market_payload["trust_policy"]
    if source_bindings["candidate_specification_sha256"] != candidate_hash:
        raise PeadSignalInputReconciliationError(
            "source reconciliation binds another candidate specification"
        )
    if source_bindings["construction_code_sha256"] != construction_hash:
        raise PeadSignalInputReconciliationError(
            "source reconciliation binds other construction code"
        )
    if source_bindings["known_by_policy_sha256"] != KNOWN_BY_POLICY_SHA256:
        raise PeadSignalInputReconciliationError(
            "source reconciliation binds another known-by policy"
        )
    if market_bindings["source_reconciliation_sha256"] != source["artifact_hash"]:
        raise PeadSignalInputReconciliationError(
            "market evidence binds another source reconciliation"
        )
    if (
        market_bindings["source_reconciliation_event_universe_sha256"]
        != source_bindings["event_universe_sha256"]
    ):
        raise PeadSignalInputReconciliationError(
            "market and source receipts bind different event universes"
        )
    expected_market_trust = {
        "candidate_specification_set_sha256": _trust_set_hash(trust_sets["candidate"]),
        "construction_code_set_sha256": _trust_set_hash(trust_sets["construction_code"]),
        "source_reconciliation_set_sha256": _trust_set_hash(trust_sets["source"]),
    }
    for field, expected in expected_market_trust.items():
        if market_trust[field] != expected:
            raise PeadSignalInputReconciliationError(
                f"market evidence {field} differs from the final external trust registry"
            )

    source_rows = source_payload["event_results"]
    market_rows = market_payload["event_results"]
    if not isinstance(source_rows, list) or not isinstance(market_rows, list):
        raise PeadSignalInputReconciliationError("upstream event results must be arrays")
    source_ids = [row.get("event_id") for row in source_rows]
    market_ids = [row.get("event_id") for row in market_rows]
    if source_ids != market_ids or len(source_ids) != len(set(source_ids)):
        raise PeadSignalInputReconciliationError(
            "market and source receipts do not preserve one identical ordered event set"
        )

    results: list[dict[str, Any]] = []
    for index, (source_row, market_row) in enumerate(zip(source_rows, market_rows, strict=True)):
        if source_row["event_key"] != market_row["event_key"]:
            raise PeadSignalInputReconciliationError(
                f"event key differs across lanes at position {index}"
            )
        source_disposition = source_row["disposition"]
        market_disposition = market_row["disposition"]
        source_blockers = list(source_row["blockers"])
        market_blockers = list(market_row["blockers"])
        identity: dict[str, Any] | None = None
        source_input = source_row["event_source_input"]
        denominator = market_row["market_denominator"]
        reconciliation_blockers: list[str] = []
        signal: dict[str, Any] | None = None

        if source_disposition == "excluded":
            if market_disposition != "upstream_excluded" or source_input is not None:
                raise PeadSignalInputReconciliationError(
                    "market evidence changes an upstream source exclusion"
                )
            reconciliation_blockers.append("source_not_reconciled")
            source_input = None
            denominator = None
        elif source_disposition == "event_source_reconciled":
            if not isinstance(source_input, Mapping):
                raise PeadSignalInputReconciliationError(
                    "reconciled source event has no source input"
                )
            lineage = market_row["lineage"]
            if lineage is not None and not isinstance(lineage, Mapping):
                raise PeadSignalInputReconciliationError(
                    "source-reconciled event has malformed market lineage"
                )
            if lineage is not None:
                identity = {
                    "ticker": lineage["ticker"],
                    "permaticker": lineage["permaticker"],
                    "identity_id": lineage["identity_id"],
                }
            if market_disposition == "market_accounting_evidenced":
                if identity is None or not isinstance(denominator, Mapping):
                    raise PeadSignalInputReconciliationError(
                        "evidenced market event has no identity and denominator"
                    )
                if (
                    source_input["known_public_by_at_utc"]
                    != market_row["timing"]["known_public_by_at_utc"]
                ):
                    raise PeadSignalInputReconciliationError(
                        "source and market known-public times differ"
                    )
                reconciliation_blockers.extend(_cross_lane_blockers(source_input))
                if not reconciliation_blockers:
                    signal = _signal(source_input, denominator)
            elif market_disposition == "market_accounting_excluded":
                reconciliation_blockers.append("market_accounting_not_evidenced")
            else:
                raise PeadSignalInputReconciliationError(
                    "source-reconciled event has an invalid market disposition"
                )
        else:
            raise PeadSignalInputReconciliationError(
                "source reconciliation has an unsupported disposition"
            )

        reconciliation_blockers = sorted(set(reconciliation_blockers))
        results.append(
            {
                "event_id": source_row["event_id"],
                "event_key": source_row["event_key"],
                "source_disposition": source_disposition,
                "market_disposition": market_disposition,
                "disposition": (
                    "signal_input_accepted" if signal is not None else "signal_input_excluded"
                ),
                "source_blockers": source_blockers,
                "market_blockers": market_blockers,
                "reconciliation_blockers": reconciliation_blockers,
                "identity": identity,
                "source_input": source_input,
                "market_denominator": denominator,
                "signal": signal,
            }
        )

    # Detect mutable specification or code replacement during both upstream replays.
    for path, before, label in (
        (candidate_path, candidate_raw, "candidate specification"),
        (construction_path, construction_raw, "construction code"),
        (signal_path, signal_raw, "signal reconciliation code"),
    ):
        if path.read_bytes() != before:
            raise PeadSignalInputReconciliationError(f"{label} changed during reconciliation")

    created_text, created_at = _utc(created_at_utc, "created_at_utc")
    latest_input = max(
        _utc(source_payload["reconciled_at_utc"], "source reconciliation time")[1],
        _utc(market_payload["created_at_utc"], "market evidence time")[1],
    )
    if created_at < latest_input:
        raise PeadSignalInputReconciliationError(
            "signal-input reconciliation predates an upstream artifact"
        )

    accepted = sum(row["disposition"] == "signal_input_accepted" for row in results)
    source_reconciled = sum(
        row["source_disposition"] == "event_source_reconciled" for row in results
    )
    market_evidenced = sum(
        row["market_disposition"] == "market_accounting_evidenced" for row in results
    )
    blocker_counts: dict[str, int] = {}
    for row in results:
        for lane, field in (
            ("source", "source_blockers"),
            ("market", "market_blockers"),
            ("reconciliation", "reconciliation_blockers"),
        ):
            for reason in row[field]:
                key = f"{lane}__{reason}"
                blocker_counts[key] = blocker_counts.get(key, 0) + 1
    allowed = accepted > 0
    evidence_class = source_payload["evidence_class"]
    trust_policy = {
        "candidate_specification_set_sha256": _trust_set_hash(trust_sets["candidate"]),
        "construction_code_set_sha256": _trust_set_hash(trust_sets["construction_code"]),
        "signal_reconciliation_code_set_sha256": _trust_set_hash(trust_sets["signal_code"]),
        "source_reconciliation_set_sha256": _trust_set_hash(trust_sets["source"]),
        "market_accounting_evidence_set_sha256": _trust_set_hash(trust_sets["market"]),
    }
    bindings = {
        "candidate_specification_sha256": candidate_hash,
        "construction_code_sha256": construction_hash,
        "signal_reconciliation_code_sha256": signal_code_hash,
        "source_reconciliation_sha256": source["artifact_hash"],
        "market_accounting_evidence_sha256": market["artifact_hash"],
        "event_universe_sha256": source_bindings["event_universe_sha256"],
        "source_reconciliation_bindings_sha256": content_hash(source_bindings),
        "market_accounting_bindings_sha256": content_hash(market_bindings),
        "market_accounting_trust_policy_sha256": content_hash(market_trust),
        "known_by_policy_sha256": KNOWN_BY_POLICY_SHA256,
        "source_reconciliation_policy_sha256": source_bindings["reconciliation_policy_sha256"],
        "market_accounting_policy_sha256": market_bindings["market_accounting_policy_sha256"],
        "signal_input_reconciliation_policy_sha256": content_hash(_POLICY),
    }
    payload = {
        "schema_version": SIGNAL_INPUT_RECONCILIATION_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "evidence_class": evidence_class,
        "created_at_utc": created_text,
        "policy": _POLICY,
        "trust_policy": trust_policy,
        "bindings": bindings,
        "event_results": results,
        "coverage": {
            "expected_event_count": len(results),
            "source_reconciled_event_count": source_reconciled,
            "market_accounting_evidenced_count": market_evidenced,
            "signal_input_accepted_count": accepted,
            "signal_input_excluded_count": len(results) - accepted,
            "exhaustive_event_accounting": [row["event_id"] for row in results] == source_ids,
            "partial_coverage": accepted < len(results),
            "blocker_counts": {key: blocker_counts[key] for key in sorted(blocker_counts)},
        },
        "qualification": {
            "has_research_consumable_signal_inputs": allowed,
            "all_expected_events_signal_accepted": bool(results and accepted == len(results)),
            "signal_input_reconciliation_allowed": allowed,
            "research_consumable": allowed,
            "historical_replication_allowed": (
                allowed and evidence_class == "historical_reconstruction"
            ),
            "prospective_accumulation_allowed": (
                allowed and evidence_class == "prospective_signal"
            ),
            "edge_claim_allowed": False,
            "paper_execution_allowed": False,
            "live_deployment_allowed": False,
        },
    }
    return validate_pead_signal_input_reconciliation_structure(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_signal_input_reconciliation_structure(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate content identity and derived math only; this is not authoritative."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "signal-input reconciliation")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "signal-input payload")
    claimed = _sha(wrapper["artifact_hash"], "signal-input artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadSignalInputReconciliationError("signal-input artifact hash mismatch")
    if payload["schema_version"] != SIGNAL_INPUT_RECONCILIATION_SCHEMA_VERSION:
        raise PeadSignalInputReconciliationError("unsupported signal-input schema")
    _text(payload["candidate_id"], "candidate_id")
    evidence_class = payload["evidence_class"]
    if evidence_class not in {"historical_reconstruction", "prospective_signal"}:
        raise PeadSignalInputReconciliationError("unsupported evidence class")
    _utc(payload["created_at_utc"], "created_at_utc")
    if payload["policy"] != _POLICY:
        raise PeadSignalInputReconciliationError("signal-input policy differs")
    trust_policy = _exact(payload["trust_policy"], _TRUST_POLICY_FIELDS, "trust_policy")
    bindings = _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    for field in sorted(_TRUST_POLICY_FIELDS):
        _sha(trust_policy[field], f"trust_policy.{field}")
    for field in sorted(_BINDING_FIELDS):
        _sha(bindings[field], f"bindings.{field}")
    if bindings["known_by_policy_sha256"] != KNOWN_BY_POLICY_SHA256:
        raise PeadSignalInputReconciliationError("known-by policy binding differs")
    if bindings["signal_input_reconciliation_policy_sha256"] != content_hash(_POLICY):
        raise PeadSignalInputReconciliationError("signal-input policy binding differs")

    rows = payload["event_results"]
    if not isinstance(rows, list):
        raise PeadSignalInputReconciliationError("event_results must be an array")
    normalized_rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    for index, raw_row in enumerate(rows):
        row = _exact(raw_row, _EVENT_RESULT_FIELDS, f"event_results[{index}]")
        try:
            event_key = validate_event_key(
                row["event_key"], label=f"event_results[{index}].event_key"
            )
        except PeadEventUniverseError as exc:
            raise PeadSignalInputReconciliationError(
                f"event_results[{index}] has invalid event key"
            ) from exc
        event_id = _sha(row["event_id"], f"event_results[{index}].event_id")
        if event_id != canonical_event_id(event_key):
            raise PeadSignalInputReconciliationError(
                f"event_results[{index}] event ID differs from its key"
            )
        source_blockers = _machine_reasons(
            row["source_blockers"], f"event_results[{index}].source_blockers"
        )
        market_blockers = _machine_reasons(
            row["market_blockers"], f"event_results[{index}].market_blockers"
        )
        reconciliation_blockers = _machine_reasons(
            row["reconciliation_blockers"],
            f"event_results[{index}].reconciliation_blockers",
        )
        for lane, reasons in (
            ("source", source_blockers),
            ("market", market_blockers),
            ("reconciliation", reconciliation_blockers),
        ):
            for reason in reasons:
                key = f"{lane}__{reason}"
                blocker_counts[key] = blocker_counts.get(key, 0) + 1
        source_disposition = row["source_disposition"]
        market_disposition = row["market_disposition"]
        disposition = row["disposition"]
        if source_disposition not in {"event_source_reconciled", "excluded"}:
            raise PeadSignalInputReconciliationError(
                f"event_results[{index}] has invalid source disposition"
            )
        if market_disposition not in {
            "market_accounting_evidenced",
            "market_accounting_excluded",
            "upstream_excluded",
        }:
            raise PeadSignalInputReconciliationError(
                f"event_results[{index}] has invalid market disposition"
            )
        if disposition not in {"signal_input_accepted", "signal_input_excluded"}:
            raise PeadSignalInputReconciliationError(
                f"event_results[{index}] has invalid final disposition"
            )

        identity: dict[str, Any] | None = None
        source_input: dict[str, Any] | None = None
        denominator: dict[str, Any] | None = None
        signal: dict[str, Any] | None = None
        if source_disposition == "excluded":
            if (
                market_disposition != "upstream_excluded"
                or disposition != "signal_input_excluded"
                or not source_blockers
                or not market_blockers
                or reconciliation_blockers != ["source_not_reconciled"]
                or row["identity"] is not None
                or row["source_input"] is not None
                or row["market_denominator"] is not None
                or row["signal"] is not None
            ):
                raise PeadSignalInputReconciliationError(
                    f"event_results[{index}] changes an upstream exclusion"
                )
        else:
            if source_blockers:
                raise PeadSignalInputReconciliationError(
                    f"event_results[{index}] reconciled source retains blockers"
                )
            ticker: str | None = None
            permaticker: int | None = None
            if row["identity"] is not None:
                raw_identity = _exact(
                    row["identity"], _IDENTITY_FIELDS, f"event_results[{index}].identity"
                )
                ticker = _text(raw_identity["ticker"], f"event_results[{index}].identity.ticker")
                permaticker = raw_identity["permaticker"]
                if (
                    isinstance(permaticker, bool)
                    or not isinstance(permaticker, int)
                    or permaticker <= 0
                ):
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] has invalid permaticker"
                    )
                identity = {
                    "ticker": ticker,
                    "permaticker": permaticker,
                    "identity_id": _sha(
                        raw_identity["identity_id"],
                        f"event_results[{index}].identity.identity_id",
                    ),
                }
            raw_source = _exact(
                row["source_input"],
                _SOURCE_INPUT_FIELDS,
                f"event_results[{index}].source_input",
            )
            if raw_source["event_id"] != event_id or raw_source["event_key"] != event_key:
                raise PeadSignalInputReconciliationError(
                    f"event_results[{index}] source input identity differs"
                )
            metric = _exact(raw_source["metric"], _METRIC_FIELDS, f"event_results[{index}].metric")
            for field in _METRIC_FIELDS - {"metric_definition_sha256"}:
                _text(metric[field], f"event_results[{index}].metric.{field}")
            _sha(metric["metric_definition_sha256"], "metric_definition_sha256")
            _decimal(raw_source["actual_value"], "source actual")
            _decimal(raw_source["consensus_value"], "source consensus")
            _decimal(raw_source["raw_surprise"], "source surprise")
            if raw_source["surprise_direction"] not in {"positive", "negative", "zero"}:
                raise PeadSignalInputReconciliationError(
                    f"event_results[{index}] has invalid surprise direction"
                )
            if (
                isinstance(raw_source["analyst_count"], bool)
                or not isinstance(raw_source["analyst_count"], int)
                or raw_source["analyst_count"] < 1
            ):
                raise PeadSignalInputReconciliationError(
                    f"event_results[{index}] has invalid analyst count"
                )
            _utc(raw_source["known_public_by_at_utc"], "known_public_by_at_utc")
            _utc(raw_source["consensus_available_at_utc"], "consensus_available_at_utc")
            _utc(
                raw_source["consensus_receipt_captured_at_utc"],
                "consensus_receipt_captured_at_utc",
            )
            if not isinstance(raw_source["provenance"], Mapping):
                raise PeadSignalInputReconciliationError("source provenance must be an object")
            source_input = _plain(raw_source)
            raw_denominator = row["market_denominator"]
            if raw_denominator is not None:
                if not isinstance(raw_denominator, Mapping):
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] denominator must be an object"
                    )
                if identity is None or (
                    raw_denominator.get("ticker") != ticker
                    or raw_denominator.get("permaticker") != permaticker
                    or raw_denominator.get("identity_id") != identity["identity_id"]
                ):
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] denominator identity differs"
                    )
                _decimal(
                    raw_denominator.get("close_split_normalized"),
                    "market denominator close",
                    positive=True,
                )
                denominator = _plain(raw_denominator)

            if disposition == "signal_input_accepted":
                expected_cross_blockers = _cross_lane_blockers(source_input)
                if (
                    market_disposition != "market_accounting_evidenced"
                    or market_blockers
                    or reconciliation_blockers
                    or expected_cross_blockers
                    or identity is None
                    or denominator is None
                    or row["signal"] is None
                ):
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] accepted signal is incomplete"
                    )
                raw_signal = _exact(row["signal"], _SIGNAL_FIELDS, f"event_results[{index}].signal")
                ratio = _exact(
                    raw_signal["exact_ratio"],
                    _RATIO_FIELDS,
                    f"event_results[{index}].signal.exact_ratio",
                )
                if (
                    isinstance(ratio["numerator"], bool)
                    or not isinstance(ratio["numerator"], int)
                    or isinstance(ratio["denominator"], bool)
                    or not isinstance(ratio["denominator"], int)
                    or ratio["denominator"] <= 0
                    or Fraction(ratio["numerator"], ratio["denominator"]).denominator
                    != ratio["denominator"]
                ):
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] exact ratio is not reduced"
                    )
                expected_signal = _signal(source_input, denominator)
                if dict(raw_signal) != expected_signal:
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] signal math is not derived"
                    )
                signal = expected_signal
            else:
                if not reconciliation_blockers or row["signal"] is not None:
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] excluded signal has no final blocker"
                    )
                if market_disposition == "market_accounting_evidenced":
                    expected_cross_blockers = _cross_lane_blockers(source_input)
                    if (
                        market_blockers
                        or identity is None
                        or denominator is None
                        or not expected_cross_blockers
                        or reconciliation_blockers != expected_cross_blockers
                    ):
                        raise PeadSignalInputReconciliationError(
                            f"event_results[{index}] cross-lane exclusion is not derived"
                        )
                elif market_disposition == "market_accounting_excluded":
                    if not market_blockers or reconciliation_blockers != [
                        "market_accounting_not_evidenced"
                    ]:
                        raise PeadSignalInputReconciliationError(
                            f"event_results[{index}] market exclusion is not preserved"
                        )
                else:
                    raise PeadSignalInputReconciliationError(
                        f"event_results[{index}] reconciled source has invalid market state"
                    )
        normalized_rows.append(
            {
                "event_id": event_id,
                "event_key": event_key,
                "source_disposition": source_disposition,
                "market_disposition": market_disposition,
                "disposition": disposition,
                "source_blockers": source_blockers,
                "market_blockers": market_blockers,
                "reconciliation_blockers": reconciliation_blockers,
                "identity": identity,
                "source_input": source_input,
                "market_denominator": denominator,
                "signal": signal,
            }
        )
    if rows != normalized_rows:
        raise PeadSignalInputReconciliationError("event_results are not canonical")
    ids = [row["event_id"] for row in normalized_rows]
    if len(ids) != len(set(ids)):
        raise PeadSignalInputReconciliationError("event_results duplicate event IDs")

    coverage = _exact(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    qualification = _exact(payload["qualification"], _QUALIFICATION_FIELDS, "qualification")
    accepted = sum(row["disposition"] == "signal_input_accepted" for row in normalized_rows)
    source_reconciled = sum(
        row["source_disposition"] == "event_source_reconciled" for row in normalized_rows
    )
    market_evidenced = sum(
        row["market_disposition"] == "market_accounting_evidenced" for row in normalized_rows
    )
    expected_coverage = {
        "expected_event_count": len(normalized_rows),
        "source_reconciled_event_count": source_reconciled,
        "market_accounting_evidenced_count": market_evidenced,
        "signal_input_accepted_count": accepted,
        "signal_input_excluded_count": len(normalized_rows) - accepted,
        "exhaustive_event_accounting": True,
        "partial_coverage": accepted < len(normalized_rows),
        "blocker_counts": {key: blocker_counts[key] for key in sorted(blocker_counts)},
    }
    if dict(coverage) != expected_coverage:
        raise PeadSignalInputReconciliationError("coverage is not derived")
    allowed = accepted > 0
    expected_qualification = {
        "has_research_consumable_signal_inputs": allowed,
        "all_expected_events_signal_accepted": bool(
            normalized_rows and accepted == len(normalized_rows)
        ),
        "signal_input_reconciliation_allowed": allowed,
        "research_consumable": allowed,
        "historical_replication_allowed": (
            allowed and evidence_class == "historical_reconstruction"
        ),
        "prospective_accumulation_allowed": (allowed and evidence_class == "prospective_signal"),
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    if dict(qualification) != expected_qualification:
        raise PeadSignalInputReconciliationError("qualification is not derived")
    return {"artifact_hash": claimed, "payload": _plain(payload)}


def verify_pead_signal_input_reconciliation(
    document: Mapping[str, Any],
    source_reconciliation: Mapping[str, Any],
    market_accounting_evidence: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Authoritatively rebuild the final receipt from every original input."""
    normalized = validate_pead_signal_input_reconciliation_structure(document)
    expected = build_pead_signal_input_reconciliation(
        source_reconciliation,
        market_accounting_evidence,
        created_at_utc=normalized["payload"]["created_at_utc"],
        **kwargs,
    )
    if normalized != expected:
        raise PeadSignalInputReconciliationError(
            "signal-input reconciliation does not replay from authoritative inputs"
        )
    return expected


def _strict_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadSignalInputReconciliationError(
            f"signal-input receipt is not a regular file: {path}"
        )
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_SIGNAL_INPUT_RECONCILIATION_BYTES:
        raise PeadSignalInputReconciliationError("signal-input receipt file size is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadSignalInputReconciliationError("signal-input receipt is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadSignalInputReconciliationError(
                    f"signal-input receipt contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadSignalInputReconciliationError(
            f"signal-input receipt contains invalid number {token}"
        )

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadSignalInputReconciliationError("invalid signal-input JSON") from exc
    if not isinstance(value, dict):
        raise PeadSignalInputReconciliationError("signal-input receipt root must be an object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadSignalInputReconciliationError(
            "signal-input receipt bytes are not canonical JSON plus one newline"
        )
    return value


def publish_pead_signal_input_reconciliation(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    authoritative_verification_kwargs: Mapping[str, Any] | None = None,
    allow_structural_only: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Create one canonical receipt without ever replacing existing bytes.

    ``allow_structural_only=True`` explicitly selects internal validation;
    otherwise authoritative replay inputs are mandatory.  In both modes the
    just-created bytes are reopened through the strict JSON loader and
    structurally revalidated before success is returned.
    """
    if authoritative_verification_kwargs is None:
        if allow_structural_only is not True:
            raise PeadSignalInputReconciliationError(
                "publication requires authoritative verification or explicit "
                "allow_structural_only=True"
            )
        normalized = validate_pead_signal_input_reconciliation_structure(document)
    else:
        if allow_structural_only is not False:
            raise PeadSignalInputReconciliationError(
                "authoritative and structural-only publication modes are mutually exclusive"
            )
        if not isinstance(authoritative_verification_kwargs, Mapping):
            raise PeadSignalInputReconciliationError(
                "authoritative_verification_kwargs must be a mapping"
            )
        if "document" in authoritative_verification_kwargs:
            raise PeadSignalInputReconciliationError(
                "authoritative_verification_kwargs may not contain document"
            )
        normalized = verify_pead_signal_input_reconciliation(
            document, **dict(authoritative_verification_kwargs)
        )

    encoded = (canonical_json(normalized) + "\n").encode("utf-8")
    if len(encoded) > MAX_SIGNAL_INPUT_RECONCILIATION_BYTES:
        raise PeadSignalInputReconciliationError(
            "signal-input receipt exceeds the publication size limit"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise PeadSignalInputReconciliationError(
            f"signal-input publication parent is not a regular directory: {target.parent}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise PeadSignalInputReconciliationError(
            f"signal-input publication destination already exists: {target}"
        ) from exc
    except OSError as exc:
        raise PeadSignalInputReconciliationError(
            f"cannot create signal-input publication: {target}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PeadSignalInputReconciliationError(
            f"cannot durably write signal-input publication: {target}"
        ) from exc

    reread = validate_pead_signal_input_reconciliation_structure(_strict_json_file(target))
    if reread != normalized:
        raise PeadSignalInputReconciliationError(
            "published signal-input receipt differs from the validated document"
        )
    return normalized, target


def load_pead_signal_input_reconciliation(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Load canonical bytes and authoritatively replay every original input."""
    return verify_pead_signal_input_reconciliation(
        _strict_json_file(Path(path)),
        **kwargs,
    )


__all__ = [
    "MAX_SIGNAL_INPUT_RECONCILIATION_BYTES",
    "PeadSignalInputReconciliationError",
    "SIGNAL_INPUT_RECONCILIATION_POLICY_SCHEMA_VERSION",
    "SIGNAL_INPUT_RECONCILIATION_SCHEMA_VERSION",
    "build_pead_signal_input_reconciliation",
    "load_pead_signal_input_reconciliation",
    "publish_pead_signal_input_reconciliation",
    "validate_pead_signal_input_reconciliation_structure",
    "verify_pead_signal_input_reconciliation",
]
