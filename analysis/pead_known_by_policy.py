"""Conservative no-lookahead policy for PEAD ``known_public_by`` evidence.

An observation that an announcement was public at time *t* proves only that it
was public no later than *t*.  It does not prove that the announcement first
became public at *t*.  Consequently, same-calendar-day consensus observations
and market closes are not safe denominators: the true release may have
preceded them.  This module freezes the stricter, independently replayable
cutoffs used by the source-qualification v3 architecture.

The policy is intentionally pure.  It selects from already validated rows but
does not decide whether the upstream source bytes or availability proof are
trustworthy.  Those are separate evidence boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any
from zoneinfo import ZoneInfo


KNOWN_BY_POLICY_SCHEMA_VERSION = "pead_known_by_cutoff_policy.v1"
MINIMUM_ANALYST_COUNT = 2

_EASTERN = ZoneInfo("America/New_York")
_HEX = frozenset("0123456789abcdef")


class PeadKnownByPolicyError(ValueError):
    """An activation, consensus vintage, or market row is not canonical."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


KNOWN_BY_POLICY = {
    "schema_version": KNOWN_BY_POLICY_SCHEMA_VERSION,
    "activation_claim": "known_public_by",
    "activation_timezone": "America/New_York",
    "consensus_rule": ("provider_as_of_date_strictly_before_activation_eastern_date"),
    "same_day_consensus_allowed": False,
    "market_rule": (
        "latest_observed_session_close_with_session_date_strictly_before_activation_eastern_date"
    ),
    "same_day_market_close_allowed": False,
    "prospective_consensus_freeze_rule": (
        "acquisition_completed_no_later_than_selected_prior_session_close"
    ),
    "minimum_analyst_count": MINIMUM_ANALYST_COUNT,
    "ambiguity_rule": "exclude_without_fallback",
}
KNOWN_BY_POLICY_SHA256 = hashlib.sha256(
    _canonical_json(KNOWN_BY_POLICY).encode("utf-8")
).hexdigest()


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadKnownByPolicyError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadKnownByPolicyError(f"{label} must be canonical UTC with Z") from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadKnownByPolicyError(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _day(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PeadKnownByPolicyError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadKnownByPolicyError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadKnownByPolicyError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadKnownByPolicyError(f"{label} must be a lowercase SHA-256")
    return value


def activation_eastern_date(known_public_by_at_utc: str) -> date:
    """Return the civil date that defines both conservative cutoffs."""
    _, activation = _utc(known_public_by_at_utc, "known_public_by_at_utc")
    return activation.astimezone(_EASTERN).date()


def activation_eastern_day_start_utc(known_public_by_at_utc: str) -> datetime:
    """Return midnight ET for the activation date, converted to UTC."""
    local_day = activation_eastern_date(known_public_by_at_utc)
    return datetime.combine(local_day, time.min, tzinfo=_EASTERN).astimezone(timezone.utc)


def _vintage_order_key(row: Mapping[str, Any]) -> tuple[date, datetime]:
    as_of = _day(row.get("provider_as_of_date"), "provider_as_of_date")
    available = row.get("trusted_available_at_utc")
    if available is None:
        available_dt = datetime.combine(as_of, time.min, tzinfo=timezone.utc)
    else:
        _, available_dt = _utc(available, "trusted_available_at_utc")
    return as_of, available_dt


def select_conservative_consensus_vintage(
    vintages: Sequence[Mapping[str, Any]],
    *,
    known_public_by_at_utc: str,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Select the latest unambiguous prior-day consensus vintage.

    Exact intraday availability does not relax the prior-day rule because a
    ``known_public_by`` observation does not establish the true release time.
    The latest eligible vintage is evaluated as a unit: an ambiguous or
    under-covered latest vintage excludes the event rather than falling back to
    an older, more favorable record.
    """
    if isinstance(vintages, (str, bytes)) or not isinstance(vintages, Sequence):
        raise PeadKnownByPolicyError("vintages must be a sequence")
    activation_day = activation_eastern_date(known_public_by_at_utc)
    activation_day_start = activation_eastern_day_start_utc(known_public_by_at_utc)
    eligible: list[dict[str, Any]] = []
    for index, raw in enumerate(vintages):
        if not isinstance(raw, Mapping):
            raise PeadKnownByPolicyError(f"vintages[{index}] must be an object")
        row = dict(raw)
        as_of, available = _vintage_order_key(row)
        exact_available = row.get("trusted_available_at_utc")
        exact_is_prior = exact_available is None or available < activation_day_start
        if as_of < activation_day and exact_is_prior:
            eligible.append(row)
    if not eligible:
        return None, ["consensus_no_eligible_prior_day_vintage"], []

    latest_key = max(_vintage_order_key(row) for row in eligible)
    latest = [row for row in eligible if _vintage_order_key(row) == latest_key]
    raw_hashes = sorted(_sha(row.get("raw_record_sha256"), "raw_record_sha256") for row in latest)
    if len(latest) != 1:
        return None, ["consensus_latest_vintage_ambiguous"], raw_hashes
    selected = latest[0]
    analyst_count = selected.get("analyst_count")
    if type(analyst_count) is not int or analyst_count < 1:
        raise PeadKnownByPolicyError("analyst_count must be a positive integer")
    if analyst_count < MINIMUM_ANALYST_COUNT:
        return (
            selected,
            ["consensus_analyst_count_below_minimum"],
            raw_hashes,
        )
    return selected, [], raw_hashes


def select_conservative_market_session(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_public_by_at_utc: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Select the latest unique positive SEP session from a prior ET date."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise PeadKnownByPolicyError("market rows must be a sequence")
    activation_day = activation_eastern_date(known_public_by_at_utc)
    _, activation = _utc(known_public_by_at_utc, "known_public_by_at_utc")
    eligible: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PeadKnownByPolicyError(f"market rows[{index}] must be an object")
        row = dict(raw)
        session_day = _day(row.get("session_date"), "session_date")
        _, close_at = _utc(row.get("session_close_at_utc"), "session_close_at_utc")
        if close_at >= activation:
            raise PeadKnownByPolicyError("a supplied market session closes at or after activation")
        for field in ("close", "closeunadj"):
            value = row.get(field)
            if not isinstance(value, str):
                raise PeadKnownByPolicyError(f"{field} must be a decimal string")
            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise PeadKnownByPolicyError(f"{field} must be a positive finite decimal") from exc
            if not number.is_finite() or number <= 0:
                raise PeadKnownByPolicyError(f"{field} must be a positive finite decimal")
        _sha(row.get("source_row_sha256"), "source_row_sha256")
        if session_day < activation_day:
            eligible.append(row)
    if not eligible:
        return None, ["market_prior_session_absent"]

    latest_day = max(_day(row["session_date"], "session_date") for row in eligible)
    latest = [row for row in eligible if _day(row["session_date"], "session_date") == latest_day]
    if len(latest) != 1:
        return None, ["market_latest_prior_session_ambiguous"]
    return latest[0], []


def validate_prospective_consensus_freeze(
    *,
    acquired_at_utc: str,
    selected_prior_session_close_at_utc: str,
) -> None:
    """Require a prospective consensus snapshot by the prior close.

    Equality is accepted because the policy says "no later than" the selected
    prior close.  Callers still need a source receipt proving the acquisition
    timestamp; this helper validates only chronology.
    """
    _, acquired = _utc(acquired_at_utc, "acquired_at_utc")
    _, close = _utc(
        selected_prior_session_close_at_utc,
        "selected_prior_session_close_at_utc",
    )
    if acquired > close:
        raise PeadKnownByPolicyError(
            "prospective consensus was acquired after the prior-session freeze"
        )


__all__ = [
    "KNOWN_BY_POLICY",
    "KNOWN_BY_POLICY_SCHEMA_VERSION",
    "KNOWN_BY_POLICY_SHA256",
    "MINIMUM_ANALYST_COUNT",
    "PeadKnownByPolicyError",
    "activation_eastern_date",
    "activation_eastern_day_start_utc",
    "select_conservative_consensus_vintage",
    "select_conservative_market_session",
    "validate_prospective_consensus_freeze",
]
