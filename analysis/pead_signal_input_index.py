"""Authoritative cross-partition index for final PEAD signal inputs.

The v3 replication target is partitioned into ten calendar-year event
universes.  Each year has its own ``pead_signal_input_reconciliation.v1``
receipt because copying every event into one more monolithic receipt would
make review and replay unnecessarily expensive.  This module binds those ten
receipts to the exact Sharadar event-replay and event-universe-index roots,
proves one ordered and exhaustive signal disposition per frozen event, and
aggregates only counts and content identities.

``validate_pead_signal_input_index_structure`` is deliberately not an
authority boundary.  It validates the compact document's identity, shape,
derived counts, and fail-closed permissions, but it cannot recover event IDs
from their hashes.  ``verify_pead_signal_input_index`` first authoritatively
replays the Sharadar event root and every annual signal receipt using
caller-supplied verification contexts, then rebuilds this index exactly.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from analysis.pead_signal_input_reconciliation import (
    PeadSignalInputReconciliationError,
    validate_pead_signal_input_reconciliation_structure,
    verify_pead_signal_input_reconciliation,
)
from data.pead_event_universe import canonical_json, content_hash
from data.pead_event_universe_index import (
    EVENT_UNIVERSE_INDEX_SCHEMA_VERSION,
    PARTITION_POLICY,
    PeadEventUniverseIndexError,
    validate_pead_event_universe_index,
)
from data.pead_sharadar_event_universe_replay import (
    TARGET_END,
    TARGET_START,
    PeadSharadarEventUniverseReplayError,
    validate_pead_sharadar_event_universe_replay_structure,
    verify_pead_sharadar_event_universe_replay,
)


SIGNAL_INPUT_INDEX_SCHEMA_VERSION = "pead_signal_input_index.v1"
SIGNAL_INPUT_INDEX_POLICY_SCHEMA_VERSION = "pead_signal_input_index_policy.v1"
TRUST_ROOT_SET_SCHEMA_VERSION = "pead_sha256_trust_root_set.v1"
MAX_SIGNAL_INPUT_INDEX_BYTES = 8 * 1024 * 1024

_HEX = frozenset("0123456789abcdef")
_MACHINE_REASON = re.compile(r"^[a-z][a-z0-9_]*$")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "indexed_at_utc",
    "target_window",
    "policy",
    "trust_policy",
    "bindings",
    "partitions",
    "coverage",
    "qualification",
}
_WINDOW_FIELDS = {"start", "end"}
_TRUST_POLICY_FIELDS = {
    "candidate_specification_set_sha256",
    "construction_code_set_sha256",
    "signal_reconciliation_code_set_sha256",
    "signal_input_reconciliation_set_sha256",
}
_BINDING_FIELDS = {
    "event_universe_replay_sha256",
    "event_universe_index_sha256",
    "candidate_specification_sha256",
    "construction_code_sha256",
    "signal_reconciliation_code_sha256",
    "event_universe_partition_policy_sha256",
    "partition_event_id_manifests_sha256",
    "signal_input_index_policy_sha256",
}
_PARTITION_FIELDS = {
    "partition_id",
    "event_window",
    "event_universe_sha256",
    "event_universe_expected_event_ids_sha256",
    "signal_input_reconciliation_sha256",
    "signal_input_event_ids_sha256",
    "signal_input_created_at_utc",
    "event_count",
    "source_reconciled_event_count",
    "market_accounting_evidenced_count",
    "signal_input_accepted_count",
    "signal_input_excluded_count",
    "blocker_counts",
    "research_consumable",
}
_COVERAGE_FIELDS = {
    "partition_count",
    "expected_event_count",
    "source_reconciled_event_count",
    "market_accounting_evidenced_count",
    "signal_input_accepted_count",
    "signal_input_excluded_count",
    "exhaustive_event_accounting",
    "cross_partition_event_ids_unique",
    "partial_coverage",
    "blocker_counts",
}
_QUALIFICATION_FIELDS = {
    "has_research_consumable_signal_inputs",
    "all_expected_events_signal_accepted",
    "all_partitions_research_consumable",
    "signal_input_index_allowed",
    "research_consumable",
    "historical_replication_allowed",
    "edge_claim_allowed",
    "paper_execution_allowed",
    "live_deployment_allowed",
}

INDEX_POLICY = {
    "schema_version": SIGNAL_INPUT_INDEX_POLICY_SCHEMA_VERSION,
    "target_window": {"start": TARGET_START, "end": TARGET_END},
    "partition_rule": "exactly_one_calendar_year_receipt_for_each_2015_through_2024_partition",
    "event_root_rule": (
        "exact_authoritatively_replayed_sharadar_event_replay_and_event_universe_index_roots"
    ),
    "child_rule": "authoritatively_replayed_pead_signal_input_reconciliation_v1_only",
    "trust_rule": "every_child_and_common_specification_and_code_hash_externally_admitted",
    "event_accounting_rule": (
        "child_event_ids_and_keys_exactly_equal_bound_universe_order_with_no_cross_year_duplicates"
    ),
    "aggregation_rule": "sum_only_replayed_child_coverage_and_blocker_counts",
    "research_rule": "only_signal_input_accepted_rows_are_research_consumable",
    "edge_claim_allowed": False,
    "paper_execution_allowed": False,
    "live_deployment_allowed": False,
}


class PeadSignalInputIndexError(ValueError):
    """The annual signal-input index is malformed or cannot replay."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadSignalInputIndexError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadSignalInputIndexError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PeadSignalInputIndexError(f"{label} must be nonempty canonical text")
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadSignalInputIndexError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadSignalInputIndexError(f"{label} must be canonical UTC with Z") from exc
    rendered = parsed.isoformat().replace("+00:00", "Z")
    if rendered != value:
        raise PeadSignalInputIndexError(f"{label} is not canonical UTC")
    return rendered, parsed


def _day(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PeadSignalInputIndexError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadSignalInputIndexError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadSignalInputIndexError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def _count(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise PeadSignalInputIndexError(f"{label} must be a {qualifier} integer")
    return value


def _trust_roots(values: Collection[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PeadSignalInputIndexError(f"{label} must be a hash collection")
    try:
        roots = tuple(sorted({_sha(value, label) for value in values}))
    except TypeError as exc:
        raise PeadSignalInputIndexError(f"{label} must be a hash collection") from exc
    if not roots:
        raise PeadSignalInputIndexError(f"{label} external trust registry is empty")
    return roots


def _trust_set_hash(values: Collection[str]) -> str:
    return content_hash(
        {
            "schema_version": TRUST_ROOT_SET_SCHEMA_VERSION,
            "members": sorted(values),
        }
    )


def _require_trusted(claimed: str, roots: Collection[str], label: str) -> None:
    if claimed not in roots:
        raise PeadSignalInputIndexError(f"{label} is absent from its external trust registry")


def _partition_windows() -> list[tuple[str, str, str]]:
    start = date.fromisoformat(TARGET_START)
    end = date.fromisoformat(TARGET_END)
    return [
        (
            str(year),
            max(start, date(year, 1, 1)).isoformat(),
            min(end, date(year, 12, 31)).isoformat(),
        )
        for year in range(start.year, end.year + 1)
    ]


def _blocker_counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise PeadSignalInputIndexError(f"{label} must be an object")
    normalized: dict[str, int] = {}
    for raw_reason, raw_count in value.items():
        if not isinstance(raw_reason, str) or _MACHINE_REASON.fullmatch(raw_reason) is None:
            raise PeadSignalInputIndexError(f"{label} contains an invalid reason")
        normalized[raw_reason] = _count(raw_count, f"{label}.{raw_reason}", positive=True)
    return {reason: normalized[reason] for reason in sorted(normalized)}


def _replay_parts(
    event_universe_replay: Mapping[str, Any],
    event_universe_index: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        replay = validate_pead_sharadar_event_universe_replay_structure(
            event_universe_replay
        )
        partitions = [year["event_universe"] for year in replay["payload"]["years"]]
        index = validate_pead_event_universe_index(
            event_universe_index, partitions=partitions
        )
    except (
        PeadSharadarEventUniverseReplayError,
        PeadEventUniverseIndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise PeadSignalInputIndexError(
            "event-universe replay and index do not validate structurally"
        ) from exc
    if index["payload"]["schema_version"] != EVENT_UNIVERSE_INDEX_SCHEMA_VERSION:
        raise PeadSignalInputIndexError("unsupported event-universe index schema")
    if replay["payload"]["target_window"] != {
        "start": TARGET_START,
        "end": TARGET_END,
    } or index["payload"]["target_window"] != {
        "start": TARGET_START,
        "end": TARGET_END,
    }:
        raise PeadSignalInputIndexError("event roots differ from the fixed research target")
    if index["payload"]["indexed_at_utc"] != replay["payload"]["created_at_utc"]:
        raise PeadSignalInputIndexError("event replay and index timestamps differ")
    if len(partitions) != 10:
        raise PeadSignalInputIndexError("event replay must contain exactly ten annual partitions")
    return replay, index, partitions


def build_pead_signal_input_index(
    event_universe_replay: Mapping[str, Any],
    event_universe_index: Mapping[str, Any],
    signal_input_reconciliations: Sequence[Mapping[str, Any]],
    *,
    indexed_at_utc: str,
    trusted_signal_input_reconciliation_sha256s: Collection[str],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
    trusted_signal_reconciliation_code_sha256s: Collection[str],
) -> dict[str, Any]:
    """Build the compact root after structural validation of every child.

    This builder requires external admissions and exact child coverage, but it
    does not authoritatively replay the children.  Use
    :func:`verify_pead_signal_input_index` for that boundary.
    """
    if isinstance(signal_input_reconciliations, (str, bytes)) or not isinstance(
        signal_input_reconciliations, Sequence
    ):
        raise PeadSignalInputIndexError("signal_input_reconciliations must be a sequence")
    if len(signal_input_reconciliations) != 10:
        raise PeadSignalInputIndexError(
            "exactly ten annual signal-input reconciliations are required"
        )
    trust_sets = {
        "signals": _trust_roots(
            trusted_signal_input_reconciliation_sha256s,
            "trusted_signal_input_reconciliation_sha256s",
        ),
        "candidate": _trust_roots(
            trusted_candidate_specification_sha256s,
            "trusted_candidate_specification_sha256s",
        ),
        "construction": _trust_roots(
            trusted_construction_code_sha256s,
            "trusted_construction_code_sha256s",
        ),
        "signal_code": _trust_roots(
            trusted_signal_reconciliation_code_sha256s,
            "trusted_signal_reconciliation_code_sha256s",
        ),
    }
    replay, universe_index, universes = _replay_parts(
        event_universe_replay, event_universe_index
    )
    indexed_text, indexed_at = _utc(indexed_at_utc, "indexed_at_utc")
    replay_time = _utc(replay["payload"]["created_at_utc"], "event replay time")[1]
    if indexed_at < replay_time:
        raise PeadSignalInputIndexError("signal-input index predates its event roots")

    replay_payload = replay["payload"]
    index_payload = universe_index["payload"]
    candidate_id = _text(replay_payload["candidate_id"], "candidate_id")
    if index_payload["candidate_id"] != candidate_id:
        raise PeadSignalInputIndexError("event replay and index candidates differ")
    candidate_hash = _sha(
        replay_payload["bindings"]["candidate_specification_sha256"],
        "candidate specification binding",
    )
    construction_hash = _sha(
        replay_payload["bindings"]["construction_code_sha256"],
        "construction code binding",
    )
    if (
        index_payload["bindings"]["candidate_specification_sha256"] != candidate_hash
        or index_payload["bindings"]["construction_code_sha256"] != construction_hash
    ):
        raise PeadSignalInputIndexError(
            "event replay and index specification/code bindings differ"
        )
    _require_trusted(candidate_hash, trust_sets["candidate"], "candidate specification")
    _require_trusted(construction_hash, trust_sets["construction"], "construction code")

    normalized_signals: list[dict[str, Any]] = []
    try:
        for raw in signal_input_reconciliations:
            normalized_signals.append(
                validate_pead_signal_input_reconciliation_structure(raw)
            )
    except (PeadSignalInputReconciliationError, TypeError, ValueError) as exc:
        raise PeadSignalInputIndexError(
            "an annual signal-input reconciliation is structurally invalid"
        ) from exc

    by_universe: dict[str, dict[str, Any]] = {}
    for signal in normalized_signals:
        signal_hash = _sha(signal["artifact_hash"], "signal-input reconciliation hash")
        _require_trusted(signal_hash, trust_sets["signals"], "signal-input reconciliation")
        universe_hash = _sha(
            signal["payload"]["bindings"]["event_universe_sha256"],
            "signal event-universe binding",
        )
        if universe_hash in by_universe:
            raise PeadSignalInputIndexError(
                "more than one signal-input receipt binds an annual event universe"
            )
        by_universe[universe_hash] = signal

    expected_universe_hashes = [partition["artifact_hash"] for partition in universes]
    if set(by_universe) != set(expected_universe_hashes):
        raise PeadSignalInputIndexError(
            "signal-input receipts do not cover every annual event universe exactly once"
        )

    expected_candidate_trust = _trust_set_hash(trust_sets["candidate"])
    expected_construction_trust = _trust_set_hash(trust_sets["construction"])
    expected_signal_code_trust = _trust_set_hash(trust_sets["signal_code"])
    signal_code_hash: str | None = None
    descriptors: list[dict[str, Any]] = []
    global_event_ids: list[str] = []
    aggregate_blockers: dict[str, int] = {}
    expected_windows = _partition_windows()
    index_descriptors = index_payload["partitions"]
    if len(index_descriptors) != len(expected_windows):
        raise PeadSignalInputIndexError("event-universe index partition count differs")

    for position, ((partition_id, start, end), universe, universe_descriptor) in enumerate(
        zip(expected_windows, universes, index_descriptors, strict=True)
    ):
        universe_payload = universe["payload"]
        expected_window = {"start": start, "end": end}
        if (
            replay_payload["years"][position]["partition_id"] != partition_id
            or replay_payload["years"][position]["event_window"] != expected_window
            or universe_payload["event_window"] != expected_window
            or universe_descriptor["partition_id"] != partition_id
            or universe_descriptor["event_window"] != expected_window
            or universe_descriptor["event_universe_sha256"] != universe["artifact_hash"]
        ):
            raise PeadSignalInputIndexError("annual event roots are not in exact target order")
        signal = by_universe[universe["artifact_hash"]]
        signal_payload = signal["payload"]
        bindings = signal_payload["bindings"]
        trust = signal_payload["trust_policy"]
        if signal_payload["candidate_id"] != candidate_id:
            raise PeadSignalInputIndexError("annual signal receipts have different candidates")
        if signal_payload["evidence_class"] != "historical_reconstruction":
            raise PeadSignalInputIndexError(
                "the fixed replication target requires historical reconstruction receipts"
            )
        if (
            bindings["candidate_specification_sha256"] != candidate_hash
            or bindings["construction_code_sha256"] != construction_hash
        ):
            raise PeadSignalInputIndexError(
                "an annual signal receipt binds another specification or construction code"
            )
        child_signal_code = _sha(
            bindings["signal_reconciliation_code_sha256"],
            "signal reconciliation code binding",
        )
        _require_trusted(
            child_signal_code, trust_sets["signal_code"], "signal reconciliation code"
        )
        if signal_code_hash is None:
            signal_code_hash = child_signal_code
        elif signal_code_hash != child_signal_code:
            raise PeadSignalInputIndexError(
                "annual signal receipts bind different final reconciliation code"
            )
        if (
            trust["candidate_specification_set_sha256"] != expected_candidate_trust
            or trust["construction_code_set_sha256"] != expected_construction_trust
            or trust["signal_reconciliation_code_set_sha256"]
            != expected_signal_code_trust
        ):
            raise PeadSignalInputIndexError(
                "an annual signal receipt differs from the common external trust registries"
            )

        expected_events = universe_payload["expected_events"]
        expected_ids = universe_payload["expected_event_ids"]
        signal_rows = signal_payload["event_results"]
        signal_ids = [row["event_id"] for row in signal_rows]
        signal_keys = [row["event_key"] for row in signal_rows]
        if signal_ids != expected_ids or signal_keys != [row["event_key"] for row in expected_events]:
            raise PeadSignalInputIndexError(
                f"partition {partition_id} signal events differ from the bound universe order"
            )
        for event in expected_events:
            period = _day(event["event_key"]["fiscal_period_end"], "fiscal period end")
            if not date.fromisoformat(start) <= period <= date.fromisoformat(end):
                raise PeadSignalInputIndexError(
                    f"partition {partition_id} contains an event outside its window"
                )
        if set(global_event_ids).intersection(signal_ids):
            raise PeadSignalInputIndexError("event IDs appear in more than one annual receipt")
        global_event_ids.extend(signal_ids)

        created_text, child_created = _utc(
            signal_payload["created_at_utc"], f"partition {partition_id} created_at_utc"
        )
        if indexed_at < child_created:
            raise PeadSignalInputIndexError(
                f"signal-input index predates partition {partition_id}"
            )
        coverage = signal_payload["coverage"]
        if coverage["expected_event_count"] != len(expected_ids):
            raise PeadSignalInputIndexError(
                f"partition {partition_id} coverage differs from its event universe"
            )
        child_blockers = _blocker_counts(
            coverage["blocker_counts"], f"partition {partition_id} blocker_counts"
        )
        for reason, count in child_blockers.items():
            aggregate_blockers[reason] = aggregate_blockers.get(reason, 0) + count
        expected_ids_hash = content_hash(expected_ids)
        descriptors.append(
            {
                "partition_id": partition_id,
                "event_window": expected_window,
                "event_universe_sha256": universe["artifact_hash"],
                "event_universe_expected_event_ids_sha256": expected_ids_hash,
                "signal_input_reconciliation_sha256": signal["artifact_hash"],
                "signal_input_event_ids_sha256": content_hash(signal_ids),
                "signal_input_created_at_utc": created_text,
                "event_count": coverage["expected_event_count"],
                "source_reconciled_event_count": coverage[
                    "source_reconciled_event_count"
                ],
                "market_accounting_evidenced_count": coverage[
                    "market_accounting_evidenced_count"
                ],
                "signal_input_accepted_count": coverage[
                    "signal_input_accepted_count"
                ],
                "signal_input_excluded_count": coverage[
                    "signal_input_excluded_count"
                ],
                "blocker_counts": child_blockers,
                "research_consumable": signal_payload["qualification"][
                    "research_consumable"
                ],
            }
        )

    if signal_code_hash is None:  # pragma: no cover - ten nonempty children are required
        raise PeadSignalInputIndexError("no final reconciliation code binding was found")
    if len(global_event_ids) != len(set(global_event_ids)):
        raise PeadSignalInputIndexError("event IDs cross annual signal-input receipts")

    event_count = sum(row["event_count"] for row in descriptors)
    accepted_count = sum(row["signal_input_accepted_count"] for row in descriptors)
    source_count = sum(row["source_reconciled_event_count"] for row in descriptors)
    market_count = sum(
        row["market_accounting_evidenced_count"] for row in descriptors
    )
    all_research = all(row["research_consumable"] for row in descriptors)
    allowed = accepted_count > 0
    manifests = [
        {
            "partition_id": row["partition_id"],
            "event_ids_sha256": row["event_universe_expected_event_ids_sha256"],
        }
        for row in descriptors
    ]
    payload = {
        "schema_version": SIGNAL_INPUT_INDEX_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "evidence_class": "historical_reconstruction",
        "indexed_at_utc": indexed_text,
        "target_window": {"start": TARGET_START, "end": TARGET_END},
        "policy": INDEX_POLICY,
        "trust_policy": {
            "candidate_specification_set_sha256": expected_candidate_trust,
            "construction_code_set_sha256": expected_construction_trust,
            "signal_reconciliation_code_set_sha256": expected_signal_code_trust,
            "signal_input_reconciliation_set_sha256": _trust_set_hash(
                trust_sets["signals"]
            ),
        },
        "bindings": {
            "event_universe_replay_sha256": replay["artifact_hash"],
            "event_universe_index_sha256": universe_index["artifact_hash"],
            "candidate_specification_sha256": candidate_hash,
            "construction_code_sha256": construction_hash,
            "signal_reconciliation_code_sha256": signal_code_hash,
            "event_universe_partition_policy_sha256": content_hash(PARTITION_POLICY),
            "partition_event_id_manifests_sha256": content_hash(manifests),
            "signal_input_index_policy_sha256": content_hash(INDEX_POLICY),
        },
        "partitions": descriptors,
        "coverage": {
            "partition_count": len(descriptors),
            "expected_event_count": event_count,
            "source_reconciled_event_count": source_count,
            "market_accounting_evidenced_count": market_count,
            "signal_input_accepted_count": accepted_count,
            "signal_input_excluded_count": event_count - accepted_count,
            "exhaustive_event_accounting": len(global_event_ids) == event_count,
            "cross_partition_event_ids_unique": len(global_event_ids)
            == len(set(global_event_ids)),
            "partial_coverage": accepted_count < event_count,
            "blocker_counts": {
                reason: aggregate_blockers[reason]
                for reason in sorted(aggregate_blockers)
            },
        },
        "qualification": {
            "has_research_consumable_signal_inputs": allowed,
            "all_expected_events_signal_accepted": bool(
                event_count and accepted_count == event_count
            ),
            "all_partitions_research_consumable": all_research,
            "signal_input_index_allowed": allowed,
            "research_consumable": allowed,
            "historical_replication_allowed": allowed,
            "edge_claim_allowed": False,
            "paper_execution_allowed": False,
            "live_deployment_allowed": False,
        },
    }
    return validate_pead_signal_input_index_structure(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_signal_input_index_structure(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate compact identity and derivations only; this is not authoritative."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "signal-input index")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "signal-input index.payload")
    claimed = _sha(wrapper["artifact_hash"], "signal-input index.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadSignalInputIndexError("signal-input index artifact hash mismatch")
    if payload["schema_version"] != SIGNAL_INPUT_INDEX_SCHEMA_VERSION:
        raise PeadSignalInputIndexError("unsupported signal-input index schema")
    _text(payload["candidate_id"], "candidate_id")
    if payload["evidence_class"] != "historical_reconstruction":
        raise PeadSignalInputIndexError("signal-input index evidence class differs")
    indexed_text, indexed_at = _utc(payload["indexed_at_utc"], "indexed_at_utc")
    target = _exact(payload["target_window"], _WINDOW_FIELDS, "target_window")
    if dict(target) != {"start": TARGET_START, "end": TARGET_END}:
        raise PeadSignalInputIndexError("signal-input index target window differs")
    if payload["policy"] != INDEX_POLICY:
        raise PeadSignalInputIndexError("signal-input index policy differs")
    trust = _exact(payload["trust_policy"], _TRUST_POLICY_FIELDS, "trust_policy")
    bindings = _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    for field in sorted(_TRUST_POLICY_FIELDS):
        _sha(trust[field], f"trust_policy.{field}")
    for field in sorted(_BINDING_FIELDS):
        _sha(bindings[field], f"bindings.{field}")
    if bindings["event_universe_partition_policy_sha256"] != content_hash(
        PARTITION_POLICY
    ):
        raise PeadSignalInputIndexError("event partition policy binding differs")
    if bindings["signal_input_index_policy_sha256"] != content_hash(INDEX_POLICY):
        raise PeadSignalInputIndexError("signal-input index policy binding differs")

    raw_partitions = payload["partitions"]
    expected_windows = _partition_windows()
    if not isinstance(raw_partitions, list) or len(raw_partitions) != len(expected_windows):
        raise PeadSignalInputIndexError(
            "signal-input index must contain exactly ten annual partitions"
        )
    partitions: list[dict[str, Any]] = []
    universe_hashes: list[str] = []
    signal_hashes: list[str] = []
    aggregate_blockers: dict[str, int] = {}
    for position, (raw, (partition_id, start, end)) in enumerate(
        zip(raw_partitions, expected_windows, strict=True)
    ):
        row = _exact(raw, _PARTITION_FIELDS, f"partitions[{position}]")
        if row["partition_id"] != partition_id:
            raise PeadSignalInputIndexError("annual partition IDs are not canonical")
        window = _exact(
            row["event_window"], _WINDOW_FIELDS, f"partitions[{position}].event_window"
        )
        if dict(window) != {"start": start, "end": end}:
            raise PeadSignalInputIndexError("annual partition windows are not canonical")
        universe_hash = _sha(
            row["event_universe_sha256"], f"partitions[{position}].event_universe_sha256"
        )
        expected_ids_hash = _sha(
            row["event_universe_expected_event_ids_sha256"],
            f"partitions[{position}].event_universe_expected_event_ids_sha256",
        )
        signal_hash = _sha(
            row["signal_input_reconciliation_sha256"],
            f"partitions[{position}].signal_input_reconciliation_sha256",
        )
        signal_ids_hash = _sha(
            row["signal_input_event_ids_sha256"],
            f"partitions[{position}].signal_input_event_ids_sha256",
        )
        if signal_ids_hash != expected_ids_hash:
            raise PeadSignalInputIndexError(
                "a partition signal event manifest differs from its event universe"
            )
        created_text, created_at = _utc(
            row["signal_input_created_at_utc"],
            f"partitions[{position}].signal_input_created_at_utc",
        )
        if indexed_at < created_at:
            raise PeadSignalInputIndexError("signal-input index predates an annual receipt")
        event_count = _count(
            row["event_count"], f"partitions[{position}].event_count", positive=True
        )
        source_count = _count(
            row["source_reconciled_event_count"],
            f"partitions[{position}].source_reconciled_event_count",
        )
        market_count = _count(
            row["market_accounting_evidenced_count"],
            f"partitions[{position}].market_accounting_evidenced_count",
        )
        accepted_count = _count(
            row["signal_input_accepted_count"],
            f"partitions[{position}].signal_input_accepted_count",
        )
        excluded_count = _count(
            row["signal_input_excluded_count"],
            f"partitions[{position}].signal_input_excluded_count",
        )
        if (
            source_count > event_count
            or market_count > event_count
            or market_count > source_count
            or accepted_count > source_count
            or accepted_count > market_count
            or accepted_count + excluded_count != event_count
        ):
            raise PeadSignalInputIndexError("annual coverage counts are inconsistent")
        child_blockers = _blocker_counts(
            row["blocker_counts"], f"partitions[{position}].blocker_counts"
        )
        for reason, count in child_blockers.items():
            aggregate_blockers[reason] = aggregate_blockers.get(reason, 0) + count
        research_consumable = row["research_consumable"]
        if type(research_consumable) is not bool or research_consumable is not (
            accepted_count > 0
        ):
            raise PeadSignalInputIndexError(
                "annual research-consumable status is not derived"
            )
        partitions.append(
            {
                "partition_id": partition_id,
                "event_window": {"start": start, "end": end},
                "event_universe_sha256": universe_hash,
                "event_universe_expected_event_ids_sha256": expected_ids_hash,
                "signal_input_reconciliation_sha256": signal_hash,
                "signal_input_event_ids_sha256": signal_ids_hash,
                "signal_input_created_at_utc": created_text,
                "event_count": event_count,
                "source_reconciled_event_count": source_count,
                "market_accounting_evidenced_count": market_count,
                "signal_input_accepted_count": accepted_count,
                "signal_input_excluded_count": excluded_count,
                "blocker_counts": child_blockers,
                "research_consumable": research_consumable,
            }
        )
        universe_hashes.append(universe_hash)
        signal_hashes.append(signal_hash)
    if len(universe_hashes) != len(set(universe_hashes)):
        raise PeadSignalInputIndexError("annual event-universe roots are duplicated")
    if len(signal_hashes) != len(set(signal_hashes)):
        raise PeadSignalInputIndexError("annual signal-input receipt roots are duplicated")
    if raw_partitions != partitions:
        raise PeadSignalInputIndexError("signal-input index partitions are not canonical")

    manifests = [
        {
            "partition_id": row["partition_id"],
            "event_ids_sha256": row["event_universe_expected_event_ids_sha256"],
        }
        for row in partitions
    ]
    if bindings["partition_event_id_manifests_sha256"] != content_hash(manifests):
        raise PeadSignalInputIndexError("partition event manifest binding is not derived")

    coverage = _exact(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    expected_event_count = sum(row["event_count"] for row in partitions)
    accepted_count = sum(row["signal_input_accepted_count"] for row in partitions)
    expected_coverage = {
        "partition_count": len(partitions),
        "expected_event_count": expected_event_count,
        "source_reconciled_event_count": sum(
            row["source_reconciled_event_count"] for row in partitions
        ),
        "market_accounting_evidenced_count": sum(
            row["market_accounting_evidenced_count"] for row in partitions
        ),
        "signal_input_accepted_count": accepted_count,
        "signal_input_excluded_count": sum(
            row["signal_input_excluded_count"] for row in partitions
        ),
        "exhaustive_event_accounting": True,
        "cross_partition_event_ids_unique": True,
        "partial_coverage": accepted_count < expected_event_count,
        "blocker_counts": {
            reason: aggregate_blockers[reason] for reason in sorted(aggregate_blockers)
        },
    }
    if dict(coverage) != expected_coverage:
        raise PeadSignalInputIndexError("signal-input index coverage is not derived")

    qualification = _exact(
        payload["qualification"], _QUALIFICATION_FIELDS, "qualification"
    )
    allowed = accepted_count > 0
    expected_qualification = {
        "has_research_consumable_signal_inputs": allowed,
        "all_expected_events_signal_accepted": bool(
            expected_event_count and accepted_count == expected_event_count
        ),
        "all_partitions_research_consumable": all(
            row["research_consumable"] for row in partitions
        ),
        "signal_input_index_allowed": allowed,
        "research_consumable": allowed,
        "historical_replication_allowed": allowed,
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    if dict(qualification) != expected_qualification:
        raise PeadSignalInputIndexError("signal-input index qualification is not derived")
    normalized_payload = {
        **payload,
        "indexed_at_utc": indexed_text,
        "partitions": partitions,
        "coverage": expected_coverage,
        "qualification": expected_qualification,
    }
    return {"artifact_hash": claimed, "payload": _plain(normalized_payload)}


def verify_pead_signal_input_index(
    document: Mapping[str, Any],
    event_universe_replay: Mapping[str, Any],
    event_universe_index: Mapping[str, Any],
    signal_input_reconciliations: Sequence[Mapping[str, Any]],
    *,
    event_replay_verification_kwargs: Mapping[str, Any],
    child_verification_contexts: Mapping[str, Mapping[str, Any]],
    trusted_signal_input_reconciliation_sha256s: Collection[str],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
    trusted_signal_reconciliation_code_sha256s: Collection[str],
) -> dict[str, Any]:
    """Authoritatively replay the event roots and every annual final receipt."""
    normalized = validate_pead_signal_input_index_structure(document)
    if not isinstance(event_replay_verification_kwargs, Mapping):
        raise PeadSignalInputIndexError(
            "event_replay_verification_kwargs must be a mapping"
        )
    replay_kwargs = dict(event_replay_verification_kwargs)
    for forbidden in ("replay", "index"):
        if forbidden in replay_kwargs:
            raise PeadSignalInputIndexError(
                f"event_replay_verification_kwargs may not contain {forbidden}"
            )
    if not isinstance(child_verification_contexts, Mapping):
        raise PeadSignalInputIndexError("child_verification_contexts must be a mapping")
    expected_ids = [partition_id for partition_id, _, _ in _partition_windows()]
    if set(child_verification_contexts) != set(expected_ids):
        raise PeadSignalInputIndexError(
            "child verification contexts must cover exactly partitions 2015 through 2024"
        )
    normalized_contexts: dict[str, dict[str, Any]] = {}
    for partition_id in expected_ids:
        context = child_verification_contexts[partition_id]
        if not isinstance(context, Mapping):
            raise PeadSignalInputIndexError(
                f"partition {partition_id} verification context must be a mapping"
            )
        child_kwargs = dict(context)
        if "document" in child_kwargs:
            raise PeadSignalInputIndexError(
                f"partition {partition_id} verification context may not contain document"
            )
        normalized_contexts[partition_id] = child_kwargs
    trust_sets = {
        "signals": _trust_roots(
            trusted_signal_input_reconciliation_sha256s,
            "trusted_signal_input_reconciliation_sha256s",
        ),
        "candidate": _trust_roots(
            trusted_candidate_specification_sha256s,
            "trusted_candidate_specification_sha256s",
        ),
        "construction": _trust_roots(
            trusted_construction_code_sha256s,
            "trusted_construction_code_sha256s",
        ),
        "signal_code": _trust_roots(
            trusted_signal_reconciliation_code_sha256s,
            "trusted_signal_reconciliation_code_sha256s",
        ),
    }
    expected_index_trust = {
        "signal_input_reconciliation_set_sha256": _trust_set_hash(
            trust_sets["signals"]
        ),
        "candidate_specification_set_sha256": _trust_set_hash(
            trust_sets["candidate"]
        ),
        "construction_code_set_sha256": _trust_set_hash(
            trust_sets["construction"]
        ),
        "signal_reconciliation_code_set_sha256": _trust_set_hash(
            trust_sets["signal_code"]
        ),
    }
    if normalized["payload"]["trust_policy"] != expected_index_trust:
        raise PeadSignalInputIndexError(
            "signal-input index differs from the supplied external trust registries"
        )
    index_bindings = normalized["payload"]["bindings"]
    _require_trusted(
        index_bindings["candidate_specification_sha256"],
        trust_sets["candidate"],
        "candidate specification",
    )
    _require_trusted(
        index_bindings["construction_code_sha256"],
        trust_sets["construction"],
        "construction code",
    )
    _require_trusted(
        index_bindings["signal_reconciliation_code_sha256"],
        trust_sets["signal_code"],
        "signal reconciliation code",
    )
    if isinstance(signal_input_reconciliations, (str, bytes)) or not isinstance(
        signal_input_reconciliations, Sequence
    ):
        raise PeadSignalInputIndexError("signal_input_reconciliations must be a sequence")
    if len(signal_input_reconciliations) != 10:
        raise PeadSignalInputIndexError(
            "exactly ten annual signal-input reconciliations are required"
        )
    structurally_by_universe: dict[str, dict[str, Any]] = {}
    try:
        for raw in signal_input_reconciliations:
            child = validate_pead_signal_input_reconciliation_structure(raw)
            _require_trusted(
                child["artifact_hash"],
                trust_sets["signals"],
                "signal-input reconciliation",
            )
            universe_hash = child["payload"]["bindings"]["event_universe_sha256"]
            if universe_hash in structurally_by_universe:
                raise PeadSignalInputIndexError(
                    "more than one child binds the same annual event universe"
                )
            structurally_by_universe[universe_hash] = child
    except PeadSignalInputIndexError:
        raise
    except (PeadSignalInputReconciliationError, TypeError, ValueError) as exc:
        raise PeadSignalInputIndexError(
            "an annual signal-input reconciliation is structurally invalid"
        ) from exc

    # The source replay can be expensive.  Run it only after every cheap local
    # shape, context, and child-structure check has passed.
    try:
        event_roots = verify_pead_sharadar_event_universe_replay(
            event_universe_replay,
            event_universe_index,
            **replay_kwargs,
        )
    except (
        PeadSharadarEventUniverseReplayError,
        PeadEventUniverseIndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise PeadSignalInputIndexError(
            "event-universe roots do not replay authoritatively"
        ) from exc
    if not isinstance(event_roots, Mapping) or set(event_roots) != {"replay", "index"}:
        raise PeadSignalInputIndexError("event-universe verifier returned malformed roots")

    replay_years = event_roots["replay"]["payload"]["years"]
    verified_children: list[dict[str, Any]] = []
    for year in replay_years:
        partition_id = year["partition_id"]
        universe_hash = year["event_universe"]["artifact_hash"]
        child = structurally_by_universe.get(universe_hash)
        if child is None:
            raise PeadSignalInputIndexError(
                f"partition {partition_id} has no structurally bound signal receipt"
            )
        child_kwargs = normalized_contexts[partition_id]
        try:
            verified = verify_pead_signal_input_reconciliation(child, **child_kwargs)
        except (PeadSignalInputReconciliationError, TypeError, ValueError) as exc:
            raise PeadSignalInputIndexError(
                f"partition {partition_id} signal receipt does not replay authoritatively"
            ) from exc
        if verified != child:
            raise PeadSignalInputIndexError(
                f"partition {partition_id} verifier returned another signal receipt"
            )
        verified_children.append(verified)

    expected = build_pead_signal_input_index(
        event_roots["replay"],
        event_roots["index"],
        verified_children,
        indexed_at_utc=normalized["payload"]["indexed_at_utc"],
        trusted_signal_input_reconciliation_sha256s=trust_sets["signals"],
        trusted_candidate_specification_sha256s=trust_sets["candidate"],
        trusted_construction_code_sha256s=trust_sets["construction"],
        trusted_signal_reconciliation_code_sha256s=trust_sets["signal_code"],
    )
    if normalized != expected:
        raise PeadSignalInputIndexError(
            "signal-input index does not replay from authoritative annual children"
        )
    return expected


def _strict_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadSignalInputIndexError(
            f"signal-input index is not a regular file: {path}"
        )
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_SIGNAL_INPUT_INDEX_BYTES:
        raise PeadSignalInputIndexError("signal-input index file size is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadSignalInputIndexError("signal-input index is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadSignalInputIndexError(
                    f"signal-input index contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadSignalInputIndexError(
            f"signal-input index contains invalid number {token}"
        )

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadSignalInputIndexError("invalid signal-input index JSON") from exc
    if not isinstance(value, dict):
        raise PeadSignalInputIndexError("signal-input index root must be an object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadSignalInputIndexError(
            "signal-input index bytes are not canonical JSON plus one newline"
        )
    return value


def publish_pead_signal_input_index(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    authoritative_verification_kwargs: Mapping[str, Any] | None = None,
    allow_structural_only: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Create one canonical index without ever replacing existing bytes."""
    if authoritative_verification_kwargs is None:
        if allow_structural_only is not True:
            raise PeadSignalInputIndexError(
                "publication requires authoritative verification or explicit "
                "allow_structural_only=True"
            )
        normalized = validate_pead_signal_input_index_structure(document)
    else:
        if allow_structural_only is not False:
            raise PeadSignalInputIndexError(
                "authoritative and structural-only publication modes are mutually exclusive"
            )
        if not isinstance(authoritative_verification_kwargs, Mapping):
            raise PeadSignalInputIndexError(
                "authoritative_verification_kwargs must be a mapping"
            )
        verification_kwargs = dict(authoritative_verification_kwargs)
        if "document" in verification_kwargs:
            raise PeadSignalInputIndexError(
                "authoritative_verification_kwargs may not contain document"
            )
        normalized = verify_pead_signal_input_index(document, **verification_kwargs)

    encoded = (canonical_json(normalized) + "\n").encode("utf-8")
    if len(encoded) > MAX_SIGNAL_INPUT_INDEX_BYTES:
        raise PeadSignalInputIndexError("signal-input index exceeds its size limit")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise PeadSignalInputIndexError(
            f"signal-input index parent is not a regular directory: {target.parent}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise PeadSignalInputIndexError(
            f"signal-input index destination already exists: {target}"
        ) from exc
    except OSError as exc:
        raise PeadSignalInputIndexError(
            f"cannot create signal-input index: {target}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PeadSignalInputIndexError(
            f"cannot durably write signal-input index: {target}"
        ) from exc
    reread = validate_pead_signal_input_index_structure(_strict_json_file(target))
    if reread != normalized:
        raise PeadSignalInputIndexError(
            "published signal-input index differs from the validated document"
        )
    return normalized, target


def load_pead_signal_input_index(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Load canonical bytes and authoritatively replay all annual inputs."""
    return verify_pead_signal_input_index(_strict_json_file(Path(path)), **kwargs)


__all__ = [
    "INDEX_POLICY",
    "MAX_SIGNAL_INPUT_INDEX_BYTES",
    "PeadSignalInputIndexError",
    "SIGNAL_INPUT_INDEX_POLICY_SCHEMA_VERSION",
    "SIGNAL_INPUT_INDEX_SCHEMA_VERSION",
    "build_pead_signal_input_index",
    "load_pead_signal_input_index",
    "publish_pead_signal_input_index",
    "validate_pead_signal_input_index_structure",
    "verify_pead_signal_input_index",
]
