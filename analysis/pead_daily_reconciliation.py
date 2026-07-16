"""PEAD-specific acceptance contract for independent daily money paths.

The generic reconciliation engine is intentionally strategy-agnostic.  A raw
``ReplicationEvidence`` object therefore cannot establish that PEAD covered
the correct cohorts, dates, identities, implementation pair, or tolerances.
This module supplies that missing strategy-specific layer.  It freezes the
4,114-key development-sample manifest, records disjoint money-calculation code
manifests, nests the fully replayable generic evidence, and retains every
qualification and execution blocker outside this narrow modeled-accounting
claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.independent_replication import (
    IndependentReplicationContract,
    ImplementationIdentity,
    NumericTolerance,
    ReplicationEvidence,
    ReplicationIntegrityError,
)
from analysis.pead_daily_ledger import (
    replication_observations as primary_replication_observations,
    validate_primary_daily_ledger,
)
from analysis.pead_daily_inputs import validate_pead_daily_input_snapshot
from analysis.pead_execution_ledger import validate_pead_execution_ledger
from analysis.pead_reference_replication import verify_reference_artifact
from data.pead_economic_evidence import canonical_json, content_hash


SCHEMA_VERSION = "pead_daily_reconciliation_receipt.v3"
CANDIDATE_ID = "pead-vq-locked-replication-v1"
PRIMARY_IMPLEMENTATION_ID = "pead-primary-daily-ledger-v2"
REFERENCE_IMPLEMENTATION_ID = "pead-independent-event-driven-v2"
PROTOCOL_SCHEMA_VERSION = "pead_daily_money_path_protocol.v2"
EXPECTED_TOTAL_KEYS = 4114
EXPECTED_FORMATION_KEYS = 330
EXPECTED_DAILY_KEYS = 3784
EXPECTED_PATHS = 88
EXPECTED_COHORTS = 16
# Updated with the content address of the frozen v2 protocol below.  The
# protocol itself also pins the exact development-sample inputs and key set.
DAILY_PROTOCOL_HASH = "622e9987d86bec5a0c36ea507ee8b87579f255c985e3f550f24e5aff40cc7f38"

EXPECTED_DEVELOPMENT_SAMPLE_BINDING = {
    "combined_data_snapshot_hash": (
        "0c047827948d9de2ac60ee8d753cca79c0499f6e789d520524feedb2291a15cc"
    ),
    "daily_input_snapshot_hash": (
        "d0b6b6ee3d42696b80528ac58c8c95d3adabfc83f294a1521d654125ba721f52"
    ),
    "economic_return_inputs_hash": (
        "c30e57998d475818c64dcf47bbc4f92e91ef584e65ffbe64d4591451254927ad"
    ),
    "expected_key_manifest_hash": (
        "933431159894ca481cf10d99a83915f2f0d73aab3014293626127bb480e0d67f"
    ),
    "independent_reference_comparison_hash": (
        "926ed1a9a9e91c2e04a04b4ca431d17d4313db04cc313925ed1cbefb1937ec8e"
    ),
    "modeled_execution_ledger_hash": (
        "b38f4f8b654b23e3046c236a046f24ba3cf597c02c143f13435f8a87e4f45cdc"
    ),
    "research_manifest_binding_hash": (
        "8348e851a0d77a1bc9069059acffe4c19f4e579591a1fe0446be9caba1981e40"
    ),
    "source_report_core_hash": (
        "94374850901408dba8ae7c2f428af66effeaf81461194baab67ee1946bc6ce92"
    ),
    "source_report_file_sha256": (
        "e2ae42ee3f6fad49748f23a40869e3b2461113fb31f4cb9016033e8560d678b7"
    ),
    "warehouse_snapshot_hash": (
        "7067cea26762257a8cbf9e8c89ad684aaf2e3bcb3b0964e2fc26d554c6cdd919"
    ),
}

EXPECTED_IMPLEMENTATION_PATHS = {
    "primary": (
        "analysis/pead_daily_ledger.py",
        "analysis/pead_execution_ledger.py",
    ),
    "reference": ("analysis/pead_daily_reference.py",),
    "shared": (
        "analysis/independent_replication.py",
        "analysis/pead_daily_acceptance.py",
        "analysis/pead_daily_inputs.py",
        "analysis/pead_daily_reconciliation.py",
        "analysis/pead_economic_returns.py",
        "analysis/pead_reference_replication.py",
        "analysis/pead_replication.py",
        "data/pead_economic_evidence.py",
    ),
}
SHARED_IMPLEMENTATION_ID = "pead-daily-shared-verification-v2"
COMPONENT_SCHEMA_VERSION = "pead_daily_component_reconciliation.v1"
COMPONENT_MONEY_TOLERANCE = Decimal("0.000000000000000005")
EXPECTED_COMPONENT_COVERAGE = {
    "cohort_daily_states": 688,
    "daily_constituent_states": 3784,
    "distribution_action_applications": 61,
}
COMPONENT_CONSTITUENT_KEY_FIELDS = (
    "cohort_id",
    "formation_date",
    "horizon_sessions",
    "sequence",
    "checkpoint",
    "session_date",
    "m_ticker",
    "permaticker",
    "source_event_key",
)
COMPONENT_COHORT_KEY_FIELDS = (
    "cohort_id",
    "formation_date",
    "horizon_sessions",
    "sequence",
    "checkpoint",
    "session_date",
)

FROZEN_TOLERANCES = {
    "signal": {"absolute": 1e-12, "relative": 1e-12},
    "rank": {"absolute": 0.0, "relative": 0.0},
    "target": {"absolute": 1e-15, "relative": 1e-12},
    "order": {"absolute": 1e-15, "relative": 1e-12},
    "position": {"absolute": 1e-15, "relative": 1e-12},
    "cash": {"absolute": 1e-12, "relative": 1e-12},
    "fees": {"absolute": 1e-15, "relative": 1e-12},
    "pnl": {"absolute": 1e-12, "relative": 1e-12},
}

REPORT_CORE_FIELDS = (
    "candidate_id",
    "combined_data_snapshot",
    "research_manifest_binding",
    "economic_return_inputs",
    "configuration",
    "normalization",
    "coverage",
    "slice_coverage",
    "raw_portfolio_observations",
    "tests",
    "multiple_testing",
)
KEY_FIELDS = {
    "candidate_id",
    "slice",
    "cohort_id",
    "formation_date",
    "horizon_sessions",
    "checkpoint",
    "session_date",
    "ticker",
    "m_ticker",
    "permaticker",
    "source_event_key",
}


class PeadDailyReconciliationError(ValueError):
    """A PEAD daily receipt is malformed, incomplete, or mismatched."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _verified_wrapper(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"artifact_hash", "payload"}:
        raise PeadDailyReconciliationError(
            f"{label} must be a content-addressed wrapper"
        )
    claimed = value["artifact_hash"]
    payload = value["payload"]
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or not isinstance(payload, Mapping)
        or content_hash(payload) != claimed
    ):
        raise PeadDailyReconciliationError(f"{label} content identity is invalid")
    return payload


def pead_reconciliation_input(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable PEAD report core, excluding receipt-derived status."""
    if not isinstance(report, Mapping):
        raise PeadDailyReconciliationError("PEAD report must be an object")
    missing = [field for field in REPORT_CORE_FIELDS if field not in report]
    if missing:
        raise PeadDailyReconciliationError(
            f"PEAD report omits reconciliation-core fields: {missing}"
        )
    if report.get("candidate_id") != CANDIDATE_ID:
        raise PeadDailyReconciliationError("PEAD report belongs to another candidate")
    payload = {
        "schema_version": "pead_reconciliation_input.v1",
        **{field: _plain(report[field]) for field in REPORT_CORE_FIELDS},
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _validated_protocol(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _verified_wrapper(protocol, "daily protocol")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise PeadDailyReconciliationError("unsupported daily protocol")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadDailyReconciliationError("daily protocol belongs to another candidate")
    if protocol.get("artifact_hash") != DAILY_PROTOCOL_HASH:
        raise PeadDailyReconciliationError("daily protocol hash is not frozen")
    if payload.get("development_sample_input_binding") != (
        EXPECTED_DEVELOPMENT_SAMPLE_BINDING
    ):
        raise PeadDailyReconciliationError(
            "daily protocol development-sample binding changed"
        )
    expected = payload.get("development_sample_expected_coverage")
    if not isinstance(expected, Mapping) or expected != {
        "admitted_pooled_cohorts": 16,
        "daily_selected_constituent_checkpoints": 3784,
        "exhaustive_formation_checkpoints": 330,
        "generic_observation_keys": 4114,
        "selected_constituent_paths": 88,
    }:
        raise PeadDailyReconciliationError("daily protocol coverage changed")
    return payload


def _code_manifest(
    implementation_id: str, paths: Sequence[Path], *, root: Path
) -> dict[str, Any]:
    files: list[dict[str, str]] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise PeadDailyReconciliationError(
                f"implementation file is not regular: {resolved}"
            )
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise PeadDailyReconciliationError(
                "implementation file is outside repository root"
            ) from exc
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            }
        )
    files.sort(key=lambda item: item["path"])
    if not files or len({item["path"] for item in files}) != len(files):
        raise PeadDailyReconciliationError("implementation manifest is empty or duplicate")
    return {
        "implementation_id": implementation_id,
        "code_hash": content_hash(files),
        "files": files,
    }


def current_implementation_manifests(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the current disjoint primary/reference money-calculation files."""
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    primary = _code_manifest(
        PRIMARY_IMPLEMENTATION_ID,
        [root / path for path in EXPECTED_IMPLEMENTATION_PATHS["primary"]],
        root=root,
    )
    reference = _code_manifest(
        REFERENCE_IMPLEMENTATION_ID,
        [root / path for path in EXPECTED_IMPLEMENTATION_PATHS["reference"]],
        root=root,
    )
    overlap = {item["path"] for item in primary["files"]} & {
        item["path"] for item in reference["files"]
    }
    if overlap:
        raise PeadDailyReconciliationError(
            f"money-calculation manifests overlap: {sorted(overlap)}"
        )
    shared = _code_manifest(
        SHARED_IMPLEMENTATION_ID,
        [root / path for path in EXPECTED_IMPLEMENTATION_PATHS["shared"]],
        root=root,
    )
    return {"primary": primary, "reference": reference, "shared": shared}


def _canonical_keys(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tokens: dict[str, dict[str, Any]] = {}
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise PeadDailyReconciliationError("replication observation is malformed")
        key = observation.get("key")
        if not isinstance(key, Mapping) or set(key) != KEY_FIELDS:
            raise PeadDailyReconciliationError("daily observation key fields changed")
        normalized = _plain(key)
        token = canonical_json(normalized)
        if token in tokens:
            raise PeadDailyReconciliationError("daily observations contain duplicate keys")
        tokens[token] = normalized
    return [tokens[token] for token in sorted(tokens)]


def _validate_key_manifest(keys: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if len(keys) != EXPECTED_TOTAL_KEYS:
        raise PeadDailyReconciliationError(
            f"expected {EXPECTED_TOTAL_KEYS} daily keys; found {len(keys)}"
        )
    formation = [key for key in keys if key["checkpoint"] == "formation_target"]
    daily = [key for key in keys if key["checkpoint"] != "formation_target"]
    if len(formation) != EXPECTED_FORMATION_KEYS or len(daily) != EXPECTED_DAILY_KEYS:
        raise PeadDailyReconciliationError("formation/daily key coverage changed")
    for key in keys:
        if key["candidate_id"] != CANDIDATE_ID or key["slice"] != "pooled":
            raise PeadDailyReconciliationError("key belongs to another candidate or slice")
        if key["horizon_sessions"] not in {21, 63}:
            raise PeadDailyReconciliationError("key horizon is not frozen")
        try:
            formation_date = date.fromisoformat(key["formation_date"])
            session_date = date.fromisoformat(key["session_date"])
        except (TypeError, ValueError) as exc:
            raise PeadDailyReconciliationError("key date is not canonical ISO") from exc
        if (
            not isinstance(key["permaticker"], int)
            or isinstance(key["permaticker"], bool)
            or key["permaticker"] <= 0
            or not isinstance(key["source_event_key"], Mapping)
            or not key["source_event_key"]
        ):
            raise PeadDailyReconciliationError("key identity evidence is incomplete")
        if key["checkpoint"] == "formation_target":
            if session_date != formation_date:
                raise PeadDailyReconciliationError("formation key uses another session")
        elif key["checkpoint"] not in {"entry_close", "mark_close", "exit_close"}:
            raise PeadDailyReconciliationError("daily checkpoint label is invalid")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for key in daily:
        grouped.setdefault((key["cohort_id"], key["m_ticker"]), []).append(key)
    if len(grouped) != EXPECTED_PATHS:
        raise PeadDailyReconciliationError("selected-path key count changed")
    for path in grouped.values():
        ordered = sorted(path, key=lambda key: key["session_date"])
        horizon = ordered[0]["horizon_sessions"]
        if len(ordered) != horizon + 1:
            raise PeadDailyReconciliationError("daily path is not horizon-inclusive")
        if ordered[0]["checkpoint"] != "entry_close":
            raise PeadDailyReconciliationError("daily path does not begin at entry")
        if ordered[-1]["checkpoint"] != "exit_close":
            raise PeadDailyReconciliationError("daily path does not end at exit")
        if any(key["checkpoint"] != "mark_close" for key in ordered[1:-1]):
            raise PeadDailyReconciliationError("interior daily path label changed")
    cohorts = {key["cohort_id"] for key in daily}
    if len(cohorts) != EXPECTED_COHORTS:
        raise PeadDailyReconciliationError("admitted cohort key count changed")
    return {
        "total": len(keys),
        "formation": len(formation),
        "daily": len(daily),
        "selected_paths": len(grouped),
        "admitted_cohorts": len(cohorts),
    }


def _tolerances(protocol_payload: Mapping[str, Any]) -> dict[str, NumericTolerance]:
    projection = protocol_payload.get("generic_projection")
    raw = projection.get("numeric_tolerances") if isinstance(projection, Mapping) else None
    if not isinstance(raw, Mapping):
        raise PeadDailyReconciliationError("daily protocol omits tolerances")
    try:
        result = {
            field: NumericTolerance(
                absolute=values["absolute"], relative=values["relative"]
            )
            for field, values in raw.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PeadDailyReconciliationError("daily protocol tolerances are invalid") from exc
    if set(result) != {
        "signal", "rank", "target", "order", "position", "cash", "fees", "pnl"
    }:
        raise PeadDailyReconciliationError("daily tolerances do not cover generic fields")
    return result


def _reference_observations(
    document: Mapping[str, Any],
    *,
    reference_artifact: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        from analysis.pead_daily_reference import replication_observations
    except ImportError as exc:  # pragma: no cover - installation defect
        raise PeadDailyReconciliationError("independent daily implementation is unavailable") from exc
    try:
        return replication_observations(
            document,
            reference_artifact=reference_artifact,
            daily_inputs=daily_inputs,
        )
    except (TypeError, ValueError) as exc:
        raise PeadDailyReconciliationError("independent daily projection is invalid") from exc


def _component_decimal(value: Any, label: str) -> str:
    if isinstance(value, bool) or value is None:
        raise PeadDailyReconciliationError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PeadDailyReconciliationError(
            f"{label} must be a finite decimal"
        ) from exc
    if not result.is_finite():
        raise PeadDailyReconciliationError(f"{label} must be a finite decimal")
    return format(result, "f")


def _component_key(
    row: Mapping[str, Any], fields: Sequence[str], label: str
) -> dict[str, Any]:
    try:
        key = {field: _plain(row[field]) for field in fields}
    except KeyError as exc:
        raise PeadDailyReconciliationError(f"{label} omits a key field") from exc
    return key


def _component_action_keys(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PeadDailyReconciliationError(f"{label} must be an array")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or not item:
            raise PeadDailyReconciliationError(f"{label} contains a malformed key")
        normalized.append(_plain(item))
    normalized.sort(key=canonical_json)
    tokens = [canonical_json(item) for item in normalized]
    if len(tokens) != len(set(tokens)):
        raise PeadDailyReconciliationError(f"{label} repeats an action key")
    return normalized


def _component_constituents(
    payload: Mapping[str, Any], *, primary: bool
) -> list[dict[str, Any]]:
    rows = payload.get("daily_constituent_states")
    if not isinstance(rows, list):
        raise PeadDailyReconciliationError("daily ledger omits constituent states")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PeadDailyReconciliationError("daily constituent state is malformed")
        prefix = f"daily constituent state {index}"
        field = (
            (lambda primary_name, reference_name: primary_name)
            if primary
            else (lambda primary_name, reference_name: reference_name)
        )
        result.append(
            {
                "key": _component_key(
                    raw, COMPONENT_CONSTITUENT_KEY_FIELDS, prefix
                ),
                "checkpoint": raw.get("checkpoint"),
                "leg": raw.get("leg"),
                "rank": raw.get("rank"),
                "sequence": raw.get("sequence"),
                "price_split_normalized": _component_decimal(
                    raw.get(field("close_split_normalized", "price_split_normalized")),
                    f"{prefix} price",
                ),
                "target": _component_decimal(raw.get("target"), f"{prefix} target"),
                "order": _component_decimal(raw.get("order"), f"{prefix} order"),
                "position": _component_decimal(
                    raw.get("position"), f"{prefix} position"
                ),
                "order_cash_flow": _component_decimal(
                    raw.get("order_cash_flow"), f"{prefix} order cash flow"
                ),
                "distribution_accrual_today": _component_decimal(
                    raw.get(
                        field(
                            "candidate_distribution_accrual",
                            "distribution_accrual_today",
                        )
                    ),
                    f"{prefix} distribution accrual",
                ),
                "signed_distribution_balance": _component_decimal(
                    raw.get(
                        field(
                            "candidate_distribution_receivable",
                            "distribution_receivable",
                        )
                    ),
                    f"{prefix} signed distribution balance",
                ),
                "market_value": _component_decimal(
                    raw.get("market_value"), f"{prefix} market value"
                ),
                "fee_today": _component_decimal(
                    raw.get(field("modeled_fee", "fee_today")),
                    f"{prefix} fee",
                ),
                "price_pnl": _component_decimal(
                    raw.get("price_pnl"), f"{prefix} price P&L"
                ),
                "net_pnl_contribution": _component_decimal(
                    raw.get("net_pnl_contribution"),
                    f"{prefix} net P&L contribution",
                ),
                "applied_distribution_action_keys": _component_action_keys(
                    raw.get("applied_distribution_action_keys"),
                    f"{prefix} distribution action keys",
                ),
            }
        )
    return sorted(result, key=lambda row: canonical_json(row["key"]))


def _component_cohorts(
    payload: Mapping[str, Any],
    constituents: Sequence[Mapping[str, Any]],
    *,
    primary: bool,
) -> list[dict[str, Any]]:
    rows = payload.get("cohort_daily_states")
    if not isinstance(rows, list):
        raise PeadDailyReconciliationError("daily ledger omits cohort states")
    reference_checkpoint_fees: dict[str, Decimal] = {}
    if not primary:
        for row in constituents:
            cohort_key = {
                field: row["key"][field] for field in COMPONENT_COHORT_KEY_FIELDS
            }
            token = canonical_json(cohort_key)
            reference_checkpoint_fees[token] = (
                reference_checkpoint_fees.get(token, Decimal("0"))
                + Decimal(row["fee_today"])
            )
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise PeadDailyReconciliationError("daily cohort state is malformed")
        prefix = f"daily cohort state {index}"
        key = _component_key(raw, COMPONENT_COHORT_KEY_FIELDS, prefix)
        checkpoint_fees = (
            raw.get("checkpoint_fees")
            if primary
            else reference_checkpoint_fees.get(canonical_json(key))
        )
        balance_field = (
            "candidate_distribution_receivable"
            if primary
            else "distribution_receivable"
        )
        pnl_field = "cumulative_pnl" if primary else "pnl"
        result.append(
            {
                "key": key,
                "checkpoint": raw.get("checkpoint"),
                "sequence": raw.get("sequence"),
                "open_position_count": raw.get("open_position_count"),
                "settled_cash": _component_decimal(
                    raw.get("settled_cash"), f"{prefix} settled cash"
                ),
                "signed_distribution_balance": _component_decimal(
                    raw.get(balance_field), f"{prefix} signed distribution balance"
                ),
                "market_value": _component_decimal(
                    raw.get("market_value"), f"{prefix} market value"
                ),
                "gross_long_market_value": _component_decimal(
                    raw.get("gross_long_market_value"),
                    f"{prefix} gross long market value",
                ),
                "gross_short_market_value": _component_decimal(
                    raw.get("gross_short_market_value"),
                    f"{prefix} gross short market value",
                ),
                "nav": _component_decimal(raw.get("nav"), f"{prefix} NAV"),
                "checkpoint_fees": _component_decimal(
                    checkpoint_fees, f"{prefix} checkpoint fees"
                ),
                "cumulative_fees": _component_decimal(
                    raw.get("cumulative_fees"), f"{prefix} cumulative fees"
                ),
                "daily_pnl": _component_decimal(
                    raw.get("daily_pnl"), f"{prefix} daily P&L"
                ),
                "cumulative_pnl": _component_decimal(
                    raw.get(pnl_field), f"{prefix} cumulative P&L"
                ),
            }
        )
    return sorted(result, key=lambda row: canonical_json(row["key"]))


def _compare_component_level(
    level: str,
    primary_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    exact_fields: Sequence[str],
    exact_numeric_fields: Sequence[str],
    money_fields: Sequence[str],
    discrepancies: list[dict[str, Any]],
    maxima: dict[str, Decimal],
) -> None:
    def indexed(
        rows: Sequence[Mapping[str, Any]], side: str
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            token = canonical_json(row["key"])
            if token in result:
                raise PeadDailyReconciliationError(
                    f"{side} {level} component projection repeats a key"
                )
            result[token] = row
        return result

    primary = indexed(primary_rows, "primary")
    reference = indexed(reference_rows, "reference")
    for token in sorted(set(primary) | set(reference)):
        left = primary.get(token)
        right = reference.get(token)
        key = _plain((left or right)["key"])  # type: ignore[index]
        if left is None or right is None:
            discrepancies.append(
                {
                    "level": level,
                    "key": key,
                    "field": "__row__",
                    "primary": "missing" if left is None else "present",
                    "reference": "missing" if right is None else "present",
                }
            )
            continue
        for field in exact_fields:
            if canonical_json(left[field]) != canonical_json(right[field]):
                discrepancies.append(
                    {
                        "level": level,
                        "key": key,
                        "field": field,
                        "primary": _plain(left[field]),
                        "reference": _plain(right[field]),
                        "tolerance": "exact",
                    }
                )
        for field in (*exact_numeric_fields, *money_fields):
            difference = abs(Decimal(left[field]) - Decimal(right[field]))
            maximum_key = f"{level}.{field}"
            maxima[maximum_key] = max(maxima.get(maximum_key, Decimal("0")), difference)
            tolerance = (
                Decimal("0")
                if field in exact_numeric_fields
                else COMPONENT_MONEY_TOLERANCE
            )
            if difference > tolerance:
                discrepancies.append(
                    {
                        "level": level,
                        "key": key,
                        "field": field,
                        "primary": left[field],
                        "reference": right[field],
                        "absolute_difference": format(difference, "f"),
                        "tolerance": format(tolerance, "f"),
                    }
                )


def _component_reconciliation(
    primary_daily_ledger: Mapping[str, Any],
    independent_daily_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    primary_payload = _verified_wrapper(primary_daily_ledger, "primary daily ledger")
    reference_payload = _verified_wrapper(
        independent_daily_ledger, "independent daily ledger"
    )
    primary_constituents = _component_constituents(primary_payload, primary=True)
    reference_constituents = _component_constituents(
        reference_payload, primary=False
    )
    primary_cohorts = _component_cohorts(
        primary_payload, primary_constituents, primary=True
    )
    reference_cohorts = _component_cohorts(
        reference_payload, reference_constituents, primary=False
    )
    discrepancies: list[dict[str, Any]] = []
    maxima: dict[str, Decimal] = {}
    _compare_component_level(
        "constituent",
        primary_constituents,
        reference_constituents,
        exact_fields=(
            "checkpoint",
            "leg",
            "rank",
            "sequence",
            "applied_distribution_action_keys",
        ),
        exact_numeric_fields=(
            "price_split_normalized",
            "target",
            "order",
            "position",
        ),
        money_fields=(
            "order_cash_flow",
            "distribution_accrual_today",
            "signed_distribution_balance",
            "market_value",
            "fee_today",
            "price_pnl",
            "net_pnl_contribution",
        ),
        discrepancies=discrepancies,
        maxima=maxima,
    )
    _compare_component_level(
        "cohort",
        primary_cohorts,
        reference_cohorts,
        exact_fields=("checkpoint", "sequence", "open_position_count"),
        exact_numeric_fields=(),
        money_fields=(
            "settled_cash",
            "signed_distribution_balance",
            "market_value",
            "gross_long_market_value",
            "gross_short_market_value",
            "nav",
            "checkpoint_fees",
            "cumulative_fees",
            "daily_pnl",
            "cumulative_pnl",
        ),
        discrepancies=discrepancies,
        maxima=maxima,
    )
    primary_actions = sum(
        len(row["applied_distribution_action_keys"])
        for row in primary_constituents
    )
    reference_actions = sum(
        len(row["applied_distribution_action_keys"])
        for row in reference_constituents
    )
    expected = EXPECTED_COMPONENT_COVERAGE
    primary_coverage = {
        "daily_constituent_states": len(primary_constituents),
        "cohort_daily_states": len(primary_cohorts),
        "distribution_action_applications": primary_actions,
    }
    reference_coverage = {
        "daily_constituent_states": len(reference_constituents),
        "cohort_daily_states": len(reference_cohorts),
        "distribution_action_applications": reference_actions,
    }
    if primary_coverage != expected or reference_coverage != expected:
        raise PeadDailyReconciliationError(
            "component reconciliation coverage differs from the frozen protocol"
        )
    primary_projection = {
        "constituents": primary_constituents,
        "cohorts": primary_cohorts,
    }
    reference_projection = {
        "constituents": reference_constituents,
        "cohorts": reference_cohorts,
    }
    return {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "passed": not discrepancies,
        "tolerances": {
            "exact_numeric_absolute": "0",
            "money_absolute": format(COMPONENT_MONEY_TOLERANCE, "f"),
        },
        "coverage": {
            "expected": expected,
            "primary": primary_coverage,
            "reference": reference_coverage,
        },
        "primary_projection_hash": content_hash(primary_projection),
        "reference_projection_hash": content_hash(reference_projection),
        "max_absolute_differences": {
            field: format(value, "f") for field, value in sorted(maxima.items())
        },
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
    }


def _generic_document(evidence: ReplicationEvidence) -> dict[str, Any]:
    return json.loads(evidence.to_json())


def _generic_evidence(document: Any) -> ReplicationEvidence:
    try:
        return ReplicationEvidence.from_json(canonical_json(document) + "\n")
    except (ReplicationIntegrityError, TypeError, ValueError) as exc:
        raise PeadDailyReconciliationError("nested generic evidence is invalid") from exc


def _source_hashes(
    report: Mapping[str, Any],
    ledger: Mapping[str, Any],
    reference: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    report_core = pead_reconciliation_input(report)
    combined = report.get("combined_data_snapshot")
    economic = report.get("economic_return_inputs")
    manifest = report.get("research_manifest_binding")
    for value, label in (
        (combined, "combined data snapshot"),
        (economic, "economic return inputs"),
        (manifest, "research manifest binding"),
    ):
        _verified_wrapper(value, label)
    daily_input_bindings = daily_inputs.get("payload", {}).get("bindings", {})
    if not isinstance(daily_input_bindings, Mapping):
        raise PeadDailyReconciliationError("daily input bindings are malformed")
    return {
        "source_report_file_sha256": hashlib.sha256(
            (canonical_json(report) + "\n").encode("utf-8")
        ).hexdigest(),
        "source_report_core_hash": report_core["artifact_hash"],
        "combined_data_snapshot_hash": combined["artifact_hash"],
        "economic_return_inputs_hash": economic["artifact_hash"],
        "research_manifest_binding_hash": manifest["artifact_hash"],
        "modeled_execution_ledger_hash": ledger["artifact_hash"],
        "independent_reference_comparison_hash": reference["artifact_hash"],
        "daily_input_snapshot_hash": daily_inputs["artifact_hash"],
        "daily_protocol_hash": protocol["artifact_hash"],
        "warehouse_snapshot_hash": daily_input_bindings.get(
            "warehouse_snapshot_version"
        ),
    }


def _validate_development_sample_sources(bindings: Mapping[str, Any]) -> None:
    actual = {
        "combined_data_snapshot_hash": bindings.get(
            "combined_data_snapshot_hash"
        ),
        "daily_input_snapshot_hash": bindings.get("daily_input_snapshot_hash"),
        "economic_return_inputs_hash": bindings.get(
            "economic_return_inputs_hash"
        ),
        "independent_reference_comparison_hash": bindings.get(
            "independent_reference_comparison_hash"
        ),
        "modeled_execution_ledger_hash": bindings.get(
            "modeled_execution_ledger_hash"
        ),
        "research_manifest_binding_hash": bindings.get(
            "research_manifest_binding_hash"
        ),
        "source_report_core_hash": bindings.get("source_report_core_hash"),
        "source_report_file_sha256": bindings.get("source_report_file_sha256"),
        "warehouse_snapshot_hash": bindings.get("warehouse_snapshot_hash"),
    }
    expected = {
        key: value
        for key, value in EXPECTED_DEVELOPMENT_SAMPLE_BINDING.items()
        if key != "expected_key_manifest_hash"
    }
    if actual != expected:
        raise PeadDailyReconciliationError(
            "daily reconciliation sources differ from the frozen development sample"
        )


def build_pead_daily_reconciliation_receipt(
    source_report: Mapping[str, Any],
    modeled_ledger: Mapping[str, Any],
    independent_reference: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
    protocol: Mapping[str, Any],
    primary_daily_ledger: Mapping[str, Any],
    independent_daily_ledger: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a PEAD-specific receipt around exhaustive generic evidence."""
    validate_pead_execution_ledger(modeled_ledger, source_report=source_report)
    verify_reference_artifact(independent_reference)
    validate_pead_daily_input_snapshot(
        daily_inputs,
        report=source_report,
        ledger=modeled_ledger,
        reference=independent_reference,
    )
    protocol_payload = _validated_protocol(protocol)
    bindings = _source_hashes(
        source_report, modeled_ledger, independent_reference, daily_inputs, protocol
    )
    _validate_development_sample_sources(bindings)
    validate_primary_daily_ledger(
        primary_daily_ledger,
        source_report=source_report,
        modeled_ledger=modeled_ledger,
        daily_inputs=daily_inputs,
        protocol=protocol,
    )
    try:
        from analysis.pead_daily_reference import validate_independent_daily_ledger

        validate_independent_daily_ledger(
            independent_daily_ledger,
            reference_artifact=independent_reference,
            daily_inputs=daily_inputs,
        )
    except ImportError as exc:  # pragma: no cover - installation defect
        raise PeadDailyReconciliationError("independent daily validator is unavailable") from exc
    implementations = current_implementation_manifests(repository_root)
    primary_output = primary_replication_observations(primary_daily_ledger)
    reference_output = _reference_observations(
        independent_daily_ledger,
        reference_artifact=independent_reference,
        daily_inputs=daily_inputs,
    )
    primary_keys = _canonical_keys(primary_output)
    reference_keys = _canonical_keys(reference_output)
    if canonical_json(primary_keys) != canonical_json(reference_keys):
        raise PeadDailyReconciliationError(
            "primary and reference expected-key derivations differ"
        )
    coverage = _validate_key_manifest(primary_keys)
    if content_hash(primary_keys) != EXPECTED_DEVELOPMENT_SAMPLE_BINDING[
        "expected_key_manifest_hash"
    ]:
        raise PeadDailyReconciliationError(
            "daily reconciliation key manifest differs from the frozen protocol"
        )
    contract = IndependentReplicationContract(
        protocol_hash=protocol["artifact_hash"],
        data_snapshot_hash=daily_inputs["artifact_hash"],
        primary=ImplementationIdentity(
            PRIMARY_IMPLEMENTATION_ID, implementations["primary"]["code_hash"]
        ),
        replication=ImplementationIdentity(
            REFERENCE_IMPLEMENTATION_ID, implementations["reference"]["code_hash"]
        ),
        expected_observation_keys=primary_keys,
        tolerances=_tolerances(protocol_payload),
    )
    evidence = ReplicationEvidence.create(
        contract,
        primary_observations=primary_output,
        replication_observations=reference_output,
    )
    component_reconciliation = _component_reconciliation(
        primary_daily_ledger, independent_daily_ledger
    )
    bounded_passed = evidence.passed and component_reconciliation["passed"]
    bindings.update(
        {
            "primary_daily_ledger_hash": primary_daily_ledger["artifact_hash"],
            "independent_daily_ledger_hash": independent_daily_ledger["artifact_hash"],
            "generic_replication_evidence_hash": evidence.evidence_hash,
            "component_reconciliation_hash": content_hash(
                component_reconciliation
            ),
        }
    )
    source_blockers: set[str] = set()
    for payload in (
        source_report,
        modeled_ledger.get("payload", {}),
        independent_reference.get("payload", {}),
        primary_daily_ledger.get("payload", {}),
        independent_daily_ledger.get("payload", {}),
    ):
        blockers = payload.get("blockers") if isinstance(payload, Mapping) else None
        if isinstance(blockers, list):
            source_blockers.update(str(item) for item in blockers)
    for cleared in (
        "independent_implementation_reconciliation_missing",
        "daily_mark_to_market_path_not_implemented",
        "independent_event_driven_money_path_reconciliation_missing",
        "event_driven_money_path_not_implemented",
        "generic_replication_evidence_not_available",
    ):
        source_blockers.discard(cleared)
    source_blockers.update(
        {
            "authoritative_distribution_semantics_and_payment_dates_missing",
            "borrow_financing_capacity_evidence_missing",
            "full_historical_and_prospective_evidence_missing",
            "observed_broker_execution_evidence_missing",
            "pooled_daily_scope_not_full_eight_cell_family",
            "split_normalized_accounting_quantities_are_not_broker_orders",
        }
    )
    if not evidence.passed:
        source_blockers.add("independent_daily_money_path_reconciliation_failed")
    if not component_reconciliation["passed"]:
        source_blockers.add("independent_daily_component_reconciliation_failed")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "development_modeled_daily_money_path_reconciliation",
        "bounded_modeled_daily_money_path_reconciliation_passed": bounded_passed,
        "qualifying_evidence": False,
        "candidate_replication_evidence_eligible": False,
        "paper_execution_evidence": False,
        "promotion_allowed": False,
        "bindings": bindings,
        "implementation_manifests": implementations,
        "key_manifest": {
            "artifact_hash": content_hash(primary_keys),
            "keys": primary_keys,
            "coverage": coverage,
        },
        "generic_replication_evidence": _generic_document(evidence),
        "component_reconciliation": component_reconciliation,
        "comparison": {
            "passed": evidence.passed,
            "discrepancy_count": len(evidence.discrepancies),
            "primary_observations": len(primary_output),
            "reference_observations": len(reference_output),
            "expected_observations": len(primary_keys),
        },
        "claim_boundary": protocol_payload["claim_boundary"],
        "blockers": sorted(source_blockers),
    }
    return {"artifact_hash": content_hash(payload), "payload": _plain(payload)}


def _validate_receipt_static(
    document: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ReplicationEvidence]:
    payload = _verified_wrapper(document, "PEAD daily reconciliation receipt")
    expected_payload_fields = {
        "schema_version",
        "candidate_id",
        "evidence_class",
        "bounded_modeled_daily_money_path_reconciliation_passed",
        "qualifying_evidence",
        "candidate_replication_evidence_eligible",
        "paper_execution_evidence",
        "promotion_allowed",
        "bindings",
        "implementation_manifests",
        "key_manifest",
        "generic_replication_evidence",
        "component_reconciliation",
        "comparison",
        "claim_boundary",
        "blockers",
    }
    if set(payload) != expected_payload_fields:
        raise PeadDailyReconciliationError("daily receipt fields changed")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PeadDailyReconciliationError("unsupported PEAD daily receipt schema")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadDailyReconciliationError("PEAD daily receipt belongs to another candidate")
    evidence = _generic_evidence(payload.get("generic_replication_evidence"))
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping) or comparison != {
        "passed": evidence.passed,
        "discrepancy_count": len(evidence.discrepancies),
        "primary_observations": EXPECTED_TOTAL_KEYS,
        "reference_observations": EXPECTED_TOTAL_KEYS,
        "expected_observations": EXPECTED_TOTAL_KEYS,
    }:
        raise PeadDailyReconciliationError("receipt comparison summary is inconsistent")
    component = payload.get("component_reconciliation")
    expected_component_fields = {
        "schema_version",
        "passed",
        "tolerances",
        "coverage",
        "primary_projection_hash",
        "reference_projection_hash",
        "max_absolute_differences",
        "discrepancy_count",
        "discrepancies",
    }
    if not isinstance(component, Mapping) or set(component) != expected_component_fields:
        raise PeadDailyReconciliationError(
            "receipt component reconciliation is malformed"
        )
    discrepancies = component.get("discrepancies")
    if (
        component.get("schema_version") != COMPONENT_SCHEMA_VERSION
        or component.get("tolerances")
        != {
            "exact_numeric_absolute": "0",
            "money_absolute": format(COMPONENT_MONEY_TOLERANCE, "f"),
        }
        or component.get("coverage")
        != {
            "expected": EXPECTED_COMPONENT_COVERAGE,
            "primary": EXPECTED_COMPONENT_COVERAGE,
            "reference": EXPECTED_COMPONENT_COVERAGE,
        }
        or not isinstance(discrepancies, list)
        or component.get("discrepancy_count") != len(discrepancies)
        or component.get("passed") is not (len(discrepancies) == 0)
    ):
        raise PeadDailyReconciliationError(
            "receipt component reconciliation is inconsistent"
        )
    for field in ("primary_projection_hash", "reference_projection_hash"):
        value = component.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise PeadDailyReconciliationError(
                "receipt component projection identity is invalid"
            )
    maxima = component.get("max_absolute_differences")
    if not isinstance(maxima, Mapping):
        raise PeadDailyReconciliationError(
            "receipt component maximum differences are malformed"
        )
    for field, value in maxima.items():
        if not isinstance(field, str):
            raise PeadDailyReconciliationError(
                "receipt component maximum-difference field is invalid"
            )
        _component_decimal(value, f"component maximum difference {field}")
    bounded_passed = evidence.passed and bool(component["passed"])
    if payload.get("bounded_modeled_daily_money_path_reconciliation_passed") is not (
        bounded_passed
    ):
        raise PeadDailyReconciliationError(
            "receipt pass claim differs from exhaustive reconciliations"
        )
    for field in (
        "qualifying_evidence",
        "candidate_replication_evidence_eligible",
        "paper_execution_evidence",
        "promotion_allowed",
    ):
        if payload.get(field) is not False:
            raise PeadDailyReconciliationError(f"receipt overstates {field}")
    key_manifest = payload.get("key_manifest")
    if not isinstance(key_manifest, Mapping) or set(key_manifest) != {
        "artifact_hash", "keys", "coverage"
    }:
        raise PeadDailyReconciliationError("receipt key manifest is malformed")
    keys = key_manifest["keys"]
    if not isinstance(keys, list) or content_hash(keys) != key_manifest["artifact_hash"]:
        raise PeadDailyReconciliationError("receipt key manifest hash mismatch")
    coverage = _validate_key_manifest(keys)
    if key_manifest["coverage"] != coverage:
        raise PeadDailyReconciliationError("receipt key coverage is inconsistent")
    if key_manifest["artifact_hash"] != EXPECTED_DEVELOPMENT_SAMPLE_BINDING[
        "expected_key_manifest_hash"
    ]:
        raise PeadDailyReconciliationError(
            "receipt key manifest differs from the frozen protocol"
        )
    contract = evidence.payload["contract"]
    if canonical_json(contract["expected_observation_keys"]) != canonical_json(keys):
        raise PeadDailyReconciliationError("generic evidence uses another key manifest")
    if contract.get("protocol_hash") != DAILY_PROTOCOL_HASH:
        raise PeadDailyReconciliationError("generic evidence uses another daily protocol")
    if canonical_json(contract.get("tolerances")) != canonical_json(FROZEN_TOLERANCES):
        raise PeadDailyReconciliationError("generic evidence tolerances are not frozen")
    bindings = payload.get("bindings")
    implementations = payload.get("implementation_manifests")
    if not isinstance(bindings, Mapping) or not isinstance(implementations, Mapping):
        raise PeadDailyReconciliationError("receipt omits bindings or code manifests")
    expected_binding_fields = {
        "source_report_file_sha256",
        "source_report_core_hash",
        "combined_data_snapshot_hash",
        "economic_return_inputs_hash",
        "research_manifest_binding_hash",
        "modeled_execution_ledger_hash",
        "independent_reference_comparison_hash",
        "daily_input_snapshot_hash",
        "daily_protocol_hash",
        "warehouse_snapshot_hash",
        "primary_daily_ledger_hash",
        "independent_daily_ledger_hash",
        "generic_replication_evidence_hash",
        "component_reconciliation_hash",
    }
    if set(bindings) != expected_binding_fields:
        raise PeadDailyReconciliationError("receipt source-binding fields changed")
    _validate_development_sample_sources(bindings)
    if set(implementations) != {"primary", "reference", "shared"}:
        raise PeadDailyReconciliationError("receipt code-manifest sections changed")
    expected_ids = {
        "primary": PRIMARY_IMPLEMENTATION_ID,
        "reference": REFERENCE_IMPLEMENTATION_ID,
        "shared": SHARED_IMPLEMENTATION_ID,
    }
    for side, expected_id in expected_ids.items():
        manifest = implementations.get(side)
        if (
            not isinstance(manifest, Mapping)
            or set(manifest) != {"implementation_id", "code_hash", "files"}
            or manifest.get("implementation_id") != expected_id
        ):
            raise PeadDailyReconciliationError("receipt implementation identity changed")
        files = manifest.get("files")
        if (
            not isinstance(files, list)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"path", "sha256"}
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64
                for item in files
            )
            or content_hash(files) != manifest.get("code_hash")
        ):
            raise PeadDailyReconciliationError("receipt code manifest is inconsistent")
    for side, expected_paths in EXPECTED_IMPLEMENTATION_PATHS.items():
        paths = [item.get("path") for item in implementations[side]["files"]]
        if paths != list(expected_paths):
            raise PeadDailyReconciliationError(
                f"receipt {side} implementation manifest changed"
            )
    if contract["primary"] != {
        "implementation_id": PRIMARY_IMPLEMENTATION_ID,
        "code_hash": implementations["primary"]["code_hash"],
    } or contract["replication"] != {
        "implementation_id": REFERENCE_IMPLEMENTATION_ID,
        "code_hash": implementations["reference"]["code_hash"],
    }:
        raise PeadDailyReconciliationError("generic evidence implementation pair changed")
    if bindings.get("generic_replication_evidence_hash") != evidence.evidence_hash:
        raise PeadDailyReconciliationError("receipt binds another generic evidence hash")
    if bindings.get("component_reconciliation_hash") != content_hash(component):
        raise PeadDailyReconciliationError(
            "receipt binds another component reconciliation hash"
        )
    if evidence.payload["protocol_hash"] != bindings.get("daily_protocol_hash"):
        raise PeadDailyReconciliationError("receipt protocol binding is inconsistent")
    if evidence.payload["data_snapshot_hash"] != bindings.get(
        "daily_input_snapshot_hash"
    ):
        raise PeadDailyReconciliationError("receipt data binding is inconsistent")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
        raise PeadDailyReconciliationError("receipt blockers are noncanonical")
    required = {
        "authoritative_distribution_semantics_and_payment_dates_missing",
        "borrow_financing_capacity_evidence_missing",
        "full_historical_and_prospective_evidence_missing",
        "observed_broker_execution_evidence_missing",
        "pooled_daily_scope_not_full_eight_cell_family",
        "split_normalized_accounting_quantities_are_not_broker_orders",
    }
    if not required.issubset(blockers):
        raise PeadDailyReconciliationError("receipt hides required limitations")
    if source_report is not None:
        core = pead_reconciliation_input(source_report)
        if bindings.get("source_report_core_hash") != core["artifact_hash"]:
            raise PeadDailyReconciliationError("receipt binds another PEAD report core")
        if bindings.get("combined_data_snapshot_hash") != source_report[
            "combined_data_snapshot"
        ]["artifact_hash"]:
            raise PeadDailyReconciliationError("receipt combined snapshot differs from report")
        if bindings.get("economic_return_inputs_hash") != source_report[
            "economic_return_inputs"
        ]["artifact_hash"]:
            raise PeadDailyReconciliationError("receipt economic inputs differ from report")
        if bindings.get("research_manifest_binding_hash") != source_report[
            "research_manifest_binding"
        ]["artifact_hash"]:
            raise PeadDailyReconciliationError("receipt research protocol differs from report")
    return payload, evidence


def validate_pead_daily_reconciliation_receipt(
    document: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
    modeled_ledger: Mapping[str, Any],
    independent_reference: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
    protocol: Mapping[str, Any],
    primary_daily_ledger: Mapping[str, Any],
    independent_daily_ledger: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fully rebuild both paths and the generic/PEAD-specific receipt."""
    _validate_receipt_static(document, source_report=source_report)
    rebuilt = build_pead_daily_reconciliation_receipt(
        source_report,
        modeled_ledger,
        independent_reference,
        daily_inputs,
        protocol,
        primary_daily_ledger,
        independent_daily_ledger,
        repository_root=repository_root,
    )
    if canonical_json(rebuilt) != canonical_json(document):
        raise PeadDailyReconciliationError("PEAD daily receipt differs from full rebuild")
    return _plain(document)


def _inspect_receipt_for_report(
    document: Mapping[str, Any], source_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Inspect self-consistency against a report core without truth validation.

    This helper is deliberately private: content/hash replay is necessary but
    insufficient to prove the stored implementation outputs were actually
    produced by the bound code. Report acceptance requires the replay-gated
    token from :mod:`analysis.pead_daily_acceptance`.
    """
    payload, evidence = _validate_receipt_static(document, source_report=source_report)
    if not evidence.passed or not payload[
        "bounded_modeled_daily_money_path_reconciliation_passed"
    ]:
        raise PeadDailyReconciliationError("PEAD daily reconciliation did not pass")
    return _plain(payload)


def _inspect_receipt_for_bindings(
    document: Mapping[str, Any],
    *,
    combined_data_snapshot_hash: str,
    economic_return_inputs_hash: str,
    research_manifest_binding_hash: str,
) -> dict[str, Any]:
    """Inspect static source bindings; never use this as report acceptance."""
    payload, evidence = _validate_receipt_static(document)
    if not evidence.passed or not payload[
        "bounded_modeled_daily_money_path_reconciliation_passed"
    ]:
        raise PeadDailyReconciliationError("PEAD daily reconciliation did not pass")
    bindings = payload["bindings"]
    expected = {
        "combined_data_snapshot_hash": combined_data_snapshot_hash,
        "economic_return_inputs_hash": economic_return_inputs_hash,
        "research_manifest_binding_hash": research_manifest_binding_hash,
    }
    for field, value in expected.items():
        if bindings.get(field) != value:
            raise PeadDailyReconciliationError(
                f"PEAD daily receipt {field} differs from current report inputs"
            )
    return _plain(payload)


__all__ = [
    "CANDIDATE_ID",
    "PeadDailyReconciliationError",
    "SCHEMA_VERSION",
    "build_pead_daily_reconciliation_receipt",
    "current_implementation_manifests",
    "pead_reconciliation_input",
    "validate_pead_daily_reconciliation_receipt",
]
