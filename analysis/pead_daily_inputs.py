"""Frozen, content-addressed inputs for bounded daily PEAD accounting.

The primary and independent daily money-path implementations must consume the
same immutable rows without sharing accounting code.  This module is that
neutral boundary.  Construction validates the exact v6 report, modeled ledger,
independent signal comparison, and warehouse bytes; the public validator is
deliberately self-contained so an independent implementation need not import
the primary ledger module.

``ACTIONS.date`` and ``ACTIONS.value`` remain explicitly unproved economic
semantics.  The snapshot preserves the rows but never upgrades them to payment
dates, spendable cash, broker quotes, or executable terminal proceeds.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
from numbers import Real
from typing import Any

from data.pead_economic_evidence import (
    PeadEconomicEvidenceError,
    canonical_json,
    content_hash,
)


SCHEMA_VERSION = "pead_daily_input_snapshot.v1"
SOURCE_REPORT_SCHEMA_VERSION = "pead_replication_report.v6"
LEDGER_SCHEMA_VERSION = "pead_modeled_execution_ledger.v1"
REFERENCE_SCHEMA_VERSION = "pead_reference_reconciliation.v4"
CANDIDATE_ID = "pead-vq-locked-replication-v1"

WAREHOUSE_TABLES = ("actions", "daily", "sep", "tickers")
HORIZONS = (21, 63)
MINIMUM_NAMES = 10
QUANTILE = Decimal("0.2")
DECIMAL_PRECISION = 50
DECIMAL_QUANTUM = Decimal("0.000000000000000001")
SIGNAL_TOLERANCE = Decimal("0.000000000001")

_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "qualifying_evidence",
    "paper_execution_evidence",
    "promotion_allowed",
    "bindings",
    "distribution_semantics",
    "envelope",
    "formation_observations",
    "selected_paths",
    "sessions",
    "prices",
    "actions",
    "currencies",
    "path_coverage",
    "coverage",
    "blockers",
}
_BINDING_FIELDS = {
    "source_report_file_sha256",
    "source_report_schema_version",
    "combined_data_snapshot_hash",
    "economic_return_inputs_hash",
    "modeled_execution_ledger_hash",
    "independent_reference_artifact_hash",
    "protocol_hash",
    "warehouse_snapshot_version",
    "warehouse_snapshot",
}
_WAREHOUSE_FIELDS = {"version", "tables", "complete", "quality_flags"}
_WAREHOUSE_TABLE_FIELDS = {"table", "sha256", "bytes"}
_SEMANTICS = {
    "action_date_role": "candidate_ex_date_unproven",
    "action_value_role": "candidate_split_normalized_cash_per_share_unproven",
    "adjustment_check_absolute_tolerance": "0.005000000000000000",
    "adjustment_check_relative_tolerance": "0.001000000000000000",
    "holding_interval": "entry_date_exclusive_exit_date_inclusive",
    "payment_date_available": False,
    "cash_settlement_allowed": False,
    "reinvestment": False,
}
_ENVELOPE_FIELDS = {"start", "end"}
_FORMATION_FIELDS = {
    "cohort_id",
    "formation_date",
    "entry_date",
    "exit_date",
    "horizon_sessions",
    "ticker",
    "m_ticker",
    "permaticker",
    "rank",
    "selected_leg",
    "cohort_status",
    "cohort_reason",
    "source_event_key",
    "signal",
}
_SELECTED_PATH_FIELDS = {
    "cohort_id",
    "formation_date",
    "entry_date",
    "exit_date",
    "horizon_sessions",
    "ticker",
    "m_ticker",
    "permaticker",
    "rank",
    "leg",
    "source_event_key",
    "signal",
}
_EVENT_KEY_FIELDS = {"m_ticker", "per_end_date", "per_type"}
_PRICE_FIELDS = {"ticker", "date", "close", "closeadj"}
_ACTION_FIELDS = {
    "date", "action", "ticker", "name", "value", "contraticker", "contraname"
}
_CURRENCY_FIELDS = {"ticker", "currency"}
_PATH_COVERAGE_FIELDS = {
    "cohort_id",
    "ticker",
    "m_ticker",
    "permaticker",
    "horizon_sessions",
    "entry_date",
    "exit_date",
    "session_count",
    "session_dates",
}
_COVERAGE_FIELDS = {
    "formation_observations",
    "selected_paths",
    "selected_tickers",
    "sessions",
    "price_rows",
    "action_rows",
    "currency_rows",
    "path_price_applications",
    "paths_by_horizon",
}
_REQUIRED_BLOCKERS = {
    "cash_distribution_payment_dates_missing",
    "cash_distribution_semantics_source_missing",
    "pooled_daily_scope_not_full_eight_cell_family",
    "split_normalized_prices_are_not_broker_quotes",
}


class PeadDailyInputError(ValueError):
    """The daily PEAD input boundary is malformed or cannot be substantiated."""


def _exact_fields(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadDailyInputError(
            f"{label} fields differ: expected {sorted(expected)}, got {actual}"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadDailyInputError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str, *, uppercase: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadDailyInputError(f"{label} must be non-empty canonical text")
    if uppercase and value != value.upper():
        raise PeadDailyInputError(f"{label} must be uppercase canonical text")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _iso_date(value: Any, label: str) -> str:
    if hasattr(value, "date") and callable(value.date):
        try:
            value = value.date().isoformat()
        except (TypeError, ValueError, AttributeError) as exc:
            raise PeadDailyInputError(f"{label} must be a canonical ISO date") from exc
    elif isinstance(value, date):
        value = value.isoformat()
    if not isinstance(value, str):
        raise PeadDailyInputError(f"{label} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadDailyInputError(f"{label} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise PeadDailyInputError(f"{label} must be a canonical ISO date")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PeadDailyInputError(
            f"{label} must be an integer of at least {minimum}"
        )
    return value


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, Real, Decimal)):
        raise PeadDailyInputError(f"{label} must be a finite decimal")
    if isinstance(value, float) and not math.isfinite(value):
        raise PeadDailyInputError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PeadDailyInputError(f"{label} must be a finite decimal") from exc
    if not result.is_finite() or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise PeadDailyInputError(f"{label} must be a finite {qualifier}decimal")
    return result


def _fixed(value: Any, label: str, *, positive: bool = False) -> str:
    parsed = _decimal(value, label, positive=positive)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        try:
            result = parsed.quantize(DECIMAL_QUANTUM)
        except InvalidOperation as exc:
            raise PeadDailyInputError(f"{label} exceeds the decimal policy") from exc
    if result == 0:
        result = abs(result)
    return format(result, "f")


def _fixed_decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or _fixed(value, label, positive=positive) != value:
        raise PeadDailyInputError(f"{label} must be a fixed-scale decimal string")
    return _decimal(value, label, positive=positive)


def _plain(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except PeadEconomicEvidenceError as exc:
        raise PeadDailyInputError("daily input is not strict JSON") from exc


def source_report_file_sha256(report: Mapping[str, Any]) -> str:
    """Hash the exact canonical one-line report bytes used by project CLIs."""
    if not isinstance(report, Mapping):
        raise PeadDailyInputError("source report must be an object")
    try:
        encoded = (canonical_json(report) + "\n").encode("utf-8")
    except PeadEconomicEvidenceError as exc:
        raise PeadDailyInputError("source report is not strict JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _verified_wrapper(value: Any, label: str) -> Mapping[str, Any]:
    wrapper = _exact_fields(value, _WRAPPER_FIELDS, label)
    claimed = _sha256(wrapper["artifact_hash"], f"{label} artifact hash")
    payload = wrapper["payload"]
    if not isinstance(payload, Mapping) or content_hash(payload) != claimed:
        raise PeadDailyInputError(f"{label} artifact hash mismatch")
    return wrapper


def _validate_warehouse_receipt(value: Any, label: str) -> dict[str, Any]:
    receipt = _exact_fields(value, _WAREHOUSE_FIELDS, label)
    if receipt["complete"] is not True or receipt["quality_flags"] != []:
        raise PeadDailyInputError(f"{label} is incomplete or has quality flags")
    rows = receipt["tables"]
    if not isinstance(rows, list) or len(rows) != len(WAREHOUSE_TABLES):
        raise PeadDailyInputError(f"{label} does not bind all required tables")
    normalized_rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for expected_table, raw in zip(WAREHOUSE_TABLES, rows, strict=True):
        row = _exact_fields(raw, _WAREHOUSE_TABLE_FIELDS, f"{label} table")
        table = _text(row["table"], f"{label} table name")
        if table != expected_table:
            raise PeadDailyInputError(f"{label} table order or membership changed")
        table_hash = _sha256(row["sha256"], f"{label} {table} hash")
        size = _integer(row["bytes"], f"{label} {table} bytes")
        digest.update(f"{table}:{table_hash}:{size}\n".encode())
        normalized_rows.append({"table": table, "sha256": table_hash, "bytes": size})
    version = _sha256(receipt["version"], f"{label} version")
    if digest.hexdigest() != version:
        raise PeadDailyInputError(f"{label} version does not match its table manifest")
    return {
        "version": version,
        "tables": normalized_rows,
        "complete": True,
        "quality_flags": [],
    }


def _validate_ledger_document(
    document: Mapping[str, Any], report: Mapping[str, Any]
) -> Mapping[str, Any]:
    # Kept local so importing this neutral module does not import primary code.
    from analysis.pead_execution_ledger import (  # noqa: PLC0415
        PeadExecutionLedgerError,
        validate_pead_execution_ledger,
    )

    try:
        return validate_pead_execution_ledger(document, source_report=report)["payload"]
    except PeadExecutionLedgerError as exc:
        raise PeadDailyInputError("modeled execution ledger failed validation") from exc


def _validate_reference_document(document: Mapping[str, Any]) -> Mapping[str, Any]:
    # Kept local for the same independence reason as the ledger validator.
    from analysis.pead_reference_replication import (  # noqa: PLC0415
        PeadReferenceError,
        verify_reference_artifact,
    )

    try:
        return verify_reference_artifact(document)
    except PeadReferenceError as exc:
        raise PeadDailyInputError("independent reference artifact failed validation") from exc


def _event_key(value: Any, label: str) -> dict[str, Any]:
    row = _exact_fields(value, _EVENT_KEY_FIELDS, label)
    m_ticker = _text(row["m_ticker"], f"{label} m_ticker")
    return {
        "m_ticker": m_ticker,
        "per_end_date": _iso_date(row["per_end_date"], f"{label} per_end_date"),
        "per_type": _text(row["per_type"], f"{label} per_type"),
    }


def _selection_rows(
    report: Mapping[str, Any], horizon: int
) -> dict[str, Mapping[str, Any]]:
    slice_coverage = report.get("slice_coverage")
    if not isinstance(slice_coverage, Mapping):
        raise PeadDailyInputError("source report omits slice coverage")
    pooled = slice_coverage.get("pooled")
    if not isinstance(pooled, Mapping):
        raise PeadDailyInputError("source report omits pooled slice coverage")
    horizons = pooled.get("horizons")
    cell = horizons.get(str(horizon)) if isinstance(horizons, Mapping) else None
    rows = cell.get("frozen_selections") if isinstance(cell, Mapping) else None
    if not isinstance(rows, list):
        raise PeadDailyInputError("source report omits frozen pooled selections")
    result: dict[str, Mapping[str, Any]] = {}
    observed_order: list[str] = []
    fields = {
        "date", "eligible_names", "names_per_leg",
        "short_m_tickers", "long_m_tickers",
    }
    for raw in rows:
        row = _exact_fields(raw, fields, "frozen pooled selection")
        formation = _iso_date(row["date"], "frozen selection date")
        if formation in result:
            raise PeadDailyInputError("frozen pooled selections repeat a formation")
        observed_order.append(formation)
        result[formation] = row
    if observed_order != sorted(observed_order):
        raise PeadDailyInputError("frozen pooled selections are not sorted")
    return result


def _ranked_reference_observations(
    reference_payload: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[dict[str, list[Mapping[str, Any]]], Mapping[str, Any]]:
    outputs = reference_payload.get("outputs")
    reference_outputs = outputs.get("reference") if isinstance(outputs, Mapping) else None
    rows = (
        reference_outputs.get("portfolio_observations")
        if isinstance(reference_outputs, Mapping)
        else None
    )
    if not isinstance(rows, list) or not rows:
        raise PeadDailyInputError("reference output omits portfolio observations")
    coverage = report.get("coverage")
    lifecycle = (
        coverage.get("security_lifecycle_diagnostics")
        if isinstance(coverage, Mapping)
        else None
    )
    if not isinstance(lifecycle, Mapping):
        raise PeadDailyInputError("source report omits security identity evidence")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise PeadDailyInputError("reference portfolio observation is malformed")
        formation = _iso_date(raw.get("formation_date"), "reference formation date")
        m_ticker = _text(raw.get("m_ticker"), "reference m_ticker")
        key = (formation, m_ticker)
        if key in seen:
            raise PeadDailyInputError("reference output repeats a formation/name")
        seen.add(key)
        _text(raw.get("ticker"), "reference ticker", uppercase=True)
        _decimal(raw.get("signal"), "reference signal")
        grouped[formation].append(raw)
    return dict(grouped), lifecycle


def _derive_upstream_rows(
    report: Mapping[str, Any],
    ledger_payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped, lifecycle = _ranked_reference_observations(reference_payload, report)
    manifests_raw = ledger_payload.get("selection_manifest")
    constituents_raw = ledger_payload.get("constituent_ledger")
    if not isinstance(manifests_raw, list) or not isinstance(constituents_raw, list):
        raise PeadDailyInputError("validated ledger omits selection paths")
    manifests: dict[str, Mapping[str, Any]] = {}
    for raw in manifests_raw:
        if not isinstance(raw, Mapping):
            raise PeadDailyInputError("ledger selection manifest is malformed")
        cohort_id = _text(raw.get("cohort_id"), "ledger cohort_id")
        if cohort_id in manifests:
            raise PeadDailyInputError("ledger repeats a selection cohort")
        manifests[cohort_id] = raw

    formation_rows: list[dict[str, Any]] = []
    expected_selected: list[dict[str, Any]] = []
    expected_cohorts: set[str] = set()
    for horizon in HORIZONS:
        frozen = _selection_rows(report, horizon)
        if set(frozen) - set(grouped):
            raise PeadDailyInputError("frozen selection has no reference formation")
        for formation, raw_group in grouped.items():
            cohort_id = f"pooled:{formation}:{horizon}"
            expected_cohorts.add(cohort_id)
            manifest = manifests.get(cohort_id)
            if manifest is None:
                raise PeadDailyInputError("ledger omits a reference formation cohort")
            ranked = sorted(
                raw_group,
                key=lambda row: (
                    _decimal(row.get("signal"), "reference signal"),
                    _text(row.get("m_ticker"), "reference m_ticker"),
                ),
            )
            count = len(ranked)
            k = max(1, int(QUANTILE * count)) if count >= MINIMUM_NAMES else 0
            selection = frozen.get(formation)
            if count < MINIMUM_NAMES:
                if selection is not None:
                    raise PeadDailyInputError("below-floor formation has a frozen selection")
                expected_status = "below_minimum_names"
                expected_reason = "formation_below_frozen_ten_name_floor"
                short_names: list[str] = []
                long_names: list[str] = []
            else:
                if selection is None:
                    raise PeadDailyInputError("floor-passing formation lacks a frozen selection")
                if (
                    _integer(selection["eligible_names"], "eligible names", minimum=1)
                    != count
                    or _integer(selection["names_per_leg"], "names per leg", minimum=1)
                    != k
                ):
                    raise PeadDailyInputError("frozen selection size differs from reference")
                short_names = [
                    _text(value, "frozen short m_ticker")
                    for value in selection["short_m_tickers"]
                ] if isinstance(selection["short_m_tickers"], list) else []
                long_names = [
                    _text(value, "frozen long m_ticker")
                    for value in selection["long_m_tickers"]
                ] if isinstance(selection["long_m_tickers"], list) else []
                expected_short = [str(row["m_ticker"]) for row in ranked[:k]]
                expected_long = [str(row["m_ticker"]) for row in ranked[-k:]]
                if short_names != expected_short or long_names != expected_long:
                    raise PeadDailyInputError(
                        "frozen pooled selection differs from independent ranking"
                    )
                expected_status = manifest.get("status")
                expected_reason = manifest.get("reason")
                if expected_status not in {"admitted", "excluded_selected_path"}:
                    raise PeadDailyInputError("floor-passing ledger cohort has invalid status")

            if (
                manifest.get("formation_date") != formation
                or manifest.get("horizon_sessions") != horizon
                or manifest.get("eligible_names") != count
                or manifest.get("names_per_leg") != k
                or manifest.get("status") != expected_status
                or manifest.get("reason") != expected_reason
                or manifest.get("short_m_tickers") != short_names
                or manifest.get("long_m_tickers") != long_names
            ):
                raise PeadDailyInputError(
                    "ledger selection manifest differs from reference/frozen selection"
                )
            ranked_manifest = manifest.get("ranked_constituents")
            if not isinstance(ranked_manifest, list) or len(ranked_manifest) != count:
                raise PeadDailyInputError("ledger ranking is not exhaustive")

            for rank, (observation, ledger_ranked) in enumerate(
                zip(ranked, ranked_manifest, strict=True), start=1
            ):
                if not isinstance(ledger_ranked, Mapping):
                    raise PeadDailyInputError("ledger ranked constituent is malformed")
                ticker = _text(observation.get("ticker"), "reference ticker", uppercase=True)
                m_ticker = _text(observation.get("m_ticker"), "reference m_ticker")
                identity = lifecycle.get(ticker)
                if not isinstance(identity, Mapping) or identity.get("status") != "validated":
                    raise PeadDailyInputError("reference ticker lacks validated identity")
                permaticker = _integer(
                    identity.get("permaticker"), "reference permaticker", minimum=1
                )
                signal = _decimal(observation.get("signal"), "reference signal")
                event_key = _event_key(
                    observation.get("source_event_key"), "reference source event key"
                )
                entry = _iso_date(observation.get("entry_date"), "reference entry date")
                exit_date = _iso_date(
                    observation.get(f"target_exit_date_{horizon}"),
                    "reference exit date",
                )
                leg = (
                    "short" if m_ticker in short_names
                    else "long" if m_ticker in long_names
                    else None
                )
                if (
                    ledger_ranked.get("rank") != rank
                    or ledger_ranked.get("ticker") != ticker
                    or ledger_ranked.get("m_ticker") != m_ticker
                    or ledger_ranked.get("permaticker") != permaticker
                    or ledger_ranked.get("selected_leg") != leg
                    or _plain(ledger_ranked.get("source_event_key")) != event_key
                    or abs(
                        _decimal(ledger_ranked.get("signal"), "ledger ranked signal")
                        - signal
                    ) > SIGNAL_TOLERANCE
                ):
                    raise PeadDailyInputError(
                        "ledger ranked constituent differs from reference identity"
                    )
                row = {
                    "cohort_id": cohort_id,
                    "formation_date": formation,
                    "entry_date": entry,
                    "exit_date": exit_date,
                    "horizon_sessions": horizon,
                    "ticker": ticker,
                    "m_ticker": m_ticker,
                    "permaticker": permaticker,
                    "rank": rank,
                    "selected_leg": leg,
                    "cohort_status": expected_status,
                    "cohort_reason": expected_reason,
                    "source_event_key": event_key,
                    "signal": _fixed(signal, "reference signal"),
                }
                formation_rows.append(row)
                if leg is not None and expected_status == "admitted":
                    expected_selected.append(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"selected_leg", "cohort_status", "cohort_reason"}
                        }
                        | {"leg": leg}
                    )

    if set(manifests) != expected_cohorts:
        raise PeadDailyInputError("ledger selection manifest has an extra cohort")

    constituent_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in constituents_raw:
        if not isinstance(raw, Mapping):
            raise PeadDailyInputError("ledger constituent path is malformed")
        key = (
            _text(raw.get("cohort_id"), "ledger path cohort_id"),
            _text(raw.get("m_ticker"), "ledger path m_ticker"),
        )
        if key in constituent_map:
            raise PeadDailyInputError("ledger repeats a selected path")
        constituent_map[key] = raw
    expected_map = {(row["cohort_id"], row["m_ticker"]): row for row in expected_selected}
    if set(constituent_map) != set(expected_map):
        raise PeadDailyInputError(
            "ledger and independent reference selected path unions differ"
        )
    for key, expected in expected_map.items():
        raw = constituent_map[key]
        exact = {
            "formation_date": expected["formation_date"],
            "entry_date": expected["entry_date"],
            "exit_date": expected["exit_date"],
            "horizon_sessions": expected["horizon_sessions"],
            "ticker": expected["ticker"],
            "m_ticker": expected["m_ticker"],
            "permaticker": expected["permaticker"],
            "rank": expected["rank"],
            "leg": expected["leg"],
            "source_event_key": expected["source_event_key"],
        }
        if any(_plain(raw.get(field)) != value for field, value in exact.items()):
            raise PeadDailyInputError(
                "ledger selected path differs from independent reference path"
            )
        if abs(
            _decimal(raw.get("signal"), "ledger selected signal")
            - _decimal(expected["signal"], "reference selected signal")
        ) > SIGNAL_TOLERANCE:
            raise PeadDailyInputError("ledger selected path signal differs from reference")

    formation_rows.sort(
        key=lambda row: (row["formation_date"], row["horizon_sessions"], row["rank"])
    )
    expected_selected.sort(
        key=lambda row: (row["formation_date"], row["horizon_sessions"], row["rank"])
    )
    return formation_rows, expected_selected


def _validate_upstream_sources(
    report: Mapping[str, Any],
    ledger: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    report_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise PeadDailyInputError("source report must be an object")
    if report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION:
        raise PeadDailyInputError("unsupported source report schema")
    if report.get("candidate_id") != CANDIDATE_ID:
        raise PeadDailyInputError("source report belongs to another candidate")
    actual_report_hash = source_report_file_sha256(report)
    if report_sha256 is not None and (
        _sha256(report_sha256, "claimed source report hash") != actual_report_hash
    ):
        raise PeadDailyInputError("source report file hash mismatch")

    combined = _verified_wrapper(
        report.get("combined_data_snapshot"), "combined data snapshot"
    )
    economic = _verified_wrapper(
        report.get("economic_return_inputs"), "economic return inputs"
    )
    if economic["payload"].get("combined_data_snapshot_hash") != combined["artifact_hash"]:
        raise PeadDailyInputError("economic inputs do not bind the combined snapshot")
    semantics = economic["payload"].get("cash_distribution_semantics")
    semantics_payload = (
        semantics.get("payload") if isinstance(semantics, Mapping) else None
    )
    actions_semantics = (
        semantics_payload.get("ACTIONS") if isinstance(semantics_payload, Mapping) else None
    )
    adjustment_tolerance = (
        semantics_payload.get("adjustment_check_tolerance")
        if isinstance(semantics_payload, Mapping)
        else None
    )
    if not isinstance(actions_semantics, Mapping) or (
        actions_semantics.get("date_role") != "candidate_ex_date_unproven"
        or actions_semantics.get("value_role")
        != "candidate_split_normalized_cash_per_share_unproven"
        or semantics_payload.get("qualification_allowed") is not False
        or not isinstance(adjustment_tolerance, Mapping)
        or _decimal(
            adjustment_tolerance.get("absolute"), "absolute adjustment tolerance"
        ) != Decimal("0.005")
        or _decimal(
            adjustment_tolerance.get("relative"), "relative adjustment tolerance"
        ) != Decimal("0.001")
    ):
        raise PeadDailyInputError("source report overstates ACTIONS semantics")

    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping) or (
        configuration.get("horizons_sessions") != list(HORIZONS)
        or configuration.get("minimum_names_per_formation_per_slice") != MINIMUM_NAMES
        or _decimal(configuration.get("quantile"), "report quantile") != QUANTILE
        or configuration.get("signal_tie_break") != "ascending stable m_ticker"
        or configuration.get("cash_distribution_holding_interval")
        != _SEMANTICS["holding_interval"]
        or configuration.get("cash_distribution_reinvestment") is not False
    ):
        raise PeadDailyInputError("source report configuration changed")

    source_snapshot = report.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise PeadDailyInputError("source report omits source snapshot")
    report_warehouse = _validate_warehouse_receipt(
        source_snapshot.get("warehouse_return_snapshot"),
        "source report warehouse snapshot",
    )
    if source_snapshot.get("warehouse_snapshot_unchanged_during_run") is not True:
        raise PeadDailyInputError("source report did not freeze the warehouse")
    combined_warehouse = _validate_warehouse_receipt(
        combined["payload"].get("warehouse_return_snapshot"),
        "combined warehouse snapshot",
    )
    if combined_warehouse != report_warehouse:
        raise PeadDailyInputError("combined and source warehouse snapshots differ")

    ledger_wrapper = _verified_wrapper(ledger, "modeled execution ledger")
    ledger_payload = _validate_ledger_document(ledger, report)
    if ledger_payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise PeadDailyInputError("unsupported modeled execution ledger schema")
    ledger_bindings = ledger_payload.get("bindings")
    if not isinstance(ledger_bindings, Mapping) or (
        ledger_bindings.get("source_report_file_sha256") != actual_report_hash
        or ledger_bindings.get("combined_data_snapshot_hash") != combined["artifact_hash"]
        or ledger_bindings.get("economic_return_inputs_hash") != economic["artifact_hash"]
    ):
        raise PeadDailyInputError("modeled ledger input bindings differ")
    protocol_hash = _sha256(ledger_bindings.get("protocol_hash"), "protocol hash")

    reference_wrapper = _verified_wrapper(reference, "independent reference")
    reference_payload = _validate_reference_document(reference)
    if reference_payload.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise PeadDailyInputError("unsupported independent reference schema")
    comparison = reference_payload.get("comparison")
    if not isinstance(comparison, Mapping) or (
        comparison.get("signal_path_passed") is not True
        or comparison.get("discrepancy_count") != 0
        or comparison.get("discrepancies") != []
    ):
        raise PeadDailyInputError("independent signal comparison did not pass")
    reference_bindings = reference_payload.get("bindings")
    if not isinstance(reference_bindings, Mapping) or (
        reference_bindings.get("primary_report_sha256") != actual_report_hash
        or reference_bindings.get("data_snapshot_hash") != combined["artifact_hash"]
        or reference_bindings.get("protocol_hash") != protocol_hash
        or _plain(reference_bindings.get("economic_return_inputs")) != _plain(economic)
    ):
        raise PeadDailyInputError("independent reference input bindings differ")
    reference_warehouse = _validate_warehouse_receipt(
        reference_bindings.get("warehouse_snapshot"),
        "independent reference warehouse snapshot",
    )
    if reference_warehouse != report_warehouse:
        raise PeadDailyInputError("independent reference warehouse snapshot differs")

    formation_rows, selected_paths = _derive_upstream_rows(
        report, ledger_payload, reference_payload
    )
    return {
        "bindings": {
            "source_report_file_sha256": actual_report_hash,
            "source_report_schema_version": SOURCE_REPORT_SCHEMA_VERSION,
            "combined_data_snapshot_hash": combined["artifact_hash"],
            "economic_return_inputs_hash": economic["artifact_hash"],
            "modeled_execution_ledger_hash": ledger_wrapper["artifact_hash"],
            "independent_reference_artifact_hash": reference_wrapper["artifact_hash"],
            "protocol_hash": protocol_hash,
            "warehouse_snapshot_version": report_warehouse["version"],
            "warehouse_snapshot": report_warehouse,
        },
        "formation_observations": formation_rows,
        "selected_paths": selected_paths,
    }


def _normalize_sessions(raw: Any, start: str, end: str) -> list[str]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) and not hasattr(raw, "__iter__"):
        raise PeadDailyInputError("market sessions must be an ordered sequence")
    try:
        sessions = [_iso_date(value, "market session") for value in raw]
    except TypeError as exc:
        raise PeadDailyInputError("market sessions must be iterable") from exc
    if not sessions or sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise PeadDailyInputError("market sessions must be nonempty, sorted, and unique")
    if sessions[0] != start or sessions[-1] != end:
        raise PeadDailyInputError("market session envelope endpoints differ from paths")
    return sessions


def _series_rows(raw: Any, ticker: str, field: str) -> list[tuple[str, str]]:
    if raw is None or isinstance(raw, (str, bytes)) or not hasattr(raw, "items"):
        raise PeadDailyInputError(f"{field} reader returned a non-series for {ticker}")
    index = getattr(raw, "index", None)
    if index is not None and bool(getattr(index, "has_duplicates", False)):
        raise PeadDailyInputError(f"{field} contains duplicate dates for {ticker}")
    rows: list[tuple[str, str]] = []
    for raw_date, raw_value in raw.items():
        rows.append(
            (
                _iso_date(raw_date, f"{ticker} {field} date"),
                _fixed(raw_value, f"{ticker} {field}", positive=True),
            )
        )
    dates = [item[0] for item in rows]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise PeadDailyInputError(f"{field} dates are not sorted and unique for {ticker}")
    return rows


def _normalize_actions(
    value: Any, tickers: set[str], start: str, end: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PeadDailyInputError("corporate-action reader must return a list")
    result: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for raw in value:
        row = _exact_fields(raw, _ACTION_FIELDS, "corporate action")
        action_date = _iso_date(row["date"], "corporate action date")
        ticker = _text(row["ticker"], "corporate action ticker", uppercase=True)
        if ticker not in tickers or not start <= action_date <= end:
            raise PeadDailyInputError("corporate action falls outside the requested slice")
        action = _text(row["action"], "corporate action kind")
        name = _text(row["name"], "corporate action name")
        contra_ticker = _optional_text(row["contraticker"], "corporate action contraticker")
        contra_name = _optional_text(row["contraname"], "corporate action contraname")
        identity = (ticker, action_date, action, name, contra_ticker, contra_name)
        if identity in identities:
            raise PeadDailyInputError("corporate-action slice repeats a primary key")
        identities.add(identity)
        result.append(
            {
                "date": action_date,
                "action": action,
                "ticker": ticker,
                "name": name,
                "value": None if row["value"] is None else _fixed(
                    row["value"], "corporate action value"
                ),
                "contraticker": contra_ticker,
                "contraname": contra_name,
            }
        )
    result.sort(
        key=lambda row: (
            row["ticker"], row["date"], row["action"], row["name"],
            row["contraticker"] or "", row["contraname"] or "",
        )
    )
    return result


def _read_provider_rows(
    provider: Any, selected_paths: list[dict[str, Any]], expected_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    if not selected_paths:
        raise PeadDailyInputError("daily input snapshot requires admitted selected paths")
    snapshot_reader = getattr(provider, "snapshot_version", None)
    session_reader = getattr(provider, "market_sessions", None)
    price_reader = getattr(provider, "prices_strict", None)
    action_reader = getattr(provider, "corporate_actions_for_tickers", None)
    currency_reader = getattr(provider, "security_currency", None)
    if not all(
        callable(reader)
        for reader in (
            snapshot_reader, session_reader, price_reader, action_reader, currency_reader
        )
    ):
        raise PeadDailyInputError("provider lacks a required strict daily input reader")
    try:
        before = _validate_warehouse_receipt(
            snapshot_reader(WAREHOUSE_TABLES), "provider warehouse snapshot before reads"
        )
    except PeadDailyInputError:
        raise
    except Exception as exc:
        raise PeadDailyInputError("provider snapshot read failed before daily reads") from exc
    if before != expected_receipt:
        raise PeadDailyInputError("provider warehouse snapshot differs from bound report")

    start = min(row["entry_date"] for row in selected_paths)
    end = max(row["exit_date"] for row in selected_paths)
    tickers = sorted({row["ticker"] for row in selected_paths})
    try:
        sessions = _normalize_sessions(session_reader(start, end), start, end)
        session_index = {value: index for index, value in enumerate(sessions)}
        path_coverage: list[dict[str, Any]] = []
        for path in selected_paths:
            entry_index = session_index.get(path["entry_date"])
            exit_index = session_index.get(path["exit_date"])
            if (
                entry_index is None
                or exit_index is None
                or exit_index - entry_index != path["horizon_sessions"]
            ):
                raise PeadDailyInputError(
                    "selected path does not span its exact global-session horizon"
                )
            path_sessions = sessions[entry_index : exit_index + 1]
            path_coverage.append(
                {
                    "cohort_id": path["cohort_id"],
                    "ticker": path["ticker"],
                    "m_ticker": path["m_ticker"],
                    "permaticker": path["permaticker"],
                    "horizon_sessions": path["horizon_sessions"],
                    "entry_date": path["entry_date"],
                    "exit_date": path["exit_date"],
                    "session_count": len(path_sessions),
                    "session_dates": path_sessions,
                }
            )

        prices: list[dict[str, Any]] = []
        available: dict[str, set[str]] = {}
        session_set = set(sessions)
        for ticker in tickers:
            close = _series_rows(price_reader(ticker, start, end, "close"), ticker, "close")
            closeadj = _series_rows(
                price_reader(ticker, start, end, "closeadj"), ticker, "closeadj"
            )
            if [item[0] for item in close] != [item[0] for item in closeadj]:
                raise PeadDailyInputError(
                    f"close and closeadj date coverage differ for {ticker}"
                )
            if any(item[0] not in session_set for item in close):
                raise PeadDailyInputError(f"price row falls outside sessions for {ticker}")
            available[ticker] = {item[0] for item in close}
            prices.extend(
                {
                    "ticker": ticker,
                    "date": close_row[0],
                    "close": close_row[1],
                    "closeadj": adjusted_row[1],
                }
                for close_row, adjusted_row in zip(close, closeadj, strict=True)
            )
        for path in path_coverage:
            missing = set(path["session_dates"]) - available[path["ticker"]]
            if missing:
                raise PeadDailyInputError(
                    "selected path is missing an internal strict price bar"
                )

        actions = _normalize_actions(
            action_reader(tickers, start, end), set(tickers), start, end
        )
        currencies: list[dict[str, str]] = []
        for ticker in tickers:
            raw_currency = _exact_fields(
                currency_reader(ticker), _CURRENCY_FIELDS, "security currency"
            )
            if raw_currency["ticker"] != ticker:
                raise PeadDailyInputError("security currency identity changed")
            currencies.append(
                {
                    "ticker": ticker,
                    "currency": _text(
                        raw_currency["currency"], "security currency", uppercase=True
                    ),
                }
            )
        after = _validate_warehouse_receipt(
            snapshot_reader(WAREHOUSE_TABLES), "provider warehouse snapshot after reads"
        )
    except PeadDailyInputError:
        raise
    except Exception as exc:
        raise PeadDailyInputError("strict provider daily input read failed") from exc
    if after != before:
        raise PeadDailyInputError("provider warehouse snapshot changed during daily reads")
    prices.sort(key=lambda row: (row["ticker"], row["date"]))
    path_coverage.sort(
        key=lambda row: (row["entry_date"], row["horizon_sessions"], row["m_ticker"])
    )
    return {
        "envelope": {"start": start, "end": end},
        "sessions": sessions,
        "prices": prices,
        "actions": actions,
        "currencies": currencies,
        "path_coverage": path_coverage,
    }


def _expected_coverage(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected_paths = payload["selected_paths"]
    return {
        "formation_observations": len(payload["formation_observations"]),
        "selected_paths": len(selected_paths),
        "selected_tickers": len({row["ticker"] for row in selected_paths}),
        "sessions": len(payload["sessions"]),
        "price_rows": len(payload["prices"]),
        "action_rows": len(payload["actions"]),
        "currency_rows": len(payload["currencies"]),
        "path_price_applications": sum(
            row["session_count"] for row in payload["path_coverage"]
        ),
        "paths_by_horizon": {
            str(horizon): sum(
                row["horizon_sessions"] == horizon for row in selected_paths
            )
            for horizon in HORIZONS
        },
    }


def build_pead_daily_input_snapshot(
    report: Mapping[str, Any],
    ledger: Mapping[str, Any],
    reference: Mapping[str, Any],
    provider: Any,
    *,
    report_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact immutable row slice shared by both daily implementations."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        upstream = _validate_upstream_sources(
            report, ledger, reference, report_sha256=report_sha256
        )
        provider_rows = _read_provider_rows(
            provider,
            upstream["selected_paths"],
            upstream["bindings"]["warehouse_snapshot"],
        )
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "evidence_class": "content_addressed_exact_daily_input_slice_nonqualifying",
            "qualifying_evidence": False,
            "paper_execution_evidence": False,
            "promotion_allowed": False,
            "bindings": upstream["bindings"],
            "distribution_semantics": dict(_SEMANTICS),
            "envelope": provider_rows["envelope"],
            "formation_observations": upstream["formation_observations"],
            "selected_paths": upstream["selected_paths"],
            "sessions": provider_rows["sessions"],
            "prices": provider_rows["prices"],
            "actions": provider_rows["actions"],
            "currencies": provider_rows["currencies"],
            "path_coverage": provider_rows["path_coverage"],
            "coverage": {},
            "blockers": sorted(_REQUIRED_BLOCKERS),
        }
        payload["coverage"] = _expected_coverage(payload)
        normalized = _plain(payload)
        document = {"artifact_hash": content_hash(normalized), "payload": normalized}
        return validate_pead_daily_input_snapshot(document)


def _validate_formation_rows(value: Any) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise PeadDailyInputError("formation observations must be nonempty")
    normalized: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for raw in value:
        row = _exact_fields(raw, _FORMATION_FIELDS, "formation observation")
        formation = _iso_date(row["formation_date"], "formation date")
        horizon = _integer(row["horizon_sessions"], "formation horizon", minimum=1)
        if horizon not in HORIZONS:
            raise PeadDailyInputError("formation observation has an unsupported horizon")
        cohort_id = _text(row["cohort_id"], "formation cohort_id")
        if cohort_id != f"pooled:{formation}:{horizon}":
            raise PeadDailyInputError("formation cohort identity is invalid")
        m_ticker = _text(row["m_ticker"], "formation m_ticker")
        token = (formation, horizon, m_ticker)
        if token in seen:
            raise PeadDailyInputError("formation observations repeat a key")
        seen.add(token)
        leg = row["selected_leg"]
        if leg not in {None, "short", "long"}:
            raise PeadDailyInputError("formation selected leg is invalid")
        status = row["cohort_status"]
        if status not in {"below_minimum_names", "admitted", "excluded_selected_path"}:
            raise PeadDailyInputError("formation cohort status is invalid")
        reason = row["cohort_reason"]
        if status == "admitted" and reason is not None:
            raise PeadDailyInputError("admitted formation cannot have a reason")
        if status != "admitted":
            _text(reason, "formation cohort reason")
        normalized_row = {
            "cohort_id": cohort_id,
            "formation_date": formation,
            "entry_date": _iso_date(row["entry_date"], "formation entry date"),
            "exit_date": _iso_date(row["exit_date"], "formation exit date"),
            "horizon_sessions": horizon,
            "ticker": _text(row["ticker"], "formation ticker", uppercase=True),
            "m_ticker": m_ticker,
            "permaticker": _integer(row["permaticker"], "formation permaticker", minimum=1),
            "rank": _integer(row["rank"], "formation rank", minimum=1),
            "selected_leg": leg,
            "cohort_status": status,
            "cohort_reason": reason,
            "source_event_key": _event_key(row["source_event_key"], "formation event key"),
            "signal": row["signal"],
        }
        _fixed_decimal(row["signal"], "formation signal")
        groups[(formation, horizon)].append(normalized_row)
        normalized.append(normalized_row)
    expected_order = sorted(
        normalized,
        key=lambda row: (row["formation_date"], row["horizon_sessions"], row["rank"]),
    )
    if normalized != expected_order:
        raise PeadDailyInputError("formation observations are not canonically sorted")

    for group in groups.values():
        if [row["rank"] for row in group] != list(range(1, len(group) + 1)):
            raise PeadDailyInputError("formation ranks are not contiguous")
        if [(Decimal(row["signal"]), row["m_ticker"]) for row in group] != sorted(
            (Decimal(row["signal"]), row["m_ticker"]) for row in group
        ):
            raise PeadDailyInputError("formation ranks violate signal/ticker ordering")
        statuses = {(row["cohort_status"], row["cohort_reason"]) for row in group}
        if len(statuses) != 1:
            raise PeadDailyInputError("formation cohort status differs within a cohort")
        count = len(group)
        if count < MINIMUM_NAMES:
            if statuses != {(
                "below_minimum_names", "formation_below_frozen_ten_name_floor"
            )} or any(row["selected_leg"] is not None for row in group):
                raise PeadDailyInputError("below-floor formation selection is inconsistent")
        else:
            if next(iter(statuses))[0] == "below_minimum_names":
                raise PeadDailyInputError("floor-passing formation marked below floor")
            k = max(1, int(QUANTILE * count))
            expected_legs = ["short"] * k + [None] * (count - 2 * k) + ["long"] * k
            if [row["selected_leg"] for row in group] != expected_legs:
                raise PeadDailyInputError("formation tail selection is inconsistent")
    return normalized, {(row["cohort_id"], row["m_ticker"]): row for row in normalized}


def _validate_selected_paths(
    value: Any, formation_map: Mapping[tuple[str, str], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PeadDailyInputError("selected paths must be nonempty")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value:
        row = _exact_fields(raw, _SELECTED_PATH_FIELDS, "selected path")
        cohort_id = _text(row["cohort_id"], "selected path cohort_id")
        m_ticker = _text(row["m_ticker"], "selected path m_ticker")
        token = (cohort_id, m_ticker)
        if token in seen:
            raise PeadDailyInputError("selected paths repeat a key")
        seen.add(token)
        formation = formation_map.get(token)
        if formation is None or formation["cohort_status"] != "admitted":
            raise PeadDailyInputError("selected path lacks an admitted formation row")
        leg = row["leg"]
        if leg not in {"short", "long"} or formation["selected_leg"] != leg:
            raise PeadDailyInputError("selected path leg differs from formation selection")
        expected = {
            key: formation[key]
            for key in _SELECTED_PATH_FIELDS
            if key != "leg"
        } | {"leg": leg}
        if _plain(row) != _plain(expected):
            raise PeadDailyInputError("selected path differs from its formation identity")
        normalized.append(dict(expected))
    expected_all = {
        key for key, row in formation_map.items()
        if row["cohort_status"] == "admitted" and row["selected_leg"] is not None
    }
    if seen != expected_all:
        raise PeadDailyInputError("selected path manifest is not exhaustive")
    expected_order = sorted(
        normalized,
        key=lambda row: (row["formation_date"], row["horizon_sessions"], row["rank"]),
    )
    if normalized != expected_order:
        raise PeadDailyInputError("selected paths are not canonically sorted")
    return normalized


def _validate_internal_payload(payload: Mapping[str, Any]) -> None:
    bindings = _exact_fields(payload["bindings"], _BINDING_FIELDS, "daily input bindings")
    for field in _BINDING_FIELDS - {
        "source_report_schema_version", "warehouse_snapshot"
    }:
        _sha256(bindings[field], f"daily input binding {field}")
    if bindings["source_report_schema_version"] != SOURCE_REPORT_SCHEMA_VERSION:
        raise PeadDailyInputError("daily input binds an unsupported report schema")
    receipt = _validate_warehouse_receipt(
        bindings["warehouse_snapshot"], "daily input warehouse snapshot"
    )
    if receipt["version"] != bindings["warehouse_snapshot_version"]:
        raise PeadDailyInputError("daily input warehouse version binding differs")
    if _exact_fields(
        payload["distribution_semantics"], set(_SEMANTICS), "distribution semantics"
    ) != _SEMANTICS:
        raise PeadDailyInputError("daily input distribution semantics changed")
    envelope = _exact_fields(payload["envelope"], _ENVELOPE_FIELDS, "daily envelope")
    start = _iso_date(envelope["start"], "daily envelope start")
    end = _iso_date(envelope["end"], "daily envelope end")
    if start > end:
        raise PeadDailyInputError("daily envelope is reversed")
    sessions = [_iso_date(value, "daily session") for value in payload["sessions"]] \
        if isinstance(payload["sessions"], list) else []
    if (
        not sessions
        or sessions != sorted(sessions)
        or len(sessions) != len(set(sessions))
        or sessions[0] != start
        or sessions[-1] != end
    ):
        raise PeadDailyInputError("daily sessions do not exactly cover the envelope")
    session_index = {value: index for index, value in enumerate(sessions)}

    formation_rows, formation_map = _validate_formation_rows(
        payload["formation_observations"]
    )
    selected_paths = _validate_selected_paths(payload["selected_paths"], formation_map)

    coverage_rows = payload["path_coverage"]
    if not isinstance(coverage_rows, list):
        raise PeadDailyInputError("path coverage must be a list")
    normalized_coverage: list[dict[str, Any]] = []
    coverage_keys: set[tuple[str, str]] = set()
    selected_map = {(row["cohort_id"], row["m_ticker"]): row for row in selected_paths}
    for raw in coverage_rows:
        row = _exact_fields(raw, _PATH_COVERAGE_FIELDS, "path coverage")
        key = (
            _text(row["cohort_id"], "path coverage cohort_id"),
            _text(row["m_ticker"], "path coverage m_ticker"),
        )
        path = selected_map.get(key)
        if path is None or key in coverage_keys:
            raise PeadDailyInputError("path coverage key is missing or duplicated")
        coverage_keys.add(key)
        dates = (
            [_iso_date(value, "path session date") for value in row["session_dates"]]
            if isinstance(row["session_dates"], list)
            else []
        )
        entry_index = session_index.get(path["entry_date"])
        exit_index = session_index.get(path["exit_date"])
        expected_dates = (
            sessions[entry_index : exit_index + 1]
            if entry_index is not None and exit_index is not None
            else []
        )
        expected = {
            "cohort_id": path["cohort_id"],
            "ticker": path["ticker"],
            "m_ticker": path["m_ticker"],
            "permaticker": path["permaticker"],
            "horizon_sessions": path["horizon_sessions"],
            "entry_date": path["entry_date"],
            "exit_date": path["exit_date"],
            "session_count": path["horizon_sessions"] + 1,
            "session_dates": expected_dates,
        }
        if (
            dates != expected_dates
            or len(expected_dates) != path["horizon_sessions"] + 1
            or _plain(row) != expected
        ):
            raise PeadDailyInputError("path coverage is not an exact session path")
        normalized_coverage.append(expected)
    if coverage_keys != set(selected_map):
        raise PeadDailyInputError("path coverage is not exhaustive")
    if normalized_coverage != sorted(
        normalized_coverage,
        key=lambda row: (row["entry_date"], row["horizon_sessions"], row["m_ticker"]),
    ):
        raise PeadDailyInputError("path coverage is not canonically sorted")

    tickers = sorted({row["ticker"] for row in selected_paths})
    price_rows = payload["prices"]
    if not isinstance(price_rows, list):
        raise PeadDailyInputError("price rows must be a list")
    price_keys: set[tuple[str, str]] = set()
    normalized_prices: list[dict[str, Any]] = []
    for raw in price_rows:
        row = _exact_fields(raw, _PRICE_FIELDS, "price row")
        ticker = _text(row["ticker"], "price ticker", uppercase=True)
        price_date = _iso_date(row["date"], "price date")
        if ticker not in tickers or price_date not in session_index:
            raise PeadDailyInputError("price row falls outside the selected envelope")
        key = (ticker, price_date)
        if key in price_keys:
            raise PeadDailyInputError("price rows repeat a ticker/date")
        price_keys.add(key)
        _fixed_decimal(row["close"], "close", positive=True)
        _fixed_decimal(row["closeadj"], "closeadj", positive=True)
        normalized_prices.append(dict(row))
    if normalized_prices != sorted(
        normalized_prices, key=lambda row: (row["ticker"], row["date"])
    ):
        raise PeadDailyInputError("price rows are not canonically sorted")
    for coverage in normalized_coverage:
        if any(
            (coverage["ticker"], value) not in price_keys
            for value in coverage["session_dates"]
        ):
            raise PeadDailyInputError("selected path is missing an internal price row")

    actions = payload["actions"]
    if not isinstance(actions, list):
        raise PeadDailyInputError("action rows must be a list")
    normalized_actions = _normalize_actions(actions, set(tickers), start, end)
    if _plain(actions) != normalized_actions:
        raise PeadDailyInputError("action rows are not canonically normalized")

    currencies = payload["currencies"]
    if not isinstance(currencies, list):
        raise PeadDailyInputError("currency rows must be a list")
    normalized_currencies: list[dict[str, str]] = []
    for raw in currencies:
        row = _exact_fields(raw, _CURRENCY_FIELDS, "currency row")
        normalized_currencies.append(
            {
                "ticker": _text(row["ticker"], "currency ticker", uppercase=True),
                "currency": _text(row["currency"], "currency", uppercase=True),
            }
        )
    if normalized_currencies != sorted(
        normalized_currencies, key=lambda row: row["ticker"]
    ) or [row["ticker"] for row in normalized_currencies] != tickers:
        raise PeadDailyInputError("currency rows are not exhaustive and sorted")

    coverage = _exact_fields(payload["coverage"], _COVERAGE_FIELDS, "daily coverage")
    if coverage != _expected_coverage(payload):
        raise PeadDailyInputError("daily input coverage is inconsistent")
    blockers = payload["blockers"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not _REQUIRED_BLOCKERS.issubset(blockers)
    ):
        raise PeadDailyInputError("daily input blockers are incomplete or noncanonical")
    # Silence an otherwise easy-to-miss accidental mutation of the validated list.
    if formation_rows != payload["formation_observations"]:
        raise PeadDailyInputError("formation observations are not canonical")


def validate_pead_daily_input_snapshot(
    document: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    report: Mapping[str, Any] | None = None,
    ledger: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate content identity and every internal coverage relationship.

    With no source arguments this function is neutral and imports no primary
    accounting code.  Supplying ``report``, ``ledger``, and ``reference`` adds
    exact upstream identity validation; the three must be supplied together.
    ``expected_bindings`` may pin any subset of binding fields independently.
    """
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        wrapper = _exact_fields(document, _WRAPPER_FIELDS, "daily input snapshot")
        claimed = _sha256(wrapper["artifact_hash"], "daily input artifact hash")
        payload = _exact_fields(wrapper["payload"], _PAYLOAD_FIELDS, "daily input payload")
        if content_hash(payload) != claimed:
            raise PeadDailyInputError("daily input snapshot hash mismatch")
        expected_scalars = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "evidence_class": "content_addressed_exact_daily_input_slice_nonqualifying",
            "qualifying_evidence": False,
            "paper_execution_evidence": False,
            "promotion_allowed": False,
        }
        for field, expected in expected_scalars.items():
            if payload[field] != expected:
                raise PeadDailyInputError(f"daily input {field} is invalid")
        _validate_internal_payload(payload)
        bindings = payload["bindings"]
        if expected_bindings is not None:
            if not isinstance(expected_bindings, Mapping):
                raise PeadDailyInputError("expected bindings must be an object")
            unknown = set(expected_bindings) - set(bindings)
            if unknown:
                raise PeadDailyInputError(f"unknown expected bindings: {sorted(unknown)}")
            for field, expected in expected_bindings.items():
                if _plain(bindings[field]) != _plain(expected):
                    raise PeadDailyInputError(f"daily input binding {field} differs")
        provided = (report is not None, ledger is not None, reference is not None)
        if any(provided) and not all(provided):
            raise PeadDailyInputError(
                "report, ledger, and reference must be supplied together"
            )
        if all(provided):
            upstream = _validate_upstream_sources(
                report, ledger, reference, report_sha256=None  # type: ignore[arg-type]
            )
            if _plain(bindings) != _plain(upstream["bindings"]):
                raise PeadDailyInputError("daily input upstream bindings differ")
            if payload["formation_observations"] != upstream["formation_observations"]:
                raise PeadDailyInputError("daily input formation universe differs upstream")
            if payload["selected_paths"] != upstream["selected_paths"]:
                raise PeadDailyInputError("daily input selected paths differ upstream")
        return _plain(document)


def verify_pead_daily_input_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    """Neutral alias used by independently written daily reconstruction code."""
    return validate_pead_daily_input_snapshot(document)


__all__ = [
    "CANDIDATE_ID",
    "PeadDailyInputError",
    "SCHEMA_VERSION",
    "build_pead_daily_input_snapshot",
    "source_report_file_sha256",
    "validate_pead_daily_input_snapshot",
    "verify_pead_daily_input_snapshot",
]
