"""Deterministic, non-broker money accounting for the locked PEAD candidate.

The primary PEAD report proves a signal path and mechanically reconstructs a
candidate close-plus-cash return.  This module takes the next deliberately
bounded step: it rebuilds the pooled selections and turns every admitted
date/horizon cohort into exact targets, split-normalized share-equivalent
quantities, fixed modeled fees, distribution accruals, exits, and P&L.

This is *not* paper execution.  ``SEP.close`` is a split-normalized accounting
basis, the dividend rows do not contain independently proved payment dates,
and there are no quotes, broker orders, fills, borrow records, margin rules, or
capital sharing between overlapping cohorts.  The artifact and its validator
therefore keep paper, promotion, and generic independent-replication
eligibility false.
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
    validate_cash_distribution_semantics,
    validate_terminal_settlement_ledger,
)


SCHEMA_VERSION = "pead_modeled_execution_ledger.v1"
SOURCE_REPORT_SCHEMA_VERSION = "pead_replication_report.v6"
CANDIDATE_ID = "pead-vq-locked-replication-v1"
ECONOMIC_INPUT_SCHEMA_VERSION = "pead_economic_return_inputs.v1"

HORIZONS = (21, 63)
MINIMUM_NAMES = 10
QUANTILE = Decimal("0.2")
INITIAL_NAV = Decimal("1")
LEG_GROSS = Decimal("1")
ONE_WAY_FEE_RATE = Decimal("0.003")
TOTAL_COHORT_FEE = Decimal("0.012")

DECIMAL_PRECISION = 50
MONEY_QUANTUM = Decimal("0.000000000000000001")
QUANTITY_QUANTUM = Decimal("0.000000000000000000000001")
SOURCE_TOLERANCE = Decimal("0.000000000001")
INTERNAL_MONEY_TOLERANCE = MONEY_QUANTUM * Decimal("32")
INTERNAL_SUMMARY_TOLERANCE = MONEY_QUANTUM * Decimal("128")

_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "qualifying_evidence",
    "replication_evidence_eligible",
    "paper_execution_evidence",
    "promotion_allowed",
    "modeled_execution_claim",
    "bindings",
    "frozen_protocol",
    "decimal_policy",
    "selection_manifest",
    "constituent_ledger",
    "cohort_summaries",
    "atomic_checkpoints",
    "coverage",
    "blockers",
}
_MODELED_CLAIM_FIELDS = {
    "claim",
    "broker",
    "account_id",
    "broker_order_ids",
    "broker_fill_ids",
    "observed_quotes",
}
_BINDING_FIELDS = {
    "source_report_file_sha256",
    "source_report_schema_version",
    "combined_data_snapshot_hash",
    "economic_return_inputs_hash",
    "protocol_hash",
}
_PROTOCOL_FIELDS = {
    "slice",
    "horizons_sessions",
    "minimum_names_per_formation",
    "quantile",
    "signal_tie_break",
    "rank_definition",
    "initial_nav_per_independent_cohort",
    "long_gross",
    "short_gross",
    "one_way_fee_rate_per_trade_per_leg",
    "fixed_total_round_trip_fee",
    "quantity_basis",
    "cash_distribution_holding_interval",
    "cash_distribution_reinvestment",
    "distribution_treatment",
    "terminal_payout_from_actions_value_allowed",
    "capital_sharing",
    "cash_yield",
    "financing",
    "borrow",
    "margin_and_short_proceeds",
    "replication_projection_units",
    "checkpoint_state_semantics",
}
_DECIMAL_POLICY_FIELDS = {
    "precision",
    "rounding",
    "money_quantum",
    "quantity_quantum",
    "rounding_stage",
}
_MANIFEST_FIELDS = {
    "cohort_id",
    "formation_date",
    "horizon_sessions",
    "eligible_names",
    "names_per_leg",
    "status",
    "reason",
    "short_m_tickers",
    "long_m_tickers",
    "ranked_constituents",
}
_RANKED_FIELDS = {
    "rank",
    "ticker",
    "m_ticker",
    "permaticker",
    "source_event_key",
    "signal",
    "selected_leg",
    "economic_path_status",
    "terminal_settlement_id",
}
_CONSTITUENT_FIELDS = {
    "cohort_id",
    "formation_date",
    "entry_date",
    "exit_date",
    "horizon_sessions",
    "ticker",
    "m_ticker",
    "permaticker",
    "source_event_key",
    "rank",
    "leg",
    "signal",
    "entry_price_split_normalized",
    "exit_price_split_normalized",
    "cash_total_per_split_normalized_share",
    "distribution_accruals",
    "source_candidate_gross_economic_return",
    "reconstructed_gross_economic_return",
    "signed_target_notional",
    "signed_split_normalized_share_equivalent_quantity",
    "price_pnl",
    "distribution_pnl",
    "entry_fee",
    "exit_fee",
    "total_fees",
    "net_pnl",
}
_DISTRIBUTION_FIELDS = {
    "date",
    "amount_per_split_normalized_share",
    "signed_accrual_pnl",
    "action_key",
}
_SUMMARY_FIELDS = {
    "cohort_id",
    "formation_date",
    "entry_date",
    "exit_date",
    "horizon_sessions",
    "names_per_leg",
    "constituent_count",
    "initial_nav",
    "long_gross_notional",
    "short_gross_notional",
    "gross_target_notional",
    "net_target_notional",
    "entry_cash_after_modeled_fees",
    "entry_fees",
    "gross_factor_return",
    "total_fees",
    "net_factor_return",
    "ledger_net_pnl",
    "terminal_cash",
    "terminal_nav",
    "terminal_open_position_count",
    "return_identity_difference",
}
_CHECKPOINT_FIELDS = {
    "cohort_id",
    "sequence",
    "checkpoint",
    "date",
    "cash",
    "fees",
    "pnl",
    "state_rows",
}
_STATE_FIELDS = {
    "ticker",
    "m_ticker",
    "permaticker",
    "rank",
    "leg",
    "target",
    "order",
    "position",
}
_COVERAGE_FIELDS = {
    "formation_horizon_cells",
    "below_floor_cells",
    "admitted_cohorts",
    "excluded_selected_path_cohorts",
    "modeled_constituent_paths",
    "atomic_checkpoints",
    "replication_projection_observations",
}

_REQUIRED_BLOCKERS = {
    "borrow_financing_capacity_evidence_missing",
    "cash_distribution_semantics_source_missing",
    "daily_mark_to_market_path_not_implemented",
    "dividend_payment_dates_missing",
    "independent_event_driven_money_path_reconciliation_missing",
    "modeled_split_normalized_quantities_are_not_broker_orders",
    "observed_broker_execution_evidence_missing",
    "pooled_cohort_accounting_not_full_eight_cell_family",
}


class PeadExecutionLedgerError(ValueError):
    """The modeled PEAD ledger is malformed or cannot be substantiated."""


def _exact_fields(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadExecutionLedgerError(
            f"{label} fields differ: expected {sorted(expected)}, got {actual}"
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadExecutionLedgerError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadExecutionLedgerError(f"{label} must be non-empty canonical text")
    return value


def _iso_date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PeadExecutionLedgerError(f"{label} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadExecutionLedgerError(
            f"{label} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise PeadExecutionLedgerError(f"{label} must be a canonical ISO date")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PeadExecutionLedgerError(
            f"{label} must be an integer of at least {minimum}"
        )
    return value


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, Real, Decimal)):
        raise PeadExecutionLedgerError(f"{label} must be a finite decimal")
    if isinstance(value, float) and not math.isfinite(value):
        raise PeadExecutionLedgerError(f"{label} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PeadExecutionLedgerError(f"{label} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        qualifier = "positive " if positive else ""
        raise PeadExecutionLedgerError(f"{label} must be a finite {qualifier}decimal")
    return parsed


def _fixed(value: Decimal, *, quantity: bool = False) -> str:
    quantum = QUANTITY_QUANTUM if quantity else MONEY_QUANTUM
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        normalized = value.quantize(quantum)
    if normalized == 0:
        normalized = abs(normalized)
    return format(normalized, "f")


def _fixed_decimal(value: Any, label: str, *, quantity: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise PeadExecutionLedgerError(f"{label} must be a fixed-scale decimal string")
    parsed = _decimal(value, label)
    if _fixed(parsed, quantity=quantity) != value:
        raise PeadExecutionLedgerError(f"{label} is not a canonical fixed-scale decimal")
    return parsed


def _plain(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except PeadEconomicEvidenceError as exc:
        raise PeadExecutionLedgerError("ledger input is not strict JSON") from exc


def source_report_file_sha256(report: Mapping[str, Any]) -> str:
    """Identity of the canonical one-line report file emitted by the PEAD CLI."""
    if not isinstance(report, Mapping):
        raise PeadExecutionLedgerError("source report must be an object")
    try:
        encoded = (canonical_json(report) + "\n").encode("utf-8")
    except PeadEconomicEvidenceError as exc:
        raise PeadExecutionLedgerError("source report is not strict JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _verified_wrapper(value: Any, label: str) -> Mapping[str, Any]:
    wrapper = _exact_fields(value, _WRAPPER_FIELDS, label)
    artifact_hash = _sha256(wrapper["artifact_hash"], f"{label} artifact hash")
    payload = wrapper["payload"]
    if not isinstance(payload, Mapping) or content_hash(payload) != artifact_hash:
        raise PeadExecutionLedgerError(f"{label} artifact hash mismatch")
    return wrapper


def _validate_report_bindings(
    report: Mapping[str, Any], claimed_report_sha256: str | None
) -> tuple[dict[str, str], Mapping[str, Any], Mapping[str, Any]]:
    if report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION:
        raise PeadExecutionLedgerError("unsupported PEAD source report schema")
    if report.get("candidate_id") != CANDIDATE_ID:
        raise PeadExecutionLedgerError("source report belongs to another candidate")
    actual_report_hash = source_report_file_sha256(report)
    if claimed_report_sha256 is not None and (
        _sha256(claimed_report_sha256, "source report file hash")
        != actual_report_hash
    ):
        raise PeadExecutionLedgerError("source report file hash mismatch")

    combined = _verified_wrapper(
        report.get("combined_data_snapshot"), "combined data snapshot"
    )
    manifest = _verified_wrapper(
        report.get("research_manifest_binding"), "research manifest binding"
    )
    economic = _verified_wrapper(
        report.get("economic_return_inputs"), "economic return inputs"
    )
    economic_payload = _exact_fields(
        economic["payload"],
        {
            "schema_version",
            "candidate_id",
            "combined_data_snapshot_hash",
            "cash_distribution_semantics",
            "terminal_settlement_ledger",
        },
        "economic return inputs payload",
    )
    if economic_payload["schema_version"] != ECONOMIC_INPUT_SCHEMA_VERSION:
        raise PeadExecutionLedgerError("unsupported economic return input schema")
    if economic_payload["candidate_id"] != CANDIDATE_ID:
        raise PeadExecutionLedgerError("economic return inputs belong elsewhere")
    if economic_payload["combined_data_snapshot_hash"] != combined["artifact_hash"]:
        raise PeadExecutionLedgerError(
            "economic return inputs do not bind the combined data snapshot"
        )
    try:
        semantics = validate_cash_distribution_semantics(
            economic_payload["cash_distribution_semantics"]
        )
        terminal = validate_terminal_settlement_ledger(
            economic_payload["terminal_settlement_ledger"]
        )
    except PeadEconomicEvidenceError as exc:
        raise PeadExecutionLedgerError(
            "economic return evidence failed validation"
        ) from exc
    if semantics["payload"]["qualification_allowed"] is not False:
        raise PeadExecutionLedgerError("distribution semantics cannot qualify this ledger")
    if terminal["payload"]["ACTIONS"]["value_allowed"] is not False:
        raise PeadExecutionLedgerError("ACTIONS.value cannot fund terminal settlements")

    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise PeadExecutionLedgerError("source report configuration is missing")
    expected_configuration = {
        "horizons_sessions": list(HORIZONS),
        "minimum_names_per_formation_per_slice": MINIMUM_NAMES,
        "signal_tie_break": "ascending stable m_ticker",
        "cash_distribution_holding_interval": (
            "entry_date_exclusive_exit_date_inclusive"
        ),
        "cash_distribution_reinvestment": False,
        "terminal_payout_from_actions_value_allowed": False,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise PeadExecutionLedgerError(f"source report configuration {key} changed")
    if _decimal(configuration.get("quantile"), "configuration.quantile") != QUANTILE:
        raise PeadExecutionLedgerError("source report quantile changed")
    if _decimal(
        configuration.get("one_way_cost_bps_per_trade_per_leg"),
        "configuration one-way fee",
    ) != Decimal("30"):
        raise PeadExecutionLedgerError("source report one-way fee changed")
    if _decimal(
        configuration.get("fixed_total_round_trip_bps"),
        "configuration total fee",
    ) != Decimal("120"):
        raise PeadExecutionLedgerError("source report round-trip fee changed")

    return (
        {
            "source_report_file_sha256": actual_report_hash,
            "source_report_schema_version": SOURCE_REPORT_SCHEMA_VERSION,
            "combined_data_snapshot_hash": combined["artifact_hash"],
            "economic_return_inputs_hash": economic["artifact_hash"],
            "protocol_hash": manifest["artifact_hash"],
        },
        economic_payload,
        terminal,
    )


def _permatickers(report: Mapping[str, Any]) -> dict[str, int]:
    coverage = report.get("coverage")
    lifecycle = (
        coverage.get("security_lifecycle_diagnostics")
        if isinstance(coverage, Mapping)
        else None
    )
    if not isinstance(lifecycle, Mapping):
        raise PeadExecutionLedgerError("source report omits security lifecycle evidence")
    result: dict[str, int] = {}
    for raw_ticker, evidence in lifecycle.items():
        ticker = _text(raw_ticker, "lifecycle ticker")
        if not isinstance(evidence, Mapping) or evidence.get("status") != "validated":
            raise PeadExecutionLedgerError(
                f"security lifecycle is not validated for {ticker}"
            )
        result[ticker] = _integer(
            evidence.get("permaticker"), f"{ticker} permaticker", minimum=1
        )
    return result


def _normalized_observations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("raw_portfolio_observations")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise PeadExecutionLedgerError("source report has no portfolio observations")
    permatickers = _permatickers(report)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise PeadExecutionLedgerError(f"observation {index} is not an object")
        formation = _iso_date(item.get("formation_date"), "formation_date")
        entry = _iso_date(item.get("entry_date"), "entry_date")
        if entry <= formation:
            raise PeadExecutionLedgerError("entry must follow formation")
        ticker = _text(item.get("ticker"), "ticker")
        m_ticker = _text(item.get("m_ticker"), "m_ticker")
        if ticker not in permatickers:
            raise PeadExecutionLedgerError(f"missing permaticker evidence for {ticker}")
        key = (formation, m_ticker)
        if key in seen:
            raise PeadExecutionLedgerError(
                "duplicate m_ticker on a formation date"
            )
        seen.add(key)
        source_event_key = item.get("source_event_key")
        if not isinstance(source_event_key, Mapping) or not source_event_key:
            raise PeadExecutionLedgerError("observation omits source_event_key")
        observation = {
            "formation_date": formation,
            "entry_date": entry,
            "ticker": ticker,
            "m_ticker": m_ticker,
            "permaticker": permatickers[ticker],
            "signal": _decimal(item.get("signal"), "signal"),
            "entry_price": _decimal(
                item.get("entry_close_split_normalized"),
                "entry split-normalized close",
                positive=True,
            ),
            "source_event_key": _plain(source_event_key),
            "raw": item,
        }
        normalized.append(observation)
    return sorted(
        normalized,
        key=lambda item: (
            item["formation_date"], item["signal"], item["m_ticker"]
        ),
    )


def _path_status(observation: Mapping[str, Any], horizon: int) -> tuple[str, Any]:
    raw = observation["raw"]
    resolution = raw.get(f"economic_return_resolution_{horizon}")
    candidate = raw.get(f"economic_forward_return_candidate_{horizon}")
    target_exit = raw.get(f"target_exit_date_{horizon}")
    if not isinstance(resolution, Mapping) or candidate is None or target_exit is None:
        return "unresolved", resolution
    if resolution.get("status") != "mechanically_reconstructed_nonqualifying":
        return "unresolved", resolution
    if resolution.get("reason") is not None:
        return "unresolved", resolution
    if resolution.get("terminal_settlement_id") is not None:
        return "terminal_path_not_supported", resolution
    if resolution.get("exit_price_split_normalized") is None:
        return "unresolved", resolution
    return "resolved_nonterminal", resolution


def _primary_frozen_selections(report: Mapping[str, Any], horizon: int) -> list[dict]:
    slice_coverage = report.get("slice_coverage")
    pooled = slice_coverage.get("pooled") if isinstance(slice_coverage, Mapping) else None
    horizons = pooled.get("horizons") if isinstance(pooled, Mapping) else None
    horizon_coverage = horizons.get(str(horizon)) if isinstance(horizons, Mapping) else None
    selections = (
        horizon_coverage.get("frozen_selections")
        if isinstance(horizon_coverage, Mapping)
        else None
    )
    if not isinstance(selections, list):
        raise PeadExecutionLedgerError(
            f"source report omits pooled {horizon}-session selections"
        )
    return selections


def _build_selection_manifest(
    report: Mapping[str, Any], observations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[item["formation_date"]].append(item)
    manifests: list[dict[str, Any]] = []
    derived_primary: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in HORIZONS}
    for formation in sorted(grouped):
        group = sorted(grouped[formation], key=lambda row: (row["signal"], row["m_ticker"]))
        count = len(group)
        for horizon in HORIZONS:
            names_per_leg = max(1, int(QUANTILE * count)) if count >= MINIMUM_NAMES else 0
            short = group[:names_per_leg] if names_per_leg else []
            long = group[-names_per_leg:] if names_per_leg else []
            short_names = [item["m_ticker"] for item in short]
            long_names = [item["m_ticker"] for item in long]
            selected = {name: "short" for name in short_names}
            selected.update({name: "long" for name in long_names})
            status = "below_minimum_names"
            reason: str | None = "formation_below_frozen_ten_name_floor"
            if names_per_leg:
                derived_primary[horizon].append(
                    {
                        "date": formation,
                        "eligible_names": count,
                        "names_per_leg": names_per_leg,
                        "short_m_tickers": short_names,
                        "long_m_tickers": long_names,
                    }
                )
                selected_statuses = [
                    _path_status(item, horizon)[0]
                    for item in (*short, *long)
                ]
                if "terminal_path_not_supported" in selected_statuses:
                    status = "excluded_selected_path"
                    reason = "selected_terminal_settlement_path_not_supported"
                elif any(value != "resolved_nonterminal" for value in selected_statuses):
                    status = "excluded_selected_path"
                    reason = "selected_economic_return_path_unresolved"
                else:
                    status = "admitted"
                    reason = None
            ranked = []
            for rank, item in enumerate(group, start=1):
                path_status, resolution = _path_status(item, horizon)
                terminal_id = (
                    resolution.get("terminal_settlement_id")
                    if isinstance(resolution, Mapping)
                    else None
                )
                ranked.append(
                    {
                        "rank": rank,
                        "ticker": item["ticker"],
                        "m_ticker": item["m_ticker"],
                        "permaticker": item["permaticker"],
                        "source_event_key": item["source_event_key"],
                        "signal": _fixed(item["signal"]),
                        "selected_leg": selected.get(item["m_ticker"]),
                        "economic_path_status": path_status,
                        "terminal_settlement_id": terminal_id,
                    }
                )
            manifests.append(
                {
                    "cohort_id": f"pooled:{formation}:{horizon}",
                    "formation_date": formation,
                    "horizon_sessions": horizon,
                    "eligible_names": count,
                    "names_per_leg": names_per_leg,
                    "status": status,
                    "reason": reason,
                    "short_m_tickers": short_names,
                    "long_m_tickers": long_names,
                    "ranked_constituents": ranked,
                }
            )
    for horizon in HORIZONS:
        if _plain(_primary_frozen_selections(report, horizon)) != derived_primary[horizon]:
            raise PeadExecutionLedgerError(
                f"independent pooled {horizon}-session selection differs from source report"
            )
    return manifests


def _economic_path(
    observation: Mapping[str, Any], horizon: int
) -> dict[str, Any]:
    status, resolution = _path_status(observation, horizon)
    if status != "resolved_nonterminal" or not isinstance(resolution, Mapping):
        raise PeadExecutionLedgerError("selected economic path is not modelable")
    raw = observation["raw"]
    entry = observation["entry_price"]
    resolution_entry = _decimal(
        resolution.get("entry_price_split_normalized"), "resolution entry price",
        positive=True,
    )
    if abs(resolution_entry - entry) > SOURCE_TOLERANCE:
        raise PeadExecutionLedgerError("resolution entry price differs from observation")
    exit_price = _decimal(
        resolution.get("exit_price_split_normalized"), "resolution exit price",
        positive=True,
    )
    cash_total = _decimal(resolution.get("cash_total"), "resolution cash total")
    if cash_total < 0:
        raise PeadExecutionLedgerError("resolution cash total cannot be negative")
    gross_terminal = _decimal(
        resolution.get("gross_terminal_value"), "resolution gross terminal value"
    )
    if abs(gross_terminal - (exit_price + cash_total)) > SOURCE_TOLERANCE:
        raise PeadExecutionLedgerError("gross terminal value does not equal close plus cash")
    # ``gross_terminal_value`` is retained as a source cross-check.  The
    # modeled return itself is rebuilt from the two independent components so
    # binary-float addition in the source report cannot leak into ledger P&L.
    reconstructed = (exit_price + cash_total) / entry - INITIAL_NAV
    source_candidate = _decimal(
        raw.get(f"economic_forward_return_candidate_{horizon}"),
        "source economic return candidate",
    )
    resolution_gross = _decimal(
        resolution.get("gross_economic_return"), "resolution gross return"
    )
    if (
        abs(source_candidate - reconstructed) > SOURCE_TOLERANCE
        or abs(resolution_gross - reconstructed) > SOURCE_TOLERANCE
    ):
        raise PeadExecutionLedgerError("source economic return does not reconcile")
    exit_date = _iso_date(
        raw.get(f"target_exit_date_{horizon}"), "target exit date"
    )
    if exit_date < observation["entry_date"]:
        raise PeadExecutionLedgerError("target exit precedes entry")
    distributions = resolution.get("cash_distributions")
    if not isinstance(distributions, list):
        raise PeadExecutionLedgerError("cash distributions must be an array")
    parsed_distributions: list[tuple[str, Decimal, Any]] = []
    distribution_total = Decimal("0")
    seen: set[str] = set()
    for index, distribution in enumerate(distributions):
        if not isinstance(distribution, Mapping):
            raise PeadExecutionLedgerError("cash distribution must be an object")
        distribution_date = _iso_date(
            distribution.get("date"), f"distribution {index} date"
        )
        if not observation["entry_date"] < distribution_date <= exit_date:
            raise PeadExecutionLedgerError(
                "cash distribution is outside the frozen holding interval"
            )
        amount = _decimal(
            distribution.get("amount"), f"distribution {index} amount", positive=True
        )
        action_key = distribution.get("action_key")
        if not isinstance(action_key, Mapping) or not action_key:
            raise PeadExecutionLedgerError("distribution omits action_key")
        token = canonical_json(action_key)
        if token in seen:
            raise PeadExecutionLedgerError("economic path repeats a distribution")
        seen.add(token)
        distribution_total += amount
        parsed_distributions.append((distribution_date, amount, _plain(action_key)))
    if abs(distribution_total - cash_total) > SOURCE_TOLERANCE:
        raise PeadExecutionLedgerError("distribution rows do not sum to cash_total")
    return {
        "entry": entry,
        "exit": exit_price,
        "cash_total": cash_total,
        "source_candidate": source_candidate,
        "reconstructed": reconstructed,
        "exit_date": exit_date,
        "distributions": parsed_distributions,
    }


def _build_constituents(
    observations: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (item["formation_date"], item["m_ticker"]): item for item in observations
    }
    result: list[dict[str, Any]] = []
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        for manifest in manifests:
            if manifest["status"] != "admitted":
                continue
            k = manifest["names_per_leg"]
            unsigned_notional = LEG_GROSS / Decimal(k)
            fee = ONE_WAY_FEE_RATE / Decimal(k)
            selected = [
                ("short", name) for name in manifest["short_m_tickers"]
            ] + [("long", name) for name in manifest["long_m_tickers"]]
            ranks = {
                row["m_ticker"]: row["rank"]
                for row in manifest["ranked_constituents"]
            }
            for leg, m_ticker in selected:
                observation = by_key[(manifest["formation_date"], m_ticker)]
                path = _economic_path(observation, manifest["horizon_sessions"])
                sign = Decimal("1") if leg == "long" else Decimal("-1")
                target = sign * unsigned_notional
                quantity = target / path["entry"]
                price_pnl = quantity * (path["exit"] - path["entry"])
                distribution_pnl = quantity * path["cash_total"]
                net_pnl = price_pnl + distribution_pnl - fee - fee
                accruals = [
                    {
                        "date": distribution_date,
                        "amount_per_split_normalized_share": _fixed(amount),
                        "signed_accrual_pnl": _fixed(quantity * amount),
                        "action_key": action_key,
                    }
                    for distribution_date, amount, action_key in path["distributions"]
                ]
                result.append(
                    {
                        "cohort_id": manifest["cohort_id"],
                        "formation_date": observation["formation_date"],
                        "entry_date": observation["entry_date"],
                        "exit_date": path["exit_date"],
                        "horizon_sessions": manifest["horizon_sessions"],
                        "ticker": observation["ticker"],
                        "m_ticker": m_ticker,
                        "permaticker": observation["permaticker"],
                        "source_event_key": observation["source_event_key"],
                        "rank": ranks[m_ticker],
                        "leg": leg,
                        "signal": _fixed(observation["signal"]),
                        "entry_price_split_normalized": _fixed(path["entry"]),
                        "exit_price_split_normalized": _fixed(path["exit"]),
                        "cash_total_per_split_normalized_share": _fixed(
                            path["cash_total"]
                        ),
                        "distribution_accruals": accruals,
                        "source_candidate_gross_economic_return": _fixed(
                            path["source_candidate"]
                        ),
                        "reconstructed_gross_economic_return": _fixed(
                            path["reconstructed"]
                        ),
                        "signed_target_notional": _fixed(target),
                        "signed_split_normalized_share_equivalent_quantity": _fixed(
                            quantity, quantity=True
                        ),
                        "price_pnl": _fixed(price_pnl),
                        "distribution_pnl": _fixed(distribution_pnl),
                        "entry_fee": _fixed(fee),
                        "exit_fee": _fixed(fee),
                        "total_fees": _fixed(fee + fee),
                        "net_pnl": _fixed(net_pnl),
                    }
                )
    return sorted(
        result,
        key=lambda row: (
            row["formation_date"],
            row["horizon_sessions"],
            0 if row["leg"] == "short" else 1,
            row["rank"],
            row["m_ticker"],
        ),
    )


def _cohort_products(
    manifests: Sequence[dict[str, Any]], constituents: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in constituents:
        grouped[row["cohort_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for manifest in manifests:
        if manifest["status"] != "admitted":
            continue
        rows = grouped[manifest["cohort_id"]]
        k = manifest["names_per_leg"]
        if len(rows) != 2 * k:
            raise PeadExecutionLedgerError("admitted cohort is not exhaustive")
        long_returns = [
            _fixed_decimal(row["reconstructed_gross_economic_return"], "long return")
            for row in rows
            if row["leg"] == "long"
        ]
        short_returns = [
            _fixed_decimal(row["reconstructed_gross_economic_return"], "short return")
            for row in rows
            if row["leg"] == "short"
        ]
        ledger_pnl = sum(
            (_fixed_decimal(row["net_pnl"], "constituent net pnl") for row in rows),
            Decimal("0"),
        )
        gross = sum(long_returns, Decimal("0")) / Decimal(k) - (
            sum(short_returns, Decimal("0")) / Decimal(k)
        )
        net = gross - TOTAL_COHORT_FEE
        difference = ledger_pnl - net
        if abs(difference) > INTERNAL_SUMMARY_TOLERANCE:
            raise PeadExecutionLedgerError("cohort P&L does not reconcile to factor return")
        entry_dates = {row["entry_date"] for row in rows}
        exit_dates = {row["exit_date"] for row in rows}
        if len(entry_dates) != 1 or len(exit_dates) != 1:
            raise PeadExecutionLedgerError("cohort constituents do not share dates")
        entry_date = next(iter(entry_dates))
        exit_date = next(iter(exit_dates))
        terminal = INITIAL_NAV + ledger_pnl
        summary = {
            "cohort_id": manifest["cohort_id"],
            "formation_date": manifest["formation_date"],
            "entry_date": entry_date,
            "exit_date": exit_date,
            "horizon_sessions": manifest["horizon_sessions"],
            "names_per_leg": k,
            "constituent_count": len(rows),
            "initial_nav": _fixed(INITIAL_NAV),
            "long_gross_notional": _fixed(LEG_GROSS),
            "short_gross_notional": _fixed(LEG_GROSS),
            "gross_target_notional": _fixed(LEG_GROSS + LEG_GROSS),
            "net_target_notional": _fixed(Decimal("0")),
            "entry_cash_after_modeled_fees": _fixed(
                INITIAL_NAV - Decimal("0.006")
            ),
            "entry_fees": _fixed(Decimal("0.006")),
            "gross_factor_return": _fixed(gross),
            "total_fees": _fixed(TOTAL_COHORT_FEE),
            "net_factor_return": _fixed(net),
            "ledger_net_pnl": _fixed(ledger_pnl),
            "terminal_cash": _fixed(terminal),
            "terminal_nav": _fixed(terminal),
            "terminal_open_position_count": 0,
            "return_identity_difference": _fixed(difference),
        }
        summaries.append(summary)

        def states(checkpoint: str) -> list[dict[str, Any]]:
            values: list[dict[str, Any]] = []
            for row in sorted(rows, key=lambda item: (item["rank"], item["m_ticker"])):
                target = row["signed_target_notional"]
                quantity = row[
                    "signed_split_normalized_share_equivalent_quantity"
                ]
                if checkpoint == "formation_target":
                    order, position = _fixed(Decimal("0"), quantity=True), _fixed(
                        Decimal("0"), quantity=True
                    )
                elif checkpoint == "modeled_entry":
                    order = quantity
                    position = quantity
                else:
                    order = _fixed(-_fixed_decimal(
                        quantity, "exit quantity", quantity=True
                    ), quantity=True)
                    position = _fixed(Decimal("0"), quantity=True)
                    target = _fixed(Decimal("0"))
                values.append(
                    {
                        "ticker": row["ticker"],
                        "m_ticker": row["m_ticker"],
                        "permaticker": row["permaticker"],
                        "rank": row["rank"],
                        "leg": row["leg"],
                        "target": target,
                        "order": order,
                        "position": position,
                    }
                )
            return values

        checkpoint_specs = [
            (0, "formation_target", manifest["formation_date"], INITIAL_NAV, Decimal("0"), Decimal("0")),
            (1, "modeled_entry", entry_date, INITIAL_NAV - Decimal("0.006"), Decimal("0.006"), Decimal("-0.006")),
            (2, "modeled_exit", exit_date, terminal, TOTAL_COHORT_FEE, ledger_pnl),
        ]
        for sequence, name, checkpoint_date, cash, fees, pnl in checkpoint_specs:
            checkpoints.append(
                {
                    "cohort_id": manifest["cohort_id"],
                    "sequence": sequence,
                    "checkpoint": name,
                    "date": checkpoint_date,
                    "cash": _fixed(cash),
                    "fees": _fixed(fees),
                    "pnl": _fixed(pnl),
                    "state_rows": states(name),
                }
            )
    return summaries, checkpoints


def _protocol() -> dict[str, Any]:
    return {
        "slice": "pooled_only",
        "horizons_sessions": list(HORIZONS),
        "minimum_names_per_formation": MINIMUM_NAMES,
        "quantile": _fixed(QUANTILE),
        "signal_tie_break": "ascending_signal_then_stable_m_ticker",
        "rank_definition": "one_based_ascending_full_formation_cross_section",
        "initial_nav_per_independent_cohort": _fixed(INITIAL_NAV),
        "long_gross": _fixed(LEG_GROSS),
        "short_gross": _fixed(LEG_GROSS),
        "one_way_fee_rate_per_trade_per_leg": _fixed(ONE_WAY_FEE_RATE),
        "fixed_total_round_trip_fee": _fixed(TOTAL_COHORT_FEE),
        "quantity_basis": (
            "split_normalized_share_equivalent_accounting_units_not_broker_shares"
        ),
        "cash_distribution_holding_interval": (
            "entry_date_exclusive_exit_date_inclusive"
        ),
        "cash_distribution_reinvestment": False,
        "distribution_treatment": (
            "holding_period_accrual_at_exit_not_a_claim_about_payment_date"
        ),
        "terminal_payout_from_actions_value_allowed": False,
        "capital_sharing": (
            "none_each_formation_horizon_is_an_independent_normalized_subledger"
        ),
        "cash_yield": "not_modeled",
        "financing": "not_modeled",
        "borrow": "not_modeled",
        "margin_and_short_proceeds": "not_modeled_no_capital_constraint_claim",
        "replication_projection_units": {
            "target": "initial_nav_fraction",
            "order": "split_normalized_share_equivalent",
            "position": "split_normalized_share_equivalent",
            "cash": "initial_nav_units",
            "fees": "initial_nav_units_cumulative",
            "pnl": "initial_nav_units_cumulative",
        },
        "checkpoint_state_semantics": (
            "one_atomic_cohort_state_repeated_for_each_constituent_projection_row"
        ),
    }


def _coverage(
    manifests: Sequence[Mapping[str, Any]],
    constituents: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    projection_count = sum(len(checkpoint["state_rows"]) for checkpoint in checkpoints)
    return {
        "formation_horizon_cells": len(manifests),
        "below_floor_cells": sum(
            item["status"] == "below_minimum_names" for item in manifests
        ),
        "admitted_cohorts": sum(item["status"] == "admitted" for item in manifests),
        "excluded_selected_path_cohorts": sum(
            item["status"] == "excluded_selected_path" for item in manifests
        ),
        "modeled_constituent_paths": len(constituents),
        "atomic_checkpoints": len(checkpoints),
        "replication_projection_observations": projection_count,
    }


def _build_pead_execution_ledger(
    report: Mapping[str, Any], *, report_sha256: str | None = None
) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise PeadExecutionLedgerError("source report must be an object")
    bindings, _, _ = _validate_report_bindings(report, report_sha256)
    observations = _normalized_observations(report)
    manifests = _build_selection_manifest(report, observations)
    constituents = _build_constituents(observations, manifests)
    summaries, checkpoints = _cohort_products(manifests, constituents)
    source_blockers = report.get("blockers")
    if not isinstance(source_blockers, list) or any(
        not isinstance(item, str) or not item for item in source_blockers
    ):
        raise PeadExecutionLedgerError("source report blockers are malformed")
    blockers = sorted(set(source_blockers) | _REQUIRED_BLOCKERS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "deterministic_modeled_cohort_accounting_nonqualifying",
        "qualifying_evidence": False,
        "replication_evidence_eligible": False,
        "paper_execution_evidence": False,
        "promotion_allowed": False,
        "modeled_execution_claim": {
            "claim": "modeled_accounting_only_not_observed_or_routeable_execution",
            "broker": None,
            "account_id": None,
            "broker_order_ids": None,
            "broker_fill_ids": None,
            "observed_quotes": None,
        },
        "bindings": bindings,
        "frozen_protocol": _protocol(),
        "decimal_policy": {
            "precision": DECIMAL_PRECISION,
            "rounding": "ROUND_HALF_EVEN",
            "money_quantum": format(MONEY_QUANTUM, "f"),
            "quantity_quantum": format(QUANTITY_QUANTUM, "f"),
            "rounding_stage": "serialization_only_all_equations_use_precision_50",
        },
        "selection_manifest": manifests,
        "constituent_ledger": constituents,
        "cohort_summaries": summaries,
        "atomic_checkpoints": checkpoints,
        "coverage": _coverage(manifests, constituents, checkpoints),
        "blockers": blockers,
    }
    normalized = _plain(payload)
    artifact = {"artifact_hash": content_hash(normalized), "payload": normalized}
    _validate_pead_execution_ledger(artifact)
    return artifact


def build_pead_execution_ledger(
    report: Mapping[str, Any], *, report_sha256: str | None = None
) -> dict[str, Any]:
    """Build the content-addressed pooled modeled-accounting artifact."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return _build_pead_execution_ledger(
            report, report_sha256=report_sha256
        )


def _validate_manifest(manifests: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(manifests, list) or not manifests:
        raise PeadExecutionLedgerError("selection manifest must be nonempty")
    result: dict[str, Mapping[str, Any]] = {}
    tokens: list[tuple[str, int]] = []
    for manifest in manifests:
        item = _exact_fields(manifest, _MANIFEST_FIELDS, "selection manifest item")
        cohort_id = _text(item["cohort_id"], "cohort_id")
        formation = _iso_date(item["formation_date"], "manifest formation date")
        horizon = _integer(item["horizon_sessions"], "manifest horizon", minimum=1)
        if horizon not in HORIZONS or cohort_id != f"pooled:{formation}:{horizon}":
            raise PeadExecutionLedgerError("manifest cohort identity is invalid")
        if cohort_id in result:
            raise PeadExecutionLedgerError("selection manifest repeats a cohort")
        eligible = _integer(item["eligible_names"], "eligible names", minimum=1)
        k = _integer(item["names_per_leg"], "names per leg")
        status = item["status"]
        if status not in {"below_minimum_names", "admitted", "excluded_selected_path"}:
            raise PeadExecutionLedgerError("manifest status is invalid")
        if status == "below_minimum_names":
            if eligible >= MINIMUM_NAMES or k != 0 or item["reason"] != (
                "formation_below_frozen_ten_name_floor"
            ):
                raise PeadExecutionLedgerError("below-floor manifest is inconsistent")
        else:
            expected_k = max(1, int(QUANTILE * eligible))
            if eligible < MINIMUM_NAMES or k != expected_k:
                raise PeadExecutionLedgerError("selected manifest has the wrong leg size")
            if status == "admitted" and item["reason"] is not None:
                raise PeadExecutionLedgerError("admitted manifest has an exclusion reason")
        ranked = item["ranked_constituents"]
        if not isinstance(ranked, list) or len(ranked) != eligible:
            raise PeadExecutionLedgerError("ranked manifest is not exhaustive")
        observed_order: list[tuple[Decimal, str]] = []
        short: list[str] = []
        long: list[str] = []
        seen_names: set[str] = set()
        for expected_rank, raw_row in enumerate(ranked, start=1):
            row = _exact_fields(raw_row, _RANKED_FIELDS, "ranked constituent")
            if row["rank"] != expected_rank:
                raise PeadExecutionLedgerError("manifest ranks are not contiguous")
            ticker = _text(row["ticker"], "ranked ticker")
            m_ticker = _text(row["m_ticker"], "ranked m_ticker")
            _integer(row["permaticker"], "ranked permaticker", minimum=1)
            if not isinstance(row["source_event_key"], Mapping) or not row[
                "source_event_key"
            ]:
                raise PeadExecutionLedgerError(
                    "ranked constituent omits source_event_key"
                )
            signal = _fixed_decimal(row["signal"], "ranked signal")
            if m_ticker in seen_names:
                raise PeadExecutionLedgerError("manifest repeats m_ticker")
            seen_names.add(m_ticker)
            observed_order.append((signal, m_ticker))
            leg = row["selected_leg"]
            if leg not in {None, "short", "long"}:
                raise PeadExecutionLedgerError("ranked selection leg is invalid")
            if leg == "short":
                short.append(m_ticker)
            elif leg == "long":
                long.append(m_ticker)
            path_status = row["economic_path_status"]
            if path_status not in {
                "resolved_nonterminal",
                "unresolved",
                "terminal_path_not_supported",
            }:
                raise PeadExecutionLedgerError("economic path status is invalid")
            terminal_id = row["terminal_settlement_id"]
            if path_status == "terminal_path_not_supported":
                _text(terminal_id, "terminal settlement id")
            elif terminal_id is not None:
                raise PeadExecutionLedgerError(
                    "nonterminal ranked path has a terminal settlement id"
                )
            if not ticker:
                raise PeadExecutionLedgerError("ranked ticker is empty")
        if observed_order != sorted(observed_order):
            raise PeadExecutionLedgerError("manifest is not sorted by signal and m_ticker")
        if short != item["short_m_tickers"] or long != item["long_m_tickers"]:
            raise PeadExecutionLedgerError("manifest leg lists differ from ranked rows")
        if k and (short != [row[1] for row in observed_order[:k]] or long != [
            row[1] for row in observed_order[-k:]
        ]):
            raise PeadExecutionLedgerError("manifest legs do not use frozen tails")
        if not k and (short or long):
            raise PeadExecutionLedgerError("below-floor manifest selects names")
        if k:
            selected_rows = [
                row for row in ranked if row["selected_leg"] is not None
            ]
            if any(
                row["economic_path_status"] == "terminal_path_not_supported"
                for row in selected_rows
            ):
                expected_status = "excluded_selected_path"
                expected_reason = "selected_terminal_settlement_path_not_supported"
            elif any(
                row["economic_path_status"] != "resolved_nonterminal"
                for row in selected_rows
            ):
                expected_status = "excluded_selected_path"
                expected_reason = "selected_economic_return_path_unresolved"
            else:
                expected_status = "admitted"
                expected_reason = None
            if status != expected_status or item["reason"] != expected_reason:
                raise PeadExecutionLedgerError(
                    "manifest admission differs from selected path evidence"
                )
        result[cohort_id] = item
        tokens.append((formation, horizon))
    if tokens != sorted(tokens) or len(tokens) != len(set(tokens)):
        raise PeadExecutionLedgerError("selection manifest order is not canonical")
    return result


def _validate_constituents(
    value: Any, manifests: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    if not isinstance(value, list):
        raise PeadExecutionLedgerError("constituent ledger must be an array")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for raw in value:
        row = _exact_fields(raw, _CONSTITUENT_FIELDS, "constituent ledger row")
        cohort_id = _text(row["cohort_id"], "constituent cohort_id")
        if cohort_id not in manifests or manifests[cohort_id]["status"] != "admitted":
            raise PeadExecutionLedgerError("constituent belongs to a non-admitted cohort")
        manifest = manifests[cohort_id]
        if (
            row["formation_date"] != manifest["formation_date"]
            or row["horizon_sessions"] != manifest["horizon_sessions"]
        ):
            raise PeadExecutionLedgerError("constituent cohort fields mismatch")
        _iso_date(row["entry_date"], "constituent entry date")
        _iso_date(row["exit_date"], "constituent exit date")
        m_ticker = _text(row["m_ticker"], "constituent m_ticker")
        ticker = _text(row["ticker"], "constituent ticker")
        permaticker = _integer(
            row["permaticker"], "constituent permaticker", minimum=1
        )
        rank = _integer(row["rank"], "constituent rank", minimum=1)
        leg = row["leg"]
        if leg not in {"short", "long"}:
            raise PeadExecutionLedgerError("constituent leg is invalid")
        ranked = {item["m_ticker"]: item for item in manifest["ranked_constituents"]}
        if not isinstance(row["source_event_key"], Mapping) or not row[
            "source_event_key"
        ]:
            raise PeadExecutionLedgerError(
                "constituent omits source_event_key provenance"
            )
        if m_ticker not in ranked or ranked[m_ticker]["rank"] != rank or (
            ranked[m_ticker]["selected_leg"] != leg
        ):
            raise PeadExecutionLedgerError("constituent differs from frozen selection")
        if (
            ranked[m_ticker]["ticker"] != ticker
            or ranked[m_ticker]["permaticker"] != permaticker
            or canonical_json(ranked[m_ticker]["source_event_key"])
            != canonical_json(row["source_event_key"])
        ):
            raise PeadExecutionLedgerError(
                "constituent provenance differs from frozen selection"
            )
        key = (cohort_id, m_ticker)
        if key in seen:
            raise PeadExecutionLedgerError("constituent ledger repeats a selected name")
        seen.add(key)
        signal = _fixed_decimal(row["signal"], "constituent signal")
        if signal != _fixed_decimal(ranked[m_ticker]["signal"], "ranked signal"):
            raise PeadExecutionLedgerError("constituent signal differs from manifest")
        entry = _fixed_decimal(row["entry_price_split_normalized"], "entry price")
        exit_price = _fixed_decimal(row["exit_price_split_normalized"], "exit price")
        cash_total = _fixed_decimal(
            row["cash_total_per_split_normalized_share"], "cash total"
        )
        if entry <= 0 or exit_price <= 0 or cash_total < 0:
            raise PeadExecutionLedgerError("constituent prices or cash are invalid")
        source_return = _fixed_decimal(
            row["source_candidate_gross_economic_return"], "source return"
        )
        reconstructed = _fixed_decimal(
            row["reconstructed_gross_economic_return"], "reconstructed return"
        )
        expected_return = (exit_price + cash_total) / entry - INITIAL_NAV
        if (
            abs(reconstructed - expected_return) > INTERNAL_MONEY_TOLERANCE
            or abs(source_return - expected_return) > SOURCE_TOLERANCE
        ):
            raise PeadExecutionLedgerError("constituent economic return is inconsistent")
        target = _fixed_decimal(row["signed_target_notional"], "target")
        quantity = _fixed_decimal(
            row["signed_split_normalized_share_equivalent_quantity"],
            "quantity",
            quantity=True,
        )
        expected_target = (Decimal("1") if leg == "long" else Decimal("-1")) / (
            Decimal(manifest["names_per_leg"])
        )
        if abs(target - expected_target) > MONEY_QUANTUM:
            raise PeadExecutionLedgerError("constituent target is inconsistent")
        if abs(quantity - expected_target / entry) > QUANTITY_QUANTUM:
            raise PeadExecutionLedgerError("constituent quantity is inconsistent")
        accruals = row["distribution_accruals"]
        if not isinstance(accruals, list):
            raise PeadExecutionLedgerError("distribution accruals must be an array")
        accrual_total = Decimal("0")
        signed_accrual_total = Decimal("0")
        for raw_accrual in accruals:
            accrual = _exact_fields(
                raw_accrual, _DISTRIBUTION_FIELDS, "distribution accrual"
            )
            accrual_date = _iso_date(accrual["date"], "distribution accrual date")
            if not row["entry_date"] < accrual_date <= row["exit_date"]:
                raise PeadExecutionLedgerError("distribution accrual is out of range")
            amount = _fixed_decimal(
                accrual["amount_per_split_normalized_share"], "distribution amount"
            )
            signed = _fixed_decimal(
                accrual["signed_accrual_pnl"], "distribution signed accrual"
            )
            if (
                amount <= 0
                or abs(signed - quantity * amount) > INTERNAL_MONEY_TOLERANCE
            ):
                raise PeadExecutionLedgerError("distribution accrual is inconsistent")
            if not isinstance(accrual["action_key"], Mapping) or not accrual["action_key"]:
                raise PeadExecutionLedgerError("distribution accrual omits action key")
            accrual_total += amount
            signed_accrual_total += signed
        if abs(accrual_total - cash_total) > INTERNAL_MONEY_TOLERANCE:
            raise PeadExecutionLedgerError("distribution accruals do not sum to cash")
        price_pnl = _fixed_decimal(row["price_pnl"], "price pnl")
        distribution_pnl = _fixed_decimal(row["distribution_pnl"], "distribution pnl")
        entry_fee = _fixed_decimal(row["entry_fee"], "entry fee")
        exit_fee = _fixed_decimal(row["exit_fee"], "exit fee")
        total_fees = _fixed_decimal(row["total_fees"], "total fees")
        net_pnl = _fixed_decimal(row["net_pnl"], "net pnl")
        expected_fee = ONE_WAY_FEE_RATE / Decimal(manifest["names_per_leg"])
        if (
            abs(entry_fee - expected_fee) > MONEY_QUANTUM
            or abs(exit_fee - expected_fee) > MONEY_QUANTUM
            or abs(total_fees - entry_fee - exit_fee) > MONEY_QUANTUM
            or abs(price_pnl - quantity * (exit_price - entry))
            > INTERNAL_MONEY_TOLERANCE
            or abs(distribution_pnl - signed_accrual_total)
            > INTERNAL_MONEY_TOLERANCE
            or abs(net_pnl - price_pnl - distribution_pnl + total_fees)
            > INTERNAL_MONEY_TOLERANCE
        ):
            raise PeadExecutionLedgerError("constituent money equation is inconsistent")
        grouped[cohort_id].append(row)
    for cohort_id, manifest in manifests.items():
        expected = 2 * manifest["names_per_leg"] if manifest["status"] == "admitted" else 0
        if len(grouped.get(cohort_id, [])) != expected:
            raise PeadExecutionLedgerError("constituent coverage is incomplete")
    return grouped


def _validate_summaries_and_checkpoints(
    payload: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    grouped: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    summaries = payload["cohort_summaries"]
    checkpoints = payload["atomic_checkpoints"]
    if not isinstance(summaries, list) or not isinstance(checkpoints, list):
        raise PeadExecutionLedgerError("summaries and checkpoints must be arrays")
    summary_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in summaries:
        summary = _exact_fields(raw, _SUMMARY_FIELDS, "cohort summary")
        cohort_id = _text(summary["cohort_id"], "summary cohort_id")
        if cohort_id in summary_by_id or cohort_id not in grouped:
            raise PeadExecutionLedgerError("summary cohort coverage is invalid")
        rows = grouped[cohort_id]
        manifest = manifests[cohort_id]
        if not rows:
            raise PeadExecutionLedgerError("summary belongs to an empty cohort")
        if (
            summary["formation_date"] != manifest["formation_date"]
            or summary["horizon_sessions"] != manifest["horizon_sessions"]
            or summary["names_per_leg"] != manifest["names_per_leg"]
            or summary["constituent_count"] != len(rows)
        ):
            raise PeadExecutionLedgerError("summary cohort fields are inconsistent")
        k = Decimal(manifest["names_per_leg"])
        long_returns = [
            _fixed_decimal(row["reconstructed_gross_economic_return"], "long return")
            for row in rows if row["leg"] == "long"
        ]
        short_returns = [
            _fixed_decimal(row["reconstructed_gross_economic_return"], "short return")
            for row in rows if row["leg"] == "short"
        ]
        ledger = sum(
            (_fixed_decimal(row["net_pnl"], "net pnl") for row in rows), Decimal("0")
        )
        gross = sum(long_returns, Decimal("0")) / k - sum(
            short_returns, Decimal("0")
        ) / k
        expected = {
            "initial_nav": INITIAL_NAV,
            "long_gross_notional": LEG_GROSS,
            "short_gross_notional": LEG_GROSS,
            "gross_target_notional": LEG_GROSS + LEG_GROSS,
            "net_target_notional": Decimal("0"),
            "entry_cash_after_modeled_fees": Decimal("0.994"),
            "entry_fees": Decimal("0.006"),
            "gross_factor_return": gross,
            "total_fees": TOTAL_COHORT_FEE,
            "net_factor_return": gross - TOTAL_COHORT_FEE,
            "ledger_net_pnl": ledger,
            "terminal_cash": INITIAL_NAV + ledger,
            "terminal_nav": INITIAL_NAV + ledger,
            "return_identity_difference": ledger - (gross - TOTAL_COHORT_FEE),
        }
        if abs(expected["return_identity_difference"]) > INTERNAL_SUMMARY_TOLERANCE:
            raise PeadExecutionLedgerError(
                "summary ledger P&L does not reconcile to factor return"
            )
        for field, value in expected.items():
            actual = _fixed_decimal(summary[field], f"summary {field}")
            if abs(actual - value) > INTERNAL_SUMMARY_TOLERANCE:
                raise PeadExecutionLedgerError(f"summary {field} is inconsistent")
        if summary["entry_date"] != rows[0]["entry_date"] or summary["exit_date"] != rows[0]["exit_date"]:
            raise PeadExecutionLedgerError("summary dates are inconsistent")
        if summary["terminal_open_position_count"] != 0:
            raise PeadExecutionLedgerError("summary does not prove a flat terminal book")
        summary_by_id[cohort_id] = summary
    admitted = {key for key, item in manifests.items() if item["status"] == "admitted"}
    if set(summary_by_id) != admitted:
        raise PeadExecutionLedgerError("summary coverage is not exhaustive")

    checkpoints_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in checkpoints:
        checkpoint = _exact_fields(raw, _CHECKPOINT_FIELDS, "atomic checkpoint")
        cohort_id = _text(checkpoint["cohort_id"], "checkpoint cohort_id")
        if cohort_id not in summary_by_id:
            raise PeadExecutionLedgerError("checkpoint belongs to an unknown cohort")
        checkpoints_by_id[cohort_id].append(checkpoint)
    for cohort_id, summary in summary_by_id.items():
        rows = grouped[cohort_id]
        observed = sorted(checkpoints_by_id.get(cohort_id, []), key=lambda item: item["sequence"])
        if [item["sequence"] for item in observed] != [0, 1, 2] or [
            item["checkpoint"] for item in observed
        ] != ["formation_target", "modeled_entry", "modeled_exit"]:
            raise PeadExecutionLedgerError("checkpoint progression is incomplete")
        expected_dates = [summary["formation_date"], summary["entry_date"], summary["exit_date"]]
        expected_state = [
            (INITIAL_NAV, Decimal("0"), Decimal("0")),
            (Decimal("0.994"), Decimal("0.006"), Decimal("-0.006")),
            (
                _fixed_decimal(summary["terminal_cash"], "terminal cash"),
                TOTAL_COHORT_FEE,
                _fixed_decimal(summary["ledger_net_pnl"], "ledger pnl"),
            ),
        ]
        row_by_name = {row["m_ticker"]: row for row in rows}
        for index, checkpoint in enumerate(observed):
            if checkpoint["date"] != expected_dates[index]:
                raise PeadExecutionLedgerError("checkpoint date is inconsistent")
            for field, expected_value in zip(("cash", "fees", "pnl"), expected_state[index]):
                if abs(_fixed_decimal(checkpoint[field], f"checkpoint {field}") - expected_value) > MONEY_QUANTUM:
                    raise PeadExecutionLedgerError(f"checkpoint {field} is inconsistent")
            states = checkpoint["state_rows"]
            if not isinstance(states, list) or len(states) != len(rows):
                raise PeadExecutionLedgerError("checkpoint state rows are incomplete")
            if [state.get("m_ticker") for state in states] != [
                row["m_ticker"] for row in sorted(rows, key=lambda item: (item["rank"], item["m_ticker"]))
            ]:
                raise PeadExecutionLedgerError("checkpoint state order is invalid")
            for raw_state in states:
                state = _exact_fields(raw_state, _STATE_FIELDS, "checkpoint state row")
                row = row_by_name.get(state["m_ticker"])
                if row is None or any(
                    state[field] != row[field]
                    for field in ("ticker", "permaticker", "rank", "leg")
                ):
                    raise PeadExecutionLedgerError("checkpoint state identity is invalid")
                target = _fixed_decimal(state["target"], "checkpoint target")
                order = _fixed_decimal(state["order"], "checkpoint order", quantity=True)
                position = _fixed_decimal(state["position"], "checkpoint position", quantity=True)
                row_target = _fixed_decimal(row["signed_target_notional"], "row target")
                row_quantity = _fixed_decimal(
                    row["signed_split_normalized_share_equivalent_quantity"],
                    "row quantity", quantity=True,
                )
                expected_triplet = (
                    (row_target, Decimal("0"), Decimal("0"))
                    if index == 0
                    else (row_target, row_quantity, row_quantity)
                    if index == 1
                    else (Decimal("0"), -row_quantity, Decimal("0"))
                )
                if (
                    abs(target - expected_triplet[0]) > MONEY_QUANTUM
                    or abs(order - expected_triplet[1]) > QUANTITY_QUANTUM
                    or abs(position - expected_triplet[2]) > QUANTITY_QUANTUM
                ):
                    raise PeadExecutionLedgerError("checkpoint state transition is invalid")


def _validate_pead_execution_ledger(
    document: Mapping[str, Any], *, source_report: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    wrapper = _exact_fields(document, _WRAPPER_FIELDS, "modeled execution ledger")
    artifact_hash = _sha256(wrapper["artifact_hash"], "ledger artifact hash")
    payload = _exact_fields(wrapper["payload"], _PAYLOAD_FIELDS, "ledger payload")
    if content_hash(payload) != artifact_hash:
        raise PeadExecutionLedgerError("modeled execution ledger hash mismatch")
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "deterministic_modeled_cohort_accounting_nonqualifying",
        "qualifying_evidence": False,
        "replication_evidence_eligible": False,
        "paper_execution_evidence": False,
        "promotion_allowed": False,
    }
    for field, expected in expected_scalars.items():
        if payload[field] != expected:
            raise PeadExecutionLedgerError(f"ledger {field} is invalid")
    claim = _exact_fields(payload["modeled_execution_claim"], _MODELED_CLAIM_FIELDS, "modeled claim")
    if claim != {
        "claim": "modeled_accounting_only_not_observed_or_routeable_execution",
        "broker": None,
        "account_id": None,
        "broker_order_ids": None,
        "broker_fill_ids": None,
        "observed_quotes": None,
    }:
        raise PeadExecutionLedgerError("modeled execution claim overstates evidence")
    bindings = _exact_fields(payload["bindings"], _BINDING_FIELDS, "ledger bindings")
    for field in bindings:
        if field != "source_report_schema_version":
            _sha256(bindings[field], f"binding {field}")
    if bindings["source_report_schema_version"] != SOURCE_REPORT_SCHEMA_VERSION:
        raise PeadExecutionLedgerError("ledger binds an unsupported report schema")
    if _exact_fields(payload["frozen_protocol"], _PROTOCOL_FIELDS, "frozen protocol") != _protocol():
        raise PeadExecutionLedgerError("ledger frozen protocol changed")
    decimal_policy = _exact_fields(payload["decimal_policy"], _DECIMAL_POLICY_FIELDS, "decimal policy")
    if decimal_policy != {
        "precision": DECIMAL_PRECISION,
        "rounding": "ROUND_HALF_EVEN",
        "money_quantum": format(MONEY_QUANTUM, "f"),
        "quantity_quantum": format(QUANTITY_QUANTUM, "f"),
        "rounding_stage": "serialization_only_all_equations_use_precision_50",
    }:
        raise PeadExecutionLedgerError("ledger decimal policy changed")
    blockers = payload["blockers"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not _REQUIRED_BLOCKERS.issubset(blockers)
    ):
        raise PeadExecutionLedgerError("ledger blockers are incomplete or noncanonical")
    manifests = _validate_manifest(payload["selection_manifest"])
    grouped = _validate_constituents(payload["constituent_ledger"], manifests)
    _validate_summaries_and_checkpoints(payload, manifests, grouped)
    coverage = _exact_fields(payload["coverage"], _COVERAGE_FIELDS, "ledger coverage")
    expected_coverage = _coverage(
        payload["selection_manifest"],
        payload["constituent_ledger"],
        payload["atomic_checkpoints"],
    )
    if coverage != expected_coverage:
        raise PeadExecutionLedgerError("ledger coverage is inconsistent")
    if source_report is not None:
        rebuilt = build_pead_execution_ledger(
            source_report,
            report_sha256=bindings["source_report_file_sha256"],
        )
        if rebuilt != _plain(document):
            raise PeadExecutionLedgerError("ledger differs from source-report rebuild")
    return _plain(document)


def validate_pead_execution_ledger(
    document: Mapping[str, Any], *, source_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every equation and rebuild from the exact bound source report."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return _validate_pead_execution_ledger(
            document, source_report=source_report
        )


def replication_observations(
    document: Mapping[str, Any], *, source_report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Project modeled checkpoints onto the generic reconciliation schema.

    The projection is intentionally primary-only.  It becomes eligible for a
    generic ``ReplicationEvidence`` artifact only after an independently
    written event-driven implementation produces the same exhaustive keys and
    values.
    """
    validated = validate_pead_execution_ledger(
        document, source_report=source_report
    )
    payload = validated["payload"]
    constituents = {
        (row["cohort_id"], row["m_ticker"]): row
        for row in payload["constituent_ledger"]
    }
    result: list[dict[str, Any]] = []
    for checkpoint in payload["atomic_checkpoints"]:
        for state in checkpoint["state_rows"]:
            row = constituents[(checkpoint["cohort_id"], state["m_ticker"])]
            result.append(
                {
                    "key": {
                        "candidate_id": CANDIDATE_ID,
                        "slice": "pooled",
                        "horizon_sessions": row["horizon_sessions"],
                        "formation_date": row["formation_date"],
                        "cohort_id": row["cohort_id"],
                        "checkpoint": checkpoint["checkpoint"],
                        "ticker": row["ticker"],
                        "m_ticker": row["m_ticker"],
                        "permaticker": row["permaticker"],
                    },
                    "eligibility": True,
                    "signal": float(Decimal(row["signal"])),
                    "rank": float(row["rank"]),
                    "target": float(Decimal(state["target"])),
                    "order": float(Decimal(state["order"])),
                    "position": float(Decimal(state["position"])),
                    "cash": float(Decimal(checkpoint["cash"])),
                    "fees": float(Decimal(checkpoint["fees"])),
                    "pnl": float(Decimal(checkpoint["pnl"])),
                }
            )
    return result


__all__ = [
    "CANDIDATE_ID",
    "PeadExecutionLedgerError",
    "SCHEMA_VERSION",
    "build_pead_execution_ledger",
    "replication_observations",
    "source_report_file_sha256",
    "validate_pead_execution_ledger",
]
