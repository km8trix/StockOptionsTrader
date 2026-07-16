"""Independent event-driven daily accounting for the locked PEAD candidate.

This module deliberately does not import the primary PEAD report or execution
ledger implementations.  It starts with the independently reconstructed
portfolio observations, re-ranks every pooled formation cross-section, and
then applies orders, fees, prices, and corporate actions one session at a time.

The result is still modeled, non-qualifying evidence.  ``SEP.close`` is a
split-normalized accounting basis; ACTIONS dividend dates and amounts are only
candidate accrual inputs; payment dates, broker shares, quotes, fills, borrow,
financing, margin, and shared capital are not available.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import json
from typing import Any

from analysis.pead_daily_inputs import verify_pead_daily_input_snapshot
from analysis.pead_reference_replication import verify_reference_artifact
from data.pead_economic_evidence import canonical_json, content_hash


SCHEMA_VERSION = "pead_independent_daily_ledger.v2"
INPUT_SCHEMA_VERSION = "pead_daily_input_snapshot.v1"
CANDIDATE_ID = "pead-vq-locked-replication-v1"

HORIZONS = (21, 63)
MINIMUM_NAMES = 10
QUANTILE = Decimal("0.2")
INITIAL_NAV = Decimal("1")
ONE_WAY_FEE_RATE = Decimal("0.003")
DECIMAL_PRECISION = 50
MONEY_QUANTUM = Decimal("0.000000000000000001")
QUANTITY_QUANTUM = Decimal("0.000000000000000000000001")
SOURCE_TOLERANCE = Decimal("0.000000000001")

_IGNORED_ISSUER_EXTERNAL_ACTIONS = frozenset({"acquisitionof"})
_TERMINAL_ACTIONS = frozenset(
    {
        "acquisitionby",
        "bankruptcyliquidation",
        "delisted",
        "mergerfrom",
        "mergerto",
        "regulatorydelisting",
        "voluntarydelisting",
    }
)
_UNSUPPORTED_HOLDER_ACTIONS = frozenset(
    {
        "adrratiosplit",
        "relation",
        "spinoff",
        "spinoffdividend",
        "spunofffrom",
        "split",
        "tickerchangefrom",
        "tickerchangeto",
    }
)
_REQUIRED_BLOCKERS = frozenset(
    {
        "borrow_financing_capacity_evidence_missing",
        "broker_execution_evidence_missing",
        "cash_distribution_semantics_source_missing",
        "dividend_payment_dates_missing",
        "modeled_close_execution_not_observed",
        "pooled_daily_scope_not_full_eight_cell_family",
        "split_normalized_share_equivalents_not_broker_shares",
    }
)


class PeadIndependentDailyLedgerError(ValueError):
    """An input or state transition violates the independent daily contract."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PeadIndependentDailyLedgerError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PeadIndependentDailyLedgerError(f"{label} must be an array")
    return value


def _text(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadIndependentDailyLedgerError(f"{label} must be canonical text")
    return value


def _iso(value: Any, label: str) -> str:
    parsed = _text(value, label)
    assert isinstance(parsed, str)
    try:
        result = date.fromisoformat(parsed)
    except ValueError as exc:
        raise PeadIndependentDailyLedgerError(f"{label} must be a canonical ISO date") from exc
    if result.isoformat() != parsed:
        raise PeadIndependentDailyLedgerError(f"{label} must be a canonical ISO date")
    return parsed


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PeadIndependentDailyLedgerError(f"{label} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise PeadIndependentDailyLedgerError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PeadIndependentDailyLedgerError(f"{label} must be a finite decimal") from exc
    if not result.is_finite() or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise PeadIndependentDailyLedgerError(f"{label} must be a {qualifier}finite decimal")
    return Decimal("0") if result == 0 else result


def _money(value: Decimal) -> str:
    return format(value.quantize(MONEY_QUANTUM), "f")


def _quantity(value: Decimal) -> str:
    return format(value.quantize(QUANTITY_QUANTUM), "f")


def _reference_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapper = _mapping(document, "independent reference artifact")
    if set(wrapper) != {"artifact_hash", "payload"}:
        raise PeadIndependentDailyLedgerError(
            "independent reference artifact wrapper fields differ"
        )
    payload = _mapping(wrapper["payload"], "independent reference payload")
    if content_hash(payload) != wrapper["artifact_hash"]:
        raise PeadIndependentDailyLedgerError("independent reference artifact hash mismatch")
    try:
        verified = verify_reference_artifact(wrapper)
    except Exception as exc:
        raise PeadIndependentDailyLedgerError("independent reference artifact is invalid") from exc
    if canonical_json(verified) != canonical_json(payload):
        raise PeadIndependentDailyLedgerError(
            "independent reference verification changed the payload"
        )
    return payload


def _input_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapper = _mapping(document, "daily input snapshot")
    if set(wrapper) != {"artifact_hash", "payload"}:
        raise PeadIndependentDailyLedgerError("daily input snapshot wrapper fields differ")
    payload = _mapping(wrapper["payload"], "daily input snapshot payload")
    if content_hash(payload) != wrapper["artifact_hash"]:
        raise PeadIndependentDailyLedgerError("daily input snapshot artifact hash mismatch")
    try:
        verified = verify_pead_daily_input_snapshot(wrapper)
    except Exception as exc:
        raise PeadIndependentDailyLedgerError("daily input snapshot is invalid") from exc
    verified_wrapper = _mapping(verified, "verified daily input snapshot")
    verified_payload = (
        _mapping(verified_wrapper["payload"], "verified daily input payload")
        if "payload" in verified_wrapper
        else verified_wrapper
    )
    if canonical_json(verified_payload) != canonical_json(payload):
        raise PeadIndependentDailyLedgerError("daily input verification changed the payload")
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise PeadIndependentDailyLedgerError("daily input schema is unsupported")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadIndependentDailyLedgerError("daily input belongs to another candidate")
    return payload


def _reference_observations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs = _mapping(payload.get("outputs"), "reference outputs")
    reference = _mapping(outputs.get("reference"), "reference-side outputs")
    raw = _array(
        reference.get("portfolio_observations"),
        "reference portfolio observations",
    )
    if not raw:
        raise PeadIndependentDailyLedgerError("reference portfolio observations are empty")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(raw):
        row = _mapping(value, f"reference observation {index}")
        formation = _iso(row.get("formation_date"), "formation_date")
        entry = _iso(row.get("entry_date"), "entry_date")
        if entry <= formation:
            raise PeadIndependentDailyLedgerError("entry must follow formation")
        ticker = _text(row.get("ticker"), "ticker")
        m_ticker = _text(row.get("m_ticker"), "m_ticker")
        assert isinstance(ticker, str) and isinstance(m_ticker, str)
        key = (formation, m_ticker)
        if key in seen:
            raise PeadIndependentDailyLedgerError(
                "reference repeats an m_ticker within a formation"
            )
        seen.add(key)
        source_key = _mapping(row.get("source_event_key"), "source_event_key")
        if not source_key:
            raise PeadIndependentDailyLedgerError("source_event_key is empty")
        signal = _decimal(row.get("signal"), "signal")
        horizons: dict[int, dict[str, Any]] = {}
        for horizon in HORIZONS:
            exit_date = _iso(
                row.get(f"target_exit_date_{horizon}"),
                f"target_exit_date_{horizon}",
            )
            resolution = _mapping(
                row.get(f"economic_return_resolution_{horizon}"),
                f"economic_return_resolution_{horizon}",
            )
            horizons[horizon] = {
                "exit_date": exit_date,
                "resolution": resolution,
                "candidate": _decimal(
                    row.get(f"economic_forward_return_candidate_{horizon}"),
                    f"economic_forward_return_candidate_{horizon}",
                ),
            }
        result.append(
            {
                "formation_date": formation,
                "entry_date": entry,
                "ticker": ticker,
                "m_ticker": m_ticker,
                "source_event_key": _plain(source_key),
                "signal": signal,
                "horizons": horizons,
            }
        )
    return sorted(
        result,
        key=lambda item: (item["formation_date"], item["signal"], item["m_ticker"]),
    )


def _input_formation_map(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    raw = _array(payload.get("formation_observations"), "formation_observations")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, f"formation_observations[{index}]")
        key = (
            _text(row.get("cohort_id"), "formation cohort_id"),
            _text(row.get("m_ticker"), "formation m_ticker"),
        )
        assert isinstance(key[0], str) and isinstance(key[1], str)
        if key in result:
            raise PeadIndependentDailyLedgerError("daily input repeats a formation observation")
        result[key] = row
    return result


def _build_formation_states(
    observations: Sequence[Mapping[str, Any]], input_payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[row["formation_date"]].append(row)
    input_rows = _input_formation_map(input_payload)
    states: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    expected_keys: set[tuple[str, str]] = set()
    for formation in sorted(grouped):
        group = sorted(grouped[formation], key=lambda item: (item["signal"], item["m_ticker"]))
        eligible = len(group)
        k = int(QUANTILE * eligible) if eligible >= MINIMUM_NAMES else 0
        for horizon in HORIZONS:
            cohort_id = f"pooled:{formation}:{horizon}"
            short = {item["m_ticker"] for item in group[:k]}
            long = {item["m_ticker"] for item in group[-k:]} if k else set()
            for rank, observation in enumerate(group, start=1):
                m_ticker = observation["m_ticker"]
                key = (cohort_id, m_ticker)
                expected_keys.add(key)
                source = input_rows.get(key)
                if source is None:
                    raise PeadIndependentDailyLedgerError(
                        "daily input omits an exhaustive formation observation"
                    )
                leg = "short" if m_ticker in short else ("long" if m_ticker in long else None)
                expected_status = "admitted" if k else "below_minimum_names"
                expected_reason = None if k else "formation_below_frozen_ten_name_floor"
                comparisons = {
                    "cohort_id": cohort_id,
                    "formation_date": formation,
                    "entry_date": observation["entry_date"],
                    "exit_date": observation["horizons"][horizon]["exit_date"],
                    "horizon_sessions": horizon,
                    "ticker": observation["ticker"],
                    "m_ticker": m_ticker,
                    "rank": rank,
                    "selected_leg": leg,
                    "cohort_status": expected_status,
                    "cohort_reason": expected_reason,
                }
                for field, expected in comparisons.items():
                    if source.get(field) != expected:
                        raise PeadIndependentDailyLedgerError(
                            f"daily input formation {field} differs from independent ranking"
                        )
                if canonical_json(source.get("source_event_key")) != canonical_json(
                    observation["source_event_key"]
                ):
                    raise PeadIndependentDailyLedgerError(
                        "daily input formation source_event_key differs"
                    )
                if (
                    abs(_decimal(source.get("signal"), "formation signal") - observation["signal"])
                    > SOURCE_TOLERANCE
                ):
                    raise PeadIndependentDailyLedgerError("daily input formation signal differs")
                permaticker = _integer(
                    source.get("permaticker"), "formation permaticker", minimum=1
                )
                target = Decimal("0")
                if leg is not None:
                    target = (Decimal("1") if leg == "long" else Decimal("-1")) / Decimal(k)
                state = {
                    "cohort_id": cohort_id,
                    "formation_date": formation,
                    "horizon_sessions": horizon,
                    "checkpoint": "formation_target",
                    "session_date": formation,
                    "ticker": observation["ticker"],
                    "m_ticker": m_ticker,
                    "permaticker": permaticker,
                    "source_event_key": observation["source_event_key"],
                    "rank": rank,
                    "eligible_names": eligible,
                    "names_per_leg": k,
                    "leg": leg,
                    "signal": _money(observation["signal"]),
                    "target": _money(target),
                    "order": _quantity(Decimal("0")),
                    "position": _quantity(Decimal("0")),
                    "settled_cash": _money(INITIAL_NAV),
                    "distribution_receivable": _money(Decimal("0")),
                    "market_value": _money(Decimal("0")),
                    "nav": _money(INITIAL_NAV),
                    "cumulative_fees": _money(Decimal("0")),
                    "pnl": _money(Decimal("0")),
                }
                states.append(state)
                if leg is not None:
                    resolution = observation["horizons"][horizon]["resolution"]
                    if (
                        resolution.get("status") != "mechanically_reconstructed_nonqualifying"
                        or resolution.get("reason") is not None
                        or resolution.get("terminal_settlement_id") is not None
                    ):
                        raise PeadIndependentDailyLedgerError(
                            "independently selected path is not a resolved nonterminal path"
                        )
                    selected.append(
                        {
                            **state,
                            "calculation_target": target,
                            "entry_date": observation["entry_date"],
                            "exit_date": observation["horizons"][horizon]["exit_date"],
                            "resolution": resolution,
                            "candidate_return": observation["horizons"][horizon]["candidate"],
                        }
                    )
    if set(input_rows) != expected_keys:
        raise PeadIndependentDailyLedgerError(
            "daily input formation observations are not exhaustive and exact"
        )
    states.sort(
        key=lambda row: (
            row["formation_date"],
            row["horizon_sessions"],
            row["rank"],
            row["m_ticker"],
        )
    )
    selected.sort(
        key=lambda row: (
            row["formation_date"],
            row["horizon_sessions"],
            row["rank"],
            row["m_ticker"],
        )
    )
    return states, selected


def _sessions(payload: Mapping[str, Any]) -> tuple[list[str], dict[str, int]]:
    raw = _array(payload.get("sessions"), "sessions")
    values = [_iso(value, "session") for value in raw]
    if not values or values != sorted(set(values)):
        raise PeadIndependentDailyLedgerError("sessions must be nonempty, unique, and ascending")
    return values, {value: index for index, value in enumerate(values)}


def _prices(payload: Mapping[str, Any]) -> dict[tuple[str, str], tuple[Decimal, Decimal]]:
    raw = _array(payload.get("prices"), "prices")
    result: dict[tuple[str, str], tuple[Decimal, Decimal]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, f"prices[{index}]")
        ticker = _text(row.get("ticker"), "price ticker")
        day = _iso(row.get("date"), "price date")
        assert isinstance(ticker, str)
        key = (ticker, day)
        if key in result:
            raise PeadIndependentDailyLedgerError("daily input repeats a price row")
        result[key] = (
            _decimal(row.get("close"), "close", positive=True),
            _decimal(row.get("closeadj"), "closeadj", positive=True),
        )
    return result


def _currencies(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = _array(payload.get("currencies"), "currencies")
    result: dict[str, str] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, f"currencies[{index}]")
        ticker = _text(row.get("ticker"), "currency ticker")
        currency = _text(row.get("currency"), "currency")
        assert isinstance(ticker, str) and isinstance(currency, str)
        if ticker in result:
            raise PeadIndependentDailyLedgerError("daily input repeats a ticker currency")
        result[ticker] = currency
    return result


def _actions(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = _array(payload.get("actions"), "actions")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[str] = set()
    for index, value in enumerate(raw):
        row = _mapping(value, f"actions[{index}]")
        required = {"date", "action", "ticker", "name", "value", "contraticker", "contraname"}
        if set(row) != required:
            raise PeadIndependentDailyLedgerError(
                "corporate-action row fields differ from the frozen contract"
            )
        normalized = {
            "date": _iso(row["date"], "action date"),
            "action": _text(row["action"], "action kind"),
            "ticker": _text(row["ticker"], "action ticker"),
            "name": _text(row["name"], "action name"),
            "value": None if row["value"] is None else _decimal(row["value"], "action value"),
            "contraticker": _text(row["contraticker"], "action contraticker", optional=True),
            "contraname": _text(row["contraname"], "action contraname", optional=True),
        }
        if normalized["action"] != str(normalized["action"]).lower():
            raise PeadIndependentDailyLedgerError("action kind must be lowercase")
        token = canonical_json(
            {
                key: (_money(child) if isinstance(child, Decimal) else child)
                for key, child in normalized.items()
            }
        )
        if token in identities:
            raise PeadIndependentDailyLedgerError("daily input repeats an action row")
        identities.add(token)
        result[str(normalized["ticker"])].append(normalized)
    for values in result.values():
        values.sort(
            key=lambda row: (
                row["date"],
                row["action"],
                row["name"],
                row["contraticker"] or "",
                row["contraname"] or "",
            )
        )
    return result


def _action_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "date": row["date"],
        "ticker": row["ticker"],
        "name": row["name"],
        "action": row["action"],
        "contraname": row["contraname"],
        "contraticker": row["contraticker"],
    }


def _selected_path_manifest(
    payload: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> None:
    raw = _array(payload.get("selected_paths"), "selected_paths")
    actual: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, f"selected_paths[{index}]")
        key = (
            _text(row.get("cohort_id"), "selected cohort_id"),
            _text(row.get("m_ticker"), "selected m_ticker"),
        )
        assert isinstance(key[0], str) and isinstance(key[1], str)
        if key in actual:
            raise PeadIndependentDailyLedgerError("selected_paths repeats a path")
        actual[key] = row
    expected_keys = {(row["cohort_id"], row["m_ticker"]) for row in selected}
    if set(actual) != expected_keys:
        raise PeadIndependentDailyLedgerError(
            "selected_paths differs from the independent pooled selection"
        )
    for row in selected:
        source = actual[(row["cohort_id"], row["m_ticker"])]
        fields = {
            "cohort_id": row["cohort_id"],
            "formation_date": row["formation_date"],
            "entry_date": row["entry_date"],
            "exit_date": row["exit_date"],
            "horizon_sessions": row["horizon_sessions"],
            "ticker": row["ticker"],
            "m_ticker": row["m_ticker"],
            "permaticker": row["permaticker"],
            "rank": row["rank"],
            "leg": row["leg"],
        }
        for field, expected in fields.items():
            if source.get(field) != expected:
                raise PeadIndependentDailyLedgerError(
                    f"selected_paths {field} differs from independent selection"
                )
        if canonical_json(source.get("source_event_key")) != canonical_json(
            row["source_event_key"]
        ):
            raise PeadIndependentDailyLedgerError("selected_paths source_event_key differs")


def _validate_path_coverage(
    payload: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    expected_dates: Mapping[tuple[str, str], Sequence[str]],
) -> None:
    raw = _array(payload.get("path_coverage"), "path_coverage")
    actual: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, f"path_coverage[{index}]")
        key = (
            _text(row.get("cohort_id"), "path coverage cohort_id"),
            _text(row.get("m_ticker"), "path coverage m_ticker"),
        )
        assert isinstance(key[0], str) and isinstance(key[1], str)
        if key in actual:
            raise PeadIndependentDailyLedgerError("path_coverage repeats a path")
        actual[key] = row
    keys = {(row["cohort_id"], row["m_ticker"]) for row in selected}
    if set(actual) != keys:
        raise PeadIndependentDailyLedgerError("path_coverage differs from selected paths")
    for key in keys:
        dates = [
            _iso(value, "path coverage session")
            for value in _array(actual[key].get("session_dates"), "session_dates")
        ]
        if dates != list(expected_dates[key]):
            raise PeadIndependentDailyLedgerError(
                "path_coverage sessions differ from the exact global-session slice"
            )


def _expected_distributions(
    selected: Mapping[str, Any],
    held_actions: Sequence[Mapping[str, Any]],
    prices: Mapping[tuple[str, str], tuple[Decimal, Decimal]],
    session_index: Mapping[str, int],
    sessions: Sequence[str],
    adjustment_absolute_tolerance: Decimal,
    adjustment_relative_tolerance: Decimal,
) -> dict[str, list[tuple[Decimal, dict[str, Any]]]]:
    ticker = selected["ticker"]
    entry = selected["entry_date"]
    exit_date = selected["exit_date"]
    by_date: dict[str, list[tuple[Decimal, dict[str, Any]]]] = defaultdict(list)
    candidates_by_date: dict[str, list[tuple[Decimal, dict[str, Any]]]] = defaultdict(list)
    observed: list[tuple[str, Decimal, dict[str, Any]]] = []
    observed_diagnostics: dict[str, tuple[str, Decimal, Decimal, Decimal]] = {}
    for action in held_actions:
        if not entry < action["date"] <= exit_date:
            continue
        kind = action["action"]
        if kind in _IGNORED_ISSUER_EXTERNAL_ACTIONS:
            continue
        if kind in _TERMINAL_ACTIONS or kind in _UNSUPPORTED_HOLDER_ACTIONS or kind != "dividend":
            raise PeadIndependentDailyLedgerError(f"unsupported held corporate action: {kind}")
        amount = action["value"]
        if not isinstance(amount, Decimal) or amount <= 0:
            raise PeadIndependentDailyLedgerError("held dividend has no positive candidate amount")
        action_day = action["date"]
        candidates_by_date[action_day].append((amount, _action_key(action)))

    for action_day in sorted(candidates_by_date):
        action_position = session_index.get(action_day)
        if action_position is None or action_position == 0:
            raise PeadIndependentDailyLedgerError("held dividend has no exact prior global session")
        prior = sessions[action_position - 1]
        try:
            prior_close, prior_adjusted = prices[(ticker, prior)]
            action_close, action_adjusted = prices[(ticker, action_day)]
        except KeyError as exc:
            raise PeadIndependentDailyLedgerError(
                "held dividend lacks exact price-adjustment evidence"
            ) from exc
        implied = prior_close * action_adjusted / prior_adjusted - action_close
        applications = candidates_by_date[action_day]
        candidate_total = sum((amount for amount, _ in applications), Decimal("0"))
        error = abs(implied - candidate_total)
        allowed = adjustment_absolute_tolerance + adjustment_relative_tolerance * max(
            abs(implied), abs(candidate_total)
        )
        if error > allowed:
            raise PeadIndependentDailyLedgerError(
                "held dividend candidate total fails the independent adjustment check"
            )
        for amount, key in applications:
            by_date[action_day].append((amount, key))
            observed.append((action_day, amount, key))
            observed_diagnostics[canonical_json(key)] = (prior, implied, error, allowed)

    resolution = selected["resolution"]
    expected_raw = _array(resolution.get("cash_distributions"), "reference cash distributions")
    expected: list[tuple[str, Decimal, dict[str, Any]]] = []
    for index, value in enumerate(expected_raw):
        item = _mapping(value, f"reference cash distribution {index}")
        expected.append(
            (
                _iso(item.get("date"), "reference distribution date"),
                _decimal(item.get("amount"), "reference distribution amount", positive=True),
                _plain(_mapping(item.get("action_key"), "reference distribution action_key")),
            )
        )
    normalize = lambda values: sorted(  # noqa: E731
        [(day, _money(amount), key) for day, amount, key in values],
        key=canonical_json,
    )
    if canonical_json(normalize(observed)) != canonical_json(normalize(expected)):
        raise PeadIndependentDailyLedgerError(
            "frozen actions differ from independently reconstructed cash distributions"
        )
    for value in expected_raw:
        item = _mapping(value, "reference cash distribution diagnostic")
        key = _plain(_mapping(item.get("action_key"), "reference distribution action_key"))
        diagnostic = observed_diagnostics[canonical_json(key)]
        if (
            _iso(
                item.get("adjustment_previous_session"),
                "reference adjustment_previous_session",
            )
            != diagnostic[0]
        ):
            raise PeadIndependentDailyLedgerError(
                "independent dividend prior session differs from reference"
            )
        for field, observed_value in zip(
            (
                "adjustment_implied_amount",
                "adjustment_absolute_error",
                "adjustment_allowed_error",
            ),
            diagnostic[1:],
            strict=True,
        ):
            if (
                abs(_decimal(item.get(field), f"reference {field}") - observed_value)
                > SOURCE_TOLERANCE
            ):
                raise PeadIndependentDailyLedgerError(
                    f"independent dividend diagnostic differs for {field}"
                )
    return by_date


def _build_daily_states(
    selected: Sequence[Mapping[str, Any]], input_payload: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    sessions, session_index = _sessions(input_payload)
    prices = _prices(input_payload)
    currencies = _currencies(input_payload)
    actions = _actions(input_payload)
    semantics = _mapping(input_payload.get("distribution_semantics"), "distribution_semantics")
    adjustment_absolute_tolerance = _decimal(
        semantics.get("adjustment_check_absolute_tolerance"),
        "adjustment_check_absolute_tolerance",
    )
    adjustment_relative_tolerance = _decimal(
        semantics.get("adjustment_check_relative_tolerance"),
        "adjustment_check_relative_tolerance",
    )
    if adjustment_absolute_tolerance < 0 or adjustment_relative_tolerance < 0:
        raise PeadIndependentDailyLedgerError(
            "distribution adjustment tolerances cannot be negative"
        )
    _selected_path_manifest(input_payload, selected)

    path_dates: dict[tuple[str, str], list[str]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        if currencies.get(row["ticker"]) != "USD":
            raise PeadIndependentDailyLedgerError(
                "independent daily accounting requires USD currency"
            )
        entry_position = session_index.get(row["entry_date"])
        exit_position = session_index.get(row["exit_date"])
        if entry_position is None or exit_position is None:
            raise PeadIndependentDailyLedgerError(
                "selected path boundary is missing from global sessions"
            )
        if exit_position - entry_position != row["horizon_sessions"]:
            raise PeadIndependentDailyLedgerError(
                "selected exit is not the exact frozen session horizon"
            )
        dates = sessions[entry_position : exit_position + 1]
        if len(dates) != row["horizon_sessions"] + 1:
            raise PeadIndependentDailyLedgerError(
                "selected daily path has the wrong number of sessions"
            )
        for day in dates:
            if (row["ticker"], day) not in prices:
                raise PeadIndependentDailyLedgerError("selected daily path has a missing price bar")
        key = (row["cohort_id"], row["m_ticker"])
        path_dates[key] = dates
        grouped[row["cohort_id"]].append(row)
    _validate_path_coverage(input_payload, selected, path_dates)

    constituents: list[dict[str, Any]] = []
    cohorts: list[dict[str, Any]] = []
    distribution_applications = 0
    for cohort_id in sorted(grouped):
        rows = sorted(grouped[cohort_id], key=lambda item: (item["rank"], item["m_ticker"]))
        if not rows:
            continue
        k = rows[0]["names_per_leg"]
        if k <= 0 or len(rows) != 2 * k:
            raise PeadIndependentDailyLedgerError(
                "admitted cohort does not contain two exhaustive tails"
            )
        dates = path_dates[(cohort_id, rows[0]["m_ticker"])]
        if any(path_dates[(cohort_id, row["m_ticker"])] != dates for row in rows):
            raise PeadIndependentDailyLedgerError(
                "cohort constituents do not share an exact session path"
            )
        quantities: dict[str, Decimal] = {}
        positions: dict[str, Decimal] = {}
        receivables: dict[str, Decimal] = {}
        name_fees: dict[str, Decimal] = {}
        distributions: dict[str, dict[str, list[tuple[Decimal, dict[str, Any]]]]] = {}
        per_name_fee = ONE_WAY_FEE_RATE / Decimal(k)
        for row in rows:
            entry_close = prices[(row["ticker"], row["entry_date"])][0]
            source_entry = _decimal(
                row["resolution"].get("entry_price_split_normalized"),
                "reference resolution entry price",
                positive=True,
            )
            if abs(entry_close - source_entry) > SOURCE_TOLERANCE:
                raise PeadIndependentDailyLedgerError(
                    "daily input entry price differs from independent reference"
                )
            target = row.get("calculation_target")
            if not isinstance(target, Decimal) or not target.is_finite() or target == 0:
                raise PeadIndependentDailyLedgerError(
                    "selected path omits its exact calculation target"
                )
            quantities[row["m_ticker"]] = target / entry_close
            positions[row["m_ticker"]] = Decimal("0")
            receivables[row["m_ticker"]] = Decimal("0")
            name_fees[row["m_ticker"]] = Decimal("0")
            distributions[row["m_ticker"]] = _expected_distributions(
                row,
                actions.get(row["ticker"], []),
                prices,
                session_index,
                sessions,
                adjustment_absolute_tolerance,
                adjustment_relative_tolerance,
            )

        cash = INITIAL_NAV
        cumulative_fees = Decimal("0")
        previous_nav = INITIAL_NAV
        for sequence, day in enumerate(dates):
            entry = sequence == 0
            exit_day = sequence == len(dates) - 1
            checkpoint = "entry_close" if entry else ("exit_close" if exit_day else "mark_close")
            pending_rows: list[dict[str, Any]] = []
            for row in rows:
                name = row["m_ticker"]
                price = prices[(row["ticker"], day)][0]
                quantity = quantities[name]
                accrual_today = Decimal("0")
                action_keys: list[dict[str, Any]] = []
                if not entry:
                    for amount, key in distributions[name].get(day, []):
                        signed = quantity * amount
                        receivables[name] += signed
                        accrual_today += signed
                        action_keys.append(key)
                        distribution_applications += 1
                order = Decimal("0")
                fee_today = Decimal("0")
                if entry:
                    order = quantity
                    positions[name] = quantity
                    fee_today = per_name_fee
                elif exit_day:
                    order = -quantity
                    positions[name] = Decimal("0")
                    fee_today = per_name_fee
                order_cash_flow = -order * price
                cash += order_cash_flow - fee_today
                cumulative_fees += fee_today
                name_fees[name] += fee_today
                prior_price = (
                    price
                    if entry
                    else prices[(row["ticker"], dates[sequence - 1])][0]
                )
                price_pnl = (
                    Decimal("0") if entry else quantity * (price - prior_price)
                )
                net_pnl_contribution = price_pnl + accrual_today - fee_today
                pending_rows.append(
                    {
                        "source": row,
                        "price": price,
                        "target": Decimal("0") if exit_day else row["calculation_target"],
                        "order": order,
                        "position": positions[name],
                        "order_cash_flow": order_cash_flow,
                        "distribution_accrual_today": accrual_today,
                        "distribution_receivable": receivables[name],
                        "fee_today": fee_today,
                        "cumulative_name_fees": name_fees[name],
                        "price_pnl": price_pnl,
                        "net_pnl_contribution": net_pnl_contribution,
                        "action_keys": action_keys,
                    }
                )
            market_value = sum(
                (positions[row["m_ticker"]] * prices[(row["ticker"], day)][0] for row in rows),
                Decimal("0"),
            )
            distribution_receivable = sum(receivables.values(), Decimal("0"))
            long_market_value = sum(
                (
                    positions[row["m_ticker"]] * prices[(row["ticker"], day)][0]
                    for row in rows
                    if row["leg"] == "long"
                ),
                Decimal("0"),
            )
            short_market_value = -sum(
                (
                    positions[row["m_ticker"]] * prices[(row["ticker"], day)][0]
                    for row in rows
                    if row["leg"] == "short"
                ),
                Decimal("0"),
            )
            nav = cash + distribution_receivable + market_value
            pnl = nav - INITIAL_NAV
            daily_pnl = nav - previous_nav
            if abs(
                daily_pnl
                - sum(
                    (row["net_pnl_contribution"] for row in pending_rows),
                    Decimal("0"),
                )
            ) > SOURCE_TOLERANCE:
                raise PeadIndependentDailyLedgerError(
                    "cohort daily P&L differs from exact constituent contributions"
                )
            previous_nav = nav
            open_positions = sum(value != 0 for value in positions.values())
            cohort_state = {
                "cohort_id": cohort_id,
                "formation_date": rows[0]["formation_date"],
                "horizon_sessions": rows[0]["horizon_sessions"],
                "sequence": sequence,
                "checkpoint": checkpoint,
                "session_date": day,
                "settled_cash": _money(cash),
                "distribution_receivable": _money(distribution_receivable),
                "market_value": _money(market_value),
                "gross_long_market_value": _money(long_market_value),
                "gross_short_market_value": _money(short_market_value),
                "nav": _money(nav),
                "cumulative_fees": _money(cumulative_fees),
                "daily_pnl": _money(daily_pnl),
                "pnl": _money(pnl),
                "open_position_count": open_positions,
            }
            cohorts.append(cohort_state)
            for pending in pending_rows:
                row = pending["source"]
                position_market_value = pending["position"] * pending["price"]
                constituents.append(
                    {
                        "cohort_id": cohort_id,
                        "formation_date": row["formation_date"],
                        "horizon_sessions": row["horizon_sessions"],
                        "sequence": sequence,
                        "checkpoint": checkpoint,
                        "session_date": day,
                        "ticker": row["ticker"],
                        "m_ticker": row["m_ticker"],
                        "permaticker": row["permaticker"],
                        "source_event_key": row["source_event_key"],
                        "rank": row["rank"],
                        "leg": row["leg"],
                        "signal": row["signal"],
                        "price_split_normalized": _money(pending["price"]),
                        "target": _money(pending["target"]),
                        "order": _quantity(pending["order"]),
                        "position": _quantity(pending["position"]),
                        "order_cash_flow": _money(pending["order_cash_flow"]),
                        "distribution_accrual_today": _money(pending["distribution_accrual_today"]),
                        "distribution_receivable": _money(pending["distribution_receivable"]),
                        "market_value": _money(position_market_value),
                        "price_pnl": _money(pending["price_pnl"]),
                        "fee_today": _money(pending["fee_today"]),
                        "net_pnl_contribution": _money(
                            pending["net_pnl_contribution"]
                        ),
                        "cumulative_name_fees": _money(pending["cumulative_name_fees"]),
                        "applied_distribution_action_keys": pending["action_keys"],
                    }
                )
        terminal = cohorts[-1]
        if terminal["cohort_id"] != cohort_id or terminal["open_position_count"] != 0:
            raise PeadIndependentDailyLedgerError(
                "independent cohort did not liquidate every position"
            )
        expected_pnl = Decimal("0")
        for row in rows:
            entry_price = prices[(row["ticker"], row["entry_date"])][0]
            exit_price = prices[(row["ticker"], row["exit_date"])][0]
            resolution = row["resolution"]
            source_exit = _decimal(
                resolution.get("exit_price_split_normalized"),
                "reference resolution exit price",
                positive=True,
            )
            cash_total = _decimal(resolution.get("cash_total"), "reference resolution cash_total")
            if abs(exit_price - source_exit) > SOURCE_TOLERANCE:
                raise PeadIndependentDailyLedgerError(
                    "daily input exit price differs from independent reference"
                )
            reconstructed = (exit_price + cash_total) / entry_price - INITIAL_NAV
            if abs(reconstructed - row["candidate_return"]) > SOURCE_TOLERANCE:
                raise PeadIndependentDailyLedgerError(
                    "independent terminal return differs from reference candidate"
                )
            expected_pnl += (
                quantities[row["m_ticker"]] * (exit_price - entry_price + cash_total)
                - Decimal("2") * per_name_fee
            )
        if abs(_decimal(terminal["pnl"], "terminal pnl") - expected_pnl) > SOURCE_TOLERANCE:
            raise PeadIndependentDailyLedgerError(
                "terminal event-loop P&L does not reconcile to constituent economics"
            )

    constituents.sort(
        key=lambda row: (
            row["formation_date"],
            row["horizon_sessions"],
            row["sequence"],
            row["rank"],
            row["m_ticker"],
        )
    )
    cohorts.sort(key=lambda row: (row["formation_date"], row["horizon_sessions"], row["sequence"]))
    return constituents, cohorts, distribution_applications


def _protocol() -> dict[str, Any]:
    return {
        "scope": "pooled_development_sample_only",
        "horizons_sessions": list(HORIZONS),
        "minimum_names_per_formation": MINIMUM_NAMES,
        "quantile": _money(QUANTILE),
        "rank_order": "ascending_signal_then_m_ticker",
        "formation_rows": "every_eligible_name_for_both_horizons",
        "daily_rows": "selected_tails_entry_through_exit_inclusive",
        "initial_nav_per_independent_cohort": _money(INITIAL_NAV),
        "long_gross": _money(Decimal("1")),
        "short_gross": _money(Decimal("1")),
        "one_way_fee_rate_per_name_target": _money(ONE_WAY_FEE_RATE),
        "quantity_basis": "signed_target_divided_by_entry_split_normalized_close",
        "orders": "entry_and_exit_at_same_session_split_normalized_close",
        "state_timing": "post_trade_post_accrual_close_state",
        "cash_distribution_interval": "entry_exclusive_exit_inclusive",
        "cash_distribution_treatment": "candidate_receivable_accrual_never_settled_cash",
        "cash_distribution_reinvestment": False,
        "cash_yield": "not_modeled",
        "capital_sharing": "none_independent_normalized_cohort_subledgers",
        "borrow": "not_modeled",
        "financing": "not_modeled",
        "margin_and_short_proceeds": "not_modeled",
        "terminal_actions": "fail_closed_without_independent_settlement_receipt",
        "decimal_precision": DECIMAL_PRECISION,
        "decimal_rounding": "ROUND_HALF_EVEN",
        "rounding_stage": "serialization_only",
    }


def _build(
    reference_artifact: Mapping[str, Any], daily_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    reference_payload = _reference_payload(reference_artifact)
    input_payload = _input_payload(daily_inputs)
    observations = _reference_observations(reference_payload)
    formation, selected = _build_formation_states(observations, input_payload)
    daily, cohorts, distribution_applications = _build_daily_states(selected, input_payload)
    bindings = {
        "independent_reference_artifact_hash": reference_artifact["artifact_hash"],
        "daily_input_snapshot_hash": daily_inputs["artifact_hash"],
        "combined_data_snapshot_hash": _mapping(
            input_payload.get("bindings"), "daily input bindings"
        ).get("combined_data_snapshot_hash"),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "independent_event_driven_daily_modeled_accounting_nonqualifying",
        "qualifying_evidence": False,
        "replication_evidence_eligible": False,
        "paper_execution_evidence": False,
        "promotion_allowed": False,
        "bindings": bindings,
        "frozen_protocol": _protocol(),
        "accounting_caveats": {
            "settled_cash_excludes_distribution_receivables": True,
            "distribution_payment_dates_known": False,
            "distribution_semantics_authoritatively_documented": False,
            "broker_share_quantities": False,
            "observed_orders_quotes_or_fills": False,
        },
        "formation_states": formation,
        "daily_constituent_states": daily,
        "cohort_daily_states": cohorts,
        "coverage": {
            "formation_horizon_cells": len({row["cohort_id"] for row in formation}),
            "formation_states": len(formation),
            "admitted_cohorts": len({row["cohort_id"] for row in selected}),
            "selected_constituent_paths": len(selected),
            "daily_constituent_states": len(daily),
            "cohort_daily_states": len(cohorts),
            "distribution_applications": distribution_applications,
            "replication_projection_observations": len(formation) + len(daily),
        },
        "blockers": sorted(_REQUIRED_BLOCKERS),
        "nonclaims": {
            "candidate_qualified": False,
            "full_eight_cell_family_reconciled": False,
            "paper_execution_observed": False,
            "broker_execution_observed": False,
            "promotion_allowed": False,
        },
    }
    normalized = _plain(payload)
    document = {"artifact_hash": content_hash(normalized), "payload": normalized}
    _validate_static(document)
    return document


def build_independent_daily_ledger(
    reference_artifact: Mapping[str, Any], daily_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the independent session-by-session PEAD modeled ledger."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return _build(reference_artifact, daily_inputs)


def _validate_static(document: Mapping[str, Any]) -> None:
    wrapper = _mapping(document, "independent daily ledger")
    if set(wrapper) != {"artifact_hash", "payload"}:
        raise PeadIndependentDailyLedgerError("independent daily ledger wrapper fields differ")
    payload = _mapping(wrapper["payload"], "independent daily ledger payload")
    if content_hash(payload) != wrapper["artifact_hash"]:
        raise PeadIndependentDailyLedgerError("independent daily ledger artifact hash mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PeadIndependentDailyLedgerError("independent daily schema changed")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadIndependentDailyLedgerError(
            "independent daily ledger belongs to another candidate"
        )
    for field in (
        "qualifying_evidence",
        "replication_evidence_eligible",
        "paper_execution_evidence",
        "promotion_allowed",
    ):
        if payload.get(field) is not False:
            raise PeadIndependentDailyLedgerError(f"independent daily ledger cannot claim {field}")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or not _REQUIRED_BLOCKERS.issubset(blockers):
        raise PeadIndependentDailyLedgerError("independent daily ledger omits required blockers")
    formation = _array(payload.get("formation_states"), "formation_states")
    daily = _array(payload.get("daily_constituent_states"), "daily_constituent_states")
    cohorts = _array(payload.get("cohort_daily_states"), "cohort_daily_states")
    coverage = _mapping(payload.get("coverage"), "coverage")
    expected = {
        "formation_horizon_cells": len({row["cohort_id"] for row in formation}),
        "formation_states": len(formation),
        "admitted_cohorts": len({row["cohort_id"] for row in daily}),
        "selected_constituent_paths": len({(row["cohort_id"], row["m_ticker"]) for row in daily}),
        "daily_constituent_states": len(daily),
        "cohort_daily_states": len(cohorts),
        "distribution_applications": sum(
            len(row["applied_distribution_action_keys"]) for row in daily
        ),
        "replication_projection_observations": len(formation) + len(daily),
    }
    if canonical_json(coverage) != canonical_json(expected):
        raise PeadIndependentDailyLedgerError("independent daily ledger coverage is inconsistent")


def validate_independent_daily_ledger(
    document: Mapping[str, Any],
    *,
    reference_artifact: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate content identity and rebuild every state from both bound inputs."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        _validate_static(document)
        rebuilt = _build(reference_artifact, daily_inputs)
        if canonical_json(rebuilt) != canonical_json(document):
            raise PeadIndependentDailyLedgerError(
                "independent daily ledger differs from its exhaustive input rebuild"
            )
        return _plain(document)


def replication_observations(
    document: Mapping[str, Any],
    *,
    reference_artifact: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project all formation and selected daily states to the generic contract."""
    validated = validate_independent_daily_ledger(
        document, reference_artifact=reference_artifact, daily_inputs=daily_inputs
    )
    payload = validated["payload"]
    result: list[dict[str, Any]] = []
    for row in payload["formation_states"]:
        result.append(
            {
                "key": {
                    "candidate_id": CANDIDATE_ID,
                    "slice": "pooled",
                    "cohort_id": row["cohort_id"],
                    "formation_date": row["formation_date"],
                    "horizon_sessions": row["horizon_sessions"],
                    "checkpoint": "formation_target",
                    "session_date": row["session_date"],
                    "ticker": row["ticker"],
                    "m_ticker": row["m_ticker"],
                    "permaticker": row["permaticker"],
                    "source_event_key": row["source_event_key"],
                },
                "eligibility": True,
                "signal": float(Decimal(row["signal"])),
                "rank": float(row["rank"]),
                "target": float(Decimal(row["target"])),
                "order": 0.0,
                "position": 0.0,
                "cash": 1.0,
                "fees": 0.0,
                "pnl": 0.0,
            }
        )
    cohorts = {
        (row["cohort_id"], row["session_date"]): row for row in payload["cohort_daily_states"]
    }
    for row in payload["daily_constituent_states"]:
        cohort = cohorts[(row["cohort_id"], row["session_date"])]
        result.append(
            {
                "key": {
                    "candidate_id": CANDIDATE_ID,
                    "slice": "pooled",
                    "cohort_id": row["cohort_id"],
                    "formation_date": row["formation_date"],
                    "horizon_sessions": row["horizon_sessions"],
                    "checkpoint": row["checkpoint"],
                    "session_date": row["session_date"],
                    "ticker": row["ticker"],
                    "m_ticker": row["m_ticker"],
                    "permaticker": row["permaticker"],
                    "source_event_key": row["source_event_key"],
                },
                "eligibility": True,
                "signal": float(Decimal(row["signal"])),
                "rank": float(row["rank"]),
                "target": float(Decimal(row["target"])),
                "order": float(Decimal(row["order"])),
                "position": float(Decimal(row["position"])),
                "cash": float(Decimal(cohort["settled_cash"])),
                "fees": float(Decimal(cohort["cumulative_fees"])),
                "pnl": float(Decimal(cohort["pnl"])),
            }
        )
    return sorted(result, key=lambda item: canonical_json(item["key"]))


__all__ = [
    "CANDIDATE_ID",
    "PeadIndependentDailyLedgerError",
    "SCHEMA_VERSION",
    "build_independent_daily_ledger",
    "replication_observations",
    "validate_independent_daily_ledger",
]
