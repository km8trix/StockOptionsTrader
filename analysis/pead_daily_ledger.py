"""Primary daily mark-to-market extension of the modeled PEAD cohort ledger.

This module deliberately extends, rather than relabels, the immutable
``pead_modeled_execution_ledger.v1`` artifact.  It uses the primary ledger's
frozen selections, quantities, fees, and candidate distribution accruals and
marks every selected constituent on every exact global SEP session from entry
through exit.  The result is synthetic normalized accounting, not broker or
paper-execution evidence.

The independently written money path lives in ``pead_daily_reference`` and is
not imported here.  A separate PEAD-specific receipt must reconcile the two
outputs before the bounded independent-money-path claim can be made.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from typing import Any

from analysis.pead_execution_ledger import validate_pead_execution_ledger
from data.pead_economic_evidence import canonical_json, content_hash


SCHEMA_VERSION = "pead_primary_daily_ledger.v2"
CANDIDATE_ID = "pead-vq-locked-replication-v1"
PROTOCOL_SCHEMA_VERSION = "pead_daily_money_path_protocol.v2"
INPUT_SCHEMA_VERSION = "pead_daily_input_snapshot.v1"
DECIMAL_PRECISION = 50
MONEY_QUANTUM = Decimal("0.000000000000000001")
QUANTITY_QUANTUM = Decimal("0.000000000000000000000001")
ONE_WAY_FEE_RATE = Decimal("0.003")
INITIAL_NAV = Decimal("1")
IDENTITY_TOLERANCE = Decimal("0.000000000001")


class PeadDailyLedgerError(ValueError):
    """The primary daily ledger is malformed or cannot be rebuilt."""


def _plain(value: Any) -> Any:
    return __import__("json").loads(canonical_json(value))


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise PeadDailyLedgerError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PeadDailyLedgerError(f"{label} must be a finite decimal") from exc
    if not result.is_finite() or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise PeadDailyLedgerError(f"{label} must be a finite {qualifier}decimal")
    return result


def _fixed(value: Decimal, *, quantity: bool = False) -> str:
    quantum = QUANTITY_QUANTUM if quantity else MONEY_QUANTUM
    return format(value.quantize(quantum, rounding=ROUND_HALF_EVEN), "f")


def _verified_wrapper(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"artifact_hash", "payload"}:
        raise PeadDailyLedgerError(f"{label} must be a content-addressed wrapper")
    artifact_hash = value["artifact_hash"]
    payload = value["payload"]
    if (
        not isinstance(artifact_hash, str)
        or len(artifact_hash) != 64
        or not isinstance(payload, Mapping)
        or content_hash(payload) != artifact_hash
    ):
        raise PeadDailyLedgerError(f"{label} content identity is invalid")
    return payload


def _validated_protocol(value: Any) -> Mapping[str, Any]:
    payload = _verified_wrapper(value, "daily money-path protocol")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise PeadDailyLedgerError("unsupported daily money-path protocol")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadDailyLedgerError("daily protocol belongs to another candidate")
    expected = payload.get("development_sample_expected_coverage")
    if not isinstance(expected, Mapping) or expected.get("generic_observation_keys") != 4114:
        raise PeadDailyLedgerError("daily protocol does not freeze exhaustive coverage")
    accounting = payload.get("accounting")
    if not isinstance(accounting, Mapping) or accounting.get("distribution_payment") != (
        "not_modeled_candidate_receivable_never_settled_cash"
    ):
        raise PeadDailyLedgerError("daily protocol overstates distribution cash timing")
    return payload


def _validated_inputs(value: Any) -> Mapping[str, Any]:
    # Import lazily so this module retains a clear one-way dependency and test
    # fixtures can report a missing input layer as a typed daily-ledger error.
    try:
        from analysis.pead_daily_inputs import validate_pead_daily_input_snapshot
    except ImportError as exc:  # pragma: no cover - installation defect
        raise PeadDailyLedgerError("daily input validator is unavailable") from exc
    try:
        validated = validate_pead_daily_input_snapshot(value)
    except (TypeError, ValueError) as exc:
        raise PeadDailyLedgerError("daily input snapshot is invalid") from exc
    payload = validated["payload"]
    if payload.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise PeadDailyLedgerError("unsupported daily input snapshot")
    return payload


def _source_event_token(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        raise PeadDailyLedgerError("source event key must be a nonempty object")
    return canonical_json(value)


def _input_maps(payload: Mapping[str, Any]) -> tuple[list[str], dict[tuple[str, str], Decimal]]:
    sessions = payload.get("sessions")
    prices = payload.get("prices")
    if not isinstance(sessions, list) or not sessions or not isinstance(prices, list):
        raise PeadDailyLedgerError("daily inputs omit sessions or prices")
    if sessions != sorted(set(sessions)):
        raise PeadDailyLedgerError("daily input sessions are not canonical")
    price_map: dict[tuple[str, str], Decimal] = {}
    for row in prices:
        if not isinstance(row, Mapping):
            raise PeadDailyLedgerError("daily input price row is malformed")
        key = (str(row.get("ticker")), str(row.get("date")))
        if key in price_map:
            raise PeadDailyLedgerError("daily input repeats a ticker/date price")
        price_map[key] = _decimal(row.get("close"), "SEP.close", positive=True)
    return sessions, price_map


def _formation_states(manifests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for manifest in manifests:
        cohort_id = manifest["cohort_id"]
        formation = manifest["formation_date"]
        horizon = manifest["horizon_sessions"]
        for row in manifest["ranked_constituents"]:
            leg = row["selected_leg"]
            target = Decimal("0")
            if leg is not None:
                sign = Decimal("1") if leg == "long" else Decimal("-1")
                target = sign / Decimal(manifest["names_per_leg"])
            result.append(
                {
                    "cohort_id": cohort_id,
                    "formation_date": formation,
                    "horizon_sessions": horizon,
                    "ticker": row["ticker"],
                    "m_ticker": row["m_ticker"],
                    "permaticker": row["permaticker"],
                    "source_event_key": _plain(row["source_event_key"]),
                    "rank": row["rank"],
                    "signal": row["signal"],
                    "selected_leg": leg,
                    "target": _fixed(target),
                }
            )
    return result


def _cohort_daily_rows(
    constituents: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    sessions: Sequence[str],
    prices: Mapping[tuple[str, str], Decimal],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session_index = {value: index for index, value in enumerate(sessions)}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in constituents:
        grouped[str(row["cohort_id"])].append(row)
    summary_map = {str(row["cohort_id"]): row for row in summaries}
    constituent_states: list[dict[str, Any]] = []
    cohort_states: list[dict[str, Any]] = []

    for cohort_id in sorted(grouped, key=lambda token: (
        grouped[token][0]["formation_date"], grouped[token][0]["horizon_sessions"]
    )):
        rows = sorted(grouped[cohort_id], key=lambda row: (row["rank"], row["m_ticker"]))
        summary = summary_map.get(cohort_id)
        if summary is None:
            raise PeadDailyLedgerError("daily cohort has no modeled-ledger summary")
        names_per_leg = summary.get("names_per_leg")
        if type(names_per_leg) is not int or names_per_leg <= 0:
            raise PeadDailyLedgerError("daily cohort names_per_leg is invalid")
        if len(rows) != 2 * names_per_leg:
            raise PeadDailyLedgerError("daily cohort does not contain two exhaustive legs")
        entries = {str(row["entry_date"]) for row in rows}
        exits = {str(row["exit_date"]) for row in rows}
        if len(entries) != 1 or len(exits) != 1:
            raise PeadDailyLedgerError("cohort constituents do not share entry and exit")
        entry = next(iter(entries))
        exit_date = next(iter(exits))
        if entry not in session_index or exit_date not in session_index:
            raise PeadDailyLedgerError("cohort entry or exit is absent from global sessions")
        start_index = session_index[entry]
        end_index = session_index[exit_date]
        horizon = int(rows[0]["horizon_sessions"])
        if end_index - start_index != horizon:
            raise PeadDailyLedgerError("cohort exit is not the exact frozen session horizon")
        path_sessions = list(sessions[start_index : end_index + 1])
        if len(path_sessions) != horizon + 1:
            raise PeadDailyLedgerError("cohort daily path is incomplete")

        parsed: list[dict[str, Any]] = []
        for row in rows:
            entry_price = prices.get((str(row["ticker"]), entry))
            if entry_price is None:
                raise PeadDailyLedgerError("selected path is missing its entry close")
            leg = row.get("leg")
            if leg not in {"long", "short"}:
                raise PeadDailyLedgerError("constituent leg is invalid")
            sign = Decimal("1") if leg == "long" else Decimal("-1")
            target = sign / Decimal(names_per_leg)
            quantity = target / entry_price
            serialized_quantity = _decimal(
                row["signed_split_normalized_share_equivalent_quantity"],
                "constituent quantity",
            )
            serialized_target = _decimal(
                row["signed_target_notional"], "constituent target"
            )
            if (
                abs(serialized_target - target) > IDENTITY_TOLERANCE
                or abs(serialized_quantity - quantity) > IDENTITY_TOLERANCE
            ):
                raise PeadDailyLedgerError(
                    "modeled-ledger target or quantity differs from exact daily basis"
                )
            accruals: dict[str, list[tuple[Decimal, dict[str, Any]]]] = defaultdict(list)
            seen_action_keys: set[str] = set()
            for accrual in row["distribution_accruals"]:
                accrual_date = str(accrual["date"])
                action_key = accrual.get("action_key")
                action_token = _source_event_token(action_key)
                if action_token in seen_action_keys:
                    raise PeadDailyLedgerError("constituent repeats a distribution action")
                seen_action_keys.add(action_token)
                accruals[accrual_date].append(
                    (
                        _decimal(
                            accrual["signed_accrual_pnl"],
                            "signed distribution accrual",
                        ),
                        _plain(action_key),
                    )
                )
            for applications in accruals.values():
                applications.sort(key=lambda value: canonical_json(value[1]))
            parsed.append(
                {
                    "source": row,
                    "quantity": quantity,
                    "target": target,
                    "entry_fee": _decimal(row["entry_fee"], "entry fee"),
                    "exit_fee": _decimal(row["exit_fee"], "exit fee"),
                    "accruals": accruals,
                    "receivable": Decimal("0"),
                }
            )

        cash = INITIAL_NAV
        receivable = Decimal("0")
        cumulative_fees = Decimal("0")
        previous_nav = INITIAL_NAV
        previous_prices: dict[str, Decimal] = {}
        for sequence, session_date in enumerate(path_sessions):
            is_entry = sequence == 0
            is_exit = sequence == horizon
            checkpoint = "entry_close" if is_entry else (
                "exit_close" if is_exit else "mark_close"
            )
            cohort_market = Decimal("0")
            gross_long = Decimal("0")
            gross_short = Decimal("0")
            daily_accrual = Decimal("0")
            checkpoint_fees = Decimal("0")
            order_cash_flow = Decimal("0")
            state_rows: list[dict[str, Any]] = []

            for item in parsed:
                row = item["source"]
                ticker = str(row["ticker"])
                close = prices.get((ticker, session_date))
                if close is None:
                    raise PeadDailyLedgerError(
                        f"selected path is missing SEP.close for {ticker} {session_date}"
                    )
                quantity = item["quantity"]
                target = item["target"]
                order = quantity if is_entry else (-quantity if is_exit else Decimal("0"))
                position = Decimal("0") if is_exit else quantity
                fee = item["entry_fee"] if is_entry else (
                    item["exit_fee"] if is_exit else Decimal("0")
                )
                cash_flow = -(order * close)
                distribution_applications = item["accruals"].get(session_date, [])
                accrual = sum(
                    (value for value, _ in distribution_applications), Decimal("0")
                )
                action_keys = [
                    _plain(action_key) for _, action_key in distribution_applications
                ]
                item["receivable"] += accrual
                market_value = position * close
                prior_close = previous_prices.get(ticker)
                price_pnl = (
                    Decimal("0") if is_entry or prior_close is None
                    else quantity * (close - prior_close)
                )
                net_pnl_today = price_pnl + accrual - fee
                order_cash_flow += cash_flow
                checkpoint_fees += fee
                daily_accrual += accrual
                cohort_market += market_value
                if market_value >= 0:
                    gross_long += market_value
                else:
                    gross_short += -market_value
                state_rows.append(
                    {
                        "cohort_id": cohort_id,
                        "formation_date": row["formation_date"],
                        "entry_date": entry,
                        "exit_date": exit_date,
                        "horizon_sessions": horizon,
                        "sequence": sequence,
                        "session_date": session_date,
                        "checkpoint": checkpoint,
                        "ticker": ticker,
                        "m_ticker": row["m_ticker"],
                        "permaticker": row["permaticker"],
                        "source_event_key": _plain(row["source_event_key"]),
                        "rank": row["rank"],
                        "leg": row["leg"],
                        "signal": row["signal"],
                        "close_split_normalized": _fixed(close),
                        "target": _fixed(Decimal("0") if is_exit else target),
                        "order": _fixed(order, quantity=True),
                        "position": _fixed(position, quantity=True),
                        "order_cash_flow": _fixed(cash_flow),
                        "candidate_distribution_accrual": _fixed(accrual),
                        "applied_distribution_action_keys": action_keys,
                        "candidate_distribution_receivable": _fixed(item["receivable"]),
                        "market_value": _fixed(market_value),
                        "price_pnl": _fixed(price_pnl),
                        "modeled_fee": _fixed(fee),
                        "net_pnl_contribution": _fixed(net_pnl_today),
                    }
                )
                previous_prices[ticker] = close

            cash += order_cash_flow - checkpoint_fees
            receivable += daily_accrual
            cumulative_fees += checkpoint_fees
            nav = cash + receivable + cohort_market
            daily_pnl = nav - previous_nav
            cumulative_pnl = nav - INITIAL_NAV
            if abs(daily_pnl - sum(
                (_decimal(row["net_pnl_contribution"], "daily contribution") for row in state_rows),
                Decimal("0"),
            )) > IDENTITY_TOLERANCE:
                raise PeadDailyLedgerError("cohort daily P&L does not equal constituent contributions")
            cohort_state = {
                "cohort_id": cohort_id,
                "formation_date": rows[0]["formation_date"],
                "entry_date": entry,
                "exit_date": exit_date,
                "horizon_sessions": horizon,
                "sequence": sequence,
                "session_date": session_date,
                "checkpoint": checkpoint,
                "settled_cash": _fixed(cash),
                "candidate_distribution_receivable": _fixed(receivable),
                "market_value": _fixed(cohort_market),
                "gross_long_market_value": _fixed(gross_long),
                "gross_short_market_value": _fixed(gross_short),
                "nav": _fixed(nav),
                "checkpoint_fees": _fixed(checkpoint_fees),
                "cumulative_fees": _fixed(cumulative_fees),
                "daily_pnl": _fixed(daily_pnl),
                "cumulative_pnl": _fixed(cumulative_pnl),
                "open_position_count": 0 if is_exit else len(parsed),
            }
            cohort_states.append(cohort_state)
            for state in state_rows:
                state["cohort_settled_cash"] = cohort_state["settled_cash"]
                state["cohort_candidate_distribution_receivable"] = cohort_state[
                    "candidate_distribution_receivable"
                ]
                state["cohort_nav"] = cohort_state["nav"]
                state["cohort_cumulative_fees"] = cohort_state["cumulative_fees"]
                state["cohort_cumulative_pnl"] = cohort_state["cumulative_pnl"]
                constituent_states.append(state)
            previous_nav = nav

        terminal = _decimal(summary["terminal_nav"], "modeled terminal NAV")
        if abs(previous_nav - terminal) > IDENTITY_TOLERANCE:
            raise PeadDailyLedgerError("daily terminal NAV differs from modeled ledger")
        if _decimal(cohort_states[-1]["market_value"], "terminal market value") != 0:
            raise PeadDailyLedgerError("daily terminal positions did not flatten")
    return constituent_states, cohort_states


def _coverage(
    formation_states: Sequence[Mapping[str, Any]],
    daily_states: Sequence[Mapping[str, Any]],
    cohort_states: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "formation_checkpoints": len(formation_states),
        "selected_constituent_paths": len({
            (row["cohort_id"], row["m_ticker"]) for row in daily_states
        }),
        "daily_selected_constituent_checkpoints": len(daily_states),
        "cohort_daily_checkpoints": len(cohort_states),
        "generic_observation_keys": len(formation_states) + len(daily_states),
    }


def build_primary_daily_ledger(
    source_report: Mapping[str, Any],
    modeled_ledger: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the primary exact-session daily accounting artifact."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        validated_ledger = validate_pead_execution_ledger(
            modeled_ledger, source_report=source_report
        )
        input_payload = _validated_inputs(daily_inputs)
        _validated_protocol(protocol)
        ledger_payload = validated_ledger["payload"]
        input_bindings = input_payload.get("bindings")
        if not isinstance(input_bindings, Mapping):
            raise PeadDailyLedgerError("daily inputs omit source bindings")
        if input_bindings.get("modeled_execution_ledger_hash") != modeled_ledger.get(
            "artifact_hash"
        ):
            raise PeadDailyLedgerError("daily inputs bind another modeled ledger")
        sessions, prices = _input_maps(input_payload)
        formation_states = _formation_states(ledger_payload["selection_manifest"])
        daily_states, cohort_states = _cohort_daily_rows(
            ledger_payload["constituent_ledger"],
            ledger_payload["cohort_summaries"],
            sessions,
            prices,
        )
        coverage = _coverage(formation_states, daily_states, cohort_states)
        expected = protocol["payload"]["development_sample_expected_coverage"]
        for field in (
            "selected_constituent_paths",
            "daily_selected_constituent_checkpoints",
            "generic_observation_keys",
        ):
            if coverage[field] != expected[field]:
                raise PeadDailyLedgerError(
                    f"primary daily coverage {field} differs from frozen protocol"
                )
        if coverage["formation_checkpoints"] != expected[
            "exhaustive_formation_checkpoints"
        ]:
            raise PeadDailyLedgerError("formation coverage differs from frozen protocol")
        source_blockers = ledger_payload.get("blockers")
        if not isinstance(source_blockers, list):
            raise PeadDailyLedgerError("modeled ledger blockers are malformed")
        blockers = set(source_blockers)
        blockers.discard("daily_mark_to_market_path_not_implemented")
        blockers.update(
            {
                "candidate_distribution_receivable_payment_unproven",
                "daily_path_is_modeled_not_broker_evidence",
                "pooled_daily_scope_not_full_eight_cell_family",
            }
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": CANDIDATE_ID,
            "evidence_class": "primary_daily_modeled_accounting_nonqualifying",
            "daily_mark_to_market_complete": True,
            "independent_money_path_reconciled": False,
            "qualifying_evidence": False,
            "paper_execution_evidence": False,
            "promotion_allowed": False,
            "bindings": {
                "source_report_file_sha256": ledger_payload["bindings"][
                    "source_report_file_sha256"
                ],
                "combined_data_snapshot_hash": ledger_payload["bindings"][
                    "combined_data_snapshot_hash"
                ],
                "economic_return_inputs_hash": ledger_payload["bindings"][
                    "economic_return_inputs_hash"
                ],
                "modeled_execution_ledger_hash": modeled_ledger["artifact_hash"],
                "daily_input_snapshot_hash": daily_inputs["artifact_hash"],
                "daily_protocol_hash": protocol["artifact_hash"],
            },
            "accounting_claim": {
                "claim": "synthetic_split_normalized_daily_cohort_accounting_only",
                "distribution_balance": "candidate_receivable_not_settled_cash",
                "broker": None,
                "account_id": None,
                "orders": None,
                "fills": None,
                "quotes": None,
            },
            "formation_states": formation_states,
            "daily_constituent_states": daily_states,
            "cohort_daily_states": cohort_states,
            "coverage": coverage,
            "blockers": sorted(blockers),
        }
        return {"artifact_hash": content_hash(payload), "payload": _plain(payload)}


def validate_primary_daily_ledger(
    document: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
    modeled_ledger: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify content identity and reproduce every state from bound sources."""
    payload = _verified_wrapper(document, "primary daily ledger")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PeadDailyLedgerError("unsupported primary daily ledger schema")
    rebuilt = build_primary_daily_ledger(
        source_report, modeled_ledger, daily_inputs, protocol
    )
    if canonical_json(rebuilt) != canonical_json(document):
        raise PeadDailyLedgerError("primary daily ledger differs from source rebuild")
    return _plain(document)


def replication_observations(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the already-validated primary artifact onto the generic schema."""
    payload = _verified_wrapper(document, "primary daily ledger")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PeadDailyLedgerError("unsupported primary daily ledger schema")
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
                    "session_date": row["formation_date"],
                    "ticker": row["ticker"],
                    "m_ticker": row["m_ticker"],
                    "permaticker": row["permaticker"],
                    "source_event_key": _plain(row["source_event_key"]),
                },
                "eligibility": True,
                "signal": float(_decimal(row["signal"], "signal")),
                "rank": float(row["rank"]),
                "target": float(_decimal(row["target"], "target")),
                "order": 0.0,
                "position": 0.0,
                "cash": 1.0,
                "fees": 0.0,
                "pnl": 0.0,
            }
        )
    for row in payload["daily_constituent_states"]:
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
                    "source_event_key": _plain(row["source_event_key"]),
                },
                "eligibility": True,
                "signal": float(_decimal(row["signal"], "signal")),
                "rank": float(row["rank"]),
                "target": float(_decimal(row["target"], "target")),
                "order": float(_decimal(row["order"], "order")),
                "position": float(_decimal(row["position"], "position")),
                "cash": float(_decimal(row["cohort_settled_cash"], "cash")),
                "fees": float(_decimal(row["cohort_cumulative_fees"], "fees")),
                "pnl": float(_decimal(row["cohort_cumulative_pnl"], "pnl")),
            }
        )
    return sorted(result, key=lambda row: canonical_json(row["key"]))


__all__ = [
    "CANDIDATE_ID",
    "PeadDailyLedgerError",
    "SCHEMA_VERSION",
    "build_primary_daily_ledger",
    "replication_observations",
    "validate_primary_daily_ledger",
]
