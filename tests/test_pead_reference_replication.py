from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from analysis.pead_reference_replication import (
    MONEY_PATH_FIELDS,
    PeadReferenceError,
    UNAVAILABLE_REPLICATION_FIELDS,
    build_reference_comparison,
    content_hash,
    reconstruct_events,
    reconstruct_portfolio_observations,
    run_reference_reconstruction,
    verify_reference_artifact,
)
from data.session_close_calendar import load_session_close_calendar_evidence
from data.pead_economic_evidence import (
    build_current_unproven_cash_distribution_semantics,
    build_empty_terminal_settlement_ledger,
)


ES_COLUMNS = [
    "m_ticker", "ticker", "currency_code", "per_end_date", "per_type",
    "act_rpt_date", "eps_mean_est", "eps_act", "eps_cnt_est",
    "eps_act_zacks_adj", "act_rpt_time", "act_rpt_code",
]
EEH_COLUMNS = [
    "m_ticker", "ticker", "currency_code", "per_end_date", "per_type",
    "obs_date", "eps_mean_est", "eps_cnt_est",
]


def _session_close_evidence():
    return load_session_close_calendar_evidence()


def _row(columns, **values):
    return {column: values.get(column) for column in columns}


def _es(**overrides):
    values = {
        "m_ticker": "ACME",
        "ticker": "ACME",
        "currency_code": "USD",
        "per_end_date": "2019-12-31",
        "per_type": "Q",
        "act_rpt_date": "2020-01-02",
        "eps_mean_est": 1.0,
        "eps_act": 1.2,
        "eps_cnt_est": 4,
        "eps_act_zacks_adj": 0.0,
        "act_rpt_time": "16:05",
        "act_rpt_code": "AMC",
    }
    values.update(overrides)
    return _row(ES_COLUMNS, **values)


def _eeh(**overrides):
    values = {
        "m_ticker": "ACME",
        "ticker": "ACME",
        "currency_code": "USD",
        "per_end_date": "2019-12-31",
        "per_type": "Q",
        "obs_date": "2019-12-31",
        "eps_mean_est": 1.0,
        "eps_cnt_est": 4,
    }
    values.update(overrides)
    return _row(EEH_COLUMNS, **values)


def _table(columns, rows):
    return {
        "columns": [{"name": column, "type": "text"} for column in columns],
        "rows": rows,
        "canonical_request": {"method": "GET"},
        "response_sha256": "a" * 64,
        "provider_metadata": {"pages": []},
    }


def _snapshot(*, es=None, eeh=None):
    payload = {
        "schema_version": "zacks_pead_snapshot.v1",
        "candidate_id": "pead-vq-locked-replication-v1",
        "source_id": "nasdaq-data-link-zacks",
        "evidence_class": "development_sample",
        "captured_at": "2026-07-13T20:00:00Z",
        "requested_window": {"start": "2020-01-01", "end": "2020-12-31"},
        "coverage": {"full_window": False, "table_ranges": {}, "blockers": []},
        "tables": {
            "ZACKS/ES": _table(ES_COLUMNS, es or [_es()]),
            "ZACKS/EEH": _table(EEH_COLUMNS, eeh or [_eeh()]),
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


class _Provider:
    def __init__(self, *, missing_exit=False, delisted=False):
        self.sessions = pd.DatetimeIndex(
            [
                "2019-11-29",
                "2019-12-24",
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
            ]
        )
        self.missing_exit = missing_exit
        self.delisted = delisted
        self.requested_price_fields = []

    def snapshot_version(self, tables):
        manifest = [
            {"table": table, "sha256": str(index) * 64, "bytes": index}
            for index, table in enumerate(sorted(set(tables)), start=1)
        ]
        digest = hashlib.sha256()
        for item in manifest:
            digest.update(
                f"{item['table']}:{item['sha256']}:{item['bytes']}\n".encode()
            )
        return {
            "version": digest.hexdigest(),
            "tables": manifest,
            "complete": True,
            "quality_flags": [],
        }

    def corporate_action_evidence(self, start, end):
        payload = {
            "schema_version": "sharadar_actions_evidence.v1",
            "acquisition_artifact_hash": "a" * 64,
            "source_snapshot_time": "2026-07-10 03:28:21 UTC",
            "parquet_sha256": "b" * 64,
            "raw_zip_sha256": "c" * 64,
            "row_count": 100,
            "min_date": "1997-12-31",
            "max_date": "2026-07-10",
            "required_window": {"start": start, "end": end},
            "complete": True,
            "blockers": [],
            "value_is_terminal_payout_per_share": False,
        }
        return {"artifact_hash": content_hash(payload), "payload": payload}

    def market_sessions(self, start, end):
        return self.sessions

    def market_session_close_calendar(self):
        return _session_close_evidence()

    def universe_asof(self, formation):
        return ["ACME"]

    def security_lifecycle(self, ticker):
        return {
            "ticker": ticker,
            "permaticker": 101,
            "isdelisted": "Y" if self.delisted else "N",
            "lastpricedate": pd.Timestamp(
                "2020-01-06" if self.delisted else "2020-01-08"
            ).date(),
            "sep_lastpricedate": (
                pd.Timestamp("2020-01-06").date() if self.delisted else None
            ),
        }

    def security_currency(self, ticker):
        return {"ticker": ticker, "currency": "USD"}

    def corporate_actions_for_tickers(self, tickers, start, end):
        del tickers, start, end
        return []

    def prices_bulk(self, tickers, start, end, *, field):
        self.requested_price_fields.append(field)
        if field == "close":
            values = {
                "2019-12-24": 2.0,
                "2020-01-02": 2.5,
                "2020-01-03": 2.75,
                "2020-01-06": 3.0,
                "2020-01-07": 3.75,
                "2020-01-08": 4.5,
            }
        elif field == "closeunadj":
            values = {
                "2019-12-24": 8.0,
                "2020-01-02": 10.0,
                "2020-01-03": 11.0,
                "2020-01-06": 12.0,
                "2020-01-07": 15.0,
                "2020-01-08": 18.0,
            }
        elif field == "closeadj":
            values = {
                "2019-12-24": 80.0,
                "2020-01-02": 100.0,
                "2020-01-03": 110.0,
                "2020-01-06": 12.0,
                "2020-01-07": 15.0,
                "2020-01-08": 18.0,
            }
        else:
            raise AssertionError(f"unexpected price field {field}")
        if self.missing_exit and field == "closeadj":
            del values["2020-01-07"]
        series = pd.Series(values, dtype=float)
        series.index = pd.DatetimeIndex(series.index)
        return {ticker: series.rename(ticker) for ticker in tickers}

    def prices_strict(self, ticker, start, end, field):
        return self.prices_bulk([ticker], start, end, field=field)[ticker]

    def daily_marketcaps_for_dates(self, tickers, formations):
        return {
            (ticker, pd.Timestamp(formation).date()): 1000.0
            for ticker in tickers
            for formation in formations
        }


def _primary_report(run):
    return {
        "candidate_id": "pead-vq-locked-replication-v1",
        "source_snapshot": {"artifact_hash": run.source_snapshot_hash},
        "combined_data_snapshot": run.combined_snapshot,
        "economic_return_inputs": run.economic_return_inputs,
        "research_manifest_binding": {"artifact_hash": "b" * 64},
        "normalization": deepcopy(run.normalized),
        "raw_portfolio_observations": [
            deepcopy(item) for item in run.portfolio_observations
        ],
        "coverage": deepcopy(run.coverage),
        "blockers": ["source_coverage_not_full_window"],
    }


def _identity(name):
    file_hash = ("1" if name == "primary" else "2") * 64
    files = [{"path": f"{name}.py", "sha256": file_hash}]
    return {
        "implementation_id": name,
        "code_hash": content_hash(files),
        "files": files,
    }


def _comparison(run, report):
    return build_reference_comparison(
        run,
        report,
        protocol_hash="b" * 64,
        primary_report_sha256="e" * 64,
        primary_implementation=_identity("primary"),
        reference_implementation=_identity("reference"),
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )


def test_reference_uses_strictly_prior_consensus_and_after_close_same_day_price():
    snapshot = _snapshot(
        eeh=[_eeh(), _eeh(obs_date="2020-01-02", eps_mean_est=99.0)]
    )
    provider = _Provider()
    run = run_reference_reconstruction(
        snapshot,
        provider,
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )
    event = run.normalized["eps_events"][0]
    assert event["consensus_obs_date"] == "2019-12-31"
    assert event["consensus"] == 1.0
    observation = run.portfolio_observations[0]
    assert observation["signal_preannouncement_close_split_normalized"] == 2.5


def test_reference_independently_reconstructs_and_compares_economic_cash_return():
    provider = _Provider()
    run = run_reference_reconstruction(
        _snapshot(),
        provider,
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
        cash_distribution_semantics=(
            build_current_unproven_cash_distribution_semantics(
                absolute_tolerance=0.01,
                relative_tolerance=0.005,
            )
        ),
        terminal_settlement_ledger=build_empty_terminal_settlement_ledger(),
    )
    observation = run.portfolio_observations[0]
    assert observation["economic_forward_return_candidate_1"] == pytest.approx(
        3.75 / 3.0 - 1.0
    )
    assert run.coverage["economic_return_reconstruction"][
        "mechanical_reconstruction_complete"
    ] is True
    comparison = _comparison(run, _primary_report(run))
    assert comparison["payload"]["comparison"]["signal_path_passed"] is True
    assert comparison["payload"]["bindings"]["economic_return_inputs"][
        "artifact_hash"
    ] == run.economic_return_inputs["artifact_hash"]
    assert observation[
        "signal_preannouncement_closeunadj_execution_evidence"
    ] == 10.0
    assert observation["signal"] == pytest.approx(0.08)
    assert observation["entry_date"] == "2020-01-06"
    assert observation["entry_close_split_normalized"] == 3.0
    assert observation["entry_closeunadj_execution_evidence"] == 12.0
    assert observation["entry_closeadj_diagnostic"] == 12.0
    assert observation["target_exit_date_1"] == "2020-01-07"
    assert observation["adjusted_forward_return_1"] == pytest.approx(0.25)
    assert observation["return_resolution_1"] == {
        "status": "resolved_diagnostic",
        "reason": None,
        "pricing_path": "SEP.closeadj_exact_global_sessions_diagnostic",
    }
    assert provider.requested_price_fields == ["close", "closeunadj", "closeadj"]
    assert run.coverage["security_lifecycle_complete"] is True


def test_reference_groups_same_date_distribution_components_for_adjustment_check():
    class SameDateDistributionProvider(_Provider):
        def corporate_actions_for_tickers(self, tickers, start, end):
            del tickers, start, end
            return [
                {
                    "date": "2020-01-07",
                    "action": "dividend",
                    "ticker": "ACME",
                    "name": "ACME Corp regular",
                    "value": 0.10,
                    "contraticker": None,
                    "contraname": None,
                },
                {
                    "date": "2020-01-07",
                    "action": "dividend",
                    "ticker": "ACME",
                    "name": "ACME Corp special",
                    "value": 0.15,
                    "contraticker": None,
                    "contraname": None,
                },
            ]

        def prices_bulk(self, tickers, start, end, *, field):
            result = super().prices_bulk(tickers, start, end, field=field)
            if field == "closeadj":
                for series in result.values():
                    series.loc[pd.Timestamp("2020-01-07")] = 16.0
            return result

    run = run_reference_reconstruction(
        _snapshot(),
        SameDateDistributionProvider(),
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
        cash_distribution_semantics=(
            build_current_unproven_cash_distribution_semantics(
                absolute_tolerance=1e-12,
                relative_tolerance=1e-12,
            )
        ),
        terminal_settlement_ledger=build_empty_terminal_settlement_ledger(),
    )

    resolution = run.portfolio_observations[0]["economic_return_resolution_1"]
    assert [row["amount"] for row in resolution["cash_distributions"]] == [
        0.10,
        0.15,
    ]
    assert resolution["cash_total"] == pytest.approx(0.25)
    assert all(
        row["adjustment_implied_amount"] == pytest.approx(0.25)
        and row["adjustment_absolute_error"] == pytest.approx(0.0)
        for row in resolution["cash_distributions"]
    )


def test_reference_uses_actual_early_close_for_announcement_visibility():
    snapshot = _snapshot(
        es=[
            _es(
                act_rpt_date="2019-12-24",
                act_rpt_time="14:00",
                act_rpt_code="DTM",
            )
        ],
        eeh=[_eeh(obs_date="2019-12-23")],
    )
    normalized = reconstruct_events(snapshot, consensus_abs_tolerance=0.01)
    provider = _Provider()

    observations, coverage = reconstruct_portfolio_observations(
        normalized,
        provider,
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
    )

    assert coverage["observed_early_close_sessions"] == 2
    assert observations[0]["signal_preannouncement_close_split_normalized"] == 2.0


def test_reference_rejects_raw_calendar_source_tampering():
    normalized = reconstruct_events(_snapshot(), consensus_abs_tolerance=0.01)
    provider = _Provider()
    evidence = deepcopy(_session_close_evidence())
    source_id = next(iter(evidence["source_documents"]))
    encoded = evidence["source_documents"][source_id]
    evidence["source_documents"][source_id] = encoded[:-1] + (
        "A" if encoded[-1] != "A" else "B"
    )
    provider.market_session_close_calendar = lambda: evidence

    with pytest.raises(PeadReferenceError, match="base64|raw document"):
        reconstruct_portfolio_observations(
            normalized,
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
        )


def test_reference_rejects_unproved_calendar_transcription():
    normalized = reconstruct_events(_snapshot(), consensus_abs_tolerance=0.01)
    provider = _Provider()
    evidence = deepcopy(_session_close_evidence())
    calendar = evidence["calendar"]
    calendar["payload"]["early_close_sessions"].append(
        {"date": "2019-12-23", "source_id": "ice-2017-2019"}
    )
    calendar["payload"]["early_close_sessions"].sort(
        key=lambda item: item["date"]
    )
    calendar["artifact_hash"] = content_hash(calendar["payload"])
    receipt = evidence["source_receipt"]
    receipt["payload"]["calendar_artifact_hash"] = calendar["artifact_hash"]
    receipt["artifact_hash"] = content_hash(receipt["payload"])
    provider.market_session_close_calendar = lambda: evidence

    with pytest.raises(PeadReferenceError, match="not proved|source union"):
        reconstruct_portfolio_observations(
            normalized,
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
        )


def test_reference_rejects_self_consistent_impossible_calendar_http_time():
    normalized = reconstruct_events(_snapshot(), consensus_abs_tolerance=0.01)
    provider = _Provider()
    evidence = deepcopy(_session_close_evidence())
    receipt = evidence["source_receipt"]
    receipt["payload"]["sources"][0]["http"]["date_utc"] = (
        "2099-01-01T00:00:00Z"
    )
    receipt["artifact_hash"] = content_hash(receipt["payload"])
    provider.market_session_close_calendar = lambda: evidence

    with pytest.raises(PeadReferenceError, match="clock skew"):
        reconstruct_portfolio_observations(
            normalized,
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
        )


def test_reference_fails_closed_without_session_close_calendar():
    normalized = reconstruct_events(_snapshot(), consensus_abs_tolerance=0.01)
    provider = _Provider()
    provider.market_session_close_calendar = None

    with pytest.raises(PeadReferenceError, match="session close times"):
        reconstruct_portfolio_observations(
            normalized,
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
        )


def test_global_session_horizon_never_slides_over_a_missing_ticker_row():
    run = run_reference_reconstruction(
        _snapshot(),
        _Provider(missing_exit=True),
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )
    assert len(run.portfolio_observations) == 1
    assert run.portfolio_observations[0]["adjusted_forward_return_1"] is None
    assert run.portfolio_observations[0]["return_resolution_1"] == {
        "status": "unresolved",
        "reason": "missing_exact_global_session_exit",
        "pricing_path": "SEP.closeadj_exact_global_sessions_diagnostic",
    }
    assert run.coverage["formation_exclusions"] == []
    assert run.coverage["horizon_return_exclusions"] == [
        {
            "formation_date": "2020-01-03",
            "ticker": "ACME",
            "m_ticker": "ACME",
            "horizon_sessions": 1,
            "target_exit_date": "2020-01-07",
            "reason": "missing_exact_global_session_exit",
        }
    ]


def test_delisting_before_target_is_unresolved_without_inventing_a_payout():
    run = run_reference_reconstruction(
        _snapshot(),
        _Provider(delisted=True),
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )
    observation = run.portfolio_observations[0]
    assert observation["adjusted_forward_return_1"] is None
    assert observation["return_resolution_1"] == {
        "status": "unresolved",
        "reason": "held_delisting_terminal_economics_unresolved",
        "pricing_path": "SEP.closeadj_exact_global_sessions_diagnostic",
    }
    assert run.coverage["security_lifecycle_diagnostics"]["ACME"] == {
        "status": "validated",
        "isdelisted": "Y",
        "permaticker": 101,
        "lastpricedate": "2020-01-06",
        "sep_lastpricedate": "2020-01-06",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"ticker": "OTHER"},
        {"permaticker": True},
        {"permaticker": 101.0},
        {"permaticker": 0},
        {"lastpricedate": "2020-01-08T00:00:00"},
        {"sep_lastpricedate": "not-a-date"},
        {"isdelisted": "Y", "sep_lastpricedate": None},
        {"isdelisted": "Y", "sep_lastpricedate": "2020-01-07"},
    ],
)
def test_reference_lifecycle_rejects_identity_coercion_and_date_conflicts(mutation):
    provider = _Provider()
    lifecycle = {
        "ticker": "ACME",
        "permaticker": 101,
        "isdelisted": "N",
        "lastpricedate": "2020-01-08",
        "sep_lastpricedate": None,
    }
    lifecycle.update(mutation)
    provider.security_lifecycle = lambda ticker: lifecycle

    run = run_reference_reconstruction(
        _snapshot(),
        provider,
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )

    assert run.coverage["security_lifecycle_complete"] is False
    assert run.coverage["security_lifecycle_diagnostics"]["ACME"] == {
        "status": "unresolved",
        "reason": "security_lifecycle_validation_failed",
    }


@pytest.mark.parametrize(
    "source_snapshot_time",
    ["2026-07-10 03:28:21", "2026-07-10T03:28:21-04:00", "not-a-time"],
)
def test_reference_actions_evidence_parses_and_requires_utc_source_time(
    source_snapshot_time,
):
    provider = _Provider()
    original = provider.corporate_action_evidence

    def invalid_time(start, end):
        evidence = original(start, end)
        evidence["payload"]["source_snapshot_time"] = source_snapshot_time
        evidence["artifact_hash"] = content_hash(evidence["payload"])
        return evidence

    provider.corporate_action_evidence = invalid_time
    with pytest.raises(PeadReferenceError, match="must identify UTC"):
        run_reference_reconstruction(
            _snapshot(),
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
            consensus_abs_tolerance=0.01,
        )


def test_reference_actions_evidence_derives_exact_range_coverage_claims():
    provider = _Provider()
    original = provider.corporate_action_evidence
    expected = [
        "actions_range_starts_after_required_window",
        "actions_range_ends_before_required_window",
    ]

    def incomplete(start, end):
        evidence = original(start, end)
        evidence["payload"].update(
            {
                "min_date": "2020-01-04",
                "max_date": "2020-01-10",
                "blockers": expected,
                "complete": False,
            }
        )
        evidence["artifact_hash"] = content_hash(evidence["payload"])
        return evidence

    provider.corporate_action_evidence = incomplete
    run = run_reference_reconstruction(
        _snapshot(),
        provider,
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )
    assert run.corporate_action_evidence["payload"]["blockers"] == expected
    assert run.corporate_action_evidence["payload"]["complete"] is False

    def reordered(start, end):
        evidence = incomplete(start, end)
        evidence["payload"]["blockers"] = list(reversed(expected))
        evidence["artifact_hash"] = content_hash(evidence["payload"])
        return evidence

    provider.corporate_action_evidence = reordered
    with pytest.raises(PeadReferenceError, match="observed coverage"):
        run_reference_reconstruction(
            _snapshot(),
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
            consensus_abs_tolerance=0.01,
        )

    def reversed_window(start, end):
        evidence = original(start, end)
        evidence["payload"]["required_window"] = {
            "start": "2020-01-04",
            "end": "2020-01-03",
        }
        evidence["artifact_hash"] = content_hash(evidence["payload"])
        return evidence

    provider.corporate_action_evidence = reversed_window
    with pytest.raises(PeadReferenceError, match="required window is invalid"):
        run_reference_reconstruction(
            _snapshot(),
            provider,
            start="2020-01-03",
            end="2020-01-03",
            horizons=[1],
            fresh_days=63,
            consensus_abs_tolerance=0.01,
        )


def test_duplicate_latest_consensus_vintage_fails_closed():
    normalized = reconstruct_events(
        _snapshot(eeh=[_eeh(), _eeh()]), consensus_abs_tolerance=0.01
    )
    assert normalized["eps_events"] == []
    assert normalized["eps_exclusions"][0]["reasons"] == [
        "duplicate_latest_consensus_vintage"
    ]


def test_comparison_is_deterministic_and_explicitly_not_money_path_evidence():
    run = run_reference_reconstruction(
        _snapshot(),
        _Provider(),
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )
    report = _primary_report(run)
    first = _comparison(run, report)
    second = _comparison(run, deepcopy(report))
    assert first == second
    assert run.combined_snapshot["payload"]["schema_version"] == (
        "pead_combined_data_snapshot.v4"
    )
    assert run.combined_snapshot["payload"]["session_close_evidence"] == (
        run.session_close_evidence
    )
    payload = verify_reference_artifact(first)
    assert payload["schema_version"] == "pead_reference_reconciliation.v4"
    assert payload["comparison"]["signal_path_passed"] is True
    assert payload["comparison"]["discrepancy_count"] == 0
    assert payload["qualifying_evidence"] is False
    assert payload["replication_evidence_eligible"] is False
    assert payload["replication_contract_inputs"][
        "unavailable_required_money_path_fields"
    ] == list(MONEY_PATH_FIELDS)
    assert payload["replication_contract_inputs"][
        "unavailable_required_replication_fields"
    ] == list(UNAVAILABLE_REPLICATION_FIELDS)


def test_comparison_preserves_primary_reference_discrepancy_and_tamper_detection():
    run = run_reference_reconstruction(
        _snapshot(),
        _Provider(),
        start="2020-01-03",
        end="2020-01-03",
        horizons=[1],
        fresh_days=63,
        consensus_abs_tolerance=0.01,
    )
    report = _primary_report(run)
    report["raw_portfolio_observations"][0]["signal"] += 0.1
    artifact = _comparison(run, report)
    assert artifact["payload"]["comparison"]["signal_path_passed"] is False
    assert artifact["payload"]["comparison"]["discrepancies"][0]["field"] == (
        "signal"
    )
    tampered = deepcopy(artifact)
    tampered["payload"]["comparison"]["discrepancy_count"] = 0
    with pytest.raises(PeadReferenceError, match="hash mismatch"):
        verify_reference_artifact(tampered)

    self_consistent = _comparison(run, _primary_report(run))
    self_consistent["payload"]["outputs"]["reference"][
        "portfolio_observations"
    ][0]["signal"] += 0.1
    self_consistent["artifact_hash"] = content_hash(self_consistent["payload"])
    with pytest.raises(PeadReferenceError, match="stored reference discrepancies"):
        verify_reference_artifact(self_consistent)


def test_reference_module_has_no_dependency_on_primary_pead_module():
    path = Path(__file__).parents[1] / "analysis" / "pead_reference_replication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "analysis.pead_replication" not in imported
