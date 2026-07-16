from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from analysis.pead_replication import (
    CANDIDATE_ID,
    PeadReplicationError,
    REPORT_SCHEMA_VERSION,
    WAREHOUSE_RETURN_TABLES,
    _locked_factor_slice,
    _locked_horizon_portfolio,
    build_combined_data_snapshot,
    build_research_manifest_binding,
    build_replication_report,
    collect_replication_observations,
    content_hash,
    normalize_source_events,
    validate_snapshot_document,
)
from analysis.independent_replication import (
    ImplementationIdentity,
    IndependentReplicationContract,
    NUMERIC_FIELDS,
    NumericTolerance,
    reconcile_implementations,
)
from data.pit_warehouse import PitWarehouse
from data.pead_economic_evidence import (
    build_current_unproven_cash_distribution_semantics,
    build_empty_terminal_settlement_ledger,
)
from data.session_close_calendar import load_session_close_calendar_evidence
from scripts.pead_replication import (
    RESEARCH_PACKAGE,
    _load_research_manifest_binding,
    main,
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


def _session_close_calendar():
    return load_session_close_calendar_evidence()


def _row(columns, **values):
    return {column: values.get(column) for column in columns}


def _table(columns, rows):
    return {
        "columns": [{"name": column, "type": "text"} for column in columns],
        "rows": rows,
        "canonical_request": {
            "method": "GET",
            "url": "https://data.nasdaq.com/api/v3/datatables/example.json",
            "params": {"qopts.export": "true"},
        },
        "response_sha256": "a" * 64,
        "provider_metadata": {"pages": 1},
    }


def _es(**overrides):
    values = {
        "m_ticker": "AAPL",
        "ticker": "AAPL",
        "currency_code": "USD",
        "per_end_date": "2019-12-31",
        "per_type": "Q",
        "act_rpt_date": "2020-01-02",
        "eps_mean_est": 1.0,
        "eps_act": 1.2,
        "eps_cnt_est": 4,
        "eps_act_zacks_adj": 0.0,
        "act_rpt_time": "08:00",
        "act_rpt_code": "BTO",
    }
    values.update(overrides)
    return _row(ES_COLUMNS, **values)


def _eeh(**overrides):
    values = {
        "m_ticker": "AAPL",
        "ticker": "AAPL",
        "currency_code": "USD",
        "per_end_date": "2019-12-31",
        "per_type": "Q",
        "obs_date": "2019-12-31",
        "eps_mean_est": 1.0,
        "eps_cnt_est": 4,
    }
    values.update(overrides)
    return _row(EEH_COLUMNS, **values)


def _warehouse_snapshot():
    manifest = [
        {"table": table, "sha256": str(index) * 64, "bytes": index}
        for index, table in enumerate(sorted(WAREHOUSE_RETURN_TABLES), start=1)
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


def _corporate_action_evidence(start="2020-01-01", end="2020-12-31"):
    payload = {
        "schema_version": "sharadar_actions_evidence.v1",
        "acquisition_artifact_hash": "a" * 64,
        "source_snapshot_time": "2026-07-10 03:28:21 UTC",
        "parquet_sha256": "b" * 64,
        "raw_zip_sha256": "c" * 64,
        "row_count": 669_891,
        "min_date": "1997-12-31",
        "max_date": "2026-07-10",
        "required_window": {"start": start, "end": end},
        "complete": True,
        "blockers": [],
        "value_is_terminal_payout_per_share": False,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _manifest_binding():
    return build_research_manifest_binding(
        candidate_file_name="candidate_specification.json",
        candidate_file_sha256="a" * 64,
        candidate_schema_version="candidate_specification.v1",
        source_file_name="source_manifest.json",
        source_file_sha256="b" * 64,
        source_schema_version="pead_source_manifest.v1",
    )


def _passing_reconciliation(*, data_snapshot_hash, protocol_hash):
    key = {"checkpoint": "pead-test"}
    contract = IndependentReplicationContract(
        protocol_hash=protocol_hash,
        data_snapshot_hash=data_snapshot_hash,
        primary=ImplementationIdentity("pead-primary-test", "c" * 64),
        replication=ImplementationIdentity("pead-replication-test", "d" * 64),
        expected_observation_keys=[key],
        tolerances={
            field: NumericTolerance(absolute=0.0, relative=0.0)
            for field in NUMERIC_FIELDS
        },
    )
    observation = {
        "key": key,
        "eligibility": True,
        **{field: 0.0 for field in NUMERIC_FIELDS},
    }
    return reconcile_implementations(
        contract,
        primary_observations=[observation],
        replication_observations=[observation],
    )


def _document(*, es=None, eeh=None, full=True, blockers=None, start="2020-01-01",
              end="2020-12-31"):
    payload = {
        "schema_version": "zacks_pead_snapshot.v1",
        "candidate_id": CANDIDATE_ID,
        "source_id": "nasdaq-data-link-zacks",
        "evidence_class": "historical_replication" if full else "development_sample",
        "captured_at": "2026-07-13T18:00:00Z",
        "requested_window": {"start": start, "end": end},
        "coverage": {
            "full_window": full,
            "table_ranges": {
                "ZACKS/ES": {
                    "date_columns": ["act_rpt_date"],
                    "min_date": start,
                    "max_date": end,
                    "row_count": len(es or [_es()]),
                },
                "ZACKS/EEH": {
                    "date_columns": ["obs_date"],
                    "min_date": start,
                    "max_date": end,
                    "row_count": len(eeh or [_eeh()]),
                },
            },
            "blockers": list(blockers or []),
        },
        "tables": {
            "ZACKS/ES": _table(ES_COLUMNS, es or [_es()]),
            "ZACKS/EEH": _table(EEH_COLUMNS, eeh or [_eeh()]),
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def test_snapshot_hash_exact_schema_and_partial_coverage_blockers():
    document = _document(full=False, blockers=["sample_only"])
    snapshot = validate_snapshot_document(
        document, start="2020-01-01", end="2020-12-31"
    )
    assert snapshot.full_window is False
    assert snapshot.coverage_blockers == (
        "sample_only", "source_coverage_not_full_window"
    )

    document["payload"]["tables"]["ZACKS/ES"]["rows"][0]["eps_act"] = 9.0
    with pytest.raises(PeadReplicationError, match="hash mismatch"):
        validate_snapshot_document(document, start="2020-01-01", end="2020-12-31")


def test_latest_strictly_prior_vintage_and_es_crosscheck_are_enforced():
    document = _document(
        eeh=[
            _eeh(obs_date="2019-12-20", eps_mean_est=0.9),
            _eeh(obs_date="2019-12-31", eps_mean_est=1.0),
            _eeh(obs_date="2020-01-02", eps_mean_est=99.0),
        ]
    )
    snapshot = validate_snapshot_document(
        document, start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    event = normalized["eps_events"][0]
    assert event["consensus_obs_date"] == "2019-12-31"
    assert event["consensus"] == 1.0
    assert event["actual"] == 1.2
    assert event["unscaled_forecast_error"] == pytest.approx(0.2)

    mismatch = _document(eeh=[_eeh(eps_mean_est=0.99)])
    snapshot = validate_snapshot_document(
        mismatch, start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.005)
    assert normalized["eps_events"] == []
    assert normalized["eps_exclusions"][0]["reasons"] == [
        "surprise_consensus_crosscheck_mismatch"
    ]


def test_consensus_requires_at_least_two_analysts():
    document = _document(eeh=[_eeh(eps_cnt_est=1)])
    snapshot = validate_snapshot_document(
        document, start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    assert normalized["eps_events"] == []
    assert "insufficient_vintage_analyst_count" in (
        normalized["eps_exclusions"][0]["reasons"]
    )


@pytest.mark.parametrize(
    ("report_time", "report_code"),
    [("10:00", "BTO"), ("08:00", "AMC"), ("16:30", "DTM")],
)
def test_report_time_code_inconsistency_is_preserved_as_exclusion(
    report_time, report_code
):
    document = _document(es=[_es(act_rpt_time=report_time, act_rpt_code=report_code)])
    snapshot = validate_snapshot_document(
        document, start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    assert not normalized["eps_events"]
    assert "report_time_code_mismatch" in normalized["eps_exclusions"][0]["reasons"]


class FakeProvider:
    def __init__(self, *, announcement_code="BTO"):
        self.sessions = pd.bdate_range("2019-12-30", "2020-03-31")
        self.sessions = self.sessions[self.sessions != pd.Timestamp("2020-01-01")]
        self.sessions = pd.DatetimeIndex(
            sorted(
                {
                    *self.sessions,
                    pd.Timestamp("2019-11-29"),
                    pd.Timestamp("2019-12-24"),
                }
            )
        )
        self.raw = pd.Series(
            np.arange(len(self.sessions), dtype=float) + 10.0,
            index=self.sessions,
        )
        self.unadjusted = self.raw * 4.0
        self.adjusted = pd.Series(
            np.arange(len(self.sessions), dtype=float) + 100.0,
            index=self.sessions,
        )
        self.announcement_code = announcement_code
        self.snapshot_requests = []

    def snapshot_version(self, tables):
        self.snapshot_requests.append(tuple(tables))
        return _warehouse_snapshot()

    def market_sessions(self, start, end):
        return self.sessions[(self.sessions >= start) & (self.sessions <= end)]

    def market_session_close_calendar(self):
        return _session_close_calendar()

    def universe_asof(self, date):
        return ["AAPL"]

    def prices_bulk(self, tickers, start, end, *, field="closeadj"):
        series = {
            "close": self.raw,
            "closeunadj": self.unadjusted,
            "closeadj": self.adjusted,
        }[field]
        return {ticker: series[(series.index >= start) & (series.index <= end)]
                for ticker in tickers}

    def daily_marketcaps_for_dates(self, tickers, dates):
        return {(ticker, pd.Timestamp(value).date()): 1_000_000.0
                for ticker in tickers for value in dates}

    def security_lifecycle(self, ticker):
        return {
            "ticker": ticker,
            "permaticker": 1,
            "isdelisted": "N",
            "lastpricedate": pd.Timestamp("2026-07-10").date(),
            "sep_lastpricedate": None,
        }

    def corporate_action_evidence(self, start, end):
        return _corporate_action_evidence(start, end)

    def corporate_actions_for_tickers(self, tickers, start, end):
        del tickers, start, end
        return []

    def security_currency(self, ticker):
        return {"ticker": ticker, "currency": "USD"}


def test_observed_session_formation_t_plus_one_and_scaled_signal():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    frame, coverage = collect_replication_observations(
        normalized,
        FakeProvider(),
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        fresh_days=63,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["date"] == pd.Timestamp("2020-01-02")
    assert frame.iloc[0]["entry_date"] == pd.Timestamp("2020-01-03")
    assert frame.iloc[0]["m_ticker"] == "AAPL"
    assert frame.iloc[0]["entry_close_split_normalized"] == FakeProvider().raw.loc[
        "2020-01-03"
    ]
    assert frame.iloc[0]["entry_closeunadj_execution_evidence"] == (
        FakeProvider().unadjusted.loc["2020-01-03"]
    )
    assert frame.iloc[0]["signal_preannouncement_close_split_normalized"] == (
        FakeProvider().raw.loc["2019-12-31"]
    )
    assert frame.iloc[0][
        "signal_preannouncement_closeunadj_execution_evidence"
    ] == FakeProvider().unadjusted.loc["2019-12-31"]
    # Pre-announcement close is 2019-12-31, not the report-day close.
    expected = (1.2 - 1.0) / FakeProvider().raw.loc["2019-12-31"]
    assert frame.iloc[0]["sue"] == pytest.approx(expected)
    assert coverage["portfolio_observations"] == 1


def test_early_close_makes_same_day_close_visible_after_1300_et():
    snapshot = validate_snapshot_document(
        _document(
            es=[
                _es(
                    per_end_date="2019-09-30",
                    act_rpt_date="2019-12-24",
                    act_rpt_time="14:00",
                    act_rpt_code="DTM",
                )
            ],
            eeh=[_eeh(per_end_date="2019-09-30", obs_date="2019-12-23")],
            start="2019-01-01",
        ),
        start="2019-01-01",
        end="2020-12-31",
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    provider = FakeProvider()

    frame, coverage = collect_replication_observations(
        normalized,
        provider,
        start="2020-01-01",
        end="2020-02-28",
        horizons=[1],
        fresh_days=63,
    )

    assert coverage["observed_early_close_sessions"] == 2
    assert frame.iloc[0]["signal_preannouncement_close_split_normalized"] == (
        provider.raw.loc["2019-12-24"]
    )


def test_calendar_cannot_invent_an_early_close_absent_from_official_bytes():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    forged = deepcopy(_session_close_calendar())
    calendar = forged["calendar"]
    calendar["payload"]["early_close_sessions"].append(
        {"date": "2020-01-02", "source_id": "ice-2018-2020"}
    )
    calendar["payload"]["early_close_sessions"].sort(key=lambda row: row["date"])
    calendar["artifact_hash"] = content_hash(calendar["payload"])
    receipt = forged["source_receipt"]
    receipt["payload"]["calendar_artifact_hash"] = calendar["artifact_hash"]
    receipt["artifact_hash"] = content_hash(receipt["payload"])
    provider = FakeProvider()
    provider.market_session_close_calendar = lambda: forged

    with pytest.raises(PeadReplicationError, match="archived official source"):
        collect_replication_observations(
            normalized,
            provider,
            start="2020-01-01",
            end="2020-01-31",
            horizons=[1],
            fresh_days=63,
        )


def test_primary_rejects_self_consistent_impossible_calendar_http_time():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    forged = deepcopy(_session_close_calendar())
    receipt = forged["source_receipt"]
    receipt["payload"]["sources"][0]["http"]["date_utc"] = (
        "2099-01-01T00:00:00Z"
    )
    receipt["artifact_hash"] = content_hash(receipt["payload"])
    provider = FakeProvider()
    provider.market_session_close_calendar = lambda: forged

    with pytest.raises(PeadReplicationError, match="clock skew"):
        collect_replication_observations(
            normalized,
            provider,
            start="2020-01-01",
            end="2020-01-31",
            horizons=[1],
            fresh_days=63,
        )


def test_missing_session_close_calendar_fails_closed():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    provider = FakeProvider()
    provider.market_session_close_calendar = None

    with pytest.raises(PeadReplicationError, match="session close times"):
        collect_replication_observations(
            normalized,
            provider,
            start="2020-01-01",
            end="2020-01-31",
            horizons=[1],
            fresh_days=63,
        )


def test_lifecycle_evidence_preserves_matching_delisted_sep_date():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    provider = FakeProvider()
    provider.security_lifecycle = lambda ticker: {
        "ticker": ticker,
        "permaticker": 101,
        "isdelisted": "Y",
        "lastpricedate": "2026-07-10",
        "sep_lastpricedate": "2026-07-10",
    }

    _, coverage = collect_replication_observations(
        normalized,
        provider,
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        fresh_days=63,
    )

    assert coverage["security_lifecycle_complete"] is True
    assert coverage["security_lifecycle_diagnostics"]["AAPL"] == {
        "status": "validated",
        "isdelisted": "Y",
        "permaticker": 101,
        "lastpricedate": "2026-07-10",
        "sep_lastpricedate": "2026-07-10",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"ticker": "MSFT"},
        {"permaticker": True},
        {"permaticker": 1.0},
        {"permaticker": 0},
        {"lastpricedate": "2026-07-10T00:00:00"},
        {"sep_lastpricedate": "not-a-date"},
        {"isdelisted": "Y", "sep_lastpricedate": None},
        {"isdelisted": "Y", "sep_lastpricedate": "2026-07-09"},
    ],
)
def test_lifecycle_evidence_rejects_coercion_identity_and_date_conflicts(mutation):
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    provider = FakeProvider()
    lifecycle = {
        "ticker": "AAPL",
        "permaticker": 101,
        "isdelisted": "N",
        "lastpricedate": "2026-07-10",
        "sep_lastpricedate": None,
    }
    lifecycle.update(mutation)
    provider.security_lifecycle = lambda ticker: lifecycle

    _, coverage = collect_replication_observations(
        normalized,
        provider,
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        fresh_days=63,
    )

    assert coverage["security_lifecycle_complete"] is False
    assert coverage["security_lifecycle_diagnostics"]["AAPL"] == {
        "status": "unresolved",
        "reason": "security_lifecycle_validation_failed",
    }


def test_t_plus_one_requires_exact_positive_unadjusted_sep_close():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    provider = FakeProvider()
    provider.raw = provider.raw.drop(pd.Timestamp("2020-01-03"))
    frame, coverage = collect_replication_observations(
        normalized,
        provider,
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        fresh_days=63,
    )
    assert frame.empty
    assert coverage["formation_exclusions"] == [
        {
            "formation_date": "2020-01-02",
            "ticker": "AAPL",
                "reason": "missing_exact_t_plus_1_split_normalized_entry",
        }
    ]


def test_forward_horizon_uses_exact_global_session_and_never_slides():
    snapshot = validate_snapshot_document(
        _document(), start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    provider = FakeProvider()
    required_exit = pd.Timestamp("2020-01-06")
    assert required_exit in provider.sessions
    provider.adjusted = provider.adjusted.drop(required_exit)

    frame, coverage = collect_replication_observations(
        normalized,
        provider,
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        fresh_days=63,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["target_exit_date_1"] == required_exit
    assert pd.isna(row["fwd_1"])
    assert row["return_resolution_1"] == {
        "status": "unresolved",
        "reason": "missing_exact_global_session_exit",
        "pricing_path": "SEP.closeadj_exact_global_sessions_diagnostic",
    }
    assert coverage["horizon_return_exclusions"] == [
        {
            "formation_date": "2020-01-02",
            "ticker": "AAPL",
            "m_ticker": "AAPL",
            "horizon_sessions": 1,
            "target_exit_date": "2020-01-06",
            "reason": "missing_exact_global_session_exit",
        }
    ]


def test_horizon_missingness_does_not_rerank_or_contaminate_other_horizon():
    rows = []
    for index in range(10):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-02"),
                "name": f"T{index}",
                "m_ticker": f"M{index:02d}",
                "sue": float(index),
                "fwd_1": float(index) / 100,
                "fwd_2": (np.nan if index == 9 else float(index) / 100),
            }
        )
    ranked, _ = _locked_factor_slice(pd.DataFrame(rows))

    one, one_coverage = _locked_horizon_portfolio(
        ranked, horizon=1, quantile=0.2
    )
    two, two_coverage = _locked_horizon_portfolio(
        ranked, horizon=2, quantile=0.2
    )

    assert one_coverage["formation_dates_admitted_to_inference"] == 1
    assert set(one["m_ticker"]) == {"M00", "M01", "M08", "M09"}
    assert two.empty
    assert two_coverage["excluded_unresolved_cohorts"][0][
        "unresolved_m_tickers"
    ] == ["M09"]


def test_locked_factor_slice_requires_ten_names_per_formation():
    rows = []
    for date_value, count in (("2020-01-02", 9), ("2020-02-03", 10)):
        for index in range(count):
            rows.append({
                "date": pd.Timestamp(date_value),
                "name": f"T{index}",
                "m_ticker": f"M{index:02d}",
                "sue": float(index),
                "fwd_1": 0.0,
            })
    filtered, coverage = _locked_factor_slice(pd.DataFrame(rows))
    assert set(filtered["date"]) == {pd.Timestamp("2020-02-03")}
    assert coverage["formation_dates_before_floor"] == 2
    assert coverage["formation_dates_after_floor"] == 1
    assert coverage["excluded_formation_dates"] == [
        {"date": "2020-01-02", "eligible_names": 9}
    ]


def test_locked_signal_ties_use_ascending_stable_m_ticker_not_ticker_order():
    rows = [
        {
            "date": pd.Timestamp("2020-01-02"),
            "name": f"TICKER-{10 - index:02d}",
            "m_ticker": f"M{index:02d}",
            "sue": 1.0,
        }
        for index in reversed(range(10))
    ]
    filtered, _ = _locked_factor_slice(pd.DataFrame(rows))
    assert filtered["m_ticker"].tolist() == [f"M{index:02d}" for index in range(10)]
    assert filtered["_pead_signal_order"].tolist() == list(range(10))

    duplicated = pd.concat([pd.DataFrame(rows), pd.DataFrame([rows[0]])])
    with pytest.raises(PeadReplicationError, match="duplicate m_ticker"):
        _locked_factor_slice(duplicated)


def test_amc_uses_same_day_close_only_when_close_strictly_precedes_release():
    document = _document(
        es=[_es(act_rpt_date="2020-01-02", act_rpt_time="16:30", act_rpt_code="AMC")]
    )
    snapshot = validate_snapshot_document(
        document, start="2020-01-01", end="2020-12-31"
    )
    normalized = normalize_source_events(snapshot, consensus_abs_tolerance=0.0)
    frame, _ = collect_replication_observations(
        normalized,
        FakeProvider(),
        start="2020-02-01",
        end="2020-02-28",
        horizons=[1],
        fresh_days=63,
    )
    assert len(frame) == 1
    expected = (1.2 - 1.0) / FakeProvider().raw.loc["2020-01-02"]
    assert frame.iloc[0]["sue"] == pytest.approx(expected)


def test_partial_snapshot_generates_nonqualifying_report_not_exception():
    snapshot = validate_snapshot_document(
        _document(full=False), start="2020-01-01", end="2020-12-31"
    )
    provider = FakeProvider()
    binding = _manifest_binding()
    report = build_replication_report(
        snapshot,
        provider,
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        cost_bps=30,
        fresh_days=63,
        quantile=0.2,
        winsor_fraction=0.01,
        consensus_abs_tolerance=0.0,
        research_manifest_binding=binding,
    )
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["status"] == "blocked_or_sample"
    assert report["qualifying_evidence"] is False
    assert report["source_snapshot"]["legacy_sf1_or_events_used"] is False
    assert "source_coverage_not_full_window" in report["blockers"]
    assert report["raw_portfolio_observations"][0]["m_ticker"] == "AAPL"
    assert report["raw_portfolio_observations"][0][
        "entry_close_split_normalized"
    ] == (
        provider.raw.loc["2020-01-03"]
    )
    assert report["raw_portfolio_observations"][0][
        "entry_closeunadj_execution_evidence"
    ] == provider.unadjusted.loc["2020-01-03"]
    assert report["raw_portfolio_observations"][0][
        "signal_preannouncement_close_split_normalized"
    ] == provider.raw.loc["2019-12-31"]
    assert report["source_snapshot"]["warehouse_return_tables"] == list(
        WAREHOUSE_RETURN_TABLES
    )
    assert report["source_snapshot"]["warehouse_return_snapshot"] == (
        _warehouse_snapshot()
    )
    assert report["source_snapshot"]["warehouse_snapshot_unchanged_during_run"]
    assert report["combined_data_snapshot"] == build_combined_data_snapshot(
        snapshot.artifact_hash,
        _warehouse_snapshot(),
        _corporate_action_evidence("2020-01-01", "2020-02-18"),
        _session_close_calendar(),
    )
    assert report["research_manifest_binding"] == binding
    assert provider.snapshot_requests == [
        WAREHOUSE_RETURN_TABLES,
        WAREHOUSE_RETURN_TABLES,
    ]


def test_mechanical_economic_return_clears_only_the_reconstruction_blocker():
    snapshot = validate_snapshot_document(
        _document(full=False), start="2020-01-01", end="2020-12-31"
    )
    provider = FakeProvider()
    report = build_replication_report(
        snapshot,
        provider,
        start="2020-01-01",
        end="2020-01-31",
        horizons=[1],
        cost_bps=30,
        fresh_days=63,
        quantile=0.2,
        winsor_fraction=0.01,
        consensus_abs_tolerance=0.0,
        research_manifest_binding=_manifest_binding(),
        cash_distribution_semantics=(
            build_current_unproven_cash_distribution_semantics(
                absolute_tolerance=0.01,
                relative_tolerance=0.005,
            )
        ),
        terminal_settlement_ledger=build_empty_terminal_settlement_ledger(),
    )

    economics = report["coverage"]["economic_return_reconstruction"]
    assert economics["holding_paths"] == 1
    assert economics["mechanically_resolved_paths"] == 1
    assert economics["mechanical_reconstruction_complete"] is True
    assert economics["qualification_ready"] is False
    assert "economic_return_reconstruction_incomplete" not in report["blockers"]
    assert "cash_distribution_semantics_source_missing" in report["blockers"]
    observation = report["raw_portfolio_observations"][0]
    expected = (
        provider.raw.loc["2020-01-06"] / provider.raw.loc["2020-01-03"] - 1.0
    )
    assert observation["economic_forward_return_candidate_1"] == pytest.approx(
        expected
    )
    assert observation["economic_return_resolution_1"]["cash_total"] == 0.0
    assert report["economic_return_inputs"]["payload"][
        "combined_data_snapshot_hash"
    ] == report["combined_data_snapshot"]["artifact_hash"]


def test_raw_generic_reconciliation_cannot_clear_pead_specific_blocker():
    snapshot = validate_snapshot_document(
        _document(full=False), start="2020-01-01", end="2020-12-31"
    )
    combined = build_combined_data_snapshot(
        snapshot.artifact_hash,
        _warehouse_snapshot(),
        _corporate_action_evidence("2020-01-01", "2020-02-18"),
        _session_close_calendar(),
    )
    binding = _manifest_binding()

    def report_for(evidence):
        return build_replication_report(
            snapshot,
            FakeProvider(),
            start="2020-01-01",
            end="2020-01-31",
            horizons=[1],
            cost_bps=30,
            fresh_days=63,
            quantile=0.2,
            winsor_fraction=0.01,
            consensus_abs_tolerance=0.0,
            independent_reconciliation=evidence,
            research_manifest_binding=binding,
            cash_distribution_semantics=(
                build_current_unproven_cash_distribution_semantics()
            ),
            terminal_settlement_ledger=build_empty_terminal_settlement_ledger(),
        )

    passing = _passing_reconciliation(
        data_snapshot_hash=combined["artifact_hash"],
        protocol_hash=binding["artifact_hash"],
    )
    report = report_for(passing)
    assert report["independent_reconciliation_hash"] is None
    assert (
        "independent_implementation_reconciliation_invalid" in report["blockers"]
    )

    zacks_only = _passing_reconciliation(
        data_snapshot_hash=snapshot.artifact_hash,
        protocol_hash=binding["artifact_hash"],
    )
    report = report_for(zacks_only)
    assert report["independent_reconciliation_hash"] is None
    assert (
        "independent_implementation_reconciliation_invalid"
        in report["blockers"]
    )

    wrong_protocol = _passing_reconciliation(
        data_snapshot_hash=combined["artifact_hash"],
        protocol_hash="e" * 64,
    )
    report = report_for(wrong_protocol)
    assert report["independent_reconciliation_hash"] is None
    assert (
        "independent_implementation_reconciliation_invalid"
        in report["blockers"]
    )

    raw_pead_receipt = json.loads(
        (RESEARCH_PACKAGE / "daily_money_path_reconciliation_v3.json").read_text()
    )
    report = report_for(raw_pead_receipt)
    assert report["independent_reconciliation_hash"] is None
    assert (
        "independent_implementation_reconciliation_invalid"
        in report["blockers"]
    )


def test_combined_snapshot_accepts_real_pitwarehouse_snapshot_receipt(tmp_path):
    for index, table in enumerate(WAREHOUSE_RETURN_TABLES, start=1):
        (tmp_path / f"{table}.parquet").write_bytes(bytes([index]) * index)
    receipt = PitWarehouse(tmp_path).snapshot_version(list(WAREHOUSE_RETURN_TABLES))
    combined = build_combined_data_snapshot(
        "f" * 64, receipt, _corporate_action_evidence(),
        _session_close_calendar(),
    )
    assert combined["payload"]["warehouse_return_snapshot"] == receipt
    assert [item["table"] for item in receipt["tables"]] == [
        "actions", "daily", "sep", "tickers"
    ]

    reordered = {**receipt, "tables": list(reversed(receipt["tables"]))}
    with pytest.raises(PeadReplicationError, match="table order"):
        build_combined_data_snapshot(
            "f" * 64, reordered, _corporate_action_evidence(),
            _session_close_calendar(),
        )


@pytest.mark.parametrize(
    "source_snapshot_time",
    ["2026-07-10 03:28:21", "2026-07-10T03:28:21+01:00", "not-a-time"],
)
def test_corporate_action_evidence_requires_parsed_utc_source_time(
    source_snapshot_time,
):
    evidence = _corporate_action_evidence()
    evidence["payload"]["source_snapshot_time"] = source_snapshot_time
    evidence["artifact_hash"] = content_hash(evidence["payload"])

    with pytest.raises(PeadReplicationError, match="timestamp|UTC"):
        build_combined_data_snapshot(
            "f" * 64, _warehouse_snapshot(), evidence,
            _session_close_calendar(),
        )


def test_corporate_action_evidence_derives_exact_range_blockers_and_completion():
    evidence = _corporate_action_evidence("1990-01-01", "2030-01-01")
    expected = [
        "actions_range_starts_after_required_window",
        "actions_range_ends_before_required_window",
    ]
    evidence["payload"]["blockers"] = expected
    evidence["payload"]["complete"] = False
    evidence["artifact_hash"] = content_hash(evidence["payload"])

    combined = build_combined_data_snapshot(
        "f" * 64, _warehouse_snapshot(), evidence,
        _session_close_calendar(),
    )
    assert combined["payload"]["corporate_action_evidence"] == evidence

    evidence["payload"]["blockers"] = list(reversed(expected))
    evidence["artifact_hash"] = content_hash(evidence["payload"])
    with pytest.raises(PeadReplicationError, match="observed date range"):
        build_combined_data_snapshot(
            "f" * 64, _warehouse_snapshot(), evidence,
            _session_close_calendar(),
        )

    evidence = _corporate_action_evidence("2021-01-01", "2020-01-01")
    evidence["artifact_hash"] = content_hash(evidence["payload"])
    with pytest.raises(PeadReplicationError, match="required window is reversed"):
        build_combined_data_snapshot(
            "f" * 64, _warehouse_snapshot(), evidence,
            _session_close_calendar(),
        )


def test_manifest_binding_hashes_exact_files_and_rejects_column_order_drift():
    candidate_path = RESEARCH_PACKAGE / "candidate_specification.json"
    source_path = RESEARCH_PACKAGE / "source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    document = {
        "payload": {
            "tables": {
                table_code: {
                    "columns": [
                        {"name": name, "type": "test"}
                        for name in entry["required_columns"]
                    ]
                }
                for table_code, entry in source["tables"].items()
            }
        }
    }
    binding = _load_research_manifest_binding(
        document,
        candidate_path=candidate_path,
        source_path=source_path,
    )
    assert binding["payload"]["candidate_specification"]["file_sha256"] == (
        hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    )
    assert binding["payload"]["source_manifest"]["file_sha256"] == (
        hashlib.sha256(source_path.read_bytes()).hexdigest()
    )

    columns = document["payload"]["tables"]["ZACKS/ES"]["columns"]
    columns[0], columns[1] = columns[1], columns[0]
    with pytest.raises(PeadReplicationError, match="exact column sequence"):
        _load_research_manifest_binding(
            document,
            candidate_path=candidate_path,
            source_path=source_path,
        )


def _frozen_cli_args(tmp_path, snapshot):
    return [
        "--snapshot", str(snapshot),
        "--warehouse-dir", str(tmp_path),
        "--start", "2015-01-01",
        "--end", "2024-09-30",
        "--output-json", str(tmp_path / "report.json"),
        "--consensus-abs-tolerance", "0.01",
    ]


def test_cli_duplicate_json_is_malformed_exit_two(tmp_path, capsys):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text('{"artifact_hash":"a","artifact_hash":"b"}')
    code = main(_frozen_cli_args(tmp_path, snapshot))
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "duplicate key" in json.loads(captured.err)["error"]


@pytest.mark.parametrize(
    "override",
    [
        ["--start", "2015-01-02"],
        ["--end", "2024-09-27"],
        ["--horizons", "21", "42"],
        ["--cost-bps", "29.99"],
        ["--fresh-days", "62"],
        ["--quantile", "0.25"],
        ["--winsor", "0.02"],
        ["--consensus-abs-tolerance", "0.009"],
    ],
)
def test_cli_rejects_every_frozen_target_configuration_drift(
    tmp_path, capsys, override
):
    code = main(_frozen_cli_args(tmp_path, tmp_path / "not-read.json") + override)
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "differs from frozen PEAD target" in json.loads(captured.err)["error"]
