from __future__ import annotations

import copy
from decimal import localcontext
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import analysis.pead_daily_inputs as daily_inputs
from analysis.pead_daily_inputs import (
    PeadDailyInputError,
    build_pead_daily_input_snapshot,
    validate_pead_daily_input_snapshot,
    verify_pead_daily_input_snapshot,
)
from data.pead_economic_evidence import content_hash


RESEARCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "pead_vq_locked_replication_v1"
)


def _warehouse_receipt(*, changed: bool = False) -> dict:
    rows = [
        {
            "table": table,
            "sha256": str(index) * 64,
            "bytes": index + (1 if changed and table == "sep" else 0),
        }
        for index, table in enumerate(daily_inputs.WAREHOUSE_TABLES, start=1)
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['table']}:{row['sha256']}:{row['bytes']}\n".encode()
        )
    return {
        "version": digest.hexdigest(),
        "tables": rows,
        "complete": True,
        "quality_flags": [],
    }


def _synthetic_upstream() -> tuple[dict, list[str]]:
    sessions = [
        value.date().isoformat()
        for value in pd.bdate_range("2020-01-03", periods=64)
    ]
    formation_rows = []
    selected_paths = []
    for horizon in (21, 63):
        for index in range(10):
            leg = "short" if index < 2 else "long" if index >= 8 else None
            row = {
                "cohort_id": f"pooled:2020-01-02:{horizon}",
                "formation_date": "2020-01-02",
                "entry_date": sessions[0],
                "exit_date": sessions[horizon],
                "horizon_sessions": horizon,
                "ticker": f"T{index:02d}",
                "m_ticker": f"M{index:02d}",
                "permaticker": 1000 + index,
                "rank": index + 1,
                "selected_leg": leg,
                "cohort_status": "admitted",
                "cohort_reason": None,
                "source_event_key": {
                    "m_ticker": f"M{index:02d}",
                    "per_end_date": "2019-12-31",
                    "per_type": "Q",
                },
                "signal": f"{index - 5:.18f}",
            }
            formation_rows.append(row)
            if leg is not None:
                selected_paths.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {
                            "selected_leg", "cohort_status", "cohort_reason"
                        }
                    }
                    | {"leg": leg}
                )
    receipt = _warehouse_receipt()
    return {
        "bindings": {
            "source_report_file_sha256": "a" * 64,
            "source_report_schema_version": "pead_replication_report.v6",
            "combined_data_snapshot_hash": "b" * 64,
            "economic_return_inputs_hash": "c" * 64,
            "modeled_execution_ledger_hash": "d" * 64,
            "independent_reference_artifact_hash": "e" * 64,
            "protocol_hash": "f" * 64,
            "warehouse_snapshot_version": receipt["version"],
            "warehouse_snapshot": receipt,
        },
        "formation_observations": formation_rows,
        "selected_paths": selected_paths,
    }, sessions


class _Provider:
    def __init__(
        self,
        sessions: list[str],
        *,
        missing_internal_bar: bool = False,
        source_changes: bool = False,
    ):
        self.sessions = pd.DatetimeIndex(sessions)
        self.missing_internal_bar = missing_internal_bar
        self.source_changes = source_changes
        self.snapshot_requests: list[tuple[str, ...]] = []
        self.price_requests: list[tuple[str, str]] = []

    def snapshot_version(self, tables):
        self.snapshot_requests.append(tuple(tables))
        changed = self.source_changes and len(self.snapshot_requests) > 1
        return _warehouse_receipt(changed=changed)

    def market_sessions(self, start, end):
        assert start == self.sessions[0].date().isoformat()
        assert end == self.sessions[-1].date().isoformat()
        return self.sessions

    def prices_strict(self, ticker, start, end, field):
        del start, end
        self.price_requests.append((ticker, field))
        values = pd.Series(
            [10.0 + index / 100 for index in range(len(self.sessions))],
            index=self.sessions,
            name=ticker,
        )
        if field == "closeadj":
            values = values + 50.0
        if self.missing_internal_bar and ticker == "T00":
            values = values.drop(self.sessions[10])
        return values

    def corporate_actions_for_tickers(self, tickers, start, end):
        del start, end
        assert tickers == ["T00", "T01", "T08", "T09"]
        return [
            {
                "date": self.sessions[10].date().isoformat(),
                "action": "dividend",
                "ticker": "T00",
                "name": "T00 INC",
                "value": 0.5,
                "contraticker": None,
                "contraname": None,
            }
        ]

    def security_currency(self, ticker):
        return {"ticker": ticker, "currency": "USD"}


def _build_synthetic(monkeypatch, **provider_options):
    upstream, sessions = _synthetic_upstream()
    monkeypatch.setattr(
        daily_inputs,
        "_validate_upstream_sources",
        lambda *args, **kwargs: copy.deepcopy(upstream),
    )
    provider = _Provider(sessions, **provider_options)
    artifact = build_pead_daily_input_snapshot({}, {}, {}, provider)
    return artifact, provider, upstream


def test_committed_sources_resolve_exact_independent_selected_path_union():
    report = json.loads(
        (RESEARCH_DIR / "development_sample_report_v6.json").read_text()
    )
    ledger = json.loads(
        (RESEARCH_DIR / "modeled_execution_ledger_v1.json").read_text()
    )
    reference = json.loads(
        (RESEARCH_DIR / "independent_reference_comparison_v5.json").read_text()
    )

    upstream = daily_inputs._validate_upstream_sources(
        report, ledger, reference, report_sha256=None
    )

    assert len(upstream["formation_observations"]) == 330
    assert len(upstream["selected_paths"]) == 88
    assert len(
        {
            (row["cohort_id"], row["m_ticker"])
            for row in upstream["selected_paths"]
        }
    ) == 88
    assert upstream["bindings"]["source_report_file_sha256"] == (
        "e2ae42ee3f6fad49748f23a40869e3b2461113fb31f4cb9016033e8560d678b7"
    )
    assert upstream["bindings"]["modeled_execution_ledger_hash"] == (
        ledger["artifact_hash"]
    )
    assert upstream["bindings"]["independent_reference_artifact_hash"] == (
        reference["artifact_hash"]
    )


def test_builds_deterministic_neutral_daily_slice(monkeypatch):
    artifact, provider, upstream = _build_synthetic(monkeypatch)
    payload = artifact["payload"]

    assert verify_pead_daily_input_snapshot(artifact) == artifact
    assert validate_pead_daily_input_snapshot(
        artifact,
        expected_bindings={
            "source_report_file_sha256": "a" * 64,
            "warehouse_snapshot": upstream["bindings"]["warehouse_snapshot"],
        },
    ) == artifact
    assert payload["coverage"] == {
        "formation_observations": 20,
        "selected_paths": 8,
        "selected_tickers": 4,
        "sessions": 64,
        "price_rows": 256,
        "action_rows": 1,
        "currency_rows": 4,
        "path_price_applications": 344,
        "paths_by_horizon": {"21": 4, "63": 4},
    }
    assert payload["distribution_semantics"] == {
        "action_date_role": "candidate_ex_date_unproven",
        "action_value_role": "candidate_split_normalized_cash_per_share_unproven",
        "adjustment_check_absolute_tolerance": "0.005000000000000000",
        "adjustment_check_relative_tolerance": "0.001000000000000000",
        "holding_interval": "entry_date_exclusive_exit_date_inclusive",
        "payment_date_available": False,
        "cash_settlement_allowed": False,
        "reinvestment": False,
    }
    assert payload["actions"][0]["value"] == "0.500000000000000000"
    assert provider.snapshot_requests == [
        daily_inputs.WAREHOUSE_TABLES,
        daily_inputs.WAREHOUSE_TABLES,
    ]
    assert {field for _, field in provider.price_requests} == {"close", "closeadj"}

    second, _, _ = _build_synthetic(monkeypatch)
    assert second == artifact
    with localcontext() as context:
        context.prec = 9
        third, _, _ = _build_synthetic(monkeypatch)
    assert third == artifact


def test_missing_internal_path_bar_fails_closed(monkeypatch):
    with pytest.raises(PeadDailyInputError, match="missing an internal strict price bar"):
        _build_synthetic(monkeypatch, missing_internal_bar=True)


def test_provider_source_change_during_reads_fails_closed(monkeypatch):
    with pytest.raises(PeadDailyInputError, match="changed during daily reads"):
        _build_synthetic(monkeypatch, source_changes=True)


def test_hash_tamper_and_self_rehashed_path_erasure_fail_closed(monkeypatch):
    artifact, _, _ = _build_synthetic(monkeypatch)

    unhashed = copy.deepcopy(artifact)
    unhashed["payload"]["prices"][0]["close"] = "99.000000000000000000"
    with pytest.raises(PeadDailyInputError, match="hash mismatch"):
        validate_pead_daily_input_snapshot(unhashed)

    erased = copy.deepcopy(artifact)
    erased["payload"]["prices"] = [
        row
        for row in erased["payload"]["prices"]
        if not (row["ticker"] == "T00" and row["date"] == erased["payload"]["sessions"][10])
    ]
    erased["artifact_hash"] = content_hash(erased["payload"])
    with pytest.raises(PeadDailyInputError, match="missing an internal price row"):
        validate_pead_daily_input_snapshot(erased)

    with pytest.raises(PeadDailyInputError, match="binding protocol_hash differs"):
        validate_pead_daily_input_snapshot(
            artifact, expected_bindings={"protocol_hash": "0" * 64}
        )


def test_source_report_mutation_cannot_reuse_committed_ledger_and_reference():
    report = json.loads(
        (RESEARCH_DIR / "development_sample_report_v6.json").read_text()
    )
    ledger = json.loads(
        (RESEARCH_DIR / "modeled_execution_ledger_v1.json").read_text()
    )
    reference = json.loads(
        (RESEARCH_DIR / "independent_reference_comparison_v5.json").read_text()
    )
    report["blockers"] = sorted(set(report["blockers"]) | {"synthetic_source_change"})

    with pytest.raises(PeadDailyInputError):
        daily_inputs._validate_upstream_sources(
            report, ledger, reference, report_sha256=None
        )


def test_partial_source_validation_request_is_rejected(monkeypatch):
    artifact, _, _ = _build_synthetic(monkeypatch)
    with pytest.raises(PeadDailyInputError, match="must be supplied together"):
        validate_pead_daily_input_snapshot(artifact, report={})
