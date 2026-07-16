from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from data.earnings_announcements import ZacksTablesClient
from data.pead_consensus_replay import (
    PROVIDER_NEUTRAL_CSV_ADAPTER,
    PROVIDER_NEUTRAL_JSON_ADAPTER,
    ZACKS_EEH_ADAPTER,
    PeadConsensusReplayError,
    build_pead_consensus_metric_profile,
    build_pead_consensus_replay,
    build_pead_consensus_source_manifest,
    canonical_json,
    content_hash,
    validate_pead_consensus_replay,
    verify_pead_consensus_replay,
)
from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
)
from data.sharadar_source_evidence import (
    PEAD_SECURITY_IDENTITY_SCHEMA_VERSION,
    PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION,
)


CANDIDATE = "pead-replay-test-v1"
KEYS = [
    {
        "cik": "0000320193",
        "fiscal_period_end": "2020-03-31",
        "fiscal_period_type": "Q",
    },
    {
        "cik": "0000789019",
        "fiscal_period_end": "2020-03-31",
        "fiscal_period_type": "Q",
    },
]
FIELD_MAP = {
    "provider_record_id": "record_id",
    "provider_security_id": "security_id",
    "ticker": "ticker",
    "cik": "cik",
    "fiscal_period_end": "period_end",
    "fiscal_period_type": "period_type",
    "provider_as_of_date": "as_of_date",
    "trusted_available_at_utc": "available_at",
    "availability_precision": "availability_precision",
    "consensus_value": "mean",
    "analyst_count": "analysts",
    "currency_code": "currency",
}


def _identity_snapshot(*, second_ticker: str = "MSFT"):
    rows = [
        {
            "cik": "0000320193",
            "permaticker": 1001,
            "ticker": "AAPL",
            "valid_from": "2010-01-01",
            "valid_through": "2030-12-31",
            "is_delisted": "N",
            "category": "Domestic Common Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
            "source_record_sha256": "1" * 64,
        },
        {
            "cik": "0000789019",
            "permaticker": 1002,
            "ticker": second_ticker,
            "valid_from": "2010-01-01",
            "valid_through": "2030-12-31",
            "is_delisted": "N",
            "category": "Domestic Common Stock",
            "exchange": "NASDAQ",
            "currency": "USD",
            "source_record_sha256": "2" * 64,
        },
    ]
    identities = []
    for row in rows:
        identity_id = content_hash({"schema_version": PEAD_SECURITY_IDENTITY_SCHEMA_VERSION, **row})
        identities.append({"identity_id": identity_id, **row})
    identities.sort(key=lambda row: row["identity_id"])
    dispositions = sorted(
        [
            {
                "source_record_sha256": row["source_record_sha256"],
                "disposition": "identity",
                "identity_id": row["identity_id"],
                "reason": None,
            }
            for row in identities
        ],
        key=lambda row: row["source_record_sha256"],
    )
    payload = {
        "schema_version": PEAD_SECURITY_IDENTITY_SNAPSHOT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE,
        "created_at_utc": "2026-07-14T11:00:00Z",
        "bindings": {
            "sharadar_source_snapshot_sha256": "a" * 64,
            "tickers_acquisition_sha256": "b" * 64,
            "tickers_parquet_sha256": "c" * 64,
        },
        "source_dispositions": dispositions,
        "identities": identities,
        "coverage": {
            "source_row_count": 2,
            "disposition_count": 2,
            "identity_count": 2,
            "identity_gap_count": 0,
            "complete": True,
        },
        "blockers": [],
        "qualification_allowed": True,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _universe(identity):
    source_ids = ["3" * 64, "4" * 64]
    query_hash = "d" * 64
    census = build_pead_event_census_receipt(
        raw_census_artifact_sha256="e" * 64,
        canonical_query_sha256=query_hash,
        source_record_ids=source_ids,
    )
    return build_pead_event_universe(
        candidate_id=CANDIDATE,
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start="2020-01-01",
        event_end="2020-12-31",
        bindings={
            "market_snapshot_sha256": "5" * 64,
            "identity_snapshot_sha256": identity["artifact_hash"],
            "candidate_specification_sha256": "6" * 64,
            "construction_code_sha256": "7" * 64,
            "canonical_query_sha256": query_hash,
        },
        census_receipt=census,
        census_dispositions=[
            {
                "source_record_id": source_id,
                "disposition": "expected_event",
                "event_id": canonical_event_id(key),
                "event_key": key,
                "reason": None,
            }
            for source_id, key in zip(source_ids, KEYS, strict=True)
        ],
    )


def _source(adapter=PROVIDER_NEUTRAL_JSON_ADAPTER):
    return build_pead_consensus_source_manifest(
        candidate_id=CANDIDATE,
        adapter_id=adapter,
        provider_id="test-pit-provider",
        dataset_id="test-consensus-export",
        evidence_class="historical_reconstruction",
        source_captured_at_utc="2026-07-14T13:00:00Z",
        provider_snapshot_at_utc="2026-07-14T12:30:00Z",
        canonical_query_sha256="8" * 64,
        records_path="$",
        field_map=FIELD_MAP,
    )


def _profile(source):
    payload = source["payload"]
    return build_pead_consensus_metric_profile(
        candidate_id=CANDIDATE,
        profile_id="adjusted-diluted-eps-v1",
        adapter_id=payload["adapter_id"],
        provider_id=payload["provider_id"],
        dataset_id=payload["dataset_id"],
        value_field="mean" if payload["adapter_id"] != ZACKS_EEH_ADAPTER else "eps_mean_est",
        analyst_count_field=(
            "analysts" if payload["adapter_id"] != ZACKS_EEH_ADAPTER else "eps_cnt_est"
        ),
        metric={
            "metric_id": "eps",
            "accounting_basis": "adjusted",
            "per_share_basis": "diluted",
            "scope": "continuing_operations",
            "canonical_share_basis": "split_restated",
            "unit": "currency_per_share",
            "metric_definition_sha256": "9" * 64,
        },
    )


def _json_raw(*, duplicate_first=False, second_ticker="MSFT"):
    first = (
        '{"record_id":"r1","security_id":"sid-aapl","ticker":"AAPL",'
        '"cik":"0000320193","period_end":"2020-03-31","period_type":"Q",'
        '"as_of_date":"2020-03-27","available_at":null,'
        '"availability_precision":"date","mean":1.2300e0,"analysts":4,'
        '"currency":"USD"}'
    )
    duplicate = (
        ',{"record_id":"r1b","security_id":"sid-aapl","ticker":"AAPL",'
        '"cik":"0000320193","period_end":"2020-03-31","period_type":"Q",'
        '"as_of_date":"2020-03-27","available_at":null,'
        '"availability_precision":"date","mean":1.24,"analysts":4,'
        '"currency":"USD"}'
        if duplicate_first
        else ""
    )
    second = (
        ',{"record_id":"r2","security_id":"sid-msft","ticker":"'
        + second_ticker
        + '","cik":"0000789019","period_end":"2020-03-31",'
        '"period_type":"Q","as_of_date":"2020-03-30","available_at":null,'
        '"availability_precision":"date","mean":2.5000,"analysts":7,'
        '"currency":"USD"}'
    )
    return ("[" + first + duplicate + second + "]").encode()


def _csv_raw():
    header = ",".join(FIELD_MAP.values())
    return (
        header
        + "\n"
        + "r1,sid-aapl,AAPL,0000320193,2020-03-31,Q,2020-03-27,,date,1.2300e0,4,USD\n"
        + "r2,sid-msft,MSFT,0000789019,2020-03-31,Q,2020-03-30,,date,2.5000,7,USD\n"
    ).encode()


def _trust(universe, identity, source, profile, raw):
    return {
        "trusted_event_universe_sha256s": {universe["artifact_hash"]},
        "trusted_identity_snapshot_sha256s": {identity["artifact_hash"]},
        "trusted_source_manifest_sha256s": {source["artifact_hash"]},
        "trusted_metric_profile_sha256s": {profile["artifact_hash"]},
        "trusted_raw_artifact_sha256s": {hashlib.sha256(raw).hexdigest()},
    }


def _build(raw=None, *, adapter=PROVIDER_NEUTRAL_JSON_ADAPTER, identity=None, **trust):
    raw = _json_raw() if raw is None else raw
    identity = _identity_snapshot() if identity is None else identity
    universe = _universe(identity)
    source = _source(adapter)
    profile = _profile(source)
    allowed = _trust(universe, identity, source, profile, raw)
    allowed.update(trust)
    receipt = build_pead_consensus_replay(
        raw_artifact=raw,
        event_universe=universe,
        identity_snapshot=identity,
        source_manifest=source,
        metric_profile=profile,
        **allowed,
    )
    return receipt, raw, universe, identity, source, profile, allowed


def test_json_replay_is_exhaustive_decimal_exact_and_authoritatively_reproducible():
    receipt, raw, _, _, _, _, trust = _build()
    payload = receipt["payload"]

    assert payload["qualification_allowed"] is True
    assert payload["blockers"] == []
    assert payload["counts"] == {
        "raw_record_count": 2,
        "ledger_record_count": 2,
        "matched_record_count": 2,
        "outside_universe_record_count": 0,
        "identity_gap_record_count": 0,
        "invalid_record_count": 0,
        "duplicate_record_count": 0,
        "expected_event_count": 2,
        "available_event_count": 2,
        "missing_event_count": 0,
        "normalized_vintage_count": 2,
    }
    values = sorted(
        event["vintages"][0]["consensus_value"]
        for event in payload["consensus_evidence"]["payload"]["event_records"]
    )
    assert values == ["1.23", "2.5"]
    for entry in payload["raw_record_ledger"]:
        assert entry["raw_record_sha256"] == content_hash(entry["source_locator"])
    assert verify_pead_consensus_replay(receipt, raw_artifact=raw, **trust) == receipt


def test_csv_adapter_replays_same_values_from_exact_csv_cells():
    raw = _csv_raw()
    receipt, _, _, _, _, _, trust = _build(raw, adapter=PROVIDER_NEUTRAL_CSV_ADAPTER)

    assert receipt["payload"]["qualification_allowed"] is True
    values = sorted(
        event["vintages"][0]["consensus_value"]
        for event in receipt["payload"]["consensus_evidence"]["payload"]["event_records"]
    )
    assert values == ["1.23", "2.5"]
    assert verify_pead_consensus_replay(receipt, raw_artifact=raw, **trust) == receipt


def test_external_trust_roots_are_required_for_qualification_and_verification():
    identity = _identity_snapshot()
    universe = _universe(identity)
    source = _source()
    profile = _profile(source)
    raw = _json_raw()
    trust = _trust(universe, identity, source, profile, raw)
    trust["trusted_raw_artifact_sha256s"] = set()
    receipt = build_pead_consensus_replay(
        raw_artifact=raw,
        event_universe=universe,
        identity_snapshot=identity,
        source_manifest=source,
        metric_profile=profile,
        **trust,
    )

    assert receipt["payload"]["qualification_allowed"] is False
    assert "raw_artifact_not_trusted" in receipt["payload"]["blockers"]
    assert verify_pead_consensus_replay(receipt, raw_artifact=raw, **trust) == receipt
    trusted = {**trust, "trusted_raw_artifact_sha256s": {hashlib.sha256(raw).hexdigest()}}
    with pytest.raises(PeadConsensusReplayError, match="does not reproduce"):
        verify_pead_consensus_replay(receipt, raw_artifact=raw, **trusted)


def test_rehashed_normalized_tamper_passes_structure_but_fails_byte_replay():
    receipt, raw, _, _, _, _, trust = _build()
    tampered = copy.deepcopy(receipt)
    consensus = tampered["payload"]["consensus_evidence"]
    vintage = consensus["payload"]["event_records"][0]["vintages"][0]
    vintage["consensus_value"] = "99"
    matching = next(
        entry
        for entry in tampered["payload"]["raw_record_ledger"]
        if entry["raw_record_sha256"] == vintage["raw_record_sha256"]
    )
    matching["normalized_vintage_sha256"] = content_hash(vintage)
    consensus["artifact_hash"] = content_hash(consensus["payload"])
    tampered["payload"]["bindings"]["consensus_evidence_sha256"] = consensus["artifact_hash"]
    tampered["artifact_hash"] = content_hash(tampered["payload"])

    assert validate_pead_consensus_replay(tampered) == tampered
    with pytest.raises(PeadConsensusReplayError, match="does not reproduce"):
        verify_pead_consensus_replay(tampered, raw_artifact=raw, **trust)


def test_wrong_raw_bytes_fail_authoritative_replay_even_if_parseable():
    receipt, raw, _, _, _, _, trust = _build()
    changed = raw.replace(b"1.2300e0", b"1.2400e0")

    with pytest.raises(PeadConsensusReplayError, match="does not reproduce"):
        verify_pead_consensus_replay(receipt, raw_artifact=changed, **trust)


def test_duplicate_natural_vintages_are_ledgered_and_excluded():
    receipt, *_ = _build(_json_raw(duplicate_first=True))
    payload = receipt["payload"]

    assert payload["qualification_allowed"] is False
    assert payload["counts"]["duplicate_record_count"] == 2
    assert payload["counts"]["missing_event_count"] == 1
    assert "raw_duplicate_vintages_present" in payload["blockers"]
    assert all(
        entry["reasons"] == ["natural_vintage_duplicate"]
        for entry in payload["raw_record_ledger"]
        if entry["disposition"] == "duplicate_natural_vintage"
    )


def test_dated_identity_mismatch_is_not_repaired_from_ticker_or_cik_alone():
    receipt, *_ = _build(_json_raw(second_ticker="MSFT.A"))
    payload = receipt["payload"]

    assert payload["counts"]["identity_gap_record_count"] == 1
    assert payload["counts"]["missing_event_count"] == 1
    assert "raw_identity_gaps_present" in payload["blockers"]


def test_generic_parsers_reject_duplicate_json_keys_and_csv_headers():
    duplicate_json = _json_raw().replace(
        b'"record_id":"r1"', b'"record_id":"r1","record_id":"forged"', 1
    )
    with pytest.raises(PeadConsensusReplayError, match="duplicate key"):
        _build(duplicate_json)

    duplicate_csv = _csv_raw().replace(b"record_id,security_id", b"record_id,record_id", 1)
    with pytest.raises(PeadConsensusReplayError, match="header is invalid"):
        _build(duplicate_csv, adapter=PROVIDER_NEUTRAL_CSV_ADAPTER)


def test_pathological_decimal_exponent_is_ledgered_invalid_without_expansion():
    raw = _json_raw().replace(b"1.2300e0", b"1e1000000", 1)
    receipt, *_ = _build(raw)

    assert receipt["payload"]["counts"]["invalid_record_count"] == 1
    assert "raw_invalid_records_present" in receipt["payload"]["blockers"]


class _Response:
    def __init__(self, body: bytes):
        self.content = body
        self.status_code = 200
        self.headers = {"X-Request-Id": "synthetic-request"}


class _Transport:
    def __init__(self, pages):
        self.pages = {key: list(value) for key, value in pages.items()}

    def __call__(self, url, *, params, timeout):
        del params, timeout
        table = Path(url).name.removesuffix(".json")
        return _Response(self.pages[table].pop(0))


def _zacks_body(columns, rows):
    return json.dumps(
        {
            "datatable": {"columns": columns, "data": rows},
            "meta": {"next_cursor_id": None},
        },
        separators=(",", ":"),
    ).encode()


def _zacks_raw():
    eeh_columns = [
        {"name": "m_ticker", "type": "String"},
        {"name": "ticker", "type": "String"},
        {"name": "currency_code", "type": "String"},
        {"name": "per_end_date", "type": "Date"},
        {"name": "per_type", "type": "String"},
        {"name": "obs_date", "type": "Date"},
        {"name": "eps_mean_est", "type": "BigDecimal"},
        {"name": "eps_cnt_est", "type": "Integer"},
    ]
    mt_columns = [
        {"name": "m_ticker", "type": "String"},
        {"name": "ticker", "type": "String"},
        {"name": "comp_cik", "type": "String"},
        {"name": "currency_code", "type": "String"},
    ]
    base = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
    times = iter([base, base + timedelta(seconds=1), base + timedelta(seconds=2)])
    snapshot = ZacksTablesClient(
        get=_Transport(
            {
                "EEH": [
                    _zacks_body(
                        eeh_columns,
                        [
                            [
                                "AAPL",
                                "AAPL",
                                "USD",
                                "2020-03-31",
                                "Q",
                                "2020-03-27",
                                1.23,
                                4,
                            ],
                            [
                                "MSFT",
                                "MSFT",
                                "USD",
                                "2020-03-31",
                                "Q",
                                "2020-03-30",
                                2.5,
                                7,
                            ],
                        ],
                    )
                ],
                "MT": [
                    _zacks_body(
                        mt_columns,
                        [
                            ["AAPL", "AAPL", "0000320193", "USD"],
                            ["MSFT", "MSFT", "0000789019", "USD"],
                        ],
                    )
                ],
            }
        ),
        clock=lambda: next(times),
        environ={"NASDAQ_DATA_LINK_API_KEY": "synthetic-only"},
    ).capture(
        mode="historical-sample",
        candidate_id=CANDIDATE,
        requested_start="2020-01-01",
        requested_end="2020-12-31",
        tables=["EEH", "MT"],
        filters_by_table={
            "EEH": {
                "ticker": "AAPL,MSFT",
                "per_end_date.gte": "2020-01-01",
                "per_end_date.lte": "2020-12-31",
                "per_type": "Q",
            },
            "MT": {"m_ticker": "AAPL,MSFT"},
        },
    )
    return snapshot.to_json().encode()


def test_zacks_adapter_reopens_pages_but_preserves_development_only_status():
    identity = _identity_snapshot()
    universe = _universe(identity)
    source = build_pead_consensus_source_manifest(
        candidate_id=CANDIDATE,
        adapter_id=ZACKS_EEH_ADAPTER,
        provider_id="nasdaq-data-link-zacks",
        dataset_id="ZACKS/EEH",
        evidence_class=None,
        source_captured_at_utc=None,
        provider_snapshot_at_utc="2026-07-14T12:00:00Z",
        canonical_query_sha256=None,
        records_path=None,
        field_map=None,
    )
    profile = _profile(source)
    raw = _zacks_raw()
    trust = _trust(universe, identity, source, profile, raw)
    receipt = build_pead_consensus_replay(
        raw_artifact=raw,
        event_universe=universe,
        identity_snapshot=identity,
        source_manifest=source,
        metric_profile=profile,
        **trust,
    )
    consensus = receipt["payload"]["consensus_evidence"]["payload"]

    assert consensus["evidence_class"] == "development_sample"
    assert receipt["payload"]["qualification_allowed"] is False
    assert "consensus:evidence_class_not_qualifying" in receipt["payload"]["blockers"]
    vintages = [
        event["vintages"][0]
        for event in consensus["event_records"]
        if event["disposition"] == "available"
    ]
    assert len(vintages) == 2
    assert all(vintage["availability_precision"] == "date" for vintage in vintages)
    assert all(vintage["trusted_available_at_utc"] is None for vintage in vintages)
    assert verify_pead_consensus_replay(receipt, raw_artifact=raw, **trust) == receipt


def test_metric_profile_cannot_redirect_generic_value_column():
    identity = _identity_snapshot()
    universe = _universe(identity)
    source = _source()
    profile = _profile(source)
    forged = copy.deepcopy(profile)
    forged["payload"]["value_field"] = "analysts"
    forged["artifact_hash"] = content_hash(forged["payload"])
    raw = _json_raw()
    trust = _trust(universe, identity, source, forged, raw)

    with pytest.raises(PeadConsensusReplayError, match="mapped raw value/count"):
        build_pead_consensus_replay(
            raw_artifact=raw,
            event_universe=universe,
            identity_snapshot=identity,
            source_manifest=source,
            metric_profile=forged,
            **trust,
        )


def test_receipt_bytes_are_canonical_content_addressed_json():
    receipt, *_ = _build()
    encoded = canonical_json(receipt)
    assert json.loads(encoded) == receipt
    assert receipt["artifact_hash"] == content_hash(receipt["payload"])
