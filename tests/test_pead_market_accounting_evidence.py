from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import duckdb
import pytest

import data.pead_market_accounting_evidence as market_module
import data.pead_sharadar_event_universe_replay as event_replay_module
from data.pead_event_universe import canonical_json, content_hash
from data.pead_market_accounting_evidence import (
    PeadMarketAccountingEvidenceError,
    build_pead_market_accounting_evidence,
    build_pead_sep_semantic_profile,
    load_pead_sep_semantic_profile,
    publish_pead_market_accounting_evidence,
    publish_pead_sep_semantic_profile,
    validate_pead_market_accounting_evidence_structure,
    validate_pead_sep_semantic_profile,
    verify_pead_market_accounting_evidence,
)


CANDIDATE = "pead-vq-source-qualification-v2"
UNIVERSE_HASH = "3" * 64
SOURCE_HASH = "4" * 64
IDENTITY_HASH = "5" * 64
REPLAY_HASH = "6" * 64
INDEX_HASH = "7" * 64
RECONCILIATION_HASH = "8" * 64
ACQUISITION_HASH = "9" * 64
RAW_ZIP_HASH = "a" * 64
PARQUET_HASH = "b" * 64
CALENDAR_HASH = "c" * 64
CALENDAR_RECEIPT_HASH = "d" * 64
OFFICIAL_SEMANTICS_RECEIPT_HASH = "e" * 64
IDENTITY_ID = "f" * 64

EVENT_KEY = {
    "cik": "0000001001",
    "fiscal_period_end": "2020-03-31",
    "fiscal_period_type": "Q",
}
EXCLUDED_EVENT_KEY = {
    "cik": "0000002002",
    "fiscal_period_end": "2020-03-31",
    "fiscal_period_type": "Q",
}
EVENT_ID = content_hash(EVENT_KEY)
EXCLUDED_EVENT_ID = content_hash(EXCLUDED_EVENT_KEY)
SEP_SCHEMA = [
    {"name": "ticker", "logical_type": "VARCHAR"},
    {"name": "date", "logical_type": "DATE"},
    {"name": "close", "logical_type": "DOUBLE"},
    {"name": "closeadj", "logical_type": "DOUBLE"},
    {"name": "closeunadj", "logical_type": "DOUBLE"},
    {"name": "lastupdated", "logical_type": "DATE"},
]
SEP_METADATA = {
    "vendor_code": "SHARADAR",
    "datatable_code": "SEP",
    "columns": [
        {"name": "ticker", "type": "String", "description": "Ticker"},
        {"name": "date", "type": "Date", "description": "Session date"},
        {"name": "close", "type": "BigDecimal", "description": "Close"},
        {"name": "closeadj", "type": "BigDecimal", "description": "Adjusted close"},
        {"name": "closeunadj", "type": "BigDecimal", "description": "Unadjusted close"},
        {"name": "lastupdated", "type": "Date", "description": "Last updated"},
    ],
}


def _wrapper(payload: dict, artifact_hash: str | None = None) -> dict:
    return {"artifact_hash": artifact_hash or content_hash(payload), "payload": payload}


def _write_sep(path: Path, rows: list[tuple]) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            "CREATE TABLE sep (ticker VARCHAR,date DATE,close DOUBLE,closeadj DOUBLE,"
            "closeunadj DOUBLE,lastupdated DATE)"
        )
        connection.executemany("INSERT INTO sep VALUES (?,?,?,?,?,?)", rows)
        connection.execute("COPY sep TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        connection.close()


def _calendar_evidence(*, early_date: str = "2020-04-30") -> dict:
    source_id = "ice-test"
    calendar = _wrapper(
        {
            "schema_version": "nyse_session_close_calendar.v1",
            "venue": "NYSE cash equities",
            "timezone": "America/New_York",
            "coverage": {"start": "2020-01-01", "end": "2020-12-31"},
            "regular_close_local_time": "16:00:00",
            "early_close_local_time": "13:00:00",
            "observed_session_rule": (
                "SEP-observed sessions use the listed 13:00 early close when present and "
                "otherwise the NYSE 16:00 core-session close; dates absent from SEP receive "
                "no inferred session."
            ),
            "early_close_sessions": [{"date": early_date, "source_id": source_id}],
            "sources": [
                {
                    "source_id": source_id,
                    "publisher": "Intercontinental Exchange / NYSE Group",
                    "url": "https://example.test/official",
                    "covered_years": [2020],
                }
            ],
        },
        CALENDAR_HASH,
    )
    receipt = _wrapper(
        {
            "schema_version": "nyse_session_close_source_receipt.v1",
            "calendar_artifact_hash": CALENDAR_HASH,
            "created_at_utc": "2026-07-13T12:00:00Z",
            "sources": [
                {
                    "source_id": source_id,
                    "extraction": {"early_close_dates": [early_date]},
                }
            ],
        },
        CALENDAR_RECEIPT_HASH,
    )
    return {"calendar": calendar, "source_receipt": receipt, "source_documents": {}}


def _install_event_replay_stub(monkeypatch, replay: dict, index: dict, calls: list) -> None:
    def verify(replay_document, index_document, **kwargs):
        calls.append((replay_document, index_document, kwargs))
        return {"replay": replay, "index": index}

    monkeypatch.setattr(event_replay_module, "verify_pead_sharadar_event_universe_replay", verify)


def _fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    evidence_class: str = "historical_reconstruction",
    capture_at: str = "2020-04-30T17:00:00Z",
    rows: list[tuple] | None = None,
) -> dict:
    parquet_path = tmp_path / "sep.parquet"
    _write_sep(
        parquet_path,
        rows
        or [
            ("AAA", date(2020, 4, 29), 10.0, 9.0, 20.0, date(2026, 7, 13)),
            ("AAA", date(2020, 4, 30), 12.0, 11.0, 24.0, date(2026, 7, 13)),
            ("AAA", date(2020, 5, 1), 99.0, 98.0, 198.0, date(2026, 7, 13)),
        ],
    )
    source_snapshot = _wrapper(
        {
            "candidate_id": CANDIDATE,
            "created_at_utc": "2026-07-13T09:00:00Z",
            "qualification_allowed": True,
            "tables": [],
        },
        SOURCE_HASH,
    )
    identity_snapshot = _wrapper(
        {
            "candidate_id": CANDIDATE,
            "created_at_utc": "2026-07-13T10:00:00Z",
            "qualification_allowed": True,
            "identities": [
                {
                    "identity_id": IDENTITY_ID,
                    "cik": EVENT_KEY["cik"],
                    "ticker": "AAA",
                    "permaticker": 101,
                    "valid_from": "2010-01-01",
                    "valid_through": "2026-12-31",
                }
            ],
        },
        IDENTITY_HASH,
    )
    lineage = {
        "event_id": EVENT_ID,
        "event_key": EVENT_KEY,
        "ticker": "AAA",
        "permaticker": 101,
        "identity_id": IDENTITY_ID,
        "representative_sf1_source_record_sha256": "0" * 64,
        "sf1_source_record_sha256s": ["0" * 64],
        "sf1_revision_count": 1,
    }
    universe = _wrapper({"candidate_id": CANDIDATE}, UNIVERSE_HASH)
    replay = _wrapper(
        {
            "candidate_id": CANDIDATE,
            "created_at_utc": "2026-07-13T11:00:00Z",
            "qualification_allowed": True,
            "bindings": {
                "source_snapshot_sha256": SOURCE_HASH,
                "identity_snapshot_sha256": IDENTITY_HASH,
            },
            "years": [
                {
                    "event_universe": universe,
                    "event_lineage": [lineage],
                }
            ],
        },
        REPLAY_HASH,
    )
    index = _wrapper({"candidate_id": CANDIDATE}, INDEX_HASH)
    reconciliation = _wrapper(
        {
            "candidate_id": CANDIDATE,
            "evidence_class": evidence_class,
            "reconciled_at_utc": "2026-07-13T13:00:00Z",
            "bindings": {"event_universe_sha256": UNIVERSE_HASH},
            "event_results": [
                {
                    "event_id": EVENT_ID,
                    "event_key": EVENT_KEY,
                    "disposition": "event_source_reconciled",
                    "event_source_input": {
                        "known_public_by_at_utc": "2020-05-01T13:00:00Z",
                        "consensus_receipt_captured_at_utc": capture_at,
                    },
                },
                {
                    "event_id": EXCLUDED_EVENT_ID,
                    "event_key": EXCLUDED_EVENT_KEY,
                    "disposition": "excluded",
                    "event_source_input": None,
                },
            ],
        },
        RECONCILIATION_HASH,
    )
    acquisition = _wrapper(
        {
            "source": {"datatable_metadata": SEP_METADATA},
            "raw_zip": {"sha256": RAW_ZIP_HASH},
            "parquet": {"sha256": PARQUET_HASH, "schema": SEP_SCHEMA},
        },
        ACQUISITION_HASH,
    )
    profile = build_pead_sep_semantic_profile(
        created_at_utc="2026-07-13T08:00:00Z",
        datatable_metadata_sha256=content_hash(SEP_METADATA),
        official_semantics_source_receipt_sha256=OFFICIAL_SEMANTICS_RECEIPT_HASH,
    )
    calls: list = []
    _install_event_replay_stub(monkeypatch, replay, index, calls)
    monkeypatch.setattr(
        market_module,
        "validate_pead_sharadar_source_snapshot",
        lambda document, **kwargs: document,
    )
    monkeypatch.setattr(
        market_module,
        "validate_pead_security_identity_snapshot",
        lambda document, **kwargs: document,
    )
    monkeypatch.setattr(
        market_module,
        "verify_pead_source_reconciliation_v2",
        lambda document, **kwargs: document,
    )
    monkeypatch.setattr(
        market_module,
        "_sep_acquisition",
        lambda document, **kwargs: (acquisition, parquet_path),
    )
    monkeypatch.setattr(
        market_module,
        "load_session_close_calendar_evidence",
        lambda **kwargs: _calendar_evidence(),
    )
    kwargs = {
        "warehouse_dir": tmp_path,
        "candidate_specification_path": tmp_path / "candidate.json",
        "construction_code_path": tmp_path / "construction.py",
        "calendar_path": tmp_path / "calendar.json",
        "calendar_receipt_path": tmp_path / "receipt.json",
        "created_at_utc": "2026-07-14T12:00:00Z",
        "source_reconciliation_verification_kwargs": {},
        "trusted_candidate_specification_sha256s": {"1" * 64},
        "trusted_construction_code_sha256s": {"2" * 64},
        "trusted_sharadar_source_snapshot_sha256s": {SOURCE_HASH},
        "trusted_security_identity_snapshot_sha256s": {IDENTITY_HASH},
        "trusted_sharadar_event_replay_sha256s": {REPLAY_HASH},
        "trusted_source_reconciliation_sha256s": {RECONCILIATION_HASH},
        "trusted_sep_semantic_profile_sha256s": {profile["artifact_hash"]},
        "trusted_nyse_calendar_sha256s": {CALENDAR_HASH},
        "trusted_nyse_source_receipt_sha256s": {CALENDAR_RECEIPT_HASH},
    }
    return {
        "source": source_snapshot,
        "identity": identity_snapshot,
        "replay": replay,
        "index": index,
        "reconciliation": reconciliation,
        "profile": profile,
        "kwargs": kwargs,
        "calls": calls,
    }


def _build(fixture: dict) -> dict:
    return build_pead_market_accounting_evidence(
        fixture["reconciliation"],
        fixture["replay"],
        fixture["index"],
        fixture["source"],
        fixture["identity"],
        fixture["profile"],
        **fixture["kwargs"],
    )


def _authoritative_publication_kwargs(fixture: dict) -> dict:
    kwargs = dict(fixture["kwargs"])
    kwargs.pop("created_at_utc")
    return {
        "source_reconciliation": fixture["reconciliation"],
        "sharadar_event_replay": fixture["replay"],
        "event_universe_index": fixture["index"],
        "sharadar_source_snapshot": fixture["source"],
        "security_identity_snapshot": fixture["identity"],
        "sep_semantic_profile": fixture["profile"],
        **kwargs,
    }


def test_authoritative_boundary_selects_unique_prior_sep_row_and_official_close(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)

    document = _build(fixture)

    result = document["payload"]["event_results"][0]
    denominator = result["market_denominator"]
    assert result["disposition"] == "market_accounting_evidenced"
    assert denominator["session_date"] == "2020-04-30"
    assert denominator["session_close_at_utc"] == "2020-04-30T17:00:00Z"
    assert denominator["session_close_kind"] == "official_early_close"
    assert denominator["close_split_normalized"] == "12"
    assert denominator["closeunadj_execution_evidence"] == "24"
    assert denominator["split_normalization_factor"] == {
        "formula": "closeunadj / close",
        "numerator": 2,
        "denominator": 1,
        "decimal_34": "2",
    }
    assert len(denominator["sep_source_row_sha256"]) == 64
    assert document["payload"]["event_results"][1]["disposition"] == "upstream_excluded"
    assert document["payload"]["coverage"] == {
        "upstream_event_count": 2,
        "upstream_excluded_count": 1,
        "source_reconciled_event_count": 1,
        "market_accounting_evidenced_count": 1,
        "market_accounting_excluded_count": 0,
        "exhaustive_upstream_accounting": True,
        "exhaustive_source_reconciled_accounting": True,
        "event_blocker_counts": {"upstream_not_event_source_reconciled": 1},
    }
    assert document["payload"]["qualification"]["market_accounting_evidence_allowed"]
    assert document["payload"]["qualification"]["research_consumable"] is False
    assert document["payload"]["qualification"]["edge_claim_allowed"] is False
    assert document["payload"]["qualification"]["paper_execution_allowed"] is False
    assert document["payload"]["qualification"]["live_deployment_allowed"] is False
    assert fixture["calls"]


def test_every_external_trust_registry_is_mandatory(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    trust_fields = [key for key in fixture["kwargs"] if key.startswith("trusted_")]

    for field in trust_fields:
        kwargs = {**fixture["kwargs"], field: set()}
        with pytest.raises(PeadMarketAccountingEvidenceError, match="external trust registry"):
            build_pead_market_accounting_evidence(
                fixture["reconciliation"],
                fixture["replay"],
                fixture["index"],
                fixture["source"],
                fixture["identity"],
                fixture["profile"],
                **kwargs,
            )


def test_same_day_sep_row_is_never_used(tmp_path, monkeypatch):
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        rows=[
            ("AAA", date(2020, 5, 1), 99.0, 98.0, 198.0, date(2026, 7, 13)),
        ],
    )

    result = _build(fixture)["payload"]["event_results"][0]

    assert result["disposition"] == "market_accounting_excluded"
    assert result["blockers"] == ["market_prior_session_absent"]
    assert result["market_denominator"] is None


def test_invalid_latest_prior_row_never_falls_back_to_older_positive_row(tmp_path, monkeypatch):
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        rows=[
            ("AAA", date(2020, 4, 29), 10.0, 9.0, 20.0, date(2026, 7, 13)),
            ("AAA", date(2020, 4, 30), 0.0, 11.0, 24.0, date(2026, 7, 13)),
        ],
    )

    result = _build(fixture)["payload"]["event_results"][0]

    assert result["disposition"] == "market_accounting_excluded"
    assert result["blockers"] == ["market_latest_prior_session_price_invalid"]
    assert result["market_denominator"] is None


def test_duplicate_latest_sep_rows_are_ambiguous_without_fallback(tmp_path, monkeypatch):
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        rows=[
            ("AAA", date(2020, 4, 30), 12.0, 11.0, 24.0, date(2026, 7, 13)),
            ("AAA", date(2020, 4, 30), 13.0, 12.0, 26.0, date(2026, 7, 13)),
        ],
    )

    result = _build(fixture)["payload"]["event_results"][0]

    assert result["disposition"] == "market_accounting_excluded"
    assert result["blockers"] == ["market_latest_prior_session_ambiguous"]


def test_prospective_consensus_capture_must_precede_selected_official_close(tmp_path, monkeypatch):
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        evidence_class="prospective_signal",
        capture_at="2020-04-30T17:00:01Z",
    )

    result = _build(fixture)["payload"]["event_results"][0]

    assert result["disposition"] == "market_accounting_excluded"
    assert result["blockers"] == ["prospective_consensus_capture_after_prior_close"]
    assert result["market_denominator"]["session_date"] == "2020-04-30"
    assert len(result["market_denominator"]["sep_source_row_sha256"]) == 64
    assert result["timing"]["prospective_freeze_required"] is True
    assert result["timing"]["prospective_freeze_passed"] is False


def test_prospective_consensus_capture_at_exact_prior_close_is_allowed(tmp_path, monkeypatch):
    fixture = _fixture(
        tmp_path,
        monkeypatch,
        evidence_class="prospective_signal",
        capture_at="2020-04-30T17:00:00Z",
    )

    result = _build(fixture)["payload"]["event_results"][0]

    assert result["disposition"] == "market_accounting_evidenced"
    assert result["timing"]["prospective_freeze_passed"] is True


def test_semantic_profile_is_closed_and_must_match_acquired_metadata(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["profile"])
    forged["payload"]["normalization"]["factor_formula"] = "SEP.close / SEP.closeunadj"
    forged["artifact_hash"] = content_hash(forged["payload"])
    with pytest.raises(PeadMarketAccountingEvidenceError, match="normalization semantics"):
        validate_pead_sep_semantic_profile(forged)

    wrong_metadata = build_pead_sep_semantic_profile(
        created_at_utc="2026-07-13T08:00:00Z",
        datatable_metadata_sha256="0" * 64,
        official_semantics_source_receipt_sha256=OFFICIAL_SEMANTICS_RECEIPT_HASH,
    )
    fixture["profile"] = wrong_metadata
    fixture["kwargs"]["trusted_sep_semantic_profile_sha256s"] = {wrong_metadata["artifact_hash"]}
    with pytest.raises(PeadMarketAccountingEvidenceError, match="official metadata"):
        _build(fixture)


def test_rehashed_denominator_tamper_passes_structure_but_fails_authoritative_replay(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    document = _build(fixture)
    forged = copy.deepcopy(document)
    forged["payload"]["event_results"][0]["market_denominator"]["close_split_normalized"] = "1200"
    forged["payload"]["event_results"][0]["market_denominator"]["split_normalization_factor"] = {
        "formula": "closeunadj / close",
        "numerator": 1,
        "denominator": 50,
        "decimal_34": "0.02",
    }
    forged["artifact_hash"] = content_hash(forged["payload"])

    assert validate_pead_market_accounting_evidence_structure(forged) == forged
    verification_kwargs = dict(fixture["kwargs"])
    verification_kwargs.pop("created_at_utc")
    with pytest.raises(PeadMarketAccountingEvidenceError, match="does not replay"):
        verify_pead_market_accounting_evidence(
            forged,
            fixture["reconciliation"],
            fixture["replay"],
            fixture["index"],
            fixture["source"],
            fixture["identity"],
            fixture["profile"],
            **verification_kwargs,
        )


def test_calendar_claim_must_equal_replayed_official_source_union(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    evidence = _calendar_evidence()
    evidence["calendar"]["payload"]["early_close_sessions"] = []
    monkeypatch.setattr(
        market_module,
        "load_session_close_calendar_evidence",
        lambda **kwargs: evidence,
    )

    with pytest.raises(PeadMarketAccountingEvidenceError, match="official source union"):
        _build(fixture)


def test_semantic_profile_loader_requires_canonical_duplicate_key_free_bytes(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        '{"artifact_hash":"a","artifact_hash":"b","payload":{}}',
        encoding="utf-8",
    )

    with pytest.raises(PeadMarketAccountingEvidenceError, match="duplicate key"):
        load_pead_sep_semantic_profile(path)


def test_semantic_profile_publication_is_canonical_exclusive_and_collision_safe(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, monkeypatch)
    target = tmp_path / "published" / "sep-profile.json"

    normalized, published = publish_pead_sep_semantic_profile(fixture["profile"], target)

    assert normalized == fixture["profile"]
    assert published == target
    expected_bytes = (canonical_json(fixture["profile"]) + "\n").encode("utf-8")
    assert target.read_bytes() == expected_bytes
    assert load_pead_sep_semantic_profile(target) == fixture["profile"]
    with pytest.raises(PeadMarketAccountingEvidenceError, match="collision"):
        publish_pead_sep_semantic_profile(fixture["profile"], target)
    assert target.read_bytes() == expected_bytes


def test_profile_publication_refuses_preexisting_different_bytes(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    target = tmp_path / "profile.json"
    target.write_bytes(b"do-not-replace")

    with pytest.raises(PeadMarketAccountingEvidenceError, match="collision"):
        publish_pead_sep_semantic_profile(fixture["profile"], target)

    assert target.read_bytes() == b"do-not-replace"


def test_market_publication_requires_authoritative_replay_by_default(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    document = _build(fixture)
    target = tmp_path / "market.json"

    with pytest.raises(
        PeadMarketAccountingEvidenceError,
        match="requires authoritative verification",
    ):
        publish_pead_market_accounting_evidence(document, target)

    assert not target.exists()


def test_market_publication_authoritatively_replays_then_strictly_rereads(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    document = _build(fixture)
    target = tmp_path / "receipts" / f"{document['artifact_hash']}.json"

    normalized, published = publish_pead_market_accounting_evidence(
        document,
        target,
        authoritative_verification_kwargs=_authoritative_publication_kwargs(fixture),
    )

    assert normalized == document
    assert published == target
    assert target.read_bytes() == (canonical_json(document) + "\n").encode("utf-8")
    with pytest.raises(PeadMarketAccountingEvidenceError, match="collision"):
        publish_pead_market_accounting_evidence(
            document,
            target,
            authoritative_verification_kwargs=_authoritative_publication_kwargs(fixture),
        )


def test_market_publication_rejects_self_consistent_tamper_before_creation(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(_build(fixture))
    denominator = forged["payload"]["event_results"][0]["market_denominator"]
    denominator["close_split_normalized"] = "1200"
    denominator["split_normalization_factor"] = {
        "formula": "closeunadj / close",
        "numerator": 1,
        "denominator": 50,
        "decimal_34": "0.02",
    }
    forged["artifact_hash"] = content_hash(forged["payload"])
    target = tmp_path / "forged.json"

    with pytest.raises(PeadMarketAccountingEvidenceError, match="does not replay"):
        publish_pead_market_accounting_evidence(
            forged,
            target,
            authoritative_verification_kwargs=_authoritative_publication_kwargs(fixture),
        )

    assert not target.exists()


def test_structural_market_publication_requires_explicit_exclusive_mode(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    document = _build(fixture)
    target = tmp_path / "development-market.json"

    normalized, _ = publish_pead_market_accounting_evidence(
        document, target, allow_structural_only=True
    )
    assert normalized == document
    with pytest.raises(PeadMarketAccountingEvidenceError, match="mutually exclusive"):
        publish_pead_market_accounting_evidence(
            document,
            tmp_path / "invalid-mode.json",
            authoritative_verification_kwargs=_authoritative_publication_kwargs(fixture),
            allow_structural_only=True,
        )
