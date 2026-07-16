from __future__ import annotations

import copy

import pytest

from data.pead_consensus_evidence import (
    PeadConsensusEvidenceError,
    build_pead_consensus_evidence,
    canonical_json,
    content_hash,
    load_pead_consensus_evidence,
    validate_pead_consensus_evidence,
)
from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
)


KEYS = [
    {"cik": "0000320193", "fiscal_period_end": "2020-03-28", "fiscal_period_type": "Q"},
    {"cik": "0000789019", "fiscal_period_end": "2020-03-31", "fiscal_period_type": "Q"},
]
EVENT_IDS = sorted(canonical_event_id(key) for key in KEYS)


def _universe():
    source_ids = ["1" * 64, "2" * 64]
    query_hash = "e" * 64
    census = build_pead_event_census_receipt(
        raw_census_artifact_sha256="f" * 64,
        canonical_query_sha256=query_hash,
        source_record_ids=source_ids,
    )
    return build_pead_event_universe(
        candidate_id="pead-test-v1",
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start="2020-01-01",
        event_end="2020-12-31",
        bindings={
            "market_snapshot_sha256": "a" * 64,
            "identity_snapshot_sha256": "b" * 64,
            "candidate_specification_sha256": "c" * 64,
            "construction_code_sha256": "d" * 64,
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


def _source(*, snapshot="2026-07-13T00:00:00Z"):
    return {
        "provider_id": "licensed-consensus-provider",
        "dataset_id": "historical-point-in-time-estimates",
        "source_manifest_sha256": "8" * 64,
        "captured_at_utc": "2026-07-14T12:00:00Z",
        "provider_snapshot_at_utc": snapshot,
    }


def _receipt(*, scoped_ids=None, terminal=True):
    body = {
        "source_captured_at_utc": "2026-07-14T11:00:00Z",
        "query_scope": {
            "scope_kind": "full_export",
            "canonical_query_sha256": "7" * 64,
            "expected_event_ids": list(EVENT_IDS if scoped_ids is None else scoped_ids),
        },
        "pagination": {
            "mode": "bulk_file",
            "terminal_page_observed": terminal,
            "page_count": 1,
            "pages": [
                {
                    "sequence": 1,
                    "request_sha256": "6" * 64,
                    "raw_response_sha256": "5" * 64,
                    "raw_response_bytes": 1234,
                    "continuation_token_sha256": None,
                }
            ],
        },
    }
    return {"receipt_sha256": content_hash(body), **body}


def _metric():
    return {
        "metric_id": "eps",
        "accounting_basis": "adjusted",
        "per_share_basis": "diluted",
        "scope": "continuing_operations",
        "canonical_share_basis": "split_restated",
        "currency_code": "USD",
        "unit": "currency_per_share",
        "metric_definition_sha256": "4" * 64,
    }


def _vintage(index, *, as_of, available=None, raw_hash=None, receipt_hash=None):
    return {
        "provider_as_of_date": as_of,
        "trusted_available_at_utc": available,
        "availability_precision": "second" if available else "date",
        "consensus_value": ("1", "1.1", "1.2")[index],
        "analyst_count": 4,
        "raw_record_sha256": raw_hash or str(index + 1) * 64,
        "acquisition_receipt_sha256": receipt_hash or _receipt()["receipt_sha256"],
        "metric": _metric(),
    }


def _records(*, missing_second=False, receipt_hash=None):
    return [
        {
            "event_id": EVENT_IDS[1],
            "disposition": "missing" if missing_second else "available",
            "missing_reason": "provider_record_absent" if missing_second else None,
            "vintages": [] if missing_second else [
                _vintage(2, as_of="2020-03-30", receipt_hash=receipt_hash)
            ],
        },
        {
            "event_id": EVENT_IDS[0],
            "disposition": "available",
            "missing_reason": None,
            "vintages": [
                _vintage(
                    1, as_of="2020-03-27", available="2020-03-27T22:00:00Z",
                    receipt_hash=receipt_hash,
                ),
                _vintage(0, as_of="2020-03-20", receipt_hash=receipt_hash),
            ],
        },
    ]


def _document(*, evidence_class="historical_reconstruction", missing=False,
              scoped_ids=None, terminal=True, snapshot="2026-07-13T00:00:00Z"):
    receipt = _receipt(scoped_ids=scoped_ids, terminal=terminal)
    return build_pead_consensus_evidence(
        candidate_id="pead-test-v1",
        evidence_class=evidence_class,
        event_universe=_universe(),
        source=_source(snapshot=snapshot),
        acquisition_receipts=[receipt],
        event_records=_records(
            missing_second=missing, receipt_hash=receipt["receipt_sha256"]
        ),
    )


def _rehash(document):
    document["artifact_hash"] = content_hash(document["payload"])


def test_complete_artifact_is_content_addressed_exhaustive_and_unselected():
    document = _document()
    payload = document["payload"]

    assert document["artifact_hash"] == content_hash(payload)
    assert [row["event_id"] for row in payload["event_records"]] == EVENT_IDS
    assert [
        row["provider_as_of_date"]
        for row in payload["event_records"][0]["vintages"]
    ] == ["2020-03-20", "2020-03-27"]
    assert "selected_vintage" not in payload
    assert payload["coverage"] == {
        "expected_event_count": 2,
        "available_event_count": 2,
        "missing_event_count": 0,
        "query_scoped_event_count": 2,
        "vintage_count": 3,
        "pagination_complete": True,
        "blockers": [],
        "qualification_allowed": True,
    }


def test_omitted_event_record_is_rejected_even_when_rehashed():
    tampered = copy.deepcopy(_document())
    tampered["payload"]["event_records"].pop()
    tampered["payload"]["coverage"]["expected_event_count"] = 1
    tampered["payload"]["coverage"]["available_event_count"] = 1
    _rehash(tampered)

    with pytest.raises(PeadConsensusEvidenceError, match="every frozen expected event"):
        validate_pead_consensus_evidence(tampered)


def test_duplicate_raw_record_hash_and_unknown_receipt_are_rejected():
    records = _records()
    records[0]["vintages"][0]["raw_record_sha256"] = records[1]["vintages"][0][
        "raw_record_sha256"
    ]
    with pytest.raises(PeadConsensusEvidenceError, match="raw consensus records must be unique"):
        build_pead_consensus_evidence(
            candidate_id="pead-test-v1",
            evidence_class="historical_reconstruction",
            event_universe=_universe(),
            source=_source(),
            acquisition_receipts=[_receipt()],
            event_records=records,
        )

    records = _records()
    records[0]["vintages"][0]["acquisition_receipt_sha256"] = "4" * 64
    with pytest.raises(PeadConsensusEvidenceError, match="unknown acquisition receipt"):
        build_pead_consensus_evidence(
            candidate_id="pead-test-v1",
            evidence_class="historical_reconstruction",
            event_universe=_universe(),
            source=_source(),
            acquisition_receipts=[_receipt()],
            event_records=records,
        )


def test_fake_completeness_and_development_claims_fail_closed():
    development = _document(evidence_class="development_sample")
    assert development["payload"]["coverage"]["qualification_allowed"] is False
    assert "evidence_class_not_qualifying" in development["payload"]["coverage"]["blockers"]

    fake = copy.deepcopy(development)
    fake["payload"]["coverage"]["blockers"] = []
    fake["payload"]["coverage"]["qualification_allowed"] = True
    _rehash(fake)
    with pytest.raises(PeadConsensusEvidenceError, match="coverage and qualification"):
        validate_pead_consensus_evidence(fake)

    missing = _document(missing=True)
    assert missing["payload"]["coverage"]["qualification_allowed"] is False
    assert "expected_events_missing_consensus" in missing["payload"]["coverage"]["blockers"]


def test_query_scope_and_pagination_are_exhaustive_not_date_envelopes():
    partial = _document(scoped_ids=[EVENT_IDS[0]], terminal=False)
    coverage = partial["payload"]["coverage"]

    assert coverage["qualification_allowed"] is False
    assert "query_scope_not_exhaustive_once" in coverage["blockers"]
    assert "pagination_not_complete" in coverage["blockers"]
    assert "full_window" not in partial["payload"]


def test_hash_tamper_and_noncanonical_sorting_are_rejected():
    tampered = copy.deepcopy(_document())
    tampered["payload"]["event_records"][0]["vintages"][0]["consensus_value"] = "999"
    with pytest.raises(PeadConsensusEvidenceError, match="artifact hash mismatch"):
        validate_pead_consensus_evidence(tampered)

    unsorted = copy.deepcopy(_document())
    unsorted["payload"]["event_records"].reverse()
    _rehash(unsorted)
    with pytest.raises(PeadConsensusEvidenceError, match="canonically sorted"):
        validate_pead_consensus_evidence(unsorted)


def test_metric_basis_and_availability_precision_are_exact():
    records = _records()
    del records[0]["vintages"][0]["metric"]["scope"]
    with pytest.raises(PeadConsensusEvidenceError, match="metric fields differ"):
        build_pead_consensus_evidence(
            candidate_id="pead-test-v1",
            evidence_class="historical_reconstruction",
            event_universe=_universe(),
            source=_source(),
            acquisition_receipts=[_receipt()],
            event_records=records,
        )


def test_receipt_identity_decimal_identity_and_capture_chronology_are_derived():
    forged_receipt = _receipt()
    forged_receipt["receipt_sha256"] = "9" * 64
    with pytest.raises(PeadConsensusEvidenceError, match="canonical body"):
        build_pead_consensus_evidence(
            candidate_id="pead-test-v1",
            evidence_class="historical_reconstruction",
            event_universe=_universe(),
            source=_source(),
            acquisition_receipts=[forged_receipt],
            event_records=_records(receipt_hash="9" * 64),
        )

    for noncanonical in (1.0, "1.0"):
        records = _records()
        records[0]["vintages"][0]["consensus_value"] = noncanonical
        with pytest.raises(PeadConsensusEvidenceError, match="canonical finite decimal string"):
            build_pead_consensus_evidence(
                candidate_id="pead-test-v1",
                evidence_class="historical_reconstruction",
                event_universe=_universe(),
                source=_source(),
                acquisition_receipts=[_receipt()],
                event_records=records,
            )

    records = _records()
    records[0]["vintages"][0]["trusted_available_at_utc"] = (
        "2026-07-14T11:00:01Z"
    )
    records[0]["vintages"][0]["availability_precision"] = "second"
    with pytest.raises(PeadConsensusEvidenceError, match="follows its acquisition"):
        build_pead_consensus_evidence(
            candidate_id="pead-test-v1",
            evidence_class="historical_reconstruction",
            event_universe=_universe(),
            source=_source(),
            acquisition_receipts=[_receipt()],
            event_records=records,
        )

    records = _records()
    records[0]["vintages"][0]["availability_precision"] = "second"
    with pytest.raises(PeadConsensusEvidenceError, match="requires date precision"):
        build_pead_consensus_evidence(
            candidate_id="pead-test-v1",
            evidence_class="historical_reconstruction",
            event_universe=_universe(),
            source=_source(),
            acquisition_receipts=[_receipt()],
            event_records=records,
        )


def test_loader_accepts_canonical_file_and_rejects_duplicate_json_keys(tmp_path):
    document = _document()
    path = tmp_path / "consensus.json"
    path.write_text(canonical_json(document) + "\n", encoding="utf-8")
    assert load_pead_consensus_evidence(path) == document

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"artifact_hash":"' + "a" * 64 + '","artifact_hash":"'
        + "b" * 64 + '","payload":{}}',
        encoding="utf-8",
    )
    with pytest.raises(PeadConsensusEvidenceError, match="duplicate key"):
        load_pead_consensus_evidence(duplicate)
