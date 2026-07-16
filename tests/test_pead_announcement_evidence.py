from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from data.pead_announcement_evidence import (
    PeadAnnouncementEvidenceError,
    build_missing_outcome,
    build_pead_announcement_evidence,
    build_sec_available_outcome,
    load_pead_announcement_evidence,
    timing_eligible_announcements,
    validate_pead_announcement_evidence,
)
from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
    content_hash,
)


KEY_A = {
    "cik": "0000320193",
    "fiscal_period_end": "2018-12-29",
    "fiscal_period_type": "Q",
}
KEY_B = {
    "cik": "0000789019",
    "fiscal_period_end": "2018-12-31",
    "fiscal_period_type": "Q",
}
HASH = "a" * 64
ACCEPTED = "2019-01-29T21:05:00Z"
RETRIEVED = "2026-07-14T15:00:10Z"
HTTP_DATE = "2026-07-14T15:00:00Z"


def _universe(*keys: dict) -> dict:
    query_hash = "b" * 64
    source_ids = [hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(len(keys))]
    receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256="c" * 64,
        canonical_query_sha256=query_hash,
        source_record_ids=source_ids,
    )
    dispositions = [
        {
            "source_record_id": source_id,
            "disposition": "expected_event",
            "event_id": canonical_event_id(key),
            "event_key": key,
            "reason": None,
        }
        for source_id, key in zip(source_ids, keys, strict=True)
    ]
    return build_pead_event_universe(
        candidate_id="pead-vq-locked-replication-v1",
        frozen_at_utc="2026-07-14T14:00:00Z",
        event_start="2018-01-01",
        event_end="2019-12-31",
        bindings={
            "market_snapshot_sha256": HASH,
            "identity_snapshot_sha256": HASH,
            "candidate_specification_sha256": HASH,
            "construction_code_sha256": HASH,
            "canonical_query_sha256": query_hash,
        },
        census_receipt=receipt,
        census_dispositions=dispositions,
    )


def _http(date_utc: str = HTTP_DATE) -> dict:
    return {
        "status_code": 200,
        "date_utc": date_utc,
        "content_type": "text/html; charset=UTF-8",
        "etag": '"abc"',
        "last_modified_at_utc": None,
    }


def _document(role: str, raw: bytes, *, date_utc: str, retrieved_at: str) -> dict:
    return {
        "role": role,
        "url": "https://www.sec.gov/Archives/edgar/data/320193/source.txt",
        "retrieved_at_utc": retrieved_at,
        "http": _http(date_utc),
        "raw_document": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "base64": base64.b64encode(raw).decode("ascii"),
        },
    }


def _actual() -> dict:
    return {
        "announced_value": "4.18",
        "canonical_value": "1.045",
        "normalization_factor": "0.25",
        "metric": "earnings_per_share",
        "source_metric_label": "Adjusted diluted earnings per share",
        "metric_definition_sha256": "f" * 64,
        "accounting_basis": "non_gaap",
        "per_share_basis": "diluted",
        "scope": "total_company",
        "currency": "USD",
        "unit": "currency_per_share",
        "announced_share_basis": "issuer_as_reported_at_publication",
        "canonical_share_basis": "split_restated_to_bound_consensus_snapshot",
        "fiscal_period_end": "2018-12-29",
        "fiscal_period_type": "Q",
        "normalization_evidence_sha256": "9" * 64,
    }


def _available(
    key: dict = KEY_A,
    *,
    first_public_at: str | None = None,
    first_public_proof: dict | None = None,
    observed_public: dict | None = None,
) -> dict:
    exhibit = b"Adjusted diluted earnings per share were $4.18."
    metadata = (
        f"<SEC-DOCUMENT>\n<SEC-HEADER>\n"
        f"<ACCESSION-NUMBER>{key['cik']}-19-000010\n"
        f"<CONFORMED-SUBMISSION-TYPE>8-K\n"
        f"<CENTRAL-INDEX-KEY>{key['cik']}\n"
        f"<ACCEPTANCE-DATETIME>20190129160500\n"
        f"<ITEMS>2.02\n</SEC-HEADER>\n"
        f"<DOCUMENT>\n<TYPE>EX-99.1\n<FILENAME>exhibit991.htm\n"
        f"</DOCUMENT>\n"
    ).encode()
    return build_sec_available_outcome(
        event_key=key,
        accession_number=f"{key['cik']}-19-000010",
        exhibit="EX-99.1",
        metadata_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019319000010/0000320193-19-000010.txt"
        ),
        metadata_retrieved_at_utc=RETRIEVED,
        metadata_http=_http(),
        metadata_bytes=metadata,
        exhibit_url=(
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019319000010/exhibit991.htm"
        ),
        exhibit_retrieved_at_utc=RETRIEVED,
        exhibit_http=_http(),
        exhibit_bytes=exhibit,
        edgar_acceptance_at_utc=ACCEPTED,
        canonical_actual=_actual(),
        extraction={
            "method": "sec_exhibit_label_value_visible_text.v1",
            "code_hash": "d" * 64,
            "reviewer": "independent-reviewer-1",
            "locator": "visible_text:Adjusted diluted earnings per share",
        },
        first_public_at_utc=first_public_at,
        first_public_proof=first_public_proof,
        observed_public=observed_public,
    )


def _rehash(document: dict) -> None:
    document["artifact_hash"] = content_hash(document["payload"])


def test_exact_artifact_accounts_for_available_and_missing_events():
    universe = _universe(KEY_A, KEY_B)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[
            _available(),
            build_missing_outcome(KEY_B, reason="sec_item_2_02_exhibit_unavailable"),
        ],
    )

    verified = validate_pead_announcement_evidence(
        document, expected_event_manifest=universe
    )
    assert verified == document
    assert verified["payload"]["expected_event_manifest_hash"] == universe[
        "artifact_hash"
    ]
    assert verified["payload"]["coverage"] == {
        "expected_events": 2,
        "available_events": 1,
        "timing_eligible_events": 0,
        "missing_events": 1,
        "event_universe_qualified": True,
        "blockers": [
            "expected_events_missing_announcement",
            "first_public_timestamps_unproven",
        ],
        "complete": False,
    }
    assert verified["payload"]["outcomes"][0]["available_record"][
        "canonical_actual"
    ]["canonical_value"] == "1.045"


def test_favorable_omission_and_duplicate_event_ids_fail_closed():
    universe = _universe(KEY_A, KEY_B)
    with pytest.raises(PeadAnnouncementEvidenceError, match="every frozen expected event"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T15:30:00Z",
            outcomes=[_available()],
        )
    with pytest.raises(PeadAnnouncementEvidenceError, match="duplicate event"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T15:30:00Z",
            outcomes=[_available(), _available()],
        )


def test_receipt_chronology_cannot_predate_universe_or_source_retrieval():
    universe = _universe(KEY_A)
    with pytest.raises(PeadAnnouncementEvidenceError, match="predates its frozen"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T13:59:59Z",
            outcomes=[build_missing_outcome(KEY_A, reason="not_yet_acquired")],
        )

    with pytest.raises(PeadAnnouncementEvidenceError, match="retrieval follows"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T14:30:00Z",
            outcomes=[_available()],
        )


def test_archived_sec_bytes_are_rehashed_even_after_self_consistent_receipt_edit():
    universe = _universe(KEY_A)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )
    tampered = copy.deepcopy(document)
    raw = tampered["payload"]["outcomes"][0]["available_record"][
        "exhibit_document"
    ]["raw_document"]
    raw["base64"] = base64.b64encode(b"forged exhibit same role").decode("ascii")
    _rehash(tampered)
    with pytest.raises(PeadAnnouncementEvidenceError, match="bytes/hash mismatch"):
        validate_pead_announcement_evidence(
            tampered, expected_event_manifest=universe
        )


def test_metadata_and_exhibit_claims_are_replayed_from_archived_bytes():
    universe = _universe(KEY_A)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )

    acceptance = copy.deepcopy(document)
    acceptance["payload"]["outcomes"][0]["available_record"][
        "edgar_acceptance_at_utc"
    ] = "2019-01-29T21:06:00Z"
    _rehash(acceptance)
    with pytest.raises(PeadAnnouncementEvidenceError, match="acceptance time differs"):
        validate_pead_announcement_evidence(
            acceptance, expected_event_manifest=universe
        )

    actual = copy.deepcopy(document)
    claimed = actual["payload"]["outcomes"][0]["available_record"][
        "canonical_actual"
    ]
    claimed["announced_value"] = "4.19"
    claimed["canonical_value"] = "1.0475"
    _rehash(actual)
    with pytest.raises(PeadAnnouncementEvidenceError, match="archived SEC exhibit"):
        validate_pead_announcement_evidence(actual, expected_event_manifest=universe)


def test_period_and_metric_definition_are_exact_canonical_actual_inputs():
    universe = _universe(KEY_A)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )
    actual = document["payload"]["outcomes"][0]["available_record"][
        "canonical_actual"
    ]
    assert actual["fiscal_period_end"] == KEY_A["fiscal_period_end"]
    assert actual["metric_definition_sha256"] == "f" * 64
    assert actual["normalization_evidence_sha256"] == "9" * 64

    mismatched = copy.deepcopy(document)
    mismatched["payload"]["outcomes"][0]["available_record"][
        "canonical_actual"
    ]["fiscal_period_end"] = "2018-12-31"
    _rehash(mismatched)
    with pytest.raises(PeadAnnouncementEvidenceError, match="differs from event key"):
        validate_pead_announcement_evidence(
            mismatched, expected_event_manifest=universe
        )


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("announced_value", "4.180"),
        ("canonical_value", "1.0450"),
        ("normalization_factor", "0.250"),
    ],
)
def test_actual_decimal_strings_have_one_cross_lane_identity(field: str, alias: str):
    universe = _universe(KEY_A)
    outcome = _available()
    outcome["available_record"]["canonical_actual"][field] = alias

    with pytest.raises(PeadAnnouncementEvidenceError, match="canonical decimal"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T15:30:00Z",
            outcomes=[outcome],
        )


def test_unqualified_event_universe_is_an_explicit_coverage_blocker():
    query_hash = "b" * 64
    source_ids = [hashlib.sha256(value).hexdigest() for value in (b"event", b"gap")]
    receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256="c" * 64,
        canonical_query_sha256=query_hash,
        source_record_ids=source_ids,
    )
    universe = build_pead_event_universe(
        candidate_id="pead-vq-locked-replication-v1",
        frozen_at_utc="2026-07-14T14:00:00Z",
        event_start="2018-01-01",
        event_end="2019-12-31",
        bindings={
            "market_snapshot_sha256": HASH,
            "identity_snapshot_sha256": HASH,
            "candidate_specification_sha256": HASH,
            "construction_code_sha256": HASH,
            "canonical_query_sha256": query_hash,
        },
        census_receipt=receipt,
        census_dispositions=[
            {
                "source_record_id": source_ids[0],
                "disposition": "expected_event",
                "event_id": canonical_event_id(KEY_A),
                "event_key": KEY_A,
                "reason": None,
            },
            {
                "source_record_id": source_ids[1],
                "disposition": "identity_gap",
                "event_id": None,
                "event_key": None,
                "reason": "period_identity_unresolved",
            },
        ],
    )
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )

    coverage = document["payload"]["coverage"]
    assert coverage["event_universe_qualified"] is False
    assert "event_universe_not_qualified" in coverage["blockers"]
    assert coverage["complete"] is False


def test_vendor_or_issuer_self_claim_cannot_replace_sec_source():
    universe = _universe(KEY_A)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )
    tampered = copy.deepcopy(document)
    record = tampered["payload"]["outcomes"][0]["available_record"]
    record["source_kind"] = "vendor_actual"
    _rehash(tampered)
    with pytest.raises(PeadAnnouncementEvidenceError, match="independent SEC"):
        validate_pead_announcement_evidence(
            tampered, expected_event_manifest=universe
        )

    issuer_url = copy.deepcopy(document)
    issuer_url["payload"]["outcomes"][0]["available_record"][
        "exhibit_document"
    ]["url"] = "https://investor.example.com/earnings.html"
    _rehash(issuer_url)
    with pytest.raises(PeadAnnouncementEvidenceError, match="SEC URL"):
        validate_pead_announcement_evidence(
            issuer_url, expected_event_manifest=universe
        )


def test_acceptance_and_later_retrieval_do_not_prove_first_public_time():
    universe = _universe(KEY_A)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )
    record = document["payload"]["outcomes"][0]["available_record"]
    assert record["edgar_acceptance_at_utc"] == ACCEPTED
    assert record["first_public_at_utc"] is None
    assert record["observed_public_by_at_utc"] is None
    assert timing_eligible_announcements(
        document, expected_event_manifest=universe
    ) == []

    relabeled = copy.deepcopy(document)
    relabeled_record = relabeled["payload"]["outcomes"][0]["available_record"]
    relabeled_record["first_public_at_utc"] = ACCEPTED
    relabeled_record["first_public_basis"] = "authoritative_sec_dissemination_timestamp"
    _rehash(relabeled)
    with pytest.raises(PeadAnnouncementEvidenceError, match="no replayable"):
        validate_pead_announcement_evidence(
            relabeled, expected_event_manifest=universe
        )


def test_http_date_proves_only_contemporaneous_observed_public_upper_bound():
    observation_date = "2019-01-29T21:05:30Z"
    observation_retrieved = "2019-01-29T21:05:35Z"
    observed = {
        "url": "https://www.sec.gov/Archives/edgar/data/320193/feed-entry",
        "retrieved_at_utc": observation_retrieved,
        "http": _http(observation_date),
        "bytes": b"contemporaneous SEC dissemination observation",
    }
    universe = _universe(KEY_A)
    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available(observed_public=observed)],
    )
    record = document["payload"]["outcomes"][0]["available_record"]
    assert record["observed_public_by_at_utc"] == observation_date
    assert record["first_public_at_utc"] is None
    assert document["payload"]["coverage"]["timing_eligible_events"] == 0

    historical_retrieval = {
        "url": observed["url"],
        "retrieved_at_utc": RETRIEVED,
        "http": _http(),
        "bytes": observed["bytes"],
    }
    with pytest.raises(PeadAnnouncementEvidenceError, match="not contemporaneous"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T15:30:00Z",
            outcomes=[_available(observed_public=historical_retrieval)],
        )


def test_nonreplayed_first_public_proof_fails_closed_and_actual_only_loads(
    tmp_path: Path,
):
    first_public = "2019-01-29T21:05:20Z"
    proof_raw = b"authoritative SEC dissemination timestamp 2019-01-29T21:05:20Z"
    proof = {
        "method": "sec_authoritative_dissemination_timestamp.v1",
        "code_hash": "e" * 64,
        "locator": "feed.acceptedAt",
        "source_document": _document(
            "first_public_proof",
            proof_raw,
            date_utc="2026-07-14T15:00:00Z",
            retrieved_at=RETRIEVED,
        ),
    }
    universe = _universe(KEY_A)
    with pytest.raises(PeadAnnouncementEvidenceError, match="no replayable"):
        build_pead_announcement_evidence(
            expected_event_manifest=universe,
            created_at_utc="2026-07-14T15:30:00Z",
            outcomes=[
                _available(
                    first_public_at=first_public,
                    first_public_proof=proof,
                )
            ],
        )

    document = build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-07-14T15:30:00Z",
        outcomes=[_available()],
    )

    path = tmp_path / "announcement.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_pead_announcement_evidence(
        path, expected_event_manifest=universe
    ) == document
