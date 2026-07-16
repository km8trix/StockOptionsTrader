from __future__ import annotations

import base64
import copy
import hashlib
import json

import pytest

from data.pead_announcement_availability import (
    CHECKPOINT_SCHEMA_VERSION,
    LICENSED_MANIFEST_SCHEMA_VERSION,
    LICENSED_RECORD_SCHEMA_VERSION,
    LICENSED_RELEASE_ADAPTER,
    SEC_HTTPS_OBSERVATION_ADAPTER,
    PeadAnnouncementAvailabilityError,
    build_licensed_distribution_outcome,
    build_missing_availability_outcome,
    build_pead_announcement_availability,
    build_sec_https_observation_outcome,
    eligible_announcement_activations,
    validate_pead_announcement_availability,
)
from data.pead_announcement_evidence import (
    build_pead_announcement_evidence,
    build_sec_available_outcome,
)
from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
    canonical_json,
    content_hash,
)


KEY = {
    "cik": "0000320193",
    "fiscal_period_end": "2026-03-31",
    "fiscal_period_type": "Q",
}
EVENT_ID = canonical_event_id(KEY)
ACCESSION = "0000320193-26-000010"
COMPACT_ACCESSION = ACCESSION.replace("-", "")
METADATA_URL = f"https://www.sec.gov/Archives/edgar/data/320193/{COMPACT_ACCESSION}/{ACCESSION}.txt"
EXHIBIT_URL = f"https://www.sec.gov/Archives/edgar/data/320193/{COMPACT_ACCESSION}/exhibit991.htm"
HASH = "a" * 64
ACTUAL_FIELDS = [
    "canonical_value",
    "metric",
    "metric_definition_sha256",
    "accounting_basis",
    "per_share_basis",
    "scope",
    "currency",
    "unit",
    "canonical_share_basis",
    "fiscal_period_end",
    "fiscal_period_type",
    "normalization_evidence_sha256",
]


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _raw(raw: bytes) -> dict:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _universe(*, frozen_at_utc: str = "2026-04-01T00:00:00Z") -> dict:
    receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256="b" * 64,
        canonical_query_sha256="c" * 64,
        source_record_ids=["d" * 64],
    )
    return build_pead_event_universe(
        candidate_id="pead-availability-test-v1",
        frozen_at_utc=frozen_at_utc,
        event_start="2026-01-01",
        event_end="2026-12-31",
        bindings={
            "market_snapshot_sha256": HASH,
            "identity_snapshot_sha256": HASH,
            "candidate_specification_sha256": HASH,
            "construction_code_sha256": HASH,
            "canonical_query_sha256": "c" * 64,
        },
        census_receipt=receipt,
        census_dispositions=[
            {
                "source_record_id": "d" * 64,
                "disposition": "expected_event",
                "event_id": EVENT_ID,
                "event_key": KEY,
                "reason": None,
            }
        ],
    )


def _actual() -> dict:
    return {
        "announced_value": "1.25",
        "canonical_value": "1.25",
        "normalization_factor": "1",
        "metric": "earnings_per_share",
        "source_metric_label": "Adjusted diluted earnings per share",
        "metric_definition_sha256": "e" * 64,
        "accounting_basis": "non_gaap",
        "per_share_basis": "diluted",
        "scope": "total_company",
        "currency": "USD",
        "unit": "currency_per_share",
        "announced_share_basis": "issuer_as_reported_at_publication",
        "canonical_share_basis": "split_restated",
        "fiscal_period_end": KEY["fiscal_period_end"],
        "fiscal_period_type": "Q",
        "normalization_evidence_sha256": "f" * 64,
    }


def _metadata_bytes() -> bytes:
    return (
        "<SEC-DOCUMENT>\n<SEC-HEADER>\n"
        f"<ACCESSION-NUMBER>{ACCESSION}\n"
        "<CONFORMED-SUBMISSION-TYPE>8-K\n"
        f"<CENTRAL-INDEX-KEY>{KEY['cik']}\n"
        "<ACCEPTANCE-DATETIME>20260501155900\n"
        "<ITEMS>2.02\n</SEC-HEADER>\n"
        "<DOCUMENT>\n<TYPE>EX-99.1\n<FILENAME>exhibit991.htm\n"
        "</DOCUMENT>\n"
    ).encode()


def _exhibit_bytes() -> bytes:
    return b"Adjusted diluted earnings per share were $1.25."


def _announcement(universe: dict) -> dict:
    http = {
        "status_code": 200,
        "date_utc": "2026-05-01T20:00:05Z",
        "content_type": "text/html; charset=UTF-8",
        "etag": '"source"',
        "last_modified_at_utc": None,
    }
    outcome = build_sec_available_outcome(
        event_key=KEY,
        accession_number=ACCESSION,
        exhibit="EX-99.1",
        metadata_url=METADATA_URL,
        metadata_retrieved_at_utc="2026-05-01T20:00:07Z",
        metadata_http=http,
        metadata_bytes=_metadata_bytes(),
        exhibit_url=EXHIBIT_URL,
        exhibit_retrieved_at_utc="2026-05-01T20:00:07Z",
        exhibit_http=http,
        exhibit_bytes=_exhibit_bytes(),
        edgar_acceptance_at_utc="2026-05-01T19:59:00Z",
        canonical_actual=_actual(),
        extraction={
            "method": "sec_exhibit_label_value_visible_text.v1",
            "code_hash": "1" * 64,
            "reviewer": "independent-test-reviewer",
            "locator": "visible_text:Adjusted diluted earnings per share",
        },
    )
    return build_pead_announcement_evidence(
        expected_event_manifest=universe,
        created_at_utc="2026-05-01T20:10:00Z",
        outcomes=[outcome],
    )


def _licensed_manifest(*, semantics="provider_attested_public_distribution_time") -> bytes:
    return _json_bytes(
        {
            "schema_version": LICENSED_MANIFEST_SCHEMA_VERSION,
            "provider_id": "licensed-release-archive",
            "dataset_id": "issuer-distributions",
            "license_evidence_sha256": "2" * 64,
            "record_schema_version": LICENSED_RECORD_SCHEMA_VERSION,
            "distribution_timestamp_field": "distribution_at_utc",
            "distribution_timestamp_semantics": semantics,
            "timestamp_precision": "second",
            "event_identity_fields": [
                "cik",
                "accession_number",
                "event_id",
                "event_key",
            ],
            "actual_binding_fields": ACTUAL_FIELDS,
            "source_document_binding": "sha256_exact_bytes",
        }
    )


def _licensed_record(announcement: dict, *, actual: dict | None = None) -> bytes:
    canonical_actual = actual or _actual()
    exhibit_hash = announcement["payload"]["outcomes"][0]["available_record"]["exhibit_document"][
        "raw_document"
    ]["sha256"]
    return _json_bytes(
        {
            "schema_version": LICENSED_RECORD_SCHEMA_VERSION,
            "provider_id": "licensed-release-archive",
            "dataset_id": "issuer-distributions",
            "provider_record_id": "release-2026-q1-aapl",
            "cik": KEY["cik"],
            "accession_number": ACCESSION,
            "event_id": EVENT_ID,
            "event_key": KEY,
            "distribution_at_utc": "2026-05-01T20:00:00Z",
            "source_document_sha256": exhibit_hash,
            "canonical_actual_sha256": content_hash(canonical_actual),
            "canonical_actual": canonical_actual,
        }
    )


def _licensed_outcome(
    announcement: dict, *, manifest: bytes | None = None, record: bytes | None = None
):
    manifest_bytes = manifest or _licensed_manifest()
    record_bytes = record or _licensed_record(announcement)
    return (
        build_licensed_distribution_outcome(
            event_key=KEY,
            provider_url="https://archive.provider.example/releases/aapl-2026-q1",
            retrieved_at_utc="2026-07-14T15:00:00Z",
            provider_manifest_bytes=manifest_bytes,
            provider_record_bytes=record_bytes,
        ),
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(record_bytes).hexdigest(),
    )


def _headers(*, date: str, body: bytes, content_type="text/plain") -> bytes:
    return (
        "HTTP/1.1 200 OK\r\n"
        f"Date: {date}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode("ascii")


def _checkpoint(bundle_hash: str, *, recorded_at="2026-05-01T20:00:10Z") -> bytes:
    return _json_bytes(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "authority_id": "independent-transparency-log",
            "authority_manifest_sha256": "3" * 64,
            "log_id": "pead-observations",
            "sequence": 42,
            "previous_checkpoint_sha256": "4" * 64,
            "observed_bundle_sha256": bundle_hash,
            "recorded_at_utc": recorded_at,
        }
    )


def _sec_outcome() -> tuple[dict, str]:
    metadata = _metadata_bytes()
    exhibit = _exhibit_bytes()
    placeholder = _checkpoint("0" * 64)
    outcome = build_sec_https_observation_outcome(
        event_key=KEY,
        metadata_url=METADATA_URL,
        metadata_received_at_utc="2026-05-01T20:00:08Z",
        metadata_headers_bytes=_headers(date="Fri, 01 May 2026 20:00:06 GMT", body=metadata),
        metadata_body_bytes=metadata,
        exhibit_url=EXHIBIT_URL,
        exhibit_received_at_utc="2026-05-01T20:00:09Z",
        exhibit_headers_bytes=_headers(
            date="Fri, 01 May 2026 20:00:07 GMT",
            body=exhibit,
            content_type="text/html",
        ),
        exhibit_body_bytes=exhibit,
        checkpoint_receipt_bytes=placeholder,
    )
    evidence = outcome["claim"]["evidence"]
    bundle_hash = content_hash(
        {
            "metadata_observation": evidence["metadata_observation"],
            "exhibit_observation": evidence["exhibit_observation"],
        }
    )
    receipt = _checkpoint(bundle_hash)
    outcome["claim"]["evidence"]["checkpoint_receipt"] = _raw(receipt)
    outcome["claim"]["known_public_by_at_utc"] = "2026-05-01T20:00:10Z"
    return outcome, hashlib.sha256(receipt).hexdigest()


def test_licensed_historical_distribution_qualifies_only_as_conservative_known_by():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement)

    artifact = build_pead_announcement_availability(
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        evidence_class="historical_reconstruction",
        created_at_utc="2026-07-14T15:01:00Z",
        outcomes=[outcome],
        trusted_provider_manifest_sha256s=[manifest_hash],
        trusted_provider_record_sha256s=[record_hash],
    )

    claim = artifact["payload"]["outcomes"][0]["claim"]
    assert claim["claim_kind"] == "known_public_by"
    assert claim["known_public_by_at_utc"] == "2026-05-01T20:00:00Z"
    assert claim["eligibility"] == {
        "claim_semantics": "conservative_known_public_by_upper_bound",
        "first_public_proven": False,
        "eligible_for_declared_evidence_class": True,
        "historical_reconstruction_allowed": True,
        "prospective_observation_allowed": False,
        "consensus_cutoff_rule": "strict_prior_eastern_calendar_date",
        "market_cutoff_rule": "strict_prior_nyse_session",
        "same_day_consensus_allowed": False,
        "same_day_market_close_allowed": False,
    }
    assert artifact["payload"]["coverage"]["complete"] is True
    assert (
        validate_pead_announcement_availability(
            artifact,
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )
        == artifact
    )

    activations = eligible_announcement_activations(
        artifact,
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        trusted_provider_manifest_sha256s=[manifest_hash],
        trusted_provider_record_sha256s=[record_hash],
    )
    assert activations == [
        {
            "event_id": EVENT_ID,
            "event_key": KEY,
            "claim_kind": "known_public_by",
            "signal_activation_at_utc": "2026-05-01T20:00:00Z",
            "adapter_id": LICENSED_RELEASE_ADAPTER,
            "consensus_cutoff_rule": "strict_prior_eastern_calendar_date",
            "market_cutoff_rule": "strict_prior_nyse_session",
            "announcement_availability_artifact_hash": artifact["artifact_hash"],
            "announcement_evidence_artifact_hash": announcement["artifact_hash"],
        }
    ]


def test_licensed_manifest_must_be_external_trust_input_not_embedded_assertion():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, _, _ = _licensed_outcome(announcement)

    with pytest.raises(PeadAnnouncementAvailabilityError, match="trust registry"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
        )


def test_licensed_record_must_be_external_trust_input_not_embedded_assertion():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, manifest_hash, _ = _licensed_outcome(announcement)

    with pytest.raises(PeadAnnouncementAvailabilityError, match="provider record"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
            trusted_provider_manifest_sha256s=[manifest_hash],
        )


def test_licensed_manifest_requires_provider_attested_timestamp_and_exact_metric_semantics():
    universe = _universe()
    announcement = _announcement(universe)
    manifest = _licensed_manifest(semantics="provider_reported_date")
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement, manifest=manifest)

    with pytest.raises(PeadAnnouncementAvailabilityError, match="does not attest"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )

    changed_actual = _actual()
    changed_actual["accounting_basis"] = "gaap"
    record = _licensed_record(announcement, actual=changed_actual)
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement, record=record)
    with pytest.raises(PeadAnnouncementAvailabilityError, match="metric/actual"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cik", "0000789019", "CIK differs"),
        ("accession_number", "0000320193-26-000011", "accession differs"),
        ("event_id", "9" * 64, "event identity differs"),
    ],
)
def test_licensed_record_must_bind_exact_event_and_accession(field, value, message):
    universe = _universe()
    announcement = _announcement(universe)
    record = json.loads(_licensed_record(announcement))
    record[field] = value
    outcome, manifest_hash, record_hash = _licensed_outcome(
        announcement, record=_json_bytes(record)
    )

    with pytest.raises(PeadAnnouncementAvailabilityError, match=message):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )


def test_licensed_historical_archive_cannot_be_relabelled_as_prospective():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement)

    with pytest.raises(PeadAnnouncementAvailabilityError, match="ineligible"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="prospective",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )


def test_prospective_sec_positive_observation_uses_external_checkpoint_as_bound():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, checkpoint_hash = _sec_outcome()

    artifact = build_pead_announcement_availability(
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        evidence_class="prospective",
        created_at_utc="2026-05-01T20:11:00Z",
        outcomes=[outcome],
        trusted_checkpoint_sha256s=[checkpoint_hash],
    )

    claim = artifact["payload"]["outcomes"][0]["claim"]
    assert claim["adapter_id"] == SEC_HTTPS_OBSERVATION_ADAPTER
    assert claim["known_public_by_at_utc"] == "2026-05-01T20:00:10Z"
    assert claim["eligibility"]["first_public_proven"] is False
    assert claim["eligibility"]["prospective_observation_allowed"] is True
    assert claim["eligibility"]["same_day_consensus_allowed"] is False
    assert artifact["payload"]["coverage"]["complete"] is True


def test_sec_http_observation_can_never_reconstruct_historical_timing():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, checkpoint_hash = _sec_outcome()

    with pytest.raises(PeadAnnouncementAvailabilityError, match="prospective-only"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-05-01T20:11:00Z",
            outcomes=[outcome],
            trusted_checkpoint_sha256s=[checkpoint_hash],
        )


def test_prospective_sec_observation_requires_ex_ante_frozen_universe():
    universe = _universe(frozen_at_utc="2026-05-01T20:05:00Z")
    announcement = _announcement(universe)
    outcome, checkpoint_hash = _sec_outcome()

    with pytest.raises(PeadAnnouncementAvailabilityError, match="not frozen before"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="prospective",
            created_at_utc="2026-05-01T20:11:00Z",
            outcomes=[outcome],
            trusted_checkpoint_sha256s=[checkpoint_hash],
        )


def test_sec_observation_requires_external_trust_and_exact_checkpoint_bundle():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, checkpoint_hash = _sec_outcome()

    with pytest.raises(PeadAnnouncementAvailabilityError, match="trust registry"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="prospective",
            created_at_utc="2026-05-01T20:11:00Z",
            outcomes=[outcome],
        )

    tampered = copy.deepcopy(outcome)
    checkpoint = json.loads(
        base64.b64decode(tampered["claim"]["evidence"]["checkpoint_receipt"]["base64"])
    )
    checkpoint["observed_bundle_sha256"] = "8" * 64
    raw = _json_bytes(checkpoint)
    tampered["claim"]["evidence"]["checkpoint_receipt"] = _raw(raw)
    tampered_hash = hashlib.sha256(raw).hexdigest()
    with pytest.raises(PeadAnnouncementAvailabilityError, match="exact positive SEC"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="prospective",
            created_at_utc="2026-05-01T20:11:00Z",
            outcomes=[tampered],
            trusted_checkpoint_sha256s=[checkpoint_hash, tampered_hash],
        )


def test_sec_observation_replays_exact_announcement_body_and_http_chronology():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, checkpoint_hash = _sec_outcome()
    wrong_body = b"Adjusted diluted earnings per share were $9.99."
    outcome["claim"]["evidence"]["exhibit_observation"]["raw_body"] = _raw(wrong_body)

    with pytest.raises(PeadAnnouncementAvailabilityError, match="body differs"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="prospective",
            created_at_utc="2026-05-01T20:11:00Z",
            outcomes=[outcome],
            trusted_checkpoint_sha256s=[checkpoint_hash],
        )

    outcome, checkpoint_hash = _sec_outcome()
    outcome["claim"]["evidence"]["metadata_observation"]["received_at_utc"] = "2026-05-01T19:59:59Z"
    with pytest.raises(PeadAnnouncementAvailabilityError, match="chronology"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="prospective",
            created_at_utc="2026-05-01T20:11:00Z",
            outcomes=[outcome],
            trusted_checkpoint_sha256s=[checkpoint_hash],
        )


def test_claim_kind_is_closed_and_never_accepts_first_public_relabeling():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement)
    outcome["claim"]["claim_kind"] = "first_public"

    with pytest.raises(PeadAnnouncementAvailabilityError, match="never first_public"):
        build_pead_announcement_availability(
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            evidence_class="historical_reconstruction",
            created_at_utc="2026-07-14T15:01:00Z",
            outcomes=[outcome],
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )


def test_missing_claim_is_exhaustive_but_explicitly_incomplete():
    universe = _universe()
    announcement = _announcement(universe)
    artifact = build_pead_announcement_availability(
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        evidence_class="historical_reconstruction",
        created_at_utc="2026-07-14T15:01:00Z",
        outcomes=[
            build_missing_availability_outcome(
                KEY, reason="licensed_distribution_archive_not_acquired"
            )
        ],
    )

    assert artifact["payload"]["coverage"] == {
        "expected_events": 1,
        "available_claims": 0,
        "eligible_claims": 0,
        "missing_claims": 1,
        "event_universe_qualified": True,
        "announcement_actuals_complete": True,
        "blockers": ["expected_events_missing_availability"],
        "complete": False,
    }


def test_rehashed_eligibility_relaxation_is_rejected_by_source_replay():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement)
    artifact = build_pead_announcement_availability(
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        evidence_class="historical_reconstruction",
        created_at_utc="2026-07-14T15:01:00Z",
        outcomes=[outcome],
        trusted_provider_manifest_sha256s=[manifest_hash],
        trusted_provider_record_sha256s=[record_hash],
    )
    tampered = copy.deepcopy(artifact)
    tampered["payload"]["outcomes"][0]["claim"]["eligibility"]["same_day_consensus_allowed"] = True
    tampered["artifact_hash"] = content_hash(tampered["payload"])

    with pytest.raises(PeadAnnouncementAvailabilityError, match="not derived exactly"):
        validate_pead_announcement_availability(
            tampered,
            expected_event_manifest=universe,
            announcement_evidence=announcement,
            trusted_provider_manifest_sha256s=[manifest_hash],
            trusted_provider_record_sha256s=[record_hash],
        )


def test_artifact_serialization_is_content_addressed_canonical_json():
    universe = _universe()
    announcement = _announcement(universe)
    outcome, manifest_hash, record_hash = _licensed_outcome(announcement)
    artifact = build_pead_announcement_availability(
        expected_event_manifest=universe,
        announcement_evidence=announcement,
        evidence_class="historical_reconstruction",
        created_at_utc="2026-07-14T15:01:00Z",
        outcomes=[outcome],
        trusted_provider_manifest_sha256s=[manifest_hash],
        trusted_provider_record_sha256s=[record_hash],
    )

    assert (
        artifact["artifact_hash"]
        == hashlib.sha256(canonical_json(artifact["payload"]).encode()).hexdigest()
    )
