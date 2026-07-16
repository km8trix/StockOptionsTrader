from __future__ import annotations

from copy import deepcopy

import pytest

import analysis.pead_signal_input_index as subject
import analysis.pead_signal_input_reconciliation as signal_subject
from analysis.pead_signal_input_index import (
    PeadSignalInputIndexError,
    build_pead_signal_input_index,
    load_pead_signal_input_index,
    publish_pead_signal_input_index,
    validate_pead_signal_input_index_structure,
    verify_pead_signal_input_index,
)
from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
    canonical_json,
    content_hash,
)
from data.pead_event_universe_index import build_pead_event_universe_index


CANDIDATE_ID = "pead-vq-source-qualification-v3"
CANDIDATE_HASH = "a" * 64
CONSTRUCTION_HASH = "b" * 64
SIGNAL_CODE_HASH = "c" * 64
IDENTITY_ID = "d" * 64
SOURCE_POLICY_HASH = "e" * 64
MARKET_POLICY_HASH = "f" * 64


def _trust_hash(values):
    return content_hash(
        {
            "schema_version": "pead_sha256_trust_root_set.v1",
            "members": sorted(values),
        }
    )


def _partition(year: int):
    end = "2024-09-30" if year == 2024 else f"{year}-12-31"
    key = {
        "cik": f"{year:010d}",
        "fiscal_period_end": f"{year}-03-31",
        "fiscal_period_type": "Q",
    }
    event_id = canonical_event_id(key)
    source_id = content_hash({"source": year})
    query = content_hash({"query": year})
    receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256=content_hash({"raw": year}),
        canonical_query_sha256=query,
        source_record_ids=[source_id],
    )
    return build_pead_event_universe(
        candidate_id=CANDIDATE_ID,
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start=f"{year}-01-01",
        event_end=end,
        bindings={
            "market_snapshot_sha256": "1" * 64,
            "identity_snapshot_sha256": "2" * 64,
            "candidate_specification_sha256": CANDIDATE_HASH,
            "construction_code_sha256": CONSTRUCTION_HASH,
            "canonical_query_sha256": query,
        },
        census_receipt=receipt,
        census_dispositions=[
            {
                "source_record_id": source_id,
                "disposition": "expected_event",
                "event_id": event_id,
                "event_key": key,
                "reason": None,
            }
        ],
    )


def _accepted_row(event):
    event_id = event["event_id"]
    event_key = event["event_key"]
    source_input = {
        "event_id": event_id,
        "event_key": event_key,
        "actual_value": "1.25",
        "consensus_value": "1",
        "raw_surprise": "0.25",
        "surprise_direction": "positive",
        "analyst_count": 2,
        "known_public_by_at_utc": "2020-05-01T20:00:00Z",
        "availability_adapter_id": "licensed_historical_actuals.v1",
        "consensus_provider_as_of_date": "2020-04-30",
        "consensus_available_at_utc": "2020-04-30T19:30:00Z",
        "consensus_availability_precision": "timestamp",
        "consensus_receipt_captured_at_utc": "2020-04-30T19:30:00Z",
        "consensus_cutoff_rule": "strict_prior_eastern_calendar_date",
        "market_cutoff_rule": "strictly_prior_observed_nyse_session",
        "metric": {
            "metric_id": "earnings_per_share",
            "accounting_basis": "non_gaap",
            "per_share_basis": "diluted",
            "scope": "total_company",
            "canonical_share_basis": "split_restated",
            "currency_code": "USD",
            "unit": "currency_per_share",
            "metric_definition_sha256": "3" * 64,
        },
        "provenance": {"consensus_raw_record_sha256": "4" * 64},
    }
    denominator = {
        "ticker": "AAA",
        "permaticker": 101,
        "identity_id": IDENTITY_ID,
        "close_split_normalized": "12",
    }
    return {
        "event_id": event_id,
        "event_key": event_key,
        "source_disposition": "event_source_reconciled",
        "market_disposition": "market_accounting_evidenced",
        "disposition": "signal_input_accepted",
        "source_blockers": [],
        "market_blockers": [],
        "reconciliation_blockers": [],
        "identity": {
            "ticker": "AAA",
            "permaticker": 101,
            "identity_id": IDENTITY_ID,
        },
        "source_input": source_input,
        "market_denominator": denominator,
        "signal": signal_subject._signal(source_input, denominator),
    }


def _excluded_row(event):
    return {
        "event_id": event["event_id"],
        "event_key": event["event_key"],
        "source_disposition": "excluded",
        "market_disposition": "upstream_excluded",
        "disposition": "signal_input_excluded",
        "source_blockers": ["consensus_event_missing"],
        "market_blockers": ["upstream_not_event_source_reconciled"],
        "reconciliation_blockers": ["source_not_reconciled"],
        "identity": None,
        "source_input": None,
        "market_denominator": None,
        "signal": None,
    }


def _signal_receipt(universe, *, accepted: bool):
    event = universe["payload"]["expected_events"][0]
    row = _accepted_row(event) if accepted else _excluded_row(event)
    blocker_counts = (
        {}
        if accepted
        else {
            "market__upstream_not_event_source_reconciled": 1,
            "reconciliation__source_not_reconciled": 1,
            "source__consensus_event_missing": 1,
        }
    )
    payload = {
        "schema_version": "pead_signal_input_reconciliation.v1",
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "historical_reconstruction",
        "created_at_utc": "2026-07-14T13:00:00Z",
        "policy": signal_subject._POLICY,
        "trust_policy": {
            "candidate_specification_set_sha256": _trust_hash([CANDIDATE_HASH]),
            "construction_code_set_sha256": _trust_hash([CONSTRUCTION_HASH]),
            "signal_reconciliation_code_set_sha256": _trust_hash([SIGNAL_CODE_HASH]),
            "source_reconciliation_set_sha256": "5" * 64,
            "market_accounting_evidence_set_sha256": "6" * 64,
        },
        "bindings": {
            "candidate_specification_sha256": CANDIDATE_HASH,
            "construction_code_sha256": CONSTRUCTION_HASH,
            "signal_reconciliation_code_sha256": SIGNAL_CODE_HASH,
            "source_reconciliation_sha256": "7" * 64,
            "market_accounting_evidence_sha256": "8" * 64,
            "event_universe_sha256": universe["artifact_hash"],
            "source_reconciliation_bindings_sha256": "9" * 64,
            "market_accounting_bindings_sha256": "0" * 64,
            "market_accounting_trust_policy_sha256": "1" * 64,
            "known_by_policy_sha256": signal_subject.KNOWN_BY_POLICY_SHA256,
            "source_reconciliation_policy_sha256": SOURCE_POLICY_HASH,
            "market_accounting_policy_sha256": MARKET_POLICY_HASH,
            "signal_input_reconciliation_policy_sha256": content_hash(
                signal_subject._POLICY
            ),
        },
        "event_results": [row],
        "coverage": {
            "expected_event_count": 1,
            "source_reconciled_event_count": int(accepted),
            "market_accounting_evidenced_count": int(accepted),
            "signal_input_accepted_count": int(accepted),
            "signal_input_excluded_count": int(not accepted),
            "exhaustive_event_accounting": True,
            "partial_coverage": not accepted,
            "blocker_counts": blocker_counts,
        },
        "qualification": {
            "has_research_consumable_signal_inputs": accepted,
            "all_expected_events_signal_accepted": accepted,
            "signal_input_reconciliation_allowed": accepted,
            "research_consumable": accepted,
            "historical_replication_allowed": accepted,
            "prospective_accumulation_allowed": False,
            "edge_claim_allowed": False,
            "paper_execution_allowed": False,
            "live_deployment_allowed": False,
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


@pytest.fixture
def bundle(monkeypatch):
    universes = [_partition(year) for year in range(2015, 2025)]
    index = build_pead_event_universe_index(
        partitions=universes,
        target_start="2015-01-01",
        target_end="2024-09-30",
        indexed_at_utc="2026-07-14T12:00:00Z",
    )
    replay_payload = {
        "candidate_id": CANDIDATE_ID,
        "created_at_utc": "2026-07-14T12:00:00Z",
        "target_window": {"start": "2015-01-01", "end": "2024-09-30"},
        "bindings": {
            "candidate_specification_sha256": CANDIDATE_HASH,
            "construction_code_sha256": CONSTRUCTION_HASH,
        },
        "years": [
            {
                "partition_id": str(year),
                "event_window": universe["payload"]["event_window"],
                "event_universe": universe,
            }
            for year, universe in zip(range(2015, 2025), universes, strict=True)
        ],
    }
    replay = {"artifact_hash": content_hash(replay_payload), "payload": replay_payload}
    monkeypatch.setattr(
        subject,
        "validate_pead_sharadar_event_universe_replay_structure",
        lambda document: deepcopy(document),
    )
    signals = [
        _signal_receipt(universe, accepted=position == 0)
        for position, universe in enumerate(universes)
    ]
    kwargs = {
        "indexed_at_utc": "2026-07-14T14:00:00Z",
        "trusted_signal_input_reconciliation_sha256s": {
            signal["artifact_hash"] for signal in signals
        },
        "trusted_candidate_specification_sha256s": {CANDIDATE_HASH},
        "trusted_construction_code_sha256s": {CONSTRUCTION_HASH},
        "trusted_signal_reconciliation_code_sha256s": {SIGNAL_CODE_HASH},
    }
    return {
        "replay": replay,
        "index": index,
        "universes": universes,
        "signals": signals,
        "kwargs": kwargs,
    }


def _build(bundle, **overrides):
    kwargs = {**bundle["kwargs"], **overrides}
    return build_pead_signal_input_index(
        bundle["replay"], bundle["index"], list(reversed(bundle["signals"])), **kwargs
    )


def test_index_binds_ten_years_in_root_order_and_aggregates_coverage(bundle):
    document = _build(bundle)

    assert [row["partition_id"] for row in document["payload"]["partitions"]] == [
        str(year) for year in range(2015, 2025)
    ]
    assert document["payload"]["coverage"] == {
        "partition_count": 10,
        "expected_event_count": 10,
        "source_reconciled_event_count": 1,
        "market_accounting_evidenced_count": 1,
        "signal_input_accepted_count": 1,
        "signal_input_excluded_count": 9,
        "exhaustive_event_accounting": True,
        "cross_partition_event_ids_unique": True,
        "partial_coverage": True,
        "blocker_counts": {
            "market__upstream_not_event_source_reconciled": 9,
            "reconciliation__source_not_reconciled": 9,
            "source__consensus_event_missing": 9,
        },
    }
    assert document["payload"]["qualification"] == {
        "has_research_consumable_signal_inputs": True,
        "all_expected_events_signal_accepted": False,
        "all_partitions_research_consumable": False,
        "signal_input_index_allowed": True,
        "research_consumable": True,
        "historical_replication_allowed": True,
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    assert validate_pead_signal_input_index_structure(document) == document


def test_exactly_one_externally_trusted_receipt_is_required_per_partition(bundle):
    duplicate = [bundle["signals"][0], bundle["signals"][0], *bundle["signals"][2:]]
    with pytest.raises(PeadSignalInputIndexError, match="more than one"):
        build_pead_signal_input_index(
            bundle["replay"], bundle["index"], duplicate, **bundle["kwargs"]
        )

    trusted = set(bundle["kwargs"]["trusted_signal_input_reconciliation_sha256s"])
    trusted.remove(bundle["signals"][5]["artifact_hash"])
    with pytest.raises(PeadSignalInputIndexError, match="external trust registry"):
        _build(bundle, trusted_signal_input_reconciliation_sha256s=trusted)


def test_child_event_identity_and_order_must_exactly_equal_annual_universe(bundle):
    forged = deepcopy(bundle["signals"])
    forged[2]["payload"]["event_results"] = deepcopy(
        forged[1]["payload"]["event_results"]
    )
    forged[2]["artifact_hash"] = content_hash(forged[2]["payload"])
    trusted = {signal["artifact_hash"] for signal in forged}

    with pytest.raises(PeadSignalInputIndexError, match="differ from the bound universe order"):
        build_pead_signal_input_index(
            bundle["replay"],
            bundle["index"],
            forged,
            **{
                **bundle["kwargs"],
                "trusted_signal_input_reconciliation_sha256s": trusted,
            },
        )


def test_common_specification_construction_and_final_code_trust_is_fail_closed(bundle):
    for field in (
        "trusted_candidate_specification_sha256s",
        "trusted_construction_code_sha256s",
        "trusted_signal_reconciliation_code_sha256s",
    ):
        with pytest.raises(PeadSignalInputIndexError, match="external trust registry"):
            _build(bundle, **{field: set()})

    forged = deepcopy(bundle["signals"])
    forged[2]["payload"]["trust_policy"][
        "signal_reconciliation_code_set_sha256"
    ] = "f" * 64
    forged[2]["artifact_hash"] = content_hash(forged[2]["payload"])
    with pytest.raises(PeadSignalInputIndexError, match="common external trust"):
        build_pead_signal_input_index(
            bundle["replay"],
            bundle["index"],
            forged,
            **{
                **bundle["kwargs"],
                "trusted_signal_input_reconciliation_sha256s": {
                    signal["artifact_hash"] for signal in forged
                },
            },
        )


def test_self_consistent_count_tampering_fails_structural_derivation(bundle):
    document = _build(bundle)
    forged = deepcopy(document)
    forged["payload"]["coverage"]["signal_input_accepted_count"] = 2
    forged["artifact_hash"] = content_hash(forged["payload"])

    with pytest.raises(PeadSignalInputIndexError, match="coverage is not derived"):
        validate_pead_signal_input_index_structure(forged)


def test_authoritative_verifier_replays_root_and_each_partition_context(
    bundle, monkeypatch
):
    document = _build(bundle)
    calls = []

    def verify_roots(replay, index, **kwargs):
        calls.append(("root", kwargs))
        return {"replay": replay, "index": index}

    def verify_child(child, **kwargs):
        calls.append((child["payload"]["bindings"]["event_universe_sha256"], kwargs))
        return child

    monkeypatch.setattr(subject, "verify_pead_sharadar_event_universe_replay", verify_roots)
    monkeypatch.setattr(subject, "verify_pead_signal_input_reconciliation", verify_child)
    contexts = {str(year): {"partition_marker": str(year)} for year in range(2015, 2025)}

    assert (
        verify_pead_signal_input_index(
            document,
            bundle["replay"],
            bundle["index"],
            bundle["signals"],
            event_replay_verification_kwargs={"root_marker": True},
            child_verification_contexts=contexts,
            **{key: value for key, value in bundle["kwargs"].items() if key != "indexed_at_utc"},
        )
        == document
    )
    assert calls[0] == ("root", {"root_marker": True})
    assert [call[1]["partition_marker"] for call in calls[1:]] == [
        str(year) for year in range(2015, 2025)
    ]

    with pytest.raises(PeadSignalInputIndexError, match="cover exactly"):
        verify_pead_signal_input_index(
            document,
            bundle["replay"],
            bundle["index"],
            bundle["signals"],
            event_replay_verification_kwargs={},
            child_verification_contexts={"2015": {}},
            **{key: value for key, value in bundle["kwargs"].items() if key != "indexed_at_utc"},
        )


def test_authoritative_child_verifier_cannot_substitute_another_receipt(bundle, monkeypatch):
    document = _build(bundle)
    monkeypatch.setattr(
        subject,
        "verify_pead_sharadar_event_universe_replay",
        lambda replay, index, **kwargs: {"replay": replay, "index": index},
    )
    monkeypatch.setattr(
        subject,
        "verify_pead_signal_input_reconciliation",
        lambda child, **kwargs: bundle["signals"][1],
    )

    with pytest.raises(PeadSignalInputIndexError, match="returned another"):
        verify_pead_signal_input_index(
            document,
            bundle["replay"],
            bundle["index"],
            bundle["signals"],
            event_replay_verification_kwargs={},
            child_verification_contexts={str(year): {} for year in range(2015, 2025)},
            **{key: value for key, value in bundle["kwargs"].items() if key != "indexed_at_utc"},
        )


def test_create_only_publication_and_strict_loader(bundle, tmp_path, monkeypatch):
    document = _build(bundle)
    path = tmp_path / "published" / "signal-input-index.json"

    with pytest.raises(PeadSignalInputIndexError, match="requires authoritative"):
        publish_pead_signal_input_index(document, path)
    assert not path.exists()

    normalized, published = publish_pead_signal_input_index(
        document, path, allow_structural_only=True
    )
    assert normalized == document
    assert published == path
    assert path.read_bytes() == (canonical_json(document) + "\n").encode("utf-8")
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(PeadSignalInputIndexError, match="already exists"):
        publish_pead_signal_input_index(document, path, allow_structural_only=True)

    monkeypatch.setattr(subject, "verify_pead_signal_input_index", lambda value, **kwargs: value)
    assert load_pead_signal_input_index(path) == document
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"artifact_hash":"x","artifact_hash":"y"}\n', encoding="utf-8")
    with pytest.raises(PeadSignalInputIndexError, match="duplicate key"):
        load_pead_signal_input_index(duplicate)


def test_index_cannot_predate_event_roots_or_annual_receipts(bundle):
    with pytest.raises(PeadSignalInputIndexError, match="predates partition"):
        _build(bundle, indexed_at_utc="2026-07-14T12:30:00Z")
