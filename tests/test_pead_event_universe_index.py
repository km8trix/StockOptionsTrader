from __future__ import annotations

import copy

import pytest

from data.pead_event_universe import (
    build_pead_event_census_receipt,
    build_pead_event_universe,
    canonical_event_id,
    content_hash,
)
from data.pead_event_universe_index import (
    PeadEventUniverseIndexError,
    build_pead_event_universe_index,
    validate_pead_event_universe_index,
)


HASHES = {character: character * 64 for character in "123456789abcdef"}


def _partition(year: int, *, cik_suffix: int):
    key = {
        "cik": f"{cik_suffix:010d}",
        "fiscal_period_end": f"{year}-03-31",
        "fiscal_period_type": "Q",
    }
    event_id = canonical_event_id(key)
    source_id = content_hash({"year": year, "source": cik_suffix})
    query = content_hash({"year": year, "dimension": "ARQ"})
    receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256=content_hash({"raw": year}),
        canonical_query_sha256=query,
        source_record_ids=[source_id],
    )
    return build_pead_event_universe(
        candidate_id="pead-index-test-v1",
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start=f"{year}-01-01",
        event_end=f"{year}-12-31",
        bindings={
            "market_snapshot_sha256": HASHES["a"],
            "identity_snapshot_sha256": HASHES["b"],
            "candidate_specification_sha256": HASHES["c"],
            "construction_code_sha256": HASHES["d"],
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


def test_index_replays_contiguous_year_partitions():
    partitions = [_partition(2020, cik_suffix=320193), _partition(2021, cik_suffix=789019)]
    document = build_pead_event_universe_index(
        partitions=partitions,
        target_start="2020-01-01",
        target_end="2021-12-31",
        indexed_at_utc="2026-07-14T13:00:00Z",
    )
    assert document["payload"]["counts"] == {
        "partition_count": 2,
        "event_count": 2,
        "source_record_count": 2,
        "identity_gap_count": 0,
    }
    assert document["payload"]["qualification"]["qualification_allowed"] is True
    assert validate_pead_event_universe_index(document, partitions=partitions) == document


def test_missing_year_and_cross_partition_event_are_rejected():
    with pytest.raises(PeadEventUniverseIndexError, match="every target calendar year"):
        build_pead_event_universe_index(
            partitions=[_partition(2020, cik_suffix=320193)],
            target_start="2020-01-01",
            target_end="2021-12-31",
            indexed_at_utc="2026-07-14T13:00:00Z",
        )

    first = _partition(2020, cik_suffix=320193)
    second = _partition(2021, cik_suffix=789019)
    second["payload"]["expected_events"][0]["event_key"]["fiscal_period_end"] = "2020-03-31"
    second["artifact_hash"] = content_hash(second["payload"])
    with pytest.raises(PeadEventUniverseIndexError):
        build_pead_event_universe_index(
            partitions=[first, second],
            target_start="2020-01-01",
            target_end="2021-12-31",
            indexed_at_utc="2026-07-14T13:00:00Z",
        )


def test_self_consistent_index_edit_cannot_survive_replay():
    partitions = [_partition(2020, cik_suffix=320193)]
    document = build_pead_event_universe_index(
        partitions=partitions,
        target_start="2020-01-01",
        target_end="2020-12-31",
        indexed_at_utc="2026-07-14T13:00:00Z",
    )
    tampered = copy.deepcopy(document)
    tampered["payload"]["counts"]["event_count"] = 2
    tampered["artifact_hash"] = content_hash(tampered["payload"])
    with pytest.raises(PeadEventUniverseIndexError, match="does not replay"):
        validate_pead_event_universe_index(tampered, partitions=partitions)
