from __future__ import annotations

import copy

import pytest

from data.pead_event_universe import (
    PeadEventUniverseError,
    build_pead_event_census_receipt,
    build_pead_event_universe,
    build_pead_event_universe_v2,
    canonical_event_id,
    content_hash,
    validate_pead_event_universe,
)


KEYS = [
    {
        "cik": "0000320193",
        "fiscal_period_end": "2020-03-28",
        "fiscal_period_type": "Q",
    },
    {
        "cik": "0000789019",
        "fiscal_period_end": "2020-03-31",
        "fiscal_period_type": "Q",
    },
]
SOURCE_IDS = ["1" * 64, "2" * 64, "3" * 64]
BINDINGS = {
    "market_snapshot_sha256": "a" * 64,
    "identity_snapshot_sha256": "b" * 64,
    "candidate_specification_sha256": "c" * 64,
    "construction_code_sha256": "d" * 64,
    "canonical_query_sha256": "e" * 64,
}


def _receipt():
    return build_pead_event_census_receipt(
        raw_census_artifact_sha256="f" * 64,
        canonical_query_sha256=BINDINGS["canonical_query_sha256"],
        source_record_ids=SOURCE_IDS,
    )


def _dispositions(*, gap: bool = False):
    rows = [
        {
            "source_record_id": SOURCE_IDS[0],
            "disposition": "expected_event",
            "event_id": canonical_event_id(KEYS[0]),
            "event_key": KEYS[0],
            "reason": None,
        },
        {
            "source_record_id": SOURCE_IDS[1],
            "disposition": "expected_event" if not gap else "identity_gap",
            "event_id": canonical_event_id(KEYS[1]) if not gap else None,
            "event_key": KEYS[1] if not gap else None,
            "reason": None if not gap else "missing_cik_bridge",
        },
        {
            "source_record_id": SOURCE_IDS[2],
            "disposition": "out_of_scope",
            "event_id": None,
            "event_key": None,
            "reason": "annual_period",
        },
    ]
    return list(reversed(rows))


def _universe(*, gap: bool = False):
    return build_pead_event_universe(
        candidate_id="pead-test-v1",
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start="2020-01-01",
        event_end="2020-12-31",
        bindings=BINDINGS,
        census_receipt=_receipt(),
        census_dispositions=_dispositions(gap=gap),
    )


def _rehash(document):
    document["artifact_hash"] = content_hash(document["payload"])


def test_event_id_and_expected_set_are_canonical_and_census_derived():
    universe = _universe()

    assert canonical_event_id(KEYS[0]) == content_hash(KEYS[0])
    assert universe["artifact_hash"] == content_hash(universe["payload"])
    assert universe["payload"]["expected_event_ids"] == sorted(
        canonical_event_id(key) for key in KEYS
    )
    assert universe["payload"]["census_counts"] == {
        "source_record_count": 3,
        "disposition_count": 3,
        "expected_event_count": 2,
        "identity_gap_count": 0,
        "out_of_scope_count": 1,
        "census_complete": True,
    }
    assert universe["payload"]["qualification_allowed"] is True


def test_omitted_census_disposition_cannot_pass_against_bound_receipt():
    tampered = copy.deepcopy(_universe())
    payload = tampered["payload"]
    payload["census_dispositions"] = payload["census_dispositions"][:-1]
    payload["expected_events"] = [
        row for row in payload["expected_events"]
        if row["event_id"] in {
            item["event_id"] for item in payload["census_dispositions"]
            if item["event_id"] is not None
        }
    ]
    payload["expected_event_ids"] = [row["event_id"] for row in payload["expected_events"]]
    payload["census_counts"]["disposition_count"] -= 1
    _rehash(tampered)

    with pytest.raises(PeadEventUniverseError, match="every receipt record exactly once"):
        validate_pead_event_universe(tampered)


def test_duplicate_census_disposition_and_event_identity_are_rejected():
    duplicate_source = _dispositions()
    duplicate_source[0]["source_record_id"] = duplicate_source[1]["source_record_id"]
    with pytest.raises(PeadEventUniverseError, match="every receipt record exactly once"):
        build_pead_event_universe(
            candidate_id="pead-test-v1",
            frozen_at_utc="2026-07-14T12:00:00Z",
            event_start="2020-01-01",
            event_end="2020-12-31",
            bindings=BINDINGS,
            census_receipt=_receipt(),
            census_dispositions=duplicate_source,
        )

    duplicate_event = _dispositions()
    expected = [row for row in duplicate_event if row["disposition"] == "expected_event"]
    expected[1]["event_id"] = expected[0]["event_id"]
    expected[1]["event_key"] = expected[0]["event_key"]
    with pytest.raises(PeadEventUniverseError, match="event identities must be unique"):
        build_pead_event_universe(
            candidate_id="pead-test-v1",
            frozen_at_utc="2026-07-14T12:00:00Z",
            event_start="2020-01-01",
            event_end="2020-12-31",
            bindings=BINDINGS,
            census_receipt=_receipt(),
            census_dispositions=duplicate_event,
        )


def test_expected_period_must_be_inside_frozen_window():
    rows = _dispositions()
    expected = next(row for row in rows if row["disposition"] == "expected_event")
    expected["event_key"] = {**expected["event_key"], "fiscal_period_end": "2019-12-31"}
    expected["event_id"] = canonical_event_id(expected["event_key"])

    with pytest.raises(PeadEventUniverseError, match="outside event_window"):
        build_pead_event_universe(
            candidate_id="pead-test-v1",
            frozen_at_utc="2026-07-14T12:00:00Z",
            event_start="2020-01-01",
            event_end="2020-12-31",
            bindings=BINDINGS,
            census_receipt=_receipt(),
            census_dispositions=rows,
        )


def test_identity_gap_is_source_record_based_and_forces_nonqualification():
    universe = _universe(gap=True)

    assert universe["payload"]["identity_gaps"] == [
        {"source_record_id": SOURCE_IDS[1], "reason": "missing_cik_bridge"}
    ]
    assert universe["payload"]["blockers"] == ["identity_gaps_present"]
    assert universe["payload"]["qualification_allowed"] is False

    fake = copy.deepcopy(universe)
    fake["payload"]["blockers"] = []
    fake["payload"]["qualification_allowed"] = True
    _rehash(fake)
    with pytest.raises(PeadEventUniverseError, match="blockers are not derived"):
        validate_pead_event_universe(fake)


def test_v2_retains_identity_gap_but_qualifies_unrelated_expected_events():
    universe = build_pead_event_universe_v2(
        candidate_id="pead-test-v2",
        frozen_at_utc="2026-07-14T12:00:00Z",
        event_start="2020-01-01",
        event_end="2020-12-31",
        bindings=BINDINGS,
        census_receipt=_receipt(),
        census_dispositions=_dispositions(gap=True),
    )

    assert universe["payload"]["schema_version"] == "pead_event_universe.v2"
    assert universe["payload"]["identity_gaps"] == [
        {"source_record_id": SOURCE_IDS[1], "reason": "missing_cik_bridge"}
    ]
    assert universe["payload"]["census_counts"]["identity_gap_count"] == 1
    assert universe["payload"]["expected_event_ids"] == [
        canonical_event_id(KEYS[0])
    ]
    assert universe["payload"]["blockers"] == []
    assert universe["payload"]["qualification_allowed"] is True
    assert validate_pead_event_universe(universe) == universe

    v1 = _universe(gap=True)
    assert v1["payload"]["schema_version"] == "pead_event_universe.v1"
    assert v1["payload"]["qualification_allowed"] is False
