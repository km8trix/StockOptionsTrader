from __future__ import annotations

import pytest

from analysis.pead_known_by_policy import (
    KNOWN_BY_POLICY,
    KNOWN_BY_POLICY_SHA256,
    PeadKnownByPolicyError,
    activation_eastern_date,
    activation_eastern_day_start_utc,
    select_conservative_consensus_vintage,
    select_conservative_market_session,
    validate_prospective_consensus_freeze,
)


HASHES = {character: character * 64 for character in "123456789abcdef"}


def _vintage(
    *,
    as_of: str,
    available: str | None = None,
    analysts: int = 4,
    raw_hash: str = HASHES["1"],
):
    return {
        "provider_as_of_date": as_of,
        "trusted_available_at_utc": available,
        "analyst_count": analysts,
        "raw_record_sha256": raw_hash,
    }


def _market(
    session_date: str,
    close_at: str,
    *,
    raw_hash: str = HASHES["2"],
):
    return {
        "session_date": session_date,
        "session_close_at_utc": close_at,
        "close": "101.25",
        "closeunadj": "50.625",
        "source_row_sha256": raw_hash,
    }


def test_policy_identity_and_dst_boundary_are_frozen():
    assert KNOWN_BY_POLICY["activation_claim"] == "known_public_by"
    assert len(KNOWN_BY_POLICY_SHA256) == 64
    assert activation_eastern_date("2020-03-09T00:30:00Z").isoformat() == ("2020-03-08")
    assert (
        activation_eastern_day_start_utc("2020-03-09T00:30:00Z").isoformat()
        == "2020-03-08T05:00:00+00:00"
    )


def test_consensus_same_day_is_excluded_even_with_exact_time():
    selected, blockers, hashes = select_conservative_consensus_vintage(
        [
            _vintage(
                as_of="2020-05-01",
                available="2020-05-01T12:00:00Z",
            )
        ],
        known_public_by_at_utc="2020-05-01T20:00:00Z",
    )
    assert selected is None
    assert blockers == ["consensus_no_eligible_prior_day_vintage"]
    assert hashes == []

    # A provider date cannot make a contradictory same-day exact timestamp
    # eligible either.
    selected, blockers, hashes = select_conservative_consensus_vintage(
        [
            _vintage(
                as_of="2020-04-30",
                available="2020-05-01T12:00:00Z",
            )
        ],
        known_public_by_at_utc="2020-05-01T20:00:00Z",
    )
    assert selected is None
    assert blockers == ["consensus_no_eligible_prior_day_vintage"]
    assert hashes == []


def test_consensus_latest_tie_or_low_count_never_falls_back():
    earlier = _vintage(as_of="2020-04-29", raw_hash=HASHES["1"])
    tied_a = _vintage(as_of="2020-04-30", raw_hash=HASHES["2"])
    tied_b = _vintage(as_of="2020-04-30", raw_hash=HASHES["3"])
    selected, blockers, hashes = select_conservative_consensus_vintage(
        [earlier, tied_a, tied_b],
        known_public_by_at_utc="2020-05-01T20:00:00Z",
    )
    assert selected is None
    assert blockers == ["consensus_latest_vintage_ambiguous"]
    assert hashes == [HASHES["2"], HASHES["3"]]

    latest = _vintage(as_of="2020-04-30", analysts=1, raw_hash=HASHES["4"])
    selected, blockers, hashes = select_conservative_consensus_vintage(
        [earlier, latest],
        known_public_by_at_utc="2020-05-01T20:00:00Z",
    )
    assert selected == latest
    assert blockers == ["consensus_analyst_count_below_minimum"]
    assert hashes == [HASHES["4"]]


def test_market_selector_requires_unique_positive_prior_date():
    same_day = _market("2020-05-01", "2020-05-01T20:00:00Z")
    prior = _market("2020-04-30", "2020-04-30T20:00:00Z")
    selected, blockers = select_conservative_market_session(
        [prior, same_day],
        known_public_by_at_utc="2020-05-01T21:00:00Z",
    )
    assert selected == prior
    assert blockers == []

    duplicate = dict(prior, source_row_sha256=HASHES["3"])
    selected, blockers = select_conservative_market_session(
        [prior, duplicate],
        known_public_by_at_utc="2020-05-01T21:00:00Z",
    )
    assert selected is None
    assert blockers == ["market_latest_prior_session_ambiguous"]


def test_market_selector_rejects_future_and_nonpositive_rows():
    with pytest.raises(PeadKnownByPolicyError, match="at or after activation"):
        select_conservative_market_session(
            [_market("2020-04-30", "2020-05-01T21:00:00Z")],
            known_public_by_at_utc="2020-05-01T21:00:00Z",
        )
    bad = _market("2020-04-30", "2020-04-30T20:00:00Z")
    bad["close"] = "0"
    with pytest.raises(PeadKnownByPolicyError, match="positive finite"):
        select_conservative_market_session([bad], known_public_by_at_utc="2020-05-01T21:00:00Z")

    # Exact decimal validation must not underflow a valid positive source token
    # merely because it lies outside binary floating-point's normal range.
    tiny = _market("2020-04-30", "2020-04-30T20:00:00Z")
    tiny["close"] = "1e-9999"
    selected, blockers = select_conservative_market_session(
        [tiny], known_public_by_at_utc="2020-05-01T21:00:00Z"
    )
    assert selected == tiny
    assert blockers == []


def test_prospective_consensus_must_be_frozen_by_prior_close():
    validate_prospective_consensus_freeze(
        acquired_at_utc="2020-04-30T20:00:00Z",
        selected_prior_session_close_at_utc="2020-04-30T20:00:00Z",
    )
    with pytest.raises(PeadKnownByPolicyError, match="after the prior-session"):
        validate_prospective_consensus_freeze(
            acquired_at_utc="2020-04-30T20:00:01Z",
            selected_prior_session_close_at_utc="2020-04-30T20:00:00Z",
        )
