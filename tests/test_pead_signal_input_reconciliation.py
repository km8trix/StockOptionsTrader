from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import analysis.pead_signal_input_reconciliation as subject
import data.pead_market_accounting_evidence as market_subject
from analysis.pead_signal_input_reconciliation import (
    PeadSignalInputReconciliationError,
    build_pead_signal_input_reconciliation,
    load_pead_signal_input_reconciliation,
    publish_pead_signal_input_reconciliation,
    validate_pead_signal_input_reconciliation_structure,
    verify_pead_signal_input_reconciliation,
)
from data.pead_event_universe import canonical_event_id, canonical_json, content_hash


EVENT_KEY = {
    "cik": "0000000123",
    "fiscal_period_end": "2020-03-31",
    "fiscal_period_type": "Q",
}
EXCLUDED_KEY = {
    "cik": "0000000456",
    "fiscal_period_end": "2020-03-31",
    "fiscal_period_type": "Q",
}
EVENT_ID = canonical_event_id(EVENT_KEY)
EXCLUDED_ID = canonical_event_id(EXCLUDED_KEY)
SOURCE_HASH = "1" * 64
MARKET_HASH = "2" * 64
UNIVERSE_HASH = "3" * 64
SOURCE_POLICY_HASH = "4" * 64
MARKET_POLICY_HASH = "5" * 64
IDENTITY_ID = "6" * 64
METRIC_HASH = "7" * 64
CANDIDATE = "pead-vq-source-qualification-v3"
FORMULA = (
    "(independent_canonical_actual_eps - selected_point_in_time_consensus_eps) / "
    "strictly_prior_split_normalized_SEP_close"
)


def _trust_hash(values) -> str:
    return content_hash(
        {
            "schema_version": "pead_sha256_trust_root_set.v1",
            "members": sorted(values),
        }
    )


def _source_input(*, share_basis="split_restated", currency="USD", unit="currency_per_share"):
    return {
        "event_id": EVENT_ID,
        "event_key": EVENT_KEY,
        "actual_value": "1.25",
        "consensus_value": "1",
        "raw_surprise": "0.25",
        "surprise_direction": "positive",
        "analyst_count": 4,
        "known_public_by_at_utc": "2020-05-01T20:00:00Z",
        "availability_adapter_id": "licensed_release_distribution_archive.v1",
        "consensus_provider_as_of_date": "2020-04-30",
        "consensus_available_at_utc": "2020-04-30T19:00:00Z",
        "consensus_availability_precision": "second",
        "consensus_receipt_captured_at_utc": "2020-04-30T19:30:00Z",
        "consensus_cutoff_rule": "strict_prior_eastern_calendar_date",
        "market_cutoff_rule": "strictly_prior_observed_nyse_session",
        "metric": {
            "metric_id": "earnings_per_share",
            "accounting_basis": "non_gaap",
            "per_share_basis": "diluted",
            "scope": "total_company",
            "canonical_share_basis": share_basis,
            "currency_code": currency,
            "unit": unit,
            "metric_definition_sha256": METRIC_HASH,
        },
        "provenance": {"consensus_raw_record_sha256": "8" * 64},
    }


def _denominator():
    return {
        "ticker": "AAA",
        "permaticker": 101,
        "identity_id": IDENTITY_ID,
        "session_date": "2020-04-30",
        "session_close_at_utc": "2020-04-30T20:00:00Z",
        "session_close_kind": "regular_core_close",
        "close_split_normalized": "12",
        "closeunadj_execution_evidence": "24",
        "split_normalization_factor": {
            "formula": "closeunadj / close",
            "numerator": 2,
            "denominator": 1,
            "decimal_34": "2",
        },
        "sep_source_row_sha256": "9" * 64,
        "sep_acquisition_sha256": "a" * 64,
        "sep_raw_zip_sha256": "b" * 64,
        "sep_parquet_sha256": "c" * 64,
    }


def _documents(candidate_hash: str, construction_hash: str, *, evidence_class=None):
    evidence_class = evidence_class or "historical_reconstruction"
    source = {
        "artifact_hash": SOURCE_HASH,
        "payload": {
            "candidate_id": CANDIDATE,
            "evidence_class": evidence_class,
            "reconciled_at_utc": "2026-07-14T12:00:00Z",
            "bindings": {
                "candidate_specification_sha256": candidate_hash,
                "construction_code_sha256": construction_hash,
                "event_universe_sha256": UNIVERSE_HASH,
                "known_by_policy_sha256": subject.KNOWN_BY_POLICY_SHA256,
                "reconciliation_policy_sha256": SOURCE_POLICY_HASH,
            },
            "event_results": [
                {
                    "event_id": EVENT_ID,
                    "event_key": EVENT_KEY,
                    "disposition": "event_source_reconciled",
                    "blockers": [],
                    "event_source_input": _source_input(),
                },
                {
                    "event_id": EXCLUDED_ID,
                    "event_key": EXCLUDED_KEY,
                    "disposition": "excluded",
                    "blockers": ["consensus_event_missing"],
                    "event_source_input": None,
                },
            ],
        },
    }
    market = {
        "artifact_hash": MARKET_HASH,
        "payload": {
            "candidate_id": CANDIDATE,
            "evidence_class": evidence_class,
            "created_at_utc": "2026-07-14T13:00:00Z",
            "trust_policy": {
                "candidate_specification_set_sha256": _trust_hash([candidate_hash]),
                "construction_code_set_sha256": _trust_hash([construction_hash]),
                "source_reconciliation_set_sha256": _trust_hash([SOURCE_HASH]),
            },
            "bindings": {
                "source_reconciliation_sha256": SOURCE_HASH,
                "source_reconciliation_event_universe_sha256": UNIVERSE_HASH,
                "market_accounting_policy_sha256": MARKET_POLICY_HASH,
            },
            "event_results": [
                {
                    "event_id": EVENT_ID,
                    "event_key": EVENT_KEY,
                    "upstream_disposition": "event_source_reconciled",
                    "disposition": "market_accounting_evidenced",
                    "blockers": [],
                    "lineage": {
                        "event_id": EVENT_ID,
                        "event_key": EVENT_KEY,
                        "ticker": "AAA",
                        "permaticker": 101,
                        "identity_id": IDENTITY_ID,
                        "representative_sf1_source_record_sha256": "d" * 64,
                        "sf1_source_record_sha256s": ["d" * 64],
                        "sf1_revision_count": 1,
                    },
                    "market_denominator": _denominator(),
                    "timing": {
                        "known_public_by_at_utc": "2020-05-01T20:00:00Z",
                        "activation_eastern_date": "2020-05-01",
                        "consensus_receipt_captured_at_utc": "2020-04-30T19:30:00Z",
                        "prospective_freeze_required": evidence_class == "prospective_signal",
                        "prospective_freeze_passed": (
                            True if evidence_class == "prospective_signal" else None
                        ),
                    },
                },
                {
                    "event_id": EXCLUDED_ID,
                    "event_key": EXCLUDED_KEY,
                    "upstream_disposition": "excluded",
                    "disposition": "upstream_excluded",
                    "blockers": ["upstream_not_event_source_reconciled"],
                    "lineage": None,
                    "market_denominator": None,
                    "timing": None,
                },
            ],
        },
    }
    return source, market


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.json"
    construction = tmp_path / "construction.py"
    signal_code = Path(subject.__file__)
    candidate.write_text(
        json.dumps(
            {
                "schema_version": "pead_candidate_specification.v3",
                "candidate_id": CANDIDATE,
                "signal_rule": {"formula": FORMULA, "minimum_analyst_count": 2},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    construction.write_bytes(b"# exact upstream construction code\n")
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    construction_hash = hashlib.sha256(construction.read_bytes()).hexdigest()
    signal_hash = hashlib.sha256(signal_code.read_bytes()).hexdigest()
    source, market = _documents(candidate_hash, construction_hash)
    calls = []

    def verify_source(document, **kwargs):
        calls.append(("source", document, kwargs))
        return document

    def verify_market(document, source_document, **kwargs):
        calls.append(("market", document, source_document, kwargs))
        return document

    monkeypatch.setattr(subject, "verify_pead_source_reconciliation_v2", verify_source)
    monkeypatch.setattr(subject, "verify_pead_market_accounting_evidence", verify_market)
    kwargs = {
        "candidate_specification_path": candidate,
        "construction_code_path": construction,
        "signal_reconciliation_code_path": signal_code,
        "created_at_utc": "2026-07-14T14:00:00Z",
        "market_accounting_verification_kwargs": {
            "source_reconciliation_verification_kwargs": {},
            "warehouse_dir": tmp_path,
        },
        "trusted_candidate_specification_sha256s": {candidate_hash},
        "trusted_construction_code_sha256s": {construction_hash},
        "trusted_signal_reconciliation_code_sha256s": {signal_hash},
        "trusted_source_reconciliation_sha256s": {SOURCE_HASH},
        "trusted_market_accounting_evidence_sha256s": {MARKET_HASH},
    }
    return {
        "source": source,
        "market": market,
        "kwargs": kwargs,
        "calls": calls,
        "hashes": {
            "candidate": candidate_hash,
            "construction": construction_hash,
            "signal": signal_hash,
        },
    }


def _build(inputs, **overrides):
    kwargs = {**inputs["kwargs"], **overrides}
    return build_pead_signal_input_reconciliation(inputs["source"], inputs["market"], **kwargs)


def test_final_boundary_replays_both_lanes_and_emits_exact_signal(inputs):
    document = _build(inputs)

    accepted = document["payload"]["event_results"][0]
    assert accepted["disposition"] == "signal_input_accepted"
    assert accepted["identity"] == {
        "ticker": "AAA",
        "permaticker": 101,
        "identity_id": IDENTITY_ID,
    }
    assert accepted["signal"] == {
        "formula": FORMULA,
        "actual_value": "1.25",
        "consensus_value": "1",
        "raw_surprise": "0.25",
        "strictly_prior_split_normalized_close": "12",
        "exact_ratio": {"numerator": 1, "denominator": 48},
        "value_decimal_34": "0.02083333333333333333333333333333333",
        "direction": "positive",
    }
    assert document["payload"]["event_results"][1]["disposition"] == ("signal_input_excluded")
    assert document["payload"]["coverage"] == {
        "expected_event_count": 2,
        "source_reconciled_event_count": 1,
        "market_accounting_evidenced_count": 1,
        "signal_input_accepted_count": 1,
        "signal_input_excluded_count": 1,
        "exhaustive_event_accounting": True,
        "partial_coverage": True,
        "blocker_counts": {
            "market__upstream_not_event_source_reconciled": 1,
            "reconciliation__source_not_reconciled": 1,
            "source__consensus_event_missing": 1,
        },
    }
    assert document["payload"]["qualification"] == {
        "has_research_consumable_signal_inputs": True,
        "all_expected_events_signal_accepted": False,
        "signal_input_reconciliation_allowed": True,
        "research_consumable": True,
        "historical_replication_allowed": True,
        "prospective_accumulation_allowed": False,
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    assert [call[0] for call in inputs["calls"]] == ["source", "market"]


def test_every_external_trust_registry_is_required(inputs):
    fields = [key for key in inputs["kwargs"] if key.startswith("trusted_")]

    for field in fields:
        with pytest.raises(PeadSignalInputReconciliationError, match="external trust registry"):
            _build(inputs, **{field: set()})


def test_final_trust_set_encoding_matches_market_accounting_contract():
    roots = ["0" * 64, "f" * 64]

    assert _trust_hash(roots) == market_subject._trust_set_hash(roots)


def test_exact_specification_and_both_code_identities_are_bound(inputs):
    document = _build(inputs)
    bindings = document["payload"]["bindings"]

    assert bindings["candidate_specification_sha256"] == inputs["hashes"]["candidate"]
    assert bindings["construction_code_sha256"] == inputs["hashes"]["construction"]
    assert bindings["signal_reconciliation_code_sha256"] == inputs["hashes"]["signal"]

    with pytest.raises(PeadSignalInputReconciliationError, match="executing implementation"):
        _build(
            inputs,
            signal_reconciliation_code_path=inputs["kwargs"]["construction_code_path"],
            trusted_signal_reconciliation_code_sha256s={inputs["hashes"]["construction"]},
        )


def test_market_must_bind_exact_source_and_event_universe(inputs):
    inputs["market"]["payload"]["bindings"]["source_reconciliation_sha256"] = "f" * 64

    with pytest.raises(PeadSignalInputReconciliationError, match="another source"):
        _build(inputs)


def test_market_trust_policy_must_equal_final_external_roots(inputs):
    inputs["market"]["payload"]["trust_policy"]["source_reconciliation_set_sha256"] = "f" * 64

    with pytest.raises(PeadSignalInputReconciliationError, match="final external trust"):
        _build(inputs)


def test_cross_lane_share_currency_and_unit_mismatches_exclude_without_imputation(inputs):
    source_input = inputs["source"]["payload"]["event_results"][0]["event_source_input"]
    source_input["metric"].update(
        {
            "canonical_share_basis": "issuer_as_reported",
            "currency_code": "CAD",
            "unit": "currency",
        }
    )

    result = _build(inputs)["payload"]["event_results"][0]

    assert result["disposition"] == "signal_input_excluded"
    assert result["signal"] is None
    assert result["market_denominator"] is not None
    assert result["reconciliation_blockers"] == [
        "canonical_share_basis_not_split_restated",
        "currency_not_usd",
        "unit_not_currency_per_share",
    ]


def test_missing_authoritative_market_lineage_is_an_exhaustive_exclusion(inputs):
    market_row = inputs["market"]["payload"]["event_results"][0]
    market_row.update(
        {
            "disposition": "market_accounting_excluded",
            "blockers": ["authoritative_event_lineage_missing"],
            "lineage": None,
            "market_denominator": None,
        }
    )

    document = _build(inputs)
    result = document["payload"]["event_results"][0]

    assert result["disposition"] == "signal_input_excluded"
    assert result["identity"] is None
    assert result["reconciliation_blockers"] == ["market_accounting_not_evidenced"]
    assert document["payload"]["coverage"]["expected_event_count"] == 2
    assert document["payload"]["coverage"]["signal_input_excluded_count"] == 2
    assert document["payload"]["qualification"]["research_consumable"] is False


def test_actual_minus_consensus_is_exact_beyond_default_decimal_context(inputs):
    source_input = inputs["source"]["payload"]["event_results"][0]["event_source_input"]
    source_input.update(
        {
            "actual_value": "10000000000000000000000000000.1",
            "consensus_value": "10000000000000000000000000000",
            "raw_surprise": "0.1",
        }
    )
    inputs["market"]["payload"]["event_results"][0]["market_denominator"][
        "close_split_normalized"
    ] = "2"

    signal = _build(inputs)["payload"]["event_results"][0]["signal"]

    assert signal["exact_ratio"] == {"numerator": 1, "denominator": 20}
    assert signal["value_decimal_34"] == "0.05"


def test_below_frozen_analyst_floor_is_exhaustively_excluded(inputs):
    inputs["source"]["payload"]["event_results"][0]["event_source_input"]["analyst_count"] = 1

    result = _build(inputs)["payload"]["event_results"][0]

    assert result["disposition"] == "signal_input_excluded"
    assert result["reconciliation_blockers"] == ["analyst_count_below_minimum"]
    assert result["signal"] is None


def test_candidate_specification_must_freeze_the_same_analyst_floor(inputs):
    path = inputs["kwargs"]["candidate_specification_path"]
    candidate = json.loads(path.read_text(encoding="utf-8"))
    candidate["signal_rule"]["minimum_analyst_count"] = 3
    path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    changed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs["source"]["payload"]["bindings"]["candidate_specification_sha256"] = changed_hash
    inputs["market"]["payload"]["trust_policy"]["candidate_specification_set_sha256"] = _trust_hash(
        [changed_hash]
    )

    with pytest.raises(PeadSignalInputReconciliationError, match="analyst-count floor"):
        _build(inputs, trusted_candidate_specification_sha256s={changed_hash})


def test_recomputed_wrapper_hash_cannot_hide_changed_signal_math(inputs):
    document = _build(inputs)
    forged = deepcopy(document)
    forged["payload"]["event_results"][0]["signal"]["value_decimal_34"] = "0.5"
    forged["artifact_hash"] = content_hash(forged["payload"])

    with pytest.raises(PeadSignalInputReconciliationError, match="signal math"):
        validate_pead_signal_input_reconciliation_structure(forged)


def test_source_and_market_event_order_must_match_exactly(inputs):
    inputs["market"]["payload"]["event_results"].reverse()

    with pytest.raises(PeadSignalInputReconciliationError, match="ordered event set"):
        _build(inputs)


def test_receipt_cannot_predate_upstream_evidence(inputs):
    with pytest.raises(PeadSignalInputReconciliationError, match="predates"):
        _build(inputs, created_at_utc="2026-07-14T12:30:00Z")


def test_prospective_receipt_allows_accumulation_but_never_execution(inputs):
    source, market = _documents(
        inputs["hashes"]["candidate"],
        inputs["hashes"]["construction"],
        evidence_class="prospective_signal",
    )
    inputs["source"] = source
    inputs["market"] = market

    qualification = _build(inputs)["payload"]["qualification"]

    assert qualification["prospective_accumulation_allowed"] is True
    assert qualification["historical_replication_allowed"] is False
    assert qualification["edge_claim_allowed"] is False
    assert qualification["paper_execution_allowed"] is False
    assert qualification["live_deployment_allowed"] is False


def test_authoritative_verifier_rebuilds_and_canonical_loader_rejects_duplicates(inputs, tmp_path):
    document = _build(inputs)
    assert (
        verify_pead_signal_input_reconciliation(
            document,
            inputs["source"],
            inputs["market"],
            **{key: value for key, value in inputs["kwargs"].items() if key != "created_at_utc"},
        )
        == document
    )

    path = tmp_path / "receipt.json"
    path.write_bytes((canonical_json(document) + "\n").encode("utf-8"))
    assert (
        load_pead_signal_input_reconciliation(
            path,
            source_reconciliation=inputs["source"],
            market_accounting_evidence=inputs["market"],
            **{key: value for key, value in inputs["kwargs"].items() if key != "created_at_utc"},
        )
        == document
    )

    path.write_text('{"artifact_hash":"x","artifact_hash":"y"}\n', encoding="utf-8")
    with pytest.raises(PeadSignalInputReconciliationError, match="duplicate key"):
        load_pead_signal_input_reconciliation(path)


def test_create_only_publisher_writes_canonical_bytes_and_refuses_every_collision(inputs, tmp_path):
    document = _build(inputs)
    path = tmp_path / "published" / "signal-input.json"

    normalized, published_path = publish_pead_signal_input_reconciliation(
        document, path, allow_structural_only=True
    )

    assert normalized == document
    assert published_path == path
    assert path.read_bytes() == (canonical_json(document) + "\n").encode("utf-8")
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(PeadSignalInputReconciliationError, match="already exists"):
        publish_pead_signal_input_reconciliation(document, path, allow_structural_only=True)

    path.write_bytes(b"different existing bytes\n")
    with pytest.raises(PeadSignalInputReconciliationError, match="already exists"):
        publish_pead_signal_input_reconciliation(document, path, allow_structural_only=True)


def test_publisher_validates_before_create_and_supports_authoritative_mode(inputs, tmp_path):
    document = _build(inputs)
    forged = deepcopy(document)
    forged["payload"]["event_results"][0]["signal"]["value_decimal_34"] = "0.5"
    forged["artifact_hash"] = content_hash(forged["payload"])
    rejected_path = tmp_path / "rejected.json"

    with pytest.raises(PeadSignalInputReconciliationError, match="requires authoritative"):
        publish_pead_signal_input_reconciliation(document, rejected_path)
    assert not rejected_path.exists()

    with pytest.raises(PeadSignalInputReconciliationError, match="signal math"):
        publish_pead_signal_input_reconciliation(forged, rejected_path, allow_structural_only=True)
    assert not rejected_path.exists()

    authoritative_path = tmp_path / "authoritative.json"
    verification_kwargs = {
        "source_reconciliation": inputs["source"],
        "market_accounting_evidence": inputs["market"],
        **{key: value for key, value in inputs["kwargs"].items() if key != "created_at_utc"},
    }
    normalized, _ = publish_pead_signal_input_reconciliation(
        document,
        authoritative_path,
        authoritative_verification_kwargs=verification_kwargs,
    )

    assert normalized == document
    assert authoritative_path.is_file()
