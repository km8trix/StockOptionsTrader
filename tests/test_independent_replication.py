from __future__ import annotations

import hashlib
import json

import pytest

from analysis.independent_replication import (
    ImplementationIdentity,
    IndependentReplicationContract,
    NUMERIC_FIELDS,
    NumericTolerance,
    ReplicationEvidence,
    ReplicationEvidenceStore,
    ReplicationIntegrityError,
    canonical_replication_json,
    reconcile_implementations,
)


PROTOCOL_HASH = "1" * 64
DATA_HASH = "2" * 64
PRIMARY_CODE_HASH = "3" * 64
REPLICATION_CODE_HASH = "4" * 64


def _tolerances(absolute: float = 1e-8, relative: float = 1e-6):
    return {
        field_name: NumericTolerance(absolute=absolute, relative=relative)
        for field_name in NUMERIC_FIELDS
    }


def _key(symbol: str):
    return {"as_of": "2026-01-02", "symbol": symbol}


def _contract(*, tolerances=None, expected_symbols=("AAPL", "MSFT")):
    return IndependentReplicationContract(
        protocol_hash=PROTOCOL_HASH,
        data_snapshot_hash=DATA_HASH,
        primary=ImplementationIdentity("vectorized-v1", PRIMARY_CODE_HASH),
        replication=ImplementationIdentity("event-driven-v1", REPLICATION_CODE_HASH),
        expected_observation_keys=[_key(symbol) for symbol in expected_symbols],
        tolerances=tolerances or _tolerances(),
    )


def _observation(symbol: str, **overrides):
    value = {
        "key": _key(symbol),
        "signal": 0.25,
        "eligibility": True,
        "rank": 0.8,
        "target": 100.0,
        "order": 25.0,
        "position": 75.0,
        "cash": 92_500.0,
        "fees": 2.5,
        "pnl": 4.5,
    }
    value.update(overrides)
    return value


def test_contract_binds_hashes_and_requires_independent_code_and_explicit_tolerances():
    contract = _contract()
    payload = contract.to_payload()
    assert payload["protocol_hash"] == PROTOCOL_HASH
    assert payload["data_snapshot_hash"] == DATA_HASH
    assert payload["expected_observation_keys"] == [_key("AAPL"), _key("MSFT")]
    assert len(contract.contract_hash) == 64
    assert contract.contract_hash != _contract(
        expected_symbols=("AAPL",)).contract_hash

    with pytest.raises(ValueError, match="distinct IDs"):
        IndependentReplicationContract(
            protocol_hash=PROTOCOL_HASH,
            data_snapshot_hash=DATA_HASH,
            primary=ImplementationIdentity("same", PRIMARY_CODE_HASH),
            replication=ImplementationIdentity("same", REPLICATION_CODE_HASH),
            expected_observation_keys=[_key("AAPL")],
            tolerances=_tolerances(),
        )
    with pytest.raises(ValueError, match="distinct code hashes"):
        IndependentReplicationContract(
            protocol_hash=PROTOCOL_HASH,
            data_snapshot_hash=DATA_HASH,
            primary=ImplementationIdentity("one", PRIMARY_CODE_HASH),
            replication=ImplementationIdentity("two", PRIMARY_CODE_HASH),
            expected_observation_keys=[_key("AAPL")],
            tolerances=_tolerances(),
        )
    with pytest.raises(ValueError, match="nonempty"):
        _contract(expected_symbols=())
    with pytest.raises(ValueError, match="unique"):
        _contract(expected_symbols=("AAPL", "AAPL"))
    incomplete = _tolerances()
    incomplete.pop("pnl")
    with pytest.raises(ValueError, match="explicitly cover"):
        _contract(tolerances=incomplete)
    with pytest.raises(ValueError, match="finite and non-negative"):
        NumericTolerance(absolute=float("nan"), relative=0.0)


def test_full_reconciliation_passes_within_tolerance_and_is_order_independent(tmp_path):
    contract = _contract()
    primary = [_observation("MSFT"), _observation("AAPL")]
    replication = [
        _observation("AAPL", signal=0.2500001, pnl=4.500004),
        _observation("MSFT", target=100.00005),
    ]

    evidence = reconcile_implementations(
        contract,
        primary_observations=primary,
        replication_observations=replication,
    )
    reversed_evidence = reconcile_implementations(
        contract,
        primary_observations=list(reversed(primary)),
        replication_observations=list(reversed(replication)),
    )

    assert evidence.passed is True
    assert evidence.discrepancies == []
    assert evidence.evidence_hash == reversed_evidence.evidence_hash
    assert evidence.payload["contract_hash"] == contract.contract_hash
    assert evidence.payload["protocol_hash"] == PROTOCOL_HASH
    assert evidence.payload["data_snapshot_hash"] == DATA_HASH

    store = ReplicationEvidenceStore(tmp_path)
    path = store.persist(evidence)
    assert path == store.path_for(evidence.evidence_hash)
    assert store.load(evidence.evidence_hash) == evidence


def test_every_field_mismatch_and_both_missing_sides_are_recorded():
    zero_tolerances = _tolerances(absolute=0.0, relative=0.0)
    contract = _contract(
        tolerances=zero_tolerances,
        expected_symbols=("AAPL", "PRIMARY_ONLY", "REPLICATION_ONLY"),
    )
    primary_common = _observation("AAPL")
    replication_common = _observation(
        "AAPL",
        signal=1.25,
        eligibility=False,
        rank=1.8,
        target=101.0,
        order=26.0,
        position=76.0,
        cash=92_501.0,
        fees=3.5,
        pnl=5.5,
    )
    evidence = ReplicationEvidence.create(
        contract,
        primary_observations=[primary_common, _observation("PRIMARY_ONLY")],
        replication_observations=[replication_common, _observation("REPLICATION_ONLY")],
    )

    assert evidence.passed is False
    kinds = [item["kind"] for item in evidence.discrepancies]
    assert kinds.count("missing_observation") == 2
    assert kinds.count("numeric_mismatch") == len(NUMERIC_FIELDS)
    assert kinds.count("exact_mismatch") == 1
    mismatched_fields = {
        item["field"]
        for item in evidence.discrepancies
        if item["kind"] in {"numeric_mismatch", "exact_mismatch"}
    }
    assert mismatched_fields == {*NUMERIC_FIELDS, "eligibility"}
    numeric = next(
        item
        for item in evidence.discrepancies
        if item["kind"] == "numeric_mismatch" and item["field"] == "signal"
    )
    assert numeric["absolute_error"] == 1.0
    assert numeric["allowed_error"] == 0.0
    assert numeric["tolerance"] == {"absolute": 0.0, "relative": 0.0}


def test_duplicate_missing_and_nonfinite_observations_fail_closed_with_evidence():
    contract = _contract(expected_symbols=("BAD", "DUP", "PRIMARY_ONLY"))
    duplicate = _observation("DUP")
    primary = [
        duplicate,
        dict(duplicate),
        _observation("BAD", pnl=float("nan")),
        _observation("PRIMARY_ONLY"),
    ]
    replication = [_observation("DUP"), _observation("BAD")]

    evidence = reconcile_implementations(
        contract,
        primary_observations=primary,
        replication_observations=replication,
    )

    assert evidence.passed is False
    assert {item["kind"] for item in evidence.discrepancies} == {
        "duplicate_observation",
        "nonfinite_value",
        "missing_observation",
        "missing_expected_observation",
    }
    duplicate_record = next(
        item for item in evidence.discrepancies if item["kind"] == "duplicate_observation"
    )
    assert duplicate_record["count"] == 2
    nonfinite_record = next(
        item for item in evidence.discrepancies if item["kind"] == "nonfinite_value"
    )
    assert nonfinite_record["field"] == "pnl"
    assert nonfinite_record["value"] == {"__replication_nonfinite__": "NaN"}
    # Invalid numeric evidence is encoded canonically rather than leaking JSON NaN.
    assert "NaN" in evidence.to_json()
    assert ":NaN" not in evidence.to_json()
    assert ReplicationEvidence.from_json(evidence.to_json()) == evidence


def test_artifact_verification_recomputes_discrepancies_even_if_attacker_rehashes():
    contract = _contract(
        tolerances=_tolerances(absolute=0.0, relative=0.0),
        expected_symbols=("AAPL",),
    )
    evidence = reconcile_implementations(
        contract,
        primary_observations=[_observation("AAPL")],
        replication_observations=[_observation("AAPL", signal=99.0)],
    )
    document = json.loads(evidence.to_json())
    document["evidence"]["discrepancies"] = []
    document["evidence"]["passed"] = True
    forged_payload = canonical_replication_json(document["evidence"])
    document["evidence_hash"] = hashlib.sha256(forged_payload.encode()).hexdigest()

    with pytest.raises(ReplicationIntegrityError, match="complete reconciliation"):
        ReplicationEvidence.from_json(json.dumps(document))


def test_observation_schema_errors_are_failures_not_partial_successes():
    contract = _contract()
    incomplete = _observation("AAPL")
    incomplete.pop("order")
    wrong_type = _observation("MSFT", eligibility=1, signal="0.25")
    evidence = reconcile_implementations(
        contract,
        primary_observations=[incomplete, wrong_type],
        replication_observations=[_observation("AAPL"), _observation("MSFT")],
    )

    assert evidence.passed is False
    kinds = [item["kind"] for item in evidence.discrepancies]
    assert "invalid_observation_fields" in kinds
    assert kinds.count("invalid_value_type") == 2


def test_both_implementations_must_cover_exact_predeclared_key_manifest():
    contract = _contract()
    incomplete = reconcile_implementations(
        contract,
        primary_observations=[_observation("AAPL")],
        replication_observations=[_observation("AAPL")],
    )

    assert incomplete.passed is False
    missing = [
        item for item in incomplete.discrepancies
        if item["kind"] == "missing_expected_observation"
    ]
    assert len(missing) == 2
    assert {item["implementation_id"] for item in missing} == {
        "vectorized-v1", "event-driven-v1",
    }
    assert all(item["key"] == _key("MSFT") for item in missing)

    empty = reconcile_implementations(
        contract,
        primary_observations=[],
        replication_observations=[],
    )
    assert empty.passed is False
    assert sum(
        item["kind"] == "missing_expected_observation"
        for item in empty.discrepancies
    ) == 4

    unexpected = reconcile_implementations(
        contract,
        primary_observations=[
            _observation("AAPL"), _observation("MSFT"), _observation("TSLA")],
        replication_observations=[
            _observation("AAPL"), _observation("MSFT"), _observation("TSLA")],
    )
    assert unexpected.passed is False
    assert sum(
        item["kind"] == "unexpected_observation"
        for item in unexpected.discrepancies
    ) == 2


def test_stored_artifact_recomputes_manifest_coverage_after_both_sides_omit_key():
    contract = _contract()
    evidence = reconcile_implementations(
        contract,
        primary_observations=[_observation("AAPL"), _observation("MSFT")],
        replication_observations=[_observation("AAPL"), _observation("MSFT")],
    )
    document = json.loads(evidence.to_json())
    for output in document["evidence"]["outputs"]:
        output["observations"] = [
            observation for observation in output["observations"]
            if observation["key"]["symbol"] != "MSFT"
        ]
    document["evidence"]["discrepancies"] = []
    document["evidence"]["passed"] = True
    forged_payload = canonical_replication_json(document["evidence"])
    document["evidence_hash"] = hashlib.sha256(forged_payload.encode()).hexdigest()

    with pytest.raises(ReplicationIntegrityError, match="complete reconciliation"):
        ReplicationEvidence.from_json(json.dumps(document))
