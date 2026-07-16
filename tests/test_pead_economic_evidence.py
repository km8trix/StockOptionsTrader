from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from data.pead_economic_evidence import (
    ACTIONS_DIVIDEND_DATE_ROLE,
    ACTIONS_DIVIDEND_VALUE_ROLE,
    CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION,
    TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION,
    UNPROVEN_SEMANTICS_EVIDENCE_STATUS,
    PeadEconomicEvidenceError,
    build_current_unproven_cash_distribution_semantics,
    build_empty_terminal_settlement_ledger,
    canonical_json,
    content_hash,
    load_cash_distribution_semantics,
    load_pead_economic_evidence,
    load_terminal_settlement_ledger,
    validate_cash_distribution_semantics,
    validate_terminal_settlement_ledger,
)


def _write_document(path: Path, document: dict) -> None:
    path.write_text(canonical_json(document) + "\n", encoding="utf-8")


def _wrap(payload: dict) -> dict:
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _terminal_document(source_bytes: bytes = b"authoritative cash settlement") -> dict:
    payload = {
        "schema_version": TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION,
        "ACTIONS": {"value_allowed": False},
        "cash_only_records": [
            {
                "ticker": "OLD",
                "permaticker": 101,
                "last_price_date": "2024-03-14",
                "settlement_date": "2024-03-18",
                "cash_per_terminal_share": 12.75,
                "source_receipts": [
                    {
                        "url": "https://www.sec.gov/Archives/example-settlement.html",
                        "retrieved_at_utc": "2026-07-13T20:15:00Z",
                        "local_path": "sources/example-settlement.html",
                        "sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "bytes": len(source_bytes),
                    }
                ],
            }
        ],
    }
    return _wrap(payload)


def test_current_semantics_builder_is_exact_nonqualifying_and_content_addressed(
    tmp_path,
):
    document = build_current_unproven_cash_distribution_semantics(
        absolute_tolerance=1e-9,
        relative_tolerance=2e-7,
    )

    assert document["artifact_hash"] == content_hash(document["payload"])
    assert document["payload"] == {
        "schema_version": CASH_DISTRIBUTION_SEMANTICS_SCHEMA_VERSION,
        "ACTIONS": {
            "action": "dividend",
            "date_role": ACTIONS_DIVIDEND_DATE_ROLE,
            "value_role": ACTIONS_DIVIDEND_VALUE_ROLE,
        },
        "evidence_status": UNPROVEN_SEMANTICS_EVIDENCE_STATUS,
        "qualification_allowed": False,
        "adjustment_check_tolerance": {
            "absolute": 1e-9,
            "relative": 2e-7,
        },
    }

    path = tmp_path / "cash-semantics.json"
    _write_document(path, document)
    assert load_cash_distribution_semantics(path) == document
    assert load_pead_economic_evidence(path) == document


def test_semantics_rejects_hash_tampering_extensions_and_rewritten_claims():
    document = build_current_unproven_cash_distribution_semantics()
    tampered = copy.deepcopy(document)
    tampered["payload"]["ACTIONS"]["date_role"] = "ex_date_proven"
    with pytest.raises(PeadEconomicEvidenceError, match="artifact hash mismatch"):
        validate_cash_distribution_semantics(tampered)

    self_consistent = copy.deepcopy(tampered)
    self_consistent["artifact_hash"] = content_hash(self_consistent["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="unproven contract"):
        validate_cash_distribution_semantics(self_consistent)

    qualifying = copy.deepcopy(document)
    qualifying["payload"]["qualification_allowed"] = True
    qualifying["artifact_hash"] = content_hash(qualifying["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="cannot allow qualification"):
        validate_cash_distribution_semantics(qualifying)

    extended = copy.deepcopy(document)
    extended["payload"]["notes"] = "looks plausible"
    extended["artifact_hash"] = content_hash(extended["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="fields differ"):
        validate_cash_distribution_semantics(extended)


@pytest.mark.parametrize("absolute, relative", [(-1.0, 0.0), (0.0, -1.0)])
def test_semantics_requires_nonnegative_finite_tolerances(absolute, relative):
    with pytest.raises(PeadEconomicEvidenceError, match="finite non-negative"):
        build_current_unproven_cash_distribution_semantics(
            absolute_tolerance=absolute,
            relative_tolerance=relative,
        )

    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(PeadEconomicEvidenceError, match="non-finite"):
            build_current_unproven_cash_distribution_semantics(
                absolute_tolerance=invalid
            )


def test_strict_loader_rejects_duplicate_keys_and_json_constants(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"artifact_hash":"' + "a" * 64 + '","artifact_hash":"' + "b" * 64
        + '","payload":{}}',
        encoding="utf-8",
    )
    with pytest.raises(PeadEconomicEvidenceError, match="duplicate key"):
        load_cash_distribution_semantics(duplicate)

    invalid_number = tmp_path / "invalid-number.json"
    invalid_number.write_text(
        '{"artifact_hash":"' + "a" * 64
        + '","payload":{"schema_version":"pead_cash_distribution_semantics.v1",'
        '"value":NaN}}',
        encoding="utf-8",
    )
    with pytest.raises(PeadEconomicEvidenceError, match="invalid number NaN"):
        load_cash_distribution_semantics(invalid_number)


def test_empty_terminal_ledger_is_an_exact_content_addressed_nonclaim(tmp_path):
    document = build_empty_terminal_settlement_ledger()

    assert document == _wrap(
        {
            "schema_version": TERMINAL_SETTLEMENT_LEDGER_SCHEMA_VERSION,
            "ACTIONS": {"value_allowed": False},
            "cash_only_records": [],
        }
    )
    path = tmp_path / "terminal-ledger.json"
    _write_document(path, document)
    assert load_terminal_settlement_ledger(path) == document
    assert load_pead_economic_evidence(path) == document


def test_terminal_record_requires_and_revalidates_archived_source_bytes(tmp_path):
    source_bytes = b"authoritative cash settlement"
    source_path = tmp_path / "sources" / "example-settlement.html"
    source_path.parent.mkdir()
    source_path.write_bytes(source_bytes)
    document = _terminal_document(source_bytes)
    ledger_path = tmp_path / "terminal-ledger.json"
    _write_document(ledger_path, document)

    assert load_terminal_settlement_ledger(ledger_path) == document

    source_path.write_bytes(b"tampered cash settlement")
    with pytest.raises(PeadEconomicEvidenceError, match="byte count mismatch"):
        load_terminal_settlement_ledger(ledger_path)

    source_path.write_bytes(b"forged bytes same byte count!!!!")
    receipt = document["payload"]["cash_only_records"][0]["source_receipts"][0]
    if source_path.stat().st_size != receipt["bytes"]:
        source_path.write_bytes(b"x" * receipt["bytes"])
    with pytest.raises(PeadEconomicEvidenceError, match="SHA-256 mismatch"):
        load_terminal_settlement_ledger(ledger_path)


def test_terminal_ledger_rejects_actions_value_and_malformed_cash_records():
    document = _terminal_document()

    actions_value = copy.deepcopy(document)
    actions_value["payload"]["ACTIONS"]["value_allowed"] = True
    actions_value["artifact_hash"] = content_hash(actions_value["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="ACTIONS.value"):
        validate_terminal_settlement_ledger(actions_value)

    invalid_cases = [
        ("permaticker", 0, "positive integer"),
        ("cash_per_terminal_share", 0.0, "must be positive"),
        ("source_receipts", [], "one or more source receipts"),
        ("settlement_date", "2024-03-13", "precedes last_price_date"),
    ]
    for field, value, message in invalid_cases:
        invalid = copy.deepcopy(document)
        invalid["payload"]["cash_only_records"][0][field] = value
        invalid["artifact_hash"] = content_hash(invalid["payload"])
        with pytest.raises(PeadEconomicEvidenceError, match=message):
            validate_terminal_settlement_ledger(invalid)

    nonfinite = copy.deepcopy(document)
    nonfinite["payload"]["cash_only_records"][0]["cash_per_terminal_share"] = (
        math.inf
    )
    with pytest.raises(PeadEconomicEvidenceError, match="non-finite"):
        validate_terminal_settlement_ledger(nonfinite)


def test_terminal_ledger_rejects_inexact_receipts_duplicates_and_path_escape(
    tmp_path,
):
    document = _terminal_document()
    record = document["payload"]["cash_only_records"][0]

    invalid_url = copy.deepcopy(document)
    invalid_url["payload"]["cash_only_records"][0]["source_receipts"][0][
        "url"
    ] = "http://example.com/settlement"
    invalid_url["artifact_hash"] = content_hash(invalid_url["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="HTTPS URL"):
        validate_terminal_settlement_ledger(invalid_url)

    absolute_path = copy.deepcopy(document)
    absolute_path["payload"]["cash_only_records"][0]["source_receipts"][0][
        "local_path"
    ] = "/tmp/source.html"
    absolute_path["artifact_hash"] = content_hash(absolute_path["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="relative path"):
        validate_terminal_settlement_ledger(absolute_path)

    duplicate_receipt = copy.deepcopy(document)
    duplicate_receipt["payload"]["cash_only_records"][0][
        "source_receipts"
    ].append(copy.deepcopy(record["source_receipts"][0]))
    duplicate_receipt["artifact_hash"] = content_hash(duplicate_receipt["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="duplicate source receipt"):
        validate_terminal_settlement_ledger(duplicate_receipt)

    duplicate_record = copy.deepcopy(document)
    duplicate_record["payload"]["cash_only_records"].append(copy.deepcopy(record))
    duplicate_record["artifact_hash"] = content_hash(duplicate_record["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="duplicate record key"):
        validate_terminal_settlement_ledger(duplicate_record)

    outside = tmp_path.parent / "outside-terminal-source.html"
    outside.write_bytes(b"authoritative cash settlement")
    escaping = copy.deepcopy(document)
    escaping["payload"]["cash_only_records"][0]["source_receipts"][0][
        "local_path"
    ] = "../outside-terminal-source.html"
    escaping["artifact_hash"] = content_hash(escaping["payload"])
    with pytest.raises(PeadEconomicEvidenceError, match="relative path"):
        validate_terminal_settlement_ledger(escaping, source_root=tmp_path)


def test_terminal_ledger_requires_exact_wrapper_payload_record_and_receipt_fields():
    document = _terminal_document()
    mutation_paths = [
        (document, "wrapper_note"),
        (document["payload"], "payload_note"),
        (document["payload"]["ACTIONS"], "actions_note"),
        (document["payload"]["cash_only_records"][0], "record_note"),
        (
            document["payload"]["cash_only_records"][0]["source_receipts"][0],
            "receipt_note",
        ),
    ]

    for target, extra_field in mutation_paths:
        invalid = copy.deepcopy(document)
        if target is document:
            invalid[extra_field] = True
        elif target is document["payload"]:
            invalid["payload"][extra_field] = True
        elif target is document["payload"]["ACTIONS"]:
            invalid["payload"]["ACTIONS"][extra_field] = True
        elif target is document["payload"]["cash_only_records"][0]:
            invalid["payload"]["cash_only_records"][0][extra_field] = True
        else:
            invalid["payload"]["cash_only_records"][0]["source_receipts"][0][
                extra_field
            ] = True
        if target is not document:
            invalid["artifact_hash"] = content_hash(invalid["payload"])
        with pytest.raises(PeadEconomicEvidenceError, match="fields differ"):
            validate_terminal_settlement_ledger(invalid)


def test_generic_loader_rejects_unknown_schema(tmp_path):
    document = _wrap({"schema_version": "pead_unknown_evidence.v1"})
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PeadEconomicEvidenceError, match="unsupported"):
        load_pead_economic_evidence(path)
