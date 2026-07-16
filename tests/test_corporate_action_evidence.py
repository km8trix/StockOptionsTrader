from __future__ import annotations

import json
from pathlib import Path
import zipfile

import duckdb
import pytest

from data.corporate_action_evidence import (
    ACTIONS_ACQUISITION_SCHEMA_VERSION,
    ACTIONS_RECEIPT_ARCHIVE_DIR,
    ACTIONS_RECEIPT_FILE,
    LEGACY_ACTIONS_ACQUISITION_SCHEMA_VERSION,
    ActionsEvidenceError,
    archive_raw_zip,
    atomic_write_json,
    build_actions_acquisition_document,
    content_hash,
    expected_datatable_metadata,
    file_sha256,
    inspect_actions_parquet,
    upgrade_actions_acquisition_receipt,
    validate_actions_evidence,
)


def _write_actions(path: Path, *, duplicate: bool = False, apple_name: str = "Apple") -> None:
    escaped_apple_name = apple_name.replace("'", "''")
    rows = [
        f"(DATE '2010-01-04','dividend','AAPL','{escaped_apple_name}',0.10,"
        "CAST(NULL AS VARCHAR),CAST(NULL AS VARCHAR))",
        "(DATE '2026-07-10','split','WMT','Walmart',3.0,"
        "CAST(NULL AS VARCHAR),CAST(NULL AS VARCHAR))",
    ]
    if duplicate:
        rows.append(rows[-1])
    duckdb.connect().execute(
        "COPY (SELECT * FROM (VALUES " + ",".join(rows) + ") AS "
        "t(date,action,ticker,name,value,contraticker,contraname)) "
        f"TO '{path}' (FORMAT PARQUET)"
    )


def _write_zip(path: Path) -> str:
    member = "SHARADAR_ACTIONS_test.csv"
    csv = (
        "date,action,ticker,name,value,contraticker,contraname\n"
        "2010-01-04,dividend,AAPL,Apple,0.10,,\n"
        "2026-07-10,split,WMT,Walmart,3.0,,\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv)
    return member


def _metadata() -> dict:
    return {
        **expected_datatable_metadata(),
        "status": {
            "expected_at": "*",
            "refreshed_at": "2026-07-10T03:18:34.000Z",
            "status": "ON TIME",
            "update_frequency": "CONTINUOUS",
        },
    }


def _published_receipt(tmp_path: Path) -> dict:
    parquet = tmp_path / "actions.parquet"
    source_zip = tmp_path / "download.zip"
    _write_actions(parquet)
    _write_zip(source_zip)
    archived = archive_raw_zip(source_zip, tmp_path)
    document = build_actions_acquisition_document(
        parquet_path=parquet,
        raw_zip_path=archived,
        raw_zip_relative_path=archived.relative_to(tmp_path).as_posix(),
        acquired_at_utc="2026-07-13T22:00:00Z",
        last_refreshed_time="2026-07-10 03:18:34 UTC",
        data_snapshot_time="2026-07-10 03:28:21 UTC",
        datatable_metadata=_metadata(),
    )
    atomic_write_json(tmp_path / ACTIONS_RECEIPT_FILE, document)
    return document


def _legacy_receipt(tmp_path: Path) -> dict:
    document = _published_receipt(tmp_path)
    payload = document["payload"]
    payload["schema_version"] = LEGACY_ACTIONS_ACQUISITION_SCHEMA_VERSION
    payload.pop("row_equivalence")
    document["artifact_hash"] = content_hash(payload)
    atomic_write_json(tmp_path / ACTIONS_RECEIPT_FILE, document)
    return document


def test_actions_receipt_revalidates_raw_and_converted_snapshot(tmp_path):
    document = _published_receipt(tmp_path)

    evidence = validate_actions_evidence(
        tmp_path, required_start="2015-01-01", required_end="2024-09-30"
    )

    assert evidence["payload"]["complete"] is True
    assert evidence["payload"]["blockers"] == []
    assert evidence["payload"]["acquisition_artifact_hash"] == (document["artifact_hash"])
    assert evidence["payload"]["row_count"] == 2
    assert evidence["payload"]["value_is_terminal_payout_per_share"] is False
    assert evidence["payload"]["parquet_sha256"] == file_sha256(tmp_path / "actions.parquet")


def test_actions_evidence_detects_parquet_and_receipt_mutation(tmp_path):
    _published_receipt(tmp_path)
    (tmp_path / "actions.parquet").write_bytes(b"not parquet")
    with pytest.raises(ActionsEvidenceError, match="identity mismatch"):
        validate_actions_evidence(tmp_path, required_start="2015-01-01", required_end="2024-09-30")

    _published_receipt(tmp_path)
    receipt = json.loads((tmp_path / ACTIONS_RECEIPT_FILE).read_text())
    receipt["payload"]["value_semantics"] = "cash per share"
    (tmp_path / ACTIONS_RECEIPT_FILE).write_text(json.dumps(receipt))
    with pytest.raises(ActionsEvidenceError, match="artifact hash mismatch"):
        validate_actions_evidence(tmp_path, required_start="2015-01-01", required_end="2024-09-30")


def test_actions_evidence_rejects_same_statistics_with_one_altered_row(tmp_path):
    document = _published_receipt(tmp_path)
    original_statistics = document["payload"]["parquet"]["statistics"]

    # Preserve every existing schema/summary check while changing one row.
    parquet = tmp_path / "actions.parquet"
    _write_actions(parquet, apple_name="Applf")
    altered_statistics = inspect_actions_parquet(parquet)
    assert altered_statistics == original_statistics

    # Model a legacy internally self-consistent receipt that binds the altered
    # Parquet bytes but has no row-equivalence commitment. Validation must
    # still independently compare the complete CSV and Parquet row multisets.
    payload = document["payload"]
    payload["schema_version"] = "sharadar_actions_acquisition.v1"
    payload.pop("row_equivalence")
    payload["parquet"]["sha256"] = file_sha256(parquet)
    payload["parquet"]["bytes"] = parquet.stat().st_size
    payload["parquet"]["statistics"] = altered_statistics
    document["artifact_hash"] = content_hash(payload)
    atomic_write_json(tmp_path / ACTIONS_RECEIPT_FILE, document)

    with pytest.raises(
        ActionsEvidenceError,
        match="CSV and Parquet normalized row multisets differ",
    ):
        validate_actions_evidence(
            tmp_path,
            required_start="2015-01-01",
            required_end="2024-09-30",
            allow_legacy=True,
        )


def test_candidate_grade_requires_v2_but_legacy_compatibility_is_explicit(
    tmp_path,
):
    legacy = _legacy_receipt(tmp_path)

    with pytest.raises(ActionsEvidenceError, match="not candidate-grade"):
        validate_actions_evidence(
            tmp_path,
            required_start="2015-01-01",
            required_end="2024-09-30",
        )

    evidence = validate_actions_evidence(
        tmp_path,
        required_start="2015-01-01",
        required_end="2024-09-30",
        allow_legacy=True,
    )
    assert evidence["payload"]["acquisition_artifact_hash"] == legacy[
        "artifact_hash"
    ]


def test_offline_upgrade_preserves_v1_metadata_and_promotes_validated_v2(
    tmp_path,
):
    legacy = _legacy_receipt(tmp_path)
    legacy_payload = json.loads(json.dumps(legacy["payload"]))

    upgraded = upgrade_actions_acquisition_receipt(tmp_path)

    assert upgraded["payload"]["schema_version"] == (
        ACTIONS_ACQUISITION_SCHEMA_VERSION
    )
    assert upgraded["payload"]["row_equivalence"]["rows"] == 2
    assert upgraded["artifact_hash"] != legacy["artifact_hash"]
    for field, value in legacy_payload.items():
        if field != "schema_version":
            assert upgraded["payload"][field] == value
    active = json.loads(
        (tmp_path / ACTIONS_RECEIPT_FILE).read_text(encoding="utf-8")
    )
    assert active == upgraded
    archived = (
        tmp_path
        / ACTIONS_RECEIPT_ARCHIVE_DIR
        / f"{upgraded['artifact_hash']}.json"
    )
    assert json.loads(archived.read_text(encoding="utf-8")) == upgraded
    assert validate_actions_evidence(
        tmp_path,
        required_start="2015-01-01",
        required_end="2024-09-30",
    )["payload"]["acquisition_artifact_hash"] == upgraded["artifact_hash"]

    # Re-running is safe and does not produce a different active receipt.
    assert upgrade_actions_acquisition_receipt(tmp_path) == upgraded
    assert json.loads(
        (tmp_path / ACTIONS_RECEIPT_FILE).read_text(encoding="utf-8")
    ) == upgraded


def test_offline_upgrade_failure_never_promotes_unproven_receipt(tmp_path):
    legacy = _legacy_receipt(tmp_path)
    active_before = (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes()
    _write_actions(tmp_path / "actions.parquet", apple_name="Applf")

    with pytest.raises(
        ActionsEvidenceError,
        match="identity mismatch|normalized row multisets differ",
    ):
        upgrade_actions_acquisition_receipt(tmp_path)

    assert (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes() == active_before
    assert json.loads(active_before)["artifact_hash"] == legacy["artifact_hash"]
    archive_root = tmp_path / ACTIONS_RECEIPT_ARCHIVE_DIR
    assert not archive_root.exists() or not list(archive_root.iterdir())


def test_offline_upgrade_stages_create_only_before_atomic_promotion(
    tmp_path, monkeypatch,
):
    import data.corporate_action_evidence as actions_evidence

    _legacy_receipt(tmp_path)
    active_before = (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes()

    def interrupt_promotion(*_args, **_kwargs):
        raise RuntimeError("promotion interrupted")

    monkeypatch.setattr(actions_evidence, "atomic_write_json", interrupt_promotion)
    with pytest.raises(RuntimeError, match="promotion interrupted"):
        upgrade_actions_acquisition_receipt(tmp_path)

    assert (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes() == active_before
    archived = list((tmp_path / ACTIONS_RECEIPT_ARCHIVE_DIR).glob("*.json"))
    assert len(archived) == 1
    staged = json.loads(archived[0].read_text(encoding="utf-8"))
    assert staged["payload"]["schema_version"] == (
        ACTIONS_ACQUISITION_SCHEMA_VERSION
    )
    assert staged["payload"]["row_equivalence"]["rows"] == 2
    assert archived[0].stem == staged["artifact_hash"]


def test_actions_acquisition_rejects_same_statistics_with_one_altered_row(
    tmp_path,
):
    original = tmp_path / "original.parquet"
    altered = tmp_path / "altered.parquet"
    source_zip = tmp_path / "download.zip"
    _write_actions(original)
    _write_actions(altered, apple_name="Applf")
    _write_zip(source_zip)
    assert inspect_actions_parquet(altered) == inspect_actions_parquet(original)
    archived = archive_raw_zip(source_zip, tmp_path)

    with pytest.raises(
        ActionsEvidenceError,
        match="CSV and Parquet normalized row multisets differ",
    ):
        build_actions_acquisition_document(
            parquet_path=altered,
            raw_zip_path=archived,
            raw_zip_relative_path=archived.relative_to(tmp_path).as_posix(),
            acquired_at_utc="2026-07-13T22:00:00Z",
            last_refreshed_time="2026-07-10 03:18:34 UTC",
            data_snapshot_time="2026-07-10 03:28:21 UTC",
            datatable_metadata=_metadata(),
        )


def test_actions_equivalence_rejects_timestamp_to_date_truncation(tmp_path):
    parquet = tmp_path / "actions.parquet"
    source_zip = tmp_path / "download.zip"
    _write_actions(parquet)
    member = "SHARADAR_ACTIONS_test.csv"
    csv = (
        "date,action,ticker,name,value,contraticker,contraname\n"
        "2010-01-04 12:34:56,dividend,AAPL,Apple,0.10,,\n"
        "2026-07-10 00:00:00,split,WMT,Walmart,3.0,,\n"
    )
    with zipfile.ZipFile(
        source_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(member, csv)
    archived = archive_raw_zip(source_zip, tmp_path)

    with pytest.raises(ActionsEvidenceError, match="schema mismatch"):
        build_actions_acquisition_document(
            parquet_path=parquet,
            raw_zip_path=archived,
            raw_zip_relative_path=archived.relative_to(tmp_path).as_posix(),
            acquired_at_utc="2026-07-13T22:00:00Z",
            last_refreshed_time="2026-07-10 03:18:34 UTC",
            data_snapshot_time="2026-07-10 03:28:21 UTC",
            datatable_metadata=_metadata(),
        )

    timestamp_parquet = tmp_path / "timestamp.parquet"
    duckdb.connect().execute(
        "COPY (SELECT TIMESTAMP '2020-01-01 12:34:56' AS date, "
        "'split' AS action, 'AAPL' AS ticker, 'Apple' AS name, 2.0 AS value, "
        "CAST(NULL AS VARCHAR) AS contraticker, CAST(NULL AS VARCHAR) AS "
        f"contraname) TO '{timestamp_parquet}' (FORMAT PARQUET)"
    )
    with pytest.raises(ActionsEvidenceError, match="schema mismatch"):
        inspect_actions_parquet(timestamp_parquet)


def test_actions_evidence_reports_required_window_gap(tmp_path):
    _published_receipt(tmp_path)

    evidence = validate_actions_evidence(
        tmp_path, required_start="2000-01-01", required_end="2027-01-01"
    )

    assert evidence["payload"]["complete"] is False
    assert evidence["payload"]["blockers"] == [
        "actions_range_starts_after_required_window",
        "actions_range_ends_before_required_window",
    ]


def test_actions_parquet_rejects_schema_and_duplicate_primary_keys(tmp_path):
    wrong = tmp_path / "wrong.parquet"
    duckdb.connect().execute(
        "COPY (SELECT DATE '2020-01-01' AS date, 'split' AS action, "
        f"'AAPL' AS ticker) TO '{wrong}' (FORMAT PARQUET)"
    )
    with pytest.raises(ActionsEvidenceError, match="schema mismatch"):
        inspect_actions_parquet(wrong)

    duplicate = tmp_path / "duplicate.parquet"
    _write_actions(duplicate, duplicate=True)
    with pytest.raises(ActionsEvidenceError, match="duplicate primary keys"):
        inspect_actions_parquet(duplicate)


def test_raw_zip_must_be_content_addressed_and_receipt_requires_fresh_metadata(
    tmp_path,
):
    parquet = tmp_path / "actions.parquet"
    source_zip = tmp_path / "download.zip"
    _write_actions(parquet)
    _write_zip(source_zip)
    archived = archive_raw_zip(source_zip, tmp_path)

    with pytest.raises(ActionsEvidenceError, match="not content-addressed"):
        build_actions_acquisition_document(
            parquet_path=parquet,
            raw_zip_path=archived,
            raw_zip_relative_path="actions.zip",
            acquired_at_utc="2026-07-13T22:00:00Z",
            last_refreshed_time="2026-07-10 03:18:34 UTC",
            data_snapshot_time="2026-07-10 03:28:21 UTC",
            datatable_metadata=_metadata(),
        )

    metadata = _metadata()
    metadata["status"]["status"] = "LATE"
    with pytest.raises(ActionsEvidenceError, match="not continuously on time"):
        build_actions_acquisition_document(
            parquet_path=parquet,
            raw_zip_path=archived,
            raw_zip_relative_path=archived.relative_to(tmp_path).as_posix(),
            acquired_at_utc="2026-07-13T22:00:00Z",
            last_refreshed_time="2026-07-10 03:18:34 UTC",
            data_snapshot_time="2026-07-10 03:28:21 UTC",
            datatable_metadata=metadata,
        )
