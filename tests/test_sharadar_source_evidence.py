from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import zipfile

import duckdb
import pytest

import data.pead_sharadar_event_universe_replay as replay_module
from data.pead_event_universe import content_hash as event_content_hash
from data.pead_event_universe_index import build_pead_event_universe_index
from data.pead_sharadar_event_universe_replay import (
    EVENT_REPLAY_RECEIPT_ROOT,
    EVENT_UNIVERSE_INDEX_RECEIPT_ROOT,
    PeadSharadarEventUniverseReplayError,
    build_pead_sharadar_event_universe_replay,
    load_pead_sharadar_event_universe_replay,
    publish_pead_sharadar_event_universe_replay,
    validate_pead_sharadar_event_universe_replay_structure,
    verify_pead_sharadar_event_universe_replay,
)
from data.sharadar_source_evidence import (
    CANDIDATE_TABLES,
    SharadarSourceEvidenceError,
    build_pead_security_identity_snapshot,
    build_pead_sharadar_source_snapshot,
    build_sharadar_table_acquisition_document,
    canonical_json,
    content_hash,
    convert_sharadar_zip_to_parquet,
    file_sha256,
    normalize_datatable_metadata,
    publish_pead_security_identity_snapshot,
    publish_pead_sharadar_source_snapshot,
    publish_sharadar_table_acquisition,
    sharadar_source_record_sha256,
    validate_pead_security_identity_snapshot,
    validate_pead_security_identity_snapshot_structure,
    validate_pead_sharadar_source_snapshot,
    validate_sharadar_table_acquisition,
)


def _metadata(name: str) -> dict:
    columns = {
        "sf1": [
            ("ticker", "text", "Sharadar ticker"),
            ("dimension", "text", "Statement dimension"),
            ("calendardate", "Date", "Calendar quarter"),
            ("datekey", "Date", "Filing date"),
            ("reportperiod", "Date", "Fiscal period end"),
            ("fiscalperiod", "text", "Fiscal period"),
            ("eps", "BigDecimal", "Basic EPS"),
            ("epsdil", "BigDecimal", "Diluted EPS"),
            ("sharesbas", "BigDecimal", "Basic shares"),
            ("sharefactor", "BigDecimal", "Share normalization"),
        ],
        "sep": [
            ("ticker", "text", "Sharadar ticker"),
            ("date", "Date", "Price date"),
            ("close", "BigDecimal", "Close adjusted for splits only"),
            ("closeadj", "BigDecimal", "Close adjusted for splits and dividends"),
            ("closeunadj", "BigDecimal", "Unadjusted close"),
            ("lastupdated", "Date", "Provider update date"),
        ],
        "tickers": [
            ("table", "text", "Source table"),
            ("permaticker", "Integer", "Permanent security identifier"),
            ("ticker", "text", "Listed ticker"),
            ("exchange", "text", "Listing exchange"),
            ("isdelisted", "text", "Delisting status"),
            ("category", "text", "Security category"),
            ("currency", "text", "Listing currency"),
            ("firstpricedate", "Date", "First listed price"),
            ("lastpricedate", "Date", "Last listed price"),
            ("secfilings", "text", "SEC issuer URL"),
            ("lastupdated", "Date", "Provider update date"),
            ("firstadded", "Date", "First added date"),
            ("firstquarter", "Date", "First fundamental quarter"),
            ("lastquarter", "Date", "Last fundamental quarter"),
        ],
    }[name]
    primary_key = {
        "sf1": ["ticker", "dimension", "datekey", "reportperiod"],
        "sep": ["ticker", "date"],
        "tickers": ["table", "permaticker", "ticker"],
    }[name]
    return {
        "vendor_code": "SHARADAR",
        "datatable_code": name.upper(),
        "name": f"Synthetic {name.upper()}",
        "description": f"Synthetic candidate {name} fixture",
        "columns": [
            {"name": column, "type": kind, "description": description}
            for column, kind, description in columns
        ],
        "filters": list(primary_key),
        "primary_key": primary_key,
        "premium": True,
        "status": {
            "expected_at": "*",
            "refreshed_at": "2026-07-13T03:00:00Z",
            "status": "ON TIME",
            "update_frequency": "DAILY",
        },
    }


def _rows(name: str) -> list[list[object]]:
    if name == "sf1":
        return [
            [
                "AAA",
                "ARQ",
                "2020-03-31",
                "2020-05-01",
                "2020-03-31",
                "2020-Q1",
                1.0,
                0.9,
                100,
                1.0,
            ],
            [
                "BBB",
                "ARQ",
                "2020-03-31",
                "2020-05-02",
                "2020-03-31",
                "2020-Q1",
                2.0,
                1.8,
                200,
                1.0,
            ],
        ]
    if name == "sep":
        return [
            ["AAA", "2020-04-29", 10.0, 10.5, 20.0, "2026-07-13"],
            ["BBB", "2020-04-29", 30.0, 31.0, 30.0, "2026-07-13"],
        ]
    return [
        [
            "SF1",
            101,
            "AAA",
            "NYSE",
            "N",
            "Domestic Common Stock",
            "USD",
            "2010-01-01",
            "2026-07-13",
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000001001",
            "2026-07-13",
            "2010-01-01",
            "2010-03-31",
            "2026-03-31",
        ],
        [
            "SF1",
            202,
            "BBB",
            "NASDAQ",
            "Y",
            "Domestic Common Stock",
            "USD",
            "2011-01-01",
            "2024-12-31",
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000002002",
            "2026-07-13",
            "2011-01-01",
            "2011-03-31",
            "2024-09-30",
        ],
    ]


def _write_zip(path: Path, name: str, rows: list[list[object]] | None = None) -> None:
    metadata = _metadata(name)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([column["name"] for column in metadata["columns"]])
    writer.writerows(rows or _rows(name))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"SHARADAR_{name.upper()}_test.csv", output.getvalue())


def _publish_table(root: Path, name: str, rows=None) -> dict:
    staging = root / "staging"
    staging.mkdir(exist_ok=True)
    raw_zip = staging / f"{name}.zip"
    parquet = staging / f"{name}.parquet"
    _write_zip(raw_zip, name, rows)
    convert_sharadar_zip_to_parquet(
        raw_zip,
        parquet,
        logical_name=name,
        datatable_metadata=_metadata(name),
    )
    document = build_sharadar_table_acquisition_document(
        logical_name=name,
        raw_zip_path=raw_zip,
        parquet_path=parquet,
        acquired_at_utc="2026-07-14T12:00:00Z",
        last_refreshed_time="2026-07-13 03:00:00 UTC",
        data_snapshot_time="2026-07-13 03:05:00 UTC",
        datatable_metadata=_metadata(name),
    )
    verified, receipt_path = publish_sharadar_table_acquisition(
        root,
        raw_zip_path=raw_zip,
        parquet_path=parquet,
        document=document,
    )
    assert receipt_path.name == f"{verified['artifact_hash']}.json"
    return verified


def _snapshot(root: Path, *, ticker_rows=None, sf1_rows=None):
    acquisitions = {
        name: _publish_table(
            root,
            name,
            ticker_rows
            if name == "tickers"
            else sf1_rows
            if name == "sf1"
            else None,
        )
        for name in CANDIDATE_TABLES
    }
    snapshot = build_pead_sharadar_source_snapshot(
        warehouse_dir=root,
        candidate_id="pead-vq-source-qualification-v2",
        created_at_utc="2026-07-14T13:00:00Z",
        acquisitions=acquisitions,
    )
    published, _ = publish_pead_sharadar_source_snapshot(root, snapshot)
    return acquisitions, published


@pytest.mark.parametrize("name", CANDIDATE_TABLES)
def test_candidate_table_receipt_replays_raw_csv_and_parquet(tmp_path, name):
    receipt = _publish_table(tmp_path, name)

    verified = validate_sharadar_table_acquisition(tmp_path, receipt)

    payload = verified["payload"]
    assert payload["logical_name"] == name
    assert payload["row_equivalence"] == {
        "schema_version": "sharadar_row_equivalence.v1",
        "method": "duckdb_bidirectional_except_all.v1",
        "rows": 2,
        "csv_minus_parquet_rows": 0,
        "parquet_minus_csv_rows": 0,
        "equivalent": True,
    }
    assert payload["source"]["canonical_request"]["parameters"]["qopts.export"] == "true"
    assert "description" in payload["source"]["datatable_metadata"]["columns"][0]
    assert payload["raw_zip"]["relative_path"].endswith(f"/{payload['raw_zip']['sha256']}.zip")
    assert payload["parquet"]["relative_path"].endswith(f"/{payload['parquet']['sha256']}.parquet")


def test_row_equivalence_rejects_self_consistent_parquet_row_mutation(tmp_path):
    receipt = _publish_table(tmp_path, "sep")
    payload = receipt["payload"]
    parquet = tmp_path / payload["parquet"]["relative_path"]
    altered = tmp_path / "altered.parquet"
    escaped = str(parquet).replace("'", "''")
    duckdb.connect().execute(
        "COPY (SELECT ticker,date,close + CASE WHEN ticker='AAA' THEN 1 ELSE 0 END AS close,"
        "closeadj,closeunadj,lastupdated FROM "
        f"read_parquet('{escaped}')) TO '{altered}' (FORMAT PARQUET)"
    )
    assert file_sha256(altered) != payload["parquet"]["sha256"]
    forged = copy.deepcopy(receipt)
    forged["payload"]["parquet"]["sha256"] = file_sha256(altered)
    forged["payload"]["parquet"]["bytes"] = altered.stat().st_size
    forged["payload"]["parquet"]["relative_path"] = (
        f"source_snapshots/sharadar/sep/parquet/{file_sha256(altered)}.parquet"
    )
    forged["artifact_hash"] = content_hash(forged["payload"])

    with pytest.raises(SharadarSourceEvidenceError, match="row multisets differ"):
        publish_sharadar_table_acquisition(
            tmp_path,
            raw_zip_path=tmp_path / payload["raw_zip"]["relative_path"],
            parquet_path=altered,
            document=forged,
        )


def test_source_snapshot_and_identity_authoritatively_replay(tmp_path):
    acquisitions, snapshot = _snapshot(tmp_path)

    assert validate_pead_sharadar_source_snapshot(snapshot, warehouse_dir=tmp_path) == snapshot
    identity = build_pead_security_identity_snapshot(
        warehouse_dir=tmp_path,
        candidate_id="pead-vq-source-qualification-v2",
        created_at_utc="2026-07-14T14:00:00Z",
        source_snapshot=snapshot,
    )
    published, _ = publish_pead_security_identity_snapshot(
        tmp_path, identity, source_snapshot=snapshot
    )

    assert published["payload"]["qualification_allowed"] is True
    assert published["payload"]["coverage"] == {
        "source_row_count": 2,
        "disposition_count": 2,
        "identity_count": 2,
        "identity_gap_count": 0,
        "complete": True,
    }
    assert {item["cik"] for item in published["payload"]["identities"]} == {
        "0000001001",
        "0000002002",
    }
    assert (
        validate_pead_security_identity_snapshot(
            published, warehouse_dir=tmp_path, source_snapshot=snapshot
        )
        == published
    )

    tampered = copy.deepcopy(published)
    tampered["payload"]["identities"][0]["exchange"] = "OTC"
    core = {
        key: value
        for key, value in tampered["payload"]["identities"][0].items()
        if key != "identity_id"
    }
    tampered["payload"]["identities"][0]["identity_id"] = content_hash(
        {"schema_version": "pead_security_identity.v1", **core}
    )
    for disposition in tampered["payload"]["source_dispositions"]:
        if disposition["source_record_sha256"] == core["source_record_sha256"]:
            disposition["identity_id"] = tampered["payload"]["identities"][0]["identity_id"]
    tampered["payload"]["identities"].sort(key=lambda item: item["identity_id"])
    tampered["artifact_hash"] = content_hash(tampered["payload"])
    validate_pead_security_identity_snapshot_structure(tampered)
    with pytest.raises(SharadarSourceEvidenceError, match="does not replay"):
        validate_pead_security_identity_snapshot(
            tampered, warehouse_dir=tmp_path, source_snapshot=snapshot
        )
    assert (
        acquisitions["tickers"]["artifact_hash"]
        == (snapshot["payload"]["tables"][2]["acquisition_artifact_hash"])
    )


def test_identity_gaps_are_explicit_but_do_not_block_unrelated_rows(tmp_path):
    rows = _rows("tickers")
    rows[1][9] = "https://example.invalid/not-a-cik"
    _, snapshot = _snapshot(tmp_path, ticker_rows=rows)

    identity = build_pead_security_identity_snapshot(
        warehouse_dir=tmp_path,
        candidate_id="pead-vq-source-qualification-v2",
        created_at_utc="2026-07-14T14:00:00Z",
        source_snapshot=snapshot,
    )

    assert identity["payload"]["qualification_allowed"] is True
    assert identity["payload"]["blockers"] == []
    assert identity["payload"]["coverage"]["identity_count"] == 1
    assert identity["payload"]["coverage"]["identity_gap_count"] == 1
    gap = next(
        item
        for item in identity["payload"]["source_dispositions"]
        if item["disposition"] == "identity_gap"
    )
    assert gap["reason"] == "invalid_cik_source_url"


def test_same_cik_multiple_securities_is_retained_without_guessing(tmp_path):
    rows = _rows("tickers")
    rows[1][9] = rows[0][9]
    _, snapshot = _snapshot(tmp_path, ticker_rows=rows)

    identity = build_pead_security_identity_snapshot(
        warehouse_dir=tmp_path,
        candidate_id="pead-vq-source-qualification-v2",
        created_at_utc="2026-07-14T14:00:00Z",
        source_snapshot=snapshot,
    )

    assert identity["payload"]["qualification_allowed"] is True
    assert len(identity["payload"]["identities"]) == 2
    assert len({item["cik"] for item in identity["payload"]["identities"]}) == 1
    assert len({item["permaticker"] for item in identity["payload"]["identities"]}) == 2


def test_source_record_digest_is_ordered_typed_and_shared(tmp_path):
    receipt = _publish_table(tmp_path, "sep")
    schema = receipt["payload"]["parquet"]["schema"]
    parquet = tmp_path / receipt["payload"]["parquet"]["relative_path"]
    row = (
        duckdb.connect()
        .execute(f"SELECT * FROM read_parquet('{parquet}') ORDER BY ticker LIMIT 1")
        .fetchone()
    )
    assert row is not None
    digest_from_sequence = sharadar_source_record_sha256("sep", schema, row)
    digest_from_mapping = sharadar_source_record_sha256(
        "sep",
        schema,
        dict(zip([item["name"] for item in schema], row, strict=True)),
    )
    assert digest_from_sequence == digest_from_mapping
    assert len(digest_from_sequence) == 64


def test_tickers_provider_primary_key_drift_fails_closed():
    metadata = _metadata("tickers")
    metadata["primary_key"] = ["table", "permaticker"]

    with pytest.raises(SharadarSourceEvidenceError, match="primary key differs"):
        normalize_datatable_metadata("tickers", metadata)


def test_sf1_provider_primary_key_drift_fails_closed():
    metadata = _metadata("sf1")
    metadata["primary_key"] = ["ticker", "dimension", "calendardate"]

    with pytest.raises(SharadarSourceEvidenceError, match="primary key differs"):
        normalize_datatable_metadata("sf1", metadata)


def test_provider_metadata_retains_nullable_status_fields_without_invention():
    metadata = _metadata("tickers")
    metadata["status"] = {
        "expected_at": None,
        "refreshed_at": "2026-07-13T23:11:59.000Z",
        "status": None,
        "update_frequency": None,
    }

    normalized = normalize_datatable_metadata("tickers", metadata)

    assert normalized["status"] == metadata["status"]

    for field in ("expected_at", "status", "update_frequency"):
        drifted = copy.deepcopy(metadata)
        drifted["status"][field] = ""
        with pytest.raises(SharadarSourceEvidenceError, match="metadata .*invalid"):
            normalize_datatable_metadata("tickers", drifted)


def test_tickers_date_ranges_accept_native_dates_and_canonical_text_only(tmp_path):
    metadata = _metadata("tickers")
    next(column for column in metadata["columns"] if column["name"] == "firstadded")["type"] = (
        "text"
    )
    raw_zip = tmp_path / "tickers-text-date.zip"
    parquet = tmp_path / "tickers-text-date.parquet"
    _write_zip(raw_zip, "tickers")
    convert_sharadar_zip_to_parquet(
        raw_zip,
        parquet,
        logical_name="tickers",
        datatable_metadata=metadata,
    )

    document = build_sharadar_table_acquisition_document(
        logical_name="tickers",
        raw_zip_path=raw_zip,
        parquet_path=parquet,
        acquired_at_utc="2026-07-14T12:00:00Z",
        last_refreshed_time="2026-07-13 03:00:00 UTC",
        data_snapshot_time="2026-07-13 03:05:00 UTC",
        datatable_metadata=metadata,
    )

    ranges = {
        item["column"]: item for item in document["payload"]["parquet"]["statistics"]["date_ranges"]
    }
    assert ranges["firstpricedate"]["min"] == "2010-01-01"  # native DATE
    assert ranges["firstadded"]["min"] == "2010-01-01"  # canonical VARCHAR

    bad_rows = _rows("tickers")
    bad_rows[0][11] = "not-a-date"
    bad_zip = tmp_path / "tickers-bad-text-date.zip"
    bad_parquet = tmp_path / "tickers-bad-text-date.parquet"
    _write_zip(bad_zip, "tickers", bad_rows)
    convert_sharadar_zip_to_parquet(
        bad_zip,
        bad_parquet,
        logical_name="tickers",
        datatable_metadata=metadata,
    )
    with pytest.raises(SharadarSourceEvidenceError, match="canonical YYYY-MM-DD"):
        build_sharadar_table_acquisition_document(
            logical_name="tickers",
            raw_zip_path=bad_zip,
            parquet_path=bad_parquet,
            acquired_at_utc="2026-07-14T12:00:00Z",
            last_refreshed_time="2026-07-13 03:00:00 UTC",
            data_snapshot_time="2026-07-13 03:05:00 UTC",
            datatable_metadata=metadata,
        )


def test_strict_snapshot_loader_rejects_duplicate_json_keys(tmp_path):
    _, snapshot = _snapshot(tmp_path)
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"artifact_hash":"a","artifact_hash":"b","payload":{}}',
        encoding="utf-8",
    )
    from data.sharadar_source_evidence import load_pead_sharadar_source_snapshot

    with pytest.raises(SharadarSourceEvidenceError, match="duplicate key"):
        load_pead_sharadar_source_snapshot(path, warehouse_dir=tmp_path)
    assert json.loads(canonical_json(snapshot))["artifact_hash"] == snapshot["artifact_hash"]


def _target_sf1_rows() -> list[list[object]]:
    rows: list[list[object]] = []
    for year in range(2015, 2024):
        period = f"{year}-03-31"
        rows.append(
            [
                "AAA",
                "ARQ",
                period,
                f"{year}-05-15",
                period,
                f"{year}-Q1",
                float(year - 2014),
                float(year - 2014) - 0.1,
                100,
                1.0,
            ]
        )
    rows.append(
        [
            "AAA",
            "ARQ",
            "2024-09-30",
            "2024-11-14",
            "2024-09-30",
            "2025-Q1",
            10.0,
            9.9,
            100,
            1.0,
        ]
    )
    # A second complete source row for the same security/issuer-period is a
    # retained revision, not a second event.
    rows.append(
        [
            "AAA",
            "ARQ",
            "2020-03-31",
            "2020-06-01",
            "2020-03-31",
            "2020-Q1",
            99.0,
            98.9,
            100,
            1.0,
        ]
    )
    return rows


def _event_replay_fixture(tmp_path: Path) -> dict:
    _, snapshot = _snapshot(tmp_path, sf1_rows=_target_sf1_rows())
    identity = build_pead_security_identity_snapshot(
        warehouse_dir=tmp_path,
        candidate_id="pead-vq-source-qualification-v2",
        created_at_utc="2026-07-14T14:00:00Z",
        source_snapshot=snapshot,
    )
    candidate_specification = tmp_path / "candidate-specification.json"
    candidate_specification.write_bytes(b'{"candidate":"pead-test"}\n')
    construction_code = Path(replay_module.__file__)
    candidate_hash = file_sha256(candidate_specification)
    code_hash = file_sha256(construction_code)
    bundle = build_pead_sharadar_event_universe_replay(
        warehouse_dir=tmp_path,
        source_snapshot=snapshot,
        identity_snapshot=identity,
        candidate_specification_path=candidate_specification,
        construction_code_path=construction_code,
        trusted_candidate_specification_sha256s={candidate_hash},
        trusted_construction_code_sha256s=frozenset({code_hash}),
        created_at_utc="2026-07-14T15:00:00Z",
    )
    return {
        "bundle": bundle,
        "snapshot": snapshot,
        "identity": identity,
        "candidate_specification": candidate_specification,
        "construction_code": construction_code,
        "candidate_hash": candidate_hash,
        "code_hash": code_hash,
    }


def _event_replay_authority_kwargs(fixture: dict) -> dict:
    return {
        "warehouse_dir": fixture["candidate_specification"].parent,
        "source_snapshot": fixture["snapshot"],
        "identity_snapshot": fixture["identity"],
        "candidate_specification_path": fixture["candidate_specification"],
        "construction_code_path": fixture["construction_code"],
        "trusted_candidate_specification_sha256s": {fixture["candidate_hash"]},
        "trusted_construction_code_sha256s": {fixture["code_hash"]},
    }


def _verify_event_replay(fixture: dict, *, bundle: dict | None = None) -> dict:
    evidence = fixture["bundle"] if bundle is None else bundle
    return verify_pead_sharadar_event_universe_replay(
        evidence["replay"],
        evidence["index"],
        **_event_replay_authority_kwargs(fixture),
    )


def _publish_event_replay(fixture: dict) -> tuple[dict, dict]:
    authority = _event_replay_authority_kwargs(fixture)
    warehouse_dir = authority.pop("warehouse_dir")
    return publish_pead_sharadar_event_universe_replay(
        warehouse_dir,
        fixture["bundle"]["replay"],
        fixture["bundle"]["index"],
        **authority,
    )


def test_sharadar_event_replay_builds_v2_years_and_retains_revisions(tmp_path):
    fixture = _event_replay_fixture(tmp_path)
    bundle = fixture["bundle"]

    assert _verify_event_replay(fixture) == bundle
    replay = bundle["replay"]
    assert replay["payload"]["schema_version"] == (
        "pead_sharadar_event_universe_replay.v1"
    )
    assert [year["partition_id"] for year in replay["payload"]["years"]] == [
        str(year) for year in range(2015, 2025)
    ]
    assert {
        year["event_universe"]["payload"]["schema_version"]
        for year in replay["payload"]["years"]
    } == {"pead_event_universe.v2"}
    assert replay["payload"]["coverage"] == {
        "partition_count": 10,
        "source_record_count": 11,
        "expected_event_count": 10,
        "identity_gap_count": 0,
        "additional_revision_count": 1,
        "complete": True,
    }
    revised = next(
        lineage
        for year in replay["payload"]["years"]
        for lineage in year["event_lineage"]
        if lineage["event_key"]["fiscal_period_end"] == "2020-03-31"
    )
    assert revised["sf1_revision_count"] == 2
    assert revised["representative_sf1_source_record_sha256"] == min(
        revised["sf1_source_record_sha256s"]
    )
    final_year = replay["payload"]["years"][-1]
    assert final_year["raw_census"]["payload"]["records"][0]["fiscalperiod"] == (
        "2025-Q1"
    )
    assert final_year["event_lineage"][0]["event_key"]["fiscal_period_end"] == (
        "2024-09-30"
    )


def test_sharadar_event_replay_rejects_self_consistent_raw_census_tamper(tmp_path):
    fixture = _event_replay_fixture(tmp_path)
    forged = copy.deepcopy(fixture["bundle"])
    year = forged["replay"]["payload"]["years"][0]
    year["raw_census"]["payload"]["records"][0]["datekey"] = "2015-05-16"
    year["raw_census"]["artifact_hash"] = event_content_hash(
        year["raw_census"]["payload"]
    )
    receipt = year["event_universe"]["payload"]["census_receipt"]
    receipt["payload"]["raw_census_artifact_sha256"] = year["raw_census"][
        "artifact_hash"
    ]
    receipt["artifact_hash"] = event_content_hash(receipt["payload"])
    year["event_universe"]["artifact_hash"] = event_content_hash(
        year["event_universe"]["payload"]
    )
    forged["index"] = build_pead_event_universe_index(
        partitions=[
            child["event_universe"]
            for child in forged["replay"]["payload"]["years"]
        ],
        target_start="2015-01-01",
        target_end="2024-09-30",
        indexed_at_utc="2026-07-14T15:00:00Z",
    )
    forged["replay"]["artifact_hash"] = event_content_hash(
        forged["replay"]["payload"]
    )

    validate_pead_sharadar_event_universe_replay_structure(forged["replay"])
    with pytest.raises(
        PeadSharadarEventUniverseReplayError,
        match="does not rederive from immutable Sharadar evidence",
    ):
        _verify_event_replay(fixture, bundle=forged)


def test_sharadar_event_census_never_selects_one_of_multiple_permatickers():
    base = {
        "calendardate": "2020-03-31",
        "datekey": "2020-05-01",
        "reportperiod": "2020-03-31",
        "fiscalperiod": "2021-Q1",
        "identity_disposition": "matched",
        "identity_reason": None,
        "cik": "0000001001",
    }
    records = [
        {
            **base,
            "source_record_sha256": "1" * 64,
            "ticker": "AAA",
            "identity_id": "a" * 64,
            "permaticker": 101,
        },
        {
            **base,
            "source_record_sha256": "2" * 64,
            "ticker": "AAB",
            "identity_id": "b" * 64,
            "permaticker": 202,
        },
    ]

    dispositions, lineages, additional = replay_module._census_and_lineage(records)

    assert lineages == []
    assert additional == 0
    assert {row["disposition"] for row in dispositions} == {"identity_gap"}
    assert {row["reason"] for row in dispositions} == {
        "multiple_permatickers_for_issuer_period"
    }


def test_sharadar_event_replay_requires_externally_trusted_spec_bytes(tmp_path):
    fixture = _event_replay_fixture(tmp_path)

    with pytest.raises(
        PeadSharadarEventUniverseReplayError,
        match="candidate specification hash is not externally trusted",
    ):
        verify_pead_sharadar_event_universe_replay(
            fixture["bundle"]["replay"],
            fixture["bundle"]["index"],
            warehouse_dir=tmp_path,
            source_snapshot=fixture["snapshot"],
            identity_snapshot=fixture["identity"],
            candidate_specification_path=fixture["candidate_specification"],
            construction_code_path=fixture["construction_code"],
            trusted_candidate_specification_sha256s={"0" * 64},
            trusted_construction_code_sha256s={fixture["code_hash"]},
        )


def test_sharadar_event_replay_create_only_publish_and_authoritative_load(tmp_path):
    fixture = _event_replay_fixture(tmp_path)

    published, paths = _publish_event_replay(fixture)

    assert published == fixture["bundle"]
    assert paths == {
        "replay": tmp_path
        / EVENT_REPLAY_RECEIPT_ROOT
        / f"{published['replay']['artifact_hash']}.json",
        "index": tmp_path
        / EVENT_UNIVERSE_INDEX_RECEIPT_ROOT
        / f"{published['index']['artifact_hash']}.json",
    }
    for name, path in paths.items():
        assert path.read_bytes() == (
            canonical_json(published[name]) + "\n"
        ).encode("utf-8")
    loaded = load_pead_sharadar_event_universe_replay(
        paths["replay"],
        paths["index"],
        **_event_replay_authority_kwargs(fixture),
    )
    assert loaded == published
    assert _publish_event_replay(fixture) == (published, paths)


def test_sharadar_event_replay_publication_refuses_content_address_collision(tmp_path):
    fixture = _event_replay_fixture(tmp_path)
    published, paths = _publish_event_replay(fixture)
    original_replay = paths["replay"].read_bytes()
    paths["replay"].write_bytes(b"{}\n")

    with pytest.raises(
        PeadSharadarEventUniverseReplayError,
        match="content-addressed event replay collision",
    ):
        _publish_event_replay(fixture)

    paths["replay"].write_bytes(original_replay)
    original_index = paths["index"].read_bytes()
    paths["index"].write_bytes(b"{}\n")
    with pytest.raises(
        PeadSharadarEventUniverseReplayError,
        match="content-addressed event-universe index collision",
    ):
        _publish_event_replay(fixture)
    paths["index"].write_bytes(original_index)
    assert published == fixture["bundle"]


def test_sharadar_event_replay_publish_verifies_before_writing(tmp_path):
    fixture = _event_replay_fixture(tmp_path)
    forged = copy.deepcopy(fixture["bundle"])
    forged["replay"]["payload"]["years"][0]["raw_census"]["payload"]["records"][
        0
    ]["datekey"] = "2015-05-16"
    forged["replay"]["artifact_hash"] = event_content_hash(
        forged["replay"]["payload"]
    )
    authority = _event_replay_authority_kwargs(fixture)
    warehouse_dir = authority.pop("warehouse_dir")

    with pytest.raises(PeadSharadarEventUniverseReplayError):
        publish_pead_sharadar_event_universe_replay(
            warehouse_dir,
            forged["replay"],
            forged["index"],
            **authority,
        )

    assert not (tmp_path / EVENT_REPLAY_RECEIPT_ROOT).exists()
    assert not (tmp_path / EVENT_UNIVERSE_INDEX_RECEIPT_ROOT).exists()


def test_sharadar_event_replay_loader_enforces_path_and_size_bound(tmp_path, monkeypatch):
    fixture = _event_replay_fixture(tmp_path)
    published, paths = _publish_event_replay(fixture)
    outside = tmp_path / "copied-replay.json"
    outside.write_bytes(paths["replay"].read_bytes())

    with pytest.raises(
        PeadSharadarEventUniverseReplayError,
        match="not at its immutable content-addressed warehouse path",
    ):
        load_pead_sharadar_event_universe_replay(
            outside,
            paths["index"],
            **_event_replay_authority_kwargs(fixture),
        )

    monkeypatch.setattr(
        replay_module, "MAX_REPLAY_BYTES", paths["replay"].stat().st_size - 1
    )
    with pytest.raises(PeadSharadarEventUniverseReplayError, match="exceeds its size limit"):
        load_pead_sharadar_event_universe_replay(
            paths["replay"],
            paths["index"],
            **_event_replay_authority_kwargs(fixture),
        )
    assert published == fixture["bundle"]


def test_sharadar_event_replay_loader_rejects_duplicate_json_keys(tmp_path):
    fixture = _event_replay_fixture(tmp_path)
    _, paths = _publish_event_replay(fixture)
    paths["replay"].write_bytes(
        b'{"artifact_hash":"a","artifact_hash":"b","payload":{}}\n'
    )

    with pytest.raises(PeadSharadarEventUniverseReplayError, match="duplicate key"):
        load_pead_sharadar_event_universe_replay(
            paths["replay"],
            paths["index"],
            **_event_replay_authority_kwargs(fixture),
        )
