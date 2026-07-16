from __future__ import annotations

from io import BytesIO
import json
import traceback
import zipfile

import pytest
import requests

from data.pit_provider import PitCache
from data.pit_warehouse import PitWarehouse
from data.sharadar_source_evidence import (
    SharadarSourceEvidenceError,
    load_sharadar_table_acquisition,
)


def _metadata() -> dict:
    return {
        "vendor_code": "SHARADAR",
        "datatable_code": "SEP",
        "name": "Equity Prices",
        "description": "Synthetic SEP metadata",
        "columns": [
            {"name": "ticker", "type": "text", "description": "Ticker"},
            {"name": "date", "type": "Date", "description": "Price date"},
            {
                "name": "close",
                "type": "BigDecimal",
                "description": "Close adjusted for splits only",
            },
            {
                "name": "closeadj",
                "type": "BigDecimal",
                "description": "Close adjusted for splits and dividends",
            },
            {
                "name": "closeunadj",
                "type": "BigDecimal",
                "description": "Unadjusted close",
            },
            {
                "name": "lastupdated",
                "type": "Date",
                "description": "Provider update date",
            },
        ],
        "filters": ["ticker", "date"],
        "primary_key": ["ticker", "date"],
        "premium": True,
        "status": {
            "expected_at": "*",
            "refreshed_at": "2026-07-13T03:00:00Z",
            "status": "ON TIME",
            "update_frequency": "DAILY",
        },
    }


def _zip_bytes(*, bad_header: bool = False) -> bytes:
    header = (
        "symbol,date,close,closeadj,closeunadj,lastupdated"
        if bad_header
        else "ticker,date,close,closeadj,closeunadj,lastupdated"
    )
    csv = (
        f"{header}\n"
        "AAA,2020-04-29,10,10.5,20,2026-07-13\n"
        "BBB,2020-04-29,30,31,30,2026-07-13\n"
    )
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SHARADAR_SEP_test.csv", csv)
    return output.getvalue()


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, *, chunk_size: int):
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset : offset + chunk_size]


def _mock_source(monkeypatch, warehouse: PitWarehouse, payload: list[bytes]) -> None:
    monkeypatch.setattr(
        warehouse,
        "_candidate_datatable_metadata",
        lambda logical_name, table: _metadata(),
    )

    def export(table, params, *, include_metadata=False, **_kwargs):
        assert table == "SHARADAR/SEP"
        assert params == {}
        assert include_metadata is True
        return "https://signed.invalid/sep.zip?secret=one-time", {
            "last_refreshed_time": "2026-07-13 03:00:00 UTC",
            "data_snapshot_time": "2026-07-13 03:05:00 UTC",
            "status": "fresh",
        }

    monkeypatch.setattr(warehouse, "_export_link", export)
    monkeypatch.setattr(
        "requests.get", lambda url, **kwargs: _Response(payload[0])
    )


def test_candidate_ingest_routes_sep_through_immutable_receipt(tmp_path, monkeypatch):
    warehouse = PitWarehouse(str(tmp_path))
    payload = [_zip_bytes()]
    _mock_source(monkeypatch, warehouse, payload)

    assert warehouse.ingest_table("sep") == 2

    active_receipt = tmp_path / "sep.acquisition.json"
    document = load_sharadar_table_acquisition(
        active_receipt, warehouse_dir=tmp_path
    )
    assert (tmp_path / "sep.parquet").is_file()
    assert document["payload"]["row_equivalence"]["equivalent"] is True
    immutable_receipt = (
        tmp_path
        / "source_snapshots"
        / "sharadar"
        / "sep"
        / "receipts"
        / f"{document['artifact_hash']}.json"
    )
    assert json.loads(immutable_receipt.read_text()) == document
    receipt_text = active_receipt.read_text(encoding="utf-8")
    assert "signed.invalid" not in receipt_text
    assert "one-time" not in receipt_text
    assert "api_key" not in receipt_text


def test_candidate_prepublication_failure_preserves_active_cache(tmp_path, monkeypatch):
    warehouse = PitWarehouse(str(tmp_path))
    payload = [_zip_bytes()]
    _mock_source(monkeypatch, warehouse, payload)
    warehouse.ingest_table("sep")
    parquet_before = (tmp_path / "sep.parquet").read_bytes()
    receipt_before = (tmp_path / "sep.acquisition.json").read_bytes()

    payload[0] = _zip_bytes(bad_header=True)
    with pytest.raises(SharadarSourceEvidenceError, match="CSV header differs"):
        warehouse.ingest_table("sep")

    assert (tmp_path / "sep.parquet").read_bytes() == parquet_before
    assert (tmp_path / "sep.acquisition.json").read_bytes() == receipt_before


def test_candidate_metadata_fetch_drops_request_and_response_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "request-only-secret")
    body = {
        "datatable": {
            **_metadata(),
            "api_key": "response-secret",
            "signed_link": "https://signed.invalid/metadata",
        }
    }

    def get_json(table, params):
        assert table == "SHARADAR/SEP/metadata"
        assert params == {"api_key": "request-only-secret"}
        return body

    monkeypatch.setattr(PitCache, "_get_json", staticmethod(get_json))
    metadata = PitWarehouse(str(tmp_path))._candidate_datatable_metadata(
        "sep", "SHARADAR/SEP"
    )

    assert metadata == _metadata()
    assert "api_key" not in metadata
    assert "signed_link" not in metadata


def test_candidate_download_failure_suppresses_signed_url_and_token(
    tmp_path, monkeypatch
):
    warehouse = PitWarehouse(str(tmp_path))
    payload = [_zip_bytes()]
    _mock_source(monkeypatch, warehouse, payload)

    class FailingResponse(_Response):
        def raise_for_status(self):
            raise requests.exceptions.RequestException(
                "403 for https://signed.invalid/sep.zip?token=do-not-print"
            )

    monkeypatch.setattr(
        "requests.get", lambda url, **kwargs: FailingResponse(payload[0])
    )
    with pytest.raises(
        SharadarSourceEvidenceError, match="candidate bulk download failed"
    ) as caught:
        warehouse.ingest_table("sep")

    rendered = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert "signed.invalid" not in rendered
    assert "do-not-print" not in rendered
    assert caught.value.__suppress_context__ is True
