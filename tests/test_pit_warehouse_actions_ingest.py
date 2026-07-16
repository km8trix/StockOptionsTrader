from __future__ import annotations

from io import BytesIO
import zipfile

import pytest

from data.corporate_action_evidence import (
    ACTIONS_RECEIPT_FILE,
    ActionsEvidenceError,
    expected_datatable_metadata,
    inspect_actions_parquet,
)
from data.pit_provider import PitCache
from data.pit_warehouse import PitWarehouse


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


def _zip_bytes(*, second_ticker: str = "WMT") -> bytes:
    csv = (
        "date,action,ticker,name,value,contraticker,contraname\n"
        "2010-01-04,dividend,AAPL,Apple,0.10,,\n"
        f"2026-07-10,split,{second_ticker},Walmart,3.0,OLD,Old Walmart\n"
    )
    output = BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("SHARADAR_ACTIONS_test.csv", csv)
    return output.getvalue()


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset:offset + chunk_size]


def _mock_actions_source(monkeypatch, warehouse: PitWarehouse, payload: list[bytes]) -> None:
    monkeypatch.setattr(
        warehouse,
        "_actions_datatable_metadata",
        lambda table: _metadata(),
    )

    def export(table, params, *, include_metadata=False, **_kwargs):
        assert table == "SHARADAR/ACTIONS"
        assert params == {}
        assert include_metadata is True
        return "https://signed.invalid/actions.zip?secret=one-time", {
            "last_refreshed_time": "2026-07-10 03:18:34 UTC",
            "data_snapshot_time": "2026-07-10 03:28:21 UTC",
            "status": "fresh",
        }

    monkeypatch.setattr(warehouse, "_export_link", export)
    monkeypatch.setattr(
        "requests.get",
        lambda url, **kwargs: _Response(payload[0]),
    )


def test_actions_ingest_publishes_verified_evidence_without_signed_link(
    tmp_path, monkeypatch,
):
    warehouse = PitWarehouse(str(tmp_path))
    payload = [_zip_bytes()]
    _mock_actions_source(monkeypatch, warehouse, payload)

    assert warehouse.ingest_table("actions") == 2

    statistics = inspect_actions_parquet(tmp_path / "actions.parquet")
    assert statistics["rows"] == 2
    evidence = warehouse.corporate_action_evidence(
        "2015-01-01", "2024-09-30"
    )
    assert evidence["payload"]["complete"] is True
    assert evidence["payload"]["value_is_terminal_payout_per_share"] is False
    receipt = (tmp_path / ACTIONS_RECEIPT_FILE).read_text(encoding="utf-8")
    assert "signed.invalid" not in receipt
    assert "secret=one-time" not in receipt
    assert "api_key" not in receipt
    archives = list((tmp_path / "source_snapshots" / "actions").glob("*.zip"))
    assert len(archives) == 1
    assert archives[0].stem == evidence["payload"]["raw_zip_sha256"]


def test_actions_prepublication_failure_preserves_active_snapshot(
    tmp_path, monkeypatch,
):
    import data.pit_warehouse as pit_warehouse

    warehouse = PitWarehouse(str(tmp_path))
    payload = [_zip_bytes()]
    _mock_actions_source(monkeypatch, warehouse, payload)
    warehouse.ingest_table("actions")
    parquet_before = (tmp_path / "actions.parquet").read_bytes()
    receipt_before = (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes()

    payload[0] = _zip_bytes(second_ticker="COST")

    def fail_before_publish(**_kwargs):
        raise ActionsEvidenceError("receipt construction failed")

    monkeypatch.setattr(
        pit_warehouse, "build_actions_acquisition_document", fail_before_publish
    )
    with pytest.raises(ActionsEvidenceError, match="receipt construction failed"):
        warehouse.ingest_table("actions")

    assert (tmp_path / "actions.parquet").read_bytes() == parquet_before
    assert (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes() == receipt_before
    assert warehouse.corporate_action_evidence(
        "2015-01-01", "2024-09-30"
    )["payload"]["complete"] is True
    # Archiving a new immutable raw source before receipt construction may
    # leave an orphan, but it never changes the active evidence pair.
    assert len(list(
        (tmp_path / "source_snapshots" / "actions").glob("*.zip")
    )) == 2


def test_actions_receipt_publish_interruption_fails_closed(
    tmp_path, monkeypatch,
):
    import data.pit_warehouse as pit_warehouse

    warehouse = PitWarehouse(str(tmp_path))
    payload = [_zip_bytes()]
    _mock_actions_source(monkeypatch, warehouse, payload)
    warehouse.ingest_table("actions")
    old_receipt = (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes()

    payload[0] = _zip_bytes(second_ticker="COST")

    def interrupt_receipt_publish(*_args, **_kwargs):
        raise RuntimeError("simulated publication interruption")

    monkeypatch.setattr(
        pit_warehouse, "atomic_write_json", interrupt_receipt_publish
    )
    with pytest.raises(RuntimeError, match="publication interruption"):
        warehouse.ingest_table("actions")

    # The publication order is Parquet then receipt.  If interrupted between
    # them, the prior receipt remains but no longer authenticates the active
    # Parquet; research therefore rejects the half-published pair.
    assert (tmp_path / ACTIONS_RECEIPT_FILE).read_bytes() == old_receipt
    with pytest.raises(ActionsEvidenceError, match="identity mismatch"):
        warehouse.corporate_action_evidence("2015-01-01", "2024-09-30")


def test_actions_evidence_reader_raises_typed_error_on_missing_receipt(tmp_path):
    warehouse = PitWarehouse(str(tmp_path))

    with pytest.raises(ActionsEvidenceError, match="receipt is missing"):
        warehouse.corporate_action_evidence("2015-01-01", "2024-09-30")
    with pytest.raises(ActionsEvidenceError, match="dates must be valid"):
        warehouse.corporate_action_evidence("not-a-date", "2024-09-30")


def test_export_link_returns_only_sanitized_bulk_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "request-only-secret")

    def get_json(table, params):
        assert table == "SHARADAR/ACTIONS"
        assert params["api_key"] == "request-only-secret"
        return {
            "datatable_bulk_download": {
                "datatable": {
                    "last_refreshed_time": "2026-07-10 03:18:34 UTC",
                },
                "file": {
                    "link": "https://signed.invalid/actions.zip?token=secret",
                    "status": "fresh",
                    "data_snapshot_time": "2026-07-10 03:28:21 UTC",
                    "next_refresh_time": "2026-07-11 03:28:21 UTC",
                    "provider_internal_secret": "never persist",
                }
            }
        }

    monkeypatch.setattr(PitCache, "_get_json", staticmethod(get_json))
    warehouse = PitWarehouse(str(tmp_path))

    link, metadata = warehouse._export_link(
        "SHARADAR/ACTIONS", {}, include_metadata=True
    )

    assert link.startswith("https://signed.invalid/")
    assert metadata == {
        "last_refreshed_time": "2026-07-10 03:18:34 UTC",
        "data_snapshot_time": "2026-07-10 03:28:21 UTC",
        "status": "fresh",
    }
    assert warehouse._export_link("SHARADAR/ACTIONS", {}) == link


def test_datatable_metadata_fetch_drops_request_and_response_secrets(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "request-only-secret")
    body = {
        "datatable": {
            **_metadata(),
            "signed_link": "https://signed.invalid/metadata",
            "api_key": "response-secret",
        }
    }

    def get_json(table, params):
        assert table == "SHARADAR/ACTIONS/metadata"
        assert params == {"api_key": "request-only-secret"}
        return body

    monkeypatch.setattr(PitCache, "_get_json", staticmethod(get_json))
    metadata = PitWarehouse(str(tmp_path))._actions_datatable_metadata(
        "SHARADAR/ACTIONS"
    )

    assert metadata == _metadata()
    assert "signed_link" not in metadata
    assert "api_key" not in metadata
