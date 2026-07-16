from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from data.earnings_announcements import (
    CANONICAL_TABLES,
    CredentialError,
    EarningsAnnouncementSnapshot,
    EarningsAnnouncementStore,
    ProviderResponseError,
    SUPPORTED_TABLES,
    SnapshotIntegrityError,
    ZacksTablesClient,
    canonical_json,
    canonical_query_params,
)
from scripts.ingest_earnings_announcements import main as ingest_main


API_KEY = "test-key-must-never-be-persisted"
BASE_TIME = datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)
DATE_COLUMNS = {
    "ES": "act_rpt_date",
    "SS": "act_rpt_date",
    "EEH": "obs_date",
    "SEH": "obs_date",
}


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, request_id: str = "req-1"):
        self.content = body
        self.status_code = status
        self.headers = {"X-Request-Id": request_id}

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


class TableTransport:
    def __init__(self, pages: dict[str, list[bytes]]):
        self.pages = {table: list(values) for table, values in pages.items()}
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, *, params: dict[str, str], timeout: float):
        table = Path(url).name.removesuffix(".json")
        self.calls.append((url, dict(params), timeout))
        return FakeResponse(self.pages[table].pop(0), request_id=f"req-{len(self.calls)}")


def _clock(*values: datetime):
    sequence = iter(values)
    return lambda: next(sequence)


def _body(columns: list[dict[str, str]], rows: list[list[object]], cursor=None) -> bytes:
    return json.dumps(
        {
            "datatable": {"data": rows, "columns": columns},
            "meta": {"next_cursor_id": cursor},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _columns(table: str) -> list[dict[str, str]]:
    date_column = DATE_COLUMNS.get(table)
    if date_column is None:
        return [{"name": "ticker", "type": "String"}]
    return [
        {"name": "ticker", "type": "String"},
        {"name": date_column, "type": "Date"},
        {"name": "value", "type": "BigDecimal"},
    ]


def _full_pages() -> dict[str, list[bytes]]:
    result: dict[str, list[bytes]] = {}
    for table in SUPPORTED_TABLES:
        columns = _columns(table)
        rows: list[list[object]] = []
        if table in DATE_COLUMNS:
            rows = [
                ["AAPL", "2020-12-31", 2.0],
                ["AAPL", "2020-01-01", 1.0],
            ]
        result[table] = [_body(columns, rows)]
    return result


def _capture(
    *,
    mode: str = "historical-sample",
    tables: tuple[str, ...] = ("ES",),
    pages: dict[str, list[bytes]] | None = None,
    start_time: datetime = BASE_TIME,
):
    transport = TableTransport(pages or {
        "ES": [_body(_columns("ES"), [["AAPL", "2020-01-01", 1.0]])]
    })
    times = [start_time + timedelta(seconds=index) for index in range(len(tables) + 1)]
    snapshot = ZacksTablesClient(
        get=transport,
        clock=_clock(*times),
        environ={"NASDAQ_DATA_LINK_API_KEY": API_KEY},
    ).capture(
        mode=mode,
        candidate_id="pead-test",
        requested_start="2020-01-01",
        requested_end="2020-12-31",
        tables=tables,
        filters_by_table={table: {"ticker": "AAPL"} for table in tables},
        per_page=100,
    )
    return snapshot, transport


def _rehash(document: dict) -> str:
    document["artifact_hash"] = hashlib.sha256(
        canonical_json(document["payload"]).encode("utf-8")
    ).hexdigest()
    return canonical_json(document)


def test_cursor_pagination_preserves_raw_bodies_and_excludes_credentials():
    page_one = _body(
        _columns("ES"), [["AAPL", "2020-12-31", 2.0]], cursor="cursor-2"
    )
    page_two = _body(_columns("ES"), [["AAPL", "2020-01-01", 1.0]])
    transport = TableTransport({"ES": [page_one, page_two]})
    snapshot = ZacksTablesClient(
        get=transport,
        clock=_clock(
            BASE_TIME,
            BASE_TIME + timedelta(microseconds=500_000),
            BASE_TIME + timedelta(seconds=1),
        ),
        environ={"NASDAQ_DATA_LINK_API_KEY": API_KEY},
    ).capture(
        mode="historical-sample",
        candidate_id="pead-test",
        requested_start="2020-01-01",
        requested_end="2020-12-31",
        tables=["ES"],
        filters_by_table={"ES": {"ticker": "AAPL"}},
        per_page=1,
    )

    assert set(snapshot.payload["tables"]) == {"ZACKS/ES"}
    assert set(snapshot.payload["coverage"]["table_ranges"]) == {"ZACKS/ES"}
    assert snapshot.payload["evidence_class"] == "development_sample"
    table = snapshot.payload["tables"]["ZACKS/ES"]
    assert table["rows"] == [
        {"act_rpt_date": "2020-01-01", "ticker": "AAPL", "value": 1.0},
        {"act_rpt_date": "2020-12-31", "ticker": "AAPL", "value": 2.0},
    ]
    pages = table["provider_metadata"]["pages"]
    assert base64.b64decode(pages[0]["response_body_base64"]) == page_one
    assert base64.b64decode(pages[1]["response_body_base64"]) == page_two
    assert pages[0]["response_body_sha256"] == hashlib.sha256(page_one).hexdigest()
    assert pages[1]["response_body_sha256"] == hashlib.sha256(page_two).hexdigest()
    assert pages[0]["captured_at"] == "2026-07-13T14:00:00.500000Z"
    assert pages[1]["canonical_request"]["params"]["qopts.cursor_id"] == "cursor-2"
    assert all(call[1]["api_key"] == API_KEY for call in transport.calls)
    assert API_KEY not in snapshot.to_json()
    assert all("api_key" not in page["canonical_request"]["params"] for page in pages)


def test_historical_full_uses_exact_coverage_fields_and_full_table_codes():
    tables = tuple(SUPPORTED_TABLES)
    snapshot, _ = _capture(
        mode="historical-full", tables=tables, pages=_full_pages()
    )

    payload = snapshot.payload
    assert payload["evidence_class"] == "historical_replication"
    assert payload["coverage"]["full_window"] is True
    assert payload["coverage"]["blockers"] == []
    assert set(payload["tables"]) == set(CANONICAL_TABLES)
    assert payload["coverage"]["table_ranges"]["ZACKS/ES"]["date_columns"] == [
        "act_rpt_date"
    ]
    assert payload["coverage"]["table_ranges"]["ZACKS/SS"]["date_columns"] == [
        "act_rpt_date"
    ]
    assert payload["coverage"]["table_ranges"]["ZACKS/EEH"]["date_columns"] == [
        "obs_date"
    ]
    assert payload["coverage"]["table_ranges"]["ZACKS/SEH"]["date_columns"] == [
        "obs_date"
    ]
    for table in ("ZACKS/MT", "ZACKS/EA"):
        assert payload["coverage"]["table_ranges"][table] == {
            "date_columns": [], "min_date": None, "max_date": None, "row_count": 0
        }
    assert EarningsAnnouncementSnapshot.from_json(snapshot.to_json()) == snapshot


def test_prospective_signal_is_always_nonqualifying():
    snapshot, _ = _capture(
        mode="prospective", tables=tuple(SUPPORTED_TABLES), pages=_full_pages()
    )
    assert snapshot.payload["evidence_class"] == "prospective_signal"
    assert snapshot.payload["coverage"]["full_window"] is False
    assert "prospective_window_not_complete" in snapshot.payload["coverage"]["blockers"]


def test_tampered_raw_body_and_derived_rows_are_rejected():
    snapshot, _ = _capture()
    raw_tamper = json.loads(snapshot.to_json())
    page = raw_tamper["payload"]["tables"]["ZACKS/ES"]["provider_metadata"][
        "pages"
    ][0]
    page["response_body_base64"] = base64.b64encode(b"{}").decode("ascii")
    with pytest.raises(SnapshotIntegrityError, match="response body hash mismatch"):
        EarningsAnnouncementSnapshot.from_json(_rehash(raw_tamper))

    row_tamper = json.loads(snapshot.to_json())
    row_tamper["payload"]["tables"]["ZACKS/ES"]["rows"][0]["value"] = 999
    with pytest.raises(SnapshotIntegrityError, match="preserved provider responses"):
        EarningsAnnouncementSnapshot.from_json(_rehash(row_tamper))

    alias_tamper = json.loads(snapshot.to_json())
    payload = alias_tamper["payload"]
    payload["tables"]["ES"] = payload["tables"].pop("ZACKS/ES")
    payload["coverage"]["table_ranges"]["ES"] = payload["coverage"][
        "table_ranges"
    ].pop("ZACKS/ES")
    with pytest.raises(SnapshotIntegrityError, match="snapshot table keys"):
        EarningsAnnouncementSnapshot.from_json(_rehash(alias_tamper))


def test_strict_inputs_provider_schema_and_clock_fail_closed():
    client = ZacksTablesClient(get=lambda *args, **kwargs: None, environ={})
    with pytest.raises(CredentialError, match="NASDAQ_DATA_LINK_API_KEY"):
        client.capture(
            mode="historical-sample",
            candidate_id="pead-test",
            requested_start="2020-01-01",
            requested_end="2020-12-31",
            tables=["ES"],
            filters_by_table={"ES": {"ticker": "AAPL"}},
        )
    with pytest.raises(ValueError, match="unsupported Zacks table"):
        _capture(tables=("SF1",))
    with pytest.raises(ValueError, match="controlled by the client"):
        canonical_query_params({"api_key": "operator-supplied"})

    reflected_secret = _body(
        _columns("ES"), [["AAPL", "2020-01-01", API_KEY]]
    )
    with pytest.raises(ProviderResponseError, match="exposed request credentials"):
        _capture(pages={"ES": [reflected_secret]})

    invalid = _body(_columns("ES"), [["AAPL", "2020-01-01", 1.0]])
    document = json.loads(invalid)
    document["meta"]["extra"] = True
    with pytest.raises(ProviderResponseError, match="provider meta requires exactly"):
        _capture(pages={"ES": [json.dumps(document).encode("utf-8")]})

    transport = TableTransport({"ES": [invalid]})
    with pytest.raises(ProviderResponseError, match="clock moved backwards"):
        ZacksTablesClient(
            get=transport,
            clock=_clock(BASE_TIME, BASE_TIME - timedelta(seconds=1)),
            environ={"NASDAQ_DATA_LINK_API_KEY": API_KEY},
        ).capture(
            mode="historical-sample",
            candidate_id="pead-test",
            requested_start="2020-01-01",
            requested_end="2020-12-31",
            tables=["ES"],
            filters_by_table={"ES": {"ticker": "AAPL"}},
        )


def test_store_is_create_only_idempotent_and_hash_chained(tmp_path):
    first, _ = _capture(start_time=BASE_TIME)
    second, _ = _capture(start_time=BASE_TIME + timedelta(hours=1))
    store = EarningsAnnouncementStore(
        tmp_path / "evidence", clock=lambda: BASE_TIME + timedelta(days=1)
    )

    first_result = store.persist(first)
    retry = store.persist(first)
    second_result = store.persist(second)
    assert retry == first_result
    events = store.verify_journal()
    assert len(events) == 2
    assert events[0]["previous_event_hash"] is None
    assert events[1]["previous_event_hash"] == first_result.journal_event_hash
    assert len(list(store.snapshots_directory.glob("*.json"))) == 2
    assert len(list(store.events_directory.glob("*.json"))) == 2

    event = json.loads(first_result.journal_event_path.read_text(encoding="utf-8"))
    event["payload"]["candidate_id"] = "tampered"
    first_result.journal_event_path.write_text(json.dumps(event), encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="journal event hash mismatch"):
        store.verify_journal()
    assert second_result.snapshot_path.exists()


def test_store_refuses_to_replace_a_conflicting_content_address(tmp_path):
    snapshot, _ = _capture()
    store = EarningsAnnouncementStore(tmp_path / "evidence", clock=lambda: BASE_TIME)
    result = store.persist(snapshot)
    result.snapshot_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError, match="refusing to overwrite"):
        store.persist(snapshot)


@pytest.mark.parametrize(
    ("mode", "expected_code", "evidence_class"),
    [
        ("historical-sample", 0, "development_sample"),
        ("historical-full", 1, "historical_replication"),
        ("prospective", 1, "prospective_signal"),
    ],
)
def test_cli_modes_persist_fail_closed_without_printing_secrets(
    tmp_path, capsys, mode, expected_code, evidence_class
):
    transport = TableTransport({
        "ES": [_body(_columns("ES"), [["AAPL", "2020-01-01", 1.0]])]
    })
    store_path = tmp_path / mode
    code = ingest_main(
        [
            mode,
            "--start", "2020-01-01",
            "--end", "2020-12-31",
            "--tables", "ES",
            "--ticker", "AAPL",
            "--store-dir", str(store_path),
        ],
        get=transport,
        clock=_clock(BASE_TIME, BASE_TIME + timedelta(seconds=1)),
        environ={"NASDAQ_DATA_LINK_API_KEY": API_KEY},
        store_clock=lambda: BASE_TIME + timedelta(days=1),
    )
    output = capsys.readouterr()
    assert code == expected_code
    assert "artifact_hash=" in output.out
    assert "snapshot_path=" in output.out
    assert API_KEY not in output.out + output.err
    artifact = EarningsAnnouncementSnapshot.from_json(
        next((store_path / "snapshots").glob("*.json")).read_text(encoding="utf-8")
    )
    assert artifact.payload["evidence_class"] == evidence_class
