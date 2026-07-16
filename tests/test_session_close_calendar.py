from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from data.session_close_calendar import (
    EXTRACTION_METHOD,
    SessionCloseCalendarError,
    build_session_close_source_receipt,
    canonical_json,
    content_hash,
    extract_early_close_dates,
    load_session_close_calendar_evidence,
    normalized_html_text,
)
from scripts.ingest_nyse_session_calendar import (
    NyseCalendarAcquisitionError,
    acquire_nyse_session_calendar_sources,
)


SOURCE_ID = "ice-test-2024"
SOURCE_URL = (
    "https://ir.theice.com/press/news-details/2024/"
    "NYSE-test-calendar/default.aspx"
)
OBSERVED_SESSION_RULE = (
    "SEP-observed sessions use the listed 13:00 early close when present and "
    "otherwise the NYSE 16:00 core-session close; dates absent from SEP receive "
    "no inferred session."
)
EARLY_CLOSE_HTML = b"""<!doctype html>
<html><head>
<style>Friday, December 31, 2099</style>
<script>Monday, January 1, 2099</script>
</head><body>
<p>Each market will close early at 1:00 p.m. on Wednesday, July 3, 2024
and Friday, November 29, 2024.</p>
<p>NYSE Group Markets holidays include Thursday, July 4, 2024.</p>
</body></html>"""


def _calendar_document() -> dict:
    payload = {
        "schema_version": "nyse_session_close_calendar.v1",
        "venue": "NYSE cash equities",
        "timezone": "America/New_York",
        "coverage": {"start": "2024-01-01", "end": "2024-12-31"},
        "regular_close_local_time": "16:00:00",
        "early_close_local_time": "13:00:00",
        "observed_session_rule": OBSERVED_SESSION_RULE,
        "early_close_sessions": [
            {"date": "2024-07-03", "source_id": SOURCE_ID},
            {"date": "2024-11-29", "source_id": SOURCE_ID},
        ],
        "sources": [
            {
                "source_id": SOURCE_ID,
                "publisher": "Intercontinental Exchange / NYSE Group",
                "url": SOURCE_URL,
                "covered_years": [2024],
            }
        ],
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _write_calendar(path: Path) -> dict:
    document = _calendar_document()
    path.write_text(canonical_json(document) + "\n", encoding="utf-8")
    return document


def _source_metadata() -> dict:
    return {
        SOURCE_ID: {
            "retrieved_at_utc": "2026-07-13T22:01:00Z",
            "http": {
                "status_code": 200,
                "date_utc": "2026-07-13T22:00:00Z",
                "content_type": "text/html; charset=utf-8",
                "etag": '"calendar-v1"',
                "last_modified_utc": "2026-07-01T12:30:00Z",
            },
        }
    }


def _publish_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    calendar_path = tmp_path / "calendar.json"
    receipt_path = tmp_path / "nyse_session_close_sources" / "receipt.json"
    _write_calendar(calendar_path)
    document = build_session_close_source_receipt(
        calendar_path,
        {SOURCE_ID: EARLY_CLOSE_HTML},
        _source_metadata(),
        created_at_utc="2026-07-13T22:02:00Z",
    )
    digest = hashlib.sha256(EARLY_CLOSE_HTML).hexdigest()
    raw_path = receipt_path.parent / "raw" / f"{digest}.html"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(EARLY_CLOSE_HTML)
    receipt_path.write_text(canonical_json(document) + "\n", encoding="utf-8")
    return calendar_path, receipt_path, raw_path, document


def test_visible_text_and_early_close_extraction_are_scoped_and_deterministic():
    text = normalized_html_text(EARLY_CLOSE_HTML)

    assert "December 31, 2099" not in text
    assert "January 1, 2099" not in text
    assert extract_early_close_dates(EARLY_CLOSE_HTML) == [
        "2024-07-03",
        "2024-11-29",
    ]


def test_strict_source_receipt_revalidates_exact_raw_bytes_and_extraction(tmp_path):
    calendar_path, receipt_path, _, document = _publish_source_fixture(tmp_path)

    evidence = load_session_close_calendar_evidence(
        calendar_path=calendar_path,
        receipt_path=receipt_path,
    )

    assert evidence["source_receipt"] == document
    assert base64.b64decode(evidence["source_documents"][SOURCE_ID]) == (
        EARLY_CLOSE_HTML
    )
    entry = document["payload"]["sources"][0]
    assert entry["extraction"] == {
        "method": EXTRACTION_METHOD,
        "normalized_text_sha256": hashlib.sha256(
            normalized_html_text(EARLY_CLOSE_HTML).encode("utf-8")
        ).hexdigest(),
        "early_close_dates": ["2024-07-03", "2024-11-29"],
    }
    assert entry["http"] == _source_metadata()[SOURCE_ID]["http"]


def test_source_evidence_rejects_raw_receipt_and_duplicate_key_tampering(tmp_path):
    calendar_path, receipt_path, raw_path, document = _publish_source_fixture(
        tmp_path
    )
    raw_path.write_bytes(EARLY_CLOSE_HTML.replace(b"July 3", b"July 2"))
    with pytest.raises(SessionCloseCalendarError, match="raw document mismatch"):
        load_session_close_calendar_evidence(calendar_path, receipt_path)

    raw_path.write_bytes(EARLY_CLOSE_HTML)
    tampered = json.loads(json.dumps(document))
    tampered["payload"]["sources"][0]["http"]["etag"] = '"forged"'
    receipt_path.write_text(canonical_json(tampered), encoding="utf-8")
    with pytest.raises(SessionCloseCalendarError, match="artifact hash mismatch"):
        load_session_close_calendar_evidence(calendar_path, receipt_path)

    receipt_path.write_text(
        '{"artifact_hash":"a","artifact_hash":"b","payload":{}}',
        encoding="utf-8",
    )
    with pytest.raises(SessionCloseCalendarError, match="duplicate key"):
        load_session_close_calendar_evidence(calendar_path, receipt_path)


def test_source_evidence_rejects_self_consistent_impossible_http_chronology(
    tmp_path,
):
    calendar_path, receipt_path, _, document = _publish_source_fixture(tmp_path)
    tampered = json.loads(json.dumps(document))
    tampered["payload"]["sources"][0]["http"]["date_utc"] = (
        "2099-01-01T00:00:00Z"
    )
    tampered["artifact_hash"] = content_hash(tampered["payload"])
    receipt_path.write_text(canonical_json(tampered) + "\n", encoding="utf-8")

    with pytest.raises(SessionCloseCalendarError, match="clock skew"):
        load_session_close_calendar_evidence(calendar_path, receipt_path)

    metadata = _source_metadata()
    metadata[SOURCE_ID]["http"]["last_modified_utc"] = "2099-01-02T00:00:00Z"
    with pytest.raises(SessionCloseCalendarError, match="Last-Modified"):
        build_session_close_source_receipt(
            calendar_path,
            {SOURCE_ID: EARLY_CLOSE_HTML},
            metadata,
            created_at_utc="2026-07-13T22:02:00Z",
        )


class _Response:
    def __init__(
        self,
        *,
        url: str = SOURCE_URL,
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
        date: str | None = "Mon, 13 Jul 2026 22:00:00 GMT",
    ):
        self.url = url
        self.status_code = status_code
        self.content = EARLY_CLOSE_HTML
        self.history = []
        self.headers = {
            "Content-Type": content_type,
            "Date": date,
            "ETag": '"calendar-v1"',
            "Last-Modified": "Wed, 01 Jul 2026 12:30:00 GMT",
        }

    def raise_for_status(self) -> None:
        return None


def test_acquisition_archives_create_only_and_normalizes_http_metadata(tmp_path):
    calendar_path = tmp_path / "calendar.json"
    receipt_path = tmp_path / "sources" / "receipt.json"
    _write_calendar(calendar_path)
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    result = acquire_nyse_session_calendar_sources(
        calendar_path=calendar_path,
        receipt_path=receipt_path,
        http_get=get,
        created_at_utc="2026-07-13T22:01:00Z",
    )

    assert calls == [
        (
            SOURCE_URL,
            {
                "headers": {
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": "StockOptionsTrader-research-evidence/1.0",
                },
                "timeout": 60.0,
                "allow_redirects": True,
            },
        )
    ]
    entry = result["receipt"]["payload"]["sources"][0]
    assert entry["retrieved_at_utc"] == "2026-07-13T22:01:00Z"
    assert entry["http"] == _source_metadata()[SOURCE_ID]["http"]
    archive = receipt_path.parent / result["raw_archives"][SOURCE_ID]
    assert archive.read_bytes() == EARLY_CLOSE_HTML
    assert archive.name == f"{hashlib.sha256(EARLY_CLOSE_HTML).hexdigest()}.html"
    receipt_archive = receipt_path.parent / result["receipt_archive"]
    assert receipt_archive.read_text(encoding="utf-8") == (
        canonical_json(result["receipt"]) + "\n"
    )
    assert load_session_close_calendar_evidence(
        calendar_path, receipt_path
    ) == result["evidence"]

    # The same bytes and metadata are idempotent and cannot rewrite the raw
    # content-addressed source.
    before = archive.stat().st_ino
    acquire_nyse_session_calendar_sources(
        calendar_path=calendar_path,
        receipt_path=receipt_path,
        http_get=get,
        created_at_utc="2026-07-13T22:01:00Z",
    )
    assert archive.stat().st_ino == before


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_Response(status_code=204), "HTTP status"),
        (_Response(content_type="application/json"), "HTML content type"),
        (_Response(date="not-a-date"), "RFC date"),
        (_Response(date=None), "candidate-grade"),
        (
            _Response(
                url=(
                    "https://ir.theice.com/press/news-details/2024/"
                    "different/default.aspx"
                )
            ),
            "frozen host/path",
        ),
    ],
)
def test_acquisition_rejects_unqualified_http_evidence(tmp_path, response, message):
    calendar_path = tmp_path / "calendar.json"
    receipt_path = tmp_path / "sources" / "receipt.json"
    _write_calendar(calendar_path)

    with pytest.raises(NyseCalendarAcquisitionError, match=message):
        acquire_nyse_session_calendar_sources(
            calendar_path=calendar_path,
            receipt_path=receipt_path,
            http_get=lambda *_args, **_kwargs: response,
            created_at_utc="2026-07-13T22:01:00Z",
        )
    assert not receipt_path.exists()
