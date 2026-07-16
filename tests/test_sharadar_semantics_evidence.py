from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

import data.sharadar_semantics_evidence as evidence_module
import scripts.acquire_sharadar_semantics as cli_module
from data.pead_event_universe import canonical_json, content_hash
from data.sharadar_semantics_evidence import (
    CANONICAL_REQUEST,
    INDICATORS_URL,
    SharadarSemanticsEvidenceError,
    build_sharadar_semantics_receipt,
    load_sharadar_semantics_receipt,
    publish_sharadar_semantics_receipt,
    validate_sharadar_semantics_receipt,
)


COLUMNS = (
    "table",
    "indicator",
    "isfilter",
    "isprimarykey",
    "title",
    "description",
    "unittype",
)
REQUIRED_ROWS = [
    [
        "SEP",
        "ticker",
        "Y",
        "Y",
        "Ticker Symbol",
        (
            "The ticker is a unique identifier for a security in the database. "
            "Where a company is delisted and the ticker subsequently recycled for use "
            "by a different company; we utilise that ticker for the currently active "
            "company and append a number to the ticker of the delisted company. The "
            "ACTIONS table provides a record of historical ticker changes."
        ),
        "text",
    ],
    [
        "SEP",
        "date",
        "Y",
        "Y",
        "Price Date",
        "The trade date of the price observations.",
        "date (YYYY-MM-DD)",
    ],
    [
        "SEP",
        "close",
        "N",
        "N",
        "Close Price - Split Adjusted",
        (
            "The official exchange close price; adjusted for stock splits and stock "
            "dividends. Not adjusted for cash dividends or spinoffs."
        ),
        "USD/share",
    ],
    [
        "SEP",
        "closeunadj",
        "N",
        "N",
        "Close Price - Unadjusted",
        (
            "The official exchange close price; not adjusted for stock splits; stock "
            "dividends; cash dividends or spinoffs."
        ),
        "USD/share",
    ],
]


def _provider_document() -> dict:
    return {
        "datatable": {
            "columns": [{"name": name, "type": "text"} for name in COLUMNS],
            "data": copy.deepcopy(REQUIRED_ROWS),
        },
        "meta": {"next_cursor_id": None},
    }


def _raw(document: dict | None = None) -> bytes:
    return json.dumps(
        _provider_document() if document is None else document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _published(tmp_path: Path) -> tuple[bytes, dict, Path]:
    raw = _raw()
    receipt, path = publish_sharadar_semantics_receipt(
        tmp_path,
        raw,
        candidate_id="pead-vq-source-qualification-v3",
        captured_at_utc="2026-07-14T12:34:56.123456Z",
    )
    return raw, receipt, path


def test_receipt_round_trips_exact_provider_bytes_and_canonical_receipt(tmp_path):
    raw, receipt, receipt_path = _published(tmp_path)

    assert receipt_path.name == f"{receipt['artifact_hash']}.json"
    assert receipt_path.read_bytes() == (canonical_json(receipt) + "\n").encode()
    raw_relative = receipt["payload"]["raw_artifact"]["relative_path"]
    assert (tmp_path / raw_relative).read_bytes() == raw
    assert load_sharadar_semantics_receipt(
        receipt_path, warehouse_dir=tmp_path
    ) == receipt
    assert receipt["payload"]["source"]["canonical_request"] == {
        "url": INDICATORS_URL,
        "params": {"qopts.per_page": "10000", "table": "SEP"},
    }
    assert "api_key" not in canonical_json(receipt)
    assert [row["indicator"] for row in receipt["payload"]["selected_semantics"]] == [
        "close",
        "closeunadj",
        "date",
        "ticker",
    ]
    assert receipt["payload"]["coverage"] == {
        "required_indicators": ["close", "closeunadj", "date", "ticker"],
        "present_indicators": ["close", "closeunadj", "date", "ticker"],
        "complete": True,
    }
    assert receipt["payload"]["qualification_allowed"] is True


def test_publish_is_idempotent_for_identical_content(tmp_path):
    raw, first, first_path = _published(tmp_path)

    second, second_path = publish_sharadar_semantics_receipt(
        tmp_path,
        raw,
        candidate_id="pead-vq-source-qualification-v3",
        captured_at_utc="2026-07-14T12:34:56.123456Z",
    )

    assert second == first
    assert second_path == first_path


def test_builder_does_not_trust_mutable_exported_request_constant():
    original = copy.deepcopy(CANONICAL_REQUEST)
    try:
        CANONICAL_REQUEST["params"]["api_key"] = "must-not-enter-artifact"
        CANONICAL_REQUEST["url"] = "https://attacker.invalid"
        receipt = build_sharadar_semantics_receipt(
            _raw(), candidate_id="candidate", captured_at_utc="2026-07-14T12:00:00Z"
        )
    finally:
        CANONICAL_REQUEST.clear()
        CANONICAL_REQUEST.update(original)

    request = receipt["payload"]["source"]["canonical_request"]
    assert request["url"] == INDICATORS_URL
    assert request["params"] == {"qopts.per_page": "10000", "table": "SEP"}


@pytest.mark.parametrize(
    ("candidate_id", "captured_at"),
    [
        ("", "2026-07-14T12:00:00Z"),
        (" leading", "2026-07-14T12:00:00Z"),
        ("trailing ", "2026-07-14T12:00:00Z"),
        ("line\nbreak", "2026-07-14T12:00:00Z"),
        ("candidate", "2026-07-14T12:00:00+00:00"),
        ("candidate", "2026-07-14T12:00Z"),
        ("candidate", "2026-07-14T08:00:00-04:00Z"),
    ],
)
def test_builder_rejects_noncanonical_identity_or_time(candidate_id, captured_at):
    with pytest.raises(SharadarSemanticsEvidenceError):
        build_sharadar_semantics_receipt(
            _raw(), candidate_id=candidate_id, captured_at_utc=captured_at
        )


def _extra_root(document):
    document["unexpected"] = True


def _datatable_extra(document):
    document["datatable"]["unexpected"] = True


def _meta_extra(document):
    document["meta"]["unexpected"] = True


def _not_final(document):
    document["meta"]["next_cursor_id"] = "cursor"


def _columns_reordered(document):
    document["datatable"]["columns"].reverse()


def _column_type_changed(document):
    document["datatable"]["columns"][0]["type"] = "String"


def _column_declaration_extra(document):
    document["datatable"]["columns"][0]["extra"] = None


def _row_too_short(document):
    document["datatable"]["data"][0].pop()


def _row_has_nontext(document):
    document["datatable"]["data"][0][2] = True


def _wrong_table(document):
    document["datatable"]["data"][0][0] = "SF1"


def _duplicate_indicator(document):
    document["datatable"]["data"].append(copy.deepcopy(document["datatable"]["data"][0]))


def _missing_required_indicator(document):
    document["datatable"]["data"] = document["datatable"]["data"][1:]


def _changed_semantics(document):
    document["datatable"]["data"][2][5] += " Changed."


@pytest.mark.parametrize(
    "mutation",
    [
        _extra_root,
        _datatable_extra,
        _meta_extra,
        _not_final,
        _columns_reordered,
        _column_type_changed,
        _column_declaration_extra,
        _row_too_short,
        _row_has_nontext,
        _wrong_table,
        _duplicate_indicator,
        _missing_required_indicator,
        _changed_semantics,
    ],
)
def test_provider_contract_fails_closed_on_shape_or_semantic_drift(mutation):
    document = _provider_document()
    mutation(document)

    with pytest.raises(SharadarSemanticsEvidenceError):
        build_sharadar_semantics_receipt(
            _raw(document),
            candidate_id="candidate",
            captured_at_utc="2026-07-14T12:00:00Z",
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff",
        b"[]",
        b'{"datatable":{},"datatable":{},"meta":{"next_cursor_id":null}}',
        b'{"datatable":NaN,"meta":{"next_cursor_id":null}}',
    ],
)
def test_provider_json_must_be_bounded_utf8_duplicate_free_and_finite(raw):
    with pytest.raises(SharadarSemanticsEvidenceError):
        build_sharadar_semantics_receipt(
            raw,
            candidate_id="candidate",
            captured_at_utc="2026-07-14T12:00:00Z",
        )


def test_provider_json_rejects_deep_non_object_input_as_evidence_error():
    raw = ("[" * 2_000 + "]" * 2_000).encode()

    with pytest.raises(SharadarSemanticsEvidenceError):
        build_sharadar_semantics_receipt(
            raw,
            candidate_id="candidate",
            captured_at_utc="2026-07-14T12:00:00Z",
        )


def test_provider_response_size_is_bounded(monkeypatch):
    raw = _raw()
    monkeypatch.setattr(evidence_module, "MAX_RAW_RESPONSE_BYTES", len(raw) - 1)

    with pytest.raises(SharadarSemanticsEvidenceError, match="size"):
        build_sharadar_semantics_receipt(
            raw,
            candidate_id="candidate",
            captured_at_utc="2026-07-14T12:00:00Z",
        )


def test_validator_rebuilds_claimed_semantics_from_raw_bytes(tmp_path):
    _, receipt, _ = _published(tmp_path)
    forged = copy.deepcopy(receipt)
    forged["payload"]["selected_semantics"][0]["description"] = "plausible forgery"
    forged["artifact_hash"] = content_hash(forged["payload"])

    with pytest.raises(SharadarSemanticsEvidenceError, match="does not replay"):
        validate_sharadar_semantics_receipt(forged, warehouse_dir=tmp_path)


def test_validator_rejects_mutated_raw_artifact(tmp_path):
    _, receipt, _ = _published(tmp_path)
    raw_path = tmp_path / receipt["payload"]["raw_artifact"]["relative_path"]
    raw_path.write_bytes(_raw() + b" ")

    with pytest.raises(SharadarSemanticsEvidenceError, match="identity"):
        validate_sharadar_semantics_receipt(receipt, warehouse_dir=tmp_path)


def test_validator_rejects_path_escape_even_with_rehashed_payload(tmp_path):
    _, receipt, _ = _published(tmp_path)
    forged = copy.deepcopy(receipt)
    forged["payload"]["raw_artifact"]["relative_path"] = "../outside.json"
    forged["artifact_hash"] = content_hash(forged["payload"])

    with pytest.raises(SharadarSemanticsEvidenceError, match="escapes"):
        validate_sharadar_semantics_receipt(forged, warehouse_dir=tmp_path)


def test_validator_and_loader_reject_symlinks(tmp_path):
    _, receipt, receipt_path = _published(tmp_path)
    raw_path = tmp_path / receipt["payload"]["raw_artifact"]["relative_path"]
    raw_copy = tmp_path / "raw-copy.json"
    raw_copy.write_bytes(raw_path.read_bytes())
    raw_path.unlink()
    raw_path.symlink_to(raw_copy)

    with pytest.raises(SharadarSemanticsEvidenceError, match="unsafe"):
        validate_sharadar_semantics_receipt(receipt, warehouse_dir=tmp_path)

    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(receipt_path)
    with pytest.raises(SharadarSemanticsEvidenceError, match="regular file"):
        load_sharadar_semantics_receipt(receipt_link, warehouse_dir=tmp_path)


@pytest.mark.parametrize(
    "render",
    [
        lambda receipt: json.dumps(receipt, indent=2).encode(),
        lambda receipt: (canonical_json(receipt) + "\n\n").encode(),
        lambda receipt: b'{"artifact_hash":"a","artifact_hash":"b","payload":{}}',
    ],
)
def test_loader_requires_duplicate_free_canonical_receipt_bytes(tmp_path, render):
    _, receipt, _ = _published(tmp_path)
    path = tmp_path / "noncanonical.json"
    path.write_bytes(render(receipt))

    with pytest.raises(SharadarSemanticsEvidenceError):
        load_sharadar_semantics_receipt(path, warehouse_dir=tmp_path)


def test_publish_rejects_content_addressed_collision(tmp_path):
    raw = _raw()
    receipt = build_sharadar_semantics_receipt(
        raw,
        candidate_id="candidate",
        captured_at_utc="2026-07-14T12:00:00Z",
    )
    destination = tmp_path / receipt["payload"]["raw_artifact"]["relative_path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"different")

    with pytest.raises(SharadarSemanticsEvidenceError, match="refusing to overwrite"):
        publish_sharadar_semantics_receipt(
            tmp_path,
            raw,
            candidate_id="candidate",
            captured_at_utc="2026-07-14T12:00:00Z",
        )


def test_publish_rejects_nested_symlink_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "source_snapshots").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SharadarSemanticsEvidenceError, match="parent is unsafe"):
        publish_sharadar_semantics_receipt(
            tmp_path,
            _raw(),
            candidate_id="candidate",
            captured_at_utc="2026-07-14T12:00:00Z",
        )
    assert list(outside.iterdir()) == []


class _Response:
    def __init__(self, chunks, *, status_code=200, stream_error=None):
        self.chunks = chunks
        self.status_code = status_code
        self.stream_error = stream_error
        self.closed = False

    def iter_content(self, *, chunk_size):
        assert chunk_size == 64 * 1024
        if self.stream_error is not None:
            raise self.stream_error
        return iter(self.chunks)

    def close(self):
        self.closed = True


def test_cli_fetches_exact_bytes_and_publishes_without_persisting_or_logging_key(
    tmp_path, capsys
):
    secret = "request-only-secret-123"
    raw = _raw()
    response = _Response([raw[:31], b"", raw[31:]])
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path), "--candidate-id", "candidate"],
        get=get,
        clock=lambda: datetime(
            2026, 7, 14, 8, 30, tzinfo=timezone(timedelta(hours=-4))
        ),
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )
    captured = capsys.readouterr()

    assert result == 0
    assert response.closed is True
    assert calls == [
        (
            INDICATORS_URL,
            {
                "params": {
                    "qopts.per_page": "10000",
                    "table": "SEP",
                    "api_key": secret,
                },
                "timeout": 60.0,
                "stream": True,
                "allow_redirects": False,
            },
        )
    ]
    assert "captured_at_utc=2026-07-14T12:30:00Z" in captured.out
    assert secret not in captured.out + captured.err
    persisted = [path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()]
    assert raw in persisted
    assert all(secret.encode() not in value for value in persisted)


def test_cli_capture_timestamp_is_taken_after_response_is_closed(tmp_path):
    secret = "request-only-secret-123"
    response = _Response([_raw()])

    def clock():
        assert response.closed is True
        return datetime(2026, 7, 14, 12, 30, tzinfo=timezone.utc)

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)],
        get=lambda *args, **kwargs: response,
        clock=clock,
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )

    assert result == 0


@pytest.mark.parametrize("argument", ["candidate", "warehouse"])
def test_cli_rejects_credential_reuse_in_artifact_arguments(
    tmp_path, capsys, argument
):
    secret = "request-only-secret-123"
    argv = ["--warehouse-dir", str(tmp_path), "--candidate-id", "candidate"]
    if argument == "candidate":
        argv[-1] = secret
    else:
        argv[1] = str(tmp_path / secret)

    result = cli_module.main(
        argv,
        get=lambda *args, **kwargs: pytest.fail("must not request"),
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )
    captured = capsys.readouterr()

    assert result == 1
    assert secret not in captured.out + captured.err


@pytest.mark.parametrize("status_code", [301, 302, 307, 401, 429, 500])
def test_cli_refuses_redirects_and_non_success_without_logging_provider_details(
    tmp_path, capsys, status_code
):
    secret = "request-only-secret-123"
    response = _Response([f"provider echoed {secret}".encode()], status_code=status_code)

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)],
        get=lambda *args, **kwargs: response,
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )
    captured = capsys.readouterr()

    assert result == 1
    assert response.closed is True
    assert secret not in captured.out + captured.err
    assert list(tmp_path.rglob("*.json")) == []


def test_cli_suppresses_network_and_stream_exception_details(tmp_path, capsys):
    secret = "request-only-secret-123"

    def failing_get(*args, **kwargs):
        raise RuntimeError(f"failed URL ?api_key={secret}")

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)],
        get=failing_get,
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )
    first = capsys.readouterr()
    assert result == 1
    assert secret not in first.out + first.err

    response = _Response([], stream_error=RuntimeError(f"stream URL ?api_key={secret}"))
    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)],
        get=lambda *args, **kwargs: response,
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )
    second = capsys.readouterr()
    assert result == 1
    assert response.closed is True
    assert secret not in second.out + second.err


def test_cli_refuses_response_that_reflects_credential(tmp_path, capsys):
    secret = "request-only-secret-123"
    raw = _raw().replace(b"Price Date", secret.encode())

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)],
        get=lambda *args, **kwargs: _Response([raw]),
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )
    captured = capsys.readouterr()

    assert result == 1
    assert secret not in captured.out + captured.err
    assert list(tmp_path.rglob("*.json")) == []


def test_cli_bounds_stream_before_publication(tmp_path, capsys, monkeypatch):
    secret = "request-only-secret-123"
    monkeypatch.setattr(cli_module, "MAX_RAW_RESPONSE_BYTES", 10)
    response = _Response([b"123456", b"78901"])

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)],
        get=lambda *args, **kwargs: response,
        environ={"NASDAQ_DATA_LINK_API_KEY": secret},
    )

    assert result == 1
    assert secret not in capsys.readouterr().err
    assert list(tmp_path.rglob("*.json")) == []


def test_cli_requires_environment_credential_before_network(tmp_path, capsys):
    called = False

    def get(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not request")

    result = cli_module.main(
        ["--warehouse-dir", str(tmp_path)], get=get, environ={}
    )

    assert result == 1
    assert called is False
    assert "NASDAQ_DATA_LINK_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), 601])
def test_fetch_rejects_unsafe_timeout_without_request(timeout):
    with pytest.raises(cli_module.SharadarSemanticsAcquisitionError, match="timeout"):
        cli_module.fetch_sharadar_semantics(
            credential="request-only-secret-123",
            get=lambda *args, **kwargs: pytest.fail("must not request"),
            timeout=timeout,
        )


@pytest.mark.parametrize("timeout", [True, "60", None])
def test_fetch_rejects_non_numeric_timeout_without_request(timeout):
    with pytest.raises(cli_module.SharadarSemanticsAcquisitionError, match="timeout"):
        cli_module.fetch_sharadar_semantics(
            credential="request-only-secret-123",
            get=lambda *args, **kwargs: pytest.fail("must not request"),
            timeout=timeout,
        )


@pytest.mark.parametrize("credential", [None, b"secret-123", "short", "space key 123"])
def test_fetch_rejects_noncanonical_credential_without_request(credential):
    with pytest.raises(cli_module.SharadarSemanticsAcquisitionError, match="credential"):
        cli_module.fetch_sharadar_semantics(
            credential=credential,
            get=lambda *args, **kwargs: pytest.fail("must not request"),
            timeout=60,
        )
