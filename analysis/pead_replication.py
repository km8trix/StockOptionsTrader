"""Locked independent-source PEAD replication.

This module deliberately does not know how to read Sharadar SF1 or EVENTS.
Its only signal input is one content-addressed Zacks snapshot containing
actual announcements and point-in-time estimate vintages.  A structurally
valid partial snapshot can be analysed as development evidence, but it can
never be reported as a completed replication.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analysis.pead_economic_returns import (
    EconomicReturnError,
    reconstruct_cash_return,
    validate_action_rows,
)
from analysis.research_stats import benjamini_hochberg
from data.pead_economic_evidence import (
    PeadEconomicEvidenceError,
    validate_cash_distribution_semantics,
    validate_terminal_settlement_ledger,
)
from scripts.factor_screen import factor_study


SCHEMA_VERSION = "zacks_pead_snapshot.v1"
REPORT_SCHEMA_VERSION = "pead_replication_report.v6"
COMBINED_DATA_SNAPSHOT_SCHEMA_VERSION = "pead_combined_data_snapshot.v4"
RESEARCH_MANIFEST_BINDING_SCHEMA_VERSION = "pead_research_manifest_binding.v1"
SESSION_CLOSE_CALENDAR_SCHEMA_VERSION = "nyse_session_close_calendar.v1"
SESSION_CLOSE_SOURCE_RECEIPT_SCHEMA_VERSION = "nyse_session_close_source_receipt.v1"
SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION = "nyse_session_close_evidence.v1"
SESSION_CLOSE_EXTRACTION_METHOD = "nyse_early_close_html_text.v1"
ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION = "pead_economic_return_inputs.v1"
CANDIDATE_ID = "pead-vq-locked-replication-v1"
SOURCE_ID = "nasdaq-data-link-zacks"
EASTERN = ZoneInfo("America/New_York")
WAREHOUSE_RETURN_TABLES = ("sep", "tickers", "daily", "actions")
_SESSION_CLOSE_RULE = (
    "SEP-observed sessions use the listed 13:00 early close when present and "
    "otherwise the NYSE 16:00 core-session close; dates absent from SEP receive "
    "no inferred session."
)

_HEX = frozenset("0123456789abcdef")
_MAX_SESSION_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_SESSION_SOURCE_CLOCK_SKEW_SECONDS = 10 * 60
_MAX_SESSION_SOURCE_RECEIPT_DURATION_SECONDS = 60 * 60
_MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}
_SESSION_SOURCE_DATE_PATTERN = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s*,?\s*"
    r"(" + "|".join(_MONTH_NUMBERS) + r")\s+(\d{1,2})\s*,\s*(\d{4})",
    re.IGNORECASE,
)
_SESSION_SOURCE_EARLY_BLOCK_PATTERN = re.compile(
    r"\*{0,4}\s*Each market will close early at 1:00 p\.m\."
    r"(.*?)"
    r"(?=\*{1,4}\s*Each market will close early at 1:00 p\.m\."
    r"|NYSE Group Markets holidays|Link to (?:NYSE|Holidays)"
    r"|About (?:NYSE|Intercontinental)|SOURCE:|$)",
    re.IGNORECASE,
)
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "source_id",
    "evidence_class",
    "captured_at",
    "requested_window",
    "coverage",
    "tables",
}
_WINDOW_FIELDS = {"start", "end"}
_COVERAGE_FIELDS = {"full_window", "table_ranges", "blockers"}
_TABLE_FIELDS = {
    "columns",
    "rows",
    "canonical_request",
    "response_sha256",
    "provider_metadata",
}
_ES_COLUMNS = {
    "m_ticker",
    "ticker",
    "currency_code",
    "per_end_date",
    "per_type",
    "act_rpt_date",
    "eps_mean_est",
    "eps_act",
    "eps_cnt_est",
    "eps_act_zacks_adj",
    "act_rpt_time",
    "act_rpt_code",
}
_EEH_COLUMNS = {
    "m_ticker",
    "ticker",
    "currency_code",
    "per_end_date",
    "per_type",
    "obs_date",
    "eps_mean_est",
    "eps_cnt_est",
}
_SS_COLUMNS = {
    "m_ticker",
    "ticker",
    "currency_code",
    "per_end_date",
    "per_type",
    "act_rpt_date",
    "sales_mean_est",
    "sales_act",
    "sales_cnt_est",
    "sales_act_zacks_adj",
    "act_rpt_time",
    "act_rpt_code",
}
_SEH_COLUMNS = {
    "m_ticker",
    "ticker",
    "currency_code",
    "per_end_date",
    "per_type",
    "obs_date",
    "sales_mean_est",
    "sales_cnt_est",
}
_MT_COLUMNS = {
    "m_ticker", "ticker", "comp_name", "exchange", "currency_code",
    "ticker_type", "active_ticker_flag", "mr_split_date", "mr_split_factor",
    "comp_cik", "country_code", "comp_type", "asset_type",
}
_EA_COLUMNS = {
    "m_ticker", "ticker", "exchange", "currency_code", "per_end_date_qr1",
    "eps_mean_est_qr1", "street_mean_est_qr1", "exp_rpt_date_qr1",
    "late_last_flag", "source_flag", "time_of_day_code", "time_of_day_desc",
    "per_end_date_qr0", "eps_act_qr0",
}


class PeadReplicationError(ValueError):
    """A snapshot or replication request is malformed."""


def canonical_json(value: Any) -> str:
    """Canonical finite JSON used for snapshot and report identities."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PeadReplicationError("PEAD evidence cannot contain non-finite JSON")
            return 0.0 if item == 0.0 else item
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PeadReplicationError("PEAD evidence keys must be strings")
                normalized[key] = normalize(child)
            return {key: normalized[key] for key in sorted(normalized)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        raise PeadReplicationError(
            f"unsupported PEAD evidence value: {type(item).__name__}"
        )

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validated_warehouse_return_snapshot(value: Any) -> dict[str, Any]:
    """Verify the exact ``PitWarehouse.snapshot_version`` receipt.

    The warehouse version is independently recomputed from its table manifest;
    this prevents a provider adapter from presenting a self-inconsistent receipt
    as the return-data identity.
    """
    snapshot = _exact_fields(
        value,
        {"version", "tables", "complete", "quality_flags"},
        "warehouse return snapshot",
    )
    version = _sha256(snapshot["version"], "warehouse return snapshot.version")
    if type(snapshot["complete"]) is not bool:
        raise PeadReplicationError("warehouse return snapshot.complete must be boolean")
    tables = snapshot["tables"]
    if not isinstance(tables, list):
        raise PeadReplicationError("warehouse return snapshot.tables must be an array")
    manifest: list[dict[str, Any]] = []
    for index, raw_item in enumerate(tables):
        item = _exact_fields(
            raw_item,
            {"table", "sha256", "bytes"},
            f"warehouse return snapshot.tables[{index}]",
        )
        table = _text(item["table"], f"warehouse return snapshot.tables[{index}].table")
        if table not in WAREHOUSE_RETURN_TABLES:
            raise PeadReplicationError(
                f"warehouse return snapshot contains unexpected table {table!r}"
            )
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, Integral) or int(size) < 0:
            raise PeadReplicationError(
                f"warehouse return snapshot.tables[{index}].bytes must be "
                "a non-negative integer"
            )
        manifest.append(
            {
                "table": table,
                "sha256": _sha256(
                    item["sha256"],
                    f"warehouse return snapshot.tables[{index}].sha256",
                ),
                "bytes": int(size),
            }
        )
    names = [item["table"] for item in manifest]
    if len(names) != len(set(names)):
        raise PeadReplicationError("warehouse return snapshot contains duplicate tables")
    present = set(names)
    allowed_manifest_orders = [
        [table for table in WAREHOUSE_RETURN_TABLES if table in present],
        [table for table in sorted(WAREHOUSE_RETURN_TABLES) if table in present],
    ]
    if names not in allowed_manifest_orders:
        raise PeadReplicationError(
            "warehouse return snapshot table order differs from PitWarehouse"
        )

    flags = snapshot["quality_flags"]
    if not isinstance(flags, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in flags
    ):
        raise PeadReplicationError(
            "warehouse return snapshot.quality_flags must be an array of trimmed strings"
        )
    missing = sorted(set(WAREHOUSE_RETURN_TABLES) - set(names))
    expected_flags = [f"missing_table:{table}" for table in missing]
    if flags != expected_flags:
        raise PeadReplicationError(
            "warehouse return snapshot quality flags do not match its manifest"
        )
    if snapshot["complete"] is not (not missing):
        raise PeadReplicationError(
            "warehouse return snapshot completeness does not match its manifest"
        )

    digest = hashlib.sha256()
    for item in manifest:
        digest.update(
            f"{item['table']}:{item['sha256']}:{item['bytes']}\n".encode("utf-8")
        )
    if digest.hexdigest() != version:
        raise PeadReplicationError("warehouse return snapshot version mismatch")
    return {
        "version": version,
        "tables": manifest,
        "complete": snapshot["complete"],
        "quality_flags": list(flags),
    }


def _validated_corporate_action_evidence(
    value: Any, *, required_start: str | None = None, required_end: str | None = None
) -> dict[str, Any]:
    """Validate the research-boundary receipt for the ACTIONS acquisition."""
    outer = _exact_fields(
        value, {"artifact_hash", "payload"}, "corporate action evidence"
    )
    payload = _exact_fields(
        outer["payload"],
        {
            "schema_version",
            "acquisition_artifact_hash",
            "source_snapshot_time",
            "parquet_sha256",
            "raw_zip_sha256",
            "row_count",
            "min_date",
            "max_date",
            "required_window",
            "complete",
            "blockers",
            "value_is_terminal_payout_per_share",
        },
        "corporate action evidence.payload",
    )
    if payload["schema_version"] != "sharadar_actions_evidence.v1":
        raise PeadReplicationError("unsupported corporate action evidence schema")
    for field in (
        "acquisition_artifact_hash", "parquet_sha256", "raw_zip_sha256"
    ):
        _sha256(payload[field], f"corporate action evidence.{field}")
    _source_utc_timestamp(
        payload["source_snapshot_time"],
        "corporate action source snapshot time",
    )
    row_count = payload["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, Integral) or row_count <= 0:
        raise PeadReplicationError("corporate action row_count must be positive")
    min_date = _iso_date(payload["min_date"], "corporate action min_date")
    max_date = _iso_date(payload["max_date"], "corporate action max_date")
    if min_date > max_date:
        raise PeadReplicationError("corporate action date range is reversed")
    window = _exact_fields(
        payload["required_window"], {"start", "end"},
        "corporate action evidence.required_window",
    )
    window_start = _iso_date(window["start"], "corporate action required start")
    window_end = _iso_date(window["end"], "corporate action required end")
    if window_start > window_end:
        raise PeadReplicationError("corporate action required window is reversed")
    if required_start is not None and window["start"] != required_start:
        raise PeadReplicationError("corporate action required start changed")
    if required_end is not None and window["end"] != required_end:
        raise PeadReplicationError("corporate action required end changed")
    blockers = payload["blockers"]
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in blockers
    ):
        raise PeadReplicationError("corporate action blockers must be trimmed strings")
    expected_blockers = []
    if min_date > window_start:
        expected_blockers.append("actions_range_starts_after_required_window")
    if max_date < window_end:
        expected_blockers.append("actions_range_ends_before_required_window")
    if blockers != expected_blockers:
        raise PeadReplicationError(
            "corporate action blockers do not match the observed date range"
        )
    expected_complete = not expected_blockers
    if (
        type(payload["complete"]) is not bool
        or payload["complete"] is not expected_complete
    ):
        raise PeadReplicationError(
            "corporate action completeness does not match the observed date range"
        )
    if payload["value_is_terminal_payout_per_share"] is not False:
        raise PeadReplicationError("corporate action value semantics are unsafe")
    claimed = _sha256(outer["artifact_hash"], "corporate action evidence hash")
    if claimed != content_hash(payload):
        raise PeadReplicationError("corporate action evidence hash mismatch")
    return {"artifact_hash": claimed, "payload": dict(payload)}


class _PrimaryVisibleTextParser(HTMLParser):
    """Minimal independent visible-text parser for archived official pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0 and data.strip():
            self.parts.append(data)


def _primary_session_source_text(raw: bytes, path: str) -> str:
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadReplicationError(f"{path} must be UTF-8 HTML") from exc
    parser = _PrimaryVisibleTextParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as exc:
        raise PeadReplicationError(f"{path} contains malformed HTML") from exc
    normalized = " ".join(" ".join(parser.parts).split())
    if not normalized:
        raise PeadReplicationError(f"{path} contains no visible text")
    return normalized


def _primary_session_source_early_dates(text: str, path: str) -> list[str]:
    dates: set[str] = set()
    for block in _SESSION_SOURCE_EARLY_BLOCK_PATTERN.findall(text):
        for month_name, day_text, year_text in _SESSION_SOURCE_DATE_PATTERN.findall(
            block
        ):
            month = next(
                number
                for name, number in _MONTH_NUMBERS.items()
                if name.lower() == month_name.lower()
            )
            try:
                value = date(int(year_text), month, int(day_text))
            except ValueError as exc:
                raise PeadReplicationError(
                    f"{path} contains an invalid early-close date"
                ) from exc
            dates.add(value.isoformat())
    return sorted(dates)


def _optional_canonical_utc(value: Any, path: str) -> str | None:
    return None if value is None else _utc_timestamp(value, path)


def _canonical_base64(value: Any, path: str) -> bytes:
    value = _text(value, path)
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise PeadReplicationError(f"{path} must be canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != value:
        raise PeadReplicationError(f"{path} must be canonical base64")
    if not raw or len(raw) > _MAX_SESSION_SOURCE_BYTES:
        raise PeadReplicationError(f"{path} has an invalid decoded size")
    return raw


def _validated_session_close_calendar(
    value: Any,
    *,
    required_start: str | None = None,
    required_end: str | None = None,
) -> dict[str, Any]:
    """Validate the content-addressed official NYSE close-time calendar."""
    outer = _exact_fields(
        value, {"artifact_hash", "payload"}, "session close calendar"
    )
    payload = _exact_fields(
        outer["payload"],
        {
            "schema_version",
            "venue",
            "timezone",
            "coverage",
            "regular_close_local_time",
            "early_close_local_time",
            "observed_session_rule",
            "early_close_sessions",
            "sources",
        },
        "session close calendar.payload",
    )
    if payload["schema_version"] != SESSION_CLOSE_CALENDAR_SCHEMA_VERSION:
        raise PeadReplicationError("unsupported session close calendar schema")
    if payload["venue"] != "NYSE cash equities":
        raise PeadReplicationError("session close calendar venue changed")
    if payload["timezone"] != "America/New_York":
        raise PeadReplicationError("session close calendar timezone changed")
    if payload["regular_close_local_time"] != "16:00:00":
        raise PeadReplicationError("session close calendar regular close changed")
    if payload["early_close_local_time"] != "13:00:00":
        raise PeadReplicationError("session close calendar early close changed")
    if payload["observed_session_rule"] != _SESSION_CLOSE_RULE:
        raise PeadReplicationError("session close calendar observed-session rule changed")

    coverage = _exact_fields(
        payload["coverage"], {"start", "end"}, "session close calendar.coverage"
    )
    coverage_start = _iso_date(coverage["start"], "session close calendar start")
    coverage_end = _iso_date(coverage["end"], "session close calendar end")
    if coverage_start > coverage_end:
        raise PeadReplicationError("session close calendar coverage is reversed")
    if (required_start is None) is not (required_end is None):
        raise PeadReplicationError("session close window must provide both endpoints")
    requested_start = (
        coverage_start
        if required_start is None
        else _iso_date(required_start, "required session close start")
    )
    requested_end = (
        coverage_end
        if required_end is None
        else _iso_date(required_end, "required session close end")
    )
    if requested_start > requested_end:
        raise PeadReplicationError("required session close window is reversed")
    if coverage_start > requested_start or coverage_end < requested_end:
        raise PeadReplicationError("session close calendar does not cover required window")

    sources = payload["sources"]
    if not isinstance(sources, list) or not sources:
        raise PeadReplicationError("session close calendar sources must be nonempty")
    normalized_sources: list[dict[str, Any]] = []
    for index, raw_source in enumerate(sources):
        source = _exact_fields(
            raw_source,
            {"source_id", "publisher", "url", "covered_years"},
            f"session close calendar.sources[{index}]",
        )
        source_id = _text(
            source["source_id"], f"session close calendar.sources[{index}].source_id"
        )
        publisher = _text(
            source["publisher"], f"session close calendar.sources[{index}].publisher"
        )
        url = _text(source["url"], f"session close calendar.sources[{index}].url")
        if not (
            url.startswith("https://ir.theice.com/")
            or url.startswith("https://www.nyse.com/")
        ):
            raise PeadReplicationError("session close calendar source is not ICE/NYSE")
        years = source["covered_years"]
        if (
            not isinstance(years, list)
            or not years
            or any(
                isinstance(year, bool) or not isinstance(year, Integral)
                for year in years
            )
        ):
            raise PeadReplicationError("session close calendar source years are invalid")
        normalized_years = [int(year) for year in years]
        if normalized_years != sorted(set(normalized_years)):
            raise PeadReplicationError(
                "session close calendar source years must be unique and sorted"
            )
        normalized_sources.append(
            {
                "source_id": source_id,
                "publisher": publisher,
                "url": url,
                "covered_years": normalized_years,
            }
        )
    source_ids = [source["source_id"] for source in normalized_sources]
    if source_ids != sorted(set(source_ids)):
        raise PeadReplicationError(
            "session close calendar sources must be uniquely sorted by source_id"
        )
    source_by_id = {source["source_id"]: source for source in normalized_sources}
    all_covered_years = {
        year for source in normalized_sources for year in source["covered_years"]
    }
    expected_years = set(range(coverage_start.year, coverage_end.year + 1))
    if not expected_years.issubset(all_covered_years):
        raise PeadReplicationError("session close calendar source-year coverage is incomplete")

    early_rows = payload["early_close_sessions"]
    if not isinstance(early_rows, list):
        raise PeadReplicationError("session close calendar early closes must be an array")
    normalized_early: list[dict[str, str]] = []
    for index, raw_row in enumerate(early_rows):
        row = _exact_fields(
            raw_row,
            {"date", "source_id"},
            f"session close calendar.early_close_sessions[{index}]",
        )
        early_date = _iso_date(
            row["date"], f"session close calendar early date {index}"
        )
        source_id = _text(
            row["source_id"], f"session close calendar early source {index}"
        )
        source = source_by_id.get(source_id)
        if source is None or early_date.year not in source["covered_years"]:
            raise PeadReplicationError(
                "session close calendar early close lacks year-matching source"
            )
        if not coverage_start <= early_date <= coverage_end:
            raise PeadReplicationError("session close calendar early close is out of range")
        normalized_early.append(
            {"date": early_date.isoformat(), "source_id": source_id}
        )
    early_dates = [row["date"] for row in normalized_early]
    if early_dates != sorted(set(early_dates)):
        raise PeadReplicationError(
            "session close calendar early closes must be uniquely sorted"
        )

    claimed = _sha256(outer["artifact_hash"], "session close calendar hash")
    if claimed != content_hash(payload):
        raise PeadReplicationError("session close calendar hash mismatch")
    return {"artifact_hash": claimed, "payload": dict(payload)}


def _validated_session_close_evidence(
    value: Any,
    *,
    required_start: str | None = None,
    required_end: str | None = None,
) -> dict[str, Any]:
    """Independently prove calendar rows from archived official HTML bytes."""
    bundle = _exact_fields(
        value,
        {"calendar", "source_receipt", "source_documents"},
        "session close evidence bundle",
    )
    calendar = _validated_session_close_calendar(
        bundle["calendar"],
        required_start=required_start,
        required_end=required_end,
    )
    receipt_outer = _exact_fields(
        bundle["source_receipt"],
        {"artifact_hash", "payload"},
        "session close source receipt",
    )
    receipt_payload = _exact_fields(
        receipt_outer["payload"],
        {"schema_version", "calendar_artifact_hash", "created_at_utc", "sources"},
        "session close source receipt.payload",
    )
    if (
        receipt_payload["schema_version"]
        != SESSION_CLOSE_SOURCE_RECEIPT_SCHEMA_VERSION
    ):
        raise PeadReplicationError("unsupported session close source receipt schema")
    if (
        _sha256(
            receipt_payload["calendar_artifact_hash"],
            "session close receipt calendar hash",
        )
        != calendar["artifact_hash"]
    ):
        raise PeadReplicationError("session close receipt binds a different calendar")
    created_at = _utc_timestamp(
        receipt_payload["created_at_utc"], "session close receipt created_at_utc"
    )
    receipt_hash = _sha256(
        receipt_outer["artifact_hash"], "session close source receipt hash"
    )
    if receipt_hash != content_hash(receipt_payload):
        raise PeadReplicationError("session close source receipt hash mismatch")

    calendar_sources = calendar["payload"]["sources"]
    receipt_sources = receipt_payload["sources"]
    if not isinstance(receipt_sources, list) or len(receipt_sources) != len(
        calendar_sources
    ):
        raise PeadReplicationError(
            "session close receipt must exactly cover calendar sources"
        )
    documents = bundle["source_documents"]
    if not isinstance(documents, Mapping):
        raise PeadReplicationError("session close source documents must be an object")
    source_ids = [source["source_id"] for source in calendar_sources]
    if set(documents) != set(source_ids) or any(
        not isinstance(source_id, str) for source_id in documents
    ):
        raise PeadReplicationError(
            "session close source documents must exactly cover calendar source IDs"
        )

    extracted_by_source: dict[str, set[str]] = {}
    visible_text_by_source: dict[str, str] = {}
    created_dt = datetime.fromisoformat(created_at[:-1] + "+00:00")
    verified_receipt_sources: list[dict[str, Any]] = []
    for index, (calendar_source, raw_receipt_source) in enumerate(
        zip(calendar_sources, receipt_sources, strict=True)
    ):
        receipt_source = _exact_fields(
            raw_receipt_source,
            {
                "source_id", "publisher", "url", "covered_years",
                "retrieved_at_utc", "http", "raw_document", "extraction",
            },
            f"session close receipt.sources[{index}]",
        )
        for field in ("source_id", "publisher", "url", "covered_years"):
            if receipt_source[field] != calendar_source[field]:
                raise PeadReplicationError(
                    "session close receipt source differs from calendar source"
                )
        source_id = calendar_source["source_id"]
        retrieved_at = _utc_timestamp(
            receipt_source["retrieved_at_utc"],
            f"session close source {source_id} retrieved_at_utc",
        )
        retrieved_dt = datetime.fromisoformat(retrieved_at[:-1] + "+00:00")
        if retrieved_dt > created_dt:
            raise PeadReplicationError(
                "session close source retrieval occurs after receipt creation"
            )
        if (
            created_dt - retrieved_dt
        ).total_seconds() > _MAX_SESSION_SOURCE_RECEIPT_DURATION_SECONDS:
            raise PeadReplicationError(
                "session close source receipt duration is implausible"
            )
        http = _exact_fields(
            receipt_source["http"],
            {"status_code", "date_utc", "content_type", "etag", "last_modified_utc"},
            f"session close source {source_id}.http",
        )
        if type(http["status_code"]) is not int or http["status_code"] != 200:
            raise PeadReplicationError("session close source HTTP status is not 200")
        content_type = _text(
            http["content_type"], f"session close source {source_id} content_type"
        )
        if "html" not in content_type.lower():
            raise PeadReplicationError("session close source content type is not HTML")
        if http["etag"] is not None:
            _text(http["etag"], f"session close source {source_id} etag")
        if http["date_utc"] is None:
            raise PeadReplicationError("session close source HTTP Date is required")
        server_date = datetime.fromisoformat(
            _utc_timestamp(
                http["date_utc"],
                f"session close source {source_id} HTTP date",
            )[:-1]
            + "+00:00"
        )
        if abs(
            (server_date - retrieved_dt).total_seconds()
        ) > _MAX_SESSION_SOURCE_CLOCK_SKEW_SECONDS:
            raise PeadReplicationError(
                "session close source HTTP Date clock skew is implausible"
            )
        last_modified_value = _optional_canonical_utc(
            http["last_modified_utc"],
            f"session close source {source_id} HTTP last-modified",
        )
        if last_modified_value is not None and datetime.fromisoformat(
            last_modified_value[:-1] + "+00:00"
        ) > server_date:
            raise PeadReplicationError(
                "session close source HTTP Last-Modified follows response Date"
            )
        raw_document = _exact_fields(
            receipt_source["raw_document"],
            {"relative_path", "sha256", "bytes"},
            f"session close source {source_id}.raw_document",
        )
        raw_hash = _sha256(
            raw_document["sha256"], f"session close source {source_id} raw hash"
        )
        if raw_document["relative_path"] != f"raw/{raw_hash}.html":
            raise PeadReplicationError(
                "session close source path must be content-addressed"
            )
        raw_size = raw_document["bytes"]
        if (
            type(raw_size) is not int
            or raw_size <= 0
            or raw_size > _MAX_SESSION_SOURCE_BYTES
        ):
            raise PeadReplicationError("session close source byte count is invalid")
        raw = _canonical_base64(
            documents[source_id], f"session close source document {source_id}"
        )
        if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != raw_hash:
            raise PeadReplicationError("session close source raw document mismatch")
        text = _primary_session_source_text(raw, f"session close source {source_id}")
        visible_text_by_source[source_id] = text
        extraction = _exact_fields(
            receipt_source["extraction"],
            {"method", "normalized_text_sha256", "early_close_dates"},
            f"session close source {source_id}.extraction",
        )
        if extraction["method"] != SESSION_CLOSE_EXTRACTION_METHOD:
            raise PeadReplicationError("session close extraction method changed")
        if _sha256(
            extraction["normalized_text_sha256"],
            f"session close source {source_id} normalized text hash",
        ) != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise PeadReplicationError("session close normalized text hash mismatch")
        dates = extraction["early_close_dates"]
        if not isinstance(dates, list) or any(
            not isinstance(value, str) for value in dates
        ):
            raise PeadReplicationError("session close extracted dates are invalid")
        validated_dates = [
            _iso_date(value, f"session close source {source_id} extracted date").isoformat()
            for value in dates
        ]
        independently_extracted = _primary_session_source_early_dates(
            text, f"session close source {source_id}"
        )
        if (
            validated_dates != sorted(set(validated_dates))
            or validated_dates != independently_extracted
        ):
            raise PeadReplicationError(
                "session close extracted dates differ from official source bytes"
            )
        extracted_by_source[source_id] = set(independently_extracted)
        verified_receipt_sources.append(dict(receipt_source))

    core_text = visible_text_by_source.get("nyse-core-hours", "")
    if re.search(
        r"Core Trading Session\s*:?\s*9:30 a\.m\.\s*to\s*4:00 p\.m\.\s*ET",
        core_text,
        re.IGNORECASE,
    ) is None:
        raise PeadReplicationError(
            "NYSE source bytes do not prove the regular 16:00 ET close"
        )

    coverage = calendar["payload"]["coverage"]
    coverage_start = date.fromisoformat(coverage["start"])
    coverage_end = date.fromisoformat(coverage["end"])
    expected_dates: set[str] = set()
    source_by_id = {
        source["source_id"]: source for source in calendar_sources
    }
    for source_id, source_dates in extracted_by_source.items():
        covered_years = set(source_by_id[source_id]["covered_years"])
        expected_dates.update(
            value
            for value in source_dates
            if date.fromisoformat(value).year in covered_years
            and coverage_start <= date.fromisoformat(value) <= coverage_end
        )
    calendar_rows = calendar["payload"]["early_close_sessions"]
    actual_dates = {row["date"] for row in calendar_rows}
    if actual_dates != expected_dates:
        raise PeadReplicationError(
            "calendar early closes differ from archived official source dates"
        )
    for row in calendar_rows:
        if row["date"] not in extracted_by_source[row["source_id"]]:
            raise PeadReplicationError(
                "calendar early close is not proved by its named source"
            )

    normalized_receipt_payload = {
        **dict(receipt_payload),
        "sources": verified_receipt_sources,
    }
    receipt = {
        "artifact_hash": receipt_hash,
        "payload": normalized_receipt_payload,
    }
    evidence_payload = {
        "schema_version": SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION,
        "calendar": calendar,
        "source_receipt": receipt,
    }
    return {
        "artifact_hash": content_hash(evidence_payload),
        "payload": evidence_payload,
    }


def _session_close_schedule(
    evidence: Mapping[str, Any],
    sessions: Sequence[Any],
    *,
    required_start: str,
    required_end: str,
) -> dict[pd.Timestamp, datetime]:
    """Derive exact UTC closes for observed sessions and fail on any gap."""
    validated_evidence = _validated_session_close_evidence(
        evidence, required_start=required_start, required_end=required_end
    )
    calendar = validated_evidence["payload"]["calendar"]
    normalized: list[pd.Timestamp] = []
    for raw_session in sessions:
        session = pd.Timestamp(raw_session)
        if pd.isna(session) or session.tzinfo is not None or session != session.normalize():
            raise PeadReplicationError(
                "observed market sessions must be unique timezone-naive dates"
            )
        normalized.append(session)
    if normalized != sorted(set(normalized)):
        raise PeadReplicationError(
            "observed market sessions must be unique and strictly sorted"
        )
    lo = _iso_date(required_start, "required session close start")
    hi = _iso_date(required_end, "required session close end")
    if any(not lo <= session.date() <= hi for session in normalized):
        raise PeadReplicationError("observed session falls outside requested close window")
    observed_dates = {session.date() for session in normalized}
    early_dates = {
        date.fromisoformat(row["date"])
        for row in calendar["payload"]["early_close_sessions"]
        if lo <= date.fromisoformat(row["date"]) <= hi
    }
    missing_early = sorted(early_dates - observed_dates)
    if missing_early:
        raise PeadReplicationError(
            "official early-close sessions are absent from observed SEP sessions: "
            + ",".join(value.isoformat() for value in missing_early)
        )
    result: dict[pd.Timestamp, datetime] = {}
    for session in normalized:
        close_time = time(13, 0) if session.date() in early_dates else time(16, 0)
        result[session] = datetime.combine(
            session.date(), close_time, tzinfo=EASTERN
        ).astimezone(timezone.utc)
    return result


def build_combined_data_snapshot(
    zacks_snapshot_hash: str,
    warehouse_return_snapshot: Mapping[str, Any],
    corporate_action_evidence: Mapping[str, Any],
    session_close_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Content-address all independent signal and warehouse return inputs."""
    payload = {
        "schema_version": COMBINED_DATA_SNAPSHOT_SCHEMA_VERSION,
        "zacks_snapshot_hash": _sha256(
            zacks_snapshot_hash, "combined data snapshot Zacks hash"
        ),
        "warehouse_return_snapshot": _validated_warehouse_return_snapshot(
            warehouse_return_snapshot
        ),
        "corporate_action_evidence": _validated_corporate_action_evidence(
            corporate_action_evidence
        ),
        "session_close_evidence": _validated_session_close_evidence(
            session_close_evidence
        ),
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def build_economic_return_inputs(
    combined_data_snapshot: Mapping[str, Any],
    cash_distribution_semantics: Mapping[str, Any],
    terminal_settlement_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind return-policy evidence to the exact multi-table data snapshot.

    The semantics document is intentionally nonqualifying today.  Binding it
    here still matters: it prevents a candidate dividend interpretation from
    floating across asynchronously refreshed SEP, TICKERS, and ACTIONS files.
    """
    combined = _exact_fields(
        combined_data_snapshot,
        {"artifact_hash", "payload"},
        "combined data snapshot",
    )
    combined_hash = _sha256(
        combined["artifact_hash"], "economic return combined snapshot hash"
    )
    if content_hash(combined["payload"]) != combined_hash:
        raise PeadReplicationError("combined data snapshot hash mismatch")
    try:
        semantics = validate_cash_distribution_semantics(
            cash_distribution_semantics
        )
        terminal = validate_terminal_settlement_ledger(
            terminal_settlement_ledger
        )
    except PeadEconomicEvidenceError as exc:
        raise PeadReplicationError(
            "economic return input evidence is invalid"
        ) from exc
    payload = {
        "schema_version": ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "combined_data_snapshot_hash": combined_hash,
        "cash_distribution_semantics": semantics,
        "terminal_settlement_ledger": terminal,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def build_research_manifest_binding(
    *,
    candidate_file_name: str,
    candidate_file_sha256: str,
    candidate_schema_version: str,
    source_file_name: str,
    source_file_sha256: str,
    source_schema_version: str,
) -> dict[str, Any]:
    """Bind the exact candidate specification and source manifest bytes."""
    payload = {
        "schema_version": RESEARCH_MANIFEST_BINDING_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "source_id": SOURCE_ID,
        "candidate_specification": {
            "file_name": _text(candidate_file_name, "candidate manifest file name"),
            "file_sha256": _sha256(
                candidate_file_sha256, "candidate manifest file hash"
            ),
            "schema_version": _text(
                candidate_schema_version, "candidate manifest schema version"
            ),
        },
        "source_manifest": {
            "file_name": _text(source_file_name, "source manifest file name"),
            "file_sha256": _sha256(source_file_sha256, "source manifest file hash"),
            "schema_version": _text(
                source_schema_version, "source manifest schema version"
            ),
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _validated_research_manifest_binding(value: Any) -> dict[str, Any]:
    outer = _exact_fields(
        value, {"artifact_hash", "payload"}, "research manifest binding"
    )
    payload = _exact_fields(
        outer["payload"],
        {
            "schema_version",
            "candidate_id",
            "source_id",
            "candidate_specification",
            "source_manifest",
        },
        "research manifest binding.payload",
    )
    if payload["schema_version"] != RESEARCH_MANIFEST_BINDING_SCHEMA_VERSION:
        raise PeadReplicationError("unsupported research manifest binding schema")
    if payload["candidate_id"] != CANDIDATE_ID or payload["source_id"] != SOURCE_ID:
        raise PeadReplicationError("research manifest binding belongs to another target")
    values: dict[str, dict[str, str]] = {}
    for name in ("candidate_specification", "source_manifest"):
        item = _exact_fields(
            payload[name],
            {"file_name", "file_sha256", "schema_version"},
            f"research manifest binding.payload.{name}",
        )
        values[name] = {
            "file_name": _text(item["file_name"], f"{name}.file_name"),
            "file_sha256": _sha256(item["file_sha256"], f"{name}.file_sha256"),
            "schema_version": _text(
                item["schema_version"], f"{name}.schema_version"
            ),
        }
    normalized_payload = {
        "schema_version": payload["schema_version"],
        "candidate_id": payload["candidate_id"],
        "source_id": payload["source_id"],
        **values,
    }
    claimed = _sha256(outer["artifact_hash"], "research manifest binding hash")
    if claimed != content_hash(normalized_payload):
        raise PeadReplicationError("research manifest binding hash mismatch")
    return {"artifact_hash": claimed, "payload": normalized_payload}


def _capture_warehouse_return_snapshot(
    provider: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    snapshot_version = getattr(provider, "snapshot_version", None)
    if not callable(snapshot_version):
        return None, "warehouse_return_snapshot_unavailable"
    try:
        raw = snapshot_version(list(WAREHOUSE_RETURN_TABLES))
    except Exception:
        return None, "warehouse_return_snapshot_unavailable"
    try:
        return _validated_warehouse_return_snapshot(raw), None
    except (PeadReplicationError, TypeError, ValueError):
        return None, "warehouse_return_snapshot_invalid"


def _capture_corporate_action_evidence(
    provider: Any, *, start: str, end: str
) -> tuple[dict[str, Any] | None, str | None]:
    reader = getattr(provider, "corporate_action_evidence", None)
    if not callable(reader):
        return None, "corporate_action_evidence_unavailable"
    try:
        raw = reader(start, end)
    except Exception:
        return None, "corporate_action_evidence_unavailable"
    try:
        return _validated_corporate_action_evidence(
            raw, required_start=start, required_end=end
        ), None
    except (PeadReplicationError, TypeError, ValueError):
        return None, "corporate_action_evidence_invalid"


def _exact_fields(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PeadReplicationError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        raise PeadReplicationError(
            f"{path} has invalid fields "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    return dict(value)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PeadReplicationError(f"{path} must be a nonempty trimmed string")
    return value


def _sha256(value: Any, path: str) -> str:
    candidate = _text(value, path).lower()
    if len(candidate) != 64 or any(character not in _HEX for character in candidate):
        raise PeadReplicationError(f"{path} must be a SHA-256 digest")
    if candidate != value:
        raise PeadReplicationError(f"{path} must use lowercase hex")
    return candidate


def _iso_date(value: Any, path: str) -> date:
    if not isinstance(value, str):
        raise PeadReplicationError(f"{path} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadReplicationError(f"{path} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise PeadReplicationError(f"{path} must be a canonical ISO date")
    return parsed


def _utc_timestamp(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadReplicationError(f"{path} must be canonical UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeadReplicationError(f"{path} must be a UTC timestamp") from exc
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if canonical != value:
        raise PeadReplicationError(f"{path} must be canonical UTC: {canonical}")
    return value


def _source_utc_timestamp(value: Any, path: str) -> str:
    """Validate a provider timestamp that explicitly identifies UTC.

    Nasdaq receipts use a space-delimited ``UTC`` suffix, while other valid
    provider representations use ``Z`` or ``+00:00``.  Parse the value instead
    of merely checking that it is non-empty, and reject naive/non-UTC offsets.
    """
    value = _text(value, path)
    candidate = value.replace(" UTC", "+00:00")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise PeadReplicationError(f"{path} must be an ISO/UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
    ):
        raise PeadReplicationError(f"{path} must identify UTC")
    return value


def _lifecycle_date(value: Any, path: str) -> date:
    """Accept only the warehouse date contract or canonical ISO equivalent."""
    if type(value) is date:
        return value
    return _iso_date(value, path)


def _contains_secret(value: Any) -> bool:
    secret_tokens = {"apikey", "api_key", "token", "secret", "authorization"}
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in secret_tokens or _contains_secret(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret(child) for child in value)
    return False


def _validate_table(value: Any, table_code: str) -> dict[str, Any]:
    table = _exact_fields(value, _TABLE_FIELDS, f"payload.tables[{table_code!r}]")
    columns = table["columns"]
    if not isinstance(columns, list) or not columns:
        raise PeadReplicationError(f"{table_code}.columns must be a nonempty array")
    column_names: list[str] = []
    for index, descriptor in enumerate(columns):
        item = _exact_fields(
            descriptor, {"name", "type"}, f"{table_code}.columns[{index}]"
        )
        column_names.append(_text(item["name"], f"{table_code}.columns[{index}].name"))
        _text(item["type"], f"{table_code}.columns[{index}].type")
    if len(column_names) != len(set(column_names)):
        raise PeadReplicationError(f"{table_code}.columns contains duplicates")
    rows = table["rows"]
    if not isinstance(rows, list):
        raise PeadReplicationError(f"{table_code}.rows must be an array")
    expected = set(column_names)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise PeadReplicationError(
                f"{table_code}.rows[{index}] must be an exact columns-keyed object"
            )
        canonical_json(row)
    if not isinstance(table["canonical_request"], Mapping):
        raise PeadReplicationError(f"{table_code}.canonical_request must be an object")
    if _contains_secret(table["canonical_request"]):
        raise PeadReplicationError(f"{table_code}.canonical_request contains a secret")
    _sha256(table["response_sha256"], f"{table_code}.response_sha256")
    if not isinstance(table["provider_metadata"], Mapping):
        raise PeadReplicationError(f"{table_code}.provider_metadata must be an object")
    if _contains_secret(table["provider_metadata"]):
        raise PeadReplicationError(f"{table_code}.provider_metadata contains a secret")
    canonical_json(table["canonical_request"])
    canonical_json(table["provider_metadata"])
    return table


@dataclass(frozen=True)
class ValidatedSnapshot:
    artifact_hash: str
    payload: Mapping[str, Any]
    requested_start: date
    requested_end: date
    coverage_blockers: tuple[str, ...]

    @property
    def full_window(self) -> bool:
        return not self.coverage_blockers


def _range_covers(value: Any, start: date, end: date) -> bool:
    """Understand the frozen range envelope without trusting its claim alone."""
    if not isinstance(value, Mapping):
        return False
    minimum = value.get("min_date", value.get("start"))
    maximum = value.get("max_date", value.get("end"))
    if minimum is None or maximum is None:
        dates = value.get("date_columns")
        if isinstance(dates, Mapping):
            candidates = [item for item in dates.values() if isinstance(item, Mapping)]
            if candidates:
                minimum = min(
                    (item.get("min_date", item.get("start")) for item in candidates),
                    default=None,
                )
                maximum = max(
                    (item.get("max_date", item.get("end")) for item in candidates),
                    default=None,
                )
    try:
        return _iso_date(minimum, "table range minimum") <= start and _iso_date(
            maximum, "table range maximum"
        ) >= end
    except PeadReplicationError:
        return False


def validate_snapshot_document(
    document: Mapping[str, Any], *, start: str, end: str
) -> ValidatedSnapshot:
    """Verify the content address and strict outer snapshot schema."""
    outer = _exact_fields(document, {"artifact_hash", "payload"}, "snapshot")
    claimed = _sha256(outer["artifact_hash"], "snapshot.artifact_hash")
    payload = _exact_fields(outer["payload"], _PAYLOAD_FIELDS, "snapshot.payload")
    actual = content_hash(payload)
    if claimed != actual:
        raise PeadReplicationError("snapshot artifact hash mismatch")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise PeadReplicationError(f"snapshot schema must be {SCHEMA_VERSION}")
    if payload["candidate_id"] != CANDIDATE_ID:
        raise PeadReplicationError("snapshot belongs to a different candidate")
    if payload["source_id"] != SOURCE_ID:
        raise PeadReplicationError("snapshot source is not Nasdaq Data Link Zacks")
    _text(payload["evidence_class"], "payload.evidence_class")
    _utc_timestamp(payload["captured_at"], "payload.captured_at")

    expected_start = _iso_date(start, "start")
    expected_end = _iso_date(end, "end")
    if expected_start > expected_end:
        raise PeadReplicationError("start cannot be after end")
    window = _exact_fields(payload["requested_window"], _WINDOW_FIELDS, "requested_window")
    source_start = _iso_date(window["start"], "requested_window.start")
    source_end = _iso_date(window["end"], "requested_window.end")
    if source_start > source_end:
        raise PeadReplicationError("snapshot requested window start cannot exceed end")

    coverage = _exact_fields(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    if type(coverage["full_window"]) is not bool:
        raise PeadReplicationError("coverage.full_window must be boolean")
    if not isinstance(coverage["table_ranges"], Mapping):
        raise PeadReplicationError("coverage.table_ranges must be an object")
    blockers = coverage["blockers"]
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item.strip() for item in blockers
    ):
        raise PeadReplicationError("coverage.blockers must be an array of strings")

    tables = payload["tables"]
    if not isinstance(tables, Mapping):
        raise PeadReplicationError("payload.tables must be an object")
    validated_tables = {
        code: _validate_table(value, code) for code, value in tables.items()
    }
    for code, required in (("ZACKS/ES", _ES_COLUMNS), ("ZACKS/EEH", _EEH_COLUMNS)):
        if code not in validated_tables:
            raise PeadReplicationError(f"snapshot is missing required table {code}")
        column_names = {item["name"] for item in validated_tables[code]["columns"]}
        missing = sorted(required - column_names)
        if missing:
            raise PeadReplicationError(f"{code} is missing required columns: {missing}")
    optional_pairs = (
        ("ZACKS/SS", _SS_COLUMNS),
        ("ZACKS/SEH", _SEH_COLUMNS),
        ("ZACKS/MT", _MT_COLUMNS),
        ("ZACKS/EA", _EA_COLUMNS),
    )
    for code, required in optional_pairs:
        if code in validated_tables:
            column_names = {item["name"] for item in validated_tables[code]["columns"]}
            missing = sorted(required - column_names)
            if missing:
                raise PeadReplicationError(f"{code} is missing required columns: {missing}")
    if ("ZACKS/SS" in validated_tables) != ("ZACKS/SEH" in validated_tables):
        raise PeadReplicationError("sales evidence requires both ZACKS/SS and ZACKS/SEH")

    reasons = list(blockers)
    if source_start != expected_start or source_end != expected_end:
        reasons.append("snapshot_requested_window_mismatch")
    if coverage["full_window"] is not True:
        reasons.append("source_coverage_not_full_window")
    for code in ("ZACKS/ES", "ZACKS/EEH"):
        if not _range_covers(
            coverage["table_ranges"].get(code), expected_start, expected_end
        ):
            reasons.append(f"{code}_range_does_not_cover_frozen_window")
    return ValidatedSnapshot(
        artifact_hash=claimed,
        payload={**payload, "tables": validated_tables},
        requested_start=source_start,
        requested_end=source_end,
        coverage_blockers=tuple(sorted(set(reasons))),
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _analyst_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value) if int(value) >= 2 else None


def _row_date(row: Mapping[str, Any], field: str) -> date | None:
    try:
        return _iso_date(row.get(field), field)
    except PeadReplicationError:
        return None


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    m_ticker = row.get("m_ticker")
    period = row.get("per_end_date")
    period_type = row.get("per_type")
    if not all(isinstance(value, str) and value.strip() for value in (m_ticker, period, period_type)):
        return None
    if _row_date(row, "per_end_date") is None:
        return None
    return str(m_ticker).strip().upper(), str(period), str(period_type).strip().upper()


def _announcement_timestamp(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    report_date = _row_date(row, "act_rpt_date")
    raw_time = row.get("act_rpt_time")
    code = row.get("act_rpt_code")
    if report_date is None or not isinstance(raw_time, str) or not isinstance(code, str):
        return None, "missing_or_invalid_report_timestamp"
    parsed_time = None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(raw_time, fmt).time()
            break
        except ValueError:
            continue
    if parsed_time is None:
        return None, "invalid_report_time"
    upper = code.strip().upper()
    if upper not in {"BTO", "DTM", "AMC"}:
        return None, "invalid_report_code"
    minute = parsed_time.hour * 60 + parsed_time.minute
    consistent = (
        (upper == "BTO" and minute < 9 * 60 + 30)
        or (upper == "DTM" and 9 * 60 + 30 <= minute < 16 * 60)
        or (upper == "AMC" and minute >= 16 * 60)
    )
    if not consistent:
        return None, "report_time_code_mismatch"
    local = datetime.combine(report_date, parsed_time, tzinfo=EASTERN)
    return local.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    ), None


def _normalize_pair(
    actual_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    *,
    value_prefix: str,
    consensus_abs_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Match one surprise table to its latest strictly prior estimate vintage."""
    actual_field = f"{value_prefix}_act"
    adjustment_field = f"{value_prefix}_act_zacks_adj"
    mean_field = f"{value_prefix}_mean_est"
    count_field = f"{value_prefix}_cnt_est"
    history: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in history_rows:
        key = _event_key(row)
        if key is not None:
            history.setdefault(key, []).append(row)
    actual_groups: dict[tuple[str, str, str] | None, list[Mapping[str, Any]]] = {}
    for row in actual_rows:
        actual_groups.setdefault(_event_key(row), []).append(row)

    matched: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for key in sorted(actual_groups, key=lambda item: canonical_json(item)):
        rows = actual_groups[key]
        reasons: list[str] = []
        representative = rows[0]
        if key is None:
            reasons.append("invalid_event_key")
        if len(rows) != 1:
            reasons.append("duplicate_actual_key")
        if key is not None and key[2] != "Q":
            reasons.append("non_quarterly_period")
        report_date = _row_date(representative, "act_rpt_date")
        announcement_at, timestamp_error = _announcement_timestamp(representative)
        if timestamp_error:
            reasons.append(timestamp_error)
        currency = representative.get("currency_code")
        if not isinstance(currency, str) or not currency.strip():
            reasons.append("missing_currency")
        elif currency.strip().upper() != "USD":
            reasons.append("non_usd_currency")
        actual = _finite(representative.get(actual_field))
        es_mean = _finite(representative.get(mean_field))
        es_count = _analyst_count(representative.get(count_field))
        if actual is None:
            reasons.append("nonfinite_actual")
        if es_mean is None:
            reasons.append("nonfinite_surprise_table_consensus")
        if es_count is None:
            reasons.append("insufficient_surprise_table_analyst_count")

        selected = None
        if key is not None and report_date is not None:
            candidates = [
                row
                for row in history.get(key, [])
                if (_row_date(row, "obs_date") is not None)
                and _row_date(row, "obs_date") < report_date
            ]
            if candidates:
                latest_date = max(_row_date(row, "obs_date") for row in candidates)
                latest = [
                    row for row in candidates if _row_date(row, "obs_date") == latest_date
                ]
                if len(latest) == 1:
                    selected = latest[0]
                else:
                    reasons.append("duplicate_latest_consensus_vintage")
            else:
                reasons.append("missing_strictly_prior_consensus")
        else:
            reasons.append("invalid_actual_report_date")

        vintage_mean = vintage_count = None
        if selected is not None:
            vintage_mean = _finite(selected.get(mean_field))
            vintage_count = _analyst_count(selected.get(count_field))
            if vintage_mean is None:
                reasons.append("nonfinite_vintage_consensus")
            if vintage_count is None:
                reasons.append("insufficient_vintage_analyst_count")
            selected_currency = selected.get("currency_code")
            if not isinstance(selected_currency, str) or not isinstance(currency, str):
                reasons.append("consensus_currency_missing")
            elif selected_currency.strip().upper() != currency.strip().upper():
                reasons.append("consensus_currency_mismatch")
            selected_ticker = str(selected.get("ticker", "")).strip().upper()
            actual_ticker = str(representative.get("ticker", "")).strip().upper()
            if not selected_ticker or selected_ticker != actual_ticker:
                reasons.append("ticker_mapping_mismatch")
        if es_mean is not None and vintage_mean is not None:
            if abs(es_mean - vintage_mean) > consensus_abs_tolerance:
                reasons.append("surprise_consensus_crosscheck_mismatch")

        key_payload = (
            {"m_ticker": key[0], "per_end_date": key[1], "per_type": key[2]}
            if key is not None
            else {"m_ticker": representative.get("m_ticker"),
                  "per_end_date": representative.get("per_end_date"),
                  "per_type": representative.get("per_type")}
        )
        if reasons:
            exclusions.append(
                {
                    "event_key": key_payload,
                    "ticker": representative.get("ticker"),
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        assert selected is not None
        assert actual is not None and es_mean is not None and vintage_mean is not None
        assert es_count is not None and vintage_count is not None and announcement_at is not None
        adjustment = _finite(representative.get(adjustment_field))
        matched.append(
            {
                "event_key": key_payload,
                "ticker": str(representative["ticker"]).strip().upper(),
                "currency_code": str(currency).strip().upper(),
                "act_rpt_date": report_date.isoformat(),
                "announcement_at_utc": announcement_at,
                "act_rpt_time": representative["act_rpt_time"],
                "act_rpt_code": str(representative["act_rpt_code"]).strip().upper(),
                "actual": actual,
                "zacks_adjustment_diagnostic": adjustment,
                "consensus": vintage_mean,
                "consensus_obs_date": _row_date(selected, "obs_date").isoformat(),
                "consensus_analyst_count": vintage_count,
                "surprise_table_consensus": es_mean,
                "surprise_table_analyst_count": es_count,
                "consensus_crosscheck_absolute_difference": abs(es_mean - vintage_mean),
                "unscaled_forecast_error": actual - vintage_mean,
            }
        )
    counts = {
        "actual_rows": len(actual_rows),
        "consensus_vintage_rows": len(history_rows),
        "matched_events": len(matched),
        "excluded_actual_events": len(exclusions),
    }
    return matched, exclusions, counts


def normalize_source_events(
    snapshot: ValidatedSnapshot, *, consensus_abs_tolerance: float
) -> dict[str, Any]:
    """Normalize EPS and optional sales evidence, preserving every exclusion."""
    tolerance = float(consensus_abs_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise PeadReplicationError("consensus tolerance must be finite and non-negative")
    tables = snapshot.payload["tables"]
    eps, eps_exclusions, eps_counts = _normalize_pair(
        tables["ZACKS/ES"]["rows"],
        tables["ZACKS/EEH"]["rows"],
        value_prefix="eps",
        consensus_abs_tolerance=tolerance,
    )
    sales: list[dict[str, Any]] = []
    sales_exclusions: list[dict[str, Any]] = []
    sales_counts = {
        "actual_rows": 0,
        "consensus_vintage_rows": 0,
        "matched_events": 0,
        "excluded_actual_events": 0,
    }
    if "ZACKS/SS" in tables:
        sales, sales_exclusions, sales_counts = _normalize_pair(
            tables["ZACKS/SS"]["rows"],
            tables["ZACKS/SEH"]["rows"],
            value_prefix="sales",
            consensus_abs_tolerance=tolerance,
        )
    sales_by_key = {canonical_json(item["event_key"]): item for item in sales}
    for event in eps:
        sales_event = sales_by_key.get(canonical_json(event["event_key"]))
        event["sales_diagnostic"] = sales_event

    identity_diagnostics = {
        "available": "ZACKS/MT" in tables,
        "validated_events": 0,
        "invalid_events": [],
    }
    if "ZACKS/MT" in tables:
        mt_by_key: dict[str, list[Mapping[str, Any]]] = {}
        for row in tables["ZACKS/MT"]["rows"]:
            value = row.get("m_ticker")
            if isinstance(value, str) and value.strip():
                mt_by_key.setdefault(value.strip().upper(), []).append(row)
        for event in eps:
            m_ticker = event["event_key"]["m_ticker"]
            identities = mt_by_key.get(m_ticker, [])
            reasons: list[str] = []
            if len(identities) != 1:
                reasons.append("missing_or_duplicate_mt_identity")
            else:
                identity = identities[0]
                if str(identity.get("ticker", "")).strip().upper() != event["ticker"]:
                    reasons.append("mt_ticker_mismatch")
                if str(identity.get("currency_code", "")).strip().upper() != event[
                    "currency_code"
                ]:
                    reasons.append("mt_currency_mismatch")
                cik = identity.get("comp_cik")
                if not isinstance(cik, (str, int)) or not str(cik).strip():
                    reasons.append("mt_cik_missing")
            if reasons:
                identity_diagnostics["invalid_events"].append(
                    {"event_key": event["event_key"], "reasons": sorted(reasons)}
                )
            else:
                identity_diagnostics["validated_events"] += 1

    schedule_diagnostics = {
        "available": "ZACKS/EA" in tables,
        "actual_event_keys": len(eps),
        "matched_event_keys": 0,
    }
    if "ZACKS/EA" in tables:
        scheduled = {
            (str(row.get("m_ticker", "")).strip().upper(), row.get("per_end_date_qr1"))
            for row in tables["ZACKS/EA"]["rows"]
        }
        schedule_diagnostics["matched_event_keys"] = sum(
            (event["event_key"]["m_ticker"], event["event_key"]["per_end_date"])
            in scheduled
            for event in eps
        )
    return {
        "consensus_absolute_tolerance": tolerance,
        "eps_events": eps,
        "eps_exclusions": eps_exclusions,
        "eps_counts": eps_counts,
        "sales_events": sales,
        "sales_exclusions": sales_exclusions,
        "sales_counts": sales_counts,
        "sales_is_diagnostic_only": True,
        "stable_identity_diagnostics": identity_diagnostics,
        "announcement_schedule_diagnostics": schedule_diagnostics,
        "primary_signal": (
            "eps_forecast_error_scaled_by_split_normalized_preannouncement_close"
        ),
    }


def monthly_formation_dates(
    sessions: pd.DatetimeIndex, *, start: str, end: str
) -> pd.DatetimeIndex:
    """First actually observed warehouse session of each calendar month."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    observed = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values().unique()
    observed = observed[(observed >= lo) & (observed <= hi)]
    if observed.empty:
        return pd.DatetimeIndex([], name="date")
    frame = pd.DataFrame({"date": observed})
    first = frame.groupby(frame["date"].dt.to_period("M"), sort=True)["date"].min()
    return pd.DatetimeIndex(first.to_list(), name="date")


def _size_terciles(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    def rank(group: pd.DataFrame) -> pd.Series:
        caps = pd.to_numeric(group["mcap"], errors="coerce")
        valid = caps.notna() & (caps > 0)
        result = pd.Series(np.nan, index=group.index)
        if int(valid.sum()) >= 3:
            ranks = caps[valid].rank(method="first")
            result.loc[valid] = np.minimum(
                2, ((ranks - 1) * 3 / valid.sum()).astype(int)
            )
        return result

    events["tercile"] = events.groupby("date", group_keys=False).apply(
        rank, include_groups=False
    )
    return events


def _locked_factor_slice(
    events: pd.DataFrame, *, minimum_names_per_formation: int = 10
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen cell-size floor and stable signal tie-break.

    ``factor_study`` only consumes the factor's ordering.  Replacing ``sue``
    with this unique ordinal therefore leaves every non-tied portfolio
    unchanged while making ties resolve by ascending stable ``m_ticker``.
    Dates below the frozen cross-sectional floor never reach the portfolio
    constructor.
    """
    if type(minimum_names_per_formation) is not int or minimum_names_per_formation < 2:
        raise PeadReplicationError(
            "minimum_names_per_formation must be an integer of at least two"
        )
    if events.empty:
        empty = events.copy()
        empty["_pead_signal_order"] = pd.Series(dtype="int64")
        return empty, {
            "minimum_names_per_formation": minimum_names_per_formation,
            "formation_dates_before_floor": 0,
            "formation_dates_after_floor": 0,
            "excluded_formation_dates": [],
        }
    required = {"date", "sue", "m_ticker"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise PeadReplicationError(f"factor observations omit fields: {missing}")
    if events.duplicated(["date", "m_ticker"]).any():
        raise PeadReplicationError(
            "factor observations contain duplicate m_ticker on a formation date"
        )
    counts = events.groupby("date", sort=True)["m_ticker"].nunique()
    kept_dates = counts[counts >= minimum_names_per_formation].index
    excluded = counts[counts < minimum_names_per_formation]
    filtered = events[events["date"].isin(kept_dates)].copy()
    filtered = filtered.sort_values(
        ["date", "sue", "m_ticker"], kind="mergesort"
    )
    filtered["_pead_signal_order"] = (
        filtered.groupby("date", sort=False).cumcount().astype(int)
    )
    return filtered, {
        "minimum_names_per_formation": minimum_names_per_formation,
        "formation_dates_before_floor": int(len(counts)),
        "formation_dates_after_floor": int(len(kept_dates)),
        "excluded_formation_dates": [
            {"date": pd.Timestamp(value).date().isoformat(), "eligible_names": int(count)}
            for value, count in excluded.items()
        ],
    }


def _locked_horizon_portfolio(
    events: pd.DataFrame, *, horizon: int, quantile: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze legs before looking at a horizon's return availability.

    Selection is based on the complete entry-time cross-section and its stable
    signal order.  If a selected long or short constituent lacks the exact
    horizon return, the entire date×horizon cohort is withheld from inference;
    the remaining names are never reranked or resized around the missing name.
    """
    if type(horizon) is not int or horizon <= 0:
        raise PeadReplicationError("horizon must be a positive integer")
    if not 0 < float(quantile) <= 0.5:
        raise PeadReplicationError("quantile must be in (0, 0.5]")
    return_column = f"fwd_{horizon}"
    required = {"date", "name", "m_ticker", "_pead_signal_order", return_column}
    missing = sorted(required - set(events.columns))
    if missing:
        raise PeadReplicationError(f"horizon observations omit fields: {missing}")
    selected_frames: list[pd.DataFrame] = []
    exclusions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for formation, raw_group in events.groupby("date", sort=True):
        group = raw_group.sort_values("_pead_signal_order", kind="mergesort")
        count = len(group)
        leg_count = max(1, int(float(quantile) * count))
        low = group.iloc[:leg_count].copy()
        high = group.iloc[-leg_count:].copy()
        low["_pead_frozen_leg"] = "short"
        high["_pead_frozen_leg"] = "long"
        selected = pd.concat([low, high], ignore_index=False)
        unresolved = selected[selected[return_column].isna()]
        selection = {
            "date": pd.Timestamp(formation).date().isoformat(),
            "eligible_names": int(count),
            "names_per_leg": int(leg_count),
            "short_m_tickers": low["m_ticker"].tolist(),
            "long_m_tickers": high["m_ticker"].tolist(),
        }
        selections.append(selection)
        if not unresolved.empty:
            exclusions.append(
                {
                    **selection,
                    "reason": "selected_constituent_return_unresolved",
                    "unresolved_m_tickers": unresolved["m_ticker"].tolist(),
                }
            )
            continue
        selected_frames.append(selected)
    result = (
        pd.concat(selected_frames, ignore_index=False)
        if selected_frames
        else events.iloc[0:0].assign(
            _pead_frozen_leg=pd.Series(dtype="object")
        )
    )
    return result, {
        "horizon_sessions": horizon,
        "formation_dates_selected": len(selections),
        "formation_dates_admitted_to_inference": len(selected_frames),
        "frozen_selections": selections,
        "excluded_unresolved_cohorts": exclusions,
    }


def _price_panel(provider: Any, tickers: Sequence[str], start: str, end: str, field: str):
    bulk = getattr(provider, "prices_bulk", None)
    if bulk is not None:
        result: dict[str, pd.Series] = {}
        names = sorted(set(tickers))
        for offset in range(0, len(names), 250):
            result.update(bulk(names[offset : offset + 250], start, end, field=field))
        return result
    return {
        ticker: provider.prices(ticker, start, end, field=field)
        for ticker in sorted(set(tickers))
    }


def _validated_economic_return_inputs(value: Any) -> dict[str, Any]:
    outer = _exact_fields(
        value, {"artifact_hash", "payload"}, "economic return inputs"
    )
    payload = _exact_fields(
        outer["payload"],
        {
            "schema_version",
            "candidate_id",
            "combined_data_snapshot_hash",
            "cash_distribution_semantics",
            "terminal_settlement_ledger",
        },
        "economic return inputs payload",
    )
    if payload["schema_version"] != ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION:
        raise PeadReplicationError("unsupported economic return inputs schema")
    if payload["candidate_id"] != CANDIDATE_ID:
        raise PeadReplicationError("economic return inputs belong to another target")
    _sha256(
        payload["combined_data_snapshot_hash"],
        "economic return inputs combined snapshot hash",
    )
    try:
        semantics = validate_cash_distribution_semantics(
            payload["cash_distribution_semantics"]
        )
        terminal = validate_terminal_settlement_ledger(
            payload["terminal_settlement_ledger"]
        )
    except PeadEconomicEvidenceError as exc:
        raise PeadReplicationError("economic return input evidence is invalid") from exc
    claimed = _sha256(outer["artifact_hash"], "economic return inputs hash")
    normalized_payload = {
        "schema_version": ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "combined_data_snapshot_hash": payload["combined_data_snapshot_hash"],
        "cash_distribution_semantics": semantics,
        "terminal_settlement_ledger": terminal,
    }
    if claimed != content_hash(normalized_payload):
        raise PeadReplicationError("economic return inputs hash mismatch")
    return {"artifact_hash": claimed, "payload": normalized_payload}


def _series_date_prices(value: pd.Series) -> dict[str, float]:
    result: dict[str, float] = {}
    seen: set[str] = set()
    for raw_day, raw_value in value.items():
        day = pd.Timestamp(raw_day)
        if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
            raise PeadReplicationError("price panel contains a noncanonical date")
        key = day.date().isoformat()
        number = _finite(raw_value)
        if key in seen:
            raise PeadReplicationError("price panel contains a duplicate date")
        seen.add(key)
        if number is not None and number > 0:
            result[key] = number
    return result


def collect_replication_observations(
    normalized: Mapping[str, Any],
    provider: Any,
    *,
    start: str,
    end: str,
    horizons: Sequence[int],
    fresh_days: int,
    session_close_evidence: Mapping[str, Any] | None = None,
    economic_return_inputs: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build PIT monthly observations using actual sessions and T+1 entry."""
    if not horizons or any(type(value) is not int or value <= 0 for value in horizons):
        raise PeadReplicationError("horizons must be positive integers")
    if type(fresh_days) is not int or fresh_days < 0:
        raise PeadReplicationError("fresh_days must be a non-negative integer")
    padding_end = (pd.Timestamp(end) + pd.Timedelta(days=max(horizons) * 3 + 15)).date()
    price_start = (pd.Timestamp(start) - pd.Timedelta(days=100)).date().isoformat()
    sessions = pd.DatetimeIndex(provider.market_sessions(start, padding_end.isoformat()))
    close_sessions = pd.DatetimeIndex(
        provider.market_sessions(price_start, padding_end.isoformat())
    )
    if session_close_evidence is None:
        calendar_reader = getattr(provider, "market_session_close_calendar", None)
        if not callable(calendar_reader):
            raise PeadReplicationError(
                "provider cannot prove authoritative session close times"
            )
        session_close_evidence = calendar_reader()
    validated_close_evidence = _validated_session_close_evidence(
        session_close_evidence,
        required_start=price_start,
        required_end=padding_end.isoformat(),
    )
    close_schedule = _session_close_schedule(
        session_close_evidence,
        close_sessions,
        required_start=price_start,
        required_end=padding_end.isoformat(),
    )
    formations = monthly_formation_dates(sessions, start=start, end=end)
    next_session: dict[pd.Timestamp, pd.Timestamp] = {}
    for formation in formations:
        position = int(sessions.searchsorted(formation, side="right"))
        if position < len(sessions):
            next_session[formation] = pd.Timestamp(sessions[position])

    membership: dict[pd.Timestamp, frozenset[str]] = {}
    for formation in formations:
        membership[formation] = frozenset(
            str(name).strip().upper() for name in provider.universe_asof(formation)
            if str(name).strip()
        )
    events = list(normalized["eps_events"])
    tickers = sorted({event["ticker"] for event in events})
    price_end = padding_end.isoformat()
    split_normalized_prices = _price_panel(
        provider, tickers, price_start, price_end, "close"
    )
    closeunadj_prices = _price_panel(
        provider, tickers, price_start, price_end, "closeunadj"
    )
    adjusted_prices = _price_panel(
        provider, tickers, price_start, price_end, "closeadj"
    )

    validated_economic_inputs = (
        _validated_economic_return_inputs(economic_return_inputs)
        if economic_return_inputs is not None
        else None
    )
    action_rows: list[dict[str, Any]] | None = None
    action_slice_hash = None
    currency_by_ticker: dict[str, str] = {}
    economic_input_reason = None
    adjustment_absolute_tolerance = 0.0
    adjustment_relative_tolerance = 0.0
    terminal_settlements: list[Mapping[str, Any]] = []
    if validated_economic_inputs is None:
        economic_input_reason = "economic_return_inputs_missing"
    else:
        semantics_payload = validated_economic_inputs["payload"][
            "cash_distribution_semantics"
        ]["payload"]
        tolerance = semantics_payload["adjustment_check_tolerance"]
        adjustment_absolute_tolerance = float(tolerance["absolute"])
        adjustment_relative_tolerance = float(tolerance["relative"])
        terminal_settlements = list(
            validated_economic_inputs["payload"]["terminal_settlement_ledger"]
            ["payload"]["cash_only_records"]
        )
        action_reader = getattr(provider, "corporate_actions_for_tickers", None)
        currency_reader = getattr(provider, "security_currency", None)
        if not callable(action_reader):
            economic_input_reason = "corporate_action_slice_reader_unavailable"
        elif not callable(currency_reader):
            economic_input_reason = "security_currency_reader_unavailable"
        else:
            try:
                action_rows = validate_action_rows(
                    action_reader(tickers, price_start, price_end),
                    requested_tickers=tickers,
                    start=price_start,
                    end=price_end,
                )
                action_slice_hash = content_hash(action_rows)
                for ticker in tickers:
                    receipt = currency_reader(ticker)
                    currency_evidence = _exact_fields(
                        receipt,
                        {"ticker", "currency"},
                        f"security currency evidence for {ticker}",
                    )
                    if currency_evidence["ticker"] != ticker:
                        raise PeadReplicationError(
                            "security currency ticker mismatch"
                        )
                    currency = currency_evidence["currency"]
                    if (
                        not isinstance(currency, str)
                        or not currency
                        or currency != currency.strip().upper()
                    ):
                        raise PeadReplicationError(
                            "security currency must be canonical uppercase text"
                        )
                    currency_by_ticker[ticker] = currency
            except EconomicReturnError as exc:
                raise PeadReplicationError(
                    "corporate-action slice failed economic validation"
                ) from exc
    economic_split_prices = {
        ticker: _series_date_prices(
            split_normalized_prices.get(ticker, pd.Series(dtype=float))
        )
        for ticker in tickers
    }
    economic_adjusted_prices = {
        ticker: _series_date_prices(
            adjusted_prices.get(ticker, pd.Series(dtype=float))
        )
        for ticker in tickers
    }

    by_ticker: dict[str, list[dict[str, Any]]] = {}
    signal_exclusions: list[dict[str, Any]] = []
    for event in events:
        ticker = event["ticker"]
        split_normalized = split_normalized_prices.get(
            ticker, pd.Series(dtype=float)
        )
        announcement = datetime.fromisoformat(
            event["announcement_at_utc"][:-1] + "+00:00"
        )
        try:
            before_mask = [
                close_schedule[pd.Timestamp(index)] < announcement
                for index in split_normalized.index
            ]
        except KeyError as exc:
            raise PeadReplicationError(
                "price date is absent from the authoritative session-close schedule"
            ) from exc
        before = split_normalized[np.asarray(before_mask, dtype=bool)]
        before = before[pd.to_numeric(before, errors="coerce") > 0]
        if before.empty:
            signal_exclusions.append(
                {"event_key": event["event_key"], "ticker": ticker,
                 "reason": "missing_positive_split_normalized_preannouncement_close"}
            )
            continue
        preclose_date = pd.Timestamp(before.index[-1])
        preclose = float(before.iloc[-1])
        closeunadj = closeunadj_prices.get(ticker, pd.Series(dtype=float))
        if preclose_date not in closeunadj.index:
            signal_exclusions.append(
                {"event_key": event["event_key"], "ticker": ticker,
                 "reason": "missing_exact_preannouncement_closeunadj_evidence"}
            )
            continue
        execution_preclose = _finite(closeunadj.loc[preclose_date])
        if execution_preclose is None or execution_preclose <= 0:
            signal_exclusions.append(
                {"event_key": event["event_key"], "ticker": ticker,
                 "reason": "invalid_preannouncement_closeunadj_evidence"}
            )
            continue
        signal = float(event["unscaled_forecast_error"]) / preclose
        if not math.isfinite(signal):
            signal_exclusions.append(
                {"event_key": event["event_key"], "ticker": ticker,
                 "reason": "nonfinite_scaled_forecast_error"}
            )
            continue
        prepared = dict(event)
        prepared["preannouncement_close_split_normalized"] = preclose
        prepared["preannouncement_closeunadj_execution_evidence"] = (
            execution_preclose
        )
        prepared["preannouncement_price_date"] = preclose_date
        prepared["forecast_error_scaled"] = signal
        prepared["announcement_datetime"] = datetime.fromisoformat(
            event["announcement_at_utc"][:-1] + "+00:00"
        )
        by_ticker.setdefault(ticker, []).append(prepared)
    for ticker in by_ticker:
        by_ticker[ticker].sort(key=lambda item: item["announcement_datetime"])

    records: list[dict[str, Any]] = []
    formation_exclusions: list[dict[str, Any]] = []
    horizon_return_exclusions: list[dict[str, Any]] = []
    economic_return_exclusions: list[dict[str, Any]] = []
    lifecycle_reader = getattr(provider, "security_lifecycle", None)
    lifecycle_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in by_ticker:
        if not callable(lifecycle_reader):
            lifecycle_by_ticker[ticker] = {
                "status": "unavailable",
                "reason": "security_lifecycle_reader_unavailable",
            }
            continue
        try:
            raw_lifecycle = lifecycle_reader(ticker)
            if not isinstance(raw_lifecycle, Mapping):
                raise PeadReplicationError("security lifecycle must be an object")
            if raw_lifecycle["ticker"] != ticker:
                raise PeadReplicationError("security lifecycle ticker mismatch")
            raw_permaticker = raw_lifecycle["permaticker"]
            if type(raw_permaticker) is not int or raw_permaticker <= 0:
                raise PeadReplicationError(
                    "security lifecycle permaticker must be an exact positive int"
                )
            lifecycle_status = raw_lifecycle["isdelisted"]
            if lifecycle_status not in {"N", "Y"}:
                raise PeadReplicationError(
                    "security lifecycle isdelisted must be literal N or Y"
                )
            final_date = _lifecycle_date(
                raw_lifecycle["lastpricedate"],
                "security lifecycle lastpricedate",
            )
            raw_sep_last_date = raw_lifecycle["sep_lastpricedate"]
            sep_last_date = (
                None
                if raw_sep_last_date is None
                else _lifecycle_date(
                    raw_sep_last_date,
                    "security lifecycle sep_lastpricedate",
                )
            )
            if lifecycle_status == "Y" and sep_last_date != final_date:
                raise PeadReplicationError(
                    "delisted security lifecycle dates do not match"
                )
            lifecycle_by_ticker[ticker] = {
                "status": "validated",
                "isdelisted": lifecycle_status,
                "permaticker": raw_permaticker,
                "lastpricedate": final_date.isoformat(),
                "sep_lastpricedate": (
                    sep_last_date.isoformat() if sep_last_date is not None else None
                ),
            }
        except (KeyError, PeadReplicationError, TypeError, ValueError):
            lifecycle_by_ticker[ticker] = {
                "status": "unresolved",
                "reason": "security_lifecycle_validation_failed",
            }
    cap_panel = getattr(provider, "daily_marketcaps_for_dates", None)
    caps = (
        cap_panel(tickers, formations)
        if cap_panel is not None
        else {
            (ticker, formation.date()): (
                (provider.daily_metric(ticker, formation) or {}).get("marketcap")
            )
            for ticker in tickers
            for formation in formations
        }
    )
    for formation in formations:
        entry_date = next_session.get(formation)
        if entry_date is None:
            continue
        session_position = int(sessions.searchsorted(entry_date))
        if (
            session_position >= len(sessions)
            or pd.Timestamp(sessions[session_position]) != entry_date
        ):
            raise PeadReplicationError("entry date is absent from the global calendar")
        try:
            cutoff = close_schedule[formation]
        except KeyError as exc:
            raise PeadReplicationError(
                "formation date is absent from the authoritative close schedule"
            ) from exc
        eligible = membership[formation]
        for ticker, ticker_events in by_ticker.items():
            if ticker not in eligible:
                continue
            visible = [
                event for event in ticker_events
                if event["announcement_datetime"] < cutoff
                and 0 <= (formation.date() - date.fromisoformat(event["act_rpt_date"])).days
                <= fresh_days
            ]
            if not visible:
                continue
            signal_event = visible[-1]
            split_normalized = split_normalized_prices.get(
                ticker, pd.Series(dtype=float)
            )
            if entry_date not in split_normalized.index:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "missing_exact_t_plus_1_split_normalized_entry",
                    }
                )
                continue
            split_entry = _finite(split_normalized.loc[entry_date])
            if split_entry is None or split_entry <= 0:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "invalid_split_normalized_entry_price",
                    }
                )
                continue
            closeunadj = closeunadj_prices.get(ticker, pd.Series(dtype=float))
            if entry_date not in closeunadj.index:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "missing_exact_t_plus_1_closeunadj_evidence",
                    }
                )
                continue
            closeunadj_entry = _finite(closeunadj.loc[entry_date])
            if closeunadj_entry is None or closeunadj_entry <= 0:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "invalid_closeunadj_entry_evidence",
                    }
                )
                continue
            adjusted = adjusted_prices.get(ticker, pd.Series(dtype=float))
            if entry_date not in adjusted.index:
                formation_exclusions.append(
                    {"formation_date": formation.date().isoformat(), "ticker": ticker,
                     "reason": "missing_exact_t_plus_1_adjusted_entry"}
                )
                continue
            entry = _finite(adjusted.loc[entry_date])
            if entry is None or entry <= 0:
                formation_exclusions.append(
                    {"formation_date": formation.date().isoformat(), "ticker": ticker,
                     "reason": "invalid_adjusted_entry_price"}
                )
                continue
            mcap = caps.get((ticker, formation.date()))
            mcap_number = _finite(mcap)
            if mcap_number is None or mcap_number <= 0:
                formation_exclusions.append(
                    {"formation_date": formation.date().isoformat(), "ticker": ticker,
                     "reason": "missing_positive_pit_marketcap"}
                )
                continue
            record = {
                "date": formation,
                "entry_date": entry_date,
                "name": ticker,
                "m_ticker": signal_event["event_key"]["m_ticker"],
                "sue": signal_event["forecast_error_scaled"],
                "mcap": mcap_number,
                "entry_close_split_normalized": split_entry,
                "entry_closeunadj_execution_evidence": closeunadj_entry,
                "entry_closeadj_diagnostic": entry,
                "signal_preannouncement_close_split_normalized": signal_event[
                    "preannouncement_close_split_normalized"
                ],
                "signal_preannouncement_closeunadj_execution_evidence": signal_event[
                    "preannouncement_closeunadj_execution_evidence"
                ],
                "source_event_key": signal_event["event_key"],
            }
            for horizon in horizons:
                target_position = session_position + horizon
                target_date = (
                    pd.Timestamp(sessions[target_position])
                    if target_position < len(sessions)
                    else None
                )
                reason = None
                exit_value = None
                lifecycle = lifecycle_by_ticker[ticker]
                if target_date is None:
                    reason = "global_horizon_outside_observed_sessions"
                elif (
                    lifecycle.get("status") == "validated"
                    and lifecycle.get("isdelisted") == "Y"
                    and date.fromisoformat(lifecycle["lastpricedate"])
                    < target_date.date()
                ):
                    reason = "held_delisting_terminal_economics_unresolved"
                elif target_date not in adjusted.index:
                    reason = "missing_exact_global_session_exit"
                else:
                    exit_value = _finite(adjusted.loc[target_date])
                    if exit_value is None or exit_value <= 0:
                        reason = "invalid_exact_global_session_exit"
                record[f"target_exit_date_{horizon}"] = target_date
                record[f"fwd_{horizon}"] = (
                    exit_value / entry - 1.0
                    if reason is None and exit_value is not None
                    else np.nan
                )
                record[f"return_resolution_{horizon}"] = {
                    "status": (
                        "resolved_diagnostic" if reason is None else "unresolved"
                    ),
                    "reason": reason,
                    "pricing_path": (
                        "SEP.closeadj_exact_global_sessions_diagnostic"
                    ),
                }
                if reason is not None:
                    horizon_return_exclusions.append(
                        {
                            "formation_date": formation.date().isoformat(),
                            "ticker": ticker,
                            "m_ticker": signal_event["event_key"]["m_ticker"],
                            "horizon_sessions": horizon,
                            "target_exit_date": (
                                target_date.date().isoformat()
                                if target_date is not None else None
                            ),
                            "reason": reason,
                        }
                    )
                economic_reason = None
                economic_resolution: dict[str, Any]
                if target_date is None:
                    economic_reason = "global_horizon_outside_observed_sessions"
                elif economic_input_reason is not None:
                    economic_reason = economic_input_reason
                elif action_rows is None:
                    economic_reason = "corporate_action_slice_unavailable"
                elif lifecycle.get("status") != "validated":
                    economic_reason = "security_lifecycle_unresolved"
                elif currency_by_ticker.get(ticker) != "USD":
                    economic_reason = "security_currency_not_usd_or_unresolved"
                if economic_reason is not None:
                    diagnostic_return = record[f"fwd_{horizon}"]
                    economic_resolution = {
                        "status": "unresolved",
                        "reason": economic_reason,
                        "pricing_path": (
                            "SEP.close_plus_explicit_cash_no_reinvestment_candidate"
                        ),
                        "entry_price_split_normalized": split_entry,
                        "exit_price_split_normalized": (
                            _finite(split_normalized.loc[target_date])
                            if target_date is not None
                            and target_date in split_normalized.index
                            else None
                        ),
                        "cash_distributions": [],
                        "cash_total": 0.0,
                        "terminal_settlement_id": None,
                        "gross_terminal_value": None,
                        "gross_economic_return": None,
                        "closeadj_diagnostic_return": (
                            float(diagnostic_return)
                            if pd.notna(diagnostic_return)
                            else None
                        ),
                        "ignored_actions": [],
                    }
                else:
                    assert target_date is not None and action_rows is not None
                    try:
                        economic_resolution = reconstruct_cash_return(
                            ticker=ticker,
                            entry_date=entry_date.date().isoformat(),
                            exit_date=target_date.date().isoformat(),
                            split_normalized_prices=economic_split_prices[ticker],
                            adjusted_prices=economic_adjusted_prices[ticker],
                            action_rows=action_rows,
                            lifecycle=lifecycle,
                            currency=currency_by_ticker[ticker],
                            terminal_settlements=terminal_settlements,
                            adjustment_absolute_tolerance=(
                                adjustment_absolute_tolerance
                            ),
                            adjustment_relative_tolerance=(
                                adjustment_relative_tolerance
                            ),
                        )
                    except EconomicReturnError as exc:
                        raise PeadReplicationError(
                            "economic return reconstruction contract failed"
                        ) from exc
                economic_value = economic_resolution["gross_economic_return"]
                record[f"economic_fwd_{horizon}"] = (
                    float(economic_value)
                    if economic_value is not None
                    else np.nan
                )
                record[f"economic_return_resolution_{horizon}"] = (
                    economic_resolution
                )
                if economic_resolution["status"] == "unresolved":
                    economic_return_exclusions.append(
                        {
                            "formation_date": formation.date().isoformat(),
                            "ticker": ticker,
                            "m_ticker": signal_event["event_key"]["m_ticker"],
                            "horizon_sessions": horizon,
                            "target_exit_date": (
                                target_date.date().isoformat()
                                if target_date is not None
                                else None
                            ),
                            "reason": economic_resolution["reason"],
                        }
                    )
            records.append(record)
    columns = [
        "date", "entry_date", "name", "m_ticker", "sue", "mcap",
        "entry_close_split_normalized", "entry_closeunadj_execution_evidence",
        "entry_closeadj_diagnostic",
        "signal_preannouncement_close_split_normalized",
        "signal_preannouncement_closeunadj_execution_evidence",
        "source_event_key",
        *(
            field
            for horizon in horizons
            for field in (
                f"target_exit_date_{horizon}", f"fwd_{horizon}",
                f"return_resolution_{horizon}",
                f"economic_fwd_{horizon}",
                f"economic_return_resolution_{horizon}",
            )
        ),
    ]
    frame = _size_terciles(pd.DataFrame(records, columns=columns)) if records else pd.DataFrame(
        columns=[*columns, "tercile"]
    )
    economic_resolutions = [
        record[f"economic_return_resolution_{horizon}"]
        for record in records
        for horizon in horizons
    ]
    distribution_applications = [
        distribution
        for resolution in economic_resolutions
        for distribution in resolution["cash_distributions"]
    ]
    unique_distribution_keys = {
        canonical_json(item["action_key"]) for item in distribution_applications
    }
    adjustment_errors = [
        float(item["adjustment_absolute_error"])
        for item in distribution_applications
    ]
    semantics_status = (
        validated_economic_inputs["payload"]["cash_distribution_semantics"]
        ["payload"]["evidence_status"]
        if validated_economic_inputs is not None
        else None
    )
    semantics_qualification_allowed = (
        validated_economic_inputs["payload"]["cash_distribution_semantics"]
        ["payload"]["qualification_allowed"]
        if validated_economic_inputs is not None
        else False
    )
    coverage = {
        "observed_sessions": len(sessions),
        "formation_dates": len(formations),
        "normalized_signal_events": len(events),
        "signals_with_preannouncement_close": sum(len(value) for value in by_ticker.values()),
        "portfolio_observations": len(frame),
        "names_with_portfolio_observations": int(frame["name"].nunique()) if len(frame) else 0,
        "signal_exclusions": signal_exclusions,
        "formation_exclusions": formation_exclusions,
        "horizon_return_exclusions": horizon_return_exclusions,
        "economic_return_exclusions": economic_return_exclusions,
        "economic_return_reconstruction": {
            "input_artifact_hash": (
                validated_economic_inputs["artifact_hash"]
                if validated_economic_inputs is not None
                else None
            ),
            "cash_distribution_semantics_status": semantics_status,
            "cash_distribution_semantics_qualification_allowed": (
                semantics_qualification_allowed
            ),
            "action_slice_status": (
                "validated" if action_rows is not None else "unavailable"
            ),
            "action_slice_reason": economic_input_reason,
            "action_slice_hash": action_slice_hash,
            "action_slice_rows": len(action_rows) if action_rows is not None else 0,
            "security_currencies": dict(sorted(currency_by_ticker.items())),
            "terminal_settlement_records": len(terminal_settlements),
            "holding_paths": len(economic_resolutions),
            "mechanically_resolved_paths": sum(
                resolution["status"] != "unresolved"
                for resolution in economic_resolutions
            ),
            "unresolved_paths": sum(
                resolution["status"] == "unresolved"
                for resolution in economic_resolutions
            ),
            "mechanical_reconstruction_complete": bool(economic_resolutions)
            and all(
                resolution["status"] != "unresolved"
                for resolution in economic_resolutions
            ),
            "cash_distribution_unique_rows": len(unique_distribution_keys),
            "cash_distribution_path_applications": len(
                distribution_applications
            ),
            "ignored_issuer_external_action_applications": sum(
                len(resolution["ignored_actions"])
                for resolution in economic_resolutions
            ),
            "maximum_dividend_adjustment_absolute_error": (
                max(adjustment_errors) if adjustment_errors else None
            ),
            "reinvestment": False,
            "cash_yield": 0.0,
            "qualification_ready": bool(economic_resolutions)
            and all(
                resolution["status"] != "unresolved"
                for resolution in economic_resolutions
            )
            and semantics_qualification_allowed is True,
        },
        "security_lifecycle_diagnostics": lifecycle_by_ticker,
        "security_lifecycle_complete": bool(lifecycle_by_ticker) and all(
            item.get("status") == "validated"
            for item in lifecycle_by_ticker.values()
        ),
        "session_close_evidence_artifact_hash": validated_close_evidence[
            "artifact_hash"
        ],
        "session_close_calendar_artifact_hash": validated_close_evidence[
            "payload"
        ]["calendar"]["artifact_hash"],
        "session_close_source_receipt_artifact_hash": validated_close_evidence[
            "payload"
        ]["source_receipt"]["artifact_hash"],
        "session_close_schedule_sessions": len(close_schedule),
        "observed_early_close_sessions": sum(
            close.astimezone(EASTERN).hour == 13
            for close in close_schedule.values()
        ),
    }
    return frame, coverage


def build_replication_report(
    snapshot: ValidatedSnapshot,
    provider: Any,
    *,
    start: str,
    end: str,
    horizons: Sequence[int],
    cost_bps: float,
    fresh_days: int,
    quantile: float,
    winsor_fraction: float | None,
    consensus_abs_tolerance: float,
    independent_reconciliation: Any | None = None,
    research_manifest_binding: Mapping[str, Any] | None = None,
    cash_distribution_semantics: Mapping[str, Any] | None = None,
    terminal_settlement_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the locked screen and return a complete canonical JSON mapping."""
    cost = float(cost_bps)
    if not math.isfinite(cost) or cost < 0:
        raise PeadReplicationError("cost_bps must be finite and non-negative")
    if not 0 < float(quantile) <= 0.5:
        raise PeadReplicationError("quantile must be in (0, 0.5]")
    manifest_binding = (
        _validated_research_manifest_binding(research_manifest_binding)
        if research_manifest_binding is not None
        else None
    )
    warehouse_snapshot_before, warehouse_snapshot_error = (
        _capture_warehouse_return_snapshot(provider)
    )
    action_required_end = (
        pd.Timestamp(end) + pd.Timedelta(days=max(horizons) * 3 + 15)
    ).date().isoformat()
    calendar_required_start = (
        pd.Timestamp(start) - pd.Timedelta(days=100)
    ).date().isoformat()
    calendar_reader = getattr(provider, "market_session_close_calendar", None)
    if not callable(calendar_reader):
        raise PeadReplicationError(
            "provider cannot prove authoritative session close times"
        )
    close_evidence_bundle_before = calendar_reader()
    close_evidence_before = _validated_session_close_evidence(
        close_evidence_bundle_before,
        required_start=calendar_required_start,
        required_end=action_required_end,
    )
    corporate_action_before, corporate_action_error = (
        _capture_corporate_action_evidence(
            provider, start=start, end=action_required_end
        )
    )
    combined_data_snapshot_before = (
        build_combined_data_snapshot(
            snapshot.artifact_hash,
            warehouse_snapshot_before,
            corporate_action_before,
            close_evidence_bundle_before,
        )
        if warehouse_snapshot_before is not None
        and corporate_action_before is not None
        else None
    )
    economic_return_inputs = None
    if (
        combined_data_snapshot_before is not None
        and cash_distribution_semantics is not None
        and terminal_settlement_ledger is not None
    ):
        economic_return_inputs = build_economic_return_inputs(
            combined_data_snapshot_before,
            cash_distribution_semantics,
            terminal_settlement_ledger,
        )
    normalized = normalize_source_events(
        snapshot, consensus_abs_tolerance=consensus_abs_tolerance
    )
    observations, run_coverage = collect_replication_observations(
        normalized,
        provider,
        start=start,
        end=end,
        horizons=tuple(horizons),
        fresh_days=fresh_days,
        session_close_evidence=close_evidence_bundle_before,
        economic_return_inputs=economic_return_inputs,
    )
    raw_slices = [("pooled", observations)] + [
        (f"tercile{value}", observations[observations["tercile"] == value])
        for value in (0, 1, 2)
    ]
    tests: list[dict[str, Any]] = []
    p_values: list[float] = []
    slice_coverage: dict[str, Any] = {}
    for label, raw_frame in raw_slices:
        frame, cell_coverage = _locked_factor_slice(
            raw_frame, minimum_names_per_formation=10
        )
        horizon_coverage: dict[str, Any] = {}
        for horizon in horizons:
            selected, selection_coverage = _locked_horizon_portfolio(
                frame, horizon=horizon, quantile=float(quantile)
            )
            horizon_coverage[str(horizon)] = selection_coverage
            results = factor_study(
                selected,
                "_pead_signal_order",
                (horizon,),
                cost,
                # ``selected`` already contains the frozen low/high q legs.
                # A 50/50 split reproduces their original sizes exactly.
                quantile=0.5,
                cheap_is_long=False,
                drop_nonpositive=False,
                winsor_returns=winsor_fraction,
                include_series=True,
            )
            result = results[0]
            result["factor"] = label
            result["signal_tie_break"] = "ascending_stable_m_ticker"
            result["minimum_names_per_formation"] = 10
            result["selection_frozen_before_return_availability"] = True
            tests.append(result)
            if not result.get("insufficient"):
                p_values.append(float(result["p"]))
        slice_coverage[label] = {
            **cell_coverage,
            "horizons": horizon_coverage,
        }
    multiple_testing = benjamini_hochberg(p_values) if p_values else {
        "m": 0,
        "alpha": 0.05,
        "rejected_bh": [],
        "n_significant_bh": 0,
    }
    warehouse_snapshot_after: Mapping[str, Any] | None = None
    warehouse_snapshot_unchanged: bool | None = None
    warehouse_snapshot_after_error: str | None = None
    if warehouse_snapshot_before is not None:
        warehouse_snapshot_after, warehouse_snapshot_after_error = (
            _capture_warehouse_return_snapshot(provider)
        )
        warehouse_snapshot_unchanged = (
            warehouse_snapshot_after is not None
            and warehouse_snapshot_after == warehouse_snapshot_before
        )
    corporate_action_after: Mapping[str, Any] | None = None
    corporate_action_unchanged: bool | None = None
    corporate_action_after_error: str | None = None
    if corporate_action_before is not None:
        corporate_action_after, corporate_action_after_error = (
            _capture_corporate_action_evidence(
                provider, start=start, end=action_required_end
            )
        )
        corporate_action_unchanged = (
            corporate_action_after is not None
            and corporate_action_after == corporate_action_before
        )
    close_evidence_bundle_after = calendar_reader()
    close_evidence_after = _validated_session_close_evidence(
        close_evidence_bundle_after,
        required_start=calendar_required_start,
        required_end=action_required_end,
    )
    close_evidence_unchanged = close_evidence_after == close_evidence_before
    combined_data_snapshot = (
        build_combined_data_snapshot(
            snapshot.artifact_hash,
            warehouse_snapshot_before,
            corporate_action_before,
            close_evidence_bundle_before,
        )
        if warehouse_snapshot_before is not None
        and warehouse_snapshot_unchanged is True
        and corporate_action_before is not None
        and corporate_action_unchanged is True
        and close_evidence_unchanged
        else None
    )
    raw_portfolio_observations = []
    for row in observations.to_dict(orient="records"):
        item = {
            "formation_date": pd.Timestamp(row["date"]).date().isoformat(),
            "entry_date": pd.Timestamp(row["entry_date"]).date().isoformat(),
            "ticker": row["name"],
            "m_ticker": row["m_ticker"],
            "signal": float(row["sue"]),
            "pit_marketcap": float(row["mcap"]),
            "entry_close_split_normalized": float(
                row["entry_close_split_normalized"]
            ),
            "entry_closeunadj_execution_evidence": float(
                row["entry_closeunadj_execution_evidence"]
            ),
            "entry_closeadj_diagnostic": float(row["entry_closeadj_diagnostic"]),
            "signal_preannouncement_close_split_normalized": float(
                row["signal_preannouncement_close_split_normalized"]
            ),
            "signal_preannouncement_closeunadj_execution_evidence": float(
                row["signal_preannouncement_closeunadj_execution_evidence"]
            ),
            "size_tercile": (
                int(row["tercile"]) if pd.notna(row["tercile"]) else None
            ),
            "source_event_key": row["source_event_key"],
        }
        for horizon in horizons:
            value = row[f"fwd_{horizon}"]
            target = row[f"target_exit_date_{horizon}"]
            item[f"target_exit_date_{horizon}"] = (
                pd.Timestamp(target).date().isoformat()
                if pd.notna(target) else None
            )
            item[f"adjusted_forward_return_{horizon}"] = (
                float(value) if pd.notna(value) else None
            )
            item[f"return_resolution_{horizon}"] = row[
                f"return_resolution_{horizon}"
            ]
            economic_value = row[f"economic_fwd_{horizon}"]
            item[f"economic_forward_return_candidate_{horizon}"] = (
                float(economic_value) if pd.notna(economic_value) else None
            )
            item[f"economic_return_resolution_{horizon}"] = row[
                f"economic_return_resolution_{horizon}"
            ]
        raw_portfolio_observations.append(item)
    blockers = list(snapshot.coverage_blockers)
    tables = snapshot.payload["tables"]
    for table_code in ("ZACKS/SS", "ZACKS/SEH", "ZACKS/MT", "ZACKS/EA"):
        if table_code not in tables:
            blockers.append(f"required_source_table_missing:{table_code}")
    if snapshot.payload["evidence_class"] not in {
        "historical_replication", "prospective_signal"
    }:
        blockers.append("source_evidence_class_not_frozen_replication")
    identity = normalized["stable_identity_diagnostics"]
    if not identity["available"] or identity["invalid_events"]:
        blockers.append("stable_mt_identity_validation_incomplete")
    schedule = normalized["announcement_schedule_diagnostics"]
    if not schedule["available"]:
        blockers.append("announcement_schedule_reconciliation_incomplete")
    elif (
        snapshot.payload["evidence_class"] == "prospective_signal"
        and schedule["matched_event_keys"] != schedule["actual_event_keys"]
    ):
        blockers.append("prospective_announcement_schedule_match_incomplete")
    if warehouse_snapshot_error is not None:
        blockers.append(warehouse_snapshot_error)
    if warehouse_snapshot_before is not None and not warehouse_snapshot_before["complete"]:
        blockers.append("warehouse_return_snapshot_incomplete")
    if warehouse_snapshot_after_error is not None:
        blockers.append(warehouse_snapshot_after_error)
    if warehouse_snapshot_before is not None and warehouse_snapshot_unchanged is not True:
        blockers.append("warehouse_return_snapshot_changed_during_run")
    if corporate_action_error is not None:
        blockers.append(corporate_action_error)
    if (
        corporate_action_before is not None
        and not corporate_action_before["payload"]["complete"]
    ):
        blockers.extend(corporate_action_before["payload"]["blockers"])
        blockers.append("corporate_action_evidence_incomplete")
    if corporate_action_after_error is not None:
        blockers.append(corporate_action_after_error)
    if corporate_action_before is not None and corporate_action_unchanged is not True:
        blockers.append("corporate_action_evidence_changed_during_run")
    if not close_evidence_unchanged:
        blockers.append("session_close_evidence_changed_during_run")
    if combined_data_snapshot is None:
        blockers.append("combined_data_snapshot_unavailable")
    if manifest_binding is None:
        blockers.append("research_manifest_binding_missing")
    if corporate_action_before is None:
        blockers.append("corporate_action_evidence_incomplete")
    if not run_coverage["security_lifecycle_complete"]:
        blockers.append("security_lifecycle_evidence_incomplete")
    if any(
        coverage["excluded_unresolved_cohorts"]
        for cell in slice_coverage.values()
        for coverage in cell["horizons"].values()
    ):
        blockers.append("selected_constituent_return_unresolved")
    economic_coverage = run_coverage["economic_return_reconstruction"]
    if economic_return_inputs is None:
        blockers.append("economic_return_input_evidence_missing")
    elif combined_data_snapshot is None or (
        economic_return_inputs["payload"]["combined_data_snapshot_hash"]
        != combined_data_snapshot["artifact_hash"]
    ):
        blockers.append("economic_return_inputs_snapshot_mismatch")
    if not economic_coverage["mechanical_reconstruction_complete"]:
        blockers.append("economic_return_reconstruction_incomplete")
    if not economic_coverage[
        "cash_distribution_semantics_qualification_allowed"
    ]:
        blockers.append("cash_distribution_semantics_source_missing")

    reconciliation_hash = None
    if independent_reconciliation is None:
        blockers.append("independent_implementation_reconciliation_missing")
    else:
        try:
            from analysis.pead_daily_acceptance import (
                ValidatedPeadDailyReconciliation,
                validate_replayed_pead_daily_reconciliation_for_bindings,
            )

            if type(independent_reconciliation) is not (
                ValidatedPeadDailyReconciliation
            ):
                raise TypeError

            if combined_data_snapshot is None:
                blockers.append(
                    "independent_reconciliation_combined_snapshot_unavailable"
                )
            elif manifest_binding is None:
                blockers.append(
                    "independent_reconciliation_protocol_manifest_unavailable"
                )
            elif economic_return_inputs is None:
                blockers.append(
                    "independent_reconciliation_economic_inputs_unavailable"
                )
            else:
                binding_validator = (
                    validate_replayed_pead_daily_reconciliation_for_bindings
                )
                validated_receipt = binding_validator(
                    independent_reconciliation,
                    combined_data_snapshot_hash=combined_data_snapshot[
                        "artifact_hash"
                    ],
                    economic_return_inputs_hash=economic_return_inputs[
                        "artifact_hash"
                    ],
                    research_manifest_binding_hash=manifest_binding[
                        "artifact_hash"
                    ],
                )
                repository_root = Path(__file__).resolve().parents[1]
                manifests = validated_receipt["implementation_manifests"]
                for section in ("primary", "reference", "shared"):
                    implementation = manifests[section]
                    files = implementation["files"]
                    for row in files:
                        path = (repository_root / row["path"]).resolve()
                        path.relative_to(repository_root)
                        if (
                            not path.is_file()
                            or path.is_symlink()
                            or hashlib.sha256(path.read_bytes()).hexdigest()
                            != row["sha256"]
                        ):
                            raise ValueError
                    if content_hash(files) != implementation["code_hash"]:
                        raise ValueError
                reconciliation_hash = independent_reconciliation.artifact_hash
                if not validated_receipt[
                    "bounded_modeled_daily_money_path_reconciliation_passed"
                ]:
                    raise ValueError
        except (ImportError, KeyError, TypeError, ValueError):
            blockers.append("independent_implementation_reconciliation_invalid")
    if not normalized["eps_events"]:
        blockers.append("no_normalized_eps_events")
    if observations.empty:
        blockers.append("no_portfolio_observations")
    if not any(not test.get("insufficient") for test in tests):
        blockers.append("insufficient_formation_history_for_inference")
    blockers = sorted(set(blockers))
    completed = not blockers
    source_evidence_class = snapshot.payload["evidence_class"]
    report_evidence_class = (
        "development_independent_source_sample"
        if source_evidence_class == "development_sample"
        else "independent_source_replication_nonqualifying"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "evidence_class": report_evidence_class,
        "qualifying_evidence": False,
        "nonqualifying_reason": (
            "The return window and candidate family were inspected during development; "
            "an independent signal source does not make those returns untouched."
        ),
        "status": "completed" if completed else "blocked_or_sample",
        "completed_full_replication": completed,
        "blockers": blockers,
        "source_snapshot": {
            "artifact_hash": snapshot.artifact_hash,
            "source_id": SOURCE_ID,
            "source_evidence_class": snapshot.payload["evidence_class"],
            "captured_at": snapshot.payload["captured_at"],
            "requested_window": dict(snapshot.payload["requested_window"]),
            "coverage_full_window": snapshot.full_window,
            "legacy_sf1_or_events_used": False,
            "warehouse_return_tables": list(WAREHOUSE_RETURN_TABLES),
            "warehouse_return_snapshot": warehouse_snapshot_before,
            "warehouse_return_snapshot_after_run": warehouse_snapshot_after,
            "warehouse_snapshot_unchanged_during_run": warehouse_snapshot_unchanged,
            "corporate_action_evidence": corporate_action_before,
            "corporate_action_evidence_after_run": corporate_action_after,
            "corporate_action_evidence_unchanged_during_run": (
                corporate_action_unchanged
            ),
            "session_close_evidence": close_evidence_before,
            "session_close_evidence_after_run": close_evidence_after,
            "session_close_evidence_unchanged_during_run": (
                close_evidence_unchanged
            ),
        },
        "combined_data_snapshot": combined_data_snapshot,
        "economic_return_inputs": economic_return_inputs,
        "research_manifest_binding": manifest_binding,
        "configuration": {
            "start": start,
            "end": end,
            "horizons_sessions": list(horizons),
            "fresh_days_calendar": fresh_days,
            "quantile": float(quantile),
            "one_way_cost_bps_per_trade_per_leg": cost,
            "fixed_total_round_trip_bps": 4.0 * cost,
            "winsor_fraction_diagnostic_only": winsor_fraction,
            "consensus_absolute_tolerance": float(consensus_abs_tolerance),
            "formation_calendar": "first_observed_warehouse_session_of_month",
            "formation_cutoff": (
                "actual NYSE regular-session close from the bound official "
                "session-close calendar (13:00 ET on published early closes; "
                "16:00 ET otherwise)"
            ),
            "entry_timing": "next_observed_session_close_t_plus_1",
            "entry_price": (
                "SEP.close split-normalized accounting basis with exact "
                "SEP.closeunadj execution evidence on T+1"
            ),
            "signal": (
                "(ZACKS_ES_eps_act-latest_strictly_prior_EEH_eps_mean_est)"
                "/split_normalized_SEP.close; closeunadj retained separately"
            ),
            "signal_per_share_basis": (
                "Zacks per-share values are split-restated; SEP.close matches "
                "that basis while SEP.closeunadj records contemporaneous dollars"
            ),
            "forward_returns": (
                "SEP.closeadj diagnostic retained as the legacy statistical "
                "series; it is not qualifying executable economics"
            ),
            "economic_return_candidate": (
                "SEP.close at the exact global-session exit plus explicit "
                "cash distributions, without reinvestment, on the "
                "split-normalized entry basis"
            ),
            "forward_return_status": (
                "economic candidate mechanically reconstructed but excluded "
                "from qualifying inference until cash-distribution semantics "
                "are authoritatively sourced"
            ),
            "cash_distribution_holding_interval": (
                "entry_date_exclusive_exit_date_inclusive"
            ),
            "cash_distribution_reinvestment": False,
            "terminal_payout_from_actions_value_allowed": False,
            "primary_leg": "EPS only; sales is preserved as an untuned diagnostic",
            "minimum_names_per_formation_per_slice": 10,
            "signal_tie_break": "ascending stable m_ticker",
        },
        "normalization": normalized,
        "coverage": run_coverage,
        "slice_coverage": slice_coverage,
        "raw_portfolio_observations": raw_portfolio_observations,
        "tests": tests,
        "multiple_testing": multiple_testing,
        "independent_reconciliation_hash": reconciliation_hash,
    }
    if reconciliation_hash is not None:
        try:
            from analysis.pead_daily_reconciliation import (
                pead_reconciliation_input,
            )

            current_core = pead_reconciliation_input(report)["artifact_hash"]
            receipt_core = independent_reconciliation.source_report_core_hash
            document_core = independent_reconciliation.document["payload"][
                "bindings"
            ]["source_report_core_hash"]
            if receipt_core != current_core or document_core != current_core:
                raise ValueError
        except (ImportError, KeyError, TypeError, ValueError):
            report["independent_reconciliation_hash"] = None
            report["blockers"] = sorted(
                set(report["blockers"])
                | {"independent_reconciliation_report_core_mismatch"}
            )
            report["completed_full_replication"] = False
            report["status"] = "blocked_or_sample"
    # Round-trip proves the returned object is strict JSON and detaches pandas/numpy values.
    return json.loads(canonical_json(report))


__all__ = [
    "CANDIDATE_ID",
    "COMBINED_DATA_SNAPSHOT_SCHEMA_VERSION",
    "PeadReplicationError",
    "REPORT_SCHEMA_VERSION",
    "RESEARCH_MANIFEST_BINDING_SCHEMA_VERSION",
    "ValidatedSnapshot",
    "WAREHOUSE_RETURN_TABLES",
    "build_combined_data_snapshot",
    "build_research_manifest_binding",
    "build_replication_report",
    "canonical_json",
    "collect_replication_observations",
    "content_hash",
    "_locked_factor_slice",
    "_locked_horizon_portfolio",
    "monthly_formation_dates",
    "normalize_source_events",
    "validate_snapshot_document",
]
