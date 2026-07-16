"""Independent reference reconstruction for the locked Zacks PEAD target.

This implementation intentionally does not import :mod:`analysis.pead_replication`
or any of its normalization, calendar, portfolio, or report helpers.  It reads the
immutable Zacks snapshot and the point-in-time warehouse through their public data
contracts, reconstructs the signal path, and compares that output with a primary
PEAD report.

The resulting artifact is deliberately narrower than ``ReplicationEvidence``.
The research screen has no order, position, cash, fee, or realised-P&L ledger, so
this module never invents those fields and never claims deployment qualification.
Instead it preserves the independently reconstructed event and portfolio records,
their discrepancies, and the exact protocol/data/code identities required by a
later event-driven money-path reconciliation.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from data.pead_economic_evidence import (
    PeadEconomicEvidenceError,
    validate_cash_distribution_semantics,
    validate_terminal_settlement_ledger,
)


SCHEMA_VERSION = "pead_reference_reconciliation.v4"
SNAPSHOT_SCHEMA_VERSION = "zacks_pead_snapshot.v1"
COMBINED_SNAPSHOT_SCHEMA_VERSION = "pead_combined_data_snapshot.v4"
MANIFEST_BINDING_SCHEMA_VERSION = "pead_research_manifest_binding.v1"
SESSION_CALENDAR_SCHEMA_VERSION = "nyse_session_close_calendar.v1"
SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION = "nyse_session_close_evidence.v1"
SESSION_SOURCE_RECEIPT_SCHEMA_VERSION = "nyse_session_close_source_receipt.v1"
SESSION_SOURCE_EXTRACTION_METHOD = "nyse_early_close_html_text.v1"
ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION = "pead_economic_return_inputs.v1"
MAX_SESSION_SOURCE_BYTES = 8 * 1024 * 1024
MAX_SESSION_SOURCE_CLOCK_SKEW_SECONDS = 10 * 60
MAX_SESSION_SOURCE_RECEIPT_DURATION_SECONDS = 60 * 60
CANDIDATE_ID = "pead-vq-locked-replication-v1"
SOURCE_ID = "nasdaq-data-link-zacks"
RETURN_TABLES = ("sep", "tickers", "daily", "actions")
EASTERN = ZoneInfo("America/New_York")
MONEY_PATH_FIELDS = ("target", "order", "position", "cash", "fees", "pnl")
UNAVAILABLE_REPLICATION_FIELDS = ("rank", *MONEY_PATH_FIELDS)
_OBSERVED_SESSION_RULE = (
    "SEP-observed sessions use the listed 13:00 early close when present and "
    "otherwise the NYSE 16:00 core-session close; dates absent from SEP receive "
    "no inferred session."
)

_HEX = frozenset("0123456789abcdef")
_MONTH_NUMBERS = {
    month: number
    for number, month in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}
_OFFICIAL_DATE_PATTERN = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s*,?\s*"
    r"(" + "|".join(_MONTH_NUMBERS) + r")\s+(\d{1,2})\s*,\s*(\d{4})",
    re.IGNORECASE,
)
_EARLY_CLOSE_STATEMENT_PATTERN = re.compile(
    r"\*{0,4}\s*Each market will close early at 1:00 p\.m\."
    r"(.*?)"
    r"(?=\*{1,4}\s*Each market will close early at 1:00 p\.m\."
    r"|NYSE Group Markets holidays|Link to (?:NYSE|Holidays)"
    r"|About (?:NYSE|Intercontinental)|SOURCE:|$)",
    re.IGNORECASE,
)
_CORE_HOURS_PATTERN = re.compile(
    r"\bcore trading session\b.{0,120}?\b9:30\s*a\.?m\.?\s*"
    r"(?:et\s*)?(?:to|[-\N{EN DASH}\N{EM DASH}])\s*"
    r"4:00\s*p\.?m\.?\s*et\b",
    re.IGNORECASE,
)
_EPS_ACTUAL_FIELDS = {
    "m_ticker", "ticker", "currency_code", "per_end_date", "per_type",
    "act_rpt_date", "eps_mean_est", "eps_act", "eps_cnt_est",
    "eps_act_zacks_adj", "act_rpt_time", "act_rpt_code",
}
_EPS_HISTORY_FIELDS = {
    "m_ticker", "ticker", "currency_code", "per_end_date", "per_type",
    "obs_date", "eps_mean_est", "eps_cnt_est",
}
_SALES_ACTUAL_FIELDS = {
    "m_ticker", "ticker", "currency_code", "per_end_date", "per_type",
    "act_rpt_date", "sales_mean_est", "sales_act", "sales_cnt_est",
    "sales_act_zacks_adj", "act_rpt_time", "act_rpt_code",
}
_SALES_HISTORY_FIELDS = {
    "m_ticker", "ticker", "currency_code", "per_end_date", "per_type",
    "obs_date", "sales_mean_est", "sales_cnt_est",
}
_MT_FIELDS = {"m_ticker", "ticker", "currency_code", "comp_cik"}
_EA_FIELDS = {"m_ticker", "per_end_date_qr1"}


class PeadReferenceError(ValueError):
    """The source, primary output, or reference request is invalid."""


class _OfficialVisibleTextParser(HTMLParser):
    """Independent visible-text reader for archived ICE/NYSE HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0 and data.strip():
            self.parts.append(data)


def _plain_json(value: Any) -> Any:
    """Convert supported values to finite, deterministic JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise PeadReferenceError("reference evidence cannot contain NaN or infinity")
        return 0.0 if number == 0.0 else number
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PeadReferenceError("reference evidence keys must be strings")
            normalized[key] = _plain_json(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    raise PeadReferenceError(
        f"unsupported reference evidence value: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise PeadReferenceError(f"{path} must be a SHA-256 digest")
    candidate = value.lower()
    if (
        candidate != value
        or len(candidate) != 64
        or any(character not in _HEX for character in candidate)
    ):
        raise PeadReferenceError(f"{path} must be lowercase SHA-256 hex")
    return candidate


def _date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _canonical_utc_timestamp(value: Any, path: str) -> datetime:
    """Require the exact UTC representation used by source receipts."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadReferenceError(f"{path} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeadReferenceError(f"{path} must be canonical UTC") from exc
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if value != canonical:
        raise PeadReferenceError(f"{path} must be canonical UTC")
    return parsed


def _official_visible_text(raw: bytes) -> str:
    """Normalize official HTML without sharing the acquisition parser."""
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadReferenceError("session source HTML is not UTF-8") from exc
    parser = _OfficialVisibleTextParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as exc:
        raise PeadReferenceError("session source HTML cannot be parsed") from exc
    return " ".join(" ".join(parser.parts).split())


def _official_early_close_dates(visible_text: str) -> list[str]:
    """Independently extract dates scoped to official 13:00 statements."""
    result: set[str] = set()
    for statement in _EARLY_CLOSE_STATEMENT_PATTERN.findall(visible_text):
        for month_name, day_text, year_text in _OFFICIAL_DATE_PATTERN.findall(
            statement
        ):
            month = next(
                number
                for name, number in _MONTH_NUMBERS.items()
                if name.lower() == month_name.lower()
            )
            try:
                parsed = date(int(year_text), month, int(day_text))
            except ValueError as exc:
                raise PeadReferenceError(
                    "official early-close statement contains an invalid date"
                ) from exc
            result.add(parsed.isoformat())
    return sorted(result)


def _provider_utc_timestamp(value: Any) -> str | None:
    """Parse the UTC timestamp representation carried by provider receipts."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return None
    candidate = value.replace(" UTC", "+00:00")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
    ):
        return None
    return value


def _lifecycle_date(value: Any) -> date | None:
    """Accept only the warehouse date contract or canonical ISO equivalent."""
    if type(value) is date:
        return value
    return _date(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _analysts(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    count = int(value)
    return count if count >= 2 else None


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    identifier = row.get("m_ticker")
    period_end = row.get("per_end_date")
    period_type = row.get("per_type")
    if not all(
        isinstance(item, str) and item.strip()
        for item in (identifier, period_end, period_type)
    ):
        return None
    if _date(period_end) is None:
        return None
    return (
        str(identifier).strip().upper(),
        str(period_end),
        str(period_type).strip().upper(),
    )


def _event_key_payload(
    key: tuple[str, str, str] | None, row: Mapping[str, Any]
) -> dict[str, Any]:
    if key is not None:
        return {"m_ticker": key[0], "per_end_date": key[1], "per_type": key[2]}
    return {
        "m_ticker": row.get("m_ticker"),
        "per_end_date": row.get("per_end_date"),
        "per_type": row.get("per_type"),
    }


def _announcement(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    day = _date(row.get("act_rpt_date"))
    clock_text = row.get("act_rpt_time")
    code = row.get("act_rpt_code")
    if day is None or not isinstance(clock_text, str) or not isinstance(code, str):
        return None, "missing_or_invalid_report_timestamp"
    parsed: time | None = None
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(clock_text, pattern).time()
            break
        except ValueError:
            pass
    if parsed is None:
        return None, "invalid_report_time"
    category = code.strip().upper()
    if category not in {"BTO", "DTM", "AMC"}:
        return None, "invalid_report_code"
    minute = parsed.hour * 60 + parsed.minute
    agrees = {
        "BTO": minute < 570,
        "DTM": 570 <= minute < 960,
        "AMC": minute >= 960,
    }[category]
    if not agrees:
        return None, "report_time_code_mismatch"
    instant = datetime.combine(day, parsed, tzinfo=EASTERN).astimezone(timezone.utc)
    return instant.isoformat(timespec="seconds").replace("+00:00", "Z"), None


def _table_rows(
    payload: Mapping[str, Any], code: str, required_fields: set[str]
) -> list[Mapping[str, Any]]:
    tables = payload.get("tables")
    if not isinstance(tables, Mapping) or code not in tables:
        raise PeadReferenceError(f"immutable snapshot is missing {code}")
    table = tables[code]
    if not isinstance(table, Mapping):
        raise PeadReferenceError(f"{code} must be an object")
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise PeadReferenceError(f"{code} must contain column and row arrays")
    names = [
        item.get("name") if isinstance(item, Mapping) else None for item in columns
    ]
    if any(not isinstance(name, str) or not name for name in names):
        raise PeadReferenceError(f"{code} contains an invalid column descriptor")
    if len(names) != len(set(names)):
        raise PeadReferenceError(f"{code} contains duplicate columns")
    missing = sorted(required_fields - set(names))
    if missing:
        raise PeadReferenceError(f"{code} is missing required columns: {missing}")
    expected = set(names)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise PeadReferenceError(
                f"{code} row {index} is not keyed by its exact column set"
            )
    return list(rows)


def validate_snapshot(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Independently validate the content address and signal-source contract."""
    if not isinstance(document, Mapping) or set(document) != {"artifact_hash", "payload"}:
        raise PeadReferenceError("snapshot must contain artifact_hash and payload only")
    payload = document["payload"]
    if not isinstance(payload, Mapping):
        raise PeadReferenceError("snapshot payload must be an object")
    claimed = _sha256(document["artifact_hash"], "snapshot artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadReferenceError("snapshot artifact hash mismatch")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise PeadReferenceError("unsupported Zacks snapshot schema")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadReferenceError("snapshot belongs to another candidate")
    if payload.get("source_id") != SOURCE_ID:
        raise PeadReferenceError("snapshot is not the independent Zacks source")
    _table_rows(payload, "ZACKS/ES", _EPS_ACTUAL_FIELDS)
    _table_rows(payload, "ZACKS/EEH", _EPS_HISTORY_FIELDS)
    return payload


def _reconstruct_pair(
    actual_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    consensus_abs_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Reference implementation of actual-to-vintage matching.

    The implementation builds a keyed, date-indexed vintage ledger first and
    then resolves each unique actual event.  It does not call or wrap the
    primary normalizer.
    """
    actual_name = f"{prefix}_act"
    adjustment_name = f"{prefix}_act_zacks_adj"
    estimate_name = f"{prefix}_mean_est"
    count_name = f"{prefix}_cnt_est"

    vintage_ledger: dict[tuple[str, str, str], dict[date, list[Mapping[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for vintage in history_rows:
        key = _event_key(vintage)
        observed = _date(vintage.get("obs_date"))
        if key is not None and observed is not None:
            vintage_ledger[key][observed].append(vintage)

    actual_ledger: dict[tuple[str, str, str] | None, list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in actual_rows:
        actual_ledger[_event_key(row)].append(row)

    events: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    ordered_keys = sorted(actual_ledger, key=canonical_json)
    for key in ordered_keys:
        group = actual_ledger[key]
        row = group[0]
        problems: set[str] = set()
        if key is None:
            problems.add("invalid_event_key")
        elif key[2] != "Q":
            problems.add("non_quarterly_period")
        if len(group) != 1:
            problems.add("duplicate_actual_key")

        report_day = _date(row.get("act_rpt_date"))
        timestamp, timestamp_problem = _announcement(row)
        if timestamp_problem:
            problems.add(timestamp_problem)
        currency_value = row.get("currency_code")
        currency = (
            currency_value.strip().upper()
            if isinstance(currency_value, str) and currency_value.strip()
            else None
        )
        if currency is None:
            problems.add("missing_currency")
        elif currency != "USD":
            problems.add("non_usd_currency")

        actual = _number(row.get(actual_name))
        embedded_mean = _number(row.get(estimate_name))
        embedded_count = _analysts(row.get(count_name))
        if actual is None:
            problems.add("nonfinite_actual")
        if embedded_mean is None:
            problems.add("nonfinite_surprise_table_consensus")
        if embedded_count is None:
            problems.add("insufficient_surprise_table_analyst_count")

        chosen: Mapping[str, Any] | None = None
        chosen_day: date | None = None
        if key is None or report_day is None:
            problems.add("invalid_actual_report_date")
        else:
            eligible_days = [
                observed
                for observed in vintage_ledger.get(key, {})
                if observed < report_day
            ]
            if not eligible_days:
                problems.add("missing_strictly_prior_consensus")
            else:
                chosen_day = max(eligible_days)
                candidates = vintage_ledger[key][chosen_day]
                if len(candidates) != 1:
                    problems.add("duplicate_latest_consensus_vintage")
                else:
                    chosen = candidates[0]

        vintage_mean = vintage_count = None
        if chosen is not None:
            vintage_mean = _number(chosen.get(estimate_name))
            vintage_count = _analysts(chosen.get(count_name))
            if vintage_mean is None:
                problems.add("nonfinite_vintage_consensus")
            if vintage_count is None:
                problems.add("insufficient_vintage_analyst_count")
            vintage_currency = chosen.get("currency_code")
            if not isinstance(vintage_currency, str) or currency is None:
                problems.add("consensus_currency_missing")
            elif vintage_currency.strip().upper() != currency:
                problems.add("consensus_currency_mismatch")
            actual_ticker = str(row.get("ticker", "")).strip().upper()
            vintage_ticker = str(chosen.get("ticker", "")).strip().upper()
            if not actual_ticker or actual_ticker != vintage_ticker:
                problems.add("ticker_mapping_mismatch")
        if embedded_mean is not None and vintage_mean is not None:
            if abs(embedded_mean - vintage_mean) > consensus_abs_tolerance:
                problems.add("surprise_consensus_crosscheck_mismatch")

        key_payload = _event_key_payload(key, row)
        if problems:
            exclusions.append(
                {
                    "event_key": key_payload,
                    "ticker": row.get("ticker"),
                    "reasons": sorted(problems),
                }
            )
            continue

        assert report_day is not None and timestamp is not None
        assert actual is not None and embedded_mean is not None
        assert embedded_count is not None and vintage_mean is not None
        assert vintage_count is not None and chosen_day is not None
        events.append(
            {
                "event_key": key_payload,
                "ticker": str(row["ticker"]).strip().upper(),
                "currency_code": str(currency),
                "act_rpt_date": report_day.isoformat(),
                "announcement_at_utc": timestamp,
                "act_rpt_time": row["act_rpt_time"],
                "act_rpt_code": str(row["act_rpt_code"]).strip().upper(),
                "actual": actual,
                "zacks_adjustment_diagnostic": _number(row.get(adjustment_name)),
                "consensus": vintage_mean,
                "consensus_obs_date": chosen_day.isoformat(),
                "consensus_analyst_count": vintage_count,
                "surprise_table_consensus": embedded_mean,
                "surprise_table_analyst_count": embedded_count,
                "consensus_crosscheck_absolute_difference": abs(
                    embedded_mean - vintage_mean
                ),
                "unscaled_forecast_error": actual - vintage_mean,
            }
        )
    return events, exclusions, {
        "actual_rows": len(actual_rows),
        "consensus_vintage_rows": len(history_rows),
        "matched_events": len(events),
        "excluded_actual_events": len(exclusions),
    }


def reconstruct_events(
    snapshot_document: Mapping[str, Any], *, consensus_abs_tolerance: float
) -> dict[str, Any]:
    tolerance = float(consensus_abs_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise PeadReferenceError("consensus tolerance must be finite and non-negative")
    payload = validate_snapshot(snapshot_document)
    eps, eps_exclusions, eps_counts = _reconstruct_pair(
        _table_rows(payload, "ZACKS/ES", _EPS_ACTUAL_FIELDS),
        _table_rows(payload, "ZACKS/EEH", _EPS_HISTORY_FIELDS),
        prefix="eps",
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
    tables = payload.get("tables")
    assert isinstance(tables, Mapping)
    if "ZACKS/SS" in tables or "ZACKS/SEH" in tables:
        if not {"ZACKS/SS", "ZACKS/SEH"}.issubset(tables):
            raise PeadReferenceError("sales reconstruction requires SS and SEH together")
        sales, sales_exclusions, sales_counts = _reconstruct_pair(
            _table_rows(payload, "ZACKS/SS", _SALES_ACTUAL_FIELDS),
            _table_rows(payload, "ZACKS/SEH", _SALES_HISTORY_FIELDS),
            prefix="sales",
            consensus_abs_tolerance=tolerance,
        )
    sales_by_key = {canonical_json(item["event_key"]): item for item in sales}
    for event in eps:
        event["sales_diagnostic"] = sales_by_key.get(
            canonical_json(event["event_key"])
        )
    identity_diagnostics: dict[str, Any] = {
        "available": "ZACKS/MT" in tables,
        "validated_events": 0,
        "invalid_events": [],
    }
    if "ZACKS/MT" in tables:
        identities: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in _table_rows(payload, "ZACKS/MT", _MT_FIELDS):
            value = row.get("m_ticker")
            if isinstance(value, str) and value.strip():
                identities[value.strip().upper()].append(row)
        for event in eps:
            rows = identities.get(event["event_key"]["m_ticker"], [])
            reasons: list[str] = []
            if len(rows) != 1:
                reasons.append("missing_or_duplicate_mt_identity")
            else:
                identity = rows[0]
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
            (
                str(row.get("m_ticker", "")).strip().upper(),
                row.get("per_end_date_qr1"),
            )
            for row in _table_rows(payload, "ZACKS/EA", _EA_FIELDS)
        }
        schedule_diagnostics["matched_event_keys"] = sum(
            (
                event["event_key"]["m_ticker"],
                event["event_key"]["per_end_date"],
            )
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


def _validated_session_calendar(
    document: Any, *, required_start: str, required_end: str
) -> dict[str, Any]:
    """Independently validate official NYSE close-time evidence."""
    if not isinstance(document, Mapping) or set(document) != {
        "artifact_hash", "payload"
    }:
        raise PeadReferenceError("session close calendar wrapper is malformed")
    payload = document["payload"]
    expected_payload = {
        "schema_version", "venue", "timezone", "coverage",
        "regular_close_local_time", "early_close_local_time",
        "observed_session_rule", "early_close_sessions", "sources",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_payload:
        raise PeadReferenceError("session close calendar payload is malformed")
    fixed_values = {
        "schema_version": SESSION_CALENDAR_SCHEMA_VERSION,
        "venue": "NYSE cash equities",
        "timezone": "America/New_York",
        "regular_close_local_time": "16:00:00",
        "early_close_local_time": "13:00:00",
        "observed_session_rule": _OBSERVED_SESSION_RULE,
    }
    if any(payload.get(field) != expected for field, expected in fixed_values.items()):
        raise PeadReferenceError("session close calendar convention changed")

    coverage = payload["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != {"start", "end"}:
        raise PeadReferenceError("session close calendar coverage is malformed")
    first = _date(coverage["start"])
    last = _date(coverage["end"])
    requested_first = _date(required_start)
    requested_last = _date(required_end)
    if (
        first is None or last is None or requested_first is None
        or requested_last is None or first > last
        or requested_first > requested_last
    ):
        raise PeadReferenceError("session close calendar dates are invalid")
    if first > requested_first or last < requested_last:
        raise PeadReferenceError("session close calendar does not span the request")

    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PeadReferenceError("session close calendar has no sources")
    sources: dict[str, set[int]] = {}
    source_order: list[str] = []
    for source in raw_sources:
        if not isinstance(source, Mapping) or set(source) != {
            "source_id", "publisher", "url", "covered_years"
        }:
            raise PeadReferenceError("session close calendar source is malformed")
        source_id = source["source_id"]
        publisher = source["publisher"]
        url = source["url"]
        years = source["covered_years"]
        if not all(
            isinstance(value, str) and value.strip() and value == value.strip()
            for value in (source_id, publisher, url)
        ):
            raise PeadReferenceError("session close calendar source text is invalid")
        if not (
            url.startswith("https://ir.theice.com/")
            or url.startswith("https://www.nyse.com/")
        ):
            raise PeadReferenceError("session close calendar source is not ICE/NYSE")
        if (
            not isinstance(years, list) or not years
            or any(type(year) is not int for year in years)
            or years != sorted(set(years))
        ):
            raise PeadReferenceError("session close calendar source years are invalid")
        source_order.append(source_id)
        sources[source_id] = set(years)
    if source_order != sorted(set(source_order)) or len(sources) != len(raw_sources):
        raise PeadReferenceError("session close calendar sources are not uniquely sorted")
    covered_years = set().union(*sources.values())
    if not set(range(first.year, last.year + 1)).issubset(covered_years):
        raise PeadReferenceError("session close calendar source years are incomplete")

    raw_early = payload["early_close_sessions"]
    if not isinstance(raw_early, list):
        raise PeadReferenceError("session close calendar early closes are malformed")
    early_order: list[str] = []
    for row in raw_early:
        if not isinstance(row, Mapping) or set(row) != {"date", "source_id"}:
            raise PeadReferenceError("session close calendar early row is malformed")
        early_day = _date(row["date"])
        source_id = row["source_id"]
        if (
            early_day is None or not isinstance(source_id, str)
            or source_id not in sources or early_day.year not in sources[source_id]
            or not first <= early_day <= last
        ):
            raise PeadReferenceError("session close calendar early row is unproved")
        early_order.append(early_day.isoformat())
    if early_order != sorted(set(early_order)):
        raise PeadReferenceError("session close calendar early dates are not unique")
    claimed = _sha256(document["artifact_hash"], "session close calendar hash")
    if claimed != content_hash(payload):
        raise PeadReferenceError("session close calendar hash mismatch")
    return {"artifact_hash": claimed, "payload": _plain_json(payload)}


def _validated_bound_session_close_evidence(
    document: Any, *, required_start: str, required_end: str
) -> dict[str, Any]:
    """Verify the compact calendar/receipt binding stored in artifacts."""
    if not isinstance(document, Mapping) or set(document) != {
        "artifact_hash", "payload"
    }:
        raise PeadReferenceError("bound session close evidence is malformed")
    payload = document["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "calendar", "source_receipt"
    }:
        raise PeadReferenceError("bound session close evidence payload is malformed")
    if payload["schema_version"] != SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION:
        raise PeadReferenceError("unsupported bound session close evidence schema")
    evidence_hash = _sha256(
        document["artifact_hash"], "bound session close evidence artifact_hash"
    )
    if evidence_hash != content_hash(payload):
        raise PeadReferenceError("bound session close evidence hash mismatch")
    calendar = _validated_session_calendar(
        payload["calendar"],
        required_start=required_start,
        required_end=required_end,
    )
    calendar_payload = calendar["payload"]
    first = date.fromisoformat(calendar_payload["coverage"]["start"])
    last = date.fromisoformat(calendar_payload["coverage"]["end"])

    receipt = payload["source_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "artifact_hash", "payload"
    }:
        raise PeadReferenceError("bound session source receipt is malformed")
    receipt_payload = receipt["payload"]
    if not isinstance(receipt_payload, Mapping) or set(receipt_payload) != {
        "schema_version", "calendar_artifact_hash", "created_at_utc", "sources"
    }:
        raise PeadReferenceError("bound session source receipt payload is malformed")
    if (
        receipt_payload["schema_version"] != SESSION_SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt_payload["calendar_artifact_hash"] != calendar["artifact_hash"]
    ):
        raise PeadReferenceError("bound session source receipt identity differs")
    _canonical_utc_timestamp(
        receipt_payload["created_at_utc"], "bound source receipt created_at_utc"
    )
    receipt_hash = _sha256(
        receipt["artifact_hash"], "bound session source receipt artifact_hash"
    )
    if receipt_hash != content_hash(receipt_payload):
        raise PeadReferenceError("bound session source receipt hash mismatch")

    calendar_sources = calendar_payload["sources"]
    receipt_sources = receipt_payload["sources"]
    if (
        not isinstance(receipt_sources, list)
        or len(receipt_sources) != len(calendar_sources)
    ):
        raise PeadReferenceError("bound session source coverage differs")
    extracted_by_source: dict[str, set[str]] = {}
    for index, (calendar_source, entry) in enumerate(
        zip(calendar_sources, receipt_sources, strict=True)
    ):
        if not isinstance(entry, Mapping) or set(entry) != {
            "source_id", "publisher", "url", "covered_years",
            "retrieved_at_utc", "http", "raw_document", "extraction",
        }:
            raise PeadReferenceError(
                f"bound session source receipt entry {index} is malformed"
            )
        if any(
            entry[field] != calendar_source[field]
            for field in ("source_id", "publisher", "url", "covered_years")
        ):
            raise PeadReferenceError(
                f"bound session source receipt entry {index} differs"
            )
        source_id = entry["source_id"]
        _canonical_utc_timestamp(
            entry["retrieved_at_utc"],
            f"bound session source {source_id} retrieved_at_utc",
        )
        http = entry["http"]
        if not isinstance(http, Mapping) or set(http) != {
            "status_code", "date_utc", "content_type", "etag",
            "last_modified_utc",
        }:
            raise PeadReferenceError(
                f"bound session source {source_id} HTTP metadata is malformed"
            )
        if type(http["status_code"]) is not int or http["status_code"] != 200:
            raise PeadReferenceError(
                f"bound session source {source_id} HTTP status differs"
            )
        if not isinstance(http["content_type"], str) or "html" not in http[
            "content_type"
        ].lower():
            raise PeadReferenceError(
                f"bound session source {source_id} content type differs"
            )
        if http["etag"] is not None and not isinstance(http["etag"], str):
            raise PeadReferenceError(
                f"bound session source {source_id} ETag is malformed"
            )
        for field in ("date_utc", "last_modified_utc"):
            if http[field] is not None:
                _canonical_utc_timestamp(
                    http[field], f"bound session source {source_id} HTTP {field}"
                )
        raw_document = entry["raw_document"]
        if not isinstance(raw_document, Mapping) or set(raw_document) != {
            "relative_path", "sha256", "bytes"
        }:
            raise PeadReferenceError(
                f"bound session source {source_id} raw metadata is malformed"
            )
        raw_hash = _sha256(
            raw_document["sha256"], f"bound session source {source_id} raw hash"
        )
        if (
            raw_document["relative_path"] != f"raw/{raw_hash}.html"
            or type(raw_document["bytes"]) is not int
            or not 0 < raw_document["bytes"] <= MAX_SESSION_SOURCE_BYTES
        ):
            raise PeadReferenceError(
                f"bound session source {source_id} raw metadata differs"
            )
        extraction = entry["extraction"]
        if not isinstance(extraction, Mapping) or set(extraction) != {
            "method", "normalized_text_sha256", "early_close_dates"
        }:
            raise PeadReferenceError(
                f"bound session source {source_id} extraction is malformed"
            )
        if extraction["method"] != SESSION_SOURCE_EXTRACTION_METHOD:
            raise PeadReferenceError(
                f"bound session source {source_id} extraction method differs"
            )
        _sha256(
            extraction["normalized_text_sha256"],
            f"bound session source {source_id} visible-text hash",
        )
        dates = extraction["early_close_dates"]
        if (
            not isinstance(dates, list)
            or any(_date(value) is None for value in dates)
            or dates != sorted(set(dates))
        ):
            raise PeadReferenceError(
                f"bound session source {source_id} extracted dates are malformed"
            )
        extracted_by_source[source_id] = set(dates)

    calendar_dates = {
        row["date"] for row in calendar_payload["early_close_sessions"]
    }
    if any(
        row["date"] not in extracted_by_source[row["source_id"]]
        for row in calendar_payload["early_close_sessions"]
    ):
        raise PeadReferenceError("bound calendar row is not proved by its source")
    extracted_union: set[str] = set()
    for source in calendar_sources:
        years = set(source["covered_years"])
        extracted_union.update(
            value
            for value in extracted_by_source[source["source_id"]]
            if first <= date.fromisoformat(value) <= last
            and date.fromisoformat(value).year in years
        )
    if extracted_union != calendar_dates:
        raise PeadReferenceError("bound calendar differs from source-date union")
    return {
        "artifact_hash": evidence_hash,
        "payload": {
            "schema_version": SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION,
            "calendar": calendar,
            "source_receipt": {
                "artifact_hash": receipt_hash,
                "payload": _plain_json(receipt_payload),
            },
        },
    }


def _validated_session_close_evidence(
    document: Any, *, required_start: str, required_end: str
) -> dict[str, Any]:
    """Validate raw official sources independently and bind their calendar."""
    if not isinstance(document, Mapping) or set(document) != {
        "calendar", "source_receipt", "source_documents"
    }:
        raise PeadReferenceError("session close evidence bundle is malformed")
    calendar = _validated_session_calendar(
        document["calendar"],
        required_start=required_start,
        required_end=required_end,
    )
    calendar_payload = calendar["payload"]
    first = date.fromisoformat(calendar_payload["coverage"]["start"])
    last = date.fromisoformat(calendar_payload["coverage"]["end"])
    calendar_sources = calendar_payload["sources"]

    receipt = document["source_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "artifact_hash", "payload"
    }:
        raise PeadReferenceError("session source receipt wrapper is malformed")
    receipt_payload = receipt["payload"]
    if not isinstance(receipt_payload, Mapping) or set(receipt_payload) != {
        "schema_version", "calendar_artifact_hash", "created_at_utc", "sources"
    }:
        raise PeadReferenceError("session source receipt payload is malformed")
    if receipt_payload["schema_version"] != SESSION_SOURCE_RECEIPT_SCHEMA_VERSION:
        raise PeadReferenceError("unsupported session source receipt schema")
    if receipt_payload["calendar_artifact_hash"] != calendar["artifact_hash"]:
        raise PeadReferenceError("session source receipt binds another calendar")
    created_at = _canonical_utc_timestamp(
        receipt_payload["created_at_utc"], "session source receipt created_at_utc"
    )
    receipt_hash = _sha256(
        receipt["artifact_hash"], "session source receipt artifact_hash"
    )
    if receipt_hash != content_hash(receipt_payload):
        raise PeadReferenceError("session source receipt hash mismatch")

    receipt_sources = receipt_payload["sources"]
    source_documents = document["source_documents"]
    if (
        not isinstance(receipt_sources, list)
        or len(receipt_sources) != len(calendar_sources)
        or not isinstance(source_documents, Mapping)
    ):
        raise PeadReferenceError("session source evidence coverage is malformed")
    calendar_source_ids = [source["source_id"] for source in calendar_sources]
    if set(source_documents) != set(calendar_source_ids):
        raise PeadReferenceError("session source documents do not exactly cover sources")

    extracted_by_source: dict[str, set[str]] = {}
    visible_by_source: dict[str, str] = {}
    for index, (calendar_source, entry) in enumerate(
        zip(calendar_sources, receipt_sources, strict=True)
    ):
        if not isinstance(entry, Mapping) or set(entry) != {
            "source_id", "publisher", "url", "covered_years",
            "retrieved_at_utc", "http", "raw_document", "extraction",
        }:
            raise PeadReferenceError(
                f"session source receipt entry {index} is malformed"
            )
        for field in ("source_id", "publisher", "url", "covered_years"):
            if entry[field] != calendar_source[field]:
                raise PeadReferenceError(
                    f"session source receipt entry {index} differs on {field}"
                )
        source_id = entry["source_id"]
        retrieved_at = _canonical_utc_timestamp(
            entry["retrieved_at_utc"],
            f"session source {source_id} retrieved_at_utc",
        )
        if retrieved_at > created_at:
            raise PeadReferenceError(
                f"session source {source_id} was retrieved after receipt creation"
            )
        if (
            created_at - retrieved_at
        ).total_seconds() > MAX_SESSION_SOURCE_RECEIPT_DURATION_SECONDS:
            raise PeadReferenceError(
                f"session source {source_id} receipt duration is implausible"
            )

        http = entry["http"]
        if not isinstance(http, Mapping) or set(http) != {
            "status_code", "date_utc", "content_type", "etag",
            "last_modified_utc",
        }:
            raise PeadReferenceError(
                f"session source {source_id} HTTP metadata is malformed"
            )
        if type(http["status_code"]) is not int or http["status_code"] != 200:
            raise PeadReferenceError(
                f"session source {source_id} was not acquired with HTTP 200"
            )
        content_type = http["content_type"]
        if (
            not isinstance(content_type, str)
            or content_type != content_type.strip()
            or "html" not in content_type.lower()
        ):
            raise PeadReferenceError(
                f"session source {source_id} content type is not HTML"
            )
        etag = http["etag"]
        if etag is not None and (
            not isinstance(etag, str) or not etag or etag != etag.strip()
        ):
            raise PeadReferenceError(f"session source {source_id} ETag is invalid")
        if http["date_utc"] is None:
            raise PeadReferenceError(
                f"session source {source_id} HTTP Date is required"
            )
        server_date = _canonical_utc_timestamp(
            http["date_utc"], f"session source {source_id} HTTP date_utc"
        )
        if abs(
            (server_date - retrieved_at).total_seconds()
        ) > MAX_SESSION_SOURCE_CLOCK_SKEW_SECONDS:
            raise PeadReferenceError(
                f"session source {source_id} HTTP Date clock skew is implausible"
            )
        last_modified = http["last_modified_utc"]
        if last_modified is not None and _canonical_utc_timestamp(
            last_modified,
            f"session source {source_id} HTTP last_modified_utc",
        ) > server_date:
            raise PeadReferenceError(
                f"session source {source_id} HTTP Last-Modified follows Date"
            )

        raw_document = entry["raw_document"]
        if not isinstance(raw_document, Mapping) or set(raw_document) != {
            "relative_path", "sha256", "bytes"
        }:
            raise PeadReferenceError(
                f"session source {source_id} raw-document metadata is malformed"
            )
        raw_hash = _sha256(
            raw_document["sha256"], f"session source {source_id} raw SHA-256"
        )
        if raw_document["relative_path"] != f"raw/{raw_hash}.html":
            raise PeadReferenceError(
                f"session source {source_id} path is not content-addressed"
            )
        byte_count = raw_document["bytes"]
        if (
            type(byte_count) is not int
            or not 0 < byte_count <= MAX_SESSION_SOURCE_BYTES
        ):
            raise PeadReferenceError(
                f"session source {source_id} byte count is invalid"
            )
        encoded = source_documents[source_id]
        if not isinstance(encoded, str) or not encoded:
            raise PeadReferenceError(
                f"session source {source_id} base64 document is invalid"
            )
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise PeadReferenceError(
                f"session source {source_id} base64 document is invalid"
            ) from exc
        if base64.b64encode(raw).decode("ascii") != encoded:
            raise PeadReferenceError(
                f"session source {source_id} base64 is not canonical"
            )
        if (
            len(raw) != byte_count
            or hashlib.sha256(raw).hexdigest() != raw_hash
        ):
            raise PeadReferenceError(
                f"session source {source_id} raw document hash differs"
            )

        visible = _official_visible_text(raw)
        if not visible:
            raise PeadReferenceError(
                f"session source {source_id} has no visible official text"
            )
        extraction = entry["extraction"]
        if not isinstance(extraction, Mapping) or set(extraction) != {
            "method", "normalized_text_sha256", "early_close_dates"
        }:
            raise PeadReferenceError(
                f"session source {source_id} extraction is malformed"
            )
        if extraction["method"] != SESSION_SOURCE_EXTRACTION_METHOD:
            raise PeadReferenceError(
                f"session source {source_id} extraction method changed"
            )
        normalized_hash = _sha256(
            extraction["normalized_text_sha256"],
            f"session source {source_id} visible-text SHA-256",
        )
        if hashlib.sha256(visible.encode("utf-8")).hexdigest() != normalized_hash:
            raise PeadReferenceError(
                f"session source {source_id} visible-text hash differs"
            )
        extracted_dates = extraction["early_close_dates"]
        if (
            not isinstance(extracted_dates, list)
            or any(_date(value) is None for value in extracted_dates)
            or extracted_dates != sorted(set(extracted_dates))
            or extracted_dates != _official_early_close_dates(visible)
        ):
            raise PeadReferenceError(
                f"session source {source_id} early-close extraction differs"
            )
        extracted_by_source[source_id] = set(extracted_dates)
        visible_by_source[source_id] = visible

    core_hours = visible_by_source.get("nyse-core-hours")
    if core_hours is None or _CORE_HOURS_PATTERN.search(core_hours) is None:
        raise PeadReferenceError(
            "NYSE source does not prove the regular 16:00 core-session close"
        )

    calendar_dates = {
        row["date"] for row in calendar_payload["early_close_sessions"]
    }
    for row in calendar_payload["early_close_sessions"]:
        if row["date"] not in extracted_by_source[row["source_id"]]:
            raise PeadReferenceError(
                "calendar early close is not proved by its named official source"
            )
    extracted_union: set[str] = set()
    for source in calendar_sources:
        years = set(source["covered_years"])
        extracted_union.update(
            value
            for value in extracted_by_source[source["source_id"]]
            if first <= date.fromisoformat(value) <= last
            and date.fromisoformat(value).year in years
        )
    if extracted_union != calendar_dates:
        raise PeadReferenceError(
            "calendar early closes differ from archived official source union"
        )

    payload = {
        "schema_version": SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION,
        "calendar": calendar,
        "source_receipt": {
            "artifact_hash": receipt_hash,
            "payload": _plain_json(receipt_payload),
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _session_closes(
    evidence: Mapping[str, Any],
    sessions: Sequence[Any],
    *,
    required_start: str,
    required_end: str,
) -> dict[pd.Timestamp, datetime]:
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != {"artifact_hash", "payload"}
        or not isinstance(evidence["payload"], Mapping)
        or set(evidence["payload"]) != {
            "schema_version", "calendar", "source_receipt"
        }
        or evidence["payload"]["schema_version"]
        != SESSION_CLOSE_EVIDENCE_SCHEMA_VERSION
        or evidence["artifact_hash"] != content_hash(evidence["payload"])
    ):
        raise PeadReferenceError("validated session close evidence is malformed")
    verified = _validated_session_calendar(
        evidence["payload"]["calendar"],
        required_start=required_start,
        required_end=required_end,
    )
    ordered: list[pd.Timestamp] = []
    for value in sessions:
        session = pd.Timestamp(value)
        if pd.isna(session) or session.tzinfo is not None or session != session.normalize():
            raise PeadReferenceError("observed sessions must be timezone-naive dates")
        ordered.append(session)
    if ordered != sorted(set(ordered)):
        raise PeadReferenceError("observed sessions are not unique and sorted")
    first = date.fromisoformat(required_start)
    last = date.fromisoformat(required_end)
    if any(not first <= value.date() <= last for value in ordered):
        raise PeadReferenceError("observed session is outside calendar request")
    observed = {value.date() for value in ordered}
    early = {
        date.fromisoformat(row["date"])
        for row in verified["payload"]["early_close_sessions"]
        if first <= date.fromisoformat(row["date"]) <= last
    }
    if not early.issubset(observed):
        raise PeadReferenceError(
            "official early close is missing from observed SEP sessions"
        )
    result: dict[pd.Timestamp, datetime] = {}
    for session in ordered:
        wall_clock = time(13) if session.date() in early else time(16)
        result[session] = datetime.combine(
            session.date(), wall_clock, tzinfo=EASTERN
        ).astimezone(timezone.utc)
    return result


def _bulk_prices(
    provider: Any, names: Sequence[str], start: str, end: str, field: str
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    distinct = sorted(set(names))
    strict_bulk = getattr(provider, "prices_bulk_strict", None)
    if callable(strict_bulk):
        return strict_bulk(distinct, start, end, field=field)
    strict_single = getattr(provider, "prices_strict", None)
    if callable(strict_single):
        return {
            name: strict_single(name, start, end, field)
            for name in distinct
        }
    bulk = getattr(provider, "prices_bulk", None)
    if callable(bulk):
        for offset in range(0, len(distinct), 250):
            result.update(bulk(distinct[offset : offset + 250], start, end, field=field))
        return result
    for name in distinct:
        result[name] = provider.prices(name, start, end, field=field)
    return result


def _first_observed_session_per_month(
    sessions: pd.DatetimeIndex, start: str, end: str
) -> list[pd.Timestamp]:
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    observed = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in sessions
            if lo <= pd.Timestamp(value).normalize() <= hi
        }
    )
    first: dict[tuple[int, int], pd.Timestamp] = {}
    for session in observed:
        first.setdefault((session.year, session.month), session)
    return [first[key] for key in sorted(first)]


def _numeric_series(series: Any) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


_REFERENCE_ACTION_FIELDS = {
    "date", "action", "ticker", "name", "value", "contraticker", "contraname"
}
_REFERENCE_TERMINAL_ACTIONS = {
    "acquisitionby", "bankruptcyliquidation", "delisted", "mergerfrom",
    "mergerto", "regulatorydelisting", "voluntarydelisting",
}
_REFERENCE_UNSUPPORTED_ACTIONS = {
    "adrratiosplit", "relation", "spinoff", "spinoffdividend", "spunofffrom",
    "split", "tickerchangefrom", "tickerchangeto",
}


def _reference_action_slice(
    provider: Any, tickers: Sequence[str], start: str, end: str
) -> list[dict[str, Any]]:
    reader = getattr(provider, "corporate_actions_for_tickers", None)
    if not callable(reader):
        raise PeadReferenceError("provider cannot expose an exact ACTIONS slice")
    raw_rows = reader(list(tickers), start, end)
    if not isinstance(raw_rows, list):
        raise PeadReferenceError("ACTIONS slice must be an array")
    allowed = set(tickers)
    lower = date.fromisoformat(start)
    upper = date.fromisoformat(end)
    rows: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != _REFERENCE_ACTION_FIELDS:
            raise PeadReferenceError(f"ACTIONS row {index} is malformed")
        try:
            action_date = date.fromisoformat(raw["date"])
        except (TypeError, ValueError) as exc:
            raise PeadReferenceError(f"ACTIONS row {index} date is invalid") from exc
        ticker = raw["ticker"]
        action = raw["action"]
        name = raw["name"]
        if (
            action_date.isoformat() != raw["date"]
            or not lower <= action_date <= upper
            or ticker not in allowed
            or not isinstance(action, str)
            or not action
            or action != action.strip().lower()
            or not isinstance(name, str)
            or not name
            or name != name.strip()
        ):
            raise PeadReferenceError(f"ACTIONS row {index} key is invalid")
        value = None if raw["value"] is None else _number(raw["value"])
        if raw["value"] is not None and value is None:
            raise PeadReferenceError(f"ACTIONS row {index} value is invalid")
        contra_ticker = raw["contraticker"]
        contra_name = raw["contraname"]
        if any(
            item is not None and (
                not isinstance(item, str) or not item or item != item.strip()
            )
            for item in (contra_ticker, contra_name)
        ):
            raise PeadReferenceError(f"ACTIONS row {index} contra identity is invalid")
        key = (action_date, ticker, name, action, contra_name, contra_ticker)
        if key in keys:
            raise PeadReferenceError("ACTIONS slice contains a duplicate key")
        keys.add(key)
        rows.append(
            {
                "date": action_date.isoformat(),
                "action": action,
                "ticker": ticker,
                "name": name,
                "value": value,
                "contraticker": contra_ticker,
                "contraname": contra_name,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["ticker"], item["date"], item["action"], item["name"],
            item["contraticker"] or "", item["contraname"] or "",
        ),
    )


def _reference_price_map(series: Any) -> dict[date, float]:
    if not isinstance(series, pd.Series):
        return {}
    result: dict[date, float] = {}
    seen: set[date] = set()
    for raw_day, raw_value in series.items():
        timestamp = pd.Timestamp(raw_day)
        if (
            pd.isna(timestamp)
            or timestamp.tzinfo is not None
            or timestamp != timestamp.normalize()
        ):
            raise PeadReferenceError("price panel contains an invalid date")
        day = timestamp.date()
        if day in seen:
            raise PeadReferenceError("price panel contains duplicate dates")
        seen.add(day)
        price = _number(raw_value)
        if price is not None and price > 0:
            result[day] = price
    return result


def _reference_unresolved_economic_return(
    reason: str,
    *,
    entry_price: float,
    exit_price: float | None,
    diagnostic_return: float | None,
    distributions: list[dict[str, Any]],
    ignored_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "unresolved",
        "reason": reason,
        "pricing_path": "SEP.close_plus_explicit_cash_no_reinvestment_candidate",
        "entry_price_split_normalized": entry_price,
        "exit_price_split_normalized": exit_price,
        "cash_distributions": distributions,
        "cash_total": float(sum(item["amount"] for item in distributions)),
        "terminal_settlement_id": None,
        "gross_terminal_value": None,
        "gross_economic_return": None,
        "closeadj_diagnostic_return": diagnostic_return,
        "ignored_actions": ignored_actions,
    }


def _reference_cash_return(
    *,
    ticker: str,
    entry_day: date,
    exit_day: date,
    close: Mapping[date, float],
    adjusted: Mapping[date, float],
    actions: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    currency: str,
    terminal_settlements: Sequence[Mapping[str, Any]],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    if entry_day not in close:
        raise PeadReferenceError("economic path omits the exact entry close")
    entry_price = close[entry_day]
    diagnostic = (
        adjusted[exit_day] / adjusted[entry_day] - 1.0
        if entry_day in adjusted and exit_day in adjusted
        else None
    )
    if lifecycle.get("status") != "validated":
        return _reference_unresolved_economic_return(
            "security_lifecycle_unresolved",
            entry_price=entry_price,
            exit_price=close.get(exit_day),
            diagnostic_return=diagnostic,
            distributions=[],
            ignored_actions=[],
        )
    if currency != "USD":
        return _reference_unresolved_economic_return(
            "security_currency_not_usd_or_unresolved",
            entry_price=entry_price,
            exit_price=close.get(exit_day),
            diagnostic_return=diagnostic,
            distributions=[],
            ignored_actions=[],
        )
    permaticker = lifecycle.get("permaticker")
    if type(permaticker) is not int or permaticker <= 0:
        raise PeadReferenceError("economic path has an invalid permaticker")
    try:
        final_day = date.fromisoformat(lifecycle["lastpricedate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PeadReferenceError("economic path has an invalid final date") from exc

    interval_actions = sorted(
        (
            dict(item)
            for item in actions
            if item["ticker"] == ticker
            and entry_day < date.fromisoformat(item["date"]) <= exit_day
        ),
        key=lambda item: (
            item["date"], item["action"], item["name"],
            item.get("contraticker") or "", item.get("contraname") or "",
        ),
    )
    distributions: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    terminal_actions: list[dict[str, Any]] = []
    dividend_actions: dict[date, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for item in interval_actions:
        kind = item["action"]
        if kind == "acquisitionof":
            ignored.append(
                {
                    "date": item["date"],
                    "action": kind,
                    "contraticker": item.get("contraticker"),
                    "reason": "issuer_external_acquisition_no_direct_holder_cash_flow",
                }
            )
            continue
        if kind in _REFERENCE_TERMINAL_ACTIONS:
            terminal_actions.append(item)
            continue
        if kind in _REFERENCE_UNSUPPORTED_ACTIONS or kind != "dividend":
            return _reference_unresolved_economic_return(
                f"held_corporate_action_terms_unresolved:{kind}",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                diagnostic_return=diagnostic,
                distributions=distributions,
                ignored_actions=ignored,
            )
        amount = _number(item["value"])
        if amount is None or amount <= 0:
            raise PeadReferenceError("dividend amount is not positive finite")
        ex_day = date.fromisoformat(item["date"])
        dividend_actions[ex_day].append((item, amount))

    # One closeadj transition represents the total distribution adjustment on
    # an ex-date. Validate regular/special components as a sum, then preserve
    # each provider action and key in the independent evidence.
    for ex_day in sorted(dividend_actions):
        actions_on_day = dividend_actions[ex_day]
        aggregate_amount = sum(amount for _, amount in actions_on_day)
        if ex_day not in close or ex_day not in adjusted:
            return _reference_unresolved_economic_return(
                "dividend_date_missing_exact_price_adjustment_evidence",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                diagnostic_return=diagnostic,
                distributions=distributions,
                ignored_actions=ignored,
            )
        prior = [day for day in close if day in adjusted and day < ex_day]
        if not prior:
            return _reference_unresolved_economic_return(
                "dividend_missing_prior_price_adjustment_evidence",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                diagnostic_return=diagnostic,
                distributions=distributions,
                ignored_actions=ignored,
            )
        previous = max(prior)
        implied = close[previous] * adjusted[ex_day] / adjusted[previous] - close[ex_day]
        error = abs(implied - aggregate_amount)
        allowed = absolute_tolerance + relative_tolerance * max(
            abs(implied), abs(aggregate_amount)
        )
        if not math.isfinite(implied) or error > allowed:
            return _reference_unresolved_economic_return(
                "cash_distribution_split_basis_or_amount_mismatch",
                entry_price=entry_price,
                exit_price=close.get(exit_day),
                diagnostic_return=diagnostic,
                distributions=distributions,
                ignored_actions=ignored,
            )
        for item, amount in actions_on_day:
            distributions.append(
                {
                    "action_key": {
                        "date": item["date"],
                        "ticker": ticker,
                        "name": item["name"],
                        "action": item["action"],
                        "contraname": item.get("contraname"),
                        "contraticker": item.get("contraticker"),
                    },
                    "date": item["date"],
                    "amount": amount,
                    "adjustment_previous_session": previous.isoformat(),
                    "adjustment_implied_amount": implied,
                    "adjustment_absolute_error": error,
                    "adjustment_allowed_error": allowed,
                }
            )

    delisted = lifecycle.get("isdelisted") == "Y" and final_day < exit_day
    terminal_id = None
    terminal_cash = None
    exit_price = close.get(exit_day)
    if delisted or terminal_actions:
        matches = [
            item
            for item in terminal_settlements
            if item.get("ticker") == ticker
            and item.get("permaticker") == permaticker
            and item.get("last_price_date") == final_day.isoformat()
        ]
        if len(matches) != 1:
            return _reference_unresolved_economic_return(
                "held_terminal_settlement_missing_or_ambiguous",
                entry_price=entry_price,
                exit_price=None,
                diagnostic_return=diagnostic,
                distributions=distributions,
                ignored_actions=ignored,
            )
        settlement = matches[0]
        settlement_day = date.fromisoformat(settlement["settlement_date"])
        if settlement_day > exit_day:
            return _reference_unresolved_economic_return(
                "terminal_cash_not_received_by_exact_horizon",
                entry_price=entry_price,
                exit_price=None,
                diagnostic_return=diagnostic,
                distributions=distributions,
                ignored_actions=ignored,
            )
        terminal_cash = _number(settlement["cash_per_terminal_share"])
        if terminal_cash is None or terminal_cash <= 0:
            raise PeadReferenceError("terminal cash is not positive finite")
        terminal_id = (
            f"{permaticker}:{final_day.isoformat()}:"
            f"{settlement_day.isoformat()}"
        )
        exit_price = None
    elif exit_price is None:
        return _reference_unresolved_economic_return(
            "missing_exact_split_normalized_exit_price",
            entry_price=entry_price,
            exit_price=None,
            diagnostic_return=diagnostic,
            distributions=distributions,
            ignored_actions=ignored,
        )
    cash_total = float(sum(item["amount"] for item in distributions))
    terminal_value = (
        terminal_cash if terminal_cash is not None else float(exit_price)
    ) + cash_total
    return {
        "status": "mechanically_reconstructed_nonqualifying",
        "reason": None,
        "pricing_path": "SEP.close_plus_explicit_cash_no_reinvestment_candidate",
        "entry_price_split_normalized": entry_price,
        "exit_price_split_normalized": exit_price,
        "cash_distributions": distributions,
        "cash_total": cash_total,
        "terminal_settlement_id": terminal_id,
        "gross_terminal_value": terminal_value,
        "gross_economic_return": terminal_value / entry_price - 1.0,
        "closeadj_diagnostic_return": diagnostic,
        "ignored_actions": ignored,
    }


def reconstruct_portfolio_observations(
    normalized: Mapping[str, Any],
    provider: Any,
    *,
    start: str,
    end: str,
    horizons: Sequence[int],
    fresh_days: int,
    session_close_evidence: Mapping[str, Any] | None = None,
    economic_return_inputs: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently form monthly records from observed warehouse sessions."""
    if not horizons or any(type(item) is not int or item <= 0 for item in horizons):
        raise PeadReferenceError("horizons must be positive integers")
    if type(fresh_days) is not int or fresh_days < 0:
        raise PeadReferenceError("fresh_days must be a non-negative integer")
    horizon_values = tuple(horizons)
    padded_end = (
        pd.Timestamp(end) + pd.Timedelta(days=max(horizon_values) * 3 + 15)
    ).date().isoformat()
    price_start = (pd.Timestamp(start) - pd.Timedelta(days=100)).date().isoformat()
    sessions = pd.DatetimeIndex(provider.market_sessions(start, padded_end))
    sessions = pd.DatetimeIndex(sorted(set(pd.to_datetime(sessions))))
    close_sessions = pd.DatetimeIndex(
        provider.market_sessions(price_start, padded_end)
    )
    if session_close_evidence is None:
        calendar_reader = getattr(provider, "market_session_close_calendar", None)
        if not callable(calendar_reader):
            raise PeadReferenceError(
                "provider cannot prove authoritative session close times"
            )
        session_close_evidence = calendar_reader()
    verified_close_evidence = _validated_session_close_evidence(
        session_close_evidence,
        required_start=price_start,
        required_end=padded_end,
    )
    close_schedule = _session_closes(
        verified_close_evidence,
        close_sessions,
        required_start=price_start,
        required_end=padded_end,
    )
    formations = _first_observed_session_per_month(sessions, start, end)
    entry_dates: dict[pd.Timestamp, pd.Timestamp] = {}
    for formation in formations:
        later = sessions[sessions > formation]
        if len(later):
            entry_dates[formation] = pd.Timestamp(later[0])

    events = list(normalized.get("eps_events", []))
    tickers = sorted({str(event["ticker"]) for event in events})
    # Zacks restates historical per-share EPS and estimates for subsequent
    # splits.  SEP ``close`` is the matching split-normalized price basis for
    # scaling that forecast error.  ``closeunadj`` is fetched separately as
    # contemporaneous execution-price evidence and is never used as the signal
    # denominator.
    signal_basis_prices = _bulk_prices(
        provider, tickers, price_start, padded_end, "close"
    )
    execution_prices = _bulk_prices(
        provider, tickers, price_start, padded_end, "closeunadj"
    )
    adjusted_prices = _bulk_prices(
        provider, tickers, price_start, padded_end, "closeadj"
    )

    action_rows: list[dict[str, Any]] | None = None
    action_slice_hash = None
    currency_by_ticker: dict[str, str] = {}
    economic_input_reason = None
    absolute_tolerance = 0.0
    relative_tolerance = 0.0
    terminal_settlements: list[Mapping[str, Any]] = []
    if economic_return_inputs is None:
        economic_input_reason = "economic_return_inputs_missing"
    else:
        if not isinstance(economic_return_inputs, Mapping) or set(
            economic_return_inputs
        ) != {"artifact_hash", "payload"}:
            raise PeadReferenceError("economic return inputs are malformed")
        economic_payload = economic_return_inputs["payload"]
        if not isinstance(economic_payload, Mapping) or set(economic_payload) != {
            "schema_version", "candidate_id", "combined_data_snapshot_hash",
            "cash_distribution_semantics", "terminal_settlement_ledger",
        }:
            raise PeadReferenceError("economic return input payload is malformed")
        if (
            economic_payload["schema_version"]
            != ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION
            or economic_payload["candidate_id"] != CANDIDATE_ID
        ):
            raise PeadReferenceError("economic return input identity changed")
        if content_hash(economic_payload) != economic_return_inputs["artifact_hash"]:
            raise PeadReferenceError("economic return input hash mismatch")
        try:
            semantics = validate_cash_distribution_semantics(
                economic_payload["cash_distribution_semantics"]
            )
            terminal = validate_terminal_settlement_ledger(
                economic_payload["terminal_settlement_ledger"]
            )
        except PeadEconomicEvidenceError as exc:
            raise PeadReferenceError("economic return evidence is invalid") from exc
        tolerance = semantics["payload"]["adjustment_check_tolerance"]
        absolute_tolerance = float(tolerance["absolute"])
        relative_tolerance = float(tolerance["relative"])
        terminal_settlements = list(terminal["payload"]["cash_only_records"])
        action_rows = _reference_action_slice(
            provider, tickers, price_start, padded_end
        )
        action_slice_hash = content_hash(action_rows)
        currency_reader = getattr(provider, "security_currency", None)
        if not callable(currency_reader):
            economic_input_reason = "security_currency_reader_unavailable"
            action_rows = None
        else:
            for ticker in tickers:
                receipt = currency_reader(ticker)
                if not isinstance(receipt, Mapping) or set(receipt) != {
                    "ticker", "currency"
                }:
                    raise PeadReferenceError("security currency evidence is malformed")
                currency = receipt["currency"]
                if (
                    receipt["ticker"] != ticker
                    or not isinstance(currency, str)
                    or not currency
                    or currency != currency.strip().upper()
                ):
                    raise PeadReferenceError("security currency evidence is invalid")
                currency_by_ticker[ticker] = currency
    economic_close_maps = {
        ticker: _reference_price_map(signal_basis_prices.get(ticker))
        for ticker in tickers
    }
    economic_adjusted_maps = {
        ticker: _reference_price_map(adjusted_prices.get(ticker))
        for ticker in tickers
    }

    prepared_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signal_exclusions: list[dict[str, Any]] = []
    for event in events:
        ticker = str(event["ticker"])
        signal_prices = _numeric_series(signal_basis_prices.get(ticker))
        announcement = datetime.fromisoformat(
            str(event["announcement_at_utc"]).replace("Z", "+00:00")
        )
        if any(pd.Timestamp(session) not in close_schedule for session in signal_prices.index):
            raise PeadReferenceError(
                "price date is missing from the session close schedule"
            )
        eligible_prices = [
            (pd.Timestamp(session), float(value))
            for session, value in signal_prices.items()
            if value > 0
            and close_schedule[pd.Timestamp(session)] < announcement
        ]
        if not eligible_prices:
            signal_exclusions.append(
                {
                    "event_key": event["event_key"],
                    "ticker": ticker,
                    "reason": (
                        "missing_positive_split_normalized_preannouncement_close"
                    ),
                }
            )
            continue
        preclose_session, preclose = eligible_prices[-1]
        execution = _numeric_series(execution_prices.get(ticker))
        execution_evidence = (
            _number(execution.loc[preclose_session])
            if preclose_session in execution.index
            else None
        )
        if execution_evidence is None or execution_evidence <= 0:
            signal_exclusions.append(
                {
                    "event_key": event["event_key"],
                    "ticker": ticker,
                    "reason": (
                        "missing_exact_closeunadj_preannouncement_execution_evidence"
                    ),
                }
            )
            continue
        scaled = float(event["unscaled_forecast_error"]) / preclose
        if not math.isfinite(scaled):
            signal_exclusions.append(
                {
                    "event_key": event["event_key"],
                    "ticker": ticker,
                    "reason": "nonfinite_scaled_forecast_error",
                }
            )
            continue
        prepared_by_ticker[ticker].append(
            {
                **event,
                "preannouncement_close_split_normalized": preclose,
                "preannouncement_closeunadj_execution_evidence": (
                    execution_evidence
                ),
                "forecast_error_scaled": scaled,
                "announcement_datetime": announcement,
            }
        )
    for ticker, ticker_events in prepared_by_ticker.items():
        ticker_events.sort(
            key=lambda item: (
                item["announcement_datetime"], canonical_json(item["event_key"])
            )
        )

    lifecycle_reader = getattr(provider, "security_lifecycle", None)
    lifecycle_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in sorted(prepared_by_ticker):
        if not callable(lifecycle_reader):
            lifecycle_by_ticker[ticker] = {
                "status": "unavailable",
                "reason": "security_lifecycle_reader_unavailable",
            }
            continue
        try:
            receipt = lifecycle_reader(ticker)
            if not isinstance(receipt, Mapping):
                raise ValueError
            if receipt["ticker"] != ticker:
                raise ValueError
            status = receipt["isdelisted"]
            if status not in {"N", "Y"}:
                raise ValueError
            permaticker = receipt["permaticker"]
            if type(permaticker) is not int or permaticker <= 0:
                raise ValueError
            parsed_final_day = _lifecycle_date(receipt["lastpricedate"])
            if parsed_final_day is None:
                raise ValueError
            raw_sep_last_day = receipt["sep_lastpricedate"]
            parsed_sep_last_day = (
                None
                if raw_sep_last_day is None
                else _lifecycle_date(raw_sep_last_day)
            )
            if raw_sep_last_day is not None and parsed_sep_last_day is None:
                raise ValueError
            if status == "Y" and parsed_sep_last_day != parsed_final_day:
                raise ValueError
            lifecycle_by_ticker[ticker] = {
                "status": "validated",
                "isdelisted": status,
                "permaticker": permaticker,
                "lastpricedate": parsed_final_day.isoformat(),
                "sep_lastpricedate": (
                    parsed_sep_last_day.isoformat()
                    if parsed_sep_last_day is not None
                    else None
                ),
            }
        except (KeyError, TypeError, ValueError):
            lifecycle_by_ticker[ticker] = {
                "status": "unresolved",
                "reason": "security_lifecycle_validation_failed",
            }

    membership = {
        formation: frozenset(
            str(item).strip().upper()
            for item in provider.universe_asof(formation)
            if str(item).strip()
        )
        for formation in formations
    }
    cap_reader = getattr(provider, "daily_marketcaps_for_dates", None)
    if callable(cap_reader):
        marketcaps = cap_reader(tickers, formations)
    else:
        marketcaps = {}
        for ticker in tickers:
            for formation in formations:
                metric = provider.daily_metric(ticker, formation)
                if isinstance(metric, Mapping):
                    marketcaps[(ticker, formation.date())] = metric.get("marketcap")

    records: list[dict[str, Any]] = []
    formation_exclusions: list[dict[str, Any]] = []
    horizon_return_exclusions: list[dict[str, Any]] = []
    economic_return_exclusions: list[dict[str, Any]] = []
    for formation in formations:
        entry_day = entry_dates.get(formation)
        if entry_day is None:
            continue
        try:
            cutoff = close_schedule[formation]
        except KeyError as exc:
            raise PeadReferenceError(
                "formation date is missing from the session close schedule"
            ) from exc
        for ticker in sorted(prepared_by_ticker):
            if ticker not in membership[formation]:
                continue
            visible = []
            for event in prepared_by_ticker[ticker]:
                report_day = date.fromisoformat(str(event["act_rpt_date"]))
                age = (formation.date() - report_day).days
                if event["announcement_datetime"] < cutoff and 0 <= age <= fresh_days:
                    visible.append(event)
            if not visible:
                continue
            event = visible[-1]
            split_normalized = _numeric_series(signal_basis_prices.get(ticker))
            if entry_day not in split_normalized.index:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "missing_exact_t_plus_1_split_normalized_entry",
                    }
                )
                continue
            split_entry = _number(split_normalized.loc[entry_day])
            if split_entry is None or split_entry <= 0:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "invalid_split_normalized_entry_price",
                    }
                )
                continue
            execution = _numeric_series(execution_prices.get(ticker))
            if entry_day not in execution.index:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "missing_exact_t_plus_1_unadjusted_entry",
                    }
                )
                continue
            raw_entry = _number(execution.loc[entry_day])
            if raw_entry is None or raw_entry <= 0:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "invalid_unadjusted_entry_price",
                    }
                )
                continue
            adjusted = _numeric_series(adjusted_prices.get(ticker))
            if entry_day not in adjusted.index:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "missing_exact_t_plus_1_adjusted_entry",
                    }
                )
                continue
            location = adjusted.index.get_loc(entry_day)
            if not isinstance(location, Integral):
                raise PeadReferenceError("adjusted prices contain duplicate entry dates")
            adjusted_entry = _number(adjusted.loc[entry_day])
            if adjusted_entry is None or adjusted_entry <= 0:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "invalid_adjusted_entry_price",
                    }
                )
                continue
            marketcap = _number(marketcaps.get((ticker, formation.date())))
            if marketcap is None or marketcap <= 0:
                formation_exclusions.append(
                    {
                        "formation_date": formation.date().isoformat(),
                        "ticker": ticker,
                        "reason": "missing_positive_pit_marketcap",
                    }
                )
                continue
            record: dict[str, Any] = {
                "formation_date": formation.date().isoformat(),
                "entry_date": entry_day.date().isoformat(),
                "ticker": ticker,
                "m_ticker": event["event_key"]["m_ticker"],
                "signal": float(event["forecast_error_scaled"]),
                "pit_marketcap": marketcap,
                "entry_close_split_normalized": split_entry,
                "entry_closeunadj_execution_evidence": raw_entry,
                "entry_closeadj_diagnostic": adjusted_entry,
                "signal_preannouncement_close_split_normalized": event[
                    "preannouncement_close_split_normalized"
                ],
                "signal_preannouncement_closeunadj_execution_evidence": event[
                    "preannouncement_closeunadj_execution_evidence"
                ],
                "source_event_key": event["event_key"],
            }
            global_entry_location = sessions.get_loc(entry_day)
            if not isinstance(global_entry_location, Integral):
                raise PeadReferenceError("global sessions contain duplicate entry dates")
            for horizon in horizon_values:
                field = f"adjusted_forward_return_{horizon}"
                exit_location = int(global_entry_location) + horizon
                exit_day = (
                    pd.Timestamp(sessions[exit_location])
                    if exit_location < len(sessions)
                    else None
                )
                record[f"target_exit_date_{horizon}"] = (
                    exit_day.date().isoformat() if exit_day is not None else None
                )
                lifecycle = lifecycle_by_ticker[ticker]
                exit_price = None
                if exit_day is None:
                    reason = "global_horizon_outside_observed_sessions"
                elif (
                    lifecycle.get("status") == "validated"
                    and lifecycle.get("isdelisted") == "Y"
                    and date.fromisoformat(str(lifecycle["lastpricedate"]))
                    < exit_day.date()
                ):
                    reason = "held_delisting_terminal_economics_unresolved"
                elif exit_day not in adjusted.index:
                    reason = "missing_exact_global_session_exit"
                else:
                    exit_price = _number(adjusted.loc[exit_day])
                    reason = (
                        None
                        if exit_price is not None and exit_price > 0
                        else "invalid_exact_global_session_exit"
                    )
                if reason is not None:
                    record[field] = None
                    record[f"return_resolution_{horizon}"] = {
                        "status": "unresolved",
                        "reason": reason,
                        "pricing_path": (
                            "SEP.closeadj_exact_global_sessions_diagnostic"
                        ),
                    }
                    horizon_return_exclusions.append(
                        {
                            "formation_date": formation.date().isoformat(),
                            "ticker": ticker,
                            "m_ticker": event["event_key"]["m_ticker"],
                            "horizon_sessions": horizon,
                            "target_exit_date": (
                                exit_day.date().isoformat()
                                if exit_day is not None
                                else None
                            ),
                            "reason": reason,
                        }
                    )
                else:
                    assert exit_price is not None
                    record[field] = exit_price / adjusted_entry - 1.0
                    record[f"return_resolution_{horizon}"] = {
                        "status": "resolved_diagnostic",
                        "reason": None,
                        "pricing_path": (
                            "SEP.closeadj_exact_global_sessions_diagnostic"
                        ),
                    }
                economic_reason = None
                if exit_day is None:
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
                    economic_resolution = _reference_unresolved_economic_return(
                        economic_reason,
                        entry_price=split_entry,
                        exit_price=(
                            economic_close_maps[ticker].get(exit_day.date())
                            if exit_day is not None
                            else None
                        ),
                        diagnostic_return=record[field],
                        distributions=[],
                        ignored_actions=[],
                    )
                else:
                    assert exit_day is not None and action_rows is not None
                    economic_resolution = _reference_cash_return(
                        ticker=ticker,
                        entry_day=entry_day.date(),
                        exit_day=exit_day.date(),
                        close=economic_close_maps[ticker],
                        adjusted=economic_adjusted_maps[ticker],
                        actions=action_rows,
                        lifecycle=lifecycle,
                        currency=currency_by_ticker[ticker],
                        terminal_settlements=terminal_settlements,
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                    )
                record[f"economic_forward_return_candidate_{horizon}"] = (
                    economic_resolution["gross_economic_return"]
                )
                record[f"economic_return_resolution_{horizon}"] = (
                    economic_resolution
                )
                if economic_resolution["status"] == "unresolved":
                    economic_return_exclusions.append(
                        {
                            "formation_date": formation.date().isoformat(),
                            "ticker": ticker,
                            "m_ticker": event["event_key"]["m_ticker"],
                            "horizon_sessions": horizon,
                            "target_exit_date": (
                                exit_day.date().isoformat()
                                if exit_day is not None
                                else None
                            ),
                            "reason": economic_resolution["reason"],
                        }
                    )
            records.append(record)

    by_formation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_formation[record["formation_date"]].append(record)
    for group in by_formation.values():
        ordered = sorted(group, key=lambda item: (item["pit_marketcap"], item["m_ticker"]))
        if len(ordered) < 3:
            for item in ordered:
                item["size_tercile"] = None
            continue
        total = len(ordered)
        for rank, item in enumerate(ordered, start=1):
            item["size_tercile"] = min(2, int((rank - 1) * 3 / total))

    records.sort(key=_portfolio_key_token)
    economic_resolutions = [
        record[f"economic_return_resolution_{horizon}"]
        for record in records
        for horizon in horizon_values
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
        semantics["payload"]["evidence_status"]
        if economic_return_inputs is not None
        else None
    )
    qualification_allowed = (
        semantics["payload"]["qualification_allowed"]
        if economic_return_inputs is not None
        else False
    )
    return records, {
        "observed_sessions": len(sessions),
        "formation_dates": len(formations),
        "normalized_signal_events": len(events),
        "signals_with_preannouncement_close": sum(
            len(items) for items in prepared_by_ticker.values()
        ),
        "portfolio_observations": len(records),
        "names_with_portfolio_observations": len(
            {record["ticker"] for record in records}
        ),
        "signal_exclusions": signal_exclusions,
        "formation_exclusions": formation_exclusions,
        "horizon_return_exclusions": horizon_return_exclusions,
        "economic_return_exclusions": economic_return_exclusions,
        "economic_return_reconstruction": {
            "input_artifact_hash": (
                economic_return_inputs["artifact_hash"]
                if economic_return_inputs is not None
                else None
            ),
            "cash_distribution_semantics_status": semantics_status,
            "cash_distribution_semantics_qualification_allowed": (
                qualification_allowed
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
                item["status"] != "unresolved" for item in economic_resolutions
            ),
            "unresolved_paths": sum(
                item["status"] == "unresolved" for item in economic_resolutions
            ),
            "mechanical_reconstruction_complete": bool(economic_resolutions)
            and all(
                item["status"] != "unresolved" for item in economic_resolutions
            ),
            "cash_distribution_unique_rows": len(unique_distribution_keys),
            "cash_distribution_path_applications": len(
                distribution_applications
            ),
            "ignored_issuer_external_action_applications": sum(
                len(item["ignored_actions"]) for item in economic_resolutions
            ),
            "maximum_dividend_adjustment_absolute_error": (
                max(adjustment_errors) if adjustment_errors else None
            ),
            "reinvestment": False,
            "cash_yield": 0.0,
            "qualification_ready": bool(economic_resolutions)
            and all(
                item["status"] != "unresolved" for item in economic_resolutions
            )
            and qualification_allowed is True,
        },
        "security_lifecycle_diagnostics": lifecycle_by_ticker,
        "security_lifecycle_complete": bool(lifecycle_by_ticker) and all(
            item.get("status") == "validated"
            for item in lifecycle_by_ticker.values()
        ),
        "session_close_evidence_artifact_hash": verified_close_evidence[
            "artifact_hash"
        ],
        "session_close_calendar_artifact_hash": verified_close_evidence["payload"][
            "calendar"
        ]["artifact_hash"],
        "session_close_source_receipt_artifact_hash": verified_close_evidence[
            "payload"
        ]["source_receipt"]["artifact_hash"],
        "session_close_schedule_sessions": len(close_schedule),
        "observed_early_close_sessions": sum(
            value.astimezone(EASTERN).hour == 13
            for value in close_schedule.values()
        ),
    }


def _event_key_token(item: Mapping[str, Any]) -> str:
    return canonical_json(item.get("event_key"))


def _exclusion_key_token(item: Mapping[str, Any]) -> str:
    return canonical_json(
        {"event_key": item.get("event_key"), "ticker": item.get("ticker")}
    )


def _portfolio_key(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "formation_date": item.get("formation_date"),
        "ticker": item.get("ticker"),
        "m_ticker": item.get("m_ticker"),
        "source_event_key": item.get("source_event_key"),
    }


def _portfolio_key_token(item: Mapping[str, Any]) -> str:
    return canonical_json(_portfolio_key(item))


def _index_unique(
    values: Sequence[Mapping[str, Any]], key_function: Any, section: str
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    result: dict[str, Mapping[str, Any]] = {}
    discrepancies: list[dict[str, Any]] = []
    for item in values:
        token = key_function(item)
        if token in result:
            discrepancies.append(
                {
                    "section": section,
                    "kind": "duplicate_key",
                    "key": json.loads(token),
                }
            )
        else:
            result[token] = item
    return result, discrepancies


def _close(left: Any, right: Any, *, absolute: float, relative: float) -> bool:
    if left is None or right is None:
        return left is right
    a = _number(left)
    b = _number(right)
    if a is None or b is None:
        return False
    return abs(a - b) <= absolute + relative * max(abs(a), abs(b))


def _compare_records(
    primary: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    section: str,
    key_function: Any,
    numeric_tolerances: Mapping[str, tuple[float, float]],
    exact_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    primary_by_key, discrepancies = _index_unique(primary, key_function, section)
    reference_by_key, reference_duplicates = _index_unique(
        reference, key_function, section
    )
    discrepancies.extend(reference_duplicates)
    tokens = sorted(set(primary_by_key) | set(reference_by_key))
    matched = 0
    for token in tokens:
        key = json.loads(token)
        if token not in primary_by_key:
            discrepancies.append(
                {"section": section, "kind": "missing_primary", "key": key}
            )
            continue
        if token not in reference_by_key:
            discrepancies.append(
                {"section": section, "kind": "missing_reference", "key": key}
            )
            continue
        before = len(discrepancies)
        left = primary_by_key[token]
        right = reference_by_key[token]
        for field in exact_fields:
            if canonical_json(left.get(field)) != canonical_json(right.get(field)):
                discrepancies.append(
                    {
                        "section": section,
                        "kind": "value_mismatch",
                        "key": key,
                        "field": field,
                        "primary": left.get(field),
                        "reference": right.get(field),
                    }
                )
        for field, (absolute, relative) in numeric_tolerances.items():
            if not _close(
                left.get(field), right.get(field),
                absolute=absolute, relative=relative,
            ):
                discrepancies.append(
                    {
                        "section": section,
                        "kind": "numeric_mismatch",
                        "key": key,
                        "field": field,
                        "primary": left.get(field),
                        "reference": right.get(field),
                        "absolute_tolerance": absolute,
                        "relative_tolerance": relative,
                    }
                )
        if len(discrepancies) == before:
            matched += 1
    return discrepancies, matched


def _warehouse_snapshot(provider: Any) -> dict[str, Any]:
    reader = getattr(provider, "snapshot_version", None)
    if not callable(reader):
        raise PeadReferenceError("provider cannot content-address return inputs")
    value = reader(list(RETURN_TABLES))
    if not isinstance(value, Mapping) or set(value) != {
        "version", "tables", "complete", "quality_flags"
    }:
        raise PeadReferenceError("warehouse snapshot receipt must be an object")
    tables = value["tables"]
    if not isinstance(tables, list):
        raise PeadReferenceError("warehouse snapshot tables must be an array")
    manifest: list[dict[str, Any]] = []
    for index, item in enumerate(tables):
        if not isinstance(item, Mapping) or set(item) != {"table", "sha256", "bytes"}:
            raise PeadReferenceError(f"warehouse table receipt {index} is malformed")
        table = item["table"]
        size = item["bytes"]
        if table not in RETURN_TABLES:
            raise PeadReferenceError(f"unexpected warehouse return table {table!r}")
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise PeadReferenceError("warehouse table byte count is invalid")
        manifest.append(
            {"table": table, "sha256": _sha256(item["sha256"], "table hash"),
             "bytes": int(size)}
        )
    names = [item["table"] for item in manifest]
    if names != sorted(set(names)):
        raise PeadReferenceError("warehouse table receipt is not uniquely sorted")
    missing = sorted(set(RETURN_TABLES) - set(names))
    expected_flags = [f"missing_table:{table}" for table in missing]
    if value["quality_flags"] != expected_flags:
        raise PeadReferenceError("warehouse quality flags do not match missing tables")
    if value["complete"] is not (not missing):
        raise PeadReferenceError("warehouse completeness does not match missing tables")
    digest = hashlib.sha256()
    for item in manifest:
        digest.update(
            f"{item['table']}:{item['sha256']}:{item['bytes']}\n".encode("utf-8")
        )
    version = _sha256(value["version"], "warehouse snapshot version")
    if digest.hexdigest() != version:
        raise PeadReferenceError("warehouse snapshot version mismatch")
    return {
        "version": version,
        "tables": manifest,
        "complete": value["complete"],
        "quality_flags": expected_flags,
    }


def _validated_corporate_action_evidence(
    value: Any, *, start: str, end: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"artifact_hash", "payload"}:
        raise PeadReferenceError("corporate action evidence is malformed")
    payload = value["payload"]
    expected = {
        "schema_version", "acquisition_artifact_hash", "source_snapshot_time",
        "parquet_sha256", "raw_zip_sha256", "row_count", "min_date",
        "max_date", "required_window", "complete", "blockers",
        "value_is_terminal_payout_per_share",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise PeadReferenceError("corporate action evidence payload is malformed")
    if payload["schema_version"] != "sharadar_actions_evidence.v1":
        raise PeadReferenceError("unsupported corporate action evidence schema")
    for field in ("acquisition_artifact_hash", "parquet_sha256", "raw_zip_sha256"):
        _sha256(payload[field], f"corporate action {field}")
    if _provider_utc_timestamp(payload["source_snapshot_time"]) is None:
        raise PeadReferenceError("corporate action snapshot time must identify UTC")
    rows = payload["row_count"]
    if isinstance(rows, bool) or not isinstance(rows, Integral) or rows <= 0:
        raise PeadReferenceError("corporate action row count must be positive")
    minimum = _date(payload["min_date"])
    maximum = _date(payload["max_date"])
    if minimum is None or maximum is None or minimum > maximum:
        raise PeadReferenceError("corporate action coverage dates are invalid")
    window = payload["required_window"]
    if not isinstance(window, Mapping) or set(window) != {"start", "end"}:
        raise PeadReferenceError("corporate action required window is malformed")
    window_start = _date(window["start"])
    window_end = _date(window["end"])
    if window_start is None or window_end is None or window_start > window_end:
        raise PeadReferenceError("corporate action required window is invalid")
    if window != {"start": start, "end": end}:
        raise PeadReferenceError("corporate action required window changed")
    blockers = payload["blockers"]
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in blockers
    ):
        raise PeadReferenceError("corporate action blockers are malformed")
    expected_blockers = []
    if minimum > window_start:
        expected_blockers.append("actions_range_starts_after_required_window")
    if maximum < window_end:
        expected_blockers.append("actions_range_ends_before_required_window")
    if blockers != expected_blockers:
        raise PeadReferenceError(
            "corporate action blockers do not match observed coverage"
        )
    expected_complete = not expected_blockers
    if (
        type(payload["complete"]) is not bool
        or payload["complete"] is not expected_complete
    ):
        raise PeadReferenceError(
            "corporate action completeness does not match observed coverage"
        )
    if payload["value_is_terminal_payout_per_share"] is not False:
        raise PeadReferenceError("ACTIONS value cannot be treated as a terminal payout")
    claimed = _sha256(value["artifact_hash"], "corporate action evidence hash")
    if claimed != content_hash(payload):
        raise PeadReferenceError("corporate action evidence hash mismatch")
    return {"artifact_hash": claimed, "payload": _plain_json(payload)}


def _corporate_action_evidence(
    provider: Any, *, start: str, end: str
) -> dict[str, Any]:
    reader = getattr(provider, "corporate_action_evidence", None)
    if not callable(reader):
        raise PeadReferenceError("provider cannot prove ACTIONS acquisition identity")
    return _validated_corporate_action_evidence(
        reader(start, end), start=start, end=end
    )


def _session_close_bundle(provider: Any) -> Any:
    reader = getattr(provider, "market_session_close_calendar", None)
    if not callable(reader):
        raise PeadReferenceError(
            "provider cannot prove authoritative session close times"
        )
    return reader()


def _combined_snapshot(
    source_hash: str,
    warehouse: Mapping[str, Any],
    corporate_actions: Mapping[str, Any],
    session_close_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": COMBINED_SNAPSHOT_SCHEMA_VERSION,
        "zacks_snapshot_hash": _sha256(source_hash, "source snapshot hash"),
        "warehouse_return_snapshot": warehouse,
        "corporate_action_evidence": corporate_actions,
        "session_close_evidence": session_close_evidence,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _economic_inputs(
    combined_snapshot: Mapping[str, Any],
    cash_distribution_semantics: Mapping[str, Any],
    terminal_settlement_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently bind economic policies to the exact return snapshot."""
    if not isinstance(combined_snapshot, Mapping) or set(combined_snapshot) != {
        "artifact_hash", "payload"
    }:
        raise PeadReferenceError("combined snapshot is malformed")
    combined_hash = _sha256(
        combined_snapshot["artifact_hash"], "combined snapshot hash"
    )
    if combined_hash != content_hash(combined_snapshot["payload"]):
        raise PeadReferenceError("combined snapshot hash mismatch")
    try:
        semantics = validate_cash_distribution_semantics(
            cash_distribution_semantics
        )
        terminal = validate_terminal_settlement_ledger(
            terminal_settlement_ledger
        )
    except PeadEconomicEvidenceError as exc:
        raise PeadReferenceError("economic input evidence is invalid") from exc
    payload = {
        "schema_version": ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION,
        "candidate_id": CANDIDATE_ID,
        "combined_data_snapshot_hash": combined_hash,
        "cash_distribution_semantics": semantics,
        "terminal_settlement_ledger": terminal,
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _implementation_identity(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "implementation_id", "code_hash", "files"
    }:
        raise PeadReferenceError(f"{path} implementation identity is malformed")
    identifier = value["implementation_id"]
    if not isinstance(identifier, str) or not identifier.strip():
        raise PeadReferenceError(f"{path} implementation ID is required")
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise PeadReferenceError(f"{path} implementation must identify source files")
    normalized_files: list[dict[str, str]] = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise PeadReferenceError(f"{path} source file {index} is malformed")
        file_path = item["path"]
        if not isinstance(file_path, str) or not file_path.strip():
            raise PeadReferenceError(f"{path} source file path is required")
        normalized_files.append(
            {"path": file_path, "sha256": _sha256(item["sha256"], "source hash")}
        )
    if normalized_files != sorted(normalized_files, key=lambda item: item["path"]):
        raise PeadReferenceError(f"{path} source files are not sorted")
    if len({item["path"] for item in normalized_files}) != len(normalized_files):
        raise PeadReferenceError(f"{path} source file paths are duplicated")
    code_hash = _sha256(value["code_hash"], f"{path} code hash")
    if code_hash != content_hash(normalized_files):
        raise PeadReferenceError(f"{path} code hash does not bind its source files")
    return {
        "implementation_id": identifier.strip(),
        "code_hash": code_hash,
        "files": normalized_files,
    }


@dataclass(frozen=True)
class ReferenceRun:
    source_snapshot_hash: str
    warehouse_snapshot: Mapping[str, Any]
    corporate_action_evidence: Mapping[str, Any]
    session_close_evidence: Mapping[str, Any]
    combined_snapshot: Mapping[str, Any]
    economic_return_inputs: Mapping[str, Any] | None
    normalized: Mapping[str, Any]
    portfolio_observations: tuple[Mapping[str, Any], ...]
    coverage: Mapping[str, Any]


def run_reference_reconstruction(
    snapshot_document: Mapping[str, Any],
    provider: Any,
    *,
    start: str,
    end: str,
    horizons: Sequence[int],
    fresh_days: int,
    consensus_abs_tolerance: float,
    cash_distribution_semantics: Mapping[str, Any] | None = None,
    terminal_settlement_ledger: Mapping[str, Any] | None = None,
) -> ReferenceRun:
    """Run the independent reference while proving inputs did not mutate."""
    validate_snapshot(snapshot_document)
    before = _warehouse_snapshot(provider)
    action_end = (
        pd.Timestamp(end) + pd.Timedelta(days=max(horizons) * 3 + 15)
    ).date().isoformat()
    actions_before = _corporate_action_evidence(
        provider, start=start, end=action_end
    )
    calendar_start = (
        pd.Timestamp(start) - pd.Timedelta(days=100)
    ).date().isoformat()
    close_bundle_before = _session_close_bundle(provider)
    close_evidence_before = _validated_session_close_evidence(
        close_bundle_before,
        required_start=calendar_start,
        required_end=action_end,
    )
    source_hash = _sha256(snapshot_document["artifact_hash"], "source snapshot hash")
    combined = _combined_snapshot(
        source_hash, before, actions_before, close_evidence_before
    )
    if (cash_distribution_semantics is None) != (
        terminal_settlement_ledger is None
    ):
        raise PeadReferenceError(
            "cash semantics and terminal ledger must be supplied together"
        )
    economic_inputs = (
        _economic_inputs(
            combined,
            cash_distribution_semantics,
            terminal_settlement_ledger,
        )
        if cash_distribution_semantics is not None
        and terminal_settlement_ledger is not None
        else None
    )
    normalized = reconstruct_events(
        snapshot_document, consensus_abs_tolerance=consensus_abs_tolerance
    )
    observations, coverage = reconstruct_portfolio_observations(
        normalized,
        provider,
        start=start,
        end=end,
        horizons=horizons,
        fresh_days=fresh_days,
        session_close_evidence=close_bundle_before,
        economic_return_inputs=economic_inputs,
    )
    after = _warehouse_snapshot(provider)
    actions_after = _corporate_action_evidence(
        provider, start=start, end=action_end
    )
    close_bundle_after = _session_close_bundle(provider)
    close_evidence_after = _validated_session_close_evidence(
        close_bundle_after,
        required_start=calendar_start,
        required_end=action_end,
    )
    if before != after:
        raise PeadReferenceError("warehouse inputs changed during reference run")
    if actions_before != actions_after:
        raise PeadReferenceError("corporate action evidence changed during reference run")
    if close_evidence_before != close_evidence_after:
        raise PeadReferenceError("session close evidence changed during reference run")
    return ReferenceRun(
        source_snapshot_hash=source_hash,
        warehouse_snapshot=before,
        corporate_action_evidence=actions_before,
        session_close_evidence=close_evidence_before,
        combined_snapshot=combined,
        economic_return_inputs=economic_inputs,
        normalized=normalized,
        portfolio_observations=tuple(observations),
        coverage=coverage,
    )


def build_reference_comparison(
    run: ReferenceRun,
    primary_report: Mapping[str, Any],
    *,
    protocol_hash: str,
    primary_report_sha256: str,
    primary_implementation: Mapping[str, Any],
    reference_implementation: Mapping[str, Any],
    start: str,
    end: str,
    horizons: Sequence[int],
    fresh_days: int,
    consensus_abs_tolerance: float,
) -> dict[str, Any]:
    """Compare primary and reference signal paths and content-address the result."""
    protocol = _sha256(protocol_hash, "protocol hash")
    report_hash = _sha256(primary_report_sha256, "primary report hash")
    if not isinstance(primary_report, Mapping):
        raise PeadReferenceError("primary report must be an object")
    if primary_report.get("candidate_id") != CANDIDATE_ID:
        raise PeadReferenceError("primary report belongs to another candidate")
    primary_source = primary_report.get("source_snapshot")
    if not isinstance(primary_source, Mapping) or primary_source.get("artifact_hash") != (
        run.source_snapshot_hash
    ):
        raise PeadReferenceError("primary and reference source snapshot hashes differ")
    primary_combined = primary_report.get("combined_data_snapshot")
    if not isinstance(primary_combined, Mapping) or primary_combined.get(
        "artifact_hash"
    ) != run.combined_snapshot["artifact_hash"]:
        raise PeadReferenceError("primary and reference combined data hashes differ")
    primary_economic_inputs = primary_report.get("economic_return_inputs")
    if run.economic_return_inputs is None:
        if primary_economic_inputs is not None:
            raise PeadReferenceError(
                "primary has economic inputs absent from the reference run"
            )
    elif (
        not isinstance(primary_economic_inputs, Mapping)
        or primary_economic_inputs.get("artifact_hash")
        != run.economic_return_inputs["artifact_hash"]
    ):
        raise PeadReferenceError(
            "primary and reference economic return input hashes differ"
        )
    primary_binding = primary_report.get("research_manifest_binding")
    if not isinstance(primary_binding, Mapping) or primary_binding.get(
        "artifact_hash"
    ) != protocol:
        raise PeadReferenceError("primary report has a different protocol binding")
    primary_normalized = primary_report.get("normalization")
    primary_portfolio = primary_report.get("raw_portfolio_observations")
    primary_coverage = primary_report.get("coverage")
    if not isinstance(primary_normalized, Mapping) or not isinstance(
        primary_portfolio, list
    ):
        raise PeadReferenceError("primary report omits comparison outputs")
    if not isinstance(primary_coverage, Mapping):
        raise PeadReferenceError("primary report omits coverage output")

    primary_identity = _implementation_identity(
        primary_implementation, "primary"
    )
    reference_identity = _implementation_identity(
        reference_implementation, "reference"
    )
    if primary_identity["implementation_id"] == reference_identity["implementation_id"]:
        raise PeadReferenceError("independent implementation IDs must differ")
    if primary_identity["code_hash"] == reference_identity["code_hash"]:
        raise PeadReferenceError("independent implementation code hashes must differ")

    event_numeric = {
        field: (1e-12, 1e-12)
        for field in (
            "actual", "zacks_adjustment_diagnostic", "consensus",
            "consensus_crosscheck_absolute_difference", "unscaled_forecast_error",
            "surprise_table_consensus",
        )
    }
    event_exact = (
        "ticker", "currency_code", "act_rpt_date", "announcement_at_utc",
        "act_rpt_time", "act_rpt_code", "consensus_obs_date",
        "consensus_analyst_count", "surprise_table_analyst_count",
        "sales_diagnostic",
    )
    event_discrepancies, matched_events = _compare_records(
        primary_normalized.get("eps_events", []),
        run.normalized["eps_events"],
        section="normalized_eps_events",
        key_function=_event_key_token,
        numeric_tolerances=event_numeric,
        exact_fields=event_exact,
    )
    eps_exclusion_discrepancies, matched_eps_exclusions = _compare_records(
        primary_normalized.get("eps_exclusions", []),
        run.normalized["eps_exclusions"],
        section="eps_exclusions",
        key_function=_exclusion_key_token,
        numeric_tolerances={},
        exact_fields=("reasons",),
    )
    sales_discrepancies, matched_sales_events = _compare_records(
        primary_normalized.get("sales_events", []),
        run.normalized["sales_events"],
        section="normalized_sales_events",
        key_function=_event_key_token,
        numeric_tolerances=event_numeric,
        exact_fields=event_exact[:-1],
    )
    sales_exclusion_discrepancies, matched_sales_exclusions = _compare_records(
        primary_normalized.get("sales_exclusions", []),
        run.normalized["sales_exclusions"],
        section="sales_exclusions",
        key_function=_exclusion_key_token,
        numeric_tolerances={},
        exact_fields=("reasons",),
    )
    horizon_fields = {
        f"adjusted_forward_return_{horizon}": (1e-12, 1e-12)
        for horizon in horizons
    }
    horizon_fields.update(
        {
            f"economic_forward_return_candidate_{horizon}": (1e-12, 1e-12)
            for horizon in horizons
        }
    )
    horizon_date_fields = tuple(
        f"target_exit_date_{horizon}" for horizon in horizons
    )
    horizon_resolution_fields = tuple(
        f"return_resolution_{horizon}" for horizon in horizons
    )
    economic_resolution_fields = tuple(
        f"economic_return_resolution_{horizon}" for horizon in horizons
    )
    portfolio_numeric = {
        "signal": (1e-12, 1e-12),
        "pit_marketcap": (1e-8, 1e-12),
        "entry_close_split_normalized": (1e-12, 1e-12),
        "entry_closeunadj_execution_evidence": (1e-12, 1e-12),
        "entry_closeadj_diagnostic": (1e-12, 1e-12),
        "signal_preannouncement_close_split_normalized": (1e-12, 1e-12),
        "signal_preannouncement_closeunadj_execution_evidence": (
            1e-12, 1e-12
        ),
        **horizon_fields,
    }
    portfolio_discrepancies, matched_portfolios = _compare_records(
        primary_portfolio,
        run.portfolio_observations,
        section="portfolio_observations",
        key_function=_portfolio_key_token,
        numeric_tolerances=portfolio_numeric,
        exact_fields=(
            "entry_date", "size_tercile", *horizon_date_fields,
            *horizon_resolution_fields, *economic_resolution_fields,
        ),
    )
    discrepancies = [
        *event_discrepancies,
        *eps_exclusion_discrepancies,
        *sales_discrepancies,
        *sales_exclusion_discrepancies,
        *portfolio_discrepancies,
    ]
    normalization_metadata_fields = (
        "consensus_absolute_tolerance", "eps_counts", "sales_counts",
        "sales_is_diagnostic_only", "stable_identity_diagnostics",
        "announcement_schedule_diagnostics", "primary_signal",
    )
    primary_metadata = {
        field: primary_normalized.get(field) for field in normalization_metadata_fields
    }
    reference_metadata = {
        field: run.normalized.get(field) for field in normalization_metadata_fields
    }
    if canonical_json(primary_metadata) != canonical_json(reference_metadata):
        discrepancies.append(
            {
                "section": "normalization_metadata",
                "kind": "value_mismatch",
                "key": {"candidate_id": CANDIDATE_ID},
                "primary": primary_metadata,
                "reference": reference_metadata,
            }
        )
    if canonical_json(primary_coverage) != canonical_json(run.coverage):
        discrepancies.append(
            {
                "section": "coverage",
                "kind": "value_mismatch",
                "key": {"candidate_id": CANDIDATE_ID},
                "primary": primary_coverage,
                "reference": run.coverage,
            }
        )
    discrepancies = sorted(discrepancies, key=canonical_json)
    source_blockers = primary_report.get("blockers")
    blockers = (
        [
            str(item)
            for item in source_blockers
            if str(item) != "independent_implementation_reconciliation_missing"
        ]
        if isinstance(source_blockers, list)
        else ["primary_report_blockers_unavailable"]
    )
    blockers.extend(
        [
            "event_driven_money_path_not_implemented",
            "generic_replication_evidence_not_available",
        ]
    )
    if discrepancies:
        blockers.append("independent_signal_reconstruction_mismatch")

    expected_keys = sorted(
        [_portfolio_key(item) for item in run.portfolio_observations],
        key=canonical_json,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "independent_pead_reference_reconciliation",
        "candidate_id": CANDIDATE_ID,
        "evidence_class": "development_independent_implementation_comparison",
        "qualifying_evidence": False,
        "replication_evidence_eligible": False,
        "nonqualifying_reason": (
            "This independently reconciles the research signal and return records only. "
            "The PEAD research layer has no event-driven order, position, cash, fee, or "
            "realised-P&L ledger, and those fields were not fabricated."
        ),
        "bindings": {
            "protocol_hash": protocol,
            "data_snapshot_hash": run.combined_snapshot["artifact_hash"],
            "source_snapshot_hash": run.source_snapshot_hash,
            "warehouse_snapshot": run.warehouse_snapshot,
            "corporate_action_evidence": run.corporate_action_evidence,
            "session_close_evidence": run.session_close_evidence,
            "economic_return_inputs": run.economic_return_inputs,
            "primary_report_sha256": report_hash,
        },
        "implementations": {
            "primary": primary_identity,
            "reference": reference_identity,
        },
        "configuration": {
            "start": start,
            "end": end,
            "horizons_sessions": list(horizons),
            "fresh_days_calendar": fresh_days,
            "consensus_absolute_tolerance": float(consensus_abs_tolerance),
            "signal_price_field": "SEP.close_split_normalized",
            "execution_price_evidence_field": "SEP.closeunadj",
            "forward_return_field": "SEP.closeadj_diagnostic_only",
            "economic_return_candidate": (
                "SEP.close_plus_explicit_cash_without_reinvestment"
            ),
            "formation_cutoff": "bound_actual_nyse_session_close",
        },
        "replication_contract_inputs": {
            "protocol_hash": protocol,
            "data_snapshot_hash": run.combined_snapshot["artifact_hash"],
            "expected_observation_keys": expected_keys,
            "numeric_tolerances": {
                **{
                    field: {"absolute": values[0], "relative": values[1]}
                    for field, values in portfolio_numeric.items()
                },
            },
            "unavailable_required_money_path_fields": list(MONEY_PATH_FIELDS),
            "unavailable_required_replication_fields": list(
                UNAVAILABLE_REPLICATION_FIELDS
            ),
        },
        "reference_coverage": _plain_json(run.coverage),
        "outputs": {
            "primary": {
                "normalized_eps_events": primary_normalized.get("eps_events", []),
                "eps_exclusions": primary_normalized.get("eps_exclusions", []),
                "normalized_sales_events": primary_normalized.get("sales_events", []),
                "sales_exclusions": primary_normalized.get("sales_exclusions", []),
                "normalization_metadata": primary_metadata,
                "portfolio_observations": primary_portfolio,
                "coverage": primary_coverage,
            },
            "reference": {
                "normalized_eps_events": run.normalized["eps_events"],
                "eps_exclusions": run.normalized["eps_exclusions"],
                "normalized_sales_events": run.normalized["sales_events"],
                "sales_exclusions": run.normalized["sales_exclusions"],
                "normalization_metadata": reference_metadata,
                "portfolio_observations": list(run.portfolio_observations),
                "coverage": run.coverage,
            },
        },
        "comparison": {
            "signal_path_passed": not discrepancies,
            "normalized_eps_events_primary": len(
                primary_normalized.get("eps_events", [])
            ),
            "normalized_eps_events_reference": len(run.normalized["eps_events"]),
            "normalized_eps_events_matched": matched_events,
            "eps_exclusions_primary": len(
                primary_normalized.get("eps_exclusions", [])
            ),
            "eps_exclusions_reference": len(run.normalized["eps_exclusions"]),
            "eps_exclusions_matched": matched_eps_exclusions,
            "normalized_sales_events_primary": len(
                primary_normalized.get("sales_events", [])
            ),
            "normalized_sales_events_reference": len(
                run.normalized["sales_events"]
            ),
            "normalized_sales_events_matched": matched_sales_events,
            "sales_exclusions_primary": len(
                primary_normalized.get("sales_exclusions", [])
            ),
            "sales_exclusions_reference": len(run.normalized["sales_exclusions"]),
            "sales_exclusions_matched": matched_sales_exclusions,
            "portfolio_observations_primary": len(primary_portfolio),
            "portfolio_observations_reference": len(run.portfolio_observations),
            "portfolio_observations_matched": matched_portfolios,
            "discrepancy_count": len(discrepancies),
            "discrepancy_reason_counts": dict(
                sorted(Counter(item["kind"] for item in discrepancies).items())
            ),
            "discrepancies": discrepancies,
            "money_path_status": "not_implemented",
            "generic_replication_evidence_status": "blocked_missing_money_path",
        },
        "blockers": sorted(set(blockers)),
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _reconcile_artifact_outputs(
    outputs: Mapping[str, Any], horizons: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recompute every stored discrepancy during artifact verification."""
    primary = outputs.get("primary")
    reference = outputs.get("reference")
    if not isinstance(primary, Mapping) or not isinstance(reference, Mapping):
        raise PeadReferenceError("reference artifact outputs are malformed")
    event_numeric = {
        field: (1e-12, 1e-12)
        for field in (
            "actual", "zacks_adjustment_diagnostic", "consensus",
            "consensus_crosscheck_absolute_difference", "unscaled_forecast_error",
            "surprise_table_consensus",
        )
    }
    event_exact = (
        "ticker", "currency_code", "act_rpt_date", "announcement_at_utc",
        "act_rpt_time", "act_rpt_code", "consensus_obs_date",
        "consensus_analyst_count", "surprise_table_analyst_count",
        "sales_diagnostic",
    )
    discrepancies: list[dict[str, Any]] = []
    matched_counts: dict[str, int] = {}
    comparisons = (
        (
            "normalized_eps_events", _event_key_token, event_numeric, event_exact,
        ),
        ("eps_exclusions", _exclusion_key_token, {}, ("reasons",)),
        (
            "normalized_sales_events", _event_key_token, event_numeric,
            event_exact[:-1],
        ),
        ("sales_exclusions", _exclusion_key_token, {}, ("reasons",)),
    )
    for section, key_function, numeric, exact in comparisons:
        left = primary.get(section)
        right = reference.get(section)
        if not isinstance(left, list) or not isinstance(right, list):
            raise PeadReferenceError(f"reference artifact omits {section}")
        found, matched = _compare_records(
            left,
            right,
            section=section,
            key_function=key_function,
            numeric_tolerances=numeric,
            exact_fields=exact,
        )
        discrepancies.extend(found)
        matched_counts[f"{section}_matched"] = matched
    portfolio_numeric = {
        "signal": (1e-12, 1e-12),
        "pit_marketcap": (1e-8, 1e-12),
        "entry_close_split_normalized": (1e-12, 1e-12),
        "entry_closeunadj_execution_evidence": (1e-12, 1e-12),
        "entry_closeadj_diagnostic": (1e-12, 1e-12),
        "signal_preannouncement_close_split_normalized": (1e-12, 1e-12),
        "signal_preannouncement_closeunadj_execution_evidence": (
            1e-12, 1e-12
        ),
        **{
            f"adjusted_forward_return_{horizon}": (1e-12, 1e-12)
            for horizon in horizons
        },
        **{
            f"economic_forward_return_candidate_{horizon}": (1e-12, 1e-12)
            for horizon in horizons
        },
    }
    left_portfolio = primary.get("portfolio_observations")
    right_portfolio = reference.get("portfolio_observations")
    if not isinstance(left_portfolio, list) or not isinstance(right_portfolio, list):
        raise PeadReferenceError("reference artifact omits portfolio observations")
    found, matched = _compare_records(
        left_portfolio,
        right_portfolio,
        section="portfolio_observations",
        key_function=_portfolio_key_token,
        numeric_tolerances=portfolio_numeric,
        exact_fields=(
            "entry_date", "size_tercile",
            *(f"target_exit_date_{horizon}" for horizon in horizons),
            *(f"return_resolution_{horizon}" for horizon in horizons),
            *(f"economic_return_resolution_{horizon}" for horizon in horizons),
        ),
    )
    discrepancies.extend(found)
    matched_counts["portfolio_observations_matched"] = matched
    for section in ("normalization_metadata", "coverage"):
        if canonical_json(primary.get(section)) != canonical_json(
            reference.get(section)
        ):
            discrepancies.append(
                {
                    "section": section,
                    "kind": "value_mismatch",
                    "key": {"candidate_id": CANDIDATE_ID},
                    "primary": primary.get(section),
                    "reference": reference.get(section),
                }
            )
    return sorted(discrepancies, key=canonical_json), matched_counts


def verify_reference_artifact(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify content identity and fail-closed non-qualification invariants."""
    if not isinstance(document, Mapping) or set(document) != {"artifact_hash", "payload"}:
        raise PeadReferenceError("reference artifact must have two outer fields")
    claimed = _sha256(document["artifact_hash"], "reference artifact hash")
    payload = document["payload"]
    if not isinstance(payload, Mapping) or content_hash(payload) != claimed:
        raise PeadReferenceError("reference artifact hash mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PeadReferenceError("unsupported reference artifact schema")
    if payload.get("candidate_id") != CANDIDATE_ID:
        raise PeadReferenceError("reference artifact belongs to another candidate")
    if payload.get("qualifying_evidence") is not False:
        raise PeadReferenceError("reference signal comparison cannot be qualifying")
    if payload.get("replication_evidence_eligible") is not False:
        raise PeadReferenceError("reference artifact cannot claim money-path eligibility")
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping):
        raise PeadReferenceError("reference artifact omits comparison")
    discrepancies = comparison.get("discrepancies")
    if not isinstance(discrepancies, list) or comparison.get(
        "discrepancy_count"
    ) != len(discrepancies):
        raise PeadReferenceError("reference discrepancy count is inconsistent")
    if comparison.get("signal_path_passed") is not (not discrepancies):
        raise PeadReferenceError("reference pass status is inconsistent")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise PeadReferenceError("reference artifact omits configuration")
    horizons = configuration.get("horizons_sessions")
    if not isinstance(horizons, list) or any(
        type(item) is not int or item <= 0 for item in horizons
    ):
        raise PeadReferenceError("reference artifact horizons are invalid")
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PeadReferenceError("reference artifact omits outputs")
    rebuilt_discrepancies, matched_counts = _reconcile_artifact_outputs(
        outputs, horizons
    )
    if canonical_json(rebuilt_discrepancies) != canonical_json(discrepancies):
        raise PeadReferenceError("stored reference discrepancies are inconsistent")
    for field, value in matched_counts.items():
        if comparison.get(field) != value:
            raise PeadReferenceError(f"stored reference {field} is inconsistent")
    count_fields = {
        "normalized_eps_events": "normalized_eps_events",
        "eps_exclusions": "eps_exclusions",
        "normalized_sales_events": "normalized_sales_events",
        "sales_exclusions": "sales_exclusions",
        "portfolio_observations": "portfolio_observations",
    }
    for output_field, count_prefix in count_fields.items():
        for side in ("primary", "reference"):
            values = outputs[side].get(output_field)
            if not isinstance(values, list) or comparison.get(
                f"{count_prefix}_{side}"
            ) != len(values):
                raise PeadReferenceError(
                    f"stored reference {count_prefix}_{side} is inconsistent"
                )
    expected_reason_counts = dict(
        sorted(Counter(item["kind"] for item in discrepancies).items())
    )
    if comparison.get("discrepancy_reason_counts") != expected_reason_counts:
        raise PeadReferenceError("stored discrepancy reason counts are inconsistent")
    implementations = payload.get("implementations")
    if not isinstance(implementations, Mapping):
        raise PeadReferenceError("reference artifact omits code identities")
    primary_identity = _implementation_identity(
        implementations.get("primary"), "primary"
    )
    reference_identity = _implementation_identity(
        implementations.get("reference"), "reference"
    )
    if primary_identity["implementation_id"] == reference_identity["implementation_id"]:
        raise PeadReferenceError("stored implementation IDs are not independent")
    if primary_identity["code_hash"] == reference_identity["code_hash"]:
        raise PeadReferenceError("stored implementation code hashes are not independent")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PeadReferenceError("reference artifact omits input bindings")
    source_hash = _sha256(bindings.get("source_snapshot_hash"), "source hash")
    data_hash = _sha256(bindings.get("data_snapshot_hash"), "data hash")
    _sha256(bindings.get("protocol_hash"), "protocol hash")
    _sha256(bindings.get("primary_report_sha256"), "primary report hash")
    warehouse = bindings.get("warehouse_snapshot")
    action_end = (
        pd.Timestamp(configuration.get("end"))
        + pd.Timedelta(days=max(horizons) * 3 + 15)
    ).date().isoformat()
    actions = _validated_corporate_action_evidence(
        bindings.get("corporate_action_evidence"),
        start=str(configuration.get("start")),
        end=action_end,
    )
    calendar_start = (
        pd.Timestamp(configuration.get("start")) - pd.Timedelta(days=100)
    ).date().isoformat()
    close_evidence = _validated_bound_session_close_evidence(
        bindings.get("session_close_evidence"),
        required_start=calendar_start,
        required_end=action_end,
    )
    if not isinstance(warehouse, Mapping) or _combined_snapshot(
        source_hash, warehouse, actions, close_evidence
    )["artifact_hash"] != data_hash:
        raise PeadReferenceError("reference combined data binding is inconsistent")
    bound_economic_inputs = bindings.get("economic_return_inputs")
    if bound_economic_inputs is not None:
        if not isinstance(bound_economic_inputs, Mapping) or set(
            bound_economic_inputs
        ) != {"artifact_hash", "payload"}:
            raise PeadReferenceError("economic return binding is malformed")
        economic_payload = bound_economic_inputs["payload"]
        if (
            not isinstance(economic_payload, Mapping)
            or economic_payload.get("schema_version")
            != ECONOMIC_RETURN_INPUTS_SCHEMA_VERSION
            or economic_payload.get("candidate_id") != CANDIDATE_ID
            or economic_payload.get("combined_data_snapshot_hash") != data_hash
            or content_hash(economic_payload)
            != bound_economic_inputs.get("artifact_hash")
        ):
            raise PeadReferenceError("economic return binding is inconsistent")
        try:
            validate_cash_distribution_semantics(
                economic_payload.get("cash_distribution_semantics")
            )
            validate_terminal_settlement_ledger(
                economic_payload.get("terminal_settlement_ledger")
            )
        except PeadEconomicEvidenceError as exc:
            raise PeadReferenceError("economic return binding is invalid") from exc
    contract = payload.get("replication_contract_inputs")
    if not isinstance(contract, Mapping) or contract.get(
        "unavailable_required_money_path_fields"
    ) != list(MONEY_PATH_FIELDS):
        raise PeadReferenceError("reference artifact hides missing money-path fields")
    if contract.get("unavailable_required_replication_fields") != list(
        UNAVAILABLE_REPLICATION_FIELDS
    ):
        raise PeadReferenceError("reference artifact hides unavailable fields")
    expected_tolerances = {
        "signal": {"absolute": 1e-12, "relative": 1e-12},
        "pit_marketcap": {"absolute": 1e-8, "relative": 1e-12},
        "entry_close_split_normalized": {
            "absolute": 1e-12, "relative": 1e-12,
        },
        "entry_closeunadj_execution_evidence": {
            "absolute": 1e-12, "relative": 1e-12,
        },
        "entry_closeadj_diagnostic": {
            "absolute": 1e-12, "relative": 1e-12,
        },
        "signal_preannouncement_close_split_normalized": {
            "absolute": 1e-12, "relative": 1e-12,
        },
        "signal_preannouncement_closeunadj_execution_evidence": {
            "absolute": 1e-12, "relative": 1e-12,
        },
        **{
            f"adjusted_forward_return_{horizon}": {
                "absolute": 1e-12, "relative": 1e-12,
            }
            for horizon in horizons
        },
        **{
            f"economic_forward_return_candidate_{horizon}": {
                "absolute": 1e-12, "relative": 1e-12,
            }
            for horizon in horizons
        },
    }
    if canonical_json(contract.get("numeric_tolerances")) != canonical_json(
        expected_tolerances
    ):
        raise PeadReferenceError("reference numeric tolerances are inconsistent")
    if contract.get("protocol_hash") != bindings.get("protocol_hash") or contract.get(
        "data_snapshot_hash"
    ) != data_hash:
        raise PeadReferenceError("replication contract inputs differ from bindings")
    reference_outputs = outputs["reference"]
    if canonical_json(payload.get("reference_coverage")) != canonical_json(
        reference_outputs.get("coverage")
    ):
        raise PeadReferenceError("reference coverage copies are inconsistent")
    expected_keys = sorted(
        [
            _portfolio_key(item)
            for item in reference_outputs["portfolio_observations"]
        ],
        key=canonical_json,
    )
    if canonical_json(contract.get("expected_observation_keys")) != canonical_json(
        expected_keys
    ):
        raise PeadReferenceError("expected observation key manifest is inconsistent")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or not {
        "event_driven_money_path_not_implemented",
        "generic_replication_evidence_not_available",
    }.issubset(blockers):
        raise PeadReferenceError("reference artifact omits required blockers")
    return payload


__all__ = [
    "CANDIDATE_ID",
    "MONEY_PATH_FIELDS",
    "UNAVAILABLE_REPLICATION_FIELDS",
    "PeadReferenceError",
    "ReferenceRun",
    "build_reference_comparison",
    "canonical_json",
    "content_hash",
    "reconstruct_events",
    "reconstruct_portfolio_observations",
    "run_reference_reconstruction",
    "validate_snapshot",
    "verify_reference_artifact",
]
