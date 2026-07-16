"""Trust-rooted Sharadar denominator evidence for PEAD source reconciliation.

This boundary is intentionally narrower than a signal receipt.  It replays a
candidate Sharadar source/identity snapshot from immutable warehouse bytes,
consumes the authoritative SF1 event-to-security lineage, revalidates the
upstream source reconciliation, and selects the unique latest SEP row on an
NYSE session date strictly before the conservative ``known_public_by``
Eastern date.  It never substitutes a same-day row, an older valid-looking
row, or another ticker.

The one-argument structural validator proves only internal consistency.  The
authoritative verifier always rebuilds the document from the warehouse,
official calendar source pages, and every original upstream input while also
requiring independently supplied trust registries.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

import duckdb

from analysis.pead_known_by_policy import (
    KNOWN_BY_POLICY,
    KNOWN_BY_POLICY_SHA256,
    PeadKnownByPolicyError,
    activation_eastern_date,
    validate_prospective_consensus_freeze,
)
from analysis.pead_source_reconciliation_v2 import (
    PeadSourceReconciliationV2Error,
    verify_pead_source_reconciliation_v2,
)
from data.pead_event_universe import (
    PeadEventUniverseError,
    canonical_event_id,
    canonical_json,
    content_hash,
    validate_event_key,
)
from data.session_close_calendar import (
    SessionCloseCalendarError,
    load_session_close_calendar_evidence,
)
from data.sharadar_source_evidence import (
    SharadarSourceEvidenceError,
    load_sharadar_table_acquisition,
    sharadar_source_record_sha256,
    validate_pead_security_identity_snapshot,
    validate_pead_sharadar_source_snapshot,
)


MARKET_ACCOUNTING_EVIDENCE_SCHEMA_VERSION = "pead_market_accounting_evidence.v1"
MARKET_ACCOUNTING_POLICY_SCHEMA_VERSION = "pead_market_accounting_policy.v1"
SEP_SEMANTIC_PROFILE_SCHEMA_VERSION = "pead_sep_semantic_profile.v1"
TRUST_ROOT_SET_SCHEMA_VERSION = "pead_sha256_trust_root_set.v1"
MAX_MARKET_ACCOUNTING_EVIDENCE_BYTES = 512 * 1024 * 1024
MAX_SEP_SEMANTIC_PROFILE_BYTES = 1024 * 1024

_EASTERN = ZoneInfo("America/New_York")
_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PROFILE_PAYLOAD_FIELDS = {
    "schema_version",
    "profile_id",
    "provider",
    "dataset",
    "created_at_utc",
    "source_evidence",
    "field_semantics",
    "normalization",
    "selection_policy",
}
_PROFILE_SOURCE_FIELDS = {
    "datatable_metadata_sha256",
    "official_semantics_source_receipt_sha256",
}
_PROFILE_FIELD_FIELDS = {"ticker", "session_date", "close", "closeunadj"}
_PROFILE_NORMALIZATION_FIELDS = {
    "factor_name",
    "factor_formula",
    "exact_representation",
    "decimal_precision",
    "decimal_rounding",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "evidence_class",
    "created_at_utc",
    "policy",
    "trust_policy",
    "bindings",
    "event_results",
    "coverage",
    "qualification",
}
_TRUST_POLICY_FIELDS = {
    "candidate_specification_set_sha256",
    "construction_code_set_sha256",
    "sharadar_source_snapshot_set_sha256",
    "security_identity_snapshot_set_sha256",
    "sharadar_event_replay_set_sha256",
    "source_reconciliation_set_sha256",
    "sep_semantic_profile_set_sha256",
    "nyse_calendar_set_sha256",
    "nyse_source_receipt_set_sha256",
}
_BINDING_FIELDS = {
    "sharadar_source_snapshot_sha256",
    "security_identity_snapshot_sha256",
    "sharadar_event_replay_sha256",
    "event_universe_index_sha256",
    "source_reconciliation_sha256",
    "source_reconciliation_event_universe_sha256",
    "sep_semantic_profile_sha256",
    "sep_acquisition_sha256",
    "sep_raw_zip_sha256",
    "sep_parquet_sha256",
    "nyse_calendar_sha256",
    "nyse_source_receipt_sha256",
    "known_by_policy_sha256",
    "market_accounting_policy_sha256",
}
_COVERAGE_FIELDS = {
    "upstream_event_count",
    "upstream_excluded_count",
    "source_reconciled_event_count",
    "market_accounting_evidenced_count",
    "market_accounting_excluded_count",
    "exhaustive_upstream_accounting",
    "exhaustive_source_reconciled_accounting",
    "event_blocker_counts",
}
_QUALIFICATION_FIELDS = {
    "has_market_accounting_evidence",
    "all_source_reconciled_events_evidenced",
    "market_accounting_evidence_allowed",
    "final_signal_receipt_required",
    "research_consumable",
    "edge_claim_allowed",
    "paper_execution_allowed",
    "live_deployment_allowed",
}
_EVENT_RESULT_FIELDS = {
    "event_id",
    "event_key",
    "upstream_disposition",
    "disposition",
    "blockers",
    "lineage",
    "market_denominator",
    "timing",
}
_LINEAGE_FIELDS = {
    "event_id",
    "event_key",
    "ticker",
    "permaticker",
    "identity_id",
    "representative_sf1_source_record_sha256",
    "sf1_source_record_sha256s",
    "sf1_revision_count",
}
_DENOMINATOR_FIELDS = {
    "ticker",
    "permaticker",
    "identity_id",
    "session_date",
    "session_close_at_utc",
    "session_close_kind",
    "close_split_normalized",
    "closeunadj_execution_evidence",
    "split_normalization_factor",
    "sep_source_row_sha256",
    "sep_acquisition_sha256",
    "sep_raw_zip_sha256",
    "sep_parquet_sha256",
}
_FACTOR_FIELDS = {"formula", "numerator", "denominator", "decimal_34"}
_TIMING_FIELDS = {
    "known_public_by_at_utc",
    "activation_eastern_date",
    "consensus_receipt_captured_at_utc",
    "prospective_freeze_required",
    "prospective_freeze_passed",
}
_MACHINE_REASON = re.compile(r"^[a-z][a-z0-9_]*$")

_OBSERVED_SESSION_RULE = (
    "SEP-observed sessions use the listed 13:00 early close when present and "
    "otherwise the NYSE 16:00 core-session close; dates absent from SEP receive "
    "no inferred session."
)
_FIELD_SEMANTICS = {
    "ticker": "Sharadar SEP security ticker used only through authoritative dated lineage",
    "session_date": "observed cash-equity session date from SEP.date",
    "close": "split-normalized closing price used with split-restated per-share signal inputs",
    "closeunadj": "contemporaneous unadjusted closing price retained as execution evidence",
}
_NORMALIZATION = {
    "factor_name": "closeunadj_per_split_normalized_close",
    "factor_formula": "SEP.closeunadj / SEP.close",
    "exact_representation": "reduced_rational_plus_decimal_34",
    "decimal_precision": 34,
    "decimal_rounding": "ROUND_HALF_EVEN",
}
_SELECTION_POLICY = {
    "known_by_policy_sha256": KNOWN_BY_POLICY_SHA256,
    "session_rule": "unique_latest_sep_session_date_strictly_before_activation_eastern_date",
    "price_rule": "selected_latest_row_must_have_positive_finite_close_and_closeunadj",
    "invalid_latest_row_rule": "exclude_without_older_fallback",
    "same_day_allowed": False,
    "ticker_fallback_allowed": False,
}
_POLICY = {
    "schema_version": MARKET_ACCOUNTING_POLICY_SCHEMA_VERSION,
    "known_by_policy": KNOWN_BY_POLICY,
    "known_by_policy_sha256": KNOWN_BY_POLICY_SHA256,
    "lineage_rule": "authoritative_sharadar_event_replay_only",
    "market_source_rule": "byte_revalidated_sharadar_sep_acquisition_only",
    "calendar_rule": "byte_replayed_official_nyse_source_receipt",
    "session_selection_rule": _SELECTION_POLICY,
    "split_normalization_rule": _NORMALIZATION,
    "prospective_freeze_rule": (
        "consensus_receipt_capture_no_later_than_selected_prior_session_close"
    ),
    "upstream_accounting_rule": "preserve_every_source_reconciliation_event_once",
    "return_accounting_allowed": False,
    "final_signal_receipt_required": True,
}


class PeadMarketAccountingEvidenceError(ValueError):
    """Market/accounting evidence is malformed or cannot replay authoritatively."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadMarketAccountingEvidenceError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadMarketAccountingEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadMarketAccountingEvidenceError(f"{label} must be nonempty canonical text")
    return value


def _utc(value: Any, label: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadMarketAccountingEvidenceError(f"{label} must be canonical UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise PeadMarketAccountingEvidenceError(f"{label} must be canonical UTC with Z") from exc
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if canonical != value:
        raise PeadMarketAccountingEvidenceError(f"{label} must be canonical UTC with Z")
    return canonical, parsed


def _day(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PeadMarketAccountingEvidenceError(f"{label} must be canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadMarketAccountingEvidenceError(f"{label} must be canonical YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise PeadMarketAccountingEvidenceError(f"{label} must be canonical YYYY-MM-DD")
    return parsed


def _trust_roots(values: Collection[str], label: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise PeadMarketAccountingEvidenceError(f"{label} must be a collection")
    return sorted({_sha(value, f"{label} entry") for value in values})


def _trust_set_hash(values: Sequence[str]) -> str:
    return content_hash({"schema_version": TRUST_ROOT_SET_SCHEMA_VERSION, "members": list(values)})


def _require_trusted(artifact_hash: str, roots: Sequence[str], label: str) -> None:
    if artifact_hash not in roots:
        raise PeadMarketAccountingEvidenceError(f"{label} is not in the external trust registry")


def build_pead_sep_semantic_profile(
    *,
    created_at_utc: str,
    datatable_metadata_sha256: str,
    official_semantics_source_receipt_sha256: str,
) -> dict[str, Any]:
    """Build the closed SEP field-semantics profile for external approval."""
    payload = {
        "schema_version": SEP_SEMANTIC_PROFILE_SCHEMA_VERSION,
        "profile_id": "sharadar-sep-close-basis-v1",
        "provider": "Sharadar",
        "dataset": "SHARADAR/SEP",
        "created_at_utc": _utc(created_at_utc, "profile created_at_utc")[0],
        "source_evidence": {
            "datatable_metadata_sha256": _sha(
                datatable_metadata_sha256, "datatable_metadata_sha256"
            ),
            "official_semantics_source_receipt_sha256": _sha(
                official_semantics_source_receipt_sha256,
                "official_semantics_source_receipt_sha256",
            ),
        },
        "field_semantics": _FIELD_SEMANTICS,
        "normalization": _NORMALIZATION,
        "selection_policy": _SELECTION_POLICY,
    }
    return validate_pead_sep_semantic_profile(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_sep_semantic_profile(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed profile structurally; external admission is separate."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "SEP semantic profile")
    payload = _exact(wrapper["payload"], _PROFILE_PAYLOAD_FIELDS, "SEP semantic profile.payload")
    claimed = _sha(wrapper["artifact_hash"], "SEP semantic profile artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadMarketAccountingEvidenceError("SEP semantic profile artifact hash mismatch")
    if payload["schema_version"] != SEP_SEMANTIC_PROFILE_SCHEMA_VERSION:
        raise PeadMarketAccountingEvidenceError("unsupported SEP semantic profile schema")
    if (
        payload["profile_id"] != "sharadar-sep-close-basis-v1"
        or payload["provider"] != "Sharadar"
        or payload["dataset"] != "SHARADAR/SEP"
    ):
        raise PeadMarketAccountingEvidenceError("SEP semantic profile identity differs")
    created = _utc(payload["created_at_utc"], "profile created_at_utc")[0]
    source = _exact(payload["source_evidence"], _PROFILE_SOURCE_FIELDS, "profile source_evidence")
    normalized_source = {
        field: _sha(source[field], f"profile source_evidence.{field}")
        for field in sorted(_PROFILE_SOURCE_FIELDS)
    }
    fields = _exact(payload["field_semantics"], _PROFILE_FIELD_FIELDS, "profile field_semantics")
    if dict(fields) != _FIELD_SEMANTICS:
        raise PeadMarketAccountingEvidenceError("SEP field semantics are not the closed profile")
    normalization = _exact(
        payload["normalization"], _PROFILE_NORMALIZATION_FIELDS, "profile normalization"
    )
    if dict(normalization) != _NORMALIZATION:
        raise PeadMarketAccountingEvidenceError("SEP normalization semantics differ")
    if payload["selection_policy"] != _SELECTION_POLICY:
        raise PeadMarketAccountingEvidenceError("SEP selection policy differs")
    normalized_payload = {
        "schema_version": SEP_SEMANTIC_PROFILE_SCHEMA_VERSION,
        "profile_id": "sharadar-sep-close-basis-v1",
        "provider": "Sharadar",
        "dataset": "SHARADAR/SEP",
        "created_at_utc": created,
        "source_evidence": normalized_source,
        "field_semantics": _FIELD_SEMANTICS,
        "normalization": _NORMALIZATION,
        "selection_policy": _SELECTION_POLICY,
    }
    if content_hash(normalized_payload) != claimed:
        raise PeadMarketAccountingEvidenceError("SEP semantic profile is not canonical")
    return {"artifact_hash": claimed, "payload": _plain(normalized_payload)}


def _resolve_under(root: Path, relative_path: str, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise PeadMarketAccountingEvidenceError(f"{label} path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PeadMarketAccountingEvidenceError(f"{label} path escapes the warehouse")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PeadMarketAccountingEvidenceError(f"{label} is missing or unsafe") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise PeadMarketAccountingEvidenceError(f"{label} is not a regular file")
    return resolved


def _sep_acquisition(
    source_snapshot: Mapping[str, Any], *, warehouse_dir: Path
) -> tuple[dict[str, Any], Path]:
    table = next(
        (row for row in source_snapshot["payload"]["tables"] if row["logical_name"] == "sep"),
        None,
    )
    if table is None:
        raise PeadMarketAccountingEvidenceError("Sharadar source snapshot has no SEP table")
    receipt_path = _resolve_under(
        warehouse_dir,
        table["acquisition_receipt_relative_path"],
        label="SEP acquisition receipt",
    )
    try:
        acquisition = load_sharadar_table_acquisition(receipt_path, warehouse_dir=warehouse_dir)
    except SharadarSourceEvidenceError as exc:
        raise PeadMarketAccountingEvidenceError(
            "SEP acquisition does not replay from immutable bytes"
        ) from exc
    if acquisition["artifact_hash"] != table["acquisition_artifact_hash"]:
        raise PeadMarketAccountingEvidenceError("SEP snapshot and acquisition identities differ")
    parquet_path = _resolve_under(
        warehouse_dir,
        acquisition["payload"]["parquet"]["relative_path"],
        label="SEP Parquet",
    )
    return acquisition, parquet_path


def _calendar_model(evidence: Mapping[str, Any]) -> dict[str, Any]:
    calendar = evidence["calendar"]
    receipt = evidence["source_receipt"]
    payload = calendar.get("payload")
    expected_fields = {
        "schema_version",
        "venue",
        "timezone",
        "coverage",
        "regular_close_local_time",
        "early_close_local_time",
        "observed_session_rule",
        "early_close_sessions",
        "sources",
    }
    _exact(payload, expected_fields, "NYSE calendar payload")
    if (
        payload["schema_version"] != "nyse_session_close_calendar.v1"
        or payload["venue"] != "NYSE cash equities"
        or payload["timezone"] != "America/New_York"
        or payload["regular_close_local_time"] != "16:00:00"
        or payload["early_close_local_time"] != "13:00:00"
        or payload["observed_session_rule"] != _OBSERVED_SESSION_RULE
    ):
        raise PeadMarketAccountingEvidenceError("NYSE calendar policy differs")
    coverage = _exact(payload["coverage"], {"start", "end"}, "NYSE calendar coverage")
    start = _day(coverage["start"], "NYSE calendar coverage.start")
    end = _day(coverage["end"], "NYSE calendar coverage.end")
    if start > end:
        raise PeadMarketAccountingEvidenceError("NYSE calendar coverage is reversed")

    receipt_sources = receipt["payload"]["sources"]
    extracted_by_source = {
        row["source_id"]: {
            _day(value, f"source {row['source_id']} early-close date")
            for value in row["extraction"]["early_close_dates"]
        }
        for row in receipt_sources
    }
    rows = payload["early_close_sessions"]
    if not isinstance(rows, list):
        raise PeadMarketAccountingEvidenceError("early_close_sessions must be an array")
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        row = _exact(raw, {"date", "source_id"}, f"early_close_sessions[{index}]")
        session_day = _day(row["date"], f"early_close_sessions[{index}].date")
        source_id = _text(row["source_id"], f"early_close_sessions[{index}].source_id")
        if (
            source_id not in extracted_by_source
            or session_day not in extracted_by_source[source_id]
        ):
            raise PeadMarketAccountingEvidenceError(
                "calendar early close is not replayed by its official source"
            )
        if not start <= session_day <= end:
            raise PeadMarketAccountingEvidenceError("calendar early close is outside coverage")
        normalized.append({"date": session_day.isoformat(), "source_id": source_id})
    if normalized != sorted(normalized, key=lambda row: row["date"]):
        raise PeadMarketAccountingEvidenceError("calendar early closes are not sorted")
    dates = [row["date"] for row in normalized]
    if len(dates) != len(set(dates)):
        raise PeadMarketAccountingEvidenceError("calendar early-close dates are duplicated")
    extracted_union = {
        item
        for source_dates in extracted_by_source.values()
        for item in source_dates
        if start <= item <= end
    }
    if {date.fromisoformat(value) for value in dates} != extracted_union:
        raise PeadMarketAccountingEvidenceError(
            "calendar early closes do not equal the official source union"
        )
    return {
        "start": start,
        "end": end,
        "early_dates": set(dates),
        "calendar": calendar,
        "receipt": receipt,
    }


def _session_close(session_day: date, calendar: Mapping[str, Any]) -> tuple[str, str]:
    if not calendar["start"] <= session_day <= calendar["end"]:
        raise PeadMarketAccountingEvidenceError("selected SEP date is outside calendar coverage")
    early = session_day.isoformat() in calendar["early_dates"]
    close_time = time(13, 0) if early else time(16, 0)
    close = datetime.combine(session_day, close_time, tzinfo=_EASTERN).astimezone(timezone.utc)
    return close.isoformat(timespec="seconds").replace("+00:00", "Z"), (
        "official_early_close" if early else "regular_core_close"
    )


def _positive_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise PeadMarketAccountingEvidenceError(f"{label} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise PeadMarketAccountingEvidenceError(f"{label} must be positive and finite")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise PeadMarketAccountingEvidenceError(f"{label} must be a decimal") from exc
    if not result.is_finite() or result <= 0:
        raise PeadMarketAccountingEvidenceError(f"{label} must be positive and finite")
    return result


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _normalization_factor(close: Decimal, closeunadj: Decimal) -> dict[str, Any]:
    exact = Fraction(closeunadj) / Fraction(close)
    with localcontext() as context:
        context.prec = 34
        context.rounding = ROUND_HALF_EVEN
        decimal_value = Decimal(exact.numerator) / Decimal(exact.denominator)
    return {
        "formula": "closeunadj / close",
        "numerator": exact.numerator,
        "denominator": exact.denominator,
        "decimal_34": _canonical_decimal(decimal_value),
    }


def _query_latest_sep_rows(
    *,
    parquet_path: Path,
    schema: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    if not requests:
        return {}
    columns = [row["name"] for row in schema]
    if len(columns) != len(set(columns)) or not {"ticker", "date", "close", "closeunadj"} <= set(
        columns
    ):
        raise PeadMarketAccountingEvidenceError("SEP schema lacks required unique columns")

    def identifier(value: str) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise PeadMarketAccountingEvidenceError("SEP schema contains an invalid identifier")
        return '"' + value.replace('"', '""') + '"'

    selected_columns = ",".join(f"s.{identifier(column)}" for column in columns)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            "CREATE TEMP TABLE event_requests "
            "(event_id VARCHAR NOT NULL, ticker VARCHAR NOT NULL, cutoff DATE NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO event_requests VALUES (?, ?, ?)",
            [(row["event_id"], row["ticker"], row["cutoff"]) for row in requests],
        )
        query = (
            f"SELECT r.event_id,{selected_columns} "
            "FROM read_parquet(?) AS s "
            "JOIN event_requests AS r ON s.ticker = r.ticker AND s.date < r.cutoff "
            "QUALIFY s.date = max(s.date) OVER (PARTITION BY r.event_id) "
            "ORDER BY r.event_id,s.date"
        )
        raw_rows = connection.execute(query, [str(parquet_path)]).fetchall()
    except duckdb.Error as exc:
        raise PeadMarketAccountingEvidenceError("exact SEP rows cannot be queried") from exc
    finally:
        connection.close()
    result: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_rows:
        event_id = raw[0]
        row = dict(zip(columns, raw[1:], strict=True))
        result.setdefault(event_id, []).append(row)
    return result


def _lineage_map(replay: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    rows: dict[str, dict[str, Any]] = {}
    universe_hashes: set[str] = set()
    for year in replay["payload"]["years"]:
        universe = year["event_universe"]
        universe_hashes.add(universe["artifact_hash"])
        for raw in year["event_lineage"]:
            event_id = raw["event_id"]
            if event_id in rows:
                raise PeadMarketAccountingEvidenceError("event replay duplicates event lineage")
            rows[event_id] = _plain(raw)
    return rows, universe_hashes


def _validate_lineage(
    value: Any,
    *,
    event_id: str,
    event_key: Mapping[str, Any],
    label: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    row = _exact(value, _LINEAGE_FIELDS, label)
    if row["event_id"] != event_id or row["event_key"] != event_key:
        raise PeadMarketAccountingEvidenceError(f"{label} differs from its event")
    ticker = _text(row["ticker"], f"{label}.ticker")
    if ticker != ticker.upper():
        raise PeadMarketAccountingEvidenceError(f"{label}.ticker must be uppercase")
    permaticker = row["permaticker"]
    if type(permaticker) is not int or permaticker <= 0:
        raise PeadMarketAccountingEvidenceError(f"{label}.permaticker must be a positive integer")
    raw_hashes = row["sf1_source_record_sha256s"]
    if not isinstance(raw_hashes, list):
        raise PeadMarketAccountingEvidenceError(
            f"{label}.sf1_source_record_sha256s must be an array"
        )
    hashes = [_sha(value, f"{label}.sf1_source_record_sha256s entry") for value in raw_hashes]
    if not hashes or hashes != sorted(set(hashes)):
        raise PeadMarketAccountingEvidenceError(
            f"{label}.sf1 source hashes must be nonempty sorted and unique"
        )
    representative = _sha(
        row["representative_sf1_source_record_sha256"],
        f"{label}.representative_sf1_source_record_sha256",
    )
    if representative not in hashes:
        raise PeadMarketAccountingEvidenceError(
            f"{label} representative SF1 row is absent from its revisions"
        )
    revision_count = row["sf1_revision_count"]
    if type(revision_count) is not int or revision_count != len(hashes):
        raise PeadMarketAccountingEvidenceError(
            f"{label}.sf1_revision_count differs from its source rows"
        )
    return {
        "event_id": event_id,
        "event_key": dict(event_key),
        "ticker": ticker,
        "permaticker": permaticker,
        "identity_id": _sha(row["identity_id"], f"{label}.identity_id"),
        "representative_sf1_source_record_sha256": representative,
        "sf1_source_record_sha256s": hashes,
        "sf1_revision_count": revision_count,
    }


def _validate_denominator(
    value: Any,
    *,
    lineage: Mapping[str, Any],
    bindings: Mapping[str, Any],
    activation_day: date,
    known_public_by: datetime,
    label: str,
) -> dict[str, Any]:
    row = _exact(value, _DENOMINATOR_FIELDS, label)
    if (
        row["ticker"] != lineage["ticker"]
        or row["permaticker"] != lineage["permaticker"]
        or row["identity_id"] != lineage["identity_id"]
    ):
        raise PeadMarketAccountingEvidenceError(f"{label} differs from event lineage")
    session_day = _day(row["session_date"], f"{label}.session_date")
    if session_day >= activation_day:
        raise PeadMarketAccountingEvidenceError(f"{label} uses a same-day or future market session")
    close_at_text, close_at = _utc(row["session_close_at_utc"], f"{label}.session_close_at_utc")
    if close_at >= known_public_by:
        raise PeadMarketAccountingEvidenceError(
            f"{label} session close is not earlier than known-public time"
        )
    if row["session_close_kind"] not in {"official_early_close", "regular_core_close"}:
        raise PeadMarketAccountingEvidenceError(f"{label} close kind is unsupported")
    close = _positive_decimal(row["close_split_normalized"], f"{label}.close_split_normalized")
    closeunadj = _positive_decimal(
        row["closeunadj_execution_evidence"],
        f"{label}.closeunadj_execution_evidence",
    )
    if row["close_split_normalized"] != _canonical_decimal(close) or row[
        "closeunadj_execution_evidence"
    ] != _canonical_decimal(closeunadj):
        raise PeadMarketAccountingEvidenceError(f"{label} price decimals are not canonical")
    factor = _exact(
        row["split_normalization_factor"],
        _FACTOR_FIELDS,
        f"{label}.split_normalization_factor",
    )
    if dict(factor) != _normalization_factor(close, closeunadj):
        raise PeadMarketAccountingEvidenceError(
            f"{label} split-normalization factor is not derived"
        )
    hash_bindings = {
        "sep_acquisition_sha256": "sep_acquisition_sha256",
        "sep_raw_zip_sha256": "sep_raw_zip_sha256",
        "sep_parquet_sha256": "sep_parquet_sha256",
    }
    for field, binding in hash_bindings.items():
        if _sha(row[field], f"{label}.{field}") != bindings[binding]:
            raise PeadMarketAccountingEvidenceError(
                f"{label}.{field} differs from the receipt binding"
            )
    return {
        "ticker": lineage["ticker"],
        "permaticker": lineage["permaticker"],
        "identity_id": lineage["identity_id"],
        "session_date": session_day.isoformat(),
        "session_close_at_utc": close_at_text,
        "session_close_kind": row["session_close_kind"],
        "close_split_normalized": _canonical_decimal(close),
        "closeunadj_execution_evidence": _canonical_decimal(closeunadj),
        "split_normalization_factor": dict(factor),
        "sep_source_row_sha256": _sha(
            row["sep_source_row_sha256"], f"{label}.sep_source_row_sha256"
        ),
        "sep_acquisition_sha256": bindings["sep_acquisition_sha256"],
        "sep_raw_zip_sha256": bindings["sep_raw_zip_sha256"],
        "sep_parquet_sha256": bindings["sep_parquet_sha256"],
    }


def _identity_matches_lineage(
    lineage: Mapping[str, Any],
    event_key: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    identity = identities.get(lineage["identity_id"])
    if identity is None:
        return None
    if (
        identity["cik"] != event_key["cik"]
        or identity["ticker"] != lineage["ticker"]
        or identity["permaticker"] != lineage["permaticker"]
        or not identity["valid_from"] <= event_key["fiscal_period_end"] <= identity["valid_through"]
    ):
        return None
    return identity


def build_pead_market_accounting_evidence(
    source_reconciliation: Mapping[str, Any],
    sharadar_event_replay: Mapping[str, Any],
    event_universe_index: Mapping[str, Any],
    sharadar_source_snapshot: Mapping[str, Any],
    security_identity_snapshot: Mapping[str, Any],
    sep_semantic_profile: Mapping[str, Any],
    *,
    warehouse_dir: str | os.PathLike[str],
    candidate_specification_path: str | os.PathLike[str],
    construction_code_path: str | os.PathLike[str],
    calendar_path: str | Path,
    calendar_receipt_path: str | Path,
    created_at_utc: str,
    source_reconciliation_verification_kwargs: Mapping[str, Any],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
    trusted_sharadar_source_snapshot_sha256s: Collection[str],
    trusted_security_identity_snapshot_sha256s: Collection[str],
    trusted_sharadar_event_replay_sha256s: Collection[str],
    trusted_source_reconciliation_sha256s: Collection[str],
    trusted_sep_semantic_profile_sha256s: Collection[str],
    trusted_nyse_calendar_sha256s: Collection[str],
    trusted_nyse_source_receipt_sha256s: Collection[str],
) -> dict[str, Any]:
    """Build authoritative per-event denominator evidence from immutable inputs."""
    if not isinstance(source_reconciliation_verification_kwargs, Mapping):
        raise PeadMarketAccountingEvidenceError(
            "source_reconciliation_verification_kwargs must be a mapping"
        )
    trust_sets = {
        "candidate_specification": _trust_roots(
            trusted_candidate_specification_sha256s,
            "trusted_candidate_specification_sha256s",
        ),
        "construction_code": _trust_roots(
            trusted_construction_code_sha256s,
            "trusted_construction_code_sha256s",
        ),
        "source_snapshot": _trust_roots(
            trusted_sharadar_source_snapshot_sha256s,
            "trusted_sharadar_source_snapshot_sha256s",
        ),
        "identity_snapshot": _trust_roots(
            trusted_security_identity_snapshot_sha256s,
            "trusted_security_identity_snapshot_sha256s",
        ),
        "event_replay": _trust_roots(
            trusted_sharadar_event_replay_sha256s,
            "trusted_sharadar_event_replay_sha256s",
        ),
        "source_reconciliation": _trust_roots(
            trusted_source_reconciliation_sha256s,
            "trusted_source_reconciliation_sha256s",
        ),
        "sep_profile": _trust_roots(
            trusted_sep_semantic_profile_sha256s,
            "trusted_sep_semantic_profile_sha256s",
        ),
        "calendar": _trust_roots(trusted_nyse_calendar_sha256s, "trusted_nyse_calendar_sha256s"),
        "calendar_receipt": _trust_roots(
            trusted_nyse_source_receipt_sha256s,
            "trusted_nyse_source_receipt_sha256s",
        ),
    }
    if not trust_sets["candidate_specification"]:
        raise PeadMarketAccountingEvidenceError(
            "candidate specification external trust registry is empty"
        )
    if not trust_sets["construction_code"]:
        raise PeadMarketAccountingEvidenceError(
            "construction code external trust registry is empty"
        )
    for document, key, label in (
        (sharadar_source_snapshot, "source_snapshot", "Sharadar source snapshot"),
        (security_identity_snapshot, "identity_snapshot", "security identity snapshot"),
        (sharadar_event_replay, "event_replay", "Sharadar event replay"),
        (source_reconciliation, "source_reconciliation", "source reconciliation"),
        (sep_semantic_profile, "sep_profile", "SEP semantic profile"),
    ):
        claimed = _sha(document.get("artifact_hash"), f"{label} artifact_hash")
        _require_trusted(claimed, trust_sets[key], label)

    root = Path(warehouse_dir).resolve()
    try:
        source_snapshot = validate_pead_sharadar_source_snapshot(
            sharadar_source_snapshot, warehouse_dir=root
        )
        identity_snapshot = validate_pead_security_identity_snapshot(
            security_identity_snapshot,
            warehouse_dir=root,
            source_snapshot=source_snapshot,
        )
    except SharadarSourceEvidenceError as exc:
        raise PeadMarketAccountingEvidenceError(
            "Sharadar source or identity snapshot fails immutable-byte replay"
        ) from exc
    if not source_snapshot["payload"]["qualification_allowed"]:
        raise PeadMarketAccountingEvidenceError("Sharadar source snapshot is not qualified")
    if not identity_snapshot["payload"]["qualification_allowed"]:
        raise PeadMarketAccountingEvidenceError("security identity snapshot is not qualified")

    try:
        from data.pead_sharadar_event_universe_replay import (
            PeadSharadarEventUniverseReplayError,
            verify_pead_sharadar_event_universe_replay,
        )
    except ImportError as exc:
        raise PeadMarketAccountingEvidenceError(
            "Sharadar event-universe replay implementation is unavailable"
        ) from exc
    try:
        verified_bundle = verify_pead_sharadar_event_universe_replay(
            sharadar_event_replay,
            event_universe_index,
            warehouse_dir=root,
            source_snapshot=source_snapshot,
            identity_snapshot=identity_snapshot,
            candidate_specification_path=candidate_specification_path,
            construction_code_path=construction_code_path,
            trusted_candidate_specification_sha256s=trust_sets["candidate_specification"],
            trusted_construction_code_sha256s=trust_sets["construction_code"],
        )
    except (PeadSharadarEventUniverseReplayError, TypeError) as exc:
        raise PeadMarketAccountingEvidenceError(
            "Sharadar event-universe replay is not authoritative"
        ) from exc
    if isinstance(verified_bundle, Mapping) and set(verified_bundle) == {"replay", "index"}:
        event_replay = verified_bundle["replay"]
        universe_index = verified_bundle["index"]
    elif isinstance(verified_bundle, tuple) and len(verified_bundle) == 2:
        event_replay, universe_index = verified_bundle
    else:
        raise PeadMarketAccountingEvidenceError("event replay verifier returned an invalid bundle")
    if not event_replay["payload"]["qualification_allowed"]:
        raise PeadMarketAccountingEvidenceError("Sharadar event replay is not qualified")

    try:
        reconciliation = verify_pead_source_reconciliation_v2(
            source_reconciliation, **dict(source_reconciliation_verification_kwargs)
        )
    except (PeadSourceReconciliationV2Error, TypeError, ValueError) as exc:
        raise PeadMarketAccountingEvidenceError(
            "source reconciliation does not replay authoritatively"
        ) from exc

    profile = validate_pead_sep_semantic_profile(sep_semantic_profile)
    acquisition, sep_parquet_path = _sep_acquisition(source_snapshot, warehouse_dir=root)
    sep_payload = acquisition["payload"]
    metadata_hash = content_hash(sep_payload["source"]["datatable_metadata"])
    if profile["payload"]["source_evidence"]["datatable_metadata_sha256"] != metadata_hash:
        raise PeadMarketAccountingEvidenceError(
            "SEP semantic profile does not bind the acquired official metadata"
        )

    try:
        calendar_evidence = load_session_close_calendar_evidence(
            calendar_path=calendar_path, receipt_path=calendar_receipt_path
        )
    except SessionCloseCalendarError as exc:
        raise PeadMarketAccountingEvidenceError(
            "NYSE calendar source evidence does not replay"
        ) from exc
    calendar = _calendar_model(calendar_evidence)
    _require_trusted(calendar["calendar"]["artifact_hash"], trust_sets["calendar"], "NYSE calendar")
    _require_trusted(
        calendar["receipt"]["artifact_hash"],
        trust_sets["calendar_receipt"],
        "NYSE source receipt",
    )

    candidate = source_snapshot["payload"]["candidate_id"]
    if any(
        document["payload"]["candidate_id"] != candidate
        for document in (identity_snapshot, event_replay, reconciliation)
    ):
        raise PeadMarketAccountingEvidenceError(
            "market evidence inputs belong to different candidates"
        )
    if (
        event_replay["payload"]["bindings"]["source_snapshot_sha256"]
        != source_snapshot["artifact_hash"]
    ):
        raise PeadMarketAccountingEvidenceError("event replay binds another source snapshot")
    if (
        event_replay["payload"]["bindings"]["identity_snapshot_sha256"]
        != identity_snapshot["artifact_hash"]
    ):
        raise PeadMarketAccountingEvidenceError("event replay binds another identity snapshot")

    lineage_by_id, replay_universe_hashes = _lineage_map(event_replay)
    reconciliation_universe_hash = reconciliation["payload"]["bindings"]["event_universe_sha256"]
    if reconciliation_universe_hash not in replay_universe_hashes:
        raise PeadMarketAccountingEvidenceError(
            "source reconciliation universe is absent from authoritative event replay"
        )
    identities = {row["identity_id"]: row for row in identity_snapshot["payload"]["identities"]}

    reconciled_rows = reconciliation["payload"]["event_results"]
    event_ids = [row["event_id"] for row in reconciled_rows]
    if len(event_ids) != len(set(event_ids)):
        raise PeadMarketAccountingEvidenceError("source reconciliation duplicates event IDs")
    prepared: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, str]] = []
    for row in reconciled_rows:
        if row["disposition"] != "event_source_reconciled":
            continue
        event_id = row["event_id"]
        source_input = row["event_source_input"]
        lineage = lineage_by_id.get(event_id)
        blockers: list[str] = []
        identity: Mapping[str, Any] | None = None
        if lineage is None:
            blockers.append("authoritative_event_lineage_missing")
        elif lineage["event_key"] != row["event_key"]:
            blockers.append("event_lineage_key_mismatch")
        else:
            identity = _identity_matches_lineage(lineage, row["event_key"], identities)
            if identity is None:
                blockers.append("event_lineage_identity_mismatch")
        try:
            activation_day = activation_eastern_date(source_input["known_public_by_at_utc"])
        except PeadKnownByPolicyError as exc:
            raise PeadMarketAccountingEvidenceError(
                "source reconciliation has invalid known-public time"
            ) from exc
        prepared[event_id] = {
            "lineage": lineage,
            "identity": identity,
            "blockers": blockers,
            "activation_day": activation_day,
            "source_input": source_input,
        }
        if not blockers and lineage is not None:
            requests.append(
                {
                    "event_id": event_id,
                    "ticker": lineage["ticker"],
                    "cutoff": activation_day.isoformat(),
                }
            )

    sep_schema = sep_payload["parquet"]["schema"]
    selected_rows = _query_latest_sep_rows(
        parquet_path=sep_parquet_path, schema=sep_schema, requests=requests
    )
    evidence_class = reconciliation["payload"]["evidence_class"]
    if evidence_class not in {"historical_reconstruction", "prospective_signal"}:
        raise PeadMarketAccountingEvidenceError("unsupported source evidence class")

    results: list[dict[str, Any]] = []
    for upstream in reconciled_rows:
        base = {
            "event_id": upstream["event_id"],
            "event_key": upstream["event_key"],
            "upstream_disposition": upstream["disposition"],
        }
        if upstream["disposition"] != "event_source_reconciled":
            results.append(
                {
                    **base,
                    "disposition": "upstream_excluded",
                    "blockers": ["upstream_not_event_source_reconciled"],
                    "lineage": None,
                    "market_denominator": None,
                    "timing": None,
                }
            )
            continue

        state = prepared[upstream["event_id"]]
        blockers = list(state["blockers"])
        lineage = state["lineage"]
        identity = state["identity"]
        candidates = selected_rows.get(upstream["event_id"], [])
        if not blockers:
            if not candidates:
                blockers.append("market_prior_session_absent")
            elif len(candidates) != 1:
                blockers.append("market_latest_prior_session_ambiguous")

        denominator: dict[str, Any] | None = None
        close_at_utc: str | None = None
        close_kind: str | None = None
        if not blockers:
            selected = candidates[0]
            raw_day = selected["date"]
            if isinstance(raw_day, datetime):
                session_day = raw_day.date()
            elif isinstance(raw_day, date):
                session_day = raw_day
            else:
                session_day = _day(raw_day, "SEP session date")
            if session_day >= state["activation_day"]:
                raise PeadMarketAccountingEvidenceError(
                    "SEP query returned a same-day or future row"
                )
            try:
                close = _positive_decimal(selected["close"], "SEP.close")
                closeunadj = _positive_decimal(selected["closeunadj"], "SEP.closeunadj")
            except PeadMarketAccountingEvidenceError:
                blockers.append("market_latest_prior_session_price_invalid")
            if identity is not None and not (
                identity["valid_from"] <= session_day.isoformat() <= identity["valid_through"]
            ):
                blockers.append("event_lineage_not_valid_on_selected_session")
            if not blockers:
                close_at_utc, close_kind = _session_close(session_day, calendar)
                source_hash = sharadar_source_record_sha256("sep", sep_schema, selected)
                denominator = {
                    "ticker": lineage["ticker"],
                    "permaticker": lineage["permaticker"],
                    "identity_id": lineage["identity_id"],
                    "session_date": session_day.isoformat(),
                    "session_close_at_utc": close_at_utc,
                    "session_close_kind": close_kind,
                    "close_split_normalized": _canonical_decimal(close),
                    "closeunadj_execution_evidence": _canonical_decimal(closeunadj),
                    "split_normalization_factor": _normalization_factor(close, closeunadj),
                    "sep_source_row_sha256": source_hash,
                    "sep_acquisition_sha256": acquisition["artifact_hash"],
                    "sep_raw_zip_sha256": sep_payload["raw_zip"]["sha256"],
                    "sep_parquet_sha256": sep_payload["parquet"]["sha256"],
                }

        source_input = state["source_input"]
        prospective_required = evidence_class == "prospective_signal"
        freeze_passed: bool | None = None
        if not blockers and prospective_required:
            try:
                validate_prospective_consensus_freeze(
                    acquired_at_utc=source_input["consensus_receipt_captured_at_utc"],
                    selected_prior_session_close_at_utc=close_at_utc,
                )
            except PeadKnownByPolicyError:
                blockers.append("prospective_consensus_capture_after_prior_close")
                freeze_passed = False
            else:
                freeze_passed = True
        timing = {
            "known_public_by_at_utc": source_input["known_public_by_at_utc"],
            "activation_eastern_date": state["activation_day"].isoformat(),
            "consensus_receipt_captured_at_utc": source_input["consensus_receipt_captured_at_utc"],
            "prospective_freeze_required": prospective_required,
            "prospective_freeze_passed": freeze_passed,
        }
        results.append(
            {
                **base,
                "disposition": (
                    "market_accounting_evidenced" if not blockers else "market_accounting_excluded"
                ),
                "blockers": sorted(set(blockers)),
                "lineage": lineage,
                "market_denominator": denominator,
                "timing": timing,
            }
        )

    source_reconciled_count = sum(
        row["upstream_disposition"] == "event_source_reconciled" for row in results
    )
    evidenced_count = sum(row["disposition"] == "market_accounting_evidenced" for row in results)
    blocker_counts: dict[str, int] = {}
    for row in results:
        for blocker in row["blockers"]:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    all_evidenced = source_reconciled_count > 0 and evidenced_count == source_reconciled_count

    created_text, created_at = _utc(created_at_utc, "created_at_utc")
    input_times = [
        _utc(source_snapshot["payload"]["created_at_utc"], "source snapshot creation")[1],
        _utc(identity_snapshot["payload"]["created_at_utc"], "identity creation")[1],
        _utc(event_replay["payload"]["created_at_utc"], "event replay creation")[1],
        _utc(reconciliation["payload"]["reconciled_at_utc"], "source reconciliation creation")[1],
        _utc(profile["payload"]["created_at_utc"], "SEP profile creation")[1],
        _utc(calendar["receipt"]["payload"]["created_at_utc"], "calendar receipt creation")[1],
    ]
    if created_at < max(input_times):
        raise PeadMarketAccountingEvidenceError("market evidence predates an input artifact")

    trust_policy = {
        "candidate_specification_set_sha256": _trust_set_hash(
            trust_sets["candidate_specification"]
        ),
        "construction_code_set_sha256": _trust_set_hash(trust_sets["construction_code"]),
        "sharadar_source_snapshot_set_sha256": _trust_set_hash(trust_sets["source_snapshot"]),
        "security_identity_snapshot_set_sha256": _trust_set_hash(trust_sets["identity_snapshot"]),
        "sharadar_event_replay_set_sha256": _trust_set_hash(trust_sets["event_replay"]),
        "source_reconciliation_set_sha256": _trust_set_hash(trust_sets["source_reconciliation"]),
        "sep_semantic_profile_set_sha256": _trust_set_hash(trust_sets["sep_profile"]),
        "nyse_calendar_set_sha256": _trust_set_hash(trust_sets["calendar"]),
        "nyse_source_receipt_set_sha256": _trust_set_hash(trust_sets["calendar_receipt"]),
    }
    bindings = {
        "sharadar_source_snapshot_sha256": source_snapshot["artifact_hash"],
        "security_identity_snapshot_sha256": identity_snapshot["artifact_hash"],
        "sharadar_event_replay_sha256": event_replay["artifact_hash"],
        "event_universe_index_sha256": universe_index["artifact_hash"],
        "source_reconciliation_sha256": reconciliation["artifact_hash"],
        "source_reconciliation_event_universe_sha256": reconciliation_universe_hash,
        "sep_semantic_profile_sha256": profile["artifact_hash"],
        "sep_acquisition_sha256": acquisition["artifact_hash"],
        "sep_raw_zip_sha256": sep_payload["raw_zip"]["sha256"],
        "sep_parquet_sha256": sep_payload["parquet"]["sha256"],
        "nyse_calendar_sha256": calendar["calendar"]["artifact_hash"],
        "nyse_source_receipt_sha256": calendar["receipt"]["artifact_hash"],
        "known_by_policy_sha256": KNOWN_BY_POLICY_SHA256,
        "market_accounting_policy_sha256": content_hash(_POLICY),
    }
    payload = {
        "schema_version": MARKET_ACCOUNTING_EVIDENCE_SCHEMA_VERSION,
        "candidate_id": candidate,
        "evidence_class": evidence_class,
        "created_at_utc": created_text,
        "policy": _POLICY,
        "trust_policy": trust_policy,
        "bindings": bindings,
        "event_results": results,
        "coverage": {
            "upstream_event_count": len(results),
            "upstream_excluded_count": sum(
                row["disposition"] == "upstream_excluded" for row in results
            ),
            "source_reconciled_event_count": source_reconciled_count,
            "market_accounting_evidenced_count": evidenced_count,
            "market_accounting_excluded_count": source_reconciled_count - evidenced_count,
            "exhaustive_upstream_accounting": [row["event_id"] for row in results] == event_ids,
            "exhaustive_source_reconciled_accounting": (
                sum(
                    row["disposition"]
                    in {"market_accounting_evidenced", "market_accounting_excluded"}
                    for row in results
                )
                == source_reconciled_count
            ),
            "event_blocker_counts": {key: blocker_counts[key] for key in sorted(blocker_counts)},
        },
        "qualification": {
            "has_market_accounting_evidence": evidenced_count > 0,
            "all_source_reconciled_events_evidenced": all_evidenced,
            "market_accounting_evidence_allowed": all_evidenced,
            "final_signal_receipt_required": True,
            "research_consumable": False,
            "edge_claim_allowed": False,
            "paper_execution_allowed": False,
            "live_deployment_allowed": False,
        },
    }
    return validate_pead_market_accounting_evidence_structure(
        {"artifact_hash": content_hash(payload), "payload": payload}
    )


def validate_pead_market_accounting_evidence_structure(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate structure and content identity only; this is not authoritative."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "market accounting evidence")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "market accounting evidence.payload")
    claimed = _sha(wrapper["artifact_hash"], "market accounting evidence artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadMarketAccountingEvidenceError("market accounting artifact hash mismatch")
    if payload["schema_version"] != MARKET_ACCOUNTING_EVIDENCE_SCHEMA_VERSION:
        raise PeadMarketAccountingEvidenceError("unsupported market accounting schema")
    _text(payload["candidate_id"], "candidate_id")
    if payload["evidence_class"] not in {"historical_reconstruction", "prospective_signal"}:
        raise PeadMarketAccountingEvidenceError("unsupported market accounting evidence class")
    _utc(payload["created_at_utc"], "created_at_utc")
    if payload["policy"] != _POLICY:
        raise PeadMarketAccountingEvidenceError("market accounting policy differs")
    trust_policy = _exact(payload["trust_policy"], _TRUST_POLICY_FIELDS, "trust_policy")
    for field in sorted(_TRUST_POLICY_FIELDS):
        _sha(trust_policy[field], f"trust_policy.{field}")
    bindings = _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    for field in sorted(_BINDING_FIELDS):
        _sha(bindings[field], f"bindings.{field}")
    coverage = _exact(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    qualification = _exact(payload["qualification"], _QUALIFICATION_FIELDS, "qualification")
    rows = payload["event_results"]
    if not isinstance(rows, list):
        raise PeadMarketAccountingEvidenceError("event_results must be an array")
    normalized_rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    evidence_class = payload["evidence_class"]
    for index, raw in enumerate(rows):
        row = _exact(raw, _EVENT_RESULT_FIELDS, f"event_results[{index}]")
        try:
            event_key = validate_event_key(
                row["event_key"], label=f"event_results[{index}].event_key"
            )
        except PeadEventUniverseError as exc:
            raise PeadMarketAccountingEvidenceError(
                f"event_results[{index}] has an invalid event key"
            ) from exc
        event_id = _sha(row["event_id"], f"event_results[{index}].event_id")
        if event_id != canonical_event_id(event_key):
            raise PeadMarketAccountingEvidenceError(
                f"event_results[{index}].event_id differs from its event key"
            )
        blockers = row["blockers"]
        if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
            raise PeadMarketAccountingEvidenceError(
                f"event_results[{index}].blockers must be sorted and unique"
            )
        for blocker in blockers:
            if not isinstance(blocker, str) or _MACHINE_REASON.fullmatch(blocker) is None:
                raise PeadMarketAccountingEvidenceError(
                    f"event_results[{index}] has an invalid blocker"
                )
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        upstream = row["upstream_disposition"]
        disposition = row["disposition"]
        if upstream not in {"event_source_reconciled", "excluded"}:
            raise PeadMarketAccountingEvidenceError(
                f"event_results[{index}] has an unsupported upstream disposition"
            )
        lineage = _validate_lineage(
            row["lineage"],
            event_id=event_id,
            event_key=event_key,
            label=f"event_results[{index}].lineage",
        )
        denominator: dict[str, Any] | None = None
        timing: dict[str, Any] | None = None
        if upstream == "excluded":
            if (
                disposition != "upstream_excluded"
                or blockers != ["upstream_not_event_source_reconciled"]
                or lineage is not None
                or row["market_denominator"] is not None
                or row["timing"] is not None
            ):
                raise PeadMarketAccountingEvidenceError(
                    f"event_results[{index}] changes an upstream exclusion"
                )
        else:
            if disposition not in {
                "market_accounting_evidenced",
                "market_accounting_excluded",
            }:
                raise PeadMarketAccountingEvidenceError(
                    f"event_results[{index}] has an unsupported market disposition"
                )
            raw_timing = _exact(row["timing"], _TIMING_FIELDS, f"event_results[{index}].timing")
            known_text, known_at = _utc(
                raw_timing["known_public_by_at_utc"],
                f"event_results[{index}].timing.known_public_by_at_utc",
            )
            activation_day = activation_eastern_date(known_text)
            if raw_timing["activation_eastern_date"] != activation_day.isoformat():
                raise PeadMarketAccountingEvidenceError(
                    f"event_results[{index}] activation Eastern date is not derived"
                )
            capture_text, capture_at = _utc(
                raw_timing["consensus_receipt_captured_at_utc"],
                f"event_results[{index}].timing.consensus_receipt_captured_at_utc",
            )
            prospective = evidence_class == "prospective_signal"
            if raw_timing["prospective_freeze_required"] is not prospective:
                raise PeadMarketAccountingEvidenceError(
                    f"event_results[{index}] prospective freeze requirement differs"
                )
            freeze_passed = raw_timing["prospective_freeze_passed"]
            if freeze_passed not in {None, True, False} or (
                not prospective and freeze_passed is not None
            ):
                raise PeadMarketAccountingEvidenceError(
                    f"event_results[{index}] prospective freeze result differs"
                )
            timing = {
                "known_public_by_at_utc": known_text,
                "activation_eastern_date": activation_day.isoformat(),
                "consensus_receipt_captured_at_utc": capture_text,
                "prospective_freeze_required": prospective,
                "prospective_freeze_passed": freeze_passed,
            }
            if disposition == "market_accounting_evidenced":
                if blockers or lineage is None or row["market_denominator"] is None:
                    raise PeadMarketAccountingEvidenceError(
                        f"event_results[{index}] evidenced disposition is incomplete"
                    )
                denominator = _validate_denominator(
                    row["market_denominator"],
                    lineage=lineage,
                    bindings=bindings,
                    activation_day=activation_day,
                    known_public_by=known_at,
                    label=f"event_results[{index}].market_denominator",
                )
                if prospective:
                    close_at = _utc(denominator["session_close_at_utc"], "session close")[1]
                    if freeze_passed is not True or capture_at > close_at:
                        raise PeadMarketAccountingEvidenceError(
                            f"event_results[{index}] prospective freeze did not pass"
                        )
                elif freeze_passed is not None:
                    raise PeadMarketAccountingEvidenceError(
                        f"event_results[{index}] historical freeze must be null"
                    )
            else:
                if not blockers:
                    raise PeadMarketAccountingEvidenceError(
                        f"event_results[{index}] excluded disposition has no blocker"
                    )
                if row["market_denominator"] is not None:
                    if lineage is None:
                        raise PeadMarketAccountingEvidenceError(
                            f"event_results[{index}] excluded denominator lacks lineage"
                        )
                    denominator = _validate_denominator(
                        row["market_denominator"],
                        lineage=lineage,
                        bindings=bindings,
                        activation_day=activation_day,
                        known_public_by=known_at,
                        label=f"event_results[{index}].market_denominator",
                    )
                    close_at = _utc(denominator["session_close_at_utc"], "session close")[1]
                    if (
                        freeze_passed is not False
                        or blockers != ["prospective_consensus_capture_after_prior_close"]
                        or capture_at <= close_at
                    ):
                        raise PeadMarketAccountingEvidenceError(
                            f"event_results[{index}] retained denominator has invalid timing"
                        )
                elif freeze_passed is False or (
                    "prospective_consensus_capture_after_prior_close" in blockers
                ):
                    raise PeadMarketAccountingEvidenceError(
                        f"event_results[{index}] late-capture exclusion lost its denominator"
                    )
        normalized_rows.append(
            {
                "event_id": event_id,
                "event_key": event_key,
                "upstream_disposition": upstream,
                "disposition": disposition,
                "blockers": blockers,
                "lineage": lineage,
                "market_denominator": denominator,
                "timing": timing,
            }
        )
    if rows != normalized_rows:
        raise PeadMarketAccountingEvidenceError("event_results are not canonical")
    ids = [row["event_id"] for row in normalized_rows]
    if len(ids) != len(set(ids)):
        raise PeadMarketAccountingEvidenceError("event_results must have unique event IDs")
    source_count = sum(
        row["upstream_disposition"] == "event_source_reconciled" for row in normalized_rows
    )
    evidenced = sum(row["disposition"] == "market_accounting_evidenced" for row in normalized_rows)
    excluded = sum(row["disposition"] == "market_accounting_excluded" for row in normalized_rows)
    upstream_excluded = sum(row["disposition"] == "upstream_excluded" for row in normalized_rows)
    expected_counts = {
        "upstream_event_count": len(rows),
        "upstream_excluded_count": upstream_excluded,
        "source_reconciled_event_count": source_count,
        "market_accounting_evidenced_count": evidenced,
        "market_accounting_excluded_count": excluded,
    }
    for field, expected in expected_counts.items():
        if coverage[field] != expected:
            raise PeadMarketAccountingEvidenceError(f"coverage.{field} is not derived")
    raw_blocker_counts = coverage["event_blocker_counts"]
    expected_blocker_counts = {key: blocker_counts[key] for key in sorted(blocker_counts)}
    if raw_blocker_counts != expected_blocker_counts:
        raise PeadMarketAccountingEvidenceError("coverage.event_blocker_counts is not derived")
    if coverage["exhaustive_upstream_accounting"] is not True or (
        coverage["exhaustive_source_reconciled_accounting"] is not True
    ):
        raise PeadMarketAccountingEvidenceError("market accounting is not exhaustive")
    all_evidenced = source_count > 0 and evidenced == source_count
    expected_qualification = {
        "has_market_accounting_evidence": evidenced > 0,
        "all_source_reconciled_events_evidenced": all_evidenced,
        "market_accounting_evidence_allowed": all_evidenced,
        "final_signal_receipt_required": True,
        "research_consumable": False,
        "edge_claim_allowed": False,
        "paper_execution_allowed": False,
        "live_deployment_allowed": False,
    }
    if dict(qualification) != expected_qualification:
        raise PeadMarketAccountingEvidenceError("market qualification is not derived")
    return {"artifact_hash": claimed, "payload": _plain(payload)}


def verify_pead_market_accounting_evidence(
    document: Mapping[str, Any],
    source_reconciliation: Mapping[str, Any],
    sharadar_event_replay: Mapping[str, Any],
    event_universe_index: Mapping[str, Any],
    sharadar_source_snapshot: Mapping[str, Any],
    security_identity_snapshot: Mapping[str, Any],
    sep_semantic_profile: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Authoritatively rebuild the receipt from all original evidence."""
    normalized = validate_pead_market_accounting_evidence_structure(document)
    expected = build_pead_market_accounting_evidence(
        source_reconciliation,
        sharadar_event_replay,
        event_universe_index,
        sharadar_source_snapshot,
        security_identity_snapshot,
        sep_semantic_profile,
        created_at_utc=normalized["payload"]["created_at_utc"],
        **kwargs,
    )
    if normalized != expected:
        raise PeadMarketAccountingEvidenceError(
            "market accounting evidence does not replay from authoritative inputs"
        )
    return expected


def _strict_json_file(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PeadMarketAccountingEvidenceError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    if not raw or len(raw) > max_bytes:
        raise PeadMarketAccountingEvidenceError(f"{label} file size is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadMarketAccountingEvidenceError(f"{label} is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadMarketAccountingEvidenceError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadMarketAccountingEvidenceError(f"{label} contains invalid number {token}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadMarketAccountingEvidenceError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise PeadMarketAccountingEvidenceError(f"{label} root must be an object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadMarketAccountingEvidenceError(
            f"{label} bytes are not canonical JSON plus one newline"
        )
    return value


def load_pead_sep_semantic_profile(path: str | Path) -> dict[str, Any]:
    """Load a duplicate-key-free canonical semantic profile."""
    return validate_pead_sep_semantic_profile(
        _strict_json_file(
            Path(path),
            label="SEP semantic profile",
            max_bytes=MAX_SEP_SEMANTIC_PROFILE_BYTES,
        )
    )


def _publish_canonical_exclusive(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    label: str,
    max_bytes: int,
) -> Path:
    """Durably create canonical JSON once and refuse every existing target."""
    try:
        encoded = (canonical_json(document) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PeadMarketAccountingEvidenceError(f"{label} cannot be encoded canonically") from exc
    if not encoded or len(encoded) > max_bytes:
        raise PeadMarketAccountingEvidenceError(f"{label} exceeds the publication size limit")
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PeadMarketAccountingEvidenceError(
            f"cannot create {label} publication directory: {target.parent}"
        ) from exc
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise PeadMarketAccountingEvidenceError(
            f"{label} publication parent is not a regular directory: {target.parent}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise PeadMarketAccountingEvidenceError(
            f"refusing {label} publication collision: {target}"
        ) from exc
    except OSError as exc:
        raise PeadMarketAccountingEvidenceError(
            f"cannot exclusively create {label} publication: {target}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            written = stream.write(encoded)
            if written != len(encoded):
                raise OSError("short canonical evidence write")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PeadMarketAccountingEvidenceError(
            f"cannot durably write {label} publication: {target}"
        ) from exc
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise PeadMarketAccountingEvidenceError(
            f"cannot durably sync {label} publication directory: {target.parent}"
        ) from exc
    return target


def publish_pead_sep_semantic_profile(
    document: Mapping[str, Any], path: str | Path
) -> tuple[dict[str, Any], Path]:
    """Validate and exclusively publish a profile for external approval."""
    normalized = validate_pead_sep_semantic_profile(document)
    target = _publish_canonical_exclusive(
        path,
        normalized,
        label="SEP semantic profile",
        max_bytes=MAX_SEP_SEMANTIC_PROFILE_BYTES,
    )
    reread = load_pead_sep_semantic_profile(target)
    if reread != normalized:
        raise PeadMarketAccountingEvidenceError(
            "published SEP semantic profile differs from the validated document"
        )
    return normalized, target


def publish_pead_market_accounting_evidence(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    authoritative_verification_kwargs: Mapping[str, Any] | None = None,
    allow_structural_only: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Verify and exclusively publish one canonical market receipt.

    Authoritative replay is the default and requires the original evidence in
    ``authoritative_verification_kwargs``. Structural-only publication exists
    solely for explicitly marked development workflows.
    """
    if authoritative_verification_kwargs is None:
        if allow_structural_only is not True:
            raise PeadMarketAccountingEvidenceError(
                "market publication requires authoritative verification or explicit "
                "allow_structural_only=True"
            )
        normalized = validate_pead_market_accounting_evidence_structure(document)
    else:
        if allow_structural_only is not False:
            raise PeadMarketAccountingEvidenceError(
                "authoritative and structural-only market publication modes are mutually exclusive"
            )
        if not isinstance(authoritative_verification_kwargs, Mapping):
            raise PeadMarketAccountingEvidenceError(
                "authoritative_verification_kwargs must be a mapping"
            )
        if "document" in authoritative_verification_kwargs:
            raise PeadMarketAccountingEvidenceError(
                "authoritative_verification_kwargs may not contain document"
            )
        normalized = verify_pead_market_accounting_evidence(
            document, **dict(authoritative_verification_kwargs)
        )
    target = _publish_canonical_exclusive(
        path,
        normalized,
        label="market accounting evidence",
        max_bytes=MAX_MARKET_ACCOUNTING_EVIDENCE_BYTES,
    )
    reread = validate_pead_market_accounting_evidence_structure(
        _strict_json_file(
            target,
            label="market accounting evidence",
            max_bytes=MAX_MARKET_ACCOUNTING_EVIDENCE_BYTES,
        )
    )
    if reread != normalized:
        raise PeadMarketAccountingEvidenceError(
            "published market accounting evidence differs from the verified document"
        )
    return normalized, target


def load_pead_market_accounting_evidence(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Load canonical bytes and authoritatively replay all original evidence."""
    return verify_pead_market_accounting_evidence(
        _strict_json_file(
            Path(path),
            label="market accounting evidence",
            max_bytes=MAX_MARKET_ACCOUNTING_EVIDENCE_BYTES,
        ),
        **kwargs,
    )


__all__ = [
    "MARKET_ACCOUNTING_EVIDENCE_SCHEMA_VERSION",
    "MARKET_ACCOUNTING_POLICY_SCHEMA_VERSION",
    "MAX_MARKET_ACCOUNTING_EVIDENCE_BYTES",
    "MAX_SEP_SEMANTIC_PROFILE_BYTES",
    "PeadMarketAccountingEvidenceError",
    "SEP_SEMANTIC_PROFILE_SCHEMA_VERSION",
    "build_pead_market_accounting_evidence",
    "build_pead_sep_semantic_profile",
    "load_pead_market_accounting_evidence",
    "load_pead_sep_semantic_profile",
    "publish_pead_market_accounting_evidence",
    "publish_pead_sep_semantic_profile",
    "validate_pead_market_accounting_evidence_structure",
    "validate_pead_sep_semantic_profile",
    "verify_pead_market_accounting_evidence",
]
