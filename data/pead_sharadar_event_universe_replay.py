"""Authoritative Sharadar SF1 replay into yearly PEAD event universes.

The replay reopens candidate-grade SF1/TICKERS acquisitions, revalidates the
dated identity snapshot, hashes every complete SF1 row with the shared typed
row helper, and accounts for every row in an annual census.  It never chooses a
share class: an issuer-period observed through more than one permaticker is an
explicit identity gap.  Multiple SF1 revisions for one unambiguous security are
retained in full; the lexicographically smallest source-row SHA is the
content-deterministic representative and all other revisions remain explicit
``out_of_scope`` census dispositions.

Children use ``pead_event_universe.v2``.  Unlike v1, exhaustive identity gaps
exclude only affected source rows when valid expected events remain.  The v1
builder and validator semantics remain unchanged for frozen earlier packages.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

import duckdb

from data.pead_event_universe import (
    PeadEventUniverseError,
    build_pead_event_census_receipt,
    build_pead_event_universe_v2,
    canonical_event_id,
    canonical_json,
    content_hash,
    validate_pead_event_universe,
)
from data.pead_event_universe_index import (
    MAX_EVENT_UNIVERSE_INDEX_BYTES,
    PeadEventUniverseIndexError,
    build_pead_event_universe_index,
    validate_pead_event_universe_index,
)
from data.sharadar_source_evidence import (
    SHARADAR_RECEIPT_ROOT,
    SharadarSourceEvidenceError,
    load_sharadar_table_acquisition,
    sharadar_source_record_sha256,
    validate_pead_security_identity_snapshot,
    validate_pead_sharadar_source_snapshot,
)


PEAD_SHARADAR_EVENT_UNIVERSE_REPLAY_SCHEMA_VERSION = (
    "pead_sharadar_event_universe_replay.v1"
)
PEAD_SHARADAR_EVENT_CENSUS_SCHEMA_VERSION = "pead_sharadar_event_census.v1"
PEAD_SHARADAR_EVENT_REPLAY_POLICY_SCHEMA_VERSION = (
    "pead_sharadar_event_replay_policy.v1"
)
TARGET_START = "2015-01-01"
TARGET_END = "2024-09-30"
MAX_REPLAY_BYTES = 1024 * 1024 * 1024
EVENT_REPLAY_RECEIPT_ROOT = f"{SHARADAR_RECEIPT_ROOT}/event_replays"
EVENT_UNIVERSE_INDEX_RECEIPT_ROOT = (
    f"{SHARADAR_RECEIPT_ROOT}/event_universe_indexes"
)

REPLAY_POLICY = {
    "schema_version": PEAD_SHARADAR_EVENT_REPLAY_POLICY_SCHEMA_VERSION,
    "source_table": "SHARADAR/SF1",
    "source_dimension": "ARQ",
    "quarterly_scope_rule": "ARQ_dimension_is_the_complete_quarterly_scope",
    "fiscalperiod_rule": (
        "retain_provider_value_as_diagnostic_only_never_use_as_event_scope_gate"
    ),
    "partition_key": "reportperiod_calendar_year",
    "event_key": ["cik", "reportperiod", "fiscal_period_type_Q"],
    "source_row_identity": "shared_complete_typed_sf1_row_sha256",
    "dated_identity_rule": (
        "exact_sf1_ticker_to_one_accepted_tickers_identity_whose_inclusive_"
        "validity_interval_contains_reportperiod"
    ),
    "missing_identity_rule": "identity_gap_exclude_affected_source_row",
    "share_class_rule": (
        "issuer_reportperiod_with_multiple_permatickers_is_identity_gap_no_selection"
    ),
    "revision_rule": (
        "retain_all_source_rows_select_lexicographically_smallest_sha_as_"
        "content_deterministic_representative"
    ),
    "additional_revision_disposition": "out_of_scope_additional_sf1_revision_retained",
    "child_schema": "pead_event_universe.v2",
    "target_window": {"start": TARGET_START, "end": TARGET_END},
}

CANONICAL_QUERY = {
    "schema_version": "pead_sharadar_sf1_event_query.v1",
    "projection": "all_bound_sf1_columns_in_provider_metadata_order",
    "relation": "immutable_content_addressed_sf1_parquet",
    "predicate": (
        "dimension_is_ARQ_by_acquisition_contract_and_cast_reportperiod_as_date_"
        "is_between_partition_start_and_partition_end_inclusive"
    ),
    "parameters": ["partition_start", "partition_end"],
    "order_by": ["cast_reportperiod_as_date", "ticker", "cast_datekey_as_date"],
    "streaming_fetch_rows": 5_000,
}

_HEX = frozenset("0123456789abcdef")
_WRAPPER_FIELDS = {"artifact_hash", "payload"}
_PAYLOAD_FIELDS = {
    "schema_version",
    "candidate_id",
    "created_at_utc",
    "target_window",
    "policy",
    "bindings",
    "years",
    "coverage",
    "blockers",
    "qualification_allowed",
}
_WINDOW_FIELDS = {"start", "end"}
_BINDING_FIELDS = {
    "source_snapshot_sha256",
    "identity_snapshot_sha256",
    "sf1_acquisition_sha256",
    "sf1_parquet_sha256",
    "tickers_acquisition_sha256",
    "tickers_parquet_sha256",
    "candidate_specification_sha256",
    "construction_code_sha256",
    "canonical_query_sha256",
}
_YEAR_FIELDS = {
    "partition_id",
    "event_window",
    "raw_census",
    "event_lineage",
    "event_universe",
}
_RAW_CENSUS_FIELDS = {
    "schema_version",
    "partition_id",
    "event_window",
    "bindings",
    "records",
}
_RAW_CENSUS_BINDING_FIELDS = {
    "source_snapshot_sha256",
    "identity_snapshot_sha256",
    "sf1_acquisition_sha256",
    "sf1_parquet_sha256",
    "canonical_query_sha256",
}
_CENSUS_RECORD_FIELDS = {
    "source_record_sha256",
    "ticker",
    "calendardate",
    "datekey",
    "reportperiod",
    "fiscalperiod",
    "identity_disposition",
    "identity_id",
    "cik",
    "permaticker",
    "identity_reason",
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
_COVERAGE_FIELDS = {
    "partition_count",
    "source_record_count",
    "expected_event_count",
    "identity_gap_count",
    "additional_revision_count",
    "complete",
}
_MACHINE_REASON = re.compile(r"^[a-z][a-z0-9_]*$")
_CIK = re.compile(r"^[0-9]{10}$")


class PeadSharadarEventUniverseReplayError(ValueError):
    """The authoritative SF1 event-universe replay cannot be trusted."""


def _plain(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PeadSharadarEventUniverseReplayError(
            f"{label} fields differ: expected {sorted(fields)}, got {actual}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be nonempty canonical text"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _day(value: Any, label: str) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if not isinstance(value, str):
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be canonical YYYY-MM-DD"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be canonical YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != value:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be canonical YYYY-MM-DD"
        )
    return value


def _utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be canonical UTC with Z"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(
            timezone.utc
        )
    except ValueError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be canonical UTC with Z"
        ) from exc
    canonical = parsed.isoformat(
        timespec="microseconds" if parsed.microsecond else "seconds"
    ).replace("+00:00", "Z")
    if canonical != value:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} must be canonical UTC with Z"
        )
    return value


def _trusted_hashes(values: Collection[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise PeadSharadarEventUniverseReplayError(f"{label} must be a hash collection")
    result = {_sha(value, label) for value in values}
    if not result:
        raise PeadSharadarEventUniverseReplayError(f"{label} must not be empty")
    return result


def _trusted_file(
    path: str | os.PathLike[str], *, trusted_hashes: set[str], label: str
) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise PeadSharadarEventUniverseReplayError(
            f"{label} is not a regular file: {source}"
        )
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if value not in trusted_hashes:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} hash is not externally trusted"
        )
    return value


def _partition_windows(start: str, end: str) -> list[tuple[str, str, str]]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    return [
        (
            str(year),
            max(first, date(year, 1, 1)).isoformat(),
            min(last, date(year, 12, 31)).isoformat(),
        )
        for year in range(first.year, last.year + 1)
    ]


def _table_entry(source_snapshot: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return next(
            row
            for row in source_snapshot["payload"]["tables"]
            if row["logical_name"] == name
        )
    except StopIteration as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"source snapshot is missing {name}"
        ) from exc


def _immutable_parquet(
    root: Path, acquisition: Mapping[str, Any], *, label: str
) -> Path:
    relative = Path(acquisition["payload"]["parquet"]["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PeadSharadarEventUniverseReplayError(f"{label} path escapes warehouse")
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} is missing or escapes warehouse"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise PeadSharadarEventUniverseReplayError(
            f"{label} is not a regular file"
        )
    return resolved


def _authoritative_sources(
    *,
    warehouse_dir: str | os.PathLike[str],
    source_snapshot: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(warehouse_dir).resolve()
    try:
        source = validate_pead_sharadar_source_snapshot(
            source_snapshot, warehouse_dir=root
        )
        identity = validate_pead_security_identity_snapshot(
            identity_snapshot,
            warehouse_dir=root,
            source_snapshot=source,
        )
        sf1_entry = _table_entry(source, "sf1")
        tickers_entry = _table_entry(source, "tickers")
        sf1 = load_sharadar_table_acquisition(
            root / sf1_entry["acquisition_receipt_relative_path"],
            warehouse_dir=root,
        )
        tickers = load_sharadar_table_acquisition(
            root / tickers_entry["acquisition_receipt_relative_path"],
            warehouse_dir=root,
        )
    except SharadarSourceEvidenceError as exc:
        raise PeadSharadarEventUniverseReplayError(
            "authoritative Sharadar source or identity validation failed"
        ) from exc
    if source["payload"]["qualification_allowed"] is not True:
        raise PeadSharadarEventUniverseReplayError(
            "Sharadar source snapshot is not qualifying"
        )
    if identity["payload"]["qualification_allowed"] is not True:
        raise PeadSharadarEventUniverseReplayError(
            "security identity snapshot has no usable identities"
        )
    if source["payload"]["candidate_id"] != identity["payload"]["candidate_id"]:
        raise PeadSharadarEventUniverseReplayError(
            "source and identity candidates differ"
        )
    return root, source, identity, sf1, tickers


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or '"' in value:
        raise PeadSharadarEventUniverseReplayError(
            f"unsafe Sharadar column name {value!r}"
        )
    return f'"{value}"'


def _optional_day(value: Any, label: str) -> str | None:
    return None if value is None else _day(value, label)


def _identity_by_ticker(identity_snapshot: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for raw in identity_snapshot["payload"]["identities"]:
        identity = dict(raw)
        result.setdefault(identity["ticker"], []).append(identity)
    for rows in result.values():
        rows.sort(
            key=lambda row: (
                row["valid_from"],
                row["valid_through"],
                row["permaticker"],
                row["identity_id"],
            )
        )
    return result


def _dated_identity(
    identities: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ticker: str,
    reportperiod: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    matches = [
        row
        for row in identities.get(ticker, ())
        if row["valid_from"] <= reportperiod <= row["valid_through"]
    ]
    if not matches:
        return None, "missing_dated_identity"
    if len(matches) != 1:
        return None, "ambiguous_dated_identity"
    return matches[0], None


def _sf1_records_for_partition(
    *,
    sf1_parquet: Path,
    sf1_schema: Sequence[Mapping[str, Any]],
    identity_snapshot: Mapping[str, Any],
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    columns = [row["name"] for row in sf1_schema]
    required = {
        "ticker",
        "dimension",
        "calendardate",
        "datekey",
        "reportperiod",
        "fiscalperiod",
    }
    if not required <= set(columns):
        raise PeadSharadarEventUniverseReplayError(
            f"SF1 schema omits event columns: {sorted(required - set(columns))}"
        )
    identities = _identity_by_ticker(identity_snapshot)
    path = str(sf1_parquet).replace("'", "''")
    connection = duckdb.connect(database=":memory:")
    records: list[dict[str, Any]] = []
    try:
        cursor = connection.execute(
            f"SELECT {', '.join(_identifier(column) for column in columns)} "
            f"FROM read_parquet('{path}') "
            "WHERE CAST(reportperiod AS DATE) >= CAST(? AS DATE) "
            "AND CAST(reportperiod AS DATE) <= CAST(? AS DATE) "
            "ORDER BY CAST(reportperiod AS DATE), ticker, CAST(datekey AS DATE)",
            [start, end],
        )
        while batch := cursor.fetchmany(5_000):
            for values in batch:
                row = dict(zip(columns, values, strict=True))
                if row.get("dimension") != "ARQ":
                    raise PeadSharadarEventUniverseReplayError(
                        "SF1 replay encountered a non-ARQ row"
                    )
                ticker = _text(row.get("ticker"), "SF1 ticker")
                reportperiod = _day(row.get("reportperiod"), "SF1 reportperiod")
                source_hash = sharadar_source_record_sha256(
                    "sf1", sf1_schema, row
                )
                identity, reason = _dated_identity(
                    identities, ticker=ticker, reportperiod=reportperiod
                )
                raw_fiscalperiod = row.get("fiscalperiod")
                fiscalperiod = (
                    raw_fiscalperiod
                    if isinstance(raw_fiscalperiod, str) and raw_fiscalperiod
                    else None
                )
                records.append(
                    {
                        "source_record_sha256": source_hash,
                        "ticker": ticker,
                        "calendardate": _optional_day(
                            row.get("calendardate"), "SF1 calendardate"
                        ),
                        "datekey": _day(row.get("datekey"), "SF1 datekey"),
                        "reportperiod": reportperiod,
                        "fiscalperiod": fiscalperiod,
                        "identity_disposition": (
                            "matched" if identity is not None else "identity_gap"
                        ),
                        "identity_id": (
                            identity["identity_id"] if identity is not None else None
                        ),
                        "cik": identity["cik"] if identity is not None else None,
                        "permaticker": (
                            identity["permaticker"] if identity is not None else None
                        ),
                        "identity_reason": reason,
                    }
                )
    except (duckdb.Error, SharadarSourceEvidenceError) as exc:
        raise PeadSharadarEventUniverseReplayError(
            "SF1 event-census replay failed"
        ) from exc
    finally:
        connection.close()
    records.sort(key=lambda row: row["source_record_sha256"])
    hashes = [row["source_record_sha256"] for row in records]
    if len(hashes) != len(set(hashes)):
        raise PeadSharadarEventUniverseReplayError(
            "SF1 replay contains duplicate source-row identities"
        )
    return records


def _census_and_lineage(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    dispositions: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        source_hash = record["source_record_sha256"]
        if record["identity_disposition"] != "matched":
            reason = record["identity_reason"]
            if not isinstance(reason, str) or _MACHINE_REASON.fullmatch(reason) is None:
                raise PeadSharadarEventUniverseReplayError(
                    "SF1 identity gap has an invalid reason"
                )
            dispositions[source_hash] = {
                "source_record_id": source_hash,
                "disposition": "identity_gap",
                "event_id": None,
                "event_key": None,
                "reason": reason,
            }
            continue
        groups.setdefault((record["cik"], record["reportperiod"]), []).append(record)

    lineages: list[dict[str, Any]] = []
    additional_revisions = 0
    for (cik, reportperiod), rows in sorted(groups.items()):
        permatickers = {row["permaticker"] for row in rows}
        identity_pairs = {(row["ticker"], row["identity_id"]) for row in rows}
        if len(permatickers) != 1:
            for row in rows:
                dispositions[row["source_record_sha256"]] = {
                    "source_record_id": row["source_record_sha256"],
                    "disposition": "identity_gap",
                    "event_id": None,
                    "event_key": None,
                    "reason": "multiple_permatickers_for_issuer_period",
                }
            continue
        if len(identity_pairs) != 1:
            for row in rows:
                dispositions[row["source_record_sha256"]] = {
                    "source_record_id": row["source_record_sha256"],
                    "disposition": "identity_gap",
                    "event_id": None,
                    "event_key": None,
                    "reason": "multiple_dated_identities_for_issuer_period",
                }
            continue
        source_hashes = sorted(row["source_record_sha256"] for row in rows)
        representative = source_hashes[0]
        event_key = {
            "cik": cik,
            "fiscal_period_end": reportperiod,
            "fiscal_period_type": "Q",
        }
        event_id = canonical_event_id(event_key)
        ticker, identity_id = next(iter(identity_pairs))
        permaticker = next(iter(permatickers))
        for source_hash in source_hashes:
            if source_hash == representative:
                dispositions[source_hash] = {
                    "source_record_id": source_hash,
                    "disposition": "expected_event",
                    "event_id": event_id,
                    "event_key": event_key,
                    "reason": None,
                }
            else:
                dispositions[source_hash] = {
                    "source_record_id": source_hash,
                    "disposition": "out_of_scope",
                    "event_id": None,
                    "event_key": None,
                    "reason": "additional_sf1_revision_retained",
                }
                additional_revisions += 1
        lineages.append(
            {
                "event_id": event_id,
                "event_key": event_key,
                "ticker": ticker,
                "permaticker": permaticker,
                "identity_id": identity_id,
                "representative_sf1_source_record_sha256": representative,
                "sf1_source_record_sha256s": source_hashes,
                "sf1_revision_count": len(source_hashes),
            }
        )
    source_ids = {row["source_record_sha256"] for row in records}
    if set(dispositions) != source_ids:
        raise PeadSharadarEventUniverseReplayError(
            "SF1 census did not disposition every source row exactly once"
        )
    lineages.sort(key=lambda row: row["event_id"])
    return list(dispositions.values()), lineages, additional_revisions


def _build_year(
    *,
    partition_id: str,
    start: str,
    end: str,
    candidate_id: str,
    created_at_utc: str,
    bindings: Mapping[str, str],
    sf1_parquet: Path,
    sf1_schema: Sequence[Mapping[str, Any]],
    identity_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    records = _sf1_records_for_partition(
        sf1_parquet=sf1_parquet,
        sf1_schema=sf1_schema,
        identity_snapshot=identity_snapshot,
        start=start,
        end=end,
    )
    raw_census_payload = {
        "schema_version": PEAD_SHARADAR_EVENT_CENSUS_SCHEMA_VERSION,
        "partition_id": partition_id,
        "event_window": {"start": start, "end": end},
        "bindings": {
            key: bindings[key] for key in sorted(_RAW_CENSUS_BINDING_FIELDS)
        },
        "records": records,
    }
    raw_census = {
        "artifact_hash": content_hash(raw_census_payload),
        "payload": raw_census_payload,
    }
    census_receipt = build_pead_event_census_receipt(
        raw_census_artifact_sha256=raw_census["artifact_hash"],
        canonical_query_sha256=bindings["canonical_query_sha256"],
        source_record_ids=[row["source_record_sha256"] for row in records],
    )
    dispositions, lineages, additional_revisions = _census_and_lineage(records)
    event_universe = build_pead_event_universe_v2(
        candidate_id=candidate_id,
        frozen_at_utc=created_at_utc,
        event_start=start,
        event_end=end,
        bindings={
            "market_snapshot_sha256": bindings["source_snapshot_sha256"],
            "identity_snapshot_sha256": bindings["identity_snapshot_sha256"],
            "candidate_specification_sha256": bindings[
                "candidate_specification_sha256"
            ],
            "construction_code_sha256": bindings["construction_code_sha256"],
            "canonical_query_sha256": bindings["canonical_query_sha256"],
        },
        census_receipt=census_receipt,
        census_dispositions=dispositions,
    )
    expected_ids = event_universe["payload"]["expected_event_ids"]
    if [row["event_id"] for row in lineages] != expected_ids:
        raise PeadSharadarEventUniverseReplayError(
            "event lineage differs from child expected events"
        )
    return {
        "partition_id": partition_id,
        "event_window": {"start": start, "end": end},
        "raw_census": raw_census,
        "event_lineage": lineages,
        "event_universe": event_universe,
    }, additional_revisions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_document(
    *,
    source: Mapping[str, Any],
    identity: Mapping[str, Any],
    sf1: Mapping[str, Any],
    tickers: Mapping[str, Any],
    candidate_specification_sha256: str,
    construction_code_sha256: str,
) -> dict[str, str]:
    return {
        "source_snapshot_sha256": source["artifact_hash"],
        "identity_snapshot_sha256": identity["artifact_hash"],
        "sf1_acquisition_sha256": sf1["artifact_hash"],
        "sf1_parquet_sha256": sf1["payload"]["parquet"]["sha256"],
        "tickers_acquisition_sha256": tickers["artifact_hash"],
        "tickers_parquet_sha256": tickers["payload"]["parquet"]["sha256"],
        "candidate_specification_sha256": candidate_specification_sha256,
        "construction_code_sha256": construction_code_sha256,
        "canonical_query_sha256": content_hash(CANONICAL_QUERY),
    }


def build_pead_sharadar_event_universe_replay(
    *,
    warehouse_dir: str | os.PathLike[str],
    source_snapshot: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    candidate_specification_path: str | os.PathLike[str],
    construction_code_path: str | os.PathLike[str],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
    created_at_utc: str,
) -> dict[str, dict[str, Any]]:
    """Replay the fixed ten-year SF1 target from immutable source evidence.

    The two allowlists are deliberately supplied by the caller.  Hashes copied
    out of either file would prove only self-consistency, not external trust.
    """
    created = _utc(created_at_utc, "created_at_utc")
    candidate_trust = _trusted_hashes(
        trusted_candidate_specification_sha256s,
        "trusted_candidate_specification_sha256s",
    )
    code_trust = _trusted_hashes(
        trusted_construction_code_sha256s,
        "trusted_construction_code_sha256s",
    )
    candidate_hash = _trusted_file(
        candidate_specification_path,
        trusted_hashes=candidate_trust,
        label="candidate specification",
    )
    code_hash = _trusted_file(
        construction_code_path,
        trusted_hashes=code_trust,
        label="construction code",
    )
    root, source, identity, sf1, tickers = _authoritative_sources(
        warehouse_dir=warehouse_dir,
        source_snapshot=source_snapshot,
        identity_snapshot=identity_snapshot,
    )
    created_datetime = datetime.fromisoformat(created[:-1] + "+00:00")
    source_datetime = datetime.fromisoformat(
        source["payload"]["created_at_utc"][:-1] + "+00:00"
    )
    identity_datetime = datetime.fromisoformat(
        identity["payload"]["created_at_utc"][:-1] + "+00:00"
    )
    if created_datetime < source_datetime:
        raise PeadSharadarEventUniverseReplayError(
            "event replay predates its Sharadar source snapshot"
        )
    if created_datetime < identity_datetime:
        raise PeadSharadarEventUniverseReplayError(
            "event replay predates its security identity snapshot"
        )

    source_sf1 = _table_entry(source, "sf1")
    source_tickers = _table_entry(source, "tickers")
    if source_sf1["acquisition_artifact_hash"] != sf1["artifact_hash"]:
        raise PeadSharadarEventUniverseReplayError(
            "SF1 acquisition differs from the source snapshot"
        )
    if source_tickers["acquisition_artifact_hash"] != tickers["artifact_hash"]:
        raise PeadSharadarEventUniverseReplayError(
            "TICKERS acquisition differs from the source snapshot"
        )
    identity_bindings = identity["payload"]["bindings"]
    if identity_bindings["tickers_acquisition_sha256"] != tickers["artifact_hash"]:
        raise PeadSharadarEventUniverseReplayError(
            "identity snapshot differs from the bound TICKERS acquisition"
        )

    sf1_parquet = _immutable_parquet(root, sf1, label="SF1 Parquet")
    tickers_parquet = _immutable_parquet(root, tickers, label="TICKERS Parquet")
    bindings = _binding_document(
        source=source,
        identity=identity,
        sf1=sf1,
        tickers=tickers,
        candidate_specification_sha256=candidate_hash,
        construction_code_sha256=code_hash,
    )
    candidate_id = source["payload"]["candidate_id"]
    sf1_schema = sf1["payload"]["parquet"]["schema"]
    years: list[dict[str, Any]] = []
    additional_revision_count = 0
    try:
        for partition_id, start, end in _partition_windows(TARGET_START, TARGET_END):
            child, child_additional = _build_year(
                partition_id=partition_id,
                start=start,
                end=end,
                candidate_id=candidate_id,
                created_at_utc=created,
                bindings=bindings,
                sf1_parquet=sf1_parquet,
                sf1_schema=sf1_schema,
                identity_snapshot=identity,
            )
            years.append(child)
            additional_revision_count += child_additional
        index = build_pead_event_universe_index(
            partitions=[year["event_universe"] for year in years],
            target_start=TARGET_START,
            target_end=TARGET_END,
            indexed_at_utc=created,
        )
    except (PeadEventUniverseError, PeadEventUniverseIndexError) as exc:
        raise PeadSharadarEventUniverseReplayError(
            "yearly event-universe construction failed"
        ) from exc

    # Detect mutable-file replacement during the replay rather than accepting
    # a digest that was valid only before or only after the query ran.
    if _file_sha256(sf1_parquet) != bindings["sf1_parquet_sha256"]:
        raise PeadSharadarEventUniverseReplayError("SF1 Parquet changed during replay")
    if _file_sha256(tickers_parquet) != bindings["tickers_parquet_sha256"]:
        raise PeadSharadarEventUniverseReplayError(
            "TICKERS Parquet changed during replay"
        )
    if (
        _trusted_file(
            candidate_specification_path,
            trusted_hashes=candidate_trust,
            label="candidate specification",
        )
        != candidate_hash
    ):
        raise PeadSharadarEventUniverseReplayError(
            "candidate specification changed during replay"
        )
    if (
        _trusted_file(
            construction_code_path,
            trusted_hashes=code_trust,
            label="construction code",
        )
        != code_hash
    ):
        raise PeadSharadarEventUniverseReplayError(
            "construction code changed during replay"
        )

    source_count = sum(
        year["event_universe"]["payload"]["census_counts"]["source_record_count"]
        for year in years
    )
    expected_count = sum(
        year["event_universe"]["payload"]["census_counts"]["expected_event_count"]
        for year in years
    )
    identity_gap_count = sum(
        year["event_universe"]["payload"]["census_counts"]["identity_gap_count"]
        for year in years
    )
    complete = (
        len(years) == len(_partition_windows(TARGET_START, TARGET_END))
        and index["payload"]["qualification"]["qualification_allowed"] is True
    )
    blockers: list[str] = []
    if not complete:
        blockers.append("yearly_event_universe_index_not_qualified")
    if expected_count == 0:
        blockers.append("expected_event_manifest_empty")
    payload = {
        "schema_version": PEAD_SHARADAR_EVENT_UNIVERSE_REPLAY_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "created_at_utc": created,
        "target_window": {"start": TARGET_START, "end": TARGET_END},
        "policy": REPLAY_POLICY,
        "bindings": bindings,
        "years": years,
        "coverage": {
            "partition_count": len(years),
            "source_record_count": source_count,
            "expected_event_count": expected_count,
            "identity_gap_count": identity_gap_count,
            "additional_revision_count": additional_revision_count,
            "complete": complete,
        },
        "blockers": blockers,
        "qualification_allowed": not blockers,
    }
    replay = {"artifact_hash": content_hash(payload), "payload": _plain(payload)}
    return {"replay": replay, "index": index}


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PeadSharadarEventUniverseReplayError(f"{label} must be a positive integer")
    return value


def _raw_census_records(
    raw_census: Mapping[str, Any],
    *,
    partition_id: str,
    start: str,
    end: str,
    bindings: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    wrapper = _exact(raw_census, _WRAPPER_FIELDS, "raw_census")
    payload = _exact(wrapper["payload"], _RAW_CENSUS_FIELDS, "raw_census.payload")
    claimed = _sha(wrapper["artifact_hash"], "raw_census.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadSharadarEventUniverseReplayError("raw census artifact hash mismatch")
    if payload["schema_version"] != PEAD_SHARADAR_EVENT_CENSUS_SCHEMA_VERSION:
        raise PeadSharadarEventUniverseReplayError("unsupported raw census schema")
    if payload["partition_id"] != partition_id:
        raise PeadSharadarEventUniverseReplayError("raw census partition differs")
    window = _exact(payload["event_window"], _WINDOW_FIELDS, "raw_census.event_window")
    if window != {"start": start, "end": end}:
        raise PeadSharadarEventUniverseReplayError("raw census window differs")
    raw_bindings = _exact(
        payload["bindings"], _RAW_CENSUS_BINDING_FIELDS, "raw_census.bindings"
    )
    for name in _RAW_CENSUS_BINDING_FIELDS:
        if _sha(raw_bindings[name], f"raw_census.bindings.{name}") != bindings[name]:
            raise PeadSharadarEventUniverseReplayError(
                f"raw census binding differs: {name}"
            )
    records = payload["records"]
    if not isinstance(records, list):
        raise PeadSharadarEventUniverseReplayError("raw census records must be an array")
    source_hashes: list[str] = []
    for index, raw in enumerate(records):
        row = _exact(raw, _CENSUS_RECORD_FIELDS, f"raw_census.records[{index}]")
        source_hash = _sha(
            row["source_record_sha256"],
            f"raw_census.records[{index}].source_record_sha256",
        )
        source_hashes.append(source_hash)
        _text(row["ticker"], f"raw_census.records[{index}].ticker")
        if row["calendardate"] is not None:
            _day(row["calendardate"], f"raw_census.records[{index}].calendardate")
        _day(row["datekey"], f"raw_census.records[{index}].datekey")
        reportperiod = _day(
            row["reportperiod"], f"raw_census.records[{index}].reportperiod"
        )
        if not start <= reportperiod <= end:
            raise PeadSharadarEventUniverseReplayError(
                "raw census reportperiod falls outside its partition"
            )
        if row["fiscalperiod"] is not None:
            _text(row["fiscalperiod"], f"raw_census.records[{index}].fiscalperiod")
        disposition = row["identity_disposition"]
        if disposition == "matched":
            _sha(row["identity_id"], f"raw_census.records[{index}].identity_id")
            cik = row["cik"]
            if not isinstance(cik, str) or _CIK.fullmatch(cik) is None or cik == "0000000000":
                raise PeadSharadarEventUniverseReplayError(
                    f"raw_census.records[{index}].cik must be a positive 10-digit CIK"
                )
            _positive_int(
                row["permaticker"], f"raw_census.records[{index}].permaticker"
            )
            if row["identity_reason"] is not None:
                raise PeadSharadarEventUniverseReplayError(
                    "matched raw census identity cannot have a gap reason"
                )
        elif disposition == "identity_gap":
            if any(row[name] is not None for name in ("identity_id", "cik", "permaticker")):
                raise PeadSharadarEventUniverseReplayError(
                    "identity-gap raw census row cannot carry accepted identity fields"
                )
            reason = row["identity_reason"]
            if not isinstance(reason, str) or _MACHINE_REASON.fullmatch(reason) is None:
                raise PeadSharadarEventUniverseReplayError(
                    "identity-gap raw census reason is invalid"
                )
        else:
            raise PeadSharadarEventUniverseReplayError(
                "raw census identity disposition is unsupported"
            )
    if source_hashes != sorted(set(source_hashes)):
        raise PeadSharadarEventUniverseReplayError(
            "raw census source identities must be sorted and unique"
        )
    return records


def _lineage_rows(
    raw_rows: Any,
    *,
    records: Sequence[Mapping[str, Any]],
    event_universe: Mapping[str, Any],
    start: str,
    end: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(raw_rows, list):
        raise PeadSharadarEventUniverseReplayError("event_lineage must be an array")
    by_source = {row["source_record_sha256"]: row for row in records}
    dispositions = {
        row["source_record_id"]: row
        for row in event_universe["payload"]["census_dispositions"]
    }
    rows: list[Mapping[str, Any]] = []
    used_sources: set[str] = set()
    for index, raw in enumerate(raw_rows):
        row = _exact(raw, _LINEAGE_FIELDS, f"event_lineage[{index}]")
        event_id = _sha(row["event_id"], f"event_lineage[{index}].event_id")
        try:
            derived_id = canonical_event_id(row["event_key"])
        except PeadEventUniverseError as exc:
            raise PeadSharadarEventUniverseReplayError(
                f"event_lineage[{index}].event_key is invalid"
            ) from exc
        if event_id != derived_id:
            raise PeadSharadarEventUniverseReplayError("lineage event identity differs")
        period = row["event_key"]["fiscal_period_end"]
        if not start <= period <= end:
            raise PeadSharadarEventUniverseReplayError(
                "lineage event falls outside its partition"
            )
        ticker = _text(row["ticker"], f"event_lineage[{index}].ticker")
        permaticker = _positive_int(
            row["permaticker"], f"event_lineage[{index}].permaticker"
        )
        identity_id = _sha(
            row["identity_id"], f"event_lineage[{index}].identity_id"
        )
        sources = row["sf1_source_record_sha256s"]
        if not isinstance(sources, list) or not sources:
            raise PeadSharadarEventUniverseReplayError(
                "lineage source-row identities must be a nonempty array"
            )
        source_hashes = [
            _sha(item, f"event_lineage[{index}].sf1_source_record_sha256s")
            for item in sources
        ]
        if source_hashes != sorted(set(source_hashes)):
            raise PeadSharadarEventUniverseReplayError(
                "lineage source-row identities must be sorted and unique"
            )
        representative = _sha(
            row["representative_sf1_source_record_sha256"],
            f"event_lineage[{index}].representative_sf1_source_record_sha256",
        )
        if representative != source_hashes[0]:
            raise PeadSharadarEventUniverseReplayError(
                "lineage representative is not the smallest complete-row hash"
            )
        if _positive_int(
            row["sf1_revision_count"], f"event_lineage[{index}].sf1_revision_count"
        ) != len(source_hashes):
            raise PeadSharadarEventUniverseReplayError(
                "lineage revision count differs from retained rows"
            )
        if used_sources.intersection(source_hashes):
            raise PeadSharadarEventUniverseReplayError(
                "an SF1 source row appears in more than one event lineage"
            )
        used_sources.update(source_hashes)
        for source_hash in source_hashes:
            source = by_source.get(source_hash)
            if source is None:
                raise PeadSharadarEventUniverseReplayError(
                    "lineage refers to an SF1 row outside the bound census"
                )
            if (
                source["identity_disposition"] != "matched"
                or source["ticker"] != ticker
                or source["permaticker"] != permaticker
                or source["identity_id"] != identity_id
                or source["cik"] != row["event_key"]["cik"]
                or source["reportperiod"] != period
            ):
                raise PeadSharadarEventUniverseReplayError(
                    "lineage fields differ from the dated SF1 identity census"
                )
            disposition = dispositions[source_hash]
            if source_hash == representative:
                if (
                    disposition["disposition"] != "expected_event"
                    or disposition["event_id"] != event_id
                    or disposition["event_key"] != row["event_key"]
                ):
                    raise PeadSharadarEventUniverseReplayError(
                        "representative SF1 disposition differs from lineage"
                    )
            elif (
                disposition["disposition"] != "out_of_scope"
                or disposition["reason"] != "additional_sf1_revision_retained"
            ):
                raise PeadSharadarEventUniverseReplayError(
                    "additional SF1 revision is not explicitly retained"
                )
        rows.append(row)
    event_ids = [row["event_id"] for row in rows]
    if event_ids != sorted(set(event_ids)):
        raise PeadSharadarEventUniverseReplayError(
            "event lineage identities must be sorted and unique"
        )
    if event_ids != event_universe["payload"]["expected_event_ids"]:
        raise PeadSharadarEventUniverseReplayError(
            "event lineage differs from child expected events"
        )
    return rows


def validate_pead_sharadar_event_universe_replay_structure(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate internal structure only; this is not an authority boundary."""
    wrapper = _exact(document, _WRAPPER_FIELDS, "event replay")
    payload = _exact(wrapper["payload"], _PAYLOAD_FIELDS, "event replay.payload")
    claimed = _sha(wrapper["artifact_hash"], "event replay.artifact_hash")
    if content_hash(payload) != claimed:
        raise PeadSharadarEventUniverseReplayError("event replay artifact hash mismatch")
    if payload["schema_version"] != PEAD_SHARADAR_EVENT_UNIVERSE_REPLAY_SCHEMA_VERSION:
        raise PeadSharadarEventUniverseReplayError("unsupported event replay schema")
    candidate_id = _text(payload["candidate_id"], "candidate_id")
    created = _utc(payload["created_at_utc"], "created_at_utc")
    target = _exact(payload["target_window"], _WINDOW_FIELDS, "target_window")
    if target != {"start": TARGET_START, "end": TARGET_END}:
        raise PeadSharadarEventUniverseReplayError(
            "event replay target window is not the fixed research target"
        )
    if payload["policy"] != REPLAY_POLICY:
        raise PeadSharadarEventUniverseReplayError("event replay policy changed")
    raw_bindings = _exact(payload["bindings"], _BINDING_FIELDS, "bindings")
    bindings = {
        name: _sha(raw_bindings[name], f"bindings.{name}")
        for name in sorted(_BINDING_FIELDS)
    }
    if bindings["canonical_query_sha256"] != content_hash(CANONICAL_QUERY):
        raise PeadSharadarEventUniverseReplayError("canonical query binding changed")
    raw_years = payload["years"]
    expected_windows = _partition_windows(TARGET_START, TARGET_END)
    if not isinstance(raw_years, list) or len(raw_years) != len(expected_windows):
        raise PeadSharadarEventUniverseReplayError(
            "event replay must contain every target calendar year"
        )
    years: list[Mapping[str, Any]] = []
    source_count = 0
    expected_count = 0
    gap_count = 0
    additional_count = 0
    for index, (raw, (partition_id, start, end)) in enumerate(
        zip(raw_years, expected_windows, strict=True)
    ):
        year = _exact(raw, _YEAR_FIELDS, f"years[{index}]")
        if year["partition_id"] != partition_id:
            raise PeadSharadarEventUniverseReplayError("year partition ID differs")
        window = _exact(year["event_window"], _WINDOW_FIELDS, f"years[{index}].event_window")
        if window != {"start": start, "end": end}:
            raise PeadSharadarEventUniverseReplayError("year partition window differs")
        records = _raw_census_records(
            year["raw_census"],
            partition_id=partition_id,
            start=start,
            end=end,
            bindings=bindings,
        )
        try:
            universe = validate_pead_event_universe(year["event_universe"])
        except PeadEventUniverseError as exc:
            raise PeadSharadarEventUniverseReplayError(
                f"years[{index}] event universe is invalid"
            ) from exc
        universe_payload = universe["payload"]
        if universe_payload["schema_version"] != "pead_event_universe.v2":
            raise PeadSharadarEventUniverseReplayError(
                "Sharadar replay children must use event-universe v2"
            )
        if (
            universe_payload["candidate_id"] != candidate_id
            or universe_payload["frozen_at_utc"] != created
            or universe_payload["event_window"] != window
        ):
            raise PeadSharadarEventUniverseReplayError(
                "year event-universe identity differs from the replay"
            )
        expected_universe_bindings = {
            "market_snapshot_sha256": bindings["source_snapshot_sha256"],
            "identity_snapshot_sha256": bindings["identity_snapshot_sha256"],
            "candidate_specification_sha256": bindings[
                "candidate_specification_sha256"
            ],
            "construction_code_sha256": bindings["construction_code_sha256"],
            "canonical_query_sha256": bindings["canonical_query_sha256"],
        }
        if universe_payload["bindings"] != expected_universe_bindings:
            raise PeadSharadarEventUniverseReplayError(
                "year event-universe bindings differ from the replay"
            )
        receipt = universe_payload["census_receipt"]["payload"]
        if (
            receipt["raw_census_artifact_sha256"]
            != year["raw_census"]["artifact_hash"]
            or receipt["source_record_ids"]
            != [row["source_record_sha256"] for row in records]
        ):
            raise PeadSharadarEventUniverseReplayError(
                "year census receipt differs from its raw census"
            )
        lineages = _lineage_rows(
            year["event_lineage"],
            records=records,
            event_universe=universe,
            start=start,
            end=end,
        )
        counts = universe_payload["census_counts"]
        if counts["source_record_count"] != len(records):
            raise PeadSharadarEventUniverseReplayError(
                "year source count differs from raw census"
            )
        source_count += len(records)
        expected_count += len(lineages)
        gap_count += counts["identity_gap_count"]
        additional_count += sum(row["sf1_revision_count"] - 1 for row in lineages)
        years.append(year)
    coverage = _exact(payload["coverage"], _COVERAGE_FIELDS, "coverage")
    expected_coverage = {
        "partition_count": len(years),
        "source_record_count": source_count,
        "expected_event_count": expected_count,
        "identity_gap_count": gap_count,
        "additional_revision_count": additional_count,
        "complete": all(
            year["event_universe"]["payload"]["qualification_allowed"]
            for year in years
        ),
    }
    if coverage != expected_coverage:
        raise PeadSharadarEventUniverseReplayError("event replay coverage is not derived")
    expected_blockers: list[str] = []
    if not expected_coverage["complete"]:
        expected_blockers.append("yearly_event_universe_index_not_qualified")
    if expected_count == 0:
        expected_blockers.append("expected_event_manifest_empty")
    if payload["blockers"] != expected_blockers:
        raise PeadSharadarEventUniverseReplayError("event replay blockers are not derived")
    if payload["qualification_allowed"] is not (not expected_blockers):
        raise PeadSharadarEventUniverseReplayError(
            "event replay qualification is not derived"
        )
    return {"artifact_hash": claimed, "payload": _plain(payload)}


def verify_pead_sharadar_event_universe_replay(
    replay: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    warehouse_dir: str | os.PathLike[str],
    source_snapshot: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    candidate_specification_path: str | os.PathLike[str],
    construction_code_path: str | os.PathLike[str],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
) -> dict[str, dict[str, Any]]:
    """Requery immutable Sharadar evidence and require an exact two-artifact replay."""
    normalized_replay = validate_pead_sharadar_event_universe_replay_structure(replay)
    partitions = [year["event_universe"] for year in normalized_replay["payload"]["years"]]
    try:
        normalized_index = validate_pead_event_universe_index(
            index, partitions=partitions
        )
    except PeadEventUniverseIndexError as exc:
        raise PeadSharadarEventUniverseReplayError(
            "event-universe index does not replay from yearly children"
        ) from exc
    if (
        normalized_index["payload"]["indexed_at_utc"]
        != normalized_replay["payload"]["created_at_utc"]
    ):
        raise PeadSharadarEventUniverseReplayError(
            "event-universe index timestamp differs from replay"
        )
    expected = build_pead_sharadar_event_universe_replay(
        warehouse_dir=warehouse_dir,
        source_snapshot=source_snapshot,
        identity_snapshot=identity_snapshot,
        candidate_specification_path=candidate_specification_path,
        construction_code_path=construction_code_path,
        trusted_candidate_specification_sha256s=(
            trusted_candidate_specification_sha256s
        ),
        trusted_construction_code_sha256s=trusted_construction_code_sha256s,
        created_at_utc=normalized_replay["payload"]["created_at_utc"],
    )
    if normalized_replay != expected["replay"]:
        raise PeadSharadarEventUniverseReplayError(
            "event replay does not rederive from immutable Sharadar evidence"
        )
    if normalized_index != expected["index"]:
        raise PeadSharadarEventUniverseReplayError(
            "event-universe index does not rederive from immutable Sharadar evidence"
        )
    return expected


def _read_regular_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    if max_bytes <= 0:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} size limit must be positive"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} is missing, unreadable, or not a regular file: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PeadSharadarEventUniverseReplayError(
                f"{label} is not a regular file: {path}"
            )
        if metadata.st_size > max_bytes:
            raise PeadSharadarEventUniverseReplayError(
                f"{label} exceeds its size limit: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise PeadSharadarEventUniverseReplayError(
                f"{label} exceeds its size limit: {path}"
            )
        return raw
    finally:
        os.close(descriptor)


def _strict_json_file(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, max_bytes=max_bytes, label=label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} is not UTF-8: {path}"
        ) from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadSharadarEventUniverseReplayError(
                    f"{label} contains duplicate key {key!r}: {path}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} contains invalid number {token}: {path}"
        )

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"invalid {label} JSON at line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadSharadarEventUniverseReplayError(
            f"{label} root must be an object: {path}"
        )
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise PeadSharadarEventUniverseReplayError(
            f"{label} bytes are not canonical JSON plus one newline: {path}"
        )
    return value


def _canonical_document_bytes(
    document: Mapping[str, Any], *, max_bytes: int, label: str
) -> bytes:
    try:
        encoded = (canonical_json(document) + "\n").encode("utf-8")
    except PeadEventUniverseError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} cannot be encoded canonically"
        ) from exc
    if len(encoded) > max_bytes:
        raise PeadSharadarEventUniverseReplayError(
            f"{label} exceeds its size limit"
        )
    return encoded


def _check_create_only_destination(
    path: Path, *, encoded: bytes, max_bytes: int, label: str
) -> None:
    if not os.path.lexists(path):
        return
    try:
        existing = _read_regular_bytes(path, max_bytes=max_bytes, label=label)
    except PeadSharadarEventUniverseReplayError as exc:
        raise PeadSharadarEventUniverseReplayError(
            f"content-addressed {label} collision: {path}"
        ) from exc
    if existing != encoded:
        raise PeadSharadarEventUniverseReplayError(
            f"content-addressed {label} collision: {path}"
        )


def _publish_json_create_only(
    path: Path, *, document: Mapping[str, Any], max_bytes: int, label: str
) -> Path:
    encoded = _canonical_document_bytes(
        document, max_bytes=max_bytes, label=label
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_create_only_destination(
        path, encoded=encoded, max_bytes=max_bytes, label=label
    )
    if os.path.lexists(path):
        return path
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _check_create_only_destination(
                path, encoded=encoded, max_bytes=max_bytes, label=label
            )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


def _event_receipt_paths(
    root: Path, *, replay_hash: str, index_hash: str
) -> dict[str, Path]:
    return {
        "replay": root / EVENT_REPLAY_RECEIPT_ROOT / f"{replay_hash}.json",
        "index": root / EVENT_UNIVERSE_INDEX_RECEIPT_ROOT / f"{index_hash}.json",
    }


def publish_pead_sharadar_event_universe_replay(
    warehouse_dir: str | os.PathLike[str],
    replay: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    candidate_specification_path: str | os.PathLike[str],
    construction_code_path: str | os.PathLike[str],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Authoritatively verify, then create-only publish replay and index receipts."""
    root = Path(warehouse_dir).resolve()
    verified = verify_pead_sharadar_event_universe_replay(
        replay,
        index,
        warehouse_dir=root,
        source_snapshot=source_snapshot,
        identity_snapshot=identity_snapshot,
        candidate_specification_path=candidate_specification_path,
        construction_code_path=construction_code_path,
        trusted_candidate_specification_sha256s=(
            trusted_candidate_specification_sha256s
        ),
        trusted_construction_code_sha256s=trusted_construction_code_sha256s,
    )
    paths = _event_receipt_paths(
        root,
        replay_hash=verified["replay"]["artifact_hash"],
        index_hash=verified["index"]["artifact_hash"],
    )
    replay_bytes = _canonical_document_bytes(
        verified["replay"], max_bytes=MAX_REPLAY_BYTES, label="event replay"
    )
    index_bytes = _canonical_document_bytes(
        verified["index"],
        max_bytes=MAX_EVENT_UNIVERSE_INDEX_BYTES,
        label="event-universe index",
    )
    # Preflight both destinations before either link is created.  A concurrent
    # exact publication remains idempotent; different bytes at either digest
    # path fail as a collision.
    _check_create_only_destination(
        paths["replay"],
        encoded=replay_bytes,
        max_bytes=MAX_REPLAY_BYTES,
        label="event replay",
    )
    _check_create_only_destination(
        paths["index"],
        encoded=index_bytes,
        max_bytes=MAX_EVENT_UNIVERSE_INDEX_BYTES,
        label="event-universe index",
    )
    _publish_json_create_only(
        paths["replay"],
        document=verified["replay"],
        max_bytes=MAX_REPLAY_BYTES,
        label="event replay",
    )
    _publish_json_create_only(
        paths["index"],
        document=verified["index"],
        max_bytes=MAX_EVENT_UNIVERSE_INDEX_BYTES,
        label="event-universe index",
    )
    published_replay = _strict_json_file(
        paths["replay"], max_bytes=MAX_REPLAY_BYTES, label="event replay"
    )
    published_index = _strict_json_file(
        paths["index"],
        max_bytes=MAX_EVENT_UNIVERSE_INDEX_BYTES,
        label="event-universe index",
    )
    if published_replay != verified["replay"]:
        raise PeadSharadarEventUniverseReplayError(
            "published event replay changed after create-only publication"
        )
    if published_index != verified["index"]:
        raise PeadSharadarEventUniverseReplayError(
            "published event-universe index changed after create-only publication"
        )
    return verified, paths


def load_pead_sharadar_event_universe_replay(
    replay_path: str | os.PathLike[str],
    index_path: str | os.PathLike[str],
    *,
    warehouse_dir: str | os.PathLike[str],
    source_snapshot: Mapping[str, Any],
    identity_snapshot: Mapping[str, Any],
    candidate_specification_path: str | os.PathLike[str],
    construction_code_path: str | os.PathLike[str],
    trusted_candidate_specification_sha256s: Collection[str],
    trusted_construction_code_sha256s: Collection[str],
) -> dict[str, dict[str, Any]]:
    """Strictly load content-addressed receipts and authoritatively replay them."""
    replay_file = Path(replay_path)
    index_file = Path(index_path)
    replay = _strict_json_file(
        replay_file, max_bytes=MAX_REPLAY_BYTES, label="event replay"
    )
    index = _strict_json_file(
        index_file,
        max_bytes=MAX_EVENT_UNIVERSE_INDEX_BYTES,
        label="event-universe index",
    )
    root = Path(warehouse_dir).resolve()
    expected_paths = _event_receipt_paths(
        root,
        replay_hash=_sha(replay.get("artifact_hash"), "event replay artifact_hash"),
        index_hash=_sha(index.get("artifact_hash"), "event-universe index artifact_hash"),
    )
    try:
        actual_replay_path = replay_file.resolve(strict=True)
        actual_index_path = index_file.resolve(strict=True)
    except OSError as exc:
        raise PeadSharadarEventUniverseReplayError(
            "event replay receipt path cannot be resolved"
        ) from exc
    if actual_replay_path != expected_paths["replay"]:
        raise PeadSharadarEventUniverseReplayError(
            "event replay is not at its immutable content-addressed warehouse path"
        )
    if actual_index_path != expected_paths["index"]:
        raise PeadSharadarEventUniverseReplayError(
            "event-universe index is not at its immutable content-addressed warehouse path"
        )
    return verify_pead_sharadar_event_universe_replay(
        replay,
        index,
        warehouse_dir=root,
        source_snapshot=source_snapshot,
        identity_snapshot=identity_snapshot,
        candidate_specification_path=candidate_specification_path,
        construction_code_path=construction_code_path,
        trusted_candidate_specification_sha256s=(
            trusted_candidate_specification_sha256s
        ),
        trusted_construction_code_sha256s=trusted_construction_code_sha256s,
    )


__all__ = [
    "CANONICAL_QUERY",
    "EVENT_REPLAY_RECEIPT_ROOT",
    "EVENT_UNIVERSE_INDEX_RECEIPT_ROOT",
    "MAX_REPLAY_BYTES",
    "PEAD_SHARADAR_EVENT_CENSUS_SCHEMA_VERSION",
    "PEAD_SHARADAR_EVENT_REPLAY_POLICY_SCHEMA_VERSION",
    "PEAD_SHARADAR_EVENT_UNIVERSE_REPLAY_SCHEMA_VERSION",
    "PeadSharadarEventUniverseReplayError",
    "REPLAY_POLICY",
    "TARGET_END",
    "TARGET_START",
    "build_pead_sharadar_event_universe_replay",
    "load_pead_sharadar_event_universe_replay",
    "publish_pead_sharadar_event_universe_replay",
    "validate_pead_sharadar_event_universe_replay_structure",
    "verify_pead_sharadar_event_universe_replay",
]
