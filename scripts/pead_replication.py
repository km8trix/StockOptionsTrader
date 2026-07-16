#!/usr/bin/env python
"""Run the locked PEAD replication from an immutable Zacks snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pead_replication import (  # noqa: E402
    CANDIDATE_ID,
    PeadReplicationError,
    build_replication_report,
    build_research_manifest_binding,
    canonical_json,
    validate_snapshot_document,
)
from analysis.independent_replication import ReplicationIntegrityError  # noqa: E402
from analysis.pead_daily_acceptance import (  # noqa: E402
    replay_and_validate_pead_daily_reconciliation,
)
from data.earnings_announcements import (  # noqa: E402
    EarningsAnnouncementSnapshot,
    SnapshotIntegrityError,
)
from data.pead_economic_evidence import (  # noqa: E402
    PeadEconomicEvidenceError,
    load_cash_distribution_semantics,
    load_terminal_settlement_ledger,
)
from data.pit_warehouse import PitWarehouse  # noqa: E402


MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_DAILY_ARTIFACT_BYTES = 64 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_PACKAGE = REPOSITORY_ROOT / "research" / "pead_vq_locked_replication_v1"
DEFAULT_DAILY_SOURCE_REPORT = RESEARCH_PACKAGE / "development_sample_report_v6.json"
DEFAULT_DAILY_MODELED_LEDGER = RESEARCH_PACKAGE / "modeled_execution_ledger_v1.json"
DEFAULT_DAILY_INDEPENDENT_REFERENCE = (
    RESEARCH_PACKAGE / "independent_reference_comparison_v5.json"
)
DEFAULT_DAILY_INPUT_SNAPSHOT = RESEARCH_PACKAGE / "daily_input_snapshot_v1.json"
DEFAULT_DAILY_PROTOCOL = RESEARCH_PACKAGE / "daily_money_path_protocol_v2.json"
DEFAULT_PRIMARY_DAILY_LEDGER = RESEARCH_PACKAGE / "primary_daily_ledger_v2.json"
DEFAULT_INDEPENDENT_DAILY_LEDGER = (
    RESEARCH_PACKAGE / "independent_daily_ledger_v2.json"
)
FROZEN_START = "2015-01-01"
FROZEN_END = "2024-09-30"
FROZEN_HORIZONS = [21, 63]
FROZEN_COST_BPS = 30.0
FROZEN_FRESH_DAYS = 63
FROZEN_QUANTILE = 0.2
FROZEN_WINSOR = 0.01
FROZEN_CONSENSUS_ABS_TOLERANCE = 0.01


class CliUsageError(ValueError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def _strict_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PeadReplicationError(f"snapshot is not a regular file: {path}")
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise PeadReplicationError("snapshot exceeds the 512 MiB limit")
    raw = path.read_text(encoding="utf-8")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadReplicationError(f"snapshot contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise PeadReplicationError(f"snapshot contains invalid number {token}")

    try:
        value = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    except json.JSONDecodeError as exc:
        raise PeadReplicationError(
            f"invalid snapshot JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadReplicationError("snapshot root must be an object")
    try:
        # Reconstruct the acquisition artifact before analysis so raw provider
        # page hashes, cursor receipts, exact rows, and the content address are
        # all checked again at the research boundary.
        verified = EarningsAnnouncementSnapshot.from_json(raw)
    except SnapshotIntegrityError as exc:
        raise PeadReplicationError(
            "snapshot failed acquisition-integrity verification"
        ) from exc
    if verified.artifact_hash != value.get("artifact_hash"):
        raise PeadReplicationError("snapshot verifier returned a different identity")
    return value


def _strict_manifest_object(
    path: Path,
    label: str,
    *,
    maximum_bytes: int = MAX_MANIFEST_BYTES,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise PeadReplicationError(f"{label} is not a regular file: {path}")
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > maximum_bytes:
        raise PeadReplicationError(
            f"{label} exceeds the {maximum_bytes} byte limit"
        )
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadReplicationError(f"{label} is not UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadReplicationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise PeadReplicationError(f"{label} contains invalid number {token}")

    try:
        value = json.loads(
            raw, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    except json.JSONDecodeError as exc:
        raise PeadReplicationError(
            f"invalid {label} JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise PeadReplicationError(f"{label} root must be an object")
    return value, hashlib.sha256(raw_bytes).hexdigest()


def _load_research_manifest_binding(
    document: Mapping[str, Any], *, candidate_path: Path, source_path: Path
) -> dict[str, Any]:
    candidate, candidate_hash = _strict_manifest_object(
        candidate_path, "candidate specification"
    )
    source, source_hash = _strict_manifest_object(source_path, "source manifest")
    if candidate.get("schema_version") != "candidate_specification.v1":
        raise PeadReplicationError("unsupported candidate specification schema")
    if candidate.get("candidate_id") != CANDIDATE_ID:
        raise PeadReplicationError("candidate specification belongs to another target")
    source_binding = candidate.get("independent_source_binding")
    if not isinstance(source_binding, Mapping):
        raise PeadReplicationError("candidate specification omits source binding")
    if source_binding.get("source_manifest_file") != source_path.name:
        raise PeadReplicationError(
            "candidate specification names a different source manifest"
        )
    if source.get("schema_version") != "pead_source_manifest.v1":
        raise PeadReplicationError("unsupported source manifest schema")
    if source.get("candidate_id") != CANDIDATE_ID:
        raise PeadReplicationError("source manifest belongs to another candidate")
    source_identity = source.get("source")
    if not isinstance(source_identity, Mapping) or source_identity.get("source_id") != (
        "nasdaq-data-link-zacks"
    ):
        raise PeadReplicationError("source manifest belongs to another source")

    manifest_tables = source.get("tables")
    required_tables = source_binding.get("required_tables")
    expected_tables = {
        "ZACKS/ES", "ZACKS/SS", "ZACKS/EEH",
        "ZACKS/SEH", "ZACKS/MT", "ZACKS/EA",
    }
    if not isinstance(manifest_tables, Mapping) or set(manifest_tables) != expected_tables:
        raise PeadReplicationError("source manifest must define the exact six Zacks tables")
    if not isinstance(required_tables, list) or set(required_tables) != expected_tables:
        raise PeadReplicationError(
            "candidate specification required tables differ from the source manifest"
        )
    payload = document.get("payload")
    snapshot_tables = payload.get("tables") if isinstance(payload, Mapping) else None
    if not isinstance(snapshot_tables, Mapping):
        raise PeadReplicationError("snapshot payload.tables must be an object")
    for table_code, table in snapshot_tables.items():
        if table_code not in manifest_tables:
            raise PeadReplicationError(
                f"snapshot table {table_code!r} is absent from source_manifest.json"
            )
        columns = table.get("columns") if isinstance(table, Mapping) else None
        if not isinstance(columns, list):
            raise PeadReplicationError(f"{table_code}.columns must be an array")
        actual_names = [
            item.get("name") if isinstance(item, Mapping) else None for item in columns
        ]
        manifest_entry = manifest_tables[table_code]
        expected_names = (
            manifest_entry.get("required_columns")
            if isinstance(manifest_entry, Mapping)
            else None
        )
        if not isinstance(expected_names, list) or actual_names != expected_names:
            raise PeadReplicationError(
                f"{table_code} exact column sequence differs from source_manifest.json"
            )
    return build_research_manifest_binding(
        candidate_file_name=candidate_path.name,
        candidate_file_sha256=candidate_hash,
        candidate_schema_version=candidate["schema_version"],
        source_file_name=source_path.name,
        source_file_sha256=source_hash,
        source_schema_version=source["schema_version"],
    )


def _atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise PeadReplicationError(
                    f"refusing to overwrite immutable PEAD report: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _emit(payload: Mapping[str, Any], *, stream=None) -> None:
    print(canonical_json(payload), file=stream or sys.stdout)


def _validate_frozen_target(args: argparse.Namespace) -> None:
    expected = {
        "start": FROZEN_START,
        "end": FROZEN_END,
        "horizons": FROZEN_HORIZONS,
        "cost_bps": FROZEN_COST_BPS,
        "fresh_days": FROZEN_FRESH_DAYS,
        "quantile": FROZEN_QUANTILE,
        "winsor": FROZEN_WINSOR,
        "consensus_abs_tolerance": FROZEN_CONSENSUS_ABS_TOLERANCE,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if mismatches:
        raise CliUsageError(
            "configuration differs from frozen PEAD target: " + "; ".join(mismatches)
        )


def parser() -> JsonArgumentParser:
    result = JsonArgumentParser(description=__doc__)
    result.add_argument("--snapshot", required=True)
    result.add_argument("--warehouse-dir", required=True)
    result.add_argument(
        "--candidate-manifest",
        default=str(RESEARCH_PACKAGE / "candidate_specification.json"),
    )
    result.add_argument(
        "--source-manifest",
        default=str(RESEARCH_PACKAGE / "source_manifest.json"),
    )
    result.add_argument("--start", required=True)
    result.add_argument("--end", required=True)
    result.add_argument("--output-json", required=True)
    result.add_argument("--horizons", nargs="+", type=int, default=FROZEN_HORIZONS)
    result.add_argument("--cost-bps", type=float, default=FROZEN_COST_BPS)
    result.add_argument("--fresh-days", type=int, default=FROZEN_FRESH_DAYS)
    result.add_argument("--quantile", type=float, default=FROZEN_QUANTILE)
    result.add_argument("--winsor", type=float, default=FROZEN_WINSOR)
    result.add_argument(
        "--consensus-abs-tolerance",
        type=float,
        required=True,
        help="frozen absolute tolerance for EEH-vs-ES consensus reconciliation",
    )
    result.add_argument(
        "--independent-reconciliation-json",
        default=None,
        help=(
            "PEAD daily reconciliation receipt; when supplied, every bound daily "
            "artifact is replayed before the report may accept it"
        ),
    )
    result.add_argument(
        "--daily-source-report",
        default=str(DEFAULT_DAILY_SOURCE_REPORT),
    )
    result.add_argument(
        "--daily-modeled-ledger",
        default=str(DEFAULT_DAILY_MODELED_LEDGER),
    )
    result.add_argument(
        "--daily-independent-reference",
        default=str(DEFAULT_DAILY_INDEPENDENT_REFERENCE),
    )
    result.add_argument(
        "--daily-input-snapshot",
        default=str(DEFAULT_DAILY_INPUT_SNAPSHOT),
    )
    result.add_argument(
        "--daily-protocol",
        default=str(DEFAULT_DAILY_PROTOCOL),
    )
    result.add_argument(
        "--primary-daily-ledger",
        default=str(DEFAULT_PRIMARY_DAILY_LEDGER),
    )
    result.add_argument(
        "--independent-daily-ledger",
        default=str(DEFAULT_INDEPENDENT_DAILY_LEDGER),
    )
    result.add_argument(
        "--cash-distribution-semantics",
        default=str(RESEARCH_PACKAGE / "cash_distribution_semantics.json"),
    )
    result.add_argument(
        "--terminal-settlement-ledger",
        default=str(RESEARCH_PACKAGE / "terminal_settlement_ledger.json"),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        _validate_frozen_target(args)
        document = _strict_json_object(Path(args.snapshot))
        manifest_binding = _load_research_manifest_binding(
            document,
            candidate_path=Path(args.candidate_manifest),
            source_path=Path(args.source_manifest),
        )
        snapshot = validate_snapshot_document(
            document, start=args.start, end=args.end
        )
        cash_distribution_semantics = load_cash_distribution_semantics(
            args.cash_distribution_semantics
        )
        terminal_settlement_ledger = load_terminal_settlement_ledger(
            args.terminal_settlement_ledger
        )
        reconciliation = None
        if args.independent_reconciliation_json is not None:
            receipt, _ = _strict_manifest_object(
                Path(args.independent_reconciliation_json),
                "PEAD daily reconciliation receipt",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            daily_source_report, _ = _strict_manifest_object(
                Path(args.daily_source_report),
                "daily source report",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            daily_modeled_ledger, _ = _strict_manifest_object(
                Path(args.daily_modeled_ledger),
                "daily modeled execution ledger",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            daily_independent_reference, _ = _strict_manifest_object(
                Path(args.daily_independent_reference),
                "daily independent reference",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            daily_input_snapshot, _ = _strict_manifest_object(
                Path(args.daily_input_snapshot),
                "daily input snapshot",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            daily_protocol, _ = _strict_manifest_object(
                Path(args.daily_protocol),
                "daily money-path protocol",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            primary_daily_ledger, _ = _strict_manifest_object(
                Path(args.primary_daily_ledger),
                "primary daily ledger",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            independent_daily_ledger, _ = _strict_manifest_object(
                Path(args.independent_daily_ledger),
                "independent daily ledger",
                maximum_bytes=MAX_DAILY_ARTIFACT_BYTES,
            )
            reconciliation = replay_and_validate_pead_daily_reconciliation(
                receipt,
                source_report=daily_source_report,
                modeled_ledger=daily_modeled_ledger,
                independent_reference=daily_independent_reference,
                daily_inputs=daily_input_snapshot,
                protocol=daily_protocol,
                primary_daily_ledger=primary_daily_ledger,
                independent_daily_ledger=independent_daily_ledger,
                repository_root=REPOSITORY_ROOT,
            )
        report = build_replication_report(
            snapshot,
            PitWarehouse(args.warehouse_dir),
            start=args.start,
            end=args.end,
            horizons=args.horizons,
            cost_bps=args.cost_bps,
            fresh_days=args.fresh_days,
            quantile=args.quantile,
            winsor_fraction=(args.winsor or None),
            consensus_abs_tolerance=args.consensus_abs_tolerance,
            independent_reconciliation=reconciliation,
            research_manifest_binding=manifest_binding,
            cash_distribution_semantics=cash_distribution_semantics,
            terminal_settlement_ledger=terminal_settlement_ledger,
        )
        report_json = canonical_json(report) + "\n"
        output = Path(args.output_json).resolve()
        _atomic_create(output, report_json)
        _emit(
            {
                "ok": report["completed_full_replication"],
                "status": report["status"],
                "candidate_id": report["candidate_id"],
                "source_snapshot_hash": snapshot.artifact_hash,
                "combined_data_snapshot_hash": (
                    report["combined_data_snapshot"]["artifact_hash"]
                    if report["combined_data_snapshot"] is not None
                    else None
                ),
                "report_sha256": hashlib.sha256(
                    report_json.encode("utf-8")
                ).hexdigest(),
                "report_path": str(output),
                "blockers": report["blockers"],
            }
        )
        return 0 if report["completed_full_replication"] else 1
    except (
        CliUsageError,
        PeadReplicationError,
        PeadEconomicEvidenceError,
        ReplicationIntegrityError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        _emit(
            {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
