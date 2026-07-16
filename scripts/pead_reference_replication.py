#!/usr/bin/env python
"""Run and reconcile the independent PEAD reference implementation."""

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

from analysis.pead_reference_replication import (  # noqa: E402
    CANDIDATE_ID,
    PeadReferenceError,
    build_reference_comparison,
    canonical_json,
    content_hash,
    run_reference_reconstruction,
    verify_reference_artifact,
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


ROOT = Path(__file__).resolve().parent.parent
RESEARCH_PACKAGE = ROOT / "research" / "pead_vq_locked_replication_v1"
FROZEN_START = "2015-01-01"
FROZEN_END = "2024-09-30"
FROZEN_HORIZONS = [21, 63]
FROZEN_FRESH_DAYS = 63
FROZEN_CONSENSUS_TOLERANCE = 0.01
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class ReferenceCliError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ReferenceCliError(message)


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadReferenceError(f"{label} is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadReferenceError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise PeadReferenceError(f"{label} contains invalid number {token}")

    try:
        document = json.loads(
            text, object_pairs_hook=unique, parse_constant=reject
        )
    except json.JSONDecodeError as exc:
        raise PeadReferenceError(
            f"invalid {label} JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(document, dict):
        raise PeadReferenceError(f"{label} root must be an object")
    return document


def _read_json(path: Path, label: str, *, maximum: int) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise PeadReferenceError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise PeadReferenceError(f"{label} exceeds its byte limit")
    return _strict_json_bytes(raw, label), raw


def _load_snapshot(path: Path) -> dict[str, Any]:
    document, raw = _read_json(
        path, "Zacks source snapshot", maximum=MAX_JSON_BYTES
    )
    try:
        verified = EarningsAnnouncementSnapshot.from_json(raw.decode("utf-8"))
    except SnapshotIntegrityError as exc:
        raise PeadReferenceError(
            "Zacks source snapshot failed acquisition-layer verification"
        ) from exc
    if verified.artifact_hash != document.get("artifact_hash"):
        raise PeadReferenceError("acquisition verifier returned another snapshot hash")
    return document


def _manifest_binding(candidate_path: Path, source_path: Path) -> dict[str, Any]:
    candidate, candidate_raw = _read_json(
        candidate_path, "candidate specification", maximum=MAX_MANIFEST_BYTES
    )
    source, source_raw = _read_json(
        source_path, "source manifest", maximum=MAX_MANIFEST_BYTES
    )
    if candidate.get("schema_version") != "candidate_specification.v1":
        raise PeadReferenceError("unsupported candidate specification schema")
    if candidate.get("candidate_id") != CANDIDATE_ID:
        raise PeadReferenceError("candidate specification belongs to another target")
    if source.get("schema_version") != "pead_source_manifest.v1":
        raise PeadReferenceError("unsupported source manifest schema")
    if source.get("candidate_id") != CANDIDATE_ID:
        raise PeadReferenceError("source manifest belongs to another target")
    identity = source.get("source")
    if not isinstance(identity, Mapping) or identity.get("source_id") != (
        "nasdaq-data-link-zacks"
    ):
        raise PeadReferenceError("source manifest does not identify Zacks")
    payload = {
        "schema_version": "pead_research_manifest_binding.v1",
        "candidate_id": CANDIDATE_ID,
        "source_id": "nasdaq-data-link-zacks",
        "candidate_specification": {
            "file_name": candidate_path.name,
            "file_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "schema_version": candidate["schema_version"],
        },
        "source_manifest": {
            "file_name": source_path.name,
            "file_sha256": hashlib.sha256(source_raw).hexdigest(),
            "schema_version": source["schema_version"],
        },
    }
    return {"artifact_hash": content_hash(payload), "payload": payload}


def _code_identity(implementation_id: str, paths: Sequence[Path]) -> dict[str, Any]:
    files = []
    for path in paths:
        raw = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    files.sort(key=lambda item: item["path"])
    return {
        "implementation_id": implementation_id,
        "code_hash": content_hash(files),
        "files": files,
    }


def _atomic_create(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise PeadReferenceError(
                    f"refusing to overwrite immutable reference artifact: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def parser() -> Parser:
    result = Parser(description=__doc__)
    result.add_argument("--snapshot", required=True)
    result.add_argument("--warehouse-dir", required=True)
    result.add_argument("--primary-report", required=True)
    result.add_argument("--output-json", required=True)
    result.add_argument(
        "--candidate-manifest",
        default=str(RESEARCH_PACKAGE / "candidate_specification.json"),
    )
    result.add_argument(
        "--source-manifest",
        default=str(RESEARCH_PACKAGE / "source_manifest.json"),
    )
    result.add_argument("--start", default=FROZEN_START)
    result.add_argument("--end", default=FROZEN_END)
    result.add_argument("--horizons", nargs="+", type=int, default=FROZEN_HORIZONS)
    result.add_argument("--fresh-days", type=int, default=FROZEN_FRESH_DAYS)
    result.add_argument(
        "--consensus-abs-tolerance",
        type=float,
        default=FROZEN_CONSENSUS_TOLERANCE,
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


def _enforce_frozen(args: argparse.Namespace) -> None:
    expected = {
        "start": FROZEN_START,
        "end": FROZEN_END,
        "horizons": FROZEN_HORIZONS,
        "fresh_days": FROZEN_FRESH_DAYS,
        "consensus_abs_tolerance": FROZEN_CONSENSUS_TOLERANCE,
    }
    differences = [
        f"{field}={getattr(args, field)!r} (expected {value!r})"
        for field, value in expected.items()
        if getattr(args, field) != value
    ]
    if differences:
        raise ReferenceCliError(
            "configuration differs from the frozen target: " + "; ".join(differences)
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        _enforce_frozen(args)
        snapshot = _load_snapshot(Path(args.snapshot))
        primary_report, primary_raw = _read_json(
            Path(args.primary_report), "primary PEAD report", maximum=MAX_JSON_BYTES
        )
        protocol = _manifest_binding(
            Path(args.candidate_manifest), Path(args.source_manifest)
        )
        cash_distribution_semantics = load_cash_distribution_semantics(
            args.cash_distribution_semantics
        )
        terminal_settlement_ledger = load_terminal_settlement_ledger(
            args.terminal_settlement_ledger
        )
        run = run_reference_reconstruction(
            snapshot,
            PitWarehouse(args.warehouse_dir),
            start=args.start,
            end=args.end,
            horizons=args.horizons,
            fresh_days=args.fresh_days,
            consensus_abs_tolerance=args.consensus_abs_tolerance,
            cash_distribution_semantics=cash_distribution_semantics,
            terminal_settlement_ledger=terminal_settlement_ledger,
        )
        comparison = build_reference_comparison(
            run,
            primary_report,
            protocol_hash=protocol["artifact_hash"],
            primary_report_sha256=hashlib.sha256(primary_raw).hexdigest(),
            primary_implementation=_code_identity(
                "pead-primary-vectorized-v1",
                [ROOT / "analysis" / "pead_replication.py",
                 ROOT / "scripts" / "pead_replication.py",
                 ROOT / "analysis" / "pead_economic_returns.py",
                 ROOT / "data" / "pit_warehouse.py",
                 ROOT / "data" / "corporate_action_evidence.py",
                 ROOT / "data" / "pead_economic_evidence.py",
                 ROOT / "data" / "session_close_calendar.py"],
            ),
            reference_implementation=_code_identity(
                "pead-independent-reference-v1",
                [ROOT / "analysis" / "pead_reference_replication.py",
                 ROOT / "scripts" / "pead_reference_replication.py",
                 ROOT / "data" / "pit_warehouse.py",
                 ROOT / "data" / "corporate_action_evidence.py",
                 ROOT / "data" / "pead_economic_evidence.py",
                 ROOT / "data" / "session_close_calendar.py"],
            ),
            start=args.start,
            end=args.end,
            horizons=args.horizons,
            fresh_days=args.fresh_days,
            consensus_abs_tolerance=args.consensus_abs_tolerance,
        )
        verify_reference_artifact(comparison)
        _atomic_create(
            Path(args.output_json), canonical_json(comparison) + "\n"
        )
        passed = comparison["payload"]["comparison"]["signal_path_passed"]
        print(
            canonical_json(
                {
                    "artifact_hash": comparison["artifact_hash"],
                    "signal_path_passed": passed,
                    "qualifying_evidence": False,
                    "replication_evidence_eligible": False,
                    "output_json": str(Path(args.output_json).resolve()),
                }
            )
        )
        return 0 if passed else 1
    except (
        OSError,
        PeadEconomicEvidenceError,
        PeadReferenceError,
        ReferenceCliError,
        ValueError,
    ) as exc:
        print(
            canonical_json(
                {"error": type(exc).__name__, "message": str(exc)}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
