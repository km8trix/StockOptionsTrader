#!/usr/bin/env python
"""Build the deterministic, non-broker PEAD modeled execution ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pead_execution_ledger import (  # noqa: E402
    PeadExecutionLedgerError,
    build_pead_execution_ledger,
    validate_pead_execution_ledger,
)
from data.pead_economic_evidence import canonical_json  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_PACKAGE = REPOSITORY_ROOT / "research" / "pead_vq_locked_replication_v1"
DEFAULT_REPORT = RESEARCH_PACKAGE / "development_sample_report_v6.json"
DEFAULT_OUTPUT = RESEARCH_PACKAGE / "modeled_execution_ledger_v1.json"
MAX_REPORT_BYTES = 64 * 1024 * 1024


class CliUsageError(ValueError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise PeadExecutionLedgerError(
            f"source report is not a regular file: {path}"
        )
    raw = path.read_bytes()
    if len(raw) > MAX_REPORT_BYTES:
        raise PeadExecutionLedgerError("source report exceeds the 64 MiB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PeadExecutionLedgerError("source report is not UTF-8") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeadExecutionLedgerError(
                    f"source report contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise PeadExecutionLedgerError(
            f"source report contains invalid number {token}"
        )

    try:
        document = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise PeadExecutionLedgerError(
            f"invalid source report JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(document, dict):
        raise PeadExecutionLedgerError("source report root must be an object")
    canonical = (canonical_json(document) + "\n").encode("utf-8")
    if raw != canonical:
        raise PeadExecutionLedgerError(
            "source report bytes are not the canonical PEAD CLI representation"
        )
    return document, raw


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
                raise PeadExecutionLedgerError(
                    f"refusing to overwrite immutable modeled ledger: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        report, raw = _strict_json(args.report)
        artifact = build_pead_execution_ledger(
            report, report_sha256=hashlib.sha256(raw).hexdigest()
        )
        validate_pead_execution_ledger(artifact, source_report=report)
        _atomic_create(args.output, canonical_json(artifact) + "\n")
        coverage = artifact["payload"]["coverage"]
        print(
            json.dumps(
                {
                    "artifact_hash": artifact["artifact_hash"],
                    "output": str(args.output),
                    "qualifying_evidence": False,
                    "paper_execution_evidence": False,
                    "coverage": coverage,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    except (CliUsageError, PeadExecutionLedgerError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
