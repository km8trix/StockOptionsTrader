#!/usr/bin/env python
"""Assess whether a candidate package is complete enough to preregister.

This command is deliberately read-only.  It cannot freeze a protocol,
register a trial, or open a holdout.  Exit status is 0 only when the package is
ready to freeze, 1 when well-formed evidence has blockers, and 2 for malformed
input or an operator error.  Every outcome is emitted as strict JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.candidate_readiness import (  # noqa: E402
    ReadinessPackageError,
    assess_candidate_readiness,
)


MAX_PACKAGE_BYTES = 2 * 1024 * 1024


class CliUsageError(ValueError):
    """Argparse error represented through the CLI's JSON error surface."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def _emit(payload: Mapping[str, Any], *, stream=None) -> None:
    destination = stream if stream is not None else sys.stdout
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        file=destination,
    )


def _strict_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"package is not a regular file: {path}")
    if path.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError(f"package exceeds the {MAX_PACKAGE_BYTES}-byte limit")
    raw = path.read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > MAX_PACKAGE_BYTES:
        raise ValueError(f"package exceeds the {MAX_PACKAGE_BYTES}-byte limit")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"package contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"package contains invalid JSON number {token}")

    try:
        document = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"package is invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(document, dict):
        raise TypeError("package must be a JSON object")
    return document


def _parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, help="strict readiness JSON")
    parser.add_argument("--warehouse-dir", required=True, help="local PIT Parquet warehouse")
    parser.add_argument("--repo-dir", default=".", help="Git checkout whose clean HEAD is assessed")
    parser.add_argument(
        "--evidence-root",
        default=None,
        help="root for relative evidence files (defaults to package directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        package_path = Path(args.package).resolve()
        package = _strict_json_object(package_path)
        evidence_root = (
            Path(args.evidence_root).resolve()
            if args.evidence_root is not None
            else package_path.parent
        )
        report = assess_candidate_readiness(
            package,
            warehouse_dir=args.warehouse_dir,
            repo_dir=args.repo_dir,
            evidence_root=evidence_root,
        )
    except (CliUsageError, ReadinessPackageError, OSError, TypeError, ValueError) as exc:
        _emit(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 2

    _emit(report.to_mapping())
    return 0 if report.ready_to_freeze else 1


if __name__ == "__main__":
    raise SystemExit(main())
