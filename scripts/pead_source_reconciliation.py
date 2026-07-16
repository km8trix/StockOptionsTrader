#!/usr/bin/env python
"""Create an immutable PEAD announcement/consensus reconciliation receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pead_source_reconciliation import (  # noqa: E402
    PeadSourceReconciliationError,
    build_pead_source_reconciliation,
    verify_pead_source_reconciliation,
)
from data.pead_announcement_evidence import (  # noqa: E402
    PeadAnnouncementEvidenceError,
    load_pead_announcement_evidence,
)
from data.pead_consensus_evidence import (  # noqa: E402
    PeadConsensusEvidenceError,
    canonical_json,
    load_pead_consensus_evidence,
)


class CliUsageError(ValueError):
    """The command-line request is incomplete or malformed."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        description=(
            "Reconcile a frozen PEAD consensus artifact with independent "
            "announcement evidence. Exit 1 means a valid receipt was created "
            "but no normalized event input reconciled. Research remains blocked "
            "until the raw-source, external-binding, and market gates pass."
        )
    )
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--announcement", type=Path, required=True)
    parser.add_argument("--reconciled-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


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
        created = False
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError as exc:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise PeadSourceReconciliationError(
                    f"refusing to overwrite immutable source reconciliation: {path}"
                ) from exc
        if created:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        consensus = load_pead_consensus_evidence(args.consensus)
        universe = consensus["payload"]["event_universe"]
        announcement = load_pead_announcement_evidence(
            args.announcement, expected_event_manifest=universe
        )
        receipt = build_pead_source_reconciliation(
            consensus,
            announcement,
            reconciled_at_utc=args.reconciled_at_utc,
        )
        verified = verify_pead_source_reconciliation(
            receipt,
            consensus_evidence=consensus,
            announcement_evidence=announcement,
        )
        _atomic_create(args.output, canonical_json(verified) + "\n")
        qualification = verified["payload"]["qualification"]
        summary = {
            "artifact_hash": verified["artifact_hash"],
            "output": str(args.output),
            "coverage": verified["payload"]["coverage"],
            "has_reconciled_event_inputs": qualification[
                "has_reconciled_event_inputs"
            ],
            "source_qualified_event_inputs_allowed": False,
            "historical_replication_allowed": qualification[
                "historical_replication_allowed"
            ],
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if qualification["has_reconciled_event_inputs"] else 1
    except (
        CliUsageError,
        PeadAnnouncementEvidenceError,
        PeadConsensusEvidenceError,
        PeadSourceReconciliationError,
        OSError,
    ) as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
