#!/usr/bin/env python
"""Build and reconcile the two bounded PEAD daily money-path implementations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pead_daily_inputs import (  # noqa: E402
    PeadDailyInputError,
    build_pead_daily_input_snapshot,
    validate_pead_daily_input_snapshot,
)
from analysis.pead_daily_acceptance import (  # noqa: E402
    replay_and_validate_pead_daily_reconciliation,
)
from analysis.pead_daily_ledger import (  # noqa: E402
    PeadDailyLedgerError,
    build_primary_daily_ledger,
    validate_primary_daily_ledger,
)
from analysis.pead_daily_reconciliation import (  # noqa: E402
    PeadDailyReconciliationError,
    build_pead_daily_reconciliation_receipt,
)
from analysis.pead_daily_reference import (  # noqa: E402
    PeadIndependentDailyLedgerError,
    build_independent_daily_ledger,
    validate_independent_daily_ledger,
)
from data.pead_economic_evidence import canonical_json, content_hash  # noqa: E402
from data.pit_warehouse import PitWarehouse, WarehouseReadError  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "research" / "pead_vq_locked_replication_v1"
DEFAULT_REPORT = PACKAGE / "development_sample_report_v6.json"
DEFAULT_MODELED_LEDGER = PACKAGE / "modeled_execution_ledger_v1.json"
DEFAULT_REFERENCE = PACKAGE / "independent_reference_comparison_v5.json"
DEFAULT_PROTOCOL = PACKAGE / "daily_money_path_protocol_v2.json"
DEFAULT_INPUTS = PACKAGE / "daily_input_snapshot_v1.json"
DEFAULT_PRIMARY = PACKAGE / "primary_daily_ledger_v2.json"
DEFAULT_INDEPENDENT = PACKAGE / "independent_daily_ledger_v2.json"
DEFAULT_GENERIC = PACKAGE / "daily_generic_replication_evidence_v2.json"
DEFAULT_KEYS = PACKAGE / "daily_money_path_key_manifest_v2.json"
DEFAULT_RECEIPT = PACKAGE / "daily_money_path_reconciliation_v3.json"
MAX_JSON_BYTES = 512 * 1024 * 1024


class DailyReconciliationCliError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise DailyReconciliationCliError(message)


def _strict_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise DailyReconciliationCliError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise DailyReconciliationCliError(f"{label} exceeds its byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DailyReconciliationCliError(f"{label} is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DailyReconciliationCliError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject(token: str) -> None:
        raise DailyReconciliationCliError(f"{label} contains invalid number {token}")

    try:
        document = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise DailyReconciliationCliError(
            f"invalid {label} JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(document, dict):
        raise DailyReconciliationCliError(f"{label} root must be an object")
    return document, raw


def _atomic_create(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(document) + "\n").encode("utf-8")
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
                raise DailyReconciliationCliError(
                    f"refusing to overwrite immutable daily artifact: {path}"
                )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def parser() -> Parser:
    result = Parser(description=__doc__)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--modeled-ledger", type=Path, default=DEFAULT_MODELED_LEDGER)
    result.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--warehouse-dir")
    result.add_argument("--inputs-output", type=Path, default=DEFAULT_INPUTS)
    result.add_argument("--primary-output", type=Path, default=DEFAULT_PRIMARY)
    result.add_argument("--independent-output", type=Path, default=DEFAULT_INDEPENDENT)
    result.add_argument("--generic-output", type=Path, default=DEFAULT_GENERIC)
    result.add_argument("--keys-output", type=Path, default=DEFAULT_KEYS)
    result.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        report, _ = _strict_json(args.report, "source report")
        modeled, _ = _strict_json(args.modeled_ledger, "modeled execution ledger")
        reference, _ = _strict_json(args.reference, "independent reference")
        protocol, _ = _strict_json(args.protocol, "daily money-path protocol")
        provider = PitWarehouse(args.warehouse_dir)
        daily_inputs = build_pead_daily_input_snapshot(
            report, modeled, reference, provider
        )
        validate_pead_daily_input_snapshot(
            daily_inputs, report=report, ledger=modeled, reference=reference
        )
        primary = build_primary_daily_ledger(
            report, modeled, daily_inputs, protocol
        )
        validate_primary_daily_ledger(
            primary,
            source_report=report,
            modeled_ledger=modeled,
            daily_inputs=daily_inputs,
            protocol=protocol,
        )
        independent = build_independent_daily_ledger(reference, daily_inputs)
        validate_independent_daily_ledger(
            independent,
            reference_artifact=reference,
            daily_inputs=daily_inputs,
        )
        receipt = build_pead_daily_reconciliation_receipt(
            report,
            modeled,
            reference,
            daily_inputs,
            protocol,
            primary,
            independent,
            repository_root=ROOT,
        )
        replay_and_validate_pead_daily_reconciliation(
            receipt,
            source_report=report,
            modeled_ledger=modeled,
            independent_reference=reference,
            daily_inputs=daily_inputs,
            protocol=protocol,
            primary_daily_ledger=primary,
            independent_daily_ledger=independent,
            repository_root=ROOT,
        )
        generic = receipt["payload"]["generic_replication_evidence"]
        key_payload = receipt["payload"]["key_manifest"]
        key_document = {
            "artifact_hash": content_hash(key_payload),
            "payload": key_payload,
        }
        outputs = (
            (args.inputs_output, daily_inputs),
            (args.primary_output, primary),
            (args.independent_output, independent),
            (args.generic_output, generic),
            (args.keys_output, key_document),
            (args.receipt_output, receipt),
        )
        for path, document in outputs:
            _atomic_create(path, document)
        comparison = receipt["payload"]["comparison"]
        print(
            canonical_json(
                {
                    "receipt_artifact_hash": receipt["artifact_hash"],
                    "bounded_modeled_daily_money_path_reconciliation_passed": (
                        receipt["payload"][
                            "bounded_modeled_daily_money_path_reconciliation_passed"
                        ]
                    ),
                    "discrepancy_count": comparison["discrepancy_count"],
                    "expected_observations": comparison["expected_observations"],
                    "qualifying_evidence": False,
                    "paper_execution_evidence": False,
                    "promotion_allowed": False,
                    "receipt_output": str(args.receipt_output.resolve()),
                }
            )
        )
        return 0 if comparison["passed"] else 1
    except (
        DailyReconciliationCliError,
        OSError,
        PeadDailyInputError,
        PeadDailyLedgerError,
        PeadDailyReconciliationError,
        PeadIndependentDailyLedgerError,
        WarehouseReadError,
    ) as exc:
        print(
            canonical_json({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
