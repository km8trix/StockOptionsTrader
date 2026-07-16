"""Replay-gated acceptance token for a PEAD daily reconciliation receipt.

JSON content addressing proves integrity, not truth.  A self-consistent receipt
can still contain fabricated outputs.  The PEAD report builder therefore does
not accept receipt mappings directly.  Callers must first supply every bound
source and both complete daily ledgers to this module; only the exhaustive
rebuild returns the in-memory token accepted by ``build_replication_report``.

This is an API trust boundary, not a cryptographic sandbox against arbitrary
Python code running in the same process.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from analysis.pead_daily_reconciliation import (
    _inspect_receipt_for_bindings,
    pead_reconciliation_input,
    validate_pead_daily_reconciliation_receipt,
)
from data.pead_economic_evidence import canonical_json


_CONSTRUCTION_TOKEN = object()


class ValidatedPeadDailyReconciliation:
    """Opaque result of a complete source-bound daily-ledger replay."""

    __slots__ = ("_document_json", "_source_report_core_hash", "_sealed")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ValidatedPeadDailyReconciliation cannot be subclassed")

    def __init__(
        self,
        construction_token: object,
        document: Mapping[str, Any],
        source_report_core_hash: str,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError(
                "ValidatedPeadDailyReconciliation must come from full replay"
            )
        object.__setattr__(self, "_document_json", canonical_json(document))
        object.__setattr__(self, "_source_report_core_hash", source_report_core_hash)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise TypeError("validated PEAD replay tokens are immutable")
        object.__setattr__(self, name, value)

    @property
    def document(self) -> dict[str, Any]:
        return json.loads(self._document_json)

    @property
    def artifact_hash(self) -> str:
        return self.document["artifact_hash"]

    @property
    def source_report_core_hash(self) -> str:
        return self._source_report_core_hash


def replay_and_validate_pead_daily_reconciliation(
    receipt: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
    modeled_ledger: Mapping[str, Any],
    independent_reference: Mapping[str, Any],
    daily_inputs: Mapping[str, Any],
    protocol: Mapping[str, Any],
    primary_daily_ledger: Mapping[str, Any],
    independent_daily_ledger: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> ValidatedPeadDailyReconciliation:
    """Replay both implementations and return the only accepted report token."""
    verified = validate_pead_daily_reconciliation_receipt(
        receipt,
        source_report=source_report,
        modeled_ledger=modeled_ledger,
        independent_reference=independent_reference,
        daily_inputs=daily_inputs,
        protocol=protocol,
        primary_daily_ledger=primary_daily_ledger,
        independent_daily_ledger=independent_daily_ledger,
        repository_root=repository_root,
    )
    core_hash = pead_reconciliation_input(source_report)["artifact_hash"]
    return ValidatedPeadDailyReconciliation(
        _CONSTRUCTION_TOKEN, verified, core_hash
    )


def validate_replayed_pead_daily_reconciliation_for_bindings(
    value: Any,
    *,
    combined_data_snapshot_hash: str,
    economic_return_inputs_hash: str,
    research_manifest_binding_hash: str,
) -> dict[str, Any]:
    """Validate current report bindings for an already source-replayed token."""
    if type(value) is not ValidatedPeadDailyReconciliation:
        raise TypeError("PEAD reconciliation acceptance requires a replay token")
    return _inspect_receipt_for_bindings(
        value.document,
        combined_data_snapshot_hash=combined_data_snapshot_hash,
        economic_return_inputs_hash=economic_return_inputs_hash,
        research_manifest_binding_hash=research_manifest_binding_hash,
    )


__all__ = [
    "ValidatedPeadDailyReconciliation",
    "replay_and_validate_pead_daily_reconciliation",
    "validate_replayed_pead_daily_reconciliation_for_bindings",
]
