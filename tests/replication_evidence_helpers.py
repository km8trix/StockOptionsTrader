"""Deterministic independent-replication evidence for promotion fixtures."""

from __future__ import annotations

from analysis.independent_replication import (
    ImplementationIdentity,
    IndependentReplicationContract,
    NUMERIC_FIELDS,
    NumericTolerance,
    ReplicationEvidence,
    ReplicationEvidenceStore,
    reconcile_implementations,
)


PRIMARY_CODE_HASH = "5" * 64
REPLICATION_CODE_HASH = "6" * 64


def replication_evidence_for(
        research_artifact, *, mismatch_field: str | None = None,
        protocol_hash: str | None = None,
        data_snapshot_hash: str | None = None) -> ReplicationEvidence:
    """Build complete, deterministic evidence bound to one research artifact."""
    research = research_artifact.evidence or {}
    checkpoint_key = {"checkpoint": "fixture-001"}
    contract = IndependentReplicationContract(
        protocol_hash=(protocol_hash or research["research_integrity"]
                       ["opening"]["protocol_hash"]),
        data_snapshot_hash=(data_snapshot_hash
                            or research["warehouse_snapshot"]["version"]),
        primary=ImplementationIdentity(
            "foundation-primary-fixture", PRIMARY_CODE_HASH),
        replication=ImplementationIdentity(
            "foundation-independent-fixture", REPLICATION_CODE_HASH),
        expected_observation_keys=[checkpoint_key],
        tolerances={
            field: NumericTolerance(absolute=0.0, relative=0.0)
            for field in NUMERIC_FIELDS
        },
    )
    primary = {
        "key": checkpoint_key,
        "eligibility": True,
        **{field: 0.0 for field in NUMERIC_FIELDS},
    }
    replication = dict(primary)
    if mismatch_field is not None:
        if mismatch_field not in NUMERIC_FIELDS:
            raise ValueError("mismatch_field must be a numeric replication field")
        replication[mismatch_field] = 1.0
    return reconcile_implementations(
        contract,
        primary_observations=[primary],
        replication_observations=[replication],
    )


def persist_passing_replication(registry, research_artifact) -> ReplicationEvidence:
    evidence = replication_evidence_for(research_artifact)
    ReplicationEvidenceStore(registry.root).persist(evidence)
    return evidence


def persist_replication(registry, evidence: ReplicationEvidence) -> ReplicationEvidence:
    ReplicationEvidenceStore(registry.root).persist(evidence)
    return evidence
