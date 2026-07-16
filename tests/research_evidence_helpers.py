"""Shared, internally consistent Foundation research fixtures.

These helpers deliberately build the same raw-report and terminal-ledger facts
that production promotion now requires.  Tests must not bypass credibility
checks with hand-written passing summaries or unattached digest strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from analysis.promotion import (
    PromotionArtifact,
    PromotionPolicy,
    _atomic_create,
    canonical_json,
)
from analysis.research_integrity import (
    ResearchIntegrityLedger,
    ResearchProtocol,
    TrialOutcome,
    TrialRegistration,
    content_hash,
)
from analysis.research_report_store import recompute_foundation_results


PROGRAM_ID = "stock-options-trader"
HOLDOUT_ID = "foundation-oos-v1"
FROZEN_AT = "2026-07-13T13:55:00Z"
OPENED_AT = "2026-07-13T14:00:00Z"
OUTCOME_AT = "2026-07-13T14:01:00Z"
DECIDED_AT = "2026-07-13T14:02:00Z"
MUTATION_AT = "2026-07-13T20:00:00Z"


@dataclass(frozen=True)
class _IntegrityFixture:
    protocol: ResearchProtocol
    trials: tuple[TrialRegistration, ...]
    opening_payload: Mapping[str, Any]


_FIXTURES: dict[str, _IntegrityFixture] = {}


def positive_foundation_report(trade_count: int = 4) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    value = 100_000.0
    for year in (2022, 2023, 2024):
        for offset, day in enumerate(
                pd.bdate_range(f"{year}-01-03", periods=30)):
            value *= 1.0008 if offset % 2 else 1.0012
            points.append({"timestamp": day, "portfolio_value": value})
    return {
        "portfolio_history": points,
        "trades": [
            {"quantity": 1, "price": 100.0, "fixture_trade": index}
            for index in range(trade_count)
        ],
        "pending_signals": [],
    }


def fixture_engine_parameters(seed: int = 7) -> dict[str, Any]:
    return {
        "initial_capital": 100_000.0,
        "commission": 0.001,
        "slippage_bps": 5.0,
        "enable_realistic_fills": True,
        "impact_coef": 0.01,
        "participation_cap": 0.01,
        "adv_window": 20,
        "reject_fills_without_adv": True,
        "seed": seed,
    }


def fixture_regimes(names=("bull", "bear", "sideways")) -> list[dict[str, str]]:
    return [
        {"name": name, "start": "2022-01-01", "end": "2024-12-31"}
        for name in names
    ]


def fixture_results(
        *, trade_count: int = 4, n_trials: int = 3, seed: int = 7,
        regime_names=("bull", "bear", "sideways")):
    return recompute_foundation_results(
        positive_foundation_report(trade_count),
        n_trials=n_trials,
        engine_parameters=fixture_engine_parameters(seed),
        regimes=fixture_regimes(regime_names),
    )


def _event_hash(
        sequence: int, previous: str | None, kind: str,
        target_type: str, target_hash: str) -> str:
    return content_hash({
        "schema_version": 1,
        "record_type": "ledger_event",
        "sequence": sequence,
        "previous_event_hash": previous,
        "event_kind": kind,
        "target_type": target_type,
        "target_hash": target_hash,
        "recorded_at": MUTATION_AT,
    })


def fixture_integrity_evidence(
        *, snapshot_sha: str, code_sha: str, seed: int = 7,
        n_trials: int = 3, identity: str = "default") -> dict[str, Any]:
    key = hashlib.sha256(
        canonical_json({
            "snapshot": snapshot_sha,
            "code": code_sha,
            "seed": seed,
            "n_trials": n_trials,
            "identity": identity,
        }).encode("utf-8")
    ).hexdigest()
    candidate_id = "foundation-target-v1"
    protocol = ResearchProtocol.create(
        program_id=PROGRAM_ID,
        protocol_id=f"foundation-fixture-{key[:20]}",
        version="1",
        objective="Exercise the immutable Foundation promotion contract.",
        hypotheses=[{"hypothesis_id": "foundation-net", "direction": "+"}],
        candidate_specifications=[{
            "candidate_id": candidate_id, "fixture_identity": key}],
        data_plan={"point_in_time": True, "fixture_identity": key},
        evaluation_plan={"primary": "aggregate net OOS"},
        decision_rules={
            "promotion_policy_id": PromotionPolicy.default().policy_id},
        holdouts={HOLDOUT_ID: {
            "start": "2022-01-01",
            "end": "2024-12-31",
            "data_artifact_hash": snapshot_sha,
            "sealed": True,
        }},
        code_version=code_sha,
        frozen_at=FROZEN_AT,
    )
    trials = tuple(
        TrialRegistration.create(
            protocol_hash=protocol.protocol_hash,
            trial_id=f"foundation-fixture-{key[:12]}-{index + 1:03d}",
            candidate_id=candidate_id,
            family="foundation",
            inputs={"fixture_identity": key, "attempt": index + 1},
            data_version="pit-sha256:" + snapshot_sha,
            code_version=code_sha,
            seed=seed + index,
            registered_at=f"2026-07-13T13:{56 + index:02d}:00Z",
        )
        for index in range(n_trials)
    )
    opening_payload = {
        "schema_version": 1,
        "record_type": "holdout_opening",
        "protocol_hash": protocol.protocol_hash,
        "holdout_id": HOLDOUT_ID,
        "holdout_spec_hash": content_hash(
            protocol.payload["holdouts"][HOLDOUT_ID]),
        "trial_hash": trials[-1].trial_hash,
        "data_artifact_hash": snapshot_sha,
        "actor": "independent-reviewer",
        "program_trial_count_at_open": n_trials,
        "protocol_trial_count_at_open": n_trials,
        "opened_at": OPENED_AT,
    }
    opening_hash = content_hash(opening_payload)
    previous = _event_hash(
        0, None, "protocol_frozen", "research_protocol",
        protocol.protocol_hash)
    for index, trial in enumerate(trials, start=1):
        previous = _event_hash(
            index, previous, "trial_registered", "research_trial",
            trial.trial_hash)
    opening_event = _event_hash(
        n_trials + 1, previous, "holdout_opened", "holdout_opening",
        opening_hash)
    _FIXTURES[protocol.protocol_hash] = _IntegrityFixture(
        protocol=protocol, trials=trials, opening_payload=opening_payload)
    return {
        "program_id": PROGRAM_ID,
        "opening_hash": opening_hash,
        "opening": opening_payload,
        "ledger_head_hash": opening_event,
    }


def persist_terminal_integrity(
        root: str | Path, artifact: PromotionArtifact) -> None:
    integrity = (artifact.evidence or {})["research_integrity"]
    fixture = _FIXTURES[integrity["opening"]["protocol_hash"]]
    ledger = ResearchIntegrityLedger(
        Path(root) / "research-integrity", program_id=PROGRAM_ID,
        _clock=lambda: datetime.fromisoformat(
            MUTATION_AT.replace("Z", "+00:00")),
    )
    ledger.freeze_protocol(fixture.protocol)
    for trial in fixture.trials:
        ledger.register_trial(trial)
    opening = ledger.get_holdout_opening(
        protocol_hash=fixture.protocol.protocol_hash,
        holdout_id=HOLDOUT_ID,
    )
    if opening is None:
        opening = ledger.open_holdout(
            protocol_hash=fixture.protocol.protocol_hash,
            holdout_id=HOLDOUT_ID,
            trial_hash=fixture.trials[-1].trial_hash,
            data_artifact_hash=fixture.opening_payload["data_artifact_hash"],
            actor=fixture.opening_payload["actor"],
            opened_at=OPENED_AT,
        )
    assert opening.opening_hash == integrity["opening_hash"]
    summary = {"promotion_level": artifact.decision.value}
    outcome = TrialOutcome.create(
        trial_hash=fixture.trials[-1].trial_hash,
        status="completed",
        result_summary=summary,
        evidence_hash=artifact.artifact_hash,
        recorded_at=OUTCOME_AT,
    )
    outcome = ledger.record_trial_outcome(outcome)
    existing = ledger.get_holdout_decision(
        protocol_hash=fixture.protocol.protocol_hash,
        holdout_id=HOLDOUT_ID,
    )
    if existing is None:
        existing = ledger.record_holdout_decision(
            opening_hash=opening.opening_hash,
            decision="pass",
            result_summary=summary,
            result_artifact_hash=artifact.artifact_hash,
            actor="independent-reviewer",
            decided_at=DECIDED_AT,
        )
    payload = {
        "schema_version": 1,
        "record_type": "research_integrity_receipt",
        "program_id": ledger.program_id,
        "artifact_hash": artifact.artifact_hash,
        "protocol_hash": fixture.protocol.protocol_hash,
        "trial_hash": fixture.trials[-1].trial_hash,
        "opening_hash": opening.opening_hash,
        "outcome_hash": outcome.record_hash,
        "decision_hash": existing.record_hash,
        "opening_event_hash": ledger.event_hash_for(
            event_kind="holdout_opened", target_hash=opening.opening_hash),
        "terminal_event_hash": ledger.event_hash_for(
            event_kind="holdout_decided", target_hash=existing.record_hash),
    }
    receipt_hash = hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()
    _atomic_create(
        Path(root) / "research-integrity-receipts"
        / f"{artifact.artifact_hash}.json",
        canonical_json({"receipt_hash": receipt_hash, "payload": payload}) + "\n",
    )
