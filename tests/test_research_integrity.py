"""Focused tests for immutable research protocols and the trial ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from analysis.research_integrity import (
    DuplicateRecordError,
    HoldoutAlreadyDecided,
    HoldoutAlreadyOpened,
    IntegrityViolation,
    ProtocolNotFrozen,
    ResearchIntegrityLedger,
    ResearchProtocol,
    TrialOutcome,
    TrialRegistration,
    UnknownRecordError,
    canonical_json,
    content_hash,
    prospective_data_commitment_hash,
)


PROGRAM = "stock-options-trader"
FROZEN_AT = "2026-07-13T14:00:00Z"
REGISTERED_AT = "2026-07-13T14:01:00Z"
OPENED_AT = "2026-07-13T14:02:00Z"
DECIDED_AT = "2026-07-13T14:03:00Z"
DATA_HASH = "a" * 64
EVIDENCE_HASH = "b" * 64
RESULT_HASH = "c" * 64
SOURCE_HASH = "d" * 64
QUERY_HASH = "e" * 64
ACQUISITION_HASH = "f" * 64
REALIZED_HASH = "8" * 64


class MutableClock:
    def __init__(self, value: str = "2026-07-13T20:00:00Z") -> None:
        self.set(value)

    def set(self, value: str) -> None:
        self.value = datetime.fromisoformat(
            value.replace("Z", "+00:00")).astimezone(timezone.utc)

    def __call__(self) -> datetime:
        return self.value


def prospective_specification() -> dict:
    return {
        "mode": "prospective",
        "start": "2026-08-01",
        "end": "2026-12-31",
        "source_manifest_hash": SOURCE_HASH,
        "query_hash": QUERY_HASH,
        "acquisition_plan_hash": ACQUISITION_HASH,
        "append_only": True,
        "sealed": True,
    }


def protocol(**overrides) -> ResearchProtocol:
    values = {
        "program_id": PROGRAM,
        "protocol_id": "issuance-midcap-ls",
        "version": "1",
        "objective": "Test whether net issuance has executable net alpha.",
        "hypotheses": [{
            "hypothesis_id": "issuance-spread",
            "claim": "Low issuance outperforms high issuance net of costs.",
        }],
        "candidate_specifications": [{
            "candidate_id": "issuance-midcap-ls-v1",
            "formation_days": 252,
            "holding_days": 63,
        }],
        "data_plan": {
            "point_in_time": True,
            "development_end": "2024-12-31",
        },
        "evaluation_plan": {
            "primary_test": "stationary-block-bootstrap",
            "cost_stress": [1, 2, 3],
        },
        "decision_rules": {
            "minimum_net_sharpe": 0.75,
            "failure_is_terminal": True,
        },
        "holdouts": {
            "historical-oos-1": {
                "start": "2025-01-01",
                "end": "2026-06-30",
                "data_artifact_hash": DATA_HASH,
            },
        },
        "code_version": "git:0123456789abcdef",
        "frozen_at": FROZEN_AT,
    }
    values.update(overrides)
    return ResearchProtocol.create(**values)


def trial(protocol_hash: str, **overrides) -> TrialRegistration:
    values = {
        "protocol_hash": protocol_hash,
        "trial_id": "issuance-dev-001",
        "candidate_id": "issuance-midcap-ls-v1",
        "family": "net-share-issuance",
        "inputs": {
            "formation_days": 252,
            "holding_days": 63,
            "universe": "pit-midcap",
        },
        "data_version": "warehouse:abc",
        "code_version": "git:0123456789abcdef",
        "seed": 7,
        "registered_at": REGISTERED_AT,
    }
    values.update(overrides)
    return TrialRegistration.create(**values)


def frozen_ledger(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    frozen = ledger.freeze_protocol(protocol())
    registered = ledger.register_trial(trial(frozen.protocol_hash))
    return ledger, frozen, registered


def prospective_ledger(tmp_path):
    clock = MutableClock()
    ledger = ResearchIntegrityLedger(
        tmp_path / "integrity", program_id=PROGRAM, _clock=clock)
    preregistration = ledger.freeze_protocol(protocol(
        protocol_id="issuance-prospective",
        holdouts={"prospective-oos-1": prospective_specification()},
    ))
    commitment = prospective_data_commitment_hash(prospective_specification())
    attempted = ledger.register_trial(trial(
        preregistration.protocol_hash,
        trial_id="issuance-prospective-001",
        data_version=f"prospective-commitment:{commitment}",
    ))
    return ledger, preregistration, attempted, commitment, clock


def test_protocol_is_canonical_and_content_addressed():
    first = protocol(
        data_plan={"point_in_time": True, "development_end": "2024-12-31"})
    second = protocol(
        data_plan={"development_end": "2024-12-31", "point_in_time": True})

    assert first.protocol_hash == second.protocol_hash
    assert first.payload_json == canonical_json(first.payload)
    assert content_hash(first.payload) == first.protocol_hash
    assert ResearchProtocol.from_json(first.to_json()) == first

    candidates = [
        {"candidate_id": "issuance-midcap-ls-v1", "holding_days": 63},
        {"candidate_id": "pead-vq-v1", "holding_days": 21},
    ]
    reordered = protocol(candidate_specifications=list(reversed(candidates)))
    declared = protocol(candidate_specifications=candidates)
    assert reordered.protocol_hash == declared.protocol_hash


def test_protocol_rejects_ambiguous_or_incomplete_preregistration():
    with pytest.raises(ValueError, match="holdouts"):
        protocol(holdouts={})
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        protocol(candidate_specifications=[
            {"candidate_id": "same", "x": 1},
            {"candidate_id": "same", "x": 2},
        ])
    with pytest.raises(ValueError, match="NaN"):
        protocol(decision_rules={"minimum_net_sharpe": float("nan")})
    with pytest.raises(TypeError, match="mapping keys"):
        protocol(data_plan={1: "not canonical"})
    with pytest.raises(ValueError, match="requires data_artifact_hash"):
        protocol(holdouts={"fixed": {"start": "2025-01-01",
                                     "end": "2025-12-31"}})
    future_shaped = prospective_specification()
    future_shaped.pop("mode")
    with pytest.raises(ValueError, match="prospective fields without mode"):
        protocol(holdouts={"downgraded": future_shaped})


def test_prospective_holdout_declaration_is_exact_and_content_addressed():
    specification = prospective_specification()
    preregistration = protocol(
        holdouts={"prospective-oos-1": specification})

    commitment = prospective_data_commitment_hash(specification)
    assert len(commitment) == 64
    assert commitment == prospective_data_commitment_hash(
        dict(reversed(list(specification.items()))))
    assert preregistration.payload["holdouts"]["prospective-oos-1"] \
        == specification

    for changed, message in (
        ({**specification, "append_only": False}, "append_only"),
        ({**specification, "sealed": False}, "sealed"),
        ({**specification, "unexpected": True}, "extra"),
        ({key: value for key, value in specification.items()
          if key != "query_hash"}, "missing"),
        ({**specification, "start": "2027-01-01"}, "after end"),
    ):
        with pytest.raises(ValueError, match=message):
            protocol(holdouts={"prospective-oos-1": changed})

    with pytest.raises(ValueError, match="duplicate holdout data commitment"):
        protocol(holdouts={
            "prospective-oos-1": specification,
            "prospective-oos-2": specification,
        })


def test_protocol_freeze_is_idempotent_but_identity_cannot_be_rebound(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    first = protocol()

    assert ledger.freeze_protocol(first) == first
    assert ledger.freeze_protocol(first) == first
    assert ledger.verify()["event_count"] == 1

    conflicting = protocol(objective="A changed objective after preregistration.")
    with pytest.raises(DuplicateRecordError, match="already frozen"):
        ledger.freeze_protocol(conflicting)
    assert ledger.verify()["event_count"] == 1


def test_ledger_root_is_permanently_bound_to_one_program(tmp_path):
    root = tmp_path / "integrity"
    ResearchIntegrityLedger(root, program_id=PROGRAM)
    with pytest.raises(IntegrityViolation, match="different research program"):
        ResearchIntegrityLedger(root, program_id="another-program")


def test_trial_requires_frozen_protocol_and_declared_candidate(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    preregistration = protocol()
    attempted = trial(preregistration.protocol_hash)
    with pytest.raises(ProtocolNotFrozen):
        ledger.register_trial(attempted)

    ledger.freeze_protocol(preregistration)
    undeclared = trial(
        preregistration.protocol_hash, trial_id="other",
        candidate_id="selected-after-looking")
    with pytest.raises(ValueError, match="not declared"):
        ledger.register_trial(undeclared)

    wrong_code = trial(
        preregistration.protocol_hash, trial_id="wrong-code",
        code_version="git:different")
    with pytest.raises(ValueError, match="code_version"):
        ledger.register_trial(wrong_code)


def test_trial_count_is_ledger_derived_and_conflicts_are_rejected(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    preregistration = ledger.freeze_protocol(protocol())
    first = trial(preregistration.protocol_hash)
    second = trial(
        preregistration.protocol_hash, trial_id="issuance-dev-002", seed=8,
        registered_at="2026-07-13T14:01:30Z")

    ledger.register_trial(first)
    ledger.register_trial(first)  # exact retry is not another research attempt
    ledger.register_trial(second)
    assert ledger.trial_count == 2
    assert ledger.protocol_trial_count(preregistration.protocol_hash) == 2

    changed = trial(
        preregistration.protocol_hash, inputs={"holding_days": 21})
    with pytest.raises(DuplicateRecordError, match="trial_id"):
        ledger.register_trial(changed)
    assert ledger.trial_count == 2


def test_new_ledger_events_persist_trusted_mutation_timestamps(tmp_path):
    ledger, _preregistration, _attempted = frozen_ledger(tmp_path)

    event_paths = sorted((tmp_path / "integrity" / "events").glob("*.json"))
    assert len(event_paths) == 2
    recorded_at = [
        json.loads(path.read_text())["payload"]["recorded_at"]
        for path in event_paths
    ]
    assert all(value.endswith("Z") for value in recorded_at)
    assert recorded_at == sorted(recorded_at)
    assert ledger.verify()["event_count"] == 2


def test_trial_terminal_outcome_is_append_only(tmp_path):
    ledger, _preregistration, attempted = frozen_ledger(tmp_path)
    outcome = TrialOutcome.create(
        trial_hash=attempted.trial_hash, status="failed",
        result_summary={"reason": "cost stress failed"},
        evidence_hash=EVIDENCE_HASH, recorded_at=DECIDED_AT)
    ledger.record_trial_outcome(outcome)
    assert ledger.record_trial_outcome(outcome) == outcome

    changed = TrialOutcome.create(
        trial_hash=attempted.trial_hash, status="completed",
        result_summary={"net_sharpe": 2.0}, evidence_hash=RESULT_HASH,
        recorded_at=DECIDED_AT)
    with pytest.raises(DuplicateRecordError, match="terminal outcome"):
        ledger.record_trial_outcome(changed)
    assert ledger.verify()["trial_outcome_count"] == 1


def test_holdout_open_requires_frozen_protocol(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    with pytest.raises(ProtocolNotFrozen, match="before its protocol"):
        ledger.open_holdout(
            protocol_hash="1" * 64, holdout_id="historical-oos-1",
            trial_hash="2" * 64, data_artifact_hash=DATA_HASH,
            actor="research-reviewer", opened_at=OPENED_AT)


def test_holdout_is_one_shot_and_captures_counts_from_verified_ledger(tmp_path):
    ledger, preregistration, attempted = frozen_ledger(tmp_path)
    second = trial(
        preregistration.protocol_hash, trial_id="issuance-dev-002", seed=8,
        registered_at="2026-07-13T14:01:30Z")
    ledger.register_trial(second)

    opening = ledger.open_holdout(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="historical-oos-1", trial_hash=attempted.trial_hash,
        data_artifact_hash=DATA_HASH, actor="independent-reviewer",
        opened_at=OPENED_AT)
    assert opening.payload["program_trial_count_at_open"] == 2
    assert opening.payload["protocol_trial_count_at_open"] == 2
    assert opening.payload["holdout_spec_hash"] == content_hash(
        preregistration.payload["holdouts"]["historical-oos-1"])
    assert ledger.get_holdout_opening(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="historical-oos-1") == opening

    with pytest.raises(HoldoutAlreadyOpened, match="another trial"):
        ledger.register_trial(trial(
            preregistration.protocol_hash,
            trial_id="issuance-dev-003", seed=9,
            registered_at="2026-07-13T14:02:30Z"))

    with pytest.raises(HoldoutAlreadyOpened, match="cannot be reopened"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="historical-oos-1", trial_hash=second.trial_hash,
            data_artifact_hash=DATA_HASH, actor="someone-else",
            opened_at="2026-07-13T15:00:00Z")


def test_opening_rejects_data_other_than_preregistered_digest(tmp_path):
    ledger, preregistration, attempted = frozen_ledger(tmp_path)
    with pytest.raises(ValueError, match="preregistered digest"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="historical-oos-1", trial_hash=attempted.trial_hash,
            data_artifact_hash="d" * 64, actor="independent-reviewer",
            opened_at=OPENED_AT)


def test_prospective_opening_binds_commitment_before_first_observation(tmp_path):
    ledger, preregistration, attempted, commitment, _clock = prospective_ledger(
        tmp_path)

    with pytest.raises(ValueError, match="cannot receive realized data"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
            data_artifact_hash=REALIZED_HASH,
            data_commitment_hash=commitment,
            actor="independent-reviewer", opened_at=OPENED_AT)
    with pytest.raises(ValueError, match="preregistered data commitment"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
            data_commitment_hash="9" * 64,
            actor="independent-reviewer", opened_at=OPENED_AT)
    with pytest.raises(ValueError, match="before its first observation"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
            data_commitment_hash=commitment,
            actor="independent-reviewer",
            opened_at="2026-08-01T00:00:00Z")

    opening = ledger.open_holdout(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
        data_commitment_hash=commitment,
        actor="independent-reviewer", opened_at=OPENED_AT)
    assert opening.payload["data_binding_mode"] == "prospective_commitment"
    assert opening.payload["data_commitment_hash"] == commitment
    assert "data_artifact_hash" not in opening.payload

    # An exact retry remains a lookup/no-op, but no new same-protocol attempt
    # can be appended after any of that protocol's holdouts has opened.
    assert ledger.register_trial(attempted) == attempted
    later = trial(
        preregistration.protocol_hash,
        trial_id="issuance-prospective-002",
        data_version=f"prospective-commitment:{commitment}",
        registered_at="2026-07-13T14:02:30Z",
    )
    with pytest.raises(HoldoutAlreadyOpened, match="another trial"):
        ledger.register_trial(later)
    assert ledger.trial_count == 1


def test_trusted_clock_blocks_backdated_open_and_future_decision(tmp_path):
    ledger, preregistration, attempted, commitment, clock = prospective_ledger(
        tmp_path)
    clock.set("2026-08-02T12:00:00Z")
    with pytest.raises(ValueError, match="before its first observation"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
            data_commitment_hash=commitment,
            actor="independent-reviewer", opened_at=OPENED_AT)

    second_root = tmp_path / "future-decision"
    ledger, preregistration, attempted, commitment, _clock = prospective_ledger(
        second_root)
    opening = ledger.open_holdout(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
        data_commitment_hash=commitment,
        actor="independent-reviewer", opened_at=OPENED_AT)
    with pytest.raises(ValueError, match="cannot postdate its ledger mutation"):
        ledger.record_holdout_decision(
            opening_hash=opening.opening_hash, decision="pass",
            result_summary={"primary_test_passed": True},
            result_artifact_hash=RESULT_HASH,
            realized_data_artifact_hash=REALIZED_HASH,
            actor="independent-reviewer",
            decided_at="2027-01-01T00:05:00Z")


def test_program_wide_data_binding_cannot_be_opened_twice(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    first_protocol = ledger.freeze_protocol(protocol(protocol_id="fixed-one"))
    second_protocol = ledger.freeze_protocol(protocol(protocol_id="fixed-two"))
    first_trial = ledger.register_trial(trial(
        first_protocol.protocol_hash, trial_id="fixed-one-trial"))
    second_trial = ledger.register_trial(trial(
        second_protocol.protocol_hash, trial_id="fixed-two-trial"))
    ledger.open_holdout(
        protocol_hash=first_protocol.protocol_hash,
        holdout_id="historical-oos-1", trial_hash=first_trial.trial_hash,
        data_artifact_hash=DATA_HASH, actor="reviewer", opened_at=OPENED_AT)

    with pytest.raises(HoldoutAlreadyOpened, match="already consumed"):
        ledger.open_holdout(
            protocol_hash=second_protocol.protocol_hash,
            holdout_id="historical-oos-1", trial_hash=second_trial.trial_hash,
            data_artifact_hash=DATA_HASH, actor="reviewer", opened_at=OPENED_AT)


def test_program_wide_prospective_commitment_cannot_be_opened_twice(tmp_path):
    clock = MutableClock()
    ledger = ResearchIntegrityLedger(
        tmp_path / "integrity", program_id=PROGRAM, _clock=clock)
    commitment = prospective_data_commitment_hash(prospective_specification())
    records = []
    for index in (1, 2):
        preregistration = ledger.freeze_protocol(protocol(
            protocol_id=f"prospective-{index}",
            holdouts={"prospective-oos-1": prospective_specification()},
        ))
        attempted = ledger.register_trial(trial(
            preregistration.protocol_hash,
            trial_id=f"prospective-trial-{index}",
            data_version=f"prospective-commitment:{commitment}",
        ))
        records.append((preregistration, attempted))
    first_protocol, first_trial = records[0]
    ledger.open_holdout(
        protocol_hash=first_protocol.protocol_hash,
        holdout_id="prospective-oos-1", trial_hash=first_trial.trial_hash,
        data_commitment_hash=commitment, actor="reviewer", opened_at=OPENED_AT)
    second_protocol, second_trial = records[1]
    with pytest.raises(HoldoutAlreadyOpened, match="already consumed"):
        ledger.open_holdout(
            protocol_hash=second_protocol.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=second_trial.trial_hash,
            data_commitment_hash=commitment, actor="reviewer",
            opened_at=OPENED_AT)


def test_prospective_opening_rejects_trial_bound_to_unrealized_bytes(tmp_path):
    ledger = ResearchIntegrityLedger(tmp_path / "integrity", program_id=PROGRAM)
    preregistration = ledger.freeze_protocol(protocol(
        protocol_id="issuance-prospective",
        holdouts={"prospective-oos-1": prospective_specification()},
    ))
    attempted = ledger.register_trial(trial(
        preregistration.protocol_hash,
        trial_id="issuance-prospective-realized-data",
        data_version=f"pit-sha256:{REALIZED_HASH}",
    ))
    commitment = prospective_data_commitment_hash(prospective_specification())

    with pytest.raises(ValueError, match="not bound to its data commitment"):
        ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
            data_commitment_hash=commitment,
            actor="independent-reviewer", opened_at=OPENED_AT)


def test_prospective_trial_cannot_record_result_before_opening(tmp_path):
    ledger, preregistration, attempted, commitment, _clock = prospective_ledger(
        tmp_path)
    with pytest.raises(ValueError, match="requires exactly one opening"):
        ledger.record_trial_outcome(TrialOutcome.create(
            trial_hash=attempted.trial_hash,
            status="completed",
            result_summary={"primary_test_passed": True},
            evidence_hash=RESULT_HASH,
            recorded_at="2026-07-13T14:01:30Z",
        ))

    opening = ledger.open_holdout(
            protocol_hash=preregistration.protocol_hash,
            holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
            data_commitment_hash=commitment,
            actor="independent-reviewer", opened_at=OPENED_AT)
    assert opening.payload["trial_hash"] == attempted.trial_hash


def test_prospective_decision_binds_realized_snapshot_only_at_finalization(
        tmp_path):
    ledger, preregistration, attempted, commitment, clock = prospective_ledger(
        tmp_path)
    opening = ledger.open_holdout(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
        data_commitment_hash=commitment,
        actor="independent-reviewer", opened_at=OPENED_AT)

    with pytest.raises(ValueError, match="realized data artifact"):
        ledger.record_holdout_decision(
            opening_hash=opening.opening_hash, decision="pass",
            result_summary={"primary_test_passed": True},
            result_artifact_hash=RESULT_HASH, actor="independent-reviewer",
            decided_at="2027-01-01T00:00:00Z")
    clock.set("2027-01-01T00:10:00Z")
    with pytest.raises(ValueError, match="predate holdout end"):
        ledger.record_holdout_decision(
            opening_hash=opening.opening_hash, decision="fail",
            result_summary={"primary_test_passed": False},
            result_artifact_hash=RESULT_HASH, actor="independent-reviewer",
            realized_data_artifact_hash=REALIZED_HASH,
            decided_at="2026-12-30T21:00:00Z")

    summary = {"primary_test_passed": True}
    with pytest.raises(ValueError, match="completed trial outcome"):
        ledger.record_holdout_decision(
            opening_hash=opening.opening_hash, decision="pass",
            result_summary=summary,
            result_artifact_hash=RESULT_HASH, actor="independent-reviewer",
            realized_data_artifact_hash=REALIZED_HASH,
            decided_at="2027-01-01T00:05:00Z")
    ledger.record_trial_outcome(TrialOutcome.create(
        trial_hash=attempted.trial_hash,
        status="completed",
        result_summary=summary,
        evidence_hash=RESULT_HASH,
        recorded_at="2027-01-01T00:05:00Z",
    ))
    clock.set("2027-01-01T00:12:00Z")

    with pytest.raises(ValueError, match="evidence differs"):
        ledger.record_holdout_decision(
            opening_hash=opening.opening_hash, decision="pass",
            result_summary=summary,
            result_artifact_hash=EVIDENCE_HASH,
            actor="independent-reviewer",
            realized_data_artifact_hash=REALIZED_HASH,
            decided_at="2027-01-01T00:11:00Z")

    decision = ledger.record_holdout_decision(
        opening_hash=opening.opening_hash, decision="pass",
        result_summary=summary,
        result_artifact_hash=RESULT_HASH, actor="independent-reviewer",
        realized_data_artifact_hash=REALIZED_HASH,
        decided_at="2027-01-01T00:11:00Z")
    assert decision.payload["data_binding_mode"] == "prospective_commitment"
    assert decision.payload["realized_data_artifact_hash"] == REALIZED_HASH
    assert ledger.verify()["holdout_decision_count"] == 1


def test_completed_prospective_outcome_cannot_be_recorded_early(tmp_path):
    ledger, preregistration, attempted, commitment, _clock = prospective_ledger(
        tmp_path)
    ledger.open_holdout(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="prospective-oos-1", trial_hash=attempted.trial_hash,
        data_commitment_hash=commitment,
        actor="independent-reviewer", opened_at=OPENED_AT)

    with pytest.raises(ValueError, match="cannot predate holdout end"):
        ledger.record_trial_outcome(TrialOutcome.create(
            trial_hash=attempted.trial_hash,
            status="completed",
            result_summary={"primary_test_passed": True},
            evidence_hash=RESULT_HASH,
            recorded_at="2026-07-13T19:00:00Z",
        ))


def test_holdout_decision_and_result_are_recorded_exactly_once(tmp_path):
    ledger, preregistration, attempted = frozen_ledger(tmp_path)
    opening = ledger.open_holdout(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="historical-oos-1", trial_hash=attempted.trial_hash,
        data_artifact_hash=DATA_HASH, actor="independent-reviewer",
        opened_at=OPENED_AT)
    decision = ledger.record_holdout_decision(
        opening_hash=opening.opening_hash, decision="fail",
        result_summary={"net_sharpe": 0.22, "cost_stress_passed": False},
        result_artifact_hash=RESULT_HASH, actor="independent-reviewer",
        decided_at=DECIDED_AT)

    assert decision.payload["decision"] == "fail"
    assert ledger.get_holdout_decision(
        protocol_hash=preregistration.protocol_hash,
        holdout_id="historical-oos-1") == decision
    assert ledger.verify() == {
        "program_id": PROGRAM,
        "event_count": 4,
        "protocol_count": 1,
        "trial_count": 1,
        "trial_outcome_count": 0,
        "holdout_opening_count": 1,
        "holdout_decision_count": 1,
        "head_hash": decision_head(ledger),
    }

    with pytest.raises(HoldoutAlreadyDecided, match="permanent decision"):
        ledger.record_holdout_decision(
            opening_hash=opening.opening_hash, decision="pass",
            result_summary={"net_sharpe": 3.0},
            result_artifact_hash="d" * 64, actor="different-reviewer",
            decided_at="2026-07-13T15:00:00Z")

    with pytest.raises(HoldoutAlreadyDecided, match="after its holdout decision"):
        ledger.record_trial_outcome(TrialOutcome.create(
            trial_hash=attempted.trial_hash,
            status="completed",
            result_summary={"net_sharpe": 0.22},
            evidence_hash=RESULT_HASH,
            recorded_at="2026-07-13T15:01:00Z",
        ))


def decision_head(ledger: ResearchIntegrityLedger) -> str:
    head = ledger.head_hash
    assert head is not None
    return head


def test_unknown_opening_cannot_receive_a_decision(tmp_path):
    ledger, _preregistration, _attempted = frozen_ledger(tmp_path)
    with pytest.raises(UnknownRecordError, match="unknown holdout"):
        ledger.record_holdout_decision(
            opening_hash="9" * 64, decision="invalid",
            result_summary={"reason": "data corrupt"},
            result_artifact_hash=RESULT_HASH, actor="reviewer",
            decided_at=DECIDED_AT)


def test_every_read_detects_record_tampering_and_external_head_mismatch(tmp_path):
    ledger, _preregistration, attempted = frozen_ledger(tmp_path)
    anchored_head = ledger.head_hash
    assert anchored_head is not None
    assert ledger.verify(expected_head_hash=anchored_head)["trial_count"] == 1

    trial_path = (
        tmp_path / "integrity" / "records" / "trials"
        / f"{attempted.trial_hash}.json")
    document = json.loads(trial_path.read_text())
    document["payload"]["seed"] = 999
    trial_path.chmod(0o644)
    trial_path.write_text(canonical_json(document) + "\n")
    with pytest.raises(IntegrityViolation, match="hash mismatch"):
        _ = ledger.trial_count

    # Restore the exact record, append a new event, and demonstrate that an
    # externally anchored old head catches otherwise-valid tail growth.
    trial_path.write_text(attempted.to_json())
    TrialRegistration.from_json(trial_path.read_text())
    second = trial(
        attempted.payload["protocol_hash"], trial_id="issuance-dev-002",
        registered_at="2026-07-13T14:05:00Z")
    ledger.register_trial(second)
    with pytest.raises(IntegrityViolation, match="external anchor"):
        ledger.verify(expected_head_hash=anchored_head)
