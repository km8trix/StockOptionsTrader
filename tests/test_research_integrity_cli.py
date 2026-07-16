"""Operator-level tests for the research-integrity CLI without subprocesses."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from analysis.research_integrity import (
    ResearchIntegrityLedger,
    prospective_data_commitment_hash,
)
from scripts.research_integrity import main


PROGRAM = "stock-options-trader"
DATA_HASH = "a" * 64
EVIDENCE_HASH = "b" * 64
RESULT_HASH = "c" * 64
REALIZED_HASH = "8" * 64


class MutableClock:
    def __init__(self, value: str) -> None:
        self.set(value)

    def set(self, value: str) -> None:
        self.value = datetime.fromisoformat(
            value.replace("Z", "+00:00")).astimezone(timezone.utc)

    def __call__(self) -> datetime:
        return self.value


def _write_json(path, value) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def _invoke(tmp_path, capsys, *arguments):
    code = main(
        [
            "--ledger-root",
            str(tmp_path / "integrity"),
            "--program-id",
            PROGRAM,
            *arguments,
        ]
    )
    captured = capsys.readouterr()
    output = captured.out if code == 0 else captured.err
    return code, json.loads(output)


def _protocol_specification():
    return {
        "program_id": PROGRAM,
        "protocol_id": "issuance-midcap-ls",
        "version": "1",
        "objective": "Test whether net issuance has executable net alpha.",
        "hypotheses": [
            {
                "hypothesis_id": "issuance-spread",
                "claim": "Low issuance outperforms high issuance net of costs.",
            }
        ],
        "candidate_specifications": [
            {
                "candidate_id": "issuance-midcap-ls-v1",
                "formation_days": 252,
                "holding_days": 63,
            }
        ],
        "data_plan": {"point_in_time": True, "development_end": "2024-12-31"},
        "evaluation_plan": {"primary_test": "stationary-block-bootstrap"},
        "decision_rules": {"minimum_net_sharpe": 0.75},
        "holdouts": {
            "historical-oos-1": {
                "start": "2025-01-01",
                "end": "2026-06-30",
                "data_artifact_hash": DATA_HASH,
            }
        },
        "code_version": "git:0123456789abcdef",
        "frozen_at": "2026-07-13T14:00:00Z",
    }


def _prospective_protocol_specification():
    specification = _protocol_specification()
    specification["protocol_id"] = "issuance-prospective"
    specification["holdouts"] = {
        "prospective-oos-1": {
            "mode": "prospective",
            "start": "2026-08-01",
            "end": "2026-12-31",
            "source_manifest_hash": "d" * 64,
            "query_hash": "e" * 64,
            "acquisition_plan_hash": "f" * 64,
            "append_only": True,
            "sealed": True,
        }
    }
    return specification


def test_cli_runs_complete_immutable_workflow_and_reports_derived_counts(
    tmp_path, capsys
):
    protocol_path = _write_json(tmp_path / "protocol.json", _protocol_specification())
    code, frozen = _invoke(
        tmp_path, capsys, "freeze-protocol", "--json", protocol_path
    )
    assert code == 0
    assert frozen["ok"] is True
    protocol_hash = frozen["protocol_hash"]
    # The explicit content timestamp makes an exact retry a no-op with the
    # same record and ledger head, rather than a second wall-clock-derived hash.
    code, retried_frozen = _invoke(
        tmp_path, capsys, "freeze-protocol", "--json", protocol_path
    )
    assert code == 0
    assert retried_frozen["protocol_hash"] == protocol_hash
    assert retried_frozen["head_hash"] == frozen["head_hash"]

    trial_path = _write_json(
        tmp_path / "trial.json",
        {
            "protocol_hash": protocol_hash,
            "trial_id": "issuance-dev-001",
            "candidate_id": "issuance-midcap-ls-v1",
            "family": "net-share-issuance",
            "inputs": {"formation_days": 252, "holding_days": 63},
            "data_version": "warehouse:abc",
            "code_version": "git:0123456789abcdef",
            "seed": 7,
            "registered_at": "2026-07-13T14:01:00Z",
        },
    )
    code, registered = _invoke(
        tmp_path, capsys, "register-trial", "--json", trial_path
    )
    assert code == 0
    assert registered["program_trial_count"] == 1
    assert registered["protocol_trial_count"] == 1
    trial_hash = registered["trial_hash"]
    code, retried_trial = _invoke(
        tmp_path, capsys, "register-trial", "--json", trial_path
    )
    assert code == 0
    assert retried_trial["trial_hash"] == trial_hash
    assert retried_trial["head_hash"] == registered["head_hash"]
    assert retried_trial["program_trial_count"] == 1

    outcome_path = _write_json(
        tmp_path / "outcome.json",
        {
            "trial_hash": trial_hash,
            "status": "completed",
            "result_summary": {"development_net_sharpe": 0.91},
            "evidence_hash": EVIDENCE_HASH,
            "recorded_at": "2026-07-13T14:02:00Z",
        },
    )
    code, outcome = _invoke(
        tmp_path, capsys, "record-trial-outcome", "--json", outcome_path
    )
    assert code == 0
    assert outcome["trial_hash"] == trial_hash
    code, retried_outcome = _invoke(
        tmp_path, capsys, "record-trial-outcome", "--json", outcome_path
    )
    assert code == 0
    assert retried_outcome["outcome_hash"] == outcome["outcome_hash"]
    assert retried_outcome["head_hash"] == outcome["head_hash"]

    code, opened = _invoke(
        tmp_path,
        capsys,
        "open-holdout",
        "--protocol-hash",
        protocol_hash,
        "--holdout-id",
        "historical-oos-1",
        "--trial-hash",
        trial_hash,
        "--data-artifact-hash",
        DATA_HASH,
        "--actor",
        "independent-reviewer",
        "--opened-at",
        "2026-07-13T14:03:00Z",
    )
    assert code == 0
    assert opened["program_trial_count_at_open"] == 1
    assert opened["protocol_trial_count_at_open"] == 1

    summary_path = _write_json(
        tmp_path / "summary.json", {"net_sharpe": 0.83, "primary_test_passed": True}
    )
    code, decided = _invoke(
        tmp_path,
        capsys,
        "decide-holdout",
        "--opening-hash",
        opened["opening_hash"],
        "--decision",
        "pass",
        "--result-summary-json",
        summary_path,
        "--result-artifact-hash",
        RESULT_HASH,
        "--actor",
        "independent-reviewer",
        "--decided-at",
        "2026-07-13T14:04:00Z",
    )
    assert code == 0
    assert decided["decision"] == "pass"

    code, verified = _invoke(
        tmp_path,
        capsys,
        "verify",
        "--expected-head-hash",
        decided["head_hash"],
    )
    assert code == 0
    assert verified == {
        "event_count": 5,
        "head_hash": decided["head_hash"],
        "holdout_decision_count": 1,
        "holdout_opening_count": 1,
        "ok": True,
        "operation": "verify",
        "program_id": PROGRAM,
        "protocol_count": 1,
        "trial_count": 1,
        "trial_outcome_count": 1,
    }


def test_cli_supports_prospective_commitment_then_realized_finalization(
        tmp_path, capsys, monkeypatch):
    clock = MutableClock("2026-07-13T20:00:00Z")
    monkeypatch.setattr(
        "scripts.research_integrity.ResearchIntegrityLedger",
        lambda root, *, program_id: ResearchIntegrityLedger(
            root, program_id=program_id, _clock=clock),
    )
    protocol_specification = _prospective_protocol_specification()
    protocol_path = _write_json(
        tmp_path / "prospective-protocol.json", protocol_specification)
    code, frozen = _invoke(
        tmp_path, capsys, "freeze-protocol", "--json", protocol_path)
    assert code == 0

    holdout = protocol_specification["holdouts"]["prospective-oos-1"]
    commitment = prospective_data_commitment_hash(holdout)
    assert frozen["prospective_data_commitments"] == {
        "prospective-oos-1": commitment}
    trial_path = _write_json(tmp_path / "prospective-trial.json", {
        "protocol_hash": frozen["protocol_hash"],
        "trial_id": "issuance-prospective-001",
        "candidate_id": "issuance-midcap-ls-v1",
        "family": "net-share-issuance",
        "inputs": {"formation_days": 252, "holding_days": 63},
        "data_version": f"prospective-commitment:{commitment}",
        "code_version": "git:0123456789abcdef",
        "seed": 7,
        "registered_at": "2026-07-13T14:01:00Z",
    })
    code, registered = _invoke(
        tmp_path, capsys, "register-trial", "--json", trial_path)
    assert code == 0

    code, opened = _invoke(
        tmp_path, capsys,
        "open-holdout",
        "--protocol-hash", frozen["protocol_hash"],
        "--holdout-id", "prospective-oos-1",
        "--trial-hash", registered["trial_hash"],
        "--data-commitment-hash", commitment,
        "--actor", "independent-reviewer",
        "--opened-at", "2026-07-13T14:02:00Z",
    )
    assert code == 0
    assert opened["data_binding_mode"] == "prospective_commitment"
    assert opened["data_artifact_hash"] is None
    assert opened["data_commitment_hash"] == commitment

    summary_path = _write_json(
        tmp_path / "prospective-summary.json",
        {"primary_test_passed": True})
    clock.set("2027-01-01T00:10:00Z")
    outcome_path = _write_json(tmp_path / "prospective-outcome.json", {
        "trial_hash": registered["trial_hash"],
        "status": "completed",
        "result_summary": {"primary_test_passed": True},
        "evidence_hash": RESULT_HASH,
        "recorded_at": "2027-01-01T00:05:00Z",
    })
    code, outcome = _invoke(
        tmp_path, capsys, "record-trial-outcome", "--json", outcome_path)
    assert code == 0
    assert outcome["status"] == "completed"
    clock.set("2027-01-01T00:12:00Z")
    code, decided = _invoke(
        tmp_path, capsys,
        "decide-holdout",
        "--opening-hash", opened["opening_hash"],
        "--decision", "pass",
        "--result-summary-json", summary_path,
        "--result-artifact-hash", RESULT_HASH,
        "--realized-data-artifact-hash", REALIZED_HASH,
        "--actor", "independent-reviewer",
        "--decided-at", "2027-01-01T00:11:00Z",
    )
    assert code == 0
    assert decided["realized_data_artifact_hash"] == REALIZED_HASH


def test_cli_rejects_duplicate_keys_before_creating_a_ledger(tmp_path, capsys):
    invalid = tmp_path / "protocol.json"
    invalid.write_text('{"program_id":"one","program_id":"two"}', encoding="utf-8")

    code, error = _invoke(
        tmp_path, capsys, "freeze-protocol", "--json", str(invalid)
    )

    assert code == 2
    assert error["ok"] is False
    assert "duplicate key" in error["error"]
    assert not (tmp_path / "integrity").exists()


def test_cli_rejects_non_object_and_non_finite_json(tmp_path, capsys):
    for name, contents, message in (
        ("array.json", "[]", "must be a JSON object"),
        ("nan.json", '{"result":NaN}', "invalid JSON number"),
    ):
        path = tmp_path / name
        path.write_text(contents, encoding="utf-8")
        code, error = _invoke(
            tmp_path, capsys, "freeze-protocol", "--json", str(path)
        )
        assert code == 2
        assert message in error["error"]


def test_cli_requires_explicit_content_timestamps(tmp_path, capsys):
    protocol = _protocol_specification()
    protocol.pop("frozen_at")
    protocol_path = _write_json(tmp_path / "protocol.json", protocol)
    trial_path = _write_json(
        tmp_path / "trial.json",
        {
            "protocol_hash": "a" * 64,
            "trial_id": "trial-1",
            "candidate_id": "candidate-1",
            "family": "family-1",
            "inputs": {},
            "data_version": "warehouse:abc",
            "code_version": "git:abc",
            "seed": 1,
        },
    )
    outcome_path = _write_json(
        tmp_path / "outcome.json",
        {
            "trial_hash": "b" * 64,
            "status": "failed",
            "result_summary": {},
            "evidence_hash": "c" * 64,
        },
    )

    for command, path, field_name in (
        ("freeze-protocol", protocol_path, "frozen_at"),
        ("register-trial", trial_path, "registered_at"),
        ("record-trial-outcome", outcome_path, "recorded_at"),
    ):
        code, error = _invoke(tmp_path, capsys, command, "--json", path)
        assert code == 2
        assert field_name in error["error"]
        assert "retry wall clock" in error["error"]


def test_cli_requires_explicit_holdout_timestamps(tmp_path, capsys):
    code, error = _invoke(
        tmp_path,
        capsys,
        "open-holdout",
        "--protocol-hash",
        "a" * 64,
        "--holdout-id",
        "holdout-1",
        "--trial-hash",
        "b" * 64,
        "--data-artifact-hash",
        "c" * 64,
        "--actor",
        "reviewer",
    )
    assert code == 2
    assert "--opened-at" in error["error"]

    code, error = _invoke(
        tmp_path,
        capsys,
        "decide-holdout",
        "--opening-hash",
        "a" * 64,
        "--decision",
        "fail",
        "--result-summary-json",
        _write_json(tmp_path / "summary.json", {}),
        "--result-artifact-hash",
        "b" * 64,
        "--actor",
        "reviewer",
    )
    assert code == 2
    assert "--decided-at" in error["error"]


def test_cli_has_no_manual_trial_count_override(tmp_path, capsys):
    code, error = _invoke(
        tmp_path,
        capsys,
        "open-holdout",
        "--protocol-hash",
        "a" * 64,
        "--holdout-id",
        "historical-oos-1",
        "--trial-hash",
        "b" * 64,
        "--data-artifact-hash",
        "c" * 64,
        "--actor",
        "reviewer",
        "--opened-at",
        "2026-07-13T14:03:00Z",
        "--trial-count",
        "1",
    )

    assert code == 2
    assert error["error_type"] == "CliUsageError"
    assert "--trial-count" in error["error"]


def test_cli_returns_json_error_when_external_head_anchor_does_not_match(
    tmp_path, capsys
):
    code, error = _invoke(
        tmp_path,
        capsys,
        "verify",
        "--expected-head-hash",
        "d" * 64,
    )

    assert code == 2
    assert error["ok"] is False
    assert error["error_type"] == "IntegrityViolation"
    assert "external anchor" in error["error"]
