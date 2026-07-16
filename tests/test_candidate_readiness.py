"""Hermetic tests for the read-only candidate readiness gate."""

from __future__ import annotations

from datetime import date
import hashlib
import json

import pytest

from analysis.candidate_readiness import (
    ReadinessPackageError,
    assess_candidate_readiness,
    validate_readiness_package,
)
from scripts.candidate_readiness import main


GIT_SHA = "1" * 40
TABLES = ["actions", "daily", "sep", "sf1", "tickers"]
SNAPSHOT_HASH = hashlib.sha256(
    "".join(f"{table}:{'3' * 64}:100\n" for table in TABLES).encode("utf-8")
).hexdigest()


def _artifact(root, name: str, payload: dict | None = None) -> tuple[str, str]:
    path = root / name
    path.write_text(
        json.dumps(payload if payload is not None else {"artifact": name}),
        encoding="utf-8",
    )
    return name, hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(root) -> dict[str, tuple[str, str]]:
    return {
        key: _artifact(root, f"{key}.json")
        for key in (
            "access",
            "acquisition",
            "borrow",
            "contamination",
            "costs",
            "holdout_data",
            "independent_data",
            "inventory",
        )
    }


def _package(root, *, prospective: bool = False) -> dict:
    evidence = _evidence(root)
    specification = _artifact(
        root,
        "specification.json",
        {
            "candidate_id": "issuance-midcap-ls-v1",
            "candidate_family": "net-share-issuance",
            "status": "final",
        },
    )
    holdout_start = "2027-01-01" if prospective else "2025-01-01"
    holdout_end = "2027-12-31" if prospective else "2026-06-30"
    data_file, data_hash = evidence["holdout_data"]
    acquisition_file, acquisition_hash = evidence["acquisition"]
    package = {
        "schema_version": 1,
        "candidate_id": "issuance-midcap-ls-v1",
        "candidate_family": "net-share-issuance",
        "candidate_author": "researcher@example.com",
        "development_source_id": "sharadar-development",
        "code_sha": GIT_SHA,
        "warehouse_snapshot_hash": SNAPSHOT_HASH,
        "required_tables": TABLES,
        "candidate_specification": {
            "artifact_file": specification[0],
            "artifact_sha256": specification[1],
            "status": "final",
        },
        "development_window": {"start": "2015-01-01", "end": "2024-12-31"},
        "historical_inventory": {
            "manifest_file": evidence["inventory"][0],
            "manifest_sha256": evidence["inventory"][1],
            "known_attempt_ids": ["issuance-dev-001", "issuance-dev-002"],
            "complete": True,
            "includes_failed_and_abandoned": True,
            "attested_by": "researcher@example.com",
            "attested_at": "2026-07-13T14:00:00Z",
        },
        "independent_review": {
            "report_file": None,
            "report_sha256": None,
            "reviewer": "reviewer@example.com",
            "independent": True,
            "approved": True,
            "reviewed_at": "2026-07-13T15:00:00Z",
            "scopes": [
                "data",
                "economics",
                "implementation",
                "statistics",
                "trial_inventory",
            ],
            "reviewed_subject_sha256": None,
            "reviewed_candidate_specification_sha256": None,
        },
        "independent_data": {
            "manifest_file": evidence["independent_data"][0],
            "manifest_sha256": evidence["independent_data"][1],
            "source_id": "sec-filings-independent",
            "independent_of_development": True,
            "point_in_time": True,
            "coverage_start": "2015-01-01",
            "coverage_end": "2026-12-31",
            "reviewed_by": "data-reviewer@example.com",
        },
        "borrow_evidence": {
            "shorting_required": True,
            "status": "complete",
            "manifest_file": evidence["borrow"][0],
            "manifest_sha256": evidence["borrow"][1],
            "locates_complete": True,
            "fees_complete": True,
            "financing_complete": True,
            "coverage_start": "2015-01-01",
            "coverage_end": "2026-12-31",
            "non_applicability_reason": None,
        },
        "cost_evidence": {
            "manifest_file": evidence["costs"][0],
            "manifest_sha256": evidence["costs"][1],
            "commissions_complete": True,
            "spreads_complete": True,
            "slippage_complete": True,
            "impact_complete": True,
            "financing_complete": True,
            "capacity_complete": True,
            "coverage_start": "2015-01-01",
            "coverage_end": "2026-12-31",
            "capital_levels": [100000, 250000, 500000],
            "calibrated_at": "2026-07-13T16:00:00Z",
        },
        "holdout_plan": {
            "mode": ("prospective_acquisition" if prospective else "existing_uncontaminated"),
            "start": holdout_start,
            "end": holdout_end,
            "source_id": "independent-holdout-source",
            "data_manifest_file": None if prospective else data_file,
            "data_manifest_sha256": None if prospective else data_hash,
            "acquisition_plan_file": acquisition_file if prospective else None,
            "acquisition_plan_sha256": acquisition_hash if prospective else None,
            "access_log_file": evidence["access"][0],
            "access_log_sha256": evidence["access"][1],
            "contamination_review_file": evidence["contamination"][0],
            "contamination_review_sha256": evidence["contamination"][1],
            "reviewed_by": "holdout-reviewer@example.com",
            "uncontaminated": True,
            "outcome_accessed": False,
            "sealed": True,
            "sealed_location": "object-lock://research/issuance-v1",
        },
    }
    review_subject = dict(package)
    review_subject.pop("independent_review")
    subject_hash = hashlib.sha256(
        json.dumps(
            review_subject,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    review_file, review_hash = _artifact(
        root,
        "review.json",
        {
            "reviewed_subject_sha256": subject_hash,
            "reviewed_candidate_specification_sha256": specification[1],
        },
    )
    package["independent_review"].update(
        {
            "report_file": review_file,
            "report_sha256": review_hash,
            "reviewed_subject_sha256": subject_hash,
            "reviewed_candidate_specification_sha256": specification[1],
        }
    )
    return package


def _git(*, clean: bool = True) -> dict:
    return {
        "available": True,
        "repo_root": "/repo",
        "sha": GIT_SHA,
        "clean": clean,
        "dirty_paths": [] if clean else [" M candidate.py"],
        "error": None,
    }


def _snapshot(*, complete: bool = True, missing: tuple[str, ...] = ()) -> dict:
    present = [table for table in TABLES if table not in missing]
    return {
        "version": SNAPSHOT_HASH,
        "complete": complete,
        "quality_flags": [f"missing_table:{table}" for table in missing],
        "tables": [{"table": table, "sha256": "3" * 64, "bytes": 100} for table in present],
    }


def _assess(root, package, *, git=None, snapshot=None, as_of=None):
    return assess_candidate_readiness(
        package,
        warehouse_dir=root / "warehouse",
        repo_dir=root,
        evidence_root=root,
        git_state=git or _git(),
        warehouse_snapshot=snapshot or _snapshot(),
        as_of=as_of,
    )


def test_complete_existing_holdout_package_is_ready_to_run(tmp_path):
    report = _assess(tmp_path, _package(tmp_path))

    assert report.ready_to_freeze is True
    assert report.ready_to_run_holdout is True
    assert report.blockers == ()
    payload = report.to_mapping()
    assert payload["ok"] is True
    assert payload["blocker_count"] == 0
    assert len(payload["package_hash"]) == 64


def test_complete_prospective_plan_is_ready_to_freeze_but_not_run(tmp_path):
    report = _assess(
        tmp_path,
        _package(tmp_path, prospective=True),
        as_of=date(2026, 7, 13),
    )

    assert report.ready_to_freeze is True
    assert report.ready_to_run_holdout is False
    assert report.blockers == ()


def test_honest_pending_draft_validates_but_never_becomes_ready(tmp_path):
    package = _package(tmp_path, prospective=True)
    specification_file, specification_hash = _artifact(
        tmp_path,
        "specification.json",
        {
            "candidate_id": package["candidate_id"],
            "candidate_family": package["candidate_family"],
            "status": "draft",
            "disposition": "stopped_development",
        },
    )
    package["candidate_specification"] = {
        "artifact_file": specification_file,
        "artifact_sha256": specification_hash,
        "status": "draft",
    }
    package["historical_inventory"].update(
        {
            "complete": False,
            "includes_failed_and_abandoned": False,
            "attested_by": None,
            "attested_at": None,
        }
    )
    package["independent_review"] = {
        "report_file": None,
        "report_sha256": None,
        "reviewer": None,
        "independent": False,
        "approved": False,
        "reviewed_at": None,
        "scopes": [],
        "reviewed_subject_sha256": None,
        "reviewed_candidate_specification_sha256": None,
    }
    package["independent_data"] = {
        "manifest_file": None,
        "manifest_sha256": None,
        "source_id": None,
        "independent_of_development": False,
        "point_in_time": False,
        "coverage_start": None,
        "coverage_end": None,
        "reviewed_by": None,
    }
    package["borrow_evidence"] = {
        "shorting_required": True,
        "status": "pending",
        "manifest_file": None,
        "manifest_sha256": None,
        "locates_complete": False,
        "fees_complete": False,
        "financing_complete": False,
        "coverage_start": None,
        "coverage_end": None,
        "non_applicability_reason": None,
    }
    package["cost_evidence"] = {
        "manifest_file": None,
        "manifest_sha256": None,
        "commissions_complete": False,
        "spreads_complete": False,
        "slippage_complete": False,
        "impact_complete": False,
        "financing_complete": False,
        "capacity_complete": False,
        "coverage_start": None,
        "coverage_end": None,
        "capital_levels": [],
        "calibrated_at": None,
    }
    package["holdout_plan"] = {
        "mode": "prospective_acquisition",
        "start": None,
        "end": None,
        "source_id": None,
        "data_manifest_file": None,
        "data_manifest_sha256": None,
        "acquisition_plan_file": None,
        "acquisition_plan_sha256": None,
        "access_log_file": None,
        "access_log_sha256": None,
        "contamination_review_file": None,
        "contamination_review_sha256": None,
        "reviewed_by": None,
        "uncontaminated": False,
        "outcome_accessed": False,
        "sealed": False,
        "sealed_location": None,
    }

    validate_readiness_package(package)
    report = _assess(tmp_path, package)
    codes = {blocker.code for blocker in report.blockers}

    assert report.ready_to_freeze is False
    assert report.ready_to_run_holdout is False
    assert {
        "borrow_evidence_incomplete",
        "candidate_specification_not_final",
        "cost_calibration_missing",
        "holdout_dates_missing",
        "review_not_approved",
        "trial_inventory_incomplete",
        "trial_inventory_unattested",
    } <= codes


def test_review_artifact_must_bind_subject_and_specification(tmp_path):
    package = _package(tmp_path)
    report_file, report_hash = _artifact(
        tmp_path,
        "review.json",
        {
            "reviewed_subject_sha256": "0" * 64,
            "reviewed_candidate_specification_sha256": "0" * 64,
        },
    )
    package["independent_review"]["report_file"] = report_file
    package["independent_review"]["report_sha256"] = report_hash

    report = _assess(tmp_path, package)

    assert report.ready_to_freeze is False
    assert any(blocker.code == "review_evidence_binding_mismatch" for blocker in report.blockers)


def test_gate_reports_all_material_evidence_blockers(tmp_path):
    package = _package(tmp_path)
    package["historical_inventory"]["complete"] = False
    package["historical_inventory"]["includes_failed_and_abandoned"] = False
    package["independent_review"]["reviewer"] = package["candidate_author"]
    package["independent_review"]["independent"] = False
    package["independent_review"]["approved"] = False
    package["independent_review"]["scopes"] = ["data"]
    package["independent_data"]["source_id"] = package["development_source_id"]
    package["independent_data"]["independent_of_development"] = False
    package["independent_data"]["point_in_time"] = False
    package["independent_data"]["reviewed_by"] = package["candidate_author"]
    package["borrow_evidence"].update(
        {
            "status": "not_applicable",
            "manifest_file": None,
            "manifest_sha256": None,
            "locates_complete": False,
            "fees_complete": False,
            "financing_complete": False,
            "coverage_start": None,
            "coverage_end": None,
            "non_applicability_reason": "none claimed",
        }
    )
    package["cost_evidence"]["spreads_complete"] = False
    package["holdout_plan"].update(
        {
            "start": "2024-01-01",
            "uncontaminated": False,
            "outcome_accessed": True,
            "sealed": False,
            "reviewed_by": package["candidate_author"],
        }
    )

    report = _assess(
        tmp_path,
        package,
        git=_git(clean=False),
        snapshot=_snapshot(complete=False, missing=("actions",)),
    )
    codes = {blocker.code for blocker in report.blockers}

    assert report.ready_to_freeze is False
    assert {
        "borrow_evidence_incomplete",
        "cost_spreads_complete",
        "data_not_independent",
        "git_dirty",
        "holdout_contaminated",
        "holdout_not_sealed",
        "holdout_overlaps_development",
        "independent_data_not_pit",
        "review_not_approved",
        "review_not_independent",
        "review_scope_incomplete",
        "trial_inventory_incomplete",
        "trial_inventory_omits_failures",
        "warehouse_snapshot_incomplete",
        "warehouse_table_missing",
    } <= codes


def test_artifact_hash_mismatch_is_a_blocker(tmp_path):
    package = _package(tmp_path)
    package["cost_evidence"]["manifest_sha256"] = "f" * 64

    report = _assess(tmp_path, package)

    assert report.ready_to_freeze is False
    assert any(
        blocker.code == "evidence_hash_mismatch" and blocker.path == "cost_evidence.manifest"
        for blocker in report.blockers
    )


def test_package_schema_rejects_missing_and_unknown_fields(tmp_path):
    package = _package(tmp_path)
    package["surprise"] = True
    with pytest.raises(ReadinessPackageError, match="extra=.*surprise"):
        validate_readiness_package(package)

    package = _package(tmp_path)
    package["cost_evidence"].pop("impact_complete")
    with pytest.raises(ReadinessPackageError, match="impact_complete"):
        validate_readiness_package(package)


def test_cli_returns_nonzero_json_with_blockers(tmp_path, capsys, monkeypatch):
    package = _package(tmp_path)
    package["historical_inventory"]["complete"] = False
    package_path = tmp_path / "readiness.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    monkeypatch.setattr("analysis.candidate_readiness._git_state", lambda _: _git())
    monkeypatch.setattr(
        "data.pit_warehouse.PitWarehouse.snapshot_version",
        lambda _self, _tables: _snapshot(),
    )

    code = main(
        [
            "--package",
            str(package_path),
            "--warehouse-dir",
            str(tmp_path / "warehouse"),
            "--repo-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 1
    assert captured.err == ""
    assert payload["ok"] is False
    assert "trial_inventory_incomplete" in {blocker["code"] for blocker in payload["blockers"]}
    assert not (tmp_path / "research-integrity").exists()


def test_cli_rejects_duplicate_json_keys(tmp_path, capsys):
    package_path = tmp_path / "readiness.json"
    package_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")

    code = main(
        [
            "--package",
            str(package_path),
            "--warehouse-dir",
            str(tmp_path / "warehouse"),
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert "duplicate key" in json.loads(captured.err)["error"]
