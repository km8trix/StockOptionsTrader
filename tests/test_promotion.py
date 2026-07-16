from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from analysis.promotion import (
    ArtifactIntegrityError,
    ArtifactStore,
    PaperValidationArtifact,
    PaperValidationPolicy,
    PaperValidationStore,
    PromotionArtifact,
    PromotionLevel,
    PromotionNotApproved,
    PromotionPolicy,
    PromotionRegistry,
    PromotionResults,
    canonical_json,
)
from analysis.research_report_store import (
    ResearchReportArtifact,
    ResearchReportStore,
)
from desks.foundation import FoundationDesk
from desks.deployment_config import FoundationDeploymentConfig
from desks.registry import create_deployed_desk
from tests.paper_evidence_helpers import (
    authoritative_paper_evidence as _authoritative_paper_evidence,
)
from tests.research_evidence_helpers import (
    fixture_engine_parameters,
    fixture_integrity_evidence,
    fixture_regimes,
    fixture_results,
    persist_terminal_integrity,
    positive_foundation_report,
)
from tests.replication_evidence_helpers import (
    persist_passing_replication,
    persist_replication,
    replication_evidence_for,
)


SNAPSHOT_SHA = "eb12241b17158c2ac21b6c18f8b23f12b3f1a0a3fde3c4e6ac94feed72a7f411"


def _fixture_report(trade_count=4, pending_count=0):
    report = positive_foundation_report(trade_count)
    report["pending_signals"] = [
        {"fixture_pending": index} for index in range(pending_count)]
    return report


def _persist_research(registry, artifact):
    evidence = artifact.evidence or {}
    report = _fixture_report(
        evidence.get("trade_count", 4),
        evidence.get("pending_signal_count", 0))
    ResearchReportStore(registry.root).persist(
        ResearchReportArtifact.create(report))
    path = registry.store.persist(artifact)
    if artifact.evidence and "research_integrity" in artifact.evidence:
        persist_terminal_integrity(registry.root, artifact)
    return path


def _research_integrity(
        snapshot_sha=SNAPSHOT_SHA, n_trials=3, *, code_sha="a" * 40,
        seed=7, identity="default"):
    return fixture_integrity_evidence(
        snapshot_sha=snapshot_sha,
        code_sha=code_sha,
        seed=seed,
        n_trials=n_trials,
        identity=identity,
    )


def results_for(level: PromotionLevel) -> PromotionResults:
    if level == PromotionLevel.LIVE_ELIGIBLE:
        return PromotionResults(
            psr=0.99, dsr=0.98, oos_total_folds=4,
            oos_testable_folds=4, oos_significant_bh=3,
            cost_model_applied=True, estimated_cost_bps=12.0,
            cost_adjusted_return=0.20, annual_turnover=3.0,
            regime_results={"bull": 0.20, "bear": 0.02, "sideways": 0.08},
        )
    if level == PromotionLevel.PAPER_ELIGIBLE:
        return PromotionResults(
            psr=0.93, dsr=0.85, oos_total_folds=2,
            oos_testable_folds=2, oos_significant_bh=1,
            cost_model_applied=True, estimated_cost_bps=75.0,
            cost_adjusted_return=0.04, annual_turnover=18.0,
            regime_results={"bull": 0.12, "bear": -0.20},
        )
    return PromotionResults(
        psr=None, dsr=None, oos_total_folds=0, oos_testable_folds=0,
        oos_significant_bh=0, cost_model_applied=False,
        estimated_cost_bps=None, cost_adjusted_return=None,
        annual_turnover=None, regime_results={},
    )


def artifact_for(level: PromotionLevel, **overrides) -> PromotionArtifact:
    config = FoundationDeploymentConfig(
        strategy_version="foundation-v1",
        capital_allocation=0.10,
        model_key="gbm",
        rsi_entry_low=40.0,
        rsi_entry_high=70.0,
        rsi_exit=70.0,
        volume_confirmation_mult=1.2,
        gate_threshold=-0.05,
        target_mode=True,
    )
    values = {
        "strategy_id": "foundation",
        "strategy_version": "foundation-v1",
        "data_version": "pit-sha256:" + SNAPSHOT_SHA,
        "universe": ["MSFT", "AAPL"],
        "parameters": config.to_mapping(),
        "seed": 7,
        "dependency_versions": {"numpy": "2.4.6", "python": "3.13.5"},
        "code_sha": "0123456789abcdef0123456789abcdef01234567",
        "results": results_for(level),
    }
    values.update(overrides)
    if "evidence" not in overrides:
        if (level == PromotionLevel.LIVE_ELIGIBLE
                and "results" not in overrides):
            values["results"] = fixture_results(seed=values["seed"])
        symbols = sorted({
            str(symbol).strip().upper()
            for symbol in values["universe"] if str(symbol).strip()
        })
        dependencies = dict(values["dependency_versions"])
        python_version = dependencies.pop("python")
        regime_names = list(values["results"].regime_results)
        values["evidence"] = {
            "runner": "foundation_research_v4",
            "window": ["2022-01-01", "2024-12-31"],
            "universe_selection": {
                "requested_as_of": "2022-01-01",
                "resolved_as_of": "2021-12-31",
                "max_symbols": len(symbols),
                "method": (
                    "full_live_universe_fixed_requested_date_ranked_on_prior_"
                    "observable_session_complete_market_cap"
                ),
                "eligible_symbols": len(symbols),
                "ranked_symbols": len(symbols),
                "market_cap_coverage_complete": True,
            },
            "warehouse_snapshot": {
                "version": SNAPSHOT_SHA,
                "complete": True,
                "tables": [
                    {"table": table, "sha256": "c" * 64, "bytes": 10}
                    for table in ("actions", "daily", "sep", "tickers")
                ],
                "quality_flags": [],
            },
            "engine_parameters": fixture_engine_parameters(values["seed"]),
            "n_trials": 3,
            "research_integrity": _research_integrity(
                code_sha=values["code_sha"],
                seed=values["seed"],
                identity=canonical_json({
                    "parameters": values["parameters"],
                    "universe": symbols,
                    "strategy_version": values["strategy_version"],
                }),
            ),
            "regimes": fixture_regimes(regime_names),
            "report_sha256": ResearchReportArtifact.create(
                _fixture_report()).report_hash,
            "trade_count": 4,
            "pending_signal_count": 0,
            "provenance": {
                "git_sha": values["code_sha"],
                "git_dirty": False,
                "seed": values["seed"],
                "python_version": python_version,
                "dependency_versions": dependencies,
            },
        }
    return PromotionArtifact.create(**values)


def authoritative_paper_evidence(
        artifact: PromotionArtifact) -> dict:
    return _authoritative_paper_evidence(artifact, fills=4)


def paper_validation_for(
        artifact: PromotionArtifact, *,
        run_overrides=None, reconciliation_overrides=None,
        audit_verified=True, evidence=None) -> PaperValidationArtifact:
    run_summary = {
        "cycles": 20, "sessions": 15, "fills": 4, "errors": 0,
        "prospective": True,
    }
    run_summary.update(run_overrides or {})
    reconciliation = {
        "checks": 41, "failures": 0, "unknown_orders": 0,
        "open_orders": 0,
    }
    reconciliation.update(reconciliation_overrides or {})
    runner_evidence = authoritative_paper_evidence(artifact)
    runner_evidence.update(evidence or {})
    return PaperValidationArtifact.create(
        research_artifact=artifact,
        run_summary=run_summary,
        reconciliation_evidence=reconciliation,
        audit_verified=audit_verified,
        evidence=runner_evidence,
    )


def persist_paper_validation(
        registry: PromotionRegistry,
        artifact: PromotionArtifact) -> PaperValidationArtifact:
    validation = paper_validation_for(artifact)
    registry.paper_validation_store.persist(validation)
    return validation


@pytest.mark.parametrize(
    ("expected", "results"),
    [
        (PromotionLevel.RESEARCH_ONLY,
         results_for(PromotionLevel.RESEARCH_ONLY)),
        (PromotionLevel.PAPER_ELIGIBLE,
         results_for(PromotionLevel.PAPER_ELIGIBLE)),
        (PromotionLevel.LIVE_ELIGIBLE,
         results_for(PromotionLevel.LIVE_ELIGIBLE)),
    ],
)
def test_policy_assigns_all_three_levels(expected, results):
    evaluation = PromotionPolicy.default().evaluate(results)
    assert evaluation.level == expected
    assert len(evaluation.paper_checks) == 10
    assert len(evaluation.live_checks) == 10


@pytest.mark.parametrize(
    "weaker_live",
    [
        lambda paper: replace(paper, min_psr=paper.min_psr - 0.01),
        lambda paper: replace(paper, min_dsr=paper.min_dsr - 0.01),
        lambda paper: replace(
            paper, min_testable_oos_folds=1,
            min_significant_oos_folds=1),
        lambda paper: replace(
            paper, max_estimated_cost_bps=paper.max_estimated_cost_bps + 1),
        lambda paper: replace(
            paper, min_cost_adjusted_return=paper.min_cost_adjusted_return - 0.01),
        lambda paper: replace(
            paper, max_annual_turnover=paper.max_annual_turnover + 1),
        lambda paper: replace(paper, min_regime_count=1),
        lambda paper: replace(
            paper, min_regime_return=paper.min_regime_return - 0.01),
        lambda paper: replace(paper, require_cost_model=False),
    ],
)
def test_live_policy_can_never_be_weaker_than_paper(weaker_live):
    default = PromotionPolicy.default()
    with pytest.raises(ValueError, match="live criteria cannot be weaker"):
        PromotionPolicy(
            name="invalid", version="1", paper=default.paper,
            live=weaker_live(default.paper))


def test_live_significant_fold_requirement_cannot_be_weaker():
    default = PromotionPolicy.default()
    paper = replace(
        default.paper, min_testable_oos_folds=3,
        min_significant_oos_folds=2)
    with pytest.raises(ValueError, match="min_significant_oos_folds"):
        PromotionPolicy(
            name="invalid", version="1", paper=paper,
            live=replace(paper, min_significant_oos_folds=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("psr", None),
        ("dsr", None),
        ("estimated_cost_bps", None),
        ("cost_adjusted_return", None),
        ("annual_turnover", None),
    ],
)
def test_missing_required_evidence_is_research_only(field, value):
    data = results_for(PromotionLevel.LIVE_ELIGIBLE).__dict__.copy()
    data[field] = value
    assert PromotionPolicy.default().evaluate(
        PromotionResults(**data)).level == PromotionLevel.RESEARCH_ONLY


def test_cost_turnover_and_regimes_are_real_gates():
    base = results_for(PromotionLevel.LIVE_ELIGIBLE).__dict__.copy()
    mutations = [
        {"cost_model_applied": False},
        {"estimated_cost_bps": 101.0},
        {"cost_adjusted_return": -0.01},
        {"annual_turnover": 25.0},
        {"regime_results": {"bull": 0.2, "crisis": -0.30}},
    ]
    for mutation in mutations:
        assert PromotionPolicy.default().evaluate(
            PromotionResults(**{**base, **mutation})
        ).level == PromotionLevel.RESEARCH_ONLY


def test_calendar_year_significance_is_diagnostic_not_a_survivor_gate():
    data = results_for(PromotionLevel.LIVE_ELIGIBLE).__dict__.copy()
    data["oos_significant_bh"] = 0
    assert PromotionPolicy.default().evaluate(
        PromotionResults(**data)).level == PromotionLevel.LIVE_ELIGIBLE


def test_results_factory_computes_psr_dsr_fold_bh_and_cost_return():
    returns = [0.001, 0.002] * 30
    labels = [2023] * 20 + [2024] * 20 + [2025] * 20
    results = PromotionResults.from_oos_returns(
        returns, labels, n_trials=4, cost_model_applied=True,
        estimated_cost_bps=8.0, annual_turnover=2.0,
        regime_results={"bull": 0.1, "bear": 0.02, "sideways": 0.04},
    )
    assert results.psr is not None and results.psr > 0.95
    assert results.dsr is not None and 0.0 <= results.dsr <= 1.0
    assert results.oos_total_folds == results.oos_testable_folds == 3
    assert results.oos_significant_bh == 3
    assert results.cost_adjusted_return > 0.0


def test_results_factory_rejects_unaligned_or_nonfinite_returns():
    with pytest.raises(ValueError, match="align"):
        PromotionResults.from_oos_returns(
            [0.01], [], n_trials=1, cost_model_applied=True,
            estimated_cost_bps=1.0, annual_turnover=1.0,
            regime_results={"all": 0.01})
    with pytest.raises(ValueError, match="finite"):
        PromotionResults.from_oos_returns(
            [float("nan")], [2025], n_trials=1, cost_model_applied=True,
            estimated_cost_bps=1.0, annual_turnover=1.0,
            regime_results={"all": 0.01})


def test_identical_semantic_inputs_reproduce_hash_and_decision():
    first = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    second = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        universe=["AAPL", "MSFT", "AAPL"],
        parameters=dict(reversed(list(first.parameters.items()))),
        dependency_versions={"python": "3.13.5", "numpy": "2.4.6"},
    )
    assert first.artifact_hash == second.artifact_hash
    assert first.payload_json == second.payload_json
    assert first.decision == second.decision == PromotionLevel.LIVE_ELIGIBLE


def test_material_input_changes_artifact_hash():
    first = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    assert artifact_for(PromotionLevel.LIVE_ELIGIBLE, seed=8).artifact_hash \
        != first.artifact_hash
    assert artifact_for(PromotionLevel.LIVE_ELIGIBLE,
                        code_sha="different").artifact_hash != first.artifact_hash


def test_research_evidence_is_canonical_and_part_of_artifact_identity():
    evidence = {
        "snapshot_manifest": "sha256:snapshot",
        "report_digest": "sha256:report",
        "trial_count": 7,
    }
    first = artifact_for(PromotionLevel.LIVE_ELIGIBLE, evidence=evidence)
    reordered = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        evidence={"trial_count": 7, "report_digest": "sha256:report",
                  "snapshot_manifest": "sha256:snapshot"})
    changed = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        evidence={**evidence, "report_digest": "sha256:new-report"})
    assert first.evidence == evidence
    assert first.artifact_hash == reordered.artifact_hash
    assert first.artifact_hash != changed.artifact_hash


def test_schema_one_research_artifact_remains_readable_without_evidence():
    document = json.loads(artifact_for(PromotionLevel.LIVE_ELIGIBLE).to_json())
    document["payload"]["schema_version"] = 1
    document["payload"].pop("policy_id")
    document["payload"].pop("evidence")
    payload_json = canonical_json(document["payload"])
    document["artifact_hash"] = hashlib.sha256(payload_json.encode()).hexdigest()
    loaded = PromotionArtifact.from_json(canonical_json(document))
    assert loaded.evidence is None
    assert loaded.decision == PromotionLevel.LIVE_ELIGIBLE


def test_registry_refuses_readable_legacy_research_for_execution(tmp_path):
    document = json.loads(artifact_for(PromotionLevel.LIVE_ELIGIBLE).to_json())
    document["payload"]["schema_version"] = 1
    document["payload"].pop("policy_id")
    document["payload"].pop("evidence")
    payload_json = canonical_json(document["payload"])
    document["artifact_hash"] = hashlib.sha256(payload_json.encode()).hexdigest()
    legacy = PromotionArtifact.from_json(canonical_json(document))
    registry = PromotionRegistry(tmp_path)
    registry.store.persist(legacy)

    with pytest.raises(
            PromotionNotApproved, match="research artifact schema 2"):
        registry.promote(
            "foundation", legacy.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


@pytest.mark.parametrize("evidence", [
    None,
    {"runner": "synthetic-contract-fixture"},
    {"runner": "foundation_research_v1"},
    {"runner": "foundation_research_v2"},
    {"runner": "foundation_research_v3"},
])
def test_registry_refuses_fabricated_or_obsolete_research_evidence(
        tmp_path, evidence):
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE, evidence=evidence)
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)

    with pytest.raises(PromotionNotApproved, match="authoritative research"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_registry_accepts_complete_v4_research_runner_attestation(tmp_path):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)

    approval = registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")

    assert approval.exists()
    assert registry.require_approved(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE) == artifact


@pytest.mark.parametrize(
    "strategy_id",
    [
        "pead",
        "pead-vq-locked-replication-v1",
        "pead-vq-source-qualification-v2",
    ],
)
@pytest.mark.parametrize(
    "level", [PromotionLevel.PAPER_ELIGIBLE, PromotionLevel.LIVE_ELIGIBLE])
def test_registry_rejects_pead_until_authoritative_verifier_exists(
        tmp_path, strategy_id, level):
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        strategy_id=strategy_id,
        strategy_version="pead-v1",
    )
    registry = PromotionRegistry(tmp_path)
    registry.store.persist(artifact)

    with pytest.raises(
            PromotionNotApproved,
            match="no authoritative research verifier is registered"):
        registry.promote(
            strategy_id,
            artifact.artifact_hash,
            level,
            actor="research-reviewer",
        )
    assert not registry.approval_dir.exists()


def test_foundation_verifier_rejects_pead_version_masquerade(tmp_path):
    strategy_version = "pead-vq-source-qualification-v2"
    parameters = FoundationDeploymentConfig(
        strategy_version=strategy_version,
        capital_allocation=0.10,
        gate_threshold=-0.05,
    ).to_mapping()
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        strategy_id="foundation",
        strategy_version=strategy_version,
        parameters=parameters,
    )
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)

    with pytest.raises(
            PromotionNotApproved,
            match="strategy version must begin with 'foundation-'"):
        registry.promote(
            "foundation",
            artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE,
            actor="research-reviewer",
        )


def test_authoritative_research_verifier_dispatch_has_no_mutable_map():
    import analysis.promotion as promotion_module

    assert not hasattr(
        promotion_module, "_AUTHORITATIVE_RESEARCH_VERIFIERS")


@pytest.mark.parametrize("strategy_id", [
    "pead",
    "pead-vq-source-qualification-v2",
    "arbitrary-unregistered-strategy",
])
@pytest.mark.parametrize("schema_version", [1, 3])
@pytest.mark.parametrize(
    "level", [PromotionLevel.PAPER_ELIGIBLE, PromotionLevel.LIVE_ELIGIBLE])
def test_unknown_strategy_direct_approval_files_remain_ineligible(
        tmp_path, strategy_id, schema_version, level):
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        strategy_id=strategy_id,
        strategy_version="pead-v1",
    )
    registry = PromotionRegistry(tmp_path)
    registry.store.persist(artifact)
    path = registry._approval_path(strategy_id, artifact.artifact_hash, level)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = {
        "schema_version": schema_version,
        "strategy_id": strategy_id,
        "artifact_hash": artifact.artifact_hash,
        "approved_level": level.value,
    }
    if schema_version == 1:
        document = fixed
    else:
        document = {
            **fixed,
            "actor": "planted-approval",
            "paper_artifact_hash": (
                "a" * 64 if level == PromotionLevel.LIVE_ELIGIBLE else None),
            "replication_artifact_hash": (
                "b" * 64 if level == PromotionLevel.LIVE_ELIGIBLE else None),
            "promotion_policy_id": registry.policy.policy_id,
            "paper_validation_policy_id": (
                registry.paper_validation_policy.policy_id),
        }
    path.write_text(canonical_json(document) + "\n", encoding="utf-8")

    with pytest.raises(
            PromotionNotApproved,
            match="no authoritative research verifier is registered"):
        registry.require_approved(strategy_id, artifact.artifact_hash, level)


@pytest.mark.parametrize(
    "level", [PromotionLevel.PAPER_ELIGIBLE, PromotionLevel.LIVE_ELIGIBLE])
def test_legacy_schema_one_approval_is_inspect_only(tmp_path, level):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)
    if level == PromotionLevel.LIVE_ELIGIBLE:
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")
    path = registry._approval_path(
        "foundation", artifact.artifact_hash, level)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json({
        "schema_version": 1,
        "strategy_id": "foundation",
        "artifact_hash": artifact.artifact_hash,
        "approved_level": level.value,
    }) + "\n", encoding="utf-8")

    with pytest.raises(
            PromotionNotApproved,
            match="schema 1 is inspect-only"):
        registry.require_approved(
            "foundation", artifact.artifact_hash, level)


def test_registry_recomputes_summary_from_complete_raw_report(tmp_path):
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        results=results_for(PromotionLevel.LIVE_ELIGIBLE),
    )
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)

    with pytest.raises(PromotionNotApproved, match="statistics differ"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_registry_requires_terminal_on_disk_research_ledger(tmp_path):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    registry = PromotionRegistry(tmp_path)
    report = _fixture_report()
    ResearchReportStore(registry.root).persist(
        ResearchReportArtifact.create(report))
    registry.store.persist(artifact)

    with pytest.raises(PromotionNotApproved, match="verified research ledger"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_registry_rejects_summary_artifact_without_raw_report(tmp_path):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    registry = PromotionRegistry(tmp_path)
    registry.store.persist(artifact)

    with pytest.raises(PromotionNotApproved, match="complete verified raw"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_registry_rejects_fabricated_ledger_count_or_holdout_payload(tmp_path):
    valid = artifact_for(PromotionLevel.LIVE_ELIGIBLE)

    count_evidence = json.loads(json.dumps(valid.evidence))
    count_evidence["n_trials"] = 1
    forged_count = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE, evidence=count_evidence)
    registry = PromotionRegistry(tmp_path / "count")
    _persist_research(registry, forged_count)
    with pytest.raises(PromotionNotApproved, match="ledger-derived"):
        registry.promote(
            "foundation", forged_count.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")

    opening_evidence = json.loads(json.dumps(valid.evidence))
    opening_evidence["research_integrity"]["opening"]["actor"] = "changed"
    forged_opening = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE, evidence=opening_evidence)
    registry = PromotionRegistry(tmp_path / "opening")
    _persist_research(registry, forged_opening)
    with pytest.raises(PromotionNotApproved, match="opening hash"):
        registry.promote(
            "foundation", forged_opening.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reject_fills_without_adv", False, "without valid ADV"),
        ("participation_cap", 0.02, "0.01"),
    ],
)
def test_registry_rejects_relaxed_research_liquidity_attestation(
        tmp_path, field, value, message):
    valid = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    evidence = json.loads(json.dumps(valid.evidence))
    evidence["engine_parameters"][field] = value
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE, evidence=evidence)
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)

    with pytest.raises(PromotionNotApproved, match=message):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_registry_recomputes_research_snapshot_manifest_digest(tmp_path):
    valid = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    evidence = json.loads(json.dumps(valid.evidence))
    evidence["warehouse_snapshot"]["version"] = "f" * 64
    artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE,
        data_version="pit-sha256:" + "f" * 64,
        evidence=evidence,
    )
    registry = PromotionRegistry(tmp_path)
    _persist_research(registry, artifact)

    with pytest.raises(PromotionNotApproved, match="table manifest"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_artifact_rejects_noncanonical_numbers():
    with pytest.raises(ValueError, match="NaN"):
        artifact_for(PromotionLevel.LIVE_ELIGIBLE,
                     parameters={"threshold": float("nan")})


def test_store_round_trip_is_idempotent_and_content_addressed(tmp_path):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    store = ArtifactStore(tmp_path)
    first = store.persist(artifact)
    second = store.persist(artifact)
    assert first == second
    assert first.name == f"{artifact.artifact_hash}.json"
    assert store.load(artifact.artifact_hash) == artifact


def test_registry_pins_policy_and_rejects_other_embedded_policy(tmp_path):
    registry = PromotionRegistry(tmp_path)
    other_policy = replace(PromotionPolicy.default(), version="other")
    other_artifact = artifact_for(
        PromotionLevel.LIVE_ELIGIBLE, policy=other_policy)
    with pytest.raises(ArtifactIntegrityError, match="not accepted"):
        registry.store.persist(other_artifact)
    with pytest.raises(ArtifactIntegrityError, match="pinned policy"):
        PromotionRegistry(tmp_path, policy=other_policy)


def test_store_never_overwrites_existing_hash_path(tmp_path):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    store = ArtifactStore(tmp_path)
    path = store.path_for(artifact.artifact_hash)
    path.parent.mkdir(parents=True)
    path.write_text("foreign bytes")
    with pytest.raises(ArtifactIntegrityError, match="refusing to overwrite"):
        store.persist(artifact)
    assert path.read_text() == "foreign bytes"


def test_load_detects_tampered_artifact(tmp_path):
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    store = ArtifactStore(tmp_path)
    path = store.persist(artifact)
    document = json.loads(path.read_text())
    document["payload"]["seed"] = 999
    path.write_text(canonical_json(document))
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        store.load(artifact.artifact_hash)


def test_load_recomputes_decision_even_if_attacker_rehashes(tmp_path):
    artifact = artifact_for(PromotionLevel.RESEARCH_ONLY)
    document = json.loads(artifact.to_json())
    document["payload"]["decision"]["level"] = "live_eligible"
    payload_json = canonical_json(document["payload"])
    forged_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    document["artifact_hash"] = forged_hash
    path = tmp_path / "artifacts" / f"{forged_hash}.json"
    path.parent.mkdir(parents=True)
    path.write_text(canonical_json(document))
    with pytest.raises(ArtifactIntegrityError, match="decision"):
        ArtifactStore(tmp_path).load(forged_hash)


def test_paper_validation_is_content_addressed_and_exactly_bound(tmp_path):
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    validation = paper_validation_for(
        research, evidence={"run_id": "paper-2026-07-13"})
    store = PaperValidationStore(tmp_path)
    path = store.persist(validation)
    assert path.name == f"{validation.artifact_hash}.json"
    assert store.load(validation.artifact_hash) == validation
    assert validation.passed
    assert validation.research_artifact_hash == research.artifact_hash
    assert validation.strategy_id == research.strategy_id
    assert validation.code_sha == research.code_sha
    assert validation.parameters == research.parameters
    assert validation.evidence["run_id"] == "paper-2026-07-13"
    assert validation.evidence["runner"] == "foundation_paper_rehearsal_v2"


def test_passing_paper_summary_cannot_invent_arbitrary_counts():
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    with pytest.raises(
            ArtifactIntegrityError,
            match="run_summary disagrees with the sealed cycle/order facts"):
        PaperValidationArtifact.create(
            research_artifact=research,
            run_summary={
                "cycles": 999, "sessions": 999, "fills": 999,
                "errors": 0, "prospective": True,
            },
            reconciliation_evidence={
                "checks": 1999, "failures": 0,
                "unknown_orders": 0, "open_orders": 0,
            },
            audit_verified=True,
            evidence=authoritative_paper_evidence(research),
        )


def test_rehashed_paper_document_cannot_invent_passing_counts():
    validation = paper_validation_for(
        artifact_for(PromotionLevel.LIVE_ELIGIBLE))
    document = json.loads(validation.to_json())
    payload = document["payload"]
    payload["run_summary"].update({
        "cycles": 999, "sessions": 999, "fills": 999,
    })
    payload["reconciliation_evidence"]["checks"] = 1999
    policy = PaperValidationPolicy.from_dict(payload["policy"])
    payload["decision"] = policy.evaluate(
        payload["run_summary"], payload["reconciliation_evidence"],
        audit_verified=payload["audit_verified"],
    ).to_dict()
    document["artifact_hash"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")).hexdigest()

    with pytest.raises(
            ArtifactIntegrityError,
            match="run_summary disagrees with the sealed cycle/order facts"):
        PaperValidationArtifact.from_json(canonical_json(document))


def test_passing_paper_artifact_requires_authoritative_runner_evidence():
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    with pytest.raises(ArtifactIntegrityError, match="is required"):
        PaperValidationArtifact.create(
            research_artifact=research,
            run_summary={
                "cycles": 20, "sessions": 15, "fills": 4,
                "errors": 0, "prospective": True,
            },
            reconciliation_evidence={
                "checks": 41, "failures": 0,
                "unknown_orders": 0, "open_orders": 0,
            },
            audit_verified=True,
            evidence=None,
        )


def test_passing_paper_rejects_tampered_checkpoint_payload():
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    evidence = authoritative_paper_evidence(research)
    evidence["model_checkpoint"]["state"]["payload"]["cadence"][
        "days_since_fit"] += 1
    with pytest.raises(ArtifactIntegrityError, match="checkpoint hash"):
        paper_validation_for(research, evidence=evidence)


def test_passing_paper_derives_errors_from_cycle_result_and_reports():
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    evidence = authoritative_paper_evidence(research)
    evidence["cycles"][0]["result"] = {
        "status": "halted",
        "reports": [{"status": "error", "order_id": None}],
    }
    with pytest.raises(ArtifactIntegrityError, match="run_summary disagrees"):
        paper_validation_for(research, evidence=evidence)


def test_passing_paper_requires_a_flat_completed_round_trip():
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    evidence = authoritative_paper_evidence(research)
    for order in evidence["broker"]["orders"]:
        order["side"] = "BUY"
    for cycle in evidence["cycles"]:
        for order in cycle["orders"]:
            order["side"] = "BUY"
    with pytest.raises(ArtifactIntegrityError, match="completed round trip"):
        paper_validation_for(research, evidence=evidence)


def test_nonpassing_paper_artifact_can_preserve_diagnostic_summary():
    validation = PaperValidationArtifact.create(
        research_artifact=artifact_for(PromotionLevel.LIVE_ELIGIBLE),
        run_summary={
            "cycles": 1, "sessions": 1, "fills": 0,
            "errors": 1, "prospective": False,
        },
        reconciliation_evidence={
            "checks": 1, "failures": 1,
            "unknown_orders": 0, "open_orders": 0,
        },
        audit_verified=False,
        evidence={"diagnostic": "runner-crashed-before-seal"},
    )
    assert validation.passed is False


@pytest.mark.parametrize(
    ("run_overrides", "reconciliation_overrides", "audit_verified"),
    [
        ({"cycles": 1}, {}, True),
        ({"sessions": 1}, {}, True),
        ({"fills": 1}, {}, True),
        ({"errors": 1}, {}, True),
        ({"prospective": False}, {}, True),
        ({}, {"checks": 1}, True),
        ({}, {"failures": 1}, True),
        ({}, {"unknown_orders": 1}, True),
        ({}, {"open_orders": 1}, True),
        ({}, {}, False),
    ],
)
def test_default_paper_validation_policy_fails_closed(
        run_overrides, reconciliation_overrides, audit_verified):
    validation = paper_validation_for(
        artifact_for(PromotionLevel.LIVE_ELIGIBLE),
        run_overrides=run_overrides,
        reconciliation_overrides=reconciliation_overrides,
        audit_verified=audit_verified)
    assert not validation.passed


def test_paper_validation_store_detects_tampering(tmp_path):
    validation = paper_validation_for(
        artifact_for(PromotionLevel.LIVE_ELIGIBLE))
    store = PaperValidationStore(tmp_path)
    path = store.persist(validation)
    document = json.loads(path.read_text())
    document["payload"]["run_summary"]["errors"] = 1
    path.write_text(canonical_json(document))
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        store.load(validation.artifact_hash)


def test_registry_rejects_paper_validation_under_another_policy(tmp_path):
    registry = PromotionRegistry(tmp_path)
    research = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, research)
    other_policy = replace(PaperValidationPolicy.default(), version="other")
    validation = PaperValidationArtifact.create(
        research_artifact=research,
        run_summary={"cycles": 2, "sessions": 2, "fills": 2, "errors": 0},
        reconciliation_evidence={
            "checks": 2, "failures": 0, "unknown_orders": 0,
            "open_orders": 0,
        },
        audit_verified=True,
        policy=other_policy,
    )
    with pytest.raises(ArtifactIntegrityError, match="not accepted"):
        registry.paper_validation_store.persist(validation)


def test_promotion_requires_persisted_passing_artifact(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.RESEARCH_ONLY)
    with pytest.raises(PromotionNotApproved, match="does not exist"):
        registry.promote("foundation", artifact.artifact_hash,
                         PromotionLevel.PAPER_ELIGIBLE)
    _persist_research(registry, artifact)
    with pytest.raises(PromotionNotApproved, match="does not satisfy"):
        registry.promote("foundation", artifact.artifact_hash,
                         PromotionLevel.PAPER_ELIGIBLE)


def test_live_promotion_requires_same_artifact_paper_approval(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    with pytest.raises(PromotionNotApproved, match="not explicitly approved"):
        registry.promote("foundation", artifact.artifact_hash,
                         PromotionLevel.LIVE_ELIGIBLE)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.PAPER_ELIGIBLE,
                     actor="research-approver")
    validation = persist_paper_validation(registry, artifact)
    replication = persist_passing_replication(registry, artifact)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.LIVE_ELIGIBLE, actor="factory-test",
                     paper_artifact_hash=validation.artifact_hash,
                     replication_artifact_hash=replication.evidence_hash)
    assert registry.require_approved(
        "foundation", artifact.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE) == artifact
    assert registry.require_live_approved(
        "foundation", artifact.artifact_hash,
        validation.artifact_hash) == (artifact, validation)


def test_live_promotion_requires_actor_and_passing_paper_evidence(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    with pytest.raises(PromotionNotApproved, match="approving actor"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver")
    passing = persist_paper_validation(registry, artifact)
    failing = paper_validation_for(artifact, run_overrides={"errors": 1})
    registry.paper_validation_store.persist(failing)

    with pytest.raises(PromotionNotApproved, match="approving actor"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE,
            paper_artifact_hash=passing.artifact_hash)
    with pytest.raises(PromotionNotApproved, match="must differ"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="research-approver",
            paper_artifact_hash=passing.artifact_hash)
    with pytest.raises(PromotionNotApproved, match="requires a paper validation"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver")
    with pytest.raises(PromotionNotApproved, match="does not satisfy"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
            paper_artifact_hash=failing.artifact_hash)


def test_live_promotion_requires_passing_matching_replication(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver")
    validation = persist_paper_validation(registry, artifact)

    with pytest.raises(PromotionNotApproved, match="independent replication"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
            paper_artifact_hash=validation.artifact_hash)

    failing = persist_replication(
        registry,
        replication_evidence_for(artifact, mismatch_field="signal"))
    with pytest.raises(PromotionNotApproved, match="unresolved discrepancies"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
            paper_artifact_hash=validation.artifact_hash,
            replication_artifact_hash=failing.evidence_hash)

    wrong_protocol = persist_replication(
        registry,
        replication_evidence_for(artifact, protocol_hash="a" * 64))
    with pytest.raises(PromotionNotApproved, match="different research evidence"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
            paper_artifact_hash=validation.artifact_hash,
            replication_artifact_hash=wrong_protocol.evidence_hash)


def test_live_approval_records_actor_and_exact_paper_hash_immutably(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver")
    first = paper_validation_for(artifact, evidence={"run": "first"})
    second = paper_validation_for(artifact, evidence={"run": "second"})
    registry.paper_validation_store.persist(first)
    registry.paper_validation_store.persist(second)
    replication = persist_passing_replication(registry, artifact)
    approval_path = registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
        paper_artifact_hash=first.artifact_hash,
        replication_artifact_hash=replication.evidence_hash)

    approval = json.loads(approval_path.read_text())
    assert approval["schema_version"] == 3
    assert approval["actor"] == "risk-approver"
    assert approval["paper_artifact_hash"] == first.artifact_hash
    assert approval["replication_artifact_hash"] == replication.evidence_hash
    assert approval["promotion_policy_id"] == registry.policy.policy_id
    assert (approval["paper_validation_policy_id"]
            == registry.paper_validation_policy.policy_id)
    assert registry.require_live_approved(
        "foundation", artifact.artifact_hash,
        first.artifact_hash) == (artifact, first)
    with pytest.raises(PromotionNotApproved, match="different paper evidence"):
        registry.require_live_approved(
            "foundation", artifact.artifact_hash,
            second.artifact_hash)
    with pytest.raises(ArtifactIntegrityError, match="refusing to overwrite"):
        registry.promote(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="other-approver",
            paper_artifact_hash=second.artifact_hash,
            replication_artifact_hash=replication.evidence_hash)


def test_live_promotion_rejects_paper_evidence_for_other_artifact(tmp_path):
    registry = PromotionRegistry(tmp_path)
    approved = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    other = artifact_for(PromotionLevel.LIVE_ELIGIBLE, seed=8)
    _persist_research(registry, other)
    registry.store.persist(approved)
    registry.promote(
        "foundation", other.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver")
    validation = persist_paper_validation(registry, approved)
    with pytest.raises(PromotionNotApproved, match="different research artifact"):
        registry.promote(
            "foundation", other.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
            paper_artifact_hash=validation.artifact_hash)


def test_require_live_approved_reverifies_paper_evidence(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver")
    validation = persist_paper_validation(registry, artifact)
    replication = persist_passing_replication(registry, artifact)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
        paper_artifact_hash=validation.artifact_hash,
        replication_artifact_hash=replication.evidence_hash)
    path = registry.paper_validation_store.path_for(validation.artifact_hash)
    document = json.loads(path.read_text())
    document["payload"]["audit_verified"] = False
    path.write_text(canonical_json(document))
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        registry.require_approved(
            "foundation", artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE)


def test_require_live_approved_reverifies_replication_evidence(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver")
    validation = persist_paper_validation(registry, artifact)
    replication = persist_passing_replication(registry, artifact)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE, actor="risk-approver",
        paper_artifact_hash=validation.artifact_hash,
        replication_artifact_hash=replication.evidence_hash)

    replication_path = (registry.root / "independent-replications"
                        / f"{replication.evidence_hash}.json")
    replication_path.write_text("{}")
    with pytest.raises(PromotionNotApproved, match="verified independent"):
        registry.require_live_approved(
            "foundation", artifact.artifact_hash,
            validation.artifact_hash)


def test_approval_is_exact_hash_and_strategy(tmp_path):
    registry = PromotionRegistry(tmp_path)
    approved = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    other = artifact_for(PromotionLevel.LIVE_ELIGIBLE, seed=8)
    _persist_research(registry, approved)
    registry.store.persist(other)
    registry.promote("foundation", approved.artifact_hash,
                     PromotionLevel.PAPER_ELIGIBLE,
                     actor="research-approver")
    with pytest.raises(PromotionNotApproved):
        registry.require_approved("foundation", other.artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)
    with pytest.raises(PromotionNotApproved, match="different strategy"):
        registry.require_approved("other", approved.artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)


def test_corrupt_approval_reference_fails_closed(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    _persist_research(registry, artifact)
    path = registry.promote("foundation", artifact.artifact_hash,
                            PromotionLevel.PAPER_ELIGIBLE,
                            actor="research-approver")
    path.write_text("{}")
    with pytest.raises(ArtifactIntegrityError, match="reference"):
        registry.require_approved("foundation", artifact.artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)


def test_runtime_factory_requires_exact_approval_code_and_parameters(
        tmp_path, monkeypatch):
    parameters = FoundationDeploymentConfig(
        strategy_version="foundation-v1",
        gate_threshold=-0.05,
    ).to_mapping()
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE,
                            parameters=parameters)
    _persist_research(registry, artifact)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.PAPER_ELIGIBLE,
                     actor="research-approver")
    validation = persist_paper_validation(registry, artifact)
    replication = persist_passing_replication(registry, artifact)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.LIVE_ELIGIBLE, actor="factory-test",
                     paper_artifact_hash=validation.artifact_hash,
                     replication_artifact_hash=replication.evidence_hash)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda error_type: artifact.code_sha,
    )

    desk = create_deployed_desk(
        "foundation", artifact_hash=artifact.artifact_hash,
        promotion_registry=registry, runtime_code_sha=artifact.code_sha,
        runtime_parameters=parameters)
    assert isinstance(desk, FoundationDesk)

    with pytest.raises(PromotionNotApproved, match="code SHA"):
        create_deployed_desk(
            "foundation", artifact_hash=artifact.artifact_hash,
            promotion_registry=registry, runtime_code_sha="new-code",
            runtime_parameters=parameters)
    with pytest.raises(PromotionNotApproved, match="parameters"):
        create_deployed_desk(
            "foundation", artifact_hash=artifact.artifact_hash,
            promotion_registry=registry, runtime_code_sha=artifact.code_sha,
            runtime_parameters={**parameters, "gate_threshold": 0.0})
