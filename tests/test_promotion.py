from __future__ import annotations

import hashlib
import json

import pytest

from analysis.promotion import (
    ArtifactIntegrityError,
    ArtifactStore,
    PromotionArtifact,
    PromotionLevel,
    PromotionNotApproved,
    PromotionPolicy,
    PromotionRegistry,
    PromotionResults,
    canonical_json,
)
from desks.foundation import FoundationDesk
from desks.registry import create_deployed_desk


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
    values = {
        "strategy_id": "foundation",
        "strategy_version": "foundation-v1",
        "data_version": "pit-sha256:abc",
        "universe": ["MSFT", "AAPL"],
        "parameters": {"model_key": None, "gate_threshold": -0.05},
        "seed": 7,
        "dependency_versions": {"numpy": "2.4.6", "python": "3.13.5"},
        "code_sha": "0123456789abcdef",
        "results": results_for(level),
    }
    values.update(overrides)
    return PromotionArtifact.create(**values)


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


def test_cost_turnover_multiple_testing_and_regimes_are_real_gates():
    base = results_for(PromotionLevel.LIVE_ELIGIBLE).__dict__.copy()
    mutations = [
        {"cost_model_applied": False},
        {"estimated_cost_bps": 101.0},
        {"cost_adjusted_return": -0.01},
        {"annual_turnover": 25.0},
        {"oos_significant_bh": 0},
        {"regime_results": {"bull": 0.2, "crisis": -0.30}},
    ]
    for mutation in mutations:
        assert PromotionPolicy.default().evaluate(
            PromotionResults(**{**base, **mutation})
        ).level == PromotionLevel.RESEARCH_ONLY


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
        parameters={"gate_threshold": -0.05, "model_key": None},
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


def test_promotion_requires_persisted_passing_artifact(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.RESEARCH_ONLY)
    with pytest.raises(PromotionNotApproved, match="does not exist"):
        registry.promote("foundation", artifact.artifact_hash,
                         PromotionLevel.PAPER_ELIGIBLE)
    registry.store.persist(artifact)
    with pytest.raises(PromotionNotApproved, match="does not satisfy"):
        registry.promote("foundation", artifact.artifact_hash,
                         PromotionLevel.PAPER_ELIGIBLE)


def test_live_promotion_requires_same_artifact_paper_approval(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    registry.store.persist(artifact)
    with pytest.raises(PromotionNotApproved, match="not explicitly approved"):
        registry.promote("foundation", artifact.artifact_hash,
                         PromotionLevel.LIVE_ELIGIBLE)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.PAPER_ELIGIBLE)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.LIVE_ELIGIBLE)
    assert registry.require_approved(
        "foundation", artifact.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE) == artifact


def test_approval_is_exact_hash_and_strategy(tmp_path):
    registry = PromotionRegistry(tmp_path)
    approved = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    other = artifact_for(PromotionLevel.LIVE_ELIGIBLE, seed=8)
    registry.store.persist(approved)
    registry.store.persist(other)
    registry.promote("foundation", approved.artifact_hash,
                     PromotionLevel.PAPER_ELIGIBLE)
    with pytest.raises(PromotionNotApproved):
        registry.require_approved("foundation", other.artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)
    with pytest.raises(PromotionNotApproved, match="different strategy"):
        registry.require_approved("other", approved.artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)


def test_corrupt_approval_reference_fails_closed(tmp_path):
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE)
    registry.store.persist(artifact)
    path = registry.promote("foundation", artifact.artifact_hash,
                            PromotionLevel.PAPER_ELIGIBLE)
    path.write_text("{}")
    with pytest.raises(ArtifactIntegrityError, match="reference"):
        registry.require_approved("foundation", artifact.artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)


def test_runtime_factory_requires_exact_approval_code_and_parameters(tmp_path):
    parameters = {"model_key": None, "gate_threshold": -0.05}
    registry = PromotionRegistry(tmp_path)
    artifact = artifact_for(PromotionLevel.LIVE_ELIGIBLE,
                            parameters=parameters)
    registry.store.persist(artifact)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.PAPER_ELIGIBLE)
    registry.promote("foundation", artifact.artifact_hash,
                     PromotionLevel.LIVE_ELIGIBLE)

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
            runtime_parameters={"gate_threshold": 0.0})
