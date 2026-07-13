"""Focused tests for artifact-bound Foundation construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from analysis.promotion import (
    PaperValidationArtifact,
    PromotionArtifact,
    PromotionLevel,
    PromotionNotApproved,
    PromotionRegistry,
    PromotionResults,
)
from desks.deployment_config import (
    FoundationDeploymentConfig,
    FoundationDeploymentIdentity,
)
from desks.foundation import FoundationDesk
from desks.ml_model import GradientBoostingModel
from desks.registry import create_deployed_desk
from tests.paper_evidence_helpers import authoritative_paper_evidence


CODE_SHA = "a" * 40
DATA_SHA = "eb12241b17158c2ac21b6c18f8b23f12b3f1a0a3fde3c4e6ac94feed72a7f411"


@pytest.fixture(autouse=True)
def _clean_approved_runtime(monkeypatch):
    """The focused artifacts use a synthetic, deterministic commit id."""
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda error_type: CODE_SHA,
    )


def _live_results() -> PromotionResults:
    return PromotionResults(
        psr=0.99,
        dsr=0.98,
        oos_total_folds=4,
        oos_testable_folds=4,
        oos_significant_bh=3,
        cost_model_applied=True,
        estimated_cost_bps=12.0,
        cost_adjusted_return=0.20,
        annual_turnover=3.0,
        regime_results={"bull": 0.20, "bear": 0.02, "sideways": 0.08},
    )


def _config(**overrides) -> FoundationDeploymentConfig:
    values = {
        "strategy_version": "foundation-v2",
        "capital_allocation": 0.35,
        "model_key": "gbm",
        "rsi_entry_low": 35.0,
        "rsi_entry_high": 66.0,
        "rsi_exit": 74.0,
        "volume_confirmation_mult": 1.4,
        "gate_threshold": -0.08,
        "target_mode": True,
    }
    values.update(overrides)
    return FoundationDeploymentConfig(**values)


def _research_evidence(universe, results: PromotionResults) -> dict:
    symbols = sorted({str(symbol).strip().upper() for symbol in universe})
    return {
        "runner": "foundation_research_v2",
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
            "version": DATA_SHA,
            "complete": True,
            "tables": [
                {"table": table, "sha256": "c" * 64, "bytes": 10}
                for table in ("actions", "daily", "sep", "tickers")
            ],
            "quality_flags": [],
        },
        "engine_parameters": {
            "initial_capital": 100_000.0,
            "commission": 0.001,
            "slippage_bps": 5.0,
            "enable_realistic_fills": True,
            "impact_coef": 0.01,
            "participation_cap": 0.10,
            "adv_window": 20,
            "seed": 7,
        },
        "n_trials": 3,
        "regimes": [
            {"name": name, "start": "2022-01-01", "end": "2024-12-31"}
            for name in results.regime_results
        ],
        "report_sha256": "d" * 64,
        "trade_count": 2,
        "pending_signal_count": 0,
        "provenance": {
            "git_sha": CODE_SHA,
            "git_dirty": False,
            "seed": 7,
            "python_version": "3.13.5",
            "dependency_versions": {"numpy": "2.4.6"},
        },
    }


def _approved(tmp_path, config: FoundationDeploymentConfig):
    registry = PromotionRegistry(tmp_path)
    results = _live_results()
    universe = ["AAPL", "MSFT"]
    artifact = PromotionArtifact.create(
        strategy_id="foundation",
        strategy_version=config.strategy_version,
        data_version="pit-sha256:" + DATA_SHA,
        universe=universe,
        parameters=config.to_mapping(),
        seed=7,
        dependency_versions={"numpy": "2.4.6", "python": "3.13.5"},
        code_sha=CODE_SHA,
        results=results,
        evidence=_research_evidence(universe, results),
    )
    registry.store.persist(artifact)
    _approve_live(registry, artifact)
    return registry, artifact


def _approve_live(registry: PromotionRegistry,
                  artifact: PromotionArtifact) -> None:
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-approver",
    )
    validation = PaperValidationArtifact.create(
        research_artifact=artifact,
        run_summary={
            "cycles": 20, "sessions": 15, "fills": 4, "errors": 0,
            "prospective": True,
        },
        reconciliation_evidence={
            "checks": 41, "failures": 0, "unknown_orders": 0,
            "open_orders": 0,
        },
        audit_verified=True,
        evidence=authoritative_paper_evidence(artifact, fills=4),
    )
    registry.paper_validation_store.persist(validation)
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE, actor="deployment-test",
        paper_artifact_hash=validation.artifact_hash,
    )


def test_canonical_mapping_contains_every_constructor_value_and_is_frozen():
    config = FoundationDeploymentConfig(strategy_version="foundation-v2")

    assert config.to_mapping() == {
        "strategy_version": "foundation-v2",
        "capital_allocation": 1.0,
        "model_key": "gbm",
        "rsi_entry_low": 40.0,
        "rsi_entry_high": 70.0,
        "rsi_exit": 70.0,
        "volume_confirmation_mult": 1.2,
        "gate_threshold": 0.0,
        "target_mode": True,
    }
    assert FoundationDeploymentConfig.from_mapping(
        config.to_mapping()) == config
    assert len(config.config_hash) == 64
    with pytest.raises(FrozenInstanceError):
        config.gate_threshold = 0.1


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda values: values.pop("rsi_exit"), "missing fields: rsi_exit"),
        (lambda values: values.update(extra=True), "unknown fields: extra"),
    ],
)
def test_mapping_parser_rejects_missing_and_unknown_fields(mutation, pattern):
    values = _config().to_mapping()
    mutation(values)
    with pytest.raises(ValueError, match=pattern):
        FoundationDeploymentConfig.from_mapping(values)


def test_deployment_config_rejects_unsafe_legacy_target_mode():
    with pytest.raises(ValueError, match="target_mode must be true"):
        _config(target_mode=False)


def test_build_threads_every_value_into_foundation():
    config = _config()
    desk = config.build()

    assert isinstance(desk, FoundationDesk)
    assert desk.capital_allocation == 0.35
    assert desk.rsi_entry_low == 35.0
    assert desk.rsi_entry_high == 66.0
    assert desk.rsi_exit == 74.0
    assert desk.volume_confirmation_mult == 1.4
    assert desk.gate_threshold == -0.08
    assert desk.target_native_enabled is True
    assert desk.deployment_version == config.strategy_version
    assert isinstance(desk._controller.model, GradientBoostingModel)
    assert desk.deployment_config is config


def test_deployed_factory_builds_only_from_artifact_and_attaches_identity(
        tmp_path):
    config = _config()
    registry, artifact = _approved(tmp_path, config)

    desk = create_deployed_desk(
        "foundation",
        artifact_hash=artifact.artifact_hash,
        promotion_registry=registry,
        runtime_code_sha=artifact.code_sha,
    )

    assert desk.deployment_config == config
    assert desk.capital_allocation == config.capital_allocation
    assert desk.gate_threshold == config.gate_threshold
    assert desk.target_native_enabled is True
    assert desk.deployment_identity == FoundationDeploymentIdentity(
        artifact_hash=artifact.artifact_hash,
        strategy_id="foundation",
        strategy_version=config.strategy_version,
        required_level="live_eligible",
        code_sha=artifact.code_sha,
        config_hash=config.config_hash,
    )
    with pytest.raises(FrozenInstanceError):
        desk.deployment_identity.artifact_hash = "different"
    with pytest.raises(AttributeError):
        desk.deployment_identity = FoundationDeploymentIdentity(
            artifact_hash="different", strategy_id="foundation",
            strategy_version=config.strategy_version,
            required_level="live_eligible", code_sha=artifact.code_sha,
            config_hash=config.config_hash,
        )


def test_legacy_runtime_arguments_are_assertions_never_overrides(tmp_path):
    config = _config()
    registry, artifact = _approved(tmp_path, config)
    common = {
        "key": "foundation",
        "artifact_hash": artifact.artifact_hash,
        "promotion_registry": registry,
        "runtime_code_sha": artifact.code_sha,
    }

    with pytest.raises(PromotionNotApproved, match="capital_allocation"):
        create_deployed_desk(**common, capital_allocation=0.9)
    with pytest.raises(PromotionNotApproved, match="model_key"):
        create_deployed_desk(**common, model_key="lightgbm")
    drifted = config.to_mapping()
    drifted["gate_threshold"] = 0.0
    with pytest.raises(PromotionNotApproved, match="parameters"):
        create_deployed_desk(**common, runtime_parameters=drifted)


def test_approval_rejects_incomplete_or_version_drifted_artifact(tmp_path):
    config = _config()
    registry, artifact = _approved(tmp_path, config)

    incomplete = PromotionArtifact.create(
        strategy_id="foundation",
        strategy_version=config.strategy_version,
        data_version="pit-sha256:" + DATA_SHA,
        universe=["AAPL"],
        parameters={"gate_threshold": -0.08},
        seed=7,
        dependency_versions={"numpy": "2.4.6", "python": "3.13.5"},
        code_sha=artifact.code_sha,
        results=_live_results(),
    )
    registry.store.persist(incomplete)
    with pytest.raises(PromotionNotApproved, match="authoritative research"):
        _approve_live(registry, incomplete)

    version_drift = _config(strategy_version="parameters-v3")
    drifted = PromotionArtifact.create(
        strategy_id="foundation",
        strategy_version="artifact-v3",
        data_version="pit-sha256:" + DATA_SHA,
        universe=["AAPL"],
        parameters=version_drift.to_mapping(),
        seed=7,
        dependency_versions={"numpy": "2.4.6", "python": "3.13.5"},
        code_sha=artifact.code_sha,
        results=_live_results(),
        evidence=_research_evidence(["AAPL"], _live_results()),
    )
    registry.store.persist(drifted)
    with pytest.raises(PromotionNotApproved, match="strategy version"):
        _approve_live(registry, drifted)


def test_factory_resolves_code_sha_internally_by_default(tmp_path):
    config = _config()
    registry, artifact = _approved(tmp_path, config)
    desk = create_deployed_desk(
        "foundation",
        artifact_hash=artifact.artifact_hash,
        promotion_registry=registry,
    )
    assert desk.deployment_identity.code_sha == artifact.code_sha
