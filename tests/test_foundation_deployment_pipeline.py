"""End-to-end contract tests for the first trustworthy deployment lane."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from analysis.foundation_research import (
    FoundationResearchOutput,
    FoundationResearchSpec,
    ResearchRegime,
    FOUNDATION_UNIVERSE_METHOD,
    _require_frozen_foundation_trial,
    _select_universe,
    _snapshot_version,
    build_foundation_artifact,
    foundation_trial_inputs,
    persist_foundation_research,
)
from analysis.promotion import (
    ArtifactStore,
    PaperValidationArtifact,
    PaperValidationPolicy,
    PromotionArtifact,
    PromotionLevel,
    PromotionNotApproved,
    PromotionPolicy,
    PromotionRegistry,
    PromotionResults,
)
from analysis.research_report_store import (
    ResearchReportArtifact,
    ResearchReportStore,
)
from analysis.research_integrity import (
    ResearchIntegrityLedger,
    ResearchProtocol,
    TrialRegistration,
)
from deployment.live import (
    FoundationLiveController,
    LiveDeploymentPreflightError,
    VerifiedFoundationDeployment,
    _BoundedLiveData,
    validate_realtime_equity_quote,
)
from deployment.rehearsal import (
    FoundationPaperRehearsal,
    RehearsalStateError,
)
from deployment.state import (
    DeploymentManifest,
    DeploymentState,
    DeploymentStateError,
    DeploymentStore,
)
from desks.deployment_config import FoundationDeploymentConfig
from desks.foundation import FoundationDesk
from core.models import AssetType
from execution.live_context import LiveContextIdentity, LiveExecutionContext
from scripts import foundation_pipeline
from tests.paper_evidence_helpers import authoritative_paper_evidence
from tests.replication_evidence_helpers import persist_passing_replication
from tests.research_evidence_helpers import (
    fixture_engine_parameters,
    fixture_integrity_evidence,
    fixture_regimes,
    fixture_results,
    persist_terminal_integrity,
    positive_foundation_report,
)
from utils.audit import AuditLog


CODE_SHA = "a" * 40
SNAPSHOT_SHA = "eb12241b17158c2ac21b6c18f8b23f12b3f1a0a3fde3c4e6ac94feed72a7f411"


def _fixture_report():
    return positive_foundation_report(2)


def test_approve_live_cli_requires_and_forwards_replication_hash(
        monkeypatch, tmp_path):
    base = [
        "approve-live",
        "--artifact", "a" * 64,
        "--paper-artifact", "b" * 64,
        "--actor", "risk-reviewer",
    ]
    with pytest.raises(SystemExit):
        foundation_pipeline.parser().parse_args(base)
    args = foundation_pipeline.parser().parse_args([
        *base,
        "--replication-artifact", "c" * 64,
    ])
    captured = {}

    class Registry:
        def promote(self, *values, **kwargs):
            captured["values"] = values
            captured["kwargs"] = kwargs
            return tmp_path / "approval.json"

    monkeypatch.setattr(foundation_pipeline, "_registry", lambda _args: Registry())
    monkeypatch.setattr(foundation_pipeline, "_json", captured.update)
    args.func(args)

    assert captured["kwargs"]["replication_artifact_hash"] == "c" * 64
    assert captured["replication_artifact_hash"] == "c" * 64


def test_research_cli_requires_explicit_window_boundaries():
    base = [
        "research",
        "--protocol-hash", "a" * 64,
        "--trial-hash", "b" * 64,
        "--holdout-id", "oos-1",
        "--reviewer", "independent-reviewer",
    ]
    with pytest.raises(SystemExit):
        foundation_pipeline.parser().parse_args(base)

    args = foundation_pipeline.parser().parse_args([
        *base,
        "--start", "2027-01-01",
        "--end", "2027-12-31",
    ])
    assert args.start == "2027-01-01"
    assert args.end == "2027-12-31"


def _research_integrity(snapshot_sha=SNAPSHOT_SHA, n_trials=3):
    return fixture_integrity_evidence(
        snapshot_sha=snapshot_sha,
        code_sha=CODE_SHA,
        n_trials=n_trials,
        identity="foundation-pipeline",
    )


def _live_results() -> PromotionResults:
    return fixture_results(
        trade_count=2, regime_names=("bull", "bear", "flat"))


def _config() -> FoundationDeploymentConfig:
    return FoundationDeploymentConfig(
        strategy_version="foundation-target-v1",
        capital_allocation=0.10,
        model_key="gbm",
        rsi_entry_low=40.0,
        rsi_entry_high=70.0,
        rsi_exit=70.0,
        volume_confirmation_mult=1.2,
        gate_threshold=-0.05,
        target_mode=True,
    )


def _research_artifact(universe=("TEST",)) -> PromotionArtifact:
    config = _config()
    symbols = sorted({str(symbol).strip().upper() for symbol in universe})
    return PromotionArtifact.create(
        strategy_id="foundation",
        strategy_version=config.strategy_version,
        data_version="pit-sha256:" + SNAPSHOT_SHA,
        universe=symbols,
        parameters=config.to_mapping(),
        seed=7,
        dependency_versions={"numpy": "2.4.6", "python": "3.13.3"},
        code_sha=CODE_SHA,
        results=_live_results(),
        evidence={
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
            "engine_parameters": fixture_engine_parameters(),
            "n_trials": 3,
            "research_integrity": _research_integrity(),
            "regimes": fixture_regimes(("bull", "bear", "flat")),
            "report_sha256": ResearchReportArtifact.create(
                _fixture_report()).report_hash,
            "trade_count": 2,
            "pending_signal_count": 0,
            "provenance": {
                "git_sha": CODE_SHA,
                "git_dirty": False,
                "seed": 7,
                "python_version": "3.13.3",
                "dependency_versions": {"numpy": "2.4.6"},
            },
        },
    )


def _registry_with_paper_approval(
        tmp_path, universe=("TEST",), paper_policy=None):
    registry = PromotionRegistry(
        tmp_path / "promotion", paper_validation_policy=paper_policy)
    research = _research_artifact(universe)
    ResearchReportStore(registry.root).persist(
        ResearchReportArtifact.create(_fixture_report()))
    registry.store.persist(research)
    persist_terminal_integrity(registry.root, research)
    registry.promote(
        "foundation", research.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer",
    )
    return registry, research


def _positive_report() -> dict:
    points = []
    value = 100_000.0
    for year in (2022, 2023, 2024):
        for offset, day in enumerate(
                pd.bdate_range(f"{year}-01-03", periods=30)):
            value *= 1.0008 if offset % 2 else 1.0012
            points.append({"timestamp": day, "portfolio_value": value})
    return {
        "portfolio_history": points,
        "trades": [
            {"quantity": 10, "price": 100.0},
            {"quantity": 10, "price": 102.0},
        ],
        "pending_signals": [],
    }


def test_research_artifact_binds_strict_inputs_and_can_reach_live_decision(
        tmp_path):
    spec = FoundationResearchSpec(
        n_trials=3,
        start="2022-01-01",
        end="2024-12-31",
        regimes=(
            ResearchRegime("r2022", "2022-01-01", "2022-12-31"),
            ResearchRegime("r2023", "2023-01-01", "2023-12-31"),
            ResearchRegime("r2024", "2024-01-01", "2024-12-31"),
        ),
    )
    snapshot = {
        "version": SNAPSHOT_SHA,
        "complete": True,
        "tables": [{"table": name, "sha256": "c" * 64, "bytes": 10}
                   for name in ("actions", "daily", "sep", "tickers")],
        "quality_flags": [],
    }
    provenance = {
        "git_sha": CODE_SHA,
        "git_dirty": False,
        "seed": 7,
        "python_version": "3.13.3",
        "dependency_versions": {"numpy": "2.4.6", "python": "3.13.3"},
    }

    artifact = build_foundation_artifact(
        report=_positive_report(),
        selected_universe=["TEST"],
        data_snapshot=snapshot,
        provenance=provenance,
        spec=spec,
        config=_config(),
        universe_selection_date="2021-12-31",
        eligible_symbol_count=1,
    )

    assert artifact.decision is PromotionLevel.LIVE_ELIGIBLE
    assert artifact.parameters == _config().to_mapping()
    assert artifact.evidence["engine_parameters"]["enable_realistic_fills"] \
        is True
    assert artifact.evidence["engine_parameters"][
        "reject_fills_without_adv"] is True
    assert artifact.evidence["engine_parameters"]["participation_cap"] \
        == pytest.approx(0.01)
    assert artifact.evidence["n_trials"] == 3
    assert artifact.evidence["runner"] == "foundation_research_v3"
    assert artifact.evidence["universe_selection"] == {
        "requested_as_of": "2022-01-01",
        "resolved_as_of": "2021-12-31",
        "max_symbols": 100,
        "method": (
            "full_live_universe_fixed_requested_date_ranked_on_prior_"
            "observable_session_complete_market_cap"
        ),
        "eligible_symbols": 1,
        "ranked_symbols": 1,
        "market_cap_coverage_complete": True,
    }
    assert artifact.payload["data_version"] == "pit-sha256:" + SNAPSHOT_SHA
    output = FoundationResearchOutput(
        artifact=artifact,
        report=_positive_report(),
        selected_universe=("TEST",),
        data_snapshot=snapshot,
    )
    store = ArtifactStore(tmp_path)
    persist_foundation_research(output, store)
    raw = ResearchReportStore(tmp_path).load(
        artifact.evidence["report_sha256"])
    assert len(raw.report["portfolio_history"]) == 90
    assert len(raw.report["trades"]) == 2


def test_qualified_research_uses_ledger_count_and_permanently_decides_holdout(
        tmp_path):
    registry = PromotionRegistry(tmp_path / "promotion")
    ledger = ResearchIntegrityLedger(
        registry.root / "research-integrity",
        program_id="stock-options-trader")
    protocol = ledger.freeze_protocol(ResearchProtocol.create(
        program_id="stock-options-trader",
        protocol_id="foundation-locked",
        version="1",
        objective="Evaluate the locked Foundation candidate net of costs.",
        hypotheses=[{"hypothesis_id": "foundation-net", "direction": "+"}],
        candidate_specifications=[{
            "candidate_id": "foundation-target-v1", "locked": True}],
        data_plan={"point_in_time": True},
        evaluation_plan={"primary": "aggregate net OOS"},
        decision_rules={"paper_policy": "promotion-v1"},
        holdouts={"foundation-oos-v1": {
            "data_artifact_hash": SNAPSHOT_SHA}},
        code_version=CODE_SHA,
    ))
    trial = ledger.register_trial(TrialRegistration.create(
        protocol_hash=protocol.protocol_hash,
        trial_id="foundation-001",
        candidate_id="foundation-target-v1",
        family="foundation",
        inputs={"configuration": "locked"},
        data_version="pit-sha256:" + SNAPSHOT_SHA,
        code_version=CODE_SHA,
        seed=7,
    ))
    opening = ledger.open_holdout(
        protocol_hash=protocol.protocol_hash,
        holdout_id="foundation-oos-v1",
        trial_hash=trial.trial_hash,
        data_artifact_hash=SNAPSHOT_SHA,
        actor="independent-reviewer",
    )
    spec = FoundationResearchSpec(
        n_trials=1,
        start="2022-01-01",
        end="2024-12-31",
        regimes=(
            ResearchRegime("r2022", "2022-01-01", "2022-12-31"),
            ResearchRegime("r2023", "2023-01-01", "2023-12-31"),
            ResearchRegime("r2024", "2024-01-01", "2024-12-31"),
        ),
    )
    snapshot = {
        "version": SNAPSHOT_SHA,
        "complete": True,
        "tables": [{"table": name, "sha256": "c" * 64, "bytes": 10}
                   for name in ("actions", "daily", "sep", "tickers")],
        "quality_flags": [],
    }
    artifact = build_foundation_artifact(
        report=_positive_report(),
        selected_universe=["TEST"],
        data_snapshot=snapshot,
        provenance={
            "git_sha": CODE_SHA,
            "git_dirty": False,
            "seed": 7,
            "python_version": "3.13.3",
            "dependency_versions": {"numpy": "2.4.6"},
        },
        spec=spec,
        config=_config(),
        universe_selection_date="2021-12-31",
        eligible_symbol_count=1,
        integrity_opening=opening,
        integrity_program_id=ledger.program_id,
        integrity_head_hash=ledger.head_hash,
    )
    assert artifact.evidence["runner"] == "foundation_research_v4"
    assert artifact.evidence["n_trials"] == ledger.trial_count == 1

    output = FoundationResearchOutput(
        artifact=artifact,
        report=_positive_report(),
        selected_universe=("TEST",),
        data_snapshot=snapshot,
        integrity_opening=opening,
    )
    persist_foundation_research(
        output, registry.store,
        integrity_ledger=ledger, reviewer="independent-reviewer")
    # A retry after durable writes verifies and reuses the exact terminal
    # records instead of manufacturing a second outcome or decision.
    persist_foundation_research(
        output, registry.store,
        integrity_ledger=ledger, reviewer="independent-reviewer")
    decision = ledger.get_holdout_decision(
        protocol_hash=protocol.protocol_hash,
        holdout_id="foundation-oos-v1",
    )
    assert decision is not None
    assert decision.payload["decision"] == "pass"
    assert decision.payload["result_artifact_hash"] == artifact.artifact_hash
    outcome = ledger.get_trial_outcome(trial.trial_hash)
    assert outcome is not None
    assert outcome.payload["status"] == "completed"
    assert outcome.payload["evidence_hash"] == artifact.artifact_hash
    verified = ledger.verify()
    assert verified["trial_outcome_count"] == 1
    assert verified["holdout_decision_count"] == 1
    registry.promote(
        "foundation", artifact.artifact_hash,
        PromotionLevel.PAPER_ELIGIBLE, actor="research-reviewer")


def test_qualifying_runner_requires_exact_frozen_trial_inputs(tmp_path):
    registry = PromotionRegistry(tmp_path / "promotion")
    ledger = ResearchIntegrityLedger(
        registry.root / "research-integrity",
        program_id="stock-options-trader")
    spec = FoundationResearchSpec(
        n_trials=1,
        start="2022-01-01",
        end="2024-12-31",
        regimes=(
            ResearchRegime("r2022", "2022-01-01", "2022-12-31"),
            ResearchRegime("r2023", "2023-01-01", "2023-12-31"),
            ResearchRegime("r2024", "2024-01-01", "2024-12-31"),
        ),
    )
    config = _config()
    policy = PromotionPolicy.default()
    protocol = ledger.freeze_protocol(ResearchProtocol.create(
        program_id="stock-options-trader",
        protocol_id="foundation-exact-v1",
        version="1",
        objective="Evaluate one locked Foundation candidate net of costs.",
        hypotheses=[{"hypothesis_id": "foundation-net", "direction": "+"}],
        candidate_specifications=[{
            "candidate_id": config.strategy_version, "locked": True}],
        data_plan={
            "point_in_time": True,
            "snapshot_hash": SNAPSHOT_SHA,
            "universe_selection_method": FOUNDATION_UNIVERSE_METHOD,
        },
        evaluation_plan={
            "primary_test": "aggregate_net_oos",
            "calendar_years": "diagnostic_only",
            "costs": "realized_execution",
            "independent_replication_required": True,
        },
        decision_rules={
            "promotion_policy_id": policy.policy_id,
            "one_shot_holdout": True,
            "failure_is_terminal": True,
        },
        holdouts={"foundation-oos-v1": {
            "start": "2022-01-01",
            "end": "2024-12-31",
            "data_artifact_hash": SNAPSHOT_SHA,
            "sealed": True,
        }},
        code_version=CODE_SHA,
    ))
    trial = ledger.register_trial(TrialRegistration.create(
        protocol_hash=protocol.protocol_hash,
        trial_id="foundation-exact-001",
        candidate_id=config.strategy_version,
        family="foundation",
        inputs=foundation_trial_inputs(spec, config, policy),
        data_version="pit-sha256:" + SNAPSHOT_SHA,
        code_version=CODE_SHA,
        seed=7,
    ))
    provenance = {
        "git_sha": CODE_SHA,
        "git_dirty": False,
        "seed": 7,
        "python_version": "3.13.3",
        "dependency_versions": {"numpy": "2.4.6"},
    }
    _require_frozen_foundation_trial(
        ledger=ledger,
        protocol_hash=protocol.protocol_hash,
        trial_hash=trial.trial_hash,
        holdout_id="foundation-oos-v1",
        data_version=SNAPSHOT_SHA,
        spec=spec,
        config=config,
        policy=policy,
        provenance=provenance,
    )
    changed = FoundationResearchSpec(
        n_trials=1,
        start=spec.start,
        end=spec.end,
        max_symbols=99,
        regimes=spec.regimes,
    )
    with pytest.raises(RuntimeError, match="registered trial"):
        _require_frozen_foundation_trial(
            ledger=ledger,
            protocol_hash=protocol.protocol_hash,
            trial_hash=trial.trial_hash,
            holdout_id="foundation-oos-v1",
            data_version=SNAPSHOT_SHA,
            spec=changed,
            config=config,
            policy=policy,
            provenance=provenance,
        )
def test_dirty_research_runtime_is_never_qualifying():
    spec = FoundationResearchSpec(
        n_trials=1, start="2022-01-01", end="2024-12-31",
        regimes=(ResearchRegime("all", "2022-01-01", "2024-12-31"),),
    )
    with pytest.raises(RuntimeError, match="clean working tree"):
        build_foundation_artifact(
            report=_positive_report(), selected_universe=["TEST"],
            data_snapshot={"version": "b" * 64, "complete": True},
            provenance={
                "git_sha": CODE_SHA, "git_dirty": True, "seed": 7,
                "python_version": "3.13.3",
                "dependency_versions": {"python": "3.13.3"},
            },
            spec=spec, config=_config(),
            universe_selection_date="2021-12-31",
            eligible_symbol_count=1,
        )


def test_research_universe_ranks_on_prior_observable_session():
    class Warehouse:
        def __init__(self):
            self.cap_dates = []

        def universe_asof(self, as_of):
            assert as_of == "2015-01-01"
            return ["SMALL", "LARGE"]

        def daily_marketcaps(self, symbols, as_of):
            self.cap_dates.append(as_of)
            assert symbols == ["LARGE", "SMALL"]
            if as_of == "2014-12-31":
                return {"LARGE": 500.0, "SMALL": 10.0}
            return {}

    warehouse = Warehouse()
    selected, resolved, eligible = _select_universe(
        warehouse, FoundationResearchSpec(n_trials=1, max_symbols=1))

    assert selected == ("LARGE",)
    assert resolved == "2014-12-31"
    assert eligible == 2
    assert warehouse.cap_dates == ["2015-01-01", "2014-12-31"]


def test_research_universe_rejects_incomplete_market_cap_coverage():
    class Warehouse:
        def universe_asof(self, _as_of):
            return ["COVERED", "MISSING"]

        def daily_marketcaps(self, _symbols, _as_of):
            return {"COVERED": 100.0}

    with pytest.raises(RuntimeError, match="market-cap coverage is incomplete"):
        _select_universe(
            Warehouse(), FoundationResearchSpec(n_trials=1, max_symbols=1))


def test_research_universe_date_cannot_look_past_backtest_start():
    with pytest.raises(ValueError, match="cannot be after the research start"):
        FoundationResearchSpec(
            n_trials=1, start="2015-01-01", universe_as_of="2024-01-01")


def test_qualifying_research_requires_fail_closed_one_percent_liquidity():
    spec = FoundationResearchSpec(n_trials=1)

    assert spec.participation_cap == pytest.approx(0.01)
    assert spec.reject_fills_without_adv is True
    assert spec.engine_parameters()["reject_fills_without_adv"] is True

    with pytest.raises(ValueError, match=r"\(0, 0\.01\]"):
        FoundationResearchSpec(n_trials=1, participation_cap=0.02)
    with pytest.raises(ValueError, match="must reject fills"):
        FoundationResearchSpec(n_trials=1, reject_fills_without_adv=False)


def test_research_snapshot_digest_must_match_table_manifest():
    snapshot = {
        "version": "f" * 64,
        "complete": True,
        "tables": [
            {"table": table, "sha256": "c" * 64, "bytes": 10}
            for table in ("actions", "daily", "sep", "tickers")
        ],
        "quality_flags": [],
    }
    with pytest.raises(RuntimeError, match="differs from its table manifest"):
        _snapshot_version(snapshot)


def test_saved_real_foundation_result_remains_research_only():
    with open("analysis/backtests/foundation.json", encoding="utf-8") as stream:
        result = json.load(stream)
    regimes = {
        name: float(row.get("return_pct", 0.0)) / 100.0
        for name, row in result["regimes"].items()
    }
    evidence = PromotionResults(
        psr=result["psr"],
        dsr=result["deflated_sharpe"],
        oos_total_folds=len(result["oos_folds"].get("folds", [])),
        oos_testable_folds=result["oos_folds"].get("n_testable_folds", 0),
        oos_significant_bh=result["oos_folds"].get("bh", {}).get(
            "n_significant_bh", 0),
        cost_model_applied=False,
        estimated_cost_bps=None,
        cost_adjusted_return=float(result["total_return_pct"]) / 100.0,
        annual_turnover=None,
        regime_results=regimes,
    )
    assert PromotionPolicy.default().evaluate(evidence).level \
        is PromotionLevel.RESEARCH_ONLY


def _momentum_frame(end: str, *, entry: bool) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp(end), periods=130)
    frame = pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 200.0,
        "volume_sma": 100.0,
        "macd": 0.0,
        "signal": 0.0,
        "rsi": 50.0,
    }, index=index)
    if entry:
        frame.iloc[-2, frame.columns.get_loc("macd")] = -1.0
        frame.iloc[-1, frame.columns.get_loc("macd")] = 1.0
    else:
        frame.iloc[-2, frame.columns.get_loc("macd")] = 1.0
        frame.iloc[-1, frame.columns.get_loc("macd")] = -1.0
        frame.iloc[-1, frame.columns.get_loc("rsi")] = 80.0
    return frame


def test_paper_rehearsal_reconciles_two_fills_and_binds_live_approval(
        tmp_path, monkeypatch):
    contract_policy = PaperValidationPolicy(
        name="synthetic-contract-paper", version="1",
        min_cycles=2, min_sessions=2, min_fills=2,
        min_reconciliation_checks=5,
    )
    registry, research = _registry_with_paper_approval(
        tmp_path, paper_policy=contract_policy)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda _error_type: CODE_SHA,
    )
    wall_clock = [datetime(2026, 1, 5, 15, tzinfo=timezone.utc)]
    monkeypatch.setattr(
        "deployment.rehearsal._wall_clock", lambda: wall_clock[0])
    db = tmp_path / "paper.db"
    rehearsal = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="run-001",
        research_artifact_hash=research.artifact_hash,
    )

    first = rehearsal.run_cycle(
        "2026-01-05T15:00:00+00:00",
        {"TEST": _momentum_frame("2026-01-02", entry=True)},
        execution_prices={
            "TEST": {
                "price": 100.0,
                "observed_at": "2026-01-05T14:59:30+00:00",
                "source": "synthetic-contract-quote",
            },
        },
    )
    # Match the operator CLI: every step is a fresh process.  The resumed
    # desk must restore its safe model/cadence checkpoint before D+1.
    rehearsal = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="run-001", resume=True,
    )
    wall_clock[0] = datetime(2026, 1, 6, 15, tzinfo=timezone.utc)
    second = rehearsal.run_cycle(
        "2026-01-06T15:00:00+00:00",
        {"TEST": _momentum_frame("2026-01-05", entry=False)},
        execution_prices={
            "TEST": {
                "price": 100.0,
                "observed_at": "2026-01-06T14:59:30+00:00",
                "source": "synthetic-contract-quote",
            },
        },
    )
    artifact = rehearsal.finalize()

    assert first["pre_reconciliation"]["ok"] is True
    assert first["post_reconciliation"]["ok"] is True
    assert second["post_reconciliation"]["ok"] is True
    assert artifact.passed is True
    assert artifact.payload["run_summary"] == {
        "cycles": 2, "sessions": 2, "fills": 2, "errors": 0,
        "prospective": True,
    }
    assert artifact.payload["reconciliation_evidence"]["failures"] == 0
    assert artifact.payload["evidence"]["broker"]["transaction_cost"] \
        == pytest.approx(2.0)
    sealed_checkpoint = artifact.payload["evidence"]["model_checkpoint"]
    assert sealed_checkpoint["state"] == rehearsal.desk.model_checkpoint_state()
    assert sealed_checkpoint["sha256"] == sealed_checkpoint["state"]["sha256"]

    # Restart reads the independent broker and system books and returns the
    # same immutable final artifact, rather than fabricating a new rehearsal.
    resumed = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="run-001", resume=True,
    )
    assert resumed.finalize().artifact_hash == artifact.artifact_hash

    replication = persist_passing_replication(registry, research)
    registry.promote(
        "foundation", research.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE,
        actor="live-risk-owner",
        paper_artifact_hash=artifact.artifact_hash,
        replication_artifact_hash=replication.evidence_hash,
    )
    bound_research, bound_paper = registry.require_live_approved(
        "foundation", research.artifact_hash, artifact.artifact_hash)
    assert bound_research == research
    assert bound_paper == artifact


def _paper_quote(observed_at: str) -> dict:
    return {"TEST": {
        "price": 100.0,
        "observed_at": observed_at,
        "source": "synthetic-contract-quote",
    }}


def test_paper_cycles_reject_replay_bad_clocks_and_noncausal_data(
        tmp_path, monkeypatch):
    registry, research = _registry_with_paper_approval(tmp_path)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda _error_type: CODE_SHA,
    )
    wall = [datetime(2026, 1, 5, 15, tzinfo=timezone.utc)]
    monkeypatch.setattr(
        "deployment.rehearsal._wall_clock", lambda: wall[0])
    rehearsal = FoundationPaperRehearsal(
        registry=registry, db_path=str(tmp_path / "rejections.db"),
        run_id="reject-invalid", research_artifact_hash=research.artifact_hash,
    )
    frame = _momentum_frame("2026-01-02", entry=True)

    with pytest.raises(RehearsalStateError, match="regular session"):
        rehearsal.run_cycle(
            "2026-01-05T14:00:00+00:00", {"TEST": frame},
            execution_prices=_paper_quote("2026-01-05T13:59:30+00:00"))
    with pytest.raises(RehearsalStateError, match="prospectively"):
        rehearsal.run_cycle(
            "2026-01-05T15:10:00+00:00", {"TEST": frame},
            execution_prices=_paper_quote("2026-01-05T15:09:30+00:00"))
    wall[0] = datetime(2026, 1, 5, 16, tzinfo=timezone.utc)
    with pytest.raises(RehearsalStateError, match="prospectively"):
        rehearsal.run_cycle(
            "2026-01-05T15:00:00+00:00", {"TEST": frame},
            execution_prices=_paper_quote("2026-01-05T14:59:30+00:00"))
    wall[0] = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)
    with pytest.raises(RehearsalStateError, match="future-dated or stale"):
        rehearsal.run_cycle(
            "2026-01-05T15:00:00+00:00", {"TEST": frame},
            execution_prices=_paper_quote("2026-01-05T14:50:00+00:00"))
    with pytest.raises(RehearsalStateError, match="future-dated or stale"):
        rehearsal.run_cycle(
            "2026-01-05T15:00:00+00:00", {"TEST": frame},
            execution_prices=_paper_quote("2026-01-05T15:00:01+00:00"))
    with pytest.raises(RehearsalStateError, match="unique and increasing"):
        rehearsal.run_cycle(
            "2026-01-05T15:00:00+00:00",
            {"TEST": frame.iloc[::-1]},
            execution_prices=_paper_quote("2026-01-05T14:59:30+00:00"))
    with pytest.raises(RehearsalStateError, match="universe differs"):
        rehearsal.run_cycle(
            "2026-01-05T15:00:00+00:00",
            {"TEST": frame, " test ": frame.copy()},
            execution_prices=_paper_quote("2026-01-05T14:59:30+00:00"))
    assert rehearsal.status().cycles == 0


def test_cli_paper_step_builds_prior_session_signals_and_uncached_quote(
        tmp_path, monkeypatch):
    _, research = _registry_with_paper_approval(tmp_path)
    history = _momentum_frame("2026-07-10", entry=True)

    class FakeLiveMarketData:
        def fetch_stock_data(self, _symbol, start, end):
            return history.copy()

        @staticmethod
        def fetch_stock_quote(_symbol):
            return {
                "price": 101.25,
                "observed_at": "2026-07-13T13:59:45+00:00",
                "source": "openbb:test-live:last",
            }

        @staticmethod
        def calculate_indicators(frame):
            return frame

    monkeypatch.setattr(
        foundation_pipeline, "MarketDataHandler", FakeLiveMarketData)
    signals, prices = foundation_pipeline._cycle_data(
        research, datetime(2026, 7, 13, 14, tzinfo=timezone.utc))

    assert signals["TEST"].index.max() == pd.Timestamp("2026-07-10")
    assert prices["TEST"]["price"] == pytest.approx(101.25)
    assert prices["TEST"]["source"] == "openbb:test-live:last"
    assert datetime.fromisoformat(prices["TEST"]["observed_at"]).tzinfo \
        is not None


def test_started_cycle_resumes_without_duplicate_order_after_commit_failure(
        tmp_path, monkeypatch):
    registry, research = _registry_with_paper_approval(tmp_path)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda _error_type: CODE_SHA,
    )
    wall = [datetime(2026, 1, 5, 15, tzinfo=timezone.utc)]
    monkeypatch.setattr(
        "deployment.rehearsal._wall_clock", lambda: wall[0])
    db = tmp_path / "crash-recovery.db"
    rehearsal = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="crash-recovery",
        research_artifact_hash=research.artifact_hash,
    )
    monkeypatch.setattr(
        rehearsal.desk, "model_checkpoint_state",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated crash")),
    )
    args = (
        "2026-01-05T15:00:00+00:00",
        {"TEST": _momentum_frame("2026-01-02", entry=True)},
    )
    prices = _paper_quote("2026-01-05T14:59:30+00:00")
    with pytest.raises(RuntimeError, match="simulated crash"):
        rehearsal.run_cycle(*args, execution_prices=prices)
    assert rehearsal.status().cycles == 0
    assert len(rehearsal.broker.orders_snapshot()) == 1

    resumed = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="crash-recovery",
        resume=True,
    )
    # Recovery remains available while the original decision and quote are
    # still live.  The deterministic client id resolves to the submitted order
    # rather than creating a second broker mutation.
    wall[0] = datetime(2026, 1, 5, 15, 4, tzinfo=timezone.utc)
    result = resumed.run_cycle(*args, execution_prices=prices)
    assert result["state"] == "COMPLETED"
    assert resumed.status().cycles == 1
    assert len(resumed.broker.orders_snapshot()) == 1


@pytest.mark.parametrize(("as_of", "quote_at", "resume_at"), [
    (
        "2026-01-05T15:00:00+00:00",
        "2026-01-05T14:59:30+00:00",
        datetime(2026, 1, 5, 15, 6, tzinfo=timezone.utc),
    ),
    (
        "2026-01-05T20:59:00+00:00",
        "2026-01-05T20:58:30+00:00",
        datetime(2026, 1, 5, 21, 1, tzinfo=timezone.utc),
    ),
])
def test_started_cycle_cannot_mutate_broker_after_live_window_expires(
        tmp_path, monkeypatch, as_of, quote_at, resume_at):
    registry, research = _registry_with_paper_approval(tmp_path)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda _error_type: CODE_SHA,
    )
    wall = [datetime.fromisoformat(as_of)]
    monkeypatch.setattr(
        "deployment.rehearsal._wall_clock", lambda: wall[0])
    db = tmp_path / f"stale-{resume_at.hour}-{resume_at.minute}.db"
    rehearsal = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="stale-recovery",
        research_artifact_hash=research.artifact_hash,
    )
    # Simulate a process dying immediately after the durable STARTED
    # reservation, before reconciliation/evaluation reaches the broker.
    monkeypatch.setattr(
        rehearsal.market, "set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated pre-execution crash")),
    )
    args = (
        as_of,
        {"TEST": _momentum_frame("2026-01-02", entry=True)},
    )
    prices = _paper_quote(quote_at)
    with pytest.raises(RuntimeError, match="pre-execution crash"):
        rehearsal.run_cycle(*args, execution_prices=prices)
    assert rehearsal.broker.orders_snapshot() == []

    wall[0] = resume_at
    resumed = FoundationPaperRehearsal(
        registry=registry, db_path=str(db), run_id="stale-recovery",
        resume=True,
    )
    with pytest.raises(RehearsalStateError, match="live recovery window"):
        resumed.run_cycle(*args, execution_prices=prices)

    # The rejected recovery did not evaluate the desk, submit/process an
    # order, or turn the incomplete reservation into qualifying evidence.
    assert resumed.broker.orders_snapshot() == []
    assert resumed.status().cycles == 0
    with pytest.raises(RehearsalStateError, match="incomplete cycle"):
        resumed.finalize()


def _passing_paper(
        registry, research, *, final_session: date = date(2026, 7, 10)):
    evidence = authoritative_paper_evidence(
        research, fills=2, final_session=final_session)
    artifact = PaperValidationArtifact.create(
        research_artifact=research,
        run_summary={
            "cycles": 20, "sessions": 15, "fills": 2, "errors": 0,
            "prospective": True,
        },
        reconciliation_evidence={
            "checks": 41, "failures": 0,
            "unknown_orders": 0, "open_orders": 0,
        },
        audit_verified=True,
        evidence=evidence,
    )
    registry.paper_validation_store.persist(artifact)
    replication = persist_passing_replication(registry, research)
    registry.promote(
        "foundation", research.artifact_hash,
        PromotionLevel.LIVE_ELIGIBLE,
        actor="live-approver", paper_artifact_hash=artifact.artifact_hash,
        replication_artifact_hash=replication.evidence_hash,
    )
    return artifact


def _manifest(research, paper, expires=None):
    return DeploymentManifest.create(
        research_artifact_hash=research.artifact_hash,
        paper_evidence_hash=paper.artifact_hash,
        strategy_version=research.strategy_version,
        code_sha=research.code_sha,
        config_hash=_config().config_hash,
        account_id_key="ACCOUNT-1",
        allowed_universe=["TEST"],
        expires_at=(expires or "2030-01-01T00:00:00+00:00"),
        created_by="deployer",
        max_order_notional=1_000.0,
        max_daily_notional=1_400.0,
        max_daily_orders=2,
    )


def test_deployment_store_requires_two_people_and_enforces_order_limits(
        tmp_path):
    registry, research = _registry_with_paper_approval(tmp_path)
    paper = _passing_paper(registry, research)
    audit = AuditLog(str(tmp_path / "live.db"), env="production")
    store = DeploymentStore(str(tmp_path / "live.db"), audit)
    manifest = _manifest(research, paper)

    store.stage(manifest)
    store.approve_risk(manifest.manifest_hash, "alice")
    with pytest.raises(DeploymentStateError, match="must differ"):
        store.approve_operations(manifest.manifest_hash, "alice")
    store.approve_operations(manifest.manifest_hash, "bob")
    store.activate(manifest.manifest_hash, "operator")
    store.arm(manifest.manifest_hash, "operator")

    # The immediate first cycle executes while ARMED; exact manifest limits
    # must therefore authorize its opening order before RUNNING is granted.
    store.authorize_order(
        manifest.manifest_hash, intent_id="intent1", side="BUY",
        symbol="TEST", quantity=5, reference_price=100.0,
        trading_date="2026-07-13", opening=True,
    )
    store.mark_running(manifest.manifest_hash, "operator")
    assert store.authorize_order(
        manifest.manifest_hash, intent_id="intent1", side="BUY",
        symbol="TEST", quantity=5, reference_price=100.0,
        trading_date="2026-07-13", opening=True,
    )["idempotent"] is True
    with pytest.raises(DeploymentStateError, match="max_daily_notional"):
        store.authorize_order(
            manifest.manifest_hash, intent_id="intent2", side="BUY",
            symbol="TEST", quantity=10, reference_price=100.0,
            trading_date="2026-07-13", opening=True,
        )
    # Exits remain possible even after opening capacity is exhausted.
    store.authorize_order(
        manifest.manifest_hash, intent_id="exit1", side="SELL",
        symbol="TEST", quantity=50, reference_price=100.0,
        trading_date="2026-07-13", opening=False,
    )


def test_production_context_rejects_raw_unverified_automation():
    context = object.__new__(LiveExecutionContext)
    context.identity = LiveContextIdentity(
        db_path="/tmp/x", env="production", account_id_key="A",
        auth_manager_id=1,
    )
    with pytest.raises(ValueError, match="verified deployment"):
        context.configure_session(
            portfolio=object(), data_fn=lambda: {}, desk=object())
    with pytest.raises(TypeError, match="only be created by preflight"):
        VerifiedFoundationDeployment(
            manifest=object(), context_identity=object(), desk=object(),
            portfolio=object(), data_fn=lambda: {}, execution_guard=object(),
            checked_at="now", _token=object(),
        )


class _Kill:
    def __init__(self):
        self.value = True

    def engaged(self):
        return self.value

    def engage(self, _reason, _actor):
        self.value = True

    def disengage(self, _actor):
        self.value = False


class _Scheduler:
    def __init__(self):
        self.running = False
        self.hold_after_first_cycle = True
        self.max_consecutive_errors = 1
        self.first_result = {"status": "ok", "reports": []}
        self.wait_error = None
        self.on_wait = None
        self.released = False

    def status(self):
        return {"running": self.running}

    def start(self):
        self.running = True
        return True

    def wait_for_first_cycle(self, _timeout):
        if self.on_wait is not None:
            self.on_wait()
        if self.wait_error is not None:
            raise self.wait_error
        return self.first_result

    def release_after_first_cycle(self):
        if self.released:
            return False
        self.released = True
        return True

    def stop(self):
        was = self.running
        self.running = False
        return was


class _Broker:
    def get_portfolio_status(self):
        return {"cash": 100_000.0, "portfolio_value": 100_000.0,
                "positions": []}

    def get_current_quote(self, _symbol):
        return {
            "bid": 99.9, "ask": 100.1, "last": 100.0,
            "observed_at": "2026-07-13T13:59:30+00:00",
            "quote_status": "REALTIME",
        }


class _Context:
    def __init__(self, db, audit, book):
        self.identity = LiveContextIdentity(
            db_path=str(db), env="production", account_id_key="ACCOUNT-1",
            auth_manager_id=1,
        )
        self.state = "ready"
        self.session = None
        self.scheduler = None
        self.verified_deployment = None
        self.kill_switch = _Kill()
        self.audit = audit
        self.auth_manager = SimpleNamespace(
            status=lambda: {"state": "connected"})
        self.local_book = book
        self.reservation_gate = SimpleNamespace(policy=SimpleNamespace(
            gross_nav_multiple=1.0, per_name_nav_fraction=0.10))
        self.broker = _Broker()
        self.reconciliation_calls = 0
        self.reconciliation_failure_call = None

    def working_orders(self):
        return []

    def reservation_snapshot(self):
        return {"reservations": []}

    def run_reconciliation(self, cash_tolerance=0.01):
        assert cash_tolerance == 0.01
        self.reconciliation_calls += 1
        if self.reconciliation_calls == self.reconciliation_failure_call:
            return {"ok": False, "mismatches": [{"type": "cash"}],
                    "checked_at": "2026-07-13T14:00:00+00:00"}
        return {"ok": True, "mismatches": [],
                "checked_at": "2026-07-13T14:00:00+00:00"}

    def configure_session(self, *, portfolio, data_fn, desk,
                          interval_minutes, verified_deployment):
        verified_deployment.bind(self, desk, interval_minutes)
        self.session = SimpleNamespace(
            portfolio=portfolio, data_fn=data_fn, desk=desk)
        self.scheduler = _Scheduler()
        self.verified_deployment = verified_deployment


def _live_harness(tmp_path, monkeypatch, clock):
    registry, research = _registry_with_paper_approval(tmp_path)
    paper = _passing_paper(registry, research)
    db = tmp_path / "controlled-live.db"
    audit = AuditLog(str(db), env="production")
    store = DeploymentStore(str(db), audit)
    manifest = _manifest(research, paper, "2027-01-01T00:00:00+00:00")
    store.stage(manifest)
    store.approve_risk(manifest.manifest_hash, "alice")
    store.approve_operations(manifest.manifest_hash, "bob")

    from brokers.local_book import LocalBook
    book = LocalBook(str(db), env="production", account_id_key="ACCOUNT-1")
    book.bootstrap_snapshot({}, 100_000.0)
    context = _Context(db, audit, book)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha",
        lambda _error_type: CODE_SHA,
    )
    controller = FoundationLiveController(
        store=store, promotion_registry=registry, context=context,
        clock=clock,
        provenance_fn=lambda: {"git_sha": CODE_SHA, "git_dirty": False},
    )
    data = {"TEST": _momentum_frame("2026-07-10", entry=True)}
    verified = controller.prepare(
        manifest.manifest_hash, data_fn=lambda: data, actor="operator")
    assert verified.desk.model_checkpoint_state() \
        == paper.evidence["model_checkpoint"]["state"]
    return controller, context, store, manifest, verified


@pytest.mark.parametrize("change", [
    {"quote_status": "DELAYED"},
    {"observed_at": "2026-07-13T13:58:00+00:00"},
    {"observed_at": None},
])
def test_controlled_live_quote_must_be_realtime_and_fresh(change):
    quote = {
        "bid": 99.9, "ask": 100.1, "last": 100.0,
        "observed_at": "2026-07-13T13:59:30+00:00",
        "quote_status": "REALTIME",
    }
    quote.update(change)
    with pytest.raises(DeploymentStateError):
        validate_realtime_equity_quote(
            quote, symbol="TEST",
            now=datetime(2026, 7, 13, 14, tzinfo=timezone.utc))


def test_live_prepare_restore_failure_never_activates(
        tmp_path, monkeypatch):
    registry, research = _registry_with_paper_approval(tmp_path)
    paper = _passing_paper(registry, research)
    db = tmp_path / "restore-failure.db"
    audit = AuditLog(str(db), env="production")
    store = DeploymentStore(str(db), audit)
    manifest = _manifest(research, paper, "2027-01-01T00:00:00+00:00")
    store.stage(manifest)
    store.approve_risk(manifest.manifest_hash, "alice")
    store.approve_operations(manifest.manifest_hash, "bob")
    from brokers.local_book import LocalBook
    book = LocalBook(str(db), env="production", account_id_key="ACCOUNT-1")
    book.bootstrap_snapshot({}, 100_000.0)
    context = _Context(db, audit, book)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha", lambda _error_type: CODE_SHA)

    def reject_checkpoint(_self, _checkpoint):
        raise ValueError("incompatible checkpoint")

    monkeypatch.setattr(
        FoundationDesk, "restore_model_checkpoint", reject_checkpoint)
    controller = FoundationLiveController(
        store=store, promotion_registry=registry, context=context,
        clock=lambda: datetime(2026, 7, 13, 14, tzinfo=timezone.utc),
        provenance_fn=lambda: {"git_sha": CODE_SHA, "git_dirty": False},
    )
    with pytest.raises(
            LiveDeploymentPreflightError, match="checkpoint is incompatible"):
        controller.prepare(
            manifest.manifest_hash,
            data_fn=lambda: {"TEST": _momentum_frame("2026-07-10", entry=True)},
            actor="operator")
    assert store.get(manifest.manifest_hash)["state"] \
        is DeploymentState.OPS_APPROVED
    assert context.session is None


def test_live_prepare_rejects_stale_paper_handoff(tmp_path, monkeypatch):
    registry, research = _registry_with_paper_approval(tmp_path)
    paper = _passing_paper(
        registry, research, final_session=date(2026, 7, 8))
    db = tmp_path / "stale-handoff.db"
    audit = AuditLog(str(db), env="production")
    store = DeploymentStore(str(db), audit)
    manifest = _manifest(research, paper, "2027-01-01T00:00:00+00:00")
    store.stage(manifest)
    store.approve_risk(manifest.manifest_hash, "alice")
    store.approve_operations(manifest.manifest_hash, "bob")
    from brokers.local_book import LocalBook
    book = LocalBook(str(db), env="production", account_id_key="ACCOUNT-1")
    book.bootstrap_snapshot({}, 100_000.0)
    context = _Context(db, audit, book)
    monkeypatch.setattr(
        "desks.registry._current_clean_code_sha", lambda _error_type: CODE_SHA)
    controller = FoundationLiveController(
        store=store, promotion_registry=registry, context=context,
        clock=lambda: datetime(2026, 7, 13, 14, tzinfo=timezone.utc),
        provenance_fn=lambda: {"git_sha": CODE_SHA, "git_dirty": False},
    )
    with pytest.raises(LiveDeploymentPreflightError, match="handoff is stale"):
        controller.prepare(
            manifest.manifest_hash,
            data_fn=lambda: {"TEST": _momentum_frame("2026-07-10", entry=True)},
            actor="operator")
    assert store.get(manifest.manifest_hash)["state"] \
        is DeploymentState.OPS_APPROVED


def test_live_controller_preflights_starts_and_pauses_exact_manifest(
        tmp_path, monkeypatch):
    def clock():
        return datetime(2026, 7, 13, 14, tzinfo=timezone.utc)
    controller, context, store, manifest, verified = _live_harness(
        tmp_path, monkeypatch, clock)
    assert store.get(manifest.manifest_hash)["state"] \
        is DeploymentState.ACTIVATED
    assert context.kill_switch.engaged() is True
    states_during_first_cycle = []
    context.scheduler.on_wait = lambda: states_during_first_cycle.append(
        store.get(manifest.manifest_hash)["state"])
    assert controller.start(verified, actor="operator") is True
    assert states_during_first_cycle == [DeploymentState.ARMED]
    assert store.get(manifest.manifest_hash)["state"] \
        is DeploymentState.RUNNING
    assert context.kill_switch.engaged() is False
    assert context.scheduler.released is True

    verified.execution_guard(
        intent=SimpleNamespace(
            asset=SimpleNamespace(symbol="TEST", asset_type=AssetType.STOCK),
            action="BUY", intent_id="target123"),
        side="BUY", quantity=5, now=clock(),
    )
    result = controller.pause(
        verified, actor="operator", reason="planned observation stop")
    assert result["state"] == "paused"
    assert context.kill_switch.engaged() is True


@pytest.mark.parametrize("first_result", [
    {"status": "halted", "reason": "intent_generation_error"},
    {"status": "pending", "reason": "unexpected_working_order"},
    {"status": "error", "reason": "evaluation_exception"},
])
def test_live_first_cycle_failure_kills_stops_and_pauses(
        tmp_path, monkeypatch, first_result):
    def clock():
        return datetime(2026, 7, 13, 14, tzinfo=timezone.utc)
    controller, context, store, manifest, verified = _live_harness(
        tmp_path, monkeypatch, clock)
    context.scheduler.first_result = first_result

    with pytest.raises(LiveDeploymentPreflightError,
                       match="first live cycle was not acceptable"):
        controller.start(verified, actor="operator")

    assert context.kill_switch.engaged() is True
    assert context.scheduler.running is False
    assert store.get(manifest.manifest_hash)["state"] is DeploymentState.PAUSED


def test_live_first_cycle_timeout_kills_stops_and_pauses(
        tmp_path, monkeypatch):
    def clock():
        return datetime(2026, 7, 13, 14, tzinfo=timezone.utc)
    controller, context, store, manifest, verified = _live_harness(
        tmp_path, monkeypatch, clock)
    context.scheduler.wait_error = TimeoutError("first cycle timed out")

    with pytest.raises(TimeoutError, match="first cycle timed out"):
        controller.start(
            verified, actor="operator", first_cycle_timeout_seconds=0.01)

    assert context.kill_switch.engaged() is True
    assert context.scheduler.running is False
    assert store.get(manifest.manifest_hash)["state"] is DeploymentState.PAUSED


def test_live_first_cycle_must_reconcile_before_running(
        tmp_path, monkeypatch):
    def clock():
        return datetime(2026, 7, 13, 14, tzinfo=timezone.utc)
    controller, context, store, manifest, verified = _live_harness(
        tmp_path, monkeypatch, clock)
    # prepare is call 1, the pre-arm check is call 2, and the independently
    # gated post-first-cycle check is call 3.
    context.reconciliation_failure_call = 3

    with pytest.raises(LiveDeploymentPreflightError,
                       match="first-cycle reconciliation failed"):
        controller.start(verified, actor="operator")

    assert context.reconciliation_calls == 3
    assert context.kill_switch.engaged() is True
    assert context.scheduler.running is False
    assert context.scheduler.released is False
    assert store.get(manifest.manifest_hash)["state"] is DeploymentState.PAUSED


def test_live_start_outside_market_hours_never_arms(
        tmp_path, monkeypatch):
    def clock():
        return datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    controller, context, store, manifest, verified = _live_harness(
        tmp_path, monkeypatch, clock)

    with pytest.raises(LiveDeploymentPreflightError,
                       match="only during NYSE market hours"):
        controller.start(verified, actor="operator")

    assert context.kill_switch.engaged() is True
    assert context.scheduler.running is False
    assert store.get(manifest.manifest_hash)["state"] \
        is DeploymentState.ACTIVATED


@pytest.mark.parametrize("mutate", [
    lambda frame: pd.concat([frame, frame.iloc[[-1]]]),
    lambda frame: frame.iloc[::-1],
])
def test_live_data_rejects_duplicate_or_nonmonotonic_indexes(mutate):
    def clock():
        return datetime(2026, 7, 13, 14, tzinfo=timezone.utc)
    frame = mutate(_momentum_frame("2026-07-10", entry=True))
    bounded = _BoundedLiveData(
        lambda: {"TEST": frame},
        SimpleNamespace(allowed_universe=("TEST",)), clock,
    )

    with pytest.raises(LiveDeploymentPreflightError,
                       match="unique and monotonic increasing"):
        bounded()


def test_live_data_requires_exact_previous_completed_exchange_session():
    def clock():
        return datetime(2026, 7, 13, 14, tzinfo=timezone.utc)
    stale = _momentum_frame("2026-07-09", entry=True)
    bounded = _BoundedLiveData(
        lambda: {"TEST": stale},
        SimpleNamespace(allowed_universe=("TEST",)), clock,
    )

    with pytest.raises(LiveDeploymentPreflightError,
                       match="must end on previous exchange session 2026-07-10"):
        bounded()


def test_live_manifest_paper_hash_cannot_be_substituted(tmp_path):
    registry, research = _registry_with_paper_approval(tmp_path)
    paper = _passing_paper(registry, research)
    with pytest.raises(PromotionNotApproved, match="different paper evidence"):
        registry.require_live_approved(
            "foundation", research.artifact_hash, "f" * 64)
    assert _manifest(research, paper).paper_evidence_hash == paper.artifact_hash


def test_controller_recovery_pauses_interrupted_running_deployment(tmp_path):
    registry, research = _registry_with_paper_approval(tmp_path)
    paper = _passing_paper(registry, research)
    db = tmp_path / "restart-live.db"
    audit = AuditLog(str(db), env="production")
    store = DeploymentStore(str(db), audit)
    manifest = _manifest(research, paper)
    store.stage(manifest)
    store.approve_risk(manifest.manifest_hash, "alice")
    store.approve_operations(manifest.manifest_hash, "bob")
    store.activate(manifest.manifest_hash, "operator")
    store.arm(manifest.manifest_hash, "operator")
    store.mark_running(manifest.manifest_hash, "operator")

    from brokers.local_book import LocalBook
    book = LocalBook(str(db), env="production", account_id_key="ACCOUNT-1")
    book.bootstrap_snapshot({}, 100_000.0)
    context = _Context(db, audit, book)
    context.kill_switch.value = False  # state left by the interrupted process

    FoundationLiveController(
        store=store, promotion_registry=registry, context=context)

    assert context.kill_switch.engaged() is True
    assert store.get(manifest.manifest_hash)["state"] is DeploymentState.PAUSED
