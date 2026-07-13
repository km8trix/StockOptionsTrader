#!/usr/bin/env python
"""Operator CLI for the Foundation research -> paper -> live evidence lane.

Nothing here auto-promotes or auto-starts trading.  Each command performs one
explicit transition and prints the exact content hash/state it affected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.foundation_research import (  # noqa: E402
    FoundationResearchSpec,
    persist_foundation_research,
    run_foundation_research,
)
from analysis.promotion import (  # noqa: E402
    PromotionLevel,
    PromotionNotApproved,
    PromotionRegistry,
)
from data.market_data import MarketDataHandler  # noqa: E402
from data.pit_warehouse import PitWarehouse  # noqa: E402
from data.warehouse_feed import WarehouseMarketData  # noqa: E402
from deployment.rehearsal import FoundationPaperRehearsal  # noqa: E402
from deployment.state import DeploymentManifest, DeploymentStore  # noqa: E402
from desks.deployment_config import (  # noqa: E402
    FoundationDeploymentConfig,
    foundation_target_v1_config,
)
from utils.audit import AuditLog  # noqa: E402
from utils.market_hours import MarketHours, NYSE_TZ  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "var" / "foundation-promotion"
DEFAULT_DB = ROOT / "var" / "foundation-control.db"


def _json(value) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str))


def _registry(args) -> PromotionRegistry:
    return PromotionRegistry(Path(args.registry_root))


def _config(args) -> FoundationDeploymentConfig:
    base = foundation_target_v1_config().to_mapping()
    if hasattr(args, "capital_allocation") and args.capital_allocation is not None:
        base["capital_allocation"] = args.capital_allocation
    if hasattr(args, "gate_threshold") and args.gate_threshold is not None:
        base["gate_threshold"] = args.gate_threshold
    return FoundationDeploymentConfig.from_mapping(base)


def command_research(args) -> None:
    registry = _registry(args)
    warehouse = PitWarehouse(args.warehouse_dir)
    spec = FoundationResearchSpec(
        n_trials=args.trials,
        start=args.start,
        end=args.end,
        universe_as_of=args.universe_as_of,
        max_symbols=args.max_symbols,
        seed=args.seed,
    )
    output = run_foundation_research(
        spec,
        config=_config(args),
        warehouse=warehouse,
        market_data=WarehouseMarketData(warehouse),
    )
    persist_foundation_research(output, registry.store)
    _json({
        "artifact_hash": output.artifact.artifact_hash,
        "decision": output.artifact.decision.value,
        "artifact_path": str(registry.store.path_for(
            output.artifact.artifact_hash)),
        "universe_size": len(output.selected_universe),
        "data_version": output.artifact.payload["data_version"],
        "paper_approval_created": False,
        "live_approval_created": False,
    })


def command_inspect(args) -> None:
    registry = _registry(args)
    artifact = registry.store.load(args.artifact)
    paper_approved = live_approved = False
    paper_hash = None
    try:
        registry.require_approved(
            artifact.strategy_id, artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE)
        paper_approved = True
    except PromotionNotApproved:
        pass
    try:
        _, paper = registry.require_live_approved(
            artifact.strategy_id, artifact.artifact_hash)
        live_approved = True
        paper_hash = paper.artifact_hash
    except PromotionNotApproved:
        pass
    _json({
        "artifact_hash": artifact.artifact_hash,
        "strategy_id": artifact.strategy_id,
        "strategy_version": artifact.strategy_version,
        "decision": artifact.decision.value,
        "paper_approved": paper_approved,
        "live_approved": live_approved,
        "bound_paper_evidence_hash": paper_hash,
        "decision_checks": artifact.payload["decision"],
    })


def command_approve_paper(args) -> None:
    registry = _registry(args)
    path = registry.promote(
        "foundation", args.artifact, PromotionLevel.PAPER_ELIGIBLE,
        actor=args.actor)
    _json({"approved": "paper_eligible", "artifact_hash": args.artifact,
           "approval_path": str(path), "actor": args.actor})


def command_paper_start(args) -> None:
    rehearsal = FoundationPaperRehearsal(
        registry=_registry(args), db_path=args.db,
        run_id=args.run_id, research_artifact_hash=args.artifact,
        initial_capital=args.initial_capital,
    )
    _json(asdict(rehearsal.status()))


def _market_date(value) -> date:
    timestamp = pd.Timestamp(value)
    return timestamp.date()


def _previous_session(as_of: datetime) -> date:
    calendar = MarketHours()
    local = as_of.astimezone(NYSE_TZ)
    if not calendar.is_market_open(as_of):
        raise RuntimeError(
            "paper-step must run during the NYSE regular session")
    candidate = local.date() - timedelta(days=1)
    for _ in range(10):
        if calendar.is_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)
    raise RuntimeError("could not resolve the prior NYSE session")


def _cycle_data(research, requested_at: datetime) -> tuple[dict, dict]:
    """Fetch D signal history and uncached D+1 quote observations."""
    provider = MarketDataHandler()
    previous_session = _previous_session(requested_at)
    start = (previous_session - timedelta(days=450)).isoformat()
    result: dict = {}
    execution_prices: dict = {}
    for symbol in research.payload["universe"]:
        frame = provider.fetch_stock_data(
            symbol, start, previous_session.isoformat())
        if frame is None or frame.empty:
            raise RuntimeError(
                f"no live signal history for {symbol} through "
                f"{previous_session}")
        dates = pd.Index([_market_date(value) for value in frame.index])
        causal = frame.loc[dates <= previous_session].copy()
        if causal.empty or _market_date(causal.index.max()) != previous_session:
            raise RuntimeError(
                f"{symbol} has no completed signal bar for "
                f"{previous_session}")
        result[symbol] = provider.calculate_indicators(causal)

        quote = provider.fetch_stock_quote(symbol)
        if not isinstance(quote, dict):
            raise RuntimeError(
                f"no provider-timestamped execution quote for {symbol}")
        if not quote.get("observed_at") or not quote.get("source"):
            raise RuntimeError(
                f"execution quote for {symbol} lacks provider provenance")
        execution_prices[symbol] = dict(quote)
    return result, execution_prices


def command_paper_step(args) -> None:
    registry = _registry(args)
    rehearsal = FoundationPaperRehearsal(
        registry=registry, db_path=args.db, run_id=args.run_id, resume=True)
    requested_at = datetime.now(timezone.utc)
    data, execution_prices = _cycle_data(
        rehearsal.research, requested_at)
    as_of = datetime.now(timezone.utc)
    result = rehearsal.run_cycle(
        as_of, data, execution_prices=execution_prices)
    _json(result)


def command_paper_status(args) -> None:
    rehearsal = FoundationPaperRehearsal(
        registry=_registry(args), db_path=args.db,
        run_id=args.run_id, resume=True)
    _json(asdict(rehearsal.status()))


def command_paper_finalize(args) -> None:
    rehearsal = FoundationPaperRehearsal(
        registry=_registry(args), db_path=args.db,
        run_id=args.run_id, resume=True)
    artifact = rehearsal.finalize()
    _json({
        "paper_artifact_hash": artifact.artifact_hash,
        "research_artifact_hash": artifact.research_artifact_hash,
        "passed": artifact.passed,
        "decision": artifact.payload["decision"],
        "artifact_path": str(
            rehearsal.registry.paper_validation_store.path_for(
                artifact.artifact_hash)),
    })


def command_approve_live(args) -> None:
    registry = _registry(args)
    path = registry.promote(
        "foundation", args.artifact, PromotionLevel.LIVE_ELIGIBLE,
        actor=args.actor, paper_artifact_hash=args.paper_artifact)
    _json({
        "approved": "live_eligible",
        "artifact_hash": args.artifact,
        "paper_artifact_hash": args.paper_artifact,
        "actor": args.actor,
        "approval_path": str(path),
    })


def _store(args) -> DeploymentStore:
    db = os.path.realpath(args.db)
    return DeploymentStore(db, AuditLog(db, env="production"))


def command_manifest_stage(args) -> None:
    registry = _registry(args)
    research, paper = registry.require_live_approved(
        "foundation", args.artifact, args.paper_artifact)
    config = FoundationDeploymentConfig.from_mapping(research.parameters)
    manifest = DeploymentManifest.create(
        research_artifact_hash=research.artifact_hash,
        paper_evidence_hash=paper.artifact_hash,
        strategy_version=research.strategy_version,
        code_sha=research.code_sha,
        config_hash=config.config_hash,
        account_id_key=args.account,
        allowed_universe=research.payload["universe"],
        expires_at=args.expires_at,
        created_by=args.actor,
        max_order_notional=args.max_order_notional,
        max_daily_notional=args.max_daily_notional,
        max_daily_orders=args.max_daily_orders,
        max_gross_nav_multiple=args.max_gross_nav_multiple,
        max_per_name_nav_fraction=args.max_per_name_nav_fraction,
        interval_minutes=args.interval_minutes,
    )
    record = _store(args).stage(manifest)
    _json({"manifest_hash": manifest.manifest_hash,
           "state": record["state"].value,
           "payload": manifest.payload})


def command_manifest_approve(args, kind: str) -> None:
    store = _store(args)
    if kind == "risk":
        record = store.approve_risk(args.manifest, args.actor)
    else:
        record = store.approve_operations(args.manifest, args.actor)
    _json({"manifest_hash": args.manifest,
           "state": record["state"].value,
           "risk_approver": record["risk_approver"],
           "ops_approver": record["ops_approver"]})


def command_manifest_status(args) -> None:
    record = _store(args).get(args.manifest)
    _json({
        "manifest_hash": record["manifest"].manifest_hash,
        "state": record["state"].value,
        "version": record["version"],
        "risk_approver": record["risk_approver"],
        "ops_approver": record["ops_approver"],
        "reason": record["reason"],
        "updated_at": record["updated_at"],
        "payload": record["manifest"].payload,
    })


def _common(subparser) -> None:
    subparser.add_argument(
        "--registry-root", default=str(DEFAULT_REGISTRY),
        help="immutable artifact/approval directory")


def _paper_common(subparser) -> None:
    _common(subparser)
    subparser.add_argument("--db", default=str(DEFAULT_DB))
    subparser.add_argument("--run-id", required=True)


def _manifest_common(subparser) -> None:
    subparser.add_argument("--db", default=str(DEFAULT_DB))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    research = commands.add_parser("research", help="run strict PIT research")
    _common(research)
    research.add_argument("--warehouse-dir", default=None)
    research.add_argument("--trials", type=int, required=True)
    research.add_argument("--start", default="2015-01-01")
    research.add_argument("--end", default="2024-12-31")
    research.add_argument("--universe-as-of", default=None)
    research.add_argument("--max-symbols", type=int, default=100)
    research.add_argument("--seed", type=int, default=7)
    research.add_argument("--capital-allocation", type=float, default=None)
    research.add_argument("--gate-threshold", type=float, default=None)
    research.set_defaults(func=command_research)

    inspect = commands.add_parser("inspect", help="show evidence/approvals")
    _common(inspect)
    inspect.add_argument("--artifact", required=True)
    inspect.set_defaults(func=command_inspect)

    approve_paper = commands.add_parser("approve-paper")
    _common(approve_paper)
    approve_paper.add_argument("--artifact", required=True)
    approve_paper.add_argument("--actor", required=True)
    approve_paper.set_defaults(func=command_approve_paper)

    paper_start = commands.add_parser("paper-start")
    _paper_common(paper_start)
    paper_start.add_argument("--artifact", required=True)
    paper_start.add_argument("--initial-capital", type=float, default=100_000)
    paper_start.set_defaults(func=command_paper_start)

    paper_step = commands.add_parser("paper-step")
    _paper_common(paper_step)
    paper_step.set_defaults(func=command_paper_step)

    paper_status = commands.add_parser("paper-status")
    _paper_common(paper_status)
    paper_status.set_defaults(func=command_paper_status)

    paper_finalize = commands.add_parser("paper-finalize")
    _paper_common(paper_finalize)
    paper_finalize.set_defaults(func=command_paper_finalize)

    approve_live = commands.add_parser("approve-live")
    _common(approve_live)
    approve_live.add_argument("--artifact", required=True)
    approve_live.add_argument("--paper-artifact", required=True)
    approve_live.add_argument("--actor", required=True)
    approve_live.set_defaults(func=command_approve_live)

    stage = commands.add_parser("manifest-stage")
    _common(stage)
    _manifest_common(stage)
    stage.add_argument("--artifact", required=True)
    stage.add_argument("--paper-artifact", required=True)
    stage.add_argument("--account", required=True)
    stage.add_argument("--expires-at", required=True)
    stage.add_argument("--actor", required=True)
    stage.add_argument("--max-order-notional", type=float, default=2_500)
    stage.add_argument("--max-daily-notional", type=float, default=10_000)
    stage.add_argument("--max-daily-orders", type=int, default=5)
    stage.add_argument("--max-gross-nav-multiple", type=float, default=1.0)
    stage.add_argument("--max-per-name-nav-fraction", type=float, default=0.10)
    stage.add_argument("--interval-minutes", type=float, default=15)
    stage.set_defaults(func=command_manifest_stage)

    risk = commands.add_parser("manifest-approve-risk")
    _manifest_common(risk)
    risk.add_argument("--manifest", required=True)
    risk.add_argument("--actor", required=True)
    risk.set_defaults(func=lambda args: command_manifest_approve(args, "risk"))

    ops = commands.add_parser("manifest-approve-ops")
    _manifest_common(ops)
    ops.add_argument("--manifest", required=True)
    ops.add_argument("--actor", required=True)
    ops.set_defaults(func=lambda args: command_manifest_approve(args, "ops"))

    status = commands.add_parser("manifest-status")
    _manifest_common(status)
    status.add_argument("--manifest", required=True)
    status.set_defaults(func=command_manifest_status)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:  # one concise operator-safe failure surface
        _json({"ok": False, "error_type": type(exc).__name__,
               "error": str(exc)})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
