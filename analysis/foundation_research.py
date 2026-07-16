"""Authoritative research runner for the first deployable Foundation strategy.

The ordinary desk backtest script is useful for exploration, but its output is
not deployment evidence.  This module deliberately has a smaller, stricter
contract: one immutable Foundation configuration, one point-in-time universe,
one content-addressed warehouse snapshot, a pinned random seed, fail-closed
realistic fills capped at one percent of ADV, and a one-shot holdout whose
research-trial count is captured by the append-only integrity ledger.

The runner never grants an execution permission.  It returns a
``PromotionArtifact`` whose policy decision is an immutable fact; an operator
must persist and separately approve that exact hash before paper rehearsal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from analysis.promotion import (
    ArtifactStore,
    PromotionArtifact,
    PromotionLevel,
    PromotionPolicy,
    PromotionResults,
    _atomic_create,
    canonical_json,
)
from analysis.research_report_store import (
    ResearchReportArtifact,
    ResearchReportStore,
    recompute_foundation_results,
)
from analysis.research_integrity import (
    HoldoutOpening,
    ResearchIntegrityLedger,
    TrialOutcome,
)
from backtesting.backtest_engine import BacktestEngine
from data.pit_warehouse import PitWarehouse
from data.warehouse_feed import WarehouseMarketData
from desks.deployment_config import (
    FoundationDeploymentConfig,
    foundation_target_v1_config,
)
from utils.provenance import capture_run_provenance


REQUIRED_SNAPSHOT_TABLES = ("actions", "daily", "sep", "tickers")
FOUNDATION_UNIVERSE_METHOD = (
    "full_live_universe_fixed_requested_date_ranked_on_prior_observable_"
    "session_complete_market_cap"
)


@dataclass(frozen=True)
class ResearchRegime:
    name: str
    start: str
    end: str

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("regime name is required")
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("regime start cannot be after end")


DEFAULT_REGIMES = (
    ResearchRegime("2018_q4_selloff", "2018-10-01", "2018-12-31"),
    ResearchRegime("2019_bull", "2019-01-01", "2019-12-31"),
    ResearchRegime("2020_covid", "2020-02-01", "2020-12-31"),
    ResearchRegime("2021_mania", "2021-01-01", "2021-12-31"),
    ResearchRegime("2022_bear", "2022-01-01", "2022-12-31"),
    ResearchRegime("2023_2024_ai_bull", "2023-01-01", "2024-12-31"),
)


@dataclass(frozen=True)
class FoundationResearchSpec:
    """Every mutable input to the qualifying research run.

    ``n_trials`` has no implicit default, but the authoritative runner does not
    trust it: the value must equal the count captured atomically when the
    ledger opens the sealed holdout.  It remains on this low-level spec so DSR
    construction and legacy artifact tests are explicit.  The operator CLI has
    no manual trial-count option. V1 uses the universe observable on the first
    day and keeps it fixed for the run. Delisted constituents remain available
    through the PIT feed.
    """

    n_trials: int
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    universe_as_of: str | None = None
    max_symbols: int = 100
    seed: int = 7
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage_bps: float = 5.0
    impact_coef: float = 0.01
    participation_cap: float = 0.01
    adv_window: int = 20
    reject_fills_without_adv: bool = True
    regimes: tuple[ResearchRegime, ...] = DEFAULT_REGIMES

    def __post_init__(self) -> None:
        start_ts, end_ts = pd.Timestamp(self.start), pd.Timestamp(self.end)
        selection_ts = pd.Timestamp(self.universe_as_of or self.start)
        if any(pd.isna(value) for value in (start_ts, end_ts, selection_ts)):
            raise ValueError("research dates must be valid calendar dates")
        start, end = start_ts.date(), end_ts.date()
        selection = selection_ts.date()
        if start > end:
            raise ValueError("research start cannot be after end")
        if selection > start:
            raise ValueError(
                "universe_as_of cannot be after the research start")
        if int(self.n_trials) < 1:
            raise ValueError("n_trials must include every attempted variant")
        if int(self.max_symbols) < 1:
            raise ValueError("max_symbols must be positive")
        if isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not math.isfinite(float(self.initial_capital)) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if not 0 <= float(self.commission) < 1:
            raise ValueError("commission must be in [0, 1)")
        if not 0 <= float(self.slippage_bps) <= 10_000:
            raise ValueError("slippage_bps must be in [0, 10000]")
        if not 0 <= float(self.impact_coef) <= 1:
            raise ValueError("impact_coef must be in [0, 1]")
        if not 0 < float(self.participation_cap) <= 0.01:
            raise ValueError(
                "participation_cap must be in (0, 0.01] for qualifying "
                "research")
        if int(self.adv_window) < 1:
            raise ValueError("adv_window must be positive")
        if self.reject_fills_without_adv is not True:
            raise ValueError(
                "qualifying research must reject fills without valid ADV")
        if not self.regimes:
            raise ValueError("at least one regime is required")
        names = [regime.name for regime in self.regimes]
        if len(names) != len(set(names)):
            raise ValueError("research regime names must be unique")
        for regime in self.regimes:
            regime_start = pd.Timestamp(regime.start).date()
            regime_end = pd.Timestamp(regime.end).date()
            if regime_start < start or regime_end > end:
                raise ValueError(
                    f"research regime {regime.name!r} is outside the run window")

    @property
    def selection_date(self) -> str:
        return pd.Timestamp(self.universe_as_of or self.start).date().isoformat()

    def engine_parameters(self) -> dict[str, Any]:
        return {
            "initial_capital": float(self.initial_capital),
            "commission": float(self.commission),
            "slippage_bps": float(self.slippage_bps),
            "enable_realistic_fills": True,
            "impact_coef": float(self.impact_coef),
            "participation_cap": float(self.participation_cap),
            "adv_window": int(self.adv_window),
            "reject_fills_without_adv": True,
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class FoundationResearchOutput:
    artifact: PromotionArtifact
    report: Mapping[str, Any]
    selected_universe: tuple[str, ...]
    data_snapshot: Mapping[str, Any]
    integrity_opening: HoldoutOpening | None = None


def foundation_trial_inputs(
        spec: FoundationResearchSpec,
        config: FoundationDeploymentConfig,
        policy: PromotionPolicy | None = None) -> dict[str, Any]:
    """Canonical material inputs that must be registered before a holdout.

    The integrity ledger already binds a trial to code, data, seed, candidate,
    and protocol identities.  This payload closes the remaining gap by binding
    every mutable Foundation research and deployment parameter to that trial.
    """
    effective_policy = policy or PromotionPolicy.default()
    return {
        "research_window": {
            "start": pd.Timestamp(spec.start).date().isoformat(),
            "end": pd.Timestamp(spec.end).date().isoformat(),
        },
        "universe": {
            "requested_as_of": spec.selection_date,
            "max_symbols": int(spec.max_symbols),
            "method": FOUNDATION_UNIVERSE_METHOD,
        },
        "engine_parameters": spec.engine_parameters(),
        "strategy_parameters": config.to_mapping(),
        "regimes": [asdict(regime) for regime in spec.regimes],
        "promotion_policy_id": effective_policy.policy_id,
    }


def _require_frozen_foundation_trial(
        *, ledger: ResearchIntegrityLedger, protocol_hash: str,
        trial_hash: str, holdout_id: str, data_version: str,
        spec: FoundationResearchSpec, config: FoundationDeploymentConfig,
        policy: PromotionPolicy, provenance: Mapping[str, Any]) -> None:
    """Fail unless the run exactly matches its frozen semantic contract."""
    protocol = ledger.get_protocol(protocol_hash)
    trial = ledger.get_trial(trial_hash)
    protocol_payload = protocol.payload
    trial_payload = trial.payload
    code_version = str(provenance["git_sha"])
    if trial_payload["protocol_hash"] != protocol.protocol_hash:
        raise RuntimeError("Foundation trial belongs to a different protocol")
    if trial_payload["family"] != "foundation":
        raise RuntimeError("Foundation trial family is not 'foundation'")
    if trial_payload["candidate_id"] != config.strategy_version:
        raise RuntimeError("Foundation trial candidate differs from the strategy")
    if trial_payload["seed"] != int(spec.seed):
        raise RuntimeError("Foundation trial seed differs from the research run")
    if (trial_payload["code_version"] != code_version
            or protocol_payload["code_version"] != code_version):
        raise RuntimeError("Foundation frozen code differs from run provenance")
    if trial_payload["data_version"] != f"pit-sha256:{data_version}":
        raise RuntimeError("Foundation trial data differs from the warehouse snapshot")
    if canonical_json(trial_payload["inputs"]) != canonical_json(
            foundation_trial_inputs(spec, config, policy)):
        raise RuntimeError("Foundation run inputs differ from the registered trial")

    try:
        holdout = protocol_payload["holdouts"][holdout_id]
    except KeyError as exc:
        raise RuntimeError("Foundation holdout is not declared by the protocol") \
            from exc
    expected_holdout = {
        "start": pd.Timestamp(spec.start).date().isoformat(),
        "end": pd.Timestamp(spec.end).date().isoformat(),
        "data_artifact_hash": data_version,
        "sealed": True,
    }
    if canonical_json(holdout) != canonical_json(expected_holdout):
        raise RuntimeError("Foundation holdout specification is not exact")
    required_data_plan = {
        "point_in_time": True,
        "snapshot_hash": data_version,
        "universe_selection_method": FOUNDATION_UNIVERSE_METHOD,
    }
    if canonical_json(protocol_payload["data_plan"]) != canonical_json(
            required_data_plan):
        raise RuntimeError("Foundation protocol data plan is not exact")
    required_evaluation = {
        "primary_test": "aggregate_net_oos",
        "calendar_years": "diagnostic_only",
        "costs": "realized_execution",
        "independent_replication_required": True,
    }
    if canonical_json(protocol_payload["evaluation_plan"]) != canonical_json(
            required_evaluation):
        raise RuntimeError("Foundation protocol evaluation plan is not exact")
    required_decision = {
        "promotion_policy_id": policy.policy_id,
        "one_shot_holdout": True,
        "failure_is_terminal": True,
    }
    if canonical_json(protocol_payload["decision_rules"]) != canonical_json(
            required_decision):
        raise RuntimeError("Foundation protocol decision rules are not exact")


def _require_reproducible_provenance(value: Mapping[str, Any]) -> None:
    sha = value.get("git_sha")
    if not isinstance(sha, str) or len(sha) != 40 \
            or any(ch not in "0123456789abcdef" for ch in sha):
        raise RuntimeError("qualifying research requires a known full Git SHA")
    if value.get("git_dirty") is not False:
        raise RuntimeError("qualifying research requires a clean working tree")
    if value.get("seed") is None:
        raise RuntimeError("qualifying research requires a pinned RNG seed")
    if not isinstance(value.get("python_version"), str) \
            or not str(value["python_version"]).strip():
        raise RuntimeError("qualifying research requires the Python version")
    dependencies = value.get("dependency_versions")
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise RuntimeError("qualifying research requires dependency versions")
    missing = sorted(key for key, item in dependencies.items() if not item)
    if missing:
        raise RuntimeError(
            "qualifying research has unknown dependency versions: "
            + ", ".join(missing)
        )


def _snapshot_version(snapshot: Mapping[str, Any]) -> str:
    version = str(snapshot.get("version") or "")
    if len(version) != 64 or any(ch not in "0123456789abcdef" for ch in version):
        raise RuntimeError("warehouse snapshot has no valid SHA-256 version")
    if snapshot.get("complete") is not True:
        flags = snapshot.get("quality_flags") or []
        raise RuntimeError(f"warehouse snapshot is incomplete: {flags}")
    tables = snapshot.get("tables")
    if not isinstance(tables, list) or len(tables) != len(
            REQUIRED_SNAPSHOT_TABLES):
        raise RuntimeError("warehouse snapshot table manifest is incomplete")
    entries: list[tuple[str, str, int]] = []
    for item in tables:
        if not isinstance(item, Mapping):
            raise RuntimeError("warehouse snapshot table entry is invalid")
        table = item.get("table")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (not isinstance(table, str)
                or not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
                or type(size) is not int or size < 1):
            raise RuntimeError("warehouse snapshot table entry is invalid")
        entries.append((table, digest, size))
    if ({table for table, _, _ in entries} != set(REQUIRED_SNAPSHOT_TABLES)
            or len({table for table, _, _ in entries}) != len(entries)):
        raise RuntimeError("warehouse snapshot has the wrong required tables")
    manifest_digest = hashlib.sha256()
    for table, digest, size in sorted(entries):
        manifest_digest.update(f"{table}:{digest}:{size}\n".encode("utf-8"))
    if manifest_digest.hexdigest() != version:
        raise RuntimeError(
            "warehouse snapshot version differs from its table manifest")
    return version


def _select_universe(
        warehouse: PitWarehouse, spec: FoundationResearchSpec,
) -> tuple[tuple[str, ...], str, int]:
    """Select the fixed PIT universe using the latest observable cap session.

    Research often starts on a weekend or exchange holiday.  Membership is
    frozen at the operator-requested as-of date, but ranking must come from an
    actual DAILY observation available no later than that date.  Missing cap
    coverage is a hard failure: sorting uncovered names alphabetically would
    silently change the claimed largest-name universe.
    """
    available = warehouse.universe_asof(
        spec.selection_date,
    )
    candidates = sorted({str(symbol).strip().upper()
                         for symbol in available if str(symbol).strip()})
    if not candidates:
        raise RuntimeError(
            f"PIT universe is empty at {spec.selection_date}; ingest tickers"
        )

    requested = pd.Timestamp(spec.selection_date).date()
    caps: dict[str, float] = {}
    resolved: str | None = None
    # Ten days covers the longest normal NYSE closure; fourteen gives a small
    # operational cushion while still refusing a materially stale ranking.
    for offset in range(15):
        candidate_date = (requested - timedelta(days=offset)).isoformat()
        raw_caps = warehouse.daily_marketcaps(candidates, candidate_date)
        if not raw_caps:
            continue
        invalid = []
        for symbol in candidates:
            try:
                value = float(raw_caps[symbol])
            except (KeyError, TypeError, ValueError):
                invalid.append(symbol)
                continue
            if not math.isfinite(value) or value <= 0:
                invalid.append(symbol)
                continue
            caps[symbol] = value
        if invalid:
            preview = ", ".join(invalid[:5])
            raise RuntimeError(
                "PIT market-cap coverage is incomplete at "
                f"{candidate_date}: {len(invalid)} of {len(candidates)} "
                f"eligible symbols are missing/invalid ({preview})"
            )
        resolved = candidate_date
        break
    if resolved is None:
        raise RuntimeError(
            "PIT market-cap ranking has no observable session within 14 days "
            f"on or before {spec.selection_date}"
        )

    candidates.sort(key=lambda symbol: (-caps[symbol], symbol))
    selected = tuple(candidates[:spec.max_symbols])
    if not selected:
        raise RuntimeError("PIT universe selection produced no symbols")
    return selected, resolved, len(candidates)


def _portfolio_series(report: Mapping[str, Any]) -> pd.Series:
    points: list[tuple[pd.Timestamp, float]] = []
    for row in report.get("portfolio_history", []):
        try:
            timestamp = pd.Timestamp(row["timestamp"])
            value = float(row["portfolio_value"])
        except (KeyError, TypeError, ValueError):
            continue
        if pd.notna(timestamp) and math.isfinite(value) and value > 0:
            points.append((timestamp, value))
    if len(points) < 2:
        raise RuntimeError("backtest produced fewer than two valid NAV snapshots")
    frame = pd.DataFrame(points, columns=["date", "value"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    return frame.set_index("date")["value"]


def _regime_returns(series: pd.Series,
                    regimes: Sequence[ResearchRegime]) -> dict[str, float]:
    results: dict[str, float] = {}
    for regime in regimes:
        window = series.loc[
            (series.index >= pd.Timestamp(regime.start))
            & (series.index <= pd.Timestamp(regime.end))
        ]
        if len(window) >= 2 and float(window.iloc[0]) > 0:
            results[regime.name] = float(window.iloc[-1] / window.iloc[0] - 1.0)
    return results


def _annual_turnover(report: Mapping[str, Any], series: pd.Series) -> float:
    notional = 0.0
    for trade in report.get("trades", []):
        try:
            quantity = abs(float(trade["quantity"]))
            price = abs(float(trade["price"]))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(quantity) and math.isfinite(price):
            notional += quantity * price
    elapsed_days = max(1, int((series.index[-1] - series.index[0]).days))
    years = elapsed_days / 365.25
    average_nav = float(series.mean())
    if not math.isfinite(average_nav) or average_nav <= 0:
        raise RuntimeError("cannot compute turnover from invalid average NAV")
    return float(notional / average_nav / years)


def _estimated_cost_bps(spec: FoundationResearchSpec) -> float:
    # Stress one side at the configured maximum participation.  Actual fills
    # pay the same square-root impact formula inside BacktestEngine.
    impact_bps = spec.impact_coef * math.sqrt(spec.participation_cap) * 10_000
    return float(spec.commission * 10_000 + spec.slippage_bps + impact_bps)


def _report_digest(report: Mapping[str, Any]) -> str:
    # This digest is also the address of the complete persisted report.  Do not
    # use ``default=str``: unknown evidence objects must fail explicitly.
    return ResearchReportArtifact.create(report).report_hash


def build_foundation_artifact(
        *, report: Mapping[str, Any], selected_universe: Sequence[str],
        data_snapshot: Mapping[str, Any], provenance: Mapping[str, Any],
        spec: FoundationResearchSpec,
        config: FoundationDeploymentConfig,
        universe_selection_date: str,
        eligible_symbol_count: int,
        policy: PromotionPolicy | None = None,
        integrity_opening: HoldoutOpening | None = None,
        integrity_program_id: str | None = None,
        integrity_head_hash: str | None = None) -> PromotionArtifact:
    """Convert a completed strict run into immutable promotion evidence.

    Without ``integrity_opening`` this produces a legacy v3 artifact for
    low-level reproducibility tests only; the promotion registry rejects it.
    The authoritative runner always supplies a verified opening and emits v4.
    """
    _require_reproducible_provenance(provenance)
    data_version = _snapshot_version(data_snapshot)
    requested_selection = pd.Timestamp(spec.selection_date).date()
    resolved_selection = pd.Timestamp(universe_selection_date).date()
    research_start = pd.Timestamp(spec.start).date()
    if resolved_selection > requested_selection or requested_selection > research_start:
        raise RuntimeError(
            "universe selection dates violate the research causality boundary")
    eligible_count = int(eligible_symbol_count)
    selected_count = len(tuple(selected_universe))
    if eligible_count < 1 or selected_count != min(spec.max_symbols, eligible_count):
        raise RuntimeError(
            "selected universe size does not match the fully ranked PIT universe")
    n_trials = int(spec.n_trials)
    integrity_evidence = None
    runner = "foundation_research_v3"
    if integrity_opening is not None:
        if not isinstance(integrity_opening, HoldoutOpening):
            raise TypeError("integrity_opening must be a HoldoutOpening")
        opening = integrity_opening.payload
        if opening["data_artifact_hash"] != data_version:
            raise RuntimeError(
                "opened holdout data differs from the research snapshot")
        n_trials = int(opening["program_trial_count_at_open"])
        if n_trials != int(spec.n_trials):
            raise RuntimeError(
                "research trial count differs from the ledger-derived count")
        if not isinstance(integrity_program_id, str) \
                or not integrity_program_id.strip():
            raise ValueError("integrity program_id is required")
        if not isinstance(integrity_head_hash, str) \
                or len(integrity_head_hash) != 64 \
                or any(ch not in "0123456789abcdef"
                       for ch in integrity_head_hash):
            raise ValueError("integrity ledger head must be a SHA-256 digest")
        integrity_evidence = {
            "program_id": integrity_program_id.strip(),
            "opening_hash": integrity_opening.opening_hash,
            "opening": opening,
            "ledger_head_hash": integrity_head_hash,
        }
        runner = "foundation_research_v4"

    results = recompute_foundation_results(
        report,
        n_trials=n_trials,
        engine_parameters=spec.engine_parameters(),
        regimes=[asdict(regime) for regime in spec.regimes],
    )
    evidence = {
        "runner": runner,
        "window": [
            pd.Timestamp(spec.start).date().isoformat(),
            pd.Timestamp(spec.end).date().isoformat(),
        ],
        "universe_selection": {
            "requested_as_of": requested_selection.isoformat(),
            "resolved_as_of": resolved_selection.isoformat(),
            "max_symbols": spec.max_symbols,
            "method": FOUNDATION_UNIVERSE_METHOD,
            "eligible_symbols": eligible_count,
            "ranked_symbols": eligible_count,
            "market_cap_coverage_complete": True,
        },
        "warehouse_snapshot": dict(data_snapshot),
        "engine_parameters": spec.engine_parameters(),
        "n_trials": n_trials,
        "regimes": [asdict(regime) for regime in spec.regimes],
        "report_sha256": _report_digest(report),
        "trade_count": len(report.get("trades", [])),
        "pending_signal_count": len(report.get("pending_signals", [])),
        "provenance": dict(provenance),
    }
    if integrity_evidence is not None:
        evidence["research_integrity"] = integrity_evidence
    dependency_versions = dict(provenance["dependency_versions"])
    dependency_versions["python"] = str(provenance["python_version"])
    return PromotionArtifact.create(
        strategy_id="foundation",
        strategy_version=config.strategy_version,
        data_version=f"pit-sha256:{data_version}",
        universe=selected_universe,
        parameters=config.to_mapping(),
        seed=int(spec.seed),
        dependency_versions=dependency_versions,
        code_sha=str(provenance["git_sha"]),
        results=results,
        policy=policy,
        evidence=evidence,
    )


def run_foundation_research(
        spec: FoundationResearchSpec, *,
        integrity_ledger: ResearchIntegrityLedger,
        protocol_hash: str,
        trial_hash: str,
        holdout_id: str,
        reviewer: str,
        config: FoundationDeploymentConfig | None = None,
        warehouse: PitWarehouse | None = None,
        market_data: WarehouseMarketData | None = None,
        provenance: Mapping[str, Any] | None = None,
        engine_factory: Callable[..., BacktestEngine] = BacktestEngine,
        policy: PromotionPolicy | None = None) -> FoundationResearchOutput:
    """Open one sealed holdout, then run the qualifying research path.

    The ledger opening occurs before universe selection or strategy execution.
    Its captured program-wide trial count is the only count accepted by the
    artifact and DSR calculation. An opened-but-undecided holdout may resume
    after an operational interruption; a decided holdout can never be rerun.
    """
    if not isinstance(integrity_ledger, ResearchIntegrityLedger):
        raise TypeError("integrity_ledger must be a ResearchIntegrityLedger")
    config = config or foundation_target_v1_config()
    effective_policy = policy or PromotionPolicy.default()
    warehouse = warehouse or PitWarehouse()
    market_data = market_data or WarehouseMarketData(warehouse)
    before = market_data.data_snapshot(REQUIRED_SNAPSHOT_TABLES)
    data_version = _snapshot_version(before)
    run_provenance = dict(provenance or capture_run_provenance(seed=spec.seed))
    _require_reproducible_provenance(run_provenance)
    if int(run_provenance["seed"]) != int(spec.seed):
        raise RuntimeError("provenance seed does not match research spec")
    _require_frozen_foundation_trial(
        ledger=integrity_ledger,
        protocol_hash=protocol_hash,
        trial_hash=trial_hash,
        holdout_id=holdout_id,
        data_version=data_version,
        spec=spec,
        config=config,
        policy=effective_policy,
        provenance=run_provenance,
    )
    if integrity_ledger.get_holdout_decision(
            protocol_hash=protocol_hash, holdout_id=holdout_id) is not None:
        raise RuntimeError("the sealed holdout already has a permanent decision")
    opening = integrity_ledger.get_holdout_opening(
        protocol_hash=protocol_hash, holdout_id=holdout_id)
    if opening is None:
        opening = integrity_ledger.open_holdout(
            protocol_hash=protocol_hash,
            holdout_id=holdout_id,
            trial_hash=trial_hash,
            data_artifact_hash=data_version,
            actor=reviewer,
        )
    else:
        opened = opening.payload
        if (opened["trial_hash"] != trial_hash
                or opened["data_artifact_hash"] != data_version):
            raise RuntimeError(
                "existing holdout opening differs from this research run")
    if int(spec.n_trials) != int(opening.payload[
            "program_trial_count_at_open"]):
        raise RuntimeError(
            "research spec trial count is not the ledger-derived opening count")
    ledger_head = integrity_ledger.head_hash
    if ledger_head is None:
        raise RuntimeError("research integrity ledger has no anchorable head")
    universe, selection_date, eligible_count = _select_universe(warehouse, spec)

    engine = engine_factory(
        desk=config.build(), market_data=market_data,
        **spec.engine_parameters(),
    )
    report = engine.run(
        list(universe), spec.start, spec.end, benchmark_symbol=None,
    )
    after = market_data.data_snapshot(REQUIRED_SNAPSHOT_TABLES)
    if canonical_json(before) != canonical_json(after):
        raise RuntimeError("warehouse snapshot changed during the research run")
    artifact = build_foundation_artifact(
        report=report, selected_universe=universe,
        data_snapshot=before, provenance=run_provenance,
        spec=spec, config=config,
        universe_selection_date=selection_date,
        eligible_symbol_count=eligible_count,
        policy=effective_policy,
        integrity_opening=opening,
        integrity_program_id=integrity_ledger.program_id,
        integrity_head_hash=ledger_head,
    )
    return FoundationResearchOutput(
        artifact=artifact, report=report,
        selected_universe=universe, data_snapshot=dict(before),
        integrity_opening=opening,
    )


def persist_foundation_research(output: FoundationResearchOutput,
                                store: ArtifactStore, *,
                                integrity_ledger: ResearchIntegrityLedger | None = None,
                                reviewer: str | None = None) -> str:
    """Persist raw evidence and permanently decide an opened holdout."""
    report = ResearchReportArtifact.create(output.report)
    expected = (output.artifact.evidence or {}).get("report_sha256")
    if report.report_hash != expected:
        raise RuntimeError(
            "raw research report does not match promotion evidence digest")
    ResearchReportStore(store.root).persist(report)
    store.persist(output.artifact)
    opening = output.integrity_opening
    if opening is not None:
        if not isinstance(integrity_ledger, ResearchIntegrityLedger):
            raise RuntimeError(
                "qualified research persistence requires its integrity ledger")
        expected_ledger_root = store.root / "research-integrity"
        if integrity_ledger.root.resolve() != expected_ledger_root.resolve():
            raise RuntimeError(
                "qualified research ledger must live inside the artifact registry")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("reviewer is required to decide the holdout")
        opening_payload = opening.payload
        existing = integrity_ledger.get_holdout_decision(
            protocol_hash=opening_payload["protocol_hash"],
            holdout_id=opening_payload["holdout_id"],
        )
        decision = ("pass" if output.artifact.decision
                    is not PromotionLevel.RESEARCH_ONLY else "fail")
        results = PromotionResults.from_dict(
            output.artifact.payload["results"])
        summary = {
            "promotion_level": output.artifact.decision.value,
            "psr": results.psr,
            "dsr": results.dsr,
            "cost_adjusted_return": results.cost_adjusted_return,
        }
        trial_hash = opening_payload["trial_hash"]
        outcome = integrity_ledger.get_trial_outcome(trial_hash)
        if outcome is None:
            outcome = integrity_ledger.record_trial_outcome(TrialOutcome.create(
                trial_hash=trial_hash,
                status="completed",
                result_summary=summary,
                evidence_hash=output.artifact.artifact_hash,
            ))
        elif (outcome.payload["status"] != "completed"
              or outcome.payload["evidence_hash"]
              != output.artifact.artifact_hash):
            raise RuntimeError(
                "trial already points to different terminal evidence")
        if existing is None:
            decision_record = integrity_ledger.record_holdout_decision(
                opening_hash=opening.opening_hash,
                decision=decision,
                result_summary=summary,
                result_artifact_hash=output.artifact.artifact_hash,
                actor=reviewer,
            )
        elif (existing.payload["decision"] != decision
              or existing.payload["result_artifact_hash"]
              != output.artifact.artifact_hash):
            raise RuntimeError(
                "holdout already points to different permanent evidence")
        else:
            decision_record = existing
        receipt_payload = {
            "schema_version": 1,
            "record_type": "research_integrity_receipt",
            "program_id": integrity_ledger.program_id,
            "artifact_hash": output.artifact.artifact_hash,
            "protocol_hash": opening_payload["protocol_hash"],
            "trial_hash": trial_hash,
            "opening_hash": opening.opening_hash,
            "outcome_hash": outcome.record_hash,
            "decision_hash": decision_record.record_hash,
            "opening_event_hash": integrity_ledger.event_hash_for(
                event_kind="holdout_opened",
                target_hash=opening.opening_hash,
            ),
            "terminal_event_hash": integrity_ledger.event_hash_for(
                event_kind="holdout_decided",
                target_hash=decision_record.record_hash,
            ),
        }
        receipt_hash = hashlib.sha256(
            canonical_json(receipt_payload).encode("utf-8")).hexdigest()
        receipt = canonical_json({
            "receipt_hash": receipt_hash,
            "payload": receipt_payload,
        }) + "\n"
        _atomic_create(
            store.root / "research-integrity-receipts"
            / f"{output.artifact.artifact_hash}.json",
            receipt,
        )
    return output.artifact.artifact_hash


__all__ = [
    "DEFAULT_REGIMES",
    "FoundationResearchOutput",
    "FoundationResearchSpec",
    "FOUNDATION_UNIVERSE_METHOD",
    "ResearchRegime",
    "build_foundation_artifact",
    "foundation_trial_inputs",
    "persist_foundation_research",
    "run_foundation_research",
]
