"""Authoritative research runner for the first deployable Foundation strategy.

The ordinary desk backtest script is useful for exploration, but its output is
not deployment evidence.  This module deliberately has a smaller, stricter
contract: one immutable Foundation configuration, one point-in-time universe,
one content-addressed warehouse snapshot, a pinned random seed, realistic
fills, and an explicit count of the research trials that preceded the run.

The runner never grants an execution permission.  It returns a
``PromotionArtifact`` whose policy decision is an immutable fact; an operator
must persist and separately approve that exact hash before paper rehearsal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from analysis.promotion import (
    ArtifactStore,
    PromotionArtifact,
    PromotionPolicy,
    PromotionResults,
    canonical_json,
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

    ``n_trials`` has no implicit default.  It is the operator's count of all
    strategy/model/parameter variants considered, including unsuccessful
    ones; forcing it into the command makes the DSR penalty hard to forget.
    V1 uses the universe observable on the first day and keeps it fixed for
    the run.  Delisted constituents remain available through the PIT feed.
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
    participation_cap: float = 0.10
    adv_window: int = 20
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
        if not 0 < float(self.participation_cap) <= 1:
            raise ValueError("participation_cap must be in (0, 1]")
        if int(self.adv_window) < 1:
            raise ValueError("adv_window must be positive")
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
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class FoundationResearchOutput:
    artifact: PromotionArtifact
    report: Mapping[str, Any]
    selected_universe: tuple[str, ...]
    data_snapshot: Mapping[str, Any]


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
    encoded = json.dumps(
        report, sort_keys=True, separators=(",", ":"), default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_foundation_artifact(
        *, report: Mapping[str, Any], selected_universe: Sequence[str],
        data_snapshot: Mapping[str, Any], provenance: Mapping[str, Any],
        spec: FoundationResearchSpec,
        config: FoundationDeploymentConfig,
        universe_selection_date: str,
        eligible_symbol_count: int,
        policy: PromotionPolicy | None = None) -> PromotionArtifact:
    """Convert a completed strict run into immutable promotion evidence."""
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
    series = _portfolio_series(report)
    returns = series.pct_change().dropna()
    if returns.empty or not all(math.isfinite(float(item)) for item in returns):
        raise RuntimeError("backtest did not produce a finite OOS return series")
    regimes = _regime_returns(series, spec.regimes)
    expected_regimes = {regime.name for regime in spec.regimes}
    if set(regimes) != expected_regimes:
        missing = sorted(expected_regimes - set(regimes))
        raise RuntimeError(
            "backtest does not cover every declared research regime: "
            + ", ".join(missing))
    results = PromotionResults.from_oos_returns(
        [float(item) for item in returns],
        [int(timestamp.year) for timestamp in returns.index],
        n_trials=spec.n_trials,
        cost_model_applied=True,
        estimated_cost_bps=_estimated_cost_bps(spec),
        annual_turnover=_annual_turnover(report, series),
        regime_results=regimes,
    )
    evidence = {
        "runner": "foundation_research_v2",
        "window": [
            pd.Timestamp(spec.start).date().isoformat(),
            pd.Timestamp(spec.end).date().isoformat(),
        ],
        "universe_selection": {
            "requested_as_of": requested_selection.isoformat(),
            "resolved_as_of": resolved_selection.isoformat(),
            "max_symbols": spec.max_symbols,
            "method": (
                "full_live_universe_fixed_requested_date_ranked_on_prior_"
                "observable_session_complete_market_cap"
            ),
            "eligible_symbols": eligible_count,
            "ranked_symbols": eligible_count,
            "market_cap_coverage_complete": True,
        },
        "warehouse_snapshot": dict(data_snapshot),
        "engine_parameters": spec.engine_parameters(),
        "n_trials": int(spec.n_trials),
        "regimes": [asdict(regime) for regime in spec.regimes],
        "report_sha256": _report_digest(report),
        "trade_count": len(report.get("trades", [])),
        "pending_signal_count": len(report.get("pending_signals", [])),
        "provenance": dict(provenance),
    }
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
        config: FoundationDeploymentConfig | None = None,
        warehouse: PitWarehouse | None = None,
        market_data: WarehouseMarketData | None = None,
        provenance: Mapping[str, Any] | None = None,
        engine_factory: Callable[..., BacktestEngine] = BacktestEngine,
        policy: PromotionPolicy | None = None) -> FoundationResearchOutput:
    """Run the qualifying research path without persisting or approving it."""
    config = config or foundation_target_v1_config()
    warehouse = warehouse or PitWarehouse()
    market_data = market_data or WarehouseMarketData(warehouse)
    before = market_data.data_snapshot(REQUIRED_SNAPSHOT_TABLES)
    _snapshot_version(before)
    universe, selection_date, eligible_count = _select_universe(warehouse, spec)
    run_provenance = dict(provenance or capture_run_provenance(seed=spec.seed))
    _require_reproducible_provenance(run_provenance)
    if int(run_provenance["seed"]) != int(spec.seed):
        raise RuntimeError("provenance seed does not match research spec")

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
        policy=policy,
    )
    return FoundationResearchOutput(
        artifact=artifact, report=report,
        selected_universe=universe, data_snapshot=dict(before),
    )


def persist_foundation_research(output: FoundationResearchOutput,
                                store: ArtifactStore) -> str:
    """Persist one immutable research artifact and return its hash."""
    store.persist(output.artifact)
    return output.artifact.artifact_hash


__all__ = [
    "DEFAULT_REGIMES",
    "FoundationResearchOutput",
    "FoundationResearchSpec",
    "ResearchRegime",
    "build_foundation_artifact",
    "persist_foundation_research",
    "run_foundation_research",
]
