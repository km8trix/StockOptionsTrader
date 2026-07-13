"""Content-addressed strategy evidence and explicit deployment approvals.

Research results are immutable facts.  Deployment approval is a separate,
equally explicit fact that points at one exact research artifact.  Keeping the
two separate prevents a re-run, a mutable registry entry, or a changed working
tree from silently changing what paper/live execution is allowed to load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, TypeGuard

from utils.market_hours import MarketHours, NYSE_TZ


class PromotionError(RuntimeError):
    """Base class for promotion-pipeline failures."""


class ArtifactIntegrityError(PromotionError):
    """A persisted artifact or approval reference failed verification."""


class PromotionNotApproved(PromotionError):
    """The exact artifact is not approved for the requested execution tier."""


class PromotionLevel(str, Enum):
    RESEARCH_ONLY = "research_only"
    PAPER_ELIGIBLE = "paper_eligible"
    LIVE_ELIGIBLE = "live_eligible"


_LEVEL_RANK = {
    PromotionLevel.RESEARCH_ONLY: 0,
    PromotionLevel.PAPER_ELIGIBLE: 1,
    PromotionLevel.LIVE_ELIGIBLE: 2,
}


def _finite(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _canonical_value(value: Any) -> Any:
    """Return a deterministic JSON value, rejecting ambiguous data."""
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("promotion artifacts cannot contain NaN or infinity")
        # JSON has a single numeric representation for -0.0 in this contract.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("promotion artifact mapping keys must be strings")
            normalized[key] = _canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported promotion artifact value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Canonical UTF-8 JSON used for both hashing and persistence."""
    return json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _validate_sha256(value: str, *, field: str) -> str:
    candidate = str(value).lower()
    if (len(candidate) != 64
            or any(ch not in "0123456789abcdef" for ch in candidate)):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return candidate


def _nonnegative_integer(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class PromotionResults:
    """Evidence produced by a fixed, out-of-sample strategy evaluation."""

    psr: float | None
    dsr: float | None
    oos_total_folds: int
    oos_testable_folds: int
    oos_significant_bh: int
    cost_model_applied: bool
    estimated_cost_bps: float | None
    cost_adjusted_return: float | None
    annual_turnover: float | None
    regime_results: Mapping[str, float]

    @classmethod
    def from_oos_returns(
            cls, returns: Sequence[float], period_labels: Sequence[Any], *,
            n_trials: int, cost_model_applied: bool,
            estimated_cost_bps: float, annual_turnover: float,
            regime_results: Mapping[str, float],
            risk_free_rate: float = 0.02,
            periods_per_year: int = 252,
            psr_threshold: float = 0.95,
            alpha: float = 0.05,
            min_period_obs: int = 20) -> PromotionResults:
        """Compute the statistical fields from one cost-adjusted OOS series.

        The same excess-return series feeds PSR, DSR and fold-BH.  Callers
        supply the trial count explicitly so multiple testing cannot silently
        default to a single research attempt.
        """
        from analysis.research_stats import (deflated_sharpe_ratio,
                                             validate_strategy_oos)

        if len(returns) != len(period_labels):
            raise ValueError("period_labels must align with OOS returns")
        if int(n_trials) < 1:
            raise ValueError("n_trials must be positive")
        clean_returns = [float(value) for value in returns]
        if any(not math.isfinite(value) for value in clean_returns):
            raise ValueError("OOS returns must all be finite")
        gate = validate_strategy_oos(
            clean_returns, period_labels, psr_threshold=psr_threshold,
            alpha=alpha, min_period_obs=min_period_obs,
            risk_free_rate=risk_free_rate, periods_per_year=periods_per_year,
        )
        excess = [value - risk_free_rate / periods_per_year
                  for value in clean_returns]
        dsr = deflated_sharpe_ratio(excess, int(n_trials))
        compounded = math.prod(1.0 + value for value in clean_returns) - 1.0
        return cls(
            psr=gate["psr"], dsr=dsr,
            oos_total_folds=len(gate["fold_labels"]),
            oos_testable_folds=int(gate["n_periods_tested"]),
            oos_significant_bh=int(gate["bh"]["n_significant_bh"]),
            cost_model_applied=bool(cost_model_applied),
            estimated_cost_bps=float(estimated_cost_bps),
            cost_adjusted_return=float(compounded),
            annual_turnover=float(annual_turnover),
            regime_results=dict(regime_results),
        )

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionResults:
        return cls(
            psr=value.get("psr"),
            dsr=value.get("dsr"),
            oos_total_folds=int(value.get("oos_total_folds", 0)),
            oos_testable_folds=int(value.get("oos_testable_folds", 0)),
            oos_significant_bh=int(value.get("oos_significant_bh", 0)),
            cost_model_applied=bool(value.get("cost_model_applied", False)),
            estimated_cost_bps=value.get("estimated_cost_bps"),
            cost_adjusted_return=value.get("cost_adjusted_return"),
            annual_turnover=value.get("annual_turnover"),
            regime_results=dict(value.get("regime_results", {})),
        )


@dataclass(frozen=True)
class TierCriteria:
    """Every criterion required for one deployment tier."""

    min_psr: float
    min_dsr: float
    min_testable_oos_folds: int
    min_significant_oos_folds: int
    require_cost_model: bool
    max_estimated_cost_bps: float
    min_cost_adjusted_return: float
    max_annual_turnover: float
    min_regime_count: int
    min_regime_return: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_psr <= 1.0:
            raise ValueError("min_psr must be in [0, 1]")
        if not 0.0 <= self.min_dsr <= 1.0:
            raise ValueError("min_dsr must be in [0, 1]")
        if self.min_testable_oos_folds < 1:
            raise ValueError("min_testable_oos_folds must be positive")
        if self.min_significant_oos_folds < 1:
            raise ValueError("min_significant_oos_folds must be positive")
        if self.min_significant_oos_folds > self.min_testable_oos_folds:
            raise ValueError("significant OOS folds cannot exceed testable folds")
        if self.max_estimated_cost_bps < 0 or self.max_annual_turnover < 0:
            raise ValueError("cost and turnover limits cannot be negative")
        if self.min_regime_count < 1:
            raise ValueError("min_regime_count must be positive")


@dataclass(frozen=True)
class PromotionCheck:
    name: str
    passed: bool
    actual: Any
    required: Any

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))


@dataclass(frozen=True)
class PromotionEvaluation:
    level: PromotionLevel
    paper_checks: tuple[PromotionCheck, ...]
    live_checks: tuple[PromotionCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "paper_checks": [check.to_dict() for check in self.paper_checks],
            "live_checks": [check.to_dict() for check in self.live_checks],
        }


def _checks(results: PromotionResults,
            criteria: TierCriteria) -> tuple[PromotionCheck, ...]:
    regimes = dict(results.regime_results)
    regime_values_are_finite = all(_finite(value) for value in regimes.values())
    worst_regime = (min(float(value) for value in regimes.values())
                    if regimes and regime_values_are_finite else None)
    estimated_cost_ok = (
        _finite(results.estimated_cost_bps)
        and float(results.estimated_cost_bps) >= 0.0
        and float(results.estimated_cost_bps) <= criteria.max_estimated_cost_bps
    )
    return (
        PromotionCheck("psr", _finite(results.psr)
                       and 0.0 <= float(results.psr) <= 1.0
                       and float(results.psr) >= criteria.min_psr,
                       results.psr, {"minimum": criteria.min_psr}),
        PromotionCheck("dsr", _finite(results.dsr)
                       and 0.0 <= float(results.dsr) <= 1.0
                       and float(results.dsr) >= criteria.min_dsr,
                       results.dsr, {"minimum": criteria.min_dsr}),
        PromotionCheck("oos_testable_folds",
                       results.oos_testable_folds >= criteria.min_testable_oos_folds
                       and results.oos_total_folds >= results.oos_testable_folds,
                       results.oos_testable_folds,
                       {"minimum": criteria.min_testable_oos_folds}),
        PromotionCheck("multiple_testing_bh",
                       results.oos_significant_bh
                       >= criteria.min_significant_oos_folds
                       and results.oos_significant_bh
                       <= results.oos_testable_folds,
                       results.oos_significant_bh,
                       {"minimum": criteria.min_significant_oos_folds}),
        PromotionCheck("cost_model", results.cost_model_applied
                       if criteria.require_cost_model else True,
                       results.cost_model_applied,
                       {"required": criteria.require_cost_model}),
        PromotionCheck("estimated_cost_bps", bool(estimated_cost_ok),
                       results.estimated_cost_bps,
                       {"maximum": criteria.max_estimated_cost_bps}),
        PromotionCheck("cost_adjusted_return",
                       _finite(results.cost_adjusted_return)
                       and float(results.cost_adjusted_return)
                       >= criteria.min_cost_adjusted_return,
                       results.cost_adjusted_return,
                       {"minimum": criteria.min_cost_adjusted_return}),
        PromotionCheck("annual_turnover",
                       _finite(results.annual_turnover)
                       and 0.0 <= float(results.annual_turnover)
                       <= criteria.max_annual_turnover,
                       results.annual_turnover,
                       {"maximum": criteria.max_annual_turnover}),
        PromotionCheck("regime_count",
                       len(regimes) >= criteria.min_regime_count
                       and regime_values_are_finite,
                       len(regimes), {"minimum": criteria.min_regime_count}),
        PromotionCheck("worst_regime_return",
                       worst_regime is not None
                       and worst_regime >= criteria.min_regime_return,
                       worst_regime,
                       {"minimum": criteria.min_regime_return}),
    )


@dataclass(frozen=True)
class PromotionPolicy:
    """Single versioned policy that assigns research/paper/live eligibility."""

    name: str
    version: str
    paper: TierCriteria
    live: TierCriteria

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("promotion policy name and version are required")
        minimum_fields = (
            "min_psr", "min_dsr", "min_testable_oos_folds",
            "min_significant_oos_folds", "min_cost_adjusted_return",
            "min_regime_count", "min_regime_return",
        )
        maximum_fields = ("max_estimated_cost_bps", "max_annual_turnover")
        for field in minimum_fields:
            if getattr(self.live, field) < getattr(self.paper, field):
                raise ValueError(
                    f"live criteria cannot be weaker than paper: {field}")
        for field in maximum_fields:
            if getattr(self.live, field) > getattr(self.paper, field):
                raise ValueError(
                    f"live criteria cannot be weaker than paper: {field}")
        if self.paper.require_cost_model and not self.live.require_cost_model:
            raise ValueError(
                "live criteria cannot be weaker than paper: require_cost_model")

    @classmethod
    def default(cls) -> PromotionPolicy:
        return cls(
            name="authoritative-promotion",
            version="1",
            paper=TierCriteria(
                min_psr=0.90, min_dsr=0.80,
                min_testable_oos_folds=2, min_significant_oos_folds=1,
                require_cost_model=True, max_estimated_cost_bps=100.0,
                min_cost_adjusted_return=0.0, max_annual_turnover=24.0,
                min_regime_count=2, min_regime_return=-0.25,
            ),
            live=TierCriteria(
                min_psr=0.95, min_dsr=0.95,
                min_testable_oos_folds=3, min_significant_oos_folds=2,
                require_cost_model=True, max_estimated_cost_bps=50.0,
                min_cost_adjusted_return=0.0, max_annual_turnover=12.0,
                min_regime_count=3, min_regime_return=-0.10,
            ),
        )

    def evaluate(self, results: PromotionResults) -> PromotionEvaluation:
        paper_checks = _checks(results, self.paper)
        live_checks = _checks(results, self.live)
        if all(check.passed for check in live_checks):
            level = PromotionLevel.LIVE_ELIGIBLE
        elif all(check.passed for check in paper_checks):
            level = PromotionLevel.PAPER_ELIGIBLE
        else:
            level = PromotionLevel.RESEARCH_ONLY
        return PromotionEvaluation(level, paper_checks, live_checks)

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @property
    def policy_id(self) -> str:
        """Stable identity of the complete policy, not only its label."""
        encoded = canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionPolicy:
        return cls(
            name=str(value["name"]), version=str(value["version"]),
            paper=TierCriteria(**dict(value["paper"])),
            live=TierCriteria(**dict(value["live"])),
        )


@dataclass(frozen=True)
class PromotionArtifact:
    """Immutable canonical strategy evidence identified by its SHA-256."""

    artifact_hash: str
    payload_json: str

    @classmethod
    def create(
            cls, *, strategy_id: str, strategy_version: str,
            data_version: str, universe: Sequence[str],
            parameters: Mapping[str, Any], seed: int,
            dependency_versions: Mapping[str, str], code_sha: str,
            results: PromotionResults,
            policy: PromotionPolicy | None = None,
            evidence: Mapping[str, Any] | None = None) -> PromotionArtifact:
        policy = policy or PromotionPolicy.default()
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("promotion evidence must be a mapping")
        normalized_universe = sorted({str(symbol).strip().upper()
                                      for symbol in universe if str(symbol).strip()})
        if not strategy_id.strip() or not strategy_version.strip():
            raise ValueError("strategy_id and strategy_version are required")
        if not data_version.strip() or not code_sha.strip():
            raise ValueError("data_version and code_sha are required")
        if not normalized_universe:
            raise ValueError("universe cannot be empty")
        evaluation = policy.evaluate(results)
        payload = {
            "schema_version": 2,
            "strategy_id": strategy_id.strip(),
            "strategy_version": strategy_version.strip(),
            "data_version": data_version.strip(),
            "universe": normalized_universe,
            "parameters": _canonical_value(parameters),
            "seed": int(seed),
            "dependency_versions": _canonical_value(dependency_versions),
            "code_sha": code_sha.strip(),
            "results": results.to_dict(),
            "policy": policy.to_dict(),
            "policy_id": policy.policy_id,
            "evidence": (_canonical_value(evidence)
                         if evidence is not None else None),
            "decision": evaluation.to_dict(),
        }
        payload_json = canonical_json(payload)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return cls(digest, payload_json)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def strategy_id(self) -> str:
        return str(self.payload["strategy_id"])

    @property
    def strategy_version(self) -> str:
        return str(self.payload["strategy_version"])

    @property
    def code_sha(self) -> str:
        return str(self.payload["code_sha"])

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self.payload["parameters"])

    @property
    def decision(self) -> PromotionLevel:
        return PromotionLevel(self.payload["decision"]["level"])

    @property
    def evidence(self) -> dict[str, Any] | None:
        evidence = self.payload.get("evidence")
        return dict(evidence) if evidence is not None else None

    @property
    def policy_id(self) -> str:
        policy = PromotionPolicy.from_dict(self.payload["policy"])
        return policy.policy_id

    def to_json(self) -> str:
        return canonical_json({"artifact_hash": self.artifact_hash,
                               "payload": self.payload}) + "\n"

    @classmethod
    def from_json(cls, value: str) -> PromotionArtifact:
        try:
            document = json.loads(value)
            claimed_hash = str(document["artifact_hash"])
            payload = document["payload"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("invalid promotion artifact document") from exc
        payload_json = canonical_json(payload)
        actual_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if claimed_hash != actual_hash:
            raise ArtifactIntegrityError("promotion artifact hash mismatch")
        try:
            schema_version = payload["schema_version"]
            if not isinstance(schema_version, int) or isinstance(
                    schema_version, bool):
                raise ArtifactIntegrityError(
                    "invalid promotion artifact schema")
            if schema_version not in (1, 2):
                raise ArtifactIntegrityError(
                    "unsupported promotion artifact schema")
            results = PromotionResults.from_dict(payload["results"])
            policy = PromotionPolicy.from_dict(payload["policy"])
            if _canonical_value(payload["policy"]) != policy.to_dict():
                raise ArtifactIntegrityError(
                    "promotion artifact policy is not canonical")
            if schema_version == 2:
                if payload.get("policy_id") != policy.policy_id:
                    raise ArtifactIntegrityError(
                        "promotion artifact policy identity mismatch")
                if "evidence" not in payload:
                    raise ArtifactIntegrityError(
                        "promotion artifact evidence field is missing")
                evidence = payload["evidence"]
                if evidence is not None and not isinstance(evidence, Mapping):
                    raise ArtifactIntegrityError(
                        "promotion artifact evidence must be a mapping or null")
            expected = policy.evaluate(results).to_dict()
            if _canonical_value(payload["decision"]) != expected:
                raise ArtifactIntegrityError("promotion decision does not match evidence")
        except ArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid promotion artifact payload") from exc
        return cls(actual_hash, payload_json)


_FOUNDATION_RESEARCH_RUNNER = "foundation_research_v2"
_FOUNDATION_RESEARCH_TABLES = frozenset({
    "actions", "daily", "sep", "tickers",
})
_FOUNDATION_UNIVERSE_METHOD = (
    "full_live_universe_fixed_requested_date_ranked_on_prior_observable_"
    "session_complete_market_cap"
)
_FOUNDATION_ENGINE_FIELDS = frozenset({
    "adv_window", "commission", "enable_realistic_fills", "impact_coef",
    "initial_capital", "participation_cap", "seed", "slippage_bps",
})


def _canonical_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must be a canonical ISO date")
    return parsed


def _positive_integer(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _authoritative_foundation_research_error(message: str) -> PromotionNotApproved:
    return PromotionNotApproved(
        f"Foundation artifact is not authoritative research evidence: {message}")


def _require_authoritative_foundation_research(
        artifact: PromotionArtifact) -> None:
    """Verify the exact research-runner attestation used by execution tiers.

    ``PromotionArtifact`` remains a generic, backwards-readable research
    container.  That compatibility must not make a legacy or hand-built result
    deployable, however.  The Foundation registry therefore applies this
    stricter contract whenever an artifact is considered for paper/live use.
    Every material runner input is bound to the artifact's top-level identity;
    an arbitrary passing ``PromotionResults`` object is insufficient.
    """
    try:
        payload = artifact.payload
        if payload.get("schema_version") != 2:
            raise ValueError("research artifact schema 2 is required")
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("research runner evidence is required")
        if evidence.get("runner") != _FOUNDATION_RESEARCH_RUNNER:
            raise ValueError(
                f"runner must be {_FOUNDATION_RESEARCH_RUNNER!r}")

        window = evidence.get("window")
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("window must contain start and end dates")
        window_start = _canonical_date(window[0], field="window start")
        window_end = _canonical_date(window[1], field="window end")
        if window_start > window_end:
            raise ValueError("research window start exceeds end")

        selection = evidence.get("universe_selection")
        if not isinstance(selection, Mapping) or set(selection) != {
                "eligible_symbols", "market_cap_coverage_complete",
                "max_symbols", "method", "ranked_symbols",
                "requested_as_of", "resolved_as_of"}:
            raise ValueError("universe selection evidence is incomplete")
        requested = _canonical_date(
            selection["requested_as_of"], field="universe requested_as_of")
        resolved = _canonical_date(
            selection["resolved_as_of"], field="universe resolved_as_of")
        if not resolved <= requested <= window_start:
            raise ValueError(
                "universe dates must satisfy resolved <= requested <= window start")
        if selection.get("method") != _FOUNDATION_UNIVERSE_METHOD:
            raise ValueError("universe selection method is not authoritative")
        max_symbols = _positive_integer(
            selection.get("max_symbols"), field="max_symbols")
        eligible_symbols = _positive_integer(
            selection.get("eligible_symbols"), field="eligible_symbols")
        ranked_symbols = _positive_integer(
            selection.get("ranked_symbols"), field="ranked_symbols")
        if (ranked_symbols != eligible_symbols
                or selection.get("market_cap_coverage_complete") is not True):
            raise ValueError("market-cap ranking coverage is incomplete")
        universe = payload.get("universe")
        if (not isinstance(universe, list) or not universe
                or len(universe) != min(max_symbols, eligible_symbols)):
            raise ValueError(
                "artifact universe size differs from the complete ranking")

        snapshot = evidence.get("warehouse_snapshot")
        if not isinstance(snapshot, Mapping) or set(snapshot) != {
                "complete", "quality_flags", "tables", "version"}:
            raise ValueError("warehouse snapshot evidence is incomplete")
        raw_snapshot_version = snapshot.get("version")
        if not isinstance(raw_snapshot_version, str):
            raise ValueError("warehouse snapshot version must be a SHA-256 digest")
        snapshot_version = _validate_sha256(
            raw_snapshot_version, field="warehouse snapshot version")
        if payload.get("data_version") != f"pit-sha256:{snapshot_version}":
            raise ValueError("warehouse snapshot version differs from data_version")
        if snapshot.get("complete") is not True \
                or snapshot.get("quality_flags") != []:
            raise ValueError("warehouse snapshot is not complete and clean")
        tables = snapshot.get("tables")
        if not isinstance(tables, list) or len(tables) != len(
                _FOUNDATION_RESEARCH_TABLES):
            raise ValueError("warehouse snapshot table manifest is incomplete")
        table_names: set[str] = set()
        snapshot_entries: list[tuple[str, str, int]] = []
        for item in tables:
            if not isinstance(item, Mapping) or set(item) != {
                    "bytes", "sha256", "table"}:
                raise ValueError("warehouse snapshot table entry is invalid")
            table = item.get("table")
            if not isinstance(table, str) or table in table_names:
                raise ValueError("warehouse snapshot table names are invalid")
            table_names.add(table)
            table_sha256 = item.get("sha256")
            if not isinstance(table_sha256, str):
                raise ValueError(
                    f"warehouse table {table} sha256 must be a SHA-256 digest")
            _validate_sha256(
                table_sha256, field=f"warehouse table {table} sha256")
            table_bytes = _positive_integer(
                item.get("bytes"), field=f"warehouse table {table} bytes")
            snapshot_entries.append((table, table_sha256, table_bytes))
        if table_names != _FOUNDATION_RESEARCH_TABLES:
            raise ValueError("warehouse snapshot has the wrong required tables")
        snapshot_digest = hashlib.sha256()
        for table, table_sha256, table_bytes in sorted(snapshot_entries):
            snapshot_digest.update(
                f"{table}:{table_sha256}:{table_bytes}\n".encode("utf-8"))
        if snapshot_digest.hexdigest() != snapshot_version:
            raise ValueError(
                "warehouse snapshot version differs from its table manifest")

        engine = evidence.get("engine_parameters")
        if not isinstance(engine, Mapping) or set(engine) != _FOUNDATION_ENGINE_FIELDS:
            raise ValueError("research engine parameters are incomplete")
        if engine.get("enable_realistic_fills") is not True:
            raise ValueError("research must enable realistic fills")
        seed = payload.get("seed")
        if type(seed) is not int or engine.get("seed") != seed:
            raise ValueError("engine seed differs from artifact seed")
        adv_window = _positive_integer(engine.get("adv_window"), field="adv_window")
        del adv_window
        initial_capital = engine.get("initial_capital")
        commission = engine.get("commission")
        slippage_bps = engine.get("slippage_bps")
        impact_coef = engine.get("impact_coef")
        participation_cap = engine.get("participation_cap")
        if not _finite(initial_capital) or float(initial_capital) <= 0:
            raise ValueError("initial_capital must be finite and positive")
        if not _finite(commission) or not 0 <= float(commission) < 1:
            raise ValueError("commission must be in [0, 1)")
        if not _finite(slippage_bps) \
                or not 0 <= float(slippage_bps) <= 10_000:
            raise ValueError("slippage_bps must be in [0, 10000]")
        if not _finite(impact_coef) or not 0 <= float(impact_coef) <= 1:
            raise ValueError("impact_coef must be in [0, 1]")
        if not _finite(participation_cap) \
                or not 0 < float(participation_cap) <= 1:
            raise ValueError("participation_cap must be in (0, 1]")

        n_trials = _positive_integer(
            evidence.get("n_trials"), field="n_trials")
        del n_trials
        regimes = evidence.get("regimes")
        if not isinstance(regimes, list) or not regimes:
            raise ValueError("research regime evidence is required")
        regime_names: set[str] = set()
        for item in regimes:
            if not isinstance(item, Mapping) or set(item) != {
                    "end", "name", "start"}:
                raise ValueError("research regime entry is invalid")
            name = item.get("name")
            if (not isinstance(name, str) or not name.strip()
                    or name in regime_names):
                raise ValueError("research regime names are invalid")
            regime_names.add(name)
            regime_start = _canonical_date(
                item.get("start"), field=f"regime {name} start")
            regime_end = _canonical_date(
                item.get("end"), field=f"regime {name} end")
            if not window_start <= regime_start <= regime_end <= window_end:
                raise ValueError(f"regime {name} is outside the research window")
        result_regimes = payload.get("results", {}).get("regime_results")
        if not isinstance(result_regimes, Mapping) \
                or set(result_regimes) != regime_names:
            raise ValueError("regime results differ from declared regimes")

        report_sha256 = evidence.get("report_sha256")
        if not isinstance(report_sha256, str):
            raise ValueError("report_sha256 must be a SHA-256 digest")
        _validate_sha256(report_sha256, field="report_sha256")
        for field in ("trade_count", "pending_signal_count"):
            value = evidence.get(field)
            if not _nonnegative_integer(value):
                raise ValueError(f"{field} must be a non-negative integer")

        provenance = evidence.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("research provenance is required")
        code_sha = payload.get("code_sha")
        if (not isinstance(code_sha, str) or len(code_sha) != 40
                or any(character not in "0123456789abcdef"
                       for character in code_sha)
                or provenance.get("git_sha") != code_sha):
            raise ValueError("provenance Git SHA differs from artifact code")
        if provenance.get("git_dirty") is not False:
            raise ValueError("research provenance is not from a clean checkout")
        if provenance.get("seed") != seed:
            raise ValueError("provenance seed differs from artifact seed")
        python_version = provenance.get("python_version")
        dependencies = provenance.get("dependency_versions")
        artifact_dependencies = payload.get("dependency_versions")
        if (not isinstance(python_version, str) or not python_version.strip()
                or not isinstance(dependencies, Mapping) or not dependencies
                or any(not isinstance(value, str) or not value.strip()
                       for value in dependencies.values())
                or not isinstance(artifact_dependencies, Mapping)):
            raise ValueError("dependency provenance is incomplete")
        expected_dependencies = dict(dependencies)
        expected_dependencies["python"] = python_version
        if _canonical_value(artifact_dependencies) != _canonical_value(
                expected_dependencies):
            raise ValueError(
                "artifact dependency versions differ from provenance")

        from desks.deployment_config import FoundationDeploymentConfig
        config = FoundationDeploymentConfig.from_mapping(payload.get("parameters"))
        if config.strategy_version != payload.get("strategy_version"):
            raise ValueError(
                "Foundation strategy version differs from its parameters")
    except PromotionNotApproved:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise _authoritative_foundation_research_error(str(exc)) from exc


@dataclass(frozen=True)
class PaperValidationEvaluation:
    """Fail-closed evaluation of one exact paper-trading rehearsal."""

    passed: bool
    checks: tuple[PromotionCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


_AUTHORITATIVE_PAPER_RUNNER = "foundation_paper_rehearsal_v2"
_PAPER_TERMINAL_ORDER_STATUSES = frozenset({
    "FILLED", "CANCELLED", "CANCELED", "REJECTED", "EXPIRED", "EXECUTED",
})
_PROSPECTIVE_OBSERVATION_TOLERANCE = timedelta(minutes=5)


def _paper_evidence_error(message: str) -> ArtifactIntegrityError:
    return ArtifactIntegrityError(
        f"authoritative paper runner evidence {message}")


def _aware_evidence_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise _paper_evidence_error(f"has no {field} timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _paper_evidence_error(
            f"has an invalid {field} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _paper_evidence_error(
            f"has a timezone-naive {field} timestamp")
    return parsed.astimezone(timezone.utc)


def _evidence_symbols(
        values: Any, *, field: str) -> frozenset[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise _paper_evidence_error(f"{field} must be a symbol sequence")
    normalized = [str(value).strip().upper() for value in values]
    if (not normalized or any(not value for value in normalized)
            or len(set(normalized)) != len(normalized)):
        raise _paper_evidence_error(
            f"{field} must contain unique non-empty symbols")
    return frozenset(normalized)


def _recompute_authoritative_paper_evidence(
        evidence: Mapping[str, Any] | None, *,
        expected_universe: Sequence[str] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Recompute every promotable paper metric from sealed runner facts.

    Summary counts are convenient for operators but are not evidence.  A
    passing artifact therefore has to carry the durable runner's cycles,
    quotes, reconciliations, orders, and audit-chain result.  This function
    independently derives the values evaluated by ``PaperValidationPolicy``.
    """
    if not isinstance(evidence, Mapping):
        raise _paper_evidence_error("is required for a passing artifact")
    if evidence.get("runner") != _AUTHORITATIVE_PAPER_RUNNER:
        raise _paper_evidence_error(
            f"must identify {_AUTHORITATIVE_PAPER_RUNNER!r}")
    if not isinstance(evidence.get("run_id"), str) \
            or not str(evidence["run_id"]).strip():
        raise _paper_evidence_error("has no run_id")

    started_at = _aware_evidence_timestamp(
        evidence.get("started_at"), field="started_at")
    raw_cycles = evidence.get("cycles")
    if not isinstance(raw_cycles, list) or not raw_cycles:
        raise _paper_evidence_error("must contain a non-empty cycle ledger")

    expected = (_evidence_symbols(expected_universe, field="research universe")
                if expected_universe is not None else None)
    declared_universe = evidence.get("universe")
    if declared_universe is not None:
        declared = _evidence_symbols(
            declared_universe, field="declared universe")
        if expected is not None and declared != expected:
            raise _paper_evidence_error(
                "declared universe differs from the research artifact")
        expected = declared

    seen_cycle_ids: set[str] = set()
    execution_dates: set[str] = set()
    previous_as_of: datetime | None = None
    cycle_errors = 0
    prospective = True
    reconciliation_failures = 0
    reported_order_ids: list[str] = []
    last_cycle_orders: Any = None
    inferred_universe: frozenset[str] | None = None
    previous_order_ids: set[str] = set()
    calendar = MarketHours()

    for index, raw_cycle in enumerate(raw_cycles):
        if not isinstance(raw_cycle, Mapping):
            raise _paper_evidence_error(f"cycle {index} is not a mapping")
        cycle_id = raw_cycle.get("cycle_id")
        if (not isinstance(cycle_id, str) or not cycle_id.strip()
                or cycle_id in seen_cycle_ids):
            raise _paper_evidence_error(
                f"cycle {index} has a missing or duplicate cycle_id")
        seen_cycle_ids.add(cycle_id)

        as_of = _aware_evidence_timestamp(
            raw_cycle.get("as_of"), field=f"cycle {index} as_of")
        expected_cycle_id = hashlib.sha256(
            as_of.isoformat().encode("utf-8")).hexdigest()
        if cycle_id != expected_cycle_id:
            raise _paper_evidence_error(
                f"cycle {index} id does not match its execution timestamp")
        if not calendar.is_market_open(as_of):
            raise _paper_evidence_error(
                f"cycle {index} did not execute during an NYSE regular session")
        observed_at = _aware_evidence_timestamp(
            raw_cycle.get("observed_at"),
            field=f"cycle {index} observed_at")
        cycle_started_at = _aware_evidence_timestamp(
            raw_cycle.get("started_at"),
            field=f"cycle {index} started_at")
        completed_at = _aware_evidence_timestamp(
            raw_cycle.get("completed_at"),
            field=f"cycle {index} completed_at")
        execution_dates.add(as_of.astimezone(NYSE_TZ).date().isoformat())
        if previous_as_of is not None and as_of <= previous_as_of:
            prospective = False
        previous_as_of = as_of
        if (as_of < started_at
                or not as_of <= observed_at <= completed_at
                or cycle_started_at != observed_at
                or cycle_started_at < started_at
                or observed_at - as_of > _PROSPECTIVE_OBSERVATION_TOLERANCE
                or observed_at > completed_at
                or as_of > completed_at):
            prospective = False

        input_hash = str(raw_cycle.get("input_hash") or "").lower()
        if (len(input_hash) != 64
                or any(character not in "0123456789abcdef"
                       for character in input_hash)):
            raise _paper_evidence_error(
                f"cycle {index} has an invalid input hash")

        prices = raw_cycle.get("execution_prices")
        if not isinstance(prices, Mapping) or not prices:
            raise _paper_evidence_error(
                f"cycle {index} has no execution-price evidence")
        price_symbols = frozenset(
            str(symbol).strip().upper() for symbol in prices)
        if (any(not symbol for symbol in price_symbols)
                or len(price_symbols) != len(prices)):
            raise _paper_evidence_error(
                f"cycle {index} has invalid execution-price symbols")
        if inferred_universe is None:
            inferred_universe = price_symbols
        elif price_symbols != inferred_universe:
            raise _paper_evidence_error(
                "execution-price symbol coverage changes between cycles")
        if expected is not None and price_symbols != expected:
            raise _paper_evidence_error(
                f"cycle {index} execution-price universe differs from research")
        for symbol, raw_quote in prices.items():
            if not isinstance(raw_quote, Mapping):
                raise _paper_evidence_error(
                    f"cycle {index} quote for {symbol} is not a mapping")
            price = raw_quote.get("price")
            if not _finite(price) or float(price) <= 0.0:
                raise _paper_evidence_error(
                    f"cycle {index} quote for {symbol} has an invalid price")
            source = raw_quote.get("source")
            if not isinstance(source, str) or not source.strip():
                raise _paper_evidence_error(
                    f"cycle {index} quote for {symbol} has no source")
            quote_observed_at = _aware_evidence_timestamp(
                raw_quote.get("observed_at"),
                field=f"cycle {index} quote for {symbol} observed_at")
            if (quote_observed_at > as_of
                    or as_of - quote_observed_at
                    > _PROSPECTIVE_OBSERVATION_TOLERANCE
                    or quote_observed_at.astimezone(NYSE_TZ).date()
                    != as_of.astimezone(NYSE_TZ).date()):
                prospective = False

        error_count = raw_cycle.get("error_count")
        if not _nonnegative_integer(error_count):
            raise _paper_evidence_error(
                f"cycle {index} has an invalid error_count")
        cycle_errors += error_count
        for field in ("pre_reconciliation", "post_reconciliation"):
            reconciliation = raw_cycle.get(field)
            if not isinstance(reconciliation, Mapping):
                raise _paper_evidence_error(
                    f"cycle {index} has no {field} fact")
            mismatches = reconciliation.get("mismatches")
            if not isinstance(mismatches, list):
                raise _paper_evidence_error(
                    f"cycle {index} {field} has no mismatch ledger")
            if reconciliation.get("ok") is True and mismatches:
                raise _paper_evidence_error(
                    f"cycle {index} {field} contradicts its mismatches")
            reconciliation_failures += int(reconciliation.get("ok") is not True)

        result = raw_cycle.get("result")
        if not isinstance(result, Mapping):
            raise _paper_evidence_error(f"cycle {index} has no result fact")
        reports = result.get("reports", [])
        if not isinstance(reports, list):
            raise _paper_evidence_error(
                f"cycle {index} reports must be a list")
        for report in reports:
            if not isinstance(report, Mapping):
                raise _paper_evidence_error(
                    f"cycle {index} contains an invalid order report")
            order_id = report.get("order_id")
            if order_id is not None:
                reported_order_ids.append(str(order_id))
            if str(report.get("status") or "").strip().lower() \
                    in {"error", "killed", "rejected"}:
                cycle_errors += 1
        if result.get("status") not in {"ok", "pending"}:
            cycle_errors += 1

        cycle_orders = raw_cycle.get("orders")
        if not isinstance(cycle_orders, list):
            raise _paper_evidence_error(
                f"cycle {index} order snapshot must be a list")
        if any(not isinstance(order, Mapping) for order in cycle_orders):
            raise _paper_evidence_error(
                f"cycle {index} contains an invalid order snapshot")
        cycle_order_ids = {
            str(order.get("order_id") or "").strip()
            for order in cycle_orders
        }
        if ("" in cycle_order_ids or len(cycle_order_ids) != len(cycle_orders)
                or not previous_order_ids.issubset(cycle_order_ids)):
            raise _paper_evidence_error(
                f"cycle {index} order snapshot is not a cumulative ledger")
        previous_order_ids = cycle_order_ids
        last_cycle_orders = cycle_orders

    model_checkpoint = evidence.get("model_checkpoint")
    if not isinstance(model_checkpoint, Mapping) or set(model_checkpoint) != {
            "cycle_id", "sha256", "state"}:
        raise _paper_evidence_error("has no complete model checkpoint")
    if model_checkpoint.get("cycle_id") != raw_cycles[-1].get("cycle_id"):
        raise _paper_evidence_error(
            "model checkpoint does not belong to the final cycle")
    checkpoint_state = model_checkpoint.get("state")
    if not isinstance(checkpoint_state, Mapping) or set(checkpoint_state) != {
            "payload", "schema_version", "sha256"}:
        raise _paper_evidence_error("has an invalid model checkpoint state")
    if checkpoint_state.get("schema_version") != 1:
        raise _paper_evidence_error("uses an unsupported model checkpoint schema")
    checkpoint_payload = checkpoint_state.get("payload")
    if not isinstance(checkpoint_payload, Mapping):
        raise _paper_evidence_error("has no model checkpoint payload")
    checkpoint_hash = str(model_checkpoint.get("sha256") or "").lower()
    state_hash = str(checkpoint_state.get("sha256") or "").lower()
    computed_hash = hashlib.sha256(
        canonical_json(checkpoint_payload).encode("utf-8")).hexdigest()
    if checkpoint_hash != state_hash or state_hash != computed_hash:
        raise _paper_evidence_error("model checkpoint hash does not match payload")
    fits = checkpoint_payload.get("fits")
    training_data = checkpoint_payload.get("training_data")
    if not isinstance(fits, list) or not fits:
        raise _paper_evidence_error("model checkpoint proves no fitted model")
    if not isinstance(training_data, list) or not training_data:
        raise _paper_evidence_error("model checkpoint has no deterministic refit data")
    checkpoint_symbols = [
        str(entry.get("symbol") or "").strip().upper()
        for entry in training_data if isinstance(entry, Mapping)
    ]
    if (len(checkpoint_symbols) != len(training_data)
            or len(set(checkpoint_symbols)) != len(checkpoint_symbols)
            or any(not symbol for symbol in checkpoint_symbols)):
        raise _paper_evidence_error("model checkpoint has invalid training symbols")
    if expected is not None and frozenset(checkpoint_symbols) != expected:
        raise _paper_evidence_error(
            "model checkpoint training universe differs from research")

    final_reconciliation = evidence.get("final_reconciliation")
    if not isinstance(final_reconciliation, Mapping):
        raise _paper_evidence_error("has no final reconciliation fact")
    final_mismatches = final_reconciliation.get("mismatches")
    if not isinstance(final_mismatches, list):
        raise _paper_evidence_error("final reconciliation has no mismatch ledger")
    if final_reconciliation.get("ok") is True and final_mismatches:
        raise _paper_evidence_error(
            "final reconciliation contradicts its mismatches")
    reconciliation_failures += int(
        final_reconciliation.get("ok") is not True)

    broker = evidence.get("broker")
    if not isinstance(broker, Mapping):
        raise _paper_evidence_error("has no broker ledger")
    orders = broker.get("orders")
    if not isinstance(orders, list) or any(
            not isinstance(order, Mapping) for order in orders):
        raise _paper_evidence_error("broker order ledger must be a list")
    order_ids: set[str] = set()
    fills = 0
    open_orders = 0
    filled_buys: dict[str, int] = {}
    filled_sells: dict[str, int] = {}
    for index, order in enumerate(orders):
        order_id = order.get("order_id")
        status = order.get("status")
        if (order_id is None or not str(order_id).strip()
                or str(order_id) in order_ids):
            raise _paper_evidence_error(
                f"broker order {index} has a missing or duplicate order_id")
        if not isinstance(status, str) or not status.strip():
            raise _paper_evidence_error(
                f"broker order {index} has no status")
        order_ids.add(str(order_id))
        normalized_status = status.strip().upper()
        fills += int(normalized_status == "FILLED")
        open_orders += int(
            normalized_status not in _PAPER_TERMINAL_ORDER_STATUSES)
        if normalized_status == "FILLED":
            symbol = str(order.get("symbol") or "").strip().upper()
            side = str(order.get("side") or "").strip().upper()
            quantity = order.get("quantity")
            filled_quantity = order.get("filled_quantity")
            if (not symbol or (expected is not None and symbol not in expected)
                    or str(order.get("asset_type") or "").strip().lower()
                    != "stock"
                    or side not in {"BUY", "SELL"}
                    or not isinstance(quantity, int) or isinstance(quantity, bool)
                    or quantity <= 0
                    or not isinstance(filled_quantity, int)
                    or isinstance(filled_quantity, bool)
                    or filled_quantity != quantity):
                raise _paper_evidence_error(
                    f"broker order {index} has invalid filled-order economics")
            book = filled_buys if side == "BUY" else filled_sells
            book[symbol] = book.get(symbol, 0) + quantity
    if _canonical_value(last_cycle_orders) != _canonical_value(orders):
        raise _paper_evidence_error(
            "final broker ledger differs from the last cycle order snapshot")
    unknown_orders = sum(
        1 for order_id in reported_order_ids if order_id not in order_ids)

    round_trip_symbols = {
        symbol for symbol in set(filled_buys) | set(filled_sells)
        if filled_buys.get(symbol, 0) > 0
        and filled_buys.get(symbol, 0) == filled_sells.get(symbol, 0)
    }
    if not round_trip_symbols:
        raise _paper_evidence_error("does not prove a completed round trip")
    if any(filled_buys.get(symbol, 0) != filled_sells.get(symbol, 0)
           for symbol in set(filled_buys) | set(filled_sells)):
        raise _paper_evidence_error("filled order ledger does not end flat")

    broker_positions = broker.get("positions")
    if not isinstance(broker_positions, list):
        raise _paper_evidence_error("broker has no final position ledger")
    if any(not isinstance(position, Mapping)
           or not _finite(position.get("quantity"))
           for position in broker_positions):
        raise _paper_evidence_error("broker final position ledger is invalid")
    if any(abs(float(position["quantity"])) > 1e-9
           for position in broker_positions):
        raise _paper_evidence_error("paper broker does not end flat")
    broker_cash = broker.get("cash")
    if not _finite(broker_cash):
        raise _paper_evidence_error("broker has no finite final cash balance")
    local_book = evidence.get("local_book")
    if not isinstance(local_book, Mapping) \
            or local_book.get("initialized") is not True:
        raise _paper_evidence_error("has no initialized independent local book")
    local_positions = local_book.get("positions")
    if not isinstance(local_positions, Mapping) or any(
            not _finite(quantity) for quantity in local_positions.values()):
        raise _paper_evidence_error("local book final positions are invalid")
    if any(abs(float(quantity)) > 1e-9
           for quantity in local_positions.values()):
        raise _paper_evidence_error("independent local book does not end flat")
    local_cash = local_book.get("cash")
    if (not _finite(local_cash)
            or abs(float(local_cash) - float(broker_cash)) > 0.01):
        raise _paper_evidence_error(
            "independent local book cash differs from broker cash")

    audit_verification = evidence.get("audit_verification")
    if not isinstance(audit_verification, Mapping):
        raise _paper_evidence_error("has no audit verification fact")
    audit_verified = (audit_verification.get("ok") is True
                      and audit_verification.get("first_bad_seq") is None)

    reconciliation = {
        "checks": len(raw_cycles) * 2 + 1,
        "failures": reconciliation_failures,
        "unknown_orders": unknown_orders,
        "open_orders": open_orders,
    }
    run_summary = {
        "cycles": len(raw_cycles),
        "sessions": len(execution_dates),
        "fills": fills,
        "errors": cycle_errors + reconciliation_failures
                  + unknown_orders + open_orders,
        "prospective": prospective,
    }
    return run_summary, reconciliation, audit_verified


def _verify_passing_paper_evidence(
        *, evaluation: PaperValidationEvaluation,
        run_summary: Mapping[str, Any],
        reconciliation_evidence: Mapping[str, Any],
        audit_verified: bool,
        evidence: Mapping[str, Any] | None,
        expected_universe: Sequence[str] | None = None) -> None:
    if not evaluation.passed:
        # Failed rehearsals remain persistable diagnostics.  They can never be
        # used for live approval, and retaining them is valuable incident data.
        return
    derived_run, derived_reconciliation, derived_audit = (
        _recompute_authoritative_paper_evidence(
            evidence, expected_universe=expected_universe))
    if _canonical_value(run_summary) != derived_run:
        raise _paper_evidence_error(
            "run_summary disagrees with the sealed cycle/order facts")
    if _canonical_value(reconciliation_evidence) != derived_reconciliation:
        raise _paper_evidence_error(
            "reconciliation_evidence disagrees with the sealed facts")
    if audit_verified is not derived_audit:
        raise _paper_evidence_error(
            "audit_verified disagrees with the sealed audit fact")


@dataclass(frozen=True)
class PaperValidationPolicy:
    """Minimum rehearsal evidence required before live approval.

    The default is a minimum forward soak: twenty cycles across at least
    fifteen distinct sessions (roughly three trading weeks), a completed
    round trip, and every cycle bracketed by reconciliation plus the final
    check. Error, reconciliation-failure, unknown-order, and open-order
    tolerances are intentionally fixed at zero rather than configurable.
    """

    name: str = "authoritative-paper-validation"
    version: str = "3"
    min_cycles: int = 20
    min_sessions: int = 15
    min_fills: int = 2
    min_reconciliation_checks: int = 41
    require_prospective: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("paper validation policy name and version are required")
        for field in ("min_cycles", "min_sessions", "min_fills",
                      "min_reconciliation_checks"):
            value = getattr(self, field)
            if not _nonnegative_integer(value) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(self.require_prospective, bool):
            raise ValueError("require_prospective must be a bool")

    @classmethod
    def default(cls) -> PaperValidationPolicy:
        return cls()

    def evaluate(
            self, run_summary: Mapping[str, Any],
            reconciliation_evidence: Mapping[str, Any], *,
            audit_verified: bool) -> PaperValidationEvaluation:
        cycles = run_summary.get("cycles")
        sessions = run_summary.get("sessions")
        fills = run_summary.get("fills")
        errors = run_summary.get("errors")
        prospective = run_summary.get("prospective")
        reconciliation_checks = reconciliation_evidence.get("checks")
        reconciliation_failures = reconciliation_evidence.get("failures")
        unknown_orders = reconciliation_evidence.get("unknown_orders")
        open_orders = reconciliation_evidence.get("open_orders")
        checks = (
            PromotionCheck(
                "paper_cycles",
                _nonnegative_integer(cycles) and cycles >= self.min_cycles,
                cycles, {"minimum": self.min_cycles}),
            PromotionCheck(
                "paper_sessions",
                _nonnegative_integer(sessions) and sessions >= self.min_sessions,
                sessions, {"minimum": self.min_sessions}),
            PromotionCheck(
                "paper_fills",
                _nonnegative_integer(fills) and fills >= self.min_fills,
                fills, {"minimum": self.min_fills}),
            PromotionCheck(
                "paper_errors",
                _nonnegative_integer(errors) and errors == 0,
                errors, {"maximum": 0}),
            PromotionCheck(
                "prospective_execution",
                (not self.require_prospective) or prospective is True,
                prospective, {"required": self.require_prospective}),
            PromotionCheck(
                "reconciliation_checks",
                _nonnegative_integer(reconciliation_checks)
                and reconciliation_checks >= self.min_reconciliation_checks,
                reconciliation_checks,
                {"minimum": self.min_reconciliation_checks}),
            PromotionCheck(
                "reconciliation_failures",
                _nonnegative_integer(reconciliation_failures)
                and reconciliation_failures == 0,
                reconciliation_failures, {"maximum": 0}),
            PromotionCheck(
                "unknown_orders",
                _nonnegative_integer(unknown_orders) and unknown_orders == 0,
                unknown_orders, {"maximum": 0}),
            PromotionCheck(
                "open_orders",
                _nonnegative_integer(open_orders) and open_orders == 0,
                open_orders, {"maximum": 0}),
            PromotionCheck(
                "audit_verified", audit_verified is True,
                audit_verified, {"required": True}),
        )
        return PaperValidationEvaluation(
            passed=all(check.passed for check in checks), checks=checks)

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @property
    def policy_id(self) -> str:
        encoded = canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PaperValidationPolicy:
        return cls(
            name=str(value["name"]), version=str(value["version"]),
            min_cycles=value["min_cycles"],
            min_sessions=value["min_sessions"],
            min_fills=value["min_fills"],
            min_reconciliation_checks=value["min_reconciliation_checks"],
            require_prospective=value["require_prospective"],
        )


@dataclass(frozen=True)
class PaperValidationArtifact:
    """Content-addressed rehearsal evidence for one research artifact."""

    artifact_hash: str
    payload_json: str

    @classmethod
    def create(
            cls, *, research_artifact: PromotionArtifact,
            run_summary: Mapping[str, Any],
            reconciliation_evidence: Mapping[str, Any],
            audit_verified: bool,
            policy: PaperValidationPolicy | None = None,
            evidence: Mapping[str, Any] | None = None,
            ) -> PaperValidationArtifact:
        research = PromotionArtifact.from_json(research_artifact.to_json())
        policy = policy or PaperValidationPolicy.default()
        if not isinstance(run_summary, Mapping):
            raise TypeError("run_summary must be a mapping")
        if not isinstance(reconciliation_evidence, Mapping):
            raise TypeError("reconciliation_evidence must be a mapping")
        if not isinstance(audit_verified, bool):
            raise TypeError("audit_verified must be a bool")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("paper validation evidence must be a mapping")
        canonical_run = _canonical_value(run_summary)
        canonical_reconciliation = _canonical_value(reconciliation_evidence)
        evaluation = policy.evaluate(
            canonical_run, canonical_reconciliation,
            audit_verified=audit_verified)
        _verify_passing_paper_evidence(
            evaluation=evaluation,
            run_summary=canonical_run,
            reconciliation_evidence=canonical_reconciliation,
            audit_verified=audit_verified,
            evidence=evidence,
            expected_universe=research.payload["universe"],
        )
        payload = {
            "schema_version": 2,
            "research_artifact_hash": research.artifact_hash,
            "strategy_id": research.strategy_id,
            "strategy_version": research.strategy_version,
            "code_sha": research.code_sha,
            "parameters": _canonical_value(research.parameters),
            "run_summary": canonical_run,
            "reconciliation_evidence": canonical_reconciliation,
            "audit_verified": audit_verified,
            "evidence": (_canonical_value(evidence)
                         if evidence is not None else None),
            "policy": policy.to_dict(),
            "policy_id": policy.policy_id,
            "decision": evaluation.to_dict(),
        }
        payload_json = canonical_json(payload)
        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return cls(digest, payload_json)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    @property
    def research_artifact_hash(self) -> str:
        return str(self.payload["research_artifact_hash"])

    @property
    def strategy_id(self) -> str:
        return str(self.payload["strategy_id"])

    @property
    def code_sha(self) -> str:
        return str(self.payload["code_sha"])

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self.payload["parameters"])

    @property
    def passed(self) -> bool:
        return self.payload["decision"]["passed"] is True

    @property
    def policy_id(self) -> str:
        policy = PaperValidationPolicy.from_dict(self.payload["policy"])
        return policy.policy_id

    @property
    def evidence(self) -> dict[str, Any] | None:
        evidence = self.payload.get("evidence")
        return dict(evidence) if evidence is not None else None

    def validate_against(self, research_artifact: PromotionArtifact) -> None:
        research = PromotionArtifact.from_json(research_artifact.to_json())
        expected = {
            "research_artifact_hash": research.artifact_hash,
            "strategy_id": research.strategy_id,
            "strategy_version": research.strategy_version,
            "code_sha": research.code_sha,
            "parameters": _canonical_value(research.parameters),
        }
        for field, value in expected.items():
            if _canonical_value(self.payload.get(field)) != value:
                raise ArtifactIntegrityError(
                    f"paper validation {field} does not match research artifact")
        policy = PaperValidationPolicy.from_dict(self.payload["policy"])
        evaluation = policy.evaluate(
            self.payload["run_summary"],
            self.payload["reconciliation_evidence"],
            audit_verified=self.payload["audit_verified"],
        )
        _verify_passing_paper_evidence(
            evaluation=evaluation,
            run_summary=self.payload["run_summary"],
            reconciliation_evidence=self.payload["reconciliation_evidence"],
            audit_verified=self.payload["audit_verified"],
            evidence=self.payload.get("evidence"),
            expected_universe=research.payload["universe"],
        )

    def to_json(self) -> str:
        return canonical_json({"artifact_hash": self.artifact_hash,
                               "payload": self.payload}) + "\n"

    @classmethod
    def from_json(cls, value: str) -> PaperValidationArtifact:
        try:
            document = json.loads(value)
            claimed_hash = str(document["artifact_hash"])
            payload = document["payload"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "invalid paper validation artifact document") from exc
        payload_json = canonical_json(payload)
        actual_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        if claimed_hash != actual_hash:
            raise ArtifactIntegrityError("paper validation artifact hash mismatch")
        try:
            schema_version = payload["schema_version"]
            if (not isinstance(schema_version, int)
                    or isinstance(schema_version, bool)
                    or schema_version != 2):
                raise ArtifactIntegrityError(
                    "unsupported paper validation artifact schema")
            _validate_sha256(
                payload["research_artifact_hash"],
                field="research_artifact_hash")
            if not str(payload["strategy_id"]).strip():
                raise ValueError("strategy_id is required")
            if not str(payload["strategy_version"]).strip():
                raise ValueError("strategy_version is required")
            if not str(payload["code_sha"]).strip():
                raise ValueError("code_sha is required")
            if not isinstance(payload["parameters"], Mapping):
                raise TypeError("parameters must be a mapping")
            run_summary = payload["run_summary"]
            reconciliation = payload["reconciliation_evidence"]
            if not isinstance(run_summary, Mapping):
                raise TypeError("run_summary must be a mapping")
            if not isinstance(reconciliation, Mapping):
                raise TypeError("reconciliation_evidence must be a mapping")
            if not isinstance(payload["audit_verified"], bool):
                raise TypeError("audit_verified must be a bool")
            evidence = payload.get("evidence")
            if "evidence" not in payload:
                raise TypeError("evidence field is missing")
            if evidence is not None and not isinstance(evidence, Mapping):
                raise TypeError("evidence must be a mapping or null")
            policy = PaperValidationPolicy.from_dict(payload["policy"])
            if _canonical_value(payload["policy"]) != policy.to_dict():
                raise ArtifactIntegrityError(
                    "paper validation policy is not canonical")
            if payload.get("policy_id") != policy.policy_id:
                raise ArtifactIntegrityError(
                    "paper validation policy identity mismatch")
            expected = policy.evaluate(
                run_summary, reconciliation,
                audit_verified=payload["audit_verified"]).to_dict()
            if _canonical_value(payload["decision"]) != expected:
                raise ArtifactIntegrityError(
                    "paper validation decision does not match evidence")
            evaluation = policy.evaluate(
                run_summary, reconciliation,
                audit_verified=payload["audit_verified"])
            _verify_passing_paper_evidence(
                evaluation=evaluation,
                run_summary=run_summary,
                reconciliation_evidence=reconciliation,
                audit_verified=payload["audit_verified"],
                evidence=evidence,
            )
        except ArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "invalid paper validation artifact payload") from exc
        return cls(actual_hash, payload_json)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create(path: Path, content: str) -> None:
    """Atomically create ``path`` without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ArtifactIntegrityError(
                    f"refusing to overwrite existing immutable file: {path}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class ArtifactStore:
    """Filesystem store for immutable, content-addressed artifacts."""

    def __init__(self, root: str | Path, *,
                 accepted_policy: PromotionPolicy | None = None):
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"
        self.accepted_policy = accepted_policy

    @staticmethod
    def _validate_hash(artifact_hash: str) -> str:
        return _validate_sha256(artifact_hash, field="artifact_hash")

    def _require_accepted_policy(self, artifact: PromotionArtifact) -> None:
        if (self.accepted_policy is not None
                and artifact.policy_id != self.accepted_policy.policy_id):
            raise ArtifactIntegrityError(
                "promotion artifact policy is not accepted by this registry")

    def path_for(self, artifact_hash: str) -> Path:
        return self.artifact_dir / f"{self._validate_hash(artifact_hash)}.json"

    def persist(self, artifact: PromotionArtifact) -> Path:
        # Re-parse before persistence so callers cannot construct an invalid
        # dataclass directly and smuggle in mismatched payload bytes.
        verified = PromotionArtifact.from_json(artifact.to_json())
        if verified.artifact_hash != artifact.artifact_hash:
            raise ArtifactIntegrityError("promotion artifact failed verification")
        self._require_accepted_policy(verified)
        path = self.path_for(artifact.artifact_hash)
        _atomic_create(path, artifact.to_json())
        return path

    def load(self, artifact_hash: str) -> PromotionArtifact:
        digest = self._validate_hash(artifact_hash)
        path = self.path_for(digest)
        try:
            artifact = PromotionArtifact.from_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PromotionNotApproved(f"promotion artifact does not exist: {digest}") from exc
        if artifact.artifact_hash != digest:
            raise ArtifactIntegrityError("artifact filename and content hash differ")
        self._require_accepted_policy(artifact)
        return artifact


class PaperValidationStore:
    """Immutable content-addressed paper-validation evidence store."""

    def __init__(
            self, root: str | Path, *,
            research_store: ArtifactStore | None = None,
            accepted_policy: PaperValidationPolicy | None = None):
        self.root = Path(root)
        self.artifact_dir = self.root / "paper-validations"
        self.research_store = research_store
        self.accepted_policy = accepted_policy

    def path_for(self, artifact_hash: str) -> Path:
        digest = _validate_sha256(artifact_hash, field="paper_artifact_hash")
        return self.artifact_dir / f"{digest}.json"

    def _verify(self, artifact: PaperValidationArtifact) -> None:
        if (self.accepted_policy is not None
                and artifact.policy_id != self.accepted_policy.policy_id):
            raise ArtifactIntegrityError(
                "paper validation policy is not accepted by this registry")
        if self.research_store is not None:
            research = self.research_store.load(
                artifact.research_artifact_hash)
            artifact.validate_against(research)
            policy = PaperValidationPolicy.from_dict(
                artifact.payload["policy"])
            evaluation = policy.evaluate(
                artifact.payload["run_summary"],
                artifact.payload["reconciliation_evidence"],
                audit_verified=artifact.payload["audit_verified"],
            )
            _verify_passing_paper_evidence(
                evaluation=evaluation,
                run_summary=artifact.payload["run_summary"],
                reconciliation_evidence=(
                    artifact.payload["reconciliation_evidence"]),
                audit_verified=artifact.payload["audit_verified"],
                evidence=artifact.evidence,
                expected_universe=research.payload["universe"],
            )

    def persist(self, artifact: PaperValidationArtifact) -> Path:
        verified = PaperValidationArtifact.from_json(artifact.to_json())
        if verified.artifact_hash != artifact.artifact_hash:
            raise ArtifactIntegrityError(
                "paper validation artifact failed verification")
        self._verify(verified)
        path = self.path_for(artifact.artifact_hash)
        _atomic_create(path, artifact.to_json())
        return path

    def load(self, artifact_hash: str) -> PaperValidationArtifact:
        digest = _validate_sha256(artifact_hash, field="paper_artifact_hash")
        path = self.path_for(digest)
        try:
            artifact = PaperValidationArtifact.from_json(
                path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PromotionNotApproved(
                f"paper validation artifact does not exist: {digest}") from exc
        if artifact.artifact_hash != digest:
            raise ArtifactIntegrityError(
                "paper validation filename and content hash differ")
        self._verify(artifact)
        return artifact


class PromotionRegistry:
    """Immutable paper/live approval references over an :class:`ArtifactStore`."""

    def __init__(
            self, root: str | Path, *,
            policy: PromotionPolicy | None = None,
            paper_validation_policy: PaperValidationPolicy | None = None):
        self.root = Path(root)
        self.policy = policy or PromotionPolicy.default()
        self.paper_validation_policy = (
            paper_validation_policy or PaperValidationPolicy.default())
        self._pin_policies()
        self.store = ArtifactStore(
            self.root, accepted_policy=self.policy)
        self.paper_validation_store = PaperValidationStore(
            self.root, research_store=self.store,
            accepted_policy=self.paper_validation_policy)
        self.approval_dir = self.root / "approvals"

    def _policy_pin_document(self) -> str:
        return canonical_json({
            "schema_version": 1,
            "promotion_policy": self.policy.to_dict(),
            "promotion_policy_id": self.policy.policy_id,
            "paper_validation_policy": self.paper_validation_policy.to_dict(),
            "paper_validation_policy_id": self.paper_validation_policy.policy_id,
        }) + "\n"

    def _pin_policies(self) -> None:
        path = self.root / "registry-policy.json"
        expected = self._policy_pin_document()
        try:
            _atomic_create(path, expected)
            actual = path.read_text(encoding="utf-8")
        except (ArtifactIntegrityError, OSError) as exc:
            raise ArtifactIntegrityError(
                "registry policy identity does not match the pinned policy") from exc
        if actual != expected:
            raise ArtifactIntegrityError(
                "registry policy identity does not match the pinned policy")

    @staticmethod
    def _strategy_key(strategy_id: str) -> str:
        if not strategy_id.strip():
            raise ValueError("strategy_id is required")
        return hashlib.sha256(strategy_id.strip().encode("utf-8")).hexdigest()

    def _approval_path(self, strategy_id: str, artifact_hash: str,
                       level: PromotionLevel) -> Path:
        tier = "paper" if level == PromotionLevel.PAPER_ELIGIBLE else "live"
        return (self.approval_dir / tier / self._strategy_key(strategy_id)
                / f"{ArtifactStore._validate_hash(artifact_hash)}.json")

    def _approval_document(
            self, strategy_id: str, artifact_hash: str,
            level: PromotionLevel, *, actor: str | None,
            paper_artifact_hash: str | None) -> str:
        return canonical_json({
            "schema_version": 2,
            "strategy_id": strategy_id,
            "artifact_hash": artifact_hash,
            "approved_level": level.value,
            "actor": actor,
            "paper_artifact_hash": paper_artifact_hash,
            "promotion_policy_id": self.policy.policy_id,
            "paper_validation_policy_id": self.paper_validation_policy.policy_id,
        }) + "\n"

    @staticmethod
    def _normalize_actor(actor: str | None, *, required: bool) -> str | None:
        if actor is None:
            if required:
                raise PromotionNotApproved(
                    "live promotion requires an approving actor")
            return None
        normalized = str(actor).strip()
        if not normalized:
            raise PromotionNotApproved("approval actor cannot be empty")
        return normalized

    def _read_approval(
            self, strategy_id: str, artifact_hash: str,
            level: PromotionLevel) -> dict[str, Any]:
        path = self._approval_path(strategy_id, artifact_hash, level)
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromotionNotApproved(
                f"artifact is not explicitly approved for {level.value}") from exc
        except OSError as exc:
            raise ArtifactIntegrityError("cannot read promotion approval") from exc
        try:
            document = json.loads(actual)
            if actual != canonical_json(document) + "\n":
                raise ArtifactIntegrityError(
                    "promotion approval reference is not canonical")
            schema_version = document["schema_version"]
            if (not isinstance(schema_version, int)
                    or isinstance(schema_version, bool)):
                raise ArtifactIntegrityError(
                    "promotion approval reference is invalid")
        except ArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "promotion approval reference is invalid") from exc

        fixed = {
            "strategy_id": strategy_id,
            "artifact_hash": artifact_hash,
            "approved_level": level.value,
        }
        if schema_version == 1:
            expected = {"schema_version": 1, **fixed}
            if document != expected:
                raise ArtifactIntegrityError(
                    "promotion approval reference is invalid")
            if level == PromotionLevel.LIVE_ELIGIBLE:
                raise PromotionNotApproved(
                    "legacy live approval has no paper validation evidence")
            return document
        if schema_version != 2:
            raise ArtifactIntegrityError(
                "unsupported promotion approval schema")
        expected_keys = {
            "schema_version", *fixed.keys(), "actor",
            "paper_artifact_hash", "promotion_policy_id",
            "paper_validation_policy_id",
        }
        if set(document) != expected_keys:
            raise ArtifactIntegrityError(
                "promotion approval reference is invalid")
        if any(document.get(key) != value for key, value in fixed.items()):
            raise ArtifactIntegrityError(
                "promotion approval reference is invalid")
        if document.get("promotion_policy_id") != self.policy.policy_id:
            raise ArtifactIntegrityError(
                "promotion approval policy identity mismatch")
        if (document.get("paper_validation_policy_id")
                != self.paper_validation_policy.policy_id):
            raise ArtifactIntegrityError(
                "paper validation approval policy identity mismatch")
        actor = document.get("actor")
        if actor is not None and (not isinstance(actor, str) or not actor.strip()):
            raise ArtifactIntegrityError("promotion approval actor is invalid")
        paper_hash = document.get("paper_artifact_hash")
        if actor is None:
            raise ArtifactIntegrityError("approval actor is missing")
        if level == PromotionLevel.LIVE_ELIGIBLE:
            try:
                _validate_sha256(paper_hash, field="paper_artifact_hash")
            except ValueError as exc:
                raise ArtifactIntegrityError(
                    "live approval paper evidence reference is invalid") from exc
        elif paper_hash is not None:
            raise ArtifactIntegrityError(
                "paper approval cannot contain live validation evidence")
        return document

    def _eligible_artifact(
            self, strategy_id: str, artifact_hash: str,
            level: PromotionLevel) -> PromotionArtifact:
        artifact = self.store.load(artifact_hash)
        if artifact.strategy_id != strategy_id:
            raise PromotionNotApproved("artifact belongs to a different strategy")
        if _LEVEL_RANK[artifact.decision] < _LEVEL_RANK[level]:
            raise PromotionNotApproved(
                f"artifact decision {artifact.decision.value} does not satisfy {level.value}")
        if artifact.strategy_id == "foundation":
            _require_authoritative_foundation_research(artifact)
        return artifact

    def promote(
            self, strategy_id: str, artifact_hash: str,
            required_level: PromotionLevel, *, actor: str | None = None,
            paper_artifact_hash: str | None = None) -> Path:
        level = PromotionLevel(required_level)
        if level == PromotionLevel.RESEARCH_ONLY:
            raise ValueError("research-only artifacts are not deployment approvals")
        artifact = self._eligible_artifact(strategy_id, artifact_hash, level)
        if level == PromotionLevel.LIVE_ELIGIBLE:
            self.require_approved(strategy_id, artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)
            paper_approval = self._read_approval(
                strategy_id, artifact_hash, PromotionLevel.PAPER_ELIGIBLE)
            approving_actor = self._normalize_actor(actor, required=True)
            paper_actor = paper_approval.get("actor")
            if not isinstance(paper_actor, str) or not paper_actor.strip():
                raise PromotionNotApproved(
                    "live promotion requires an attributable paper approver")
            if approving_actor == paper_actor:
                raise PromotionNotApproved(
                    "live approver must differ from paper approver")
            if paper_artifact_hash is None:
                raise PromotionNotApproved(
                    "live promotion requires a paper validation artifact")
            evidence = self.paper_validation_store.load(paper_artifact_hash)
            if evidence.research_artifact_hash != artifact.artifact_hash:
                raise PromotionNotApproved(
                    "paper validation belongs to a different research artifact")
            evidence.validate_against(artifact)
            if not evidence.passed:
                raise PromotionNotApproved(
                    "paper validation evidence does not satisfy live policy")
            approved_paper_hash = evidence.artifact_hash
        else:
            approving_actor = self._normalize_actor(actor, required=True)
            if paper_artifact_hash is not None:
                raise ValueError(
                    "paper validation evidence is only valid for live approval")
            approved_paper_hash = None
        path = self._approval_path(strategy_id, artifact_hash, level)
        _atomic_create(path, self._approval_document(
            strategy_id, artifact.artifact_hash, level,
            actor=approving_actor,
            paper_artifact_hash=approved_paper_hash))
        return path

    def require_approved(self, strategy_id: str, artifact_hash: str,
                         required_level: PromotionLevel) -> PromotionArtifact:
        level = PromotionLevel(required_level)
        if level == PromotionLevel.RESEARCH_ONLY:
            raise ValueError("an execution approval must be paper or live")
        if level == PromotionLevel.LIVE_ELIGIBLE:
            artifact, _ = self.require_live_approved(
                strategy_id, artifact_hash)
            return artifact
        artifact = self._eligible_artifact(strategy_id, artifact_hash, level)
        self._read_approval(strategy_id, artifact.artifact_hash, level)
        return artifact

    def require_live_approved(
            self, strategy_id: str, artifact_hash: str,
            paper_artifact_hash: str | None = None,
            ) -> tuple[PromotionArtifact, PaperValidationArtifact]:
        """Return both exact artifacts after re-verifying a live approval.

        Supplying ``paper_artifact_hash`` additionally proves that the caller's
        expected rehearsal is the one recorded by the immutable approval.
        """
        artifact = self._eligible_artifact(
            strategy_id, artifact_hash, PromotionLevel.LIVE_ELIGIBLE)
        self.require_approved(
            strategy_id, artifact.artifact_hash,
            PromotionLevel.PAPER_ELIGIBLE)
        approval = self._read_approval(
            strategy_id, artifact.artifact_hash,
            PromotionLevel.LIVE_ELIGIBLE)
        approved_paper_hash = str(approval["paper_artifact_hash"])
        if paper_artifact_hash is not None:
            expected_hash = _validate_sha256(
                paper_artifact_hash, field="paper_artifact_hash")
            if approved_paper_hash != expected_hash:
                raise PromotionNotApproved(
                    "live approval is bound to different paper evidence")
        evidence = self.paper_validation_store.load(approved_paper_hash)
        if evidence.research_artifact_hash != artifact.artifact_hash:
            raise PromotionNotApproved(
                "paper validation belongs to a different research artifact")
        evidence.validate_against(artifact)
        if not evidence.passed:
            raise PromotionNotApproved(
                "paper validation evidence does not satisfy live policy")
        return artifact, evidence
