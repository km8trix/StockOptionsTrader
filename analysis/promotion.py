"""Content-addressed strategy evidence and explicit deployment approvals.

Research results are immutable facts.  Deployment approval is a separate,
equally explicit fact that points at one exact research artifact.  Keeping the
two separate prevents a re-run, a mutable registry entry, or a changed working
tree from silently changing what paper/live execution is allowed to load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, TypeGuard


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
            policy: PromotionPolicy | None = None) -> PromotionArtifact:
        policy = policy or PromotionPolicy.default()
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
            "schema_version": 1,
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
            results = PromotionResults.from_dict(payload["results"])
            policy = PromotionPolicy.from_dict(payload["policy"])
            expected = policy.evaluate(results).to_dict()
            if _canonical_value(payload["decision"]) != expected:
                raise ArtifactIntegrityError("promotion decision does not match evidence")
            if int(payload["schema_version"]) != 1:
                raise ArtifactIntegrityError("unsupported promotion artifact schema")
        except ArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid promotion artifact payload") from exc
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

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"

    @staticmethod
    def _validate_hash(artifact_hash: str) -> str:
        candidate = str(artifact_hash).lower()
        if len(candidate) != 64 or any(ch not in "0123456789abcdef"
                                       for ch in candidate):
            raise ValueError("artifact_hash must be a SHA-256 hex digest")
        return candidate

    def path_for(self, artifact_hash: str) -> Path:
        return self.artifact_dir / f"{self._validate_hash(artifact_hash)}.json"

    def persist(self, artifact: PromotionArtifact) -> Path:
        # Re-parse before persistence so callers cannot construct an invalid
        # dataclass directly and smuggle in mismatched payload bytes.
        verified = PromotionArtifact.from_json(artifact.to_json())
        if verified.artifact_hash != artifact.artifact_hash:
            raise ArtifactIntegrityError("promotion artifact failed verification")
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
        return artifact


class PromotionRegistry:
    """Immutable paper/live approval references over an :class:`ArtifactStore`."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.store = ArtifactStore(self.root)
        self.approval_dir = self.root / "approvals"

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

    @staticmethod
    def _approval_document(strategy_id: str, artifact_hash: str,
                           level: PromotionLevel) -> str:
        return canonical_json({
            "schema_version": 1,
            "strategy_id": strategy_id,
            "artifact_hash": artifact_hash,
            "approved_level": level.value,
        }) + "\n"

    def promote(self, strategy_id: str, artifact_hash: str,
                required_level: PromotionLevel) -> Path:
        level = PromotionLevel(required_level)
        if level == PromotionLevel.RESEARCH_ONLY:
            raise ValueError("research-only artifacts are not deployment approvals")
        artifact = self.store.load(artifact_hash)
        if artifact.strategy_id != strategy_id:
            raise PromotionNotApproved("artifact belongs to a different strategy")
        if _LEVEL_RANK[artifact.decision] < _LEVEL_RANK[level]:
            raise PromotionNotApproved(
                f"artifact decision {artifact.decision.value} does not satisfy {level.value}")
        if level == PromotionLevel.LIVE_ELIGIBLE:
            # A live approval is never allowed to skip paper approval for this
            # exact immutable version.
            self.require_approved(strategy_id, artifact_hash,
                                  PromotionLevel.PAPER_ELIGIBLE)
        path = self._approval_path(strategy_id, artifact_hash, level)
        _atomic_create(path, self._approval_document(
            strategy_id, artifact.artifact_hash, level))
        return path

    def require_approved(self, strategy_id: str, artifact_hash: str,
                         required_level: PromotionLevel) -> PromotionArtifact:
        level = PromotionLevel(required_level)
        if level == PromotionLevel.RESEARCH_ONLY:
            raise ValueError("an execution approval must be paper or live")
        artifact = self.store.load(artifact_hash)
        if artifact.strategy_id != strategy_id:
            raise PromotionNotApproved("artifact belongs to a different strategy")
        if _LEVEL_RANK[artifact.decision] < _LEVEL_RANK[level]:
            raise PromotionNotApproved("artifact evidence does not satisfy requested tier")

        candidate_levels = ([PromotionLevel.PAPER_ELIGIBLE,
                             PromotionLevel.LIVE_ELIGIBLE]
                            if level == PromotionLevel.PAPER_ELIGIBLE
                            else [PromotionLevel.LIVE_ELIGIBLE])
        for approved_level in candidate_levels:
            path = self._approval_path(strategy_id, artifact_hash, approved_level)
            if not path.exists():
                continue
            expected = self._approval_document(
                strategy_id, artifact.artifact_hash, approved_level)
            try:
                actual = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ArtifactIntegrityError("cannot read promotion approval") from exc
            if actual != expected:
                raise ArtifactIntegrityError("promotion approval reference is invalid")
            return artifact
        raise PromotionNotApproved(
            f"artifact is not explicitly approved for {level.value}")
