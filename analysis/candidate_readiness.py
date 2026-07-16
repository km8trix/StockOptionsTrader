"""Fail-closed readiness checks that run before research preregistration.

This module does not freeze a protocol, register a trial, or open a holdout.
It answers the narrower question that must come first: is the candidate's
research package complete enough to freeze without filling material gaps after
outcomes become visible?

The package is intentionally candidate-neutral.  It binds the clean source
revision and exact warehouse snapshot, inventories prior attempts, and requires
content-addressed evidence for independent review/data, borrow, costs, and an
uncontaminated holdout or prospective acquisition plan.  A claimed attestation
is never silently converted into evidence: every referenced artifact must be a
present, nonempty file whose bytes match the declared SHA-256 digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping, Sequence

from data.pit_warehouse import PitWarehouse


SCHEMA_VERSION = 1
BASE_EQUITY_TABLES = frozenset({"actions", "daily", "sep", "tickers"})
KNOWN_WAREHOUSE_TABLES = frozenset(
    {
        "actions",
        "daily",
        "events",
        "sep",
        "sf1",
        "sf2",
        "sf3",
        "tickers",
    }
)
REQUIRED_REVIEW_SCOPES = frozenset(
    {
        "data",
        "economics",
        "implementation",
        "statistics",
        "trial_inventory",
    }
)
_HEX = frozenset("0123456789abcdef")


class ReadinessPackageError(ValueError):
    """The readiness package is malformed, ambiguous, or not exact-schema."""


@dataclass(frozen=True, slots=True)
class ReadinessBlocker:
    """One independently actionable reason a package cannot be frozen."""

    code: str
    path: str
    message: str

    def to_mapping(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True, slots=True)
class CandidateReadinessReport:
    """Deterministic assessment result suitable for CLI JSON output."""

    candidate_id: str
    package_hash: str
    review_subject_hash: str
    ready_to_freeze: bool
    ready_to_run_holdout: bool
    git: Mapping[str, Any]
    warehouse_snapshot: Mapping[str, Any]
    blockers: tuple[ReadinessBlocker, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ok": self.ready_to_freeze,
            "candidate_id": self.candidate_id,
            "package_hash": self.package_hash,
            "review_subject_hash": self.review_subject_hash,
            "ready_to_freeze": self.ready_to_freeze,
            "ready_to_run_holdout": self.ready_to_run_holdout,
            "git": dict(self.git),
            "warehouse_snapshot": dict(self.warehouse_snapshot),
            "blocker_count": len(self.blockers),
            "blockers": [blocker.to_mapping() for blocker in self.blockers],
        }


_TOP_FIELDS = {
    "schema_version",
    "candidate_id",
    "candidate_family",
    "candidate_author",
    "development_source_id",
    "code_sha",
    "warehouse_snapshot_hash",
    "required_tables",
    "candidate_specification",
    "development_window",
    "historical_inventory",
    "independent_review",
    "independent_data",
    "borrow_evidence",
    "cost_evidence",
    "holdout_plan",
}
_SPECIFICATION_FIELDS = {"artifact_file", "artifact_sha256", "status"}
_WINDOW_FIELDS = {"start", "end"}
_INVENTORY_FIELDS = {
    "manifest_file",
    "manifest_sha256",
    "known_attempt_ids",
    "complete",
    "includes_failed_and_abandoned",
    "attested_by",
    "attested_at",
}
_REVIEW_FIELDS = {
    "report_file",
    "report_sha256",
    "reviewer",
    "independent",
    "approved",
    "reviewed_at",
    "scopes",
    "reviewed_subject_sha256",
    "reviewed_candidate_specification_sha256",
}
_INDEPENDENT_DATA_FIELDS = {
    "manifest_file",
    "manifest_sha256",
    "source_id",
    "independent_of_development",
    "point_in_time",
    "coverage_start",
    "coverage_end",
    "reviewed_by",
}
_BORROW_FIELDS = {
    "shorting_required",
    "status",
    "manifest_file",
    "manifest_sha256",
    "locates_complete",
    "fees_complete",
    "financing_complete",
    "coverage_start",
    "coverage_end",
    "non_applicability_reason",
}
_COST_FIELDS = {
    "manifest_file",
    "manifest_sha256",
    "commissions_complete",
    "spreads_complete",
    "slippage_complete",
    "impact_complete",
    "financing_complete",
    "capacity_complete",
    "coverage_start",
    "coverage_end",
    "capital_levels",
    "calibrated_at",
}
_HOLDOUT_FIELDS = {
    "mode",
    "start",
    "end",
    "source_id",
    "data_manifest_file",
    "data_manifest_sha256",
    "acquisition_plan_file",
    "acquisition_plan_sha256",
    "access_log_file",
    "access_log_sha256",
    "contamination_review_file",
    "contamination_review_sha256",
    "reviewed_by",
    "uncontaminated",
    "outcome_accessed",
    "sealed",
    "sealed_location",
}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReadinessPackageError(
            "readiness package must contain only finite JSON-native values"
        ) from exc


def _exact_fields(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadinessPackageError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReadinessPackageError(f"{path} has invalid fields (missing={missing}, extra={extra})")
    return dict(value)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReadinessPackageError(f"{path} must be a nonempty trimmed string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ReadinessPackageError(f"{path} must be a boolean")
    return value


def _sha256(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    candidate = _text(value, path).lower()
    if len(candidate) != 64 or any(character not in _HEX for character in candidate):
        raise ReadinessPackageError(f"{path} must be a SHA-256 hex digest")
    return candidate


def _git_revision(value: Any, path: str) -> str:
    candidate = _text(value, path).lower()
    if len(candidate) != 40 or any(character not in _HEX for character in candidate):
        raise ReadinessPackageError(f"{path} must be a full 40-character Git SHA")
    return candidate


def _iso_date(value: Any, path: str, *, nullable: bool = False) -> date | None:
    if value is None and nullable:
        return None
    candidate = _text(value, path)
    try:
        parsed = date.fromisoformat(candidate)
    except ValueError as exc:
        raise ReadinessPackageError(f"{path} must be an ISO calendar date") from exc
    if candidate != parsed.isoformat():
        raise ReadinessPackageError(f"{path} must be canonical: {parsed.isoformat()}")
    return parsed


def _nullable_text(value: Any, path: str) -> str | None:
    return None if value is None else _text(value, path)


def _utc_timestamp(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    candidate = _text(value, path)
    if not candidate.endswith("Z"):
        raise ReadinessPackageError(f"{path} must be canonical UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(candidate[:-1] + "+00:00")
    except ValueError as exc:
        raise ReadinessPackageError(f"{path} must be a valid UTC timestamp") from exc
    parsed = parsed.astimezone(timezone.utc)
    timespec = "microseconds" if parsed.microsecond else "seconds"
    canonical = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if candidate != canonical:
        raise ReadinessPackageError(f"{path} must be canonical: {canonical}")
    return canonical


def _artifact_file(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    candidate = _text(value, path)
    pure = PurePosixPath(candidate)
    if pure.is_absolute() or ".." in pure.parts or candidate != pure.as_posix():
        raise ReadinessPackageError(f"{path} must be a normalized relative POSIX path")
    return candidate


def _text_list(value: Any, path: str, *, nonempty: bool = True) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReadinessPackageError(f"{path} must be an array of strings")
    result = [_text(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if nonempty and not result:
        raise ReadinessPackageError(f"{path} cannot be empty")
    if len(result) != len(set(result)):
        raise ReadinessPackageError(f"{path} cannot contain duplicates")
    if result != sorted(result):
        raise ReadinessPackageError(f"{path} must be sorted")
    return result


def validate_readiness_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the strict schema and return a detached JSON-native mapping."""
    top = _exact_fields(package, _TOP_FIELDS, "package")
    if type(top["schema_version"]) is not int or top["schema_version"] != SCHEMA_VERSION:
        raise ReadinessPackageError(f"package.schema_version must be {SCHEMA_VERSION}")
    for field in (
        "candidate_id",
        "candidate_family",
        "candidate_author",
        "development_source_id",
    ):
        _text(top[field], f"package.{field}")
    _git_revision(top["code_sha"], "package.code_sha")
    _sha256(top["warehouse_snapshot_hash"], "package.warehouse_snapshot_hash")

    tables = _text_list(top["required_tables"], "package.required_tables")
    unknown = sorted(set(tables) - KNOWN_WAREHOUSE_TABLES)
    if unknown:
        raise ReadinessPackageError(f"unknown required warehouse tables: {unknown}")
    missing_base = sorted(BASE_EQUITY_TABLES - set(tables))
    if missing_base:
        raise ReadinessPackageError(
            f"package.required_tables omits baseline equity tables: {missing_base}"
        )

    specification = _exact_fields(
        top["candidate_specification"],
        _SPECIFICATION_FIELDS,
        "package.candidate_specification",
    )
    _artifact_file(specification["artifact_file"], "candidate_specification.artifact_file")
    _sha256(specification["artifact_sha256"], "candidate_specification.artifact_sha256")
    if specification["status"] not in {"draft", "final"}:
        raise ReadinessPackageError("candidate_specification.status must be 'draft' or 'final'")

    development = _exact_fields(
        top["development_window"], _WINDOW_FIELDS, "package.development_window"
    )
    development_start = _iso_date(development["start"], "package.development_window.start")
    development_end = _iso_date(development["end"], "package.development_window.end")
    if development_start > development_end:
        raise ReadinessPackageError("development window start cannot exceed end")

    inventory = _exact_fields(
        top["historical_inventory"],
        _INVENTORY_FIELDS,
        "package.historical_inventory",
    )
    _artifact_file(inventory["manifest_file"], "historical_inventory.manifest_file")
    _sha256(inventory["manifest_sha256"], "historical_inventory.manifest_sha256")
    _text_list(
        inventory["known_attempt_ids"],
        "historical_inventory.known_attempt_ids",
        nonempty=False,
    )
    _boolean(inventory["complete"], "historical_inventory.complete")
    _boolean(
        inventory["includes_failed_and_abandoned"],
        "historical_inventory.includes_failed_and_abandoned",
    )
    _nullable_text(inventory["attested_by"], "historical_inventory.attested_by")
    _utc_timestamp(inventory["attested_at"], "historical_inventory.attested_at", nullable=True)

    review = _exact_fields(top["independent_review"], _REVIEW_FIELDS, "package.independent_review")
    _artifact_file(review["report_file"], "independent_review.report_file", nullable=True)
    _sha256(review["report_sha256"], "independent_review.report_sha256", nullable=True)
    _nullable_text(review["reviewer"], "independent_review.reviewer")
    _boolean(review["independent"], "independent_review.independent")
    _boolean(review["approved"], "independent_review.approved")
    _utc_timestamp(review["reviewed_at"], "independent_review.reviewed_at", nullable=True)
    _text_list(review["scopes"], "independent_review.scopes", nonempty=False)
    _sha256(
        review["reviewed_subject_sha256"],
        "independent_review.reviewed_subject_sha256",
        nullable=True,
    )
    _sha256(
        review["reviewed_candidate_specification_sha256"],
        "independent_review.reviewed_candidate_specification_sha256",
        nullable=True,
    )

    independent_data = _exact_fields(
        top["independent_data"],
        _INDEPENDENT_DATA_FIELDS,
        "package.independent_data",
    )
    _artifact_file(
        independent_data["manifest_file"],
        "independent_data.manifest_file",
        nullable=True,
    )
    _sha256(
        independent_data["manifest_sha256"],
        "independent_data.manifest_sha256",
        nullable=True,
    )
    _nullable_text(independent_data["source_id"], "independent_data.source_id")
    _boolean(
        independent_data["independent_of_development"],
        "independent_data.independent_of_development",
    )
    _boolean(independent_data["point_in_time"], "independent_data.point_in_time")
    _iso_date(
        independent_data["coverage_start"],
        "independent_data.coverage_start",
        nullable=True,
    )
    _iso_date(
        independent_data["coverage_end"],
        "independent_data.coverage_end",
        nullable=True,
    )
    _nullable_text(independent_data["reviewed_by"], "independent_data.reviewed_by")

    borrow = _exact_fields(top["borrow_evidence"], _BORROW_FIELDS, "package.borrow_evidence")
    _boolean(borrow["shorting_required"], "borrow_evidence.shorting_required")
    if borrow["status"] not in {"complete", "not_applicable", "pending"}:
        raise ReadinessPackageError(
            "borrow_evidence.status must be complete, not_applicable, or pending"
        )
    _artifact_file(borrow["manifest_file"], "borrow_evidence.manifest_file", nullable=True)
    _sha256(borrow["manifest_sha256"], "borrow_evidence.manifest_sha256", nullable=True)
    for field in ("locates_complete", "fees_complete", "financing_complete"):
        _boolean(borrow[field], f"borrow_evidence.{field}")
    _iso_date(borrow["coverage_start"], "borrow_evidence.coverage_start", nullable=True)
    _iso_date(borrow["coverage_end"], "borrow_evidence.coverage_end", nullable=True)
    if borrow["non_applicability_reason"] is not None:
        _text(
            borrow["non_applicability_reason"],
            "borrow_evidence.non_applicability_reason",
        )
    costs = _exact_fields(top["cost_evidence"], _COST_FIELDS, "package.cost_evidence")
    _artifact_file(costs["manifest_file"], "cost_evidence.manifest_file", nullable=True)
    _sha256(costs["manifest_sha256"], "cost_evidence.manifest_sha256", nullable=True)
    for field in (
        "commissions_complete",
        "spreads_complete",
        "slippage_complete",
        "impact_complete",
        "financing_complete",
        "capacity_complete",
    ):
        _boolean(costs[field], f"cost_evidence.{field}")
    _iso_date(costs["coverage_start"], "cost_evidence.coverage_start", nullable=True)
    _iso_date(costs["coverage_end"], "cost_evidence.coverage_end", nullable=True)
    levels = costs["capital_levels"]
    if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
        raise ReadinessPackageError("cost_evidence.capital_levels must be an array")
    normalized_levels: list[float] = []
    for index, level in enumerate(levels):
        if isinstance(level, bool) or not isinstance(level, (int, float)):
            raise ReadinessPackageError(f"cost_evidence.capital_levels[{index}] must be a number")
        value = float(level)
        if not math.isfinite(value) or value <= 0:
            raise ReadinessPackageError(
                f"cost_evidence.capital_levels[{index}] must be finite and positive"
            )
        normalized_levels.append(value)
    if normalized_levels != sorted(set(normalized_levels)):
        raise ReadinessPackageError("cost_evidence.capital_levels must be sorted and unique")
    _utc_timestamp(costs["calibrated_at"], "cost_evidence.calibrated_at", nullable=True)

    holdout = _exact_fields(top["holdout_plan"], _HOLDOUT_FIELDS, "package.holdout_plan")
    if holdout["mode"] not in {"existing_uncontaminated", "prospective_acquisition"}:
        raise ReadinessPackageError(
            "holdout_plan.mode must be existing_uncontaminated or prospective_acquisition"
        )
    holdout_start = _iso_date(holdout["start"], "holdout_plan.start", nullable=True)
    holdout_end = _iso_date(holdout["end"], "holdout_plan.end", nullable=True)
    if holdout_start is not None and holdout_end is not None and holdout_start > holdout_end:
        raise ReadinessPackageError("holdout start cannot exceed end")
    _nullable_text(holdout["source_id"], "holdout_plan.source_id")
    for field in (
        "data_manifest_file",
        "acquisition_plan_file",
        "access_log_file",
        "contamination_review_file",
    ):
        _artifact_file(holdout[field], f"holdout_plan.{field}", nullable=True)
    for field in (
        "data_manifest_sha256",
        "acquisition_plan_sha256",
        "access_log_sha256",
        "contamination_review_sha256",
    ):
        _sha256(holdout[field], f"holdout_plan.{field}", nullable=True)
    _nullable_text(holdout["reviewed_by"], "holdout_plan.reviewed_by")
    for field in ("uncontaminated", "outcome_accessed", "sealed"):
        _boolean(holdout[field], f"holdout_plan.{field}")
    _nullable_text(holdout["sealed_location"], "holdout_plan.sealed_location")

    # Round-trip through strict JSON so callers cannot mutate nested references
    # after validation and every package hash has one deterministic meaning.
    return json.loads(_canonical_json(top))


def _git_state(repo_dir: str | Path) -> dict[str, Any]:
    root = Path(repo_dir).resolve()

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )

    try:
        top = run("rev-parse", "--show-toplevel")
        revision = run("rev-parse", "HEAD")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "repo_root": str(root),
            "sha": None,
            "clean": False,
            "dirty_paths": [],
            "error": str(exc),
        }
    if any(result.returncode != 0 for result in (top, revision, status)):
        error = next(
            (result.stderr.strip() for result in (top, revision, status) if result.returncode != 0),
            "git command failed",
        )
        return {
            "available": False,
            "repo_root": str(root),
            "sha": None,
            "clean": False,
            "dirty_paths": [],
            "error": error,
        }
    sha = revision.stdout.strip().lower()
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "available": True,
        "repo_root": top.stdout.strip(),
        "sha": sha,
        "clean": not dirty,
        "dirty_paths": dirty,
        "error": None,
    }


def _normalize_git_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "available": value.get("available") is True,
        "repo_root": str(value.get("repo_root") or ""),
        "sha": value.get("sha") if isinstance(value.get("sha"), str) else None,
        "clean": value.get("clean") is True,
        "dirty_paths": [str(item) for item in value.get("dirty_paths", [])],
        "error": str(value["error"]) if value.get("error") is not None else None,
    }


def _verify_artifact(
    *,
    evidence_root: Path,
    file_value: str | None,
    digest_value: str | None,
    path: str,
    blockers: list[ReadinessBlocker],
) -> Path | None:
    if file_value is None or digest_value is None:
        blockers.append(
            ReadinessBlocker(
                "evidence_reference_missing",
                path,
                "both artifact file and SHA-256 digest are required",
            )
        )
        return None
    root = evidence_root.resolve()
    unresolved = root / file_value
    target = unresolved.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        blockers.append(
            ReadinessBlocker(
                "evidence_path_escape",
                path,
                "artifact path escapes the evidence root",
            )
        )
        return None
    relative_parts = PurePosixPath(file_value).parts
    if any(
        (root.joinpath(*relative_parts[:index])).is_symlink()
        for index in range(1, len(relative_parts) + 1)
    ):
        blockers.append(
            ReadinessBlocker(
                "evidence_symlink_rejected",
                path,
                f"artifact path contains a symlink: {file_value}",
            )
        )
        return None
    if not target.is_file():
        blockers.append(
            ReadinessBlocker(
                "evidence_file_missing",
                path,
                f"artifact is not a present regular file: {file_value}",
            )
        )
        return None
    if target.stat().st_size <= 0:
        blockers.append(
            ReadinessBlocker(
                "evidence_file_empty",
                path,
                f"artifact is empty: {file_value}",
            )
        )
        return None
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != digest_value:
        blockers.append(
            ReadinessBlocker(
                "evidence_hash_mismatch",
                path,
                f"artifact bytes do not match the declared digest: {file_value}",
            )
        )
        return None
    return target


def _verified_json_object(
    target: Path | None,
    *,
    path: str,
    blockers: list[ReadinessBlocker],
) -> dict[str, Any] | None:
    """Read a small, duplicate-key-free JSON evidence object."""
    if target is None:
        return None
    if target.stat().st_size > 2 * 1024 * 1024:
        blockers.append(
            ReadinessBlocker(
                "evidence_json_too_large",
                path,
                "structured evidence JSON exceeds the 2 MiB limit",
            )
        )
        return None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"invalid JSON number {token}")

    try:
        document = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        blockers.append(
            ReadinessBlocker(
                "evidence_json_invalid",
                path,
                f"structured evidence is not strict JSON: {exc}",
            )
        )
        return None
    if not isinstance(document, dict):
        blockers.append(
            ReadinessBlocker(
                "evidence_json_invalid",
                path,
                "structured evidence must be a JSON object",
            )
        )
        return None
    return document


def _coverage_blockers(
    *,
    section: Mapping[str, Any],
    path: str,
    required_start: date,
    required_end: date,
    blockers: list[ReadinessBlocker],
    nullable: bool = False,
) -> None:
    start = _iso_date(section["coverage_start"], f"{path}.coverage_start", nullable=nullable)
    end = _iso_date(section["coverage_end"], f"{path}.coverage_end", nullable=nullable)
    if start is None or end is None:
        blockers.append(
            ReadinessBlocker(
                "coverage_missing",
                path,
                "dated evidence coverage is required",
            )
        )
        return
    if start > end:
        blockers.append(
            ReadinessBlocker(
                "coverage_invalid",
                path,
                "coverage start is after coverage end",
            )
        )
    elif start > required_start or end < required_end:
        blockers.append(
            ReadinessBlocker(
                "coverage_incomplete",
                path,
                "evidence does not cover the complete required research window",
            )
        )


def _coverage_contains(section: Mapping[str, Any], start: date, end: date) -> bool:
    coverage_start = _iso_date(section["coverage_start"], "coverage_start", nullable=True)
    coverage_end = _iso_date(section["coverage_end"], "coverage_end", nullable=True)
    return (
        coverage_start is not None
        and coverage_end is not None
        and coverage_start <= start
        and coverage_end >= end
    )


def assess_candidate_readiness(
    package: Mapping[str, Any],
    *,
    warehouse_dir: str | Path,
    repo_dir: str | Path = ".",
    evidence_root: str | Path = ".",
    git_state: Mapping[str, Any] | None = None,
    warehouse_snapshot: Mapping[str, Any] | None = None,
    as_of: date | None = None,
) -> CandidateReadinessReport:
    """Assess a package without mutating any research or holdout state."""
    validated = validate_readiness_package(package)
    blockers: list[ReadinessBlocker] = []
    author = validated["candidate_author"]
    development_start = _iso_date(
        validated["development_window"]["start"], "development_window.start"
    )
    development_end = _iso_date(validated["development_window"]["end"], "development_window.end")
    holdout = validated["holdout_plan"]
    holdout_start = _iso_date(holdout["start"], "holdout_plan.start", nullable=True)
    holdout_end = _iso_date(holdout["end"], "holdout_plan.end", nullable=True)
    specification = validated["candidate_specification"]
    review_subject = dict(validated)
    # The independent-review section is excluded to avoid a self-referential
    # digest. The resulting subject still binds every research input and the
    # content hash of the executable candidate specification.
    review_subject.pop("independent_review")
    review_subject_hash = hashlib.sha256(
        _canonical_json(review_subject).encode("utf-8")
    ).hexdigest()

    actual_git = _normalize_git_state(git_state or _git_state(repo_dir))
    sha = actual_git["sha"]
    if not actual_git["available"]:
        blockers.append(
            ReadinessBlocker(
                "git_unavailable",
                "git",
                actual_git["error"] or "Git state is unavailable",
            )
        )
    elif not isinstance(sha, str) or len(sha) != 40 or any(ch not in _HEX for ch in sha):
        blockers.append(
            ReadinessBlocker(
                "git_sha_invalid",
                "git.sha",
                "Git HEAD is not a full SHA-1 revision",
            )
        )
    elif sha != validated["code_sha"]:
        blockers.append(
            ReadinessBlocker(
                "git_sha_mismatch",
                "package.code_sha",
                "the package is bound to a different source revision",
            )
        )
    if not actual_git["clean"]:
        blockers.append(
            ReadinessBlocker(
                "git_dirty",
                "git.clean",
                "qualifying preregistration requires a clean working tree",
            )
        )

    if warehouse_snapshot is None:
        try:
            warehouse_snapshot = PitWarehouse(str(warehouse_dir)).snapshot_version(
                validated["required_tables"]
            )
        except Exception as exc:  # warehouse read errors are blockers, never passes
            warehouse_snapshot = {
                "version": None,
                "tables": [],
                "complete": False,
                "quality_flags": [f"snapshot_error:{type(exc).__name__}"],
            }
            blockers.append(
                ReadinessBlocker(
                    "warehouse_snapshot_error",
                    "warehouse_snapshot",
                    str(exc),
                )
            )
    snapshot = dict(warehouse_snapshot)
    manifest = snapshot.get("tables") if isinstance(snapshot.get("tables"), list) else []
    if (
        not isinstance(snapshot.get("version"), str)
        or len(snapshot["version"]) != 64
        or any(character not in _HEX for character in snapshot["version"])
    ):
        blockers.append(
            ReadinessBlocker(
                "warehouse_version_invalid",
                "warehouse_snapshot.version",
                "warehouse snapshot has no valid SHA-256 version",
            )
        )
    seen_tables: set[str] = set()
    manifest_entries: list[tuple[str, str, int]] = []
    manifest_is_valid = True
    for index, item in enumerate(manifest):
        item_path = f"warehouse_snapshot.tables[{index}]"
        if not isinstance(item, Mapping):
            blockers.append(
                ReadinessBlocker(
                    "warehouse_manifest_invalid",
                    item_path,
                    "warehouse manifest entry must be an object",
                )
            )
            manifest_is_valid = False
            continue
        table = item.get("table")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (
            not isinstance(table, str)
            or table in seen_tables
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
            or type(size) is not int
            or size <= 0
        ):
            blockers.append(
                ReadinessBlocker(
                    "warehouse_manifest_invalid",
                    item_path,
                    "warehouse manifest entry has invalid or duplicate fields",
                )
            )
            manifest_is_valid = False
        else:
            manifest_entries.append((table, digest, size))
        if isinstance(table, str):
            seen_tables.add(table)
    present = {
        item.get("table") for item in manifest if isinstance(item, Mapping) and item.get("table")
    }
    for table in sorted(set(validated["required_tables"]) - present):
        blockers.append(
            ReadinessBlocker(
                "warehouse_table_missing",
                f"warehouse.{table}",
                f"required table {table!r} is absent from the content manifest",
            )
        )
    for table in sorted(present - set(validated["required_tables"])):
        blockers.append(
            ReadinessBlocker(
                "warehouse_table_unexpected",
                f"warehouse.{table}",
                f"table {table!r} was not declared by the package",
            )
        )
    if manifest_is_valid:
        manifest_digest = hashlib.sha256()
        for table, digest, size in sorted(manifest_entries):
            manifest_digest.update(f"{table}:{digest}:{size}\n".encode("utf-8"))
        if snapshot.get("version") != manifest_digest.hexdigest():
            blockers.append(
                ReadinessBlocker(
                    "warehouse_manifest_hash_mismatch",
                    "warehouse_snapshot.version",
                    "warehouse snapshot version differs from its table manifest",
                )
            )
    if snapshot.get("complete") is not True:
        blockers.append(
            ReadinessBlocker(
                "warehouse_snapshot_incomplete",
                "warehouse_snapshot.complete",
                "the required warehouse snapshot is incomplete",
            )
        )
    if snapshot.get("version") != validated["warehouse_snapshot_hash"]:
        blockers.append(
            ReadinessBlocker(
                "warehouse_snapshot_mismatch",
                "package.warehouse_snapshot_hash",
                "the package is bound to different warehouse bytes",
            )
        )

    root = Path(evidence_root)
    specification_target = _verify_artifact(
        evidence_root=root,
        file_value=specification["artifact_file"],
        digest_value=specification["artifact_sha256"],
        path="candidate_specification.artifact",
        blockers=blockers,
    )
    review_section = validated["independent_review"]
    review_target = _verify_artifact(
        evidence_root=root,
        file_value=review_section["report_file"],
        digest_value=review_section["report_sha256"],
        path="independent_review.report",
        blockers=blockers,
    )
    references = (
        (
            validated["historical_inventory"],
            "manifest_file",
            "manifest_sha256",
            "historical_inventory.manifest",
        ),
        (
            validated["independent_data"],
            "manifest_file",
            "manifest_sha256",
            "independent_data.manifest",
        ),
        (validated["cost_evidence"], "manifest_file", "manifest_sha256", "cost_evidence.manifest"),
        (holdout, "access_log_file", "access_log_sha256", "holdout_plan.access_log"),
        (
            holdout,
            "contamination_review_file",
            "contamination_review_sha256",
            "holdout_plan.contamination_review",
        ),
    )
    for section, file_field, hash_field, path in references:
        _verify_artifact(
            evidence_root=root,
            file_value=section[file_field],
            digest_value=section[hash_field],
            path=path,
            blockers=blockers,
        )

    specification_document = _verified_json_object(
        specification_target,
        path="candidate_specification.artifact",
        blockers=blockers,
    )
    if specification_document is not None:
        expected_identity = {
            "candidate_id": validated["candidate_id"],
            "candidate_family": validated["candidate_family"],
            "status": specification["status"],
        }
        for field, expected in expected_identity.items():
            if specification_document.get(field) != expected:
                blockers.append(
                    ReadinessBlocker(
                        "candidate_specification_identity_mismatch",
                        f"candidate_specification.artifact.{field}",
                        f"candidate specification {field!r} does not match the package",
                    )
                )

    if specification["status"] != "final":
        blockers.append(
            ReadinessBlocker(
                "candidate_specification_not_final",
                "candidate_specification.status",
                "the executable candidate specification is still a draft",
            )
        )

    inventory = validated["historical_inventory"]
    if inventory["complete"] is not True:
        blockers.append(
            ReadinessBlocker(
                "trial_inventory_incomplete",
                "historical_inventory.complete",
                "all known research attempts must be inventoried before freezing",
            )
        )
    if inventory["includes_failed_and_abandoned"] is not True:
        blockers.append(
            ReadinessBlocker(
                "trial_inventory_omits_failures",
                "historical_inventory.includes_failed_and_abandoned",
                "failed and abandoned attempts must be included",
            )
        )
    if inventory["attested_by"] is None or inventory["attested_at"] is None:
        blockers.append(
            ReadinessBlocker(
                "trial_inventory_unattested",
                "historical_inventory.attested_by",
                "the completed inventory requires a named, timestamped attestation",
            )
        )

    review = validated["independent_review"]
    review_document = _verified_json_object(
        review_target,
        path="independent_review.report",
        blockers=blockers,
    )
    if review_document is not None:
        evidence_bindings = {
            "reviewed_subject_sha256": review_subject_hash,
            "reviewed_candidate_specification_sha256": specification["artifact_sha256"],
        }
        for field, expected in evidence_bindings.items():
            if review_document.get(field) != expected:
                blockers.append(
                    ReadinessBlocker(
                        "review_evidence_binding_mismatch",
                        f"independent_review.report.{field}",
                        f"review artifact does not bind the expected {field!r}",
                    )
                )
    if review["reviewer"] is None:
        blockers.append(
            ReadinessBlocker(
                "reviewer_missing",
                "independent_review.reviewer",
                "an independent reviewer has not been assigned",
            )
        )
    if review["reviewer"] == author or review["independent"] is not True:
        blockers.append(
            ReadinessBlocker(
                "review_not_independent",
                "independent_review.reviewer",
                "the package requires a reviewer independent of the candidate author",
            )
        )
    if review["approved"] is not True:
        blockers.append(
            ReadinessBlocker(
                "review_not_approved",
                "independent_review.approved",
                "independent review has not approved the package for freezing",
            )
        )
    if review["reviewed_at"] is None:
        blockers.append(
            ReadinessBlocker(
                "review_timestamp_missing",
                "independent_review.reviewed_at",
                "independent review has no completion timestamp",
            )
        )
    if review["reviewed_subject_sha256"] != review_subject_hash:
        blockers.append(
            ReadinessBlocker(
                "review_subject_unbound",
                "independent_review.reviewed_subject_sha256",
                "review does not bind the current non-review readiness package",
            )
        )
    if review["reviewed_candidate_specification_sha256"] != specification["artifact_sha256"]:
        blockers.append(
            ReadinessBlocker(
                "review_specification_unbound",
                "independent_review.reviewed_candidate_specification_sha256",
                "review does not bind the current candidate specification",
            )
        )
    missing_scopes = sorted(REQUIRED_REVIEW_SCOPES - set(review["scopes"]))
    if missing_scopes:
        blockers.append(
            ReadinessBlocker(
                "review_scope_incomplete",
                "independent_review.scopes",
                f"independent review is missing scopes: {missing_scopes}",
            )
        )

    independent_data = validated["independent_data"]
    if independent_data["source_id"] is None:
        blockers.append(
            ReadinessBlocker(
                "independent_data_source_missing",
                "independent_data.source_id",
                "an independent data source has not been selected",
            )
        )
    if (
        independent_data["source_id"] == validated["development_source_id"]
        or independent_data["independent_of_development"] is not True
    ):
        blockers.append(
            ReadinessBlocker(
                "data_not_independent",
                "independent_data.source_id",
                "replication data must be acquired independently of development data",
            )
        )
    if independent_data["point_in_time"] is not True:
        blockers.append(
            ReadinessBlocker(
                "independent_data_not_pit",
                "independent_data.point_in_time",
                "independent replication data must be point-in-time",
            )
        )
    if independent_data["reviewed_by"] is None:
        blockers.append(
            ReadinessBlocker(
                "independent_data_reviewer_missing",
                "independent_data.reviewed_by",
                "independent data has not been reviewed",
            )
        )
    elif independent_data["reviewed_by"] == author:
        blockers.append(
            ReadinessBlocker(
                "independent_data_self_reviewed",
                "independent_data.reviewed_by",
                "independent data cannot be reviewed only by the candidate author",
            )
        )
    _coverage_blockers(
        section=independent_data,
        path="independent_data",
        required_start=development_start,
        required_end=development_end,
        blockers=blockers,
        nullable=True,
    )

    borrow = validated["borrow_evidence"]
    if borrow["shorting_required"]:
        _verify_artifact(
            evidence_root=root,
            file_value=borrow["manifest_file"],
            digest_value=borrow["manifest_sha256"],
            path="borrow_evidence.manifest",
            blockers=blockers,
        )
        if borrow["status"] != "complete":
            blockers.append(
                ReadinessBlocker(
                    "borrow_evidence_incomplete",
                    "borrow_evidence.status",
                    "a short strategy requires complete dated borrow evidence",
                )
            )
        for field in ("locates_complete", "fees_complete", "financing_complete"):
            if borrow[field] is not True:
                blockers.append(
                    ReadinessBlocker(
                        f"borrow_{field}",
                        f"borrow_evidence.{field}",
                        f"borrow evidence field {field!r} must be true",
                    )
                )
        _coverage_blockers(
            section=borrow,
            path="borrow_evidence",
            required_start=development_start,
            required_end=development_end,
            blockers=blockers,
            nullable=True,
        )
    else:
        if borrow["status"] != "not_applicable" or not borrow["non_applicability_reason"]:
            blockers.append(
                ReadinessBlocker(
                    "borrow_non_applicability_unexplained",
                    "borrow_evidence",
                    "a no-shorting strategy must document why borrow is not applicable",
                )
            )

    costs = validated["cost_evidence"]
    for field in (
        "commissions_complete",
        "spreads_complete",
        "slippage_complete",
        "impact_complete",
        "financing_complete",
        "capacity_complete",
    ):
        if costs[field] is not True:
            blockers.append(
                ReadinessBlocker(
                    f"cost_{field}",
                    f"cost_evidence.{field}",
                    f"cost evidence field {field!r} must be true",
                )
            )
    _coverage_blockers(
        section=costs,
        path="cost_evidence",
        required_start=development_start,
        required_end=development_end,
        blockers=blockers,
        nullable=True,
    )

    if costs["calibrated_at"] is None:
        blockers.append(
            ReadinessBlocker(
                "cost_calibration_missing",
                "cost_evidence.calibrated_at",
                "cost evidence has not been calibrated",
            )
        )
    if not costs["capital_levels"]:
        blockers.append(
            ReadinessBlocker(
                "cost_capital_levels_missing",
                "cost_evidence.capital_levels",
                "capacity evidence requires at least one intended capital level",
            )
        )

    if holdout_start is None or holdout_end is None:
        blockers.append(
            ReadinessBlocker(
                "holdout_dates_missing",
                "holdout_plan.start",
                "the holdout or prospective acquisition window is unresolved",
            )
        )
    elif holdout_start <= development_end:
        blockers.append(
            ReadinessBlocker(
                "holdout_overlaps_development",
                "holdout_plan.start",
                "the temporal holdout must begin after the development window",
            )
        )
    if holdout["source_id"] is None:
        blockers.append(
            ReadinessBlocker(
                "holdout_source_missing",
                "holdout_plan.source_id",
                "the holdout acquisition source is unresolved",
            )
        )
    if holdout["reviewed_by"] is None:
        blockers.append(
            ReadinessBlocker(
                "holdout_reviewer_missing",
                "holdout_plan.reviewed_by",
                "the holdout contamination plan has not been independently reviewed",
            )
        )
    elif holdout["reviewed_by"] == author:
        blockers.append(
            ReadinessBlocker(
                "holdout_self_reviewed",
                "holdout_plan.reviewed_by",
                "holdout contamination review must be independent",
            )
        )
    if holdout["uncontaminated"] is not True or holdout["outcome_accessed"] is not False:
        blockers.append(
            ReadinessBlocker(
                "holdout_contaminated",
                "holdout_plan",
                "the holdout must be attested uncontaminated with outcomes unaccessed",
            )
        )
    if holdout["sealed"] is not True:
        blockers.append(
            ReadinessBlocker(
                "holdout_not_sealed",
                "holdout_plan.sealed",
                "the holdout or acquisition plan must be sealed",
            )
        )
    if holdout["sealed_location"] is None:
        blockers.append(
            ReadinessBlocker(
                "holdout_seal_location_missing",
                "holdout_plan.sealed_location",
                "no immutable seal location has been selected",
            )
        )

    mode = holdout["mode"]
    if mode == "existing_uncontaminated":
        _verify_artifact(
            evidence_root=root,
            file_value=holdout["data_manifest_file"],
            digest_value=holdout["data_manifest_sha256"],
            path="holdout_plan.data_manifest",
            blockers=blockers,
        )
        if (
            holdout["acquisition_plan_file"] is not None
            or holdout["acquisition_plan_sha256"] is not None
        ):
            blockers.append(
                ReadinessBlocker(
                    "holdout_mode_conflict",
                    "holdout_plan.acquisition_plan_file",
                    "an existing holdout cannot also claim a prospective acquisition plan",
                )
            )
    else:
        _verify_artifact(
            evidence_root=root,
            file_value=holdout["acquisition_plan_file"],
            digest_value=holdout["acquisition_plan_sha256"],
            path="holdout_plan.acquisition_plan",
            blockers=blockers,
        )
        if holdout["data_manifest_file"] is not None or holdout["data_manifest_sha256"] is not None:
            blockers.append(
                ReadinessBlocker(
                    "holdout_mode_conflict",
                    "holdout_plan.data_manifest_file",
                    "prospective acquisition cannot claim that holdout data already exists",
                )
            )
        today = as_of or datetime.now(timezone.utc).date()
        if holdout_start is not None and holdout_start <= today:
            blockers.append(
                ReadinessBlocker(
                    "prospective_holdout_not_future",
                    "holdout_plan.start",
                    "prospective acquisition must begin after the assessment date",
                )
            )

    blockers = sorted(blockers, key=lambda item: (item.code, item.path, item.message))
    ready = not blockers
    run_coverage_complete = False
    if holdout_start is not None and holdout_end is not None:
        run_coverage_complete = (
            _coverage_contains(independent_data, holdout_start, holdout_end)
            and _coverage_contains(costs, holdout_start, holdout_end)
            and (
                not borrow["shorting_required"]
                or _coverage_contains(borrow, holdout_start, holdout_end)
            )
        )
    package_hash = hashlib.sha256(_canonical_json(validated).encode("utf-8")).hexdigest()
    return CandidateReadinessReport(
        candidate_id=validated["candidate_id"],
        package_hash=package_hash,
        review_subject_hash=review_subject_hash,
        ready_to_freeze=ready,
        ready_to_run_holdout=(
            ready and mode == "existing_uncontaminated" and run_coverage_complete
        ),
        git=actual_git,
        warehouse_snapshot=snapshot,
        blockers=tuple(blockers),
    )


__all__ = [
    "BASE_EQUITY_TABLES",
    "CandidateReadinessReport",
    "ReadinessBlocker",
    "ReadinessPackageError",
    "SCHEMA_VERSION",
    "assess_candidate_readiness",
    "validate_readiness_package",
]
