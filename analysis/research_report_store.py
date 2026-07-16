"""Immutable storage for the raw evidence behind research summaries.

A promotion artifact that stores only a report digest is not independently
recomputable.  This module persists the complete normalized report under that
digest.  The summary artifact may then point at the blob while an auditor can
load the underlying returns, trades, pending orders, and portfolio history and
recompute every statistic.

The store is deliberately content-addressed and create-only: a report can be
added or verified, never replaced in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from analysis.promotion import ArtifactIntegrityError


def _strict_json_loads(value: str) -> Any:
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ArtifactIntegrityError(
                    f"duplicate research report JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(token):
        raise ArtifactIntegrityError(
            f"invalid research report JSON number: {token}")

    try:
        return json.loads(
            value, object_pairs_hook=unique_object,
            parse_constant=invalid_constant)
    except ArtifactIntegrityError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("invalid research report document") from exc


def _normalize(value: Any) -> Any:
    """Return a deterministic JSON value without lossy ``default=str``.

    Research evidence commonly contains numpy scalars and pandas timestamps;
    those have exact, explicit conversions.  Unknown objects are rejected so a
    hash can never silently describe a repr string rather than the evidence.
    """
    if isinstance(value, Enum):
        return _normalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError("research evidence cannot contain NaT")
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("research evidence cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("research evidence mapping keys must be strings")
            normalized[key] = _normalize(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(
        f"unsupported research evidence value: {type(value).__name__}")


def canonical_report_json(report: Mapping[str, Any]) -> str:
    if not isinstance(report, Mapping):
        raise TypeError("research report must be a mapping")
    return json.dumps(
        _normalize(report), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def recompute_foundation_results(
        report: Mapping[str, Any], *, n_trials: int,
        engine_parameters: Mapping[str, Any],
        regimes: list[Mapping[str, Any]]):
    """Recompute every promotable Foundation statistic from raw evidence.

    This intentionally validates, rather than skips, malformed rows.  The same
    function is used when an artifact is built and when it is promoted, making
    a self-consistent raw report—not a caller-supplied summary—the authority.
    """
    from analysis.promotion import PromotionResults

    history = report.get("portfolio_history")
    trades = report.get("trades")
    if not isinstance(history, list) or len(history) < 2:
        raise ValueError("raw report needs at least two NAV observations")
    if not isinstance(trades, list):
        raise ValueError("raw report trades must be a list")
    points: list[tuple[pd.Timestamp, float]] = []
    for row in history:
        if not isinstance(row, Mapping):
            raise ValueError("raw report NAV observation must be a mapping")
        try:
            timestamp = pd.Timestamp(row["timestamp"])
            nav = float(row["portfolio_value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw report NAV observation is invalid") from exc
        if pd.isna(timestamp) or not math.isfinite(nav) or nav <= 0:
            raise ValueError("raw report NAV observation is invalid")
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        points.append((timestamp, nav))
    points.sort(key=lambda item: item[0])
    if len({timestamp for timestamp, _ in points}) != len(points):
        raise ValueError("raw report has duplicate NAV timestamps")
    series = pd.Series(
        [nav for _, nav in points],
        index=pd.DatetimeIndex([timestamp for timestamp, _ in points]),
        dtype=float,
    )
    returns = series.pct_change().dropna()
    if returns.empty or not all(math.isfinite(float(item)) for item in returns):
        raise ValueError("raw report has no finite OOS return series")

    regime_results: dict[str, float] = {}
    for item in regimes:
        if not isinstance(item, Mapping):
            raise ValueError("research regime must be a mapping")
        try:
            name = str(item["name"])
            start = pd.Timestamp(item["start"])
            end = pd.Timestamp(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("research regime is invalid") from exc
        window = series.loc[(series.index >= start) & (series.index <= end)]
        if not name or len(window) < 2:
            raise ValueError(f"raw report does not cover regime {name!r}")
        regime_results[name] = float(window.iloc[-1] / window.iloc[0] - 1.0)
    if len(regime_results) != len(regimes):
        raise ValueError("research regime names must be unique")

    notional = 0.0
    for trade in trades:
        if not isinstance(trade, Mapping):
            raise ValueError("raw report trade must be a mapping")
        try:
            quantity = abs(float(trade["quantity"]))
            price = abs(float(trade["price"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw report trade economics are invalid") from exc
        if not math.isfinite(quantity) or not math.isfinite(price):
            raise ValueError("raw report trade economics are invalid")
        notional += quantity * price
    elapsed_days = max(1, int((series.index[-1] - series.index[0]).days))
    years = elapsed_days / 365.25
    average_nav = float(series.mean())
    if not math.isfinite(average_nav) or average_nav <= 0:
        raise ValueError("raw report average NAV is invalid")
    annual_turnover = float(notional / average_nav / years)

    try:
        commission = float(engine_parameters["commission"])
        slippage_bps = float(engine_parameters["slippage_bps"])
        impact_coef = float(engine_parameters["impact_coef"])
        participation_cap = float(engine_parameters["participation_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research execution economics are incomplete") from exc
    estimated_cost_bps = float(
        commission * 10_000
        + slippage_bps
        + impact_coef * math.sqrt(participation_cap) * 10_000
    )
    return PromotionResults.from_oos_returns(
        [float(item) for item in returns],
        [int(timestamp.year) for timestamp in returns.index],
        n_trials=int(n_trials),
        cost_model_applied=True,
        estimated_cost_bps=estimated_cost_bps,
        annual_turnover=annual_turnover,
        regime_results=regime_results,
    )


@dataclass(frozen=True)
class ResearchReportArtifact:
    """One complete raw report identified by its canonical SHA-256."""

    report_hash: str
    payload_json: str

    @classmethod
    def create(cls, report: Mapping[str, Any]) -> "ResearchReportArtifact":
        payload = canonical_report_json(report)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return cls(digest, payload)

    @property
    def report(self) -> dict[str, Any]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise ArtifactIntegrityError("research report payload is not an object")
        return value

    def to_json(self) -> str:
        return json.dumps(
            {"report_hash": self.report_hash, "report": self.report},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "ResearchReportArtifact":
        try:
            document = _strict_json_loads(value)
            if not isinstance(document, Mapping) or set(document) != {
                    "report_hash", "report"}:
                raise ArtifactIntegrityError(
                    "research report document has invalid fields")
            claimed = document["report_hash"]
            if (not isinstance(claimed, str) or len(claimed) != 64
                    or any(ch not in "0123456789abcdef" for ch in claimed)):
                raise ArtifactIntegrityError(
                    "research report hash must be a SHA-256 digest")
            report = document["report"]
        except ArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid research report document") from exc
        payload = canonical_report_json(report)
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if claimed != actual:
            raise ArtifactIntegrityError("research report hash mismatch")
        return cls(actual, payload)


def _atomic_create(path: Path, content: str) -> None:
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
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ArtifactIntegrityError(
                    f"refusing to overwrite immutable research report: {path}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class ResearchReportStore:
    """Create-only filesystem store for complete research reports."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.report_dir = self.root / "research-reports"

    @staticmethod
    def _validate_hash(report_hash: str) -> str:
        candidate = str(report_hash).lower()
        if (len(candidate) != 64
                or any(ch not in "0123456789abcdef" for ch in candidate)):
            raise ValueError("report_hash must be a SHA-256 hex digest")
        return candidate

    def path_for(self, report_hash: str) -> Path:
        return self.report_dir / f"{self._validate_hash(report_hash)}.json"

    def persist(self, artifact: ResearchReportArtifact) -> Path:
        verified = ResearchReportArtifact.from_json(artifact.to_json())
        if verified.report_hash != artifact.report_hash:
            raise ArtifactIntegrityError("research report failed verification")
        path = self.path_for(verified.report_hash)
        _atomic_create(path, verified.to_json())
        return path

    def load(self, report_hash: str) -> ResearchReportArtifact:
        digest = self._validate_hash(report_hash)
        try:
            artifact = ResearchReportArtifact.from_json(
                self.path_for(digest).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"research report does not exist: {digest}") from exc
        if artifact.report_hash != digest:
            raise ArtifactIntegrityError(
                "research report filename and content hash differ")
        return artifact


__all__ = [
    "ResearchReportArtifact",
    "ResearchReportStore",
    "canonical_report_json",
    "recompute_foundation_results",
]
