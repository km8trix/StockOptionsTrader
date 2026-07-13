"""Fail-closed preparation, activation, and rollback for Foundation live use.

The only public path to an autonomous production session consumes a staged
``DeploymentManifest``.  Preparation re-verifies the research and paper
evidence, exact build/configuration, account identity, authentication, audit
chain, local book, open orders, reservations, risk policy, and reconciliation
while the kill switch remains engaged.  A capability created by that check is
then required by ``LiveExecutionContext.configure_session`` in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from analysis.promotion import PromotionLevel, canonical_json
from core.models import Asset, AssetType, Position
from deployment.state import (
    DeploymentManifest,
    DeploymentState,
    DeploymentStateError,
    DeploymentStore,
)
from desks.deployment_config import FoundationDeploymentConfig
from desks.registry import create_deployed_desk
from portfolio.manager import PortfolioManager
from utils.provenance import capture_run_provenance
from utils.market_hours import MarketHours, NYSE_TZ


_CAPABILITY_TOKEN = object()
FIRST_CYCLE_TIMEOUT_SECONDS = 11 * 60.0
MAX_LIVE_QUOTE_AGE_SECONDS = 60.0
_NONTERMINAL = frozenset({
    "OPEN", "PARTIAL", "PARTIALLY_FILLED", "PENDING", "PENDING_REVIEW",
    "QUEUED", "CANCEL_REQUESTED",
})


class LiveDeploymentPreflightError(RuntimeError):
    """One production activation invariant was not proven."""


def validate_realtime_equity_quote(
        quote: Any, *, symbol: str, now: datetime,
        max_age_seconds: float = MAX_LIVE_QUOTE_AGE_SECONDS,
        ) -> dict[str, float]:
    """Validate a real-time E*TRADE quote for controlled execution."""
    if not isinstance(quote, Mapping):
        raise DeploymentStateError(f"no live quote for {symbol}")
    if quote.get("quote_status") != "REALTIME":
        raise DeploymentStateError(
            f"controlled deployment requires a REALTIME quote for {symbol}")
    try:
        observed_at = datetime.fromisoformat(str(quote["observed_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentStateError(
            f"live quote for {symbol} has no provider timestamp") from exc
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise DeploymentStateError(
            f"live quote for {symbol} has a timezone-naive timestamp")
    if now.tzinfo is None or now.utcoffset() is None:
        raise DeploymentStateError("live execution clock must be timezone-aware")
    age = (now.astimezone(timezone.utc)
           - observed_at.astimezone(timezone.utc)).total_seconds()
    if age < -5.0 or age > float(max_age_seconds):
        raise DeploymentStateError(
            f"live quote for {symbol} is future-dated or stale")
    try:
        bid = float(quote["bid"])
        ask = float(quote["ask"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentStateError(
            f"live quote for {symbol} has no bid/ask market") from exc
    if (not math.isfinite(bid) or not math.isfinite(ask)
            or bid <= 0 or ask <= 0 or bid > ask):
        raise DeploymentStateError(
            f"live quote for {symbol} has an invalid bid/ask market")
    last = quote.get("last")
    price = (bid + ask) / 2.0
    if last is not None and _finite_live_price(last):
        price = float(last)
    return {"bid": bid, "ask": ask, "price": price}


def _finite_live_price(value: Any) -> bool:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(price) and price > 0


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _require_clean_runtime(provenance: Mapping[str, Any],
                           manifest: DeploymentManifest) -> None:
    sha = provenance.get("git_sha")
    if sha != manifest.code_sha:
        raise LiveDeploymentPreflightError(
            "runtime Git SHA does not match deployment manifest")
    if provenance.get("git_dirty") is not False:
        raise LiveDeploymentPreflightError(
            "production deployment requires a clean working tree")


def _paper_from_registry(registry, manifest: DeploymentManifest):
    """Use the evidence-aware registry API; older registries fail closed."""
    verifier = getattr(registry, "require_live_approved", None)
    if not callable(verifier):
        raise LiveDeploymentPreflightError(
            "promotion registry cannot verify live paper evidence")
    try:
        result = verifier(
            "foundation", manifest.research_artifact_hash,
            manifest.paper_evidence_hash,
        )
    except TypeError:
        # Permit a keyword spelling while keeping missing-evidence APIs closed.
        result = verifier(
            "foundation", manifest.research_artifact_hash,
            paper_artifact_hash=manifest.paper_evidence_hash,
        )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    research = getattr(result, "research_artifact", None)
    paper = getattr(result, "paper_artifact", None)
    if research is None or paper is None:
        raise LiveDeploymentPreflightError(
            "promotion registry returned no bound paper evidence")
    return research, paper


def _previous_exchange_session(day) -> Any:
    calendar = MarketHours()
    candidate = day - timedelta(days=1)
    for _ in range(10):
        if calendar.is_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)
    raise LiveDeploymentPreflightError(
        "cannot resolve the previous exchange session")


def _paper_handoff_checkpoint(paper, now: datetime) -> Mapping[str, Any]:
    """Return the exact recent checkpoint sealed by the paper artifact."""
    evidence = getattr(paper, "evidence", None)
    if not isinstance(evidence, Mapping):
        raise LiveDeploymentPreflightError(
            "paper artifact has no authoritative runner evidence")
    cycles = evidence.get("cycles")
    checkpoint = evidence.get("model_checkpoint")
    if (not isinstance(cycles, list) or not cycles
            or not isinstance(cycles[-1], Mapping)
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("cycle_id") != cycles[-1].get("cycle_id")
            or not isinstance(checkpoint.get("state"), Mapping)):
        raise LiveDeploymentPreflightError(
            "paper artifact has no final-cycle model checkpoint")
    try:
        final_execution = datetime.fromisoformat(str(cycles[-1]["as_of"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveDeploymentPreflightError(
            "paper artifact has an invalid final execution timestamp") from exc
    if final_execution.tzinfo is None or final_execution.utcoffset() is None:
        raise LiveDeploymentPreflightError(
            "paper artifact final execution timestamp is timezone-naive")
    final_execution = final_execution.astimezone(timezone.utc)
    if final_execution > now + timedelta(seconds=1):
        raise LiveDeploymentPreflightError(
            "paper handoff is future-dated")
    local_today = now.astimezone(NYSE_TZ).date()
    prior_session = _previous_exchange_session(local_today)
    allowed_dates = {prior_session}
    if MarketHours().is_trading_day(local_today):
        allowed_dates.add(local_today)
    final_date = final_execution.astimezone(NYSE_TZ).date()
    if final_date not in allowed_dates:
        raise LiveDeploymentPreflightError(
            "paper handoff is stale; its final cycle must be from the current "
            "or immediately preceding NYSE session")

    state = checkpoint["state"]
    try:
        cadence = state["payload"]["cadence"]
        last_seen_date = cadence["last_seen_date"]
    except (KeyError, TypeError) as exc:
        raise LiveDeploymentPreflightError(
            "paper checkpoint has no cadence handoff") from exc
    expected_signal_date = _previous_exchange_session(final_date).isoformat()
    if last_seen_date != expected_signal_date:
        raise LiveDeploymentPreflightError(
            "paper checkpoint cadence does not match its final signal session")
    return state


class _BoundedLiveData:
    """Refuse unknown, missing, future, or stale daily inputs per manifest."""

    def __init__(self, source: Callable[[], Mapping[str, pd.DataFrame]],
                 manifest: DeploymentManifest,
                 clock: Callable[[], datetime]):
        if not callable(source):
            raise TypeError("data_fn must be callable")
        self.source = source
        self.manifest = manifest
        self.clock = clock

    def __call__(self) -> dict[str, pd.DataFrame]:
        raw = self.source()
        if not isinstance(raw, Mapping):
            raise LiveDeploymentPreflightError(
                "live data source must return a symbol mapping")
        normalized = {str(key).strip().upper(): value
                      for key, value in raw.items()}
        if len(normalized) != len(raw):
            raise LiveDeploymentPreflightError(
                "live data contains duplicate normalized symbols")
        approved = set(self.manifest.allowed_universe)
        actual = set(normalized)
        if actual != approved:
            missing, unknown = sorted(approved - actual), sorted(actual - approved)
            raise LiveDeploymentPreflightError(
                f"live data universe mismatch (missing={missing}, unknown={unknown})")
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise LiveDeploymentPreflightError(
                "live data clock must be timezone-aware")
        local_now = now.astimezone(NYSE_TZ)
        calendar = MarketHours()
        if not calendar.is_trading_day(local_now.date()):
            raise LiveDeploymentPreflightError(
                "live data was requested outside an exchange session")
        previous_session = local_now.date() - timedelta(days=1)
        for _ in range(10):
            if calendar.is_trading_day(previous_session):
                break
            previous_session -= timedelta(days=1)
        else:  # pragma: no cover - static calendar defensive bound
            raise LiveDeploymentPreflightError(
                "cannot resolve the previous exchange session")
        for symbol, frame in normalized.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise LiveDeploymentPreflightError(
                    f"live data is empty for {symbol}")
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(frame.columns):
                raise LiveDeploymentPreflightError(
                    f"live data for {symbol} lacks OHLCV columns")
            try:
                index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            except (TypeError, ValueError) as exc:
                raise LiveDeploymentPreflightError(
                    f"live data index for {symbol} is not datetime-like") \
                    from exc
            if (index.hasnans or not index.is_unique
                    or not index.is_monotonic_increasing):
                raise LiveDeploymentPreflightError(
                    f"live data index for {symbol} must be unique and "
                    "monotonic increasing")
            if index.max().date() != previous_session:
                raise LiveDeploymentPreflightError(
                    f"live signal data for {symbol} must end on previous "
                    f"exchange session {previous_session}")
        return normalized


class FoundationExecutionGuard:
    """Final exact-quantity limit check directly before broker execution."""

    def __init__(self, manifest: DeploymentManifest,
                 store: DeploymentStore, broker):
        self.manifest = manifest
        self.store = store
        self.broker = broker
        self._arm_notional_envelope: Optional[Callable[..., None]] = None

    def bind_executor(self, executor: Any) -> None:
        """Bind durable authorization to every patient child/fill.

        ``authorize_order`` persists a conservative notional reservation.  A
        later quote must not let a cancel/replace child spend beyond it, so the
        verified composition root requires the executor's one-shot envelope
        protocol before automation can be configured.
        """
        enabler = getattr(
            executor, "require_controlled_notional_envelopes", None)
        armer = getattr(executor, "arm_controlled_notional_envelope", None)
        if not callable(enabler) or not callable(armer):
            raise LiveDeploymentPreflightError(
                "controlled live executor cannot enforce durable notional "
                "envelopes")
        enabler()
        self._arm_notional_envelope = armer

    def __call__(self, *, intent, side: str, quantity: int,
                 now: datetime) -> None:
        if intent.asset.asset_type is not AssetType.STOCK:
            raise DeploymentStateError(
                "Foundation deployment may execute equities only")
        action = str(intent.action).upper()
        if action not in {"BUY", "SELL"}:
            raise DeploymentStateError(
                f"Foundation deployment does not permit {action}")
        if side != action:
            raise DeploymentStateError("execution side differs from intent")
        intent_id = str(getattr(intent, "intent_id", "") or "").strip()
        if not intent_id:
            raise DeploymentStateError(
                "controlled deployment requires a deterministic intent id")
        quote_reader = getattr(self.broker, "get_current_quote", None)
        if not callable(quote_reader):
            raise DeploymentStateError(
                "controlled live broker cannot prove quote freshness")
        market = validate_realtime_equity_quote(
            quote_reader(intent.asset.symbol),
            symbol=intent.asset.symbol, now=now)
        # Authorize opening notional at the executable side of the market,
        # never a potentially older last trade or optimistic midpoint.
        price = market["ask"] if side == "BUY" else market["bid"]
        opening = action == "BUY"
        authorization = self.store.authorize_order(
            self.manifest.manifest_hash,
            intent_id=intent_id,
            side=side,
            symbol=intent.asset.symbol,
            quantity=int(quantity),
            reference_price=float(price),
            trading_date=now.astimezone(timezone.utc).date().isoformat(),
            opening=opening,
        )
        # In canonical production wiring bind_executor() is mandatory.  Keep
        # the guard independently usable for preflight/store diagnostics where
        # no execution follows (including offline operator inspection).
        if self._arm_notional_envelope is not None:
            self._arm_notional_envelope(
                execution_id=intent_id,
                side=side,
                symbol=intent.asset.symbol,
                quantity=int(quantity),
                opening=opening,
                max_notional=(authorization["notional"]
                              if opening else None),
            )


@dataclass(frozen=True, slots=True)
class VerifiedFoundationDeployment:
    """Opaque result of a successful production preflight."""

    manifest: DeploymentManifest
    context_identity: Any
    desk: Any
    portfolio: PortfolioManager
    data_fn: Callable[[], Mapping[str, pd.DataFrame]]
    execution_guard: FoundationExecutionGuard
    checked_at: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedFoundationDeployment can only be created by preflight")

    def bind(self, context, desk, interval_minutes: float) -> FoundationExecutionGuard:
        if context.identity != self.context_identity:
            raise LiveDeploymentPreflightError(
                "verified deployment belongs to another live context")
        if desk is not self.desk:
            raise LiveDeploymentPreflightError(
                "configured desk is not the preflighted instance")
        identity = getattr(desk, "deployment_identity", None)
        if identity is None or identity.artifact_hash \
                != self.manifest.research_artifact_hash:
            raise LiveDeploymentPreflightError(
                "desk identity differs from manifest research artifact")
        if float(interval_minutes) != float(self.manifest.interval_minutes):
            raise LiveDeploymentPreflightError(
                "scheduler interval differs from manifest")
        return self.execution_guard


class FoundationLiveController:
    """Prepare, start, pause, and revoke one Foundation deployment."""

    def __init__(self, *, store: DeploymentStore, promotion_registry,
                 context, clock: Callable[[], datetime] = _now_utc,
                 provenance_fn: Callable[..., Mapping[str, Any]] =
                 capture_run_provenance):
        self.store = store
        self.registry = promotion_registry
        self.context = context
        self.clock = clock
        self.provenance_fn = provenance_fn
        self._recover_interrupted_deployments()

    def _recover_interrupted_deployments(self) -> None:
        """A process restart never inherits an armed/running permission.

        Scheduler threads are process-local but the manifest and kill switch
        survive.  If a previous process died after disengaging the switch, the
        next controller construction engages it first and records PAUSED.  A
        new manifest/approval is then required; there is no silent resume.
        """
        active = self.store.records([
            DeploymentState.ACTIVATED,
            DeploymentState.ARMED,
            DeploymentState.RUNNING,
        ])
        for record in active:
            manifest = record["manifest"]
            if (manifest.environment != self.context.identity.env
                    or manifest.account_id_key
                    != self.context.identity.account_id_key):
                continue
            reason = "process restart requires explicit redeployment"
            self.context.kill_switch.engage(reason, "deployment_recovery")
            scheduler = getattr(self.context, "scheduler", None)
            if scheduler is not None:
                scheduler.stop()
            self.store.pause(
                manifest.manifest_hash, "deployment_recovery", reason)

    def _basic_preflight(
            self, manifest: DeploymentManifest, *, require_unconfigured: bool,
            ) -> tuple[Any, Any, Mapping[str, Any]]:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise LiveDeploymentPreflightError("preflight clock must be aware")
        if datetime.fromisoformat(manifest.expires_at) <= now:
            raise LiveDeploymentPreflightError("deployment manifest has expired")
        if self.context.identity.env != "production" \
                or self.context.identity.env != manifest.environment:
            raise LiveDeploymentPreflightError(
                "live context environment does not match production manifest")
        if self.context.identity.account_id_key != manifest.account_id_key:
            raise LiveDeploymentPreflightError(
                "live context account does not match manifest")
        if self.context.state != "ready":
            raise LiveDeploymentPreflightError("live context is not ready")
        if require_unconfigured and (self.context.session is not None
                                     or self.context.scheduler is not None):
            raise LiveDeploymentPreflightError(
                "live context already has an automation session")
        if not self.context.kill_switch.engaged():
            raise LiveDeploymentPreflightError(
                "kill switch must remain engaged throughout preflight")
        chain = self.context.audit.verify_chain()
        if chain.get("ok") is not True:
            raise LiveDeploymentPreflightError("audit chain verification failed")
        status_reader = getattr(self.context.auth_manager, "status", None)
        if not callable(status_reader) or status_reader().get("state") != "connected":
            raise LiveDeploymentPreflightError(
                "broker authentication is not connected")
        if not self.context.local_book.is_initialized():
            raise LiveDeploymentPreflightError("local book is uninitialized")

        research, paper = _paper_from_registry(self.registry, manifest)
        if research.artifact_hash != manifest.research_artifact_hash:
            raise LiveDeploymentPreflightError("research evidence hash mismatch")
        if getattr(paper, "artifact_hash", None) != manifest.paper_evidence_hash:
            raise LiveDeploymentPreflightError("paper evidence hash mismatch")
        config = FoundationDeploymentConfig.from_mapping(research.parameters)
        if research.strategy_version != manifest.strategy_version \
                or config.strategy_version != manifest.strategy_version:
            raise LiveDeploymentPreflightError("strategy version mismatch")
        if config.config_hash != manifest.config_hash:
            raise LiveDeploymentPreflightError("strategy configuration mismatch")
        if research.code_sha != manifest.code_sha:
            raise LiveDeploymentPreflightError("research code identity mismatch")
        if set(manifest.allowed_universe) != set(research.payload["universe"]):
            raise LiveDeploymentPreflightError(
                "manifest universe must exactly match researched universe")
        _require_clean_runtime(dict(self.provenance_fn()), manifest)
        checkpoint = _paper_handoff_checkpoint(paper, now)

        policy = getattr(self.context.reservation_gate, "policy", None)
        if policy is None:
            raise LiveDeploymentPreflightError("live risk gate has no policy")
        if float(policy.gross_nav_multiple) \
                > float(manifest.max_gross_nav_multiple):
            raise LiveDeploymentPreflightError(
                "runtime gross risk cap is weaker than manifest")
        if float(policy.per_name_nav_fraction) \
                > float(manifest.max_per_name_nav_fraction):
            raise LiveDeploymentPreflightError(
                "runtime per-name risk cap is weaker than manifest")
        return research, paper, checkpoint

    def _require_no_working_risk(self) -> None:
        orders = self.context.working_orders()
        if any(str(item.get("status") or "").upper() in _NONTERMINAL
               for item in orders):
            raise LiveDeploymentPreflightError(
                "broker has nonterminal working orders")
        reservations = self.context.reservation_snapshot().get(
            "reservations", [])
        if any(str(item.get("status") or "").upper() == "ACTIVE"
               for item in reservations):
            raise LiveDeploymentPreflightError(
                "risk ledger has active reservations")

    def _flat_portfolio(self, manifest: DeploymentManifest) -> PortfolioManager:
        snapshot = self.context.local_book.reconciliation_snapshot()
        positions = {key: value for key, value in snapshot["positions"].items()
                     if abs(float(value)) > 1e-9}
        if manifest.require_flat_start and positions:
            raise LiveDeploymentPreflightError(
                "controlled V1 deployment requires a flat account")
        broker_status = self.context.broker.get_portfolio_status()
        broker_positions = [row for row in broker_status.get("positions", [])
                            if abs(float(row.get("quantity", 0))) > 1e-9]
        if manifest.require_flat_start and broker_positions:
            raise LiveDeploymentPreflightError(
                "broker account is not flat")
        cash = float(broker_status.get("cash", snapshot["cash"]))
        total = float(broker_status.get("portfolio_value", cash) or cash)
        if not math.isfinite(cash) or not math.isfinite(total) or total <= 0:
            raise LiveDeploymentPreflightError("broker NAV is invalid")
        portfolio = PortfolioManager(total)
        portfolio.cash = cash
        # Future non-flat manifests can reconstruct stock holdings, but V1 is
        # intentionally isolated/flat.  Keep the logic exact if that flag is
        # ever relaxed for stock-only accounts.
        if not manifest.require_flat_start:
            for row in broker_positions:
                key = str(row["symbol"])
                if " " in key:
                    raise LiveDeploymentPreflightError(
                        "non-flat V1 supports stock positions only")
                price = float(row.get("current_price", 0))
                quantity = float(row["quantity"])
                if not quantity.is_integer():
                    raise LiveDeploymentPreflightError(
                        "non-flat V1 requires whole-share stock positions")
                portfolio.add_position(Position(
                    Asset(key, AssetType.STOCK), int(quantity),
                    price, price, self.clock()))
        return portfolio

    def prepare(self, manifest_hash: str, *, data_fn: Callable,
                actor: str) -> VerifiedFoundationDeployment:
        record = self.store.get(manifest_hash)
        if record["state"] != DeploymentState.OPS_APPROVED:
            raise DeploymentStateError(
                "deployment must have distinct risk and operations approvals")
        manifest = record["manifest"]
        _research, _paper, checkpoint = self._basic_preflight(
            manifest, require_unconfigured=True)
        self._require_no_working_risk()
        reconciliation = self.context.run_reconciliation(cash_tolerance=0.01)
        if reconciliation.get("ok") is not True:
            raise LiveDeploymentPreflightError("preflight reconciliation failed")
        portfolio = self._flat_portfolio(manifest)
        desk = create_deployed_desk(
            "foundation",
            artifact_hash=manifest.research_artifact_hash,
            promotion_registry=self.registry,
            required_level=PromotionLevel.LIVE_ELIGIBLE,
        )
        try:
            desk.restore_model_checkpoint(checkpoint)
            restored = desk.model_checkpoint_state()
        except Exception as exc:
            raise LiveDeploymentPreflightError(
                "paper model checkpoint is incompatible with the approved "
                "live build") from exc
        if canonical_json(restored) != canonical_json(checkpoint):
            raise LiveDeploymentPreflightError(
                "restored live model state differs from paper checkpoint")
        guard = FoundationExecutionGuard(manifest, self.store,
                                         self.context.broker)
        bounded_data = _BoundedLiveData(data_fn, manifest, self.clock)

        self.store.activate(manifest.manifest_hash, actor)
        verified = VerifiedFoundationDeployment(
            manifest=manifest,
            context_identity=self.context.identity,
            desk=desk,
            portfolio=portfolio,
            data_fn=bounded_data,
            execution_guard=guard,
            checked_at=self.clock().astimezone(timezone.utc).isoformat(),
            _token=_CAPABILITY_TOKEN,
        )
        try:
            self.context.configure_session(
                portfolio=portfolio,
                data_fn=bounded_data,
                desk=desk,
                interval_minutes=manifest.interval_minutes,
                verified_deployment=verified,
            )
        except Exception:
            self.store.pause(
                manifest.manifest_hash, actor,
                "session configuration failed after preflight")
            raise
        return verified

    def _validate_capability(self, verified: VerifiedFoundationDeployment
                             ) -> DeploymentManifest:
        if not isinstance(verified, VerifiedFoundationDeployment) \
                or verified._token is not _CAPABILITY_TOKEN:
            raise LiveDeploymentPreflightError(
                "a verified deployment capability is required")
        if verified.context_identity != self.context.identity:
            raise LiveDeploymentPreflightError(
                "verified deployment belongs to another context")
        if getattr(self.context, "verified_deployment", None) is not verified:
            raise LiveDeploymentPreflightError(
                "live context is not bound to this verified deployment")
        return verified.manifest

    def start(self, verified: VerifiedFoundationDeployment, *, actor: str,
              first_cycle_timeout_seconds: float =
              FIRST_CYCLE_TIMEOUT_SECONDS) -> bool:
        manifest = self._validate_capability(verified)
        if (isinstance(first_cycle_timeout_seconds, bool)
                or not isinstance(first_cycle_timeout_seconds, (int, float))
                or not math.isfinite(float(first_cycle_timeout_seconds))
                or float(first_cycle_timeout_seconds) <= 0):
            raise ValueError(
                "first_cycle_timeout_seconds must be positive and finite")
        if self.store.get(manifest.manifest_hash)["state"] \
                != DeploymentState.ACTIVATED:
            raise DeploymentStateError("deployment is not activated")
        try:
            market_open = MarketHours().is_market_open(self.clock())
        except (TypeError, ValueError) as exc:
            raise LiveDeploymentPreflightError(
                "cannot prove that the NYSE regular session is open") from exc
        if not market_open:
            raise LiveDeploymentPreflightError(
                "controlled deployment may start only during NYSE market hours")
        _research, _paper, checkpoint = self._basic_preflight(
            manifest, require_unconfigured=False)
        if canonical_json(verified.desk.model_checkpoint_state()) \
                != canonical_json(checkpoint):
            raise LiveDeploymentPreflightError(
                "live model state changed between preparation and arming")
        self._require_no_working_risk()
        if self.context.run_reconciliation(cash_tolerance=0.01).get("ok") is not True:
            raise LiveDeploymentPreflightError("arming reconciliation failed")
        scheduler = self.context.scheduler
        if scheduler is None or scheduler.status().get("running"):
            raise LiveDeploymentPreflightError(
                "scheduler is missing or already running")
        waiter = getattr(scheduler, "wait_for_first_cycle", None)
        releaser = getattr(scheduler, "release_after_first_cycle", None)
        if (not callable(waiter) or not callable(releaser)
                or getattr(scheduler, "hold_after_first_cycle", None) is not True
                or getattr(scheduler, "max_consecutive_errors", None) != 1):
            raise LiveDeploymentPreflightError(
                "scheduler lacks the one-error first-cycle activation policy")
        self.store.arm(manifest.manifest_hash, actor)
        try:
            self.context.kill_switch.disengage(actor)
            started = scheduler.start()
            if not started:
                raise LiveDeploymentPreflightError("scheduler refused to start")
            first_result = waiter(float(first_cycle_timeout_seconds))
            if not isinstance(first_result, Mapping) \
                    or first_result.get("status") != "ok":
                status = (first_result.get("status")
                          if isinstance(first_result, Mapping) else None)
                reason = (first_result.get("reason")
                          if isinstance(first_result, Mapping) else None)
                raise LiveDeploymentPreflightError(
                    "first live cycle was not acceptable "
                    f"(status={status!r}, reason={reason!r})")
            unsafe_reports = [
                report for key in ("reports", "structure_reports")
                for report in (first_result.get(key) or [])
                if not isinstance(report, Mapping)
                or report.get("status") in {"error", "killed"}
            ]
            if unsafe_reports:
                raise LiveDeploymentPreflightError(
                    "first live cycle returned an unsafe execution report")
            if self.context.kill_switch.engaged():
                raise LiveDeploymentPreflightError(
                    "kill switch engaged during the first live cycle")
            scheduler_status = scheduler.status()
            if (scheduler_status.get("running") is not True
                    or scheduler_status.get("paused_reason") is not None):
                raise LiveDeploymentPreflightError(
                    "scheduler did not remain healthy after its first cycle")
            # The scheduler is now held between cycles.  Re-engage the kill
            # switch while we prove that the first cycle left no working risk
            # or broker/system-book drift; only then grant RUNNING.
            self.context.kill_switch.engage(
                "first live cycle complete; validating activation", actor)
            self._require_no_working_risk()
            post_cycle_reconciliation = self.context.run_reconciliation(
                cash_tolerance=0.01)
            if post_cycle_reconciliation.get("ok") is not True:
                raise LiveDeploymentPreflightError(
                    "first-cycle reconciliation failed")
            self.store.mark_running(manifest.manifest_hash, actor)
            self.context.kill_switch.disengage(actor)
            if releaser() is not True:
                raise LiveDeploymentPreflightError(
                    "scheduler refused to release its first-cycle hold")
            return True
        except BaseException as exc:
            # A cancellation/KeyboardInterrupt is no safer than an ordinary
            # exception here: once ARMED has been entered, every escape path
            # must attempt all three rollback rails before propagating.
            cleanup_failures = []
            try:
                self.context.kill_switch.engage(
                    f"deployment first-cycle activation failed: "
                    f"{type(exc).__name__}: {exc}", actor)
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_failures.append(
                    f"kill switch: {type(cleanup_exc).__name__}: {cleanup_exc}")
            try:
                scheduler.stop()
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_failures.append(
                    f"scheduler stop: {type(cleanup_exc).__name__}: "
                    f"{cleanup_exc}")
            try:
                state = self.store.get(manifest.manifest_hash)["state"]
                if state in {DeploymentState.ACTIVATED, DeploymentState.ARMED,
                             DeploymentState.RUNNING}:
                    self.store.pause(
                        manifest.manifest_hash, actor,
                        "deployment first-cycle activation failed")
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_failures.append(
                    f"manifest pause: {type(cleanup_exc).__name__}: "
                    f"{cleanup_exc}")
            if cleanup_failures:
                raise LiveDeploymentPreflightError(
                    "activation rollback was incomplete ("
                    + "; ".join(cleanup_failures) + ")") from exc
            raise

    def pause(self, verified: VerifiedFoundationDeployment, *, actor: str,
              reason: str) -> dict[str, Any]:
        manifest = self._validate_capability(verified)
        self.context.kill_switch.engage(reason, actor)
        if self.context.scheduler is not None:
            self.context.scheduler.stop()
        try:
            reconciliation = self.context.run_reconciliation(
                cash_tolerance=0.01)
        except Exception as exc:  # state still must become visibly paused
            reconciliation = {"ok": False, "error": str(exc),
                              "error_type": type(exc).__name__}
        record = self.store.pause(manifest.manifest_hash, actor, reason)
        return {"state": record["state"].value,
                "reconciliation": reconciliation}

    def revoke(self, verified: VerifiedFoundationDeployment, *, actor: str,
               reason: str) -> dict[str, Any]:
        manifest = self._validate_capability(verified)
        self.context.kill_switch.engage(reason, actor)
        if self.context.scheduler is not None:
            self.context.scheduler.stop()
        try:
            reconciliation = self.context.run_reconciliation(
                cash_tolerance=0.01)
        except Exception as exc:
            reconciliation = {"ok": False, "error": str(exc),
                              "error_type": type(exc).__name__}
        record = self.store.revoke(manifest.manifest_hash, actor, reason)
        return {"state": record["state"].value,
                "reconciliation": reconciliation}


__all__ = [
    "FoundationExecutionGuard",
    "FoundationLiveController",
    "FIRST_CYCLE_TIMEOUT_SECONDS",
    "LiveDeploymentPreflightError",
    "VerifiedFoundationDeployment",
]
