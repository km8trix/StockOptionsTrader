"""
Desk framework — the firm-persona abstraction every trading desk implements.

A Desk owns a strategy stack, a capital allocation (fraction of the whole
portfolio it may deploy), a RiskManager wired into its order flow, and a
running log of trader's notes explaining every decision with real numbers.
Desks emit DeskIntent objects; the BacktestEngine (desk mode) fills approved
intents at the next bar's open exactly like strategy-mode signals.

Phases 6-8 (Renaissance / Citadel / Jane Street desks) subclass Desk and
plug their models into desks.walk_forward.WalkForwardController so that
every model fit/predict honors the no-future-leakage invariant by
construction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date as date_type, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import Asset, AssetType
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Allowed trader-note categories.
NOTE_CATEGORIES = ('signal', 'risk', 'allocation', 'model', 'info')


def json_safe(value):
    """Coerce a value (recursively) into JSON-serializable builtins.

    numpy scalars/arrays become Python ints/floats/bools/lists; datetimes
    and dates become ISO strings; dicts/lists/tuples are walked recursively.
    Anything already JSON-native passes through unchanged.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (datetime, date_type)):
        # pd.Timestamp is a datetime subclass, so it is covered here too.
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


@dataclass
class TraderNote:
    """One timestamped, categorized desk decision with its supporting data.

    timestamp carries the SIMULATION date (the engine drives Desk.set_clock
    each day), never the wall clock, so notes line up with the bars that
    produced them. data values are coerced to JSON-serializable builtins at
    construction time.
    """
    timestamp: datetime
    desk: str
    category: str
    message: str
    data: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.category not in NOTE_CATEGORIES:
            raise ValueError(
                f"Invalid note category '{self.category}'; "
                f"must be one of {NOTE_CATEGORIES}")
        self.data = {str(k): json_safe(v) for k, v in self.data.items()}

    def to_dict(self) -> Dict:
        """Serialize to the report shape (contract C3)."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'desk': self.desk,
            'category': self.category,
            'message': self.message,
            'data': self.data,
        }


#: Allowed desk-intent actions (contract C4). SHORT/COVER are desk-mode
#: only — strategy mode never produces them.
INTENT_ACTIONS = ('BUY', 'SELL', 'SHORT', 'COVER')

#: Actions that OPEN exposure (gated by the daily-loss circuit and the
#: position-size check); SELL/COVER close exposure and always pass.
OPENING_ACTIONS = ('BUY', 'SHORT')


@dataclass
class DeskIntent:
    """A desk's wish to trade: BUY/SELL/SHORT/COVER an asset, sized.

    size_fraction is a fraction of the DESK'S capital (the engine converts
    it to dollars as portfolio_value * desk.capital_allocation *
    size_fraction at fill time). SELL intents always close the full long
    position and COVER intents always close the full short position,
    regardless of size_fraction.

    Flipping in one intent is forbidden (contract C4): a SHORT against an
    open long — or a BUY against an open short — is blocked at apply_risk
    with a 'risk' note; the desk must close first and open on a later day.

    quantity (OPTIONAL, Phase 8, additive): an ABSOLUTE size — contracts
    for options, shares for stock. When set, it OVERRIDES the
    size_fraction dollar sizing at fill time (multi-leg option structures
    need every leg filled in exact contract counts, not value-rounded
    shares). size_fraction is still required and is what the shared risk
    checks (position-size limit) evaluate, so set it to the intent's
    approximate capital fraction.

    desk_keys (OPTIONAL, fund mode only): the key(s) of the desk(s) whose
    opening view this intent carries, set by the FundOrchestrator's
    netting. The engine stamps them onto the Position it opens (see
    core.models.Position.owners) so each desk's book logic only touches
    positions it owns. Never set outside fund mode.

    intent_id (OPTIONAL, target mode only): a deterministic logical execution
    identity produced by the target-position delta builder.  Patient-order
    replacements derive distinct broker client IDs from this stable parent,
    so a retry or restart cannot silently turn one target delta into duplicate
    exposure.  Legacy intent producers leave it unset.
    """
    asset: Asset
    action: str
    size_fraction: float
    reason: str
    quantity: Optional[int] = None
    desk_keys: Optional[tuple] = None
    intent_id: Optional[str] = None

    def __post_init__(self):
        if self.action not in INTENT_ACTIONS:
            raise ValueError(
                f"Invalid intent action '{self.action}'; "
                f"must be one of {INTENT_ACTIONS}")
        if not (0.0 < self.size_fraction <= 1.0):
            raise ValueError(
                f"size_fraction {self.size_fraction} must be in (0, 1]")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError(
                f"quantity {self.quantity} must be a positive integer "
                f"(direction comes from the action, never the sign)")
        if self.intent_id is not None:
            if not isinstance(self.intent_id, str) or not self.intent_id.strip():
                raise ValueError(
                    "intent_id must be a non-empty string when provided")
            self.intent_id = self.intent_id.strip()


class Desk(ABC):
    """Abstract base class for firm-persona trading desks.

    Subclasses implement generate_intents(); the shared apply_risk() wires
    the desk's RiskManager into its order flow (position sizing, stop
    losses, daily-loss circuit breaker) so no desk can bypass risk checks.
    """

    def __init__(self, key: str, name: str, description: str, accent: str,
                 capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None):
        self.key = key
        self.name = name
        self.description = description
        self.accent = accent
        self.capital_allocation = capital_allocation
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()
        self.notes: List[TraderNote] = []
        # Simulation clock; the engine calls set_clock(date) each simulated
        # day so notes carry simulation dates, not the wall clock.
        self._clock: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Simulation clock + notes
    # ------------------------------------------------------------------
    def set_clock(self, date) -> None:
        """Set the simulation date used to stamp subsequent notes."""
        self._clock = pd.Timestamp(date)

    def note(self, category: str, message: str, **data) -> TraderNote:
        """Record a TraderNote stamped with the simulation clock.

        Falls back to the wall clock only if the engine never set a clock
        (e.g. ad-hoc use outside a simulation).
        """
        timestamp = self._clock if self._clock is not None else datetime.now()
        trader_note = TraderNote(timestamp=timestamp, desk=self.name,
                                 category=category, message=message, data=data)
        self.notes.append(trader_note)
        logger.debug("[%s] %s note: %s", self.name, category, message)
        return trader_note

    def risk_note_data(self, asset: Optional[Asset]) -> Dict:
        """Extra data merged into the shared apply_risk notes.

        Desks whose note contracts tag EVERY note (Jane Street's C14
        data.book) override this to map the blocked/stopped asset —
        ``None`` for desk-level notes like the daily-loss circuit — to
        the owning tag(s). The keys returned must not collide with the
        note's own kwargs. The base returns {} so the apply_risk notes
        of desks without such contracts stay byte-identical.
        """
        return {}

    def _owns_position(self, position) -> bool:
        """Fund-mode ownership scoping (core.models.Position.owners).

        False when the position is owner-tagged and this desk's key is
        NOT among the owners — it belongs to another desk in the fund
        and must be invisible to this desk's sweep/exit logic (never
        closed by it). An UNTAGGED position (owners is None — always
        the case outside fund mode) is owned: single-desk behavior is
        byte-identical. Entry-blocking held-checks deliberately stay
        unscoped: entering a symbol another desk holds would co-mingle
        the books into one position.
        """
        return position.owners is None or self.key in position.owners

    # ------------------------------------------------------------------
    # Strategy surface
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """Produce the desk's trade intents for the current simulation day.

        all_data values are indicator-enriched frames sliced through the
        current simulation date — the engine guarantees no row beyond
        `date` is ever present.
        """

    def apply_risk(self, intents: List[DeskIntent],
                   portfolio: PortfolioManager,
                   all_data: Dict[str, pd.DataFrame],
                   date) -> List[DeskIntent]:
        """Run the desk's RiskManager over its intents (shared, concrete).

        1. Daily-loss circuit: if today's realized drawdown (current value
           vs the previous snapshot) breaches check_daily_loss_limit, ALL
           new BUYs AND SHORTs are blocked for the day (noted once).
           Closing intents (SELL/COVER) always pass.
        2. Stop losses, both-sided, STOCK POSITIONS ONLY: a long closes
           (full-size SELL) when should_close_position fires — price <=
           entry * (1 - stop); a short closes (full-size COVER) when
           price >= entry * (1 + stop). Both are noted with
           entry/current/stop. OPTION positions are EXEMPT (Phase 8):
           per-leg price stops on defined-risk structures are
           nonsensical — a long hedge wing decaying with theta while the
           structure WINS would trigger a SELL that strips the hedge,
           and short legs routinely move +/-5% daily. Options risk is
           managed at the STRUCTURE level (profit target / stop-loss on
           cost-to-close / time exit / regime flatten) plus the engine's
           expiry-settlement backstop. Stock behavior is unchanged.
        3. Position sizing: a BUY or SHORT whose absolute dollar value
           (portfolio_value * capital_allocation * size_fraction) violates
           check_position_size is blocked (noted with the numbers).
        4. No one-step flips (contract C4): a BUY against an open short or
           a SHORT against an open long is blocked with a 'risk' note.

        SELL and COVER intents always pass.

        MARGIN IS NOT MODELED: shorts use a cash-account approximation —
        short-sale proceeds are held as cash and no margin requirement or
        borrow cost is simulated. Live shorting is gated to Phase 9.
        """
        approved: List[DeskIntent] = []
        portfolio_value = portfolio.get_portfolio_value()

        # --- Daily-loss circuit breaker -------------------------------
        daily_pnl = 0.0
        if portfolio.portfolio_history:
            daily_pnl = (portfolio_value
                         - portfolio.portfolio_history[-1]['portfolio_value'])
        buys_allowed = self.risk_manager.check_daily_loss_limit(
            daily_pnl, portfolio_value)
        if not buys_allowed:
            self.note(
                'risk',
                f"Daily-loss circuit breaker: P&L {daily_pnl:,.2f} on "
                f"portfolio {portfolio_value:,.2f} breaches the "
                f"{self.risk_manager.max_daily_loss:.1%} limit; all new "
                f"BUYs blocked today",
                daily_pnl=daily_pnl, portfolio_value=portfolio_value,
                max_daily_loss=self.risk_manager.max_daily_loss,
                **self.risk_note_data(None))

        # --- Stop-loss closes for open positions (both-sided) ----------
        already_closing = {intent.asset for intent in intents
                           if intent.action in ('SELL', 'COVER')}
        for asset, position in list(portfolio.positions.items()):
            if asset in already_closing:
                continue
            if asset.asset_type is not AssetType.STOCK:
                # Option legs are exempt from per-leg price stops (see
                # the docstring): structure-level exits own that risk.
                continue
            if position.quantity > 0:
                if self.risk_manager.should_close_position(position):
                    stop_price = self.risk_manager.calculate_position_stop_loss(
                        position.avg_entry_price)
                    approved.append(DeskIntent(
                        asset=asset, action='SELL', size_fraction=1.0,
                        reason=(f"stop-loss: {position.current_price:.4f} <= "
                                f"stop {stop_price:.4f}")))
                    already_closing.add(asset)
                    self.note(
                        'risk',
                        f"Stop-loss SELL {asset.symbol}: entry "
                        f"{position.avg_entry_price:.4f}, current "
                        f"{position.current_price:.4f} breached stop "
                        f"{stop_price:.4f} "
                        f"({self.risk_manager.position_stop_loss:.1%} below entry)",
                        entry_price=position.avg_entry_price,
                        current_price=position.current_price,
                        stop_price=stop_price,
                        stop_loss_pct=self.risk_manager.position_stop_loss,
                        **self.risk_note_data(asset))
            elif position.quantity < 0:
                # Short stop: adverse move is the price RISING above
                # entry * (1 + stop). Mirror of the long-side semantics.
                entry_price = position.avg_entry_price
                if entry_price is None or entry_price <= 0:
                    continue
                stop_price = entry_price * (
                    1 + self.risk_manager.position_stop_loss)
                if position.current_price >= stop_price:
                    approved.append(DeskIntent(
                        asset=asset, action='COVER', size_fraction=1.0,
                        reason=(f"stop-loss: {position.current_price:.4f} >= "
                                f"stop {stop_price:.4f}")))
                    already_closing.add(asset)
                    self.note(
                        'risk',
                        f"Stop-loss COVER {asset.symbol}: entry "
                        f"{entry_price:.4f}, current "
                        f"{position.current_price:.4f} breached stop "
                        f"{stop_price:.4f} "
                        f"({self.risk_manager.position_stop_loss:.1%} above entry)",
                        entry_price=entry_price,
                        current_price=position.current_price,
                        stop_price=stop_price,
                        stop_loss_pct=self.risk_manager.position_stop_loss,
                        **self.risk_note_data(asset))

        # --- Per-intent checks -----------------------------------------
        for intent in intents:
            if intent.action in ('SELL', 'COVER'):
                approved.append(intent)
                continue

            # Index levels ('^VIX', '^GSPC', ...) are auxiliary signal
            # series, not tradeable instruments. A universe may include one
            # (the VIX desk needs ^VIX injected), and every desk sees every
            # all_data key — this shared guard keeps any desk from opening a
            # position on the level itself. Closes still pass above.
            if intent.asset.symbol.startswith('^'):
                self.note(
                    'risk',
                    f"Blocked {intent.action} {intent.asset.symbol}: index "
                    f"levels are auxiliary data, not tradeable instruments",
                    symbol=intent.asset.symbol, action=intent.action,
                    **self.risk_note_data(intent.asset))
                continue

            if not buys_allowed:
                # Already noted once above; the circuit blocks all new
                # exposure (BUYs and SHORTs alike).
                continue

            # No one-step flips (contract C4): close first, open later.
            current_position = portfolio.get_position(intent.asset)
            if intent.action == 'SHORT' and current_position is not None \
                    and current_position.quantity > 0:
                self.note(
                    'risk',
                    f"Blocked SHORT {intent.asset.symbol}: open long of "
                    f"{current_position.quantity} shares — flipping in one intent "
                    f"is forbidden (SELL first)",
                    symbol=intent.asset.symbol,
                    position_quantity=current_position.quantity,
                    action=intent.action,
                    **self.risk_note_data(intent.asset))
                continue
            if intent.action == 'BUY' and current_position is not None \
                    and current_position.quantity < 0:
                self.note(
                    'risk',
                    f"Blocked BUY {intent.asset.symbol}: open short of "
                    f"{abs(current_position.quantity)} shares — flipping in one "
                    f"intent is forbidden (COVER first)",
                    symbol=intent.asset.symbol,
                    position_quantity=current_position.quantity,
                    action=intent.action,
                    **self.risk_note_data(intent.asset))
                continue

            trade_value = abs(portfolio_value * self.capital_allocation
                              * intent.size_fraction)
            # An absolute STOCK quantity must not bypass the fractional risk
            # budget.  Use the latest causally available close to evaluate the
            # real requested notional; option quantities are structure-sized
            # by their desk's defined-risk/max-loss model and remain on that
            # risk-budget convention.
            if (intent.quantity is not None
                    and intent.asset.asset_type is AssetType.STOCK):
                frame = all_data.get(intent.asset.symbol)
                close = None
                if frame is not None and not frame.empty and 'close' in frame:
                    candidate = float(frame['close'].iloc[-1])
                    if np.isfinite(candidate) and candidate > 0:
                        close = candidate
                if close is None:
                    self.note(
                        'risk',
                        f"Blocked {intent.action} {intent.asset.symbol}: "
                        "absolute quantity cannot be valued from current data",
                        symbol=intent.asset.symbol, action=intent.action,
                        quantity=intent.quantity,
                        **self.risk_note_data(intent.asset))
                    continue
                requested_notional = abs(float(intent.quantity)) * close
                if requested_notional > trade_value + 1e-9:
                    self.note(
                        'risk',
                        f"Blocked {intent.action} {intent.asset.symbol}: "
                        f"absolute quantity requests {requested_notional:,.2f} "
                        f"against a {trade_value:,.2f} risk budget",
                        symbol=intent.asset.symbol, action=intent.action,
                        quantity=intent.quantity,
                        requested_notional=requested_notional,
                        trade_value=trade_value,
                        **self.risk_note_data(intent.asset))
                    continue
                trade_value = requested_notional
            if not self.risk_manager.check_position_size(portfolio_value,
                                                         trade_value):
                self.note(
                    'risk',
                    f"Blocked {intent.action} {intent.asset.symbol}: trade "
                    f"value {trade_value:,.2f} is "
                    f"{trade_value / portfolio_value:.1%} of portfolio "
                    f"{portfolio_value:,.2f}, exceeding the "
                    f"{self.risk_manager.max_position_size:.1%} "
                    f"position-size limit",
                    symbol=intent.asset.symbol, trade_value=trade_value,
                    portfolio_value=portfolio_value,
                    size_fraction=intent.size_fraction,
                    capital_allocation=self.capital_allocation,
                    max_position_size=self.risk_manager.max_position_size,
                    **self.risk_note_data(intent.asset))
                continue

            approved.append(intent)

        return approved

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        """Snapshot of the desk for dashboards and reports."""
        return {
            'key': self.key,
            'name': self.name,
            'capital_allocation': self.capital_allocation,
            'notes_count': len(self.notes),
            'last_note': self.notes[-1].to_dict() if self.notes else None,
        }

    @property
    def walk_forward_fits(self) -> List:
        """Walk-forward fit events (desks with controllers override this)."""
        return []
