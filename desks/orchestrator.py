"""
Fund orchestrator — run N desks against ONE shared account (the fund seam).

Every Desk defaults capital_allocation=1.0 and, run alone, deploys the whole
account; the backtest/live paths drive exactly one desk. FundOrchestrator is
the seam that lets several desks share a single PortfolioManager without
over-leveraging it, delivering the cross-desk book the platform was built for.

CAPITAL MODEL (raw, validated — NOT up-normalized): each desk keeps its own
capital_allocation and deploys exactly that fraction of the account; the
orchestrator validates the allocations sum to <= 1.0 and RAISES if they do
not (no silent rescale — an over-allocated fund is a configuration error, not
something to quietly paper over). Leftover allocation simply stays in cash. A
desk therefore behaves in the fund EXACTLY as it does alone, scaled by its
allocation — predictable, and no surprise capital efficiency.

ONE ACCOUNT-WIDE RISK GATE: each desk's generate_intents runs independently
(so a desk's OWN internal controls — e.g. Jane Street's short_vega risk-book
check inside generate_intents — still fire). The orchestrator then NETS the
intents and runs ONE unified apply_risk over the shared book via a private
account desk (capital_allocation=1.0): a single account-level daily-loss
circuit, one account-wide position-size ceiling, and stop-losses computed once
on the unified book. Per-desk apply_risk is deliberately NOT called on the
shared portfolio — a desk's apply_risk iterates portfolio.positions and would
otherwise stop-loss OTHER desks' positions with its own parameters.

INTENT NETTING (per asset):
  * STOCKS net across desks by ALGEBRAIC EXPOSURE. Every desk's opening view is
    honored — opposing desks BOTH trade and their account-absolute sizes OFFSET
    (BUY positive, SHORT negative). The fund carries the RESIDUAL net position
    (a unified book holds one position per symbol); equal-and-opposite views net
    flat. Same-direction opens simply sum. An open mixed with a close on the
    same name, or disagreeing closes (SELL vs COVER), stays ambiguous on a
    one-position book and is dropped with a 'risk' note; same-direction closes
    collapse to one full-size close.
  * OPTION legs BYPASS cross-desk netting entirely: each passes through
    independently. Netting or dropping one leg of a defined-risk structure
    would leave a naked wing. (Only one desk trades options today, so
    cross-desk option overlap does not arise in practice.)

SIZING: intents leave the orchestrator expressed as ACCOUNT-ABSOLUTE fractions
(capital_allocation * size_fraction), so the BacktestEngine fills them through
one path with no per-desk capital math. Option intents keep their absolute
contract `quantity` (Jane Street sizes structures in exact contracts).

POSITION OWNERSHIP (fund mode): every netted opening intent carries the
originating desk key(s) in DeskIntent.desk_keys (winning side only on an
opposing-open residual), and the engine stamps them onto the opened
Position (core.models.Position.owners). ENFORCEMENT: every desk family
scopes its sweep/exit logic to positions it owns (Desk._owns_position) —
CrossSectionalLongShortDesk's reconcile/sweep/retention, the Renaissance
and Citadel reconciles + orphan sweeps, Foundation's MACD/RSI exits, the
trend follower's regime/ATR exits, and Jane Street's relative-value
reconcile/exits — a position owned by another desk is invisible to all of
them (pinned by tests/test_legacy_desk_ownership.py). Entry-blocking
held-checks stay deliberately UNSCOPED (one position per symbol: entering
a symbol another desk holds would co-mingle books). Residual edge:
Renaissance/Citadel signal exits fire off tracked state, so a tracked leg
whose fill actually belongs to another desk can still be closed during the
reconcile grace (2 days) before its bookkeeping drops.

ADDITIVE: single-desk and strategy backtests are untouched (the engine only
enters the orchestrator path when an orchestrator is supplied). Reports gain an
additive 'orchestrator' block; single-desk reports stay byte-identical.

Live wiring (utils/live_session.py) lands in a later step; this module is the
backtest seam plus the reusable composition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent, TraderNote
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

if TYPE_CHECKING:  # annotation only
    from portfolio.risk_aggregator import PortfolioRiskAggregator
    from desks.rl_execution import RLExecutionThrottle

logger = logging.getLogger(__name__)

#: Float tolerance on the capital_allocation sum (<= 1.0) check.
CAPITAL_SUM_TOL = 1e-9

#: Below this |net account fraction|, algebraically-netted opposing opens are
#: treated as fully offset (the fund stays flat) — also avoids emitting a
#: DeskIntent with size_fraction 0 (which is invalid).
NET_EPS = 1e-9

_OPENING = ('BUY', 'SHORT')
_CLOSING = ('SELL', 'COVER')
_OPTION_TYPES = (AssetType.CALL, AssetType.PUT)


class _AccountDesk(Desk):
    """Private account-wide desk: reuses Desk.apply_risk over the unified
    book at capital_allocation=1.0. Never generates intents of its own."""

    def __init__(self, risk_manager: Optional[RiskManager] = None):
        super().__init__(
            key='fund', name='Fund Orchestrator',
            description="Account-wide risk overlay across the fund's desks.",
            accent='#6e7681', capital_allocation=1.0,
            risk_manager=risk_manager)

    def generate_intents(self, all_data, date,
                         portfolio) -> List[DeskIntent]:  # pragma: no cover
        return []


@dataclass
class _Tagged:
    """A desk's intent plus its account-absolute size (capital_allocation *
    size_fraction) and the originating desk, used during netting."""
    intent: DeskIntent
    desk: Desk
    account_fraction: float


class FundOrchestrator:
    """Composite that runs several desks against one shared account.

    See the module docstring for the capital model, the single account-wide
    risk gate, and the netting rules. Exposes the read surface the
    BacktestEngine reads off a desk (key/name/notes/walk_forward_fits/
    price_option) so the engine's desk-report path works unchanged.
    """

    def __init__(self, desks: List[Desk],
                 risk_manager: Optional[RiskManager] = None,
                 risk_aggregator: Optional['PortfolioRiskAggregator'] = None,
                 sizing_modulator: Optional['RLExecutionThrottle'] = None):
        if not desks:
            raise ValueError("FundOrchestrator requires at least one desk")
        keys = [desk.key for desk in desks]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Desk keys must be unique; got {keys}")
        total = sum(desk.capital_allocation for desk in desks)
        if total > 1.0 + CAPITAL_SUM_TOL:
            raise ValueError(
                f"Desk capital_allocation sums to {total:.4f}; must be <= 1.0 "
                f"(no over-leverage). Allocations are used as-is — lower them.")

        self.desks = list(desks)
        self.active_capital = total
        self.key = 'fund'
        self.name = 'Fund Orchestrator'
        #: The orchestrator owns the whole account (1.0); the BacktestEngine
        #: reads this when it sizes fund intents (which are already
        #: account-absolute, so 1.0 * size_fraction == the intended size).
        self.capital_allocation = 1.0

        self._account = _AccountDesk(risk_manager=risk_manager)
        self.risk_manager = self._account.risk_manager
        # Optional account-level overlay (gross leverage / correlation /
        # sector) run AFTER the unified apply_risk; None -> not enforced.
        self.risk_aggregator = risk_aggregator
        # OPTIONAL, OFF-BY-DEFAULT, GATED RL execution throttle (Phase F unit 2).
        # When None (the default) step() is byte-identical to before — the
        # modulator call is simply skipped. The throttle is STRICTLY
        # SUBTRACTIVE: it can only ever shrink opening-stock sizes, never grow
        # them, so the fund gross is always <= baseline. See desks.rl_execution.
        self.sizing_modulator = sizing_modulator
        #: Conflict notes the orchestrator itself raises (kept separate from
        #: the account desk's risk notes and the sub-desks' own notes).
        self._own_notes: List[TraderNote] = []
        self.conflicts_resolved = 0
        #: Desks able to price options, for held-option marking/settlement and
        #: pending option fills (only Jane Street today).
        self._option_desks = [desk for desk in desks
                              if hasattr(desk, 'price_option')]
        self._clock: Optional[pd.Timestamp] = None

        # A zero/negative-allocation desk deploys nothing and is skipped
        # every step; record it ONCE so the no-op is auditable in the notes
        # stream rather than silently invisible.
        for desk in self.desks:
            if desk.capital_allocation <= 0:
                self._note('info',
                           f"Desk '{desk.key}' has capital_allocation "
                           f"{desk.capital_allocation:g} (<= 0): it deploys "
                           f"nothing new (only closing intents pass)",
                           desk_key=desk.key,
                           capital_allocation=desk.capital_allocation)

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------
    def set_clock(self, date) -> None:
        """Set the simulation clock on the orchestrator AND every sub-desk
        and the account desk, so all notes carry the simulation date."""
        self._clock = pd.Timestamp(date)
        self._account.set_clock(date)
        for desk in self.desks:
            desk.set_clock(date)

    def _note(self, category: str, message: str, **data) -> TraderNote:
        timestamp = (self._clock if self._clock is not None
                     else pd.Timestamp.now())
        note = TraderNote(timestamp=timestamp, desk=self.name,
                          category=category, message=message, data=data)
        self._own_notes.append(note)
        return note

    # ------------------------------------------------------------------
    # One simulated step
    # ------------------------------------------------------------------
    def step(self, all_data: Dict[str, pd.DataFrame], date,
             portfolio: PortfolioManager) -> List[DeskIntent]:
        """Fan out to each desk, net the intents, run one account-wide
        apply_risk, and return the approved intents (account-absolute
        size_fraction; option `quantity` preserved)."""
        tagged: List[_Tagged] = []
        for desk in self.desks:
            zero_weight = desk.capital_allocation <= 0
            intents = desk.generate_intents(all_data, date, portfolio)
            for intent in intents:
                # A zero-allocation desk deploys nothing NEW, but must still
                # be able to exit positions it opened while it had capital —
                # DynamicReweighter's mean_variance/max_diversification modes
                # clip weights to exactly 0.0 mid-run, and dropping the desk
                # entirely orphans its open book (closes are full-size, so
                # account_fraction is irrelevant for them).
                if zero_weight and intent.action not in _CLOSING:
                    continue
                account_fraction = (desk.capital_allocation
                                    * intent.size_fraction)
                tagged.append(_Tagged(intent, desk, account_fraction))

        netted = self._net_intents(tagged)
        # OPTIONAL gated RL throttle (Phase F unit 2): runs BEFORE apply_risk so
        # the risk gate only ever sees the final, SMALLER sizes — strictly
        # safer. Off by default (None) => byte-identical to before. At this
        # point intents are ACCOUNT-ABSOLUTE size_fraction.
        if self.sizing_modulator is not None:
            netted = self.sizing_modulator.apply(netted, portfolio, all_data,
                                                 date)
        self._account.set_clock(date)
        approved = self._account.apply_risk(netted, portfolio, all_data, date)

        # Account-level overlay (gross leverage / correlation / sector) on the
        # netted, risk-approved batch. Blocked opens are dropped + noted; this
        # catches book-wide breaches the per-name apply_risk cannot (e.g. many
        # in-limit names summing past the gross cap).
        if self.risk_aggregator is not None:
            approved, blocked = self.risk_aggregator.check(
                approved, portfolio, all_data, date)
            for intent, reason in blocked:
                self._note(
                    'risk',
                    f"Account risk aggregator blocked {intent.action} "
                    f"{intent.asset.symbol}: {reason}",
                    symbol=intent.asset.symbol, action=intent.action,
                    reason=reason)
        return approved

    # ------------------------------------------------------------------
    # Netting
    # ------------------------------------------------------------------
    def _net_intents(self, tagged: List[_Tagged]) -> List[DeskIntent]:
        """Options bypass netting (one intent per leg); stocks net per asset."""
        netted: List[DeskIntent] = []
        by_asset: Dict[Asset, List[_Tagged]] = {}
        for item in tagged:
            if item.intent.asset.asset_type in _OPTION_TYPES:
                netted.append(self._account_scaled(item))
            else:
                by_asset.setdefault(item.intent.asset, []).append(item)
        for asset, group in by_asset.items():
            netted.extend(self._net_stock_asset(asset, group))
        return netted

    def _account_scaled(self, item: _Tagged) -> DeskIntent:
        """Re-express an intent's size_fraction as its account-absolute
        fraction (preserving action, asset and the absolute `quantity`)."""
        return DeskIntent(
            asset=item.intent.asset, action=item.intent.action,
            size_fraction=min(1.0, item.account_fraction),
            reason=item.intent.reason, quantity=item.intent.quantity,
            desk_keys=(item.desk.key,))

    def _net_stock_asset(self, asset: Asset,
                         group: List[_Tagged]) -> List[DeskIntent]:
        opens = [t for t in group if t.intent.action in _OPENING]
        closes = [t for t in group if t.intent.action in _CLOSING]
        close_dirs = {t.intent.action for t in closes}

        # An open mixed with a close on the same name, or disagreeing closes
        # (SELL vs COVER), stays ambiguous on a unified one-position book —
        # drop and record it. (Opposing OPENS are NETTED below, not dropped:
        # both desks trade and their exposures offset.)
        if (opens and closes) or len(close_dirs) > 1:
            desk_actions = sorted(f"{t.desk.key}:{t.intent.action}"
                                  for t in group)
            self.conflicts_resolved += 1
            self._note(
                'risk',
                f"Fund conflict on {asset.symbol}: desks disagree "
                f"({', '.join(desk_actions)}); dropping all intents for it",
                symbol=asset.symbol, desk_actions=desk_actions)
            return []

        if opens:
            # Algebraic exposure netting: every desk's directional view is
            # honored — opposing desks BOTH trade and their account-absolute
            # sizes OFFSET (BUY positive, SHORT negative). The fund carries the
            # residual net position; equal-and-opposite views net flat.
            net = sum(t.account_fraction if t.intent.action == 'BUY'
                      else -t.account_fraction for t in opens)
            desk_keys = sorted({t.desk.key for t in opens})
            opposed = len({t.intent.action for t in opens}) > 1
            if opposed:
                self._note(
                    'allocation',
                    f"Fund netted opposing opens on {asset.symbol} "
                    f"({'+'.join(desk_keys)}): residual {net:+.4f} of account",
                    symbol=asset.symbol, residual=net, desk_keys=desk_keys)
            if abs(net) < NET_EPS:
                return []  # views fully offset -> the fund stays flat
            action = 'BUY' if net > 0 else 'SHORT'
            magnitude = min(1.0, abs(net))
            # Ownership: only the desks on the WINNING side of the net own
            # the residual position (a netted-away short view must not let
            # its desk manage the long that survived).
            owners = tuple(sorted({t.desk.key for t in opens
                                   if t.intent.action == action}))
            return [DeskIntent(
                asset=asset, action=action, size_fraction=magnitude,
                reason=(f"fund {action} {asset.symbol}: {'+'.join(desk_keys)} "
                        f"(net {magnitude:.4f} of account"
                        f"{', opposing views offset' if opposed else ''})"),
                desk_keys=owners)]

        # Only closes, all the same action: one full-size close.
        action = closes[0].intent.action
        desk_keys = sorted({t.desk.key for t in closes})
        return [DeskIntent(
            asset=asset, action=action, size_fraction=1.0,
            reason=f"fund {action} {asset.symbol}: {'+'.join(desk_keys)}")]

    # ------------------------------------------------------------------
    # Option pricing routing (held-option marking / settlement / fills)
    # ------------------------------------------------------------------
    def price_option(self, asset: Asset, underlying_frame: pd.DataFrame,
                     date, spot: float) -> Optional[float]:
        """Route option pricing to the first options-capable desk that
        returns a price (only Jane Street today). None if none can."""
        for desk in self._option_desks:
            price = desk.price_option(asset, underlying_frame, date, spot)
            if price is not None:
                return price
        return None

    # ------------------------------------------------------------------
    # Read surface the BacktestEngine reads off a desk
    # ------------------------------------------------------------------
    @property
    def notes(self) -> List[TraderNote]:
        """All notes: account-overlay + orchestrator conflicts + every
        sub-desk's own notes (each TraderNote carries its desk + timestamp)."""
        merged: List[TraderNote] = list(self._account.notes)
        merged.extend(self._own_notes)
        for desk in self.desks:
            merged.extend(desk.notes)
        return merged

    @property
    def walk_forward_fits(self) -> List:
        """Every sub-desk's walk-forward fit events, concatenated."""
        return [fit for desk in self.desks for fit in desk.walk_forward_fits]

    def get_status(self) -> Dict:
        return {
            'key': self.key,
            'name': self.name,
            'active_capital': self.active_capital,
            'conflicts_resolved': self.conflicts_resolved,
            'desks': [{'key': desk.key, 'name': desk.name,
                       'capital_allocation': desk.capital_allocation}
                      for desk in self.desks],
        }
