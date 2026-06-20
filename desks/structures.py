"""
Defined-risk multi-leg options structures — builders, sizing, lifecycle.

STRIKE PLACEMENT (all builders): short strikes sit 1.0 expected-move
standard deviations from spot, long (wing) strikes 1.5 — where one
standard deviation of the expected move to expiry is
spot * sigma_annual * sqrt(dte_days / 365). Strikes are rounded to
sensible increments: $1 when spot < 100, else $5. When rounding
collapses a wing onto its short strike (tiny sigma / short DTE), the
wing is pushed one increment further out — a zero-width wing has no
defined risk and an undefined max loss.

MAX-LOSS MATH (per contract, the number sizing lives on):

    max_loss_per_contract = wing_width * 100 - net_credit_per_contract

where wing_width is the short-to-long strike distance of the spread —
for iron condors the WIDER wing side governs (the worst case is the
underlying blowing through one side only; the other side's legs expire
worthless and contribute nothing to the loss). Structure-level
max_loss = max_loss_per_contract * contracts.

SIZING RULE:

    contracts = floor(risk_budget / max_loss_per_contract)
    risk_budget = 0.02 * desk_capital * book_weight   (the caller's job)

0 contracts -> the caller must skip the trade (with an 'info' note).

EXIT CHECKS (boundary semantics are INCLUSIVE — equality triggers),
evaluated in this priority order:
    profit_target  cost-to-close <= 50% of entry credit
    stop_loss      cost-to-close >= 2x entry credit
    time_exit      DTE <= 21 calendar days

cost-to-close is the structure-level dollar cost of unwinding every leg
at the supplied marks: pay for legs that are short, receive for legs
that are long, x100 x contracts.

PRICING REALITY: leg marks come from the synthetic Black-Scholes engine
(desks/options_pricing) — see that module's banner before trusting any
backtest number.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import pandas as pd

from core.models import Asset
from desks.base import DeskIntent

logger = logging.getLogger(__name__)

#: C11 structure types.
STRUCTURE_TYPES = ('iron_condor', 'put_credit_spread', 'call_credit_spread')

#: Short strike distance in expected-move standard deviations.
SHORT_STRIKE_STD = 1.0
#: Long (wing) strike distance in expected-move standard deviations.
LONG_STRIKE_STD = 1.5


def strike_increment(spot: float) -> float:
    """$1 strikes under $100 spot, $5 at or above (module docstring)."""
    return 1.0 if spot < 100.0 else 5.0


def _round_strike(value: float, increment: float) -> float:
    return round(value / increment) * increment


def _spread_strikes(spot: float, sigma_annual: float, dte_days: int,
                    direction: int) -> Dict[str, float]:
    """Short/long strikes one side of spot (direction -1 puts, +1 calls).

    Guards: strikes are floored at one increment and a wing that rounds
    onto its short strike is pushed one increment further out.
    """
    std_move = spot * sigma_annual * math.sqrt(dte_days / 365.0)
    increment = strike_increment(spot)
    short = _round_strike(spot + direction * SHORT_STRIKE_STD * std_move,
                          increment)
    long = _round_strike(spot + direction * LONG_STRIKE_STD * std_move,
                         increment)
    short = max(short, increment)
    long = max(long, increment)
    if long == short:
        long = short + direction * increment
        long = max(long, increment)
        if long == short:  # floored put side: push the SHORT up instead
            short = long + increment
    return {'short': float(short), 'long': float(long)}


def build_put_credit_spread(spot: float, sigma_annual: float,
                            dte_days: int) -> List[Dict]:
    """Leg specs for a short put vertical: SHORT the 1.0-std put, BUY the
    1.5-std put below it. Each leg: {'action', 'right', 'strike'}."""
    strikes = _spread_strikes(spot, sigma_annual, dte_days, direction=-1)
    return [
        {'action': 'SHORT', 'right': 'put', 'strike': strikes['short']},
        {'action': 'BUY', 'right': 'put', 'strike': strikes['long']},
    ]


def build_call_credit_spread(spot: float, sigma_annual: float,
                             dte_days: int) -> List[Dict]:
    """Leg specs for a short call vertical: SHORT the 1.0-std call, BUY
    the 1.5-std call above it."""
    strikes = _spread_strikes(spot, sigma_annual, dte_days, direction=+1)
    return [
        {'action': 'SHORT', 'right': 'call', 'strike': strikes['short']},
        {'action': 'BUY', 'right': 'call', 'strike': strikes['long']},
    ]


def build_iron_condor(spot: float, sigma_annual: float,
                      dte_days: int) -> List[Dict]:
    """Leg specs for a short iron condor: put credit spread below spot
    mirrored by a call credit spread above (module docstring)."""
    return (build_put_credit_spread(spot, sigma_annual, dte_days)
            + build_call_credit_spread(spot, sigma_annual, dte_days))


def wing_width(legs: List[Dict]) -> float:
    """Governing wing width (module docstring: condors take the WIDER
    side). legs carry 'action'/'right'/'strike' (builder output) or the
    same keys on tracker legs."""
    widths = []
    for right in ('put', 'call'):
        short = [leg['strike'] for leg in legs
                 if leg['right'] == right and leg['action'] == 'SHORT']
        long = [leg['strike'] for leg in legs
                if leg['right'] == right and leg['action'] == 'BUY']
        if short and long:
            widths.append(abs(short[0] - long[0]))
    if not widths:
        raise ValueError("legs contain no short/long spread pair")
    return float(max(widths))


def max_loss_per_contract(legs: List[Dict],
                          credit_per_contract: float) -> float:
    """wing_width * 100 - net_credit_per_contract (module docstring)."""
    return wing_width(legs) * 100.0 - credit_per_contract


def size_contracts(risk_budget: float,
                   max_loss_per_contract_: float) -> int:
    """floor(risk_budget / max_loss_per_contract); 0 when the budget
    cannot absorb one contract's worst case (callers skip with a note)."""
    if max_loss_per_contract_ <= 0 or risk_budget <= 0:
        return 0
    return int(math.floor(risk_budget / max_loss_per_contract_))


class StructureTracker:
    """Lifecycle bookkeeping for open defined-risk structures.

    Internal status: 'open' -> 'closing' (close intents emitted, fills
    pending) -> 'closed' | 'expired'. C11 serialization maps 'closing'
    to 'open' (legs are still on). Deterministic ids JS-0001, JS-0002...
    """

    def __init__(self):
        self._structures: Dict[str, Dict] = {}
        self._counter = 0
        #: leg Asset -> owning structure id (legs are unique per
        #: structure: strike/expiry/right collisions across concurrent
        #: structures on one underlying are prevented by the one-open-
        #: structure-per-underlying entry rule).
        self.leg_owner: Dict[Asset, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open_structure(self, structure_type: str, underlying: str,
                       legs: List[Dict], credit: float, max_loss: float,
                       contracts: int, date) -> str:
        """Register a structure. legs: [{'asset': Asset, 'action':
        'SHORT'|'BUY'}] (all legs share one expiry); credit/max_loss are
        STRUCTURE-LEVEL dollars (x contracts) — estimates at entry,
        restated from actual fills by the desk's reconcile."""
        if structure_type not in STRUCTURE_TYPES:
            raise ValueError(f"Unknown structure type {structure_type!r}")
        if contracts <= 0:
            raise ValueError(f"contracts must be > 0, got {contracts}")
        self._counter += 1
        structure_id = f"JS-{self._counter:04d}"
        expiries = {leg['asset'].expiration_date for leg in legs}
        if len(expiries) != 1:
            raise ValueError(f"structure legs must share one expiry, "
                             f"got {sorted(expiries)}")
        self._structures[structure_id] = {
            'id': structure_id,
            'type': structure_type,
            'underlying': underlying,
            'opened': pd.Timestamp(date).date(),
            'closed': None,
            'credit': float(credit),
            'max_loss': float(max_loss),
            'contracts': int(contracts),
            'status': 'open',
            'pnl': None,
            'close_reason': None,
            'legs': list(legs),
            'expiry': pd.Timestamp(expiries.pop()).date(),
            'entry_filled': False,
            'last_cost_to_close': None,
        }
        for leg in legs:
            self.leg_owner[leg['asset']] = structure_id
        return structure_id

    def get(self, structure_id: str) -> Dict:
        return self._structures[structure_id]

    def remove(self, structure_id: str) -> None:
        """Drop a structure whose entry never filled (desk reconcile)."""
        structure = self._structures.pop(structure_id)
        for leg in structure['legs']:
            self.leg_owner.pop(leg['asset'], None)

    def live_ids(self) -> List[str]:
        """Ids with legs still on ('open' or 'closing'), sorted."""
        return sorted(sid for sid, s in self._structures.items()
                      if s['status'] in ('open', 'closing'))

    def open_ids(self) -> List[str]:
        """Ids still 'open' (no close emitted yet), sorted."""
        return sorted(sid for sid, s in self._structures.items()
                      if s['status'] == 'open')

    def has_open_on(self, underlying: str) -> bool:
        return any(s['underlying'] == underlying
                   and s['status'] in ('open', 'closing')
                   for s in self._structures.values())

    # ------------------------------------------------------------------
    # Marks + exits
    # ------------------------------------------------------------------
    def cost_to_close(self, structure_id: str,
                      leg_prices: Dict[Asset, float]) -> Optional[float]:
        """Structure-level $ cost of unwinding now: pay to buy back
        SHORT legs, receive for selling BUY legs, x100 x contracts.
        None when any leg lacks a mark. The result is recorded on the
        structure as last_cost_to_close (daily MTM)."""
        structure = self._structures[structure_id]
        per_contract = 0.0
        for leg in structure['legs']:
            price = leg_prices.get(leg['asset'])
            if price is None:
                return None
            per_contract += price if leg['action'] == 'SHORT' else -price
        cost = per_contract * 100.0 * structure['contracts']
        structure['last_cost_to_close'] = cost
        return cost

    def check_exits(self, structure_id: str, leg_prices: Dict[Asset, float],
                    date, profit_target_pct: float = 0.5,
                    stop_loss_mult: float = 2.0,
                    time_exit_dte: int = 21) -> Optional[str]:
        """Exit reason or None (module docstring: inclusive boundaries,
        priority profit_target > stop_loss > time_exit)."""
        structure = self._structures[structure_id]
        credit = structure['credit']
        cost = self.cost_to_close(structure_id, leg_prices)
        if cost is not None and credit > 0:
            if cost <= profit_target_pct * credit:
                return 'profit_target'
            if cost >= stop_loss_mult * credit:
                return 'stop_loss'
        dte = (structure['expiry'] - pd.Timestamp(date).date()).days
        if dte <= time_exit_dte:
            return 'time_exit'
        return None

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------
    def build_closing_intents(self, structure_id: str,
                              reason: str) -> List[DeskIntent]:
        """Per-leg closing intents (SHORT legs COVER, BUY legs SELL —
        full-size closes) and flip the structure to 'closing' with the
        close_reason recorded. Idempotent guard: raises if not 'open'."""
        structure = self._structures[structure_id]
        if structure['status'] != 'open':
            raise ValueError(f"structure {structure_id} is "
                             f"{structure['status']}, not open")
        intents = []
        for leg in structure['legs']:
            action = 'COVER' if leg['action'] == 'SHORT' else 'SELL'
            intents.append(DeskIntent(
                asset=leg['asset'], action=action, size_fraction=1.0,
                reason=f"{structure['type']} {structure_id} close: {reason}"))
        structure['status'] = 'closing'
        structure['close_reason'] = reason
        return intents

    def finalize(self, structure_id: str, closed_date, pnl: float,
                 status: str = 'closed',
                 close_reason: Optional[str] = None) -> None:
        """Mark a structure done once every leg is flat. pnl is the
        realized $ from the legs' closed trades (multiplier-aware,
        commissions cash-only as everywhere else)."""
        if status not in ('closed', 'expired'):
            raise ValueError(f"final status must be closed/expired, "
                             f"got {status!r}")
        structure = self._structures[structure_id]
        structure['status'] = status
        structure['closed'] = pd.Timestamp(closed_date).date()
        structure['pnl'] = float(pnl)
        if close_reason is not None:
            structure['close_reason'] = close_reason
        for leg in structure['legs']:
            self.leg_owner.pop(leg['asset'], None)

    # ------------------------------------------------------------------
    # C11 serialization
    # ------------------------------------------------------------------
    def to_report(self) -> List[Dict]:
        """Contract C11 list (id order). 'closing' serializes as 'open'
        (legs still on); credit is total $ received at entry, max_loss
        total $ positive, pnl realized $ once closed/expired else null."""
        report = []
        for structure_id in sorted(self._structures):
            s = self._structures[structure_id]
            report.append({
                'id': s['id'],
                'type': s['type'],
                'underlying': s['underlying'],
                'opened': s['opened'].strftime('%Y-%m-%d'),
                'closed': (s['closed'].strftime('%Y-%m-%d')
                           if s['closed'] is not None else None),
                'credit': s['credit'],
                'max_loss': s['max_loss'],
                'contracts': s['contracts'],
                'status': 'open' if s['status'] == 'closing' else s['status'],
                'pnl': s['pnl'],
                'close_reason': s['close_reason'],
                'legs': [
                    {
                        'instrument': str(leg['asset']),
                        'action': leg['action'],
                        'strike': float(leg['asset'].strike_price),
                        'expiry': leg['asset'].expiration_date,
                        'right': leg['asset'].asset_type.value,
                    }
                    for leg in s['legs']
                ],
            })
        return report
