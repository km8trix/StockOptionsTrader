"""
Dynamic capital reweighter — in-run, walk-forward-honest risk parity.

CrossDeskCapitalAllocator turns per-desk return series into risk-parity
weights, but only at CONSTRUCTION time (run each desk solo, weight once, then
build the fund). This module lifts that to a DYNAMIC, in-run rebalance using
the shadow-solo-book design chosen for the fund: the caller precomputes each
desk's STANDALONE equity curve (a single-desk backtest over the same window,
run at full NAV so the curve is an undistorted measure of that desk's solo
risk), and the reweighter slices those curves to the rebalance date, derives
returns, risk-parity-weights them, and writes the weights into the LIVE fund
desks' capital_allocation for the next window.

WHY SHADOW-SOLO (not per-desk attribution on the unified book): the fund is one
netted book with no per-desk P&L ledger; attributing realized pnl back to desks
through blended entry prices, partial closes and the account-wide apply_risk is
both treacherous AND produces a signal COUPLED across desks (a desk's measured
return would depend on whatever netted against it). Each desk's solo curve is
the clean, textbook risk-parity input and needs no attribution surgery.

WALK-FORWARD HONESTY: on_day only ever reads curve points dated on/before the
rebalance date, and a desk's solo curve is itself causal (the single-desk
engine uses expanding-window indicators and the desk's own walk-forward
fitting), so no future information can reach a weight. Weights are FRACTIONS of
the account (they sum to the allocator's target_gross <= 1.0), so applying them
never moves cash or breaks the equity curve — only the NEXT window's sizing
changes.

ADDITIVE: a fund run without a reweighter is unchanged; single-desk and
strategy runs never touch this path.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

from desks.capital_allocator import (CrossDeskCapitalAllocator,
                                     set_desk_capital_allocation)

if TYPE_CHECKING:  # annotation only — avoids a runtime import cycle
    from desks.base import Desk

logger = logging.getLogger(__name__)

#: Tolerance on the post-rebalance weight sum (<= 1.0) defence-in-depth check.
_SUM_TOL = 1e-9


class DynamicReweighter:
    """Periodically reweights a fund's desks from their standalone curves.

    Pre-load each desk's solo equity curve via set_curves, then call on_day
    once per simulated trading day; on a rebalance boundary it sets each live
    desk's capital_allocation in place and records the rebalance.
    """

    #: Supported weighting modes -> the allocator method each invokes.
    _WEIGHTING_MODES = ('risk_parity', 'performance', 'risk_parity_cov',
                        'max_diversification', 'mean_variance')

    def __init__(self,
                 allocator: Optional[CrossDeskCapitalAllocator] = None,
                 rebalance_every: int = 21,
                 warmup: int = 0,
                 weighting: str = 'risk_parity'):
        if rebalance_every < 1:
            raise ValueError(
                f"rebalance_every must be >= 1; got {rebalance_every}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0; got {warmup}")
        if weighting not in self._WEIGHTING_MODES:
            raise ValueError(
                f"weighting must be one of {self._WEIGHTING_MODES}; "
                f"got {weighting!r}")
        self.allocator = allocator or CrossDeskCapitalAllocator()
        self.rebalance_every = int(rebalance_every)
        self.warmup = int(warmup)
        #: 'risk_parity' (default, byte-identical) -> inverse-vol; 'performance'
        #: -> opt-in guarded risk-adjusted performance tilt; 'risk_parity_cov'
        #: -> opt-in full-covariance risk parity (equalizes risk contributions
        #: over the full covariance matrix, not the inverse-vol diagonal);
        #: 'max_diversification' -> opt-in max diversification-ratio (w ∝ Σ⁻¹σ);
        #: 'mean_variance' -> opt-in long-only max-Sharpe (w ∝ Σ⁻¹μ). Every
        #: opt-in mode degrades to the inverse-vol path.
        self.weighting = weighting
        self._curves: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        #: Append-only audit of every rebalance actually applied.
        self.rebalance_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Curve loading
    # ------------------------------------------------------------------
    def set_curves(self,
                   curves: Dict[str, Sequence[Tuple[object, float]]]) -> None:
        """Load each desk's standalone equity curve as (timestamp, value)
        pairs. Stored sorted by timestamp with values coerced to float;
        replaces any previously loaded curves. A desk with no curve here is
        degenerate at rebalance time and triggers the allocator's WHOLE-FUND
        equal-weight fallback (every desk shares equally, not just the missing
        one); on_day records that in the rebalance log so it is not mistaken
        for a genuine risk-parity result."""
        stored: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        for key, pairs in curves.items():
            seq = [(pd.Timestamp(ts), float(val)) for ts, val in pairs]
            seq.sort(key=lambda pair: pair[0])
            stored[key] = seq
        self._curves = stored

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------
    def is_rebalance_day(self, day_number: int) -> bool:
        """True on a rebalance boundary: strictly past the warmup, then every
        ``rebalance_every`` trading days (day_number is 1-based)."""
        if day_number <= self.warmup:
            return False
        return (day_number - self.warmup) % self.rebalance_every == 0

    # ------------------------------------------------------------------
    # Per-day hook
    # ------------------------------------------------------------------
    def on_day(self, desks: Sequence['Desk'], date,
               day_number: int) -> Optional[Dict[str, float]]:
        """Called once per simulated day. On a rebalance boundary, recompute
        risk-parity weights from each desk's curve sliced to ``date`` and write
        them into desk.capital_allocation, returning the weights. Returns None
        on non-boundary days (a no-op)."""
        if not self.is_rebalance_day(day_number):
            return None
        as_of = pd.Timestamp(date)
        returns_by_desk: Dict[str, List[float]] = {}
        for desk in desks:
            points = self._curves.get(desk.key, [])
            # Walk-forward honesty: only points observable on/before today.
            values = [value for ts, value in points if ts <= as_of]
            returns_by_desk[desk.key] = \
                self.allocator.returns_from_equity(values)

        # Record WHY (if at all) this rebalance is degenerate, so an
        # equal-weight fallback is auditable rather than masquerading as a real
        # risk-parity decision (a flat/missing/short solo curve forces the
        # allocator's WHOLE-FUND equal-weight fallback). degenerate_desks is the
        # SAME gate both weighting modes use, so it audits either mode.
        degraded = self.allocator.degenerate_desks(returns_by_desk)
        # Mode selection: default 'risk_parity' calls risk_parity_weights
        # EXACTLY as before (byte-identical); 'performance' opts into the
        # guarded risk-adjusted tilt; 'risk_parity_cov', 'max_diversification'
        # and 'mean_variance' opt into the covariance-aware optimizer modes. All
        # opt-in modes degrade to the inverse-vol path. The covariance modes can
        # degrade for NON-degenerate-desk reasons (short overlap, singular cov, a
        # negatively-correlated hedge desk); capture WHY via the *_with_status
        # variants so the audit below records an honest fallback rather than
        # fallback=False on a rebalance that did not use its intended weighting.
        degrade_reason: Optional[str] = None
        if self.weighting == 'performance':
            weights = self.allocator.performance_weights(returns_by_desk)
        elif self.weighting == 'risk_parity_cov':
            weights, degrade_reason = \
                self.allocator.risk_parity_cov_weights_with_status(
                    returns_by_desk)
        elif self.weighting == 'max_diversification':
            weights, degrade_reason = \
                self.allocator.max_diversification_weights_with_status(
                    returns_by_desk)
        elif self.weighting == 'mean_variance':
            weights, degrade_reason = \
                self.allocator.mean_variance_weights_with_status(
                    returns_by_desk)
        else:
            weights = self.allocator.risk_parity_weights(returns_by_desk)

        # Defence-in-depth at the mutation site: NEVER write an over-leveraged
        # set to the live book. Compare against the allocator's OWN target_gross
        # (not a literal 1.0 — a sub-1.0 target must be honored too); on a
        # breach, renormalize the weights DOWN to target_gross rather than only
        # logging and deploying past target. Both built-in modes already sum to
        # target_gross exactly, so this is a no-op on every normal path (the
        # default risk-parity weights are written unchanged).
        target_gross = self.allocator.target_gross
        total = sum(weights.values())
        if total > target_gross + _SUM_TOL:
            logger.error(
                "Reweight on %s (day %d) produced weights summing to %.6f > "
                "target_gross %.4g — renormalizing down (allocator contract "
                "breach?)", as_of.date(), day_number, total, target_gross)
            if total > 0.0:
                weights = {key: target_gross * weight / total
                           for key, weight in weights.items()}

        for desk in desks:
            if desk.key in weights:
                set_desk_capital_allocation(desk, weights[desk.key])

        # fallback is True for EITHER a degenerate-desk equal-weight fallback OR
        # a cov-mode numerical degrade to inverse-vol, so the audit never reads
        # fallback=False on a rebalance that did not use its intended weighting.
        # degrade_reason names the cov numerical cause (None otherwise); the two
        # are mutually exclusive (the degenerate gate short-circuits cov).
        fell_back = bool(degraded) or degrade_reason is not None
        self.rebalance_log.append({
            'date': as_of,
            'day_number': day_number,
            'weights': dict(weights),
            'fallback': fell_back,
            'degraded_desks': sorted(degraded),
            'degrade_reason': degrade_reason,
            'mode': self.weighting,
        })
        if degraded:
            logger.warning(
                "Fund reweight on %s (day %d) fell back to EQUAL WEIGHT; "
                "degraded desks: %s", as_of.date(), day_number, degraded)
        else:
            logger.info("Fund reweight on %s (day %d): %s",
                        as_of.date(), day_number,
                        {key: round(weight, 4)
                         for key, weight in weights.items()})
        return weights
