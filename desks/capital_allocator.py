"""
Cross-desk capital allocator — sizes each desk's slice of the fund.

FundOrchestrator runs N desks on one account, each deploying its own
capital_allocation (validated to sum <= 1.0). Hand-setting those weights is
fine, but a fund usually sizes strategies by their RISK: this allocator derives
the per-desk capital_allocation from each desk's return series using RISK
PARITY (inverse-volatility weighting), normalized to sum to a target gross
<= 1.0. It lifts Citadel's pod-level weighting up to the desk level.

THE WORKFLOW (no per-desk attribution required): the fund's portfolio is
unified — there is no per-desk P&L ledger to read mid-run — so this allocator
works on return series the caller supplies, the natural source being each
desk's STANDALONE backtest equity curve (run each desk solo, take its
portfolio_history, derive returns, allocate, THEN build the fund with those
weights). Dynamic in-run reweighting would require per-desk attribution the
unified book does not provide and is intentionally out of scope here.

RISK PARITY: weight_i proportional to 1/vol_i (vol = stdev of the desk's
returns), then scaled so the weights sum to target_gross. A desk with no
usable or degenerate (non-finite / zero) volatility cannot be inverse-vol
weighted; per the Phase 2 decision the allocator then FALLS BACK to EQUAL
WEIGHT across all desks (a try/except guard, NO scipy/sklearn dependency).

WALK-FORWARD HONESTY: the allocator computes purely from the returns passed in;
the caller is responsible for passing only returns observable at the rebalance
(no look-ahead). NAV CONTINUITY: weights are FRACTIONS of the current account,
not dollar reallocations, so applying them never moves cash or breaks the
equity curve.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np

from desks.base import Desk

logger = logging.getLogger(__name__)

#: Float tolerance on the target_gross (<= 1.0) bound.
_TARGET_TOL = 1e-9


class CrossDeskCapitalAllocator:
    """Risk-parity (inverse-vol) capital weights across desks, summing to a
    target gross <= 1.0, with an equal-weight fallback."""

    def __init__(self, target_gross: float = 1.0):
        if not (0.0 < target_gross <= 1.0 + _TARGET_TOL):
            raise ValueError(
                f"target_gross {target_gross} must be in (0, 1]")
        self.target_gross = min(target_gross, 1.0)

    # ------------------------------------------------------------------
    # Weighting
    # ------------------------------------------------------------------
    @staticmethod
    def degenerate_desks(returns_by_desk: Dict[str, Sequence[float]]
                         ) -> Dict[str, str]:
        """Map each desk whose returns cannot be inverse-vol weighted to the
        reason it is degenerate ('<2 finite returns', 'degenerate vol ...', or
        'unusable returns (...)'). An empty dict means risk parity is
        well-defined for EVERY desk. This is the single definition of the
        degeneracy rule — risk_parity_weights consults it before deciding to
        fall back, and callers (e.g. DynamicReweighter) consult it to record
        WHY a rebalance fell back to equal weight, instead of silently
        emitting equal weights that look like a real risk-parity decision.
        """
        degenerate: Dict[str, str] = {}
        for key, returns in returns_by_desk.items():
            try:
                arr = np.asarray(list(returns), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size < 2:
                    degenerate[key] = '<2 finite returns'
                    continue
                vol = float(arr.std())
                if not np.isfinite(vol) or vol <= 0.0:
                    degenerate[key] = f'degenerate vol {vol:g}'
            except Exception as exc:  # malformed / non-numeric series
                degenerate[key] = f'unusable returns ({exc})'
        return degenerate

    def risk_parity_weights(self,
                            returns_by_desk: Dict[str, Sequence[float]]
                            ) -> Dict[str, float]:
        """Inverse-volatility weights over the desks, scaled to sum to
        target_gross. Falls back to EQUAL WEIGHT (still summing to
        target_gross) when ANY desk's volatility is unusable (too few points,
        non-finite, or zero) — risk parity is undefined there, so the
        conservative whole-fund fallback avoids over-weighting a desk merely
        because its sample looked calm.
        """
        keys = list(returns_by_desk.keys())
        if not keys:
            return {}
        degenerate = self.degenerate_desks(returns_by_desk)
        if degenerate:
            logger.warning(
                "Risk-parity allocation fell back to equal weight: %s",
                degenerate)
            return self._equal_weight(keys)
        inv_vol: Dict[str, float] = {}
        for key in keys:
            arr = np.asarray(list(returns_by_desk[key]), dtype=float)
            arr = arr[np.isfinite(arr)]
            inv_vol[key] = 1.0 / float(arr.std())
        total = sum(inv_vol.values())
        if not np.isfinite(total) or total <= 0.0:
            logger.warning(
                "Risk-parity inverse-vol weights do not sum positive (%s); "
                "falling back to equal weight", total)
            return self._equal_weight(keys)
        return {key: self.target_gross * inv_vol[key] / total
                for key in keys}

    def _equal_weight(self, keys: Sequence[str]) -> Dict[str, float]:
        keys = list(keys)
        if not keys:
            return {}
        weight = self.target_gross / len(keys)
        return {key: weight for key in keys}

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def allocate(self, desks: List[Desk],
                 returns_by_desk: Dict[str, Sequence[float]]
                 ) -> Dict[str, float]:
        """Compute risk-parity weights for `desks` and SET each desk's
        capital_allocation in place; returns the weights {desk.key: weight}.
        A desk with no supplied returns is treated as an empty series (which
        triggers the equal-weight fallback). The resulting weights sum to
        target_gross <= 1.0, so a FundOrchestrator built from these desks
        passes its sum<=1.0 validation.
        """
        weights = self.risk_parity_weights(
            {desk.key: returns_by_desk.get(desk.key, []) for desk in desks})
        for desk in desks:
            desk.capital_allocation = weights[desk.key]
        return weights

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def returns_from_equity(equity_curve: Sequence[float]) -> List[float]:
        """Simple period-over-period returns from an equity curve (e.g. a
        desk's standalone portfolio_history values). Non-finite or
        non-positive points break the curve into segments — returns are taken
        only across consecutive valid points (never across a gap)."""
        values = [float(v) for v in equity_curve]
        out: List[float] = []
        for prev, curr in zip(values, values[1:]):
            if (np.isfinite(prev) and np.isfinite(curr) and prev > 0.0
                    and curr > 0.0):
                out.append(curr / prev - 1.0)
        return out
