"""
Citadel Desk — strategy pods, a central risk book, and performance-
weighted capital reallocation with drawdown-based pod cuts.

PODS: three by default (desks.pods) at initial weights {momentum: 1/3,
mean_reversion: 1/3, stat_arb: 1/3}. Pods emit per-symbol actions; the
desk owns sizing, conflict resolution, pod accounting and risk.

POD ACCOUNTING / ATTRIBUTION (the identity that must hold to the penny):
every position is tagged with its owning pod at entry; a sticky
symbol->pod map additionally covers positions whose tracking was dropped
(late fills, external closes) so NO desk position's P&L is ever
unattributed. Each day, BEFORE any weight changes:

    pod_daily_pnl  = (unrealized_today - unrealized_yesterday)
                     + realized P&L of the pod's trades closed today
    allocated_cap  = start_of_day desk capital * pod weight in force
                     (start of day = the previous portfolio snapshot,
                      i.e. yesterday's close; weights = those recorded in
                      yesterday's pod_history entry)
    pod_nav       *= 1 + pod_daily_pnl / allocated_cap

so sum_pods(allocated_cap * (nav_t/nav_{t-1} - 1)) == the desk's daily
position P&L (realized + unrealized change) exactly, commission-free on
both sides. The NAV index starts at 1.0 and CONTINUES across
reallocations — weight changes rescale future P&L, never the history.
A CUT pod keeps attributing residual P&L (positions awaiting their
flatten fill, e.g. slippage between the cut-day mark and the next open)
against its last positive allocated capital, then freezes. That base is
published as 'residual_capital' on the pod's C8 pod_history entries
from the cut day onward (an ADDITIVE field, present only while status
is 'cut'), so the identity stays reconstructible purely from the
report:

    base_t(pod) = start_capital_t * weight_{t-1}(pod)  if the weight
                  recorded in yesterday's pod_history entry is > 0,
                  else yesterday's residual_capital (0.0 if absent)
    sum_pods(base_t * (nav_t/nav_{t-1} - 1)) == desk position P&L day t

DRAWDOWN CUTS (boundary semantics are exact: ``<=`` triggers):
    drawdown = (nav - running_max_nav) / running_max_nav
    drawdown <= probation_drawdown (-5%):  PROBATION — weight halved
        immediately, the excess redistributed pro-rata to the other
        ACTIVE pods (falling back to probation pods if none are active).
        When NO recipient pod remains (every other pod is cut), the
        weight is NOT halved — retiring live mass would break weight
        conservation — and probation degrades to a status flip with a
        note.
    drawdown <= cut_drawdown (-8%):  CUT — the pod is stopped permanently
        for the run, all its positions get closing intents (SELL/COVER,
        filling at the next bar's open), and its weight is redistributed
        pro-rata (retired only in the true all-cut case, where the
        desk's live weight is 0.0). A probation pod that keeps falling
        is cut the same way.
    A probation pod returns to normal eligibility at the next scheduled
    reallocation IFF its drawdown has recovered STRICTLY above
    recovery_drawdown (-2.5%); at exactly the threshold it stays held.

REALLOCATION (every realloc_every_days=21 trading days, skipping cut
pods): per participating pod, over the trailing sharpe_window_days=63
daily NAV returns —

    sharpe = mean(r) / std(r, ddof=0) * sqrt(252)   (0.0 when std == 0)
    score  = max(0.0, sharpe)
    vol    = max(vol_floor=0.05, std(r, ddof=0) * sqrt(252))
    raw    = max(score_floor=0.05, score) / vol

weights = raw normalized over participating pods to the unheld mass,
clamped to [weight_min=0.10, weight_max=0.50] and renormalized via
clamp_renormalize (fixed point and the two infeasibility directions
documented there, max 10 iterations; the min bound yields when the
unheld mass cannot support it, so held + participating weight never
exceeds 1.0). When the unheld mass exceeds len(participating) *
weight_max (e.g. a lone survivor after cuts), the recompute is SKIPPED
and the participants are held at their current weights — the hard max
cap would otherwise under-allocate the desk. A pod with fewer than
min_nav_days=21 NAV returns keeps its current weight (held out, like
an unrecovered probation pod).

WEIGHT CONSERVATION: constructor weights must sum to exactly 1.0 and
every redistribution path conserves that total, so active + probation
weights sum to 1.0 (within 1e-9) on every pod_history entry; mass is
retired only in the true all-cut case, where the desk's live weight
is exactly 0.0.

CONFLICTS: first-claim — when two pods signal an entry for the same
symbol on the same day, the pod with the higher CURRENT weight wins,
ties broken alphabetically by pod key (same pattern as the Renaissance
books' symbol exclusivity); the loser's signal is dropped with an 'info'
note carrying data.pod. A symbol already owned by any pod (or held in
the portfolio at all) is off-limits to every other pod.

ORDER FLOW each day: pods generate signals -> conflict resolution ->
sizing (size_fraction = pod_weight * min(0.25, 1/max(4, n)) where n is
the pod's actionable entry count today) -> CentralRiskBook.check (closes
first, then opens by pod weight desc, then symbol; every block noted
with data.pod) -> the engine then runs the shared base apply_risk
(stops, daily-loss circuit).

CONTRACTS: pod_history per C8 (one entry per simulated day; drawdown_pct
is a percentage <= 0, matching the drawdown_series convention); notes
per C9 (data.pod on pod-specific notes, data.allocations + data.reason
on reallocations, data.pod + data.drawdown_pct on cuts/probations);
registry per C10. walk_forward_fits aggregates pod controllers tagged
with the pod key (additive, same pattern as the Renaissance desk).

MARGIN IS NOT MODELED: shorts use a cash-account approximation (proceeds
held as cash); live shorting is gated to Phase 9.
"""

from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent, OPENING_ACTIONS
from desks.factor_risk import FactorRiskModel
from desks.pods import (MeanReversionPod, MomentumPod, Pod, POD_ACTIONS,
                        StatArbPod)
from desks.renaissance import TaggedWalkForwardFit
from desks.risk_book import CentralRiskBook
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Trading days a tracked entry may stay unfilled/absent before its
#: bookkeeping is dropped (same engine timing as the Renaissance desk).
RECONCILE_GRACE_DAYS = 2

#: Pod status values surfaced through pod_history (contract C8).
POD_STATUSES = ('active', 'probation', 'cut')


def _stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def pod_score_inputs(returns: Sequence[float], vol_floor: float = 0.05,
                     score_floor: float = 0.05,
                     robust: bool = False) -> Dict[str, float]:
    """Reallocation inputs for one pod from its daily NAV returns.

    Default (``robust=False``): sharpe = mean/std * sqrt(252) with the
    POPULATION std (ddof=0, the project-wide convention); 0.0 when std == 0.
    ``robust=True``: a MAD-based robust Sharpe — location = median(r), scale =
    median(|r - median(r)|) * 1.4826 (the normal-consistency constant, so MAD
    estimates sigma). It resists return outliers — a single blow-up day no
    longer dominates a pod's score or vol the way mean/std let it.

    Both modes: score floors sharpe at 0; vol floors the annualized scale at
    vol_floor; raw = max(score_floor, score) / vol.
    """
    values = np.asarray(list(returns), dtype=float)
    if robust:
        loc = float(np.median(values)) if values.size else 0.0
        mad = float(np.median(np.abs(values - loc))) if values.size else 0.0
        scale = mad * 1.4826  # MAD -> sigma for ~normal data
    else:
        loc = float(np.mean(values)) if values.size else 0.0
        scale = float(np.std(values)) if values.size else 0.0
    sharpe = 0.0 if scale <= 0 else loc / scale * float(np.sqrt(252.0))
    score = max(0.0, sharpe)
    vol = max(vol_floor, scale * float(np.sqrt(252.0)))
    raw = max(score_floor, score) / vol
    return {'sharpe': sharpe, 'score': score, 'vol': vol, 'raw': raw}


def clamp_renormalize(raw: Dict[str, float], target: float,
                      weight_min: float, weight_max: float,
                      max_iterations: int = 10) -> Dict[str, float]:
    """Normalize raw scores to weights summing to `target`, each clamped
    to [weight_min, weight_max].

    THE FIXED POINT: every weight is either pinned at a bound or
    proportional to its raw score within the unpinned mass. The solver
    is a water-filling iteration over a PINNED set: each pass recomputes
    the unpinned weights proportionally from the RAW scores over the
    remaining (target - pinned) mass — never by rescaling previously
    clamped values, which would destroy the raw proportions — then pins
    only the bound violations in the direction the clamped sum
    indicates:
    * clamped sum < remaining mass: mass was cut at weight_max, so the
      true multiplier is larger and every key whose proportional share
      reaches weight_max is genuinely capped — pin those at weight_max;
    * clamped sum > remaining mass: mass was added at the floor, so the
      true multiplier is smaller and every key whose proportional share
      falls to the floor is genuinely floored — pin those at the floor.
    Each pass either converges or pins at least one more key, so the
    fixed point is reached in <= len(raw) + 1 passes; max_iterations
    (10) is a hard safety bound.

    INFEASIBLE BOUNDS — the two directions are deliberately asymmetric,
    and the returned sum NEVER exceeds `target` either way:
    * Under-allocation (target > n * weight_max): the max bound is HARD;
      every key pins at weight_max and the returned sum falls short of
      the target — the desk deliberately runs below full allocation.
    * Over-allocation (target < n * weight_min): the min bound is SOFT;
      the per-key floor drops to target/n so the result still sums to
      the target exactly. A hard floor here would return MORE mass than
      requested — at the desk level that would push held + participating
      weight above 1.0, violating the total-weight invariant.
    """
    if not raw:
        return {}
    # Soft min floor: never let the floors alone overshoot the target
    # (over-allocation infeasibility, documented above).
    floor = min(weight_min, max(0.0, target) / len(raw))
    total_raw = sum(raw.values())
    if total_raw <= 0:  # degenerate: no signal -> equal scores
        scores = {key: 1.0 for key in raw}
    else:
        scores = dict(raw)

    pinned: Dict[str, float] = {}
    unpinned = sorted(scores)
    weights: Dict[str, float] = {}
    for _ in range(max_iterations):
        if not unpinned:
            break
        remaining = max(0.0, target - sum(pinned.values()))
        score_sum = sum(scores[key] for key in unpinned)
        if score_sum > 0:
            proportional = {key: scores[key] / score_sum * remaining
                            for key in unpinned}
        else:  # zero-score stragglers: split the remaining mass evenly
            proportional = {key: remaining / len(unpinned)
                            for key in unpinned}
        weights = {key: min(weight_max, max(floor, value))
                   for key, value in proportional.items()}
        total = sum(weights.values())
        if abs(total - remaining) <= 1e-12:
            break
        if total < remaining:
            newly = [key for key in unpinned
                     if proportional[key] >= weight_max]
            bound = weight_max
        else:
            newly = [key for key in unpinned
                     if proportional[key] <= floor]
            bound = floor
        if not newly:  # numerical safety; unreachable mathematically
            break
        for key in newly:
            pinned[key] = bound
        unpinned = [key for key in unpinned if key not in pinned]

    result = {key: value for key, value in weights.items()
              if key not in pinned}
    result.update(pinned)
    return {key: result[key] for key in raw}


class CitadelDesk(Desk):
    """Pod-based multi-strategy desk (see module docstring)."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 pods: Optional[List[Pod]] = None,
                 pod_weights: Optional[Dict[str, float]] = None,
                 risk_book: Optional[CentralRiskBook] = None,
                 probation_drawdown: float = -0.05,
                 cut_drawdown: float = -0.08,
                 recovery_drawdown: float = -0.025,
                 realloc_every_days: int = 21,
                 sharpe_window_days: int = 63,
                 min_nav_days: int = 21,
                 weight_min: float = 0.10,
                 weight_max: float = 0.50,
                 vol_floor: float = 0.05,
                 score_floor: float = 0.05,
                 robust_pod_sharpe: bool = False,
                 max_factor_exposure: Optional[float] = 0.25,
                 factor_risk_model: Optional[FactorRiskModel] = None):
        super().__init__(
            key='citadel',
            name='Citadel Desk',
            description=('Strategy pods with their own capital, a central '
                         'risk book, and performance-weighted capital '
                         'allocation with drawdown-based pod cuts.'),
            accent='#bc8cff',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )

        pods = pods if pods is not None else [MomentumPod(),
                                              MeanReversionPod(),
                                              StatArbPod()]
        if not pods:
            raise ValueError("CitadelDesk needs at least one pod")
        keys = [pod.key for pod in pods]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate pod keys: {sorted(keys)}")
        self._pods: Dict[str, Pod] = {pod.key: pod for pod in pods}

        if pod_weights is None:
            weights = {key: 1.0 / len(pods) for key in self._pods}
        else:
            unknown = set(pod_weights) - set(self._pods)
            missing = set(self._pods) - set(pod_weights)
            if unknown or missing:
                raise ValueError(
                    f"pod_weights keys {sorted(pod_weights)} must match pod "
                    f"keys {sorted(self._pods)}")
            weights = dict(pod_weights)
        for key, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Pod weight for '{key}' must be >= 0, got {weight}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            # Sub-1.0 sums are rejected too: every redistribution path
            # (probation, cuts, reallocation) conserves total weight, and
            # _maybe_reallocate targets the full unheld mass — a desk
            # constructed under-allocated would silently jump to full
            # deployment at the first scheduled reallocation.
            raise ValueError(
                f"Pod weights sum to {total:.4f}; must sum to 1.0")
        self._weights: Dict[str, float] = weights

        if not (cut_drawdown <= probation_drawdown < 0):
            raise ValueError(
                f"Need cut_drawdown ({cut_drawdown}) <= probation_drawdown "
                f"({probation_drawdown}) < 0")
        if not (probation_drawdown <= recovery_drawdown <= 0):
            raise ValueError(
                f"Need probation_drawdown ({probation_drawdown}) <= "
                f"recovery_drawdown ({recovery_drawdown}) <= 0")
        if not (0 <= weight_min <= weight_max <= 1):
            raise ValueError(
                f"Need 0 <= weight_min ({weight_min}) <= weight_max "
                f"({weight_max}) <= 1")
        self.probation_drawdown = probation_drawdown
        self.cut_drawdown = cut_drawdown
        self.recovery_drawdown = recovery_drawdown
        self.realloc_every_days = realloc_every_days
        self.sharpe_window_days = sharpe_window_days
        self.min_nav_days = min_nav_days
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.vol_floor = vol_floor
        self.score_floor = score_floor
        self.robust_pod_sharpe = robust_pod_sharpe

        self.risk_book = (risk_book if risk_book is not None
                          else CentralRiskBook(
                              capital_allocation=capital_allocation))

        # --- Factor-neutrality band (the Phase E sharpening) ------------
        # A FactorRiskModel measures the aggregate book's net loading on the
        # common price factors (momentum/reversal/low_vol/risk_adj_mom,
        # reused from the AQR factor set); the central risk book caps how
        # directional the WHOLE book may get on any single factor so no pod
        # (nor the pods jointly) can quietly load the desk on one common
        # driver. ON by default (max_factor_exposure=0.25 of desk capital);
        # set max_factor_exposure to None to recover the old, non-factor-
        # gated behavior (the band gate then always passes). Registered ONCE
        # here; _refresh_factor_loadings updates the loadings the gate reads
        # each simulated day, so the check is leakage-free by construction.
        if max_factor_exposure is not None and max_factor_exposure <= 0:
            raise ValueError(
                f"max_factor_exposure must be > 0 or None, got "
                f"{max_factor_exposure}")
        self.max_factor_exposure = max_factor_exposure
        self.factor_risk_model = (factor_risk_model
                                  if factor_risk_model is not None
                                  else FactorRiskModel())
        #: live per-symbol loadings the registered factor limit reads; the
        #: desk refreshes this each day from the date-sliced all_data.
        self._factor_loadings_ref: Dict[str, Dict] = {'loadings': {}}
        #: latest aggregate book net factor exposure (transparency surface).
        self._book_factor_exposure: Dict[str, float] = {}
        if max_factor_exposure is not None:
            self.risk_book.register_limit(
                'factor_neutrality',
                CentralRiskBook.factor_neutrality_limit(
                    self.factor_risk_model, self._factor_loadings_ref,
                    max_factor_exposure))

        # --- Pod accounting state ---------------------------------------
        self._statuses: Dict[str, str] = {key: 'active' for key in self._pods}
        self._navs: Dict[str, float] = {key: 1.0 for key in self._pods}
        self._nav_max: Dict[str, float] = {key: 1.0 for key in self._pods}
        #: Daily NAV returns per pod (reallocation inputs).
        self._nav_returns: Dict[str, List[float]] = {key: []
                                                     for key in self._pods}
        #: Yesterday's unrealized P&L per pod (for the daily delta).
        self._prev_unrealized: Dict[str, float] = {key: 0.0
                                                   for key in self._pods}
        #: Last positive allocated capital per pod (residual attribution
        #: for cut pods whose flatten fills are still pending).
        self._last_allocated: Dict[str, float] = {key: 0.0
                                                  for key in self._pods}
        self._n_closed_trades_seen = 0

        # --- Position tracking (Renaissance pattern) --------------------
        #: asset -> {'pod', 'direction', 'entry_day'}
        self._pod_positions: Dict[Asset, Dict] = {}
        #: symbol -> pod that last emitted an entry for it (sticky; scopes
        #: the orphan sweep AND the attribution fallback).
        self._traded_pods: Dict[str, str] = {}

        # --- Clocks / report state ---------------------------------------
        self._day_index = 0
        self._last_seen_date: Optional[date_type] = None
        self._days_since_realloc = 0
        self._pod_history: List[Dict] = []
        self._fit_counts: Dict[str, int] = {key: 0 for key in self._pods}
        self._all_cut_noted = False

    # ------------------------------------------------------------------
    # Introspection (C3+ / C8)
    # ------------------------------------------------------------------
    @property
    def walk_forward_fits(self) -> List[TaggedWalkForwardFit]:
        """Pod controllers' fits, each tagged with its pod key (C3+)."""
        tagged: List[TaggedWalkForwardFit] = []
        for key in sorted(self._pods):
            controller = self._pods[key].controller
            if controller is None:
                continue
            tagged.extend(TaggedWalkForwardFit(fit=fit, model=key)
                          for fit in controller.fits)
        tagged.sort(key=lambda t: (t.fit.fit_date, t.model))
        return tagged

    @property
    def pod_history(self) -> List[Dict]:
        """C8 series: one {'date', 'pods'} entry per simulated day."""
        return self._pod_history

    def get_status(self) -> Dict:
        status = super().get_status()
        status['pods'] = self._pod_snapshot()
        # AQR-style transparency: the operator sees the book's net factor
        # exposures (fraction of desk capital) and the neutrality band in
        # force. None band -> the gate is off; the field reports that.
        status['factor_neutrality'] = {
            'band': self.max_factor_exposure,
            'book_factor_exposure': {
                f: float(v) for f, v in self._book_factor_exposure.items()},
        }
        return status

    def _pod_snapshot(self) -> Dict[str, Dict]:
        """Per-pod {'weight','nav','drawdown_pct','status'} (C8 shape).

        Cut pods additionally carry 'residual_capital' — the last
        positive allocated-capital base their residual P&L (flatten
        fills still pending) is attributed against — so the attribution
        identity is reconstructible purely from the report even on the
        flatten-fill day, when the recorded weight is already 0 (module
        docstring; additive to contract C8)."""
        snapshot: Dict[str, Dict] = {}
        for key in sorted(self._pods):
            entry = {
                'weight': float(self._weights[key]),
                'nav': float(self._navs[key]),
                'drawdown_pct': float(self._drawdown(key) * 100.0),
                'status': self._statuses[key],
            }
            if self._statuses[key] == 'cut':
                entry['residual_capital'] = float(
                    self._last_allocated.get(key, 0.0))
            snapshot[key] = entry
        return snapshot

    def _drawdown(self, pod_key: str) -> float:
        """(nav - running_max)/running_max, always <= 0 (max >= nav)."""
        peak = self._nav_max[pod_key]
        return min(0.0, (self._navs[pod_key] - peak) / peak)

    # ------------------------------------------------------------------
    # Intent generation — one simulated day
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """One day: attribute P&L -> reconcile -> drawdown policy ->
        (maybe) reallocate -> pod signals -> conflicts/sizing -> risk
        book. Records the day's pod_history entry last, with the weights
        now in force."""
        self._advance_day(date)
        self._update_pod_navs(portfolio)
        self._refresh_factor_loadings(all_data, date)

        tagged: List[Tuple[str, DeskIntent]] = []
        tagged.extend(self._reconcile_pods(portfolio))
        tagged.extend(self._apply_drawdown_policy())
        self._maybe_reallocate()
        tagged.extend(self._collect_pod_intents(all_data, date, portfolio))

        approved = self._apply_risk_book(tagged, portfolio, all_data)
        # Surface the held-book factor exposure EVERY day — even on
        # intent-less days — so drift on already-held positions is visible:
        # the gate is an accumulation backstop on new opens, not a realized
        # cap, so the held book can drift as loadings rotate.
        self._note_book_factor_exposure(
            portfolio, self._latest_prices(all_data))
        self._record_pod_history(date)
        return approved

    def _advance_day(self, date) -> None:
        current = pd.Timestamp(date).date()
        if self._last_seen_date is not None and current != self._last_seen_date:
            self._day_index += 1
            self._days_since_realloc += 1
        self._last_seen_date = current

    # ------------------------------------------------------------------
    # Factor-neutrality: refresh the loadings the risk-book gate reads
    # ------------------------------------------------------------------
    def _refresh_factor_loadings(self, all_data: Dict[str, pd.DataFrame],
                                 date) -> None:
        """Recompute the day's per-symbol factor loadings and publish them
        into the reference the registered factor limit reads.

        Leakage-free: ``all_data`` is the engine's date-sliced panel
        (index <= date) and FactorRiskModel.loadings re-slices defensively,
        so loadings depend only on data on-or-before ``date``. Graceful: on
        thin/empty data the model returns ``{}`` and the gate simply passes
        (no false block). When the band is disabled this is a cheap no-op
        (no gate reads the reference), so it is skipped to keep the old path
        untouched."""
        if self.max_factor_exposure is None:
            return
        try:
            loadings = self.factor_risk_model.loadings(all_data, date)
        except Exception as exc:  # noqa: BLE001 — never crash a run
            logger.warning(
                "Citadel: factor loadings failed (%s); factor gate passes "
                "this day", exc)
            loadings = {}
        self._factor_loadings_ref['loadings'] = loadings

    def _measure_book_factor_exposure(
            self, portfolio: PortfolioManager,
            prices: Dict[str, float]) -> Dict[str, float]:
        """Aggregate book net factor exposure (fraction of desk capital).

        Marks the held book at ``prices`` (latest closes) via the risk
        book's own per-symbol netting, then dollar-weights each name by its
        current factor loadings. Returns {} when the band is disabled or
        there is no usable base/loadings (graceful)."""
        if self.max_factor_exposure is None:
            return {}
        loadings = self._factor_loadings_ref.get('loadings') or {}
        if not loadings:
            return {}
        desk_capital = (portfolio.get_portfolio_value()
                        * self.capital_allocation)
        if desk_capital <= 0:
            return {}
        per_symbol = self.risk_book.compute_exposures(
            portfolio, prices)['per_symbol']
        return self.factor_risk_model.book_exposure(
            per_symbol, loadings, desk_capital)

    # ------------------------------------------------------------------
    # Pod accounting: daily P&L attribution and NAV indexing
    # ------------------------------------------------------------------
    def _owning_pod(self, asset: Asset) -> Optional[str]:
        """Tracking first, then the sticky symbol->pod map (late fills,
        externally closed positions, cut-pod residuals)."""
        state = self._pod_positions.get(asset)
        if state is not None:
            return state['pod']
        return self._traded_pods.get(asset.symbol)

    def _update_pod_navs(self, portfolio: PortfolioManager) -> None:
        """Attribute today's position P&L per pod and roll the NAVs.

        Runs BEFORE any weight change today, so the capital base is the
        previous snapshot's portfolio value (start-of-day capital) times
        the weights recorded in yesterday's pod_history entry — exactly
        what the attribution identity reconstructs from the report.
        """
        realized: Dict[str, float] = {key: 0.0 for key in self._pods}
        new_trades = portfolio.closed_trades[self._n_closed_trades_seen:]
        self._n_closed_trades_seen = len(portfolio.closed_trades)
        for trade in new_trades:
            pod_key = self._owning_pod(trade.asset)
            if pod_key is None:
                logger.warning(
                    "Citadel attribution: closed trade in %s has no owning "
                    "pod; P&L %.2f unattributed", trade.asset.symbol,
                    trade.pnl)
                continue
            realized[pod_key] += trade.pnl

        unrealized: Dict[str, float] = {key: 0.0 for key in self._pods}
        for asset in sorted(portfolio.positions, key=lambda a: a.symbol):
            position = portfolio.positions[asset]
            if position.quantity == 0:
                continue
            pod_key = self._owning_pod(asset)
            if pod_key is None:
                logger.warning(
                    "Citadel attribution: open position in %s has no owning "
                    "pod; P&L %.2f unattributed", asset.symbol,
                    position.pnl())
                continue
            unrealized[pod_key] += position.pnl()

        if portfolio.portfolio_history:
            start_value = portfolio.portfolio_history[-1]['portfolio_value']
        else:
            start_value = portfolio.get_portfolio_value()
        start_capital = start_value * self.capital_allocation

        for pod_key in sorted(self._pods):
            pnl = (unrealized[pod_key]
                   - self._prev_unrealized.get(pod_key, 0.0)
                   + realized[pod_key])
            allocated = start_capital * self._weights[pod_key]
            if allocated > 0:
                self._last_allocated[pod_key] = allocated
            else:
                # Cut pod residual: attribute against its last positive
                # capital base until its book is flat (module docstring).
                allocated = self._last_allocated.get(pod_key, 0.0)

            if allocated > 0:
                day_return = pnl / allocated
            elif pnl != 0.0:
                day_return = 0.0
                logger.warning(
                    "Citadel attribution: pod %s has P&L %.2f but no "
                    "capital base; NAV left unchanged", pod_key, pnl)
            else:
                day_return = 0.0

            self._navs[pod_key] *= (1.0 + day_return)
            self._nav_max[pod_key] = max(self._nav_max[pod_key],
                                         self._navs[pod_key])
            if self._day_index >= 1:
                self._nav_returns[pod_key].append(day_return)

        self._prev_unrealized = unrealized

    # ------------------------------------------------------------------
    # Bookkeeping reconcile + orphan sweep (Renaissance pattern)
    # ------------------------------------------------------------------
    def _reconcile_pods(self, portfolio: PortfolioManager
                        ) -> List[Tuple[str, DeskIntent]]:
        """Drop tracking for entries that never filled or were closed
        externally (stop losses), then sweep ORPHANS: the engine holds a
        pending intent for up to 5 trading days — longer than the 2-day
        grace — so an entry can fill AFTER its tracking was dropped.
        Untracked positions in desk-traded symbols are closed immediately
        (they would otherwise escape every pod-level exit and block
        re-entry forever); attribution stays correct throughout via the
        sticky symbol->pod map."""
        for asset in sorted(self._pod_positions, key=lambda a: a.symbol):
            state = self._pod_positions[asset]
            position = portfolio.get_position(asset)
            if position is not None and position.quantity != 0:
                continue
            if self._day_index - state['entry_day'] < RECONCILE_GRACE_DAYS:
                continue  # intent may still be pending its next-open fill
            del self._pod_positions[asset]
            self.note(
                'info',
                f"Pod cleanup: dropped {asset.symbol} from the "
                f"{state['pod']} pod (entry never filled or position "
                f"closed externally)",
                pod=state['pod'], symbol=asset.symbol,
                direction=state['direction'])

        intents: List[Tuple[str, DeskIntent]] = []
        for symbol in sorted(self._traded_pods):
            asset = _stock(symbol)
            if asset in self._pod_positions:
                continue  # properly tracked
            position = portfolio.get_position(asset)
            if position is None or position.quantity == 0:
                continue
            pod_key = self._traded_pods[symbol]
            direction = 'long' if position.quantity > 0 else 'short'
            intents.append((pod_key, DeskIntent(
                asset=asset,
                action=self._close_action(direction),
                size_fraction=1.0,
                reason='orphan sweep: untracked position in a desk-traded '
                       'symbol (entry filled after its pod tracking was '
                       'reconciled away)')))
            self.note(
                'risk',
                f"Orphan sweep: closing untracked {direction} position of "
                f"{abs(position.quantity)} {symbol} — the entry filled "
                f"after its {pod_key} pod tracking was reconciled away",
                pod=pod_key, symbol=symbol, direction=direction,
                quantity=float(position.quantity))
        return intents

    def _close_action(self, direction: str) -> str:
        return 'SELL' if direction == 'long' else 'COVER'

    def _symbol_is_free(self, symbol: str,
                        portfolio: PortfolioManager) -> bool:
        """Enterable only if no pod tracks it AND the portfolio holds no
        position in it (any direction)."""
        asset = _stock(symbol)
        if asset in self._pod_positions:
            return False
        position = portfolio.get_position(asset)
        return position is None or position.quantity == 0

    # ------------------------------------------------------------------
    # Drawdown policy: probation and cuts
    # ------------------------------------------------------------------
    def _apply_drawdown_policy(self) -> List[Tuple[str, DeskIntent]]:
        """<= cut_drawdown cuts (from active OR probation); else
        <= probation_drawdown puts an active pod on probation. Boundary
        semantics are exact: equality triggers."""
        intents: List[Tuple[str, DeskIntent]] = []
        for pod_key in sorted(self._pods):
            if self._statuses[pod_key] == 'cut':
                continue
            drawdown = self._drawdown(pod_key)
            if drawdown <= self.cut_drawdown:
                intents.extend(self._cut_pod(pod_key, drawdown))
            elif self._statuses[pod_key] == 'active' \
                    and drawdown <= self.probation_drawdown:
                self._place_on_probation(pod_key, drawdown)

        if not self._all_cut_noted and all(
                status == 'cut' for status in self._statuses.values()):
            self._all_cut_noted = True
            self.note(
                'risk',
                "All pods cut — the desk is flat for the remainder of "
                "the run",
                pods=sorted(self._pods))
        return intents

    def _place_on_probation(self, pod_key: str, drawdown: float) -> None:
        old_weight = self._weights[pod_key]
        if self._has_redistribution_recipients(exclude=pod_key):
            excess = old_weight / 2.0
            self._weights[pod_key] = old_weight - excess
            self._statuses[pod_key] = 'probation'
            self._redistribute(excess, exclude=pod_key)
            self.note(
                'risk',
                f"Pod {pod_key} on PROBATION: drawdown {drawdown:.2%} <= "
                f"{self.probation_drawdown:.2%}; weight halved "
                f"{old_weight:.3f} -> {self._weights[pod_key]:.3f}, excess "
                f"redistributed pro-rata",
                pod=pod_key, drawdown_pct=drawdown * 100.0,
                old_weight=old_weight, new_weight=self._weights[pod_key])
        else:
            # No recipient pod remains (every other pod is cut): halving
            # would RETIRE live mass and break the invariant that active
            # + probation weights sum to 1.0, so the weight stays put —
            # probation degrades to a status flip with a note.
            self._statuses[pod_key] = 'probation'
            self.note(
                'risk',
                f"Pod {pod_key} on PROBATION: drawdown {drawdown:.2%} <= "
                f"{self.probation_drawdown:.2%}; weight {old_weight:.3f} "
                f"unchanged — no recipient pod remains to absorb the "
                f"usual halving",
                pod=pod_key, drawdown_pct=drawdown * 100.0,
                old_weight=old_weight, new_weight=self._weights[pod_key])

    def _has_redistribution_recipients(self, exclude: str) -> bool:
        """True if any OTHER pod could absorb redistributed weight
        (active or probation — mirrors _redistribute's recipient
        search)."""
        return any(key != exclude
                   and self._statuses[key] in ('active', 'probation')
                   for key in self._pods)

    def _cut_pod(self, pod_key: str,
                 drawdown: float) -> List[Tuple[str, DeskIntent]]:
        old_weight = self._weights[pod_key]
        self._weights[pod_key] = 0.0
        self._statuses[pod_key] = 'cut'
        self._redistribute(old_weight, exclude=pod_key)

        intents: List[Tuple[str, DeskIntent]] = []
        closed_symbols: List[str] = []
        for asset in sorted(list(self._pod_positions), key=lambda a: a.symbol):
            state = self._pod_positions[asset]
            if state['pod'] != pod_key:
                continue
            intents.append((pod_key, DeskIntent(
                asset=asset,
                action=self._close_action(state['direction']),
                size_fraction=1.0,
                reason=(f"pod cut: drawdown {drawdown:.2%} <= "
                        f"{self.cut_drawdown:.2%}"))))
            del self._pod_positions[asset]
            closed_symbols.append(asset.symbol)

        self.note(
            'risk',
            f"Pod {pod_key} CUT: drawdown {drawdown:.2%} <= "
            f"{self.cut_drawdown:.2%}; stopped permanently, weight "
            f"{old_weight:.3f} redistributed, flattening "
            f"{len(closed_symbols)} position(s)"
            + (f" ({', '.join(closed_symbols)})" if closed_symbols else ""),
            pod=pod_key, drawdown_pct=drawdown * 100.0,
            old_weight=old_weight, closed_symbols=closed_symbols)
        return intents

    def _redistribute(self, mass: float, exclude: str) -> None:
        """Return weight mass to the pool pro-rata to other ACTIVE pods'
        current weights (probation pods as a fallback). With no
        recipients the mass is retired with a log — reachable only from
        the CUT path, where it is the deliberate all-cut derisk (live
        weight 0.0); probation pre-checks recipients and never retires
        live mass."""
        if mass <= 0:
            return
        recipients = [key for key in sorted(self._pods)
                      if key != exclude and self._statuses[key] == 'active']
        if not recipients:
            recipients = [key for key in sorted(self._pods)
                          if key != exclude
                          and self._statuses[key] == 'probation']
        if not recipients:
            logger.info(
                "Citadel: weight mass %.3f from pod %s retired (no "
                "recipient pods remain)", mass, exclude)
            return
        total = sum(self._weights[key] for key in recipients)
        if total > 0:
            for key in recipients:
                self._weights[key] += mass * self._weights[key] / total
        else:
            share = mass / len(recipients)
            for key in recipients:
                self._weights[key] += share

    # ------------------------------------------------------------------
    # Scheduled reallocation
    # ------------------------------------------------------------------
    def _maybe_reallocate(self) -> None:
        """Every realloc_every_days trading days: recompute participating
        pods' weights from trailing NAV-return Sharpe/vol (module
        docstring). Cut pods are skipped (weight stays 0); unrecovered
        probation pods and pods with fewer than min_nav_days NAV returns
        are HELD at their current weight and excluded from the recompute,
        which then targets the remaining mass."""
        if self._days_since_realloc < self.realloc_every_days:
            return
        self._days_since_realloc = 0

        non_cut = [key for key in sorted(self._pods)
                   if self._statuses[key] != 'cut']
        if not non_cut:
            return

        old_weights = {key: float(self._weights[key])
                       for key in sorted(self._pods)}
        held: Dict[str, float] = {}
        recovered: List[str] = []
        participating: List[str] = []
        for pod_key in non_cut:
            if self._statuses[pod_key] == 'probation':
                drawdown = self._drawdown(pod_key)
                if drawdown > self.recovery_drawdown:
                    # Strictly above the recovery threshold: back to
                    # normal eligibility (boundary: equality stays held).
                    self._statuses[pod_key] = 'active'
                    recovered.append(pod_key)
                else:
                    held[pod_key] = self._weights[pod_key]
                    continue
            returns = self._nav_returns[pod_key][-self.sharpe_window_days:]
            if len(returns) < self.min_nav_days:
                held[pod_key] = self._weights[pod_key]
                continue
            participating.append(pod_key)

        target = max(0.0, 1.0 - sum(held.values()))
        cap_infeasible = bool(participating) and \
            target - len(participating) * self.weight_max > 1e-12
        if cap_infeasible:
            # The unheld mass cannot fit under the HARD weight_max cap
            # (e.g. a lone survivor carrying weight 1.0 after cuts).
            # Recomputing would deliberately under-allocate
            # (clamp_renormalize's hard-cap direction) and silently
            # shrink desk deployment, so the participants are HELD at
            # their current weights instead: active + probation weight
            # stays exactly 1.0 (weight conservation; module docstring).
            for pod_key in participating:
                held[pod_key] = self._weights[pod_key]
            participating = []

        inputs: Dict[str, Dict] = {}
        if participating:
            raw: Dict[str, float] = {}
            for pod_key in participating:
                returns = self._nav_returns[pod_key][
                    -self.sharpe_window_days:]
                info = pod_score_inputs(returns, vol_floor=self.vol_floor,
                                        score_floor=self.score_floor,
                                        robust=self.robust_pod_sharpe)
                inputs[pod_key] = info
                raw[pod_key] = info['raw']
            new_weights = clamp_renormalize(raw, target, self.weight_min,
                                            self.weight_max)
            self._weights.update(new_weights)

        allocations = {key: float(self._weights[key])
                       for key in sorted(self._pods)}
        reason = (f"scheduled performance-weighted reallocation (every "
                  f"{self.realloc_every_days} trading days, trailing "
                  f"{self.sharpe_window_days}d Sharpe / vol)")
        changes = ", ".join(
            f"{key} {old_weights[key]:.3f}->{allocations[key]:.3f}"
            for key in sorted(self._pods))
        self.note(
            'allocation',
            f"Reallocation: {changes}"
            + (f"; held {', '.join(sorted(held))}" if held else "")
            + (f"; recovered {', '.join(recovered)}" if recovered else "")
            + ("; participants held at current weights — the unheld mass "
               "exceeds the weight_max cap" if cap_infeasible else ""),
            allocations=allocations, reason=reason, old_weights=old_weights,
            inputs=inputs, held=sorted(held), recovered=recovered,
            cap_infeasible=cap_infeasible)

    # ------------------------------------------------------------------
    # Pod signals -> conflict resolution -> sizing
    # ------------------------------------------------------------------
    def _collect_pod_intents(self, all_data: Dict[str, pd.DataFrame], date,
                             portfolio: PortfolioManager
                             ) -> List[Tuple[str, DeskIntent]]:
        """Run non-cut pods in first-claim order (current weight desc,
        then pod key): closes act only on the signaling pod's own
        positions; entries are filtered through ownership + same-day
        claims, then sized as pod_weight * min(0.25, 1/max(4, n)) where
        n is the pod's actionable entry count today."""
        order = sorted((key for key in self._pods
                        if self._statuses[key] != 'cut'),
                       key=lambda key: (-self._weights[key], key))
        claimed_today: Dict[str, str] = {}
        tagged: List[Tuple[str, DeskIntent]] = []

        for pod_key in order:
            pod = self._pods[pod_key]
            signals = pod.generate_signals(all_data, date)
            self._note_new_fits(pod)
            weight = self._weights[pod_key]

            entries: List[Tuple[str, str, str]] = []
            for symbol in sorted(signals):
                action = signals[symbol]
                if action not in POD_ACTIONS:
                    logger.warning(
                        "Pod %s emitted invalid action %r for %s; dropped",
                        pod_key, action, symbol)
                    continue
                asset = _stock(symbol)

                if action in ('SELL', 'COVER'):
                    state = self._pod_positions.get(asset)
                    if state is None or state['pod'] != pod_key:
                        continue  # only the owner manages its position
                    want = 'long' if action == 'SELL' else 'short'
                    if state['direction'] != want:
                        continue
                    tagged.append((pod_key, DeskIntent(
                        asset=asset, action=action, size_fraction=1.0,
                        reason=f"{pod_key} pod exit signal")))
                    del self._pod_positions[asset]
                    self.note(
                        'signal',
                        f"{pod.name} exit {symbol}: closing the "
                        f"{state['direction']} position",
                        pod=pod_key, symbol=symbol,
                        direction=state['direction'])
                    continue

                # Opening signal (BUY/SHORT)
                if symbol in claimed_today:
                    self.note(
                        'info',
                        f"Conflict: {pod_key} pod's {action} {symbol} "
                        f"dropped — first claim went to the "
                        f"{claimed_today[symbol]} pod (higher current "
                        f"weight wins, ties alphabetical)",
                        pod=pod_key, symbol=symbol, action=action,
                        winner=claimed_today[symbol])
                    continue
                if not self._symbol_is_free(symbol, portfolio):
                    continue
                direction = 'long' if action == 'BUY' else 'short'
                entries.append((symbol, action, direction))

            if not entries or weight <= 0:
                continue
            n_signals = len(entries)
            size_fraction = weight * min(0.25, 1.0 / max(4, n_signals))
            if size_fraction <= 0:
                continue
            for symbol, action, direction in entries:
                asset = _stock(symbol)
                claimed_today[symbol] = pod_key
                tagged.append((pod_key, DeskIntent(
                    asset=asset, action=action, size_fraction=size_fraction,
                    reason=(f"{pod_key} pod {direction} entry; "
                            f"{size_fraction:.2%} of desk capital"))))
                self._pod_positions[asset] = {'pod': pod_key,
                                              'direction': direction,
                                              'entry_day': self._day_index}
                self._traded_pods[symbol] = pod_key
                self.note(
                    'signal',
                    f"{pod.name} {action} {symbol}: size "
                    f"{size_fraction:.2%} of desk capital "
                    f"(weight {weight:.3f}, {n_signals} signal(s) today)",
                    pod=pod_key, symbol=symbol, direction=direction,
                    size_fraction=size_fraction, n_signals=n_signals)
        return tagged

    def _note_new_fits(self, pod: Pod) -> None:
        """Note walk-forward refits a pod's controller performed today."""
        if pod.controller is None:
            return
        n_fits = len(pod.controller.fits)
        if n_fits > self._fit_counts.get(pod.key, 0):
            fit = pod.controller.fits[-1]
            self.note(
                'model',
                f"{pod.name} model refit #{n_fits}: trained on "
                f"{fit.n_samples} samples ({fit.train_start} .. "
                f"{fit.train_end})",
                pod=pod.key, model=pod.key, **fit.to_dict())
            self._fit_counts[pod.key] = n_fits

    # ------------------------------------------------------------------
    # Central risk book
    # ------------------------------------------------------------------
    @staticmethod
    def _latest_prices(all_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Latest-close mark per symbol (the risk book's preferred mark)."""
        prices: Dict[str, float] = {}
        for symbol, frame in all_data.items():
            if frame is None or frame.empty or 'close' not in frame.columns:
                continue
            close = frame['close'].iloc[-1]
            if pd.notna(close):
                prices[symbol] = float(close)
        return prices

    def _apply_risk_book(self, tagged: List[Tuple[str, DeskIntent]],
                         portfolio: PortfolioManager,
                         all_data: Dict[str, pd.DataFrame]
                         ) -> List[DeskIntent]:
        """Deterministic order (closes first, then pod weight desc, then
        symbol), cumulative evaluation; every block is noted with
        data.pod and the blocked entry's tracking is rolled back."""
        if not tagged:
            return []

        def sort_key(item: Tuple[str, DeskIntent]):
            pod_key, intent = item
            closing = 0 if intent.action in ('SELL', 'COVER') else 1
            return (closing, -self._weights.get(pod_key, 0.0),
                    intent.asset.symbol)

        ordered = sorted(tagged, key=sort_key)
        prices = self._latest_prices(all_data)

        pod_by_intent = {id(intent): pod_key for pod_key, intent in ordered}
        approved, blocked = self.risk_book.check(
            [intent for _, intent in ordered], portfolio, prices)

        for intent, reason in blocked:
            pod_key = pod_by_intent[id(intent)]
            self.note(
                'risk',
                f"Central risk book blocked {intent.action} "
                f"{intent.asset.symbol} ({pod_key} pod): {reason}",
                pod=pod_key, symbol=intent.asset.symbol,
                action=intent.action, reason=reason)
            if intent.action in OPENING_ACTIONS:
                state = self._pod_positions.get(intent.asset)
                if state is not None and state['pod'] == pod_key \
                        and state['entry_day'] == self._day_index:
                    del self._pod_positions[intent.asset]
        return approved

    def _note_book_factor_exposure(self, portfolio: PortfolioManager,
                                   prices: Dict[str, float]) -> None:
        """Measure the aggregate book net factor exposure and surface it.

        Stores the latest reading for get_status (the AQR-style operator
        transparency: the book's net loading per factor, as a fraction of
        desk capital) and emits one 'info' note per day (deliberately NOT
        'allocation', to keep the C9 reallocation-note count exact — see the
        inline comment below) when the book has a measurable factor exposure.
        Called once per simulated day from generate_intents (NOT only on
        intent days), so held-book drift is surfaced even when the pods are
        quiet. Graceful/quiet when the band is disabled or there is no usable
        exposure (no note, no churn on empty/thin days)."""
        exposure = self._measure_book_factor_exposure(portfolio, prices)
        self._book_factor_exposure = exposure
        if not exposure:
            return
        if not any(v != 0.0 for v in exposure.values()):
            return
        worst_factor = max(exposure, key=lambda f: abs(exposure[f]))
        summary = ", ".join(f"{f} {exposure[f]:+.2%}"
                            for f in sorted(exposure))
        # 'info' (not 'allocation'): kept clear of the C9 reallocation
        # contract so counting 'allocation' notes stays exact; this is the
        # AQR-style factor-exposure transparency note carrying
        # data.factor_exposure + data.band.
        self.note(
            'info',
            f"Book factor exposure (net, fraction of desk capital): "
            f"{summary}; largest |{worst_factor}| "
            f"{abs(exposure[worst_factor]):.2%} vs the "
            f"+/-{self.max_factor_exposure:.2%} neutrality band",
            factor_exposure={f: float(v) for f, v in exposure.items()},
            band=self.max_factor_exposure, worst_factor=worst_factor)

    # ------------------------------------------------------------------
    # C8 history
    # ------------------------------------------------------------------
    def _record_pod_history(self, date) -> None:
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        entry = {'date': date_str, 'pods': self._pod_snapshot()}
        if self._pod_history and self._pod_history[-1]['date'] == date_str:
            self._pod_history[-1] = entry  # idempotent per simulated day
        else:
            self._pod_history.append(entry)
