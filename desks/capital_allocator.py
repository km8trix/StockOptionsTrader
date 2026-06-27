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
    target gross <= 1.0, with an equal-weight fallback.

    Also offers an OPT-IN, overfitting-guarded PERFORMANCE weighting
    (performance_weights) that tilts toward desks with better realized
    risk-adjusted out-of-sample performance, and an OPT-IN FULL-COVARIANCE risk
    parity (risk_parity_cov_weights) that equalizes risk contributions over the
    full covariance matrix instead of the inverse-vol diagonal approximation.
    Both opt-in modes degrade conservatively to the inverse-vol path, which
    itself falls back to equal weight — so the risk-parity default path is
    byte-identical and the new modes are purely additive.
    """

    def __init__(self, target_gross: float = 1.0,
                 weight_min: float = 0.05, weight_max: float = 0.60,
                 shrinkage: float = 0.5):
        if not (0.0 < target_gross <= 1.0 + _TARGET_TOL):
            raise ValueError(
                f"target_gross {target_gross} must be in (0, 1]")
        if not (0.0 <= weight_min <= weight_max):
            raise ValueError(
                f"require 0 <= weight_min ({weight_min}) <= weight_max "
                f"({weight_max})")
        if not (0.0 <= shrinkage <= 1.0):
            raise ValueError(
                f"shrinkage {shrinkage} must be in [0, 1]")
        self.target_gross = min(target_gross, 1.0)
        # Guards for performance_weights ONLY (risk_parity_weights ignores them).
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.shrinkage = float(shrinkage)

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

    def risk_parity_cov_weights(self,
                                returns_by_desk: Dict[str, Sequence[float]]
                                ) -> Dict[str, float]:
        """OPT-IN FULL-COVARIANCE risk parity, scaled to sum to target_gross.

        risk_parity_weights is inverse-vol only — a DIAGONAL-covariance
        approximation that ignores cross-desk correlation, so two correlated
        desks together carry more combined risk than intended. This mode does
        TRUE risk parity over the full covariance matrix: it equalizes each
        desk's RISK CONTRIBUTION rc_i = w_i * (Cov @ w)_i / port_vol, so a
        correlated pair is jointly down-weighted relative to inverse-vol.

        Method (numpy only — scipy/sklearn forbidden): align the per-desk return
        series on their overlapping window (equal length, finite rows only),
        compute the sample covariance, seed w from inverse-vol
        (risk_parity_weights), then run ~5 fixed-point iterations on the marginal
        risk contributions:
            port_vol = sqrt(w @ cov @ w)
            mrc      = (cov @ w) / port_vol
            rc       = w * mrc
            w       *= ((port_vol / n) / (rc + 1e-12)) ** 0.5   # damped
            w        = clip(w, 0, None); w /= w.sum()
        then rescale the converged simplex weights to target_gross (no per-desk
        band — unlike performance_weights, this mode has no weight_min/max).

        DAMPING (the ** 0.5): the raw undamped ratio (power 1) OVERSHOOTS and
        oscillates/diverges on correlated desks — it 2-cycles on a symmetric
        covariance and can drive a single desk to ~100% on noisy samples, the
        opposite of risk parity. The square-root step is the standard stable
        form of this fixed point; it converges to EQUAL risk contributions in a
        handful of iterations (RC spread < 1e-2 by iter 5 across correlated and
        uncorrelated mixes), which is the whole point of the mode.

        DEGRADE conservatively, exactly like the rest of the allocator: fall back
        to risk_parity_weights (inverse-vol) — which itself falls back to
        _equal_weight — when the overlap is too short, any desk's vol is
        degenerate (the SAME degenerate_desks gate), or the covariance is
        singular / non-finite. The whole computation is wrapped in try/except so
        any numerical failure degrades rather than raises.
        """
        keys = list(returns_by_desk.keys())
        if not keys:
            return {}
        n = len(keys)
        # Single desk: nothing to de-correlate; inverse-vol == full-cov.
        if n == 1:
            return self.risk_parity_weights(returns_by_desk)
        # SAME degeneracy gate as risk_parity_weights: any too-few / zero-vol
        # desk -> inverse-vol (which then equal-weights the whole fund). This
        # gate logs its OWN reason via risk_parity_weights, so it is not routed
        # through _degrade below (which is for the cov-specific numerical paths).
        if self.degenerate_desks(returns_by_desk):
            return self.risk_parity_weights(returns_by_desk)

        def _degrade(reason: str) -> Dict[str, float]:
            # The cov-specific degrades (short overlap, singular/non-finite cov,
            # non-finite iterate) are NOT caught by the degenerate-desk gate, so
            # without this they would silently return inverse-vol weights that
            # the reweighter then logs as fallback=False — indistinguishable from
            # a genuine full-cov result. Log WHY the mode degraded so the audit
            # trail is honest. INFO, not WARNING: an expected, recoverable
            # fallback (e.g. early in a backtest before enough overlap accrues).
            logger.info(
                "Full-covariance risk parity degraded to inverse-vol (%s)",
                reason)
            return self.risk_parity_weights(returns_by_desk)

        try:
            # Align on the overlapping window: equal length across desks (take
            # the trailing min-length tail), then drop any row where a desk is
            # non-finite (one ragged series must not silently shorten only some
            # columns).
            series = [np.asarray(list(returns_by_desk[k]), dtype=float)
                      for k in keys]
            length = min(s.size for s in series)
            if length < n + 1:
                # Too few overlapping rows for a usable n-desk covariance.
                return _degrade(
                    f'overlap of {length} rows < {n + 1} needed for {n} desks')
            matrix = np.vstack([s[-length:] for s in series])  # n x length
            matrix = matrix[:, np.all(np.isfinite(matrix), axis=0)]
            if matrix.shape[1] < n + 1:
                return _degrade(
                    f'only {matrix.shape[1]} finite rows < {n + 1} needed')
            cov = np.cov(matrix)
            if not np.all(np.isfinite(cov)):
                return _degrade('non-finite covariance')

            # Seed w from inverse-vol, work on the simplex (sum 1) during
            # iteration, rescale to target_gross at the end.
            seed = self.risk_parity_weights(returns_by_desk)
            w = np.array([seed[k] for k in keys], dtype=float)
            w = w / w.sum()

            # A singular/ill-conditioned cov can drive port_var or rc negative
            # mid-iteration (transient nan/inf); the guards below catch it and
            # degrade, so silence the expected numpy warnings on that edge.
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                for _ in range(5):
                    port_var = float(w @ cov @ w)
                    if not np.isfinite(port_var) or port_var <= 0.0:
                        return _degrade(
                            f'non-positive portfolio variance ({port_var:g})')
                    port_vol = np.sqrt(port_var)
                    mrc = (cov @ w) / port_vol
                    rc = w * mrc
                    # Damped (sqrt) step: the undamped ratio oscillates/diverges
                    # on correlated desks; sqrt is the standard convergent form.
                    w = w * ((port_vol / n) / (rc + 1e-12)) ** 0.5
                    w = np.clip(w, 0.0, None)
                    total = w.sum()
                    if not np.isfinite(total) or total <= 0.0:
                        # A negative marginal contribution (e.g. a
                        # negatively-correlated hedge desk) makes the damped
                        # step non-finite; risk parity is ill-defined there.
                        return _degrade(
                            'non-finite weights mid-iteration '
                            '(e.g. negatively-correlated desk)')
                    w = w / total

            if not np.all(np.isfinite(w)) or w.sum() <= 0.0:
                return _degrade('non-finite converged weights')
            scaled = w * (self.target_gross / w.sum())
            return {key: float(scaled[i]) for i, key in enumerate(keys)}
        except Exception as exc:  # singular cov / numerical blow-up -> degrade
            logger.warning(
                "Full-covariance risk parity failed (%s); falling back to "
                "inverse-vol risk parity", exc)
            return self.risk_parity_weights(returns_by_desk)

    def performance_weights(self,
                            returns_by_desk: Dict[str, Sequence[float]]
                            ) -> Dict[str, float]:
        """OPT-IN performance weighting: tilt capital toward desks with better
        REALIZED RISK-ADJUSTED out-of-sample performance, scaled to sum to
        target_gross, with overfitting guards on by default.

        The returns are the SAME walk-forward-honest, OOS solo returns
        risk_parity_weights consumes (the caller slices the curve to the
        rebalance date), so no look-ahead is introduced here.

        SCORE: a Sharpe-like score per desk = mean(returns) / std(returns)
        (risk-ADJUSTED, not raw return, so a desk is not rewarded merely for
        carrying more risk). The degeneracy gate is identical to
        risk_parity_weights: degenerate_desks() flags any desk with too few
        finite returns or zero/non-finite vol, and ANY such desk forces the
        conservative WHOLE-FUND equal-weight fallback.

        NON-NEGATIVE: each score is clipped at 0 — a negative-Sharpe (drawing
        down) desk is floored, never shorted or assigned a negative weight. If
        EVERY score clips to 0 (no desk shows positive risk-adjusted edge), the
        whole fund falls back to equal weight.

        SHRINKAGE: the performance tilt is blended toward the equal-weight prior
        by ``shrinkage`` in [0, 1]: raw = (1 - shrinkage) * equal + shrinkage *
        perf. shrinkage=0 -> pure equal-weight prior (ignore performance),
        shrinkage=1 -> pure performance tilt. This tempers noisy short-window
        scores.

        BOUNDS: each desk weight is clipped to [weight_min, weight_max] (as a
        FRACTION of target_gross) then renormalized so the bounded weights sum
        EXACTLY to target_gross (clip-then-renormalize). When the bounds cannot
        all be satisfied simultaneously (e.g. weight_min * n > 1 or weight_max *
        n < 1) the fund falls back to equal weight rather than emit a set that
        violates a bound or the sum.

        Deterministic; same fallback and sum<=target_gross guarantees as
        risk_parity_weights.
        """
        keys = list(returns_by_desk.keys())
        if not keys:
            return {}
        n = len(keys)

        # Same degeneracy discipline as risk_parity_weights: any too-few /
        # zero-vol desk -> whole-fund equal weight.
        degenerate = self.degenerate_desks(returns_by_desk)
        if degenerate:
            logger.warning(
                "Performance allocation fell back to equal weight "
                "(degenerate desks): %s", degenerate)
            return self._equal_weight(keys)

        # If the bounds are jointly infeasible across n desks, no bounded set
        # can sum to 1 — fall back to equal weight (which itself may breach a
        # bound, but equal weight is the documented degenerate answer and keeps
        # the sum exact, vs. silently emitting an out-of-bound performance set).
        if self.weight_min * n > 1.0 + _TARGET_TOL \
                or self.weight_max * n < 1.0 - _TARGET_TOL:
            logger.warning(
                "Performance weight bounds [%.4g, %.4g] are infeasible for %d "
                "desks; falling back to equal weight",
                self.weight_min, self.weight_max, n)
            return self._equal_weight(keys)

        # Sharpe-like score per desk; non-negative clip (never short/negative).
        scores: Dict[str, float] = {}
        for key in keys:
            arr = np.asarray(list(returns_by_desk[key]), dtype=float)
            arr = arr[np.isfinite(arr)]
            vol = float(arr.std())
            sharpe = float(arr.mean()) / vol  # vol > 0: degeneracy gate passed
            scores[key] = max(0.0, sharpe)

        total_score = sum(scores.values())
        if not np.isfinite(total_score) or total_score <= 0.0:
            # No desk shows positive risk-adjusted edge -> equal weight.
            logger.warning(
                "Performance scores all clip to 0 (no positive risk-adjusted "
                "edge); falling back to equal weight")
            return self._equal_weight(keys)

        # Performance fractions of target_gross (sum to 1 before shrinkage).
        perf = {key: scores[key] / total_score for key in keys}
        equal = 1.0 / n
        # SHRINKAGE toward the equal-weight prior (sum stays 1).
        blended = {key: (1.0 - self.shrinkage) * equal
                   + self.shrinkage * perf[key]
                   for key in keys}

        bounded = self._bounded_renormalize(blended)
        weights = {key: self.target_gross * bounded[key] for key in keys}
        # Post-condition (defence-in-depth): the bounded fractions sum to 1, so
        # the weights sum to target_gross. If a numerical edge ever breaches it
        # the desk must NOT over-leverage — fall back to the documented
        # equal-weight degenerate answer rather than emit an out-of-budget set.
        if abs(sum(weights.values()) - self.target_gross) > _TARGET_TOL:
            logger.error(
                "Performance weights summed to %.6f != target_gross %.4g; "
                "falling back to equal weight",
                sum(weights.values()), self.target_gross)
            return self._equal_weight(keys)
        return weights

    def _bounded_renormalize(self, fractions: Dict[str, float]
                             ) -> Dict[str, float]:
        """Project ``fractions`` onto {sum == 1, weight_min <= x <= weight_max}
        by FILLING the budget.

        Start every desk at the floor ``weight_min``, then distribute the
        remaining budget (1 - weight_min*n) across desks in proportion to their
        fraction, capping any desk that would exceed ``weight_max`` and
        REDISTRIBUTING the capped overflow among the still-free desks, iterating
        until the budget is placed. When the free desks have no positive
        fraction left (e.g. a zero-score desk under full shrinkage), the
        residual is split EQUALLY among them so the full target is deployed
        rather than left idle — and never beyond a cap.

        This fills to sum == 1 exactly without the over-leverage of naive
        clip-then-pin (which could pin one desk at its cap while flooring
        several others, summing past 1 and over-leveraging the book — the bug
        this replaces). Feasible by the caller's weight_min*n <= 1 <=
        weight_max*n gate. Deterministic (tie-broken by sorted key).
        """
        keys = list(fractions.keys())
        n = len(keys)
        lo, hi = self.weight_min, self.weight_max
        w = {k: max(0.0, float(fractions[k])) for k in keys}

        result = {k: lo for k in keys}
        budget = 1.0 - lo * n            # >= 0 by the feasibility gate
        free = list(keys)

        # At most n redistribution passes: each pass caps >= 1 desk or ends.
        for _ in range(n + 1):
            if budget <= _TARGET_TOL or not free:
                break
            free_w = sum(w[k] for k in free)
            if free_w > 0.0:
                add = {k: budget * w[k] / free_w for k in free}
            else:
                # No score signal left among the free desks -> split equally so
                # the budget is deployed rather than left idle.
                share = budget / len(free)
                add = {k: share for k in free}
            newly_capped = [k for k in free
                            if result[k] + add[k] > hi + _TARGET_TOL]
            if not newly_capped:
                for k in free:
                    result[k] += add[k]
                budget = 0.0
                break
            # Cap them at hi, consume only the room they had, redistribute the
            # rest among the still-free desks on the next pass.
            for k in sorted(newly_capped):
                budget -= (hi - result[k])
                result[k] = hi
                free.remove(k)
        return result

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
