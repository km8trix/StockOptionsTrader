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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from desks.base import Desk

logger = logging.getLogger(__name__)

#: Float tolerance on the target_gross (<= 1.0) bound.
_TARGET_TOL = 1e-9


def set_desk_capital_allocation(desk: Desk, weight: float) -> None:
    """Set a desk weight and keep its internal risk-book base in sync.

    Citadel and Jane Street copy their construction-time allocation into a
    ``CentralRiskBook``. Fund reweighting changes the desk allocation later,
    so mutating only the outer desk leaves risk checks using stale capital.
    Desks without a risk book retain the original single assignment.
    """
    desk.capital_allocation = weight
    risk_book = getattr(desk, 'risk_book', None)
    if risk_book is not None and hasattr(risk_book, 'capital_allocation'):
        risk_book.capital_allocation = weight

#: Reciprocal-machine-epsilon ceiling on a covariance's condition number, above
#: which it is treated as singular/ill-conditioned. np.linalg.solve may EITHER
#: raise LinAlgError OR (depending on the LAPACK build) silently return finite
#: garbage on a singular matrix, so the optimizer modes guard on this SVD-based
#: condition number — deterministic across platforms — rather than relying on
#: solve raising. ~4.5e15 for float64.
_MAX_COND = 1.0 / np.finfo(float).eps


class CrossDeskCapitalAllocator:
    """Risk-parity (inverse-vol) capital weights across desks, summing to a
    target gross <= 1.0, with an equal-weight fallback.

    Also offers an OPT-IN, overfitting-guarded PERFORMANCE weighting
    (performance_weights) that tilts toward desks with better realized
    risk-adjusted out-of-sample performance, an OPT-IN FULL-COVARIANCE risk
    parity (risk_parity_cov_weights) that equalizes risk contributions over the
    full covariance matrix instead of the inverse-vol diagonal approximation,
    and two OPT-IN portfolio-optimizer modes — MAX DIVERSIFICATION
    (max_diversification_weights, w ∝ Σ⁻¹σ) and MEAN-VARIANCE
    (mean_variance_weights, long-only max-Sharpe w ∝ Σ⁻¹μ). Every opt-in mode
    degrades conservatively to the inverse-vol path, which itself falls back to
    equal weight — so the risk-parity default path is byte-identical and the new
    modes are purely additive.
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
        return self.risk_parity_cov_weights_with_status(returns_by_desk)[0]

    def risk_parity_cov_weights_with_status(
            self, returns_by_desk: Dict[str, Sequence[float]]
            ) -> Tuple[Dict[str, float], Optional[str]]:
        """Same as risk_parity_cov_weights, but also returns WHY it degraded.

        Returns ``(weights, reason)``: ``reason`` is None when full-covariance
        risk parity was actually computed, else a short string naming the
        numerical fallback to inverse-vol (short overlap, singular/non-finite
        covariance, or a non-finite iterate — e.g. a negatively-correlated hedge
        desk). DynamicReweighter uses this to record an HONEST fallback flag in
        its audit log. The degenerate-desk gate is NOT reported here (reason
        stays None) because the reweighter already captures it separately via
        degenerate_desks().
        """
        keys = list(returns_by_desk.keys())
        if not keys:
            return {}, None
        n = len(keys)
        # Single desk: nothing to de-correlate; inverse-vol == full-cov.
        if n == 1:
            return self.risk_parity_weights(returns_by_desk), None
        # SAME degeneracy gate as risk_parity_weights: any too-few / zero-vol
        # desk -> inverse-vol (which then equal-weights the whole fund). This
        # gate logs its OWN reason via risk_parity_weights and the reweighter
        # captures it via degenerate_desks(), so reason stays None here (this is
        # a degenerate-desk fallback, not a cov-specific numerical degrade).
        if self.degenerate_desks(returns_by_desk):
            return self.risk_parity_weights(returns_by_desk), None

        def _degrade(reason: str) -> Tuple[Dict[str, float], str]:
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
            return self.risk_parity_weights(returns_by_desk), reason

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
            return {key: float(scaled[i]) for i, key in enumerate(keys)}, None
        except Exception as exc:  # singular cov / numerical blow-up -> degrade
            logger.warning(
                "Full-covariance risk parity failed (%s); falling back to "
                "inverse-vol risk parity", exc)
            return (self.risk_parity_weights(returns_by_desk),
                    'covariance computation failed')

    def _aligned_matrix(self, returns_by_desk: Dict[str, Sequence[float]],
                        keys: Sequence[str]
                        ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Align the per-desk return series on their overlapping window into an
        ``n x m`` matrix (rows = desks in ``keys`` order, finite columns only),
        the shared input both portfolio-optimizer modes build a covariance from.

        Mirrors the alignment risk_parity_cov_weights_with_status does inline:
        take the trailing min-length tail across desks, then drop any column
        where a desk is non-finite (one ragged series must not silently shorten
        only some rows). Returns ``(matrix, None)`` on success, or ``(None,
        reason)`` when the overlap is too short for an ``n``-desk covariance.
        """
        n = len(keys)
        series = [np.asarray(list(returns_by_desk[k]), dtype=float)
                  for k in keys]
        length = min(s.size for s in series)
        if length < n + 1:
            return None, (
                f'overlap of {length} rows < {n + 1} needed for {n} desks')
        matrix = np.vstack([s[-length:] for s in series])  # n x length
        matrix = matrix[:, np.all(np.isfinite(matrix), axis=0)]
        if matrix.shape[1] < n + 1:
            return None, f'only {matrix.shape[1]} finite rows < {n + 1} needed'
        return matrix, None

    def max_diversification_weights(self,
                                    returns_by_desk: Dict[str, Sequence[float]]
                                    ) -> Dict[str, float]:
        """OPT-IN MAX-DIVERSIFICATION weighting, scaled to sum to target_gross.

        Maximizes the diversification ratio ``DR(w) = (wᵀσ) / sqrt(wᵀΣw)`` — the
        ratio of the weighted-average desk vol to the realized portfolio vol — so
        a desk that is REDUNDANT (highly correlated with the rest) earns LESS
        weight than its standalone vol alone would justify under inverse-vol,
        which ignores correlation. The argmax has the closed form
        ``w ∝ Σ⁻¹σ`` (Choueifaty & Coignard's Most-Diversified Portfolio), made
        long-only by clipping negatives to 0 and renormalizing to target_gross.

        Method (numpy only — scipy/sklearn forbidden): align the series on the
        overlapping window, build the sample covariance Σ and per-desk vol vector
        σ = sqrt(diag Σ), solve ``Σ w = σ``, clip w at 0, rescale to
        target_gross. DEGRADE conservatively (same discipline as the rest of the
        allocator) to risk_parity_weights (inverse-vol) — which itself falls back
        to _equal_weight — on a too-short overlap, any degenerate-vol desk (the
        SAME degenerate_desks gate), a singular/non-finite covariance, or a
        long-only solution that clips away to nothing.
        """
        return self.max_diversification_weights_with_status(returns_by_desk)[0]

    def max_diversification_weights_with_status(
            self, returns_by_desk: Dict[str, Sequence[float]]
            ) -> Tuple[Dict[str, float], Optional[str]]:
        """Same as max_diversification_weights, but also returns WHY it degraded.

        Returns ``(weights, reason)``: ``reason`` is None when the diversification
        optimizer actually ran, else a short string naming the numerical fallback
        to inverse-vol. As with risk_parity_cov, the degenerate-desk gate keeps
        ``reason`` None (the reweighter captures it separately via
        degenerate_desks()); only cov-specific numerical degrades name a reason.
        """
        keys = list(returns_by_desk.keys())
        if not keys:
            return {}, None
        n = len(keys)
        if n == 1:
            return self.risk_parity_weights(returns_by_desk), None
        if self.degenerate_desks(returns_by_desk):
            return self.risk_parity_weights(returns_by_desk), None

        def _degrade(reason: str) -> Tuple[Dict[str, float], str]:
            logger.info(
                "Max-diversification weighting degraded to inverse-vol (%s)",
                reason)
            return self.risk_parity_weights(returns_by_desk), reason

        try:
            matrix, reason = self._aligned_matrix(returns_by_desk, keys)
            if matrix is None:
                return _degrade(reason)
            cov = np.cov(matrix)
            if not np.all(np.isfinite(cov)):
                return _degrade('non-finite covariance')
            cond = np.linalg.cond(cov)
            if not np.isfinite(cond) or cond > _MAX_COND:
                return _degrade('singular/ill-conditioned covariance')
            sigma = np.sqrt(np.diag(cov))
            if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
                return _degrade('degenerate covariance diagonal')
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                raw = np.linalg.solve(cov, sigma)   # w ∝ Σ⁻¹σ (MDP)
            if not np.all(np.isfinite(raw)):
                return _degrade('non-finite solution')
            w = np.clip(raw, 0.0, None)             # long-only
            total = w.sum()
            if not np.isfinite(total) or total <= 0.0:
                return _degrade('non-positive long-only weights')
            scaled = w * (self.target_gross / total)
            return {key: float(scaled[i]) for i, key in enumerate(keys)}, None
        except np.linalg.LinAlgError as exc:    # singular covariance
            return _degrade(f'singular covariance ({exc})')
        except Exception as exc:                # any other numerical blow-up
            logger.warning(
                "Max-diversification weighting failed (%s); falling back to "
                "inverse-vol risk parity", exc)
            return (self.risk_parity_weights(returns_by_desk),
                    'covariance computation failed')

    def mean_variance_weights(self,
                              returns_by_desk: Dict[str, Sequence[float]]
                              ) -> Dict[str, float]:
        """OPT-IN MEAN-VARIANCE weighting, scaled to sum to target_gross.

        Long-only max-Sharpe tangency portfolio: ``w ∝ Σ⁻¹μ`` with μ the per-desk
        MEAN return over the aligned window and Σ the sample covariance. This
        tilts capital toward desks with a better risk-adjusted mean (jointly,
        accounting for correlation), clips negatives to 0 (long-only — a desk
        with no positive marginal edge is dropped, never shorted), and rescales
        to target_gross.

        The returns are the SAME walk-forward-honest OOS solo returns the rest of
        the allocator consumes, so no look-ahead is introduced. Method is numpy
        only (scipy/sklearn forbidden). DEGRADE conservatively to
        risk_parity_weights (inverse-vol) — which itself falls back to
        _equal_weight — on a too-short overlap, any degenerate-vol desk (the SAME
        degenerate_desks gate), a singular/non-finite covariance, or when no desk
        has a positive risk-adjusted mean (every weight clips to 0).
        """
        return self.mean_variance_weights_with_status(returns_by_desk)[0]

    def mean_variance_weights_with_status(
            self, returns_by_desk: Dict[str, Sequence[float]]
            ) -> Tuple[Dict[str, float], Optional[str]]:
        """Same as mean_variance_weights, but also returns WHY it degraded.

        Returns ``(weights, reason)``: ``reason`` is None when the mean-variance
        optimizer actually ran, else a short string naming the numerical fallback
        to inverse-vol. As with risk_parity_cov, the degenerate-desk gate keeps
        ``reason`` None (captured separately via degenerate_desks()).
        """
        keys = list(returns_by_desk.keys())
        if not keys:
            return {}, None
        n = len(keys)
        if n == 1:
            return self.risk_parity_weights(returns_by_desk), None
        if self.degenerate_desks(returns_by_desk):
            return self.risk_parity_weights(returns_by_desk), None

        def _degrade(reason: str) -> Tuple[Dict[str, float], str]:
            logger.info(
                "Mean-variance weighting degraded to inverse-vol (%s)", reason)
            return self.risk_parity_weights(returns_by_desk), reason

        try:
            matrix, reason = self._aligned_matrix(returns_by_desk, keys)
            if matrix is None:
                return _degrade(reason)
            cov = np.cov(matrix)
            if not np.all(np.isfinite(cov)):
                return _degrade('non-finite covariance')
            cond = np.linalg.cond(cov)
            if not np.isfinite(cond) or cond > _MAX_COND:
                return _degrade('singular/ill-conditioned covariance')
            mu = matrix.mean(axis=1)            # per-desk mean return
            if not np.all(np.isfinite(mu)):
                return _degrade('non-finite mean returns')
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                raw = np.linalg.solve(cov, mu)  # w ∝ Σ⁻¹μ (max-Sharpe tangency)
            if not np.all(np.isfinite(raw)):
                return _degrade('non-finite solution')
            w = np.clip(raw, 0.0, None)         # long-only
            total = w.sum()
            if not np.isfinite(total) or total <= 0.0:
                return _degrade('no desk with positive risk-adjusted mean')
            scaled = w * (self.target_gross / total)
            return {key: float(scaled[i]) for i, key in enumerate(keys)}, None
        except np.linalg.LinAlgError as exc:    # singular covariance
            return _degrade(f'singular covariance ({exc})')
        except Exception as exc:                # any other numerical blow-up
            logger.warning(
                "Mean-variance weighting failed (%s); falling back to "
                "inverse-vol risk parity", exc)
            return (self.risk_parity_weights(returns_by_desk),
                    'covariance computation failed')

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
            set_desk_capital_allocation(desk, weights[desk.key])
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
