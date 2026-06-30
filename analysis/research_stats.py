"""
Research-integrity statistics: Probabilistic and Deflated Sharpe ratios.

A raw Sharpe ratio answers "how good did this look?" but not "how likely is it
real?". These two statistics (Bailey & Lopez de Prado) close that gap:

  * Probabilistic Sharpe Ratio (PSR): the probability that the TRUE Sharpe
    exceeds a benchmark (default 0), given the SAMPLE Sharpe, the number of
    observations, and the returns' skew and kurtosis. A short, fat-tailed,
    left-skewed track record gets a lower PSR than a clean Gaussian one at the
    same headline Sharpe. PSR(0) is always meaningful and needs no assumptions
    about how many strategies were tried.

  * Deflated Sharpe Ratio (DSR): PSR measured against a benchmark RAISED to the
    expected maximum Sharpe you would see from ``n_trials`` independent lucky
    draws. It penalizes multiple testing / strategy selection — the more
    configurations you tried, the higher the bar a Sharpe must clear to be
    credible. With n_trials <= 1 (no selection) DSR reduces exactly to PSR(0).

CONVENTIONS / HONESTY:
  - These use the PER-PERIOD Sharpe of the supplied returns (mean/std, sample
    std ddof=1) — NOT the annualized excess Sharpe reported elsewhere. They are
    a probability in [0, 1], or None when undefined (<2 finite returns, or a
    degenerate/zero variance).
  - DSR's deflation needs the cross-trial DISPERSION of Sharpe estimates. With
    a single backtest we do not have per-trial Sharpes, so the dispersion is
    APPROXIMATED by this backtest's own Sharpe standard error. That is a
    documented simplification; treat DSR as indicative, not exact.
  - ``n_trials`` supplied by callers (e.g. the count of walk-forward refits) is
    itself a PROXY for the true multiple-testing breadth and, with overlapping
    train windows, an over-count — which biases DSR conservatively (downward).
    Conservative is the safe direction for a research-integrity gate.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kurtosis, norm, skew
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)

#: Euler-Mascheroni constant, used in the expected-maximum-of-normals estimate.
_EULER_GAMMA = 0.5772156649015329


def _sharpe_components(
        returns: Sequence[float]) -> Optional[Tuple[float, int, float, float]]:
    """(per-period Sharpe, N, skew, kurtosis[normal=3]) over finite returns, or
    None when undefined (<2 finite points, or non-finite/zero std)."""
    arr = np.asarray([r for r in returns], dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        logger.debug("Sharpe stats undefined: <2 finite returns (n=%d)", n)
        return None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # std must be MEANINGFULLY positive relative to the level. A near-constant
    # series yields a float-noise std (catastrophic cancellation, e.g.
    # std([0.01]*10) == 1.8e-18, not 0.0) which then produces an astronomical,
    # meaningless Sharpe and a scipy moment-precision warning. Treat it as
    # degenerate (None) rather than letting the noise through.
    if not np.isfinite(std) or std <= 1e-12 * (abs(mean) + 1.0):
        logger.debug("Sharpe stats undefined: degenerate std=%.3e "
                     "(mean=%.3e, n=%d)", std, mean, n)
        return None
    sharpe = mean / std
    # bias=False -> sample skew/kurtosis; kurtosis non-excess (normal == 3.0).
    sk = float(skew(arr, bias=False)) if n >= 3 else 0.0
    kt = float(kurtosis(arr, fisher=False, bias=False)) if n >= 4 else 3.0
    if not (np.isfinite(sk) and np.isfinite(kt)):
        logger.debug("Sharpe stats undefined: non-finite skew=%s / kurt=%s "
                     "(n=%d)", sk, kt, n)
        return None
    return sharpe, n, sk, kt


def _psr_variance_term(sharpe: float, sk: float, kt: float) -> float:
    """Variance term of the Sharpe estimator (Bailey & Lopez de Prado):
    1 - skew*SR + ((kurt - 1)/4)*SR^2."""
    return 1.0 - sk * sharpe + ((kt - 1.0) / 4.0) * sharpe * sharpe


def probabilistic_sharpe_ratio(returns: Sequence[float],
                               sr_benchmark: float = 0.0) -> Optional[float]:
    """P(true per-period Sharpe > sr_benchmark) in [0, 1], or None if
    undefined."""
    comp = _sharpe_components(returns)
    if comp is None:
        return None
    sharpe, n, sk, kt = comp
    var_term = _psr_variance_term(sharpe, sk, kt)
    if not np.isfinite(var_term) or var_term <= 0.0:
        logger.debug("Sharpe-prob undefined: non-positive variance term %.4f "
                     "(sharpe=%.4f, skew=%.4f, kurt=%.4f)",
                     var_term, sharpe, sk, kt)
        return None
    z = (sharpe - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def expected_max_sharpe_z(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` i.i.d. standard normals (the
    multiple-testing inflation, in standard-error units). 0.0 for n_trials<=1
    (no selection).

    Uses the Gumbel-limit approximation (Bailey & Lopez de Prado). It slightly
    UNDER-estimates at T=2 (~0.52 vs the exact 0.564, making DSR marginally
    optimistic there) and slightly over-estimates for T>=5 (conservative);
    walk-forward refit counts are almost always >=5, where it errs safe."""
    t = int(n_trials)
    if t < 2:
        return 0.0
    high = float(norm.ppf(1.0 - 1.0 / t))
    low = float(norm.ppf(1.0 - 1.0 / (t * math.e)))
    return (1.0 - _EULER_GAMMA) * high + _EULER_GAMMA * low


def deflated_sharpe_ratio(returns: Sequence[float],
                          n_trials: int) -> Optional[float]:
    """PSR against the expected-maximum Sharpe from ``n_trials`` independent
    trials. With n_trials <= 1 this equals probabilistic_sharpe_ratio(returns,
    0.0). Returns None when undefined.

    The cross-trial Sharpe dispersion is approximated by this backtest's Sharpe
    standard error sqrt(var_term / (N-1)) (see the module docstring)."""
    comp = _sharpe_components(returns)
    if comp is None:
        return None
    if int(n_trials) <= 1:
        return probabilistic_sharpe_ratio(returns, 0.0)
    sharpe, n, sk, kt = comp
    var_term = _psr_variance_term(sharpe, sk, kt)
    if not np.isfinite(var_term) or var_term <= 0.0:
        logger.debug("Sharpe-prob undefined: non-positive variance term %.4f "
                     "(sharpe=%.4f, skew=%.4f, kurt=%.4f)",
                     var_term, sharpe, sk, kt)
        return None
    se_sharpe = math.sqrt(var_term / (n - 1))
    sr_star = se_sharpe * expected_max_sharpe_z(n_trials)
    return probabilistic_sharpe_ratio(returns, sr_star)


# ----------------------------------------------------------------------------
# Per-fold out-of-sample significance + multiple-testing correction (Step 4).
#
# A walk-forward backtest produces a sequence of refits; the realized account
# returns between consecutive refits form per-"fold" OOS samples. These helpers
# score each fold's OOS edge (one-sided t-test that the mean return > 0) and
# correct the family of fold p-values for multiple testing (Bonferroni and
# Benjamini-Hochberg). As with DSR's n_trials, overlapping train windows mean
# the fold count over-states INDEPENDENT trials, so the correction is a
# documented heuristic upper bound on multiple-testing severity, NOT exact
# FWER/FDR control. The fold-significance layer and the DSR deflation are two
# distinct lenses on the same over-counted breadth and must NOT be combined.
# ----------------------------------------------------------------------------


def fold_oos_tstat(returns: Sequence[float]) -> Optional[float]:
    """One-sample t-statistic that the per-period mean OOS return is > 0.

    ``t = mean / (std_ddof1 / sqrt(N)) = (per-period Sharpe) * sqrt(N)``. Reuses
    :func:`_sharpe_components`, so it inherits the SAME ddof=1 sample-std
    convention and the SAME ``<2``-finite / degenerate-std guard as PSR/DSR — a
    near-constant or too-short fold returns None (never a meaningless,
    astronomical t). The sign is informative: a losing fold yields a negative t.
    """
    comp = _sharpe_components(returns)
    if comp is None:
        return None
    sharpe, n, _sk, _kt = comp
    return float(sharpe * math.sqrt(n))


def fold_oos_pvalue(returns: Sequence[float]) -> Optional[float]:
    """One-sided p-value for H0: mean OOS return <= 0 vs H1: mean > 0.

    Upper tail of :func:`fold_oos_tstat` under a Student-t(df=N-1) null, or None
    when the t-stat is undefined. One-sided BY DESIGN: a fold with a large
    NEGATIVE mean gets a p-value near 1 (correctly not a significant positive
    edge), so a loss can never masquerade as a discovery.
    """
    comp = _sharpe_components(returns)
    if comp is None:
        return None
    sharpe, n, _sk, _kt = comp
    t_stat = sharpe * math.sqrt(n)
    return float(student_t.sf(t_stat, df=n - 1))


def bonferroni_alpha(alpha: float, m: int) -> Optional[float]:
    """Bonferroni per-comparison significance threshold ``alpha / m`` for ``m``
    simultaneous tests, or None when ``m < 1``.

    Pure arithmetic: the family size ``m`` (the number of TESTABLE folds) is
    supplied by the caller, so the "overlapping walk-forward windows over-count
    independent trials" caveat lives at the call site, not here.
    """
    m = int(m)
    if m < 1:
        return None
    return float(alpha) / m


def benjamini_hochberg(pvalues: Sequence[Optional[float]],
                       alpha: float = 0.05) -> Dict:
    """Benjamini-Hochberg step-up FDR control over a family of p-values.

    None / non-finite entries — folds whose t-stat was undefined — are DROPPED
    from the family and never counted in ``m``, so an untestable fold neither
    inflates nor relaxes the correction. Returns::

        {'m': int,                  # number of finite p-values tested
         'alpha': float,
         'bh_threshold': float|None,# largest p_(k) with p_(k) <= (k/m)*alpha
         'n_significant_bh': int,
         'rejected_bh': list[bool]} # reject flags ALIGNED to the input order
                                    # (None / dropped entries -> False)

    Standard step-up: sort the ``m`` finite p-values ascending; find the largest
    rank ``k`` with ``p_(k) <= (k/m)*alpha``; reject every hypothesis with
    ``p <= p_(k)``. ``m == 0`` -> threshold None, no rejections.
    """
    flags = [False] * len(pvalues)
    finite = [(i, float(p)) for i, p in enumerate(pvalues)
              if p is not None and np.isfinite(p)]
    m = len(finite)
    if m == 0:
        return {'m': 0, 'alpha': float(alpha), 'bh_threshold': None,
                'n_significant_bh': 0, 'rejected_bh': flags}
    ordered = sorted(finite, key=lambda ip: ip[1])
    bh_threshold: Optional[float] = None
    cutoff_rank = 0
    for rank, (_idx, p) in enumerate(ordered, start=1):
        if p <= (rank / m) * alpha:
            bh_threshold = p
            cutoff_rank = rank
    # Step-up: reject the cutoff_rank smallest p-values (all p <= p_(k*)).
    for rank, (idx, _p) in enumerate(ordered, start=1):
        if rank <= cutoff_rank:
            flags[idx] = True
    return {'m': m, 'alpha': float(alpha), 'bh_threshold': bh_threshold,
            'n_significant_bh': cutoff_rank, 'rejected_bh': flags}


def validate_strategy_oos(
        returns: Sequence[float],
        period_labels: Optional[Sequence] = None,
        *,
        psr_threshold: float = 0.95,
        alpha: float = 0.05,
        min_period_obs: int = 20,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252) -> Dict:
    """Pass/fail research gate for a strategy's OUT-OF-SAMPLE return series.

    The honest bar for a single, FIXED-rule strategy: no parameter search, so
    n_trials = 1 and DSR == PSR (no Sharpe deflation). The benchmark is the
    RISK-FREE rate, NOT zero: ``risk_free_rate`` (annualized) is converted to a
    per-period hurdle and subtracted from every return, so the gate tests
    "credibly beats cash", not merely "drifts up". (Benchmark 0 passes ANY
    upward-drifting series — buy-and-hold SPY included — which defeats the
    purpose.) Two COMPLEMENTARY lenses, combinable because they do not both
    penalize the same multiple-testing breadth (unlike DSR-deflation + fold-BH
    over the SAME folds, which the module docstring forbids):

      1. Overall edge          — PSR(excess) >= ``psr_threshold`` (the
         probability the true Sharpe beats the risk-free rate).
      2. Sub-period robustness  — split the EXCESS returns by ``period_labels``
         (e.g. calendar year), one-sided t-test (mean excess > 0) per period,
         Benjamini-Hochberg across the family; require >= 1 BH-significant
         positive period, so the edge is not one lucky stretch.

    ``passed`` is True iff BOTH hold. Periods with fewer than ``min_period_obs``
    finite returns are untestable (p-value None) and BH drops them. With
    ``period_labels=None`` the whole series is a single period.

    Returns a dict carrying ``passed`` plus every component for audit/logging.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = [r - rf_per_period for r in returns]

    psr = probabilistic_sharpe_ratio(excess, 0.0)  # P(Sharpe > risk-free)
    psr_pass = psr is not None and psr >= psr_threshold

    # Group EXCESS returns into sub-periods, preserving first-seen label order.
    if period_labels is None:
        groups = [(None, excess)]
    else:
        if len(period_labels) != len(excess):
            raise ValueError(
                f"period_labels length {len(period_labels)} != returns length "
                f"{len(excess)}")
        ordered_labels: list = []
        buckets: Dict = {}
        for label, r in zip(period_labels, excess):
            if label not in buckets:
                buckets[label] = []
                ordered_labels.append(label)
            buckets[label].append(r)
        groups = [(label, buckets[label]) for label in ordered_labels]

    fold_labels: list = []
    fold_pvalues: list = []
    for label, grp in groups:
        finite = [r for r in grp if r is not None and np.isfinite(r)]
        fold_labels.append(label)
        if len(finite) < min_period_obs:
            fold_pvalues.append(None)        # untestable -> BH drops it
        else:
            fold_pvalues.append(fold_oos_pvalue(finite))

    bh = benjamini_hochberg(fold_pvalues, alpha=alpha)
    fold_pass = bh['n_significant_bh'] >= 1
    passed = bool(psr_pass and fold_pass)

    return {
        'passed': passed,
        'psr': psr,
        'psr_threshold': float(psr_threshold),
        'psr_pass': psr_pass,
        'risk_free_rate': float(risk_free_rate),
        'n_trials': 1,
        'n_periods_tested': sum(1 for p in fold_pvalues if p is not None),
        'fold_pass': fold_pass,
        'fold_labels': fold_labels,
        'fold_pvalues': fold_pvalues,
        'bh': bh,
    }
