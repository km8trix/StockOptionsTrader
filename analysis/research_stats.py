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
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kurtosis, norm, skew

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
