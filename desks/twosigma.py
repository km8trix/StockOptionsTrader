"""
Two Sigma Desk — a systematic, ML-driven cross-sectional long/short equity
book that showcases the Phase A/B model zoo.

The desk runs ONE leakage-proof alpha signal each day, ranks the scored
universe cross-sectionally, and holds a dollar-balanced book: LONG the top
quantile, SHORT the bottom quantile, sized so gross exposure is roughly
``target_gross`` of desk capital and net exposure is roughly zero. This is
the same decile-book mechanic the Renaissance stat-arb book uses
(``_stat_arb_intents``), distilled into a single-purpose systematic desk.

THE CROSS-SECTIONAL BOOK ITSELF lives in
:class:`desks.cross_sectional.CrossSectionalLongShortDesk` — the shared,
recently-bug-fixed book mechanics (one-reconcile-per-day, closing-state,
dollar-balanced sizing with the max_name_size clamp, orphan sweep, refit
scheduling, tagged ``walk_forward_fits``). This desk supplies only the Two
Sigma IDENTITY (branding + note voice) and the Two Sigma ALPHA (a
committee of zoo controllers averaged into one centered score). The AQR
desk reuses the SAME base with a different identity and a factor-model
alpha — neither desk re-implements the book.

MODELS (the Two Sigma multi-model ethos):
    The DEFAULT is a single ``stacking`` controller — a fast, deterministic
    logistic ensemble of gradient boosting + LightGBM — so backtests and
    tests stay fast and reproducible. Optionally a ``models=`` list of zoo
    ids (gbm/lightgbm/stacking/mlp/lstm) builds a COMMITTEE: every member
    runs through its OWN WalkForwardController and their centered scores
    (P(up) - 0.5) are AVERAGED per symbol into one alpha. A symbol is only
    scored when EVERY committee member that has data scores it; the average
    spans the members that produced a score. Tests may inject an explicit
    ``controller=`` (single controller) exactly like the Foundation desk.

ALPHA -> BOOK each simulated day (mirrors the renaissance flow):
    1. advance the trading-day clock;
    2. refit-if-due every controller (note each fit, tagged with its model);
    3. score the universe;
    4. if the model is unfitted (scores is None) or fewer than
       ``min_scored`` symbols have a usable score, FLATTEN the book (close
       everything) and emit no new opens (graceful degrade, noted);
    5. otherwise rank cross-sectionally, then reconcile ONCE against the
       desired book — names that keep their side are HELD untouched, names
       that left their side are closed (a flip waits for the close to fill),
       dead bookkeeping is dropped after the grace, and positions in
       desk-traded symbols the book no longer tracks are ORPHAN-SWEPT (the
       engine holds pending intents up to MAX_PENDING_DAYS, longer than the
       reconcile grace, so an entry can fill after its tracking was
       reconciled away) — then open the newly-selected legs.

Only symbols with a bar TODAY (frame.index[-1] == date) are scored or
traded — stale bars never drive a decision.

SIZING: dollar-balanced. With k_long longs and k_short shorts each side
gets half of ``target_gross``: a long is ~ (0.5 * target_gross) / k_long
of desk capital, a short ~ (0.5 * target_gross) / k_short, each CLAMPED to
``max_name_size`` (default 0.10) so a small selected set stays inside the
shared position-size cap instead of being blocked to a flat book. Gross is
then ~ min(target_gross, 2 * k * max_name_size) of desk capital and net ~ 0.
The shared ``Desk.apply_risk`` owns the position-size cap, the daily-loss
circuit, stop losses, and the no-one-step-flip rule — this desk only emits
standard DeskIntents and lets reconcile close opposite positions before
re-opening.

MARGIN IS NOT MODELED: shorts use the shared cash-account approximation
(proceeds held as cash); live shorting is gated to Phase 9.

Every controller fit is recorded in ``walk_forward_fits`` tagged with its
model name (TaggedWalkForwardFit, the additive C3 key shared with the
Renaissance desk).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from desks.cross_sectional import (CrossSectionalLongShortDesk,
                                    RECONCILE_GRACE_DAYS)
from desks.walk_forward import WalkForwardController
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: The default single-controller model id (fast, deterministic ensemble).
DEFAULT_MODEL_KEY = 'stacking'

# RECONCILE_GRACE_DAYS is re-exported from the shared base for backward
# compatibility (callers/tests that imported it from this module still work).
__all__ = ['TwoSigmaDesk', 'DEFAULT_MODEL_KEY', 'RECONCILE_GRACE_DAYS']


class TwoSigmaDesk(CrossSectionalLongShortDesk):
    """Systematic cross-sectional long/short equity desk (module docstring).

    Reuses the shared cross-sectional book; adds the Two Sigma identity and
    a committee-averaged ML alpha.
    """

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 controller: Optional[WalkForwardController] = None,
                 model_key: Optional[str] = None,
                 models: Optional[List[str]] = None,
                 quantile: float = 0.2,
                 target_gross: float = 1.0,
                 max_name_size: float = 0.10,
                 min_scored: int = 4,
                 exit_quantile: Optional[float] = None,
                 min_holding_days: int = 0,
                 size_by_signal_strength: bool = False):
        # Controller precedence (documented contract, mirrors Foundation):
        #   1. an explicit ``controller`` wins, untouched — a single
        #      controller (tests inject this);
        #   2. else a ``models`` list builds a COMMITTEE of controllers, one
        #      per zoo id, whose centered scores are averaged;
        #   3. else a ``model_key`` selects one zoo model (default
        #      ``stacking``) wrapped in a default WalkForwardController.
        # The model-registry import is local so the explicit-controller path
        # never pulls in optional model deps.
        committee: List[Tuple[str, WalkForwardController]]
        if controller is not None:
            if models is not None or model_key is not None:
                raise ValueError(
                    "Provide either an explicit controller= OR "
                    "model_key=/models=, not both")
            committee = [('custom', controller)]
        elif models is not None:
            if not models:
                raise ValueError("models= must list at least one model id")
            from desks.models import build_model
            committee = [
                (key, WalkForwardController(build_model(key)))
                for key in models
            ]
        else:
            from desks.models import build_model
            key = model_key if model_key is not None else DEFAULT_MODEL_KEY
            committee = [
                (key, WalkForwardController(build_model(key)))
            ]
        model_label = '+'.join(name for name, _ in committee)

        super().__init__(
            key='twosigma',
            name='Two Sigma Desk',
            description=('Systematic cross-sectional long/short equity: a '
                         'walk-forward ML model (single stacking ensemble by '
                         'default, optional multi-model committee) ranks the '
                         'universe; long the top quantile, short the bottom, '
                         'dollar-balanced.'),
            accent='#3fb950',
            note_label='Two-Sigma',
            reason_prefix='two-sigma',
            committee=committee,
            model_label=model_label,
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
            target_gross=target_gross,
            max_name_size=max_name_size,
            min_scored=min_scored,
            exit_quantile=exit_quantile,
            min_holding_days=min_holding_days,
            size_by_signal_strength=size_by_signal_strength,
        )

    # ------------------------------------------------------------------
    # Two Sigma alpha: committee-averaged centered scores
    # ------------------------------------------------------------------
    def _alpha_scores(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Optional[Dict[str, float]]:
        """Centered alpha per symbol: the committee's averaged P(up)-0.5.

        Returns None until EVERY controller has fitted at least once (the
        committee has no alpha to offer before its first complete fit). Each
        symbol's alpha is the mean of the members that produced a score for
        it — so a member that degrades to {} on a given day simply abstains.
        """
        per_symbol_sum: Dict[str, float] = {}
        per_symbol_count: Dict[str, int] = {}
        for _, controller in self._committee:
            result = controller.predict(all_data, date)
            if result is None:
                # This member is unfitted: the committee has no consensus yet.
                return None
            for symbol, score in result.items():
                per_symbol_sum[symbol] = per_symbol_sum.get(symbol, 0.0) + score
                per_symbol_count[symbol] = per_symbol_count.get(symbol, 0) + 1

        if not per_symbol_count:
            return {}
        return {symbol: per_symbol_sum[symbol] / per_symbol_count[symbol]
                for symbol in per_symbol_count}
