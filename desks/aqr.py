"""
AQR Desk — the TRANSPARENT classical-quant counterpart to Two Sigma.

Where Two Sigma ranks the universe with an ML model zoo (a black box of
gradient boosting / neural nets), AQR ranks it with a GLASS-BOX
:class:`desks.models.factor.FactorModel`: a handful of named, economically-
signed PRICE FACTORS (12-1 momentum, short-term reversal, low-volatility,
risk-adjusted momentum), each standardized cross-sectionally per date and
combined by ridge regression. The book that turns those scores into a
dollar-balanced long/short portfolio is the EXACT SAME shared
:class:`desks.cross_sectional.CrossSectionalLongShortDesk` mechanics Two
Sigma uses — one reconcile per day, the closing-state guard, the orphan
sweep, the max_name_size sizing clamp. Neither desk re-implements the book.

THE AQR SIGNATURE — FACTOR-WEIGHT TRANSPARENCY. On every factor-model
refit, the desk reads the freshly-fitted weights off the FactorModel and
records them in a ``model`` trader note, so an operator can see at a glance
which factors are driving the book (and whether the fit fell back to the
equal-weight prior on thin data). This is the AQR transparency promise made
visible in the trader's notes.

NOT MODEL-SELECTABLE. AQR is defined by its factor model; unlike Foundation
and Two Sigma it does NOT accept a ``model_key`` (the registry does not add
``aqr`` to the model-selectable set). The FactorModel is exposed to the GUI
model picker via the zoo (``build_model('factor')`` / ``available_models``),
but the AQR desk itself always runs the factor model.

MARGIN IS NOT MODELED: shorts use the shared cash-account approximation
(proceeds held as cash); live shorting is gated to Phase 9.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from desks.cross_sectional import CrossSectionalLongShortDesk
from desks.models.factor import FACTOR_COLUMNS, FactorModel
from desks.walk_forward import WalkForwardController
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: The AQR model label (single fixed factor controller).
_MODEL_KEY = 'factor'

__all__ = ['AqrDesk']


class AqrDesk(CrossSectionalLongShortDesk):
    """Transparent factor-driven cross-sectional long/short desk.

    Reuses the shared cross-sectional book; supplies the AQR identity, a
    fixed :class:`FactorModel` controller, and factor-weight transparency
    notes on each refit.
    """

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 controller: Optional[WalkForwardController] = None,
                 quantile: float = 0.2,
                 target_gross: float = 1.0,
                 max_name_size: float = 0.10,
                 min_scored: int = 4):
        # A single fixed factor controller. Tests may inject an explicit
        # ``controller=`` (e.g. a stub) exactly like the other desks; the
        # default wraps a fresh FactorModel. AQR is NOT model-selectable —
        # there is deliberately no model_key / models parameter.
        if controller is None:
            controller = WalkForwardController(FactorModel())
        committee: List[Tuple[str, WalkForwardController]] = [
            (_MODEL_KEY, controller)]

        super().__init__(
            key='aqr',
            name='AQR Desk',
            description=('Transparent classical-quant cross-sectional '
                         'long/short equity: price-based factors (momentum, '
                         'low-volatility, reversal, risk-adjusted momentum) '
                         'standardized cross-sectionally and combined by '
                         'ridge regression rank the universe — long the top '
                         'quantile, short the bottom, dollar-balanced.'),
            accent='#f0883e',
            note_label='AQR',
            reason_prefix='aqr',
            committee=committee,
            model_label=_MODEL_KEY,
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
            target_gross=target_gross,
            max_name_size=max_name_size,
            min_scored=min_scored,
        )

    # ------------------------------------------------------------------
    # AQR transparency: report the fitted factor weights on each refit
    # ------------------------------------------------------------------
    def _refit_models(self, all_data: Dict[str, pd.DataFrame], date) -> None:
        """Refit the factor controller, then note the fitted factor weights.

        Detects a fresh fit by the controller's fit count, emits the shared
        refit note (via the base behavior, replicated here so the AQR
        weight note rides alongside it), and records the factor weights —
        the AQR attribution signature. Falls back gracefully if the
        controller's model is not a FactorModel (e.g. an injected stub).
        """
        for model_name, controller in self._committee:
            before = len(controller.fits)
            if not controller.maybe_refit(all_data, date):
                continue
            fit = controller.fits[-1]
            # Shared refit note (identical shape to the base).
            self.note(
                'model',
                f"{model_name} model refit #{len(controller.fits)}: "
                f"trained on {fit.n_samples} samples "
                f"({fit.train_start} .. {fit.train_end})",
                model=model_name, **fit.to_dict())

            # AQR signature: surface the freshly-fitted factor weights.
            model = getattr(controller, 'model', None)
            weights = getattr(model, 'factor_weights', None)
            if not isinstance(weights, dict) or not weights:
                continue
            degraded = bool(getattr(model, 'is_degraded', False))
            ordered = {c: weights[c] for c in FACTOR_COLUMNS if c in weights}
            summary = ', '.join(f"{c} {w:+.3f}" for c, w in ordered.items())
            mode = ('equal-weight fallback (thin/degenerate data)'
                    if degraded else 'ridge-fitted')
            self.note(
                'model',
                f"AQR factor weights ({mode}): {summary}",
                model=model_name, factor_weights=ordered, degraded=degraded)
            logger.debug("AQR refit #%d factor weights: %s (degraded=%s)",
                         before + 1, ordered, degraded)

    # ------------------------------------------------------------------
    # AQR alpha: the transparent factor composite
    # ------------------------------------------------------------------
    def _alpha_scores(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Optional[Dict[str, float]]:
        """Per-symbol factor composite score from the factor controller.

        Returns None until the controller has fitted at least once (no alpha
        before the first fit); otherwise the FactorModel's per-symbol
        composite (a weighted sum of standardized factor exposures). The base
        ranks these cross-sectionally, so absolute scale is irrelevant.
        """
        _, controller = self._committee[0]
        return controller.predict(all_data, date)

    # ------------------------------------------------------------------
    # Attribution surface (AQR transparency)
    # ------------------------------------------------------------------
    @property
    def factor_weights(self) -> Dict[str, float]:
        """The factor model's current fitted weights (empty before first fit).

        Convenience accessor mirroring the per-refit notes, so a report or
        dashboard can read the live attribution without scraping notes.
        """
        _, controller = self._committee[0]
        model = getattr(controller, 'model', None)
        weights = getattr(model, 'factor_weights', None)
        if isinstance(weights, dict):
            return dict(weights)
        return {}
