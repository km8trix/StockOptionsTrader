"""
Walk-forward fit/predict harness — leakage-proof by construction.

THE INVARIANT (enforced here, not by caller discipline):

    Every row any model ever receives — in fit() OR predict() — has
    index <= the current simulation date, and every prediction made at
    date D comes from a model whose train_end <= D.

WalkForwardController is the only object that hands data to a model. Both
entry points slice every frame to ``index <= date`` before the model sees
it (predict re-slices even if the caller already did — defense in depth, a
malicious or buggy caller passing future rows changes nothing), and fit
data is additionally capped to the trailing ``train_window_days`` rows per
symbol. Models therefore cannot leak the future even if they try to: they
hold no data beyond what fit/predict hand them.

Protocol contract for Phase 6+ implementers (HMM regime models, gradient
boosting cross-sectional models, cointegration pairs all plug in here):

    class MyModel(WalkForwardModel):
        def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
            # train_data: symbol -> OHLCV(+indicator) frame, already
            # sliced/capped by the controller. Train on it; keep ONLY
            # fitted parameters — never retain the frames themselves.

        def predict(self, data: dict[str, pd.DataFrame], date) -> dict[str, float]:
            # data: symbol -> frame sliced to index <= date.
            # Return symbol -> score; positive = long signal strength.
            # Return {} (or omit symbols) when there is nothing to say.

    desk._controller = WalkForwardController(MyModel())
    # each simulated day:  controller.maybe_refit(all_data, date)
    #                      scores = controller.predict(all_data, date)
    #                      (scores is None until the first fit)

Refit cadence: the first fit happens once any symbol has at least
``min_train_days`` rows of history; thereafter the model is refit every
``refit_every_days`` distinct trading days. Each fit is recorded as a
WalkForwardFit (serialized into desk-mode backtest reports).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as date_type
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardModel(ABC):
    """Abstract fit/predict model managed by WalkForwardController.

    Implementations must be stateless apart from fitted parameters: they
    hold NO market data outside what fit()/predict() receive, so the
    controller's slicing is the single source of temporal truth.

    One sanctioned exception: a predict-side DERIVED cache is permitted
    when every read is validated as an exact prefix/extension of the
    data the controller just handed in and every output is bit-identical
    to a stateless recompute on that handed slice (RegimeHMMModel's
    incremental forward recursion). The leakage invariant is preserved
    because the cache can never contribute information beyond the
    controller-sliced input that produced it.
    """

    @abstractmethod
    def fit(self, train_data: Dict[str, pd.DataFrame]) -> None:
        """Train on symbol -> frame data (pre-sliced by the controller)."""

    @abstractmethod
    def predict(self, data: Dict[str, pd.DataFrame], date) -> Dict[str, float]:
        """Score symbols at `date`: positive = long signal strength."""


@dataclass
class WalkForwardFit:
    """One recorded model fit: when it happened and what it trained on."""
    fit_date: date_type
    train_start: date_type
    train_end: date_type
    n_samples: int

    def to_dict(self) -> Dict:
        """Serialize to the report shape (contract C3)."""
        return {
            'fit_date': self.fit_date.strftime('%Y-%m-%d'),
            'train_start': self.train_start.strftime('%Y-%m-%d'),
            'train_end': self.train_end.strftime('%Y-%m-%d'),
            'n_samples': self.n_samples,
        }


def _as_date(value) -> date_type:
    """Normalize a Timestamp/datetime/date to a plain datetime.date."""
    return pd.Timestamp(value).date()


class WalkForwardController:
    """Schedules model refits and guarantees temporal hygiene of all data.

    See the module docstring for the leakage invariant this class enforces
    by construction.
    """

    def __init__(self, model: WalkForwardModel,
                 train_window_days: int = 252,
                 refit_every_days: int = 21,
                 min_train_days: int = 120):
        self.model = model
        self.train_window_days = train_window_days
        self.refit_every_days = refit_every_days
        self.min_train_days = min_train_days
        self.fits: List[WalkForwardFit] = []
        self.last_fit_date: Optional[date_type] = None
        # Distinct trading days observed since the last fit (the engine
        # calls maybe_refit once per simulated day; duplicate calls on the
        # same date are not double-counted).
        self._days_since_fit = 0
        self._last_seen_date: Optional[date_type] = None

    @property
    def is_fitted(self) -> bool:
        return bool(self.fits)

    # ------------------------------------------------------------------
    # Temporal slicing — the construction guarantee
    # ------------------------------------------------------------------
    def _slice_through(self, all_data: Dict[str, pd.DataFrame], date,
                       cap_window: bool) -> Dict[str, pd.DataFrame]:
        """Slice each frame to index <= date; optionally cap to the
        trailing train_window_days rows. Empty results are dropped."""
        cutoff = pd.Timestamp(date)
        sliced: Dict[str, pd.DataFrame] = {}
        for symbol, data in all_data.items():
            if data is None or data.empty:
                continue
            # Engine-driven callers pass frames already bounded through the
            # simulation date, making the mask an all-True full-frame copy;
            # share the frame instead (models are pure readers). The
            # monotonic guard is MANDATORY: index[-1] <= cutoff only implies
            # all-rows-bounded for a sorted index, and the mask path is the
            # leakage guarantee for anything unsorted.
            if data.index.is_monotonic_increasing and data.index[-1] <= cutoff:
                window = data
            else:
                window = data[data.index <= cutoff]
            if cap_window:
                window = window.tail(self.train_window_days)
            if not window.empty:
                sliced[symbol] = window
        return sliced

    # ------------------------------------------------------------------
    # Fit scheduling
    # ------------------------------------------------------------------
    def maybe_refit(self, all_data: Dict[str, pd.DataFrame], date) -> bool:
        """Refit the model if due; return True when a fit happened.

        Due when (a) never fitted and at least one symbol has
        min_train_days rows of history through `date`, or (b)
        refit_every_days distinct trading days have elapsed since the
        last fit. The data handed to model.fit is sliced to index <= date
        AND capped to the trailing train_window_days rows per symbol.
        """
        current = _as_date(date)
        if self._last_seen_date is not None and current != self._last_seen_date:
            self._days_since_fit += 1
        self._last_seen_date = current

        # Fitted and not yet due: return before slicing — on this path the
        # sliced history is discarded regardless of its contents, and this
        # runs every simulated day per controller.
        if self.is_fitted and self._days_since_fit < self.refit_every_days:
            return False

        history = self._slice_through(all_data, date, cap_window=False)
        if not history:
            return False

        if not self.is_fitted:
            # Depth is measured on the UNCAPPED history: the train-window
            # cap below bounds what the model sees, not whether enough
            # history exists to fit at all.
            depth = max(len(frame) for frame in history.values())
            if depth < self.min_train_days:
                return False

        train_data = {symbol: frame.tail(self.train_window_days)
                      for symbol, frame in history.items()}

        self.model.fit(train_data)

        train_start = min(frame.index.min() for frame in train_data.values())
        train_end = max(frame.index.max() for frame in train_data.values())
        n_samples = sum(len(frame) for frame in train_data.values())
        fit = WalkForwardFit(fit_date=current,
                             train_start=_as_date(train_start),
                             train_end=_as_date(train_end),
                             n_samples=n_samples)
        self.fits.append(fit)
        self.last_fit_date = current
        self._days_since_fit = 0
        logger.info("Walk-forward fit on %s: train %s..%s (%d samples)",
                    fit.fit_date, fit.train_start, fit.train_end, n_samples)
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, all_data: Dict[str, pd.DataFrame],
                date) -> Optional[Dict[str, float]]:
        """Score symbols at `date`, or None if the model was never fitted.

        Defense in depth: frames are re-sliced to index <= date here even
        though callers are expected to pass already-bounded data — the
        model NEVER sees a row beyond the simulation date.
        """
        if not self.is_fitted:
            return None
        data = self._slice_through(all_data, date, cap_window=False)
        return self.model.predict(data, date)
