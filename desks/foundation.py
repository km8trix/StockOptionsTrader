"""
Foundation Desk — the framework's proof desk (house strategy).

Momentum entries (MACD cross up + RSI band + volume confirmation, in the
spirit of strategies.base.MomentumStrategy) with an optional walk-forward
gradient-boosting overlay: once the WalkForwardController has fitted its
first model, momentum BUYs are only taken when the model score is
positive. Before the first fit, momentum signals pass ungated (each
decision notes which mode it used). Exits are momentum SELL crosses plus
the shared apply_risk stop losses.

This desk exists to prove the Phase 5 plumbing end to end — Desk ABC,
DeskIntent flow, risk wiring, trader's notes, and the walk-forward
harness — so Phases 6-8 can focus on their models.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent
from desks.ml_model import GradientBoostingModel
from desks.walk_forward import WalkForwardController, WalkForwardFit
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Minimum bars of history before the desk will signal on a symbol
#: (matches the warmup MomentumStrategy uses for its indicators).
MIN_HISTORY_DAYS = 50


class FoundationDesk(Desk):
    """House momentum desk with a walk-forward ML gate on entries."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 controller: Optional[WalkForwardController] = None,
                 model_key: Optional[str] = None,
                 rsi_entry_low: float = 40.0,
                 rsi_entry_high: float = 70.0,
                 rsi_exit: float = 70.0,
                 volume_confirmation_mult: float = 1.2):
        super().__init__(
            key='foundation',
            name='Foundation Desk',
            description=('House momentum desk: MACD cross entries with RSI '
                         'band and volume confirmation, gated by a '
                         'walk-forward gradient-boosting model once fitted.'),
            accent='#4493f8',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )
        self.rsi_entry_low = rsi_entry_low
        self.rsi_entry_high = rsi_entry_high
        self.rsi_exit = rsi_exit
        self.volume_confirmation_mult = volume_confirmation_mult
        # Controller precedence (documented contract):
        #   1. an explicit ``controller`` wins, untouched (unchanged path);
        #   2. else a ``model_key`` selects the model via the model
        #      registry, wrapped in a default WalkForwardController;
        #   3. else the historical default — GradientBoostingModel.
        # ``model_key=None`` and no controller is therefore BYTE-IDENTICAL
        # to the pre-existing behavior (same default model, same controller
        # defaults). The model registry import is local so the default and
        # explicit-controller paths never pull in optional model deps.
        if controller is not None:
            self._controller = controller
        elif model_key is not None:
            from desks.models import build_model
            self._controller = WalkForwardController(build_model(model_key))
        else:
            self._controller = WalkForwardController(GradientBoostingModel())

    @property
    def walk_forward_fits(self) -> List[WalkForwardFit]:
        """Fit events recorded by the desk's walk-forward controller."""
        return self._controller.fits

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """Momentum entries/exits for the day, ML-gated once fitted.

        all_data frames are indicator-enriched and sliced through `date`
        (engine guarantee); the controller re-slices regardless.
        """
        if self._controller.maybe_refit(all_data, date):
            fit = self._controller.fits[-1]
            self.note(
                'model',
                f"Walk-forward refit #{len(self._controller.fits)}: trained "
                f"on {fit.n_samples} samples ({fit.train_start} .. "
                f"{fit.train_end})",
                **fit.to_dict())

        scores = self._controller.predict(all_data, date)

        size_fraction = min(0.10, 1.0 / max(1, len(all_data)))
        intents: List[DeskIntent] = []
        current_date = pd.Timestamp(date)

        for symbol, data in all_data.items():
            if len(data) < MIN_HISTORY_DAYS:
                continue
            if data.index[-1] != current_date:
                continue  # symbol has no bar today; signals would be stale

            current = data.iloc[-1]
            prev = data.iloc[-2]
            if pd.isna(current['macd']) or pd.isna(current['signal']) \
                    or pd.isna(current['rsi']):
                continue

            asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
            position = portfolio.get_position(asset)

            macd_cross_up = (current['macd'] > current['signal']
                             and prev['macd'] <= prev['signal'])
            macd_cross_down = (current['macd'] < current['signal']
                               and prev['macd'] >= prev['signal'])
            volume_ratio = (current['volume'] / current['volume_sma']
                            if current['volume_sma'] > 0 else 0.0)

            # ---- Exits -------------------------------------------------
            if position is not None and position.quantity > 0:
                if macd_cross_down or current['rsi'] > self.rsi_exit:
                    reason = ('MACD cross down' if macd_cross_down
                              else f"RSI {current['rsi']:.1f} > {self.rsi_exit:.0f}")
                    intents.append(DeskIntent(asset=asset, action='SELL',
                                              size_fraction=1.0,
                                              reason=f"momentum exit: {reason}"))
                    self.note(
                        'signal',
                        f"Momentum exit {symbol} @ {current['close']:.2f}: "
                        f"{reason} (MACD {current['macd']:.4f} vs signal "
                        f"{current['signal']:.4f})",
                        symbol=symbol, close=current['close'],
                        macd=current['macd'], macd_signal=current['signal'],
                        rsi=current['rsi'])
                continue

            # ---- Entries -----------------------------------------------
            rsi_in_band = (self.rsi_entry_low <= current['rsi']
                           <= self.rsi_entry_high)
            volume_confirmed = (volume_ratio
                                >= self.volume_confirmation_mult)
            if not (macd_cross_up and rsi_in_band and volume_confirmed):
                continue

            score = scores.get(symbol) if scores is not None else None
            if scores is not None and score is not None and score <= 0:
                self.note(
                    'model',
                    f"Blocked momentum long {symbol} @ "
                    f"{current['close']:.2f}: walk-forward score "
                    f"{score:+.4f} <= 0",
                    symbol=symbol, close=current['close'], wf_score=score)
                continue

            if scores is None:
                mode = 'momentum-only (walk-forward model not yet fitted)'
            elif score is None:
                mode = 'momentum-only (no walk-forward score for symbol)'
            else:
                mode = f"walk-forward gated (score {score:+.4f} > 0)"

            intents.append(DeskIntent(
                asset=asset, action='BUY', size_fraction=size_fraction,
                reason=f"momentum entry, {mode}"))
            self.note(
                'signal',
                f"Momentum long {symbol} @ {current['close']:.2f}: MACD "
                f"{current['macd']:.4f} > signal {current['signal']:.4f} "
                f"(cross up), RSI {current['rsi']:.1f} in "
                f"[{self.rsi_entry_low:.0f}, {self.rsi_entry_high:.0f}], "
                f"vol {volume_ratio:.1f}x avg; size "
                f"{size_fraction:.0%} of desk capital; {mode}",
                symbol=symbol, close=current['close'], macd=current['macd'],
                macd_signal=current['signal'], rsi=current['rsi'],
                volume_ratio=volume_ratio, size_fraction=size_fraction,
                wf_score=score)

        return intents
