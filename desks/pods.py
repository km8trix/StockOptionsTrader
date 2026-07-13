"""
Strategy pods — the Citadel desk's unit of capital and signal generation.

A Pod emits per-symbol intent ACTIONS (BUY/SELL/SHORT/COVER) before any
sizing: position sizing, cross-pod conflict resolution, pod accounting and
risk checks are all the DESK'S job (desks.citadel.CitadelDesk). Pods that
need walk-forward models own their WalkForwardController — the leakage
chokepoint — and never slice data themselves.

The three concrete pods ADAPT existing, tested logic rather than
duplicating it:

    MomentumPod        — long-only; wraps strategies.base.MomentumStrategy
        (MACD cross + SMA-50 trend filter entries, MACD-cross-down /
        RSI-overbought exits) on the indicator-enriched frames, plus a
        volume-confirmation gate on entries (volume >= mult * volume_sma).
    MeanReversionPod   — long-only and UNgated (unlike the Renaissance
        desk's regime-conditioned book); wraps
        strategies.base.MeanReversionStrategy (Bollinger band + RSI
        reversion entries/exits).
    StatArbPod         — long/short; GradientBoostingModel through its own
        WalkForwardController (train 252 / refit 21 / min 120). Rebalances
        only on Mondays or refit days (mirroring the Renaissance stat-arb
        book's turnover limit): LONG the top max(1, n//5) ranked symbols,
        SHORT the bottom max(1, n//5). A symbol flipping sides emits only
        the close this rebalance; the new-side entry waits for the next
        rebalance (the close must fill first).

Pods are stateless apart from their own bookkeeping — no module-level
mutable state, so two pod instances never share signal state.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from core.models import Asset, AssetType
from desks.ml_model import GradientBoostingModel
from desks.walk_forward import WalkForwardController
from strategies.base import MeanReversionStrategy, MomentumStrategy

logger = logging.getLogger(__name__)

#: Actions a pod may emit per symbol (sizing-free desk-intent actions).
POD_ACTIONS = ('BUY', 'SELL', 'SHORT', 'COVER')

#: Indicator columns the strategy-wrapping pods require on a frame before
#: they will signal on it (the engine's enrichment provides all of them).
_MOMENTUM_COLUMNS = ('close', 'macd', 'signal', 'rsi', 'sma_20', 'sma_50',
                     'volume', 'volume_sma')
_MEAN_REVERSION_COLUMNS = ('close', 'rsi', 'bb_upper', 'bb_lower')


def _stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


class Pod(ABC):
    """One strategy pod: a key, a display name, and a signal stream.

    generate_signals returns {symbol: action} with action in POD_ACTIONS —
    intents BEFORE sizing. Pods with a walk-forward model expose it via
    ``controller`` so the desk can aggregate fit events into the report.
    """

    def __init__(self, key: str, name: str,
                 controller: Optional[WalkForwardController] = None):
        self.key = key
        self.name = name
        self.controller = controller

    @abstractmethod
    def generate_signals(self, all_data: Dict[str, pd.DataFrame],
                         date) -> Dict[str, str]:
        """Per-symbol intent actions for the current simulation day.

        all_data values are indicator-enriched frames sliced through
        `date` (the desk passes them straight from the engine, which
        guarantees no row beyond `date` is present).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_bar_today(frame: pd.DataFrame, date) -> bool:
        return (frame is not None and not frame.empty
                and frame.index[-1] == pd.Timestamp(date))

    @staticmethod
    def _has_columns(frame: pd.DataFrame, columns) -> bool:
        return all(column in frame.columns for column in columns)


class MomentumPod(Pod):
    """Long-only momentum: MomentumStrategy signals + a volume gate.

    Entries are MomentumStrategy BUYs (MACD cross up with the close above
    the 50-day SMA) confirmed by volume >= volume_confirmation_mult *
    volume_sma; exits are MomentumStrategy SELLs (MACD cross down or RSI
    overbought). The desk only acts on a SELL if this pod owns a long in
    the symbol, so the stateless strategy signal stays safe to re-emit.
    """

    def __init__(self, volume_confirmation_mult: float = 1.0):
        super().__init__(key='momentum', name='Momentum Pod')
        self.volume_confirmation_mult = volume_confirmation_mult
        self._strategy = MomentumStrategy()

    def generate_signals(self, all_data: Dict[str, pd.DataFrame],
                         date) -> Dict[str, str]:
        signals: Dict[str, str] = {}
        for symbol in sorted(all_data):
            frame = all_data[symbol]
            if not self._has_bar_today(frame, date):
                continue
            if not self._has_columns(frame, _MOMENTUM_COLUMNS):
                continue
            signal = self._strategy.generate_signals(frame, _stock(symbol))
            if signal == 'SELL':
                signals[symbol] = 'SELL'
            elif signal == 'BUY':
                current = frame.iloc[-1]
                volume_sma = current['volume_sma']
                if pd.isna(volume_sma) or volume_sma <= 0:
                    continue
                if (current['volume'] / volume_sma
                        >= self.volume_confirmation_mult):
                    signals[symbol] = 'BUY'
        return signals


class MeanReversionPod(Pod):
    """Long-only Bollinger/RSI reversion, UNgated by any regime model.

    Wraps strategies.base.MeanReversionStrategy directly: BUY at the lower
    band with RSI oversold, SELL at the upper band with RSI overbought.
    Where the Renaissance mean-reversion book goes dark outside the
    mean-reverting regime, this pod trades every day — the Citadel desk's
    drawdown cuts are its safety valve instead.
    """

    def __init__(self):
        super().__init__(key='mean_reversion', name='Mean-Reversion Pod')
        self._strategy = MeanReversionStrategy()

    def generate_signals(self, all_data: Dict[str, pd.DataFrame],
                         date) -> Dict[str, str]:
        signals: Dict[str, str] = {}
        for symbol in sorted(all_data):
            frame = all_data[symbol]
            if not self._has_bar_today(frame, date):
                continue
            if not self._has_columns(frame, _MEAN_REVERSION_COLUMNS):
                continue
            signal = self._strategy.generate_signals(frame, _stock(symbol))
            if signal in ('BUY', 'SELL'):
                signals[symbol] = signal
        return signals


class StatArbPod(Pod):
    """Long/short cross-sectional stat-arb (mirrors the Renaissance book).

    Owns a WalkForwardController around GradientBoostingModel (train 252 /
    refit 21 / min 120 by default). On rebalance days (Mondays or refit
    days) it ranks today's scored symbols, targets LONG the top
    max(1, n//5) and SHORT the bottom max(1, n//5), and emits closes for
    holdings that left their side plus entries for new targets. Before the
    first fit — or when the model degrades to no scores — it emits
    NOTHING and keeps its current target book (fail-safe: no information,
    no turnover).

    The pod's target book (_desired) tracks what it has EMITTED entries
    for; the desk remains the source of truth for actual positions. A
    side flip emits only the close — the entry is re-derived at the next
    rebalance once the close has had a chance to fill.
    """

    def __init__(self, controller: Optional[WalkForwardController] = None):
        super().__init__(
            key='stat_arb', name='Stat-Arb Pod',
            controller=(controller if controller is not None
                        else WalkForwardController(GradientBoostingModel(),
                                                   train_window_days=252,
                                                   refit_every_days=21,
                                                   min_train_days=120)))
        #: symbol -> 'long' | 'short' target book (entries emitted so far).
        self._desired: Dict[str, str] = {}

    def generate_signals(self, all_data: Dict[str, pd.DataFrame],
                         date) -> Dict[str, str]:
        controller = self.controller
        assert controller is not None
        refit_today = controller.maybe_refit(all_data, date)
        is_rebalance_day = (pd.Timestamp(date).weekday() == 0) or refit_today
        if not is_rebalance_day:
            return {}

        scores = controller.predict(all_data, date)
        if not scores:  # None (never fitted) or {} (model degraded)
            return {}

        ranked = sorted(
            ((symbol, score) for symbol, score in scores.items()
             if self._has_bar_today(all_data.get(symbol), date)),
            key=lambda item: (-item[1], item[0]))
        if not ranked:
            return {}

        top_n = max(1, len(ranked) // 5)
        longs = [symbol for symbol, _ in ranked[:top_n]]
        # Bottom-N, excluding anything already designated long (tiny
        # universes can overlap; the top ranking wins).
        shorts = [symbol for symbol, _ in ranked[-top_n:]
                  if symbol not in longs]
        desired = {symbol: 'long' for symbol in longs}
        desired.update({symbol: 'short' for symbol in shorts})

        signals: Dict[str, str] = {}
        new_book: Dict[str, str] = {}

        # Closes: holdings that left their side of the ranking (including
        # side flips, which close now and re-enter at the next rebalance).
        for symbol in sorted(self._desired):
            direction = self._desired[symbol]
            if desired.get(symbol) == direction:
                new_book[symbol] = direction
            else:
                signals[symbol] = 'SELL' if direction == 'long' else 'COVER'

        # Entries: new targets (a flipping symbol waits for its close).
        for symbol in sorted(desired):
            if symbol in signals or symbol in new_book:
                continue
            direction = desired[symbol]
            signals[symbol] = 'BUY' if direction == 'long' else 'SHORT'
            new_book[symbol] = direction

        self._desired = new_book
        return signals
