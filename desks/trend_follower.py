"""
Trend Follower Desk — MTP-style daily trend following (PLANNING.md §6).

REIMPLEMENTED from public Turtle-style mechanics, NOT the closed Market
Timing Pro product: long only while price is above the 200-day SMA (a
regime auto-off below it), entries on a new N-day closing high, incremental
pyramiding as the trend extends, and a wide ATR-based protective stop. The
shared apply_risk() still owns the daily-loss circuit, the position-size
cap, and its own percentage stop; this desk adds the regime filter and the
ATR stop on top.

Signal-core only: generate_intents reads injected, point-in-time-correct
frames (sliced through `date` by the engine) and emits DeskIntents. No I/O,
no wall-clock — parity with live by construction. walk_forward_fits stays
empty (a parameter-fixed rule has no model to refit, so n_trials=1); the
walk-forward OOS-fold wiring is a later slice step.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Bars of history required before signalling — the 200-SMA needs 200 closes.
MIN_HISTORY_DAYS = 200


class TrendFollowerDesk(Desk):
    """Daily trend follower: 200-SMA regime filter, breakout entries with
    pyramiding, and a wide ATR stop."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 atr_stop_mult: float = 3.5,
                 breakout_window: int = 50,
                 entry_size: float = 0.10,
                 max_units: int = 4):
        if risk_manager is None:
            # The default RiskManager is tuned for a diversified ~10%-per-NAME
            # book. A concentrated, pyramiding trend-follower needs (a) a
            # per-trade cap STRICTLY above its entry unit — a 0.10 unit
            # otherwise lands at 0.1000000002 of capital and the strict '>'
            # position-size check blocks every entry, leaving the book in cash
            # — and (b) a wide price stop so the 200-SMA regime filter and the
            # 3.5*ATR stop are the real exits, not a 2% whipsaw. Cumulative risk
            # is bounded by this desk's own max_exposure (entry_size*max_units).
            risk_manager = RiskManager(max_position_size=entry_size * 1.5,
                                       position_stop_loss=0.50)
        super().__init__(
            key='trend_follower',
            name='Trend Follower Desk',
            description=('MTP-style daily trend following (reimplemented '
                         'Turtle mechanics): long above the 200-SMA, breakout '
                         'entries with pyramiding, wide ATR stop, regime '
                         'auto-off below the 200-SMA.'),
            accent='#2da44e',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )
        self.atr_stop_mult = atr_stop_mult
        self.breakout_window = breakout_window
        self.entry_size = entry_size
        self.max_units = max_units
        # Pyramiding cap as a fraction of desk capital: each add is one
        # entry_size unit, capped at max_units of them.
        self.max_exposure = entry_size * max_units

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """Trend-follow entries/exits for the day.

        Per symbol, in priority order: (1) regime auto-off — exit any long
        once price closes below the 200-SMA; (2) ATR stop — exit when price
        has fallen atr_stop_mult ATRs below the average entry; (3) breakout
        entry / pyramid — add one entry_size unit on a new breakout_window-day
        closing high while the trend is on, capped at max_exposure.
        """
        intents: List[DeskIntent] = []
        current_date = pd.Timestamp(date)
        portfolio_value = portfolio.get_portfolio_value()

        for symbol, data in all_data.items():
            if len(data) < MIN_HISTORY_DAYS:
                continue
            if data.index[-1] != current_date:
                continue  # no bar today; signals would be stale

            current = data.iloc[-1]
            price = current['close']
            sma200 = current['sma_200']
            atr = current['atr']
            if pd.isna(price) or pd.isna(sma200):
                continue  # warmup: 200-SMA not yet defined

            asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
            position = portfolio.get_position(asset)
            qty = position.quantity if position is not None else 0
            holding = qty > 0

            # 1. Regime auto-off: no longs below the 200-SMA.
            if price < sma200:
                if holding:
                    intents.append(DeskIntent(
                        asset=asset, action='SELL', size_fraction=1.0,
                        reason=(f"regime off: close {price:.2f} < "
                                f"SMA200 {sma200:.2f}")))
                    self.note(
                        'signal',
                        f"Regime exit {symbol} @ {price:.2f}: closed below "
                        f"SMA200 {sma200:.2f}",
                        symbol=symbol, close=price, sma_200=sma200)
                continue

            # 2. ATR protective stop (regime on, holding only).
            if holding and not pd.isna(atr) and atr > 0:
                stop = position.avg_entry_price - self.atr_stop_mult * atr
                if price <= stop:
                    intents.append(DeskIntent(
                        asset=asset, action='SELL', size_fraction=1.0,
                        reason=(f"ATR stop: close {price:.2f} <= entry "
                                f"{position.avg_entry_price:.2f} - "
                                f"{self.atr_stop_mult}*ATR {atr:.2f}")))
                    self.note(
                        'risk',
                        f"ATR stop {symbol} @ {price:.2f}: entry "
                        f"{position.avg_entry_price:.2f}, stop {stop:.2f} "
                        f"({self.atr_stop_mult}x ATR {atr:.2f})",
                        symbol=symbol, close=price,
                        entry_price=position.avg_entry_price,
                        stop_price=stop, atr=atr, atr_mult=self.atr_stop_mult)
                    continue

            # 3. Breakout entry / pyramid: new N-day closing high, trend on.
            window = data['close'].iloc[-self.breakout_window:]
            if price < window.max():
                continue  # not a new high — no entry, hold existing

            desk_capital = portfolio_value * self.capital_allocation
            if desk_capital <= 0:
                continue
            exposure = (qty * price) / desk_capital if holding else 0.0
            if exposure >= self.max_exposure - 1e-9:
                continue  # at the pyramiding cap

            intents.append(DeskIntent(
                asset=asset, action='BUY', size_fraction=self.entry_size,
                reason=(f"new {self.breakout_window}-day high {price:.2f} "
                        f"above SMA200 {sma200:.2f}")))
            self.note(
                'signal',
                f"Trend entry/pyramid {symbol} @ {price:.2f}: new "
                f"{self.breakout_window}-day high above SMA200 {sma200:.2f}; "
                f"+{self.entry_size:.0%} (exposure {exposure:.0%} of "
                f"{self.max_exposure:.0%} cap)",
                symbol=symbol, close=price, sma_200=sma200,
                exposure=exposure, entry_size=self.entry_size)

        return intents
