"""VIX Mean Reversion Desk — buy equity panic, exit when vol normalizes.

Economic hypothesis: volatility spikes mean-revert, and the equity drawdowns
that drive them partially retrace as vol normalizes (the volatility-risk-
premium / panic-reversion effect). Expression is LONG-ONLY the trade symbol
(default SPY) — no VIX ETPs (contango bleed, blowup risk), no shorting.

Mechanics (parameters pre-specified in docs/vix_pead_desks_spec.md BEFORE any
backtest; a fixed rule, so walk_forward_fits stays [] and the research gate
runs at n_trials=1):

- SIGNAL ON the day VIX close CROSSES above spike_mult * sma_20(VIX)
  (yesterday below, today at/above). Episode-level entries only — a crossing
  cannot fire on consecutive days, so there is no averaging down.
- ENTRY: BUY trade_symbol at the next open (engine T+1), sized by the
  reverse-martingale overlay (base_entry pressed up after winning episodes,
  reset on a loss — risk-shaping, not alpha).
- EXIT: VIX close <= sma_20(VIX) (spike resolved) OR max_hold_days trading
  days in the position, whichever first. A stalled exit (pending SELL dropped
  by the engine after MAX_PENDING_DAYS without a usable bar) is re-emitted.
- STOP: the shared apply_risk percentage stop (position_stop_loss=0.10) —
  buying panics can catch falling knives; a stop-out ends the episode and a
  fresh crossing is required to re-enter.

OWNERSHIP: the desk only ever manages a position it opened itself — a fill is
adopted only while the desk's own BUY is pending, so on a shared fund book a
trade_symbol position opened by ANOTHER desk is left untouched (no adopted
entries, no foreign liquidations) and the desk simply stays out while one
exists. Residual fund-mode limitation: SELL closes the FULL shared position
(engine semantics), so if another desk adds to trade_symbol AFTER this desk's
fill, this desk's exit would flatten those shares too — run it standalone or
with a disjoint universe for clean attribution.

Round-trip classification: a stop-out (position gone without a desk exit) is
a LOSS by construction; a desk-emitted exit is classified on the day the
position is observed flat, against that day's OPEN (the T+1 fill's bar), not
the signal-day close.

The VIX symbol is an AUXILIARY, NEVER-TRADED input: this desk emits intents
only for trade_symbol, and the shared apply_risk additionally blocks any
opening intent on '^'-prefixed index levels desk-wide. Run universes must
include both symbols (e.g. ['SPY', '^VIX']); if the VIX frame is absent the
desk degrades to a flat book with one note.

Signal-core only: generate_intents reads injected, point-in-time-correct
frames and emits DeskIntents. No I/O, no wall-clock — parity by construction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent
from desks.sizing import ReverseMartingaleSizer
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Bars of VIX history required before signalling (sma_20 warmup + slack).
MIN_HISTORY_DAYS = 60

#: Give up on an unfilled BUY after this many observed flat days (the engine
#: itself drops a pending intent after 5 bar-less days; 7 adds slack).
PENDING_BUY_MAX_DAYS = 7

#: Re-emit a desk exit if the position is still open this many days after the
#: exit intent went out (covers the engine dropping a pending SELL).
EXIT_RETRY_DAYS = 3


class VixReversionDesk(Desk):
    """Long-only SPY mean-reversion on VIX spike episodes, reverse-martingale
    sized."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 vix_symbol: str = '^VIX',
                 trade_symbol: str = 'SPY',
                 spike_mult: float = 1.25,
                 base_entry: float = 0.20,
                 max_hold_days: int = 21,
                 sizer_step: float = 0.5,
                 sizer_max_mult: float = 2.0):
        if risk_manager is None:
            # Cap must clear base_entry * sizer_max_mult (0.40) with headroom
            # (the strict '>' size check + float noise — trend-follower
            # lesson); the 10% stop is DELIBERATELY tight: panic entries need
            # a real knife-catch stop, and it doubles as the episode's loss
            # classifier for the reverse-martingale overlay.
            risk_manager = RiskManager(
                max_position_size=base_entry * sizer_max_mult * 1.5,
                position_stop_loss=0.10)
        super().__init__(
            key='vixrevert',
            name='VIX Reversion Desk',
            description=('Volatility mean reversion: buys the equity index '
                         '(long-only SPY) when VIX spikes above '
                         f'{spike_mult:.2f}x its 20-day average, exits when '
                         'the spike resolves or after a time stop; '
                         'reverse-martingale sized. Requires ^VIX as an '
                         'auxiliary (never traded) symbol in the universe.'),
            accent='#bf3989',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )
        self.vix_symbol = vix_symbol
        self.trade_symbol = trade_symbol
        self.spike_mult = spike_mult
        self.base_entry = base_entry
        self.max_hold_days = max_hold_days
        self.sizer = ReverseMartingaleSizer(base_entry, step=sizer_step,
                                            max_mult=sizer_max_mult)
        # Episode state (desk-internal, deterministic).
        self._pending_buy = False                # own BUY emitted, unfilled
        self._pending_age = 0                    # flat days since emission
        self._entry_px: Optional[float] = None   # OUR observed fill price
        self._days_held = 0                      # bars since fill observed
        self._exit_emitted_at: Optional[int] = None  # _days_held at exit emit
        self._warned_no_vix = False

    def _reset_episode(self) -> None:
        self._entry_px = None
        self._days_held = 0
        self._exit_emitted_at = None

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """One VIX-episode state step: classify any completed round-trip,
        then exit on reversion/time-stop, or enter on a fresh spike crossing.

        Emits intents ONLY for trade_symbol — the VIX frame is read-only.
        """
        current_date = pd.Timestamp(date)
        vix = all_data.get(self.vix_symbol)
        spy = all_data.get(self.trade_symbol)
        if vix is None or spy is None:
            if not self._warned_no_vix:
                self.note(
                    'info',
                    f"Flat: universe lacks {self.vix_symbol!r} and/or "
                    f"{self.trade_symbol!r}; this desk needs both (VIX is "
                    f"auxiliary, never traded)",
                    vix_symbol=self.vix_symbol, trade_symbol=self.trade_symbol)
                self._warned_no_vix = True
            return []
        if len(vix) < MIN_HISTORY_DAYS or len(spy) < 2:
            return []
        if vix.index[-1] != current_date or spy.index[-1] != current_date:
            return []  # stale bar on either series; signals would be stale

        vix_close = vix['close'].iloc[-1]
        vix_sma = vix['sma_20'].iloc[-1]
        prev_close = vix['close'].iloc[-2]
        prev_sma = vix['sma_20'].iloc[-2]
        spy_close = spy['close'].iloc[-1]
        if any(pd.isna(v) for v in
               (vix_close, vix_sma, prev_close, prev_sma, spy_close)) \
                or vix_sma <= 0 or prev_sma <= 0:
            return []

        asset = Asset(symbol=self.trade_symbol, asset_type=AssetType.STOCK)
        position = portfolio.get_position(asset)
        has_pos = position is not None and position.quantity > 0

        # --- Round-trip bookkeeping (fills are T+1: observe, don't assume) --
        if has_pos:
            if self._entry_px is None and self._pending_buy:
                # OUR pending BUY filled — adopt it. A position with no
                # pending BUY of ours is another desk's (shared fund book):
                # never adopted, never exited by this desk.
                self._entry_px = position.avg_entry_price
                self._pending_buy = False
                self._pending_age = 0
                self._days_held = 0
                self._exit_emitted_at = None
            if self._entry_px is not None:
                self._days_held += 1
        else:
            if self._entry_px is not None:
                # Our episode just closed.
                if self._exit_emitted_at is None:
                    # Gone without a desk exit: the shared apply_risk stop
                    # took it — a loss by construction.
                    self.sizer.record(False)
                    self.note(
                        'risk',
                        f"Episode closed by stop: streak reset, next entry "
                        f"{self.sizer.size():.0%}",
                        entry_price=self._entry_px,
                        win_streak=self.sizer.win_streak)
                else:
                    # Desk exit filled at THIS morning's open (T+1): classify
                    # against the fill bar, not the signal-day close.
                    fill_basis = spy_close
                    if 'open' in spy.columns:
                        o = spy['open'].iloc[-1]
                        if not pd.isna(o) and o > 0:
                            fill_basis = o
                    won = fill_basis > self._entry_px
                    self.sizer.record(won)
                    self.note(
                        'signal',
                        f"Episode closed @ ~{fill_basis:.2f} vs entry "
                        f"{self._entry_px:.2f}: {'WIN' if won else 'LOSS'}, "
                        f"streak {self.sizer.win_streak}, next entry "
                        f"{self.sizer.size():.0%}",
                        fill_basis=fill_basis, entry_price=self._entry_px,
                        won=won, win_streak=self.sizer.win_streak)
                self._reset_episode()
            if self._pending_buy:
                self._pending_age += 1
                if self._pending_age > PENDING_BUY_MAX_DAYS:
                    self._pending_buy = False    # engine dropped it; move on
                    self._pending_age = 0

        ours = has_pos and self._entry_px is not None

        # --- Exit: spike resolved or time stop (our position only) -----
        if ours:
            if self._exit_emitted_at is None:
                reverted = vix_close <= vix_sma
                timed_out = self._days_held >= self.max_hold_days
                if reverted or timed_out:
                    self._exit_emitted_at = self._days_held
                    why = ('VIX reverted to its 20d mean' if reverted
                           else f'{self.max_hold_days}-day time stop')
                    self.note(
                        'signal',
                        f"Exit {self.trade_symbol} signalled @ "
                        f"{spy_close:.2f} ({why}); outcome classified at "
                        f"the fill",
                        close=spy_close, entry_price=self._entry_px,
                        vix=vix_close, vix_sma=vix_sma,
                        days_held=self._days_held)
                    return [DeskIntent(
                        asset=asset, action='SELL', size_fraction=1.0,
                        reason=(f"vix-revert exit: {why} (VIX "
                                f"{vix_close:.2f} vs SMA20 {vix_sma:.2f})"))]
                return []
            if self._days_held - self._exit_emitted_at >= EXIT_RETRY_DAYS:
                # Pending SELL was dropped (no usable bar for 5 days) —
                # re-emit rather than ride a dead exit forever.
                self._exit_emitted_at = self._days_held
                return [DeskIntent(
                    asset=asset, action='SELL', size_fraction=1.0,
                    reason='vix-revert exit retry: prior exit never filled')]
            return []
        if has_pos:
            return []  # another desk's position — stay out entirely

        # --- Entry: fresh spike crossing, flat and nothing pending ------
        if self._pending_buy:
            return []
        ratio = vix_close / vix_sma
        prev_ratio = prev_close / prev_sma
        crossed = prev_ratio < self.spike_mult <= ratio
        if not crossed:
            return []
        size = self.sizer.size()
        self._pending_buy = True
        self._pending_age = 0
        self.note(
            'signal',
            f"VIX spike entry: VIX {vix_close:.2f} crossed above "
            f"{self.spike_mult:.2f}x SMA20 {vix_sma:.2f} (ratio {ratio:.2f}); "
            f"BUY {self.trade_symbol} {size:.0%} (streak "
            f"{self.sizer.win_streak})",
            vix=vix_close, vix_sma=vix_sma, ratio=ratio,
            size_fraction=size, win_streak=self.sizer.win_streak)
        return [DeskIntent(
            asset=asset, action='BUY', size_fraction=size,
            reason=(f"vix-revert entry: VIX {vix_close:.2f} >= "
                    f"{self.spike_mult:.2f}x SMA20 {vix_sma:.2f}"))]
