"""
Renaissance Desk — HMM regime detection driving three capital books.

Books (capital weights are fractions of the DESK'S capital, constructor-
tunable, validated to sum <= 1.0):

    mean_reversion (0.4) — regime-conditioned short-horizon mean
        reversion. Active ONLY while the fitted regime model reports
        P(mean_reverting) above `mr_prob_threshold`. Per symbol,
        z = (3d return - 60d rolling mean of 3d returns) / 60d rolling
        std: LONG below -`mr_entry_z`, SHORT above +`mr_entry_z`. Exits:
        position age >= `mr_max_age_days` trading days, z crossing 0, or
        the regime leaving mean_reverting (the whole book flattens with
        one note).
    stat_arb (0.4) — cross-sectional gradient-boosting scores via the
        desk's own walk-forward controller: LONG the top max(1, n//5)
        ranked symbols, SHORT the bottom max(1, n//5). To limit turnover
        the book rebalances ONLY on Mondays or on stat-arb refit days.
    pairs (0.2) — Engle-Granger cointegration pairs (refit every 63
        days): enter when |z| > `pairs_entry_z` (short the rich leg, long
        the cheap leg), exit when |z| < `pairs_exit_z` or the pair
        vanishes at a refit.

Book/position bookkeeping: each open position is tagged with its owning
book; a symbol claimed by one book is off-limits to the others (positions
are keyed by asset, so two books cannot hold opposing exposure in one
symbol). Intents fill at the NEXT bar's open — entry ages count from the
day the intent was emitted, and tracking for intents that never fill (or
positions closed externally, e.g. by the shared stop-loss) is reconciled
away after 2 trading days. A pairs leg lost that way marks the pair
broken and the surviving leg is closed.

ORPHAN SWEEP: the engine holds a pending intent for up to 5 trading days
(MAX_PENDING_DAYS), longer than the 2-day reconcile grace — so an entry
can FILL AFTER its tracking was reconciled away. Such an untracked
position would otherwise be invisible to every book-level exit (regime
flatten, rebalance, pair close), block re-entry through _symbol_is_free
forever, and for a pairs leg leave a naked unhedged leg. _reconcile_books
therefore sweeps daily: any portfolio position in a desk-traded symbol
that no book tracks is closed immediately (SELL/COVER, 'risk' note tagged
with the originating book).

MARGIN IS NOT MODELED: shorts use a cash-account approximation (proceeds
held as cash); live shorting is gated to Phase 9.

All three models run through WalkForwardController — the leakage
chokepoint — and every fit they record is tagged with its model name in
walk_forward_fits (additive C3 key). The regime posterior per day is
exposed as `regime_series` (contract C5); every book-specific note
carries data.book (contract C7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent
from desks.ml_model import GradientBoostingModel
from desks.pairs import PairsCointegrationModel
from desks.regime import RegimeHMMModel
from desks.walk_forward import WalkForwardController, WalkForwardFit
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Default capital weights per book (fractions of the desk's capital).
DEFAULT_BOOK_WEIGHTS = {
    'mean_reversion': 0.4,
    'stat_arb': 0.4,
    'pairs': 0.2,
}

#: Trading days a tracked entry may stay unfilled/absent before its
#: bookkeeping is dropped (intent emitted day T fills day T+1).
RECONCILE_GRACE_DAYS = 2


@dataclass
class TaggedWalkForwardFit:
    """A WalkForwardFit tagged with the model that produced it (C3+)."""
    fit: WalkForwardFit
    model: str

    def to_dict(self) -> Dict:
        payload = self.fit.to_dict()
        payload['model'] = self.model
        return payload


def _stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


class RenaissanceDesk(Desk):
    """Regime-gated multi-book quant desk (see module docstring)."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 book_weights: Optional[Dict[str, float]] = None,
                 mr_prob_threshold: float = 0.6,
                 mr_entry_z: float = 1.5,
                 mr_max_age_days: int = 5,
                 mr_return_days: int = 3,
                 mr_z_window: int = 60,
                 pairs_entry_z: float = 2.0,
                 pairs_exit_z: float = 0.5,
                 regime_controller: Optional[WalkForwardController] = None,
                 stat_arb_controller: Optional[WalkForwardController] = None,
                 pairs_controller: Optional[WalkForwardController] = None):
        super().__init__(
            key='renaissance',
            name='Renaissance Desk',
            description=('HMM regime detection, regime-conditioned '
                         'short-horizon mean reversion, nonlinear stat-arb, '
                         'and cointegration pairs.'),
            accent='#58a6ff',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )

        weights = dict(DEFAULT_BOOK_WEIGHTS)
        if book_weights is not None:
            unknown = set(book_weights) - set(DEFAULT_BOOK_WEIGHTS)
            if unknown:
                raise ValueError(
                    f"Unknown book(s) {sorted(unknown)}; valid books are "
                    f"{sorted(DEFAULT_BOOK_WEIGHTS)}")
            weights.update(book_weights)
        for book, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Book weight for '{book}' must be >= 0, got {weight}")
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Book weights sum to {total:.4f}; must be <= 1.0")
        self.book_weights = weights

        self.mr_prob_threshold = mr_prob_threshold
        self.mr_entry_z = mr_entry_z
        self.mr_max_age_days = mr_max_age_days
        self.mr_return_days = mr_return_days
        self.mr_z_window = mr_z_window
        self.pairs_entry_z = pairs_entry_z
        self.pairs_exit_z = pairs_exit_z

        self._regime_controller = (
            regime_controller if regime_controller is not None
            else WalkForwardController(RegimeHMMModel(),
                                       train_window_days=252,
                                       refit_every_days=21,
                                       min_train_days=120))
        self._stat_arb_controller = (
            stat_arb_controller if stat_arb_controller is not None
            else WalkForwardController(GradientBoostingModel(),
                                       train_window_days=252,
                                       refit_every_days=21,
                                       min_train_days=120))
        self._pairs_controller = (
            pairs_controller if pairs_controller is not None
            else WalkForwardController(PairsCointegrationModel(),
                                       train_window_days=252,
                                       refit_every_days=63,
                                       min_train_days=120))

        # --- Desk state -------------------------------------------------
        #: C5 series: {'date','state','probs'} per day with a fitted model.
        self._regime_series: List[Dict] = []
        self._current_regime: Optional[Dict] = None
        #: asset -> {'book','direction','entry_day','pair_key'(pairs only)}
        self._book_positions: Dict[Asset, Dict] = {}
        #: symbol -> book that last emitted an entry intent for it; scopes
        #: the orphan sweep to desk-traded symbols (module docstring).
        self._traded_books: Dict[str, str] = {}
        #: (a, b) -> {'beta','direction','entry_day','broken'}
        self._open_pairs: Dict[Tuple[str, str], Dict] = {}
        self._day_index = 0
        self._last_seen_date: Optional[date_type] = None

    # ------------------------------------------------------------------
    # Introspection (C3+ / C5)
    # ------------------------------------------------------------------
    @property
    def walk_forward_fits(self) -> List[TaggedWalkForwardFit]:
        """All three controllers' fits, each tagged with its model name."""
        tagged: List[TaggedWalkForwardFit] = []
        for model_name, controller in self._controllers():
            tagged.extend(TaggedWalkForwardFit(fit=fit, model=model_name)
                          for fit in controller.fits)
        tagged.sort(key=lambda t: (t.fit.fit_date, t.model))
        return tagged

    @property
    def regime_series(self) -> List[Dict]:
        """C5 regime series: entries only for dates with a fitted model."""
        return self._regime_series

    def get_status(self) -> Dict:
        status = super().get_status()
        status['regime'] = (self._current_regime['state']
                            if self._current_regime else None)
        status['book_weights'] = dict(self.book_weights)
        return status

    def _controllers(self):
        return (('regime', self._regime_controller),
                ('stat_arb', self._stat_arb_controller),
                ('pairs', self._pairs_controller))

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """One simulated day: reconcile/sweep the books, refit-if-due,
        update the regime, run the books."""
        self._advance_day(date)
        intents: List[DeskIntent] = []
        intents.extend(self._reconcile_books(portfolio))

        refits = self._refit_models(all_data, date)
        self._update_regime(all_data, date)

        intents.extend(self._mean_reversion_intents(all_data, date, portfolio))
        intents.extend(self._stat_arb_intents(all_data, date, portfolio,
                                              refits['stat_arb']))
        intents.extend(self._pairs_intents(all_data, date, portfolio,
                                           refits['pairs']))
        return intents

    def _advance_day(self, date) -> None:
        current = pd.Timestamp(date).date()
        if self._last_seen_date is not None and current != self._last_seen_date:
            self._day_index += 1
        self._last_seen_date = current

    def _reconcile_books(self, portfolio: PortfolioManager) -> List[DeskIntent]:
        """Drop bookkeeping for entries that never filled or were closed
        externally (e.g. by the shared stop-loss) — a pairs leg lost this
        way marks its pair broken so the surviving leg gets closed — then
        sweep ORPHANS: the engine holds pending intents for up to
        MAX_PENDING_DAYS (5) trading days, longer than the reconcile
        grace, so an entry can fill AFTER its tracking was dropped here.
        Any portfolio position in a desk-traded symbol that no book
        tracks is closed immediately (SELL/COVER with a 'risk' note);
        otherwise it would escape every book-level exit, block re-entry
        via _symbol_is_free forever, and leave a pairs leg unhedged."""
        for asset in sorted(self._book_positions,
                            key=lambda a: a.symbol):
            state = self._book_positions[asset]
            position = portfolio.get_position(asset)
            if position is not None and position.quantity != 0:
                continue
            if self._day_index - state['entry_day'] < RECONCILE_GRACE_DAYS:
                continue  # intent may still be pending its next-open fill
            del self._book_positions[asset]
            pair_key = state.get('pair_key')
            if pair_key is not None and pair_key in self._open_pairs:
                self._open_pairs[pair_key]['broken'] = True
            self.note(
                'info',
                f"Book cleanup: dropped {asset.symbol} from the "
                f"{state['book']} book (entry never filled or position "
                f"closed externally)",
                book=state['book'], symbol=asset.symbol,
                direction=state['direction'])

        # ---- Orphan sweep (desk-traded symbols only) -------------------
        intents: List[DeskIntent] = []
        for symbol in sorted(self._traded_books):
            asset = _stock(symbol)
            if asset in self._book_positions:
                continue  # properly tracked by a book
            position = portfolio.get_position(asset)
            if position is None or position.quantity == 0:
                continue
            book = self._traded_books[symbol]
            direction = 'long' if position.quantity > 0 else 'short'
            intents.append(DeskIntent(
                asset=asset,
                action=self._close_action(direction),
                size_fraction=1.0,
                reason='orphan sweep: untracked position in a desk-traded '
                       'symbol (entry filled after its book tracking was '
                       'reconciled away)'))
            self.note(
                'risk',
                f"Orphan sweep: closing untracked {direction} position of "
                f"{abs(position.quantity)} {symbol} — the entry filled "
                f"after its {book} book tracking was reconciled away",
                book=book, symbol=symbol, direction=direction,
                quantity=float(position.quantity))
        return intents

    # ------------------------------------------------------------------
    # Models: refits + regime
    # ------------------------------------------------------------------
    def _refit_models(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Dict[str, bool]:
        """maybe_refit all three controllers; note each fit with its book."""
        labels = {'regime': 'Regime', 'stat_arb': 'Stat-arb',
                  'pairs': 'Pairs'}
        refits: Dict[str, bool] = {}
        for model_name, controller in self._controllers():
            did_fit = controller.maybe_refit(all_data, date)
            refits[model_name] = did_fit
            if did_fit:
                fit = controller.fits[-1]
                self.note(
                    'model',
                    f"{labels[model_name]} model refit "
                    f"#{len(controller.fits)}: trained on {fit.n_samples} "
                    f"samples ({fit.train_start} .. {fit.train_end})",
                    book=model_name, model=model_name, **fit.to_dict())
        return refits

    def _update_regime(self, all_data: Dict[str, pd.DataFrame], date) -> None:
        """Refresh the regime posterior; record the C5 series entry and
        note state transitions with probabilities."""
        result = self._regime_controller.predict(all_data, date)
        if result is None:  # regime model never fitted yet
            return
        if not result:
            # The controller has fitted but the model has nothing to say
            # ({}: degenerate fit / insufficient features). Fail SAFE to
            # no-regime — the mean-reversion book must go dark rather
            # than keep trading a stale regime estimate.
            self._current_regime = None
            return
        previous = (self._current_regime['state']
                    if self._current_regime else None)
        state = result['state']
        probs = {label: float(p) for label, p in result['probs'].items()}
        self._current_regime = {'state': state, 'probs': probs}
        self._regime_series.append({
            'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
            'state': state,
            'probs': probs,
        })
        if state != previous:
            was = f" (was {previous})" if previous is not None else ""
            self.note(
                'model',
                f"Regime: {state} p={probs[state]:.2f}{was}",
                book='regime', state=state, probs=probs,
                previous_state=previous)

    # ------------------------------------------------------------------
    # Book helpers
    # ------------------------------------------------------------------
    def _close_action(self, direction: str) -> str:
        return 'SELL' if direction == 'long' else 'COVER'

    def _track_entry(self, asset: Asset, **state) -> None:
        """Record a book entry AND remember the symbol as desk-traded so
        the orphan sweep covers it for the rest of the run."""
        self._book_positions[asset] = state
        self._traded_books[asset.symbol] = state['book']

    def _has_bar_today(self, frame: pd.DataFrame, date) -> bool:
        return (frame is not None and not frame.empty
                and frame.index[-1] == pd.Timestamp(date))

    def _symbol_is_free(self, symbol: str,
                        portfolio: PortfolioManager) -> bool:
        """A symbol is enterable only if no book tracks it AND the
        portfolio holds no position in it (any direction)."""
        asset = _stock(symbol)
        if asset in self._book_positions:
            return False
        position = portfolio.get_position(asset)
        return position is None or position.quantity == 0

    # ------------------------------------------------------------------
    # Mean-reversion book
    # ------------------------------------------------------------------
    def _mr_zscore(self, frame: pd.DataFrame) -> Optional[float]:
        """z of today's `mr_return_days`-day return against its
        `mr_z_window`-day rolling mean/std (pandas std, ddof=1)."""
        if frame is None or 'close' not in frame.columns:
            return None
        if len(frame) < self.mr_z_window + self.mr_return_days + 1:
            return None
        ret = frame['close'].pct_change(self.mr_return_days)
        mean = ret.rolling(self.mr_z_window).mean().iloc[-1]
        std = ret.rolling(self.mr_z_window).std().iloc[-1]
        latest = ret.iloc[-1]
        if pd.isna(latest) or pd.isna(mean) or pd.isna(std) or std <= 0:
            return None
        return float((latest - mean) / std)

    def _mean_reversion_intents(self, all_data: Dict[str, pd.DataFrame],
                                date, portfolio: PortfolioManager
                                ) -> List[DeskIntent]:
        weight = self.book_weights['mean_reversion']
        book_assets = {asset: state
                       for asset, state in self._book_positions.items()
                       if state['book'] == 'mean_reversion'}

        prob_mr = (self._current_regime['probs'].get('mean_reverting', 0.0)
                   if self._current_regime else 0.0)
        active = (weight > 0 and self._current_regime is not None
                  and prob_mr > self.mr_prob_threshold)

        intents: List[DeskIntent] = []

        if not active:
            if book_assets:
                # Regime left mean_reverting: flatten the whole book, one note.
                symbols = sorted(asset.symbol for asset in book_assets)
                for asset in sorted(book_assets, key=lambda a: a.symbol):
                    state = book_assets[asset]
                    intents.append(DeskIntent(
                        asset=asset,
                        action=self._close_action(state['direction']),
                        size_fraction=1.0,
                        reason='mean-reversion book flatten: regime exited '
                               'mean_reverting'))
                    del self._book_positions[asset]
                self.note(
                    'signal',
                    f"Mean-reversion book flattened: regime exited "
                    f"mean_reverting (P={prob_mr:.2f} <= "
                    f"{self.mr_prob_threshold:.2f}); closed "
                    f"{', '.join(symbols)}",
                    book='mean_reversion', symbols=symbols,
                    prob_mean_reverting=prob_mr)
            return intents

        # ---- Exits: age, or z crossing 0 -----------------------------
        for asset in sorted(book_assets, key=lambda a: a.symbol):
            state = book_assets[asset]
            frame = all_data.get(asset.symbol)
            z_score = (self._mr_zscore(frame)
                       if self._has_bar_today(frame, date) else None)
            age = self._day_index - state['entry_day']
            crossed = (z_score is not None
                       and ((state['direction'] == 'long' and z_score >= 0.0)
                            or (state['direction'] == 'short'
                                and z_score <= 0.0)))
            if age >= self.mr_max_age_days or crossed:
                reason = (f"z crossed 0 (z={z_score:+.2f})" if crossed
                          else f"max age {self.mr_max_age_days}d reached")
                intents.append(DeskIntent(
                    asset=asset,
                    action=self._close_action(state['direction']),
                    size_fraction=1.0,
                    reason=f"mean-reversion exit: {reason}"))
                del self._book_positions[asset]
                self.note(
                    'signal',
                    f"Mean-reversion exit {asset.symbol}: {reason} "
                    f"(held {age} trading day(s))",
                    book='mean_reversion', symbol=asset.symbol,
                    direction=state['direction'], z=z_score, age_days=age)

        # ---- Entries ---------------------------------------------------
        candidates: List[Tuple[str, float, str]] = []
        for symbol in sorted(all_data):
            frame = all_data[symbol]
            if not self._has_bar_today(frame, date):
                continue
            if not self._symbol_is_free(symbol, portfolio):
                continue
            z_score = self._mr_zscore(frame)
            if z_score is None:
                continue
            if z_score < -self.mr_entry_z:
                candidates.append((symbol, z_score, 'long'))
            elif z_score > self.mr_entry_z:
                candidates.append((symbol, z_score, 'short'))

        if not candidates:
            return intents

        n_signals = len(candidates)
        size_fraction = weight * min(0.25, 1.0 / max(4, n_signals))
        for symbol, z_score, direction in candidates:
            asset = _stock(symbol)
            action = 'BUY' if direction == 'long' else 'SHORT'
            intents.append(DeskIntent(
                asset=asset, action=action, size_fraction=size_fraction,
                reason=(f"mean-reversion {direction}: z={z_score:+.2f} "
                        f"beyond +/-{self.mr_entry_z:.1f} "
                        f"(P(mean_reverting)={prob_mr:.2f})")))
            self._track_entry(asset, book='mean_reversion',
                              direction=direction,
                              entry_day=self._day_index)
            self.note(
                'signal',
                f"Mean-reversion {direction.upper()} {symbol}: "
                f"z={z_score:+.2f} ({self.mr_return_days}d return vs "
                f"{self.mr_z_window}d window), P(mean_reverting)="
                f"{prob_mr:.2f}; size {size_fraction:.1%} of desk capital",
                book='mean_reversion', symbol=symbol, direction=direction,
                z=z_score, size_fraction=size_fraction,
                prob_mean_reverting=prob_mr, n_signals=n_signals)
        return intents

    # ------------------------------------------------------------------
    # Stat-arb book
    # ------------------------------------------------------------------
    def _stat_arb_intents(self, all_data: Dict[str, pd.DataFrame], date,
                          portfolio: PortfolioManager,
                          refit_today: bool) -> List[DeskIntent]:
        weight = self.book_weights['stat_arb']
        if weight <= 0:
            return []
        # Turnover limit: rebalance only on Mondays or on refit days.
        is_rebalance_day = (pd.Timestamp(date).weekday() == 0) or refit_today
        if not is_rebalance_day:
            return []

        scores = self._stat_arb_controller.predict(all_data, date)
        if not scores:  # None (unfitted) or {} (model degraded)
            return []

        # Rank symbols that have a usable bar today; deterministic order.
        ranked = sorted(
            ((symbol, score) for symbol, score in scores.items()
             if self._has_bar_today(all_data.get(symbol), date)),
            key=lambda item: (-item[1], item[0]))
        n_scored = len(ranked)
        if n_scored == 0:
            return []

        top_n = max(1, n_scored // 5)
        longs = [symbol for symbol, _ in ranked[:top_n]]
        # Bottom-N, excluding anything already designated long (tiny
        # universes can overlap; the top ranking wins).
        shorts = [symbol for symbol, _ in ranked[-top_n:]
                  if symbol not in longs]
        desired = {symbol: 'long' for symbol in longs}
        desired.update({symbol: 'short' for symbol in shorts})
        size_fraction = weight * min(0.25, 1.0 / (2 * top_n))

        intents: List[DeskIntent] = []
        book_assets = {asset: state
                       for asset, state in self._book_positions.items()
                       if state['book'] == 'stat_arb'}

        # ---- Exits: holdings that left their side of the ranking ------
        for asset in sorted(book_assets, key=lambda a: a.symbol):
            state = book_assets[asset]
            if desired.get(asset.symbol) == state['direction']:
                continue
            intents.append(DeskIntent(
                asset=asset,
                action=self._close_action(state['direction']),
                size_fraction=1.0,
                reason='stat-arb rebalance: symbol left the '
                       f"{state['direction']} book"))
            del self._book_positions[asset]
            self.note(
                'signal',
                f"Stat-arb exit {asset.symbol}: no longer in the "
                f"{state['direction']} top/bottom-{top_n} ranking",
                book='stat_arb', symbol=asset.symbol,
                direction=state['direction'], score=scores.get(asset.symbol))

        # ---- Entries (a flip waits for the close to fill first) -------
        for symbol in longs + shorts:
            direction = desired[symbol]
            if not self._symbol_is_free(symbol, portfolio):
                continue
            asset = _stock(symbol)
            action = 'BUY' if direction == 'long' else 'SHORT'
            score = scores[symbol]
            intents.append(DeskIntent(
                asset=asset, action=action, size_fraction=size_fraction,
                reason=(f"stat-arb {direction}: score {score:+.4f} ranks in "
                        f"the {'top' if direction == 'long' else 'bottom'} "
                        f"{top_n} of {n_scored}")))
            self._track_entry(asset, book='stat_arb', direction=direction,
                              entry_day=self._day_index)
            self.note(
                'signal',
                f"Stat-arb {direction.upper()} {symbol}: score "
                f"{score:+.4f} ({'top' if direction == 'long' else 'bottom'}"
                f"-{top_n} of {n_scored}); size {size_fraction:.1%} of "
                f"desk capital",
                book='stat_arb', symbol=symbol, direction=direction,
                score=score, size_fraction=size_fraction, top_n=top_n)

        if intents:
            self.note(
                'allocation',
                f"Stat-arb rebalance ({'refit day' if refit_today else 'Monday'}): "
                f"{n_scored} scored, longs {longs}, shorts {shorts}, "
                f"{size_fraction:.1%} per leg",
                book='stat_arb', longs=longs, shorts=shorts, top_n=top_n,
                size_fraction=size_fraction)
        return intents

    # ------------------------------------------------------------------
    # Pairs book
    # ------------------------------------------------------------------
    def _close_pair_legs(self, pair_key: Tuple[str, str],
                         intents: List[DeskIntent], reason: str) -> List[str]:
        """Emit closes for whichever legs of the pair are still tracked."""
        closed: List[str] = []
        for asset in sorted(list(self._book_positions),
                            key=lambda a: a.symbol):
            state = self._book_positions[asset]
            if state.get('pair_key') != pair_key:
                continue
            intents.append(DeskIntent(
                asset=asset,
                action=self._close_action(state['direction']),
                size_fraction=1.0,
                reason=f"pairs exit: {reason}"))
            del self._book_positions[asset]
            closed.append(asset.symbol)
        return closed

    def _pairs_intents(self, all_data: Dict[str, pd.DataFrame], date,
                       portfolio: PortfolioManager,
                       refit_today: bool) -> List[DeskIntent]:
        weight = self.book_weights['pairs']
        intents: List[DeskIntent] = []

        # ---- Broken pairs (a leg vanished externally): close survivor --
        for pair_key in sorted(self._open_pairs):
            if not self._open_pairs[pair_key].get('broken'):
                continue
            closed = self._close_pair_legs(
                pair_key, intents, 'pair broken (a leg closed externally)')
            del self._open_pairs[pair_key]
            self.note(
                'risk',
                f"Pairs book: {pair_key[0]}/{pair_key[1]} broken — a leg "
                f"was closed externally; closing surviving leg(s) "
                f"{', '.join(closed) if closed else '(none open)'}",
                book='pairs', pair=f"{pair_key[0]}/{pair_key[1]}",
                closed_legs=closed)

        prediction = self._pairs_controller.predict(all_data, date)
        if prediction is None:  # model never fitted: nothing else to do
            return intents
        priced = {(p['a'], p['b']): p for p in prediction.get('pairs', [])}

        # ---- Exits: |z| < exit threshold, or vanished at refit ---------
        model_pairs = {(p['a'], p['b'])
                       for p in self._pairs_controller.model.pairs}
        for pair_key in sorted(self._open_pairs):
            info = self._open_pairs[pair_key]
            pair_label = f"{pair_key[0]}/{pair_key[1]}"
            if refit_today and pair_key not in model_pairs:
                closed = self._close_pair_legs(pair_key, intents,
                                               'pair vanished at refit')
                del self._open_pairs[pair_key]
                self.note(
                    'signal',
                    f"Pairs exit {pair_label}: pair vanished at refit; "
                    f"closed {', '.join(closed)}",
                    book='pairs', pair=pair_label, beta=info['beta'],
                    closed_legs=closed)
                continue
            quote = priced.get(pair_key)
            if quote is not None and abs(quote['z']) < self.pairs_exit_z:
                closed = self._close_pair_legs(
                    pair_key, intents,
                    f"|z|={abs(quote['z']):.2f} < {self.pairs_exit_z:.2f}")
                del self._open_pairs[pair_key]
                self.note(
                    'signal',
                    f"Pairs exit {pair_label}: z={quote['z']:+.2f} inside "
                    f"+/-{self.pairs_exit_z:.2f}; closed {', '.join(closed)}",
                    book='pairs', pair=pair_label, beta=info['beta'],
                    z=quote['z'], closed_legs=closed)

        # ---- Entries: |z| > entry threshold ----------------------------
        if weight <= 0:
            return intents
        for quote in prediction.get('pairs', []):
            pair_key = (quote['a'], quote['b'])
            if pair_key in self._open_pairs:
                continue
            z_score = quote['z']
            if abs(z_score) <= self.pairs_entry_z:
                continue
            if not (self._symbol_is_free(quote['a'], portfolio)
                    and self._symbol_is_free(quote['b'], portfolio)):
                continue

            # Equal notional per leg; n_open counts THIS pair as open.
            n_open = len(self._open_pairs) + 1
            leg_fraction = weight * min(0.5, 1.0 / max(2, 2 * n_open))
            if leg_fraction <= 0:
                continue

            if z_score > 0:
                # Spread rich: short the rich leg (a), long the cheap (b).
                legs = ((quote['a'], 'short', 'SHORT'),
                        (quote['b'], 'long', 'BUY'))
                direction = 'short_spread'
            else:
                legs = ((quote['a'], 'long', 'BUY'),
                        (quote['b'], 'short', 'SHORT'))
                direction = 'long_spread'

            pair_label = f"{quote['a']}/{quote['b']}"
            for symbol, leg_direction, action in legs:
                asset = _stock(symbol)
                intents.append(DeskIntent(
                    asset=asset, action=action,
                    size_fraction=leg_fraction,
                    reason=(f"pairs entry {pair_label}: z={z_score:+.2f} "
                            f"beyond +/-{self.pairs_entry_z:.1f}, "
                            f"beta={quote['beta']:.3f}")))
                self._track_entry(asset, book='pairs',
                                  direction=leg_direction,
                                  entry_day=self._day_index,
                                  pair_key=pair_key)
            self._open_pairs[pair_key] = {
                'beta': quote['beta'], 'direction': direction,
                'entry_day': self._day_index, 'broken': False,
            }
            self.note(
                'signal',
                f"Pairs entry {pair_label}: z={z_score:+.2f}, "
                f"beta={quote['beta']:.3f} — "
                f"{'SHORT' if z_score > 0 else 'LONG'} {quote['a']} / "
                f"{'LONG' if z_score > 0 else 'SHORT'} {quote['b']}; "
                f"{leg_fraction:.1%} of desk capital per leg",
                book='pairs', pair=pair_label, a=quote['a'], b=quote['b'],
                beta=quote['beta'], z=z_score, leg_fraction=leg_fraction)
        return intents
