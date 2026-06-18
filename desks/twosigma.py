"""
Two Sigma Desk — a systematic, ML-driven cross-sectional long/short equity
book that showcases the Phase A/B model zoo.

The desk runs ONE leakage-proof alpha signal each day, ranks the scored
universe cross-sectionally, and holds a dollar-balanced book: LONG the top
quantile, SHORT the bottom quantile, sized so gross exposure is roughly
``target_gross`` of desk capital and net exposure is roughly zero. This is
the same decile-book mechanic the Renaissance stat-arb book uses
(``_stat_arb_intents``), distilled into a single-purpose systematic desk.

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
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent
from desks.renaissance import TaggedWalkForwardFit
from desks.walk_forward import WalkForwardController
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: The default single-controller model id (fast, deterministic ensemble).
DEFAULT_MODEL_KEY = 'stacking'

#: Trading days a tracked entry may stay unfilled/absent before its
#: bookkeeping is dropped (intent emitted day T fills day T+1). Mirrors the
#: Renaissance reconcile grace so the orphan-sweep timing is identical.
RECONCILE_GRACE_DAYS = 2


def _stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


class TwoSigmaDesk(Desk):
    """Systematic cross-sectional long/short equity desk (module docstring)."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 controller: Optional[WalkForwardController] = None,
                 model_key: Optional[str] = None,
                 models: Optional[List[str]] = None,
                 quantile: float = 0.2,
                 target_gross: float = 1.0,
                 max_name_size: float = 0.10,
                 min_scored: int = 4):
        super().__init__(
            key='twosigma',
            name='Two Sigma Desk',
            description=('Systematic cross-sectional long/short equity: a '
                         'walk-forward ML model (single stacking ensemble by '
                         'default, optional multi-model committee) ranks the '
                         'universe; long the top quantile, short the bottom, '
                         'dollar-balanced.'),
            accent='#3fb950',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )

        if not (0.0 < quantile <= 0.5):
            raise ValueError(
                f"quantile {quantile} must be in (0, 0.5]")
        if not (0.0 < target_gross <= 1.0):
            raise ValueError(
                f"target_gross {target_gross} must be in (0, 1]")
        if not (0.0 < max_name_size <= 1.0):
            raise ValueError(
                f"max_name_size {max_name_size} must be in (0, 1]")
        if min_scored < 2:
            raise ValueError(
                f"min_scored {min_scored} must be >= 2 "
                f"(need both a long and a short side)")
        self.quantile = quantile
        self.target_gross = target_gross
        self.max_name_size = max_name_size
        self.min_scored = min_scored

        # Controller precedence (documented contract, mirrors Foundation):
        #   1. an explicit ``controller`` wins, untouched — a single
        #      controller (tests inject this);
        #   2. else a ``models`` list builds a COMMITTEE of controllers, one
        #      per zoo id, whose centered scores are averaged;
        #   3. else a ``model_key`` selects one zoo model (default
        #      ``stacking``) wrapped in a default WalkForwardController.
        # The model-registry import is local so the explicit-controller path
        # never pulls in optional model deps.
        self._committee: List[Tuple[str, WalkForwardController]]
        if controller is not None:
            if models is not None or model_key is not None:
                raise ValueError(
                    "Provide either an explicit controller= OR "
                    "model_key=/models=, not both")
            self._committee = [('custom', controller)]
        elif models is not None:
            if not models:
                raise ValueError("models= must list at least one model id")
            from desks.models import build_model
            self._committee = [
                (key, WalkForwardController(build_model(key)))
                for key in models
            ]
        else:
            from desks.models import build_model
            key = model_key if model_key is not None else DEFAULT_MODEL_KEY
            self._committee = [
                (key, WalkForwardController(build_model(key)))
            ]
        #: The model label exposed in notes/status (the committee's ids).
        self._model_label = '+'.join(name for name, _ in self._committee)

        # --- Desk state -------------------------------------------------
        #: asset -> {'direction','entry_day'} for each tracked book leg.
        self._book_positions: Dict[Asset, Dict] = {}
        #: symbol -> True for every symbol the desk has ever entered; scopes
        #: the orphan sweep to desk-traded symbols.
        self._traded_symbols: Dict[str, bool] = {}
        self._day_index = 0
        self._last_seen_date: Optional[date_type] = None

    # ------------------------------------------------------------------
    # Introspection (C3+)
    # ------------------------------------------------------------------
    @property
    def walk_forward_fits(self) -> List[TaggedWalkForwardFit]:
        """Every controller's fits, each tagged with its model name."""
        tagged: List[TaggedWalkForwardFit] = []
        for model_name, controller in self._committee:
            tagged.extend(TaggedWalkForwardFit(fit=fit, model=model_name)
                          for fit in controller.fits)
        tagged.sort(key=lambda t: (t.fit.fit_date, t.model))
        return tagged

    def get_status(self) -> Dict:
        status = super().get_status()
        status['model'] = self._model_label
        status['models'] = [name for name, _ in self._committee]
        status['quantile'] = self.quantile
        status['target_gross'] = self.target_gross
        status['max_name_size'] = self.max_name_size
        longs = sum(1 for s in self._book_positions.values()
                    if s['direction'] == 'long')
        shorts = sum(1 for s in self._book_positions.values()
                     if s['direction'] == 'short')
        status['long_count'] = longs
        status['short_count'] = shorts
        return status

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """One simulated day: reconcile/sweep, refit-if-due, score, rebalance."""
        self._advance_day(date)
        intents: List[DeskIntent] = []

        self._refit_models(all_data, date)
        scores = self._alpha_scores(all_data, date)

        if scores is None:
            # No alpha yet: flatten any open book and hold flat (no opens).
            intents.extend(self._reconcile_book(portfolio, desired={}))
            self.note(
                'model',
                f"No alpha yet ({self._model_label} unfitted): holding the "
                f"book flat, no new opens",
                model=self._model_label)
            return intents

        # Only rank symbols with a usable score AND a bar today; deterministic
        # order (highest score first, ties broken by symbol).
        ranked = sorted(
            ((symbol, score) for symbol, score in scores.items()
             if self._has_bar_today(all_data.get(symbol), date)),
            key=lambda item: (-item[1], item[0]))
        n_scored = len(ranked)
        if n_scored < self.min_scored:
            # Too thin to form a book today: flatten and hold flat.
            intents.extend(self._reconcile_book(portfolio, desired={}))
            self.note(
                'info',
                f"Degrade: only {n_scored} symbol(s) scored with a bar today "
                f"(need {self.min_scored}); no new opens",
                n_scored=n_scored, min_scored=self.min_scored,
                model=self._model_label)
            return intents

        k = max(1, int(self.quantile * n_scored))
        longs = [symbol for symbol, _ in ranked[:k]]
        # Bottom-k, excluding anything already designated long (tiny
        # universes can overlap; the top ranking wins).
        shorts = [symbol for symbol, _ in ranked[-k:] if symbol not in longs]
        rank_of = {symbol: idx for idx, (symbol, _) in enumerate(ranked)}

        desired: Dict[str, str] = {symbol: 'long' for symbol in longs}
        desired.update({symbol: 'short' for symbol in shorts})

        # ONE reconcile against the freshly-desired book: names that keep
        # their side are HELD untouched, names that left their side are closed
        # (a flip waits for the close to fill), dead bookkeeping/orphans swept.
        intents.extend(self._reconcile_book(portfolio, desired=desired))

        # Per-leg size: half of target_gross spread across the side, clamped to
        # max_name_size so a small selected set stays inside the shared
        # position-size cap (else apply_risk blocks every leg and the desk
        # no-ops to a flat book under the default RiskManager).
        k_long = max(1, len(longs))
        k_short = max(1, len(shorts))
        long_size = min(self.max_name_size, (0.5 * self.target_gross) / k_long)
        short_size = min(self.max_name_size,
                         (0.5 * self.target_gross) / k_short)

        opened_longs: List[str] = []
        opened_shorts: List[str] = []
        for symbol in longs:
            if not self._symbol_is_free(symbol, portfolio):
                continue
            asset = _stock(symbol)
            score = scores[symbol]
            intents.append(DeskIntent(
                asset=asset, action='BUY', size_fraction=long_size,
                reason=(f"two-sigma long: score {score:+.4f} ranks "
                        f"#{rank_of[symbol] + 1} of {n_scored} (top {k})")))
            self._track_entry(asset, direction='long')
            opened_longs.append(symbol)
            self.note(
                'signal',
                f"Two-Sigma LONG {symbol}: score {score:+.4f}, rank "
                f"#{rank_of[symbol] + 1}/{n_scored} (top {k}); size "
                f"{long_size:.1%} of desk capital",
                symbol=symbol, direction='long', score=score,
                rank=rank_of[symbol] + 1, n_scored=n_scored,
                size_fraction=long_size, model=self._model_label)

        for symbol in shorts:
            if not self._symbol_is_free(symbol, portfolio):
                continue
            asset = _stock(symbol)
            score = scores[symbol]
            intents.append(DeskIntent(
                asset=asset, action='SHORT', size_fraction=short_size,
                reason=(f"two-sigma short: score {score:+.4f} ranks "
                        f"#{rank_of[symbol] + 1} of {n_scored} (bottom {k})")))
            self._track_entry(asset, direction='short')
            opened_shorts.append(symbol)
            self.note(
                'signal',
                f"Two-Sigma SHORT {symbol}: score {score:+.4f}, rank "
                f"#{rank_of[symbol] + 1}/{n_scored} (bottom {k}); size "
                f"{short_size:.1%} of desk capital",
                symbol=symbol, direction='short', score=score,
                rank=rank_of[symbol] + 1, n_scored=n_scored,
                size_fraction=short_size, model=self._model_label)

        if opened_longs or opened_shorts:
            self.note(
                'allocation',
                f"Two-Sigma rebalance: {n_scored} scored, opened longs "
                f"{opened_longs}, shorts {opened_shorts} (top/bottom {k}); "
                f"long {long_size:.1%} / short {short_size:.1%} per name, "
                f"gross ~{self.target_gross:.0%} of desk capital",
                longs=opened_longs, shorts=opened_shorts, k=k,
                n_scored=n_scored, long_size=long_size, short_size=short_size,
                target_gross=self.target_gross, model=self._model_label)
        return intents

    # ------------------------------------------------------------------
    # Bookkeeping helpers (mirror renaissance)
    # ------------------------------------------------------------------
    def _advance_day(self, date) -> None:
        current = pd.Timestamp(date).date()
        if self._last_seen_date is not None and current != self._last_seen_date:
            self._day_index += 1
        self._last_seen_date = current

    def _close_action(self, direction: str) -> str:
        return 'SELL' if direction == 'long' else 'COVER'

    def _track_entry(self, asset: Asset, direction: str) -> None:
        """Record a book entry AND remember the symbol as desk-traded so the
        orphan sweep covers it for the rest of the run."""
        self._book_positions[asset] = {'direction': direction,
                                        'entry_day': self._day_index}
        self._traded_symbols[asset.symbol] = True

    def _has_bar_today(self, frame: Optional[pd.DataFrame], date) -> bool:
        return (frame is not None and not frame.empty
                and frame.index[-1] == pd.Timestamp(date))

    def _symbol_is_free(self, symbol: str,
                        portfolio: PortfolioManager) -> bool:
        """Enterable only if no book leg tracks it AND the portfolio holds no
        position in it (any direction)."""
        asset = _stock(symbol)
        if asset in self._book_positions:
            return False
        position = portfolio.get_position(asset)
        return position is None or position.quantity == 0

    def _reconcile_book(self, portfolio: PortfolioManager,
                        desired: Dict[str, str]) -> List[DeskIntent]:
        """Close tracked legs no longer wanted, drop stale bookkeeping, and
        orphan-sweep untracked desk-traded positions.

        ``desired`` maps symbol -> wanted direction for THIS day's book. A
        tracked leg is closed when the desired direction differs from the
        one held (it left its side of the ranking, or the book went flat).
        Bookkeeping for legs whose position never filled (or was closed
        externally, e.g. by the shared stop-loss) is dropped after the
        reconcile grace. Finally, any portfolio position in a desk-traded
        symbol that no book leg tracks is closed immediately (the engine
        holds pending intents longer than the grace, so an entry can fill
        after its tracking was reconciled away)."""
        intents: List[DeskIntent] = []

        # ---- Close legs that left their side / drop dead bookkeeping ----
        for asset in sorted(self._book_positions, key=lambda a: a.symbol):
            state = self._book_positions[asset]
            position = portfolio.get_position(asset)
            held = position is not None and position.quantity != 0

            if held:
                if state.get('closing'):
                    # Close already emitted; KEEP tracking (so the orphan
                    # sweep never re-fires on our own in-flight close, and the
                    # name is not re-opened) until the fill flattens it.
                    continue
                want = desired.get(asset.symbol)
                if want == state['direction']:
                    continue  # still wanted on the same side — HELD untouched
                # Left its side (or the book went flat): emit the close ONCE
                # and mark the leg closing. Bookkeeping is retained until the
                # position is actually flat, so the leg is neither re-closed,
                # re-opened, nor orphan-swept while its close is in flight.
                state['closing'] = True
                intents.append(DeskIntent(
                    asset=asset,
                    action=self._close_action(state['direction']),
                    size_fraction=1.0,
                    reason=('two-sigma rebalance: symbol left the '
                            f"{state['direction']} book")))
                self.note(
                    'signal',
                    f"Two-Sigma exit {asset.symbol}: no longer in the "
                    f"{state['direction']} side of the book",
                    symbol=asset.symbol, direction=state['direction'])
                continue

            # Not held (flat).
            if state.get('closing'):
                # Our rebalance close filled — drop the settled leg cleanly.
                del self._book_positions[asset]
                continue
            # Entry never filled — wait out the grace, then drop.
            if self._day_index - state['entry_day'] < RECONCILE_GRACE_DAYS:
                continue
            del self._book_positions[asset]
            self.note(
                'info',
                f"Book cleanup: dropped {asset.symbol} from the "
                f"{state['direction']} book (entry never filled or position "
                f"closed externally)",
                symbol=asset.symbol, direction=state['direction'])

        # ---- Orphan sweep (desk-traded symbols only) -------------------
        for symbol in sorted(self._traded_symbols):
            asset = _stock(symbol)
            if asset in self._book_positions:
                continue  # properly tracked
            position = portfolio.get_position(asset)
            if position is None or position.quantity == 0:
                continue
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
                f"{abs(position.quantity)} {symbol} — the entry filled after "
                f"its book tracking was reconciled away",
                symbol=symbol, direction=direction,
                quantity=float(position.quantity))
        return intents

    # ------------------------------------------------------------------
    # Models: refits + alpha
    # ------------------------------------------------------------------
    def _refit_models(self, all_data: Dict[str, pd.DataFrame], date) -> None:
        """maybe_refit every committee controller; note each fit."""
        for model_name, controller in self._committee:
            if controller.maybe_refit(all_data, date):
                fit = controller.fits[-1]
                self.note(
                    'model',
                    f"{model_name} model refit #{len(controller.fits)}: "
                    f"trained on {fit.n_samples} samples "
                    f"({fit.train_start} .. {fit.train_end})",
                    model=model_name, **fit.to_dict())

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
