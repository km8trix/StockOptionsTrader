"""Tests for desks.renaissance.RenaissanceDesk.

Unit tests drive generate_intents day by day with STUB walk-forward
models (scripted regimes/scores/pair quotes — every number controlled) to
pin down the book mechanics: regime gating, flatten-on-regime-exit,
top/bottom-N stat-arb, pairs z thresholds, sizing, and note bookkeeping
(contract C7). The end-to-end test runs the real BacktestEngine with the
real HMM/GBM/cointegration models on synthetic data and checks the C3 +
C5 + C7 report shapes. Offline, seeded, deterministic.
"""

from __future__ import annotations

import contextlib
import json

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.foundation import FoundationDesk
from desks.ml_model import GradientBoostingModel
from desks.regime import REGIME_LABELS
from desks.renaissance import RenaissanceDesk
from desks.walk_forward import WalkForwardController, WalkForwardModel
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


@contextlib.contextmanager
def _patched_fetch(frames):
    """Monkeypatch MarketDataHandler.fetch_stock_data to serve `frames`
    (offline, deterministic) for the duration of the context."""
    original = MarketDataHandler.fetch_stock_data

    def fake_fetch(self, symbol, start_date, end_date):
        return frames.get(symbol, pd.DataFrame())

    MarketDataHandler.fetch_stock_data = fake_fetch
    try:
        yield
    finally:
        MarketDataHandler.fetch_stock_data = original


def probs_for(state: str, p: float) -> dict:
    others = [label for label in REGIME_LABELS if label != state]
    rest = (1.0 - p) / 2
    return {'state': state,
            'probs': {state: p, others[0]: rest, others[1]: rest}}


# ----------------------------------------------------------------------
# Stub walk-forward models (fit immediately via min_train_days=1)
# ----------------------------------------------------------------------
class StubRegimeModel(WalkForwardModel):
    """Scripted regime by date; {} for unscripted dates (no-regime)."""

    def __init__(self, schedule=None):
        self.schedule = {pd.Timestamp(d): r
                         for d, r in (schedule or {}).items()}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        return self.schedule.get(pd.Timestamp(date), {})


class StubScoreModel(WalkForwardModel):
    """Fixed per-symbol stat-arb scores (restricted to present symbols)."""

    def __init__(self, scores=None):
        self.scores = scores or {}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        return {symbol: score for symbol, score in self.scores.items()
                if symbol in data}


class DateScoreModel(WalkForwardModel):
    """Per-date stat-arb scores: scripted scores by date, falling back to
    `default` for unscripted dates (restricted to present symbols). Lets a
    test re-rank the universe between rebalances to exercise the rank-
    stability band, minimum-hold period, and flip rules deterministically.
    """

    def __init__(self, schedule=None, default=None):
        self.schedule = {pd.Timestamp(d): scores
                         for d, scores in (schedule or {}).items()}
        self.default = default or {}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        scores = self.schedule.get(pd.Timestamp(date), self.default)
        return {symbol: score for symbol, score in scores.items()
                if symbol in data}


class StubPairsModel(WalkForwardModel):
    """Scripted pair quotes by date; exposes .pairs like the real model."""

    def __init__(self, schedule=None, pairs=None):
        self.schedule = {pd.Timestamp(d): quotes
                         for d, quotes in (schedule or {}).items()}
        self.pairs = pairs if pairs is not None else []

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        return {'pairs': self.schedule.get(pd.Timestamp(date), [])}


def make_desk(regime_schedule=None, scores=None, pairs_schedule=None,
              model_pairs=None, pairs_refit_every=21, **kwargs):
    kwargs.setdefault('risk_manager', RiskManager(
        max_position_size=1.0, max_daily_loss=0.99,
        position_stop_loss=0.90))  # wide: book mechanics stay pure
    return RenaissanceDesk(
        regime_controller=WalkForwardController(
            StubRegimeModel(regime_schedule), min_train_days=1),
        stat_arb_controller=WalkForwardController(
            StubScoreModel(scores), min_train_days=1),
        pairs_controller=WalkForwardController(
            StubPairsModel(pairs_schedule, pairs=model_pairs),
            min_train_days=1, refit_every_days=pairs_refit_every),
        **kwargs)


def make_stat_arb_desk(score_model, stat_arb_refit_every=21, **kwargs):
    """A desk wired with a custom (date-aware) stat-arb score model and no
    regime/pairs activity, for the stat-arb turnover-control regressions.
    New stat-arb params (band mult, min-hold, conviction) pass via kwargs."""
    kwargs.setdefault('risk_manager', RiskManager(
        max_position_size=1.0, max_daily_loss=0.99,
        position_stop_loss=0.90))  # wide: book mechanics stay pure
    return RenaissanceDesk(
        regime_controller=WalkForwardController(
            StubRegimeModel(None), min_train_days=1),
        stat_arb_controller=WalkForwardController(
            score_model, min_train_days=1,
            refit_every_days=stat_arb_refit_every),
        pairs_controller=WalkForwardController(
            StubPairsModel(None, pairs=[]), min_train_days=1),
        **kwargs)


# ----------------------------------------------------------------------
# Synthetic frames + a one-desk driver (no engine; fills are manual)
# ----------------------------------------------------------------------
def noisy_frame(n=90, seed=5, start='2023-01-02',
                drop_window=None, drop=-0.02,
                overrides=None) -> pd.DataFrame:
    """Low-noise random walk; optional run of equal daily drops (so the
    3d-return z-score plunges deterministically) plus per-day overrides."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.002, n)
    if drop_window is not None:
        lo, hi = drop_window
        rets[lo:hi] = drop
    for idx, value in (overrides or {}).items():
        rets[idx] = value
    close = 100.0 * np.cumprod(1.0 + rets)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': np.full(n, 500_000.0),
    }, index=index)


def drive(desk, frames, dates, portfolio):
    """Call generate_intents once per date on expanding slices (the same
    contract the engine honors); returns {date: [intents]}."""
    by_date = {}
    for date in dates:
        sliced = {symbol: frame[frame.index <= date]
                  for symbol, frame in frames.items()}
        sliced = {symbol: frame for symbol, frame in sliced.items()
                  if not frame.empty}
        desk.set_clock(date)
        by_date[date] = desk.generate_intents(sliced, date, portfolio)
    return by_date


class TestConstructionAndStatus:
    def test_default_book_weights_sum_within_allocation(self):
        desk = RenaissanceDesk()
        assert desk.book_weights == {'mean_reversion': 0.4,
                                     'stat_arb': 0.4, 'pairs': 0.2}
        assert sum(desk.book_weights.values()) <= 1.0

    def test_overweight_books_raise(self):
        with pytest.raises(ValueError, match='sum'):
            RenaissanceDesk(book_weights={'mean_reversion': 0.6,
                                          'stat_arb': 0.6})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            RenaissanceDesk(book_weights={'pairs': -0.1})

    def test_unknown_book_raises(self):
        with pytest.raises(ValueError, match='Unknown book'):
            RenaissanceDesk(book_weights={'vol_arb': 0.2})

    def test_get_status_includes_current_regime(self):
        dates = pd.bdate_range('2023-01-02', periods=2)
        desk = make_desk(regime_schedule={
            dates[0]: probs_for('trending', 0.8)})
        assert desk.get_status()['regime'] is None
        frames = {'AAA': noisy_frame(n=2)}
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert desk.get_status()['regime'] == 'trending'


class TestRegimeSeriesAndNotes:
    def test_regime_series_and_transition_notes(self):
        frames = {'AAA': noisy_frame(n=10, seed=3)}
        dates = frames['AAA'].index[:4]
        schedule = {
            dates[0]: probs_for('trending', 0.80),
            dates[1]: probs_for('trending', 0.70),
            dates[2]: probs_for('mean_reverting', 0.81),
            # dates[3] unscripted: {} -> no series entry, regime held.
        }
        desk = make_desk(regime_schedule=schedule)
        portfolio = PortfolioManager(100000.0)
        drive(desk, frames, dates, portfolio)

        series = desk.regime_series
        assert [entry['date'] for entry in series] == \
            [d.strftime('%Y-%m-%d') for d in dates[:3]]
        for entry in series:
            assert set(entry) == {'date', 'state', 'probs'}
            assert entry['state'] in REGIME_LABELS
            assert set(entry['probs']) == set(REGIME_LABELS)

        transitions = [n for n in desk.notes if n.category == 'model'
                       and n.message.startswith('Regime:')]
        assert len(transitions) == 2  # None->trending, trending->mr
        assert 'Regime: trending p=0.80' in transitions[0].message
        assert 'Regime: mean_reverting p=0.81 (was trending)' \
            in transitions[1].message
        for note in transitions:
            assert note.data['book'] == 'regime'

    def test_walk_forward_fits_are_tagged_per_model(self):
        frames = {'AAA': noisy_frame(n=5)}
        dates = frames['AAA'].index[:1]
        desk = make_desk()
        drive(desk, frames, dates, PortfolioManager(100000.0))

        fits = desk.walk_forward_fits
        assert sorted(fit.model for fit in fits) == \
            ['pairs', 'regime', 'stat_arb']
        for fit in fits:
            payload = fit.to_dict()
            assert set(payload) == {'fit_date', 'train_start', 'train_end',
                                    'n_samples', 'model'}
        refit_notes = [n for n in desk.notes if n.category == 'model'
                       and 'refit' in n.message]
        assert sorted(n.data['book'] for n in refit_notes) == \
            ['pairs', 'regime', 'stat_arb']


class TestMeanReversionBook:
    DROP_START = 73  # first of three -2% days: z trips that very day

    def _frames(self, **kwargs):
        kwargs.setdefault('drop_window', (self.DROP_START,
                                          self.DROP_START + 3))
        return {'MRX': noisy_frame(n=90, seed=5, **kwargs)}

    def _enter(self, desk, frames, portfolio):
        """Drive up to the trigger day, assert the single entry, and
        simulate the engine's next-open fill."""
        dates = frames['MRX'].index
        out = drive(desk, frames, dates[70:self.DROP_START + 1], portfolio)
        entry_intents = out[dates[self.DROP_START]]
        assert [i.action for i in entry_intents] == ['BUY']
        portfolio.add_position(Position(
            asset=stock('MRX'), quantity=20, avg_entry_price=95.0,
            current_price=95.0, timestamp=dates[self.DROP_START + 1]))
        return dates

    def test_entry_only_when_prob_exceeds_threshold(self):
        frames = self._frames()
        dates = frames['MRX'].index
        trigger_date = dates[self.DROP_START]

        # P(mean_reverting) = 0.55 <= 0.6: the book must stay dark.
        gated = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.55) for d in dates})
        out = drive(gated, frames, dates[70:self.DROP_START + 1],
                    PortfolioManager(100000.0))
        assert all(intents == [] for intents in out.values())
        assert [n for n in gated.notes
                if n.data.get('book') == 'mean_reversion'] == []

        # P(mean_reverting) = 0.9 > 0.6: the same drop produces a LONG.
        active = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.90) for d in dates})
        out = drive(active, frames, dates[70:self.DROP_START + 1],
                    PortfolioManager(100000.0))
        intents = out[trigger_date]
        assert len(intents) == 1
        intent = intents[0]
        assert intent.action == 'BUY'
        assert intent.asset == stock('MRX')
        # size = w_mr * min(0.25, 1/max(4, 1)) = 0.4 * 0.25.
        assert intent.size_fraction == pytest.approx(0.4 * 0.25)

        entry_notes = [n for n in active.notes
                       if n.data.get('book') == 'mean_reversion']
        assert len(entry_notes) == 1
        assert entry_notes[0].data['z'] < -1.5
        assert entry_notes[0].data['direction'] == 'long'

    def test_short_side_entry_on_positive_z(self):
        frames = self._frames(drop=+0.02)  # melt-up: z > +1.5 -> SHORT
        dates = frames['MRX'].index
        desk = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.90) for d in dates})
        out = drive(desk, frames, dates[70:self.DROP_START + 1],
                    PortfolioManager(100000.0))
        intents = out[dates[self.DROP_START]]
        assert [i.action for i in intents] == ['SHORT']
        assert desk.notes[-1].data['z'] > 1.5

    def _two_symbol_frames(self):
        # Two LONG signals triggering the same day with different |z|:
        # MRB's deeper drop (-5%) gives a larger |z| than MRA's (-2%).
        win = (self.DROP_START, self.DROP_START + 3)
        return {
            'MRA': noisy_frame(n=90, seed=5, drop_window=win, drop=-0.02),
            'MRB': noisy_frame(n=90, seed=5, drop_window=win, drop=-0.05),
        }

    def test_conviction_sizing_scales_by_z_excess(self):
        # CONVICTION SIZING: the stronger |z|-excess signal (MRB) sizes
        # larger than the weaker (MRA); the strongest sits at the per-entry
        # cap and no entry exceeds it (book gross stays <= the old flat
        # total). With 2 signals the cap is 0.4 * min(0.25, 1/4) = 0.1.
        frames = self._two_symbol_frames()
        dates = frames['MRA'].index
        desk = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.90) for d in dates})
        out = drive(desk, frames, dates[70:self.DROP_START + 1],
                    PortfolioManager(100000.0))
        intents = out[dates[self.DROP_START]]
        sizes = {i.asset.symbol: i.size_fraction for i in intents}
        zs = {n.data['symbol']: n.data['z'] for n in desk.notes
              if n.data.get('book') == 'mean_reversion'
              and n.data.get('symbol')}
        assert abs(zs['MRB']) > abs(zs['MRA'])     # MRB the stronger signal
        assert sizes['MRB'] > sizes['MRA']          # ... gets the larger size
        per_entry_cap = 0.4 * min(0.25, 1.0 / max(4, len(intents)))
        assert sizes['MRB'] == pytest.approx(per_entry_cap)  # strongest at cap
        for size in sizes.values():
            assert 0 < size <= per_entry_cap + 1e-12
        assert sum(sizes.values()) <= len(intents) * per_entry_cap + 1e-12

    def test_flat_sizing_recovers_old_per_entry_size(self):
        # RECOVERABILITY: mr_size_by_conviction=False gives every entry the
        # flat per-entry cap exactly, reproducing the pre-Phase-E sizing.
        frames = self._two_symbol_frames()
        dates = frames['MRA'].index
        desk = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.90) for d in dates},
            mr_size_by_conviction=False)
        out = drive(desk, frames, dates[70:self.DROP_START + 1],
                    PortfolioManager(100000.0))
        intents = out[dates[self.DROP_START]]
        per_entry_cap = 0.4 * min(0.25, 1.0 / max(4, len(intents)))
        for intent in intents:
            assert intent.size_fraction == pytest.approx(per_entry_cap)

    def test_exit_when_z_crosses_zero(self):
        # A +6% bounce 3 days after the trigger flips the 3d return
        # positive, so z crosses 0 at age 3 — before the age-5 exit.
        frames = self._frames(overrides={self.DROP_START + 3: 0.06})
        desk = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.90)
            for d in frames['MRX'].index})
        portfolio = PortfolioManager(100000.0)
        dates = self._enter(desk, frames, portfolio)

        out = drive(desk, frames,
                    dates[self.DROP_START + 1:self.DROP_START + 4],
                    portfolio)
        exits = [(d, i) for d, intents in out.items() for i in intents]
        assert len(exits) == 1
        exit_date, exit_intent = exits[0]
        assert exit_intent.action == 'SELL'
        assert 'z crossed 0' in exit_intent.reason
        assert list(dates).index(exit_date) - self.DROP_START == 3
        exit_notes = [n for n in desk.notes
                      if n.data.get('book') == 'mean_reversion'
                      and 'exit' in n.message.lower()]
        assert len(exit_notes) == 1
        assert exit_notes[0].data['z'] >= 0

        # Simulate the engine's next-open exit fill: no duplicate close
        # and no orphan sweep may follow (a fresh ENTRY is legitimate —
        # the symbol is free again and z is still stretched).
        portfolio.remove_position(stock('MRX'))
        out_after = drive(desk, frames,
                          dates[self.DROP_START + 4:self.DROP_START + 5],
                          portfolio)
        assert [i for intents in out_after.values() for i in intents
                if i.action in ('SELL', 'COVER')] == []

    def test_exit_at_max_age_when_z_stays_stretched(self):
        # The drop keeps going: z never crosses 0, so age (5d) exits.
        frames = {'MRX': noisy_frame(
            n=95, seed=5,
            drop_window=(self.DROP_START, self.DROP_START + 14))}
        desk = make_desk(regime_schedule={
            d: probs_for('mean_reverting', 0.90)
            for d in frames['MRX'].index})
        portfolio = PortfolioManager(100000.0)
        dates = self._enter(desk, frames, portfolio)

        out = drive(desk, frames,
                    dates[self.DROP_START + 1:self.DROP_START + 6],
                    portfolio)
        exits = [(d, i) for d, intents in out.items() for i in intents]
        assert len(exits) == 1
        exit_date, exit_intent = exits[0]
        assert exit_intent.action == 'SELL'
        assert 'max age' in exit_intent.reason
        assert list(dates).index(exit_date) - self.DROP_START == 5

        # Simulate the engine's next-open exit fill: no duplicate close
        # and no orphan sweep may follow (a fresh ENTRY is legitimate —
        # the symbol is free again and z is still stretched).
        portfolio.remove_position(stock('MRX'))
        out_after = drive(desk, frames,
                          dates[self.DROP_START + 6:self.DROP_START + 7],
                          portfolio)
        assert [i for intents in out_after.values() for i in intents
                if i.action in ('SELL', 'COVER')] == []

    def test_flatten_on_regime_exit_with_one_note(self):
        # The drop persists (no z-cross) and the regime flips to
        # trending 2 days after entry: the whole book flattens at once.
        frames = {'MRX': noisy_frame(
            n=90, seed=5,
            drop_window=(self.DROP_START, self.DROP_START + 6))}
        dates = frames['MRX'].index
        flip_at = self.DROP_START + 2
        schedule = {
            date: probs_for('mean_reverting' if i < flip_at else 'trending',
                            0.9)
            for i, date in enumerate(dates)}
        desk = make_desk(regime_schedule=schedule)
        portfolio = PortfolioManager(100000.0)
        self._enter(desk, frames, portfolio)

        out = drive(desk, frames, dates[self.DROP_START + 1:flip_at + 1],
                    portfolio)
        flatten_date = dates[flip_at]
        assert [i.action for i in out[flatten_date]] == ['SELL']
        assert out[dates[self.DROP_START + 1]] == []  # held until the flip

        flatten_notes = [n for n in desk.notes
                         if 'flattened' in n.message]
        assert len(flatten_notes) == 1  # the whole book, ONE note
        assert flatten_notes[0].data['book'] == 'mean_reversion'
        assert flatten_notes[0].data['symbols'] == ['MRX']
        assert flatten_notes[0].timestamp == flatten_date


class TestStatArbBook:
    def _universe(self, n_symbols=10, n_days=10):
        return {f'S{i:02d}': noisy_frame(n=n_days, seed=100 + i)
                for i in range(n_symbols)}

    def _scores(self, n_symbols=10):
        # S00 highest .. S09 lowest, strictly monotone.
        return {f'S{i:02d}': 0.9 - 0.2 * i for i in range(n_symbols)}

    def test_long_short_counts_match_top_bottom_n(self):
        frames = self._universe()
        dates = frames['S00'].index  # starts Monday 2023-01-02
        desk = make_desk(scores=self._scores())
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))

        intents = out[dates[0]]  # refit day -> rebalance
        buys = [i for i in intents if i.action == 'BUY']
        shorts = [i for i in intents if i.action == 'SHORT']
        top_n = max(1, 10 // 5)
        assert len(buys) == top_n == 2
        assert len(shorts) == top_n == 2
        assert sorted(i.asset.symbol for i in buys) == ['S00', 'S01']
        assert sorted(i.asset.symbol for i in shorts) == ['S08', 'S09']

        # Conviction-sized by default (stat_arb_size_by_conviction=True):
        # each leg is bounded by the per-leg cap and the book gross is
        # bounded by the old flat total (n_entries * per_leg_cap).
        per_leg_cap = 0.4 * min(0.25, 1.0 / (2 * top_n))  # 0.1
        for intent in intents:
            assert 0 < intent.size_fraction <= per_leg_cap + 1e-12
        gross = sum(i.size_fraction for i in intents)
        assert gross <= len(intents) * per_leg_cap + 1e-12

        book_notes = [n for n in desk.notes
                      if n.data.get('book') == 'stat_arb']
        assert {n.category for n in book_notes} == {'model', 'signal',
                                                    'allocation'}

    def test_flat_sizing_recovers_old_per_leg_size(self):
        # RECOVERABILITY: stat_arb_size_by_conviction=False reproduces the
        # old flat per-leg size = w_sa * min(0.25, 1/(2*top_n)) = 0.1.
        frames = self._universe()
        dates = frames['S00'].index
        desk = make_desk(scores=self._scores(),
                         stat_arb_size_by_conviction=False)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        for intent in out[dates[0]]:
            assert intent.size_fraction == pytest.approx(0.4 * 0.25)

    def test_conviction_sizing_orders_legs_by_abs_score(self):
        # CONVICTION SIZING: a higher-|score| leg gets a larger
        # size_fraction than a lower-|score| leg on the same side. Scores
        # S00..S09 are strictly monotone (0.9 .. -0.9), so |S00| > |S01|
        # on the long side and |S09| > |S08| on the short side.
        frames = self._universe()
        dates = frames['S00'].index
        desk = make_desk(scores=self._scores())  # conviction on by default
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        sizes = {i.asset.symbol: i.size_fraction for i in out[dates[0]]}
        # Long side: S00 (|0.9|) heavier than S01 (|0.7|).
        assert sizes['S00'] > sizes['S01']
        # Short side: S09 (|0.9|) heavier than S08 (|0.7|).
        assert sizes['S09'] > sizes['S08']
        # Equal |score| top legs (S00 long, S09 short both |0.9|) match.
        assert sizes['S00'] == pytest.approx(sizes['S09'])

    def test_no_rebalance_off_mondays_without_refit(self):
        frames = self._universe()
        dates = frames['S00'].index
        assert dates[0].weekday() == 0 and dates[1].weekday() == 1
        desk = make_desk(scores=self._scores())
        portfolio = PortfolioManager(100000.0)
        out = drive(desk, frames, dates[:2], portfolio)
        assert len(out[dates[0]]) == 4    # Monday/refit-day rebalance
        assert out[dates[1]] == []        # Tuesday: turnover limited

    def test_tiny_universe_never_overlaps_long_and_short(self):
        frames = self._universe(n_symbols=1)
        dates = frames['S00'].index
        desk = make_desk(scores={'S00': 0.5})
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        # n=1 -> top_n=1; the single symbol ranks top, never both sides.
        assert [i.action for i in out[dates[0]]] == ['BUY']


class TestStatArbTurnoverControls:
    """Phase E stat-arb turnover controls: rank-stability band (hysteresis),
    minimum holding period, conviction sizing, and the recoverability of the
    pre-Phase-E behavior via the new params. The universe has 10 scored
    symbols (top_n = 10//5 = 2; default band exit_n = ceil(2.0*2) = 4)."""

    UNIVERSE_N = 10

    def _frames(self, n_days=12):
        return {f'S{i:02d}': noisy_frame(n=n_days, seed=100 + i)
                for i in range(self.UNIVERSE_N)}

    def _monotone(self):
        # S00 highest (+0.9) .. S09 lowest (-0.9), strictly monotone.
        return {f'S{i:02d}': 0.9 - 0.2 * i for i in range(self.UNIVERSE_N)}

    def _fill(self, portfolio, intents, fill_date):
        for intent in intents:
            qty = 100 if intent.action == 'BUY' else -100
            portfolio.add_position(Position(
                asset=intent.asset, quantity=qty, avg_entry_price=100.0,
                current_price=100.0, timestamp=fill_date))

    # ---- Rank-stability band (hysteresis) ----------------------------
    def test_name_inside_wider_band_is_held_not_exited(self):
        # A held long that slips from the top-2 ENTRY set to rank 3 stays
        # inside the top-4 HOLD band (exit_n = ceil(2.0*2) = 4): it must NOT
        # be exited. Mondays fall on day 0 and day 5; on day 5 the entry's
        # age is 5 so the minimum-hold (default 5) does not interfere.
        frames = self._frames()
        dates = frames['S00'].index
        base = self._monotone()
        # Day 5: S00 drops to rank 3 (still in band) — S02 climbs into top-2.
        day5 = dict(base)
        day5['S00'] = 0.45  # below S01(.7), S02(.5); above S03(.3): rank 3
        sm = DateScoreModel(schedule={dates[5]: day5}, default=base)
        desk = make_stat_arb_desk(sm)  # band 2.0, min-hold 5, conviction on
        portfolio = PortfolioManager(1_000_000.0)

        out0 = drive(desk, frames, dates[:1], portfolio)
        assert sorted(i.asset.symbol for i in out0[dates[0]]
                      if i.action == 'BUY') == ['S00', 'S01']
        self._fill(portfolio, out0[dates[0]], dates[1])

        out = drive(desk, frames, dates[1:6], portfolio)
        # No rebalance Tue..Fri; only day 5 (Monday) rebalances.
        for d in dates[1:5]:
            assert out[d] == []
        day5_acts = {(i.asset.symbol, i.action) for i in out[dates[5]]}
        assert ('S00', 'SELL') not in day5_acts   # held: still in band
        assert ('S02', 'BUY') in day5_acts          # new entrant top-2

    def test_name_leaving_wider_band_is_exited(self):
        # The same setup but S00 falls BELOW the top-4 band (to rank 7):
        # it must be exited with a "left the long hold band" reason.
        frames = self._frames()
        dates = frames['S00'].index
        base = self._monotone()
        day5 = dict(base)
        day5['S00'] = -0.5  # rank 7 of 10: out of the top-4 hold band
        sm = DateScoreModel(schedule={dates[5]: day5}, default=base)
        desk = make_stat_arb_desk(sm)
        portfolio = PortfolioManager(1_000_000.0)

        out0 = drive(desk, frames, dates[:1], portfolio)
        self._fill(portfolio, out0[dates[0]], dates[1])
        out = drive(desk, frames, dates[1:6], portfolio)
        s00 = [i for i in out[dates[5]] if i.asset.symbol == 'S00']
        assert len(s00) == 1
        assert s00[0].action == 'SELL'
        assert 'left the long hold band' in s00[0].reason

    def test_band_mult_one_recovers_exit_on_leaving_entry_decile(self):
        # RECOVERABILITY: stat_arb_exit_band_mult=1.0 makes exit_n == top_n,
        # so a name that merely leaves the top-2 ENTRY set (here to rank 3,
        # which the default band would HOLD) is exited immediately.
        frames = self._frames()
        dates = frames['S00'].index
        base = self._monotone()
        day5 = dict(base)
        day5['S00'] = 0.45  # rank 3: inside default band, outside entry top-2
        sm = DateScoreModel(schedule={dates[5]: day5}, default=base)
        desk = make_stat_arb_desk(sm, stat_arb_exit_band_mult=1.0,
                                  stat_arb_min_hold_days=0)
        portfolio = PortfolioManager(1_000_000.0)

        out0 = drive(desk, frames, dates[:1], portfolio)
        self._fill(portfolio, out0[dates[0]], dates[1])
        out = drive(desk, frames, dates[1:6], portfolio)
        s00 = [i for i in out[dates[5]] if i.asset.symbol == 'S00']
        assert [i.action for i in s00] == ['SELL']  # old behavior: exited

    # ---- Minimum holding period --------------------------------------
    def test_min_hold_blocks_early_exit_then_allows_it(self):
        # With refit every day, every day is a rebalance day. S00 leaves the
        # band starting day 1 but is NOT a flip; the default 5-day minimum
        # hold keeps it until its age reaches 5 (day 5), then it exits.
        frames = self._frames()
        dates = frames['S00'].index
        base = self._monotone()
        later = dict(base)
        later['S00'] = -0.5  # rank 7: out of band, but NOT bottom-2 (no flip)
        sm = DateScoreModel(schedule={d: later for d in dates[1:]},
                            default=base)
        desk = make_stat_arb_desk(sm, stat_arb_refit_every=1)  # daily rebal
        portfolio = PortfolioManager(1_000_000.0)

        out0 = drive(desk, frames, dates[:1], portfolio)
        self._fill(portfolio, out0[dates[0]], dates[1])
        out = drive(desk, frames, dates[1:6], portfolio)
        # Days 1..4 (age 1..4 < 5): held despite leaving the band.
        for d in dates[1:5]:
            assert [i for i in out[d] if i.asset.symbol == 'S00'] == []
        # Day 5 (age 5, no longer < min-hold): finally exited.
        s00 = [i for i in out[dates[5]] if i.asset.symbol == 'S00']
        assert [i.action for i in s00] == ['SELL']

    def test_min_hold_zero_recovers_immediate_exit(self):
        # RECOVERABILITY: stat_arb_min_hold_days=0 exits the name the moment
        # it leaves the band (here band mult 1.0 too, the full old behavior),
        # on the very next rebalance — no holding grace.
        frames = self._frames()
        dates = frames['S00'].index
        base = self._monotone()
        later = dict(base)
        later['S00'] = -0.5
        sm = DateScoreModel(schedule={d: later for d in dates[1:]},
                            default=base)
        desk = make_stat_arb_desk(sm, stat_arb_refit_every=1,
                                  stat_arb_exit_band_mult=1.0,
                                  stat_arb_min_hold_days=0)
        portfolio = PortfolioManager(1_000_000.0)
        out0 = drive(desk, frames, dates[:1], portfolio)
        self._fill(portfolio, out0[dates[0]], dates[1])
        out = drive(desk, frames, dates[1:3], portfolio)
        s00 = [i for i in out[dates[1]] if i.asset.symbol == 'S00']
        assert [i.action for i in s00] == ['SELL']  # exited at age 1

    def test_flip_is_exempt_from_min_hold(self):
        # A held leg that FLIPS to the opposite side must close immediately
        # (no-flip rule) even with age < min-hold: the close has to fill
        # before the opposite-side entry. S00 (long, age 1) is driven to the
        # very bottom so it now ranks SHORT.
        frames = self._frames()
        dates = frames['S00'].index
        base = self._monotone()
        flip = dict(base)
        flip['S00'] = -10.0  # rank last: desired side flips long -> short
        sm = DateScoreModel(schedule={dates[1]: flip}, default=base)
        desk = make_stat_arb_desk(sm, stat_arb_refit_every=1)  # min-hold 5
        portfolio = PortfolioManager(1_000_000.0)
        out0 = drive(desk, frames, dates[:1], portfolio)
        self._fill(portfolio, out0[dates[0]], dates[1])
        out = drive(desk, frames, dates[1:2], portfolio)
        s00 = [i for i in out[dates[1]] if i.asset.symbol == 'S00']
        assert len(s00) == 1
        assert s00[0].action == 'SELL'
        assert 'flip to opposite side' in s00[0].reason

    # ---- Conviction sizing -------------------------------------------
    def test_conviction_sizing_bounds_gross_by_old_flat_total(self):
        # Conviction-sized legs never exceed the per-leg cap, and book gross
        # stays <= the old flat book's total (n_entries * per_leg_cap).
        frames = self._frames()
        dates = frames['S00'].index
        sm = DateScoreModel(default=self._monotone())
        desk = make_stat_arb_desk(sm)  # conviction on by default
        out = drive(desk, frames, dates[:1], PortfolioManager(1_000_000.0))
        intents = out[dates[0]]
        top_n = max(1, self.UNIVERSE_N // 5)
        per_leg_cap = 0.4 * min(0.25, 1.0 / (2 * top_n))
        for intent in intents:
            assert 0 < intent.size_fraction <= per_leg_cap + 1e-12
        gross = sum(i.size_fraction for i in intents)
        assert gross <= len(intents) * per_leg_cap + 1e-12
        # Conviction comes in strictly UNDER the flat total here (the weaker
        # legs price below the cap), proving it is not just the flat book.
        assert gross < len(intents) * per_leg_cap

    def test_near_zero_score_leg_is_floored_not_dropped(self):
        # A near-zero-|score| leg in a batch alongside a strong leg is
        # floored to a small, strictly-positive size (DeskIntent requires
        # size_fraction > 0) rather than rounding to 0; the floor is well
        # below the per-leg cap so it cannot inflate gross.
        desk = make_stat_arb_desk(DateScoreModel(default=self._monotone()))
        top_n = max(1, self.UNIVERSE_N // 5)
        per_leg_cap = 0.4 * min(0.25, 1.0 / (2 * top_n))
        floor = max(per_leg_cap * 0.05, 1e-6)
        # S01 strong (|0.7|), S00 near-zero (|1e-9|): the weak leg floors.
        # Both on the SAME side so the per-side budget covers the pair.
        scores = {'S01': 0.7, 'S00': 1e-9}
        desired = {'S01': 'short', 'S00': 'short'}
        sizes = desk._stat_arb_sizes(['S01', 'S00'], scores, desired,
                                     0.4, top_n)
        assert sizes['S00'] == pytest.approx(floor)
        assert 0 < sizes['S00'] < sizes['S01'] <= per_leg_cap + 1e-12

    def test_conviction_sizing_is_dollar_neutral_across_sides(self):
        # REGRESSION for the per-side budgeting fix: conviction sizing must
        # NOT leak net directional exposure when one side is more confident
        # than the other. Budgeting per side keeps long gross ~ short gross
        # even under strong cross-side |score| asymmetry; pooling both sides
        # (the bug) would drive the confident side to the cap and starve the
        # other, taking a large net directional position on a desk that is
        # supposed to be market-neutral.
        desk = make_stat_arb_desk(DateScoreModel(default=self._monotone()))
        top_n = max(1, self.UNIVERSE_N // 5)
        per_leg_cap = 0.4 * min(0.25, 1.0 / (2 * top_n))
        # Confident longs, weak shorts (the exact asymmetry that leaked beta).
        entry = ['L0', 'L1', 'S0', 'S1']
        desired = {'L0': 'long', 'L1': 'long', 'S0': 'short', 'S1': 'short'}
        scores = {'L0': 0.40, 'L1': 0.35, 'S0': -0.08, 'S1': -0.06}
        sizes = desk._stat_arb_sizes(entry, scores, desired, 0.4, top_n)
        long_gross = sizes['L0'] + sizes['L1']
        short_gross = sizes['S0'] + sizes['S1']
        # Per side, gross stays near the (equal) per-side budget; the net is a
        # small residual from per-leg-cap clamping, FAR below the directional
        # tilt that pooled budgeting would produce (~0.13 here).
        assert abs(long_gross - short_gross) < 0.5 * per_leg_cap
        # Each side is still conviction-tilted internally and cap-bounded.
        assert sizes['L0'] > sizes['L1']
        assert sizes['S0'] > sizes['S1']
        assert max(sizes.values()) <= per_leg_cap + 1e-12

    def test_all_zero_scores_fall_back_to_flat(self):
        # All-zero |score| -> conviction is undefined; sizing falls back to
        # the flat per-leg cap so every leg still trades.
        frames = self._frames()
        dates = frames['S00'].index
        zero_scores = {f'S{i:02d}': 0.0 for i in range(self.UNIVERSE_N)}
        sm = DateScoreModel(default=zero_scores)
        desk = make_stat_arb_desk(sm)
        out = drive(desk, frames, dates[:1], PortfolioManager(1_000_000.0))
        top_n = max(1, self.UNIVERSE_N // 5)
        per_leg_cap = 0.4 * min(0.25, 1.0 / (2 * top_n))
        for intent in out[dates[0]]:
            assert intent.size_fraction == pytest.approx(per_leg_cap)

    # ---- Turnover reduction (the key regression) ---------------------
    def test_turnover_drops_vs_recovered_old_behavior(self):
        # KEY REGRESSION: on a synthetic universe whose marginal names
        # oscillate across the decile boundary every Monday, the NEW
        # defaults (band 2.0 + min-hold 5) round-trip FEWER stat-arb legs
        # than the params set to recover the old behavior (band 1.0 +
        # min-hold 0) — while still trading. Pins the hysteresis so a future
        # change that re-introduces churn fails here.
        n_days = 40
        frames = {f'S{i:02d}': noisy_frame(n=n_days, seed=300 + i)
                  for i in range(self.UNIVERSE_N)}
        dates = frames['S00'].index
        base = self._monotone()
        rng = np.random.default_rng(7)
        schedule = {}
        for d in dates:                # jitter the ranking every Monday
            if d.weekday() == 0:
                schedule[d] = {s: base[s] + rng.normal(0.0, 0.25)
                               for s in base}

        def run(recover_old):
            sm = DateScoreModel(schedule=schedule, default=base)
            kwargs = {}
            if recover_old:
                kwargs = dict(stat_arb_exit_band_mult=1.0,
                              stat_arb_min_hold_days=0,
                              stat_arb_size_by_conviction=False)
            desk = make_stat_arb_desk(sm, **kwargs)
            engine = BacktestEngine(desk=desk, initial_capital=1_000_000.0,
                                    commission=0.0)
            with _patched_fetch(frames):
                engine.run(sorted(frames), '2023-01-01', '2023-04-30',
                           benchmark_symbol=None)
            exits = [n for n in desk.notes
                     if n.data.get('book') == 'stat_arb'
                     and n.message.startswith('Stat-arb exit')]
            entries = [n for n in desk.notes
                       if n.data.get('book') == 'stat_arb'
                       and (n.message.startswith('Stat-arb LONG')
                            or n.message.startswith('Stat-arb SHORT'))]
            return len(exits), len(entries)

        new_exits, new_entries = run(recover_old=False)
        old_exits, old_entries = run(recover_old=True)
        assert new_entries > 0           # the new book still trades
        assert new_exits < old_exits     # fewer round-trips than old
        # And no churn pathology: new book does not exit more than it enters.
        assert new_exits <= new_entries


class TestPairsBook:
    def _frames(self, n=10):
        return {'AA': noisy_frame(n=n, seed=21),
                'BB': noisy_frame(n=n, seed=22)}

    def test_entry_above_z_threshold_shorts_rich_longs_cheap(self):
        frames = self._frames()
        dates = frames['AA'].index
        quote = {'a': 'AA', 'b': 'BB', 'beta': 1.5, 'z': 2.5}
        desk = make_desk(
            pairs_schedule={dates[0]: [quote]},
            model_pairs=[{'a': 'AA', 'b': 'BB'}])
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))

        intents = {i.asset.symbol: i for i in out[dates[0]]}
        assert set(intents) == {'AA', 'BB'}
        assert intents['AA'].action == 'SHORT'  # rich leg (z > 0)
        assert intents['BB'].action == 'BUY'    # cheap leg
        # leg size = w_pairs * min(0.5, 1/max(2, 2*1)) = 0.2 * 0.5 = 0.1.
        for intent in intents.values():
            assert intent.size_fraction == pytest.approx(0.2 * 0.5)

        entry_notes = [n for n in desk.notes
                       if n.data.get('book') == 'pairs'
                       and 'entry' in n.message.lower()]
        assert len(entry_notes) == 1
        assert entry_notes[0].data['pair'] == 'AA/BB'
        assert entry_notes[0].data['beta'] == pytest.approx(1.5)
        assert entry_notes[0].data['z'] == pytest.approx(2.5)

    def test_negative_z_entry_longs_a_shorts_b(self):
        frames = self._frames()
        dates = frames['AA'].index
        quote = {'a': 'AA', 'b': 'BB', 'beta': 1.2, 'z': -2.4}
        desk = make_desk(pairs_schedule={dates[0]: [quote]},
                         model_pairs=[{'a': 'AA', 'b': 'BB'}])
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        intents = {i.asset.symbol: i.action for i in out[dates[0]]}
        assert intents == {'AA': 'BUY', 'BB': 'SHORT'}

    def test_no_entry_inside_threshold(self):
        frames = self._frames()
        dates = frames['AA'].index
        quote = {'a': 'AA', 'b': 'BB', 'beta': 1.5, 'z': 1.8}
        desk = make_desk(pairs_schedule={dates[0]: [quote]},
                         model_pairs=[{'a': 'AA', 'b': 'BB'}])
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []

    def _open_pair(self, desk, frames, dates, portfolio):
        """Drive the entry day and simulate the engine fills."""
        out = drive(desk, frames, dates[:1], portfolio)
        assert len(out[dates[0]]) == 2
        portfolio.add_position(Position(
            asset=stock('AA'), quantity=-50, avg_entry_price=100.0,
            current_price=100.0, timestamp=dates[1]))
        portfolio.add_position(Position(
            asset=stock('BB'), quantity=50, avg_entry_price=100.0,
            current_price=100.0, timestamp=dates[1]))

    def test_exit_when_z_reverts_inside_band(self):
        frames = self._frames()
        dates = frames['AA'].index
        entry = {'a': 'AA', 'b': 'BB', 'beta': 1.5, 'z': 2.5}
        flat = {'a': 'AA', 'b': 'BB', 'beta': 1.5, 'z': 0.3}
        desk = make_desk(
            pairs_schedule={dates[0]: [entry], dates[1]: [entry],
                            dates[2]: [flat]},
            model_pairs=[{'a': 'AA', 'b': 'BB'}])
        portfolio = PortfolioManager(100000.0)
        self._open_pair(desk, frames, dates, portfolio)

        out = drive(desk, frames, dates[1:3], portfolio)
        assert out[dates[1]] == []  # |z| still wide: hold
        exits = {i.asset.symbol: i.action for i in out[dates[2]]}
        assert exits == {'AA': 'COVER', 'BB': 'SELL'}
        exit_notes = [n for n in desk.notes
                      if n.data.get('book') == 'pairs'
                      and 'exit' in n.message.lower()]
        assert len(exit_notes) == 1
        assert exit_notes[0].data['z'] == pytest.approx(0.3)

    def test_exit_when_pair_vanishes_at_refit(self):
        frames = self._frames()
        dates = frames['AA'].index
        entry = {'a': 'AA', 'b': 'BB', 'beta': 1.5, 'z': 2.5}
        desk = make_desk(pairs_schedule={dates[0]: [entry]},
                         model_pairs=[{'a': 'AA', 'b': 'BB'}],
                         pairs_refit_every=1)  # refit every day
        portfolio = PortfolioManager(100000.0)
        self._open_pair(desk, frames, dates, portfolio)

        # The day-2 refit drops the pair from the model.
        desk._pairs_controller.model.pairs = []
        out = drive(desk, frames, dates[1:2], portfolio)
        exits = {i.asset.symbol: i.action for i in out[dates[1]]}
        assert exits == {'AA': 'COVER', 'BB': 'SELL'}
        vanish_notes = [n for n in desk.notes
                        if 'vanished at refit' in n.message]
        assert len(vanish_notes) == 1
        assert vanish_notes[0].data['book'] == 'pairs'


class TestOrphanSweep:
    """Regression: the engine holds a pending intent for up to
    MAX_PENDING_DAYS (5) trading days, but the desk reconciles tracking
    away after RECONCILE_GRACE_DAYS (2) — so an entry can FILL after its
    tracking was dropped. Before the orphan sweep, such a position was
    never closed by the desk (it escaped regime flattening, rebalance
    exits, and pair closes; _symbol_is_free blocked re-entry forever,
    and only a stop-loss could ever touch it)."""

    def test_late_fill_after_cleanup_is_closed_by_the_desk(
            self, monkeypatch):
        # One trading week, Monday start. GAP has NO bars on the two
        # days after the day-0 intent; its bar returns on day 3.
        n_days = 5
        frames = {f'S{i:02d}': noisy_frame(n=n_days, seed=200 + i)
                  for i in range(5)}
        dates = frames['S00'].index
        assert dates[0].weekday() == 0
        gap = noisy_frame(n=n_days, seed=199)
        frames['GAP'] = gap.drop(index=[dates[1], dates[2]])

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)

        # GAP scores lowest: the stat-arb book SHORTs it on day 0
        # (6 symbols -> top/bottom 1; refit day = rebalance day).
        scores = {f'S{i:02d}': 0.9 - 0.1 * i for i in range(5)}
        scores['GAP'] = -0.9
        desk = make_desk(scores=scores)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(sorted(frames), '2023-01-01', '2023-01-31',
                            benchmark_symbol=None)

        # Tracking was reconciled away on day 2, BEFORE the day-3 fill.
        cleanup = [n for n in desk.notes if 'Book cleanup' in n.message
                   and n.data.get('symbol') == 'GAP']
        assert len(cleanup) == 1
        assert cleanup[0].timestamp == dates[2]

        # The orphan SHORT filled on day 3 and the desk's sweep COVERed
        # it on day 4 — the position must not survive the run.
        gap_trades = [t for t in report['trades'] if t['symbol'] == 'GAP']
        assert [t['action'] for t in gap_trades] == ['SHORT', 'COVER']
        assert gap_trades[0]['date'] == dates[3]
        assert gap_trades[1]['date'] == dates[4]
        position = engine.portfolio.get_position(stock('GAP'))
        assert position is None or position.quantity == 0

        sweep_notes = [n for n in desk.notes
                       if 'Orphan sweep' in n.message]
        assert len(sweep_notes) == 1
        assert sweep_notes[0].category == 'risk'
        assert sweep_notes[0].timestamp == dates[3]
        assert sweep_notes[0].data['book'] == 'stat_arb'  # C7 filterable
        assert sweep_notes[0].data['symbol'] == 'GAP'
        assert sweep_notes[0].data['direction'] == 'short'


class TestRegimeGuardAndRecoverability:
    """The Phase E sharpening must not perturb the (untouched) HMM regime
    engine, and the full pre-Phase-E behavior must be recoverable by setting
    the new params to their old values."""

    def _universe(self, n_symbols=10, n_days=12):
        return {f'S{i:02d}': noisy_frame(n=n_days, seed=100 + i)
                for i in range(n_symbols)}

    def _scores(self, n_symbols=10):
        return {f'S{i:02d}': 0.9 - 0.2 * i for i in range(n_symbols)}

    def test_regime_series_unchanged_by_stat_arb_and_mr_params(self):
        # REGIME GUARD: the regime posterior series depends only on the
        # regime controller; toggling every new stat-arb/MR param must leave
        # regime_series byte-identical (the regime engine was not touched).
        frames = self._universe()
        dates = frames['S00'].index
        schedule = {
            dates[0]: probs_for('trending', 0.80),
            dates[1]: probs_for('mean_reverting', 0.75),
            dates[2]: probs_for('trending', 0.66),
        }

        def regime_series_for(**params):
            desk = make_desk(regime_schedule=schedule,
                             scores=self._scores(), **params)
            drive(desk, frames, dates[:3], PortfolioManager(1_000_000.0))
            return desk.regime_series

        sharpened = regime_series_for()  # all sharpened defaults
        old = regime_series_for(stat_arb_exit_band_mult=1.0,
                                stat_arb_min_hold_days=0,
                                stat_arb_size_by_conviction=False,
                                mr_size_by_conviction=False)
        assert sharpened == old
        # And the series is the scripted regime, untouched by either config.
        assert [e['state'] for e in sharpened] == \
            ['trending', 'mean_reverting', 'trending']

    def test_full_old_config_recovers_flat_immediate_exit_book(self):
        # RECOVERABILITY: the documented old-behavior config (band 1.0,
        # min-hold 0, flat stat-arb + MR sizing) reproduces the pre-Phase-E
        # stat-arb book — flat per-leg sizing AND exit-the-moment-a-name-
        # leaves-the-entry-decile.
        frames = self._universe()
        dates = frames['S00'].index
        base = self._scores()
        # Day 5 (Monday): S00 drops to rank 3 — held by the new band, but
        # exited under the old (band 1.0) behavior.
        day5 = dict(base)
        day5['S00'] = 0.45
        sm = DateScoreModel(schedule={dates[5]: day5}, default=base)
        desk = make_stat_arb_desk(
            sm, stat_arb_exit_band_mult=1.0, stat_arb_min_hold_days=0,
            stat_arb_size_by_conviction=False)
        portfolio = PortfolioManager(1_000_000.0)

        out0 = drive(desk, frames, dates[:1], portfolio)
        # Old flat sizing: every entered leg at 0.4 * 0.25 = 0.1.
        for intent in out0[dates[0]]:
            assert intent.size_fraction == pytest.approx(0.4 * 0.25)
        for intent in out0[dates[0]]:
            qty = 100 if intent.action == 'BUY' else -100
            portfolio.add_position(Position(
                asset=intent.asset, quantity=qty, avg_entry_price=100.0,
                current_price=100.0, timestamp=dates[1]))

        out = drive(desk, frames, dates[1:6], portfolio)
        s00 = [i for i in out[dates[5]] if i.asset.symbol == 'S00']
        # Old behavior: leaving the top-2 entry decile exits immediately.
        assert [i.action for i in s00] == ['SELL']


class TestEndToEndWithEngine:
    """Real models, real engine, synthetic universe (fetch monkeypatched)."""

    N_DAYS = 300

    @pytest.fixture
    def universe(self):
        rng = np.random.default_rng(42)
        index = pd.bdate_range('2022-01-03', periods=self.N_DAYS)

        def frame(log_close, volume):
            close = np.exp(log_close)
            return pd.DataFrame({
                'open': close, 'high': close * 1.001, 'low': close * 0.999,
                'close': close,
                'volume': rng.integers(400_000, int(volume), self.N_DAYS
                                       ).astype(float),
            }, index=index)

        log_b = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, self.N_DAYS))
        noise = np.empty(self.N_DAYS)
        prev = 0.0
        for i in range(self.N_DAYS):
            prev = 0.5 * prev + rng.normal(0.0, 0.01)
            noise[i] = prev
        frames = {
            'PA': frame(1.5 * log_b + 0.5 + noise, 900_000),
            'PB': frame(log_b, 850_000),
        }
        for i, symbol in enumerate(('N1', 'N2', 'N3', 'N4')):
            walk = np.log(80.0 + 10 * i) + np.cumsum(
                rng.normal(0.0005, 0.012, self.N_DAYS))
            frames[symbol] = frame(walk, 800_000 - 50_000 * i)
        return frames

    @pytest.fixture
    def report_and_desk(self, universe, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        desk = RenaissanceDesk(
            stat_arb_controller=WalkForwardController(
                GradientBoostingModel(n_estimators=10),
                train_window_days=252, refit_every_days=21,
                min_train_days=120))
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=0.001)
        report = engine.run(sorted(universe), '2022-01-01', '2023-12-31',
                            benchmark_symbol=None)
        return report, desk, engine

    def test_report_has_c3_c5_shapes(self, report_and_desk):
        report, desk, _ = report_and_desk
        assert report['desk'] == {'key': 'renaissance',
                                  'name': 'Renaissance Desk'}
        assert report['strategy'] == 'Renaissance Desk'

        # --- C3+: walk_forward fits from all three models, tagged ----
        walk_forward = report['walk_forward']
        assert walk_forward
        models_seen = {fit['model'] for fit in walk_forward}
        assert models_seen == {'regime', 'stat_arb', 'pairs'}
        for fit in walk_forward:
            assert set(fit) == {'fit_date', 'train_start', 'train_end',
                                'n_samples', 'model'}
            assert fit['train_end'] <= fit['fit_date']

        # --- C5: regime_series present and well-shaped ---------------
        series = report['regime_series']
        assert series  # non-empty once the regime model fitted
        first_regime_fit = min(fit['fit_date'] for fit in walk_forward
                               if fit['model'] == 'regime')
        for entry in series:
            assert set(entry) == {'date', 'state', 'probs'}
            assert entry['state'] in REGIME_LABELS
            assert set(entry['probs']) == set(REGIME_LABELS)
            assert sum(entry['probs'].values()) == pytest.approx(1.0)
            assert entry['date'] >= first_regime_fit
        json.dumps(report['regime_series'])
        json.dumps(report['walk_forward'])
        json.dumps(report['trader_notes'])

        # Phase 7 regression (C8): pod_history is a citadel-only key.
        assert 'pod_history' not in report

        assert desk.get_status()['regime'] in REGIME_LABELS

    def test_book_notes_are_filterable_per_c7(self, report_and_desk):
        report, _, _ = report_and_desk
        allowed = {'regime', 'mean_reversion', 'stat_arb', 'pairs'}
        book_notes = [note for note in report['trader_notes']
                      if 'book' in note['data']]
        assert book_notes
        for note in book_notes:
            assert note['data']['book'] in allowed
        # All three model books are represented at minimum via refits.
        assert {'regime', 'stat_arb', 'pairs'} <= \
            {note['data']['book'] for note in book_notes}

    def test_no_mean_reversion_entries_when_prob_at_or_below_threshold(
            self, report_and_desk):
        report, _, _ = report_and_desk
        probs_by_date = {entry['date']: entry['probs']
                         for entry in report['regime_series']}
        entry_notes = [
            note for note in report['trader_notes']
            if note['data'].get('book') == 'mean_reversion'
            and (note['message'].startswith('Mean-reversion LONG')
                 or note['message'].startswith('Mean-reversion SHORT'))]
        for note in entry_notes:
            day = note['timestamp'][:10]
            # Every entry day has a regime_series row whose
            # P(mean_reverting) clears the 0.6 gate (contract: the book
            # is dark whenever P <= 0.6 or the model has no answer).
            assert probs_by_date[day]['mean_reverting'] > 0.6

    def test_trades_only_in_desk_actions_and_shorts_are_covered(
            self, report_and_desk):
        report, _, engine = report_and_desk
        assert report['trades']  # the desk did act on 300 days of data
        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}
        # Every open short was opened by SHORT and only closed by COVER:
        # net per-symbol short exposure can never be doubled.
        for symbol in {t['symbol'] for t in report['trades']}:
            position = 0
            for trade in report['trades']:
                if trade['symbol'] != symbol:
                    continue
                if trade['action'] == 'BUY':
                    assert position == 0
                    position += trade['quantity']
                elif trade['action'] == 'SELL':
                    assert position > 0
                    position -= trade['quantity']
                elif trade['action'] == 'SHORT':
                    assert position == 0
                    position -= trade['quantity']
                else:  # COVER
                    assert position < 0
                    position += trade['quantity']

    def test_foundation_desk_report_gains_no_regime_content(
            self, universe, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = FoundationDesk()
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(['N1', 'N2'], '2022-01-01', '2023-12-31',
                            benchmark_symbol=None)
        # C5: absent or [] are both acceptable; it must never be non-empty.
        assert not report.get('regime_series')
        # Phase 7 regression (C8): same for pod_history.
        assert not report.get('pod_history')
