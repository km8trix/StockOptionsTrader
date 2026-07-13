"""Tests for desks.twosigma.TwoSigmaDesk.

Unit tests drive generate_intents day by day with a STUB walk-forward
score model (every per-symbol score controlled) to pin the cross-sectional
long/short book mechanics: quantile selection, dollar-balanced sizing,
reconcile/exit when a name leaves its side, graceful degrade (unfitted
model, too-few-scored, tiny/single-symbol universe), and the committee
averaging + tagged walk_forward_fits. The end-to-end tests run the real
BacktestEngine with a real GradientBoostingModel controller on synthetic
data and check determinism, the no-one-step-flip invariant, the orphan
sweep, the research-integrity report blocks, and FundOrchestrator interop.
Offline, seeded, deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.foundation import FoundationDesk
from desks.ml_model import GradientBoostingModel
from desks.orchestrator import FundOrchestrator
from desks.twosigma import (DEFAULT_MODEL_KEY, RECONCILE_GRACE_DAYS,
                            TwoSigmaDesk)
from desks.walk_forward import WalkForwardController, WalkForwardModel
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


@pytest.fixture
def patch_market_data(monkeypatch):
    """Patch MarketDataHandler.fetch_stock_data to serve canned frames."""

    def _patch(frames_by_symbol):
        def fake_fetch(self, symbol, start_date, end_date):
            return frames_by_symbol.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

    return _patch


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def wide_risk() -> RiskManager:
    """Risk manager opened wide so the pure book mechanics show through
    (no position-size cap, no daily-loss circuit, no stop-loss)."""
    return RiskManager(max_position_size=1.0, max_daily_loss=0.99,
                       position_stop_loss=0.90)


# ----------------------------------------------------------------------
# Stub walk-forward score model (centered P(up)-0.5 per symbol)
# ----------------------------------------------------------------------
class StubScoreModel(WalkForwardModel):
    """Per-symbol centered scores by date.

    ``schedule`` maps a Timestamp -> {symbol: score}; unscripted dates fall
    back to ``default``. Only symbols present in the day's data are scored
    (mirrors the real model contract). ``None`` as a day's mapping makes the
    model return {} for that date (fitted-but-nothing-scored).
    """

    def __init__(self, default=None, schedule=None):
        self.default = default or {}
        self.schedule = {pd.Timestamp(d): s
                         for d, s in (schedule or {}).items()}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        scores = self.schedule.get(pd.Timestamp(date), self.default)
        if scores is None:
            return {}
        return {symbol: score for symbol, score in scores.items()
                if symbol in data}


def stub_controller(default=None, schedule=None,
                    min_train_days=1) -> WalkForwardController:
    """Controller that fits immediately (min_train_days=1) over a stub."""
    return WalkForwardController(StubScoreModel(default, schedule),
                                 min_train_days=min_train_days)


def make_desk(default=None, schedule=None, min_train_days=1,
              **kwargs) -> TwoSigmaDesk:
    kwargs.setdefault('risk_manager', wide_risk())
    return TwoSigmaDesk(
        controller=stub_controller(default, schedule, min_train_days),
        **kwargs)


# ----------------------------------------------------------------------
# Synthetic frames + a one-desk driver (no engine; fills are manual)
# ----------------------------------------------------------------------
def quiet_frame(n=8, seed=5, start='2023-01-02') -> pd.DataFrame:
    """Low-noise random walk; a bar on every business day in the window."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.002, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': np.full(n, 500_000.0),
    }, index=index)


def universe(n_symbols=10, n_days=8):
    return {f'S{i:02d}': quiet_frame(n=n_days, seed=100 + i)
            for i in range(n_symbols)}


def monotone_scores(n_symbols=10):
    """S00 highest .. S(n-1) lowest, strictly monotone and centered."""
    return {f'S{i:02d}': 0.4 - 0.08 * i for i in range(n_symbols)}


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


# ----------------------------------------------------------------------
# Construction + status + validation
# ----------------------------------------------------------------------
class TestConstructionAndStatus:
    def test_module_constants(self):
        assert DEFAULT_MODEL_KEY == 'stacking'
        assert RECONCILE_GRACE_DAYS == 2

    def test_default_is_single_stacking_controller(self):
        desk = TwoSigmaDesk()
        assert desk.get_status()['models'] == ['stacking']
        assert desk.get_status()['model'] == 'stacking'

    def test_model_key_selects_single_controller(self):
        desk = TwoSigmaDesk(model_key='lightgbm')
        assert desk.get_status()['models'] == ['lightgbm']
        assert desk.get_status()['model'] == 'lightgbm'

    def test_models_list_builds_a_committee(self):
        desk = TwoSigmaDesk(models=['gbm', 'lightgbm'])
        assert desk.get_status()['models'] == ['gbm', 'lightgbm']
        # The joined label is the committee's ids.
        assert desk.get_status()['model'] == 'gbm+lightgbm'

    def test_explicit_controller_takes_precedence(self):
        desk = make_desk(default=monotone_scores())
        assert desk.get_status()['models'] == ['custom']

    @pytest.mark.parametrize('quantile', [0.0, -0.1, 0.6, 1.0])
    def test_bad_quantile_raises(self, quantile):
        with pytest.raises(ValueError, match='quantile'):
            TwoSigmaDesk(quantile=quantile)

    @pytest.mark.parametrize('target_gross', [0.0, -0.5, 1.5])
    def test_bad_target_gross_raises(self, target_gross):
        with pytest.raises(ValueError, match='target_gross'):
            TwoSigmaDesk(target_gross=target_gross)

    @pytest.mark.parametrize('min_scored', [0, 1, -3])
    def test_min_scored_below_two_raises(self, min_scored):
        with pytest.raises(ValueError, match='min_scored'):
            TwoSigmaDesk(min_scored=min_scored)

    def test_controller_with_model_key_raises(self):
        controller = stub_controller(default={})
        with pytest.raises(ValueError, match='not both'):
            TwoSigmaDesk(controller=controller, model_key='gbm')

    def test_controller_with_models_raises(self):
        controller = stub_controller(default={})
        with pytest.raises(ValueError, match='not both'):
            TwoSigmaDesk(controller=controller, models=['gbm'])

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match='at least one'):
            TwoSigmaDesk(models=[])

    def test_status_exposes_book_counts(self):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        status = desk.get_status()
        assert status['quantile'] == 0.2
        assert status['target_gross'] == 1.0
        assert status['long_count'] == 2
        assert status['short_count'] == 2


# ----------------------------------------------------------------------
# Cross-sectional selection + dollar-balanced sizing
# ----------------------------------------------------------------------
class TestSelectionAndSizing:
    def test_longs_top_quantile_shorts_bottom_quantile(self):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))

        intents = out[dates[0]]
        longs = sorted(i.asset.symbol for i in intents if i.action == 'BUY')
        shorts = sorted(i.asset.symbol for i in intents if i.action == 'SHORT')
        # k = int(0.2 * 10) = 2: top two ranked, bottom two ranked.
        assert longs == ['S00', 'S01']
        assert shorts == ['S08', 'S09']

    def test_k_respects_the_quantile(self):
        frames = universe(10)
        dates = frames['S00'].index
        # quantile 0.3 -> k = int(0.3 * 10) = 3.
        desk = make_desk(default=monotone_scores(), quantile=0.3,
                         target_gross=1.0)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        intents = out[dates[0]]
        longs = sorted(i.asset.symbol for i in intents if i.action == 'BUY')
        shorts = sorted(i.asset.symbol for i in intents if i.action == 'SHORT')
        assert longs == ['S00', 'S01', 'S02']
        assert shorts == ['S07', 'S08', 'S09']

    def test_book_is_dollar_balanced(self):
        frames = universe(10)
        dates = frames['S00'].index
        # max_name_size raised to 0.30 so the intended unclamped relationship
        # (0.5*target_gross)/k is what binds — not the clamp — exercising the
        # dollar-balance mechanic with the documented size formula.
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0, max_name_size=0.30)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        intents = out[dates[0]]
        longs = [i for i in intents if i.action == 'BUY']
        shorts = [i for i in intents if i.action == 'SHORT']

        # Each side gets half of target_gross spread over its k names, clamped
        # to max_name_size: min(0.30, (0.5 * 1.0) / 2) = min(0.30, 0.25) = 0.25.
        expected = min(0.30, (0.5 * 1.0) / 2)
        assert expected == pytest.approx(0.25)
        for intent in longs + shorts:
            assert intent.size_fraction == pytest.approx(expected)
        long_gross = sum(i.size_fraction for i in longs)
        short_gross = sum(i.size_fraction for i in shorts)
        assert long_gross == pytest.approx(0.5)        # half of gross
        assert short_gross == pytest.approx(0.5)
        # Net ~ 0 (equal dollars each side); gross ~ target_gross.
        assert long_gross - short_gross == pytest.approx(0.0)
        assert long_gross + short_gross == pytest.approx(1.0)

    def test_default_max_name_size_clamps_each_leg(self):
        # Under the DEFAULT max_name_size (0.10), the per-leg size is clamped:
        # min(0.10, (0.5 * 1.0) / 2) = min(0.10, 0.25) = 0.10. The book stays
        # dollar-balanced (equal per-leg size on both sides, net ~ 0).
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)  # max_name_size defaults to 0.10
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        intents = out[dates[0]]
        longs = [i for i in intents if i.action == 'BUY']
        shorts = [i for i in intents if i.action == 'SHORT']

        expected = min(0.10, (0.5 * 1.0) / 2)
        assert expected == pytest.approx(0.10)  # the clamp binds
        for intent in longs + shorts:
            assert intent.size_fraction == pytest.approx(expected)
        long_gross = sum(i.size_fraction for i in longs)
        short_gross = sum(i.size_fraction for i in shorts)
        # Equal dollars each side -> net ~ 0; gross ~ 2 * k * max_name_size.
        assert long_gross == pytest.approx(short_gross)
        assert long_gross - short_gross == pytest.approx(0.0)
        assert long_gross + short_gross == pytest.approx(2 * 2 * 0.10)

    def test_target_gross_scales_each_leg(self):
        frames = universe(10)
        dates = frames['S00'].index
        # max_name_size raised above the per-leg target so target_gross is the
        # binding constraint (clamp does not bite): min(0.30, (0.5*0.6)/2).
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=0.6, max_name_size=0.30)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        intents = out[dates[0]]
        # min(0.30, (0.5 * 0.6) / 2) = min(0.30, 0.15) = 0.15 per leg.
        expected = min(0.30, (0.5 * 0.6) / 2)
        assert expected == pytest.approx(0.15)
        for intent in intents:
            assert intent.size_fraction == pytest.approx(expected)
        gross = sum(i.size_fraction for i in intents)
        assert gross == pytest.approx(0.6)

    def test_signal_and_allocation_notes_carry_real_numbers(self):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))

        signal_notes = [n for n in desk.notes if n.category == 'signal']
        assert len(signal_notes) == 4  # 2 longs + 2 shorts
        for note in signal_notes:
            assert note.data['symbol'] in {'S00', 'S01', 'S08', 'S09'}
            assert note.data['direction'] in ('long', 'short')
            assert note.data['n_scored'] == 10
            assert 'score' in note.data and 'rank' in note.data

        alloc = [n for n in desk.notes if n.category == 'allocation']
        assert len(alloc) == 1
        assert alloc[0].data['k'] == 2
        assert sorted(alloc[0].data['longs']) == ['S00', 'S01']
        assert sorted(alloc[0].data['shorts']) == ['S08', 'S09']
        assert alloc[0].data['target_gross'] == pytest.approx(1.0)

    def test_only_symbols_with_a_bar_today_are_traded(self):
        # S09 has no bar on the last day -> it cannot be scored/traded even
        # though the stub would score it.
        frames = universe(10)
        dates = frames['S00'].index
        frames['S09'] = frames['S09'].drop(index=dates[-1])
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        out = drive(desk, frames, dates[-1:], PortfolioManager(100000.0))
        traded = {i.asset.symbol for i in out[dates[-1]]}
        assert 'S09' not in traded
        # 9 scored -> k = int(0.2 * 9) = 1; bottom is now S08.
        shorts = sorted(i.asset.symbol for i in out[dates[-1]]
                        if i.action == 'SHORT')
        assert shorts == ['S08']


# ----------------------------------------------------------------------
# Reconcile: a name leaving its side closes next day
# ----------------------------------------------------------------------
class TestReconcile:
    """Reconcile is exercised through the real engine so pending-intent
    dedup and next-open fills behave exactly as in production (driving
    generate_intents by hand double-counts the twice-per-day reconcile)."""

    def _run(self, frames, default, schedule, monkeypatch, **desk_kwargs):
        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = TwoSigmaDesk(
            controller=stub_controller(default, schedule),
            risk_manager=wide_risk(), **desk_kwargs)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(sorted(frames), '2023-01-01', '2023-01-31',
                            benchmark_symbol=None)
        return desk, engine, report

    def test_name_leaving_its_side_is_closed_and_not_reopened(
            self, monkeypatch):
        frames = universe(10)
        dates = frames['S00'].index
        base = monotone_scores()
        # From day 2 onward S00 collapses to the middle and leaves the long
        # book for good; the rest of the ranking is unchanged.
        schedule = {d: {**base, 'S00': 0.0} for d in dates[2:]}
        desk, engine, report = self._run(
            frames, base, schedule, monkeypatch,
            quantile=0.2, target_gross=1.0)

        s00 = [t['action'] for t in report['trades'] if t['symbol'] == 'S00']
        # S00 was bought, sold when it left the book, and NEVER re-opened.
        assert s00 == ['BUY', 'SELL']
        # It is no longer tracked, and no open S00 position survives.
        assert stock('S00') not in desk._book_positions
        position = engine.portfolio.get_position(stock('S00'))
        assert position is None or position.quantity == 0

        # A name that keeps its seat (S09, always the bottom short) is opened
        # ONCE and HELD continuously — no daily close/reopen churn. (Before the
        # reconcile fix the whole book was blind-closed every day, so S09
        # churned with multiple SHORTs; now it is held untouched.)
        s09 = [t['action'] for t in report['trades'] if t['symbol'] == 'S09']
        assert s09.count('SHORT') == 1
        assert s09.count('COVER') == 0       # never closed — held the whole run
        # It is still tracked as a short, with a live short position open.
        assert stock('S09') in desk._book_positions
        assert desk._book_positions[stock('S09')]['direction'] == 'short'
        s09_position = engine.portfolio.get_position(stock('S09'))
        assert s09_position is not None and s09_position.quantity < 0
        # Only the four desk actions ever appear.
        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}

    def test_unfilled_leg_is_dropped_after_grace_then_reentered(self):
        # A name is opened but its entry never fills; after the reconcile
        # grace the stale leg is dropped (info note) and the symbol is free
        # to re-enter the SAME day (a fresh leg with a later entry_day),
        # proving it was never permanently blocked. Driven by hand to
        # control the missing fill precisely.
        frames = universe(10, n_days=12)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        portfolio = PortfolioManager(100000.0)
        out = drive(desk, frames, dates[:1], portfolio)
        assert ('BUY', 'S00') in {(i.action, i.asset.symbol)
                                  for i in out[dates[0]]}
        assert desk._book_positions[stock('S00')]['entry_day'] == 0

        # Day 1 is still inside the grace (day_index 1 - entry_day 0 < 2):
        # the stale leg is held untouched, no cleanup yet.
        drive(desk, frames, dates[1:2], portfolio)
        assert desk._book_positions[stock('S00')]['entry_day'] == 0
        assert not [n for n in desk.notes if 'Book cleanup' in n.message
                    and n.data.get('symbol') == 'S00']

        # Day 2 hits the grace boundary (2 - 0 >= RECONCILE_GRACE_DAYS): the
        # stale leg is dropped (info note) AND the symbol — now free — is
        # re-entered the same day with a fresh leg.
        out = drive(desk, frames, dates[2:3], portfolio)
        cleanup = [n for n in desk.notes if n.category == 'info'
                   and 'Book cleanup' in n.message
                   and n.data.get('symbol') == 'S00']
        assert len(cleanup) == 1
        reopened = {(i.action, i.asset.symbol)
                    for d in out.values() for i in d}
        assert ('BUY', 'S00') in reopened
        # The tracked leg is now a NEW one (entry_day advanced past 0): the
        # original stale leg was dropped, not silently kept.
        assert desk._book_positions[stock('S00')]['entry_day'] > 0


# ----------------------------------------------------------------------
# Graceful degrade (never raises, sane/empty book)
# ----------------------------------------------------------------------
class TestGracefulDegrade:
    def test_unfitted_model_yields_no_opens_with_model_note(self):
        frames = universe(6)
        dates = frames['S00'].index
        # min_train_days far beyond the window -> predict() returns None.
        desk = make_desk(default=monotone_scores(6), min_train_days=999,
                         quantile=0.2)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []
        model_notes = [n for n in desk.notes if n.category == 'model'
                       and 'unfitted' in n.message]
        assert len(model_notes) == 1
        assert desk.walk_forward_fits == []

    def test_too_few_scored_degrades_with_info_note(self):
        # Only 3 symbols, default min_scored = 4 -> degrade, no opens.
        frames = universe(3)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(3), quantile=0.2)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []
        info = [n for n in desk.notes if n.category == 'info'
                and 'Degrade' in n.message]
        assert len(info) == 1
        assert info[0].data['n_scored'] == 3
        assert info[0].data['min_scored'] == 4

    def test_empty_universe_does_not_crash(self):
        desk = make_desk(default={}, quantile=0.2)
        out = desk.generate_intents({}, pd.Timestamp('2023-01-02'),
                                    PortfolioManager(100000.0))
        assert out == []

    def test_all_equal_scores_still_picks_a_deterministic_side(self):
        # All scores identical -> ranking falls back to the symbol tiebreak,
        # so the book is still well-formed (no overlap, no crash).
        frames = universe(10)
        dates = frames['S00'].index
        flat = {f'S{i:02d}': 0.05 for i in range(10)}
        desk = make_desk(default=flat, quantile=0.2, target_gross=1.0)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        longs = sorted(i.asset.symbol for i in out[dates[0]]
                       if i.action == 'BUY')
        shorts = sorted(i.asset.symbol for i in out[dates[0]]
                        if i.action == 'SHORT')
        # Deterministic symbol tiebreak: S00/S01 top, S08/S09 bottom.
        assert longs == ['S00', 'S01']
        assert shorts == ['S08', 'S09']
        assert not (set(longs) & set(shorts))  # never both sides

    def test_min_scored_two_allows_single_long_single_short(self):
        # min_scored=2 with exactly 2 symbols -> k=1, one long one short.
        frames = universe(2)
        dates = frames['S00'].index
        desk = make_desk(default={'S00': 0.3, 'S01': -0.3}, quantile=0.5,
                         target_gross=1.0, min_scored=2)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        actions = {i.asset.symbol: i.action for i in out[dates[0]]}
        assert actions == {'S00': 'BUY', 'S01': 'SHORT'}

    def test_single_symbol_universe_never_overlaps(self):
        # One symbol -> below min_scored: degrade, no opens, no crash.
        frames = universe(1)
        dates = frames['S00'].index
        desk = make_desk(default={'S00': 0.2}, quantile=0.5, min_scored=2)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []


# ----------------------------------------------------------------------
# Walk-forward refit cadence + tagged fits + committee averaging
# ----------------------------------------------------------------------
class TestWalkForwardAndCommittee:
    def test_predict_before_fit_yields_no_opens(self):
        # A controller that hasn't fitted yet -> scores is None -> no opens.
        frames = universe(6)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(6), min_train_days=999)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []

    def test_refit_is_noted_and_fits_are_tagged(self, make_ohlcv,
                                                patch_market_data):
        df = make_ohlcv(n_days=160, seed=7)
        patch_market_data({'TEST': df})
        controller = WalkForwardController(
            GradientBoostingModel(n_estimators=10),
            train_window_days=252, refit_every_days=21, min_train_days=120)
        desk = TwoSigmaDesk(controller=controller, risk_manager=wide_risk())
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=0.001)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)

        # min_train_days 120, refit every 21: fits on days 120 and 141.
        walk_forward = report['walk_forward']
        assert len(walk_forward) == 2
        for fit in walk_forward:
            # Each fit is tagged with its model id (here the injected
            # single controller is the 'custom' tag).
            assert set(fit) == {'fit_date', 'train_start', 'train_end',
                                'n_samples', 'model'}
            assert fit['model'] == 'custom'
            assert fit['train_end'] <= fit['fit_date']
        refit_notes = [n for n in desk.notes if n.category == 'model'
                       and 'refit' in n.message]
        assert len(refit_notes) == 2

    def test_walk_forward_fits_sorted_and_tagged_per_member(self):
        # Two-member committee, each member injected through its own
        # controller via the committee path; every fit carries its tag.
        frames = universe(8, n_days=8)
        dates = frames['S00'].index
        desk = TwoSigmaDesk(models=['gbm', 'lightgbm'],
                            risk_manager=wide_risk())
        # Replace the freshly-built committee controllers with fast stubs so
        # both members fit on day 0 (keeps the test offline and quick).
        desk._committee = [
            ('gbm', stub_controller(monotone_scores(8))),
            ('lightgbm', stub_controller(monotone_scores(8))),
        ]
        desk._model_label = 'gbm+lightgbm'
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        fits = desk.walk_forward_fits
        assert sorted(f.model for f in fits) == ['gbm', 'lightgbm']
        # Sorted by (fit_date, model): same date, so 'gbm' before 'lightgbm'.
        assert [f.model for f in fits] == ['gbm', 'lightgbm']

    def test_committee_averages_member_scores(self):
        # Two members disagree on magnitude; the desk averages their
        # centered scores. Member A ranks S00 highest, member B ranks it
        # lowest of the two extremes — the average still ranks the universe
        # so the book is well-formed and order-independent.
        frames = universe(10)
        dates = frames['S00'].index
        member_a = {f'S{i:02d}': 0.4 - 0.08 * i for i in range(10)}
        member_b = {f'S{i:02d}': 0.2 - 0.04 * i for i in range(10)}
        desk = TwoSigmaDesk(models=['gbm', 'lightgbm'],
                            risk_manager=wide_risk(), quantile=0.2,
                            target_gross=1.0)
        desk._committee = [
            ('gbm', stub_controller(member_a)),
            ('lightgbm', stub_controller(member_b)),
        ]
        desk._model_label = 'gbm+lightgbm'
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        longs = sorted(i.asset.symbol for i in out[dates[0]]
                       if i.action == 'BUY')
        shorts = sorted(i.asset.symbol for i in out[dates[0]]
                        if i.action == 'SHORT')
        assert longs == ['S00', 'S01']
        assert shorts == ['S08', 'S09']
        # The averaged score on each opened name is the mean of the members.
        sig = {n.data['symbol']: n.data['score'] for n in desk.notes
               if n.category == 'signal'}
        assert sig['S00'] == pytest.approx((0.4 + 0.2) / 2)

    def test_committee_none_until_every_member_fitted(self):
        # One member is fitted, the other never fits -> the committee has no
        # consensus, so scores is None and there are no opens.
        frames = universe(8)
        dates = frames['S00'].index
        desk = TwoSigmaDesk(models=['gbm', 'lightgbm'],
                            risk_manager=wide_risk())
        desk._committee = [
            ('gbm', stub_controller(monotone_scores(8))),
            ('lightgbm', stub_controller(monotone_scores(8),
                                         min_train_days=999)),  # never fits
        ]
        desk._model_label = 'gbm+lightgbm'
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []
        assert [n for n in desk.notes if n.category == 'model'
                and 'unfitted' in n.message]


# ----------------------------------------------------------------------
# End-to-end with the real engine + models (fetch monkeypatched)
# ----------------------------------------------------------------------
class TestEndToEndWithEngine:
    N_DAYS = 320

    @pytest.fixture
    def synthetic_universe(self):
        index = pd.bdate_range('2022-01-03', periods=self.N_DAYS)

        def frame(seed, drift):
            rng = np.random.default_rng(seed)
            rets = rng.normal(drift, 0.012, self.N_DAYS)
            close = 100.0 * np.cumprod(1.0 + rets)
            return pd.DataFrame({
                'open': close, 'high': close * 1.002, 'low': close * 0.998,
                'close': close,
                'volume': rng.integers(400_000, 900_000,
                                       self.N_DAYS).astype(float),
            }, index=index)

        # A cross-sectional drift spread gives the model a real ranking.
        return {f'S{i:02d}': frame(100 + i, 0.0004 * (i - 6))
                for i in range(12)}

    def _build_desk(self, **kwargs):
        kwargs.setdefault('risk_manager', wide_risk())
        kwargs.setdefault('target_gross', 0.6)
        return TwoSigmaDesk(
            controller=WalkForwardController(
                GradientBoostingModel(n_estimators=10),
                train_window_days=120, refit_every_days=21,
                min_train_days=80),
            **kwargs)

    @pytest.fixture
    def report_and_desk(self, synthetic_universe, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return synthetic_universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = self._build_desk()
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=0.001)
        report = engine.run(sorted(synthetic_universe),
                            '2022-01-01', '2023-06-30', benchmark_symbol=None)
        return report, desk, engine

    def test_report_carries_desk_and_walk_forward_blocks(self,
                                                         report_and_desk):
        report, desk, _ = report_and_desk
        assert report['desk'] == {'key': 'twosigma',
                                  'name': 'Two Sigma Desk'}
        assert report['strategy'] == 'Two Sigma Desk'

        walk_forward = report['walk_forward']
        assert walk_forward
        for fit in walk_forward:
            assert set(fit) == {'fit_date', 'train_start', 'train_end',
                                'n_samples', 'model'}
            assert fit['model'] == 'custom'
            assert fit['train_end'] <= fit['fit_date']

        notes = report['trader_notes']
        assert notes
        json.dumps(notes)  # JSON-safe end to end
        categories = {n['category'] for n in notes}
        assert {'signal', 'model', 'allocation'} <= categories
        for note in notes:
            assert note['desk'] == 'Two Sigma Desk'

        # Phase regression: twosigma is not a regime/pod/structures desk.
        assert 'regime_series' not in report
        assert 'pod_history' not in report
        assert 'orchestrator' not in report  # single-desk run

    def test_report_carries_research_integrity_machinery(self,
                                                         report_and_desk):
        report, _, _ = report_and_desk
        summary = report['summary']
        # Deflated / probabilistic Sharpe present (may be None on a flat
        # synthetic series, but the KEYS must exist).
        assert 'psr' in summary
        assert 'deflated_sharpe' in summary
        assert 'n_trials' in summary
        # n_trials deflates by the number of walk-forward fits in desk mode.
        assert summary['n_trials'] == max(1, len(report['walk_forward']))

        # Per-fold OOS significance block, available for a single desk.
        oos = report['oos_folds']
        assert oos['available'] is True
        assert set(oos) >= {'available', 'alpha', 'n_folds',
                            'n_testable_folds', 'folds',
                            'n_significant_bonferroni', 'n_significant_bh'}
        json.dumps(oos)

    def test_run_is_deterministic_trades_and_final_value(
            self, synthetic_universe, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return synthetic_universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        def run_once():
            desk = self._build_desk()
            engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                    commission=0.001)
            return engine.run(sorted(synthetic_universe),
                              '2022-01-01', '2023-06-30',
                              benchmark_symbol=None)

        first = run_once()
        second = run_once()
        assert first['trades'] == second['trades']
        assert (first['portfolio_history'][-1]['portfolio_value']
                == second['portfolio_history'][-1]['portfolio_value'])

    def test_trades_only_in_desk_actions_and_no_one_step_flips(
            self, report_and_desk):
        report, _, _ = report_and_desk
        assert report['trades']  # the desk acted over the window
        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}
        # Per symbol the running position never flips in one step: a long is
        # always opened from flat and closed by SELL; a short from flat and
        # closed by COVER. Never doubled, never flipped without closing.
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


# ----------------------------------------------------------------------
# Regression: the HIGH reconcile bug — no phantom daily churn
# ----------------------------------------------------------------------
class TestHeldAcrossDaysRegression:
    """Pins the HIGH bug fix: previously generate_intents blind-closed the
    ENTIRE book every day (a pre-scoring reconcile against desired={}), so
    every held name churned (close+reopen) daily and every close routed
    through the emergency orphan sweep. The fix reconciles ONCE per day,
    AFTER scoring, against the real desired book — a name that keeps its side
    is HELD untouched, and the orphan sweep falls silent in steady state.

    These run through the REAL BacktestEngine with a STABLE (monotone) stub
    ranking so a top/bottom name keeps its seat every single day — exactly the
    case the old code mis-handled. With the bug present each name would show
    ~one BUY/SELL (or SHORT/COVER) cycle per trading day and the orphan-sweep
    count would dominate the closes; with the fix each is opened once and held,
    and orphan-sweep notes are ~0.
    """

    def _run_stable(self, monkeypatch, n_days, n_symbols=10):
        frames = {f'S{i:02d}': quiet_frame(n=n_days, seed=400 + i)
                  for i in range(n_symbols)}

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        # A fixed monotone ranking every day: S00 always top, S(n-1) bottom.
        desk = TwoSigmaDesk(
            controller=stub_controller(monotone_scores(n_symbols)),
            risk_manager=wide_risk(), quantile=0.2, target_gross=1.0)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(sorted(frames), '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)
        return desk, engine, report

    def test_top_ranked_name_is_held_across_many_days_not_churned(
            self, monkeypatch):
        n_days = 30  # ~22 trading days inside the window
        desk, engine, report = self._run_stable(monkeypatch, n_days)
        n_trading_days = len(report['portfolio_history'])
        assert n_trading_days >= 15  # many days, so churn would be obvious

        # S00 (always top long) and S09 (always bottom short) keep their seat
        # the WHOLE run: opened EXACTLY ONCE, never closed. With the bug each
        # would have churned once per trading day.
        s00 = [t['action'] for t in report['trades'] if t['symbol'] == 'S00']
        s09 = [t['action'] for t in report['trades'] if t['symbol'] == 'S09']
        assert s00 == ['BUY']            # held long the entire window
        assert s09 == ['SHORT']          # held short the entire window
        assert 'SELL' not in s00 and 'COVER' not in s09

        # Still tracked and still open on the right side at the end.
        assert desk._book_positions[stock('S00')]['direction'] == 'long'
        assert desk._book_positions[stock('S09')]['direction'] == 'short'
        assert engine.portfolio.get_position(stock('S00')).quantity > 0
        assert engine.portfolio.get_position(stock('S09')).quantity < 0

        # The whole stable book is opened once and held: exactly 4 opens
        # (2 longs + 2 shorts), zero closes, far fewer than the ~4 * trading
        # days the daily-churn bug would have produced.
        opens = [t for t in report['trades']
                 if t['action'] in ('BUY', 'SHORT')]
        closes = [t for t in report['trades']
                  if t['action'] in ('SELL', 'COVER')]
        assert len(opens) == 4
        assert len(closes) == 0
        assert len(report['trades']) < n_trading_days  # NOT one churn per day

    def test_orphan_sweep_is_silent_in_steady_state(self, monkeypatch):
        # With the reconcile fix, a stable held book never trips the emergency
        # orphan sweep: orphan-sweep notes are 0. (Before the fix, 100% of the
        # daily blind-closes routed through the sweep.)
        desk, engine, report = self._run_stable(monkeypatch, n_days=30)
        orphan_notes = sum('Orphan sweep' in n.message for n in desk.notes)
        assert orphan_notes == 0

    def test_drift_universe_rebalance_exits_carry_closes_not_sweeps(
            self, monkeypatch):
        # A noisier, realistic ranking (cross-sectional drift through the real
        # GBM model) DOES rebalance names in and out — but every close routes
        # through a normal 'no longer in the ... side' rebalance exit, and the
        # orphan sweep stays ~0 (a tiny fraction of total closes at worst).
        N_DAYS = 320
        index = pd.bdate_range('2022-01-03', periods=N_DAYS)

        def frame(seed, drift):
            rng = np.random.default_rng(seed)
            rets = rng.normal(drift, 0.012, N_DAYS)
            close = 100.0 * np.cumprod(1.0 + rets)
            return pd.DataFrame({
                'open': close, 'high': close * 1.002, 'low': close * 0.998,
                'close': close,
                'volume': rng.integers(400_000, 900_000,
                                       N_DAYS).astype(float),
            }, index=index)

        universe_ = {f'S{i:02d}': frame(100 + i, 0.0004 * (i - 6))
                     for i in range(12)}

        def fake_fetch(self, symbol, start_date, end_date):
            return universe_.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = TwoSigmaDesk(
            controller=WalkForwardController(
                GradientBoostingModel(n_estimators=10),
                train_window_days=120, refit_every_days=21, min_train_days=80),
            risk_manager=wide_risk(), target_gross=0.6)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=0.001)
        report = engine.run(sorted(universe_),
                            '2022-01-01', '2023-06-30', benchmark_symbol=None)

        closes = sum(1 for t in report['trades']
                     if t['action'] in ('SELL', 'COVER'))
        assert closes > 0  # this universe really does rebalance
        orphan_notes = sum('Orphan sweep' in n.message for n in desk.notes)
        rebalance_exits = sum('no longer in the' in n.message
                              for n in desk.notes)
        # The HIGH-bug signature was 100% orphan-sweep closes; the fix routes
        # closes through rebalance exits and keeps the sweep ~0.
        assert orphan_notes == 0
        assert rebalance_exits >= closes  # exits cover every close (+ in-flight)


# ----------------------------------------------------------------------
# Regression: trades under the DEFAULT RiskManager (the max_name_size clamp)
# ----------------------------------------------------------------------
class TestDefaultRiskManagerAndClamp:
    def test_trades_under_default_risk_manager(self):
        # A desk built with the DEFAULT RiskManager (10% position cap) must
        # actually OPEN positions: the max_name_size clamp (0.10) keeps every
        # leg inside the cap. Before the clamp, (0.5*1.0)/2 = 0.25 > 0.10 cap,
        # so apply_risk blocked every leg and the desk no-opped to flat.
        frames = universe(10)
        dates = frames['S00'].index
        desk = TwoSigmaDesk(
            controller=stub_controller(monotone_scores()),
            risk_manager=RiskManager(),  # DEFAULT: 10% position cap
            quantile=0.2, target_gross=1.0)
        portfolio = PortfolioManager(100000.0)
        out = drive(desk, frames, dates[:1], portfolio)
        intents = out[dates[0]]
        opens = [i for i in intents if i.action in ('BUY', 'SHORT')]
        assert len(opens) == 4  # 2 longs + 2 shorts actually opened
        # Each clamped to the cap, so apply_risk admits them.
        for intent in opens:
            assert intent.size_fraction == pytest.approx(0.10)

    def test_degrade_unfitted_flattens_a_held_book(self):
        # A book is opened on day 0; on day 1 the model goes UNFITTED (scores
        # is None) -> the degrade branch FLATTENS the held book (closes
        # emitted) and opens nothing. Driven by hand to flip the fit state.
        frames = universe(10, n_days=4)
        dates = frames['S00'].index
        # A controller whose stub never fits would never open on day 0. Use a
        # normal stub for day 0, then force the controller unfitted for day 1
        # by swapping in a predict() that returns None.
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        portfolio = PortfolioManager(100000.0)
        out0 = drive(desk, frames, dates[:1], portfolio)
        opens0 = [i for i in out0[dates[0]] if i.action in ('BUY', 'SHORT')]
        assert len(opens0) == 4  # book is held going into day 1
        # Simulate the fills so the positions are actually open on day 1.
        for intent in opens0:
            qty = 100 if intent.action == 'BUY' else -100
            portfolio.positions[intent.asset] = Position(
                asset=intent.asset, quantity=qty, avg_entry_price=100.0,
                current_price=100.0, timestamp=pd.Timestamp(dates[0]))

        # Force the model unfitted for day 1: predict() now returns None.
        _, controller = desk._committee[0]
        controller.predict = lambda data, date: None

        out1 = drive(desk, frames, dates[1:2], portfolio)
        intents1 = out1[dates[1]]
        # Degrade branch: closes emitted for the held book, NO new opens.
        assert [i for i in intents1
                if i.action in ('BUY', 'SHORT')] == []
        closes = [i for i in intents1 if i.action in ('SELL', 'COVER')]
        assert len(closes) == 4  # the whole held book is flattened
        model_notes = [n for n in desk.notes if n.category == 'model'
                       and 'unfitted' in n.message]
        assert model_notes  # the degrade was noted

    def test_degrade_too_few_scored_flattens_a_held_book(self):
        # Same flatten path via the OTHER degrade branch: a held book, then a
        # day where too few symbols score (n_scored < min_scored) flattens it.
        frames = universe(10, n_days=4)
        dates = frames['S00'].index
        # Day 0 scores all 10; day 1 scores only 3 (< default min_scored 4).
        thin = {f'S{i:02d}': 0.4 - 0.08 * i for i in range(3)}
        schedule = {dates[1]: thin}
        desk = make_desk(default=monotone_scores(), schedule=schedule,
                         quantile=0.2, target_gross=1.0)
        portfolio = PortfolioManager(100000.0)
        out0 = drive(desk, frames, dates[:1], portfolio)
        opens0 = [i for i in out0[dates[0]] if i.action in ('BUY', 'SHORT')]
        assert len(opens0) == 4
        for intent in opens0:
            qty = 100 if intent.action == 'BUY' else -100
            portfolio.positions[intent.asset] = Position(
                asset=intent.asset, quantity=qty, avg_entry_price=100.0,
                current_price=100.0, timestamp=pd.Timestamp(dates[0]))

        out1 = drive(desk, frames, dates[1:2], portfolio)
        intents1 = out1[dates[1]]
        assert [i for i in intents1 if i.action in ('BUY', 'SHORT')] == []
        closes = [i for i in intents1 if i.action in ('SELL', 'COVER')]
        assert len(closes) == 4
        info = [n for n in desk.notes if n.category == 'info'
                and 'Degrade' in n.message]
        assert info

    @pytest.mark.parametrize('bad', [0.0, -0.1, 1.5, 2.0])
    def test_bad_max_name_size_raises(self, bad):
        with pytest.raises(ValueError, match='max_name_size'):
            TwoSigmaDesk(max_name_size=bad)

    def test_max_name_size_one_is_allowed(self):
        # The boundary value 1.0 is valid (0 < x <= 1).
        desk = TwoSigmaDesk(max_name_size=1.0)
        assert desk.get_status()['max_name_size'] == 1.0

    def test_get_status_exposes_max_name_size(self):
        desk = TwoSigmaDesk(max_name_size=0.25)
        assert desk.get_status()['max_name_size'] == 0.25
        # Default surfaces too.
        assert TwoSigmaDesk().get_status()['max_name_size'] == pytest.approx(
            0.10)

    def test_clamp_binds_as_documented(self):
        # Per-leg size is min(max_name_size, (0.5*target_gross)/k). With a
        # SMALL max_name_size the clamp binds; with a LARGE one the per-leg
        # target binds — assert both regimes.
        frames = universe(10)
        dates = frames['S00'].index
        # k = int(0.2*10) = 2 per side; (0.5*1.0)/2 = 0.25 unclamped.
        clamped = make_desk(default=monotone_scores(), quantile=0.2,
                            target_gross=1.0, max_name_size=0.05)
        out = drive(clamped, frames, dates[:1], PortfolioManager(100000.0))
        for intent in out[dates[0]]:
            assert intent.size_fraction == pytest.approx(0.05)  # clamp wins

        unclamped = make_desk(default=monotone_scores(), quantile=0.2,
                             target_gross=1.0, max_name_size=0.50)
        out = drive(unclamped, frames, dates[:1], PortfolioManager(100000.0))
        for intent in out[dates[0]]:
            assert intent.size_fraction == pytest.approx(0.25)  # target wins


# ----------------------------------------------------------------------
# Orphan sweep through the real engine (mirrors renaissance regression)
# ----------------------------------------------------------------------
class TestOrphanSweep:
    """The engine holds a pending intent up to MAX_PENDING_DAYS (5) trading
    days, but the desk reconciles tracking away after RECONCILE_GRACE_DAYS
    (2) — so an entry can FILL after its tracking was dropped. The orphan
    sweep must close such an untracked desk-traded position."""

    def test_late_fill_after_cleanup_is_swept(self, monkeypatch):
        n_days = 6
        frames = {f'S{i:02d}': quiet_frame(n=n_days, seed=200 + i)
                  for i in range(6)}
        dates = frames['S00'].index
        assert dates[0].weekday() == 0  # Monday start
        # GAP has NO bars on the two days after the day-0 intent; its bar
        # returns on day 3 — after the desk reconciled its tracking away.
        gap = quiet_frame(n=n_days, seed=199)
        frames['GAP'] = gap.drop(index=[dates[1], dates[2]])

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)

        # GAP scores lowest on day 0 -> shorted (6 S-names + GAP, quantile
        # 0.2 -> k=1, GAP is the single bottom name). Inject a stub so the
        # ranking is fully controlled and the model fits on day 0. AFTER GAP's
        # bar returns (day 3) its score is middling, so the desk does NOT
        # re-select it once the sweep frees it — isolating the sweep COVER.
        base = {f'S{i:02d}': 0.4 - 0.08 * i for i in range(6)}
        base['GAP'] = -0.9
        schedule = {d: {**base, 'GAP': 0.1} for d in dates[3:]}
        desk = TwoSigmaDesk(
            controller=stub_controller(base, schedule),
            risk_manager=wide_risk(),
            quantile=0.2, target_gross=1.0, min_scored=2)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(sorted(frames), '2023-01-01', '2023-01-31',
                            benchmark_symbol=None)

        # Tracking was reconciled away on day 2, BEFORE the day-3 fill.
        cleanup = [n for n in desk.notes if 'Book cleanup' in n.message
                   and n.data.get('symbol') == 'GAP']
        assert len(cleanup) == 1
        assert cleanup[0].timestamp == dates[2]

        # The orphan SHORT filled on day 3 and the desk's sweep COVERed it.
        gap_trades = [t for t in report['trades'] if t['symbol'] == 'GAP']
        assert [t['action'] for t in gap_trades] == ['SHORT', 'COVER']
        assert gap_trades[0]['date'] == dates[3]
        assert gap_trades[1]['date'] == dates[4]
        position = engine.portfolio.get_position(stock('GAP'))
        assert position is None or position.quantity == 0

        # The GAP-specific sweep fired on the fill day (day 3) as 'risk'
        # note(s). Reconcile runs twice per day, so the untracked late fill
        # is swept in both passes — the engine dedups the COVER intent to a
        # single trade (asserted above), but both notes are stamped day 3.
        # (Other churning names produce their own sweeps under the
        # close-flat-then-reopen mechanic; we filter to GAP.)
        gap_sweeps = [n for n in desk.notes if 'Orphan sweep' in n.message
                      and n.data.get('symbol') == 'GAP']
        assert gap_sweeps  # at least one
        for note in gap_sweeps:
            assert note.category == 'risk'
            assert note.timestamp == dates[3]
            assert note.data['direction'] == 'short'


# ----------------------------------------------------------------------
# FundOrchestrator interop: twosigma composes with foundation
# ----------------------------------------------------------------------
class TestFundOrchestratorInterop:
    N_DAYS = 280

    @pytest.fixture
    def synthetic_universe(self):
        index = pd.bdate_range('2022-01-03', periods=self.N_DAYS)

        def frame(seed, drift):
            rng = np.random.default_rng(seed)
            rets = rng.normal(drift, 0.012, self.N_DAYS)
            close = 100.0 * np.cumprod(1.0 + rets)
            return pd.DataFrame({
                'open': close, 'high': close * 1.002, 'low': close * 0.998,
                'close': close,
                'volume': rng.integers(400_000, 900_000,
                                       self.N_DAYS).astype(float),
            }, index=index)

        return {f'S{i:02d}': frame(300 + i, 0.0004 * (i - 4))
                for i in range(8)}

    def test_two_desk_fund_composes_and_nets(self, synthetic_universe,
                                             monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return synthetic_universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        twosigma = TwoSigmaDesk(
            capital_allocation=0.5, risk_manager=wide_risk(),
            target_gross=0.6,
            controller=WalkForwardController(
                GradientBoostingModel(n_estimators=10),
                train_window_days=120, refit_every_days=21,
                min_train_days=80))
        foundation = FoundationDesk(capital_allocation=0.5)
        orchestrator = FundOrchestrator([twosigma, foundation])
        engine = BacktestEngine(orchestrator=orchestrator,
                                initial_capital=100000.0, commission=0.001)
        report = engine.run(sorted(synthetic_universe),
                            '2022-01-01', '2023-03-31', benchmark_symbol=None)

        # Both desks appear in the fund roster; capital nets to the active
        # allocation; the run completed without error.
        assert 'orchestrator' in report
        roster = report['orchestrator']['desks']
        assert [d['key'] for d in roster] == ['twosigma', 'foundation']
        assert report['orchestrator']['active_capital'] == pytest.approx(1.0)

        # Per-fold OOS is N/A for a netted multi-desk book (by contract).
        assert report['oos_folds']['available'] is False
        # The fund still carries the deflated/probabilistic Sharpe summary.
        assert 'deflated_sharpe' in report['summary']
        assert 'psr' in report['summary']

        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}


class TestSignalStrengthSizing:
    """Phase 5: opt-in score-excess (|score|) sizing on the shared
    cross-sectional book. Default off -> equal-weight (byte-identical);
    on -> each side's flat budget is split WITHIN the side by |score|,
    water-filled through the name cap, keeping long gross == short gross."""

    # Symmetric scores: |top-3| mirrors |bottom-3| so the two sides carry
    # the SAME conviction distribution and stay dollar-neutral exactly.
    SYMMETRIC = {'S00': 0.40, 'S01': 0.20, 'S02': 0.10,
                 'S03': 0.0, 'S04': 0.0, 'S05': 0.0, 'S06': 0.0,
                 'S07': -0.10, 'S08': -0.20, 'S09': -0.40}

    def _sizes(self, **kwargs):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=self.SYMMETRIC, quantile=0.3,
                         target_gross=1.0, max_name_size=0.30, **kwargs)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        return {i.asset.symbol: i.size_fraction for i in out[dates[0]]}

    def test_default_is_equal_weight(self):
        # Flag off: every leg on a side gets the identical flat size.
        sizes = self._sizes()  # size_by_signal_strength defaults False
        flat = (0.5 * 1.0) / 3  # half gross / k, cap (0.30) not binding
        for sym in ('S00', 'S01', 'S02', 'S07', 'S08', 'S09'):
            assert sizes[sym] == pytest.approx(flat)

    def test_conviction_orders_within_side_and_caps_gross(self):
        sizes = self._sizes(size_by_signal_strength=True)
        flat = (0.5 * 1.0) / 3
        # Stronger |score| -> more capital, WITHIN each side.
        assert sizes['S00'] > sizes['S01'] > sizes['S02']
        assert sizes['S09'] > sizes['S08'] > sizes['S07']
        # Concentration can rise above the average, but never the name cap.
        assert max(sizes.values()) > flat
        assert all(v <= 0.30 + 1e-12 for v in sizes.values())

    def test_dollar_neutrality_preserved(self):
        sizes = self._sizes(size_by_signal_strength=True)
        long_gross = sizes['S00'] + sizes['S01'] + sizes['S02']
        short_gross = sizes['S07'] + sizes['S08'] + sizes['S09']
        # Each side deploys its intended half-gross exactly.
        assert long_gross == pytest.approx(short_gross)
        assert long_gross == pytest.approx(0.5)

    def test_asymmetric_scores_and_binding_cap_stay_dollar_neutral(self):
        # Regression: the old min(flat_size, proportional_size) clamp silently
        # discarded the strong long's rejected allocation. Since the short
        # scores have a different shape, the two discarded amounts differed
        # and produced a net book. Water-filling must re-spread both sides to
        # the same 50% budget, even when the strongest long hits 30%.
        scores = {
            'S00': 10.0, 'S01': 2.0, 'S02': 1.0,
            'S03': 0.04, 'S04': 0.03, 'S05': 0.02, 'S06': 0.01,
            'S07': -0.10, 'S08': -0.20, 'S09': -0.30,
        }
        frames = universe(10)
        date = frames['S00'].index[0]
        desk = make_desk(
            default=scores, quantile=0.3, target_gross=1.0,
            max_name_size=0.30, size_by_signal_strength=True)
        out = drive(desk, frames, [date], PortfolioManager(100000.0))[date]
        longs = [i.size_fraction for i in out if i.action == 'BUY']
        shorts = [i.size_fraction for i in out if i.action == 'SHORT']

        assert sum(longs) == pytest.approx(0.5)
        assert sum(shorts) == pytest.approx(0.5)
        assert max(longs) == pytest.approx(0.30)  # cap actually binds
        assert all(size <= 0.30 + 1e-12 for size in longs + shorts)


class TestUncertaintyScaledSizing:
    """Kronos idea #3: opt-in committee-disagreement sizing. Default off ->
    byte-identical; on -> a name's conviction size is shrunk by
    1 / (1 + lambda * normalized_dispersion), where dispersion is the std of
    the committee members' per-symbol scores, normalized by the cross-section
    median. Concentrates risk on the names the ensemble AGREES on while
    renormalizing per side so gross and dollar-neutrality are preserved."""

    def test_bad_disagreement_lambda_raises(self):
        with pytest.raises(ValueError):
            make_desk(default=monotone_scores(), disagreement_lambda=-0.1)

    def test_conviction_size_shrinks_with_disagreement(self):
        # The literal acceptance test: two EQUAL-|score| legs on one side, the
        # one the committee disagrees about more is sized SMALLER. Exercises
        # _conviction_sizes directly so the multiplier is isolated from ranking.
        desk = make_desk(default=monotone_scores(), max_name_size=0.20,
                         shrink_by_disagreement=True, disagreement_lambda=1.0)
        sizes = desk._conviction_sizes(
            ['A', 'B'], {'A': 0.2, 'B': 0.2}, flat_size=0.10,
            disagreement={'A': 2.0, 'B': 0.0})  # A disagrees more
        assert sizes['A'] < sizes['B']

    def test_committee_dispersion_is_std_across_members(self):
        # Two members; per-symbol dispersion is the population std across them.
        frames = universe(4)
        dates = frames['S00'].index
        desk = TwoSigmaDesk(models=['gbm', 'lightgbm'],
                            risk_manager=wide_risk())
        desk._committee = [
            ('gbm', stub_controller({'S00': 0.4, 'S01': 0.2})),
            ('lightgbm', stub_controller({'S00': 0.0, 'S01': 0.2})),
        ]
        for _, controller in desk._committee:
            controller.maybe_refit(frames, dates[0])
        disp = desk._committee_dispersion(frames, dates[0])
        assert disp['S00'] == pytest.approx(0.2)  # pstdev([0.4, 0.0])
        assert disp['S01'] == pytest.approx(0.0)  # members agree -> no spread

    def test_single_member_committee_has_no_dispersion(self):
        # One member -> nothing to disagree with -> dispersion empty, and the
        # multiplier degrades to 1 (a single-member desk is unaffected).
        desk = make_desk(default=monotone_scores(),
                         shrink_by_disagreement=True)
        frames = universe(10)
        dates = frames['S00'].index
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        sliced = {s: f[f.index <= dates[0]] for s, f in frames.items()}
        assert desk._committee_dispersion(sliced, dates[0]) == {}

    # A 2-member committee whose averaged scores rank S00..S09 monotone, with
    # SYMMETRIC disagreement: S00 (top long) and S09 (bottom short) carry 4x
    # the cross-section's median dispersion, every other name carries the
    # baseline. member = mean +/- spread, so the mean is preserved and the
    # per-symbol pstdev equals `spread`.
    MEAN = {'S00': 0.40, 'S01': 0.20, 'S02': 0.10, 'S03': 0.0, 'S04': 0.0,
            'S05': 0.0, 'S06': 0.0, 'S07': -0.10, 'S08': -0.20, 'S09': -0.40}
    SPREAD = {s: (0.20 if s in ('S00', 'S09') else 0.05) for s in MEAN}

    def _committee_sizes(self, **kwargs):
        frames = universe(10)
        dates = frames['S00'].index
        desk = TwoSigmaDesk(models=['gbm', 'lightgbm'], risk_manager=wide_risk(),
                            quantile=0.3, target_gross=1.0, max_name_size=0.30,
                            **kwargs)
        member_a = {s: self.MEAN[s] + self.SPREAD[s] for s in self.MEAN}
        member_b = {s: self.MEAN[s] - self.SPREAD[s] for s in self.MEAN}
        desk._committee = [('gbm', stub_controller(member_a)),
                           ('lightgbm', stub_controller(member_b))]
        desk._model_label = 'gbm+lightgbm'
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        return {i.asset.symbol: i.size_fraction for i in out[dates[0]]}

    def test_committee_off_is_equal_weight(self):
        # Flag off: the 2-member committee sizes equal-weight (flat) — the new
        # path is inert, so the book is byte-identical to before.
        sizes = self._committee_sizes()  # shrink_by_disagreement defaults False
        flat = (0.5 * 1.0) / 3
        for sym in ('S00', 'S01', 'S02', 'S07', 'S08', 'S09'):
            assert sizes[sym] == pytest.approx(flat)

    def test_committee_on_shrinks_the_disagreed_name(self):
        sizes = self._committee_sizes(shrink_by_disagreement=True,
                                      disagreement_lambda=1.0)
        flat = (0.5 * 1.0) / 3
        # The high-disagreement leg on each side is sized SMALLER than its
        # equal-conviction, low-disagreement peers.
        assert sizes['S00'] < sizes['S01']
        assert sizes['S00'] < sizes['S02']
        assert sizes['S09'] < sizes['S08']
        assert sizes['S09'] < sizes['S07']
        # Rejected allocation is re-spread, so agreed names may exceed the
        # average while every leg remains below max_name_size.
        assert sizes['S01'] > flat
        assert sizes['S08'] > flat
        assert all(v <= 0.30 + 1e-12 for v in sizes.values())

    def test_committee_on_preserves_dollar_neutrality(self):
        sizes = self._committee_sizes(shrink_by_disagreement=True,
                                      disagreement_lambda=1.0)
        long_gross = sizes['S00'] + sizes['S01'] + sizes['S02']
        short_gross = sizes['S07'] + sizes['S08'] + sizes['S09']
        # Both sides retain their full budget (the book stays neutral).
        assert long_gross == pytest.approx(short_gross)
        assert long_gross == pytest.approx(0.5)
