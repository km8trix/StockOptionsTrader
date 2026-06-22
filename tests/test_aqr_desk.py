"""Tests for desks.aqr.AqrDesk — the TRANSPARENT classical-quant
cross-sectional long/short desk, the factor-model counterpart to Two Sigma.

AqrDesk REUSES the shared desks.cross_sectional.CrossSectionalLongShortDesk
book verbatim (one reconcile per day, the closing-state guard, the orphan
sweep, dollar-balanced sizing with the max_name_size clamp); it supplies the
AQR identity, a fixed FactorModel controller, and the AQR signature —
per-refit factor-weight transparency notes. It is NOT model-selectable.

Unit tests drive generate_intents day by day with a STUB walk-forward score
model (so the book mechanics — quantile selection, dollar-balanced sizing,
reconcile/exit, degrade — are pinned independent of the factor model, exactly
as the Two Sigma unit tests do). The end-to-end tests run the REAL
BacktestEngine with a REAL FactorModel controller on synthetic data and check
the held-across-days regression (orphan sweep ~0), trades under the DEFAULT
RiskManager, determinism, the no-one-step-flip invariant, the factor-weight
transparency notes, the research-integrity report blocks, and
FundOrchestrator interop.

Offline, seeded, deterministic — synthetic frames only, no network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType
from data.market_data import MarketDataHandler
from desks.aqr import AqrDesk
from desks.cross_sectional import CrossSectionalLongShortDesk
from desks.foundation import FoundationDesk
from desks.models.factor import FACTOR_COLUMNS, FactorModel
from desks.orchestrator import FundOrchestrator
from desks.registry import create_desk, list_desks
from desks.walk_forward import WalkForwardController, WalkForwardModel
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def wide_risk() -> RiskManager:
    """Risk manager opened wide so the pure book mechanics show through."""
    return RiskManager(max_position_size=1.0, max_daily_loss=0.99,
                       position_stop_loss=0.90)


# ----------------------------------------------------------------------
# Stub walk-forward score model (mirrors tests/test_twosigma_desk.py)
# ----------------------------------------------------------------------
class StubScoreModel(WalkForwardModel):
    """Per-symbol centered scores by date; ``None`` for a date -> {}."""

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
    return WalkForwardController(StubScoreModel(default, schedule),
                                 min_train_days=min_train_days)


def make_desk(default=None, schedule=None, min_train_days=1,
              **kwargs) -> AqrDesk:
    kwargs.setdefault('risk_manager', wide_risk())
    return AqrDesk(
        controller=stub_controller(default, schedule, min_train_days),
        **kwargs)


# ----------------------------------------------------------------------
# Synthetic frames + a one-desk driver (no engine; fills manual)
# ----------------------------------------------------------------------
def quiet_frame(n=8, seed=5, start='2023-01-02') -> pd.DataFrame:
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
    return {f'S{i:02d}': 0.4 - 0.08 * i for i in range(n_symbols)}


def drive(desk, frames, dates, portfolio):
    by_date = {}
    for date in dates:
        sliced = {symbol: frame[frame.index <= date]
                  for symbol, frame in frames.items()}
        sliced = {symbol: frame for symbol, frame in sliced.items()
                  if not frame.empty}
        desk.set_clock(date)
        by_date[date] = desk.generate_intents(sliced, date, portfolio)
    return by_date


# ======================================================================
# Registry + construction + identity
# ======================================================================
class TestRegistryAndIdentity:
    def test_aqr_is_ready_in_the_roster(self):
        by_key = {entry['key']: entry for entry in list_desks()}
        assert 'aqr' in by_key
        assert by_key['aqr']['status'] == 'ready'
        assert by_key['aqr']['activates_in_phase'] is None
        assert by_key['aqr']['accent'] == '#f0883e'
        assert by_key['aqr']['firm_inspiration'] == 'AQR Capital Management'

    def test_create_desk_aqr_builds_the_desk(self):
        desk = create_desk('aqr')
        assert isinstance(desk, AqrDesk)
        assert isinstance(desk, CrossSectionalLongShortDesk)
        assert desk.key == 'aqr'
        assert desk.name == 'AQR Desk'
        assert desk.accent == '#f0883e'
        assert desk.capital_allocation == 1.0

    def test_capital_allocation_threads_through(self):
        desk = create_desk('aqr', capital_allocation=0.35)
        assert desk.capital_allocation == 0.35

    def test_aqr_is_not_model_selectable(self):
        # AQR is defined by its factor model — create_desk must reject a
        # model_key (it is NOT in _MODEL_SELECTABLE_DESKS).
        with pytest.raises(ValueError,
                           match='does not support model selection'):
            create_desk('aqr', model_key='factor')

    def test_default_controller_wraps_a_factor_model(self):
        desk = AqrDesk()
        _, controller = desk._committee[0]
        assert isinstance(controller, WalkForwardController)
        assert isinstance(controller.model, FactorModel)
        assert desk.get_status()['models'] == ['factor']
        assert desk.get_status()['model'] == 'factor'

    def test_no_model_key_parameter(self):
        # Unlike TwoSigma/Foundation, AqrDesk has no model_key / models kwarg.
        with pytest.raises(TypeError):
            AqrDesk(model_key='factor')

    def test_injected_controller_is_used(self):
        ctrl = stub_controller(monotone_scores())
        desk = AqrDesk(controller=ctrl)
        assert desk._committee[0][1] is ctrl

    @pytest.mark.parametrize('quantile', [0.0, -0.1, 0.6, 1.0])
    def test_bad_quantile_raises(self, quantile):
        with pytest.raises(ValueError, match='quantile'):
            AqrDesk(quantile=quantile)

    @pytest.mark.parametrize('target_gross', [0.0, -0.5, 1.5])
    def test_bad_target_gross_raises(self, target_gross):
        with pytest.raises(ValueError, match='target_gross'):
            AqrDesk(target_gross=target_gross)

    @pytest.mark.parametrize('min_scored', [0, 1, -3])
    def test_min_scored_below_two_raises(self, min_scored):
        with pytest.raises(ValueError, match='min_scored'):
            AqrDesk(min_scored=min_scored)

    @pytest.mark.parametrize('bad', [0.0, -0.1, 1.5])
    def test_bad_max_name_size_raises(self, bad):
        with pytest.raises(ValueError, match='max_name_size'):
            AqrDesk(max_name_size=bad)

    def test_factor_weights_empty_before_first_fit(self):
        # The attribution accessor is empty until the factor model fits.
        assert AqrDesk().factor_weights == {}

    def test_factor_weights_empty_for_stub_controller(self):
        # An injected stub (no factor_weights attr) yields {} gracefully.
        desk = AqrDesk(controller=stub_controller(monotone_scores()))
        assert desk.factor_weights == {}


# ======================================================================
# Cross-sectional selection + dollar-balanced sizing (shared book)
# ======================================================================
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
        assert longs == ['S00', 'S01']
        assert shorts == ['S08', 'S09']

    def test_book_is_dollar_balanced(self):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0, max_name_size=0.30)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        intents = out[dates[0]]
        longs = [i for i in intents if i.action == 'BUY']
        shorts = [i for i in intents if i.action == 'SHORT']
        expected = min(0.30, (0.5 * 1.0) / 2)
        assert expected == pytest.approx(0.25)
        for intent in longs + shorts:
            assert intent.size_fraction == pytest.approx(expected)
        long_gross = sum(i.size_fraction for i in longs)
        short_gross = sum(i.size_fraction for i in shorts)
        assert long_gross == pytest.approx(0.5)
        assert short_gross == pytest.approx(0.5)
        assert long_gross - short_gross == pytest.approx(0.0)
        assert long_gross + short_gross == pytest.approx(1.0)

    def test_intent_reasons_carry_the_aqr_prefix(self):
        # The reason_prefix is parametrized to 'aqr' so the desk's intents
        # read in its own voice (distinct from two-sigma).
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        for intent in out[dates[0]]:
            assert intent.reason.startswith('aqr ')

    def test_signal_notes_use_the_aqr_label(self):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        signal_notes = [n for n in desk.notes if n.category == 'signal']
        assert len(signal_notes) == 4  # 2 longs + 2 shorts
        for note in signal_notes:
            assert note.message.startswith('AQR ')
            assert note.data['symbol'] in {'S00', 'S01', 'S08', 'S09'}
            assert note.data['n_scored'] == 10

    def test_status_exposes_book_counts(self):
        frames = universe(10)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(), quantile=0.2,
                         target_gross=1.0)
        drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        status = desk.get_status()
        assert status['long_count'] == 2
        assert status['short_count'] == 2
        assert status['quantile'] == 0.2


# ======================================================================
# Graceful degrade (shared book) — never raises
# ======================================================================
class TestGracefulDegrade:
    def test_unfitted_controller_yields_no_opens(self):
        # Before the controller's FIRST fit, predict() is None -> the desk
        # holds flat with a model note. (Note: the FactorModel itself never
        # goes unfitted, but the CONTROLLER returns None until its first fit.)
        frames = universe(6)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(6), min_train_days=999,
                         quantile=0.2)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []
        model_notes = [n for n in desk.notes if n.category == 'model'
                       and 'unfitted' in n.message]
        assert len(model_notes) == 1
        assert desk.walk_forward_fits == []

    def test_too_few_scored_degrades(self):
        frames = universe(3)
        dates = frames['S00'].index
        desk = make_desk(default=monotone_scores(3), quantile=0.2)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        assert out[dates[0]] == []
        info = [n for n in desk.notes if n.category == 'info'
                and 'Degrade' in n.message]
        assert len(info) == 1
        assert info[0].data['n_scored'] == 3

    def test_empty_universe_does_not_crash(self):
        desk = make_desk(default={}, quantile=0.2)
        out = desk.generate_intents({}, pd.Timestamp('2023-01-02'),
                                    PortfolioManager(100000.0))
        assert out == []

    def test_no_one_step_flip_longs_and_shorts_disjoint(self):
        frames = universe(10)
        dates = frames['S00'].index
        flat = {f'S{i:02d}': 0.05 for i in range(10)}
        desk = make_desk(default=flat, quantile=0.2, target_gross=1.0)
        out = drive(desk, frames, dates[:1], PortfolioManager(100000.0))
        longs = {i.asset.symbol for i in out[dates[0]] if i.action == 'BUY'}
        shorts = {i.asset.symbol for i in out[dates[0]] if i.action == 'SHORT'}
        assert not (longs & shorts)


# ======================================================================
# AQR signature: factor-weight transparency notes (unit, with a real
# FactorModel controller that fits immediately on a stub schedule)
# ======================================================================
class TestFactorWeightTransparency:
    def _learnable_universe(self, n_symbols=12, n=160):
        mid = n_symbols / 2

        def frame(seed, drift):
            rng = np.random.default_rng(seed)
            rets = rng.normal(drift, 0.012, n)
            close = 100.0 * np.cumprod(1.0 + rets)
            return pd.DataFrame({
                'open': close, 'high': close * 1.002, 'low': close * 0.998,
                'close': close, 'volume': np.full(n, 500_000.0),
            }, index=pd.bdate_range('2022-01-03', periods=n))

        return {f'S{i:02d}': frame(100 + i, 0.0006 * (i - mid))
                for i in range(n_symbols)}

    def test_refit_emits_a_factor_weight_note(self):
        frames = self._learnable_universe()
        dates = frames['S00'].index
        # A real FactorModel controller fits once the sliced history reaches
        # min_train_days rows; drive a single LATE date (126 rows of history)
        # so the first refit fires and the AQR factor-weight note is emitted.
        desk = AqrDesk(
            controller=WalkForwardController(
                FactorModel(), min_train_days=120, refit_every_days=21),
            risk_manager=wide_risk(), quantile=0.2, target_gross=1.0)
        drive(desk, frames, dates[125:126], PortfolioManager(100000.0))

        fw_notes = [n for n in desk.notes if n.category == 'model'
                    and 'factor weights' in n.message]
        assert len(fw_notes) == 1
        note = fw_notes[0]
        # The note carries the structured attribution payload.
        assert set(note.data) >= {'factor_weights', 'degraded', 'model'}
        assert note.data['model'] == 'factor'
        assert list(note.data['factor_weights']) == list(FACTOR_COLUMNS)
        assert isinstance(note.data['degraded'], bool)
        # The shared refit note rides alongside it (same category).
        refit_notes = [n for n in desk.notes if n.category == 'model'
                       and 'model refit' in n.message]
        assert len(refit_notes) == 1

    def test_desk_factor_weights_accessor_matches_the_model(self):
        frames = self._learnable_universe()
        dates = frames['S00'].index
        model = FactorModel()
        desk = AqrDesk(
            controller=WalkForwardController(
                model, min_train_days=120, refit_every_days=21),
            risk_manager=wide_risk(), quantile=0.2, target_gross=1.0)
        # Drive a single late date so the controller fits (see above).
        drive(desk, frames, dates[125:126], PortfolioManager(100000.0))
        assert desk.factor_weights == model.factor_weights
        assert list(desk.factor_weights) == list(FACTOR_COLUMNS)


# ======================================================================
# End-to-end with the REAL engine + REAL FactorModel
# ======================================================================
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

        return {f'S{i:02d}': frame(100 + i, 0.0004 * (i - 6))
                for i in range(12)}

    def _build_desk(self, **kwargs):
        kwargs.setdefault('risk_manager', wide_risk())
        kwargs.setdefault('target_gross', 0.6)
        return AqrDesk(
            controller=WalkForwardController(
                FactorModel(), train_window_days=120, refit_every_days=21,
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
        assert report['desk'] == {'key': 'aqr', 'name': 'AQR Desk'}
        assert report['strategy'] == 'AQR Desk'

        walk_forward = report['walk_forward']
        assert walk_forward
        for fit in walk_forward:
            assert set(fit) == {'fit_date', 'train_start', 'train_end',
                                'n_samples', 'model'}
            assert fit['model'] == 'factor'
            assert fit['train_end'] <= fit['fit_date']

        notes = report['trader_notes']
        assert notes
        json.dumps(notes)  # JSON-safe end to end
        categories = {n['category'] for n in notes}
        assert {'signal', 'model', 'allocation'} <= categories
        for note in notes:
            assert note['desk'] == 'AQR Desk'

        # Phase regression: aqr is not a regime/pod/structures desk.
        assert 'regime_series' not in report
        assert 'pod_history' not in report
        assert 'orchestrator' not in report  # single-desk run

    def test_report_carries_factor_weight_transparency_notes(self,
                                                             report_and_desk):
        report, desk, _ = report_and_desk
        notes = report['trader_notes']
        fw_notes = [n for n in notes if n['category'] == 'model'
                    and 'factor weights' in n['message']]
        # One factor-weight note per refit.
        assert len(fw_notes) == len(report['walk_forward'])
        for note in fw_notes:
            assert set(note['data']) >= {'factor_weights', 'degraded'}
            assert list(note['data']['factor_weights']) == list(FACTOR_COLUMNS)
        # The live accessor exposes the same attribution surface.
        assert list(desk.factor_weights) == list(FACTOR_COLUMNS)

    def test_report_carries_research_integrity_machinery(self,
                                                         report_and_desk):
        report, _, _ = report_and_desk
        summary = report['summary']
        assert 'psr' in summary
        assert 'deflated_sharpe' in summary
        assert 'n_trials' in summary
        # n_trials deflates by the number of walk-forward fits in desk mode.
        assert summary['n_trials'] == max(1, len(report['walk_forward']))

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


# ======================================================================
# Held-across-days regression (the shared reconcile fix) — orphan sweep ~0
# ======================================================================
class TestHeldAcrossDaysRegression:
    """The AQR desk inherits the exact reconcile fix from the shared base: a
    stable held book is opened ONCE and held, and the orphan sweep stays
    silent in steady state. Driven through the REAL BacktestEngine with a
    STABLE (monotone) stub ranking so a top/bottom name keeps its seat."""

    def _run_stable(self, monkeypatch, n_days, n_symbols=10):
        frames = {f'S{i:02d}': quiet_frame(n=n_days, seed=400 + i)
                  for i in range(n_symbols)}

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = AqrDesk(
            controller=stub_controller(monotone_scores(n_symbols)),
            risk_manager=wide_risk(), quantile=0.2, target_gross=1.0)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(sorted(frames), '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)
        return desk, engine, report

    def test_stable_book_is_held_not_churned(self, monkeypatch):
        desk, engine, report = self._run_stable(monkeypatch, n_days=30)
        n_trading_days = len(report['portfolio_history'])
        assert n_trading_days >= 15

        s00 = [t['action'] for t in report['trades'] if t['symbol'] == 'S00']
        s09 = [t['action'] for t in report['trades'] if t['symbol'] == 'S09']
        assert s00 == ['BUY']            # held long the whole window
        assert s09 == ['SHORT']          # held short the whole window
        assert 'SELL' not in s00 and 'COVER' not in s09

        opens = [t for t in report['trades']
                 if t['action'] in ('BUY', 'SHORT')]
        closes = [t for t in report['trades']
                  if t['action'] in ('SELL', 'COVER')]
        assert len(opens) == 4  # 2 longs + 2 shorts, opened once
        assert len(closes) == 0
        assert len(report['trades']) < n_trading_days  # NOT one churn per day

    def test_orphan_sweep_is_silent_in_steady_state(self, monkeypatch):
        desk, engine, report = self._run_stable(monkeypatch, n_days=30)
        orphan_notes = sum('Orphan sweep' in n.message for n in desk.notes)
        assert orphan_notes == 0

    def test_rebalance_exits_carry_closes_not_sweeps(self, monkeypatch):
        # A name that LEAVES its side closes via a normal rebalance exit; the
        # orphan sweep stays ~0 (the shared closing-state guard). Driven with a
        # stub whose ranking moves S00 out of the long book midway.
        n_days = 30
        n_symbols = 10
        frames = {f'S{i:02d}': quiet_frame(n=n_days, seed=500 + i)
                  for i in range(n_symbols)}
        dates = frames['S00'].index
        base = monotone_scores(n_symbols)
        schedule = {d: {**base, 'S00': 0.0} for d in dates[3:]}

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = AqrDesk(
            controller=stub_controller(base, schedule),
            risk_manager=wide_risk(), quantile=0.2, target_gross=1.0)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0)
        report = engine.run(sorted(frames), '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)

        s00 = [t['action'] for t in report['trades'] if t['symbol'] == 'S00']
        assert s00 == ['BUY', 'SELL']  # opened, then exited, never re-opened
        orphan_notes = sum('Orphan sweep' in n.message for n in desk.notes)
        rebalance_exits = sum('no longer in the' in n.message
                              for n in desk.notes)
        assert orphan_notes == 0
        assert rebalance_exits >= 1


# ======================================================================
# Trades under the DEFAULT RiskManager (the max_name_size clamp)
# ======================================================================
class TestDefaultRiskManager:
    def test_trades_under_default_risk_manager_unit(self):
        # The default RiskManager has a 10% position cap; the max_name_size
        # clamp (0.10) keeps every leg inside it so the desk actually opens.
        frames = universe(10)
        dates = frames['S00'].index
        desk = AqrDesk(
            controller=stub_controller(monotone_scores()),
            risk_manager=RiskManager(),  # DEFAULT 10% cap
            quantile=0.2, target_gross=1.0)
        portfolio = PortfolioManager(100000.0)
        out = drive(desk, frames, dates[:1], portfolio)
        opens = [i for i in out[dates[0]] if i.action in ('BUY', 'SHORT')]
        assert len(opens) == 4
        for intent in opens:
            assert intent.size_fraction == pytest.approx(0.10)

    def test_trades_under_default_risk_manager_end_to_end(self, monkeypatch):
        # The full engine path: a real FactorModel desk under the DEFAULT
        # RiskManager must actually TRADE (the clamp keeps legs admissible).
        N = 320
        index = pd.bdate_range('2022-01-03', periods=N)

        def frame(seed, drift):
            rng = np.random.default_rng(seed)
            rets = rng.normal(drift, 0.012, N)
            close = 100.0 * np.cumprod(1.0 + rets)
            return pd.DataFrame({
                'open': close, 'high': close * 1.002, 'low': close * 0.998,
                'close': close,
                'volume': rng.integers(400_000, 900_000, N).astype(float),
            }, index=index)

        uni = {f'S{i:02d}': frame(100 + i, 0.0004 * (i - 6))
               for i in range(12)}

        def fake_fetch(self, symbol, start_date, end_date):
            return uni.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        desk = AqrDesk(
            controller=WalkForwardController(
                FactorModel(), train_window_days=120, refit_every_days=21,
                min_train_days=80),
            risk_manager=RiskManager(),  # DEFAULT cap, not wide
            target_gross=0.6)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=0.001)
        report = engine.run(sorted(uni), '2022-01-01', '2023-06-30',
                            benchmark_symbol=None)
        assert report['trades']  # the desk actually opened positions
        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}


# ======================================================================
# FundOrchestrator interop: aqr composes with another desk
# ======================================================================
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

    def test_aqr_plus_foundation_fund_composes(self, synthetic_universe,
                                               monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return synthetic_universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        aqr = AqrDesk(
            capital_allocation=0.5, risk_manager=wide_risk(),
            target_gross=0.6,
            controller=WalkForwardController(
                FactorModel(), train_window_days=120, refit_every_days=21,
                min_train_days=80))
        foundation = FoundationDesk(capital_allocation=0.5)
        orchestrator = FundOrchestrator([aqr, foundation])
        engine = BacktestEngine(orchestrator=orchestrator,
                                initial_capital=100000.0, commission=0.001)
        report = engine.run(sorted(synthetic_universe),
                            '2022-01-01', '2023-03-31', benchmark_symbol=None)

        assert 'orchestrator' in report
        roster = report['orchestrator']['desks']
        assert [d['key'] for d in roster] == ['aqr', 'foundation']
        assert report['orchestrator']['active_capital'] == pytest.approx(1.0)

        # Per-fold OOS is N/A for a netted multi-desk book (by contract).
        assert report['oos_folds']['available'] is False
        # The fund still carries the deflated/probabilistic Sharpe summary.
        assert 'deflated_sharpe' in report['summary']
        assert 'psr' in report['summary']

        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}

    def test_aqr_plus_twosigma_two_systematic_desks_compose(
            self, synthetic_universe, monkeypatch):
        # Two cross-sectional desks (transparent + ML) sharing the same book
        # base compose in one fund without interfering.
        from desks.twosigma import TwoSigmaDesk
        from desks.ml_model import GradientBoostingModel

        def fake_fetch(self, symbol, start_date, end_date):
            return synthetic_universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        aqr = AqrDesk(
            capital_allocation=0.5, risk_manager=wide_risk(),
            target_gross=0.6,
            controller=WalkForwardController(
                FactorModel(), train_window_days=120, refit_every_days=21,
                min_train_days=80))
        twosigma = TwoSigmaDesk(
            capital_allocation=0.5, risk_manager=wide_risk(),
            target_gross=0.6,
            controller=WalkForwardController(
                GradientBoostingModel(n_estimators=10),
                train_window_days=120, refit_every_days=21, min_train_days=80))
        orchestrator = FundOrchestrator([aqr, twosigma])
        engine = BacktestEngine(orchestrator=orchestrator,
                                initial_capital=100000.0, commission=0.001)
        report = engine.run(sorted(synthetic_universe),
                            '2022-01-01', '2023-03-31', benchmark_symbol=None)

        roster = report['orchestrator']['desks']
        assert [d['key'] for d in roster] == ['aqr', 'twosigma']
        assert {t['action'] for t in report['trades']} <= \
            {'BUY', 'SELL', 'SHORT', 'COVER'}


# ======================================================================
# Smoke: 'factor' is selectable for foundation/twosigma via the zoo (the
# AQR desk itself stays non-model-selectable, asserted above).
# ======================================================================
class TestFactorModelIsZooSelectableForOtherDesks:
    def test_twosigma_can_select_the_factor_model(self):
        from desks.twosigma import TwoSigmaDesk
        desk = create_desk('twosigma', model_key='factor')
        assert isinstance(desk, TwoSigmaDesk)
        assert desk.get_status()['models'] == ['factor']
        assert isinstance(desk._committee[0][1].model, FactorModel)

    def test_foundation_can_select_the_factor_model(self):
        desk = FoundationDesk(model_key='factor')
        assert isinstance(desk._controller.model, FactorModel)
