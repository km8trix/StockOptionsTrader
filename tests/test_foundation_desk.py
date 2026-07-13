"""Tests for desks.foundation.FoundationDesk.

Unit tests drive generate_intents with hand-built indicator frames (every
number controlled); the end-to-end tests run the real BacktestEngine in
desk mode on synthetic data with fetches monkeypatched, and hand-compute
one fill including capital_allocation * size_fraction * slippage *
commission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.foundation import FoundationDesk
from desks.walk_forward import WalkForwardController, WalkForwardModel
from desks.ml_model import GradientBoostingModel
from portfolio.manager import PortfolioManager
from portfolio.targets import PortfolioSnapshot

COMMISSION = 0.001
DEFAULT_SLIPPAGE_BPS = 5.0

#: Report keys that strategy mode has always produced (must survive).
LEGACY_REPORT_KEYS = {'strategy', 'summary', 'benchmark', 'drawdown_series',
                      'trades', 'closed_trades', 'portfolio_history',
                      'pending_signals'}


@pytest.fixture
def patch_market_data(monkeypatch):
    """Patch MarketDataHandler.fetch_stock_data to serve canned frames."""

    def _patch(frames_by_symbol):
        def fake_fetch(self, symbol, start_date, end_date):
            return frames_by_symbol.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

    return _patch


def permissive_desk(**kwargs) -> FoundationDesk:
    """FoundationDesk whose only entry condition is the MACD cross (the
    RSI band and volume confirmation are opened wide), so synthetic
    random-walk data produces deterministic trades."""
    kwargs.setdefault('volume_confirmation_mult', 0.0)
    kwargs.setdefault('rsi_entry_low', 0.0)
    kwargs.setdefault('rsi_entry_high', 100.0)
    return FoundationDesk(**kwargs)


def enriched_frame(n: int = 60, cross: str = 'up',
                   rsi: float = 55.0, volume: float = 600_000.0,
                   volume_sma: float = 400_000.0) -> pd.DataFrame:
    """Hand-built indicator-enriched frame with a MACD cross on the LAST
    bar ('up', 'down', or 'none'); every indicator value is controlled."""
    index = pd.bdate_range('2023-01-02', periods=n)
    close = np.full(n, 100.0)
    signal_line = np.zeros(n)
    if cross == 'up':
        macd = np.full(n, -1.0)
        macd[-1] = 0.84  # crosses above the signal line today
    elif cross == 'down':
        macd = np.full(n, 1.0)
        macd[-1] = -1.0
    else:
        macd = np.full(n, -1.0)
    return pd.DataFrame({
        'open': close, 'high': close * 1.01, 'low': close * 0.99,
        'close': close,
        'volume': np.full(n, volume),
        'macd': macd, 'signal': signal_line,
        'rsi': np.full(n, rsi),
        'volume_sma': np.full(n, volume_sma),
    }, index=index)


class StubModel(WalkForwardModel):
    """Walk-forward model with a fixed score for every symbol."""

    def __init__(self, score: float):
        self.score = score

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        return {symbol: self.score for symbol in data}


def stub_controller(score: float) -> WalkForwardController:
    """Controller that fits immediately and scores every symbol `score`."""
    return WalkForwardController(StubModel(score), min_train_days=1)


class TestMomentumSignals:
    def test_macd_cross_up_emits_sized_buy_with_signal_note(self):
        desk = FoundationDesk()  # default controller: unfitted at 60 bars
        frame = enriched_frame(cross='up')
        date = frame.index[-1]
        desk.set_clock(date)
        portfolio = PortfolioManager(initial_capital=100000.0)

        intents = desk.generate_intents({'SYM': frame}, date, portfolio)

        assert len(intents) == 1
        intent = intents[0]
        assert intent.action == 'BUY'
        assert intent.asset == Asset(symbol='SYM', asset_type=AssetType.STOCK)
        # Equal weight: min(0.10, 1/1) with a single-symbol universe.
        assert intent.size_fraction == pytest.approx(0.10)
        assert 'not yet fitted' in intent.reason  # ungated before first fit

        signal_notes = [n for n in desk.notes if n.category == 'signal']
        assert len(signal_notes) == 1
        note = signal_notes[0]
        assert 'SYM' in note.message and 'not yet fitted' in note.message
        assert note.data['macd'] == pytest.approx(0.84)
        assert note.data['rsi'] == pytest.approx(55.0)
        assert note.data['volume_ratio'] == pytest.approx(1.5)
        assert note.timestamp == date

    def test_size_fraction_is_equal_weight_capped_at_ten_percent(self):
        desk = FoundationDesk()
        date = enriched_frame().index[-1]
        portfolio = PortfolioManager(initial_capital=100000.0)
        # 20 symbols -> 1/20 = 0.05 (the cap does not bind).
        frames = {f'S{i:02d}': enriched_frame(cross='none') for i in range(19)}
        frames['HIT'] = enriched_frame(cross='up')
        intents = desk.generate_intents(frames, date, portfolio)
        assert len(intents) == 1
        assert intents[0].size_fraction == pytest.approx(1.0 / 20)

    def test_no_cross_no_intents(self):
        desk = FoundationDesk()
        frame = enriched_frame(cross='none')
        portfolio = PortfolioManager(initial_capital=100000.0)
        assert desk.generate_intents({'SYM': frame}, frame.index[-1],
                                     portfolio) == []

    def test_rsi_band_and_volume_confirmation_filter_entries(self):
        portfolio = PortfolioManager(initial_capital=100000.0)
        # RSI outside the default [40, 70] band blocks the entry.
        frame = enriched_frame(cross='up', rsi=75.0)
        desk = FoundationDesk()
        assert desk.generate_intents({'SYM': frame}, frame.index[-1],
                                     portfolio) == []
        # Volume below 1.2x average blocks the entry.
        frame = enriched_frame(cross='up', volume=400_000.0)
        desk = FoundationDesk()
        assert desk.generate_intents({'SYM': frame}, frame.index[-1],
                                     portfolio) == []

    def test_short_history_is_skipped(self):
        desk = FoundationDesk()
        frame = enriched_frame(n=30, cross='up')  # < MIN_HISTORY_DAYS
        portfolio = PortfolioManager(initial_capital=100000.0)
        assert desk.generate_intents({'SYM': frame}, frame.index[-1],
                                     portfolio) == []

    def test_macd_cross_down_exits_open_position(self):
        desk = FoundationDesk()
        frame = enriched_frame(cross='down')
        date = frame.index[-1]
        desk.set_clock(date)
        asset = Asset(symbol='SYM', asset_type=AssetType.STOCK)
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=asset, quantity=10, avg_entry_price=100.0,
            current_price=100.0, timestamp=frame.index[0]))

        intents = desk.generate_intents({'SYM': frame}, date, portfolio)

        assert len(intents) == 1
        assert intents[0].action == 'SELL'
        assert intents[0].size_fraction == 1.0
        exit_notes = [n for n in desk.notes if n.category == 'signal']
        assert len(exit_notes) == 1
        assert 'exit' in exit_notes[0].message.lower()


class TestWalkForwardGate:
    def test_negative_score_blocks_entry_with_model_note(self):
        desk = permissive_desk(controller=stub_controller(-0.2))
        frame = enriched_frame(cross='up')
        date = frame.index[-1]
        desk.set_clock(date)
        portfolio = PortfolioManager(initial_capital=100000.0)

        intents = desk.generate_intents({'SYM': frame}, date, portfolio)

        assert intents == []
        blocked = [n for n in desk.notes
                   if n.category == 'model' and 'Blocked' in n.message]
        assert len(blocked) == 1
        assert blocked[0].data['wf_score'] == pytest.approx(-0.2)

    def test_positive_score_gates_entry_through(self):
        desk = permissive_desk(controller=stub_controller(0.07))
        frame = enriched_frame(cross='up')
        date = frame.index[-1]
        desk.set_clock(date)
        portfolio = PortfolioManager(initial_capital=100000.0)

        intents = desk.generate_intents({'SYM': frame}, date, portfolio)

        assert len(intents) == 1
        assert intents[0].action == 'BUY'
        assert 'walk-forward gated' in intents[0].reason
        assert '+0.0700' in intents[0].reason

    def test_refit_is_noted_with_fit_numbers(self):
        desk = permissive_desk(controller=stub_controller(0.1))
        frame = enriched_frame(cross='none')
        date = frame.index[-1]
        desk.set_clock(date)
        portfolio = PortfolioManager(initial_capital=100000.0)

        desk.generate_intents({'SYM': frame}, date, portfolio)

        refit_notes = [n for n in desk.notes
                       if n.category == 'model' and 'refit' in n.message]
        assert len(refit_notes) == 1
        assert refit_notes[0].data['n_samples'] == len(frame)
        assert desk.walk_forward_fits == desk._controller.fits
        assert len(desk.walk_forward_fits) == 1


class TestTargetNativeSignals:
    @staticmethod
    def snapshot(*, filled=None, reserved=None, version=1):
        return PortfolioSnapshot(
            filled_quantities=filled or {},
            reserved_deltas=reserved or {},
            version=version,
            as_of=datetime(2026, 7, 12, tzinfo=timezone.utc),
        )

    def test_repeated_entry_emits_the_same_exact_target_without_pyramiding(self):
        desk = FoundationDesk(target_mode=True)
        frame = enriched_frame(cross='up')
        portfolio = PortfolioManager(initial_capital=100000.0)
        state = self.snapshot()

        first = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio, state)
        second = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio, state)

        assert desk.target_native_enabled is True
        assert first == second
        assert first[0].target_quantity == 100
        assert first[0].owner == 'foundation'
        assert first[0].strategy == 'foundation'
        assert first[0].metadata == {
            'deployment_id': 'foundation-v1',
            'size_fraction': pytest.approx(0.10),
        }

    def test_pending_partial_fill_echoes_effective_target(self):
        desk = FoundationDesk(target_mode=True)
        frame = enriched_frame(cross='up')
        asset = Asset('SYM', AssetType.STOCK)
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=asset, quantity=40, avg_entry_price=100.0,
            current_price=100.0, timestamp=frame.index[0]))
        state = self.snapshot(filled={asset: 40}, reserved={asset: 60})

        targets = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio, state)

        assert len(targets) == 1
        assert targets[0].target_quantity == 100
        assert 'maintain' in targets[0].reason

    @pytest.mark.parametrize('cross, rsi', [('down', 55.0), ('none', 75.0)])
    def test_macd_or_rsi_exit_emits_explicit_zero(self, cross, rsi):
        desk = FoundationDesk(target_mode=True)
        frame = enriched_frame(cross=cross, rsi=rsi)
        asset = Asset('SYM', AssetType.STOCK)
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=asset, quantity=100, avg_entry_price=100.0,
            current_price=100.0, timestamp=frame.index[0]))

        targets = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio,
            self.snapshot(filled={asset: 100}))

        assert len(targets) == 1
        assert targets[0].target_quantity == 0
        assert targets[0].metadata['size_fraction'] == 1.0

    def test_no_signal_echoes_held_target(self):
        desk = FoundationDesk(target_mode=True)
        frame = enriched_frame(cross='none')
        asset = Asset('SYM', AssetType.STOCK)
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=asset, quantity=73, avg_entry_price=100.0,
            current_price=100.0, timestamp=frame.index[0]))

        targets = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio,
            self.snapshot(filled={asset: 73}))

        assert targets[0].target_quantity == 73

    def test_negative_model_gate_keeps_a_flat_symbol_flat(self):
        desk = permissive_desk(
            controller=stub_controller(-0.2), target_mode=True)
        frame = enriched_frame(cross='up')
        portfolio = PortfolioManager(initial_capital=100000.0)

        targets = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio, self.snapshot())

        assert targets[0].target_quantity == 0
        assert 'blocked' in targets[0].reason

    def test_foreign_owned_position_remains_outside_foundation_targets(self):
        desk = FoundationDesk(target_mode=True)
        frame = enriched_frame(cross='down')
        asset = Asset('SYM', AssetType.STOCK)
        portfolio = PortfolioManager(initial_capital=100000.0)
        portfolio.add_position(Position(
            asset=asset, quantity=100, avg_entry_price=100.0,
            current_price=100.0, timestamp=frame.index[0],
            owners=('renaissance',)))

        targets = desk.generate_targets(
            {'SYM': frame}, frame.index[-1], portfolio,
            self.snapshot(filled={asset: 100}))

        assert targets == []

    def test_target_mode_does_not_change_legacy_intent_contract(self):
        frame = enriched_frame(cross='up')
        portfolio = PortfolioManager(initial_capital=100000.0)
        legacy_desk = FoundationDesk()
        legacy = legacy_desk.generate_intents(
            {'SYM': frame}, frame.index[-1], portfolio)
        opted_in = FoundationDesk(target_mode=True).generate_intents(
            {'SYM': frame}, frame.index[-1], portfolio)

        assert opted_in == legacy
        assert legacy_desk.target_native_enabled is False
        assert opted_in[0].action == 'BUY'
        assert opted_in[0].size_fraction == pytest.approx(0.10)


class TestDeskModeEndToEnd:
    def test_engine_run_produces_trades_report_and_exact_fill_math(
            self, make_ohlcv, patch_market_data):
        df = make_ohlcv(n_days=120, seed=11)
        patch_market_data({'TEST': df})

        capital_allocation = 0.5
        desk = permissive_desk(capital_allocation=capital_allocation)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)

        # --- trades occurred, BUY first (no position to sell before) ----
        assert len(engine.trades_log) >= 2
        first = engine.trades_log[0]
        assert first['action'] == 'BUY'

        # --- hand-computed fill: next-day open, slippage, desk sizing ---
        signal_pos = df.index.get_loc(first['signal_date'])
        fill_date = df.index[signal_pos + 1]
        assert first['date'] == fill_date

        expected_fill = float(df.loc[fill_date, 'open']) * (
            1 + DEFAULT_SLIPPAGE_BPS / 10000.0)
        assert first['price'] == pytest.approx(expected_fill)

        # First fill happens on an all-cash portfolio: trade value =
        # 100000 * capital_allocation(0.5) * size_fraction(0.10) = 5000.
        size_fraction = min(0.10, 1.0 / 1)
        expected_qty = int(
            100000.0 * capital_allocation * size_fraction / expected_fill)
        assert first['quantity'] == expected_qty
        assert first['cost'] == pytest.approx(
            expected_qty * expected_fill * (1 + COMMISSION))

        # --- report: every legacy key plus the C3 desk additions --------
        assert LEGACY_REPORT_KEYS <= set(report)
        assert report['desk'] == {'key': 'foundation',
                                  'name': 'Foundation Desk'}
        assert report['strategy'] == 'Foundation Desk'
        assert isinstance(report['walk_forward'], list)

        notes = report['trader_notes']
        assert notes  # non-empty
        json.dumps(notes)  # JSON-safe end to end
        categories = {n['category'] for n in notes}
        assert 'signal' in categories
        # Seed 11 trips a stop-loss during this run.
        assert 'risk' in categories
        for note in notes:
            assert set(note) == {'timestamp', 'desk', 'category', 'message',
                                 'data'}
            assert note['desk'] == 'Foundation Desk'
            # Simulation dates, not the wall clock.
            assert str(df.index[0].date()) <= note['timestamp'][:10] \
                <= str(df.index[-1].date())

    def test_sell_intents_close_the_full_position(self, make_ohlcv,
                                                  patch_market_data):
        df = make_ohlcv(n_days=120, seed=11)
        patch_market_data({'TEST': df})

        desk = permissive_desk()
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=COMMISSION)
        engine.run(['TEST'], '2023-01-01', '2023-12-31',
                   benchmark_symbol=None)

        position = 0
        for trade in engine.trades_log:
            if trade['action'] == 'BUY':
                position += trade['quantity']
            else:
                assert trade['quantity'] == position  # full close
                position = 0

    def test_walk_forward_fits_appear_in_long_runs(self, make_ohlcv,
                                                   patch_market_data):
        df = make_ohlcv(n_days=160, seed=7)
        patch_market_data({'TEST': df})

        controller = WalkForwardController(
            GradientBoostingModel(n_estimators=10))
        desk = permissive_desk(controller=controller)
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=COMMISSION)
        report = engine.run(['TEST'], '2023-01-01', '2023-12-31',
                            benchmark_symbol=None)

        # min_train_days 120, refit every 21: fits on days 120 and 141.
        walk_forward = report['walk_forward']
        assert len(walk_forward) == 2
        assert walk_forward[0]['fit_date'] == str(df.index[119].date())
        assert walk_forward[1]['fit_date'] == str(df.index[140].date())
        for fit in walk_forward:
            assert sorted(fit) == ['fit_date', 'n_samples', 'train_end',
                                   'train_start']
            assert fit['train_end'] <= fit['fit_date']
            assert isinstance(fit['n_samples'], int)
        json.dumps(walk_forward)

    def test_desk_status_reflects_run_activity(self, make_ohlcv,
                                               patch_market_data):
        df = make_ohlcv(n_days=120, seed=7)
        patch_market_data({'TEST': df})

        desk = permissive_desk()
        engine = BacktestEngine(desk=desk, initial_capital=100000.0,
                                commission=COMMISSION)
        engine.run(['TEST'], '2023-01-01', '2023-12-31',
                   benchmark_symbol=None)

        status = desk.get_status()
        assert status['key'] == 'foundation'
        assert status['notes_count'] == len(desk.notes) > 0
        assert status['last_note'] == desk.notes[-1].to_dict()
