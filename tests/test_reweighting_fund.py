"""Tests for the dynamic reweighting seam and coordinator.

Two layers:
  * BacktestEngine fund-mode reweighter seam — the reweighter= guard, the
    once-per-day hook actually shifting desk capital_allocation mid-run, the
    additive 'reweight_log' report key, and non-reweighted-fund additivity.
  * ReweightingFundBacktest coordinator — injected providers (solo curves +
    orchestrator) drive it deterministically, and the DEFAULT path (real
    single-desk solo backtests + registry orchestrator) wired with scripted
    desks via monkeypatched create_desk.

Offline and deterministic — no network. Reuses the orchestrator test helpers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import pytest

import backtesting.reweighting_fund as rf_module
import desks.registry as registry
from backtesting.backtest_engine import BacktestEngine
from backtesting.reweighting_fund import ReweightingFundBacktest
from data.market_data import MarketDataHandler
from desks.capital_allocator import CrossDeskCapitalAllocator
from desks.dynamic_reweighter import DynamicReweighter
from desks.orchestrator import FundOrchestrator
from tests.test_fund_orchestrator import (ScriptedDesk, flat_frame, intent,
                                          stock, wide_risk)

START, END = '2022-01-01', '2022-12-31'
N_DAYS = 60


def _dates() -> pd.DatetimeIndex:
    return flat_frame(n=N_DAYS).index


def _curve(values: List[float]) -> List[Tuple[pd.Timestamp, float]]:
    return list(zip(_dates(), [float(v) for v in values]))


# Aligned low-/high-vol curves over the run window (distinct, finite vol).
LOW = _curve([100.0 + (0.5 if i % 2 else 0.0) for i in range(N_DAYS)])
HIGH = _curve([100.0 + (5.0 if i % 2 else 0.0) for i in range(N_DAYS)])


def _patch_fetch(monkeypatch, universe: Dict[str, pd.DataFrame]) -> None:
    def fake_fetch(self, symbol, start, end):
        return universe.get(symbol, pd.DataFrame())
    monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)


# ======================================================================
# Engine seam
# ======================================================================
class TestEngineReweighterGuard:
    def test_reweighter_requires_orchestrator_in_desk_mode(self):
        desk = ScriptedDesk('solo', 1.0)
        with pytest.raises(ValueError, match='requires orchestrator'):
            BacktestEngine(desk=desk, reweighter=DynamicReweighter())

    def test_reweighter_rejected_in_strategy_mode(self):
        from strategies.base import Strategy

        class _NoopStrategy(Strategy):
            def generate_signals(self, data, asset):
                return 'HOLD'

        with pytest.raises(ValueError, match='requires orchestrator'):
            BacktestEngine(strategy=_NoopStrategy('noop'),
                           reweighter=DynamicReweighter())


class TestEngineReweightSeam:
    def test_reweight_shifts_capital_mid_run_and_logs(self, monkeypatch):
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})
        a = ScriptedDesk('asim', 0.5, risk_manager=wide_risk())
        b = ScriptedDesk('bsim', 0.5, risk_manager=wide_risk())
        orch = FundOrchestrator([a, b], risk_manager=wide_risk())

        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'asim': LOW, 'bsim': HIGH})  # asim is the calm desk
        engine = BacktestEngine(orchestrator=orch, reweighter=rw,
                                initial_capital=100_000.0)
        report = engine.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)

        assert 'error' not in report
        # Rebalanced at day 21 and 42 over a 60-day run.
        assert [e['day_number'] for e in report['reweight_log']] == [21, 42]
        # Genuine (non-degenerate) curves -> real risk parity, not a fallback.
        assert all(e['fallback'] is False for e in report['reweight_log'])
        assert all(e['degraded_desks'] == [] for e in report['reweight_log'])
        # Capital actually moved off the 0.5/0.5 construction weights, toward
        # the lower-vol desk.
        assert a.capital_allocation != pytest.approx(0.5)
        assert a.capital_allocation > b.capital_allocation
        # active_capital was recomputed to the live sum (still <= 1.0).
        assert report['orchestrator']['active_capital'] == pytest.approx(
            a.capital_allocation + b.capital_allocation)
        assert report['orchestrator']['active_capital'] <= 1.0 + 1e-9

    def test_three_desks_one_missing_curve_falls_back_whole_fund(
            self, monkeypatch):
        # One degenerate desk forces the WHOLE fund to equal weight (1/3 each),
        # and the log names the offending desk.
        _patch_fetch(monkeypatch, {'AAA': flat_frame(n=N_DAYS)})
        a = ScriptedDesk('asim', 1 / 3, risk_manager=wide_risk())
        b = ScriptedDesk('bsim', 1 / 3, risk_manager=wide_risk())
        c = ScriptedDesk('csim', 1 / 3, risk_manager=wide_risk())
        orch = FundOrchestrator([a, b, c], risk_manager=wide_risk())
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'asim': LOW, 'bsim': HIGH})  # csim has NO curve
        engine = BacktestEngine(orchestrator=orch, reweighter=rw,
                                initial_capital=100_000.0)
        report = engine.run(['AAA'], START, END, benchmark_symbol=None)

        entry = report['reweight_log'][-1]
        assert entry['fallback'] is True
        assert entry['degraded_desks'] == ['csim']
        for key in ('asim', 'bsim', 'csim'):
            assert entry['weights'][key] == pytest.approx(1 / 3)
        assert (a.capital_allocation == pytest.approx(1 / 3)
                and c.capital_allocation == pytest.approx(1 / 3))

    def test_non_reweighted_fund_has_no_reweight_log(self, monkeypatch):
        _patch_fetch(monkeypatch, {'AAA': flat_frame(n=N_DAYS)})
        orch = FundOrchestrator([ScriptedDesk('solo', 1.0)],
                                risk_manager=wide_risk())
        engine = BacktestEngine(orchestrator=orch, initial_capital=100_000.0)
        report = engine.run(['AAA'], START, END, benchmark_symbol=None)
        assert 'reweight_log' not in report

    def test_too_short_run_never_rebalances(self, monkeypatch):
        # Fewer trading days than rebalance_every -> no boundary, empty log.
        _patch_fetch(monkeypatch, {'AAA': flat_frame(n=10)})
        a = ScriptedDesk('asim', 0.5, risk_manager=wide_risk())
        b = ScriptedDesk('bsim', 0.5, risk_manager=wide_risk())
        orch = FundOrchestrator([a, b], risk_manager=wide_risk())
        rw = DynamicReweighter(rebalance_every=21)
        rw.set_curves({'asim': LOW, 'bsim': HIGH})
        engine = BacktestEngine(orchestrator=orch, reweighter=rw,
                                initial_capital=100_000.0)
        report = engine.run(['AAA'], START, END, benchmark_symbol=None)
        assert report['reweight_log'] == []
        assert a.capital_allocation == 0.5 and b.capital_allocation == 0.5


# ======================================================================
# Coordinator
# ======================================================================
class TestCoordinatorConstruction:
    def test_empty_allocations_rejected(self):
        with pytest.raises(ValueError, match='at least one desk'):
            ReweightingFundBacktest({})


class TestCoordinatorInjected:
    def test_injected_providers_drive_solo_and_fund(self, monkeypatch):
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})

        solo_calls: List[str] = []

        def fake_solo(key, symbols, start, end):
            solo_calls.append(key)
            assert list(symbols) == ['AAA', 'BBB']
            return LOW if key == 'asim' else HIGH

        def factory(allocations):
            desks = [ScriptedDesk(k, v, risk_manager=wide_risk())
                     for k, v in allocations.items()]
            return FundOrchestrator(desks, risk_manager=wide_risk())

        rfb = ReweightingFundBacktest(
            {'asim': 0.5, 'bsim': 0.5}, rebalance_every=21,
            solo_curve_provider=fake_solo, orchestrator_factory=factory)
        report = rfb.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)

        assert sorted(solo_calls) == ['asim', 'bsim']  # one solo pass per desk
        assert 'error' not in report
        assert report['reweight_log']
        last = report['reweight_log'][-1]['weights']
        assert last['asim'] > last['bsim']  # capital flowed to the calm desk

    def test_warmup_delays_first_rebalance_end_to_end(self, monkeypatch):
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})

        def fake_solo(key, symbols, start, end):
            return LOW if key == 'asim' else HIGH

        def factory(allocations):
            desks = [ScriptedDesk(k, v, risk_manager=wide_risk())
                     for k, v in allocations.items()]
            return FundOrchestrator(desks, risk_manager=wide_risk())

        rfb = ReweightingFundBacktest(
            {'asim': 0.5, 'bsim': 0.5}, rebalance_every=15, warmup=20,
            solo_curve_provider=fake_solo, orchestrator_factory=factory)
        report = rfb.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)
        # First boundary is warmup + rebalance_every = 35, then 50 over 60 days.
        assert [e['day_number'] for e in report['reweight_log']] == [35, 50]

    def test_seed_threaded_into_engines(self, monkeypatch):
        # seed reaches the BacktestEngine(s) the coordinator builds (here only
        # the fund engine, since the injected solo provider bypasses the
        # default solo-curve engines).
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})
        captured = []
        real_engine = rf_module.BacktestEngine

        def spy(*args, **kwargs):
            captured.append(kwargs.get('seed'))
            return real_engine(*args, **kwargs)
        monkeypatch.setattr(rf_module, 'BacktestEngine', spy)

        def factory(allocations):
            desks = [ScriptedDesk(k, v, risk_manager=wide_risk())
                     for k, v in allocations.items()]
            return FundOrchestrator(desks, risk_manager=wide_risk())

        rfb = ReweightingFundBacktest(
            {'asim': 0.5, 'bsim': 0.5}, rebalance_every=21, seed=7,
            solo_curve_provider=lambda *a: LOW, orchestrator_factory=factory)
        rfb.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)
        assert captured and all(s == 7 for s in captured)

    def test_target_gross_passed_through(self, monkeypatch):
        _patch_fetch(monkeypatch, {'AAA': flat_frame(n=N_DAYS)})

        def fake_solo(key, symbols, start, end):
            return LOW if key == 'asim' else HIGH

        def factory(allocations):
            desks = [ScriptedDesk(k, v, risk_manager=wide_risk())
                     for k, v in allocations.items()]
            return FundOrchestrator(desks, risk_manager=wide_risk())

        rfb = ReweightingFundBacktest(
            {'asim': 0.3, 'bsim': 0.3}, rebalance_every=21, target_gross=0.6,
            solo_curve_provider=fake_solo, orchestrator_factory=factory)
        report = rfb.run(['AAA'], START, END, benchmark_symbol=None)
        last = report['reweight_log'][-1]['weights']
        assert sum(last.values()) == pytest.approx(0.6)


class TestCoordinatorDefaultPath:
    def test_default_path_runs_solo_then_fund(self, monkeypatch):
        # Drive the REAL default providers (single-desk solo backtests +
        # registry orchestrator) with scripted desks via monkeypatched
        # create_desk in BOTH the registry and the coordinator module.
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})

        def fake_create_desk(key, capital_allocation=1.0):
            return ScriptedDesk(key, capital_allocation,
                                risk_manager=wide_risk())

        monkeypatch.setattr(registry, 'create_desk', fake_create_desk)
        monkeypatch.setattr(rf_module, 'create_desk', fake_create_desk)

        rfb = ReweightingFundBacktest({'asim': 0.5, 'bsim': 0.5},
                                      rebalance_every=21)
        report = rfb.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)

        assert 'error' not in report
        # Solo curves are flat (no scripted trades) -> degenerate vol ->
        # equal-weight fallback, but the rebalance still fires, logs, AND
        # records the fallback so it isn't mistaken for a real 0.5/0.5.
        assert [e['day_number'] for e in report['reweight_log']] == [21, 42]
        for entry in report['reweight_log']:
            assert entry['weights']['asim'] == pytest.approx(0.5)
            assert entry['weights']['bsim'] == pytest.approx(0.5)
            assert entry['fallback'] is True
            assert set(entry['degraded_desks']) == {'asim', 'bsim'}
