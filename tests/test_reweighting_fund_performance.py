"""Tests for the OPT-IN performance weighting through the fund seam.

Two layers, mirroring tests/test_reweighting_fund.py:
  * BacktestEngine fund-mode seam driven by a performance-mode reweighter —
    capital actually shifts toward the realized higher-Sharpe desk mid-run.
  * ReweightingFundBacktest coordinator — the additive ``weighting='performance'``
    flag flows into the internal DynamicReweighter, and the DEFAULT
    ('risk_parity') is byte-identical to the existing path.

NOTE on the log: the engine's report['reweight_log'] is a fixed projection
(date/day_number/weights/fallback/degraded_desks) that intentionally does NOT
carry the additive 'mode' key — only the reweighter's own rebalance_log does.
So 'mode is recorded' is asserted on a DynamicReweighter the test owns; the
fund tests assert the realized REALLOCATION via report['reweight_log'] weights.

Offline and deterministic — no network. Reuses the orchestrator test helpers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from backtesting.reweighting_fund import ReweightingFundBacktest
from data.market_data import MarketDataHandler
from desks.dynamic_reweighter import DynamicReweighter
from desks.orchestrator import FundOrchestrator
from tests.test_fund_orchestrator import ScriptedDesk, flat_frame, wide_risk

START, END = '2022-01-01', '2022-12-31'
N_DAYS = 60


def _dates() -> pd.DatetimeIndex:
    return flat_frame(n=N_DAYS).index


def _curve(values: List[float]) -> List[Tuple[pd.Timestamp, float]]:
    return list(zip(_dates(), [float(v) for v in values]))


# High-Sharpe (steady compounding drift + tiny jitter for finite-but-low vol)
# vs near-zero-Sharpe (choppy ~0 mean, larger vol) curves over the run window.
# The high-Sharpe desk's realized Sharpe dominates at every rebalance slice.
def _high_sharpe_vals() -> List[float]:
    out: List[float] = []
    level = 100.0
    for i in range(N_DAYS):
        level *= (1.0 + 0.003 + (0.0003 if i % 2 else -0.0003))
        out.append(level)
    return out


HIGH_SHARPE = _curve(_high_sharpe_vals())
LOW_SHARPE = _curve([100.0 + (3.0 if i % 2 else 0.0) for i in range(N_DAYS)])
# Aligned low-/high-vol curves for the risk-parity default comparison.
LOW_VOL = _curve([100.0 + (0.5 if i % 2 else 0.0) for i in range(N_DAYS)])
HIGH_VOL = _curve([100.0 + (5.0 if i % 2 else 0.0) for i in range(N_DAYS)])


def _patch_fetch(monkeypatch, universe: Dict[str, pd.DataFrame]) -> None:
    def fake_fetch(self, symbol, start, end):
        return universe.get(symbol, pd.DataFrame())
    monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)


def _factory(allocations):
    desks = [ScriptedDesk(k, v, risk_manager=wide_risk())
             for k, v in allocations.items()]
    return FundOrchestrator(desks, risk_manager=wide_risk())


# ======================================================================
# Engine seam (performance-mode reweighter)
# ======================================================================
class TestEnginePerformanceSeam:
    def test_capital_shifts_to_higher_sharpe_desk_mid_run(self, monkeypatch):
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})
        a = ScriptedDesk('asim', 0.5, risk_manager=wide_risk())
        b = ScriptedDesk('bsim', 0.5, risk_manager=wide_risk())
        orch = FundOrchestrator([a, b], risk_manager=wide_risk())

        rw = DynamicReweighter(rebalance_every=21, weighting='performance')
        rw.set_curves({'asim': HIGH_SHARPE, 'bsim': LOW_SHARPE})
        engine = BacktestEngine(orchestrator=orch, reweighter=rw,
                                initial_capital=100_000.0)
        report = engine.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)

        assert 'error' not in report
        assert [e['day_number'] for e in report['reweight_log']] == [21, 42]
        # Genuine (non-degenerate) curves -> real performance tilt, no fallback.
        assert all(e['fallback'] is False for e in report['reweight_log'])
        # Capital moved off the 0.5/0.5 construction weights toward the
        # higher-Sharpe desk, bounded by the default cap.
        assert a.capital_allocation != pytest.approx(0.5)
        assert a.capital_allocation > b.capital_allocation
        assert a.capital_allocation <= 0.60 + 1e-9
        # No over-leverage: live active_capital sums to <= 1.0.
        assert report['orchestrator']['active_capital'] == pytest.approx(
            a.capital_allocation + b.capital_allocation)
        assert report['orchestrator']['active_capital'] <= 1.0 + 1e-9
        # The reweighter's own log records the performance mode every boundary.
        assert all(e['mode'] == 'performance' for e in rw.rebalance_log)

    def test_performance_differs_from_risk_parity_on_same_curves(
            self, monkeypatch):
        # Same curves, two modes -> different realized weights (performance
        # tilts by Sharpe, risk parity by inverse vol).
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})

        def run_mode(weighting):
            a = ScriptedDesk('asim', 0.5, risk_manager=wide_risk())
            b = ScriptedDesk('bsim', 0.5, risk_manager=wide_risk())
            orch = FundOrchestrator([a, b], risk_manager=wide_risk())
            rw = DynamicReweighter(rebalance_every=21, weighting=weighting)
            rw.set_curves({'asim': HIGH_SHARPE, 'bsim': LOW_SHARPE})
            engine = BacktestEngine(orchestrator=orch, reweighter=rw,
                                    initial_capital=100_000.0)
            report = engine.run(['AAA', 'BBB'], START, END,
                                benchmark_symbol=None)
            return report['reweight_log'][-1]['weights']

        perf = run_mode('performance')
        rp = run_mode('risk_parity')
        assert perf != rp


# ======================================================================
# Coordinator (ReweightingFundBacktest)
# ======================================================================
class TestCoordinatorPerformanceMode:
    def test_default_weighting_is_risk_parity(self):
        rfb = ReweightingFundBacktest({'asim': 0.5, 'bsim': 0.5})
        assert rfb.weighting == 'risk_parity'

    def test_performance_flag_reallocates_by_realized_sharpe(
            self, monkeypatch):
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})

        def fake_solo(key, symbols, start, end):
            return HIGH_SHARPE if key == 'asim' else LOW_SHARPE

        rfb = ReweightingFundBacktest(
            {'asim': 0.5, 'bsim': 0.5}, rebalance_every=21,
            weighting='performance',
            solo_curve_provider=fake_solo, orchestrator_factory=_factory)
        report = rfb.run(['AAA', 'BBB'], START, END, benchmark_symbol=None)

        assert 'error' not in report
        assert [e['day_number'] for e in report['reweight_log']] == [21, 42]
        for entry in report['reweight_log']:
            # Capital flowed to the realized higher-Sharpe desk, bounded + summing
            # to target_gross (no over-leverage).
            assert entry['weights']['asim'] > entry['weights']['bsim']
            assert entry['weights']['asim'] <= 0.60 + 1e-9
            assert entry['weights']['bsim'] >= 0.05 - 1e-9
            assert sum(entry['weights'].values()) == pytest.approx(1.0)

    def test_performance_target_gross_passed_through(self, monkeypatch):
        _patch_fetch(monkeypatch, {'AAA': flat_frame(n=N_DAYS)})

        def fake_solo(key, symbols, start, end):
            return HIGH_SHARPE if key == 'asim' else LOW_SHARPE

        rfb = ReweightingFundBacktest(
            {'asim': 0.3, 'bsim': 0.3}, rebalance_every=21, target_gross=0.6,
            weighting='performance',
            solo_curve_provider=fake_solo, orchestrator_factory=_factory)
        report = rfb.run(['AAA'], START, END, benchmark_symbol=None)
        last = report['reweight_log'][-1]['weights']
        assert sum(last.values()) == pytest.approx(0.6)
        assert sum(last.values()) <= 0.6 + 1e-9

    def test_default_path_byte_identical_to_risk_parity(self, monkeypatch):
        # Headline: a DEFAULT-weighting coordinator produces the SAME
        # reweight_log (weights + every asserted field) as an explicitly
        # risk_parity coordinator on identical curves.
        _patch_fetch(monkeypatch,
                     {'AAA': flat_frame(n=N_DAYS), 'BBB': flat_frame(n=N_DAYS)})

        def fake_solo(key, symbols, start, end):
            return LOW_VOL if key == 'asim' else HIGH_VOL

        def run(weighting=None):
            kwargs = dict(rebalance_every=21,
                          solo_curve_provider=fake_solo,
                          orchestrator_factory=_factory)
            if weighting is not None:
                kwargs['weighting'] = weighting
            rfb = ReweightingFundBacktest({'asim': 0.5, 'bsim': 0.5}, **kwargs)
            return rfb.run(['AAA', 'BBB'], START, END,
                           benchmark_symbol=None)['reweight_log']

        default_log = run()                      # omit weighting -> default
        rp_log = run('risk_parity')              # explicit risk parity
        assert default_log == rp_log
        # And the default genuinely produced risk-parity weights (lower-vol
        # desk gets more), not an equal-weight fallback.
        last = default_log[-1]['weights']
        assert last['asim'] > last['bsim']
        assert all(e['fallback'] is False for e in default_log)
