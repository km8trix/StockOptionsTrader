"""Constructor-level tests for scripts/vq_fund_gate.run_fund weighting arms.

The A/B instrumentation must not drift: the default risk_parity arm has to
construct ReweightingFundBacktest with exactly the args it always used
(byte-identical behavior), and the static arm has to bypass it entirely —
one BacktestEngine with reweighter=None (fixed construction weights) on the
same capital/seed/feed. Everything expensive is stubbed; no market data.
"""

import pandas as pd
import pytest

import scripts.vq_fund_gate as vfg

#: Minimal fund report: enough for the gate path + the deployment stats.
#: Day-1 is all-cash by construction (T+1 fills) and must be SKIPPED by
#: _deployment_stats; deployed fractions for the counted snapshots:
#: 0.5, 0.8, 0.8 -> mean 0.7, median 0.8, min 0.5 (not 0.0).
_HISTORY = [
    {'timestamp': pd.Timestamp('2015-01-01'),
     'portfolio_value': 100.0, 'cash': 100.0},
    {'timestamp': pd.Timestamp('2015-01-02'),
     'portfolio_value': 100.0, 'cash': 50.0},
    {'timestamp': pd.Timestamp('2015-01-05'),
     'portfolio_value': 100.0, 'cash': 20.0},
    {'timestamp': pd.Timestamp('2015-01-06'),
     'portfolio_value': 200.0, 'cash': 40.0},
]
_REPORT = {'summary': {'total_return_pct': 1.0},
           'portfolio_history': _HISTORY, 'closed_trades': [1, 2]}


@pytest.fixture
def stubbed(monkeypatch):
    """Stub every expensive collaborator run_fund touches, keeping only its
    own construction/dispatch logic under test."""
    monkeypatch.setattr(vfg, 'PitWarehouse', lambda: 'WH')
    monkeypatch.setattr(vfg, '_universe',
                        lambda wh, s, e, limit, seed: ['AAA', 'BBB'])
    monkeypatch.setattr(vfg, 'WarehouseMarketData', lambda wh: 'FEED')
    monkeypatch.setattr(vfg, '_daily_returns_with_years',
                        lambda history: ('RETURNS', 'YEARS'))
    monkeypatch.setattr(vfg, 'validate_strategy_oos',
                        lambda returns, years, psr_threshold: {'psr': 0.5})
    monkeypatch.setattr(
        vfg, '_make_desk',
        lambda key, wh, *, capital_allocation=1.0, dating='filing':
        (key, capital_allocation, dating))
    monkeypatch.setattr(
        vfg, 'FundOrchestrator',
        lambda desks, risk_manager=None: ('ORCH', tuple(desks), risk_manager))
    monkeypatch.setattr(vfg, 'RiskManager',
                        lambda position_stop_loss=None:
                        ('RM', position_stop_loss))


def test_default_risk_parity_construction_unchanged(stubbed, monkeypatch):
    # Pins byte-identity of the default arm: same ReweightingFundBacktest
    # kwargs as before the --weighting flag existed, and no direct engine.
    captured = {}

    class FakeFund:
        def __init__(self, allocations, **kwargs):
            captured['allocations'] = allocations
            captured['kwargs'] = kwargs

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            captured['run'] = (tuple(symbols), start, end, benchmark_symbol)
            return dict(_REPORT)

    class BoomEngine:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                'risk_parity arm must go through ReweightingFundBacktest, '
                'never build the engine directly')

    monkeypatch.setattr(vfg, 'ReweightingFundBacktest', FakeFund)
    monkeypatch.setattr(vfg, 'BacktestEngine', BoomEngine)

    summary, gate, n_trades, n_names, deploy = vfg.run_fund(
        '2015-01-01', '2024-12-31', limit=2, seed=42)

    assert captured['allocations'] == vfg.FUND_ALLOCATIONS
    kwargs = captured['kwargs']
    assert set(kwargs) == {'initial_capital', 'seed', 'weighting',
                           'market_data', 'solo_curve_provider',
                           'orchestrator_factory'}
    assert kwargs['weighting'] == 'risk_parity'
    assert kwargs['initial_capital'] == 100_000.0
    assert kwargs['seed'] == 42
    assert kwargs['market_data'] == 'FEED'
    assert callable(kwargs['solo_curve_provider'])
    assert callable(kwargs['orchestrator_factory'])
    assert captured['run'] == (('AAA', 'BBB'), '2015-01-01', '2024-12-31',
                               None)
    assert (summary, gate) == (_REPORT['summary'], {'psr': 0.5})
    assert n_trades == 2 and n_names == 2
    assert deploy == {'mean': pytest.approx(0.7),
                      'median': pytest.approx(0.8),
                      'min': pytest.approx(0.5)}


def test_static_builds_engine_with_no_reweighter(stubbed, monkeypatch):
    # The static arm must skip the N+1 ReweightingFundBacktest path and run
    # ONE engine at fixed construction weights: reweighter=None, same
    # factory-built orchestrator (50/50 legs + wide 0.50 stop), capital,
    # seed and feed as the risk-parity arm.
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured['engine_kwargs'] = kwargs

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            captured['run'] = (tuple(symbols), start, end, benchmark_symbol)
            return dict(_REPORT)

    class BoomFund:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                'static arm must skip ReweightingFundBacktest '
                '(no solo shadow passes)')

    monkeypatch.setattr(vfg, 'BacktestEngine', FakeEngine)
    monkeypatch.setattr(vfg, 'ReweightingFundBacktest', BoomFund)

    summary, gate, n_trades, n_names, deploy = vfg.run_fund(
        '2015-01-01', '2024-12-31', limit=2, seed=7, dating='announce',
        weighting='static')

    kwargs = captured['engine_kwargs']
    assert kwargs['reweighter'] is None
    assert kwargs['initial_capital'] == 100_000.0
    assert kwargs['seed'] == 7
    assert kwargs['market_data'] == 'FEED'
    orch, desks, risk_manager = kwargs['orchestrator']
    assert orch == 'ORCH'
    assert desks == (('pead_micro', 0.5, 'announce'),
                     ('value_quality', 0.5, 'announce'))
    assert risk_manager == ('RM', 0.50)
    assert captured['run'] == (('AAA', 'BBB'), '2015-01-01', '2024-12-31',
                               None)
    assert n_trades == 2 and n_names == 2
    assert deploy == {'mean': pytest.approx(0.7),
                      'median': pytest.approx(0.8),
                      'min': pytest.approx(0.5)}


def test_deployment_stats_mean_median_min():
    assert vfg._deployment_stats(_HISTORY) == {
        'mean': pytest.approx(0.7), 'median': pytest.approx(0.8),
        'min': pytest.approx(0.5)}


def test_deployment_stats_empty_history_is_none():
    assert vfg._deployment_stats([]) is None


def test_allocation_keys_match_desk_keys():
    # THE reweighting contract: ReweightingFundBacktest stores solo curves
    # under the allocation key; DynamicReweighter looks them up by desk.key.
    # A mismatch does not crash — every rebalance silently degenerates to
    # the whole-fund equal-weight fallback (how the pre-2026-07-10 'pead'
    # vs 'pead_micro' bug made the risk-parity arm a de facto static fund).
    # Real desk classes, no stubs: the .key values are what production sees.
    for allocations in (vfg.FUND_ALLOCATIONS, vfg.LEGACY_FUND_ALLOCATIONS,
                        vfg.THREE_LEG_ALLOCATIONS):
        for key in allocations:
            desk = vfg._make_desk(key, object())
            assert desk.key == key, (
                f"allocation key {key!r} builds desk.key {desk.key!r}: "
                f"solo curves would never reach the reweighter")


def test_three_leg_allocations_are_the_preregistered_equal_split():
    # The third-leg combine is pre-registered: exactly the two incumbent
    # legs + issuance at 1/3 each. Any weight tuning must fail this test and
    # be argued in review, not slipped in.
    assert vfg.THREE_LEG_ALLOCATIONS == {
        'pead_micro': pytest.approx(1 / 3),
        'value_quality': pytest.approx(1 / 3),
        'issuance': pytest.approx(1 / 3)}


def test_issuance_fund_builds_three_leg_orchestrator(stubbed, monkeypatch):
    # run_fund(issuance=True) must hand the THREE_LEG allocations to the
    # orchestrator factory — same static-arm engine construction otherwise.
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured['engine_kwargs'] = kwargs

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            return dict(_REPORT)

    monkeypatch.setattr(vfg, 'BacktestEngine', FakeEngine)

    vfg.run_fund('2015-01-01', '2024-12-31', limit=2, seed=42,
                 dating='announce', weighting='static', issuance=True)

    kwargs = captured['engine_kwargs']
    assert kwargs['reweighter'] is None
    orch, desks, risk_manager = kwargs['orchestrator']
    assert desks == (('pead_micro', pytest.approx(1 / 3), 'announce'),
                     ('value_quality', pytest.approx(1 / 3), 'announce'),
                     ('issuance', pytest.approx(1 / 3), 'announce'))
    assert risk_manager == ('RM', 0.50)
