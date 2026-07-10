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
        lambda key, wh, *, capital_allocation=1.0, dating='filing',
        vq_variant='base':
        (key, capital_allocation, dating, vq_variant))
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
    assert desks == (('pead_micro', 0.5, 'announce', 'base'),
                     ('value_quality', 0.5, 'announce', 'base'))
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
    assert desks == (
        ('pead_micro', pytest.approx(1 / 3), 'announce', 'base'),
        ('value_quality', pytest.approx(1 / 3), 'announce', 'base'),
        ('issuance', pytest.approx(1 / 3), 'announce', 'base'))
    assert risk_manager == ('RM', 0.50)


def test_vq_variant_kwargs_are_the_preregistered_trials():
    # The strengthening round is pre-registered: base is EMPTY (byte-
    # identical default desk) and exactly three trial variants exist —
    # equal-weight gp_assets, the top-quintile issuance long-filter, and
    # their composition. Adding or tuning an entry must fail this test and
    # be argued in review, not slipped in.
    assert vfg.VQ_VARIANT_KWARGS == {
        'base': {},
        'gpa': {'include_gp_assets': True},
        'issfilter': {'issuance_filter': True},
        'both': {'include_gp_assets': True, 'issuance_filter': True}}


def test_make_desk_maps_variant_to_vq_kwargs():
    # Real desk classes, no stubs: every variant reaches the VQ constructor
    # as its registered kwargs, the key contract survives all of them, and
    # the default is the unchanged base desk.
    for variant, kwargs in vfg.VQ_VARIANT_KWARGS.items():
        desk = vfg._make_desk('value_quality', object(), vq_variant=variant)
        assert desk.key == 'value_quality'
        assert desk._include_gp_assets is kwargs.get('include_gp_assets',
                                                     False)
        assert desk._issuance_filter is kwargs.get('issuance_filter', False)
    base = vfg._make_desk('value_quality', object())
    assert base._include_gp_assets is False
    assert base._issuance_filter is False
    # Non-VQ legs are untouched by the variant.
    assert vfg._make_desk('pead_micro', object(),
                          vq_variant='both').key == 'pead_micro'


def test_run_vq_applies_variant_kwargs(stubbed, monkeypatch):
    captured = {}

    class FakeDesk:
        def __init__(self, **kwargs):
            captured['desk_kwargs'] = kwargs

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            return dict(_REPORT)

    monkeypatch.setattr(vfg, 'ValueQualityDesk', FakeDesk)
    monkeypatch.setattr(vfg, 'BacktestEngine', FakeEngine)

    vfg.run_vq('2015-01-01', '2024-12-31', limit=2, seed=42,
               long_only=True, vq_variant='both')
    assert captured['desk_kwargs'] == {
        'provider': 'WH', 'long_only': True,
        'include_gp_assets': True, 'issuance_filter': True}
    # Default: NO variant kwargs at all — the constructor call is
    # byte-identical to before --vq-variant existed.
    vfg.run_vq('2015-01-01', '2024-12-31', limit=2, seed=42, long_only=True)
    assert captured['desk_kwargs'] == {'provider': 'WH', 'long_only': True}


def test_run_fund_threads_vq_variant_to_the_orchestrator(stubbed,
                                                         monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured['engine_kwargs'] = kwargs

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            return dict(_REPORT)

    monkeypatch.setattr(vfg, 'BacktestEngine', FakeEngine)

    vfg.run_fund('2015-01-01', '2024-12-31', limit=2, seed=42,
                 dating='announce', weighting='static', vq_variant='gpa')
    orch, desks, _ = captured['engine_kwargs']['orchestrator']
    # The stub echoes vq_variant for every leg; only the VQ leg acts on it
    # (test_make_desk_maps_variant_to_vq_kwargs pins that).
    assert desks == (('pead_micro', 0.5, 'announce', 'gpa'),
                     ('value_quality', 0.5, 'announce', 'gpa'))


def test_risk_parity_solo_curves_carry_the_variant(stubbed, monkeypatch):
    # The reweighting arm's solo shadow passes must build the SAME variant
    # desk as the fund legs, or the risk-parity weights would be computed
    # off a different book than the one trading.
    captured = {}

    class FakeFund:
        def __init__(self, allocations, **kwargs):
            captured['solo'] = kwargs['solo_curve_provider']

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            return dict(_REPORT)

    class FakeEngine:
        def __init__(self, **kwargs):
            captured['desk'] = kwargs['desk']

        def run(self, symbols, start, end, benchmark_symbol='SPY'):
            return dict(_REPORT)

    monkeypatch.setattr(vfg, 'ReweightingFundBacktest', FakeFund)
    monkeypatch.setattr(vfg, 'BacktestEngine', FakeEngine)

    vfg.run_fund('2015-01-01', '2024-12-31', limit=2, seed=42,
                 vq_variant='issfilter')
    captured['solo']('value_quality', ['AAA'], '2015-01-01', '2024-12-31')
    assert captured['desk'] == ('value_quality', 1.0, 'filing', 'issfilter')
