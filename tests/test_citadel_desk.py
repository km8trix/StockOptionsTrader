"""Tests for desks.citadel.CitadelDesk.

Unit tests drive generate_intents day by day with SCRIPTED pods and
hand-built portfolios (engine-faithful day order: mark -> intents ->
snapshot; fills simulated explicitly) so every number is controlled:
probation/cut boundary semantics are pinned with binary-exact thresholds
(<= triggers exactly), reallocation math is pinned at the pure-function
level AND through the desk, and the cross-pod first-claim rule is
asserted directly. The end-to-end tests run the real BacktestEngine on
synthetic data and check the C8/C9 report shapes plus the attribution
identity: sum over pods of (capital base x nav-implied daily P&L)
reconciles with the desk's portfolio position P&L to 1e-6 every day —
reconstructed PURELY from the report, including the cut-pod residual
path (weight already 0, flatten fill still pending) via the additive
residual_capital field. Offline, seeded, deterministic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from data.market_data import MarketDataHandler
from desks.citadel import CitadelDesk, clamp_renormalize, pod_score_inputs
from desks.foundation import FoundationDesk
from desks.ml_model import GradientBoostingModel
from desks.pods import MeanReversionPod, MomentumPod, Pod, StatArbPod
from desks.renaissance import RenaissanceDesk
from desks.risk_book import CentralRiskBook
from desks.walk_forward import WalkForwardController
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager
from strategies.base import MomentumStrategy

DATES = pd.bdate_range('2023-01-02', periods=40)


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


class ScriptedPod(Pod):
    """Emits exactly the scripted {symbol: action} map per date."""

    def __init__(self, key, schedule=None):
        super().__init__(key=key, name=f"{key.title()} Pod")
        self.schedule = {pd.Timestamp(d): dict(actions)
                         for d, actions in (schedule or {}).items()}

    def generate_signals(self, all_data, date):
        return dict(self.schedule.get(pd.Timestamp(date), {}))


def make_desk(pods, weights=None, **kwargs):
    """Citadel desk with wide base-risk and risk-book limits so the pod
    mechanics under test stay pure (the limits have their own tests)."""
    kwargs.setdefault('risk_manager', RiskManager(
        max_position_size=1.0, max_daily_loss=0.99,
        position_stop_loss=0.90))
    kwargs.setdefault('risk_book', CentralRiskBook(
        max_gross=2.0, max_symbol=1.0, max_net=2.0))
    return CitadelDesk(pods=pods, pod_weights=weights, **kwargs)


# ----------------------------------------------------------------------
# Engine-faithful single-day driver (fills are simulated by the caller)
# ----------------------------------------------------------------------
def drive_day(desk, portfolio, date, frames=None, marks=None):
    """Mark positions to today's price (engine PHASE 2), generate intents
    (PHASE 3), record the snapshot (PHASE 4)."""
    for symbol, price in (marks or {}).items():
        position = portfolio.get_position(stock(symbol))
        if position is not None:
            position.current_price = price
    desk.set_clock(date)
    sliced = {}
    for symbol, frame in (frames or {}).items():
        window = frame[frame.index <= date]
        if not window.empty:
            sliced[symbol] = window
    intents = desk.generate_intents(sliced, date, portfolio)
    portfolio.record_snapshot(date)
    return intents


def fill_buy(portfolio, symbol, quantity, price, date):
    """Simulate the engine's next-open BUY fill (no commission)."""
    portfolio.cash -= quantity * price
    portfolio.add_position(Position(
        asset=stock(symbol), quantity=quantity, avg_entry_price=price,
        current_price=price, timestamp=date))


def fill_close(portfolio, symbol, price, date):
    """Simulate the engine's next-open SELL/COVER fill (no commission)."""
    position = portfolio.get_position(stock(symbol))
    portfolio.cash += position.quantity * price
    portfolio.close_position(stock(symbol), price, position.quantity,
                             position.timestamp, date)


def assert_attribution_identity(report, capital_allocation=1.0):
    """The report-level attribution identity (desks.citadel module
    docstring), reconstructed PURELY from the report: per pod, the
    capital base is start-of-day capital times the weight recorded in
    yesterday's pod_history entry; a pod whose recorded weight is
    already 0 (cut) attributes against yesterday's residual_capital
    instead. The summed nav-implied P&L must reconcile with the desk's
    daily position P&L (realized + unrealized change) to 1e-6."""
    history = report['portfolio_history']
    pod_history = report['pod_history']
    assert len(history) == len(pod_history)
    for t in range(1, len(history)):
        previous, current = pod_history[t - 1], pod_history[t]
        capital = history[t - 1]['portfolio_value'] * capital_allocation
        nav_implied_pnl = 0.0
        for key, prev_pod in previous['pods'].items():
            base = capital * prev_pod['weight']
            if base <= 0:
                base = prev_pod.get('residual_capital', 0.0)
            nav_implied_pnl += base * (current['pods'][key]['nav']
                                       / prev_pod['nav'] - 1.0)
        desk_position_pnl = (
            (history[t]['realized_pnl'] + history[t]['unrealized_pnl'])
            - (history[t - 1]['realized_pnl']
               + history[t - 1]['unrealized_pnl']))
        assert nav_implied_pnl == pytest.approx(desk_position_pnl,
                                                abs=1e-6), \
            f"attribution identity broken on day {t}"


def assert_live_weight_conservation(pod_history):
    """Round-1 item 6: at every daily pod_history entry the active +
    probation weights sum to exactly 1.0 (1e-9); 0.0 only when ALL pods
    are cut."""
    assert pod_history
    for entry in pod_history:
        live = [pod for pod in entry['pods'].values()
                if pod['status'] != 'cut']
        total = sum(pod['weight'] for pod in live)
        expected = 1.0 if live else 0.0
        assert total == pytest.approx(expected, abs=1e-9), \
            f"live weight {total} != {expected} on {entry['date']}"


def assert_c8_pod_entry_shape(pod):
    """C8 per-pod shape; 'residual_capital' is the documented ADDITIVE
    field, present only on cut pods (residual attribution base)."""
    required = {'weight', 'nav', 'drawdown_pct', 'status'}
    assert required <= set(pod)
    assert set(pod) - required <= {'residual_capital'}
    if 'residual_capital' in pod:
        assert pod['status'] == 'cut'
        assert isinstance(pod['residual_capital'], float)
    assert isinstance(pod['weight'], float)
    assert isinstance(pod['nav'], float)
    assert pod['drawdown_pct'] <= 0
    assert pod['status'] in ('active', 'probation', 'cut')


def probation_setup(**desk_kwargs):
    """alpha (weight 0.5) owns 100 X bought at 100; desk capital pinned
    at 100,000 -> alpha's allocated capital is exactly 50,000."""
    pods = [ScriptedPod('alpha', {DATES[0]: {'X': 'BUY'}}),
            ScriptedPod('beta'), ScriptedPod('gamma')]
    desk = make_desk(pods,
                     weights={'alpha': 0.5, 'beta': 0.25, 'gamma': 0.25},
                     **desk_kwargs)
    portfolio = PortfolioManager(100_000.0)
    intents = drive_day(desk, portfolio, DATES[0])
    assert [(i.action, i.asset.symbol) for i in intents] == [('BUY', 'X')]
    fill_buy(portfolio, 'X', 100, 100.0, DATES[1])
    drive_day(desk, portfolio, DATES[1])  # P&L 0; NAVs stay 1.0
    return desk, portfolio


# ======================================================================
# Constructor: pod weights must sum to exactly 1.0
# ======================================================================
class TestPodWeightValidation:
    def _pods(self):
        return [ScriptedPod('alpha'), ScriptedPod('beta')]

    def test_under_allocated_weights_are_rejected(self):
        # Sub-1.0 sums would silently jump to full deployment at the
        # first scheduled reallocation (which targets the full unheld
        # mass), so they are rejected up front.
        with pytest.raises(ValueError, match='must sum to 1.0'):
            CitadelDesk(pods=self._pods(),
                        pod_weights={'alpha': 0.5, 'beta': 0.4})

    def test_over_allocated_weights_are_rejected(self):
        with pytest.raises(ValueError, match='must sum to 1.0'):
            CitadelDesk(pods=self._pods(),
                        pod_weights={'alpha': 0.6, 'beta': 0.5})

    def test_exact_sum_of_one_is_accepted(self):
        desk = CitadelDesk(pods=self._pods(),
                           pod_weights={'alpha': 0.6, 'beta': 0.4})
        pods = desk.get_status()['pods']
        assert pods['alpha']['weight'] == 0.6
        assert pods['beta']['weight'] == 0.4


# ======================================================================
# (b) Probation: boundary-exact drawdown semantics
# ======================================================================
class TestProbationBoundary:
    def test_exact_threshold_triggers_probation_with_halved_weight(self):
        # Binary-exact engineering: threshold -0.0625 (= -1/16), P&L
        # -3,125 on allocated 50,000 -> NAV 0.9375 and drawdown exactly
        # -0.0625. The <= boundary must trigger AT the threshold.
        desk, portfolio = probation_setup(probation_drawdown=-0.0625,
                                          cut_drawdown=-0.5,
                                          recovery_drawdown=-0.03125)
        drive_day(desk, portfolio, DATES[2], marks={'X': 68.75})

        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'probation'
        assert pods['alpha']['nav'] == pytest.approx(0.9375)
        assert pods['alpha']['drawdown_pct'] == pytest.approx(-6.25)
        # Weight halved; the excess redistributed pro-rata to the others.
        assert pods['alpha']['weight'] == pytest.approx(0.25)
        assert pods['beta']['weight'] == pytest.approx(0.375)
        assert pods['gamma']['weight'] == pytest.approx(0.375)

        notes = [n for n in desk.notes if 'PROBATION' in n.message]
        assert len(notes) == 1
        assert notes[0].category == 'risk'
        assert notes[0].data['pod'] == 'alpha'  # contract C9
        assert notes[0].data['drawdown_pct'] == pytest.approx(-6.25)

    def test_epsilon_inside_threshold_does_not_trigger(self):
        # P&L -3,121.875 -> drawdown -0.0624375, strictly above -0.0625.
        desk, portfolio = probation_setup(probation_drawdown=-0.0625,
                                          cut_drawdown=-0.5,
                                          recovery_drawdown=-0.03125)
        drive_day(desk, portfolio, DATES[2], marks={'X': 68.78125})
        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'active'
        assert pods['alpha']['weight'] == pytest.approx(0.5)
        assert [n for n in desk.notes if 'PROBATION' in n.message] == []

    def test_default_thresholds_minus_4_99_pct_does_not_trigger(self):
        desk, portfolio = probation_setup()  # defaults: -5% / -8%
        drive_day(desk, portfolio, DATES[2], marks={'X': 75.05})  # -4.99%
        assert desk.get_status()['pods']['alpha']['status'] == 'active'

    def test_default_thresholds_minus_5_5_pct_triggers(self):
        desk, portfolio = probation_setup()
        drive_day(desk, portfolio, DATES[2], marks={'X': 72.5})  # -5.5%
        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'probation'
        assert pods['alpha']['weight'] == pytest.approx(0.25)


# ======================================================================
# (c) Cuts: stop, flatten, redistribute
# ======================================================================
class TestCutPod:
    def test_cut_redistributes_weight_and_emits_flatten_intents(self):
        desk, portfolio = probation_setup()  # defaults: -5% / -8%
        # -5,000 on 50,000 -> -10% <= -8%: CUT (checked before probation).
        intents = drive_day(desk, portfolio, DATES[2], marks={'X': 50.0})

        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'cut'
        assert pods['alpha']['weight'] == 0.0
        assert pods['beta']['weight'] == pytest.approx(0.5)
        assert pods['gamma']['weight'] == pytest.approx(0.5)

        assert [(i.action, i.asset.symbol, i.size_fraction)
                for i in intents] == [('SELL', 'X', 1.0)]

        notes = [n for n in desk.notes if 'CUT' in n.message]
        assert len(notes) == 1
        assert notes[0].data['pod'] == 'alpha'  # contract C9
        assert notes[0].data['drawdown_pct'] == pytest.approx(-10.0)
        assert notes[0].data['closed_symbols'] == ['X']

    def test_flatten_fill_next_day_keeps_attribution_and_stays_cut(self):
        desk, portfolio = probation_setup()
        drive_day(desk, portfolio, DATES[2], marks={'X': 50.0})
        # The engine fills the SELL at the next bar's open; filling at
        # the marked price realizes exactly the already-attributed loss,
        # so alpha's NAV must NOT move again.
        fill_close(portfolio, 'X', 50.0, DATES[3])
        intents = drive_day(desk, portfolio, DATES[3])
        assert intents == []  # no resurrection, no orphan sweep
        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'cut'
        assert pods['alpha']['nav'] == pytest.approx(0.90)
        assert portfolio.get_position(stock('X')) is None

    def test_flatten_fill_slippage_attributes_residual_capital(self):
        desk, portfolio = probation_setup()
        drive_day(desk, portfolio, DATES[2], marks={'X': 50.0})  # cut
        # The cut-day entry already publishes the residual base: alpha's
        # last positive allocated capital (100,000 * 0.5).
        cut_entry = desk.pod_history[-1]['pods']['alpha']
        assert cut_entry['status'] == 'cut'
        assert cut_entry['weight'] == 0.0
        assert cut_entry['residual_capital'] == pytest.approx(50_000.0)

        # The engine fills the SELL at the next open BELOW the cut-day
        # mark (slippage): the extra -100 realized is residual P&L,
        # attributed against that base even though alpha's recorded
        # weight is already 0 — NAV moves by exactly -100 / 50,000.
        fill_close(portfolio, 'X', 49.0, DATES[3])
        drive_day(desk, portfolio, DATES[3])
        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'cut'
        assert pods['alpha']['nav'] == pytest.approx(
            0.90 * (1.0 - 100.0 / 50_000.0))
        assert pods['alpha']['residual_capital'] == pytest.approx(50_000.0)
        # Active pods carry NO residual_capital field (C8 base shape).
        assert 'residual_capital' not in pods['beta']

    def test_cut_exact_boundary_triggers_and_epsilon_inside_does_not(self):
        # cut_drawdown -0.125 (= -1/8, binary exact): P&L -6,250 on
        # 50,000 lands EXACTLY on the boundary -> cut.
        desk, portfolio = probation_setup(probation_drawdown=-0.05,
                                          cut_drawdown=-0.125)
        drive_day(desk, portfolio, DATES[2], marks={'X': 37.5})
        assert desk.get_status()['pods']['alpha']['status'] == 'cut'

        # Epsilon inside (-0.1249): NOT cut — it degrades to probation
        # (still <= -5%), proving the two thresholds are evaluated in
        # order with exact boundaries.
        desk2, portfolio2 = probation_setup(probation_drawdown=-0.05,
                                            cut_drawdown=-0.125)
        drive_day(desk2, portfolio2, DATES[2], marks={'X': 37.55})
        assert desk2.get_status()['pods']['alpha']['status'] == 'probation'

    def test_cut_pod_is_stopped_permanently(self):
        pods = [ScriptedPod('alpha', {DATES[0]: {'X': 'BUY'},
                                      DATES[3]: {'Y': 'BUY'}}),
                ScriptedPod('beta'), ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 0.5, 'beta': 0.25,
                                        'gamma': 0.25})
        portfolio = PortfolioManager(100_000.0)
        drive_day(desk, portfolio, DATES[0])
        fill_buy(portfolio, 'X', 100, 100.0, DATES[1])
        drive_day(desk, portfolio, DATES[1])
        drive_day(desk, portfolio, DATES[2], marks={'X': 50.0})  # cut
        fill_close(portfolio, 'X', 50.0, DATES[3])
        # alpha's scripted day-3 BUY Y must be ignored: the pod is cut.
        intents = drive_day(desk, portfolio, DATES[3])
        assert intents == []
        assert stock('Y') not in desk._pod_positions


# ======================================================================
# (d) Reallocation: pure math + the day-21 schedule through the desk
# ======================================================================
class TestReallocationMath:
    def test_clamp_renormalize_reaches_documented_fixed_point(self):
        # One dominant raw score: normalize -> (0.994, 0.003, 0.003),
        # clamp pins a at 0.5, the remaining mass renormalizes over b/c:
        # fixed point exactly (0.5, 0.25, 0.25).
        weights = clamp_renormalize({'a': 100.0, 'b': 0.315, 'c': 0.315},
                                    target=1.0, weight_min=0.10,
                                    weight_max=0.50)
        assert weights['a'] == pytest.approx(0.50, rel=1e-12)
        assert weights['b'] == pytest.approx(0.25, rel=1e-12)
        assert weights['c'] == pytest.approx(0.25, rel=1e-12)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-12)

    def test_clamp_renormalize_unpinned_keep_raw_proportions(self):
        # Canonical Round-1 scenario: the dominant pod (Sharpe ~2, low
        # vol) pins at weight_max, and BOTH small pods transiently clamp
        # up to the 0.10 floor on the first pass. That intermediate
        # flooring must NOT destroy their raw proportions: the fixed
        # point reallocates the remaining 0.50 proportionally to raw
        # (0.137 vs 0.255 — a 1.9x difference), not equal-weighted.
        weights = clamp_renormalize({'a': 25.07, 'b': 0.137, 'c': 0.255},
                                    target=1.0, weight_min=0.10,
                                    weight_max=0.50)
        assert weights['a'] == pytest.approx(0.50, rel=1e-12)
        assert weights['b'] == pytest.approx(0.5 * 0.137 / 0.392)  # 0.1747
        assert weights['c'] == pytest.approx(0.5 * 0.255 / 0.392)  # 0.3253
        # The unpinned pods keep their raw ratio exactly.
        assert weights['b'] / weights['c'] \
            == pytest.approx(0.137 / 0.255, rel=1e-9)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-12)

    def test_clamp_renormalize_feasible_case_is_plain_normalization(self):
        weights = clamp_renormalize({'a': 1.0, 'b': 1.0, 'c': 1.0},
                                    target=1.0, weight_min=0.10,
                                    weight_max=0.50)
        for value in weights.values():
            assert value == pytest.approx(1.0 / 3.0)

    def test_clamp_renormalize_infeasible_returns_clamped_vector(self):
        # Under-allocation: a single pod cannot absorb target 1.0 under
        # the HARD max bound 0.5 — the desk deliberately runs below
        # full allocation (documented).
        assert clamp_renormalize({'a': 1.0}, 1.0, 0.10, 0.50) == {'a': 0.50}

    def test_clamp_renormalize_over_allocation_min_bound_yields(self):
        # Over-allocation (target < n * weight_min): the min bound is
        # SOFT — the floor drops to target/n so the returned mass sums
        # to the target EXACTLY. A hard floor would return 0.20 > 0.15,
        # pushing total desk weight (held + participating) above 1.0.
        low = clamp_renormalize({'a': 1.0, 'b': 1.0}, 0.15, 0.10, 0.50)
        assert low['a'] == pytest.approx(0.075)
        assert low['b'] == pytest.approx(0.075)
        assert sum(low.values()) == pytest.approx(0.15)

        # Unequal raw scores: the floored key pins at target/n and the
        # dominant key absorbs the remainder; mass is still conserved.
        skew = clamp_renormalize({'a': 100.0, 'b': 1.0}, 0.15, 0.10, 0.50)
        assert skew['b'] == pytest.approx(0.075)
        assert sum(skew.values()) == pytest.approx(0.15)

        # Degenerate target 0 (everything held elsewhere): zeros, not
        # weight_min each.
        zero = clamp_renormalize({'a': 1.0, 'b': 1.0}, 0.0, 0.10, 0.50)
        assert zero == {'a': 0.0, 'b': 0.0}

    def test_pod_score_inputs_floors_and_sharpe(self):
        # Zero-dispersion returns: sharpe 0, vol floored, raw = floor/floor.
        flat = pod_score_inputs([0.0] * 63)
        assert flat == {'sharpe': 0.0, 'score': 0.0, 'vol': 0.05,
                        'raw': 1.0}

        # Alternating +2%/0%: mean 0.01, population std 0.01 ->
        # sharpe = sqrt(252), and raw = sharpe/vol = mean/std^2 = 100.
        winner = pod_score_inputs([0.02, 0.0] * 31)
        assert winner['sharpe'] == pytest.approx(float(np.sqrt(252.0)))
        assert winner['vol'] == pytest.approx(0.01 * float(np.sqrt(252.0)))
        assert winner['raw'] == pytest.approx(100.0)

        # Losing pod: negative sharpe floors the score at 0, raw uses
        # the 0.05 score floor over the real vol.
        loser = pod_score_inputs([-0.02, 0.0] * 31)
        assert loser['sharpe'] < 0
        assert loser['score'] == 0.0
        assert loser['raw'] == pytest.approx(
            0.05 / (0.01 * float(np.sqrt(252.0))))


class TestReallocationSchedule:
    def _flat_desk(self):
        pods = [ScriptedPod('alpha'), ScriptedPod('beta'),
                ScriptedPod('gamma')]
        return make_desk(pods, weights={'alpha': 0.5, 'beta': 0.3,
                                        'gamma': 0.2})

    def test_reallocation_fires_at_day_21_with_handcomputable_weights(self):
        desk = self._flat_desk()
        portfolio = PortfolioManager(100_000.0)
        for date in DATES[:21]:  # days 0..20: no reallocation yet
            drive_day(desk, portfolio, date)
        assert [n for n in desk.notes if n.category == 'allocation'] == []
        assert desk.get_status()['pods']['alpha']['weight'] == 0.5

        drive_day(desk, portfolio, DATES[21])  # trading day 21: realloc
        notes = [n for n in desk.notes if n.category == 'allocation']
        assert len(notes) == 1
        data = notes[0].data
        # All pods flat: sharpe 0, raw = 0.05/0.05 = 1 each -> exactly
        # equal thirds regardless of the starting weights (contract C9:
        # data.allocations + data.reason).
        assert set(data['allocations']) == {'alpha', 'beta', 'gamma'}
        for value in data['allocations'].values():
            assert value == pytest.approx(1.0 / 3.0)
        assert data['old_weights'] == {'alpha': 0.5, 'beta': 0.3,
                                       'gamma': 0.2}
        assert isinstance(data['reason'], str) and data['reason']
        for inputs in data['inputs'].values():
            assert set(inputs) == {'sharpe', 'score', 'vol', 'raw'}
        pods = desk.get_status()['pods']
        for key in ('alpha', 'beta', 'gamma'):
            assert pods[key]['weight'] == pytest.approx(1.0 / 3.0)
        assert_live_weight_conservation(desk.pod_history)

    def test_realloc_with_held_mass_keeps_total_weight_at_one(self):
        # Pathological config: two unrecovered probation pods hold 0.92
        # of the desk, so the sole participating pod's target (0.08) is
        # BELOW weight_min. The soft min floor must yield: held +
        # participating == 1.0 exactly (a hard floor would clamp gamma
        # to 0.10 -> total 1.02).
        pods = [ScriptedPod('alpha'), ScriptedPod('beta'),
                ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 0.46, 'beta': 0.46,
                                        'gamma': 0.08},
                         realloc_every_days=1, min_nav_days=1)
        portfolio = PortfolioManager(100_000.0)
        drive_day(desk, portfolio, DATES[0])
        # White-box: alpha and beta sit on unrecovered probation
        # (drawdown -7% is below the -2.5% recovery threshold).
        for key in ('alpha', 'beta'):
            desk._statuses[key] = 'probation'
            desk._navs[key] = 0.93
            desk._nav_max[key] = 1.0
        drive_day(desk, portfolio, DATES[1])  # day 1: realloc due

        notes = [n for n in desk.notes if n.category == 'allocation']
        assert len(notes) == 1
        assert notes[0].data['held'] == ['alpha', 'beta']
        pods_now = desk.get_status()['pods']
        assert pods_now['alpha']['weight'] == pytest.approx(0.46)
        assert pods_now['beta']['weight'] == pytest.approx(0.46)
        assert pods_now['gamma']['weight'] == pytest.approx(0.08)
        assert sum(p['weight'] for p in pods_now.values()) \
            == pytest.approx(1.0)
        assert_live_weight_conservation(desk.pod_history)

    def test_pods_with_too_few_nav_days_keep_current_weight(self):
        desk = self._flat_desk()
        desk.min_nav_days = 50  # more history than the run provides
        portfolio = PortfolioManager(100_000.0)
        for date in DATES[:22]:
            drive_day(desk, portfolio, date)
        notes = [n for n in desk.notes if n.category == 'allocation']
        assert len(notes) == 1
        assert notes[0].data['allocations'] == {'alpha': 0.5, 'beta': 0.3,
                                                'gamma': 0.2}
        assert sorted(notes[0].data['held']) == ['alpha', 'beta', 'gamma']


class TestSingleSurvivorConservation:
    """Weight conservation (Round-1 item 6) through the REAL
    generate_intents flow in the two degenerate single-survivor paths:
    a scheduled reallocation whose unheld mass exceeds the survivors'
    hard weight_max cap (held at current weights, NOT under-allocated),
    and a probation with no recipient pods left (weight NOT halved into
    retirement). Active + probation weights must sum to exactly 1.0 at
    every daily pod_history entry whenever any non-cut pod remains."""

    def _survivor_desk(self):
        pods = [ScriptedPod('alpha'), ScriptedPod('beta'),
                ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 1.0 / 3.0,
                                        'beta': 1.0 / 3.0,
                                        'gamma': 1.0 / 3.0})
        portfolio = PortfolioManager(100_000.0)
        drive_day(desk, portfolio, DATES[0])
        drive_day(desk, portfolio, DATES[1])
        # White-box: beta and gamma crash through the -8% cut threshold.
        for key in ('beta', 'gamma'):
            desk._navs[key] = 0.90
            desk._nav_max[key] = 1.0
        drive_day(desk, portfolio, DATES[2])  # both cut on day 2
        pods_now = desk.get_status()['pods']
        assert pods_now['beta']['status'] == 'cut'
        assert pods_now['gamma']['status'] == 'cut'
        # The survivor absorbed everything: weight exactly 1.0.
        assert pods_now['alpha']['status'] == 'active'
        assert pods_now['alpha']['weight'] == pytest.approx(1.0, abs=1e-9)
        return desk, portfolio

    def test_survivor_realloc_holds_weight_at_one(self):
        # Day-21 scheduled reallocation with one participant and target
        # 1.0 > 1 * weight_max (0.5): the recompute is skipped and the
        # survivor is HELD at 1.0 — the desk must NOT silently halve its
        # deployment for the rest of the run.
        desk, portfolio = self._survivor_desk()
        for date in DATES[3:22]:
            drive_day(desk, portfolio, date)  # DATES[21]: realloc fires

        notes = [n for n in desk.notes if n.category == 'allocation']
        assert len(notes) == 1
        data = notes[0].data
        assert data['cap_infeasible'] is True
        assert data['allocations']['alpha'] == pytest.approx(1.0, abs=1e-9)
        assert 'alpha' in data['held']
        pods_now = desk.get_status()['pods']
        assert pods_now['alpha']['status'] == 'active'
        assert pods_now['alpha']['weight'] == pytest.approx(1.0, abs=1e-9)
        # Conservation after the reallocation and on every daily entry.
        assert_live_weight_conservation(desk.pod_history)

    def test_survivor_probation_without_recipients_keeps_weight(self):
        # The lone survivor drifts to -6%: probation triggers, but with
        # every other pod cut there is no recipient for the usual
        # halving — the weight must stay 1.0 (status still flips, note
        # still carries C9 data.pod + data.drawdown_pct).
        desk, portfolio = self._survivor_desk()
        desk._navs['alpha'] = 0.94
        drive_day(desk, portfolio, DATES[3])

        pods_now = desk.get_status()['pods']
        assert pods_now['alpha']['status'] == 'probation'
        assert pods_now['alpha']['weight'] == pytest.approx(1.0, abs=1e-9)
        notes = [n for n in desk.notes if 'PROBATION' in n.message]
        assert len(notes) == 1
        assert notes[0].data['pod'] == 'alpha'
        assert notes[0].data['drawdown_pct'] == pytest.approx(-6.0)
        assert notes[0].data['old_weight'] == pytest.approx(1.0, abs=1e-9)
        assert notes[0].data['new_weight'] == pytest.approx(1.0, abs=1e-9)
        assert_live_weight_conservation(desk.pod_history)

    def test_true_all_cut_still_retires_to_zero(self):
        # The deliberate exception stays: when the survivor itself is
        # cut there is no live pod left and total weight retires to 0.0.
        desk, portfolio = self._survivor_desk()
        desk._navs['alpha'] = 0.90  # -10% <= -8%: cut
        drive_day(desk, portfolio, DATES[3])

        pods_now = desk.get_status()['pods']
        assert all(pod['status'] == 'cut' for pod in pods_now.values())
        assert sum(pod['weight'] for pod in pods_now.values()) == 0.0
        assert [n for n in desk.notes if 'All pods cut' in n.message]
        assert_live_weight_conservation(desk.pod_history)


# ======================================================================
# (e) Probation recovery rule at the next scheduled reallocation
# ======================================================================
class TestProbationRecovery:
    KWARGS = dict(realloc_every_days=5, min_nav_days=2)

    def _probation_at_minus_6pct(self, **extra):
        desk, portfolio = probation_setup(**self.KWARGS, **extra)
        # -3,000 on 50,000 -> -6% <= -5%: probation, weight 0.25.
        drive_day(desk, portfolio, DATES[2], marks={'X': 70.0})
        assert desk.get_status()['pods']['alpha']['status'] == 'probation'
        return desk, portfolio

    def test_recovered_pod_returns_to_eligibility(self):
        desk, portfolio = self._probation_at_minus_6pct()
        # Full price recovery: NAV climbs above its peak -> drawdown 0.
        drive_day(desk, portfolio, DATES[3], marks={'X': 100.0})
        drive_day(desk, portfolio, DATES[4])
        drive_day(desk, portfolio, DATES[5])  # day 5: scheduled realloc

        notes = [n for n in desk.notes if n.category == 'allocation']
        assert len(notes) == 1
        assert notes[0].data['recovered'] == ['alpha']
        assert notes[0].data['held'] == []
        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'active'
        # alpha's V-shaped NAV gives it a dominant raw score: it clamps
        # at weight_max and the flat pods split the rest equally.
        assert pods['alpha']['weight'] == pytest.approx(0.50)
        assert pods['beta']['weight'] == pytest.approx(0.25)
        assert pods['gamma']['weight'] == pytest.approx(0.25)

    def test_unrecovered_pod_stays_held_at_halved_weight(self):
        desk, portfolio = self._probation_at_minus_6pct()
        for date in DATES[3:6]:  # drawdown stays at -6%
            drive_day(desk, portfolio, date)

        notes = [n for n in desk.notes if n.category == 'allocation']
        assert len(notes) == 1
        assert notes[0].data['held'] == ['alpha']
        assert notes[0].data['recovered'] == []
        pods = desk.get_status()['pods']
        assert pods['alpha']['status'] == 'probation'
        assert pods['alpha']['weight'] == 0.25  # untouched, exactly
        # The flat pods split the remaining 0.75 equally.
        assert pods['beta']['weight'] == pytest.approx(0.375)
        assert pods['gamma']['weight'] == pytest.approx(0.375)

    def test_recovery_boundary_is_strict(self):
        # recovery_drawdown -0.03125 (= -1/32, binary exact). A pod
        # sitting EXACTLY at the threshold is NOT recovered ('above'
        # is strict); epsilon above it is.
        for nav, expected_status in ((0.96875, 'probation'),
                                     (0.969, 'active')):
            pods = [ScriptedPod('alpha'), ScriptedPod('beta'),
                    ScriptedPod('gamma')]
            desk = make_desk(pods, weights={'alpha': 0.25, 'beta': 0.375,
                                            'gamma': 0.375},
                             realloc_every_days=1, min_nav_days=1,
                             recovery_drawdown=-0.03125)
            portfolio = PortfolioManager(100_000.0)
            drive_day(desk, portfolio, DATES[0])
            # White-box: place alpha on probation at the engineered NAV.
            desk._statuses['alpha'] = 'probation'
            desk._navs['alpha'] = nav
            desk._nav_max['alpha'] = 1.0
            drive_day(desk, portfolio, DATES[1])  # day 1: realloc due
            assert desk.get_status()['pods']['alpha']['status'] \
                == expected_status


# ======================================================================
# (f) Cross-pod conflicts: first claim
# ======================================================================
class TestConflictFirstClaim:
    def test_higher_current_weight_wins(self):
        pods = [ScriptedPod('alpha', {DATES[0]: {'XYZ': 'BUY'}}),
                ScriptedPod('beta', {DATES[0]: {'XYZ': 'BUY'}}),
                ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 0.3, 'beta': 0.5,
                                        'gamma': 0.2})
        intents = drive_day(desk, PortfolioManager(100_000.0), DATES[0])
        assert [(i.action, i.asset.symbol) for i in intents] \
            == [('BUY', 'XYZ')]
        assert desk._pod_positions[stock('XYZ')]['pod'] == 'beta'

        conflicts = [n for n in desk.notes if 'Conflict' in n.message]
        assert len(conflicts) == 1
        assert conflicts[0].data['pod'] == 'alpha'  # the loser, per C9
        assert conflicts[0].data['winner'] == 'beta'

    def test_tie_breaks_alphabetically_by_pod_key(self):
        pods = [ScriptedPod('beta', {DATES[0]: {'XYZ': 'BUY'}}),
                ScriptedPod('alpha', {DATES[0]: {'XYZ': 'BUY'}}),
                ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 0.4, 'beta': 0.4,
                                        'gamma': 0.2})
        drive_day(desk, PortfolioManager(100_000.0), DATES[0])
        assert desk._pod_positions[stock('XYZ')]['pod'] == 'alpha'
        conflicts = [n for n in desk.notes if 'Conflict' in n.message]
        assert conflicts[0].data['pod'] == 'beta'

    def test_owned_symbol_is_off_limits_to_other_pods(self):
        pods = [ScriptedPod('alpha', {DATES[0]: {'XYZ': 'BUY'}}),
                ScriptedPod('beta', {DATES[2]: {'XYZ': 'BUY'}}),
                ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 0.5, 'beta': 0.3,
                                        'gamma': 0.2})
        portfolio = PortfolioManager(100_000.0)
        drive_day(desk, portfolio, DATES[0])
        fill_buy(portfolio, 'XYZ', 100, 100.0, DATES[1])
        drive_day(desk, portfolio, DATES[1])
        intents = drive_day(desk, portfolio, DATES[2])
        assert intents == []  # beta cannot enter alpha's symbol
        assert desk._pod_positions[stock('XYZ')]['pod'] == 'alpha'


class TestSizingFormula:
    @pytest.mark.parametrize('n_signals,expected_per_signal', [
        (1, 0.5 * 0.25),          # min(0.25, 1/max(4, 1)) = 0.25
        (2, 0.5 * 0.25),          # 1/max(4, 2) = 0.25
        (6, 0.5 * (1.0 / 6.0)),   # 1/max(4, 6) = 1/6
    ])
    def test_size_fraction_formula(self, n_signals, expected_per_signal):
        symbols = {f'S{i:02d}': 'BUY' for i in range(n_signals)}
        pods = [ScriptedPod('alpha', {DATES[0]: symbols}),
                ScriptedPod('beta'), ScriptedPod('gamma')]
        desk = make_desk(pods, weights={'alpha': 0.5, 'beta': 0.25,
                                        'gamma': 0.25})
        intents = drive_day(desk, PortfolioManager(100_000.0), DATES[0])
        assert len(intents) == n_signals
        for intent in intents:
            assert intent.size_fraction == pytest.approx(expected_per_signal)


# ======================================================================
# (a) + (g) End-to-end with the real engine: attribution identity + C8
# ======================================================================
class TestEndToEndWithEngine:
    N_DAYS = 40

    @pytest.fixture
    def universe(self):
        rng = np.random.default_rng(42)

        def frame(seed_offset, start_price):
            rets = np.random.default_rng(100 + seed_offset).normal(
                0.0005, 0.01, self.N_DAYS)
            close = start_price * np.cumprod(1.0 + rets)
            open_ = np.empty(self.N_DAYS)
            open_[0] = start_price
            open_[1:] = close[:-1]
            return pd.DataFrame({
                'open': open_,
                'high': np.maximum(open_, close) * 1.001,
                'low': np.minimum(open_, close) * 0.999,
                'close': close,
                'volume': rng.integers(400_000, 900_000,
                                       self.N_DAYS).astype(float),
            }, index=DATES[:self.N_DAYS])

        return {'AAA': frame(1, 100.0), 'BBB': frame(2, 50.0),
                'CCC': frame(3, 80.0)}

    @pytest.fixture
    def report_and_desk(self, universe, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        pods = [
            ScriptedPod('alpha', {DATES[1]: {'AAA': 'BUY'},
                                  DATES[12]: {'AAA': 'SELL'}}),
            ScriptedPod('beta', {DATES[3]: {'BBB': 'SHORT'},
                                 DATES[14]: {'BBB': 'COVER'}}),
            ScriptedPod('gamma', {DATES[5]: {'CCC': 'BUY'}}),
        ]
        # DEFAULT base RiskManager and CentralRiskBook: stops and limits
        # may fire — the attribution identity must hold regardless.
        desk = CitadelDesk(pods=pods)
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0,
                                commission=0.001)
        report = engine.run(sorted(universe), '2023-01-01', '2023-03-01',
                            benchmark_symbol=None)
        return report, desk, engine

    def test_attribution_identity_holds_daily_to_1e6(self, report_and_desk):
        report, desk, _ = report_and_desk
        assert len(report['portfolio_history']) \
            == len(report['pod_history']) == self.N_DAYS
        assert report['trades']  # the scripted pods actually traded
        assert_attribution_identity(report, desk.capital_allocation)

    def test_pod_history_has_c8_shape_daily(self, report_and_desk):
        report, _, _ = report_and_desk
        pod_history = report['pod_history']
        assert [entry['date'] for entry in pod_history] == \
            [d.strftime('%Y-%m-%d') for d in DATES[:self.N_DAYS]]
        for entry in pod_history:
            assert set(entry) == {'date', 'pods'}
            assert set(entry['pods']) == {'alpha', 'beta', 'gamma'}
            for pod in entry['pods'].values():
                assert_c8_pod_entry_shape(pod)
        assert_live_weight_conservation(pod_history)
        json.dumps(report['pod_history'])
        json.dumps(report['trader_notes'])

    def test_desk_report_and_notes_carry_c9_pod_tags(self, report_and_desk):
        report, desk, _ = report_and_desk
        assert report['desk'] == {'key': 'citadel', 'name': 'Citadel Desk'}
        signal_notes = [note for note in report['trader_notes']
                        if note['category'] == 'signal']
        assert signal_notes
        for note in signal_notes:
            assert note['data']['pod'] in {'alpha', 'beta', 'gamma'}
        assert set(desk.get_status()['pods']) == {'alpha', 'beta', 'gamma'}


# ======================================================================
# Real default-style pods through the engine: stat-arb fit tagging (C3+)
# ======================================================================
class TestRealPodsEndToEnd:
    """The real momentum/mean-reversion/stat-arb pods through the real
    engine (lighter GBM, same pattern as the Renaissance e2e test):
    walk_forward fits are tagged with the pod key 'stat_arb', the report
    completes with daily C8 pod_history, and the attribution identity
    holds on an organic (non-scripted) run too."""

    N_DAYS = 160

    def test_stat_arb_fits_tagged_and_identity_holds(self, monkeypatch):
        volume_rng = np.random.default_rng(21)
        index = pd.bdate_range('2022-06-01', periods=self.N_DAYS)
        frames = {}
        for i in range(6):
            rets = np.random.default_rng(300 + i).normal(0.0005, 0.012,
                                                         self.N_DAYS)
            close = (80.0 + 10 * i) * np.cumprod(1.0 + rets)
            frames[f'U{i}'] = pd.DataFrame({
                'open': close, 'high': close * 1.001, 'low': close * 0.999,
                'close': close,
                'volume': volume_rng.integers(
                    400_000, 900_000, self.N_DAYS).astype(float),
            }, index=index)

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        desk = CitadelDesk(pods=[
            MomentumPod(), MeanReversionPod(),
            StatArbPod(controller=WalkForwardController(
                GradientBoostingModel(n_estimators=10),
                train_window_days=252, refit_every_days=21,
                min_train_days=120)),
        ])
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        report = engine.run(sorted(frames), '2022-06-01', '2023-02-01',
                            benchmark_symbol=None)

        assert 'summary' in report
        assert len(report['pod_history']) == self.N_DAYS

        # C3+: the stat-arb pod's controller fits, tagged with its key.
        fits = report['walk_forward']
        assert fits  # 160 days clears the 120-day minimum
        assert {fit['model'] for fit in fits} == {'stat_arb'}
        for fit in fits:
            assert set(fit) == {'fit_date', 'train_start', 'train_end',
                                'n_samples', 'model'}
        json.dumps(report['walk_forward'])
        json.dumps(report['pod_history'])

        # The attribution identity holds on the organic run as well,
        # and live (active+probation) weight is conserved daily.
        assert_attribution_identity(report, desk.capital_allocation)
        assert_live_weight_conservation(report['pod_history'])


# ======================================================================
# (h) All pods cut: desk flat, final note, engine completes cleanly
# ======================================================================
class TestAllPodsCutEndToEnd:
    N_DAYS = 18
    CRASH_START = 10

    @pytest.fixture
    def crashing_universe(self):
        def frame():
            close = np.empty(self.N_DAYS)
            close[:self.CRASH_START] = 100.0
            for i in range(self.CRASH_START, self.N_DAYS):
                close[i] = close[i - 1] * 0.75  # -25% a day
            open_ = np.empty(self.N_DAYS)
            open_[0] = close[0]
            open_[1:] = close[:-1]
            return pd.DataFrame({
                'open': open_, 'high': np.maximum(open_, close),
                'low': np.minimum(open_, close), 'close': close,
                'volume': np.full(self.N_DAYS, 500_000.0),
            }, index=DATES[:self.N_DAYS])

        return {'AAA': frame(), 'BBB': frame()}

    @pytest.fixture
    def report_and_desk(self, crashing_universe, monkeypatch):
        def fake_fetch(self, symbol, start_date, end_date):
            return crashing_universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        pods = [ScriptedPod('alpha', {DATES[1]: {'AAA': 'BUY'}}),
                ScriptedPod('beta', {DATES[1]: {'BBB': 'BUY'}})]
        desk = make_desk(pods, weights={'alpha': 0.5, 'beta': 0.5})
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        report = engine.run(sorted(crashing_universe), '2023-01-01',
                            '2023-02-01', benchmark_symbol=None)
        return report, desk, engine

    def test_all_pods_cut_desk_flat_with_final_note(self, report_and_desk):
        report, desk, engine = report_and_desk

        # Both pods end the run cut with weight 0.
        final = report['pod_history'][-1]['pods']
        for key in ('alpha', 'beta'):
            assert final[key]['status'] == 'cut'
            assert final[key]['weight'] == 0.0
            assert final[key]['drawdown_pct'] <= -8.0

        # Individual cut notes carry C9 data; the desk-flat note is last.
        cut_notes = [n for n in desk.notes if 'CUT' in n.message]
        assert {n.data['pod'] for n in cut_notes} == {'alpha', 'beta'}
        all_cut = [n for n in desk.notes if 'All pods cut' in n.message]
        assert len(all_cut) == 1
        assert all_cut[0].data['pods'] == ['alpha', 'beta']

        # Positions were flattened the day AFTER each cut (next open).
        date_index = {d.strftime('%Y-%m-%d'): i
                      for i, d in enumerate(DATES[:self.N_DAYS])}
        symbol_for = {'alpha': 'AAA', 'beta': 'BBB'}
        for note in cut_notes:
            cut_day = date_index[note.timestamp.strftime('%Y-%m-%d')]
            sells = [t for t in report['trades']
                     if t['symbol'] == symbol_for[note.data['pod']]
                     and t['action'] == 'SELL']
            assert len(sells) == 1
            assert date_index[sells[0]['date'].strftime('%Y-%m-%d')] \
                == cut_day + 1

        # The desk ends flat and the engine completed a full report.
        assert engine.portfolio.positions == {}
        assert 'summary' in report
        assert len(report['pod_history']) == self.N_DAYS
        # Live weight conserved at 1.0 daily until the true all-cut day.
        assert_live_weight_conservation(report['pod_history'])
        for symbol in ('AAA', 'BBB'):
            actions = [t['action'] for t in report['trades']
                       if t['symbol'] == symbol]
            assert actions == ['BUY', 'SELL']

    def test_attribution_identity_holds_through_cut_and_flatten(
            self, report_and_desk):
        # THE residual boundary: each cut pod's flatten SELL fills at
        # the NEXT bar's open (default engine slippage), earning P&L on
        # a day its recorded weight is already 0. The identity must
        # reconstruct purely from the report via the additive C8
        # residual_capital field — first prove the boundary is actually
        # exercised (a zero-weight pod's NAV moves), then reconcile.
        report, desk, _ = report_and_desk
        pod_history = report['pod_history']
        residual_moves = [
            (t, key)
            for t in range(1, len(pod_history))
            for key, prev in pod_history[t - 1]['pods'].items()
            if prev['weight'] == 0.0
            and pod_history[t]['pods'][key]['nav'] != prev['nav']
        ]
        assert residual_moves  # cut-pod residual P&L actually occurred
        assert_attribution_identity(report, desk.capital_allocation)

    def test_cut_entries_carry_residual_capital(self, report_and_desk):
        # C8 (additive): from the cut day onward each cut pod's entry
        # publishes its residual attribution base; entries stay JSON-
        # serializable and shape-exact throughout.
        report, _, _ = report_and_desk
        for entry in report['pod_history']:
            for pod in entry['pods'].values():
                assert_c8_pod_entry_shape(pod)
        final = report['pod_history'][-1]['pods']
        for key in ('alpha', 'beta'):
            assert final[key]['status'] == 'cut'
            assert final[key]['residual_capital'] > 0.0
        json.dumps(report['pod_history'])


# ======================================================================
# Regression: other desks and strategy mode gain NO pod_history
# ======================================================================
class TestNoPodHistoryRegression:
    @pytest.fixture
    def quiet_universe(self, monkeypatch):
        rng = np.random.default_rng(11)
        n_days = 60
        index = pd.bdate_range('2023-01-02', periods=n_days)
        frames = {}
        for i, symbol in enumerate(('N1', 'N2')):
            close = 100.0 * np.cumprod(
                1.0 + rng.normal(0.0005, 0.01, n_days))
            frames[symbol] = pd.DataFrame({
                'open': close, 'high': close * 1.001, 'low': close * 0.999,
                'close': close,
                'volume': np.full(n_days, 500_000.0),
            }, index=index)

        def fake_fetch(self, symbol, start_date, end_date):
            return frames.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        return frames

    def test_strategy_mode_report_has_no_pod_history(self, quiet_universe):
        engine = BacktestEngine(strategy=MomentumStrategy(),
                                initial_capital=100_000.0)
        report = engine.run(['N1', 'N2'], '2023-01-01', '2023-04-01',
                            benchmark_symbol=None)
        assert 'pod_history' not in report

    def test_foundation_report_has_no_pod_history(self, quiet_universe):
        engine = BacktestEngine(desk=FoundationDesk(),
                                initial_capital=100_000.0)
        report = engine.run(['N1', 'N2'], '2023-01-01', '2023-04-01',
                            benchmark_symbol=None)
        assert 'pod_history' not in report

    def test_renaissance_report_has_no_pod_history(self, quiet_universe):
        engine = BacktestEngine(desk=RenaissanceDesk(),
                                initial_capital=100_000.0)
        report = engine.run(['N1', 'N2'], '2023-01-01', '2023-04-01',
                            benchmark_symbol=None)
        assert 'pod_history' not in report
