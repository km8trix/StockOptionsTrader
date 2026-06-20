"""QA tests for the gated RL execution throttle (Phase F unit 2).

Every test here maps to ONE numbered safety-contract item from the spec; the
class/method docstrings name the contract item so a failure points straight at
the violated invariant. The headline property is SUBTRACTIVE-ONLY: the throttle
can only ever shrink or leave a desk-intended size, never grow it.

The suite is offline and deterministic — no network, no live path. torch is the
optional dependency that backs ThrottlePolicy/ThrottlePolicyTrainer; those tests
``importorskip('torch')`` so a torch-less install still runs the OFF-path and
identity tests (and the explicit "torch-absent => ImportError" contract test).

Reuses the fund-orchestrator test helpers (ScriptedDesk, intent, stock, call,
flat_frame, wide_risk) so the wiring tests drive the real orchestrator.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import List

import numpy as np
import pandas as pd
import pytest

from core.models import Asset, AssetType
from desks.base import DeskIntent
from portfolio.manager import PortfolioManager

import desks.rl_execution as rl
from desks.rl_execution import (
    FeatureScaler,
    N_THROTTLE_FEATURES,
    RLExecutionThrottle,
    ThrottleDecision,
    extract_throttle_features,
    validate_throttle_oos,
)

from tests.test_fund_orchestrator import (
    call,
    flat_frame,
    intent,
    stock,
    wide_risk,
)

DATE = pd.Timestamp('2022-01-10')


# ----------------------------------------------------------------------
# Local helpers
# ----------------------------------------------------------------------
def _portfolio_with_history(values, start='2021-12-01'):
    """A PortfolioManager carrying a scripted equity history (<= DATE)."""
    pm = PortfolioManager(float(values[0]))
    idx = pd.bdate_range(start, periods=len(values))
    pm.portfolio_history = [
        {'timestamp': ts, 'portfolio_value': float(v)}
        for ts, v in zip(idx, values)
    ]
    return pm


def _all_data(symbols, n=40, price=100.0, vol=0.0):
    """Per-symbol OHLC frames indexed <= DATE with optional return noise."""
    out = {}
    idx = pd.bdate_range('2021-11-15', periods=n)
    rng = np.random.default_rng(0)
    for s in symbols:
        if vol > 0:
            steps = 1.0 + rng.normal(0, vol, size=n)
            closes = price * np.cumprod(steps)
        else:
            closes = np.full(n, price)
        out[s] = pd.DataFrame(
            {'open': closes, 'high': closes * 1.001,
             'low': closes * 0.999, 'close': closes,
             'volume': np.full(n, 1e6)}, index=idx)
    return out


class _IdentityPolicy:
    """A stand-in policy that always returns a fixed throttle, so the
    RLExecutionThrottle logic is testable WITHOUT torch."""

    def __init__(self, value: float, scale_min=0.5, scale_max=1.0):
        self.value = float(value)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def predict_throttle(self, features):
        return self.value


# ======================================================================
# Contract 1 — SUBTRACTIVE-ONLY (constructor clamps + fuzz)
# ======================================================================
class TestSubtractiveOnly:
    """Contract #1: s in [scale_min, scale_max<=1.0]; every emitted size <=
    original and in (0,1]; total emitted gross <= input gross; constructor
    clamps scale_max>1 to 1.0 and raises on degenerate bounds."""

    def test_constructor_clamps_scale_max_above_one(self):
        thr = RLExecutionThrottle(policy=_IdentityPolicy(1.0), scale_max=3.5)
        assert thr.scale_max == 1.0

    def test_constructor_raises_on_nonpositive_scale_min(self):
        with pytest.raises(ValueError):
            RLExecutionThrottle(scale_min=0.0)
        with pytest.raises(ValueError):
            RLExecutionThrottle(scale_min=-0.2)

    def test_constructor_raises_when_min_exceeds_max(self):
        with pytest.raises(ValueError):
            RLExecutionThrottle(scale_min=0.9, scale_max=0.5)

    def test_constructor_clamp_then_min_exceeds_clamped_max_raises(self):
        # scale_max=5 clamps to 1.0; scale_min=1.2 > 1.0 must then raise.
        with pytest.raises(ValueError):
            RLExecutionThrottle(scale_min=1.2, scale_max=5.0)

    def test_eps_must_be_positive(self):
        with pytest.raises(ValueError):
            RLExecutionThrottle(policy=_IdentityPolicy(1.0), eps=0.0)

    def test_fuzz_subtractive_only_many_vectors_and_configs(self):
        """Across many random configs + opening-stock intents, every emitted
        size <= its original, in (0,1], and total emitted gross <= input."""
        rng = random.Random(20240620)
        for _ in range(400):
            scale_min = rng.uniform(0.01, 0.99)
            # scale_max drawn possibly >1 to also exercise the hard clamp.
            scale_max_raw = rng.uniform(scale_min, 1.5)
            # Random throttle the policy will return (clamped inside apply).
            policy_val = rng.uniform(0.0, 2.0)
            thr = RLExecutionThrottle(
                policy=_IdentityPolicy(policy_val),
                scale_min=scale_min, scale_max=scale_max_raw)
            assert thr.scale_max <= 1.0

            n = rng.randint(1, 6)
            intents: List[DeskIntent] = []
            for i in range(n):
                size = rng.uniform(1e-3, 1.0)
                action = rng.choice(['BUY', 'SHORT'])
                intents.append(intent(stock(f'S{i}'), action, size))
            original_sizes = {it.asset: it.size_fraction for it in intents}
            total_in = sum(it.size_fraction for it in intents)

            pm = _portfolio_with_history([100.0, 101.0, 100.5])
            data = _all_data([f'S{i}' for i in range(n)])
            emitted = thr.apply(intents, pm, data, DATE)

            total_out = 0.0
            for it in emitted:
                assert 0.0 < it.size_fraction <= 1.0
                assert it.size_fraction <= original_sizes[it.asset] + 1e-12
                total_out += it.size_fraction
            assert total_out <= total_in + 1e-9

    def test_post_condition_raises_on_a_size_increasing_policy(self):
        """A buggy policy that returns >1 must be caught by the belt-and-
        suspenders clamp (so the post-condition never trips); confirm the
        emitted size still never exceeds the original."""
        thr = RLExecutionThrottle(policy=_IdentityPolicy(5.0),
                                  scale_min=0.5, scale_max=1.0)
        it = intent(stock('AAA'), 'BUY', 0.4)
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([it], pm, _all_data(['AAA']), DATE)
        assert out[0].size_fraction <= 0.4 + 1e-12


# ======================================================================
# Contract 2 — DIRECTION NEVER CHANGES
# ======================================================================
class TestDirectionUntouched:
    """Contract #2: action identical on every emitted intent; only
    size_fraction may change."""

    def test_action_preserved_on_throttled_opens(self):
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.6),
                                  scale_min=0.5, scale_max=1.0)
        intents = [intent(stock('AAA'), 'BUY', 0.4),
                   intent(stock('BBB'), 'SHORT', 0.4)]
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply(intents, pm, _all_data(['AAA', 'BBB']), DATE)
        by_symbol = {i.asset.symbol: i for i in out}
        assert by_symbol['AAA'].action == 'BUY'
        assert by_symbol['BBB'].action == 'SHORT'
        # Asset, reason, quantity preserved too (only size_fraction differs).
        assert by_symbol['AAA'].reason == 'test'
        assert by_symbol['AAA'].size_fraction < 0.4


# ======================================================================
# Contract 3 — CLOSING INTENTS PASS THROUGH BYTE-IDENTICAL
# ======================================================================
class TestClosingPassThrough:
    """Contract #3: SELL/COVER pass through unchanged (same object)."""

    @pytest.mark.parametrize('action', ['SELL', 'COVER'])
    def test_close_is_same_object(self, action):
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.5))
        it = intent(stock('AAA'), action, 0.4)
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([it], pm, _all_data(['AAA']), DATE)
        assert len(out) == 1
        assert out[0] is it  # byte-identical: untouched object passes through
        # And nothing was logged for a pass-through close.
        assert thr.decision_log == []


# ======================================================================
# Contract 4 — OPTION / ABSOLUTE-QUANTITY INTENTS PASS THROUGH
# ======================================================================
class TestOptionPassThrough:
    """Contract #4: option intents (CALL/PUT) or any intent carrying an
    absolute quantity pass through byte-identical."""

    def test_call_option_is_same_object(self):
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.5))
        it = intent(call('AAA', 100.0), 'BUY', 0.4)
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([it], pm, _all_data(['AAA']), DATE)
        assert out[0] is it
        assert thr.decision_log == []

    def test_put_option_is_same_object(self):
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.5))
        put = Asset(symbol='AAA', asset_type=AssetType.PUT,
                    strike_price=90.0, expiration_date='2022-06-17')
        it = intent(put, 'SHORT', 0.4)
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([it], pm, _all_data(['AAA']), DATE)
        assert out[0] is it

    def test_stock_with_absolute_quantity_is_same_object(self):
        """A plain-stock OPENING intent that nonetheless carries an absolute
        quantity (defined-risk leg) passes through unchanged."""
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.5))
        it = intent(stock('AAA'), 'BUY', 0.4, quantity=10)
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([it], pm, _all_data(['AAA']), DATE)
        assert out[0] is it
        assert thr.decision_log == []


# ======================================================================
# Contract 5 — DROP-ON-TINY
# ======================================================================
class TestDropOnTiny:
    """Contract #5: an intent scaled to <= eps is DROPPED (recorded), never
    emitted as an invalid DeskIntent."""

    def test_tiny_scaled_intent_is_dropped_and_logged(self):
        # scale_min tiny so the product falls to <= eps; eps generous.
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.001),
                                  scale_min=0.001, scale_max=1.0, eps=0.01)
        it = intent(stock('AAA'), 'BUY', 0.005)  # 0.005 * 0.001 = 5e-6 <= eps
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([it], pm, _all_data(['AAA']), DATE)
        assert out == []  # dropped, not emitted invalid
        assert len(thr.decision_log) == 1
        assert thr.decision_log[0].dropped is True
        assert thr.decision_log[0].symbol == 'AAA'

    def test_dropped_intent_never_violates_post_condition(self):
        # Mix a droppable intent with a survivor; post-condition still holds.
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.5),
                                  scale_min=0.5, scale_max=1.0, eps=0.01)
        tiny = intent(stock('TINY'), 'BUY', 0.015)   # 0.015*0.5=0.0075 <= eps
        big = intent(stock('BIG'), 'BUY', 0.4)       # survives
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply([tiny, big], pm, _all_data(['TINY', 'BIG']), DATE)
        syms = {i.asset.symbol for i in out}
        assert syms == {'BIG'}


# ======================================================================
# Contract 6 — OFF-BY-DEFAULT BYTE-IDENTICAL
# ======================================================================
class TestOffByDefaultIdentity:
    """Contract #6: with no modulator (or a None/disabled one) the path is
    byte-identical to today."""

    def test_disabled_throttle_returns_same_list_object(self):
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.5), enabled=False)
        intents = [intent(stock('AAA'), 'BUY', 0.4)]
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply(intents, pm, _all_data(['AAA']), DATE)
        assert out is intents  # identity fast-path: SAME list object

    def test_none_policy_returns_same_list_object(self):
        thr = RLExecutionThrottle(policy=None)
        intents = [intent(stock('AAA'), 'BUY', 0.4)]
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply(intents, pm, _all_data(['AAA']), DATE)
        assert out is intents

    def test_orchestrator_step_byte_identical_with_none_modulator(
            self, monkeypatch):
        """A full fund backtest with sizing_modulator=None produces a report
        byte-identical to the same run with no modulator wired at all."""
        from backtesting.backtest_engine import BacktestEngine
        from data.market_data import MarketDataHandler
        from desks.orchestrator import FundOrchestrator
        from tests.test_fund_orchestrator import ScriptedDesk

        universe = {'AAA': flat_frame(n=30), 'BBB': flat_frame(n=30)}

        def fake_fetch(self, symbol, start, end):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)
        dates = universe['AAA'].index
        sig = dates[1]

        def build_report(modulator_kwargs):
            a = ScriptedDesk('asim', 0.5,
                             per_date={sig: [intent(stock('AAA'), 'BUY', 0.05)]},
                             risk_manager=wide_risk())
            b = ScriptedDesk('bsim', 0.5,
                             per_date={sig: [intent(stock('BBB'), 'BUY', 0.05)]},
                             risk_manager=wide_risk())
            orch = FundOrchestrator([a, b], **modulator_kwargs)
            engine = BacktestEngine(orchestrator=orch,
                                    initial_capital=100_000.0)
            return engine.run(['AAA', 'BBB'], '2022-01-01', '2022-12-31',
                              benchmark_symbol=None)

        baseline = build_report({})  # no modulator wired at all
        with_none = build_report({'sizing_modulator': None})  # explicit None

        # Equity curves byte-identical.
        base_curve = [(s['timestamp'], s['portfolio_value'])
                      for s in baseline['portfolio_history']]
        none_curve = [(s['timestamp'], s['portfolio_value'])
                      for s in with_none['portfolio_history']]
        assert base_curve == none_curve
        # Headline summary stats byte-identical.
        assert baseline['summary'] == with_none['summary']

    def test_reweighting_fund_default_modulator_is_byte_identical(
            self, monkeypatch):
        """ReweightingFundBacktest with sizing_modulator=None matches a run
        with no modulator param (default) — the additive ctor param does not
        perturb the default path. Uses injected providers (no market data)."""
        from backtesting.reweighting_fund import ReweightingFundBacktest
        from desks.orchestrator import FundOrchestrator
        from data.market_data import MarketDataHandler
        from tests.test_fund_orchestrator import ScriptedDesk

        universe = {'AAA': flat_frame(n=30)}

        def fake_fetch(self, symbol, start, end):
            return universe.get(symbol, pd.DataFrame())
        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data', fake_fetch)

        dates = universe['AAA'].index
        sig = dates[1]

        def orch_factory(allocations):
            desks = [ScriptedDesk(
                k, v, per_date={sig: [intent(stock('AAA'), 'BUY', 0.05)]},
                risk_manager=wide_risk()) for k, v in allocations.items()]
            return FundOrchestrator(desks)

        def solo_provider(key, symbols, start, end):
            return [(d, 100_000.0) for d in dates]

        def run(extra):
            fb = ReweightingFundBacktest(
                {'asim': 1.0}, initial_capital=100_000.0,
                solo_curve_provider=solo_provider,
                orchestrator_factory=orch_factory, **extra)
            return fb.run(['AAA'], '2022-01-01', '2022-12-31',
                          benchmark_symbol=None)

        baseline = run({})
        with_none = run({'sizing_modulator': None})
        base_curve = [(s['timestamp'], s['portfolio_value'])
                      for s in baseline['portfolio_history']]
        none_curve = [(s['timestamp'], s['portfolio_value'])
                      for s in with_none['portfolio_history']]
        assert base_curve == none_curve
        assert baseline['summary'] == with_none['summary']


# ======================================================================
# Contract 7 — NEVER LIVE
# ======================================================================
class TestNeverLive:
    """Contract #7: the module imports no live-session / order-routing path."""

    def test_no_live_or_order_routing_imports(self):
        import inspect
        src = inspect.getsource(rl)
        forbidden = ['etrade', 'e_trade', 'live_session', 'order_routing',
                     'place_order', 'submit_order', 'broker', 'OAuth',
                     'requests.post', 'live_trading']
        low = src.lower()
        for token in forbidden:
            assert token.lower() not in low, (
                f"rl_execution must not reference '{token}' (never-live)")

    def test_module_imports_are_backtest_only(self):
        """The module's actual imported modules contain nothing live."""
        import sys
        # Confirm no e*trade / live module got pulled in by importing rl.
        live_markers = ['etrade', 'live_session', 'order_router']
        for name in list(sys.modules):
            low = name.lower()
            if any(m in low for m in live_markers):
                # If such a module exists at all, rl_execution must not be the
                # thing that imported it: assert it's not in rl's source.
                import inspect
                assert name.split('.')[-1].lower() not in \
                    inspect.getsource(rl).lower()


# ======================================================================
# Contract 8 — DETERMINISM (+ torch-absent ImportError)
# ======================================================================
class TestDeterminism:
    """Contract #8: same seed + same inputs => byte-identical throttles across
    two constructions. torch-absent => ThrottlePolicy raises ImportError while
    the OFF path still works."""

    def test_same_seed_byte_identical_throttles(self):
        torch = pytest.importorskip('torch')
        from desks.rl_execution import ThrottlePolicy
        feats = [np.array([0.4, 0.1, 0.02, 0.03, 1.0]),
                 np.array([0.2, 0.3, 0.05, 0.01, 1.0]),
                 np.array([0.8, 0.0, 0.0, 0.0, 1.0])]
        p1 = ThrottlePolicy(random_state=123)
        p2 = ThrottlePolicy(random_state=123)
        out1 = [p1.predict_throttle(f) for f in feats]
        out2 = [p2.predict_throttle(f) for f in feats]
        assert out1 == out2  # exact equality, not approx
        # Stronger: the actual weights (init AND the no-op-prior overwrite, both
        # now pinned inside _deterministic_torch) are byte-identical — catches a
        # regression where prior init drifts outside the deterministic context.
        sd1, sd2 = p1._net.state_dict(), p2._net.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            assert torch.equal(sd1[key], sd2[key]), f"weight drift in {key}"

    def test_untrained_policy_respects_noop_prior(self):
        """No-op prior: an untrained policy outputs ~scale_max (near 1.0)."""
        pytest.importorskip('torch')
        from desks.rl_execution import ThrottlePolicy
        p = ThrottlePolicy(scale_min=0.5, scale_max=1.0)
        s = p.predict_throttle(np.array([0.4, 0.1, 0.02, 0.03, 1.0]))
        assert 0.5 <= s <= 1.0
        assert s > 0.9  # the no-op prior keeps it near scale_max

    def test_predict_throttle_in_interval(self):
        pytest.importorskip('torch')
        from desks.rl_execution import ThrottlePolicy
        p = ThrottlePolicy(scale_min=0.3, scale_max=0.8)
        for _ in range(20):
            f = np.random.default_rng().normal(size=N_THROTTLE_FEATURES)
            s = p.predict_throttle(f)
            assert 0.3 <= s <= 0.8

    def test_off_path_works_without_a_policy_regardless_of_torch(self):
        """Even if torch is missing, the identity (policy=None) path runs."""
        thr = RLExecutionThrottle(policy=None)
        intents = [intent(stock('AAA'), 'BUY', 0.4)]
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply(intents, pm, _all_data(['AAA']), DATE)
        assert out is intents

    def test_torch_absent_raises_importerror(self, monkeypatch):
        """When torch is None, constructing a ThrottlePolicy raises ImportError
        (mirrors desks.models.neural)."""
        from desks.rl_execution import ThrottlePolicy
        monkeypatch.setattr(rl, 'torch', None)
        with pytest.raises(ImportError):
            ThrottlePolicy()

    def test_trainer_determinism_same_seed(self):
        """Same seed + same train decisions => byte-identical fitted throttles."""
        pytest.importorskip('torch')
        from desks.rl_execution import ThrottlePolicyTrainer
        rng = np.random.default_rng(7)
        decisions = [
            ThrottleDecision(features=rng.normal(size=N_THROTTLE_FEATURES),
                             forward_reward=float(rng.normal()))
            for _ in range(20)]
        d2 = [ThrottleDecision(features=d.features.copy(),
                               forward_reward=d.forward_reward)
              for d in decisions]
        t1 = ThrottlePolicyTrainer(random_state=99, epochs=20)
        t2 = ThrottlePolicyTrainer(random_state=99, epochs=20)
        p1 = t1.fit(decisions)
        p2 = t2.fit(d2)
        feats = [d.features for d in decisions]
        assert [p1.predict_throttle(f) for f in feats] == \
               [p2.predict_throttle(f) for f in feats]

    def test_trainer_no_decisions_returns_noop_policy(self):
        pytest.importorskip('torch')
        from desks.rl_execution import ThrottlePolicyTrainer
        p = ThrottlePolicyTrainer().fit([])
        s = p.predict_throttle(np.array([0.4, 0.1, 0.02, 0.03, 1.0]))
        assert s > 0.9  # no learnable signal => stays near scale_max


# ======================================================================
# Contract 9 — LEAKAGE-FREE
# ======================================================================
class TestLeakageFree:
    """Contract #9: rows dated AFTER the decision date cannot change a feature."""

    def test_future_price_rows_do_not_change_features(self):
        it = intent(stock('AAA'), 'BUY', 0.4)
        pm = _portfolio_with_history([100.0, 101.0, 99.0, 102.0])

        # Frame extends well past DATE, with wild future moves.
        idx = pd.bdate_range('2021-11-15', periods=60)
        rng = np.random.default_rng(1)
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.03, size=60))
        full = pd.DataFrame(
            {'open': closes, 'high': closes * 1.01, 'low': closes * 0.99,
             'close': closes, 'volume': np.full(60, 1e6)}, index=idx)
        truncated = full.loc[full.index <= DATE].copy()

        feats_full = extract_throttle_features(it, pm, {'AAA': full}, DATE)
        feats_trunc = extract_throttle_features(
            it, pm, {'AAA': truncated}, DATE)
        np.testing.assert_array_equal(feats_full, feats_trunc)

    def test_future_equity_snapshots_do_not_change_features(self):
        it = intent(stock('AAA'), 'BUY', 0.4)
        # Build history that straddles DATE; future snapshots are huge.
        idx = pd.bdate_range('2021-12-20', periods=40)
        pm_full = PortfolioManager(100.0)
        pm_full.portfolio_history = [
            {'timestamp': ts,
             'portfolio_value': (100.0 if ts <= DATE else 1e9)}
            for ts in idx]
        pm_trunc = PortfolioManager(100.0)
        pm_trunc.portfolio_history = [
            h for h in pm_full.portfolio_history
            if h['timestamp'] <= DATE]

        data = _all_data(['AAA'])
        feats_full = extract_throttle_features(it, pm_full, data, DATE)
        feats_trunc = extract_throttle_features(it, pm_trunc, data, DATE)
        np.testing.assert_array_equal(feats_full, feats_trunc)

    def test_feature_vector_has_fixed_length(self):
        it = intent(stock('AAA'), 'BUY', 0.4)
        pm = _portfolio_with_history([100.0, 101.0])
        feats = extract_throttle_features(it, pm, _all_data(['AAA']), DATE)
        assert feats.shape == (N_THROTTLE_FEATURES,)

    def test_scaler_applied_when_supplied(self):
        it = intent(stock('AAA'), 'BUY', 0.4)
        pm = _portfolio_with_history([100.0, 101.0])
        data = _all_data(['AAA'])
        raw = extract_throttle_features(it, pm, data, DATE)
        scaler = FeatureScaler().fit(np.vstack([raw, raw * 1.5 + 0.1]))
        scaled = extract_throttle_features(it, pm, data, DATE, scalers=scaler)
        np.testing.assert_array_almost_equal(scaled, scaler.transform(raw))


# ======================================================================
# Contract 10 — GATE (validate_throttle_oos, no side effects)
# ======================================================================
class TestValidationGate:
    """Contract #10: the gate returns a DSR / OOS-fold verdict + passed bool;
    passing does NOT mutate or enable any orchestrator."""

    def test_returns_verdict_dict(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 0.01, size=100)
        thr = base + rng.normal(0, 0.001, size=100)
        verdict = validate_throttle_oos(base.tolist(), thr.tolist(), n_trials=5)
        for key in ('deflated_sharpe_ratio', 'probabilistic_sharpe_ratio',
                    'passed', 'enables_agent'):
            assert key in verdict
        assert isinstance(verdict['passed'], bool)
        # The gate NEVER claims to enable the agent.
        assert verdict['enables_agent'] is False

    def test_strong_incremental_edge_passes_with_folds(self):
        """A clearly-positive incremental series with a significant OOS fold
        should pass the gate. Note the edge MUST carry variance: a Sharpe ratio
        is undefined on a zero-variance (constant) series, so DSR correctly
        returns None and the gate fails — we therefore use a noisy positive
        edge whose mean dominates its std."""
        rng = np.random.default_rng(5)
        n = 240
        incremental = 0.01 + rng.normal(0, 0.002, size=n)  # strong, finite vol
        base = np.zeros(n)
        thr = base + incremental
        # Folds are sliced from the incremental series INTERNALLY — the caller
        # passes only a fold COUNT, never the (potentially mis-built) fold data.
        verdict = validate_throttle_oos(
            base.tolist(), thr.tolist(), n_trials=1, n_folds=3)
        assert verdict['n_folds_used'] == 3
        assert verdict['passed'] is True
        assert verdict['benjamini_hochberg'].get('n_significant_bh', 0) > 0

    def test_zero_edge_does_not_pass(self):
        n = 120
        base = np.linspace(0.0, 0.001, n)
        # throttled == baseline => incremental is all-zero => no edge.
        verdict = validate_throttle_oos(base.tolist(), base.tolist(),
                                        n_trials=10)
        assert verdict['passed'] is False

    def test_gate_has_no_side_effects_on_orchestrator(self):
        """Running the gate must not enable/mutate any throttle/orchestrator."""
        from desks.orchestrator import FundOrchestrator
        from tests.test_fund_orchestrator import ScriptedDesk

        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.6), enabled=True)
        orch = FundOrchestrator([ScriptedDesk('a', 1.0)],
                                sizing_modulator=thr)
        # Snapshot the relevant state.
        before_enabled = thr.enabled
        before_log_len = len(thr.decision_log)
        before_mod = orch.sizing_modulator

        n = 240
        incremental = np.full(n, 0.01)
        validate_throttle_oos(np.zeros(n).tolist(),
                              incremental.tolist(), n_trials=1,
                              n_folds=2)

        assert thr.enabled == before_enabled
        assert len(thr.decision_log) == before_log_len
        assert orch.sizing_modulator is before_mod


# ======================================================================
# Wiring — registry factory builds a detached, off-by-default throttle
# ======================================================================
class TestRegistryWiring:
    def test_factory_returns_detached_throttle(self):
        from desks.registry import create_rl_execution_throttle
        thr = create_rl_execution_throttle(policy=None, scale_min=0.4,
                                           scale_max=2.0)
        assert isinstance(thr, RLExecutionThrottle)
        assert thr.scale_max == 1.0  # hard-clamped
        assert thr.scale_min == 0.4
        # Detached: identity until explicitly wired AND given a policy.
        intents = [intent(stock('AAA'), 'BUY', 0.4)]
        pm = _portfolio_with_history([100.0, 100.0])
        out = thr.apply(intents, pm, _all_data(['AAA']), DATE)
        assert out is intents  # policy=None => identity

    def test_create_fund_orchestrator_accepts_modulator(self, monkeypatch):
        """The registry orchestrator factory threads sizing_modulator through."""
        import desks.registry as registry
        from tests.test_fund_orchestrator import ScriptedDesk

        monkeypatch.setattr(
            registry, 'create_desk',
            lambda key, alloc=1.0: ScriptedDesk(key, alloc,
                                                risk_manager=wide_risk()))
        thr = RLExecutionThrottle(policy=_IdentityPolicy(0.6))
        orch = registry.create_fund_orchestrator(
            {'a': 1.0}, sizing_modulator=thr)
        assert orch.sizing_modulator is thr
