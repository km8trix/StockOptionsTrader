"""Tests for desks.pods — the Citadel desk's strategy pods.

Each pod's signal stream is pinned on crafted, deterministic frames
(enriched with the same indicator pipeline the engine uses): the
momentum cross fires BUY, Bollinger/RSI reversion fires BUY/SELL, the
stat-arb pod emits nothing before its first fit and longs the top /
shorts the bottom quintile after it. Pod isolation (no shared mutable
state across instances) is asserted explicitly. Offline, deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.market_data import MarketDataHandler
from desks.pods import MeanReversionPod, MomentumPod, StatArbPod
from desks.walk_forward import WalkForwardController, WalkForwardModel


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Indicator-enrich a raw OHLCV frame exactly like the engine does."""
    return MarketDataHandler().calculate_indicators(frame.copy())


def make_frame(returns, start='2023-01-02', volume=500_000.0,
               start_price=100.0) -> pd.DataFrame:
    rets = np.asarray(returns, dtype=float)
    close = start_price * np.cumprod(1.0 + rets)
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': np.full(len(close), volume),
    }, index=index)


def momentum_frame() -> pd.DataFrame:
    """Uptrend, dip, renewed rally: the dip pulls MACD below its signal
    line and the recovery crosses it back up while the close sits above
    the still-rising 50-day SMA (entry); the late rally drives RSI
    overbought (exit)."""
    return enrich(make_frame([0.004] * 40 + [-0.005] * 8 + [0.008] * 22))


class StubScoreModel(WalkForwardModel):
    """Fixed per-symbol scores (restricted to symbols present in data)."""

    def __init__(self, scores=None):
        self.scores = scores or {}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        return {symbol: score for symbol, score in self.scores.items()
                if symbol in data}


class TestMomentumPod:
    def _signals_by_day(self, pod, frame, start=50):
        """Drive the pod on expanding slices (all indicators are causal,
        so slicing the once-enriched frame matches engine enrichment)."""
        out = {}
        for i in range(start, len(frame)):
            date = frame.index[i]
            out[date] = pod.generate_signals({'MOM': frame.iloc[:i + 1]},
                                             date)
        return out

    def _expected_entry_days(self, frame, start=50):
        """Independent restatement of the entry rule: MACD cross up with
        the close above the 50-day SMA (volume gate passes at mult=1.0
        because constant volume gives ratio exactly 1.0)."""
        days = []
        for i in range(start, len(frame)):
            current, prev = frame.iloc[i], frame.iloc[i - 1]
            if (current['macd'] > current['signal']
                    and prev['macd'] <= prev['signal']
                    and current['close'] > current['sma_50']):
                days.append(frame.index[i])
        return days

    def test_macd_cross_fires_buy(self):
        frame = momentum_frame()
        expected = self._expected_entry_days(frame)
        assert expected  # the fixture must actually produce a cross
        out = self._signals_by_day(MomentumPod(), frame)
        for date in expected:
            assert out[date].get('MOM') == 'BUY'

    def test_volume_gate_blocks_unconfirmed_entries(self):
        # Constant volume -> ratio exactly 1.0 < 2.0: every BUY is gated.
        frame = momentum_frame()
        assert self._expected_entry_days(frame)  # strategy would buy
        out = self._signals_by_day(
            MomentumPod(volume_confirmation_mult=2.0), frame)
        assert all(signals.get('MOM') != 'BUY' for signals in out.values())

    def test_sell_fires_on_cross_down_or_overbought(self):
        frame = momentum_frame()
        out = self._signals_by_day(MomentumPod(), frame)
        sell_days = [date for date, signals in out.items()
                     if signals.get('MOM') == 'SELL']
        assert sell_days  # the late rally drives RSI overbought
        for date in sell_days:
            current = frame.loc[date]
            prev = frame.iloc[frame.index.get_loc(date) - 1]
            cross_down = (current['macd'] < current['signal']
                          and prev['macd'] >= prev['signal'])
            assert cross_down or current['rsi'] > 70

    def test_long_only_actions(self):
        out = self._signals_by_day(MomentumPod(), momentum_frame())
        actions = {action for signals in out.values()
                   for action in signals.values()}
        assert actions <= {'BUY', 'SELL'}

    def test_no_signal_without_todays_bar(self):
        frame = momentum_frame()
        beyond = frame.index[-1] + pd.tseries.offsets.BDay(1)
        assert MomentumPod().generate_signals({'MOM': frame}, beyond) == {}

    def test_no_signal_during_warmup(self):
        frame = momentum_frame().iloc[:30]
        assert MomentumPod().generate_signals(
            {'MOM': frame}, frame.index[-1]) == {}


class TestMeanReversionPod:
    def _jitter_then(self, tail_returns):
        jitter = [0.0005 if i % 2 == 0 else -0.0005 for i in range(60)]
        return enrich(make_frame(jitter + tail_returns))

    def test_lower_band_oversold_fires_buy(self):
        frame = self._jitter_then([-0.05, -0.05, -0.05])
        current = frame.iloc[-1]
        # Self-validating fixture: the documented conditions really hold.
        assert current['close'] <= current['bb_lower']
        assert current['rsi'] < 30
        signals = MeanReversionPod().generate_signals(
            {'MRV': frame}, frame.index[-1])
        assert signals == {'MRV': 'BUY'}

    def test_upper_band_overbought_fires_sell(self):
        frame = self._jitter_then([0.05, 0.05, 0.05])
        current = frame.iloc[-1]
        assert current['close'] >= current['bb_upper']
        assert current['rsi'] > 70
        signals = MeanReversionPod().generate_signals(
            {'MRV': frame}, frame.index[-1])
        assert signals == {'MRV': 'SELL'}

    def test_no_signal_mid_band(self):
        frame = self._jitter_then([0.0005, -0.0005])
        signals = MeanReversionPod().generate_signals(
            {'MRV': frame}, frame.index[-1])
        assert signals == {}

    def test_no_signal_without_todays_bar(self):
        frame = self._jitter_then([-0.05, -0.05, -0.05])
        beyond = frame.index[-1] + pd.tseries.offsets.BDay(1)
        assert MeanReversionPod().generate_signals({'MRV': frame},
                                                   beyond) == {}


class TestStatArbPod:
    def _universe(self, n_symbols=10, n_days=15):
        rng = np.random.default_rng(7)
        return {f'S{i:02d}': make_frame(rng.normal(0.0, 0.005, n_days))
                for i in range(n_symbols)}

    def _scores(self, n_symbols=10, reverse=False):
        # S00 highest .. S09 lowest (or reversed), strictly monotone.
        scores = {f'S{i:02d}': 0.9 - 0.2 * i for i in range(n_symbols)}
        if reverse:
            scores = {symbol: -score for symbol, score in scores.items()}
        return scores

    def _pod(self, scores, refit_every_days=99):
        return StatArbPod(controller=WalkForwardController(
            StubScoreModel(scores), min_train_days=1,
            refit_every_days=refit_every_days))

    @staticmethod
    def _through(frames, date):
        """Slice frames to index <= date — the engine's data contract."""
        return {symbol: frame[frame.index <= date]
                for symbol, frame in frames.items()}

    def test_pre_fit_emits_nothing(self):
        # Default controller: min_train_days=120 >> the 15-day history,
        # so the model never fits and the pod stays silent.
        pod = StatArbPod()
        frames = self._universe()
        monday = frames['S00'].index[0]
        assert monday.weekday() == 0
        assert pod.generate_signals(self._through(frames, monday),
                                    monday) == {}
        assert pod.controller.is_fitted is False

    def test_post_fit_longs_top_shorts_bottom_quintile(self):
        frames = self._universe()
        monday = frames['S00'].index[0]
        pod = self._pod(self._scores())
        signals = pod.generate_signals(self._through(frames, monday), monday)
        # n=10 -> top/bottom max(1, 10//5) = 2 per side.
        assert signals == {'S00': 'BUY', 'S01': 'BUY',
                           'S08': 'SHORT', 'S09': 'SHORT'}

    def test_no_rebalance_off_monday_without_refit(self):
        frames = self._universe()
        dates = frames['S00'].index
        assert dates[0].weekday() == 0 and dates[1].weekday() == 1
        pod = self._pod(self._scores())
        assert pod.generate_signals(
            self._through(frames, dates[0]), dates[0])  # Monday + refit day
        assert pod.generate_signals(
            self._through(frames, dates[1]), dates[1]) == {}  # Tuesday

    def test_flip_closes_first_then_enters_next_rebalance(self):
        frames = self._universe()
        dates = frames['S00'].index
        mondays = [d for d in dates if d.weekday() == 0]
        assert len(mondays) >= 3
        pod = self._pod(self._scores())

        week1 = pod.generate_signals(self._through(frames, mondays[0]),
                                     mondays[0])
        assert week1 == {'S00': 'BUY', 'S01': 'BUY',
                         'S08': 'SHORT', 'S09': 'SHORT'}

        # Reversed scores: every holding flips sides -> closes ONLY.
        pod.controller.model.scores = self._scores(reverse=True)
        week2 = pod.generate_signals(self._through(frames, mondays[1]),
                                     mondays[1])
        assert week2 == {'S00': 'SELL', 'S01': 'SELL',
                         'S08': 'COVER', 'S09': 'COVER'}

        # Next rebalance: the closes have had a chance to fill; the
        # new-side entries arrive now.
        week3 = pod.generate_signals(self._through(frames, mondays[2]),
                                     mondays[2])
        assert week3 == {'S09': 'BUY', 'S08': 'BUY',
                         'S01': 'SHORT', 'S00': 'SHORT'}

    def test_degraded_model_emits_nothing_and_keeps_book(self):
        frames = self._universe()
        mondays = [d for d in frames['S00'].index if d.weekday() == 0]
        pod = self._pod(self._scores())
        pod.generate_signals(self._through(frames, mondays[0]), mondays[0])
        held = dict(pod._desired)
        pod.controller.model.scores = {}  # model degrades to no scores
        assert pod.generate_signals(self._through(frames, mondays[1]),
                                    mondays[1]) == {}
        assert pod._desired == held  # fail-safe: no information, no churn

    def test_pod_isolation_no_shared_mutable_state(self):
        frames = self._universe()
        monday = frames['S00'].index[0]
        pod_a = self._pod(self._scores())
        pod_b = self._pod(self._scores())
        assert pod_a._desired is not pod_b._desired
        assert pod_a.controller is not pod_b.controller
        pod_a.generate_signals(self._through(frames, monday), monday)
        assert pod_a._desired  # pod_a built a book...
        assert pod_b._desired == {}  # ...pod_b is untouched
        assert pod_b.controller.is_fitted is False

        momentum_a, momentum_b = MomentumPod(), MomentumPod()
        assert momentum_a._strategy is not momentum_b._strategy
        reversion_a, reversion_b = MeanReversionPod(), MeanReversionPod()
        assert reversion_a._strategy is not reversion_b._strategy
