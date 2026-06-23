"""Turnover control on the cross-sectional long/short book (Phase 5).

Exercises the two opt-in knobs added to CrossSectionalLongShortDesk via a
TwoSigmaDesk + stub score model:

  * exit_quantile (hysteresis band) — a HELD name that only slips out of the
    top/bottom `quantile` entry set is retained while it stays inside the wider
    `exit_quantile` band, instead of churning;
  * min_holding_days — a freshly-opened name is held for at least N trading
    days before it can be dropped to flat.

Both default to a no-op, and neither blocks a genuine flip. Offline, seeded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models import Asset, AssetType, Position
from desks.twosigma import TwoSigmaDesk
from desks.walk_forward import WalkForwardController, WalkForwardModel
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager


def stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


def wide_risk() -> RiskManager:
    return RiskManager(max_position_size=1.0, max_daily_loss=0.99,
                       position_stop_loss=0.90)


class StubScoreModel(WalkForwardModel):
    """Per-date {symbol: score}; unscripted dates use `default`."""

    def __init__(self, default=None, schedule=None):
        self.default = default or {}
        self.schedule = {pd.Timestamp(d): s for d, s in (schedule or {}).items()}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        scores = self.schedule.get(pd.Timestamp(date), self.default)
        return {s: v for s, v in scores.items() if s in data}


def make_desk(default=None, schedule=None, **kwargs) -> TwoSigmaDesk:
    kwargs.setdefault('risk_manager', wide_risk())
    controller = WalkForwardController(StubScoreModel(default, schedule),
                                       min_train_days=1)
    return TwoSigmaDesk(controller=controller, **kwargs)


def quiet_frame(n=6, seed=5, start='2023-01-02') -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, n))
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': np.full(n, 500_000.0),
    }, index=index)


def universe(n_symbols=10, n_days=6):
    return {f'S{i:02d}': quiet_frame(n=n_days, seed=100 + i)
            for i in range(n_symbols)}


def monotone(n=10):
    """S00 highest .. S09 lowest, centered."""
    return {f'S{i:02d}': 0.4 - 0.08 * i for i in range(n)}


def drive(desk, frames, dates, portfolio):
    out = {}
    for date in dates:
        sliced = {s: f[f.index <= date] for s, f in frames.items()}
        sliced = {s: f for s, f in sliced.items() if not f.empty}
        desk.set_clock(date)
        out[date] = desk.generate_intents(sliced, date, portfolio)
    return out


def fill(portfolio, opens, date):
    """Simulate next-open fills for emitted entry intents."""
    for intent in opens:
        qty = 100 if intent.action == 'BUY' else -100
        portfolio.positions[intent.asset] = Position(
            asset=intent.asset, quantity=qty, avg_entry_price=100.0,
            current_price=100.0, timestamp=pd.Timestamp(date))


def closes_for(intents, symbol):
    return [i for i in intents
            if i.asset.symbol == symbol and i.action in ('SELL', 'COVER')]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_exit_quantile_below_quantile_raises(self):
        with pytest.raises(ValueError, match='exit_quantile'):
            TwoSigmaDesk(quantile=0.2, exit_quantile=0.1)

    def test_exit_quantile_above_half_raises(self):
        with pytest.raises(ValueError, match='exit_quantile'):
            TwoSigmaDesk(exit_quantile=0.6)

    def test_negative_min_holding_days_raises(self):
        with pytest.raises(ValueError, match='min_holding_days'):
            TwoSigmaDesk(min_holding_days=-1)

    def test_defaults_construct(self):
        desk = TwoSigmaDesk()
        assert desk._exit_quantile == desk.quantile
        assert desk.min_holding_days == 0


# ---------------------------------------------------------------------------
# Behaviour. Day 0 opens the top/bottom-2 of 10 (quantile 0.2 -> k=2); day 1
# re-ranks so the held longs S00/S01 slip to ranks 2-3 (out of the top-2 entry
# set but inside a top-4 band).
# ---------------------------------------------------------------------------
DAY1_SLIP = {'S02': 0.5, 'S03': 0.45, 'S00': 0.40, 'S01': 0.35,
             'S04': 0.0, 'S05': -0.05, 'S06': -0.1, 'S07': -0.15,
             'S08': -0.30, 'S09': -0.35}


def _open_then_rerank(**desk_kwargs):
    frames = universe(10, n_days=4)
    dates = list(frames['S00'].index)
    desk = make_desk(default=monotone(), schedule={dates[1]: DAY1_SLIP},
                     quantile=0.2, **desk_kwargs)
    portfolio = PortfolioManager(100000.0)
    out0 = drive(desk, frames, dates[:1], portfolio)
    opens0 = [i for i in out0[dates[0]] if i.action in ('BUY', 'SHORT')]
    fill(portfolio, opens0, dates[0])
    out1 = drive(desk, frames, dates[1:2], portfolio)
    return out1[dates[1]]


class TestHysteresisBand:
    def test_default_closes_slipped_name(self):
        # No band: a held long that drops out of the top-2 entry set is closed.
        intents = _open_then_rerank()
        assert closes_for(intents, 'S00')  # SELL emitted
        assert closes_for(intents, 'S01')

    def test_band_retains_slipped_name(self):
        # exit_quantile 0.4 -> top-4 band; S00/S01 at ranks 2-3 stay held.
        intents = _open_then_rerank(exit_quantile=0.4)
        assert not closes_for(intents, 'S00')  # held, no churn
        assert not closes_for(intents, 'S01')
        # The new top-2 (S02/S03) still open this day.
        opened = {i.asset.symbol for i in intents if i.action == 'BUY'}
        assert {'S02', 'S03'} <= opened


class TestMinHoldingPeriod:
    def test_young_name_retained_until_period_elapses(self):
        # S00 leaves its side entirely on day 1 but is < min_holding_days old.
        frames = universe(10, n_days=4)
        dates = list(frames['S00'].index)
        # Day 1: S00 falls to the middle (rank ~4), outside any band.
        day1 = {'S01': 0.5, 'S02': 0.45, 'S03': 0.4, 'S04': 0.35,
                'S00': 0.0, 'S05': -0.05, 'S06': -0.1, 'S07': -0.15,
                'S08': -0.3, 'S09': -0.35}
        desk = make_desk(default=monotone(), schedule={dates[1]: day1},
                         quantile=0.2, min_holding_days=3)
        portfolio = PortfolioManager(100000.0)
        out0 = drive(desk, frames, dates[:1], portfolio)
        fill(portfolio, [i for i in out0[dates[0]]
                         if i.action in ('BUY', 'SHORT')], dates[0])
        intents1 = drive(desk, frames, dates[1:2], portfolio)[dates[1]]
        assert not closes_for(intents1, 'S00')  # held: 1 day < 3

    def test_flip_not_blocked_by_min_hold(self):
        # A genuine reversal (S00 -> bottom-2) still closes the long to flip,
        # even though min_holding_days has not elapsed.
        frames = universe(10, n_days=4)
        dates = list(frames['S00'].index)
        day1 = {f'S{i:02d}': 0.4 - 0.08 * i for i in range(1, 9)}
        day1['S00'] = -0.9  # crashes to the very bottom -> short entry set
        desk = make_desk(default=monotone(), schedule={dates[1]: day1},
                         quantile=0.2, min_holding_days=5)
        portfolio = PortfolioManager(100000.0)
        out0 = drive(desk, frames, dates[:1], portfolio)
        fill(portfolio, [i for i in out0[dates[0]]
                         if i.action in ('BUY', 'SHORT')], dates[0])
        intents1 = drive(desk, frames, dates[1:2], portfolio)[dates[1]]
        assert closes_for(intents1, 'S00')  # long closed for the flip
