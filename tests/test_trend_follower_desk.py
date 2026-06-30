"""TrendFollowerDesk.generate_intents signal logic — regime filter, ATR stop,
breakout entry, and pyramiding cap. No engine, no network: intents are checked
directly against synthetic point-in-time frames and a fake portfolio.
"""

from __future__ import annotations

import pandas as pd

from core.models import Asset, AssetType, Position
from desks.trend_follower import TrendFollowerDesk
from portfolio.manager import PortfolioManager

SPY = Asset(symbol='SPY', asset_type=AssetType.STOCK)


class _FakePortfolio:
    """Minimal portfolio surface generate_intents touches."""

    def __init__(self, positions=None, value=100_000.0):
        self._positions = positions or {}
        self._value = value

    def get_position(self, asset):
        return self._positions.get(asset)

    def get_portfolio_value(self):
        return self._value


def _frame(closes, sma200, atr):
    """Build a >=200-bar OHLC-ish frame with the columns the desk reads."""
    n = len(closes)
    idx = pd.bdate_range(end='2024-06-28', periods=n)
    return pd.DataFrame(
        {'close': list(closes), 'sma_200': [sma200] * n, 'atr': [atr] * n},
        index=idx,
    )


def _position(qty, entry, current):
    return Position(asset=SPY, quantity=qty, avg_entry_price=entry,
                    current_price=current, timestamp=pd.Timestamp('2024-06-28'))


def _run(frame, portfolio):
    desk = TrendFollowerDesk()
    return desk.generate_intents({'SPY': frame}, frame.index[-1], portfolio)


# --- Regime filter ---------------------------------------------------------

def test_regime_off_while_holding_exits():
    frame = _frame([90.0] * 220, sma200=100.0, atr=2.0)  # close < SMA200
    pf = _FakePortfolio({SPY: _position(50, entry=110.0, current=90.0)})
    intents = _run(frame, pf)
    assert len(intents) == 1
    assert intents[0].action == 'SELL'
    assert intents[0].size_fraction == 1.0
    assert 'regime off' in intents[0].reason


def test_regime_off_while_flat_does_nothing():
    frame = _frame([90.0] * 220, sma200=100.0, atr=2.0)
    assert _run(frame, _FakePortfolio()) == []


# --- Breakout entry / no-signal -------------------------------------------

def test_new_high_above_sma_enters_long():
    frame = _frame(range(100, 320), sma200=200.0, atr=5.0)  # last=319 is the max
    intents = _run(frame, _FakePortfolio())
    assert len(intents) == 1
    assert intents[0].action == 'BUY'
    assert intents[0].size_fraction == 0.10
    assert 'new 50-day high' in intents[0].reason


def test_regime_on_no_new_high_flat_does_nothing():
    # Ramp up then dip: last close (250) is below recent highs -> not a new high.
    frame = _frame(list(range(100, 319)) + [250.0], sma200=200.0, atr=5.0)
    assert _run(frame, _FakePortfolio()) == []


# --- ATR stop --------------------------------------------------------------

def test_atr_stop_exits_when_price_falls_below_entry_minus_atrs():
    # Regime on (250 > 200), not a new high, but price <= entry - 3.5*ATR.
    # stop = 320 - 3.5*20 = 250; close 250 <= 250 -> exit.
    frame = _frame(list(range(100, 319)) + [250.0], sma200=200.0, atr=20.0)
    pf = _FakePortfolio({SPY: _position(10, entry=320.0, current=250.0)})
    intents = _run(frame, pf)
    assert len(intents) == 1
    assert intents[0].action == 'SELL'
    assert 'ATR stop' in intents[0].reason


# --- Pyramiding ------------------------------------------------------------

def test_pyramids_on_new_high_below_cap():
    # New high, holding a small position (exposure ~16% < 40% cap), wide ATR
    # (no stop) -> adds another unit.
    frame = _frame(range(100, 320), sma200=200.0, atr=5.0)  # last=319
    pf = _FakePortfolio({SPY: _position(50, entry=300.0, current=319.0)})
    intents = _run(frame, pf)
    assert len(intents) == 1
    assert intents[0].action == 'BUY'


def test_no_pyramid_at_exposure_cap():
    # exposure = 130*319/100000 ~= 41% >= 40% cap -> no add despite a new high.
    frame = _frame(range(100, 320), sma200=200.0, atr=5.0)
    pf = _FakePortfolio({SPY: _position(130, entry=300.0, current=319.0)})
    assert _run(frame, pf) == []


# --- Warmup guards ---------------------------------------------------------

def test_short_history_is_skipped():
    frame = _frame(range(100, 250), sma200=200.0, atr=5.0)[:150]  # < 200 bars
    assert _run(frame, _FakePortfolio()) == []


def test_nan_sma_is_skipped():
    frame = _frame(range(100, 320), sma200=float('nan'), atr=5.0)
    assert _run(frame, _FakePortfolio()) == []


# --- Risk-config integration (regression) ----------------------------------

def test_risk_cap_strictly_exceeds_entry_unit():
    """Regression invariant: a per-trade cap EQUAL to the entry unit is
    float-fragile — a 0.10 unit lands at 0.10000000000000002 of capital, so the
    strict '>' check blocks every entry. The desk's default cap must STRICTLY
    exceed its entry unit."""
    desk = TrendFollowerDesk()
    assert desk.risk_manager.max_position_size > desk.entry_size


def test_entry_unit_is_approved_at_a_realistic_balance():
    """End-to-end: a 10% entry on a NON-round book (where pv*0.10/pv exceeds
    0.10 in float) must be approved, not boundary-blocked, so the trend book
    actually gets invested (regression: 815/843 entries were blocked)."""
    desk = TrendFollowerDesk()
    frame = _frame(range(100, 320), sma200=200.0, atr=5.0)   # regime-on new high
    pm = PortfolioManager(99_998.32)                         # non-round balance
    all_data = {'SPY': frame}
    intents = desk.generate_intents(all_data, frame.index[-1], pm)
    approved = desk.apply_risk(intents, pm, all_data, frame.index[-1])
    assert any(i.action == 'BUY' for i in intents)           # signal fired
    assert any(i.action == 'BUY' for i in approved)          # and was NOT blocked


def test_wide_atr_regime_design_not_undercut_by_tight_base_stop():
    """The desk's exits are the regime filter and the wide ATR stop; the base
    apply_risk percentage stop must be loose enough not to preempt them. A long
    sitting ~3% below entry (inside a ~3.5*ATR stop) must NOT be force-closed by
    the base stop."""
    desk = TrendFollowerDesk()
    frame = _frame([103.0] * 220, sma200=100.0, atr=5.0)     # regime on, price 103
    asset = Asset(symbol='SPY', asset_type=AssetType.STOCK)
    pm = PortfolioManager(100_000)
    pm.positions[asset] = Position(asset=asset, quantity=100, avg_entry_price=106.0,
                                   current_price=103.0,
                                   timestamp=pd.Timestamp('2024-06-28'))
    approved = desk.apply_risk([], pm, {'SPY': frame}, frame.index[-1])
    # ~2.8% below entry, well inside 3.5*ATR (~17 wide) -> no base-stop SELL.
    assert not any(i.action == 'SELL' for i in approved)
