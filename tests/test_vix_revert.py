"""Tests for the VIX Reversion desk + the reverse-martingale sizer.

Pure-frame tests: synthetic ^VIX/SPY frames with explicit sma_20 columns and a
stub portfolio, driving the desk's episode state machine directly (entries on
fresh crossings only, VIX never traded, win/loss classification feeding the
sizer, stop-outs counted as losses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desks.sizing import ReverseMartingaleSizer
from desks.vix_revert import VixReversionDesk


# ---------------------------------------------------------------------------
# ReverseMartingaleSizer
# ---------------------------------------------------------------------------
class TestReverseMartingaleSizer:
    def test_streak_math_and_cap(self):
        s = ReverseMartingaleSizer(0.20, step=0.5, max_mult=2.0)
        assert s.size() == pytest.approx(0.20)          # streak 0
        s.record(True)
        assert s.size() == pytest.approx(0.30)          # 1 + 0.5*1
        s.record(True)
        assert s.size() == pytest.approx(0.40)          # capped at 2x from here
        s.record(True)
        assert s.size() == pytest.approx(0.40)          # max_mult binds
        s.record(False)
        assert s.win_streak == 0
        assert s.size() == pytest.approx(0.20)          # loss resets to base

    def test_reset(self):
        s = ReverseMartingaleSizer(0.1)
        s.record(True)
        s.reset()
        assert s.win_streak == 0

    def test_validation(self):
        with pytest.raises(ValueError):
            ReverseMartingaleSizer(0.0)
        with pytest.raises(ValueError):
            ReverseMartingaleSizer(0.1, step=-0.1)
        with pytest.raises(ValueError):
            ReverseMartingaleSizer(0.1, max_mult=0.9)


# ---------------------------------------------------------------------------
# VixReversionDesk
# ---------------------------------------------------------------------------
class _Pos:
    def __init__(self, quantity, avg_entry_price):
        self.quantity = quantity
        self.avg_entry_price = avg_entry_price


class _StubPM:
    """Duck-typed PortfolioManager surface generate_intents touches."""

    def __init__(self):
        self.pos = None

    def get_position(self, asset):
        return self.pos


def _frames(n, vix_close, vix_sma, spy_close=None, spy_open=None):
    """Build {^VIX, SPY} frames of length n with explicit indicator columns."""
    idx = pd.bdate_range('2023-01-02', periods=n)
    vix = pd.DataFrame({'close': np.asarray(vix_close, float),
                        'sma_20': np.asarray(vix_sma, float)}, index=idx)
    spy_close = spy_close if spy_close is not None else np.full(n, 400.0)
    spy = pd.DataFrame({'close': np.asarray(spy_close, float)}, index=idx)
    if spy_open is not None:
        spy['open'] = np.asarray(spy_open, float)
    return {'^VIX': vix, 'SPY': spy}, idx


def _calm(n):
    """n bars of calm VIX (close 15, sma 15 => ratio 1.0)."""
    return [15.0] * n, [15.0] * n


class TestVixReversionDesk:
    def _spiked(self, n_calm, n_spike, spy=None):
        """Calm bars then spike bars (close 21, sma 15 => ratio 1.4)."""
        close, sma = _calm(n_calm)
        close = close + [21.0] * n_spike
        sma = sma + [15.0] * n_spike
        return _frames(n_calm + n_spike, close, sma, spy)

    def test_enters_only_on_fresh_crossing_and_never_trades_vix(self):
        all_data, idx = self._spiked(60, 2)
        pm = _StubPM()
        desk = VixReversionDesk()
        # crossing day: ratio went 1.0 -> 1.4 across the 1.25 threshold
        day = {k: v.iloc[:61] for k, v in all_data.items()}
        intents = desk.generate_intents(day, idx[60], pm)
        assert [i.action for i in intents] == ['BUY']
        assert intents[0].asset.symbol == 'SPY'
        assert intents[0].size_fraction == pytest.approx(0.20)
        # signal still ON next day, but no fresh crossing -> no new entry
        assert desk.generate_intents(all_data, idx[61], pm) == []

    def test_exit_on_reversion_classifies_win_at_fill_and_presses_entry(self):
        close, sma = _calm(60)
        close += [21.0, 21.0, 14.0, 14.0]    # spike, hold, reverted, fill day
        sma += [15.0, 15.0, 15.0, 15.0]
        spy = [400.0] * 60 + [400.0, 401.0, 402.0, 419.0]
        opens = [400.0] * 63 + [420.0]       # exit fills at day-63's open
        all_data, idx = _frames(64, close, sma, spy, spy_open=opens)
        pm = _StubPM()
        desk = VixReversionDesk()
        assert desk.generate_intents(
            {k: v.iloc[:61] for k, v in all_data.items()}, idx[60], pm)
        pm.pos = _Pos(50, 400.0)             # T+1 fill observed
        assert desk.generate_intents(
            {k: v.iloc[:62] for k, v in all_data.items()}, idx[61], pm) == []
        intents = desk.generate_intents(
            {k: v.iloc[:63] for k, v in all_data.items()}, idx[62], pm)
        assert [i.action for i in intents] == ['SELL']
        assert 'reverted' in intents[0].reason
        # Outcome is NOT classified at the signal — only at the fill.
        assert desk.sizer.win_streak == 0
        pm.pos = None                        # SELL filled at day-63's open
        assert desk.generate_intents(all_data, idx[63], pm) == []
        # fill-day open 420 > entry 400 => WIN => next entry pressed to 0.30
        assert desk.sizer.win_streak == 1
        close2, sma2 = _calm(60)
        close2 += [21.0]
        sma2 += [15.0]
        all_data2, idx2 = _frames(61, close2, sma2)
        # streak carries over: same desk, new spike later
        intents2 = desk.generate_intents(all_data2, idx2[60], pm)
        assert intents2 and intents2[0].size_fraction == pytest.approx(0.30)

    def test_time_stop_exit(self):
        n_spike = 25                          # > max_hold_days=21
        all_data, idx = self._spiked(60, n_spike)
        pm = _StubPM()
        desk = VixReversionDesk()
        assert desk.generate_intents(
            {k: v.iloc[:61] for k, v in all_data.items()}, idx[60], pm)
        pm.pos = _Pos(50, 400.0)
        sell = None
        for d in range(61, 60 + n_spike):
            day = {k: v.iloc[:d + 1] for k, v in all_data.items()}
            out = desk.generate_intents(day, idx[d], pm)
            if out:
                sell = (d, out)
                break
        assert sell is not None
        _, out = sell
        assert out[0].action == 'SELL' and 'time stop' in out[0].reason

    def test_stop_out_is_a_loss_and_resets_streak(self):
        all_data, idx = self._spiked(60, 3)
        pm = _StubPM()
        desk = VixReversionDesk()
        desk.sizer.record(True)               # pretend a prior win
        assert desk.sizer.win_streak == 1
        assert desk.generate_intents(
            {k: v.iloc[:61] for k, v in all_data.items()}, idx[60], pm)
        pm.pos = _Pos(50, 400.0)
        desk.generate_intents(
            {k: v.iloc[:62] for k, v in all_data.items()}, idx[61], pm)
        # shared apply_risk stop closed it; desk never emitted an exit
        pm.pos = None
        desk.generate_intents(all_data, idx[62], pm)
        assert desk.sizer.win_streak == 0     # loss by construction

    def test_foreign_position_is_ignored(self):
        """A shared-book SPY position the desk did not open is never adopted,
        never exited, and blocks new entries (fund-mode safety)."""
        close, sma = _calm(60)
        close += [14.0]                       # calm: 'reverted' if we held
        sma += [15.0]
        all_data, idx = _frames(61, close, sma)
        pm = _StubPM()
        pm.pos = _Pos(80, 390.0)              # another desk's SPY long
        desk = VixReversionDesk()
        assert desk.generate_intents(all_data, idx[60], pm) == []
        assert desk._entry_px is None         # not adopted
        # ...and a spike crossing while the foreign position exists: no BUY.
        close2, sma2 = _calm(60)
        close2 += [21.0]
        sma2 += [15.0]
        all_data2, idx2 = _frames(61, close2, sma2)
        assert desk.generate_intents(all_data2, idx2[60], pm) == []

    def test_stalled_exit_is_reemitted(self):
        n_spike = 30
        all_data, idx = self._spiked(60, n_spike)
        pm = _StubPM()
        desk = VixReversionDesk(max_hold_days=2)   # exit signal fires fast
        assert desk.generate_intents(
            {k: v.iloc[:61] for k, v in all_data.items()}, idx[60], pm)
        pm.pos = _Pos(50, 400.0)
        sells = []
        for d in range(61, 60 + n_spike):
            day = {k: v.iloc[:d + 1] for k, v in all_data.items()}
            out = desk.generate_intents(day, idx[d], pm)
            sells.extend(out)                 # position never goes flat
        assert len(sells) >= 2                # original exit + >=1 retry
        assert any('retry' in i.reason for i in sells[1:])
        assert all(i.action == 'SELL' for i in sells)

    def test_apply_risk_blocks_index_level_opens(self):
        from core.models import Asset, AssetType
        from desks.base import DeskIntent
        from portfolio.manager import PortfolioManager
        desk = VixReversionDesk()
        pm = PortfolioManager(initial_capital=100_000.0)
        vix_asset = Asset(symbol='^VIX', asset_type=AssetType.STOCK)
        spy_asset = Asset(symbol='SPY', asset_type=AssetType.STOCK)
        intents = [
            DeskIntent(asset=vix_asset, action='BUY', size_fraction=0.1,
                       reason='rogue index buy'),
            DeskIntent(asset=spy_asset, action='BUY', size_fraction=0.1,
                       reason='legit'),
        ]
        approved = desk.apply_risk(intents, pm, {}, pd.Timestamp('2023-06-01'))
        assert [i.asset.symbol for i in approved] == ['SPY']
        assert any(n.category == 'risk' and '^VIX' in n.message
                   for n in desk.notes)

    def test_missing_vix_degrades_flat_with_one_info_note(self):
        idx = pd.bdate_range('2023-01-02', periods=70)
        spy = pd.DataFrame({'close': np.full(70, 400.0)}, index=idx)
        pm = _StubPM()
        desk = VixReversionDesk()
        assert desk.generate_intents({'SPY': spy}, idx[-1], pm) == []
        assert desk.generate_intents({'SPY': spy}, idx[-1], pm) == []
        infos = [n for n in desk.notes if n.category == 'info']
        assert len(infos) == 1                # warned once, not daily

    def test_fixed_rule_has_no_walk_forward_fits(self):
        assert VixReversionDesk().walk_forward_fits == []
