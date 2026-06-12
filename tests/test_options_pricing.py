"""Tests for desks.options_pricing — the synthetic fair-value engine.

Black-Scholes prices pin to the textbook values; greeks pin to central
finite differences of the price; the IV model's no-lookahead slicing is
asserted by construction (a frame where including the pricing date's
close changes HV20 detectably). Offline, deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.models import Asset, AssetType
from desks.options_pricing import (HV_WINDOW, IV_FLOOR,
                                   OpenBBEarningsCalendar,
                                   SyntheticEarningsCalendar,
                                   SyntheticIVModel, black_scholes_greeks,
                                   black_scholes_price, price_option)


class TestBlackScholesPrice:
    def test_known_call_value(self):
        # S=100, K=100, T=1, sigma=0.2, r=0.05 (rate passed EXPLICITLY —
        # the module default is 0.045) -> 10.4506 per the textbook.
        price = black_scholes_price(100, 100, 1.0, 0.2, rate=0.05,
                                    right='call')
        assert price == pytest.approx(10.4506, abs=1e-4)

    def test_known_put_value(self):
        price = black_scholes_price(100, 100, 1.0, 0.2, rate=0.05,
                                    right='put')
        assert price == pytest.approx(5.5735, abs=1e-4)

    @pytest.mark.parametrize('spot,strike,t,vol,rate', [
        (100, 100, 1.0, 0.2, 0.05),
        (105, 95, 0.5, 0.35, 0.045),
        (50, 60, 0.25, 0.15, 0.01),
    ])
    def test_put_call_parity(self, spot, strike, t, vol, rate):
        call = black_scholes_price(spot, strike, t, vol, rate, 'call')
        put = black_scholes_price(spot, strike, t, vol, rate, 'put')
        assert call - put == pytest.approx(
            spot - strike * math.exp(-rate * t), abs=1e-10)

    @pytest.mark.parametrize('t', [0.0, -0.1])
    def test_t_leq_zero_is_intrinsic(self, t):
        assert black_scholes_price(110, 100, t, 0.2, 0.05, 'call') == 10.0
        assert black_scholes_price(90, 100, t, 0.2, 0.05, 'call') == 0.0
        assert black_scholes_price(90, 100, t, 0.2, 0.05, 'put') == 10.0
        assert black_scholes_price(110, 100, t, 0.2, 0.05, 'put') == 0.0

    def test_invalid_right_raises(self):
        with pytest.raises(ValueError, match="right must be"):
            black_scholes_price(100, 100, 1.0, 0.2, 0.05, 'straddle')


class TestBlackScholesGreeks:
    SPOT, STRIKE, T, VOL, RATE = 105.0, 100.0, 0.4, 0.25, 0.045

    @pytest.mark.parametrize('right', ['call', 'put'])
    def test_delta_vs_central_difference(self, right):
        h = 1e-5
        greeks = black_scholes_greeks(self.SPOT, self.STRIKE, self.T,
                                      self.VOL, self.RATE, right)
        fd = (black_scholes_price(self.SPOT + h, self.STRIKE, self.T,
                                  self.VOL, self.RATE, right)
              - black_scholes_price(self.SPOT - h, self.STRIKE, self.T,
                                    self.VOL, self.RATE, right)) / (2 * h)
        assert greeks['delta'] == pytest.approx(fd, abs=1e-4)

    @pytest.mark.parametrize('right', ['call', 'put'])
    def test_gamma_vs_second_central_difference(self, right):
        # Second differences amplify float rounding as eps/h^2, so the
        # gamma bump is 1e-4 (rounding ~4e-7) instead of 1e-5 (~4e-5,
        # uncomfortably near the 1e-4 tolerance).
        h = 1e-4
        greeks = black_scholes_greeks(self.SPOT, self.STRIKE, self.T,
                                      self.VOL, self.RATE, right)
        up = black_scholes_price(self.SPOT + h, self.STRIKE, self.T,
                                 self.VOL, self.RATE, right)
        mid = black_scholes_price(self.SPOT, self.STRIKE, self.T,
                                  self.VOL, self.RATE, right)
        down = black_scholes_price(self.SPOT - h, self.STRIKE, self.T,
                                   self.VOL, self.RATE, right)
        fd = (up - 2 * mid + down) / (h * h)
        assert greeks['gamma'] == pytest.approx(fd, abs=1e-4)

    @pytest.mark.parametrize('right', ['call', 'put'])
    def test_vega_vs_central_difference(self, right):
        h = 1e-5
        greeks = black_scholes_greeks(self.SPOT, self.STRIKE, self.T,
                                      self.VOL, self.RATE, right)
        fd = (black_scholes_price(self.SPOT, self.STRIKE, self.T,
                                  self.VOL + h, self.RATE, right)
              - black_scholes_price(self.SPOT, self.STRIKE, self.T,
                                    self.VOL - h, self.RATE, right)) / (2 * h)
        # vega is per 1.00 vol point — exactly dP/dvol.
        assert greeks['vega'] == pytest.approx(fd, abs=1e-4)

    @pytest.mark.parametrize('right', ['call', 'put'])
    def test_theta_vs_central_difference_per_calendar_day(self, right):
        h = 1e-5
        greeks = black_scholes_greeks(self.SPOT, self.STRIKE, self.T,
                                      self.VOL, self.RATE, right)
        # dP/dT (time-to-expiry); theta (calendar decay) = -dP/dT / 365.
        dp_dt = (black_scholes_price(self.SPOT, self.STRIKE, self.T + h,
                                     self.VOL, self.RATE, right)
                 - black_scholes_price(self.SPOT, self.STRIKE, self.T - h,
                                       self.VOL, self.RATE, right)) / (2 * h)
        assert greeks['theta'] == pytest.approx(-dp_dt / 365.0, abs=1e-4)

    def test_price_matches_pricer(self):
        greeks = black_scholes_greeks(self.SPOT, self.STRIKE, self.T,
                                      self.VOL, self.RATE, 'call')
        assert greeks['price'] == pytest.approx(
            black_scholes_price(self.SPOT, self.STRIKE, self.T, self.VOL,
                                self.RATE, 'call'), abs=1e-12)

    def test_expired_greeks_are_settlement_deltas(self):
        itm_call = black_scholes_greeks(110, 100, 0.0, 0.2, 0.05, 'call')
        assert itm_call == {'price': 10.0, 'delta': 1.0, 'gamma': 0.0,
                            'theta': 0.0, 'vega': 0.0}
        otm_call = black_scholes_greeks(90, 100, 0.0, 0.2, 0.05, 'call')
        assert otm_call['delta'] == 0.0 and otm_call['price'] == 0.0
        itm_put = black_scholes_greeks(90, 100, -1.0, 0.2, 0.05, 'put')
        assert itm_put['delta'] == -1.0 and itm_put['price'] == 10.0
        otm_put = black_scholes_greeks(110, 100, 0.0, 0.2, 0.05, 'put')
        assert otm_put['delta'] == 0.0 and otm_put['vega'] == 0.0


# ----------------------------------------------------------------------
# Synthetic IV model
# ----------------------------------------------------------------------
def frame_with_jump(n: int = 40, jump_at: int = 30,
                    jump: float = 0.25) -> pd.DataFrame:
    """Alternating small returns, one violent +25% day at `jump_at` —
    HV20 windows including that close differ DETECTABLY from windows
    excluding it."""
    returns = np.array([0.004 if i % 2 == 0 else -0.004
                        for i in range(n)])
    returns[jump_at] = jump
    close = 100.0 * np.cumprod(1.0 + returns)
    index = pd.bdate_range('2023-01-02', periods=n)
    return pd.DataFrame({'open': close, 'close': close,
                         'high': close, 'low': close,
                         'volume': np.full(n, 1e6)}, index=index)


def hv20(closes: pd.Series) -> float:
    return float(closes.pct_change().rolling(HV_WINDOW).std().iloc[-1]
                 * np.sqrt(252))


class TestSyntheticIVModel:
    def test_hv_window_excludes_the_pricing_dates_close(self):
        frame = frame_with_jump()
        jump_date = frame.index[30]
        model = SyntheticIVModel(vrp_multiplier=1.10)

        # Pricing ON the jump day: the jump close is index 30 == `date`,
        # so it must be EXCLUDED (a fill at the open can't know it).
        iv_on = model.iv('XYZ', frame, jump_date)
        expected_excl = hv20(frame['close'].iloc[:30]) * 1.10
        assert iv_on == pytest.approx(expected_excl, rel=1e-12)

        # Pricing the NEXT day: the jump close is now strictly prior and
        # must be INCLUDED — and it changes HV20 detectably.
        iv_after = model.iv('XYZ', frame, frame.index[31])
        expected_incl = hv20(frame['close'].iloc[:31]) * 1.10
        assert iv_after == pytest.approx(expected_incl, rel=1e-12)
        assert iv_after > iv_on * 3  # the jump dominates the window

    def test_insufficient_prior_history_returns_none(self):
        frame = frame_with_jump()
        # Date at index HV_WINDOW: only HV_WINDOW closes strictly before
        # it — one short of the HV_WINDOW + 1 needed for 20 returns.
        assert SyntheticIVModel().iv('XYZ', frame,
                                     frame.index[HV_WINDOW]) is None
        assert SyntheticIVModel().iv(
            'XYZ', frame, frame.index[HV_WINDOW + 1]) is not None

    def test_iv_floor(self):
        n = 40
        close = np.full(n, 100.0)
        close[1::2] = 100.001  # nearly-zero realized vol
        frame = pd.DataFrame(
            {'close': close},
            index=pd.bdate_range('2023-01-02', periods=n))
        iv = SyntheticIVModel().iv('XYZ', frame, frame.index[-1])
        assert iv == IV_FLOOR

    def test_vrp_multiplier_scales_hv(self):
        frame = frame_with_jump()
        date = frame.index[28]
        plain = SyntheticIVModel(vrp_multiplier=1.0).iv('XYZ', frame, date)
        marked_up = SyntheticIVModel(vrp_multiplier=1.25).iv('XYZ', frame,
                                                             date)
        assert marked_up == pytest.approx(plain * 1.25, rel=1e-12)

    def test_earnings_bump_rises_into_event_and_dies_after(self):
        frame = frame_with_jump(n=40, jump_at=39)  # jump never in window
        earnings_date = frame.index[34].date()
        calendar = SyntheticEarningsCalendar({'XYZ': [earnings_date]})
        bumped = SyntheticIVModel(earnings_calendar=calendar,
                                  earnings_bump=0.40)
        plain = SyntheticIVModel()

        # Monotone RISING multiplier into E (exp(-days/3) grows as the
        # event nears), peaking at 1 + bump on the event date itself.
        ratios = []
        for idx in range(25, 35):  # dates <= E
            date = frame.index[idx]
            ratio = (bumped.iv('XYZ', frame, date)
                     / plain.iv('XYZ', frame, date))
            days_to_e = (earnings_date - date.date()).days
            assert ratio == pytest.approx(
                1.0 + 0.40 * math.exp(-days_to_e / 3.0), rel=1e-12)
            ratios.append(ratio)
        assert ratios == sorted(ratios)  # monotone rising into E
        assert ratios[-1] == pytest.approx(1.40, rel=1e-12)  # on E itself

        # The day AFTER E the bump is GONE — the crush.
        after = frame.index[35]
        assert bumped.iv('XYZ', frame, after) == pytest.approx(
            plain.iv('XYZ', frame, after), rel=1e-12)


class TestEarningsCalendars:
    def test_synthetic_calendar_next_earnings(self):
        calendar = SyntheticEarningsCalendar(
            {'AAA': ['2023-03-15', '2023-06-14']})
        assert calendar.next_earnings('AAA', '2023-01-01') == \
            pd.Timestamp('2023-03-15').date()
        # On the date itself it still counts (date <= E).
        assert calendar.next_earnings('AAA', '2023-03-15') == \
            pd.Timestamp('2023-03-15').date()
        assert calendar.next_earnings('AAA', '2023-03-16') == \
            pd.Timestamp('2023-06-14').date()
        assert calendar.next_earnings('AAA', '2023-07-01') is None
        assert calendar.next_earnings('ZZZ', '2023-01-01') is None

    def test_openbb_stub_degrades_to_none_offline(self, monkeypatch):
        # Guarded + lazy: the stub must swallow an openbb import failure
        # and return None instead of raising (live wiring is Phase 9).
        # The import is FORCED to fail so no network is ever attempted.
        import sys
        monkeypatch.setitem(sys.modules, 'openbb', None)
        assert OpenBBEarningsCalendar().next_earnings(
            'AAA', '2023-01-01') is None


class TestPriceOption:
    def test_prices_with_model_iv_and_calendar_dte(self):
        frame = frame_with_jump()
        date = frame.index[28]
        expiry = (date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        asset = Asset('XYZ', AssetType.CALL, strike_price=100.0,
                      expiration_date=expiry)
        model = SyntheticIVModel()
        price = price_option(asset, frame, date, spot=102.0, iv_model=model)
        iv = model.iv('XYZ', frame, date)
        assert price == pytest.approx(
            black_scholes_price(102.0, 100.0, 30 / 365.0, iv,
                                rate=0.045, right='call'), abs=1e-12)

    def test_expired_option_is_intrinsic_without_iv(self):
        # 5 rows of history — far too little for HV20 — but an expired
        # option needs no vol at all.
        frame = frame_with_jump().iloc[:5]
        asset = Asset('XYZ', AssetType.PUT, strike_price=100.0,
                      expiration_date='2023-01-04')
        price = price_option(asset, frame, frame.index[4], spot=92.0,
                             iv_model=SyntheticIVModel())
        assert price == 8.0

    def test_unpriceable_without_history_returns_none(self):
        frame = frame_with_jump().iloc[:5]
        asset = Asset('XYZ', AssetType.CALL, strike_price=100.0,
                      expiration_date='2024-01-04')
        assert price_option(asset, frame, frame.index[4], spot=100.0,
                            iv_model=SyntheticIVModel()) is None

    def test_stock_asset_raises(self):
        with pytest.raises(ValueError, match="needs an option asset"):
            price_option(Asset('XYZ', AssetType.STOCK), frame_with_jump(),
                         '2023-02-01', 100.0, SyntheticIVModel())
