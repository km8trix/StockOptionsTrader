"""Phase E (Jane Street vol surface + IV-rank earnings exclusion) tests.

Two OPT-IN features sharpen the Jane Street desk:

    1. A skew-aware VOL SURFACE (desks.options_pricing.skew_adjusted_iv,
       wired through JaneStreetDesk via vol_skew_slope). Each structure
       leg is priced at its OWN strike moneyness — equity skew: OTM puts
       richer, OTM calls cheaper.
    2. An IV-RANK EARNINGS EXCLUSION (exclude_earnings_from_iv_rank):
       observations inside an earnings window are dropped from the
       _iv_rank percentile so the VRP signal is not fooled by earnings
       vol run-ups.

CRITICAL INVARIANT — both features DEFAULT OFF and the DEFAULTS reproduce
the current behavior EXACTLY. The desk carries a byte-identical greeks
golden the project has held across every phase; these tests PIN that the
greeks_series / full-report hashes are unchanged with defaults (the
headline guard) and that the ON behaviors do what the spec claims.

Offline, seeded, deterministic. PRICING REALITY: every option price is a
synthetic Black-Scholes value (desks/options_pricing). Mirrors the style
and fixtures of tests/test_janestreet_desk.py and
tests/test_options_pricing.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import AssetType
from data.market_data import MarketDataHandler
from desks.janestreet import JaneStreetDesk
from desks.options_pricing import (IV_FLOOR, SyntheticEarningsCalendar,
                                   SyntheticIVModel, black_scholes_price,
                                   skew_adjusted_iv)
from desks.regime import REGIME_LABELS
from desks.walk_forward import WalkForwardController, WalkForwardModel
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

# ----------------------------------------------------------------------
# Shared fixtures (mirrored from tests/test_janestreet_desk.py)
# ----------------------------------------------------------------------
SHIFT = 80


def regime(state, p=0.9):
    others = [label for label in REGIME_LABELS if label != state]
    rest = (1.0 - p) / 2
    return {'state': state,
            'probs': {state: p, others[0]: rest, others[1]: rest}}


class StubRegimeModel(WalkForwardModel):
    """Fixed default regime, optionally overridden per date (a fresh
    instance per controller — WalkForwardController owns its model)."""

    def __init__(self, default=None, schedule=None):
        self.default = default
        self.schedule = {pd.Timestamp(d): r
                         for d, r in (schedule or {}).items()}

    def fit(self, train_data):
        pass

    def predict(self, data, date):
        result = self.schedule.get(pd.Timestamp(date), self.default)
        return result if result is not None else {}


def wide_risk():
    return RiskManager(max_position_size=1.0, max_daily_loss=0.99,
                       position_stop_loss=0.90)


def make_desk(default_regime='trending', schedule=None, fitted=True,
              **kwargs):
    controller = WalkForwardController(
        StubRegimeModel(default=regime(default_regime) if default_regime
                        else None, schedule=schedule),
        min_train_days=1 if fitted else 10 ** 9)
    kwargs.setdefault('risk_manager', wide_risk())
    kwargs.setdefault('iv_rank_min_obs', 30)
    kwargs.setdefault('iv_rank_window', 60)
    return JaneStreetDesk(regime_controller=controller, **kwargs)


def vol_frame(n=120, shift_at=None, start='2022-01-03', base=100.0,
              pre=(0.008, 0.004), post=0.02):
    """Alternating-sign returns: magnitudes DECLINE linearly pre-shift
    then jump to `post`. Deterministic by construction — no RNG."""
    mags = np.linspace(pre[0], pre[1], n)
    if shift_at is not None:
        mags[shift_at:] = post
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = base * np.cumprod(1.0 + mags * signs)
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame({'open': close, 'high': close * 1.001,
                         'low': close * 0.999, 'close': close,
                         'volume': np.full(n, 1e6)}, index=index)


def drive(desk, frames, dates, portfolio):
    intents_by_date = {}
    for date in dates:
        sliced = {symbol: frame[frame.index <= date]
                  for symbol, frame in frames.items()}
        sliced = {s: f for s, f in sliced.items() if not f.empty}
        desk.set_clock(date)
        intents_by_date[date] = desk.generate_intents(sliced, date,
                                                      portfolio)
    return intents_by_date


def option_intents(intents):
    return [i for i in intents
            if i.asset.asset_type in (AssetType.CALL, AssetType.PUT)]


# ----------------------------------------------------------------------
# The headline guard: GREEKS GOLDEN byte-identity with defaults
# ----------------------------------------------------------------------
#: Captured BEFORE any Phase E change (backend brief) and re-confirmed
#: AFTER all changes with both features at their defaults.  The full-report
#: hash was intentionally re-minted when headline ``total_return`` was fixed to
#: reconcile to NAV (and therefore include cash commissions); the independent
#: greeks-series hash stayed unchanged, proving pricing/trading behavior did
#: not move.  Do not regold for Phase-E feature work.
GOLDEN_FULL_SHA = (
    "984eea194695abcd035df1ec1e32fe0cf53e76b5d462d9724a813c17e017c20a")
GOLDEN_GREEKS_SHA = (
    "955bd0621c15f2e565d396ed03fea8134df3dd6dfdbae809567311cdeea1f1f4")

# These frozen sha256 goldens were minted on macOS. libm/BLAS floating-point
# differences make a byte-identical hash non-portable across platforms (the
# project records the same caveat for NN goldens), so the exact-hash assertions
# run only on the mint platform. Cross-platform CI still exercises the full
# greeks path via the relative byte-identity test below and the determinism
# harness; a regold is caught locally — the dev platform — where it is reviewed.
golden_platform_only = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="frozen greeks/report hash golden is platform-specific (macOS mint)")


class TestGreeksGoldenByteIdentity:
    """The byte-identical greeks golden the project has held across every
    phase. With BOTH Phase E features at their defaults the end-to-end
    report — and the greeks_series specifically — must hash to the exact
    same value captured before the change. This is the opt-out guarantee
    in its strongest form: a regold here is a real regression."""

    N_DAYS = 160
    EARNINGS_IDX = 40

    def universe(self):
        return {'AAA': vol_frame(n=self.N_DAYS, shift_at=SHIFT),
                'BBB': vol_frame(n=self.N_DAYS, shift_at=SHIFT)}

    def build_report(self, monkeypatch, **desk_kwargs):
        """The SAME end-to-end harness the determinism test pins
        (TestEndToEndWithEngine.build_report), optionally constructing the
        desk with extra Phase E kwargs."""
        universe = self.universe()
        dates = universe['AAA'].index
        earnings = dates[self.EARNINGS_IDX].date()

        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        desk = make_desk(
            earnings_calendar=SyntheticEarningsCalendar({'BBB': [earnings]}),
            **desk_kwargs)
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        report = engine.run(sorted(universe), '2022-01-01', '2022-12-31',
                            benchmark_symbol=None)
        return report

    @staticmethod
    def _full_fingerprint(report):
        return json.dumps({
            'structures': report['structures'],
            'greeks_series': report['greeks_series'],
            'trader_notes': report['trader_notes'],
            'closed_trades': report['closed_trades'],
            'summary': report['summary'],
        }, sort_keys=True, default=str)

    @staticmethod
    def _sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @golden_platform_only
    def test_default_desk_reproduces_the_full_report_golden(self,
                                                            monkeypatch):
        # Defaults only — vol_skew_slope None, exclude_earnings off.
        report = self.build_report(monkeypatch)
        assert self._sha(self._full_fingerprint(report)) == GOLDEN_FULL_SHA

    @golden_platform_only
    def test_default_desk_reproduces_the_greeks_series_golden(self,
                                                              monkeypatch):
        report = self.build_report(monkeypatch)
        greeks = json.dumps(report['greeks_series'], sort_keys=True,
                            default=str)
        assert self._sha(greeks) == GOLDEN_GREEKS_SHA

    @golden_platform_only
    def test_explicit_default_values_are_byte_identical(self, monkeypatch):
        # Passing the documented DEFAULTS explicitly must match the
        # implicit-default golden bit-for-bit — proves the new params'
        # defaults reproduce the current behavior.
        report = self.build_report(
            monkeypatch,
            vol_skew_slope=None,
            exclude_earnings_from_iv_rank=False,
            iv_rank_earnings_window_days=5)
        assert self._sha(self._full_fingerprint(report)) == GOLDEN_FULL_SHA
        greeks = json.dumps(report['greeks_series'], sort_keys=True,
                            default=str)
        assert self._sha(greeks) == GOLDEN_GREEKS_SHA

    @golden_platform_only
    def test_zero_slope_is_a_flat_surface_byte_identical(self, monkeypatch):
        # vol_skew_slope == 0.0 is the explicit flat surface; the
        # _surface_iv switch / skew_adjusted_iv both short-circuit to the
        # base IV, so the full report stays byte-identical to the golden.
        report = self.build_report(monkeypatch, vol_skew_slope=0.0)
        assert self._sha(self._full_fingerprint(report)) == GOLDEN_FULL_SHA

    def _run_no_calendar(self, monkeypatch, **desk_kwargs):
        """Identical end-to-end harness but with NO earnings calendar
        injected (so the only variable is the Phase E kwarg under test)."""
        universe = self.universe()

        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        desk = make_desk(**desk_kwargs)  # no earnings_calendar
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        return engine.run(sorted(universe), '2022-01-01', '2022-12-31',
                          benchmark_symbol=None)

    def test_exclusion_on_without_a_calendar_is_byte_identical(
            self, monkeypatch):
        # The exclusion needs a calendar to flag anything; with NO
        # calendar _in_earnings_window is always False, so the flags are
        # all-False and _iv_rank takes the unchanged raw-window branch.
        # Isolating the flag (both arms have NO calendar, so the earnings
        # book is idle in BOTH and is not the variable): turning the
        # exclusion ON must leave the full report byte-identical to the
        # exclusion-OFF run.
        off = self._run_no_calendar(
            monkeypatch, exclude_earnings_from_iv_rank=False)
        on = self._run_no_calendar(
            monkeypatch, exclude_earnings_from_iv_rank=True)
        assert (self._sha(self._full_fingerprint(on))
                == self._sha(self._full_fingerprint(off)))


# ----------------------------------------------------------------------
# skew_adjusted_iv (the module-level vol-surface primitive)
# ----------------------------------------------------------------------
class TestSkewAdjustedIV:
    SPOT = 100.0
    BASE = 0.30

    def test_atm_reproduces_base_iv_exactly(self):
        # k = ln(strike/spot) = 0 at the money -> base_iv UNCHANGED.
        assert skew_adjusted_iv(self.BASE, self.SPOT, self.SPOT, 0.5) == \
            self.BASE

    def test_positive_slope_is_the_equity_smirk(self):
        # POSITIVE slope: OTM puts (k<0, below spot) richer; OTM calls
        # (k>0, above spot) cheaper. IV is a DECREASING function of strike.
        put_wing = skew_adjusted_iv(self.BASE, self.SPOT, 85.0, 0.5)
        put_body = skew_adjusted_iv(self.BASE, self.SPOT, 95.0, 0.5)
        atm = skew_adjusted_iv(self.BASE, self.SPOT, 100.0, 0.5)
        call_body = skew_adjusted_iv(self.BASE, self.SPOT, 105.0, 0.5)
        call_wing = skew_adjusted_iv(self.BASE, self.SPOT, 115.0, 0.5)
        assert put_wing > put_body > atm > call_body > call_wing
        assert atm == self.BASE

    def test_matches_the_closed_form(self):
        strike, slope = 90.0, 0.5
        k = math.log(strike / self.SPOT)
        assert skew_adjusted_iv(self.BASE, self.SPOT, strike, slope) == \
            pytest.approx(self.BASE * (1.0 - slope * k), rel=1e-12)

    def test_floored_at_iv_floor_on_a_steep_wing(self):
        # A steep slope on a far OTM call would push IV non-positive; the
        # floor keeps Black-Scholes safe.
        iv = skew_adjusted_iv(0.10, self.SPOT, 1000.0, 5.0)
        assert iv == IV_FLOOR
        assert iv > 0.0

    def test_t_years_is_not_consumed_by_the_linear_form(self):
        # t_years is plumbed for future term structure but the linear
        # form ignores it — same IV regardless of t.
        a = skew_adjusted_iv(self.BASE, self.SPOT, 90.0, 0.5, t_years=0.05)
        b = skew_adjusted_iv(self.BASE, self.SPOT, 90.0, 0.5, t_years=2.0)
        assert a == b

    @pytest.mark.parametrize('slope', [None, 0.0])
    def test_opt_out_returns_base_iv_unchanged(self, slope):
        # slope None or 0.0 -> the flat surface -> base_iv verbatim for
        # every strike (the byte-identical default path).
        for strike in (80.0, 100.0, 130.0):
            assert skew_adjusted_iv(self.BASE, self.SPOT, strike, slope) == \
                self.BASE

    def test_determinism_pure_function_of_inputs(self):
        for _ in range(5):
            assert skew_adjusted_iv(self.BASE, self.SPOT, 88.0, 0.4) == \
                skew_adjusted_iv(self.BASE, self.SPOT, 88.0, 0.4)

    def test_non_positive_spot_or_strike_returns_base(self):
        # Defensive: degenerate inputs short-circuit to base_iv (no log
        # of a non-positive number).
        assert skew_adjusted_iv(self.BASE, 0.0, 90.0, 0.5) == self.BASE
        assert skew_adjusted_iv(self.BASE, 100.0, 0.0, 0.5) == self.BASE


# ----------------------------------------------------------------------
# Vol surface wired through the desk
# ----------------------------------------------------------------------
class TestDeskVolSurface:
    def base_desk(self, **kwargs):
        return make_desk(**kwargs)

    def test_condor_leg_ivs_are_strictly_skew_ordered_when_on(self):
        # The desk builds a condor as [SHORT put, BUY put wing, SHORT
        # call, BUY call wing]; with the surface on the four leg IVs must
        # be strictly ordered put_wing > put_body > call_body > call_wing
        # (equity skew) — the put wing prices RICHER than the call wing.
        desk = self.base_desk(vol_skew_slope=0.5)
        spot, base_iv, t_years = 100.0, 0.30, 35 / 365.0
        candidate = desk._build_condor('AAA', spot, base_iv, 35, 'vrp',
                                       '2022-03-01',
                                       PortfolioManager(100_000.0))
        assert candidate is not None
        ivs = {}
        for leg in candidate['legs']:
            asset = leg['asset']
            key = (leg['action'], asset.asset_type.value)
            ivs[key] = desk._surface_iv(base_iv, spot, asset.strike_price,
                                        t_years)
        put_wing = ivs[('BUY', 'put')]      # lowest strike
        put_body = ivs[('SHORT', 'put')]
        call_body = ivs[('SHORT', 'call')]
        call_wing = ivs[('BUY', 'call')]    # highest strike
        assert put_wing > put_body > call_body > call_wing
        # The put wing is RICHER than the call wing — the smirk.
        assert put_wing > call_wing
        # ATM base IV sits strictly between the two short legs.
        assert put_body > base_iv > call_body

    def test_legs_differ_by_strike_and_structure_stays_defined_risk(self):
        # Different strikes -> different IVs (the surface), but the
        # structure is still a bounded-loss iron condor.
        desk = self.base_desk(vol_skew_slope=0.5)
        spot, base_iv = 100.0, 0.30
        candidate = desk._build_condor('AAA', spot, base_iv, 35, 'vrp',
                                       '2022-03-01',
                                       PortfolioManager(100_000.0))
        strikes = [leg['asset'].strike_price for leg in candidate['legs']]
        assert len(set(strikes)) == 4  # four distinct strikes
        t_years = 35 / 365.0
        leg_ivs = {leg['asset'].strike_price:
                   desk._surface_iv(base_iv, spot, leg['asset'].strike_price,
                                    t_years)
                   for leg in candidate['legs']}
        assert len(set(leg_ivs.values())) == 4  # four distinct IVs
        # Defined risk: positive bounded max loss, >= 1 contract.
        assert candidate['max_loss'] > 0.0
        assert candidate['contracts'] >= 1
        assert len(candidate['legs']) == 4

    def test_surface_off_leg_ivs_equal_the_flat_base_iv(self):
        # Disabled (default): _surface_iv returns the flat base IV for
        # EVERY strike — the pre-surface pricing, byte-identical.
        desk = self.base_desk()  # vol_skew_slope None
        spot, base_iv, t_years = 100.0, 0.30, 35 / 365.0
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            assert desk._surface_iv(base_iv, spot, strike, t_years) == base_iv

    def test_surface_off_matches_flat_black_scholes_exactly(self):
        # With the surface off, a leg's mark equals flat-IV Black-Scholes
        # at the base IV (the literal pre-change pricing path).
        desk = self.base_desk()
        spot, base_iv, t_years, rate = 100.0, 0.30, 35 / 365.0, desk.rate
        strike = 90.0
        leg_iv = desk._surface_iv(base_iv, spot, strike, t_years)
        assert leg_iv == base_iv
        priced = black_scholes_price(spot, strike, t_years, leg_iv, rate,
                                     'put')
        flat = black_scholes_price(spot, strike, t_years, base_iv, rate,
                                   'put')
        assert priced == flat

    def test_surface_iv_is_deterministic(self):
        desk = self.base_desk(vol_skew_slope=0.4)
        t = 35 / 365.0
        first = [desk._surface_iv(0.25, 100.0, k, t)
                 for k in (85.0, 95.0, 105.0, 115.0)]
        second = [desk._surface_iv(0.25, 100.0, k, t)
                  for k in (85.0, 95.0, 105.0, 115.0)]
        assert first == second

    def test_negative_slope_inverts_the_skew(self):
        # A NEGATIVE slope is a (non-equity) reverse skew — OTM calls
        # richer. Asserts the sign plumbs through the desk faithfully.
        desk = self.base_desk(vol_skew_slope=-0.5)
        spot, base_iv, t = 100.0, 0.30, 35 / 365.0
        put_wing = desk._surface_iv(base_iv, spot, 85.0, t)
        call_wing = desk._surface_iv(base_iv, spot, 115.0, t)
        assert call_wing > base_iv > put_wing


class TestPriceOptionSurfaceSeam:
    """The engine's price_option hook and the desk's internal _surface_iv
    must agree on ONE surface so fills/marks never disagree with the
    desk's credit/greeks. price_option ON re-derives base IV from the
    model (same no-lookahead slice) then applies the same skew."""

    def frame(self, n=60):
        return vol_frame(n=n, shift_at=None)

    def _option(self, symbol, strike, expiry):
        from core.models import Asset
        return Asset(symbol, AssetType.PUT, strike_price=strike,
                     expiration_date=expiry)

    def test_price_option_on_equals_flat_when_skew_is_zero_at_atm(self):
        # At an ATM strike (k=0) the skew vanishes, so price_option with
        # the surface ON must equal the flat path EXACTLY — proves the two
        # code paths (module helper vs the desk's re-derivation) agree at
        # the seam.
        frame = self.frame()
        date = frame.index[40]
        spot = float(frame['close'].iloc[-1])  # used as the passed spot
        # ATM: strike == spot -> k = 0 -> skew is identity.
        expiry = (date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        from core.models import Asset
        asset = Asset('AAA', AssetType.CALL, strike_price=spot,
                      expiration_date=expiry)
        desk_off = make_desk()
        desk_on = make_desk(vol_skew_slope=0.7)
        price_off = desk_off.price_option(asset, frame, date, spot)
        price_on = desk_on.price_option(asset, frame, date, spot)
        assert price_off is not None
        assert price_on == pytest.approx(price_off, rel=1e-12)

    def test_price_option_on_otm_put_richer_otm_call_cheaper(self):
        # The surface lifts an OTM put and cheapens an OTM call vs the
        # flat model price (positive equity slope).
        frame = self.frame()
        date = frame.index[40]
        spot = float(frame['close'].iloc[-1])
        expiry = (date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        from core.models import Asset
        otm_put = Asset('AAA', AssetType.PUT, strike_price=spot * 0.9,
                        expiration_date=expiry)
        otm_call = Asset('AAA', AssetType.CALL, strike_price=spot * 1.1,
                         expiration_date=expiry)
        off = make_desk()
        on = make_desk(vol_skew_slope=0.6)
        put_flat = off.price_option(otm_put, frame, date, spot)
        put_skew = on.price_option(otm_put, frame, date, spot)
        call_flat = off.price_option(otm_call, frame, date, spot)
        call_skew = on.price_option(otm_call, frame, date, spot)
        assert put_skew > put_flat   # OTM put richer
        assert call_skew < call_flat  # OTM call cheaper

    def test_price_option_off_delegates_to_flat_model(self):
        # Surface off -> price_option returns the flat-model fair value
        # (the pre-change byte-identical path).
        from desks.options_pricing import price_option as module_price
        frame = self.frame()
        date = frame.index[40]
        spot = 101.0
        expiry = (date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        from core.models import Asset
        asset = Asset('AAA', AssetType.PUT, strike_price=95.0,
                      expiration_date=expiry)
        desk = make_desk()  # surface off
        desk_price = desk.price_option(asset, frame, date, spot)
        flat = module_price(asset, frame, date, spot, desk.iv_model,
                            rate=desk.rate)
        assert desk_price == flat

    def test_price_option_no_lookahead_base_iv_slices_before_date(self):
        # ON path still derives base IV from closes strictly before
        # `date` — appending the pricing day's bar must not change the
        # priced value (no lookahead leaks through the surface).
        frame = vol_frame(n=60, shift_at=None)
        date = frame.index[40]
        spot = 100.0
        expiry = (date + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        from core.models import Asset
        asset = Asset('AAA', AssetType.PUT, strike_price=92.0,
                      expiration_date=expiry)
        on = make_desk(vol_skew_slope=0.6)
        sliced_before = frame[frame.index < date]
        sliced_incl = frame[frame.index <= date]
        # The model slices to index < date internally, so the value
        # priced from the frame INCLUDING today's bar equals the value
        # priced from the frame WITHOUT it (today's close leaks nothing).
        with_today = on.price_option(asset, sliced_incl, date, spot)
        without_today = on.price_option(asset, sliced_before, date, spot)
        assert with_today == without_today


# ----------------------------------------------------------------------
# IV-rank earnings exclusion
# ----------------------------------------------------------------------
class TestIVRankEarningsExclusion:
    CALENDAR = SyntheticEarningsCalendar({'AAA': ['2022-05-02']})

    def _seed(self, desk, symbol, history, flags):
        desk._iv_history[symbol] = list(history)
        desk._iv_earnings_flags[symbol] = list(flags)

    def test_exclusion_changes_the_rank_vs_unfiltered(self):
        # Constructed window: 40 low NON-earnings obs, then 30 HIGH
        # earnings-window spikes (flagged), then TODAY a high-but-below-
        # spike non-earnings value. Unfiltered, the spikes sit ABOVE today
        # and suppress its rank; dropping them ranks today at the TOP of
        # the non-earnings regime — the two ranks differ sharply.
        history = [0.10] * 40 + [0.99] * 30 + [0.60]
        flags = [False] * 40 + [True] * 30 + [False]
        off = make_desk(iv_rank_window=252,
                        exclude_earnings_from_iv_rank=False)
        on = make_desk(iv_rank_window=252, earnings_calendar=self.CALENDAR,
                       exclude_earnings_from_iv_rank=True)
        self._seed(off, 'AAA', history, flags)
        self._seed(on, 'AAA', history, flags)
        unfiltered = off._iv_rank('AAA')
        filtered = on._iv_rank('AAA')
        # 41 of 71 obs (the 40 lows + today) are <= today=0.60 -> 57.7%.
        assert unfiltered == pytest.approx(100.0 * 41 / 71)
        # The 30 spikes dropped -> 41 non-earnings obs, all <= today -> 100.
        assert filtered == pytest.approx(100.0)
        assert filtered != unfiltered

    def test_flagged_count_matches_the_earnings_window(self):
        # The flag count over a driven series equals the number of
        # observation dates that fell within iv_rank_earnings_window_days
        # calendar days before the scheduled earnings (inclusive of E).
        earnings = pd.Timestamp('2022-05-02').date()
        calendar = SyntheticEarningsCalendar({'AAA': [earnings]})
        frames = {'AAA': vol_frame(n=120, shift_at=None)}
        desk = make_desk(earnings_calendar=calendar,
                         exclude_earnings_from_iv_rank=True,
                         iv_rank_earnings_window_days=5)
        drive(desk, frames, frames['AAA'].index,
              PortfolioManager(100_000.0))
        flags = desk._iv_earnings_flags['AAA']
        history = desk._iv_history['AAA']
        assert len(flags) == len(history)  # parallel lists, one-for-one
        # Recompute the expected window membership independently: the
        # observation dates are the frame dates that produced an IV
        # (HV20 needs 21 prior closes, so the first 21 produce none).
        obs_dates = [d for d in frames['AAA'].index
                     if desk.iv_model.iv('AAA', frames['AAA'][
                         frames['AAA'].index <= d], d) is not None]
        expected = []
        for d in obs_dates:
            days_to_e = (earnings - d.date()).days
            expected.append(0 <= days_to_e <= 5)
        assert flags == expected
        assert any(flags)  # the window is actually exercised
        assert sum(flags) == sum(expected)

    def test_off_and_no_calendar_rank_is_byte_identical(self):
        # Feature off OR no calendar -> the comparison set is the raw
        # trailing window -> the rank equals the unfiltered percentile
        # exactly (the byte-identical prior behavior).
        history = [0.10] * 30 + [0.50] * 10 + [0.80]
        flags = [True] * 41  # flags exist but must be IGNORED when off
        # Feature OFF: flags ignored.
        off = make_desk(iv_rank_window=60,
                        exclude_earnings_from_iv_rank=False)
        self._seed(off, 'AAA', history, flags)
        raw = 100.0 * sum(1 for v in history if v <= history[-1]) \
            / len(history)
        assert off._iv_rank('AAA') == raw
        # Feature ON but NO calendar: _in_earnings_window is always False,
        # but even if seeded flags were True the OFF semantics are what a
        # no-calendar desk would have produced — assert against a desk
        # driven with no calendar so its flags are genuinely all-False.
        on_no_cal = make_desk(iv_rank_window=60,
                              exclude_earnings_from_iv_rank=True)
        self._seed(on_no_cal, 'AAA', history, [False] * 41)
        assert on_no_cal._iv_rank('AAA') == raw

    def test_thin_non_earnings_sample_returns_none(self):
        # Backend caveat (3): the min-obs gate counts NON-earnings
        # observations. A window dominated by earnings flags can have
        # fewer than iv_rank_min_obs survivors -> None (never rank on a
        # thin post-exclusion sample), even though the raw series clears
        # the gate.
        history = [0.90] * 20 + [0.10] * 20 + [0.50]  # 41 obs
        flags = [True] * 20 + [False] * 20 + [False]  # 21 non-earnings
        on = make_desk(iv_rank_min_obs=30, iv_rank_window=60,
                       earnings_calendar=self.CALENDAR,
                       exclude_earnings_from_iv_rank=True)
        self._seed(on, 'AAA', history, flags)
        # Raw series has 41 obs >= 30 -> the unfiltered desk DOES rank.
        off = make_desk(iv_rank_min_obs=30, iv_rank_window=60,
                        exclude_earnings_from_iv_rank=False)
        self._seed(off, 'AAA', history, flags)
        assert off._iv_rank('AAA') is not None
        # Filtered: only 21 non-earnings obs < 30 -> None.
        assert on._iv_rank('AAA') is None

    def test_today_stays_the_reference_even_when_flagged(self):
        # If TODAY itself is flagged it is still the reference value (the
        # question is where today sits within the NON-earnings regime).
        history = [0.10] * 35 + [0.90]  # today is the 0.90 spike
        flags = [False] * 35 + [True]   # today flagged
        on = make_desk(iv_rank_min_obs=30, iv_rank_window=60,
                       earnings_calendar=self.CALENDAR,
                       exclude_earnings_from_iv_rank=True)
        self._seed(on, 'AAA', history, flags)
        # comparison drops today (flagged) -> 35 lows; today 0.90 > all of
        # them -> 100% rank over the non-earnings set.
        assert on._iv_rank('AAA') == pytest.approx(100.0)

    def test_no_lookahead_flag_frozen_at_observation_time(self):
        # The flag at date D is computed from D + the pre-announced
        # schedule ONLY. Appending FUTURE bars must not change a past
        # observation's flag. Drive a prefix, snapshot the flags, drive
        # the rest, and assert the prefix flags are unchanged.
        earnings = pd.Timestamp('2022-05-02').date()
        calendar = SyntheticEarningsCalendar({'AAA': [earnings]})
        frames = {'AAA': vol_frame(n=120, shift_at=None)}
        dates = frames['AAA'].index
        desk = make_desk(earnings_calendar=calendar,
                         exclude_earnings_from_iv_rank=True,
                         iv_rank_earnings_window_days=5)
        loc = dates.get_loc(pd.Timestamp('2022-05-02'))
        # Drive only up to and including the earnings date.
        drive(desk, frames, dates[:loc + 1], PortfolioManager(100_000.0))
        prefix_flags = list(desk._iv_earnings_flags['AAA'])
        prefix_len = len(prefix_flags)
        assert any(prefix_flags)  # the window was crossed in the prefix
        # Now drive the REST (all future bars).
        drive(desk, frames, dates[loc + 1:], PortfolioManager(100_000.0))
        # The already-recorded flags are frozen — future bars never
        # retroactively rewrite a past observation's earnings flag.
        assert desk._iv_earnings_flags['AAA'][:prefix_len] == prefix_flags

    def test_in_earnings_window_boundary_and_off_default(self):
        # _in_earnings_window spot-checks: inclusive 0..window before E;
        # False outside, after E (next_earnings None), and ALWAYS False
        # when the feature is off or no calendar is present.
        earnings = pd.Timestamp('2022-05-02').date()
        calendar = SyntheticEarningsCalendar({'AAA': [earnings]})
        on = make_desk(earnings_calendar=calendar,
                       exclude_earnings_from_iv_rank=True,
                       iv_rank_earnings_window_days=5)
        # On E itself: days_to_e == 0 -> in window.
        assert on._in_earnings_window('AAA', '2022-05-02') is True
        # 5 calendar days before (inclusive boundary) -> in window.
        assert on._in_earnings_window('AAA', '2022-04-27') is True
        # 6 calendar days before -> outside.
        assert on._in_earnings_window('AAA', '2022-04-26') is False
        # After E (no upcoming earnings) -> False.
        assert on._in_earnings_window('AAA', '2022-05-03') is False
        # Far before -> False.
        assert on._in_earnings_window('AAA', '2022-01-10') is False
        # Feature OFF -> always False even inside the window.
        off = make_desk(earnings_calendar=calendar,
                        exclude_earnings_from_iv_rank=False)
        assert off._in_earnings_window('AAA', '2022-05-02') is False
        # ON but no calendar -> always False.
        no_cal = make_desk(exclude_earnings_from_iv_rank=True)
        assert no_cal._in_earnings_window('AAA', '2022-05-02') is False


# ----------------------------------------------------------------------
# VRP entry skip inside the earnings window (Fix #3)
# ----------------------------------------------------------------------
class TestVRPSkipsEarningsWindowEntry:
    """Fix #3, in _structure_entries (the VRP book): when
    exclude_earnings_from_iv_rank is ON, the VRP book SKIPS a new
    short-condor entry for any symbol whose TODAY is inside the earnings
    window — `if self._in_earnings_window(symbol, date): continue` is
    checked BEFORE the _iv_rank trigger. Today's IV is itself
    earnings-bumped, so a high rank vs the calm regime would otherwise
    sell premium INTO the pre-earnings run-up; that vol belongs to the
    earnings book.

    DEFAULT path is byte-identical: _in_earnings_window returns False when
    the feature is off or no calendar is present, so the skip is a no-op
    (the greeks golden — TestGreeksGoldenByteIdentity — is unchanged).

    These drive the REAL _structure_entries VRP path (not a seeded
    _iv_rank). The IV model is injected WITHOUT a calendar so there is no
    earnings IV bump to muddy the comparison; the earnings BOOK weight is
    zeroed so the only thing the desk's calendar drives here is the
    _in_earnings_window skip. The vol_frame jumps late (shift_at=100) so
    today's IV ranks at the top of the calm trailing window and the VRP
    trigger fires day after day from late May on — giving in-window days
    that WOULD enter absent the skip. Offline, seeded, deterministic.
    """

    N = 120
    SHIFT = 100  # late vol jump -> today ranks high -> VRP triggers
    # 2022-05-25 is the FIRST day the VRP IV-rank trigger fires on this
    # frame (verified below by the feature-off control), and earnings on
    # 2022-05-27 puts it (and 05-27) inside the 5-calendar-day window.
    EARNINGS = '2022-05-27'
    WINDOW = {'2022-05-23', '2022-05-24', '2022-05-25', '2022-05-26',
              '2022-05-27'}

    def _frames(self):
        return {'AAA': vol_frame(n=self.N, shift_at=self.SHIFT)}

    def _entry_days(self, exclude, calendar):
        """Drive the desk over the full frame; return the SET of dates
        (str) on which the VRP book emitted option (condor) intents. The
        IV model carries no calendar (no IV bump) and the earnings book
        weight is 0, so the desk's `calendar` only feeds
        _in_earnings_window."""
        frames = self._frames()
        kwargs = dict(iv_rank_window=252, iv_rank_min_obs=30,
                      exclude_earnings_from_iv_rank=exclude,
                      iv_model=SyntheticIVModel(),  # no calendar -> no bump
                      book_weights={'earnings': 0.0, 'vrp': 0.5,
                                    'relative_value': 0.2})
        if calendar is not None:
            kwargs['earnings_calendar'] = calendar
        desk = make_desk(**kwargs)
        intents_by_date = drive(desk, frames, frames['AAA'].index,
                                PortfolioManager(100_000.0))
        return {d.strftime('%Y-%m-%d')
                for d, intents in intents_by_date.items()
                if option_intents(intents)}

    def test_feature_off_control_enters_inside_the_window(self):
        # The control that makes the skip meaningful: with the feature
        # OFF (but the SAME calendar present), the VRP book DOES open
        # condors on days that sit inside the earnings window — proving
        # the data/trigger would otherwise enter there.
        calendar = SyntheticEarningsCalendar({'AAA': [self.EARNINGS]})
        off_days = self._entry_days(exclude=False, calendar=calendar)
        entered_in_window = off_days & self.WINDOW
        assert entered_in_window  # at least one in-window entry
        assert '2022-05-25' in entered_in_window  # the first VRP day

    def test_feature_on_skips_every_in_window_entry(self):
        # Fix #3: ON, the VRP book opens NO condor on any in-window day.
        # The first entry slides to the first VRP-trigger day AFTER the
        # window closes (one-open-per-symbol gating shifts the cadence,
        # but the invariant is simply: zero in-window entries).
        calendar = SyntheticEarningsCalendar({'AAA': [self.EARNINGS]})
        on_days = self._entry_days(exclude=True, calendar=calendar)
        assert not (on_days & self.WINDOW)  # nothing entered in the window
        assert on_days  # the book is NOT dead — it still trades post-window
        assert min(on_days) > '2022-05-27'  # first entry is after the window

    def test_default_off_path_is_unaffected_by_the_skip(self):
        # The skip must be invisible on the DEFAULT path. With NO calendar
        # (so _in_earnings_window is always False), turning the exclusion
        # ON yields the EXACT same VRP entry days as the feature OFF —
        # byte-identical entry cadence, no skip.
        off_no_cal = self._entry_days(exclude=False, calendar=None)
        on_no_cal = self._entry_days(exclude=True, calendar=None)
        assert on_no_cal == off_no_cal
        assert off_no_cal & self.WINDOW  # and it really does enter in window

    def test_single_day_skip_is_gated_purely_by_the_window(self):
        # Tightest isolation: feature ON in BOTH arms; the ONLY thing that
        # differs is whether 2022-05-25 is inside the window. With earnings
        # on 05-27 the day IS in the window -> VRP skips (0 option intents);
        # with earnings far off (07-01) the SAME day is OUT of the window
        # -> the VRP condor enters (4 legs). Proves the skip is driven by
        # _in_earnings_window, checked BEFORE the _iv_rank trigger.
        frames = self._frames()
        dates = frames['AAA'].index
        target = pd.Timestamp('2022-05-25')
        loc = dates.get_loc(target)

        def drive_to_target(calendar):
            desk = make_desk(
                iv_rank_window=252, iv_rank_min_obs=30,
                exclude_earnings_from_iv_rank=True,
                iv_model=SyntheticIVModel(),
                earnings_calendar=calendar,
                book_weights={'earnings': 0.0, 'vrp': 0.5,
                              'relative_value': 0.2})
            intents_by_date = drive(desk, frames, dates[:loc + 1],
                                    PortfolioManager(100_000.0))
            return desk, option_intents(intents_by_date[target])

        in_cal = SyntheticEarningsCalendar({'AAA': ['2022-05-27']})
        out_cal = SyntheticEarningsCalendar({'AAA': ['2022-07-01']})
        desk_in, legs_in = drive_to_target(in_cal)
        desk_out, legs_out = drive_to_target(out_cal)
        # Sanity: the window membership is the only differing condition.
        assert desk_in._in_earnings_window('AAA', target) is True
        assert desk_out._in_earnings_window('AAA', target) is False
        # In the window -> skipped; out of the window -> the condor enters.
        assert legs_in == []
        assert len(legs_out) == 4


# ----------------------------------------------------------------------
# Construction / parameter surface
# ----------------------------------------------------------------------
class TestPhaseEConstruction:
    def test_defaults_are_off(self):
        desk = JaneStreetDesk()
        assert desk.vol_skew_slope is None
        assert desk.exclude_earnings_from_iv_rank is False
        assert desk.iv_rank_earnings_window_days == 5

    def test_params_are_stored_as_attrs(self):
        desk = JaneStreetDesk(vol_skew_slope=0.4,
                              exclude_earnings_from_iv_rank=True,
                              iv_rank_earnings_window_days=7)
        assert desk.vol_skew_slope == 0.4
        assert desk.exclude_earnings_from_iv_rank is True
        assert desk.iv_rank_earnings_window_days == 7

    def test_status_is_byte_identical_on_the_default_path(self):
        # get_status adds Phase E keys ONLY when a feature is enabled, so
        # the default status dict is unchanged from the pre-feature shape.
        default_status = JaneStreetDesk().get_status()
        assert 'vol_skew_slope' not in default_status
        assert 'exclude_earnings_from_iv_rank' not in default_status
        assert 'iv_rank_earnings_window_days' not in default_status

    def test_status_surfaces_features_only_when_enabled(self):
        skew = JaneStreetDesk(vol_skew_slope=0.4).get_status()
        assert skew['vol_skew_slope'] == 0.4
        assert 'exclude_earnings_from_iv_rank' not in skew
        excl = JaneStreetDesk(exclude_earnings_from_iv_rank=True,
                              iv_rank_earnings_window_days=8).get_status()
        assert excl['exclude_earnings_from_iv_rank'] is True
        assert excl['iv_rank_earnings_window_days'] == 8
        assert 'vol_skew_slope' not in excl


# ----------------------------------------------------------------------
# Recoverability / determinism with features ON
# ----------------------------------------------------------------------
class TestRecoverabilityAndDeterminism:
    N_DAYS = 160
    EARNINGS_IDX = 40

    def universe(self):
        return {'AAA': vol_frame(n=self.N_DAYS, shift_at=SHIFT),
                'BBB': vol_frame(n=self.N_DAYS, shift_at=SHIFT)}

    def _run(self, monkeypatch, **desk_kwargs):
        universe = self.universe()
        dates = universe['AAA'].index
        earnings = dates[self.EARNINGS_IDX].date()

        def fake_fetch(self, symbol, start_date, end_date):
            return universe.get(symbol, pd.DataFrame())

        monkeypatch.setattr(MarketDataHandler, 'fetch_stock_data',
                            fake_fetch)
        desk = make_desk(
            earnings_calendar=SyntheticEarningsCalendar({'BBB': [earnings]}),
            **desk_kwargs)
        engine = BacktestEngine(desk=desk, initial_capital=100_000.0)
        return engine.run(sorted(universe), '2022-01-01', '2022-12-31',
                          benchmark_symbol=None)

    @staticmethod
    def _fingerprint(report):
        return json.dumps({
            'structures': report['structures'],
            'greeks_series': report['greeks_series'],
            'trader_notes': report['trader_notes'],
            'closed_trades': report['closed_trades'],
            'summary': report['summary'],
        }, sort_keys=True, default=str)

    def test_skew_on_is_deterministic_across_runs(self, monkeypatch):
        first = self._fingerprint(self._run(monkeypatch, vol_skew_slope=0.5))
        assert self._fingerprint(
            self._run(monkeypatch, vol_skew_slope=0.5)) == first

    def test_exclusion_on_is_deterministic_across_runs(self, monkeypatch):
        first = self._fingerprint(
            self._run(monkeypatch, exclude_earnings_from_iv_rank=True))
        assert self._fingerprint(self._run(
            monkeypatch, exclude_earnings_from_iv_rank=True)) == first

    def test_skew_on_actually_moves_the_greeks_series(self, monkeypatch):
        # Sanity that the ON path is NOT a no-op (so the byte-identical
        # default is a real opt-out, not the only behavior): with the
        # skew on, the greeks_series differs from the default golden.
        on = json.dumps(self._run(monkeypatch, vol_skew_slope=0.5)
                        ['greeks_series'], sort_keys=True, default=str)
        off = json.dumps(self._run(monkeypatch)['greeks_series'],
                         sort_keys=True, default=str)
        assert on != off

    def test_both_features_at_defaults_match_the_plain_default_run(
            self, monkeypatch):
        # The opt-out guarantee end-to-end: constructing the desk with
        # BOTH features explicitly defaulted yields the byte-identical
        # report a desk with NO Phase E kwargs produces.
        explicit = self._fingerprint(self._run(
            monkeypatch, vol_skew_slope=None,
            exclude_earnings_from_iv_rank=False,
            iv_rank_earnings_window_days=5))
        plain = self._fingerprint(self._run(monkeypatch))
        assert explicit == plain
