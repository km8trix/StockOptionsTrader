"""
Options fair-value engine — Black-Scholes pricing with a synthetic IV model.

==========================================================================
PRICING REALITY (read this before trusting any backtest number)
==========================================================================
Free data providers carry NO historical options chains, so every BACKTEST
option price in this system is SYNTHETIC: Black-Scholes computed from the
underlying's OHLCV with implied volatility MODELED from realized
volatility (HV20 x a variance-risk-premium multiplier, plus an
earnings-event bump when a calendar is provided). This is a clearly
labeled approximation good for validating desk LOGIC — entries, exits,
sizing, Greeks plumbing — not for measuring real-world options edge.
Real chain pricing arrives in Phase 9 via E*TRADE quotes.

NO-LOOKAHEAD RULE for synthetic pricing:
    * Volatility inputs use closes STRICTLY BEFORE the pricing date — a
      fill at the open cannot know that day's close (it hasn't printed).
      SyntheticIVModel.iv() therefore slices the underlying frame to
      index < date before computing HV20.
    * Spot is passed in by the CALLER: the fill-day OPEN for fills, the
      session CLOSE for end-of-day mark-to-market.

CONVENTIONS (used everywhere in this module and its consumers):
    * t_years is CALENDAR days to expiry / 365.
    * theta is the price decay PER CALENDAR DAY (annual theta / 365).
    * vega is the price change PER 1.00 VOL POINT (e.g. sigma 0.20 ->
      1.20), NOT per 1%.
    * rate defaults to 0.045 (risk-free, annualized, continuous).
    * t <= 0 is valued at intrinsic with ZERO greeks except delta, which
      is 0 or +/-1 depending on moneyness (settlement delta).
"""

from __future__ import annotations

import logging
import math
from datetime import date as date_type
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.models import Asset, AssetType

logger = logging.getLogger(__name__)

#: Annualized IV floor — synthetic IV never drops below this.
IV_FLOOR = 0.05

#: Rolling window (sessions) for the realized-vol estimate behind the IV.
HV_WINDOW = 20

#: Default annualized risk-free rate for Black-Scholes.
DEFAULT_RATE = 0.045


def _validate_right(right: str) -> str:
    right = right.lower()
    if right not in ('call', 'put'):
        raise ValueError(f"right must be 'call' or 'put', got {right!r}")
    return right


def black_scholes_price(spot: float, strike: float, t_years: float,
                        vol: float, rate: float = DEFAULT_RATE,
                        right: str = 'call') -> float:
    """European Black-Scholes price (module conventions apply).

    t_years <= 0 returns intrinsic value. vol <= 0 with t > 0 also
    degenerates to discounted intrinsic on the forward (handled by the
    closed form via a tiny vol floor would distort greeks, so it is
    valued explicitly as intrinsic of the discounted strike).
    """
    right = _validate_right(right)
    if t_years <= 0:
        if right == 'call':
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)
    if vol <= 0:
        disc_strike = strike * math.exp(-rate * t_years)
        if right == 'call':
            return max(0.0, spot - disc_strike)
        return max(0.0, disc_strike - spot)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike)
          + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if right == 'call':
        price = (spot * norm.cdf(d1)
                 - strike * math.exp(-rate * t_years) * norm.cdf(d2))
    else:
        price = (strike * math.exp(-rate * t_years) * norm.cdf(-d2)
                 - spot * norm.cdf(-d1))
    return float(max(price, 0.0))


def black_scholes_greeks(spot: float, strike: float, t_years: float,
                         vol: float, rate: float = DEFAULT_RATE,
                         right: str = 'call') -> Dict[str, float]:
    """Analytic Black-Scholes price + per-share greeks.

    Returns {'price', 'delta', 'gamma', 'theta', 'vega'} where theta is
    PER CALENDAR DAY and vega is PER 1.00 VOL POINT (module conventions).

    t <= 0 (or vol <= 0): intrinsic price, zero greeks except delta,
    which is the settlement delta — calls: 1.0 if spot > strike else
    0.0; puts: -1.0 if spot < strike else 0.0 (at-the-money exactly:
    0.0, the option settles worthless).
    """
    right = _validate_right(right)
    price = black_scholes_price(spot, strike, t_years, vol, rate, right)
    if t_years <= 0 or vol <= 0:
        if right == 'call':
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return {'price': price, 'delta': delta, 'gamma': 0.0,
                'theta': 0.0, 'vega': 0.0}

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike)
          + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t  # per 1.00 vol point
    if right == 'call':
        delta = norm.cdf(d1)
        theta_annual = (-(spot * pdf_d1 * vol) / (2.0 * sqrt_t)
                        - rate * strike * math.exp(-rate * t_years)
                        * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1.0
        theta_annual = (-(spot * pdf_d1 * vol) / (2.0 * sqrt_t)
                        + rate * strike * math.exp(-rate * t_years)
                        * norm.cdf(-d2))
    return {
        'price': price,
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta_annual / 365.0),  # per CALENDAR day
        'vega': float(vega),
    }


# ----------------------------------------------------------------------
# Earnings calendars
# ----------------------------------------------------------------------
@runtime_checkable
class EarningsCalendar(Protocol):
    """Protocol: next_earnings(symbol, date) -> date | None.

    Returns the next earnings date ON OR AFTER `date` for `symbol`, or
    None when no upcoming earnings date is known.
    """

    def next_earnings(self, symbol: str, date) -> Optional[date_type]:
        ...


class SyntheticEarningsCalendar:
    """Constructor-injected earnings calendar for tests and backtests.

    Takes {symbol: [dates]}; dates may be strings, datetimes or
    pd.Timestamps and are normalized to datetime.date and sorted.
    """

    def __init__(self, schedule: Dict[str, List]):
        self._schedule: Dict[str, List[date_type]] = {
            symbol: sorted(pd.Timestamp(d).date() for d in dates)
            for symbol, dates in schedule.items()
        }

    def next_earnings(self, symbol: str, date) -> Optional[date_type]:
        """Earliest scheduled earnings date >= `date`, else None."""
        current = pd.Timestamp(date).date()
        for earnings_date in self._schedule.get(symbol, ()):
            if earnings_date >= current:
                return earnings_date
        return None


class OpenBBEarningsCalendar:
    """STUB: OpenBB-backed earnings calendar for later LIVE use.

    Guarded and lazy: openbb is imported only inside next_earnings, and
    any import/network failure degrades to None (no upcoming earnings)
    with a WARNING — backtests and tests must NEVER take this path (no
    network in tests; inject SyntheticEarningsCalendar instead). Wired
    for real in Phase 9 alongside E*TRADE chain quotes.
    """

    def next_earnings(self, symbol: str, date) -> Optional[date_type]:
        try:  # pragma: no cover — live-only path, never hit offline
            from openbb import obb  # lazy: never imported at module load
            current = pd.Timestamp(date).date()
            results = obb.equity.calendar.earnings(symbol=symbol).results
            dates = sorted(pd.Timestamp(r.report_date).date()
                           for r in results if r.report_date is not None)
            for earnings_date in dates:
                if earnings_date >= current:
                    return earnings_date
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("OpenBB earnings calendar unavailable for %s: %s",
                           symbol, exc)
            return None


# ----------------------------------------------------------------------
# Synthetic IV model
# ----------------------------------------------------------------------
class SyntheticIVModel:
    """Models implied volatility from realized volatility (PRICING REALITY).

    iv(symbol, underlying_frame, date) =
        HV20 (annualized 20-day realized vol of closes STRICTLY BEFORE
        `date` — calculate_volatility semantics on the sliced frame:
        pct_change().rolling(20).std() * sqrt(252) at the last row)
        x vrp_multiplier (the variance risk premium: options habitually
        trade above realized vol; default 1.10)
        x the EARNINGS EVENT BUMP when a calendar is provided
        floored at IV_FLOOR (0.05).

    EARNINGS EVENT BUMP — the IV run-up/crush phenomenon the earnings
    book harvests: with an upcoming earnings date E (date <= E),

        iv *= 1 + earnings_bump * exp(-days_to_E / 3.0)

    The bump RISES monotonically into the event (days_to_E shrinking ->
    exp term growing toward 1, so the multiplier peaks at
    1 + earnings_bump on the event date itself) and is GONE the first
    day after E — the crush. The 3-day e-folding time concentrates the
    run-up in the final week, matching the empirical pattern of
    pre-earnings IV inflation followed by an overnight collapse.

    Returns None when fewer than HV_WINDOW + 1 closes exist strictly
    before `date` (HV20 needs 20 returns = 21 closes).
    """

    def __init__(self, vrp_multiplier: float = 1.10,
                 earnings_calendar: Optional[EarningsCalendar] = None,
                 earnings_bump: float = 0.40):
        if vrp_multiplier <= 0:
            raise ValueError(
                f"vrp_multiplier must be > 0, got {vrp_multiplier}")
        if earnings_bump < 0:
            raise ValueError(
                f"earnings_bump must be >= 0, got {earnings_bump}")
        self.vrp_multiplier = vrp_multiplier
        self.earnings_calendar = earnings_calendar
        self.earnings_bump = earnings_bump

    def iv(self, symbol: str, underlying_frame: pd.DataFrame,
           date) -> Optional[float]:
        """Annualized synthetic IV at `date` (class docstring), or None
        when there is not enough PRIOR history to estimate HV20."""
        if underlying_frame is None or underlying_frame.empty \
                or 'close' not in underlying_frame.columns:
            return None
        # NO-LOOKAHEAD: closes STRICTLY BEFORE the pricing date only —
        # a fill at the open cannot see the fill day's close.
        closes = underlying_frame.loc[
            underlying_frame.index < pd.Timestamp(date), 'close'].dropna()
        if len(closes) < HV_WINDOW + 1:
            return None
        hv20 = float(closes.pct_change()
                     .rolling(window=HV_WINDOW).std().iloc[-1]
                     * np.sqrt(252))
        if not np.isfinite(hv20):
            return None

        iv = hv20 * self.vrp_multiplier
        if self.earnings_calendar is not None:
            current = pd.Timestamp(date).date()
            earnings = self.earnings_calendar.next_earnings(symbol, current)
            if earnings is not None and current <= earnings:
                days_to_e = (earnings - current).days
                iv *= 1.0 + self.earnings_bump * math.exp(-days_to_e / 3.0)
        return max(IV_FLOOR, iv)


def price_option(asset: Asset, underlying_frame: pd.DataFrame, date,
                 spot: float, iv_model: SyntheticIVModel,
                 rate: float = DEFAULT_RATE) -> Optional[float]:
    """Synthetic fair value of an option Asset at `date` (PRICING REALITY).

    Vol comes from iv_model (which slices the frame to index < date per
    the no-lookahead rule); spot is passed in by the CALLER — fill-day
    OPEN for fills, session CLOSE for mark-to-market. Returns None when
    the IV model has insufficient prior history.
    """
    if asset.asset_type not in (AssetType.CALL, AssetType.PUT):
        raise ValueError(f"price_option needs an option asset, got {asset}")
    if asset.strike_price is None or asset.expiration_date is None:
        raise ValueError(f"option asset missing strike/expiry: {asset}")

    expiry = pd.Timestamp(asset.expiration_date).date()
    t_years = (expiry - pd.Timestamp(date).date()).days / 365.0
    if t_years <= 0:
        # Expired (or expiring today): intrinsic, no vol needed.
        return black_scholes_price(spot, asset.strike_price, 0.0, 0.0,
                                   rate, asset.asset_type.value)

    vol = iv_model.iv(asset.symbol, underlying_frame, date)
    if vol is None:
        return None
    return black_scholes_price(spot, asset.strike_price, t_years, vol,
                               rate, asset.asset_type.value)
