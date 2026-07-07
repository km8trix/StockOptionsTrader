"""
Jane Street Desk — fair-value-driven defined-risk premium selling.

==========================================================================
PRICING REALITY: every option price this desk trades against in a
backtest is SYNTHETIC (Black-Scholes on the underlying's OHLCV with IV
modeled from realized vol — see desks/options_pricing's banner and the
no-lookahead rule). The desk validates LOGIC; real chain pricing arrives
in Phase 9 via E*TRADE quotes.
==========================================================================

Books (capital weights are fractions of the DESK'S capital):

    vrp (0.5) — the variance-risk-premium book. Per underlying, daily:
        the synthetic IV series (the SAME SyntheticIVModel the fills
        price with) is accumulated; IV-rank = percentile of today's IV
        within the trailing 252 observations (>= 126 required). ENTER a
        short iron condor (vrp_dte_days, default 35, in the 30-45 DTE
        band) when IV-rank > 60 AND the regime model is fitted AND the
        regime != high_vol AND no open structure exists on that
        underlying. EXITS: StructureTracker checks daily (profit target
        50% of credit / stop 2x credit / time exit DTE <= 21, inclusive
        boundaries) + REGIME FLATTEN: the day the regime becomes
        high_vol, ALL vrp structures close ('regime_flatten').

    earnings (0.3) — the IV-crush book. With an injected calendar, a
        short iron condor (earnings_dte_days, default 21, in the 14-30
        band) is entered exactly 2 TRADING DAYS before each earnings
        date (np.busday_count(date, E) == 2 — trading-day distance is
        approximated with the weekday calendar, which matches the
        business-day frames the backtests run on), same sizing rule,
        independent of IV-rank but still regime-gated. Exit on the FIRST
        trading day after E ('time_exit'), noting the crush capture with
        the model's entry/exit IV. The standard tracker checks do NOT
        run on earnings structures (a 21-DTE entry would trip the
        DTE<=21 time exit immediately); the post-E exit and the engine's
        expiry backstop own them. No calendar -> the book is idle with
        one 'info' note at the start.

    relative_value (0.2) — stock legs only, reusing the pairs mechanics:
        a designated etf_symbol (constructor; DEFAULT: the first symbol
        alphabetically — a deliberate, documented convention so default
        runs are deterministic) against the equal-weight basket of the
        rest. spread = log(etf) - mean(log(basket closes)); z over a
        rolling 60d window. |z| > 2 -> long the cheap side / short the
        rich side (stock SHORT support, cash-account approximation);
        exit |z| < 0.5. Legs sized like the Renaissance pairs legs: the
        ETF side gets half the book weight — clamped STRICTLY INSIDE
        the RiskManager's position-size cap (at registry defaults
        0.2 * 0.5 sits exactly ON the 10% cap and floating-point drift
        in the ratio check blocked every entry) — and the basket half
        mirrors the ETF half, split equally across its symbols.

NOTHING trades before the regime model's first fit (Renaissance
precedent: an ungated desk would trade the warm-up period on no
information). The regime controller runs 252/21/120 through
WalkForwardController; fits are tagged 'regime' in walk_forward_fits.

SIZING RULE (desks/structures): contracts = floor(risk_budget /
max_loss_per_contract) with risk_budget = 0.02 * desk_capital *
book_weight; 0 contracts -> skip with an 'info' note.

GREEKS + THE CENTRAL RISK BOOK: daily desk-level portfolio Greeks
(contract C12; convention documented on greeks_series) are computed from
open option positions; the desk registers a 'short_vega' limit on its
CentralRiskBook — NEW premium-selling structures are blocked when the
post-trade |portfolio vega| would exceed vega_limit (default 50,000 per
$100k of desk capital); closes always pass. UNITS: vega_limit is in the
SAME share-equivalent per-1.00-vol-point units as greeks_series — 50,000
per $100k equals $500 per 1% vol point per $100k. One 2%-risk-budget
condor carries net vega around -1,800 in these units, so a per-1%-style
number (e.g. 500) as the default would block EVERY entry at any capital
size (Phase 8 validation regression: both the limit and a structure's
vega scale linearly with capital). The whole structure is evaluated
atomically (its net vega is stamped on the first short leg) — blocking
single legs of a defined-risk structure would leave naked wings.

CONTRACTS: C11 (structures), C12 (greeks_series), C14 (every note
carries data.book in {'regime','vrp','earnings','relative_value'} —
including the SHARED Desk.apply_risk notes, tagged via the
risk_note_data hook; structure lifecycle notes additionally carry
data.structure_id + data.structure), C15 (registry: 'ready', accent
'#d29922').

MARGIN IS NOT MODELED: short options and stock shorts use the
cash-account approximation (proceeds held as cash); live trading is
gated to Phase 9.
"""

from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import Asset, AssetType
from desks.base import Desk, DeskIntent
from desks.options_pricing import (DEFAULT_RATE, EarningsCalendar,
                                   SyntheticIVModel, american_binomial_price,
                                   black_scholes_greeks, black_scholes_price,
                                   skew_adjusted_iv)
from desks.options_pricing import price_option as _price_option
from desks.regime import RegimeHMMModel
from desks.renaissance import TaggedWalkForwardFit
from desks.risk_book import CentralRiskBook
from desks.structures import (StructureTracker, build_iron_condor,
                              max_loss_per_contract, size_contracts)
from desks.walk_forward import WalkForwardController
from portfolio.greeks_aggregator import PortfolioGreeksAggregator
from portfolio.manager import PortfolioManager
from portfolio.risk_manager import RiskManager

logger = logging.getLogger(__name__)

#: Default capital weights per book (fractions of the desk's capital).
DEFAULT_BOOK_WEIGHTS = {
    'vrp': 0.5,
    'earnings': 0.3,
    'relative_value': 0.2,
}

#: Trading days an emitted entry may stay unfilled before its
#: bookkeeping is dropped (engine fills at the next bar's open).
RECONCILE_GRACE_DAYS = 2

#: `_rv_zscore_cached` refusal sentinel: fall through to the verbatim
#: pandas path (distinct from None, which is a real "no z" answer).
_RV_REFUSE = object()
#: "No aligned rows yet" frontier for the relative-value spread cache.
_RV_TS_MIN = np.iinfo(np.int64).min


def _stock(symbol: str) -> Asset:
    return Asset(symbol=symbol, asset_type=AssetType.STOCK)


class JaneStreetDesk(Desk):
    """Premium-selling / relative-value desk (see module docstring)."""

    def __init__(self, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 book_weights: Optional[Dict[str, float]] = None,
                 earnings_calendar: Optional[EarningsCalendar] = None,
                 iv_model: Optional[SyntheticIVModel] = None,
                 regime_controller: Optional[WalkForwardController] = None,
                 risk_book: Optional[CentralRiskBook] = None,
                 etf_symbol: Optional[str] = None,
                 iv_rank_entry: float = 60.0,
                 iv_rank_window: int = 252,
                 iv_rank_min_obs: int = 126,
                 vrp_dte_days: int = 35,
                 earnings_dte_days: int = 21,
                 earnings_lead_days: int = 2,
                 rv_z_window: int = 60,
                 rv_entry_z: float = 2.0,
                 rv_exit_z: float = 0.5,
                 rv_max_fraction: Optional[float] = None,
                 risk_budget_fraction: float = 0.02,
                 # Per-1.00-vol-point units (the greeks_series
                 # convention): 50,000 per $100k == $500 per 1% vol
                 # point per $100k. See the module docstring before
                 # changing — a per-1% number here blocks every entry.
                 vega_limit: float = 50_000.0,
                 profit_target_pct: float = 0.5,
                 stop_loss_mult: float = 2.0,
                 time_exit_dte: int = 21,
                 rate: float = DEFAULT_RATE,
                 # --- Phase E opt-in features (defaults reproduce the
                 # current behavior EXACTLY; greeks golden byte-identical) ---
                 #: Vol-surface skew slope on log-moneyness k=ln(K/spot):
                 #: iv(k) = base_iv * (1 - slope*k), floored positive.
                 #: None / 0.0 -> FLAT surface = current per-symbol IV on
                 #: every leg (the byte-identical default). Positive ->
                 #: equity skew (OTM puts richer, OTM calls cheaper).
                 vol_skew_slope: Optional[float] = None,
                 #: Drop earnings-window observations from the _iv_rank
                 #: percentile so the VRP signal reflects the non-earnings
                 #: vol regime. False (or no earnings_calendar) -> the rank
                 #: is byte-identical to current.
                 exclude_earnings_from_iv_rank: bool = False,
                 #: Calendar days BEFORE a scheduled earnings date that
                 #: count as the earnings window for the rank exclusion
                 #: (inclusive of the earnings date itself). Only consulted
                 #: when exclude_earnings_from_iv_rank is on.
                 iv_rank_earnings_window_days: int = 5,
                 #: Exercise style for the desk's option marks/fills.
                 #: 'european' (default, Black-Scholes -> byte-identical) or
                 #: 'american' (CRR binomial early exercise). Every Jane
                 #: Street option leg is a single-name equity option (the vrp
                 #: and earnings books trade iron condors on the underlyings;
                 #: relative_value is stock-only), which are American in
                 #: reality -> 'american' adds the early-exercise premium the
                 #: put wings carry. PRICE LAYER ONLY (fills + structure
                 #: marks); GREEKS stay Black-Scholes either way (American
                 #: greeks is a follow-up).
                 exercise_style: str = 'european'):
        super().__init__(
            key='janestreet',
            name='Jane Street Desk',
            description=('Fair-value engine, market-making simulator '
                         '(simulation-only), and IV-rank-driven '
                         'defined-risk premium selling with an earnings '
                         'IV-crush module.'),
            accent='#d29922',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
        )
        if not (30 <= vrp_dte_days <= 45):
            raise ValueError(
                f"vrp_dte_days {vrp_dte_days} must be in the 30-45 band")
        if not (14 <= earnings_dte_days <= 30):
            raise ValueError(f"earnings_dte_days {earnings_dte_days} must "
                             f"be in the 14-30 band")
        if exercise_style not in ('european', 'american'):
            raise ValueError(
                "exercise_style must be 'european' or 'american', got "
                f"{exercise_style!r}")

        weights = dict(DEFAULT_BOOK_WEIGHTS)
        if book_weights is not None:
            unknown = set(book_weights) - set(DEFAULT_BOOK_WEIGHTS)
            if unknown:
                raise ValueError(
                    f"Unknown book(s) {sorted(unknown)}; valid books are "
                    f"{sorted(DEFAULT_BOOK_WEIGHTS)}")
            weights.update(book_weights)
        for book, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Book weight for '{book}' must be >= 0, got {weight}")
        if sum(weights.values()) > 1.0 + 1e-9:
            raise ValueError(f"Book weights sum to "
                             f"{sum(weights.values()):.4f}; must be <= 1.0")
        self.book_weights = weights

        self.earnings_calendar = earnings_calendar
        self.iv_model = (iv_model if iv_model is not None
                         else SyntheticIVModel(
                             earnings_calendar=earnings_calendar))
        self.etf_symbol = etf_symbol
        self.iv_rank_entry = iv_rank_entry
        self.iv_rank_window = iv_rank_window
        self.iv_rank_min_obs = iv_rank_min_obs
        self.vrp_dte_days = vrp_dte_days
        self.earnings_dte_days = earnings_dte_days
        self.earnings_lead_days = earnings_lead_days
        self.rv_z_window = rv_z_window
        self.rv_entry_z = rv_entry_z
        self.rv_exit_z = rv_exit_z
        # The relative-value book trades an UNBOUNDED-loss stock pair (ETF vs
        # basket), sized by a flat ~10% notional fraction — far hotter than
        # the defined-risk VRP/earnings option books (risk-budgeted to ~2% of
        # desk capital). rv_max_fraction (None = byte-identical) hard-caps the
        # per-leg fraction so the RV book can't spiral the account to ruin.
        self.rv_max_fraction = rv_max_fraction
        self.risk_budget_fraction = risk_budget_fraction
        self.vega_limit = vega_limit
        self.profit_target_pct = profit_target_pct
        self.stop_loss_mult = stop_loss_mult
        self.time_exit_dte = time_exit_dte
        self.rate = rate
        self.vol_skew_slope = vol_skew_slope
        self.exclude_earnings_from_iv_rank = exclude_earnings_from_iv_rank
        self.iv_rank_earnings_window_days = iv_rank_earnings_window_days
        self.exercise_style = exercise_style

        self._regime_controller = (
            regime_controller if regime_controller is not None
            else WalkForwardController(RegimeHMMModel(),
                                       train_window_days=252,
                                       refit_every_days=21,
                                       min_train_days=120))

        # Central risk book carries the desk's short_vega limit; the
        # check function reads the desk's live vega state (module
        # docstring; risk-book contract via register_limit).
        self.risk_book = (risk_book if risk_book is not None
                          else CentralRiskBook(
                              capital_allocation=capital_allocation))
        self.risk_book.register_limit('short_vega', self._short_vega_check)

        # --- Desk state ---------------------------------------------------
        self.tracker = StructureTracker()
        #: structure id -> book ('vrp' | 'earnings')
        self._structure_books: Dict[str, str] = {}
        #: structure id -> earnings bookkeeping for the crush note
        self._earnings_meta: Dict[str, Dict] = {}
        #: (symbol, earnings date) pairs already traded (one shot per E)
        self._earnings_traded: set = set()
        #: structure id -> closing Trade pnl accumulated from the
        #: portfolio's closed_trades (one per leg when fully closed)
        self._leg_trades: Dict[str, List] = {}
        #: structure id -> day_index the entry was emitted
        self._entry_days: Dict[str, int] = {}
        #: symbol -> rolling synthetic IV observations (one per day with
        #: a computable IV; the IV-rank window slides over this)
        self._iv_history: Dict[str, List[float]] = {}
        #: symbol -> PARALLEL list of bools, one per _iv_history entry:
        #: True when that observation's date fell inside an earnings
        #: window (computed AT observation time from the pre-announced
        #: schedule — see _update_iv_series; NO lookahead). Consulted
        #: only by _iv_rank when exclude_earnings_from_iv_rank is on; the
        #: default rank reads _iv_history untouched, so it is
        #: byte-identical to the pre-feature behavior.
        self._iv_earnings_flags: Dict[str, List[bool]] = {}
        self._today_iv: Dict[str, float] = {}
        #: relative-value spread position (None when flat)
        self._rv_position: Optional[Dict] = None
        #: byte-validated aligned-closes-block cache for _rv_zscore.
        #: In-run state only; every read prefix-validates all symbols'
        #: raw closes/index bytes, so it can never serve stale values.
        self._rv_cache: Optional[Dict] = None
        #: C5-style regime series + C12 greeks series
        self._regime_series: List[Dict] = []
        self._current_regime: Optional[Dict] = None
        self._greeks_series: List[Dict] = []
        #: Desk-agnostic C12 aggregator; _update_greeks delegates to it.
        #: _leg_greeks (the provider) takes only an Asset, so it reads the
        #: per-day market context set at the top of _update_greeks.
        self._greeks_aggregator = PortfolioGreeksAggregator(self._leg_greeks)
        self._greeks_all_data: Dict[str, pd.DataFrame] = {}
        self._greeks_date = None
        #: live vega state read by the short_vega risk-book limit
        self._current_vega = 0.0
        self._committed_vega = 0.0
        self._intent_vegas: Dict[int, float] = {}
        self._n_trades_seen = 0
        self._day_index = 0
        self._last_seen_date: Optional[date_type] = None
        self._started = False

    # ------------------------------------------------------------------
    # Introspection (C3+/C5/C11/C12)
    # ------------------------------------------------------------------
    @property
    def walk_forward_fits(self) -> List[TaggedWalkForwardFit]:
        """Regime controller fits, tagged 'regime' (C3+)."""
        return [TaggedWalkForwardFit(fit=fit, model='regime')
                for fit in self._regime_controller.fits]

    @property
    def regime_series(self) -> List[Dict]:
        """C5 regime series (entries only once the model is fitted)."""
        return self._regime_series

    @property
    def structures_report(self) -> List[Dict]:
        """Contract C11: every structure's lifecycle record."""
        return self.tracker.to_report()

    @property
    def greeks_series(self) -> List[Dict]:
        """Contract C12: one {'date','delta','gamma','theta','vega'} per
        simulated day.

        CONVENTION (the one convention, used for the series AND the
        short_vega limit): SHARE-EQUIVALENT GREEKS — each option
        position contributes its per-share Black-Scholes greek x its
        SIGNED contract count x the 100 multiplier; the desk value is
        the sum over option positions. delta is therefore the
        equivalent share exposure, gamma its change per $1 of spot,
        theta $ decay per CALENDAR day, vega $ per 1.00 vol point.
        Option-free days report all zeros.
        """
        return self._greeks_series

    def get_status(self) -> Dict:
        status = super().get_status()
        status['regime'] = (self._current_regime['state']
                            if self._current_regime else None)
        status['book_weights'] = dict(self.book_weights)
        status['open_structures'] = len(self.tracker.live_ids())
        # Phase E transparency: surface the opt-in features ONLY when
        # enabled, so the default status dict is byte-identical to before.
        if self.vol_skew_slope is not None:
            status['vol_skew_slope'] = self.vol_skew_slope
        if self.exclude_earnings_from_iv_rank:
            status['exclude_earnings_from_iv_rank'] = True
            status['iv_rank_earnings_window_days'] = \
                self.iv_rank_earnings_window_days
        return status

    # ------------------------------------------------------------------
    # Engine pricing hook (no-lookahead: see desks/options_pricing)
    # ------------------------------------------------------------------
    def _price_leg(self, spot: float, strike: float, t_years: float,
                   iv: float, right: str) -> float:
        """One option leg's price under the desk's ``exercise_style``:
        Black-Scholes ('european', the default -> byte-identical) or the CRR
        binomial tree ('american'), which adds the early-exercise premium
        real single-name equity options carry (chiefly on the put wings;
        an American call with no modeled dividend equals its European value).
        Marks/fills only — greeks stay Black-Scholes.
        """
        if self.exercise_style == 'american':
            return american_binomial_price(spot, strike, t_years, iv,
                                           self.rate, right)
        return black_scholes_price(spot, strike, t_years, iv, self.rate, right)

    def price_option(self, asset: Asset, underlying_frame: pd.DataFrame,
                     date, spot: float) -> Optional[float]:
        """Synthetic fair value the engine fills/marks against — the
        SAME SyntheticIVModel the books trade on, so the desk and the
        fills can never disagree about the vol surface.

        VOL SURFACE (opt-in, vol_skew_slope): when enabled, the engine's
        fills/marks use the per-STRIKE surface IV, the same skew the
        desk's internal credit/greeks/marks apply — keeping the desk and
        the fills on ONE surface. When disabled, this delegates to the
        shared price_option with the flat model IV, byte-identical to the
        pre-surface path. NO-LOOKAHEAD preserved: the base IV still slices
        the frame to closes strictly before `date`; the skew is a pure
        function of the caller-supplied spot and the leg strike.
        """
        if self.vol_skew_slope is None:
            return _price_option(asset, underlying_frame, date, spot,
                                 self.iv_model, rate=self.rate,
                                 exercise=self.exercise_style)
        if asset.asset_type not in (AssetType.CALL, AssetType.PUT):
            raise ValueError(
                f"price_option needs an option asset, got {asset}")
        if asset.strike_price is None or asset.expiration_date is None:
            # Mirror the module price_option guard so both pricing paths
            # reject a malformed option asset with the SAME clear ValueError
            # (a None strike/expiry would otherwise surface as a vaguer
            # TypeError/NaT here). Unreachable for real legs (always carry
            # strike+expiry); defensive symmetry only.
            raise ValueError(f"option asset missing strike/expiry: {asset}")
        expiry = pd.Timestamp(asset.expiration_date).date()
        t_years = (expiry - pd.Timestamp(date).date()).days / 365.0
        if t_years <= 0:
            return black_scholes_price(spot, asset.strike_price, 0.0, 0.0,
                                       self.rate, asset.asset_type.value)
        base_iv = self.iv_model.iv(asset.symbol, underlying_frame, date)
        if base_iv is None:
            return None
        leg_iv = self._surface_iv(base_iv, spot, asset.strike_price, t_years)
        return self._price_leg(spot, asset.strike_price, t_years, leg_iv,
                               asset.asset_type.value)

    # ------------------------------------------------------------------
    # One simulated day
    # ------------------------------------------------------------------
    def generate_intents(self, all_data: Dict[str, pd.DataFrame], date,
                         portfolio: PortfolioManager) -> List[DeskIntent]:
        """Order: clocks -> IV series -> reconcile fills/closes ->
        Greeks -> regime -> structure exits -> entries (risk-book
        checked atomically) -> relative value."""
        self._advance_day(date)
        if not self._started:
            self._started = True
            if self.earnings_calendar is None:
                self.note('info',
                          "Earnings book idle: no earnings calendar "
                          "injected (synthetic or OpenBB); the book "
                          "trades nothing this run",
                          book='earnings')

        self._update_iv_series(all_data, date)
        intents: List[DeskIntent] = []
        intents.extend(self._reconcile(all_data, date, portfolio))
        self._update_greeks(all_data, date, portfolio)

        self._regime_refit(all_data, date)
        regime_now = self._update_regime(all_data, date)

        intents.extend(self._structure_exits(all_data, date, regime_now))
        intents.extend(self._structure_entries(all_data, date, portfolio))
        intents.extend(self._relative_value_intents(all_data, date,
                                                    portfolio))
        return intents

    def _advance_day(self, date) -> None:
        current = pd.Timestamp(date).date()
        if self._last_seen_date is not None and current != self._last_seen_date:
            self._day_index += 1
        self._last_seen_date = current

    def _has_bar_today(self, frame: Optional[pd.DataFrame], date) -> bool:
        return (frame is not None and not frame.empty
                and frame.index[-1] == pd.Timestamp(date))

    # ------------------------------------------------------------------
    # IV series + IV-rank (vrp book inputs)
    # ------------------------------------------------------------------
    def _update_iv_series(self, all_data: Dict[str, pd.DataFrame],
                          date) -> None:
        """Append today's synthetic IV per symbol (closes strictly
        before today — the same number a fill at today's open would
        price with).

        Alongside each appended IV we record whether TODAY fell inside an
        earnings window (computed here, at observation time, from the
        pre-announced schedule via _in_earnings_window). The flag is
        consulted only by _iv_rank under the opt-in exclusion; the IV
        list itself is unchanged, so the default rank path is
        byte-identical.
        """
        self._today_iv = {}
        for symbol in sorted(all_data):
            frame = all_data[symbol]
            if not self._has_bar_today(frame, date):
                continue
            iv = self.iv_model.iv(symbol, frame, date)
            if iv is None:
                continue
            self._today_iv[symbol] = iv
            self._iv_history.setdefault(symbol, []).append(iv)
            self._iv_earnings_flags.setdefault(symbol, []).append(
                self._in_earnings_window(symbol, date))

    def _in_earnings_window(self, symbol: str, date) -> bool:
        """True when `date` is within iv_rank_earnings_window_days
        calendar days BEFORE the next scheduled earnings date for
        `symbol` (inclusive of the earnings date itself).

        NO-LOOKAHEAD: this uses next_earnings(symbol, date) — the
        earnings schedule is known in advance, so judging an observation
        against the NEXT pre-announced date as of that same observation
        date introduces no future information. The flag is frozen when
        the observation is appended and never recomputed retroactively
        against a later-announced date. Returns False unless the
        exclusion feature is enabled AND a calendar is present (so the
        flag is always False on the default path).
        """
        if not self.exclude_earnings_from_iv_rank \
                or self.earnings_calendar is None:
            return False
        current = pd.Timestamp(date).date()
        earnings = self.earnings_calendar.next_earnings(symbol, current)
        if earnings is None:
            return False
        days_to_e = (earnings - current).days
        return 0 <= days_to_e <= self.iv_rank_earnings_window_days

    def _iv_rank(self, symbol: str) -> Optional[float]:
        """Percentile of today's IV within the trailing iv_rank_window
        observations (inclusive of today): 100 * fraction of window
        observations <= today's IV. None until iv_rank_min_obs
        observations exist.

        EARNINGS EXCLUSION (opt-in, exclude_earnings_from_iv_rank): when
        enabled, observations whose date fell inside an earnings window
        (the parallel _iv_earnings_flags, frozen at observation time) are
        DROPPED from the comparison set before the percentile is taken,
        so an earnings IV run-up cannot inflate the VRP rank. Today's IV
        stays the reference value even if today itself is flagged — the
        question is "where does today's vol sit within the NON-earnings
        regime". The min-obs gate also requires iv_rank_min_obs
        non-earnings observations, so the rank never rests on a thin
        post-exclusion sample. DEFAULT (feature off): the comparison set
        is the raw trailing window, byte-identical to the prior behavior.
        """
        history = self._iv_history.get(symbol, ())
        if len(history) < self.iv_rank_min_obs:
            return None
        window = history[-self.iv_rank_window:]
        today = window[-1]
        if self.exclude_earnings_from_iv_rank:
            flags = self._iv_earnings_flags.get(symbol, ())
            window_flags = flags[-self.iv_rank_window:]
            # The IV list and its earnings-flag list are appended in lockstep
            # in _update_iv_series; fail LOUD rather than let zip() silently
            # truncate (misaligning flags to values) if a future code path
            # ever desyncs them.
            if len(window_flags) != len(window):
                raise RuntimeError(
                    f"IV/earnings-flag desync for {symbol}: {len(window)} "
                    f"IVs vs {len(window_flags)} flags")
            comparison = [value for value, flagged
                          in zip(window, window_flags) if not flagged]
            if len(comparison) < self.iv_rank_min_obs:
                return None
            return 100.0 * sum(1 for value in comparison
                               if value <= today) / len(comparison)
        return 100.0 * sum(1 for value in window if value <= today) / len(window)

    # ------------------------------------------------------------------
    # Regime (gating controller, fits tagged 'regime')
    # ------------------------------------------------------------------
    def _regime_refit(self, all_data: Dict[str, pd.DataFrame], date) -> None:
        if self._regime_controller.maybe_refit(all_data, date):
            fit = self._regime_controller.fits[-1]
            self.note(
                'model',
                f"Regime model refit #{len(self._regime_controller.fits)}: "
                f"trained on {fit.n_samples} samples "
                f"({fit.train_start} .. {fit.train_end})",
                book='regime', model='regime', **fit.to_dict())

    def _update_regime(self, all_data: Dict[str, pd.DataFrame],
                       date) -> Optional[str]:
        """Refresh the posterior; returns today's state (None = no
        regime -> every book stays dark, the fail-safe default)."""
        result = self._regime_controller.predict(all_data, date)
        if result is None:
            return None  # never fitted: NOTHING trades (module docstring)
        if not result:
            self._current_regime = None  # degraded model: fail safe
            return None
        previous = (self._current_regime['state']
                    if self._current_regime else None)
        state = result['state']
        probs = {label: float(p) for label, p in result['probs'].items()}
        self._current_regime = {'state': state, 'probs': probs}
        self._regime_series.append({
            'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
            'state': state, 'probs': probs,
        })
        if state != previous:
            was = f" (was {previous})" if previous is not None else ""
            self.note('model', f"Regime: {state} p={probs[state]:.2f}{was}",
                      book='regime', state=state, probs=probs,
                      previous_state=previous)
        return state

    # ------------------------------------------------------------------
    # Greeks (C12 convention on the greeks_series docstring)
    # ------------------------------------------------------------------
    def _leg_inputs(self, asset: Asset, all_data: Dict[str, pd.DataFrame],
                    date) -> Optional[Tuple[float, float, float]]:
        """(spot, iv, t_years) for an option leg at today's close, or
        None when the underlying has no bar/IV today.

        VOL SURFACE (opt-in, vol_skew_slope): when enabled, the returned
        iv is the per-STRIKE surface IV — the flat per-symbol base IV
        tilted by the moneyness skew at THIS leg's strike (see
        _surface_iv / skew_adjusted_iv). When disabled (slope None/0.0),
        iv is the exact flat base IV, so every leg on a symbol shares one
        IV and the pricing/greeks are byte-identical to the pre-surface
        path. spot and strike are both same-day inputs -> no lookahead.
        """
        frame = all_data.get(asset.symbol)
        if not self._has_bar_today(frame, date):
            return None
        base_iv = self._today_iv.get(asset.symbol)
        if base_iv is None:
            return None
        spot = float(frame['close'].iloc[-1])
        expiry = pd.Timestamp(asset.expiration_date).date()
        t_years = (expiry - pd.Timestamp(date).date()).days / 365.0
        iv = self._surface_iv(base_iv, spot, asset.strike_price, t_years)
        return spot, iv, t_years

    def _surface_iv(self, base_iv: float, spot: float,
                    strike: Optional[float], t_years: float) -> float:
        """Per-strike surface IV via the deterministic moneyness skew, or
        the flat base_iv when the surface is disabled. Centralizes the
        opt-in switch so every pricing site (greeks, marks, entry sizing)
        applies the SAME surface. vol_skew_slope None/0.0 -> base_iv
        unchanged (byte-identical default)."""
        if self.vol_skew_slope is None or strike is None:
            return base_iv
        return skew_adjusted_iv(base_iv, spot, strike, self.vol_skew_slope,
                                t_years)

    def _leg_greeks(self, asset: Asset) -> Optional[Dict[str, float]]:
        """GreeksProvider for the portfolio aggregator (C12): this option
        leg's per-share Black-Scholes greeks at today's close, or None when
        the underlying has no bar/IV today — the leg then contributes
        nothing (the same skip + WARNING the old inline loop emitted).
        Reads the per-day context set by _update_greeks because a
        GreeksProvider takes only an Asset."""
        inputs = self._leg_inputs(asset, self._greeks_all_data,
                                  self._greeks_date)
        if inputs is None:
            logger.warning("Greeks: no inputs for %s on %s; leg "
                           "contributes nothing today", str(asset),
                           self._greeks_date)
            return None
        spot, iv, t_years = inputs
        return black_scholes_greeks(spot, asset.strike_price, t_years, iv,
                                    self.rate, asset.asset_type.value)

    def _update_greeks(self, all_data: Dict[str, pd.DataFrame], date,
                       portfolio: PortfolioManager) -> None:
        """Append today's C12 share-equivalent portfolio Greeks. Delegates
        the aggregation to PortfolioGreeksAggregator (the desk-agnostic
        primitive) via the _leg_greeks provider; the totals are bit-for-bit
        identical to the previous inline loop (golden-file gated)."""
        self._greeks_all_data = all_data
        self._greeks_date = date
        totals = self._greeks_aggregator.aggregate(portfolio)
        self._greeks_series.append({
            'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
            **{key: float(value) for key, value in totals.items()},
        })
        self._current_vega = totals['vega']
        self._committed_vega = 0.0  # fresh risk-book day
        self._intent_vegas = {}

    def _short_vega_check(self, intent: DeskIntent,
                          state: Dict) -> Optional[str]:
        """CentralRiskBook custom limit 'short_vega' (registered at
        construction): blocks a NEW premium-selling structure when the
        post-trade |portfolio vega| would exceed vega_limit per $100k of
        desk capital. UNITS: vega_limit, the structure vegas and the
        running portfolio vega are all in the greeks_series convention
        (share-equivalent $ per 1.00 vol point — see that docstring);
        the default 50,000 per $100k is $500 per 1% vol point per $100k.
        The structure's NET vega (short legs minus wings) is stamped on
        its first short leg; other legs carry 0 so the structure is
        judged atomically. Closing intents never reach custom limits
        (CentralRiskBook approves closes unconditionally).
        """
        structure_vega = self._intent_vegas.get(id(intent))
        if structure_vega is None:
            return None  # not a structure head (stock legs, wings)
        candidate = (self._current_vega + self._committed_vega
                     + structure_vega)
        limit = self.vega_limit * state['desk_capital'] / 100_000.0
        if abs(candidate) > limit:
            return (f"portfolio vega would reach {candidate:,.0f}, beyond "
                    f"the +/-{limit:,.0f} short_vega limit "
                    f"({self.vega_limit:,.0f} per $100k)")
        self._committed_vega += structure_vega
        return None

    # ------------------------------------------------------------------
    # C14: book tags for the SHARED apply_risk notes
    # ------------------------------------------------------------------
    def risk_note_data(self, asset: Optional[Asset]) -> Dict:
        """Contract C14: EVERY janestreet note carries data.book — the
        shared Desk.apply_risk notes (stop losses, position-size and
        flip blocks, the daily-loss circuit) included; the base class
        merges this hook's result into each of them. Mapping: option
        legs belong to their structure's book (tracker.leg_owner; 'vrp'
        when the leg is unowned), stock legs trade only in the
        relative_value book, and desk-level notes (asset is None: the
        daily-loss circuit) land on 'regime', the desk-wide gating book.
        """
        if asset is None:
            return {'book': 'regime'}
        if asset.asset_type in (AssetType.CALL, AssetType.PUT):
            owner = self.tracker.leg_owner.get(asset)
            return {'book': self._structure_books.get(owner, 'vrp')}
        return {'book': 'relative_value'}

    # ------------------------------------------------------------------
    # Reconcile: entry fills, finalized closes, expiries, broken RV legs
    # ------------------------------------------------------------------
    def _reconcile(self, all_data: Dict[str, pd.DataFrame], date,
                   portfolio: PortfolioManager) -> List[DeskIntent]:
        # --- collect new closed trades for structure legs --------------
        new_trades = portfolio.closed_trades[self._n_trades_seen:]
        self._n_trades_seen = len(portfolio.closed_trades)
        for trade in new_trades:
            owner = self.tracker.leg_owner.get(trade.asset)
            if owner is not None:
                self._leg_trades.setdefault(owner, []).append(trade)

        # Close intents for broken (partially-filled) structures — see the
        # partial-entry branch below. Returned alongside the RV intents.
        cleanup_intents: List[DeskIntent] = []
        for structure_id in self.tracker.live_ids():
            structure = self.tracker.get(structure_id)
            book = self._structure_books.get(structure_id, 'vrp')
            positions = [portfolio.get_position(leg['asset'])
                         for leg in structure['legs']]
            n_open = sum(1 for p in positions
                         if p is not None and p.quantity != 0)

            if not structure['entry_filled']:
                if n_open == len(structure['legs']):
                    self._restate_from_fills(structure_id, positions, book)
                elif n_open == 0 and not self._leg_trades.get(structure_id) \
                        and (self._day_index
                             - self._entry_days.get(structure_id, 0)
                             ) >= RECONCILE_GRACE_DAYS:
                    # Entry never filled (no bars): drop the bookkeeping.
                    self.note(
                        'info',
                        f"Structure cleanup: {structure['type']} "
                        f"{structure_id} on {structure['underlying']} "
                        f"never filled; dropped",
                        book=book, structure_id=structure_id,
                        structure=structure['type'])
                    self.tracker.remove(structure_id)
                    self._structure_books.pop(structure_id, None)
                    continue
                elif 0 < n_open < len(structure['legs']) \
                        and (self._day_index
                             - self._entry_days.get(structure_id, 0)
                             ) >= RECONCILE_GRACE_DAYS:
                    # Partial entry (RUIN GUARD): some legs never filled —
                    # typically a protective wing whose BUY was dropped for
                    # insufficient cash while the short leg (which only credits
                    # cash) filled — leaving the open legs with UNDEFINED risk.
                    # A defined-risk desk must never ride a broken structure to
                    # expiry (that is the path to a naked-short blow-up), so
                    # flatten every open leg now and drop the bookkeeping.
                    for leg, pos in zip(structure['legs'], positions):
                        if pos is not None and pos.quantity != 0:
                            cleanup_intents.append(DeskIntent(
                                asset=leg['asset'],
                                action='SELL' if pos.quantity > 0 else 'COVER',
                                size_fraction=1.0,
                                reason=(f"{structure['type']} {structure_id} "
                                        f"broken-entry cleanup: {n_open}/"
                                        f"{len(structure['legs'])} legs filled; "
                                        f"flattening open legs (no defined risk)")))
                    self.note(
                        'risk',
                        f"Broken structure {structure_id} on "
                        f"{structure['underlying']}: only {n_open}/"
                        f"{len(structure['legs'])} legs filled — flattening "
                        f"open legs to avoid naked exposure",
                        book=book, structure_id=structure_id,
                        structure=structure['type'], n_open=n_open)
                    self.tracker.remove(structure_id)
                    self._structure_books.pop(structure_id, None)
                    self._earnings_meta.pop(structure_id, None)
                    continue

            # --- all legs flat: finalize from the legs' closed trades --
            trades = self._leg_trades.get(structure_id, [])
            if structure['entry_filled'] and n_open == 0 \
                    and len(trades) >= len(structure['legs']):
                pnl = float(sum(trade.pnl for trade in trades))
                expired = structure['status'] == 'open'
                # Options are exempt from per-leg stops, so a flat book
                # the desk never asked to close can only be the engine's
                # expiry settlement backstop.
                status = 'expired' if expired else 'closed'
                reason = 'expiry' if expired else structure['close_reason']
                closed_date = max(trade.exit_time for trade in trades)
                self.tracker.finalize(structure_id, closed_date, pnl,
                                      status=status, close_reason=reason)
                message = ('expiry settlement closed' if expired
                           else f"closed ({reason})")
                self.note(
                    'signal' if not expired else 'risk',
                    f"Structure {structure_id} {message}: "
                    f"{structure['type']} on {structure['underlying']}, "
                    f"realized P&L {pnl:,.2f}",
                    book=book, structure_id=structure_id,
                    structure=structure['type'], pnl=pnl,
                    close_reason=reason)
                if structure_id in self._earnings_meta:
                    del self._earnings_meta[structure_id]

        # --- relative-value legs: broken spread handling ----------------
        return cleanup_intents + self._reconcile_rv(portfolio)

    def _restate_from_fills(self, structure_id: str, positions: List,
                            book: str) -> None:
        """All legs filled: restate credit/max_loss from ACTUAL entry
        prices (the sizing-time numbers were model estimates at the
        prior close; fills happen at the next open with spread costs).
        The exit checks then reference the credit really received."""
        structure = self.tracker.get(structure_id)
        credit_per_contract = 0.0
        for leg, position in zip(structure['legs'], positions):
            if leg['action'] == 'SHORT':
                credit_per_contract += position.avg_entry_price
            else:
                credit_per_contract -= position.avg_entry_price
        contracts = structure['contracts']
        credit = credit_per_contract * 100.0 * contracts
        spec_legs = [{'action': leg['action'],
                      'right': leg['asset'].asset_type.value,
                      'strike': leg['asset'].strike_price}
                     for leg in structure['legs']]
        max_loss = (max_loss_per_contract(spec_legs,
                                          credit_per_contract * 100.0)
                    * contracts)
        structure['credit'] = float(credit)
        structure['max_loss'] = float(max_loss)
        structure['entry_filled'] = True
        self.note(
            'info',
            f"Structure {structure_id} filled: credit {credit:,.2f} "
            f"received, max loss {max_loss:,.2f} "
            f"({contracts} contract(s))",
            book=book, structure_id=structure_id,
            structure=structure['type'], credit=credit, max_loss=max_loss,
            contracts=contracts)

    # ------------------------------------------------------------------
    # Structure exits (vrp checks + regime flatten + earnings post-E)
    # ------------------------------------------------------------------
    def _leg_marks(self, structure_id: str,
                   all_data: Dict[str, pd.DataFrame],
                   date) -> Dict[Asset, float]:
        marks: Dict[Asset, float] = {}
        for leg in self.tracker.get(structure_id)['legs']:
            inputs = self._leg_inputs(leg['asset'], all_data, date)
            if inputs is None:
                continue
            spot, iv, t_years = inputs
            marks[leg['asset']] = self._price_leg(
                spot, leg['asset'].strike_price, t_years, iv,
                leg['asset'].asset_type.value)
        return marks

    def _close_structure(self, structure_id: str, reason: str,
                         message: str) -> List[DeskIntent]:
        structure = self.tracker.get(structure_id)
        book = self._structure_books.get(structure_id, 'vrp')
        intents = self.tracker.build_closing_intents(structure_id, reason)
        self.note('signal', message, book=book, structure_id=structure_id,
                  structure=structure['type'], close_reason=reason,
                  cost_to_close=structure['last_cost_to_close'],
                  credit=structure['credit'])
        return intents

    def _structure_exits(self, all_data: Dict[str, pd.DataFrame], date,
                         regime_now: Optional[str]) -> List[DeskIntent]:
        intents: List[DeskIntent] = []
        flatten = regime_now == 'high_vol'
        for structure_id in self.tracker.open_ids():
            structure = self.tracker.get(structure_id)
            book = self._structure_books.get(structure_id, 'vrp')
            if not structure['entry_filled']:
                continue  # nothing on yet; exits act on filled books

            if book == 'earnings':
                meta = self._earnings_meta.get(structure_id)
                if meta is None:
                    continue
                if pd.Timestamp(date).date() > meta['earnings_date']:
                    exit_iv = self._today_iv.get(structure['underlying'])
                    entry_iv = meta['entry_iv']
                    crush = ((entry_iv - exit_iv) / entry_iv * 100.0
                             if exit_iv is not None and entry_iv else None)
                    intents.extend(self._close_structure(
                        structure_id, 'time_exit',
                        f"Earnings crush exit {structure_id} "
                        f"({structure['underlying']}): first session after "
                        f"earnings {meta['earnings_date']}; model IV "
                        f"{entry_iv:.3f} -> "
                        f"{exit_iv if exit_iv is not None else float('nan'):.3f}"
                        f" ({'%.1f%% crush' % crush if crush is not None else 'n/a'})"))
                    note = self.notes[-1]
                    note.data.update(entry_iv=entry_iv, exit_iv=exit_iv,
                                     earnings_date=str(meta['earnings_date']))
                continue

            # --- vrp book ------------------------------------------------
            if flatten:
                intents.extend(self._close_structure(
                    structure_id, 'regime_flatten',
                    f"Regime flatten {structure_id} "
                    f"({structure['underlying']}): regime entered high_vol; "
                    f"all vrp structures close"))
                continue
            marks = self._leg_marks(structure_id, all_data, date)
            reason = self.tracker.check_exits(
                structure_id, marks, date,
                profit_target_pct=self.profit_target_pct,
                stop_loss_mult=self.stop_loss_mult,
                time_exit_dte=self.time_exit_dte)
            if reason is not None:
                intents.extend(self._close_structure(
                    structure_id, reason,
                    f"VRP exit {structure_id} ({structure['underlying']}): "
                    f"{reason} (cost to close "
                    f"{structure['last_cost_to_close'] if structure['last_cost_to_close'] is not None else float('nan'):,.2f}"
                    f" vs credit {structure['credit']:,.2f})"))
        return intents

    # ------------------------------------------------------------------
    # Structure entries (vrp + earnings), risk-book checked atomically
    # ------------------------------------------------------------------
    def _build_condor(self, symbol: str, spot: float, iv: float,
                      dte_days: int, book: str, date,
                      portfolio: PortfolioManager) -> Optional[Dict]:
        """Candidate condor: legs, model credit estimate, sizing. None
        (with the documented 'info' note) when sizing reaches 0."""
        specs = build_iron_condor(spot, iv, dte_days)
        expiry = (pd.Timestamp(date) + pd.Timedelta(days=dte_days)
                  ).strftime('%Y-%m-%d')
        t_years = dte_days / 365.0
        credit_pc = 0.0
        vega_per_contract = 0.0
        legs = []
        for spec in specs:
            asset = Asset(symbol=symbol,
                          asset_type=(AssetType.PUT if spec['right'] == 'put'
                                      else AssetType.CALL),
                          strike_price=spec['strike'],
                          expiration_date=expiry)
            # Per-strike surface IV when the surface is on; the flat base
            # `iv` (byte-identical) when it is off.
            leg_iv = self._surface_iv(iv, spot, spec['strike'], t_years)
            greeks = black_scholes_greeks(spot, spec['strike'], t_years,
                                          leg_iv, self.rate, spec['right'])
            sign = 1.0 if spec['action'] == 'SHORT' else -1.0
            credit_pc += sign * greeks['price']
            vega_per_contract += -sign * greeks['vega'] * 100.0
            legs.append({'asset': asset, 'action': spec['action']})
        credit_pc *= 100.0  # per-contract $ credit at model mid
        loss_pc = max_loss_per_contract(specs, credit_pc)

        desk_capital = (portfolio.get_portfolio_value()
                        * self.capital_allocation)
        risk_budget = (self.risk_budget_fraction * desk_capital
                       * self.book_weights[book])
        contracts = size_contracts(risk_budget, loss_pc)
        if contracts <= 0:
            self.note(
                'info',
                f"{book} entry skipped on {symbol}: risk budget "
                f"{risk_budget:,.2f} cannot absorb one contract's max "
                f"loss {loss_pc:,.2f}",
                book=book, symbol=symbol, risk_budget=risk_budget,
                max_loss_per_contract=loss_pc)
            return None
        return {
            'symbol': symbol, 'legs': legs, 'contracts': contracts,
            'credit': credit_pc * contracts,
            'max_loss': loss_pc * contracts,
            'vega': vega_per_contract * contracts,
            'expiry': expiry, 'book': book,
            'size_fraction': min(1.0, self.risk_budget_fraction
                                 * self.book_weights[book]),
        }

    def _candidate_intents(self, candidate: Dict) -> List[DeskIntent]:
        intents = []
        head_stamped = False
        for leg in candidate['legs']:
            intent = DeskIntent(
                asset=leg['asset'], action=leg['action'],
                size_fraction=candidate['size_fraction'],
                reason=(f"{candidate['book']} short iron condor on "
                        f"{candidate['symbol']} "
                        f"({candidate['contracts']} contract(s))"),
                quantity=candidate['contracts'])
            if leg['action'] == 'SHORT' and not head_stamped:
                # Structure head: carries the WHOLE structure's net vega
                # so the short_vega limit judges the package atomically.
                self._intent_vegas[id(intent)] = candidate['vega']
                head_stamped = True
            intents.append(intent)
        return intents

    def _structure_entries(self, all_data: Dict[str, pd.DataFrame], date,
                           portfolio: PortfolioManager) -> List[DeskIntent]:
        if not self._regime_controller.is_fitted \
                or self._current_regime is None \
                or self._current_regime['state'] == 'high_vol':
            return []

        candidates: List[Dict] = []
        current = pd.Timestamp(date).date()
        for symbol in sorted(all_data):
            frame = all_data[symbol]
            if not self._has_bar_today(frame, date):
                continue
            if self.tracker.has_open_on(symbol):
                continue  # one open structure per underlying, all books
            iv = self._today_iv.get(symbol)
            if iv is None:
                continue
            spot = float(frame['close'].iloc[-1])

            # --- earnings book: exactly N trading days before E ---------
            if self.earnings_calendar is not None \
                    and self.book_weights['earnings'] > 0:
                earnings = self.earnings_calendar.next_earnings(symbol,
                                                                current)
                if earnings is not None \
                        and (symbol, earnings) not in self._earnings_traded \
                        and int(np.busday_count(current, earnings)) \
                        == self.earnings_lead_days:
                    candidate = self._build_condor(
                        symbol, spot, iv, self.earnings_dte_days,
                        'earnings', date, portfolio)
                    if candidate is not None:
                        candidate['earnings_date'] = earnings
                        candidate['entry_iv'] = iv
                        candidates.append(candidate)
                        self._earnings_traded.add((symbol, earnings))
                        continue  # one structure per underlying

            # --- vrp book: IV-rank trigger -------------------------------
            if self.book_weights['vrp'] <= 0:
                continue
            # When the IV-rank earnings exclusion is on, keep the VRP book OUT
            # of the earnings window entirely: today's IV is itself
            # earnings-bumped, so a high rank measured vs the calm
            # (non-earnings) regime would otherwise sell premium INTO the
            # pre-earnings run-up — that vol belongs to the earnings book.
            # No-op on the default path (_in_earnings_window returns False
            # when the exclusion feature is off or no calendar is present),
            # so the greeks golden stays byte-identical.
            if self._in_earnings_window(symbol, date):
                continue
            iv_rank = self._iv_rank(symbol)
            if iv_rank is None or iv_rank <= self.iv_rank_entry:
                continue
            candidate = self._build_condor(symbol, spot, iv,
                                           self.vrp_dte_days, 'vrp', date,
                                           portfolio)
            if candidate is not None:
                candidate['iv_rank'] = iv_rank
                candidates.append(candidate)

        if not candidates:
            return []

        # --- central risk book: atomic per-structure approval -----------
        all_legs: List[DeskIntent] = []
        by_structure: List[Tuple[Dict, List[DeskIntent]]] = []
        for candidate in candidates:
            leg_intents = self._candidate_intents(candidate)
            by_structure.append((candidate, leg_intents))
            all_legs.extend(leg_intents)
        prices = {symbol: float(frame['close'].iloc[-1])
                  for symbol, frame in all_data.items()
                  if self._has_bar_today(frame, date)}
        _, blocked = self.risk_book.check(all_legs, portfolio, prices)
        blocked_ids = {id(intent) for intent, _ in blocked}
        reasons = {id(intent): reason for intent, reason in blocked}

        approved: List[DeskIntent] = []
        for candidate, leg_intents in by_structure:
            hit = [reasons[id(i)] for i in leg_intents
                   if id(i) in blocked_ids]
            if hit:
                self.note(
                    'risk',
                    f"Central risk book blocked the {candidate['book']} "
                    f"iron condor on {candidate['symbol']}: {hit[0]}",
                    book=candidate['book'], symbol=candidate['symbol'],
                    reason=hit[0])
                if candidate['book'] == 'earnings':
                    self._earnings_traded.discard(
                        (candidate['symbol'], candidate['earnings_date']))
                continue
            structure_id = self.tracker.open_structure(
                'iron_condor', candidate['symbol'], candidate['legs'],
                candidate['credit'], candidate['max_loss'],
                candidate['contracts'], date)
            self._structure_books[structure_id] = candidate['book']
            self._entry_days[structure_id] = self._day_index
            if candidate['book'] == 'earnings':
                self._earnings_meta[structure_id] = {
                    'earnings_date': candidate['earnings_date'],
                    'entry_iv': candidate['entry_iv'],
                }
                detail = (f"{self.earnings_lead_days} trading day(s) before "
                          f"earnings {candidate['earnings_date']}, entry IV "
                          f"{candidate['entry_iv']:.3f}")
            else:
                detail = (f"IV-rank {candidate['iv_rank']:.1f} > "
                          f"{self.iv_rank_entry:.0f}")
            self.note(
                'signal',
                f"{candidate['book']} entry {structure_id}: short iron "
                f"condor on {candidate['symbol']} ({detail}); "
                f"{candidate['contracts']} contract(s), est. credit "
                f"{candidate['credit']:,.2f}, max loss "
                f"{candidate['max_loss']:,.2f}, net vega "
                f"{candidate['vega']:,.0f}",
                book=candidate['book'], structure_id=structure_id,
                structure='iron_condor', symbol=candidate['symbol'],
                contracts=candidate['contracts'],
                credit=candidate['credit'], max_loss=candidate['max_loss'],
                vega=candidate['vega'],
                **({'iv_rank': candidate['iv_rank']}
                   if 'iv_rank' in candidate else
                   {'earnings_date': str(candidate['earnings_date'])}))
            approved.extend(leg_intents)
        return approved

    # ------------------------------------------------------------------
    # Relative-value book (stock legs only; pairs mechanics)
    # ------------------------------------------------------------------
    def _rv_universe(self, all_data: Dict[str, pd.DataFrame]
                     ) -> Tuple[Optional[str], List[str]]:
        symbols = sorted(all_data)
        if len(symbols) < 2:
            return None, []
        etf = self.etf_symbol if self.etf_symbol is not None else symbols[0]
        if etf not in all_data:
            return None, []
        return etf, [s for s in symbols if s != etf]

    def _rv_zscore(self, all_data: Dict[str, pd.DataFrame], date,
                   etf: str, basket: List[str]) -> Optional[float]:
        """ETF-vs-basket log-spread z at `date` (None when the spread is
        not computable today).

        Served through a byte-validated cache of the ALIGNED CLOSES BLOCK
        (the `pd.DataFrame(closes).dropna()` output — the O(N*T) join
        that dominated this call at full-universe widths). The cache
        appends newly-aligned rows incrementally; everything downstream
        (np.log, basket row-mean, FULL-LENGTH rolling mean/std) re-runs
        the full path's own expressions each day over a block that is
        bit-identical, C-contiguous and column-ordered exactly like the
        full path's — so no floating-point op is ever hand-replicated
        (log/row-mean reductions are memory-layout-sensitive; rolling on
        a trailing slice is ~1-ulp wrong). Every symbol's cached raw
        prefix is byte-compared against the incoming frame on every
        call; any mismatch rebuilds through the full path's own
        construction. Inputs the cache does not model (non-float64
        closes, non-DatetimeIndex/tz/duplicate/unsorted indexes) fall
        through to the verbatim full path."""
        z = self._rv_zscore_cached(all_data, date, etf, basket)
        if z is not _RV_REFUSE:
            return z
        return self._rv_zscore_full(all_data, date, etf, basket)

    def _rv_zscore_full(self, all_data: Dict[str, pd.DataFrame], date,
                        etf: str, basket: List[str]) -> Optional[float]:
        """The original uncached body, preserved VERBATIM: both the
        fallback for inputs the cache refuses and the oracle the
        equivalence tests compare against."""
        closes = {}
        for symbol in [etf] + basket:
            frame = all_data.get(symbol)
            if frame is None or frame.empty or 'close' not in frame.columns:
                return None
            closes[symbol] = frame['close']
        aligned = pd.DataFrame(closes).dropna()
        if len(aligned) < self.rv_z_window + 1 \
                or aligned.index[-1] != pd.Timestamp(date):
            return None
        log_px = np.log(aligned)
        spread = log_px[etf] - log_px[basket].mean(axis=1)
        mean = spread.rolling(self.rv_z_window).mean().iloc[-1]
        std = spread.rolling(self.rv_z_window).std().iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std <= 0:
            return None
        return float((spread.iloc[-1] - mean) / std)

    @staticmethod
    def _rv_frame_arrays(frame: pd.DataFrame):
        """(closes float64 array, int64 index array, index dtype) of a
        raw symbol frame, or None when the shape is one the cache does
        not model. The int64 timestamps are in the INDEX'S OWN datetime64
        unit (pandas 3 indexes are not always nanoseconds), so the cache
        also carries the dtype and requires every symbol to share it —
        otherwise the integer keys would not be comparable."""
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
            return None
        closes = frame['close']
        if not isinstance(closes, pd.Series):
            return None  # duplicate 'close' columns
        vals = closes.to_numpy()
        if vals.dtype != np.float64 or vals.ndim != 1:
            return None
        return (np.ascontiguousarray(vals),
                np.ascontiguousarray(index.asi8), index.dtype)

    def _rv_zscore_cached(self, all_data: Dict[str, pd.DataFrame], date,
                          etf: str, basket: List[str]):
        """Cache-backed `_rv_zscore` body: validate + consume each
        symbol's new rows, emit newly-common timestamps into the aligned
        block, then compute over the whole cached block. Returns the z
        (float or None) or `_RV_REFUSE` to fall through."""
        cols = [etf] + basket
        frames = {}
        for symbol in cols:  # the full path's early gate, same order
            frame = all_data.get(symbol)
            if frame is None or frame.empty or 'close' not in frame.columns:
                return None
            frames[symbol] = frame
        cache = self._rv_cache
        if cache is None or cache['cols'] != tuple(cols):
            return self._rv_rebuild(frames, cols, date)
        for symbol in cols:
            arrs = self._rv_frame_arrays(frames[symbol])
            if arrs is None:
                return _RV_REFUSE
            vals, ts, dt_dtype = arrs
            if dt_dtype != cache['dt_dtype']:
                return self._rv_rebuild(frames, cols, date)
            c = cache['consumed'][symbol]
            n = len(vals)
            if n < c:
                return self._rv_rebuild(frames, cols, date)
            if not (np.array_equal(vals[:c].view(np.int64),
                                   cache['raw_vals'][symbol].view(np.int64))
                    and np.array_equal(ts[:c], cache['raw_ts'][symbol])):
                return self._rv_rebuild(frames, cols, date)
            if n > c:
                new_vals, new_ts = vals[c:], ts[c:]
                if (c and new_ts[0] <= cache['raw_ts'][symbol][-1]) \
                        or (len(new_ts) > 1
                            and (np.diff(new_ts) <= 0).any()):
                    return _RV_REFUSE  # non-increasing raw index
                cache['raw_vals'][symbol] = np.concatenate(
                    [cache['raw_vals'][symbol], new_vals])
                cache['raw_ts'][symbol] = np.concatenate(
                    [cache['raw_ts'][symbol], new_ts])
                cache['consumed'][symbol] = n
                keep = new_vals == new_vals  # drops NaN, keeps ±inf
                pend = cache['pending'][symbol]
                for t, v in zip(new_ts[keep].tolist(),
                                new_vals[keep].tolist()):
                    pend[t] = v
        # A timestamp joins the aligned block once EVERY symbol has a
        # non-NaN close for it (exactly `dropna` on the outer join).
        pending = cache['pending']
        common = set(pending[cols[0]])
        for symbol in cols[1:]:
            if not common:
                break
            common &= pending[symbol].keys()
        if common:
            order = sorted(common)
            if order[0] <= cache['last_aligned_ts']:
                # would land out of aligned order: restatement — rebuild
                return self._rv_rebuild(frames, cols, date)
            add = np.empty((len(cols), len(order)))
            for j, symbol in enumerate(cols):
                pend = pending[symbol]
                add[j] = [pend.pop(t) for t in order]
            cache['block'] = np.concatenate([cache['block'], add], axis=1)
            cache['ts'] = np.concatenate(
                [cache['ts'], np.asarray(order, dtype=np.int64)])
            cache['last_aligned_ts'] = int(cache['ts'][-1])
        return self._rv_compute(cache, date, etf, basket)

    def _rv_rebuild(self, frames: Dict[str, pd.DataFrame], cols: List[str],
                    date):
        """Cold start or any validation failure: rebuild the cache
        through the FULL PATH's own construction (`pd.DataFrame(dict)
        .dropna()`), so the cached rows are the full path's rows by
        construction, then answer from the cache."""
        self._rv_cache = None  # never leave a half-primed cache behind
        per = {}
        dt_dtype = None
        for symbol in cols:
            arrs = self._rv_frame_arrays(frames[symbol])
            if arrs is None:
                return _RV_REFUSE
            vals, ts, dtype = arrs
            if dt_dtype is None:
                dt_dtype = dtype
            elif dtype != dt_dtype:
                return _RV_REFUSE  # mixed datetime64 units across symbols
            if len(ts) > 1 and (np.diff(ts) <= 0).any():
                return _RV_REFUSE
            per[symbol] = (vals, ts)
        closes = {symbol: frames[symbol]['close'] for symbol in cols}
        aligned = pd.DataFrame(closes).dropna()
        if aligned.index.dtype != dt_dtype:
            return _RV_REFUSE  # join changed the unit: keys incomparable
        block = np.ascontiguousarray(aligned.values.T)
        ts_aligned = np.ascontiguousarray(aligned.index.asi8)
        last = int(ts_aligned[-1]) if len(ts_aligned) else _RV_TS_MIN
        pending = {}
        for symbol in cols:
            vals, ts = per[symbol]
            # rows above the aligned frontier can become common later;
            # rows at/below it only via a restatement, which the prefix
            # byte-check catches and turns into another rebuild
            sel = (vals == vals) & (ts > last)
            pending[symbol] = dict(zip(ts[sel].tolist(),
                                       vals[sel].tolist()))
        cache = {
            'cols': tuple(cols),
            'dt_dtype': dt_dtype,
            'consumed': {s: len(per[s][0]) for s in cols},
            'raw_vals': {s: per[s][0].copy() for s in cols},
            'raw_ts': {s: per[s][1].copy() for s in cols},
            'pending': pending,
            'block': block,
            'ts': ts_aligned,
            'last_aligned_ts': last,
        }
        self._rv_cache = cache
        return self._rv_compute(cache, date, cols[0], cols[1:])

    def _rv_compute(self, cache: Dict, date, etf: str,
                    basket: List[str]) -> Optional[float]:
        """Gates + spread/rolling over the cached aligned block — the
        full path's expressions verbatim on a zero-copy DataFrame wrap
        of the block (same values, dtype, C-layout and column order, so
        every downstream kernel sees exactly what the full path's fresh
        `aligned` frame would give it)."""
        ts = cache['ts']
        if len(ts) < self.rv_z_window + 1:
            return None
        index = pd.DatetimeIndex(ts.view(cache['dt_dtype']))
        if index[-1] != pd.Timestamp(date):
            return None
        aligned = pd.DataFrame(cache['block'].T, index=index,
                               columns=list(cache['cols']), copy=False)
        log_px = np.log(aligned)
        spread = log_px[etf] - log_px[basket].mean(axis=1)
        mean = spread.rolling(self.rv_z_window).mean().iloc[-1]
        std = spread.rolling(self.rv_z_window).std().iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std <= 0:
            return None
        return float((spread.iloc[-1] - mean) / std)

    def _reconcile_rv(self, portfolio: PortfolioManager) -> List[DeskIntent]:
        """Pairs-style leg reconcile: a leg closed externally (stock
        stop-loss) breaks the spread — close the survivors; an entry
        that never filled is dropped after the grace period."""
        if self._rv_position is None:
            return []
        state = self._rv_position
        if self._day_index - state['entry_day'] < RECONCILE_GRACE_DAYS:
            return []
        open_legs = {symbol: direction
                     for symbol, direction in state['legs'].items()
                     if (portfolio.get_position(_stock(symbol)) is not None
                         and portfolio.get_position(_stock(symbol)).quantity
                         != 0)}
        if len(open_legs) == len(state['legs']):
            return []
        intents: List[DeskIntent] = []
        if not open_legs:
            self.note('info',
                      "Relative-value cleanup: spread entry never filled "
                      "or fully closed externally; tracking dropped",
                      book='relative_value',
                      legs=sorted(state['legs']))
            self._rv_position = None
            return []
        for symbol in sorted(open_legs):
            direction = open_legs[symbol]
            intents.append(DeskIntent(
                asset=_stock(symbol),
                action='SELL' if direction == 'long' else 'COVER',
                size_fraction=1.0,
                reason='relative-value spread broken: a leg was closed '
                       'externally'))
        self.note('risk',
                  f"Relative-value spread broken: leg(s) "
                  f"{sorted(set(state['legs']) - set(open_legs))} closed "
                  f"externally; closing survivors {sorted(open_legs)}",
                  book='relative_value', closed_legs=sorted(open_legs))
        self._rv_position = None
        return intents

    def _relative_value_intents(self, all_data: Dict[str, pd.DataFrame],
                                date, portfolio: PortfolioManager
                                ) -> List[DeskIntent]:
        weight = self.book_weights['relative_value']
        if weight <= 0 or not self._regime_controller.is_fitted:
            return []  # nothing before the first regime fit
        etf, basket = self._rv_universe(all_data)
        if etf is None:
            return []
        z_score = self._rv_zscore(all_data, date, etf, basket)
        if z_score is None:
            return []
        intents: List[DeskIntent] = []

        if self._rv_position is not None:
            if abs(z_score) < self.rv_exit_z:
                state = self._rv_position
                for symbol in sorted(state['legs']):
                    direction = state['legs'][symbol]
                    position = portfolio.get_position(_stock(symbol))
                    if position is None or position.quantity == 0:
                        continue
                    intents.append(DeskIntent(
                        asset=_stock(symbol),
                        action='SELL' if direction == 'long' else 'COVER',
                        size_fraction=1.0,
                        reason=(f"relative-value exit: |z|="
                                f"{abs(z_score):.2f} < "
                                f"{self.rv_exit_z:.2f}")))
                self.note('signal',
                          f"Relative-value exit: z={z_score:+.2f} inside "
                          f"+/-{self.rv_exit_z:.2f}; closing "
                          f"{sorted(state['legs'])}",
                          book='relative_value', z=z_score,
                          legs=sorted(state['legs']))
                self._rv_position = None
            return intents

        if abs(z_score) <= self.rv_entry_z:
            return intents
        # ETF rich (z > 0): short the ETF, long the basket — and mirror.
        etf_direction = 'short' if z_score > 0 else 'long'
        basket_direction = 'long' if z_score > 0 else 'short'
        legs: Dict[str, str] = {etf: etf_direction}
        for symbol in basket:
            legs[symbol] = basket_direction
        # Symbol exclusivity: every leg must be free of stock positions.
        for symbol in legs:
            position = portfolio.get_position(_stock(symbol))
            if position is not None and position.quantity != 0:
                return intents
        # Size the ETF leg STRICTLY INSIDE the shared position-size cap
        # (Phase 8 validation: at registry defaults weight * 0.5 = 0.10
        # lands exactly ON the RiskManager's 10% cap, and the apply_risk
        # ratio check dies to floating-point drift whenever the
        # portfolio is above water — the book was effectively dead).
        # The basket half mirrors the (possibly clamped) ETF half so the
        # spread stays balanced.
        cap = (self.risk_manager.max_position_size
               / max(self.capital_allocation, 1e-12))
        etf_fraction = min(weight * 0.5, cap * (1.0 - 1e-6))
        if self.rv_max_fraction is not None:
            etf_fraction = min(etf_fraction, self.rv_max_fraction)
        basket_fraction = etf_fraction / len(basket)
        for symbol in sorted(legs):
            direction = legs[symbol]
            fraction = etf_fraction if symbol == etf else basket_fraction
            intents.append(DeskIntent(
                asset=_stock(symbol),
                action='BUY' if direction == 'long' else 'SHORT',
                size_fraction=fraction,
                reason=(f"relative-value {direction}: z={z_score:+.2f} "
                        f"beyond +/-{self.rv_entry_z:.1f} "
                        f"({'ETF rich' if z_score > 0 else 'ETF cheap'})")))
        self._rv_position = {'legs': legs, 'entry_day': self._day_index,
                             'z_entry': z_score}
        self.note('signal',
                  f"Relative-value entry: z={z_score:+.2f} — "
                  f"{etf_direction.upper()} {etf} "
                  f"({etf_fraction:.1%}) vs {basket_direction.upper()} "
                  f"basket {basket} ({basket_fraction:.1%} per leg)",
                  book='relative_value', z=z_score, etf=etf,
                  etf_direction=etf_direction, basket=basket,
                  etf_fraction=etf_fraction,
                  basket_fraction=basket_fraction)
        return intents
