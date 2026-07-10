"""
Backtesting Engine - Simulates trading strategies on historical data

Execution model (no same-bar lookahead):
    A signal generated from day T's data is queued as a pending intent and
    filled at day T+1's OPEN, adjusted for slippage. Signals generated on the
    final simulated day therefore never fill; they are surfaced in the report
    under 'pending_signals'.

Two driving modes (exactly one per engine instance):
    strategy mode — a strategies.base.Strategy emits per-symbol BUY/SELL
        signals; position sizing is the run() position_size fraction.
    desk mode — a desks.base.Desk emits DeskIntent objects (generate_intents
        then apply_risk each day); each fill is sized as portfolio_value *
        desk.capital_allocation * intent.size_fraction at fill time. The
        report additionally carries 'desk', 'trader_notes' and
        'walk_forward' (contract C3), plus 'regime_series' when the desk
        exposes a non-empty regime_series property (contract C5).

        Desk mode also supports SHORT/COVER intents (contract C4): SHORT
        opens a negative position (a short is a SELL, so adverse slippage
        LOWERS the fill); COVER closes it (a buy: adverse slippage raises
        the fill). Margin is NOT modeled — cash-account approximation,
        short proceeds held as cash. Strategy mode has no short path.

OPTIONS IN DESK MODE (Phase 8 — SYNTHETIC PRICING, see desks/options_pricing):
    Free providers carry no historical option chains, so option prices are
    SYNTHETIC — Black-Scholes from the underlying's OHLCV with IV modeled
    from realized vol. NO-LOOKAHEAD: fills price with vol from closes
    STRICTLY BEFORE the fill date and spot = the fill-day OPEN; end-of-day
    mark-to-market uses spot = the session CLOSE. Real chain pricing
    arrives in Phase 9 (E*TRADE).

    OPTIONS COST MODEL (replaces slippage_bps for option assets): traded
    price = model price + haircut when buying / - haircut when selling,
    haircut = max(model price * spread_pct, $0.05) (spread_pct default
    0.02), plus per_contract_commission (default $0.65) per contract per
    leg, cash-only. All option cash flows, position values and P&L carry
    the x100 contract multiplier. The fractional stock commission and
    slippage_bps do NOT apply to options.

    EXPIRY SETTLEMENT (safety net): any option position whose expiry has
    PASSED (expiration_date < current date) is force-settled at intrinsic
    value computed from the underlying's close on the last session <= the
    expiry date — cash settle, x100, no spread cost, no commission, trade
    logged with reason 'expiry settlement'. Desks normally exit before
    expiry; this is the backstop.
"""

from __future__ import annotations

import logging
import math

import pandas as pd
import numpy as np
import threading
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from core.models import Asset, AssetType, Position
from desks.base import Desk
from desks.features import enrich_extended
from portfolio.manager import PortfolioManager
from data.market_data import MarketDataHandler
from strategies.base import Strategy
from analysis.research_stats import (benjamini_hochberg, bonferroni_alpha,
                                     fold_oos_pvalue, fold_oos_tstat)

if TYPE_CHECKING:  # annotation only — avoids any import-order coupling
    from desks.orchestrator import FundOrchestrator
    from desks.dynamic_reweighter import DynamicReweighter

logger = logging.getLogger(__name__)

#: Serializes SEEDED runs only. np.random.seed pins numpy's PROCESS-GLOBAL
#: RNG, so two concurrently-running seeded backtests (JobManager uses daemon
#: threads) would otherwise interleave draws and silently break each other's
#: reproducibility. Held for the duration of a seeded run; unseeded runs (the
#: default) never touch it and stay fully concurrent. True per-run isolation
#: would need a np.random.Generator threaded through every RNG consumer — a
#: larger refactor deferred beyond this step.
_SEEDED_RUN_LOCK = threading.Lock()

#: Trading days an intent may wait for a usable bar before being dropped.
MAX_PENDING_DAYS = 5

#: Minimum absolute half-spread haircut per option fill ($/share).
MIN_OPTION_HAIRCUT = 0.05

#: Cap on the realistic-fill market-impact fraction (Step 6). Bounds a huge
#: (uncapped close) order so the impacted fill price can never reach/cross zero
#: — mirrors the options cost model flooring its traded price at 0.0.
_MAX_IMPACT_FRACTION = 0.5


class BacktestEngine:
    """Simulates trading strategies on historical data"""

    def __init__(self, strategy: Optional[Strategy] = None,
                 initial_capital: float = 100000,
                 commission: float = 0.001, slippage_bps: float = 5.0,
                 desk: Optional[Desk] = None,
                 spread_pct: float = 0.02,
                 per_contract_commission: float = 0.65,
                 orchestrator: Optional['FundOrchestrator'] = None,
                 reweighter: Optional['DynamicReweighter'] = None,
                 seed: Optional[int] = None,
                 enable_realistic_fills: bool = False,
                 impact_coef: float = 0.1,
                 participation_cap: float = 0.1,
                 adv_window: int = 20,
                 market_data: Optional[MarketDataHandler] = None):
        if sum(driver is not None
               for driver in (strategy, desk, orchestrator)) != 1:
            raise ValueError(
                "Provide exactly one of strategy=, desk= or orchestrator= "
                "to BacktestEngine")
        if reweighter is not None and orchestrator is None:
            raise ValueError(
                "reweighter= requires orchestrator= (dynamic reweighting only "
                "applies to a fund of desks)")
        self.strategy = strategy
        self.desk = desk
        # Fund mode: an orchestrator drives N desks against the shared
        # portfolio. It exposes the desk read surface the engine needs
        # (capital_allocation, price_option, notes, walk_forward_fits), so
        # the desk-mode fill/mark/settle/report paths work for it too.
        self.orchestrator = orchestrator
        # Optional dynamic reweighter (fund mode): consulted once per day after
        # the snapshot to shift desk capital_allocation for the next window.
        # None -> a fund runs at its construction-time weights, unchanged.
        self.reweighter = reweighter
        # Reproducibility (Phase 3): records the RNG seed; the actual
        # np.random.seed() happens at the START of run() (under a lock), not
        # here, so nothing consumed between construction and run() can leak into
        # the simulation's RNG stream. Seeding pins numpy's LEGACY global RNG —
        # it covers desks/strategies/models that draw from np.random.* (HMM
        # init, ML estimators reading global state) but NOT components with
        # their own np.random.Generator (e.g. SyntheticLOB's default_rng, which
        # is independently seeded). None -> unpinned (prior behavior).
        self.rng_seed = seed
        self.portfolio = PortfolioManager(initial_capital)
        self.commission = commission
        self.slippage_bps = slippage_bps
        # Options cost model (module docstring); option assets only.
        self.spread_pct = spread_pct
        self.per_contract_commission = per_contract_commission
        # Realistic execution (Phase 3 Step 6) — OPT-IN, default OFF so every
        # existing backtest and the greeks golden stay BYTE-IDENTICAL. When on:
        # square-root ADV market impact (impact_coef*sqrt(filled/ADV)) is added
        # to slippage on STOCK fills, and an order is capped at
        # participation_cap*ADV per day with the remainder re-queued to fill
        # over subsequent days (accumulating into the position). ADV is the
        # trailing adv_window-day mean of volume STRICTLY BEFORE the fill date
        # (no lookahead). Options are unaffected (synthetic, no contract volume).
        self.enable_realistic_fills = enable_realistic_fills
        self.impact_coef = impact_coef
        self.participation_cap = participation_cap
        self.adv_window = adv_window
        # Injectable price source. Default = live MarketDataHandler (OpenBB).
        # Pass a WarehouseMarketData to run a SURVIVORSHIP-FREE backtest that can
        # hold delisted names (the live feed silently drops them).
        self.market_data = market_data or MarketDataHandler()
        self.trades_log: List[Dict] = []
        self.signals_log: List[Dict] = []
        # Pending intents keyed by Asset. Each value is a dict with keys:
        # 'signal' ('BUY'/'SELL'), 'signal_date', and 'days_waiting' (count of
        # trading days the intent has waited because the symbol had no usable
        # bar). Desk-mode BUY intents additionally carry 'size_fraction'
        # (fraction of the DESK'S capital). A new intent for an asset
        # replaces an older pending one.
        self.pending_intents: Dict[Asset, Dict] = {}

    @property
    def _desk_mode(self) -> bool:
        """True when a desk OR an orchestrator drives the engine — the modes
        that settle/mark/queue options and emit desk reports."""
        return self.desk is not None or self.orchestrator is not None

    @property
    def _driver(self):
        """The desk or orchestrator driving this run (None in strategy mode).
        Both expose key/name/notes/walk_forward_fits/price_option, so the
        desk-mode paths read whichever is set off this one accessor."""
        return self.desk if self.desk is not None else self.orchestrator

    def _price_option(self, asset: Asset, underlying_frame: pd.DataFrame,
                      date, spot: float) -> Optional[float]:
        """Synthetic option fair value via the driver (desk or orchestrator;
        the orchestrator routes to the owning options desk)."""
        return self._driver.price_option(asset, underlying_frame, date, spot)

    def run(self, symbols: List[str], start_date: str, end_date: str,
            position_size: float = 0.1,
            progress_callback: Optional[Callable[[float], None]] = None,
            benchmark_symbol: Optional[str] = 'SPY') -> Dict:
        """Run the backtest, pinning numpy's global RNG when a seed was given.

        Thin wrapper over _run_impl: when ``self.rng_seed`` is set, it seeds the
        global RNG at the very start of the run AND holds _SEEDED_RUN_LOCK for
        the whole run so a concurrent seeded run cannot interleave RNG draws.
        Unseeded runs skip the lock entirely and stay fully concurrent.
        """
        if self.rng_seed is None:
            return self._run_impl(symbols, start_date, end_date, position_size,
                                  progress_callback, benchmark_symbol)
        with _SEEDED_RUN_LOCK:
            np.random.seed(self.rng_seed)
            return self._run_impl(symbols, start_date, end_date, position_size,
                                  progress_callback, benchmark_symbol)

    def _run_impl(self, symbols: List[str], start_date: str, end_date: str,
                  position_size: float = 0.1,
                  progress_callback: Optional[Callable[[float], None]] = None,
                  benchmark_symbol: Optional[str] = 'SPY') -> Dict:
        """Run backtest on given symbols.

        Each simulated trading day executes, in this order:
            1. Fill pending intents (queued on a previous day) at TODAY'S OPEN
               (slippage applied; falls back to today's close if the open is
               missing/NaN). Intents whose symbol has no bar today stay
               pending for up to MAX_PENDING_DAYS trading days, then drop.
            2. Mark open positions to today's close.
            3. Compute indicators/signals on data through today (expanding
               window) and queue resulting BUY/SELL intents for the next day.
            4. Record the portfolio snapshot.

        progress_callback, when given, is invoked after each simulated day
        with the percentage of trading days processed (float 0-100,
        monotonically nondecreasing, exactly 100.0 once at completion).

        benchmark_symbol, when set, adds a buy-and-hold benchmark equity
        curve to the report under 'benchmark' (None on any benchmark data
        failure — the backtest itself never fails because of the benchmark).
        """
        all_data = {}
        for symbol in symbols:
            data = self.market_data.fetch_stock_data(symbol, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data

        if not all_data:
            logger.warning("Backtest aborted: no data available for %s", symbols)
            return {'error': 'No data available'}

        # Indicators are prefix-stable (rolling/ewm/diff/shift are forward-
        # only), so computing them ONCE on the full frame and slicing through
        # the simulated date is byte-identical to recomputing on each
        # expanding window — and removes the O(days^2) indicator recompute
        # from the daily loop. Slicing through `date` still guarantees no
        # future context reaches a strategy or desk. enrich_extended layers
        # the extended-feature extras/seasonal columns on by the same
        # prefix-stability argument; they are inert for every consumer that
        # selects named columns and let the ML models' predict fast paths
        # read the last row instead of rebuilding O(history) features daily.
        self._enriched_all = {
            symbol: enrich_extended(
                self.market_data.calculate_indicators(data.copy()))
            for symbol, data in all_data.items()
        }

        # Align trading dates across all downloaded symbols
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        sorted_dates = sorted(all_dates)

        self.pending_intents = {}

        # Simulate sequential historical trading day by day
        total_days = len(sorted_dates)
        for day_number, date in enumerate(sorted_dates, start=1):
            # --- PHASE 0 (desk mode): SETTLE OPTIONS PAST EXPIRY ---
            if self._desk_mode:
                self._settle_expired_options(all_data, date)

            # --- PHASE 1: FILL PENDING INTENTS AT TODAY'S OPEN ---
            self._fill_pending_intents(all_data, date, position_size)

            # --- PHASE 2: MARK POSITIONS TO TODAY'S CLOSE ---
            for symbol, data in all_data.items():
                if date not in data.index:
                    continue
                asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
                existing_pos = self.portfolio.get_position(asset)
                if existing_pos:
                    close = float(data.loc[date, 'close'])
                    # A non-finite close (NaN from a missing/halted bar, or a
                    # +/-inf from a provider parse glitch) must NOT overwrite
                    # the mark: a single bad value flows through
                    # get_portfolio_value() and turns the WHOLE equity curve —
                    # and every downstream metric (Sharpe, drawdown, Calmar) —
                    # into NaN/inf while the backtest still "completes"
                    # silently. np.isfinite catches NaN AND inf (np.isnan
                    # would let inf through). Keep the prior mark and surface.
                    if not np.isfinite(close):
                        logger.warning(
                            "Non-finite close (%s) for %s on %s; keeping prior "
                            "mark %s", close, symbol, date,
                            existing_pos.current_price)
                    else:
                        existing_pos.current_price = close
            if self._desk_mode:
                self._mark_option_positions(all_data, date)

            # --- PHASE 3: SIGNALS ON DATA THROUGH TODAY, QUEUED FOR TOMORROW ---
            if self.orchestrator is not None:
                self._run_orchestrator_step(all_data, date)
            elif self.desk is not None:
                self._queue_desk_intents(all_data, date)
            else:
                for symbol, data in all_data.items():
                    if date not in data.index:
                        continue

                    # Precomputed indicators sliced through today only
                    historical_data_with_indicators = self._enriched_through(
                        symbol, date)

                    asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
                    signal = self.strategy.generate_signals(historical_data_with_indicators, asset)

                    if signal in ('BUY', 'SELL'):
                        # A new intent for an asset replaces an older pending one.
                        self._queue_pending_intent(asset, {
                            'signal': signal,
                            'signal_date': date,
                            'days_waiting': 0,
                        })

            # --- PHASE 4: RECORD SNAPSHOT ---
            self.portfolio.record_snapshot(date)

            # --- FUND REWEIGHT (optional, fund mode): on a rebalance boundary
            # shift each desk's capital_allocation from its standalone curve so
            # the NEXT day's intent generation/sizing uses the new weights.
            # Runs AFTER the snapshot so today's close is observable and no
            # already-generated intent is retroactively resized. ---
            if self.reweighter is not None:
                weights = self.reweighter.on_day(
                    self.orchestrator.desks, date, day_number)
                if weights is not None:
                    self.orchestrator.active_capital = sum(
                        desk.capital_allocation
                        for desk in self.orchestrator.desks)

            if progress_callback is not None:
                if day_number == total_days:
                    pct = 100.0
                else:
                    # Guard against float rounding ever reporting an early
                    # 100.0: only the final day may emit exactly 100.0.
                    pct = min(100.0 * day_number / total_days, 99.99)
                progress_callback(pct)

        return self._generate_report(benchmark_symbol=benchmark_symbol,
                                     start_date=start_date, end_date=end_date)

    def _enriched_through(self, symbol: str, date) -> pd.DataFrame:
        """Slice the precomputed indicator frame through `date` (inclusive).

        Equivalent to masking `index <= date` — the searchsorted fast path
        needs a sorted index; unsorted frames take the mask path so the
        no-future-leakage guarantee never depends on sortedness.
        """
        frame = self._enriched_all[symbol]
        if frame.index.is_monotonic_increasing:
            pos = frame.index.searchsorted(date, side='right')
            return frame.iloc[:pos]
        return frame[frame.index <= date]

    def _queue_desk_intents(self, all_data: Dict[str, pd.DataFrame],
                            date) -> None:
        """Desk-mode PHASE 3: ask the desk for intents, risk-check, queue.

        Builds the per-symbol indicator-enriched frames sliced through the
        current simulation date (same expanding-window pattern as strategy
        mode, so no future context can leak into the desk), sets the
        desk's simulation clock, then runs generate_intents -> apply_risk.
        Approved intents are queued as pending fills for the next bar's
        open; BUY records carry the intent's size_fraction.
        """
        self.desk.set_clock(date)

        enriched: Dict[str, pd.DataFrame] = {}
        for symbol in all_data:
            # Precomputed indicators sliced through today only
            historical_data = self._enriched_through(symbol, date)
            if historical_data.empty:
                continue
            enriched[symbol] = historical_data

        if not enriched:
            return

        intents = self.desk.generate_intents(enriched, date, self.portfolio)
        approved = self.desk.apply_risk(intents, self.portfolio, enriched, date)

        for intent in approved:
            # A new intent for an asset replaces an older pending one.
            self._queue_pending_intent(intent.asset, {
                'signal': intent.action,
                'signal_date': date,
                'days_waiting': 0,
                'size_fraction': intent.size_fraction,
                # Absolute size override (contracts/shares); None defers
                # to size_fraction dollar sizing at fill time.
                'quantity': intent.quantity,
            })

    def _run_orchestrator_step(self, all_data: Dict[str, pd.DataFrame],
                               date) -> None:
        """Fund-mode PHASE 3: drive N desks through the orchestrator and
        queue the netted, risk-approved intents for the next bar's open.

        Mirrors _queue_desk_intents: builds the same per-symbol
        indicator-enriched frames sliced through today (no future leakage),
        sets the orchestrator clock (which cascades to every sub-desk), and
        runs orchestrator.step (generate_intents per desk -> net -> one
        account-wide apply_risk). Approved intents carry an ACCOUNT-ABSOLUTE
        size_fraction, so _fill_intent sizes them with no per-desk capital
        math (orchestrator.capital_allocation is 1.0).
        """
        self.orchestrator.set_clock(date)

        enriched: Dict[str, pd.DataFrame] = {}
        for symbol in all_data:
            historical_data = self._enriched_through(symbol, date)
            if historical_data.empty:
                continue
            enriched[symbol] = historical_data

        if not enriched:
            return

        approved = self.orchestrator.step(enriched, date, self.portfolio)
        for intent in approved:
            # A new intent for an asset replaces an older pending one.
            self._queue_pending_intent(intent.asset, {
                'signal': intent.action,
                'signal_date': date,
                'days_waiting': 0,
                'size_fraction': intent.size_fraction,
                'quantity': intent.quantity,
                # Fund-mode ownership: stamped onto the opened Position so
                # each desk's book logic only touches positions it owns.
                'desk_keys': intent.desk_keys,
            })

    # ------------------------------------------------------------------
    # Realistic execution (Step 6, opt-in) — STOCK fills only
    # ------------------------------------------------------------------
    def _average_daily_volume(self, data: Optional[pd.DataFrame],
                              date) -> Optional[float]:
        """Trailing adv_window-day mean of volume STRICTLY BEFORE `date`.

        No lookahead: the fill is at today's open, before today's volume is
        known, so today's bar is excluded. Returns None when volume is
        unavailable or non-positive (the caller then applies no impact/cap).
        """
        if data is None or 'volume' not in getattr(data, 'columns', []):
            return None
        prior = data[data.index < pd.Timestamp(date)].sort_index()
        if prior.empty:
            return None
        vols = prior['volume'].tail(self.adv_window).dropna()
        if vols.empty:
            return None
        adv = float(vols.mean())
        return adv if adv > 0 else None

    def _impact_fraction(self, quantity: int,
                         adv: Optional[float]) -> float:
        """Square-root market impact (Almgren) as an adverse price fraction:
        ``impact_coef * sqrt(quantity / ADV)``. 0.0 when ADV is unknown or
        non-positive, or quantity <= 0."""
        if not adv or adv <= 0 or quantity <= 0:
            return 0.0
        return min(_MAX_IMPACT_FRACTION,
                   self.impact_coef * math.sqrt(quantity / adv))

    def _capped_fill_quantity(self, desired: int,
                              adv: Optional[float]):
        """Participation cap: at most ``participation_cap * ADV`` units fill
        today. Returns ``(filled, remainder)``; no cap (remainder 0) when ADV
        is unknown or the cap is not binding."""
        if not adv or adv <= 0:
            return desired, 0
        # Floor the cap at 1 share: int(participation_cap*ADV) rounds to 0 for
        # very thin names — exactly where the cap matters MOST — so without the
        # floor the whole order would dump uncapped. Thin names trickle instead.
        cap = max(1, int(self.participation_cap * adv))
        if desired <= cap:
            return desired, 0
        return cap, desired - cap

    def _requeue_remainder(self, intent: Dict, remainder: int) -> Optional[Dict]:
        """Build a follow-on intent for the un-filled remainder (cap-and-requeue),
        or None when nothing remains. The remainder is an ABSOLUTE share count
        and carries 'accumulate' so its next-day fill adds to the position
        opened today rather than opening a fresh one. Returns None unless
        realistic fills are enabled."""
        if remainder <= 0 or not self.enable_realistic_fills:
            return None
        follow = dict(intent)
        follow['quantity'] = remainder
        follow['accumulate'] = True
        follow['days_waiting'] = 0
        return follow

    def _queue_pending_intent(self, asset, intent: Dict) -> None:
        """Queue a new pending intent for ``asset``, replacing any older one.

        Surfaces the realistic-fills edge where a fresh signal supersedes an
        in-flight cap-and-requeue remainder: the unfilled shares are abandoned,
        so warn instead of dropping them silently. Off-path no intent ever
        carries 'accumulate', so this is a plain assignment (byte-identical).
        """
        prior = self.pending_intents.get(asset)
        if prior is not None and prior.get('accumulate'):
            logger.warning(
                "Abandoning %s unfilled shares of %s: a new %s signal "
                "supersedes the in-flight partial fill",
                prior.get('quantity'), asset.symbol, intent.get('signal'))
        self.pending_intents[asset] = intent

    def _fill_pending_intents(self, all_data: Dict[str, pd.DataFrame],
                              date, position_size: float) -> None:
        """Fill intents queued on previous days at today's open price.

        Intents without a usable bar today (no row, or both open and close
        NaN) accrue a waiting day and are dropped after MAX_PENDING_DAYS.

        OPTION assets (desk mode): the symbol is the UNDERLYING ticker;
        the fill base is the desk's synthetic fair value priced with
        spot = the underlying's open today and vol from closes strictly
        before today (no-lookahead, module docstring). An unpriceable
        option (insufficient prior history) waits like a missing bar.
        """
        for asset in list(self.pending_intents.keys()):
            intent = self.pending_intents[asset]
            data = all_data.get(asset.symbol)

            fill_base = float('nan')
            if data is not None and date in data.index:
                row = data.loc[date]
                if 'open' in row.index:
                    fill_base = row['open']
                if pd.isna(fill_base) and 'close' in row.index:
                    # Missing/NaN open: fall back to today's close.
                    fill_base = row['close']

            is_option = asset.asset_type in (AssetType.CALL, AssetType.PUT)
            if is_option and not pd.isna(fill_base):
                # Fill spot = the underlying's OPEN today (fill_base).
                model_price = self._price_option(
                    asset, data[data.index <= date], date,
                    spot=float(fill_base))
                fill_base = (float(model_price) if model_price is not None
                             else float('nan'))

            if pd.isna(fill_base):
                intent['days_waiting'] += 1
                if intent['days_waiting'] >= MAX_PENDING_DAYS:
                    logger.warning(
                        "Dropping %s intent for %s (signal %s): no usable bar "
                        "for %d trading days",
                        intent['signal'], str(asset), intent['signal_date'],
                        intent['days_waiting'])
                    del self.pending_intents[asset]
                continue

            if is_option:
                self._fill_option_intent(asset, intent, float(fill_base), date)
                del self.pending_intents[asset]
            else:
                adv = (self._average_daily_volume(data, date)
                       if self.enable_realistic_fills else None)
                remainder = self._fill_intent(asset, intent, float(fill_base),
                                              date, position_size, adv=adv)
                # Cap-and-requeue: a partial stock fill returns a follow-on
                # intent for the remainder (same asset) to fill on later days;
                # otherwise the intent is consumed. With realistic fills off,
                # _fill_intent always returns None -> identical to before.
                if remainder is not None:
                    self.pending_intents[asset] = remainder
                else:
                    del self.pending_intents[asset]

    # ------------------------------------------------------------------
    # Options: cost model, fills, MTM, expiry settlement (desk mode)
    # ------------------------------------------------------------------
    def _option_traded_price(self, model_price: float, buying: bool) -> float:
        """Traded price under the options cost model (module docstring).

        haircut = max(model_price * spread_pct, MIN_OPTION_HAIRCUT);
        buys pay model + haircut, sells receive model - haircut (floored
        at 0.0 — nobody pays you to take a worthless option).
        """
        haircut = max(model_price * self.spread_pct, MIN_OPTION_HAIRCUT)
        if buying:
            return model_price + haircut
        return max(0.0, model_price - haircut)

    def _fill_option_intent(self, asset: Asset, intent: Dict,
                            model_price: float, fill_date) -> None:
        """Fill a queued OPTION intent at the synthetic fair value with
        the options cost model and the x100 contract multiplier.

        Sizing: intent['quantity'] (absolute contracts) when set, else
        value-sized like stock: contracts = int(trade_value / (traded
        price x 100)). SELL closes the full long, COVER the full short.
        Commission = per_contract_commission x contracts, cash-only
        (never baked into entry/exit prices — consistent with stock).
        """
        signal = intent['signal']
        existing_pos = self.portfolio.get_position(asset)

        if signal in ('BUY', 'SHORT') \
                and (not existing_pos or existing_pos.quantity == 0):
            buying = signal == 'BUY'
            fill_price = self._option_traded_price(model_price, buying=buying)
            if fill_price <= 0:
                # Guards BOTH open directions: an unguarded SHORT open at the
                # 0.0 sell floor would collect zero premium, pay commission,
                # and book unbounded risk (sizing at the haircut floor can
                # short thousands of contracts for $0 credit).
                logger.warning("Skipping %s fill for %s on %s: non-positive "
                               "traded price %.4f", signal, str(asset),
                               fill_date, fill_price)
                return
            quantity = intent.get('quantity')
            if quantity is None:
                if self.orchestrator is not None:
                    # Orchestrator intents are already account-absolute.
                    size_fraction = intent.get('size_fraction', 0.0)
                else:
                    size_fraction = (self.desk.capital_allocation
                                     * intent.get('size_fraction', 0.0))
                trade_value = (self.portfolio.get_portfolio_value()
                               * size_fraction)
                per_contract = max(fill_price, MIN_OPTION_HAIRCUT) * asset.multiplier
                quantity = int(trade_value / per_contract)
            if quantity <= 0:
                logger.warning(
                    "Dropping %s intent for %s on %s: sizes to 0 contracts",
                    signal, str(asset), fill_date)
                return

            commission = quantity * self.per_contract_commission
            if buying:
                cost = quantity * fill_price * asset.multiplier + commission
                if self.portfolio.cash < cost:
                    logger.warning(
                        "Dropping BUY intent for %s on %s: insufficient cash "
                        "(needed %.2f, available %.2f)",
                        str(asset), fill_date, cost, self.portfolio.cash)
                    return
                self.portfolio.cash -= cost
                signed_quantity = quantity
                log_extra = {'cost': cost}
            else:
                # Sell-to-open: proceeds credited (cash-account approx).
                proceeds = (quantity * fill_price * asset.multiplier
                            - commission)
                self.portfolio.cash += proceeds
                signed_quantity = -quantity
                log_extra = {'proceeds': proceeds}

            self.portfolio.add_position(Position(
                asset=asset, quantity=signed_quantity,
                avg_entry_price=fill_price, current_price=fill_price,
                timestamp=fill_date, owners=intent.get('desk_keys')))
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset),
                'action': signal, 'quantity': quantity,
                'price': fill_price, 'commission': commission, **log_extra})
            logger.info("%s %d %s @ %.4f on %s (model %.4f)", signal,
                        quantity, str(asset), fill_price, fill_date,
                        model_price)

        elif signal == 'SELL' and existing_pos and existing_pos.quantity > 0:
            fill_price = self._option_traded_price(model_price, buying=False)
            quantity = existing_pos.quantity
            commission = quantity * self.per_contract_commission
            proceeds = quantity * fill_price * asset.multiplier - commission
            self.portfolio.cash += proceeds
            self.portfolio.close_position(asset, fill_price, quantity,
                                          existing_pos.timestamp, fill_date)
            self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset),
                'action': 'SELL', 'quantity': quantity, 'price': fill_price,
                'commission': commission, 'proceeds': proceeds})
            logger.info("SELL %d %s @ %.4f on %s", quantity, str(asset),
                        fill_price, fill_date)

        elif signal == 'COVER' and existing_pos and existing_pos.quantity < 0:
            # Buy-to-close the short: debit cash; closes always execute.
            fill_price = self._option_traded_price(model_price, buying=True)
            quantity = existing_pos.quantity  # negative
            commission = abs(quantity) * self.per_contract_commission
            cost = abs(quantity) * fill_price * asset.multiplier + commission
            self.portfolio.cash -= cost
            self.portfolio.close_position(asset, fill_price, quantity,
                                          existing_pos.timestamp, fill_date)
            self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset),
                'action': 'COVER', 'quantity': abs(quantity),
                'price': fill_price, 'commission': commission, 'cost': cost})
            logger.info("COVER %d %s @ %.4f on %s", abs(quantity),
                        str(asset), fill_price, fill_date)

    def _mark_option_positions(self, all_data: Dict[str, pd.DataFrame],
                               date) -> None:
        """Mark option positions to the synthetic fair value at the CLOSE
        (spot = today's close; the IV model still uses strictly-prior
        closes — see desks/options_pricing). Positions whose underlying
        has no bar (or no priceable IV) keep their last mark."""
        for asset, position in self.portfolio.positions.items():
            if asset.asset_type not in (AssetType.CALL, AssetType.PUT):
                continue
            data = all_data.get(asset.symbol)
            if data is None or date not in data.index:
                continue
            close = data.loc[date, 'close']
            if pd.isna(close):
                continue
            price = self._price_option(
                asset, data[data.index <= date], date, spot=float(close))
            if price is not None:
                position.current_price = float(price)

    def _settle_expired_options(self, all_data: Dict[str, pd.DataFrame],
                                date) -> None:
        """Force-settle option positions whose expiry has PASSED at
        intrinsic value (module docstring: cash settle, x100, no spread,
        no commission). Intrinsic uses the underlying's close on the
        last session <= the expiry date; with no usable close the option
        settles at 0.0 (logged) — the desk should never let it get
        there."""
        current = pd.Timestamp(date).date()
        for asset in sorted(
                (a for a in self.portfolio.positions
                 if a.asset_type in (AssetType.CALL, AssetType.PUT)),
                key=str):
            expiry = pd.Timestamp(asset.expiration_date).date()
            if expiry >= current:
                continue
            position = self.portfolio.positions[asset]
            intrinsic = 0.0
            data = all_data.get(asset.symbol)
            if data is not None and not data.empty:
                closes = data.loc[data.index <= pd.Timestamp(expiry),
                                  'close'].dropna()
                if not closes.empty:
                    spot = float(closes.iloc[-1])
                    if asset.asset_type is AssetType.CALL:
                        intrinsic = max(0.0, spot - asset.strike_price)
                    else:
                        intrinsic = max(0.0, asset.strike_price - spot)
                else:
                    logger.warning("Expiry settlement of %s: no underlying "
                                   "close <= %s; settling at 0.0",
                                   str(asset), expiry)
            quantity = position.quantity
            # Sign-correct both ways: longs are PAID intrinsic, shorts
            # PAY it (quantity carries the sign).
            settlement = quantity * intrinsic * asset.multiplier
            self.portfolio.cash += settlement
            self.portfolio.close_position(asset, intrinsic, quantity,
                                          position.timestamp, date)
            self.portfolio.remove_position(asset)
            action = 'SELL' if quantity > 0 else 'COVER'
            entry = {
                'date': date, 'signal_date': date, 'symbol': asset.symbol,
                'instrument': str(asset), 'action': action,
                'quantity': abs(quantity), 'price': intrinsic,
                'reason': 'expiry settlement',
            }
            entry['proceeds' if quantity > 0 else 'cost'] = abs(settlement)
            self.trades_log.append(entry)
            logger.info(
                "Expiry settlement: %s %d %s at intrinsic %.4f on %s",
                action, abs(quantity), str(asset), intrinsic, date)

    def _accumulate_position(self, position: Position, add_quantity: int,
                             fill_price: float) -> None:
        """Add to an existing same-direction position (cap-and-requeue
        remainder), updating the size-weighted average entry price.

        add_quantity is POSITIVE for a long add, NEGATIVE for a short add; the
        position's sign is preserved. Only ever called under realistic fills.
        """
        old_qty = position.quantity
        new_qty = old_qty + add_quantity
        if new_qty == 0:
            return
        position.avg_entry_price = (
            (old_qty * position.avg_entry_price + add_quantity * fill_price)
            / new_qty)
        position.quantity = new_qty
        position.current_price = fill_price

    def _fill_intent(self, asset: Asset, intent: Dict, base_price: float,
                     fill_date, position_size: float,
                     adv: Optional[float] = None) -> Optional[Dict]:
        """Fill a queued intent at base_price (today's open) with slippage.

        Slippage is always adverse: buys (BUY, COVER) fill at
        base_price * (1 + slippage_bps/10000); sells (SELL, SHORT) fill at
        base_price * (1 - slippage_bps/10000). Position size is the
        position_size fraction of portfolio value evaluated at fill time;
        desk-mode intents instead size to desk.capital_allocation *
        intent['size_fraction'] (still a fraction of fill-time portfolio
        value). Commission semantics: cost = qty * fill * (1 + commission)
        on a buy; proceeds = qty * fill * (1 - commission) on a sell.

        SHORT/COVER (desk OR fund mode, contract C4): SHORT opens a NEGATIVE
        position of -qty shares with proceeds credited to cash (margin is
        not modeled — cash-account approximation); COVER closes the full
        short, debiting cash, and records the Trade with the negative
        quantity (Trade.pnl = qty * (exit - entry) is sign-correct). Gated on
        self._desk_mode, so a fund's netted-SHORT residual and account-level
        COVER fill the same way a single desk's do.

        REALISTIC FILLS (Step 6, opt-in via enable_realistic_fills; ``adv`` is
        the trailing volume passed by the caller): adds square-root ADV market
        impact to slippage and caps the filled size at participation_cap*ADV,
        re-queuing the remainder. Returns a follow-on intent dict for that
        remainder (to fill on later days, accumulating into the position) or
        None when the intent is fully consumed. With realistic fills OFF this
        always returns None and every fill is byte-identical to before.

        NOTE: dollar-sized orders (strategy mode) compute the share count against
        the PRE-impact price, so an uncapped realistic fill can exceed the dollar
        budget by up to the impact fraction (bounded by the cash guard).
        """
        signal = intent['signal']
        slippage = self.slippage_bps / 10000.0
        realistic = self.enable_realistic_fills
        if self.orchestrator is not None and 'size_fraction' in intent:
            # Orchestrator intents are already account-absolute fractions.
            position_size = intent['size_fraction']
        elif self.desk is not None and 'size_fraction' in intent:
            position_size = self.desk.capital_allocation * intent['size_fraction']

        if base_price <= 0:
            logger.warning("Skipping %s fill for %s on %s: non-positive price %s",
                           signal, asset.symbol, fill_date, base_price)
            return None

        existing_pos = self.portfolio.get_position(asset)

        # A re-queued remainder ('accumulate') whose position is gone (closed or
        # flipped between fills) cannot top up anything — drop it deterministically
        # rather than silently re-opening a fresh position or falling through.
        if realistic and intent.get('accumulate'):
            same_direction = existing_pos is not None and (
                (signal == 'BUY' and existing_pos.quantity > 0)
                or (signal == 'SHORT' and existing_pos.quantity < 0))
            if not same_direction:
                logger.warning(
                    "Dropping orphaned %s remainder for %s on %s: the position "
                    "it was accumulating into is gone", signal, asset.symbol,
                    fill_date)
                return None

        if signal == 'BUY' and (
                (not existing_pos or existing_pos.quantity == 0)
                or (realistic and intent.get('accumulate')
                    and existing_pos and existing_pos.quantity > 0)):
            base_fill = base_price * (1 + slippage)
            portfolio_value = self.portfolio.get_portfolio_value()
            trade_value = portfolio_value * position_size
            # Desk intents may carry an absolute share count (Phase 8); it
            # overrides the dollar sizing. Strategy mode never sets it. A
            # re-queued remainder (realistic fills) also carries it.
            desired = intent.get('quantity') or int(trade_value / base_fill)

            if desired == 0:
                logger.warning(
                    "Dropping BUY intent for %s on %s: position sizes to 0 "
                    "shares (trade value %.2f at fill price %.4f)",
                    asset.symbol, fill_date, trade_value, base_fill)
                return None

            if realistic:
                quantity, remainder = self._capped_fill_quantity(desired, adv)
                fill_price = base_price * (
                    1 + slippage + self._impact_fraction(quantity, adv))
            else:
                quantity, remainder = desired, 0
                fill_price = base_fill

            cost = quantity * fill_price * (1 + self.commission)
            if self.portfolio.cash < cost:
                logger.warning(
                    "Dropping BUY intent for %s on %s: insufficient cash "
                    "(needed %.2f, available %.2f)",
                    asset.symbol, fill_date, cost, self.portfolio.cash)
                return None

            self.portfolio.cash -= cost
            if existing_pos and existing_pos.quantity > 0:
                # Accumulate the re-queued remainder into the existing long.
                self._accumulate_position(existing_pos, quantity, fill_price)
            else:
                self.portfolio.add_position(Position(
                    asset=asset, quantity=quantity, avg_entry_price=fill_price,
                    current_price=fill_price, timestamp=fill_date,
                    owners=intent.get('desk_keys')))
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset), 'action': 'BUY',
                'quantity': quantity, 'price': fill_price, 'cost': cost
            })
            logger.info("BUY %d %s @ %.4f on %s (signal %s)",
                        quantity, asset.symbol, fill_price, fill_date,
                        intent['signal_date'])
            return self._requeue_remainder(intent, remainder)

        elif signal == 'SELL' and existing_pos and existing_pos.quantity > 0:
            quantity = existing_pos.quantity
            # Closes are not size-capped (a full exit), but pay market impact.
            if realistic:
                fill_price = base_price * (
                    1 - slippage - self._impact_fraction(quantity, adv))
            else:
                fill_price = base_price * (1 - slippage)
            proceeds = quantity * fill_price * (1 - self.commission)
            self.portfolio.cash += proceeds
            self.portfolio.close_position(
                asset, fill_price, quantity,
                existing_pos.timestamp, fill_date
            )
            self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset), 'action': 'SELL',
                'quantity': quantity, 'price': fill_price, 'proceeds': proceeds
            })
            logger.info("SELL %d %s @ %.4f on %s (signal %s)",
                        quantity, asset.symbol, fill_price, fill_date,
                        intent['signal_date'])

        elif signal == 'SHORT' and self._desk_mode and (
                (not existing_pos or existing_pos.quantity == 0)
                or (realistic and intent.get('accumulate')
                    and existing_pos and existing_pos.quantity < 0)):
            # A short is a SELL: adverse slippage means a LOWER fill.
            base_fill = base_price * (1 - slippage)
            portfolio_value = self.portfolio.get_portfolio_value()
            trade_value = portfolio_value * position_size
            desired = intent.get('quantity') or int(trade_value / base_fill)

            if desired == 0:
                logger.warning(
                    "Dropping SHORT intent for %s on %s: position sizes to 0 "
                    "shares (trade value %.2f at fill price %.4f)",
                    asset.symbol, fill_date, trade_value, base_fill)
                return None

            if realistic:
                quantity, remainder = self._capped_fill_quantity(desired, adv)
                fill_price = base_price * (
                    1 - slippage - self._impact_fraction(quantity, adv))
            else:
                quantity, remainder = desired, 0
                fill_price = base_fill

            proceeds = quantity * fill_price * (1 - self.commission)
            self.portfolio.cash += proceeds
            if existing_pos and existing_pos.quantity < 0:
                # Accumulate the re-queued remainder into the existing short.
                self._accumulate_position(existing_pos, -quantity, fill_price)
            else:
                self.portfolio.add_position(Position(
                    asset=asset, quantity=-quantity, avg_entry_price=fill_price,
                    current_price=fill_price, timestamp=fill_date,
                    owners=intent.get('desk_keys')))
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset), 'action': 'SHORT',
                'quantity': quantity, 'price': fill_price,
                'proceeds': proceeds
            })
            logger.info("SHORT %d %s @ %.4f on %s (signal %s)",
                        quantity, asset.symbol, fill_price, fill_date,
                        intent['signal_date'])
            return self._requeue_remainder(intent, remainder)

        elif signal == 'COVER' and self._desk_mode \
                and existing_pos and existing_pos.quantity < 0:
            # A cover is a BUY: adverse slippage means a HIGHER fill.
            quantity = existing_pos.quantity  # negative
            # Closes are not size-capped (a full exit), but pay market impact.
            if realistic:
                fill_price = base_price * (
                    1 + slippage + self._impact_fraction(abs(quantity), adv))
            else:
                fill_price = base_price * (1 + slippage)
            cost = abs(quantity) * fill_price * (1 + self.commission)
            # Closes always execute — even if cash dips negative the desk
            # must be able to exit a losing short (margin is not modeled).
            self.portfolio.cash -= cost
            self.portfolio.close_position(
                asset, fill_price, quantity,
                existing_pos.timestamp, fill_date
            )
            self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset), 'action': 'COVER',
                'quantity': abs(quantity), 'price': fill_price, 'cost': cost
            })
            logger.info("COVER %d %s @ %.4f on %s (signal %s)",
                        abs(quantity), asset.symbol, fill_price, fill_date,
                        intent['signal_date'])

        # No matching branch (e.g. a BUY when already long without accumulate,
        # or a close with no position) -> consume the intent. Realistic-fill
        # remainders are the only thing re-queued (returned above).
        return None

    def _build_benchmark(self, benchmark_symbol: str, start_date: str,
                         end_date: str) -> Optional[Dict]:
        """Buy-and-hold benchmark equity curve over the backtest date range.

        initial_capital is notionally invested at the first available
        benchmark close in range; the position is marked at each session
        close, so value[0] == initial_capital by construction. Fetched via
        self.market_data so the persistent OHLCV cache applies. Returns None
        (with a WARNING log) on any data failure — the caller must treat the
        benchmark as strictly optional.
        """
        try:
            data = self.market_data.fetch_stock_data(
                benchmark_symbol, start_date, end_date)
            if data is None or data.empty or 'close' not in data.columns:
                raise ValueError(f"no usable data for {benchmark_symbol}")

            closes = data['close'].dropna()
            if closes.empty:
                raise ValueError(f"all closes NaN for {benchmark_symbol}")

            base_close = float(closes.iloc[0])
            if base_close <= 0:
                raise ValueError(
                    f"non-positive base close {base_close} for {benchmark_symbol}")

            initial_capital = self.portfolio.initial_capital
            equity_curve = [
                {
                    'date': ts.strftime('%Y-%m-%d'),
                    'value': float(close) / base_close * initial_capital,
                }
                for ts, close in closes.items()
            ]
            return {'symbol': benchmark_symbol, 'equity_curve': equity_curve}
        except Exception as e:
            logger.warning("Benchmark %s unavailable (%s..%s): %s",
                           benchmark_symbol, start_date, end_date, e)
            return None

    def _compute_oos_folds(self, desk, alpha: float = 0.05) -> Dict:
        """Per-fold out-of-sample significance for a SINGLE-desk run.

        Reconstructs OOS folds by slicing the realized account return stream at
        the desk's distinct walk-forward fit dates: fold k spans
        ``[fit_date_k, fit_date_{k+1})`` (the last fold is open-ended). Each
        fold's returns are scored with a one-sided t-test that the mean OOS
        return > 0 (fold_oos_tstat / fold_oos_pvalue), and the family of fold
        p-values is corrected for multiple testing (Bonferroni + BH). Returns
        with no active model (before the first fit) are excluded.

        HONEST SCOPE: the sliced series is the BLENDED account return (T+1 fill
        lag, netting, stops) over each window, NOT the isolated P&L of the model
        refit at fit_date_k. Overlapping train windows over-count independent
        trials, so the correction is a heuristic upper bound, NOT exact
        FWER/FDR control. This layer is independent of the deflated-Sharpe
        n_trials lens and must NOT be combined with it.

        'folds' is [] when the desk has no fits.
        """
        fits = list(desk.walk_forward_fits)
        # Distinct, sorted fit-date boundaries. Read the date via to_dict()
        # (contract C3) — the SAME serialization the 'walk_forward' report key
        # uses — so this works for both WalkForwardFit and the TaggedWalkForwardFit
        # wrapper (which has no .fit_date attribute). to_dict()['fit_date'] is a
        # 'YYYY-MM-DD' string; parse to a plain date to compare against the
        # snapshot-derived fold dates.
        boundaries = sorted({pd.Timestamp(fit.to_dict()['fit_date']).date()
                             for fit in fits})
        dated = self.portfolio.get_daily_returns_with_dates()

        folds: List[Dict] = []
        pvalues: List[Optional[float]] = []
        for k, start in enumerate(boundaries):
            end = boundaries[k + 1] if k + 1 < len(boundaries) else None
            fold_returns = [r for (day, r) in dated
                            if day >= start and (end is None or day < end)]
            tstat = fold_oos_tstat(fold_returns)
            pvalue = fold_oos_pvalue(fold_returns)
            pvalues.append(pvalue)
            folds.append({
                'fit_date': start.strftime('%Y-%m-%d'),
                'oos_start': start.strftime('%Y-%m-%d'),
                'oos_end': end.strftime('%Y-%m-%d') if end is not None else None,
                'n_returns': len(fold_returns),
                'mean_return': (float(np.mean(fold_returns))
                                if fold_returns else None),
                'tstat': tstat,
                'pvalue': pvalue,
            })

        bh = benjamini_hochberg(pvalues, alpha)
        m = bh['m']
        bonf_alpha = bonferroni_alpha(alpha, m)
        n_significant_bonferroni = 0
        for fold, pvalue, rejected_bh in zip(folds, pvalues, bh['rejected_bh']):
            fold['significant_bh'] = rejected_bh
            sig_bonf = (pvalue is not None and bonf_alpha is not None
                        and pvalue <= bonf_alpha)
            fold['significant_bonferroni'] = sig_bonf
            if sig_bonf:
                n_significant_bonferroni += 1

        return {
            'available': True,
            'alpha': alpha,
            'test': 'one-sided (mean OOS return > 0)',
            'n_folds': len(folds),
            'n_testable_folds': m,
            'bonferroni_alpha': bonf_alpha,
            'bh_threshold': bh['bh_threshold'],
            'n_significant_bonferroni': n_significant_bonferroni,
            'n_significant_bh': bh['n_significant_bh'],
            'folds': folds,
            'caveat': ('Account-level OOS folds sliced at distinct walk-forward '
                       'fit dates: blended account returns (T+1 fill lag, '
                       'netting, stops), not isolated per-model P&L. The first '
                       'return in each fold is realized under the PRIOR model '
                       'state (T+1 fill lag), so a boundary return reflects the '
                       'prior model, not the refit at that boundary. Overlapping '
                       'refit windows over-count independent trials, so '
                       'Bonferroni/BH is a heuristic upper bound on '
                       'multiple-testing severity, NOT exact FWER/FDR control. '
                       'Independent of the deflated-Sharpe n_trials lens; do not '
                       'combine.'),
        }

    def _generate_report(self, benchmark_symbol: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict:
        """Generate backtest report.

        Strategy-mode reports are unchanged. Desk-mode reports carry the
        same keys plus 'desk', 'trader_notes' and 'walk_forward'
        (contract C3); 'strategy' holds the desk name in that mode.
        """
        # Deflate the Sharpe for multiple testing by the number of walk-forward
        # refits (desk/fund mode). This is a conservative PROXY for the true
        # multiple-testing breadth — overlapping train windows mean it over-
        # counts independent trials, which biases the deflated Sharpe downward
        # (the safe direction). Strategy mode has no refits -> n_trials=1 (no
        # deflation, so deflated_sharpe == psr).
        n_trials = 1
        if self._desk_mode:
            n_trials = max(1, len(self._driver.walk_forward_fits))
        summary = self.portfolio.get_summary(n_trials=n_trials)

        benchmark = None
        if benchmark_symbol:
            benchmark = self._build_benchmark(benchmark_symbol,
                                              start_date, end_date)

        report = {
            'strategy': (self.strategy.name if self.strategy is not None
                         else self._driver.name),
            'summary': summary,
            'benchmark': benchmark,
            'drawdown_series': self.portfolio.get_drawdown_series(),
            'trades': self.trades_log,
            'closed_trades': [
                {
                    'symbol': trade.asset.symbol,
                    # Contract C13 (additive): full instrument string —
                    # stocks are the bare symbol, options the contract.
                    'instrument': str(trade.asset),
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'entry_time': trade.entry_time.strftime('%Y-%m-%d'),
                    'exit_time': trade.exit_time.strftime('%Y-%m-%d'),
                }
                for trade in self.portfolio.closed_trades
            ],
            'portfolio_history': self.portfolio.portfolio_history,
            # Intents still queued when the simulation ended (e.g. signals
            # generated on the final day, which correctly never fill).
            'pending_signals': [
                {
                    'symbol': asset.symbol,
                    'signal': intent['signal'],
                    'signal_date': intent['signal_date'],
                }
                for asset, intent in self.pending_intents.items()
            ],
        }

        if self._desk_mode:
            driver = self._driver
            report['desk'] = {'key': driver.key, 'name': driver.name}
            report['trader_notes'] = [note.to_dict()
                                      for note in driver.notes]
            report['walk_forward'] = [fit.to_dict()
                                      for fit in driver.walk_forward_fits]
            # Per-fold OOS significance + multiple-testing (Step 4). Coherent
            # ONLY for a single desk: one controller -> one fold timeline. In a
            # fund the desks refit on different cadences over a NETTED book, so
            # folds are not attributable to any single model -> marked N/A.
            if self.orchestrator is not None:
                report['oos_folds'] = {
                    'available': False,
                    'reason': ('netted multi-desk fund book — per-fold OOS '
                               'significance is not attributable to a single '
                               'model'),
                }
            else:
                # Defense-in-depth: this additive research-integrity feature
                # runs BEFORE the golden-protected greeks_series/structures keys
                # below, so it must never be able to abort the rest of the
                # report. Degrade to the same N/A marker on any failure (the GUI
                # already handles available=False), mirroring _build_benchmark.
                try:
                    report['oos_folds'] = self._compute_oos_folds(self.desk,
                                                                  alpha=0.05)
                except Exception as exc:  # noqa: BLE001 - defensive boundary
                    logger.warning(
                        "OOS fold computation failed; reporting N/A: %s", exc)
                    report['oos_folds'] = {
                        'available': False,
                        'reason': f'computation failed: {exc}',
                    }
            # Contract C5: desks with a regime model expose regime_series;
            # included only when non-empty (FoundationDesk stays unchanged).
            regime_series = getattr(driver, 'regime_series', None)
            if regime_series:
                report['regime_series'] = list(regime_series)
            # Contract C8: desks with pod accounting expose pod_history;
            # included only when non-empty (Foundation/Renaissance reports
            # stay byte-identical).
            pod_history = getattr(driver, 'pod_history', None)
            if pod_history:
                report['pod_history'] = list(pod_history)
            # Contract C11: desks with multi-leg options structures
            # expose structures_report (janestreet only; other desks
            # lack the attribute, so their reports are byte-identical).
            structures = getattr(driver, 'structures_report', None)
            if structures is not None:
                report['structures'] = list(structures)
            # Contract C12: daily desk-level portfolio Greeks series.
            greeks_series = getattr(driver, 'greeks_series', None)
            if greeks_series is not None:
                report['greeks_series'] = list(greeks_series)
            # Fund mode (additive): per-desk roster + capital + conflicts.
            # Single-desk reports carry NO 'orchestrator' key (byte-identical).
            if self.orchestrator is not None:
                report['orchestrator'] = {
                    'desks': [{'key': d.key, 'name': d.name,
                               'capital_allocation': d.capital_allocation,
                               'notes_count': len(d.notes)}
                              for d in self.orchestrator.desks],
                    'active_capital': self.orchestrator.active_capital,
                    'conflicts_resolved': self.orchestrator.conflicts_resolved,
                }
                # Dynamic reweighting audit (additive; only when a reweighter
                # drove this fund). Non-reweighted fund reports are unchanged.
                if self.reweighter is not None:
                    report['reweight_log'] = [
                        {'date': entry['date'].strftime('%Y-%m-%d'),
                         'day_number': entry['day_number'],
                         'weights': entry['weights'],
                         'fallback': entry['fallback'],
                         'degraded_desks': entry['degraded_desks'],
                         'degrade_reason': entry['degrade_reason']}
                        for entry in self.reweighter.rebalance_log]

        return report
