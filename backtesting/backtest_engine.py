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

SURVIVORSHIP-AWARE STOCK DELISTINGS:
    When the injected market-data feed exposes ``delisting_date(symbol)``
    (WarehouseMarketData does), a held stock is force-liquidated at its final
    observable close on that last listed session. Ordinary provider feeds do
    not expose the hook and retain their prior behavior.

ABSOLUTE-QUANTITY SAFETY:
    ``DeskIntent.quantity`` may choose exact shares/contracts, but it cannot
    enlarge the opening beyond the approved ``size_fraction`` risk budget.
    The risk layer values exact stock quantities from the latest causal close,
    and the fill layer enforces the budget again at the traded price.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from datetime import timezone

import pandas as pd
import numpy as np
import threading
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from core.models import Asset, AssetType, Position
from desks.base import Desk, DeskIntent
from desks.features import enrich_extended
from portfolio.manager import PortfolioManager
from portfolio.mechanics import (
    AccountMode,
    AuthorizationDecision,
    ExposureKind,
    ExposureRequest,
    PortfolioMechanics,
    ShortStockExposure,
)
from portfolio.option_lifecycle import (
    OptionLifecycleEvent,
    OptionLifecyclePolicy,
)
from portfolio.structures import StructureIntent
from portfolio.targets import (PortfolioSnapshot, build_order_deltas,
                               filled_quantities_from_portfolio)
from data.market_data import MarketDataHandler
from strategies.base import Strategy
from backtesting.reporting import (
    BacktestReportState,
    build_benchmark,
    compute_oos_folds,
    generate_report,
)
from backtesting.liquidity import (
    capped_fill_quantity,
    has_executable_share_capacity,
    requeue_remainder,
    require_credible_delisting_terms,
    trailing_average_daily_volume,
)

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
                 market_data: Optional[MarketDataHandler] = None,
                 cash_yield=None,
                 portfolio_mechanics: Optional[PortfolioMechanics] = None,
                 option_lifecycle_policy: Optional[
                     OptionLifecyclePolicy] = None,
                 dividend_fn=None,
                 reject_fills_without_adv: bool = False):
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
        # to slippage on STOCK fills, and an opening order is capped at
        # participation_cap*ADV per day with the remainder re-queued to fill
        # over subsequent days (accumulating into the position). ADV is the
        # trailing adv_window-day mean of volume STRICTLY BEFORE the fill date
        # (no lookahead). When reject_fills_without_adv is true, exits are also
        # capped, and an intent waits (then expires under MAX_PENDING_DAYS) when
        # valid whole-share capacity cannot be computed. Options are unaffected
        # (synthetic, no contract volume).
        self.enable_realistic_fills = enable_realistic_fills
        self.impact_coef = impact_coef
        self.participation_cap = participation_cap
        self.adv_window = adv_window
        self.reject_fills_without_adv = reject_fills_without_adv
        # Idle-cash yield (OPT-IN, default None = byte-identical: cash earns
        # zero, exactly as before the param existed). Pass a date->annual-rate
        # callable OR a pd.Series (datetime index -> annualized decimal rate,
        # e.g. data.riskfree.load_dtb3()); a Series is wrapped in the
        # forward-fill rate_asof lookup. When set, POSITIVE cash accrues
        # cash * rate(date)/252 each simulated day just before the snapshot.
        # HONESTY: only a DATED series makes sense here — a flat retro rate
        # ("cash earned 2% through ZIRP") would be dishonest; the gate's
        # benchmark stays flat-2% rf (the fund EARNS real dated yield, is
        # JUDGED vs 2%), and comparators must be rerun under the same setting.
        if cash_yield is not None and not callable(cash_yield):
            from data.riskfree import rate_asof
            series = cash_yield
            cash_yield = lambda date: rate_asof(series, date)  # noqa: E731
        self.cash_yield = cash_yield
        if (portfolio_mechanics is not None
                and not isinstance(portfolio_mechanics, PortfolioMechanics)):
            raise ValueError(
                "portfolio_mechanics must be a PortfolioMechanics or None")
        if (option_lifecycle_policy is not None
                and not isinstance(option_lifecycle_policy,
                                   OptionLifecyclePolicy)):
            raise ValueError(
                "option_lifecycle_policy must be an OptionLifecyclePolicy "
                "or None")
        if dividend_fn is not None and not callable(dividend_fn):
            raise ValueError("dividend_fn must be callable or None")
        self.portfolio_mechanics = portfolio_mechanics
        self.option_lifecycle_policy = option_lifecycle_policy
        self.dividend_fn = dividend_fn
        self.mechanics_log: List[Dict] = []
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
        # Optional structure-native path. One entry is one indivisible package;
        # legacy per-asset pending intents remain in the historical mapping
        # above and never pass through this surface.
        self.pending_structures: Dict[str, Dict] = {}
        # Monotonic generation for opt-in target-native desk snapshots.  The
        # legacy intent path never reads or mutates it.
        self._target_snapshot_version = 0

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

        # Optional survivorship-aware feed contract.  WarehouseMarketData
        # exposes the final listed session for delisted securities; ordinary
        # live/provider feeds do not, so their behavior is unchanged.  Resolve
        # once per run rather than querying the warehouse inside the daily loop.
        delisting_date = getattr(self.market_data, 'delisting_date', None)
        self._delisting_dates = {}
        if callable(delisting_date):
            for symbol in all_data:
                final_session = delisting_date(symbol)
                if final_session is not None:
                    self._delisting_dates[symbol] = pd.Timestamp(final_session)

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
        self.pending_structures = {}
        self._target_snapshot_version = 0

        # Simulate sequential historical trading day by day
        total_days = len(sorted_dates)
        for day_number, date in enumerate(sorted_dates, start=1):
            # --- PHASE 0 (desk mode): SETTLE OPTIONS PAST EXPIRY ---
            if self._desk_mode:
                self._settle_expired_options(all_data, date)

            # --- PHASE 1: FILL PENDING INTENTS AT TODAY'S OPEN ---
            self._fill_pending_structures(all_data, date)
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
                if self.option_lifecycle_policy is not None:
                    self._process_early_option_lifecycle(all_data, date)

            # A warehouse-held delisted stock must not remain at its final mark
            # forever.  Liquidate it at the final observable close (including
            # normal exit commission) on its last listed session.
            self._liquidate_delisted_stocks(all_data, date)

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

            # --- CASH YIELD (opt-in): idle cash accrues the dated T-bill
            # rate for today, BEFORE the snapshot so the equity curve (and
            # every gate metric downstream of it) includes the interest. ---
            if self.cash_yield is not None:
                self._accrue_cash_yield(date)
            if self.portfolio_mechanics is not None:
                self._accrue_financing(date)

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

    def _liquidate_delisted_stocks(self, all_data: Dict[str, pd.DataFrame],
                                   date) -> None:
        """Close warehouse positions on their final listed session.

        This hook is inert for ordinary market-data providers because only the
        point-in-time warehouse feed exposes ``delisting_date``.  The final
        usable close is an observable liquidation value; if even that is
        unavailable the position is conservatively written down to zero.
        """
        if not getattr(self, '_delisting_dates', None):
            return
        current = pd.Timestamp(date).normalize()
        for asset in sorted(list(self.portfolio.positions), key=str):
            if asset.asset_type is not AssetType.STOCK:
                continue
            final_session = self._delisting_dates.get(asset.symbol)
            if final_session is None or current < final_session.normalize():
                continue

            position = self.portfolio.positions[asset]
            frame = all_data.get(asset.symbol)
            exit_price = 0.0
            if frame is not None and not frame.empty:
                closes = frame.loc[frame.index <= date, 'close'].dropna()
                finite = closes[np.isfinite(closes.to_numpy(dtype=float))]
                if not finite.empty:
                    exit_price = float(finite.iloc[-1])
            if exit_price <= 0:
                logger.warning(
                    "Delisting liquidation of %s on %s has no usable final "
                    "close; writing the position down at 0.0",
                    asset.symbol, date)

            payout_source = 'final_tradable_close'
            quality_flags = ['delisting_terms_unavailable']
            payout_reader = getattr(self.market_data, 'delisting_payout', None)
            if callable(payout_reader):
                payout = payout_reader(asset.symbol, exit_price)
                if not isinstance(payout, dict):
                    raise ValueError(
                        "delisting_payout must return a mapping")
                modeled_price = float(payout.get('price'))
                if not math.isfinite(modeled_price) or modeled_price < 0:
                    raise ValueError(
                        f"invalid delisting payout for {asset.symbol}")
                exit_price = modeled_price
                payout_source = str(payout.get('source') or 'unknown')
                quality_flags = list(payout.get('quality_flags') or [])
            require_credible_delisting_terms(
                self.reject_fills_without_adv, payout_source, quality_flags, asset.symbol)

            quantity = position.quantity
            commission = abs(quantity) * exit_price * self.commission
            if quantity > 0:
                cash_flow = quantity * exit_price - commission
                self.portfolio.cash += cash_flow
                action = 'SELL'
                flow_key = 'proceeds'
            else:
                cash_flow = abs(quantity) * exit_price + commission
                self.portfolio.cash -= cash_flow
                action = 'COVER'
                flow_key = 'cost'

            self.portfolio.close_position(
                asset, exit_price, quantity, position.timestamp, date)
            self.portfolio.remove_position(asset)
            self.pending_intents.pop(asset, None)
            self.trades_log.append({
                'date': date,
                'signal_date': date,
                'symbol': asset.symbol,
                'instrument': str(asset),
                'action': action,
                'quantity': abs(quantity),
                'price': exit_price,
                'commission': commission,
                flow_key: cash_flow,
                'reason': ('delisting settlement' if self.reject_fills_without_adv
                           else 'delisting liquidation'),
                'payout_source': payout_source,
                'data_quality_flags': quality_flags,
            })
            logger.info(
                "Delisting liquidation: %s %d %s @ %.4f on %s",
                action, abs(quantity), asset.symbol, exit_price, date)

    def _accrue_cash_yield(self, date) -> None:
        """One day of idle-cash interest: cash += cash * rate(date) / 252.

        POSITIVE cash only. NEGATIVE cash (a margin debit — shouldn't occur
        in the long-only books, but guard anyway) accrues NOTHING: earning
        yield ON a debit would be free leverage. This seam models a cash
        sweep, not a margin facility — a real facility would CHARGE interest
        on the debit, so skipping the charge is GENEROUS to a debit-carrying
        book; acceptable only because the books here are long-only and a
        debit is a bug to surface, not an economics to model."""
        if self.portfolio.cash > 0:
            self.portfolio.cash += (self.portfolio.cash
                                    * self.cash_yield(date) / 252.0)

    def _mechanics_buying_power(self) -> float:
        """Available initial buying power under the opt-in account policy."""
        mechanics = self.portfolio_mechanics
        if mechanics is None:
            return max(0.0, self.portfolio.cash)
        if mechanics.margin_policy.mode is AccountMode.CASH:
            return max(0.0, self.portfolio.cash)
        requirement = 0.0
        for position in self.portfolio.positions.values():
            quantity = abs(int(position.quantity))
            if quantity == 0:
                continue
            if position.asset.asset_type is AssetType.STOCK:
                kind = (ExposureKind.LONG_STOCK if position.quantity > 0
                        else ExposureKind.SHORT_STOCK)
                multiplier = 1
            elif position.quantity > 0:
                kind = ExposureKind.LONG_OPTION
                multiplier = position.asset.multiplier
            else:
                # Naked short options are prohibited on the mechanics path;
                # conservatively hold their full marked notional if a caller
                # injects an existing one.
                requirement += (
                    quantity * max(0.0, position.current_price)
                    * position.asset.multiplier)
                continue
            if position.current_price <= 0:
                continue
            request = ExposureRequest(
                request_id=f"held:{position.asset}",
                symbol=position.asset.symbol,
                kind=kind,
                quantity=quantity,
                unit_price=float(position.current_price),
                contract_multiplier=multiplier,
            )
            requirement += mechanics.requirements(request).initial_requirement
        return max(
            0.0, self.portfolio.get_portfolio_value() - requirement)

    def _record_authorization(self, authorization, date) -> None:
        borrow = authorization.borrow
        self.mechanics_log.append({
            'date': pd.Timestamp(date),
            'event': 'pretrade_authorization',
            'request_id': authorization.request.request_id,
            'symbol': authorization.request.symbol,
            'kind': authorization.request.kind.value,
            'quantity': authorization.request.quantity,
            'exposure_value': authorization.requirement.exposure_value,
            'initial_requirement': (
                authorization.requirement.initial_requirement),
            'maintenance_requirement': (
                authorization.requirement.maintenance_requirement),
            'available_buying_power': authorization.available_buying_power,
            'approved': authorization.approved,
            'decision': authorization.decision.value,
            'reason': authorization.reason,
            'borrow_rate': (borrow.quote.annual_fee_rate
                            if borrow is not None and borrow.quote is not None
                            else None),
        })

    def _authorize_exposure(
            self, *, request_id: str, symbol: str, kind: ExposureKind,
            quantity: int, date, unit_price: Optional[float] = None,
            contract_multiplier: Optional[int] = None,
            max_loss_per_package: Optional[float] = None) -> bool:
        mechanics = self.portfolio_mechanics
        if mechanics is None:
            return True
        request = ExposureRequest(
            request_id=request_id,
            symbol=symbol,
            kind=kind,
            quantity=int(quantity),
            unit_price=unit_price,
            contract_multiplier=contract_multiplier,
            max_loss_per_package=max_loss_per_package,
        )
        already_borrowed = sum(
            abs(int(position.quantity))
            for asset, position in self.portfolio.positions.items()
            if (asset.asset_type is AssetType.STOCK
                and asset.symbol == symbol and position.quantity < 0)
        )
        authorization = mechanics.authorize(
            request,
            self._mechanics_buying_power(),
            as_of=pd.Timestamp(date).date(),
            already_borrowed_quantity=already_borrowed,
        )
        self._record_authorization(authorization, date)
        return authorization.approved

    def _block_naked_option_short(self, asset: Asset, quantity: int,
                                  date) -> bool:
        if self.portfolio_mechanics is None:
            return False
        self.mechanics_log.append({
            'date': pd.Timestamp(date),
            'event': 'pretrade_authorization',
            'request_id': f"{pd.Timestamp(date).date()}:{asset}:SHORT",
            'symbol': asset.symbol,
            'kind': 'SHORT_OPTION',
            'quantity': int(quantity),
            'approved': False,
            'decision': AuthorizationDecision.ACCOUNT_MODE_PROHIBITED.value,
            'reason': ('naked short options require an atomic defined-risk '
                       'package under portfolio mechanics'),
        })
        return True

    def _accrue_financing(self, date) -> None:
        mechanics = self.portfolio_mechanics
        if mechanics is None:
            return
        shorts = [
            ShortStockExposure(
                asset.symbol, abs(int(position.quantity)),
                float(position.current_price))
            for asset, position in self.portfolio.positions.items()
            if (asset.asset_type is AssetType.STOCK
                and position.quantity < 0 and position.current_price > 0)
        ]
        accrual = mechanics.accrue(
            pd.Timestamp(date).date(), self.portfolio.cash, shorts)
        self.portfolio.cash += accrual.cash_delta
        self.mechanics_log.append({
            'date': pd.Timestamp(date),
            'event': 'financing_accrual',
            'debit_principal': accrual.debit_principal,
            'debit_interest': accrual.debit_interest,
            'borrow_fees': sum(item.fee for item in accrual.borrow_fees),
            'total_charge': accrual.total_charge,
            'cash_delta': accrual.cash_delta,
            'compliance_flags': list(accrual.compliance_flags),
        })

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
        desk = self.desk
        if desk is None:
            raise RuntimeError("Desk intent queue requires desk mode")
        desk.set_clock(date)

        enriched: Dict[str, pd.DataFrame] = {}
        for symbol in all_data:
            # Precomputed indicators sliced through today only
            historical_data = self._enriched_through(symbol, date)
            if historical_data.empty:
                continue
            enriched[symbol] = historical_data

        if not enriched:
            return

        if getattr(desk, 'target_native_enabled', False):
            self._queue_desk_structures(enriched, date)
            self._queue_desk_targets(enriched, date)
            return

        self._queue_desk_structures(enriched, date)

        intents = desk.generate_intents(enriched, date, self.portfolio)
        approved = desk.apply_risk(intents, self.portfolio, enriched, date)

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

    def _queue_desk_structures(
            self, enriched: Dict[str, pd.DataFrame], date) -> None:
        """Queue each optional canonical structure as one pending object.

        Desks without ``generate_structure_intents`` do no additional work,
        preserving the legacy per-leg path exactly. Re-emitting the same
        stable intent while it waits for prices preserves its waiting state.
        """
        desk = self.desk
        if desk is None:
            raise RuntimeError("Structure intent queue requires desk mode")
        generate = getattr(desk, 'generate_structure_intents', None)
        if not callable(generate):
            return
        structures = generate(enriched, date, self.portfolio)
        if structures is None:
            return
        for structure in structures:
            if not isinstance(structure, StructureIntent):
                logger.warning(
                    "Dropping invalid atomic structure from %s on %s: %r",
                    desk.key, date, structure)
                continue
            intent_id = str(structure.intent_id)
            existing = self.pending_structures.get(intent_id)
            if existing is not None:
                if existing['structure'] != structure:
                    raise ValueError(
                        f"structure intent id collision for {intent_id}")
                continue
            self.pending_structures[intent_id] = {
                'structure': structure,
                'signal_date': date,
                'days_waiting': 0,
            }

    def _pending_target_deltas(self) -> Dict[Asset, int]:
        """Return signed, still-working units from simulated pending orders.

        Target-native orders always carry an exact ``quantity``.  A legacy
        close can still be represented if a caller enables the target path
        mid-run; value-sized legacy opens fail closed because their remaining
        native quantity is unknowable before a fill price exists.
        """
        reserved: Dict[Asset, int] = {}
        signs = {'BUY': 1, 'COVER': 1, 'SELL': -1, 'SHORT': -1}
        for asset, intent in self.pending_intents.items():
            signal = intent.get('signal')
            if signal not in signs:
                raise ValueError(f"Unsupported pending signal {signal!r}")
            quantity = intent.get('quantity')
            if quantity is None:
                position = self.portfolio.get_position(asset)
                if signal == 'SELL' and position and position.quantity > 0:
                    quantity = position.quantity
                elif signal == 'COVER' and position and position.quantity < 0:
                    quantity = abs(position.quantity)
                else:
                    raise ValueError(
                        f"Pending {signal} for {asset} has no exact quantity")
            signed = signs[signal] * int(quantity)
            reserved[asset] = reserved.get(asset, 0) + signed
        return reserved

    def _target_snapshot(self, date) -> PortfolioSnapshot:
        """Build the target contract's filled-plus-working book snapshot."""
        self._target_snapshot_version += 1
        timestamp = pd.Timestamp(date)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone.utc)
        else:
            timestamp = timestamp.tz_convert(timezone.utc)
        return PortfolioSnapshot(
            filled_quantities=filled_quantities_from_portfolio(self.portfolio),
            reserved_deltas=self._pending_target_deltas(),
            version=self._target_snapshot_version,
            as_of=timestamp.to_pydatetime(),
        )

    @staticmethod
    def _intent_action_for_delta(delta) -> str:
        """Map a signed target delta onto the legacy four-action vocabulary."""
        if delta.signed_quantity > 0:
            return 'COVER' if delta.effective_quantity < 0 else 'BUY'
        return 'SELL' if delta.effective_quantity > 0 else 'SHORT'

    def _queue_desk_targets(self, enriched: Dict[str, pd.DataFrame], date) -> None:
        """Opt-in target-native desk path with pending-order idempotency.

        The first diff validates/coalesces the target set.  When a changed
        target supersedes an existing simulated order, that order is canceled
        before a fresh snapshot and final diff.  This is essential for a zero
        target: diffing zero against an unfilled BUY reservation must cancel
        the BUY, not manufacture a SELL against shares never owned.
        """
        desk = self.desk
        if desk is None:
            raise RuntimeError("Target-native queueing requires desk mode")
        generate_targets = getattr(desk, 'generate_targets', None)
        if not callable(generate_targets):
            raise RuntimeError(
                "target_native_enabled requires a callable generate_targets")

        snapshot = self._target_snapshot(date)
        targets = tuple(generate_targets(
            enriched, date, self.portfolio, snapshot))

        # Validate the complete target set before mutating pending state.
        build_order_deltas(targets, snapshot)
        by_asset = {target.asset: target for target in targets}
        canceled = False
        for asset in list(self.pending_intents):
            target = by_asset.get(asset)
            if (target is not None
                    and target.target_quantity != snapshot.effective_quantity(asset)):
                logger.info(
                    "Canceling simulated pending %s for %s: target changed to %d",
                    self.pending_intents[asset].get('signal'), str(asset),
                    target.target_quantity)
                del self.pending_intents[asset]
                canceled = True

        if canceled:
            snapshot = self._target_snapshot(date)
        deltas = build_order_deltas(targets, snapshot)

        intents: List[DeskIntent] = []
        intent_ids: Dict[Asset, str] = {}
        for delta in deltas:
            size_fraction = delta.metadata.get('size_fraction')
            if size_fraction is None:
                raise ValueError(
                    f"Target for {delta.asset} must provide metadata.size_fraction")
            intent = DeskIntent(
                asset=delta.asset,
                action=self._intent_action_for_delta(delta),
                size_fraction=float(size_fraction),
                reason=(delta.reason
                        or f"target position {delta.target_quantity}"),
                quantity=delta.quantity,
                intent_id=delta.intent_id,
            )
            intents.append(intent)
            intent_ids[delta.asset] = delta.intent_id

        approved = desk.apply_risk(
            intents, self.portfolio, enriched, date)
        for intent in approved:
            pending = {
                'signal': intent.action,
                'signal_date': date,
                'days_waiting': 0,
                'size_fraction': intent.size_fraction,
                'quantity': intent.quantity,
                'target_native': True,
            }
            intent_id = getattr(intent, 'intent_id', None)
            if intent_id is None:
                intent_id = intent_ids.get(intent.asset)
            if intent_id is not None:
                pending['intent_id'] = intent_id
            self._queue_pending_intent(intent.asset, pending)

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
    def _impact_fraction(self, quantity: int,
                         adv: Optional[float]) -> float:
        """Square-root market impact (Almgren) as an adverse price fraction:
        ``impact_coef * sqrt(quantity / ADV)``. 0.0 when ADV is unknown or
        non-positive, or quantity <= 0."""
        if not adv or adv <= 0 or quantity <= 0:
            return 0.0
        return min(_MAX_IMPACT_FRACTION,
                   self.impact_coef * math.sqrt(quantity / adv))

    def _capped_fill_quantity(self, desired: int, adv: Optional[float]):
        return capped_fill_quantity(
            desired, adv, self.participation_cap,
            strict=self.reject_fills_without_adv)

    def _queue_pending_intent(self, asset, intent: Dict) -> None:
        """Queue a new pending intent for ``asset``, replacing any older one.

        Surfaces the realistic-fills edge where a fresh signal supersedes an
        in-flight cap-and-requeue remainder: the unfilled shares are abandoned,
        so warn instead of dropping them silently. Off-path no intent ever
        carries 'accumulate', so this is a plain assignment (byte-identical).
        """
        prior = self.pending_intents.get(asset)
        if (prior is not None
                and (prior.get('liquidity_remainder')
                     or prior.get('accumulate'))):
            logger.warning(
                "Abandoning %s unfilled shares of %s: a new %s signal "
                "supersedes the in-flight partial fill",
                prior.get('quantity'), asset.symbol, intent.get('signal'))
        self.pending_intents[asset] = intent

    def _structure_model_prices(
            self, structure: StructureIntent,
            all_data: Dict[str, pd.DataFrame], date
            ) -> Optional[Dict[Asset, float]]:
        """Return all causal leg marks, or ``None`` until all are priceable."""
        symbol = structure.legs[0].asset.symbol
        data = all_data.get(symbol)
        if data is None or date not in data.index:
            return None
        row = data.loc[date]
        spot = row.get('open', float('nan'))
        if pd.isna(spot):
            spot = row.get('close', float('nan'))
        if pd.isna(spot) or not math.isfinite(float(spot)) or float(spot) <= 0:
            return None
        history = data[data.index <= date]
        prices: Dict[Asset, float] = {}
        for leg in structure.legs:
            try:
                model_price = self._price_option(
                    leg.asset, history, date, spot=float(spot))
            except Exception as exc:  # noqa: BLE001 - whole package waits
                logger.warning(
                    "Atomic structure %s leg %s is not priceable on %s: %s",
                    structure.intent_id, str(leg.asset), date, exc)
                return None
            if model_price is None or not math.isfinite(float(model_price)):
                return None
            price = float(model_price)
            if price < 0:
                raise ValueError(
                    f"negative model price for structure leg {leg.asset}")
            prices[leg.asset] = price
        return prices

    def _structure_fill_plan(
            self, structure: StructureIntent,
            model_prices: Dict[Asset, float], fill_date=None
            ) -> tuple[List[Dict], float]:
        """Preflight every leg and return mutations plus aggregate cash delta."""
        assets = [leg.asset for leg in structure.legs]
        if len(set(assets)) != len(assets):
            raise ValueError("atomic structures require unique leg assets")

        plans: List[Dict] = []
        cash_delta = 0.0
        total_commission = 0.0
        for leg in structure.legs:
            contracts = structure.quantity * leg.ratio
            buying = leg.action.sign > 0
            fill_price = self._option_traded_price(
                model_prices[leg.asset], buying=buying)
            if not math.isfinite(fill_price) or fill_price <= 0:
                raise ValueError(
                    f"non-positive executable price for {leg.asset}")
            commission = contracts * self.per_contract_commission
            notional = contracts * fill_price * leg.asset.multiplier
            leg_cash_delta = (-notional if buying else notional) - commission
            existing = self.portfolio.get_position(leg.asset)

            if structure.opening:
                signed_quantity = leg.action.sign * contracts
                if (existing is not None and existing.quantity != 0
                        and ((existing.quantity > 0)
                             != (signed_quantity > 0))):
                    raise ValueError(
                        f"opening package conflicts with {leg.asset} position")
                close_quantity = None
            else:
                expected_position_sign = -leg.action.sign
                if (existing is None or existing.quantity == 0
                        or (1 if existing.quantity > 0 else -1)
                        != expected_position_sign
                        or abs(existing.quantity) < contracts):
                    raise ValueError(
                        f"closing package cannot close {contracts} contracts "
                        f"of {leg.asset}")
                signed_quantity = None
                # PortfolioManager.close_position expects the signed quantity
                # currently held: negative for covering a short, positive for
                # selling a long.
                close_quantity = expected_position_sign * contracts

            plans.append({
                'leg': leg,
                'contracts': contracts,
                'fill_price': fill_price,
                'model_price': model_prices[leg.asset],
                'commission': commission,
                'cash_delta': leg_cash_delta,
                'signed_quantity': signed_quantity,
                'close_quantity': close_quantity,
            })
            cash_delta += leg_cash_delta
            total_commission += commission

        if not math.isfinite(cash_delta):
            raise ValueError("atomic structure cash flow is not finite")
        if structure.opening:
            required_cash = max(
                float(structure.max_loss) + total_commission,
                max(0.0, -cash_delta))
            if self.portfolio_mechanics is not None:
                if fill_date is None or not self._authorize_exposure(
                        request_id=str(structure.intent_id),
                        symbol=structure.legs[0].asset.symbol,
                        kind=ExposureKind.DEFINED_RISK_PACKAGE,
                        quantity=structure.quantity,
                        date=fill_date,
                        max_loss_per_package=(
                            structure.max_loss / structure.quantity)):
                    raise ValueError(
                        "portfolio mechanics rejected atomic package")
            elif self.portfolio.cash + 1e-9 < required_cash:
                raise ValueError(
                    f"insufficient cash for atomic package: required "
                    f"{required_cash:.2f}, available {self.portfolio.cash:.2f}")
        return plans, cash_delta

    def _commit_structure_fill(
            self, pending: Dict, plans: List[Dict], cash_delta: float,
            fill_date) -> None:
        """Commit positions, cash and trade records as one in-memory unit."""
        structure: StructureIntent = pending['structure']
        positions_before = deepcopy(self.portfolio.positions)
        closed_before = deepcopy(self.portfolio.closed_trades)
        realized_before = self.portfolio._realized_pnl
        cash_before = self.portfolio.cash
        trades_length = len(self.trades_log)
        try:
            self.portfolio.cash += cash_delta
            for plan in plans:
                leg = plan['leg']
                existing = self.portfolio.get_position(leg.asset)
                if structure.opening:
                    signed_quantity = plan['signed_quantity']
                    if existing is not None and existing.quantity != 0:
                        self._accumulate_position(
                            existing, signed_quantity, plan['fill_price'])
                    else:
                        owners = ((structure.owner,)
                                  if structure.owner is not None else None)
                        self.portfolio.add_position(Position(
                            asset=leg.asset, quantity=signed_quantity,
                            avg_entry_price=plan['fill_price'],
                            current_price=plan['fill_price'],
                            timestamp=fill_date, owners=owners))
                else:
                    assert existing is not None
                    self.portfolio.close_position(
                        leg.asset, plan['fill_price'], plan['close_quantity'],
                        existing.timestamp, fill_date)

                buying = leg.action.sign > 0
                cash_amount = abs(
                    plan['contracts'] * plan['fill_price']
                    * leg.asset.multiplier
                    + (plan['commission'] if buying
                       else -plan['commission']))
                trade = {
                    'date': fill_date,
                    'signal_date': pending['signal_date'],
                    'symbol': leg.asset.symbol,
                    'instrument': str(leg.asset),
                    'action': leg.action.value,
                    'quantity': plan['contracts'],
                    'price': plan['fill_price'],
                    'model_price': plan['model_price'],
                    'commission': plan['commission'],
                    'package_id': structure.intent_id,
                    'package_quantity': structure.quantity,
                    'ratio': leg.ratio,
                    'reason': structure.reason,
                }
                trade['cost' if buying else 'proceeds'] = cash_amount
                self.trades_log.append(trade)
        except Exception:
            self.portfolio.positions = positions_before
            self.portfolio.closed_trades = closed_before
            self.portfolio._realized_pnl = realized_before
            self.portfolio.cash = cash_before
            del self.trades_log[trades_length:]
            raise

    def _fill_pending_structures(
            self, all_data: Dict[str, pd.DataFrame], date) -> None:
        """Fill or wait each package; no leg can survive independently."""
        for intent_id in list(self.pending_structures):
            pending = self.pending_structures[intent_id]
            structure: StructureIntent = pending['structure']
            try:
                model_prices = self._structure_model_prices(
                    structure, all_data, date)
                if model_prices is None:
                    pending['days_waiting'] += 1
                    if pending['days_waiting'] >= MAX_PENDING_DAYS:
                        logger.warning(
                            "Dropping atomic structure %s (signal %s): not all "
                            "legs priceable for %d trading days",
                            intent_id, pending['signal_date'],
                            pending['days_waiting'])
                        del self.pending_structures[intent_id]
                    continue
                plans, cash_delta = self._structure_fill_plan(
                    structure, model_prices, date)
                self._commit_structure_fill(
                    pending, plans, cash_delta, date)
                logger.info(
                    "Filled atomic structure %s: %d packages / %d legs on %s",
                    intent_id, structure.quantity, len(structure.legs), date)
            except ValueError as exc:
                # Invalid package/portfolio/cash state consumes the package as
                # one rejected object. No mutation occurs because planning is
                # complete before commit.
                logger.warning(
                    "Dropping atomic structure %s on %s: %s",
                    intent_id, date, exc)
            del self.pending_structures[intent_id]

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
                adv = (trailing_average_daily_volume(
                    data, date, self.adv_window)
                       if self.enable_realistic_fills else None)
                if (self.enable_realistic_fills
                        and self.reject_fills_without_adv
                        and not has_executable_share_capacity(
                            adv, self.participation_cap)):
                    intent['days_waiting'] += 1
                    if intent['days_waiting'] >= MAX_PENDING_DAYS:
                        logger.warning(
                            "Dropping %s intent for %s (signal %s): no valid "
                            "trailing ADV capacity for %d trading days",
                            intent['signal'], str(asset), intent['signal_date'],
                            intent['days_waiting'])
                        del self.pending_intents[asset]
                    else:
                        logger.warning(
                            "Deferring %s intent for %s (signal %s): no valid "
                            "trailing ADV capacity on %s",
                            intent['signal'], str(asset), intent['signal_date'],
                            date)
                    continue
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

        target_add = bool(intent.get('target_native') and existing_pos and (
            (signal == 'BUY' and existing_pos.quantity > 0)
            or (signal == 'SHORT' and existing_pos.quantity < 0)))
        if signal in ('BUY', 'SHORT') \
                and ((not existing_pos or existing_pos.quantity == 0)
                     or target_add):
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
            if self.orchestrator is not None:
                # Orchestrator intents are already account-absolute.
                size_fraction = intent.get('size_fraction', 0.0)
            else:
                size_fraction = (self.desk.capital_allocation
                                 * intent.get('size_fraction', 0.0))
            trade_value = (self.portfolio.get_portfolio_value()
                           * size_fraction)
            quantity = intent.get('quantity')
            if quantity is None:
                per_contract = max(fill_price, MIN_OPTION_HAIRCUT) * asset.multiplier
                quantity = int(trade_value / per_contract)
            else:
                requested_notional = quantity * fill_price * asset.multiplier
                if requested_notional > trade_value + 1e-9:
                    logger.warning(
                        "Dropping %s intent for %s on %s: absolute quantity "
                        "requests %.2f notional against %.2f risk budget",
                        signal, str(asset), fill_date, requested_notional,
                        trade_value)
                    return
            if quantity <= 0:
                logger.warning(
                    "Dropping %s intent for %s on %s: sizes to 0 contracts",
                    signal, str(asset), fill_date)
                return

            if buying:
                if not self._authorize_exposure(
                        request_id=(f"{pd.Timestamp(fill_date).date()}:"
                                    f"{asset}:BUY"),
                        symbol=asset.symbol,
                        kind=ExposureKind.LONG_OPTION,
                        quantity=int(quantity),
                        date=fill_date,
                        unit_price=fill_price,
                        contract_multiplier=asset.multiplier):
                    logger.warning(
                        "Dropping BUY intent for %s on %s: portfolio "
                        "mechanics rejected exposure", str(asset), fill_date)
                    return
            elif self._block_naked_option_short(asset, int(quantity),
                                                 fill_date):
                logger.warning(
                    "Dropping SHORT intent for %s on %s: naked options "
                    "are prohibited by portfolio mechanics",
                    str(asset), fill_date)
                return

            commission = quantity * self.per_contract_commission
            if buying:
                cost = quantity * fill_price * asset.multiplier + commission
                if (self.portfolio_mechanics is None
                        and self.portfolio.cash < cost):
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

            if target_add:
                self._accumulate_position(
                    existing_pos, signed_quantity, fill_price)
            else:
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
            if intent.get('target_native') and intent.get('quantity') is not None:
                quantity = min(int(intent['quantity']), existing_pos.quantity)
            else:
                quantity = existing_pos.quantity
            commission = quantity * self.per_contract_commission
            proceeds = quantity * fill_price * asset.multiplier - commission
            self.portfolio.cash += proceeds
            self.portfolio.close_position(asset, fill_price, quantity,
                                          existing_pos.timestamp, fill_date)
            if not intent.get('target_native'):
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
            if intent.get('target_native') and intent.get('quantity') is not None:
                quantity = -min(int(intent['quantity']), abs(existing_pos.quantity))
            else:
                quantity = existing_pos.quantity  # negative
            commission = abs(quantity) * self.per_contract_commission
            cost = abs(quantity) * fill_price * asset.multiplier + commission
            self.portfolio.cash -= cost
            self.portfolio.close_position(asset, fill_price, quantity,
                                          existing_pos.timestamp, fill_date)
            if not intent.get('target_native'):
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
            spot = 0.0
            spot_available = False
            data = all_data.get(asset.symbol)
            if data is not None and not data.empty:
                closes = data.loc[data.index <= pd.Timestamp(expiry),
                                  'close'].dropna()
                if not closes.empty:
                    spot = float(closes.iloc[-1])
                    spot_available = True
                    if asset.asset_type is AssetType.CALL:
                        intrinsic = max(0.0, spot - asset.strike_price)
                    else:
                        intrinsic = max(0.0, asset.strike_price - spot)
                else:
                    logger.warning("Expiry settlement of %s: no underlying "
                                   "close <= %s; settling at 0.0",
                                   str(asset), expiry)
            quantity = position.quantity
            if self.option_lifecycle_policy is not None:
                if not spot_available:
                    raise ValueError(
                        f"physical settlement of {asset} requires an "
                        "underlying close on or before expiry")
                event = self.option_lifecycle_policy.plan(
                    asset,
                    int(quantity),
                    spot=spot,
                    effective_date=pd.Timestamp(date).date(),
                )
                self._apply_option_lifecycle_event(
                    event, spot, date)
                continue
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

    def _apply_option_lifecycle_event(
            self, event: OptionLifecycleEvent, spot: float, timestamp) -> None:
        """Atomically remove the option and apply physical stock/strike cash."""
        option_position = self.portfolio.get_position(event.option)
        if option_position is None \
                or option_position.quantity != event.signed_contracts:
            raise ValueError(
                "option lifecycle event no longer matches the open position")
        positions_before = deepcopy(self.portfolio.positions)
        closed_before = deepcopy(self.portfolio.closed_trades)
        realized_before = self.portfolio._realized_pnl
        cash_before = self.portfolio.cash
        trades_length = len(self.trades_log)
        mechanics_length = len(self.mechanics_log)
        try:
            self.portfolio.close_position(
                event.option, event.settlement_price,
                event.signed_contracts, option_position.timestamp, timestamp)
            self.portfolio.remove_position(event.option)
            self.portfolio.cash += event.cash_delta

            if event.stock_delta:
                stock = Asset(event.option.symbol, AssetType.STOCK)
                existing = self.portfolio.get_position(stock)
                strike = float(event.option.strike_price)
                remaining = event.stock_delta
                if existing is not None and existing.quantity != 0:
                    same_direction = ((existing.quantity > 0)
                                      == (remaining > 0))
                    if same_direction:
                        self._accumulate_position(existing, remaining, strike)
                        remaining = 0
                    else:
                        close_quantity = ((1 if existing.quantity > 0 else -1)
                                          * min(abs(existing.quantity),
                                                abs(remaining)))
                        self.portfolio.close_position(
                            stock, strike, close_quantity,
                            existing.timestamp, timestamp)
                        remaining += close_quantity
                if remaining:
                    self.portfolio.add_position(Position(
                        asset=stock,
                        quantity=remaining,
                        avg_entry_price=strike,
                        current_price=float(spot),
                        timestamp=timestamp,
                        owners=option_position.owners,
                    ))

            event_row = {
                'date': pd.Timestamp(timestamp),
                'event': 'option_lifecycle',
                'instrument': str(event.option),
                'contracts': event.signed_contracts,
                'settlement_price': event.settlement_price,
                'stock_delta': event.stock_delta,
                'cash_delta': event.cash_delta,
                'reason': event.reason.value,
            }
            self.mechanics_log.append(event_row)
            self.trades_log.append({
                'date': timestamp,
                'signal_date': timestamp,
                'symbol': event.option.symbol,
                'instrument': str(event.option),
                'action': event.reason.value.upper(),
                'quantity': abs(event.signed_contracts),
                'price': event.settlement_price,
                'stock_delta': event.stock_delta,
                'cash_delta': event.cash_delta,
                'reason': event.reason.value,
            })
        except Exception:
            self.portfolio.positions = positions_before
            self.portfolio.closed_trades = closed_before
            self.portfolio._realized_pnl = realized_before
            self.portfolio.cash = cash_before
            del self.trades_log[trades_length:]
            del self.mechanics_log[mechanics_length:]
            raise

    def _process_early_option_lifecycle(
            self, all_data: Dict[str, pd.DataFrame], date) -> None:
        """Apply opt-in American early exercise/assignment at today's close."""
        policy = self.option_lifecycle_policy
        if policy is None or not policy.enable_early_exercise:
            return
        current = pd.Timestamp(date).date()
        for asset in sorted(list(self.portfolio.positions), key=str):
            if asset.asset_type not in {AssetType.CALL, AssetType.PUT}:
                continue
            expiry = pd.Timestamp(asset.expiration_date).date()
            days_to_expiry = (expiry - current).days
            if days_to_expiry <= 0:
                continue
            data = all_data.get(asset.symbol)
            if data is None or date not in data.index:
                continue
            spot = float(data.loc[date, 'close'])
            position = self.portfolio.get_position(asset)
            if (position is None or not math.isfinite(spot) or spot < 0
                    or not math.isfinite(float(position.current_price))):
                continue
            annual_rate = 0.0
            if self.cash_yield is not None:
                annual_rate = float(self.cash_yield(date))
            dividend = (float(self.dividend_fn(asset.symbol, date))
                        if self.dividend_fn is not None else 0.0)
            decision = policy.early_exercise_decision(
                asset,
                spot=spot,
                option_mark=float(position.current_price),
                days_to_expiry=days_to_expiry,
                annual_rate=annual_rate,
                dividend=dividend,
            )
            if not decision.exercise:
                continue
            event = policy.plan(
                asset,
                int(position.quantity),
                spot=spot,
                effective_date=current,
                early=True,
                option_mark=float(position.current_price),
                days_to_expiry=days_to_expiry,
                annual_rate=annual_rate,
                dividend=dividend,
            )
            self._apply_option_lifecycle_event(event, spot, date)

    def _accumulate_position(self, position: Position, add_quantity: int,
                             fill_price: float) -> None:
        """Add a remainder or target delta to a same-direction position.

        add_quantity is POSITIVE for a long add, NEGATIVE for a short add; the
        position's sign is preserved.
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
        not modeled — cash-account approximation); COVER closes the requested
        short quantity, debiting cash, and records the Trade with the negative
        quantity (Trade.pnl = qty * (exit - entry) is sign-correct). Gated on
        self._desk_mode, so a fund's netted-SHORT residual and account-level
        COVER fill the same way a single desk's do.

        REALISTIC FILLS (Step 6, opt-in via enable_realistic_fills; ``adv`` is
        the trailing volume passed by the caller): adds square-root ADV market
        impact to slippage and caps openings at participation_cap*ADV; strict
        fail-closed mode caps exits too. Returns a follow-on intent dict for the
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
                    and existing_pos and existing_pos.quantity > 0)
                or (intent.get('target_native')
                    and existing_pos and existing_pos.quantity > 0)):
            base_fill = base_price * (1 + slippage)
            portfolio_value = self.portfolio.get_portfolio_value()
            trade_value = portfolio_value * position_size
            # Desk intents may carry an absolute share count (Phase 8); it
            # overrides the dollar sizing. Strategy mode never sets it. A
            # re-queued remainder (realistic fills) also carries it.
            explicit_quantity = intent.get('quantity')
            desired = (explicit_quantity if explicit_quantity is not None
                       else int(trade_value / base_fill))

            if (explicit_quantity is not None and not intent.get('accumulate')
                    and desired * base_fill > trade_value + 1e-9):
                logger.warning(
                    "Dropping BUY intent for %s on %s: absolute quantity "
                    "requests %.2f notional against %.2f risk budget",
                    asset.symbol, fill_date, desired * base_fill, trade_value)
                return None

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
            if not self._authorize_exposure(
                    request_id=(f"{pd.Timestamp(fill_date).date()}:"
                                f"{asset}:BUY"),
                    symbol=asset.symbol,
                    kind=ExposureKind.LONG_STOCK,
                    quantity=int(quantity),
                    date=fill_date,
                    unit_price=fill_price,
                    contract_multiplier=1):
                logger.warning(
                    "Dropping BUY intent for %s on %s: portfolio mechanics "
                    "rejected exposure", asset.symbol, fill_date)
                return None
            if (self.portfolio_mechanics is None
                    and self.portfolio.cash < cost):
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
            return requeue_remainder(intent, remainder, realistic)

        elif signal == 'SELL' and existing_pos and existing_pos.quantity > 0:
            exact = intent.get('target_native') or intent.get(
                'liquidity_remainder')
            if exact and intent.get('quantity') is not None:
                desired = min(int(intent['quantity']), existing_pos.quantity)
            else:
                desired = existing_pos.quantity
            if realistic and self.reject_fills_without_adv:
                quantity, remainder = self._capped_fill_quantity(desired, adv)
            else:
                quantity, remainder = desired, 0
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
            if not intent.get('target_native') and remainder == 0:
                # Legacy behavior explicitly removed after close_position.
                self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset), 'action': 'SELL',
                'quantity': quantity, 'price': fill_price, 'proceeds': proceeds
            })
            logger.info("SELL %d %s @ %.4f on %s (signal %s)",
                        quantity, asset.symbol, fill_price, fill_date,
                        intent['signal_date'])
            return requeue_remainder(intent, remainder, realistic)

        elif signal == 'SHORT' and self._desk_mode and (
                (not existing_pos or existing_pos.quantity == 0)
                or (realistic and intent.get('accumulate')
                    and existing_pos and existing_pos.quantity < 0)
                or (intent.get('target_native')
                    and existing_pos and existing_pos.quantity < 0)):
            # A short is a SELL: adverse slippage means a LOWER fill.
            base_fill = base_price * (1 - slippage)
            portfolio_value = self.portfolio.get_portfolio_value()
            trade_value = portfolio_value * position_size
            explicit_quantity = intent.get('quantity')
            desired = (explicit_quantity if explicit_quantity is not None
                       else int(trade_value / base_fill))

            if (explicit_quantity is not None and not intent.get('accumulate')
                    and desired * base_fill > trade_value + 1e-9):
                logger.warning(
                    "Dropping SHORT intent for %s on %s: absolute quantity "
                    "requests %.2f notional against %.2f risk budget",
                    asset.symbol, fill_date, desired * base_fill, trade_value)
                return None

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

            if not self._authorize_exposure(
                    request_id=(f"{pd.Timestamp(fill_date).date()}:"
                                f"{asset}:SHORT"),
                    symbol=asset.symbol,
                    kind=ExposureKind.SHORT_STOCK,
                    quantity=int(quantity),
                    date=fill_date,
                    unit_price=fill_price,
                    contract_multiplier=1):
                logger.warning(
                    "Dropping SHORT intent for %s on %s: portfolio mechanics "
                    "rejected exposure", asset.symbol, fill_date)
                return None

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
            return requeue_remainder(intent, remainder, realistic)

        elif signal == 'COVER' and self._desk_mode \
                and existing_pos and existing_pos.quantity < 0:
            # A cover is a BUY: adverse slippage means a HIGHER fill.
            exact = intent.get('target_native') or intent.get(
                'liquidity_remainder')
            if exact and intent.get('quantity') is not None:
                desired = min(int(intent['quantity']), abs(existing_pos.quantity))
            else:
                desired = abs(existing_pos.quantity)
            if realistic and self.reject_fills_without_adv:
                filled, remainder = self._capped_fill_quantity(desired, adv)
            else:
                filled, remainder = desired, 0
            quantity = -filled
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
            if not intent.get('target_native') and remainder == 0:
                # Legacy behavior explicitly removed after close_position.
                self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'instrument': str(asset), 'action': 'COVER',
                'quantity': abs(quantity), 'price': fill_price, 'cost': cost
            })
            logger.info("COVER %d %s @ %.4f on %s (signal %s)",
                        abs(quantity), asset.symbol, fill_price, fill_date,
                        intent['signal_date'])
            return requeue_remainder(intent, remainder, realistic)

        # No matching branch (e.g. a BUY when already long without accumulate,
        # or a close with no position) -> consume the intent. Realistic-fill
        # remainders are the only thing re-queued (returned above).
        return None

    def _build_benchmark(self, benchmark_symbol: str, start_date: str,
                         end_date: str) -> Optional[Dict]:
        """Compatibility facade over the optional benchmark service."""
        return build_benchmark(
            self.market_data,
            self.portfolio.initial_capital,
            benchmark_symbol,
            start_date,
            end_date,
            log=logger,
        )

    def _compute_oos_folds(self, desk, alpha: float = 0.05) -> Dict:
        """Compatibility facade over account-level OOS fold analysis."""
        return compute_oos_folds(self.portfolio, desk, alpha=alpha)

    def _generate_report(self, benchmark_symbol: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict:
        """Generate the stable report schema from completed engine state."""
        state = BacktestReportState(
            strategy=self.strategy,
            driver=self._driver,
            desk=self.desk,
            desk_mode=self._desk_mode,
            orchestrator=self.orchestrator,
            reweighter=self.reweighter,
            portfolio=self.portfolio,
            trades_log=self.trades_log,
            pending_intents=self.pending_intents,
            pending_structures=self.pending_structures,
            mechanics_enabled=(
                self.portfolio_mechanics is not None
                or self.option_lifecycle_policy is not None
            ),
            mechanics_log=self.mechanics_log,
        )
        return generate_report(
            state,
            benchmark_symbol=benchmark_symbol,
            start_date=start_date,
            end_date=end_date,
            benchmark_loader=self._build_benchmark,
            oos_folds_loader=self._compute_oos_folds,
            log=logger,
        )
