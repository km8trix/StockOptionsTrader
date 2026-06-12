"""
Backtesting Engine - Simulates trading strategies on historical data

Execution model (no same-bar lookahead):
    A signal generated from day T's data is queued as a pending intent and
    filled at day T+1's OPEN, adjusted for slippage. Signals generated on the
    final simulated day therefore never fill; they are surfaced in the report
    under 'pending_signals'.
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional
from core.models import Asset, AssetType, Order, OrderType, OrderStatus, Position
from portfolio.manager import PortfolioManager
from data.market_data import MarketDataHandler
from strategies.base import Strategy

logger = logging.getLogger(__name__)

#: Trading days an intent may wait for a usable bar before being dropped.
MAX_PENDING_DAYS = 5


class BacktestEngine:
    """Simulates trading strategies on historical data"""

    def __init__(self, strategy: Strategy, initial_capital: float = 100000,
                 commission: float = 0.001, slippage_bps: float = 5.0):
        self.strategy = strategy
        self.portfolio = PortfolioManager(initial_capital)
        self.commission = commission
        self.slippage_bps = slippage_bps
        self.market_data = MarketDataHandler()
        self.trades_log: List[Dict] = []
        self.signals_log: List[Dict] = []
        # Pending intents keyed by Asset. Each value is a dict with keys:
        # 'signal' ('BUY'/'SELL'), 'signal_date', and 'days_waiting' (count of
        # trading days the intent has waited because the symbol had no usable
        # bar). A new intent for an asset replaces an older pending one.
        self.pending_intents: Dict[Asset, Dict] = {}

    def run(self, symbols: List[str], start_date: str, end_date: str,
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
                # Indicators are computed on expanding slices inside the loop
                # below, so no global indicator state can leak future context.
                all_data[symbol] = data

        if not all_data:
            logger.warning("Backtest aborted: no data available for %s", symbols)
            return {'error': 'No data available'}

        # Align trading dates across all downloaded symbols
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        sorted_dates = sorted(all_dates)

        self.pending_intents = {}

        # Simulate sequential historical trading day by day
        total_days = len(sorted_dates)
        for day_number, date in enumerate(sorted_dates, start=1):
            # --- PHASE 1: FILL PENDING INTENTS AT TODAY'S OPEN ---
            self._fill_pending_intents(all_data, date, position_size)

            # --- PHASE 2: MARK POSITIONS TO TODAY'S CLOSE ---
            for symbol, data in all_data.items():
                if date not in data.index:
                    continue
                asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
                existing_pos = self.portfolio.get_position(asset)
                if existing_pos:
                    existing_pos.current_price = float(data.loc[date, 'close'])

            # --- PHASE 3: SIGNALS ON DATA THROUGH TODAY, QUEUED FOR TOMORROW ---
            for symbol, data in all_data.items():
                if date not in data.index:
                    continue

                # Slice historical data strictly up to the current simulated date
                historical_data = data[data.index <= date]

                # Run indicator calculations purely on the known past data window
                historical_data_with_indicators = self.market_data.calculate_indicators(
                    historical_data.copy())

                asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
                signal = self.strategy.generate_signals(historical_data_with_indicators, asset)

                if signal in ('BUY', 'SELL'):
                    # A new intent for an asset replaces an older pending one.
                    self.pending_intents[asset] = {
                        'signal': signal,
                        'signal_date': date,
                        'days_waiting': 0,
                    }

            # --- PHASE 4: RECORD SNAPSHOT ---
            self.portfolio.record_snapshot(date)

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

    def _fill_pending_intents(self, all_data: Dict[str, pd.DataFrame],
                              date, position_size: float) -> None:
        """Fill intents queued on previous days at today's open price.

        Intents without a usable bar today (no row, or both open and close
        NaN) accrue a waiting day and are dropped after MAX_PENDING_DAYS.
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

            if pd.isna(fill_base):
                intent['days_waiting'] += 1
                if intent['days_waiting'] >= MAX_PENDING_DAYS:
                    logger.warning(
                        "Dropping %s intent for %s (signal %s): no usable bar "
                        "for %d trading days",
                        intent['signal'], asset.symbol, intent['signal_date'],
                        intent['days_waiting'])
                    del self.pending_intents[asset]
                continue

            self._fill_intent(asset, intent, float(fill_base), date, position_size)
            del self.pending_intents[asset]

    def _fill_intent(self, asset: Asset, intent: Dict, base_price: float,
                     fill_date, position_size: float) -> None:
        """Fill a queued intent at base_price (today's open) with slippage.

        BUY fills at base_price * (1 + slippage_bps/10000); SELL fills at
        base_price * (1 - slippage_bps/10000). Position size is the
        position_size fraction of portfolio value evaluated at fill time.
        Commission semantics: cost = qty * fill * (1 + commission) on buy;
        proceeds = qty * fill * (1 - commission) on sell.
        """
        signal = intent['signal']
        slippage = self.slippage_bps / 10000.0

        if base_price <= 0:
            logger.warning("Skipping %s fill for %s on %s: non-positive price %s",
                           signal, asset.symbol, fill_date, base_price)
            return

        existing_pos = self.portfolio.get_position(asset)

        if signal == 'BUY' and (not existing_pos or existing_pos.quantity == 0):
            fill_price = base_price * (1 + slippage)
            portfolio_value = self.portfolio.get_portfolio_value()
            trade_value = portfolio_value * position_size
            quantity = int(trade_value / fill_price)

            if quantity == 0:
                logger.warning(
                    "Dropping BUY intent for %s on %s: position sizes to 0 "
                    "shares (trade value %.2f at fill price %.4f)",
                    asset.symbol, fill_date, trade_value, fill_price)
                return

            cost = quantity * fill_price * (1 + self.commission)
            if self.portfolio.cash < cost:
                logger.warning(
                    "Dropping BUY intent for %s on %s: insufficient cash "
                    "(needed %.2f, available %.2f)",
                    asset.symbol, fill_date, cost, self.portfolio.cash)
                return

            self.portfolio.cash -= cost
            position = Position(
                asset=asset,
                quantity=quantity,
                avg_entry_price=fill_price,
                current_price=fill_price,
                timestamp=fill_date
            )
            self.portfolio.add_position(position)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'action': 'BUY',
                'quantity': quantity, 'price': fill_price, 'cost': cost
            })
            logger.info("BUY %d %s @ %.4f on %s (signal %s)",
                        quantity, asset.symbol, fill_price, fill_date,
                        intent['signal_date'])

        elif signal == 'SELL' and existing_pos and existing_pos.quantity > 0:
            fill_price = base_price * (1 - slippage)
            quantity = existing_pos.quantity
            proceeds = quantity * fill_price * (1 - self.commission)
            self.portfolio.cash += proceeds
            self.portfolio.close_position(
                asset, fill_price, quantity,
                existing_pos.timestamp, fill_date
            )
            self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': fill_date, 'signal_date': intent['signal_date'],
                'symbol': asset.symbol, 'action': 'SELL',
                'quantity': quantity, 'price': fill_price, 'proceeds': proceeds
            })
            logger.info("SELL %d %s @ %.4f on %s (signal %s)",
                        quantity, asset.symbol, fill_price, fill_date,
                        intent['signal_date'])

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

    def _generate_report(self, benchmark_symbol: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict:
        """Generate backtest report"""
        summary = self.portfolio.get_summary()

        benchmark = None
        if benchmark_symbol:
            benchmark = self._build_benchmark(benchmark_symbol,
                                              start_date, end_date)

        return {
            'strategy': self.strategy.name,
            'summary': summary,
            'benchmark': benchmark,
            'drawdown_series': self.portfolio.get_drawdown_series(),
            'trades': self.trades_log,
            'closed_trades': [
                {
                    'symbol': trade.asset.symbol,
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
