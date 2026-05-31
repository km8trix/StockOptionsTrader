"""
Backtesting Engine - Simulates trading strategies on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.models import Asset, AssetType, Order, OrderType, OrderStatus, Position
from portfolio.manager import PortfolioManager
from data.market_data import MarketDataHandler
from strategies.base import Strategy


class BacktestEngine:
    """Simulates trading strategies on historical data"""
    
    def __init__(self, strategy: Strategy, initial_capital: float = 100000,
                 commission: float = 0.001):
        self.strategy = strategy
        self.portfolio = PortfolioManager(initial_capital)
        self.commission = commission
        self.market_data = MarketDataHandler()
        self.trades_log: List[Dict] = []
        self.signals_log: List[Dict] = []
    
    # StockOptionsTrader/backtesting/backtest_engine.py

    def run(self, symbols: List[str], start_date: str, end_date: str,
            position_size: float = 0.1) -> Dict:
        """Run backtest on given symbols safely"""
        all_data = {}
        for symbol in symbols:
            data = self.market_data.fetch_stock_data(symbol, start_date, end_date)
            if not data.empty:
                # To completely eliminate global indicators leaking future context:
                # Indicators can be calculated cleanly on expanding data slices below.
                all_data[symbol] = data
        
        if not all_data:
            return {'error': 'No data available'}
        
        # Align calendar dates across all downloaded symbols
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index.strftime('%Y-%m-%d'))
        sorted_dates = sorted(list(all_dates))
        
        # Simulate sequential historical trading day by day
        for date_str in sorted_dates:
            date = pd.to_datetime(date_str)
            
            # --- PHASE 1: UNIVERSAL CURRENT PRICE SYNCHRONIZATION ---
            # Update all asset current prices cleanly before running any trading logic or snapshots
            for symbol, data in all_data.items():
                if date_str in data.index:
                    asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
                    current_day_close = data.loc[date_str, 'close']
                    
                    # Update active portfolio position object tracking variables
                    existing_pos = self.portfolio.get_position(asset)
                    if existing_pos:
                        existing_pos.current_price = current_day_close

            # --- PHASE 2: STRATEGY SIGNAL EXECUTION ---
            for symbol, data in all_data.items():
                if date_str not in data.index:
                    continue
                
                # Slice historical data strictly up to the current simulated date
                historical_data = data[data.index <= date]
                
                # Run indicator calculations dynamically purely on the known past data window
                historical_data_with_indicators = self.market_data.calculate_indicators(historical_data.copy())
                
                asset = Asset(symbol=symbol, asset_type=AssetType.STOCK)
                signal = self.strategy.generate_signals(historical_data_with_indicators, asset)
                price = historical_data_with_indicators.iloc[-1]['close']
                
                # Execute signals knowing position prices are accurate
                self._execute_signal(asset, signal, price, date, position_size)
            
            # --- PHASE 3: RECORD SNAPSHOT ---
            self.portfolio.record_snapshot(date)
        
        return self._generate_report()

    def _execute_signal(self, asset: Asset, signal: str, price: float,
                       date: datetime, position_size: float):
        """Execute trade based on signal (Price syncing logic removed from here)"""
        portfolio_value = self.portfolio.get_portfolio_value()
        trade_value = portfolio_value * position_size
        quantity = int(trade_value / price)
        
        if quantity == 0:
            return
        
        existing_pos = self.portfolio.get_position(asset)
        
        if signal == 'BUY' and (not existing_pos or existing_pos.quantity == 0):
            cost = quantity * price * (1 + self.commission)
            if self.portfolio.cash >= cost:
                self.portfolio.cash -= cost
                position = Position(
                    asset=asset,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price,
                    timestamp=date
                )
                self.portfolio.add_position(position)
                self.trades_log.append({
                    'date': date, 'symbol': asset.symbol, 'action': 'BUY',
                    'quantity': quantity, 'price': price, 'cost': cost
                })
        
        elif signal == 'SELL' and existing_pos and existing_pos.quantity > 0:
            proceeds = existing_pos.quantity * price * (1 - self.commission)
            self.portfolio.cash += proceeds
            self.portfolio.close_position(
                asset, price, existing_pos.quantity, 
                existing_pos.timestamp, date
            )
            self.portfolio.remove_position(asset)
            self.trades_log.append({
                'date': date, 'symbol': asset.symbol, 'action': 'SELL',
                'quantity': existing_pos.quantity, 'price': price, 'proceeds': proceeds
            })
            
    def _generate_report(self) -> Dict:
        """Generate backtest report"""
        summary = self.portfolio.get_summary()
        
        return {
            'strategy': self.strategy.name,
            'summary': summary,
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
            'portfolio_history': self.portfolio.portfolio_history
        }
