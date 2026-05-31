"""
Portfolio Manager - Tracks positions, cash, and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from core.models import Asset, Position, Trade, Order, OrderType


class PortfolioManager:
    """Manages positions, cash balance, and portfolio metrics"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[Asset, Position] = {}
        self.closed_trades: List[Trade] = []
        self.order_history: List[Order] = []
        self.portfolio_history: List[Dict] = []
        
    def add_position(self, position: Position):
        """Add or update a position"""
        self.positions[position.asset] = position
    
    def remove_position(self, asset: Asset):
        """Remove a position"""
        if asset in self.positions:
            del self.positions[asset]
    
    def get_position(self, asset: Asset) -> Optional[Position]:
        """Get position for an asset"""
        return self.positions.get(asset)
    
    def update_cash(self, amount: float):
        """Update cash balance"""
        self.cash += amount
    
    def close_position(self, asset: Asset, exit_price: float, quantity: int, 
                       entry_time: datetime, exit_time: datetime):
        """Close a position and record the trade"""
        if asset not in self.positions:
            return
        
        position = self.positions[asset]
        trade = Trade(
            asset=asset,
            entry_price=position.avg_entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_time=entry_time,
            exit_time=exit_time
        )
        self.closed_trades.append(trade)
        
        if position.quantity == quantity:
            self.remove_position(asset)
        else:
            position.quantity -= quantity
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        position_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        return self.cash + position_value
    
    def get_portfolio_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.pnl() for pos in self.positions.values())
    
    def get_portfolio_pnl_pct(self) -> float:
        """Calculate portfolio P&L percentage"""
        portfolio_value = self.get_portfolio_value()
        if portfolio_value == 0:
            return 0
        return ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
    
    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed trades"""
        return sum(trade.pnl for trade in self.closed_trades)
    
    def get_total_return(self) -> float:
        """Get total return including realized and unrealized P&L"""
        return self.get_realized_pnl() + self.get_portfolio_pnl()
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown with safety safeguards"""
        if not self.portfolio_history:
            return 0.0
        
        values = [h['portfolio_value'] for h in self.portfolio_history]
        running_max = np.maximum.accumulate(values)
        
        # Avoid division runtime warnings if running_max somehow encounters 0
        running_max = np.where(running_max == 0, 1.0, running_max)
        
        drawdowns = (np.array(values) - running_max) / running_max
        return float(np.min(drawdowns) * 100)

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio safely avoiding zero standard deviation crashes"""
        if not self.closed_trades:
            return 0.0
        
        returns = [trade.pnl_pct for trade in self.closed_trades]
        if len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - (risk_free_rate / 252)
        std_dev = np.std(excess_returns)
        
        # Prevent division by zero runtime crash
        if std_dev == 0:
            return 0.0
            
        return np.mean(excess_returns) / std_dev * np.sqrt(252)
    
    def get_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.closed_trades:
            return 0
        
        wins = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        return (wins / len(self.closed_trades)) * 100
    
    def record_snapshot(self, timestamp: datetime):
        """Record portfolio snapshot for history"""
        snapshot = {
            'timestamp': timestamp,
            'portfolio_value': self.get_portfolio_value(),
            'cash': self.cash,
            'positions_count': len(self.positions),
            'unrealized_pnl': self.get_portfolio_pnl(),
            'realized_pnl': self.get_realized_pnl(),
        }
        self.portfolio_history.append(snapshot)
    
    def get_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'initial_capital': self.initial_capital,
            'current_value': self.get_portfolio_value(),
            'cash': self.cash,
            'total_return': self.get_total_return(),
            'total_return_pct': self.get_portfolio_pnl_pct(),
            'realized_pnl': self.get_realized_pnl(),
            'unrealized_pnl': self.get_portfolio_pnl(),
            'positions_count': len(self.positions),
            'closed_trades': len(self.closed_trades),
            'win_rate': self.get_win_rate(),
            'max_drawdown': self.get_max_drawdown(),
            'sharpe_ratio': self.get_sharpe_ratio(),
        }
