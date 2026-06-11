"""
Paper Trader - Simulated trading for live market data
"""

from __future__ import annotations

import logging

import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List
from core.models import Asset, AssetType, Order, OrderStatus, OrderType, Position
from portfolio.manager import PortfolioManager
from data.market_data import MarketDataHandler
from brokers.base import ExecutionBroker

logger = logging.getLogger(__name__)


class PaperTrader(ExecutionBroker):
    """Simulated trading with real market data"""

    def __init__(self, initial_capital: float = 100000):
        self.portfolio = PortfolioManager(initial_capital)
        self.market_data = MarketDataHandler()
        self.pending_orders: List[Order] = []
        self.order_id_counter = 0

    def place_order(self, asset: Asset, order_type: OrderType, quantity: int,
                   limit_price: Optional[float]) -> str:
        """Place a new order"""
        self.order_id_counter += 1
        order_id = f"ORD-{self.order_id_counter:06d}"
        
        order = Order(
            asset=asset,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            timestamp=datetime.now(),
            order_id=order_id
        )
        
        self.pending_orders.append(order)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by its order id.

        Returns True when the order was found and removed from the pending
        queue (its status is set to CANCELLED), False otherwise.
        """
        for order in self.pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self.pending_orders.remove(order)
                logger.info("Cancelled pending order %s (%s %d %s)",
                            order_id, order.order_type.value, order.quantity,
                            order.asset.symbol)
                return True
        logger.warning("cancel_order: no pending order with id %s", order_id)
        return False

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            data = self.market_data.fetch_stock_data(symbol,
                                                     datetime.now().strftime('%Y-%m-%d'),
                                                     datetime.now().strftime('%Y-%m-%d'))
            if not data.empty:
                return data.iloc[-1]['close']
        except Exception as e:
            logger.warning("Failed to fetch current price for %s: %s", symbol, e)
        return None
    
    def process_orders(self):
        """Process pending orders at current market prices"""
        for order in self.pending_orders[:]:
            current_price = self.get_current_price(order.asset.symbol)
            
            if current_price is None:
                continue
            
            # Check if order can be filled
            filled = False

            if order.price is None:
                # Market order (limit_price=None per ExecutionBroker contract):
                # immediately marketable for both BUY and SELL at current price.
                filled = True
            elif order.order_type == OrderType.BUY and current_price <= order.price:
                filled = True
            elif order.order_type == OrderType.SELL and current_price >= order.price:
                filled = True
            
            if filled:
                self._execute_order(order, current_price)
                self.pending_orders.remove(order)
    
    def _execute_order(self, order: Order, execution_price: float):
        """Execute an order"""
        cost = order.quantity * execution_price * 1.001  # Add small slippage
        
        if order.order_type == OrderType.BUY:
            if self.portfolio.cash >= cost:
                self.portfolio.cash -= cost
                existing_pos = self.portfolio.get_position(order.asset)
                
                if existing_pos:
                    # Average up
                    new_avg = (existing_pos.avg_entry_price * existing_pos.quantity + 
                              execution_price * order.quantity) / (existing_pos.quantity + order.quantity)
                    existing_pos.quantity += order.quantity
                    existing_pos.avg_entry_price = new_avg
                else:
                    position = Position(
                        asset=order.asset,
                        quantity=order.quantity,
                        avg_entry_price=execution_price,
                        current_price=execution_price,
                        timestamp=datetime.now()
                    )
                    self.portfolio.add_position(position)
        
        elif order.order_type == OrderType.SELL:
            existing_pos = self.portfolio.get_position(order.asset)
            if existing_pos and existing_pos.quantity >= order.quantity:
                proceeds = order.quantity * execution_price * 0.999
                self.portfolio.cash += proceeds
                
                if existing_pos.quantity == order.quantity:
                    self.portfolio.remove_position(order.asset)
                else:
                    existing_pos.quantity -= order.quantity
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        self.process_orders()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cash': self.portfolio.cash,
            'portfolio_value': self.portfolio.get_portfolio_value(),
            'unrealized_pnl': self.portfolio.get_portfolio_pnl(),
            'positions': [
                {
                    'symbol': pos.asset.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl(),
                    'pnl_pct': pos.pnl_pct()
                }
                for pos in self.portfolio.positions.values()
            ],
            'pending_orders': len(self.pending_orders)
        }
