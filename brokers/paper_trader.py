"""
Paper Trader - Simulated trading for live market data
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, Optional, List
from core.models import Asset, AssetType, Order, OrderStatus, OrderType, Position
from portfolio.manager import PortfolioManager
from data.market_data import MarketDataHandler
from brokers.base import ExecutionBroker

logger = logging.getLogger(__name__)


class PaperTrader(ExecutionBroker):
    """Simulated trading with real market data"""

    def __init__(self, initial_capital: float = 100000,
                 price_sanity_threshold: Optional[float] = None):
        self.portfolio = PortfolioManager(initial_capital)
        self.market_data = MarketDataHandler()
        # Opt-in fat-finger guard (None disables); see ExecutionBroker.
        self.price_sanity_threshold = price_sanity_threshold
        self.pending_orders: List[Order] = []
        self.order_id_counter = 0
        # Date of the close that served each symbol's last price quote.
        self.last_price_dates: Dict[str, date] = {}
        # Guards order/portfolio mutation under gunicorn --threads: two
        # overlapping status polls both run process_orders(), and without a
        # lock both can pass the fill check for the same pending order
        # (double fill, then ValueError on the second remove). RLock, not
        # Lock, because get_portfolio_status() calls process_orders().
        self._lock = threading.RLock()
        # Filled orders kept queryable by order_status() so a PatientExecutor
        # driving the paper broker can poll/bank fills exactly like live (paper
        # fills all-or-nothing, so a filled order reports its full quantity).
        # Keyed by order_id; bounded by the orders placed this session.
        self.filled_orders: Dict[str, Order] = {}

    def place_order(self, asset: Asset, order_type: OrderType, quantity: int,
                   limit_price: Optional[float]) -> str:
        """Place a new order"""
        self._check_price_sanity(asset.symbol, limit_price)
        with self._lock:
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

            # Only STOCK is priceable here: get_current_price() returns the
            # UNDERLYING equity close, so an option order would limit-check
            # and "fill" at the stock price with no x100 multiplier —
            # silently corrupting the paper book. Reject until a real option
            # quote source exists (LiveEtradeBroker handles options).
            if asset.asset_type != AssetType.STOCK:
                logger.warning(
                    "PaperTrader rejects non-stock order %s (%s %s): no "
                    "option pricing in paper mode", order_id,
                    asset.asset_type.value, asset.symbol)
                order.status = OrderStatus.REJECTED
                self.filled_orders[order_id] = order
                return order_id

            self.pending_orders.append(order)
            return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by its order id.

        Returns True when the order was found and removed from the pending
        queue (its status is set to CANCELLED), False otherwise.

        Processes pending fills FIRST: a fill that just became marketable is
        captured (the order moves to filled_orders) before the cancel, so a
        fill racing the cancel is never silently dropped — an already-filled
        order is then no longer cancellable and returns False, mirroring live
        cancel-vs-fill behavior.

        Like get_portfolio_status(), this runs process_orders() under the
        RLock, so the lock is held across the price fetch — an intentional,
        pre-existing pattern (paper-only; cancel latency gains one quote
        round-trip in exchange for never losing a racing fill).
        """
        with self._lock:
            self.process_orders()
            for order in self.pending_orders:
                if order.order_id == order_id:
                    order.status = OrderStatus.CANCELLED
                    self.pending_orders.remove(order)
                    logger.info("Cancelled pending order %s (%s %d %s)",
                                order_id, order.order_type.value,
                                order.quantity, order.asset.symbol)
                    return True
        logger.warning("cancel_order: no pending order with id %s", order_id)
        return False

    def order_status(self, order_id: str) -> Optional[Dict]:
        """Status of an order for PatientExecutor fill polling.

        Returns {'status', 'filled_quantity', 'avg_fill_price'}:
        'OPEN' with filled_quantity 0 while pending, 'FILLED' with the full
        quantity and fill price once executed — or None if the order id is
        unknown. This is the same contract LiveEtradeBroker.order_status()
        satisfies, so a PatientExecutor can drive paper rehearsal exactly as
        it drives the live broker (closing a paper/live parity gap).
        """
        with self._lock:
            for order in self.pending_orders:
                if order.order_id == order_id:
                    return {"status": "OPEN",
                            "filled_quantity": order.filled_quantity,
                            "avg_fill_price": order.filled_price}
            done = self.filled_orders.get(order_id)
            if done is not None:
                status = ("REJECTED" if done.status == OrderStatus.REJECTED
                          else "FILLED")
                return {"status": status,
                        "filled_quantity": done.filled_quantity,
                        "avg_fill_price": done.filled_price}
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the most recent close for a symbol.

        Fetches a window of the last 10 calendar days through today (a
        same-day-only request returns nothing on weekends/holidays) and
        uses the latest close. The date that close traded is recorded in
        self.last_price_dates, and a WARNING is logged when it is more
        than 3 business days old. Returns None when the window is empty.
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=10)
            data = self.market_data.fetch_stock_data(
                symbol,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
            )
            if data is None or data.empty:
                logger.warning(
                    "No price data for %s in the last 10 calendar days", symbol
                )
                return None

            price = float(data['close'].iloc[-1])
            try:
                price_date = pd.Timestamp(data.index[-1]).date()
                self.last_price_dates[symbol] = price_date
                age_bdays = int(np.busday_count(price_date, end.date()))
                if age_bdays > 3:
                    logger.warning(
                        "Stale price for %s: latest close %.2f is from %s "
                        "(%d business days old)",
                        symbol, price, price_date.isoformat(), age_bdays,
                    )
            except Exception as e:
                logger.warning(
                    "Could not assess price staleness for %s: %s", symbol, e
                )
            return price
        except Exception as e:
            logger.warning("Failed to fetch current price for %s: %s", symbol, e)
        return None
    
    def process_orders(self):
        """Process pending orders at current market prices"""
        with self._lock:
            for order in self.pending_orders[:]:
                current_price = self.get_current_price(order.asset.symbol)

                if current_price is None:
                    continue

                # Check if order can be filled
                filled = False

                if order.price is None:
                    # Market order (limit_price=None per ExecutionBroker
                    # contract): immediately marketable for both BUY and
                    # SELL at current price.
                    filled = True
                elif order.order_type == OrderType.BUY and current_price <= order.price:
                    filled = True
                elif order.order_type == OrderType.SELL and current_price >= order.price:
                    filled = True

                if filled:
                    executed = self._execute_order(order, current_price)
                    if executed:
                        # Record the fill on the order and keep it queryable
                        # via order_status() (paper fills fully or not at all).
                        order.filled_quantity = order.quantity
                        order.filled_price = current_price
                        order.status = OrderStatus.FILLED
                    else:
                        # Marketable but declined (insufficient cash / no
                        # position): REJECTED, never a phantom fill — the
                        # portfolio did not change, so order_status must not
                        # report a fill PatientExecutor would bank.
                        order.status = OrderStatus.REJECTED
                    self.filled_orders[order.order_id] = order
                    self.pending_orders.remove(order)
    
    def _execute_order(self, order: Order, execution_price: float) -> bool:
        """Execute an order against the portfolio.

        Returns True only when the portfolio was actually mutated; False when
        the order was declined (insufficient cash / no position), so the
        caller never reports a phantom fill.
        """
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
                return True
            logger.warning(
                "Paper BUY %s x%d rejected: cost %.2f exceeds cash %.2f",
                order.asset.symbol, order.quantity, cost, self.portfolio.cash)
            return False

        if order.order_type == OrderType.SELL:
            existing_pos = self.portfolio.get_position(order.asset)
            if existing_pos and existing_pos.quantity >= order.quantity:
                proceeds = order.quantity * execution_price * 0.999
                self.portfolio.cash += proceeds

                if existing_pos.quantity == order.quantity:
                    self.portfolio.remove_position(order.asset)
                else:
                    existing_pos.quantity -= order.quantity
                return True
            logger.warning(
                "Paper SELL %s x%d rejected: position %s insufficient",
                order.asset.symbol, order.quantity,
                existing_pos.quantity if existing_pos else "missing")
            return False
        return False
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        with self._lock:
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
