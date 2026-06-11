"""
Execution broker interface.

All order routing (paper or live) goes through the ExecutionBroker ABC so
that strategies and trading desks stay agnostic to where orders actually
fill. Concrete implementations: brokers.paper_trader.PaperTrader and
brokers.live_trader.LiveEtradeBroker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from core.models import Asset, OrderType


class ExecutionBroker(ABC):
    """Abstract interface that every execution venue must implement."""

    @abstractmethod
    def place_order(self, asset: Asset, order_type: OrderType, quantity: int,
                    limit_price: float | None) -> str:
        """Place an order and return a broker-assigned order id.

        Args:
            asset: the asset to trade.
            order_type: OrderType.BUY or OrderType.SELL.
            quantity: number of shares/contracts (positive).
            limit_price: limit price, or None for a market order.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order; return True if an order was cancelled."""

    @abstractmethod
    def get_portfolio_status(self) -> Dict:
        """Return a snapshot of cash, portfolio value, and open positions."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float | None:
        """Return the latest known price for symbol, or None if unavailable."""
