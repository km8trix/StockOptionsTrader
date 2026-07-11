"""
Execution broker interface.

All order routing (paper or live) goes through the ExecutionBroker ABC so
that strategies and trading desks stay agnostic to where orders actually
fill. Concrete implementations: brokers.paper_trader.PaperTrader and
brokers.live_trader.LiveEtradeBroker.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.models import Asset, OrderType

logger = logging.getLogger(__name__)


class ExecutionBroker(ABC):
    """Abstract interface that every execution venue must implement."""

    #: Optional fractional distance from the current price beyond which a
    #: limit price is flagged as a likely fat-finger (e.g. 0.5 = 50%). None
    #: disables the check; concrete brokers expose it via their constructor.
    price_sanity_threshold: Optional[float] = None

    def _check_price_sanity(self, symbol: str,
                            limit_price: Optional[float]) -> None:
        """Warn when a limit price is implausibly far from the current price.

        Opt-in and WARN-ONLY: does nothing unless price_sanity_threshold is
        set, and never blocks the order (a legitimately far-from-market limit
        must still place). It exists to surface obvious fat-fingers — a 0.01
        limit on a $200 name, or a 10x typo — in the logs rather than silently
        routing them to the broker.
        """
        threshold = self.price_sanity_threshold
        if threshold is None or limit_price is None:
            return
        try:
            current = self.get_current_price(symbol)
        except Exception:  # noqa: BLE001 - a sanity check must never break place
            return
        if current is None or current <= 0:
            return
        distance = abs(limit_price - current) / current
        if distance > threshold:
            logger.warning(
                "Limit price %.4f for %s is %.1f%% away from current %.4f "
                "(threshold %.0f%%) — possible fat-finger",
                limit_price, symbol, distance * 100, current, threshold * 100)

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
    def order_status(self, order_id: str) -> Optional[Dict]:
        """Status of an order, for fill polling and confirmation.

        Contract (Step 5 — previously duck-typed, satisfied by all
        implementations; now part of the ABC so the operational loop can
        rely on it):

            {'status': str,            # 'OPEN' while working; terminal:
                                       # 'FILLED'/'EXECUTED'/'CANCELLED'/
                                       # 'REJECTED'/'EXPIRED'
             'filled_quantity': float, # cumulative units filled so far
             'avg_fill_price': float | None}  # avg price of those fills

        Returns None when the broker does not know the order id. This is
        the same contract execution.patient_executor polls (its module
        docstring documents the terminal-status vocabulary) and the one
        utils.live_session uses to confirm fills into the persistent
        local book.
        """

    @abstractmethod
    def get_portfolio_status(self) -> Dict:
        """Return a snapshot of cash, portfolio value, and open positions."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float | None:
        """Return the latest known price for symbol, or None if unavailable."""
