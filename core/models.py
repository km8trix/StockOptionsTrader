"""
Core data models for the trading system
"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional


class AssetType(Enum):
    """Asset types supported by the trading system"""
    STOCK = "stock"
    CALL = "call"
    PUT = "put"


class OrderType(Enum):
    """Order types"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Asset:
    """Represents a tradeable asset"""
    symbol: str
    asset_type: AssetType
    strike_price: Optional[float] = None
    expiration_date: Optional[str] = None

    @property
    def multiplier(self) -> int:
        """Contract multiplier: 100 for options (CALL/PUT), 1 for stock.

        Every position value, P&L and cash flow involving an option is
        the per-share price x this multiplier — the classic place sign
        and scale errors blow up an options book, so it lives in ONE
        spot and every consumer (Position, Trade, PortfolioManager,
        BacktestEngine) goes through it. Stock paths multiply by 1 and
        are byte-identical to their pre-Phase-8 behavior.
        """
        return 100 if self.asset_type in (AssetType.CALL, AssetType.PUT) else 1

    def __eq__(self, other):
        """Ensures assets with identical options parameters evaluate as the same key."""
        if not isinstance(other, Asset):
            return False
        return (self.symbol == other.symbol and 
                self.asset_type == other.asset_type and 
                self.strike_price == other.strike_price and 
                self.expiration_date == other.expiration_date)

    def __hash__(self):
        """Allows Asset objects to be safely hashed using all unique properties."""
        return hash((self.symbol, self.asset_type, self.strike_price, self.expiration_date))

    def __str__(self):
        if self.asset_type == AssetType.STOCK:
            return f"{self.symbol}"
        else:
            return f"{self.symbol} {self.expiration_date or 'N/A'} ${self.strike_price or 0.00} {self.asset_type.value}"


@dataclass
class Order:
    """Represents a trading order"""
    asset: Asset
    order_type: OrderType
    quantity: int
    price: Optional[float]  # None for MARKET orders (no limit price)
    timestamp: datetime
    order_id: str
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0


@dataclass
class Position:
    """Represents a current position in an asset.

    quantity is NEGATIVE for short positions (desk-mode SHORT fills).
    pnl() is sign-correct by construction: quantity * (current - entry).

    For OPTION assets quantity is in CONTRACTS and prices are per-share,
    so pnl() and position value carry the asset's contract multiplier
    (x100); stock paths multiply by 1 and are unchanged.
    """
    asset: Asset
    quantity: int
    avg_entry_price: float
    current_price: float
    timestamp: datetime

    def pnl(self) -> float:
        """Calculate unrealized P&L (contract-multiplier aware)."""
        return (self.quantity * (self.current_price - self.avg_entry_price)
                * self.asset.multiplier)

    def pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage (direction-aware).

        For shorts (quantity < 0) the sign is flipped so that a falling
        price reports a POSITIVE percentage, consistent with pnl().
        """
        if self.avg_entry_price == 0:
            return 0
        direction = -1.0 if self.quantity < 0 else 1.0
        return direction * ((self.current_price - self.avg_entry_price)
                            / self.avg_entry_price) * 100


@dataclass
class Trade:
    """Represents a completed trade.

    quantity is NEGATIVE for closed short trades (desk-mode COVER fills).
    pnl = quantity * (exit - entry) is sign-correct for negatives: a short
    covered below entry (exit < entry, quantity < 0) yields a POSITIVE
    pnl. pnl_pct mirrors that direction.

    For OPTION assets quantity is in CONTRACTS and prices per-share, so
    pnl carries the asset's contract multiplier (x100): a short option
    sold at 2.00 and covered at 1.00 (quantity -3) realizes
    -3 * (1.00 - 2.00) * 100 = +300. Stock pnl is unchanged (x1).
    """
    asset: Asset
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime

    def __post_init__(self):
        self._pnl = (self.quantity * (self.exit_price - self.entry_price)
                     * self.asset.multiplier)
        if self.entry_price != 0:
            direction = -1.0 if self.quantity < 0 else 1.0
            self._pnl_pct = direction * (
                (self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self._pnl_pct = 0
    
    @property
    def pnl(self) -> float:
        return self._pnl
    
    @property
    def pnl_pct(self) -> float:
        return self._pnl_pct