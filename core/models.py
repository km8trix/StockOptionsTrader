"""
Core data models for the trading system
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


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
    
    def __init__(self, symbol: str, asset_type):
        self.symbol = symbol
        self.asset_type = asset_type

    def __eq__(self, other):
        """Ensures assets with identical symbols and types evaluate as the same key."""
        if not isinstance(other, Asset):
            return False
        return self.symbol == other.symbol and self.asset_type == other.asset_type

    def __hash__(self):
        """Allows Asset objects to be safely hashed and used as dictionary keys."""
        return hash((self.symbol, self.asset_type))

    def __str__(self):
        if self.asset_type == AssetType.STOCK:
            return f"{self.symbol}"
        else:
            return f"{self.symbol} {self.expiration_date} {self.strike_price} {self.asset_type.value}"


@dataclass
class Order:
    """Represents a trading order"""
    asset: Asset
    order_type: OrderType
    quantity: int
    price: float
    timestamp: datetime
    order_id: str
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0


@dataclass
class Position:
    """Represents a current position in an asset"""
    asset: Asset
    quantity: int
    avg_entry_price: float
    current_price: float
    timestamp: datetime
    
    def pnl(self) -> float:
        """Calculate unrealized P&L"""
        return self.quantity * (self.current_price - self.avg_entry_price)
    
    def pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.avg_entry_price == 0:
            return 0
        return ((self.current_price - self.avg_entry_price) / self.avg_entry_price) * 100


@dataclass
class Trade:
    """Represents a completed trade"""
    asset: Asset
    entry_price: float
    exit_price: float
    quantity: int
    entry_time: datetime
    exit_time: datetime
    
    def __post_init__(self):
        self._pnl = self.quantity * (self.exit_price - self.entry_price)
        if self.entry_price != 0:
            self._pnl_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self._pnl_pct = 0
    
    @property
    def pnl(self) -> float:
        return self._pnl
    
    @property
    def pnl_pct(self) -> float:
        return self._pnl_pct
