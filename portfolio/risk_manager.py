"""
Risk Management Module
Enforces trading rules and position limits
"""

from __future__ import annotations

from typing import Dict, List
from core.models import Asset, Position
from datetime import datetime


class RiskManager:
    """
    Manages risk across the portfolio
    Enforces position limits, stop-losses, and risk thresholds
    """
    
    def __init__(self,
                 max_position_size: float = 0.1,        # Max 10% of portfolio per position
                 max_sector_exposure: float = 0.3,      # Max 30% in one sector
                 max_correlation: float = 0.8,          # Max correlation between positions
                 max_leverage: float = 1.0,             # Max leverage
                 max_daily_loss: float = 0.05,          # Max 5% daily loss before stopping
                 position_stop_loss: float = 0.02):     # 2% stop loss per position
        
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.max_leverage = max_leverage
        self.max_daily_loss = max_daily_loss
        self.position_stop_loss = position_stop_loss
        
        self.daily_loss = 0.0
        self.trading_allowed = True
        self.violations: List[str] = []

    def _fail_closed_on_invalid_portfolio_value(self, portfolio_value: float,
                                                check_name: str) -> bool:
        """Record a violation for a degenerate (zero or negative) portfolio value.

        Every ratio-based check divides by portfolio_value; a non-positive
        value makes the ratio meaningless. Risk checks must fail CLOSED in
        that case — record a violation and return False — rather than raise
        ZeroDivisionError into the order flow.
        """
        self.violations.append(
            f"{check_name}: portfolio value {portfolio_value:.2f} is not positive; "
            f"failing closed"
        )
        return False

    def check_position_size(self, portfolio_value: float, position_size: float) -> bool:
        """Check if position size is acceptable.

        A zero/negative portfolio value is an automatic violation (fail closed).
        """
        if portfolio_value <= 0:
            return self._fail_closed_on_invalid_portfolio_value(
                portfolio_value, 'Position size check')

        position_pct = position_size / portfolio_value
        
        if position_pct > self.max_position_size:
            self.violations.append(
                f"Position size {position_pct:.1%} exceeds max {self.max_position_size:.1%}"
            )
            return False
        
        return True
    
    def check_sector_exposure(self, positions: Dict[str, Position], new_sector: str, new_size: float, 
                             portfolio_value: float) -> bool:
        """Check sector concentration risk.

        A zero/negative portfolio value is an automatic violation (fail closed).
        """
        if portfolio_value <= 0:
            return self._fail_closed_on_invalid_portfolio_value(
                portfolio_value, 'Sector exposure check')

        sector_exposure = new_size
        
        for pos in positions.values():
            if hasattr(pos.asset, 'sector') and pos.asset.sector == new_sector:
                sector_exposure += pos.current_price * pos.quantity
        
        sector_pct = sector_exposure / portfolio_value
        
        if sector_pct > self.max_sector_exposure:
            self.violations.append(
                f"Sector {new_sector} exposure {sector_pct:.1%} exceeds max {self.max_sector_exposure:.1%}"
            )
            return False
        
        return True
    
    def check_daily_loss_limit(self, daily_pnl: float, portfolio_value: float) -> bool:
        """Check if daily loss exceeds limit.

        A zero/negative portfolio value is an automatic violation (fail
        closed) and halts trading, since this check is the kill switch and
        the loss percentage cannot be computed.
        """
        if portfolio_value <= 0:
            self.trading_allowed = False
            return self._fail_closed_on_invalid_portfolio_value(
                portfolio_value, 'Daily loss check')

        daily_loss_pct = abs(daily_pnl) / portfolio_value if daily_pnl < 0 else 0
        
        if daily_loss_pct > self.max_daily_loss:
            self.violations.append(
                f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.max_daily_loss:.1%}"
            )
            self.trading_allowed = False
            return False
        
        return True
    
    def check_leverage(self, total_notional: float, portfolio_value: float) -> bool:
        """Check leverage ratio.

        A zero/negative portfolio value is an automatic violation (fail closed).
        """
        if portfolio_value <= 0:
            return self._fail_closed_on_invalid_portfolio_value(
                portfolio_value, 'Leverage check')

        leverage = total_notional / portfolio_value
        
        if leverage > self.max_leverage:
            self.violations.append(
                f"Leverage {leverage:.2f}x exceeds max {self.max_leverage:.2f}x"
            )
            return False
        
        return True
    
    def calculate_position_stop_loss(self, entry_price: float) -> float:
        """Calculate stop-loss price for a position"""
        return entry_price * (1 - self.position_stop_loss)
    
    def should_close_position(self, position: Position) -> bool:
        """Check if position should be closed due to stop-loss.

        Uses Position.avg_entry_price (see core.models.Position). Returns
        False when the entry price is missing or non-positive, since no
        meaningful stop level can be computed in that case.
        """
        entry_price = getattr(position, 'avg_entry_price', None)
        if entry_price is None or entry_price <= 0:
            return False

        stop_price = self.calculate_position_stop_loss(entry_price)
        return position.current_price <= stop_price
    
    def check_all_constraints(self, positions: Dict[str, Position], 
                             portfolio_value: float, daily_pnl: float) -> bool:
        """Check all risk constraints"""
        self.violations = []
        
        # Check daily loss limit
        self.check_daily_loss_limit(daily_pnl, portfolio_value)
        
        # Check leverage
        total_notional = sum(pos.current_price * pos.quantity for pos in positions.values())
        self.check_leverage(total_notional, portfolio_value)
        
        return len(self.violations) == 0
    
    def get_max_trade_size(self, portfolio_value: float) -> float:
        """Get maximum allowed position size"""
        return portfolio_value * self.max_position_size
    
    def get_report(self) -> Dict:
        """Generate risk report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_allowed': self.trading_allowed,
            'violations': self.violations,
            'daily_loss': self.daily_loss,
            'max_daily_loss': self.max_daily_loss,
        }
