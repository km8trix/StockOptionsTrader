"""
Risk Management Module
Enforces trading rules and position limits
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from core.models import Asset, Position
from datetime import datetime

import numpy as np


class RiskManager:
    """
    Manages risk across the portfolio
    Enforces position limits, stop-losses, and risk thresholds

    Limit enforcement, by where it actually runs today:
      - max_position_size, position_stop_loss, max_daily_loss, max_leverage:
        live — wired into the desk path and check_all_constraints.
      - max_sector_exposure: enforced by check_sector_exposure (prospective
        trade) and check_portfolio_sector_concentration (current book); the
        latter runs inside check_all_constraints ONLY when positions carry a
        'sector' tag (core.models.Asset has none yet).
      - max_correlation: enforced by check_correlation, evaluated inside
        check_all_constraints ONLY when a returns matrix is supplied.

    When sector/correlation data is absent these limits are recorded in
    `self.unevaluated` (NOT silently passed). Plumbing sector tags and a
    returns matrix into the live order path is the Phase 2 portfolio-risk
    aggregator's job; until then `unevaluated` makes the gap explicit.
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
        # Limits that could not be evaluated in the last check_all_constraints
        # call for lack of input data (sector tags / returns matrix). Tracked
        # so a configured limit is never SILENTLY ignored.
        self.unevaluated: List[str] = []

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

        BOUNDARY SEMANTICS: this desk rail is STRICT (loss > limit
        violates; a loss exactly AT the limit still trades) and
        deliberately diverges from the live daily-loss breaker
        (brokers.circuit_breaker.DailyLossCircuitBreaker), which is
        INCLUSIVE (loss >= limit breaches). The divergence fails safe:
        in live trading the inclusive breaker is the stricter outer
        rail, so do not "align" the two conventions.
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
    
    def check_portfolio_sector_concentration(self, positions: Dict[str, Position],
                                             portfolio_value: float) -> bool:
        """No single sector across the CURRENT book may exceed max_sector_exposure.

        Distinct from check_sector_exposure, which sizes a PROSPECTIVE new
        trade; this evaluates the existing holdings. Positions whose asset
        carries no 'sector' attribute are ignored (core.models.Asset has none
        today), so on a sector-less book this is a no-op that returns True.
        Uses absolute notional so shorts still count toward concentration.
        Fails closed on a non-positive portfolio value.
        """
        if portfolio_value <= 0:
            return self._fail_closed_on_invalid_portfolio_value(
                portfolio_value, 'Sector concentration check')

        by_sector: Dict[str, float] = {}
        for pos in positions.values():
            sector = getattr(pos.asset, 'sector', None)
            if not sector:
                continue
            by_sector[sector] = by_sector.get(sector, 0.0) + abs(
                pos.current_price * pos.quantity)

        ok = True
        for sector, notional in by_sector.items():
            sector_pct = notional / portfolio_value
            if sector_pct > self.max_sector_exposure:
                self.violations.append(
                    f"Sector {sector} concentration {sector_pct:.1%} exceeds "
                    f"max {self.max_sector_exposure:.1%}"
                )
                ok = False
        return ok

    def check_correlation(self,
                          returns: Dict[str, Sequence[float]]) -> bool:
        """Max pairwise |Pearson correlation| among held names <= max_correlation.

        `returns` maps symbol -> a sequence of periodic returns. Series shorter
        than 3 points are dropped; the remaining series must be of equal length
        for a correlation matrix to be defined. With fewer than 2 usable,
        equal-length series the check is a documented no-op (returns True and
        records the reason in `self.unevaluated`). On breach, records a
        violation naming the most-correlated pair.
        """
        if not returns or len(returns) < 2:
            return True

        series = {k: np.asarray(v, dtype=float)
                  for k, v in returns.items() if v is not None and len(v) >= 3}
        lengths = {arr.shape[0] for arr in series.values()}
        if len(series) < 2 or len(lengths) != 1:
            self.unevaluated.append(
                'correlation: returns matrix missing, ragged, or too short')
            return True

        symbols = list(series.keys())
        matrix = np.vstack([series[s] for s in symbols])
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(matrix)

        worst_abs = 0.0
        worst: Optional[tuple] = None
        n = len(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                c = corr[i, j]
                if np.isnan(c):
                    continue
                if abs(c) > worst_abs:
                    worst_abs = abs(c)
                    worst = (symbols[i], symbols[j], float(c))

        if worst is None:
            # Every pair was NaN — e.g. a zero-variance (flat/halted) series
            # makes np.corrcoef NaN, and those pairs are skipped above. The
            # cap could NOT be evaluated; record it rather than silently pass
            # (mirrors the ragged/short path that records unevaluated too).
            self.unevaluated.append(
                'correlation: all pairs NaN (zero-variance/constant series)')
            return True

        if worst_abs > self.max_correlation:
            a, b, c = worst
            self.violations.append(
                f"Correlation {abs(c):.2f} between {a} and {b} exceeds "
                f"max {self.max_correlation:.2f}"
            )
            return False
        return True

    def check_all_constraints(self, positions: Dict[str, Position],
                             portfolio_value: float, daily_pnl: float,
                             returns: Optional[Dict[str, Sequence[float]]] = None
                             ) -> bool:
        """Check all risk constraints.

        Always evaluates the daily-loss and leverage limits. Additionally
        evaluates sector concentration (only when positions carry 'sector'
        tags and the portfolio value is usable) and cross-position correlation
        (only when `returns` is supplied). Limits that cannot be evaluated for
        lack of input data are recorded in `self.unevaluated` rather than
        silently passed — see the class docstring for the enforcement map.
        """
        self.violations = []
        self.unevaluated = []

        # Check daily loss limit
        self.check_daily_loss_limit(daily_pnl, portfolio_value)

        # Check leverage
        total_notional = sum(pos.current_price * pos.quantity for pos in positions.values())
        self.check_leverage(total_notional, portfolio_value)

        # Sector concentration across the CURRENT book — only meaningful with
        # sector tags and a usable portfolio value (the degenerate-value case
        # is already failed closed above, so we do not re-flag it here).
        has_sector = any(getattr(p.asset, 'sector', None) for p in positions.values())
        if portfolio_value > 0 and has_sector:
            self.check_portfolio_sector_concentration(positions, portfolio_value)
        elif self.max_sector_exposure < 1.0:
            self.unevaluated.append(
                'sector concentration: positions carry no sector tags')

        # Cross-position correlation — only when a returns matrix is supplied.
        if returns:
            self.check_correlation(returns)
        elif self.max_correlation < 1.0:
            self.unevaluated.append('correlation: no returns matrix supplied')

        return len(self.violations) == 0
    
    def get_max_trade_size(self, portfolio_value: float) -> float:
        """Get maximum allowed position size"""
        return portfolio_value * self.max_position_size
    
    def get_report(self) -> Dict:
        """Generate risk report.

        Includes `unevaluated`: limits that the last check_all_constraints
        call could not evaluate for lack of input data. Exposing it on the
        reporting surface (not just self) prevents a false green — a consumer
        seeing violations=[] must be able to tell that the correlation/sector
        caps were SKIPPED, not passed.
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_allowed': self.trading_allowed,
            'violations': self.violations,
            'unevaluated': list(self.unevaluated),
            'daily_loss': self.daily_loss,
            'max_daily_loss': self.max_daily_loss,
        }
