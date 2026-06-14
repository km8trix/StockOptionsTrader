"""
Portfolio Manager - Tracks positions, cash, and performance metrics
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from core.models import Asset, Position, Trade, Order, OrderType
from analysis.research_stats import (deflated_sharpe_ratio,
                                     probabilistic_sharpe_ratio)


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
        """Close (all or part of) a position and record the trade.

        Sign convention for shorts: a short position has NEGATIVE quantity,
        and `quantity` must carry the same sign as the position — a full
        cover of a -100 position passes quantity=-100, a partial cover of
        40 shares passes quantity=-40 (leaving -60 open). Trade.pnl =
        quantity * (exit - entry) is sign-correct for negatives.
        """
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
        """Calculate total portfolio value (cash + positions).

        Short positions (negative quantity) contribute NEGATIVE position
        value — correct under the cash-account approximation, because the
        short-sale proceeds already sit in cash; the net is the liquidation
        value of covering at the current price.

        OPTION positions (asset_type CALL/PUT) are valued per-share price
        x quantity x the contract multiplier (100); stock positions
        multiply by 1, so mixed stock/option books are exact and pure
        stock books are byte-identical to pre-Phase-8 behavior.
        """
        position_value = sum(
            pos.quantity * pos.current_price * pos.asset.multiplier
            for pos in self.positions.values())
        return self.cash + position_value
    
    def get_portfolio_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.pnl() for pos in self.positions.values())
    
    def get_portfolio_pnl_pct(self) -> float:
        """Calculate portfolio P&L percentage relative to initial capital.

        Returns 0.0 when initial_capital is zero or negative: a percentage
        return has no meaningful base in that case (the divisor is
        initial_capital, NOT the current portfolio value).
        """
        if self.initial_capital <= 0:
            return 0.0
        portfolio_value = self.get_portfolio_value()
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

    def get_drawdown_series(self) -> List[Dict]:
        """Per-snapshot drawdown from the running portfolio-value maximum.

        Returns [{'date': 'YYYY-MM-DD', 'drawdown_pct': float <= 0}, ...],
        one entry per portfolio_history snapshot, using the same math as
        get_max_drawdown (so min(drawdown_pct) == get_max_drawdown()).
        Returns [] when there is no history.
        """
        if not self.portfolio_history:
            return []

        values = [h['portfolio_value'] for h in self.portfolio_history]
        running_max = np.maximum.accumulate(values)

        # Avoid division runtime warnings if running_max somehow encounters 0
        running_max = np.where(running_max == 0, 1.0, running_max)

        drawdowns = (np.array(values) - running_max) / running_max * 100

        series: List[Dict] = []
        for snapshot, drawdown_pct in zip(self.portfolio_history, drawdowns):
            timestamp = snapshot['timestamp']
            if hasattr(timestamp, 'strftime'):
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                date_str = str(timestamp)[:10]
            series.append({'date': date_str,
                           'drawdown_pct': float(drawdown_pct)})
        return series

    def get_daily_returns(self) -> List[float]:
        """Simple returns between consecutive portfolio_history snapshots.

        Computed from each snapshot's 'portfolio_value'. Returns [] when
        there are fewer than 2 snapshots. Pairs whose previous value is
        non-positive are skipped defensively (a simple return is undefined
        for a zero or negative base).
        """
        if len(self.portfolio_history) < 2:
            return []

        values = [h['portfolio_value'] for h in self.portfolio_history]
        returns: List[float] = []
        for prev, curr in zip(values[:-1], values[1:]):
            if prev > 0:
                returns.append(curr / prev - 1.0)
        return returns

    def get_daily_returns_with_dates(self) -> List[Tuple[date, float]]:
        """Per-period simple returns, each tagged with the DATE of the later
        snapshot in the pair — the day the return realized.

        The return VALUES are identical, one-for-one, to get_daily_returns()
        (same 'previous value must be > 0' skip). Returns [] with fewer than 2
        snapshots. Used to partition the account return stream into walk-forward
        out-of-sample folds at fit-date boundaries (research integrity); the
        date is taken from each snapshot's 'timestamp'.
        """
        if len(self.portfolio_history) < 2:
            return []

        out: List[Tuple[date, float]] = []
        history = self.portfolio_history
        for prev_snap, curr_snap in zip(history[:-1], history[1:]):
            prev = prev_snap['portfolio_value']
            if prev > 0:
                ts = curr_snap['timestamp']
                day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
                out.append((day, curr_snap['portfolio_value'] / prev - 1.0))
        return out

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Annualized Sharpe ratio computed from DAILY portfolio returns.

        daily excess return = daily return - risk_free_rate / 252
        sharpe = mean(excess) / std(excess) * sqrt(252)

        Uses the SAMPLE standard deviation (np.std, ddof=1) — the conventional,
        more conservative estimator for a Sharpe computed from a return SAMPLE
        (this is what pandas .std() and most performance libraries use). This is
        a Phase 3 research-integrity change; the prior convention was the
        population std (ddof=0), which slightly understated the denominator and
        so overstated the Sharpe. Returns 0.0 when there are fewer than 2 daily
        returns or the std is zero.
        """
        returns = self.get_daily_returns()
        if len(returns) < 2:
            return 0.0

        excess_returns = np.array(returns) - (risk_free_rate / 252)
        std_dev = np.std(excess_returns, ddof=1)

        # Prevent division by zero runtime crash
        if std_dev == 0:
            return 0.0

        return float(np.mean(excess_returns) / std_dev * np.sqrt(252))

    def get_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Annualized Sortino ratio computed from DAILY portfolio returns.

        Numerator matches the Sharpe numerator (mean daily excess return).
        Denominator is the target downside deviation (target semideviation)
        about a zero excess return, summing squared downside excursions over
        ALL N daily returns but normalized by N-1 (the SAMPLE convention,
        ddof=1):

            downside_dev = sqrt(sum(min(excess_returns, 0) ** 2) / (N - 1))

        This measures the magnitude of returns below the target — the standard
        Sortino denominator — not the dispersion of the losing subset about its
        own mean. The N-1 normalization is the Phase 3 research-integrity change
        matching the Sharpe ddof=1 switch (was /N, ddof=0, before).

        Convention: returns 0.0 when there are fewer than 2 daily returns or
        when the downside deviation is zero (no returns below the target;
        downside risk is treated as undefined rather than infinite).
        """
        returns = self.get_daily_returns()
        if len(returns) < 2:
            return 0.0

        excess_returns = np.array(returns) - (risk_free_rate / 252)
        downside_dev = np.sqrt(
            np.sum(np.minimum(excess_returns, 0.0) ** 2)
            / (len(excess_returns) - 1))
        if downside_dev == 0:
            return 0.0

        return float(np.mean(excess_returns) / downside_dev * np.sqrt(252))

    def get_calmar_ratio(self) -> float:
        """Calmar ratio: annualized return / abs(max drawdown as a FRACTION).

        Annualized return = (last_value / first_value) ** (252 / (n_snapshots - 1)) - 1.
        n_snapshots history entries span only n_snapshots - 1 (daily) return
        periods — the first snapshot is the base value — so the exponent uses
        the period count, not the snapshot count.

        NOTE: get_max_drawdown() returns a PERCENTAGE (e.g. -12.5 for -12.5%),
        so it is divided by 100 here to obtain the fractional drawdown used in
        the denominator.

        Returns 0.0 when there are fewer than 2 snapshots, when the first or
        last portfolio value is non-positive, or when the drawdown is zero.
        """
        n_snapshots = len(self.portfolio_history)
        if n_snapshots < 2:
            return 0.0

        first_value = self.portfolio_history[0]['portfolio_value']
        last_value = self.portfolio_history[-1]['portfolio_value']
        if first_value <= 0 or last_value <= 0:
            return 0.0

        max_dd_fraction = abs(self.get_max_drawdown()) / 100.0
        if max_dd_fraction == 0:
            return 0.0

        annualized_return = (last_value / first_value) ** (252.0 / (n_snapshots - 1)) - 1.0
        return float(annualized_return / max_dd_fraction)
    
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
    
    def get_summary(self, n_trials: int = 1,
                    risk_free_rate: float = 0.02) -> Dict:
        """Get portfolio summary.

        ``n_trials`` (default 1) is the number of strategy trials / walk-forward
        refits used to DEFLATE the Sharpe for multiple testing (the engine
        passes the walk-forward fit count; 1 means no deflation, so
        deflated_sharpe == psr). See analysis/research_stats for the PSR/DSR
        conventions and caveats. The research-integrity keys are additive — they
        only ADD to the prior summary shape.

        PSR/DSR are computed on EXCESS daily returns (return - risk_free_rate /
        252) with the SAME risk_free_rate used for the reported Sharpe, so
        'psr' reads as P(true EXCESS Sharpe > 0) — consistent with the headline
        Sharpe rather than a raw-return Sharpe.
        """
        returns = self.get_daily_returns()
        excess_returns = [r - risk_free_rate / 252 for r in returns]
        n_trials = max(1, int(n_trials))
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
            'sharpe_ratio': self.get_sharpe_ratio(risk_free_rate),
            'sortino_ratio': self.get_sortino_ratio(risk_free_rate),
            'calmar_ratio': self.get_calmar_ratio(),
            # Research integrity (Phase 3): PSR = P(true excess Sharpe > 0);
            # deflated_sharpe = PSR vs the multiple-testing-inflated benchmark
            # from n_trials. Both None when undefined (<2 returns / zero var).
            'psr': probabilistic_sharpe_ratio(excess_returns, 0.0),
            'deflated_sharpe': deflated_sharpe_ratio(excess_returns, n_trials),
            'n_trials': n_trials,
        }
