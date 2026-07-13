"""Tests for portfolio.manager.PortfolioManager.

Performance-metric expectations are hand-computed with plain Python math
(independent of the numpy implementation under test).
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from core.models import Asset, AssetType, Position, Trade
from portfolio.manager import PortfolioManager

RISK_FREE = 0.02
DAILY_RF = RISK_FREE / 252

# A small, hand-checkable equity curve: up, down, up, down.
HISTORY_VALUES = [100000.0, 101000.0, 100500.0, 101500.0, 101000.0]


def _seed_history(pm: PortfolioManager, values, start: str = '2023-01-02') -> None:
    """Build portfolio_history by hand (only 'portfolio_value' is consumed
    by the return/drawdown metrics)."""
    dates = pd.bdate_range(start=start, periods=len(values))
    for ts, value in zip(dates, values):
        pm.portfolio_history.append({'timestamp': ts, 'portfolio_value': value})


def _samp_std(xs) -> float:
    """Sample standard deviation (ddof=1) — matches get_sharpe_ratio's
    np.std(ddof=1) convention (Phase 3 research-integrity switch from the
    population std / ddof=0)."""
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def _expected_daily_returns(values):
    return [values[i + 1] / values[i] - 1.0 for i in range(len(values) - 1)]


def _target_downside_dev(excess) -> float:
    """Target semideviation about zero excess return (Sortino denominator):
    sqrt(sum(min(x, 0)^2) / (N-1)) — the SAMPLE (ddof=1) convention matching
    get_sortino_ratio (Phase 3 switch from /N, ddof=0)."""
    return math.sqrt(sum(min(x, 0.0) ** 2 for x in excess) / (len(excess) - 1))


class TestPositionAccounting:
    def _position(self, symbol='AAPL', quantity=100, entry=150.0, current=155.0):
        return Position(
            asset=Asset(symbol, AssetType.STOCK),
            quantity=quantity,
            avg_entry_price=entry,
            current_price=current,
            timestamp=datetime(2023, 1, 2),
        )

    def test_initial_state(self):
        pm = PortfolioManager(100000.0)
        assert pm.cash == 100000.0
        assert pm.get_portfolio_value() == 100000.0
        assert pm.positions == {}

    def test_add_and_get_position(self):
        pm = PortfolioManager(100000.0)
        pos = self._position()
        pm.add_position(pos)
        assert pm.get_position(pos.asset) is pos

    def test_portfolio_value_is_cash_plus_positions(self):
        pm = PortfolioManager(100000.0)
        pm.update_cash(-15000.0)  # paid for the shares
        pm.add_position(self._position(quantity=100, entry=150.0, current=155.0))
        assert pm.cash == pytest.approx(85000.0)
        assert pm.get_portfolio_value() == pytest.approx(85000.0 + 100 * 155.0)

    def test_total_return_reconciles_to_nav_and_includes_cash_commissions(self):
        pm = PortfolioManager(100000.0)
        # Flat-price round trip whose only loss is two $10 cash commissions.
        pm.cash = 99980.0
        assert pm.get_realized_pnl() == 0.0
        assert pm.get_portfolio_pnl() == 0.0
        assert pm.get_total_return() == pytest.approx(-20.0)
        summary = pm.get_summary()
        assert summary['total_return'] == pytest.approx(
            summary['current_value'] - summary['initial_capital'])

    def test_unrealized_pnl(self):
        pm = PortfolioManager(100000.0)
        pm.add_position(self._position(quantity=100, entry=150.0, current=155.0))
        assert pm.get_portfolio_pnl() == pytest.approx(500.0)

    def test_close_position_full_records_trade_and_removes(self):
        pm = PortfolioManager(100000.0)
        pos = self._position(quantity=100, entry=150.0, current=160.0)
        pm.add_position(pos)

        pm.close_position(pos.asset, exit_price=160.0, quantity=100,
                          entry_time=pos.timestamp, exit_time=datetime(2023, 2, 1))

        assert pm.get_position(pos.asset) is None
        assert len(pm.closed_trades) == 1
        assert pm.closed_trades[0].pnl == pytest.approx(100 * (160.0 - 150.0))
        assert pm.get_realized_pnl() == pytest.approx(1000.0)

    def test_close_position_partial_decrements_quantity(self):
        pm = PortfolioManager(100000.0)
        pos = self._position(quantity=100, entry=150.0, current=160.0)
        pm.add_position(pos)

        pm.close_position(pos.asset, exit_price=160.0, quantity=40,
                          entry_time=pos.timestamp, exit_time=datetime(2023, 2, 1))

        remaining = pm.get_position(pos.asset)
        assert remaining is not None
        assert remaining.quantity == 60
        assert len(pm.closed_trades) == 1
        assert pm.closed_trades[0].pnl == pytest.approx(40 * 10.0)

    def test_close_unknown_position_is_noop(self):
        pm = PortfolioManager(100000.0)
        pm.close_position(Asset('ZZZ', AssetType.STOCK), 10.0, 1,
                          datetime(2023, 1, 2), datetime(2023, 1, 3))
        assert pm.closed_trades == []

    def test_zero_initial_capital_pnl_pct_is_zero_not_crash(self):
        # Regression: the guard used to test portfolio_value instead of
        # initial_capital (the actual divisor), so a zero-capital portfolio
        # that later gained value raised ZeroDivisionError — including
        # transitively through get_summary().
        pm = PortfolioManager(0.0)
        pm.update_cash(100.0)
        assert pm.get_portfolio_pnl_pct() == 0.0
        assert pm.get_summary()['total_return_pct'] == 0.0

    def test_total_loss_pnl_pct_is_minus_100(self):
        # A wiped-out portfolio is a real -100%, not 0% (the old
        # portfolio_value == 0 guard masked this).
        pm = PortfolioManager(100000.0)
        pm.update_cash(-100000.0)
        assert pm.get_portfolio_pnl_pct() == pytest.approx(-100.0)


class TestDailyReturns:
    def test_empty_history_returns_empty_list(self):
        pm = PortfolioManager(100000.0)
        assert pm.get_daily_returns() == []

    def test_single_snapshot_returns_empty_list(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, [100000.0])
        assert pm.get_daily_returns() == []

    def test_simple_returns_between_consecutive_snapshots(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, HISTORY_VALUES)
        expected = _expected_daily_returns(HISTORY_VALUES)
        assert pm.get_daily_returns() == pytest.approx(expected)


class TestSharpeRatio:
    def test_hand_computed_value(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, HISTORY_VALUES)

        returns = _expected_daily_returns(HISTORY_VALUES)
        excess = [r - DAILY_RF for r in returns]
        expected = (sum(excess) / len(excess)) / _samp_std(excess) * math.sqrt(252)

        assert pm.get_sharpe_ratio(risk_free_rate=RISK_FREE) == pytest.approx(expected)

    def test_uses_portfolio_history_not_closed_trades(self):
        # Regression for the old per-trade annualization bug: closed trades
        # alone must not produce a Sharpe ratio.
        pm = PortfolioManager(100000.0)
        pm.closed_trades.append(Trade(
            asset=Asset('AAPL', AssetType.STOCK),
            entry_price=100.0, exit_price=110.0, quantity=10,
            entry_time=datetime(2023, 1, 2), exit_time=datetime(2023, 1, 10),
        ))
        assert pm.get_sharpe_ratio() == 0.0

    def test_fewer_than_two_returns_is_zero(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, [100000.0, 101000.0])  # exactly one return
        assert pm.get_sharpe_ratio() == 0.0

    def test_constant_values_zero_std_is_zero(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, [100000.0] * 5)
        assert pm.get_sharpe_ratio() == 0.0


class TestSortinoRatio:
    def test_hand_computed_value(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, HISTORY_VALUES)

        returns = _expected_daily_returns(HISTORY_VALUES)
        excess = [r - DAILY_RF for r in returns]
        assert len([x for x in excess if x < 0]) >= 2  # sanity: real downside in fixture
        expected = (sum(excess) / len(excess)) / _target_downside_dev(excess) * math.sqrt(252)

        assert pm.get_sortino_ratio(risk_free_rate=RISK_FREE) == pytest.approx(expected)

    def test_no_returns_below_target_is_zero(self):
        # Monotonically rising curve, returns far above the daily risk-free
        # rate: target semideviation is exactly zero, and the documented
        # convention is 0.0 rather than infinity.
        pm = PortfolioManager(100000.0)
        _seed_history(pm, [100000.0, 101000.0, 102500.0, 104000.0])
        assert pm.get_sortino_ratio() == 0.0

    def test_single_negative_excess_return_is_finite(self):
        # Target semideviation is well-defined for a single loss. (The old
        # subset-std denominator was 0 here — the std of one point — and the
        # ratio degenerated to 0.0.)
        values = [100000.0, 99000.0, 100500.0, 102000.0]
        pm = PortfolioManager(100000.0)
        _seed_history(pm, values)

        returns = _expected_daily_returns(values)
        excess = [r - DAILY_RF for r in returns]
        assert len([x for x in excess if x < 0]) == 1
        expected = (sum(excess) / len(excess)) / _target_downside_dev(excess) * math.sqrt(252)

        assert pm.get_sortino_ratio(risk_free_rate=RISK_FREE) == pytest.approx(expected)

    def test_two_near_identical_losses_do_not_explode(self):
        # Regression: the old subset-std denominator was ~1e-17 for two
        # near-identical -1% losses, producing ratios around 6e12. The target
        # semideviation keeps the denominator at ~1% and the ratio sane.
        values = [100000.0, 99000.0, 98010.1]  # two ~-1% days, slightly different
        pm = PortfolioManager(100000.0)
        _seed_history(pm, values)

        returns = _expected_daily_returns(values)
        excess = [r - DAILY_RF for r in returns]
        expected = (sum(excess) / len(excess)) / _target_downside_dev(excess) * math.sqrt(252)

        result = pm.get_sortino_ratio(risk_free_rate=RISK_FREE)
        assert result == pytest.approx(expected)
        assert result < 0          # an all-loss curve must score negative
        assert abs(result) < 100   # ...at a sane magnitude, not 1e12

    def test_empty_history_is_zero(self):
        assert PortfolioManager(100000.0).get_sortino_ratio() == 0.0


class TestCalmarRatio:
    def test_hand_computed_value(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, HISTORY_VALUES)

        # n snapshots span only n - 1 return periods (the first snapshot is
        # the base value), so the annualization exponent uses n - 1.
        n = len(HISTORY_VALUES)
        annualized = (HISTORY_VALUES[-1] / HISTORY_VALUES[0]) ** (252 / (n - 1)) - 1.0
        # Max drawdown (fraction): trough 100500 against peak 101000.
        max_dd_fraction = (101000.0 - 100500.0) / 101000.0
        expected = annualized / max_dd_fraction

        # Cross-check that get_max_drawdown really reports a PERCENTAGE.
        assert pm.get_max_drawdown() == pytest.approx(-max_dd_fraction * 100)
        assert pm.get_calmar_ratio() == pytest.approx(expected)

    def test_zero_drawdown_is_zero(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, [100000.0, 101000.0, 102000.0])  # never draws down
        assert pm.get_calmar_ratio() == 0.0

    def test_insufficient_history_is_zero(self):
        pm = PortfolioManager(100000.0)
        assert pm.get_calmar_ratio() == 0.0
        _seed_history(pm, [100000.0])
        assert pm.get_calmar_ratio() == 0.0


class TestSummary:
    def test_summary_keeps_existing_keys_and_adds_new_ratios(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, HISTORY_VALUES)
        summary = pm.get_summary()

        expected_keys = {
            # Pre-existing keys the GUI depends on:
            'initial_capital', 'current_value', 'cash', 'total_return',
            'total_return_pct', 'realized_pnl', 'unrealized_pnl',
            'positions_count', 'closed_trades', 'win_rate', 'max_drawdown',
            'sharpe_ratio',
            # New in Phase 1:
            'sortino_ratio', 'calmar_ratio',
        }
        assert expected_keys.issubset(summary.keys())
        assert summary['sharpe_ratio'] == pytest.approx(pm.get_sharpe_ratio())
        assert summary['sortino_ratio'] == pytest.approx(pm.get_sortino_ratio())
        assert summary['calmar_ratio'] == pytest.approx(pm.get_calmar_ratio())

    def test_summary_has_research_integrity_keys(self):
        # Phase 3: psr / deflated_sharpe / n_trials are additive.
        pm = PortfolioManager(100000.0)
        _seed_history(pm, HISTORY_VALUES)
        summary = pm.get_summary()
        assert {'psr', 'deflated_sharpe', 'n_trials'} <= summary.keys()
        assert summary['n_trials'] == 1
        assert summary['psr'] is None or 0.0 <= summary['psr'] <= 1.0
        # n_trials == 1 -> no deflation, so DSR equals PSR exactly.
        assert summary['deflated_sharpe'] == summary['psr']

    def test_summary_n_trials_deflates_sharpe(self):
        pm = PortfolioManager(100000.0)
        # Positive-but-noisy curve: PSR is high yet < 1 and deflation bites,
        # so the STRICT inequality below actually exercises the n_trials wiring
        # (if n_trials were ignored both would equal PSR and the test fails).
        rets = [0.01, -0.006, 0.012, -0.004, 0.008, -0.007] * 10
        prices = [100000.0]
        for r in rets:
            prices.append(prices[-1] * (1.0 + r))
        _seed_history(pm, prices)
        base = pm.get_summary(n_trials=1)
        deflated = pm.get_summary(n_trials=50)
        assert deflated['n_trials'] == 50
        assert base['psr'] is not None
        assert deflated['deflated_sharpe'] is not None
        assert deflated['deflated_sharpe'] < base['psr']

    def test_summary_psr_none_when_history_too_short(self):
        pm = PortfolioManager(100000.0)
        _seed_history(pm, [100000.0, 100500.0])  # only one daily return
        summary = pm.get_summary()
        assert summary['psr'] is None
        assert summary['deflated_sharpe'] is None
