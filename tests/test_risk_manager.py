"""Tests for portfolio.risk_manager.RiskManager — every check, plus the
regression test for the entry_price -> avg_entry_price crash (Phase 1 B1)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from core.models import Asset, AssetType, Position
from portfolio.risk_manager import RiskManager


def _position(symbol='AAPL', quantity=100, entry=100.0, current=100.0):
    return Position(
        asset=Asset(symbol, AssetType.STOCK),
        quantity=quantity,
        avg_entry_price=entry,
        current_price=current,
        timestamp=datetime(2023, 1, 2),
    )


class TestCheckPositionSize:
    def test_within_limit_passes(self):
        rm = RiskManager(max_position_size=0.1)
        assert rm.check_position_size(portfolio_value=100000.0, position_size=5000.0)
        assert rm.violations == []

    def test_at_limit_passes(self):
        rm = RiskManager(max_position_size=0.1)
        assert rm.check_position_size(100000.0, 10000.0)

    def test_above_limit_fails_and_records_violation(self):
        rm = RiskManager(max_position_size=0.1)
        assert not rm.check_position_size(100000.0, 15000.0)
        assert len(rm.violations) == 1
        assert 'Position size' in rm.violations[0]


class TestCheckSectorExposure:
    def test_within_limit_passes(self):
        rm = RiskManager(max_sector_exposure=0.3)
        assert rm.check_sector_exposure({}, 'tech', new_size=20000.0,
                                        portfolio_value=100000.0)

    def test_above_limit_fails(self):
        rm = RiskManager(max_sector_exposure=0.3)
        assert not rm.check_sector_exposure({}, 'tech', new_size=40000.0,
                                            portfolio_value=100000.0)
        assert any('tech' in v for v in rm.violations)

    def test_existing_positions_without_sector_attribute_are_ignored(self):
        # core.models.Asset has no 'sector' field; the check must not crash
        # and must not count those positions toward sector exposure.
        rm = RiskManager(max_sector_exposure=0.3)
        positions = {'AAPL': _position(quantity=250, current=100.0)}  # 25k notional
        assert rm.check_sector_exposure(positions, 'tech', new_size=20000.0,
                                        portfolio_value=100000.0)


class TestCheckDailyLossLimit:
    def test_loss_within_limit_passes(self):
        rm = RiskManager(max_daily_loss=0.05)
        assert rm.check_daily_loss_limit(daily_pnl=-1000.0, portfolio_value=100000.0)
        assert rm.trading_allowed

    def test_loss_beyond_limit_halts_trading(self):
        rm = RiskManager(max_daily_loss=0.05)
        assert not rm.check_daily_loss_limit(daily_pnl=-6000.0, portfolio_value=100000.0)
        assert not rm.trading_allowed
        assert any('Daily loss' in v for v in rm.violations)

    def test_gains_never_trigger(self):
        rm = RiskManager(max_daily_loss=0.05)
        assert rm.check_daily_loss_limit(daily_pnl=50000.0, portfolio_value=100000.0)
        assert rm.trading_allowed


class TestCheckLeverage:
    def test_within_limit_passes(self):
        rm = RiskManager(max_leverage=1.0)
        assert rm.check_leverage(total_notional=50000.0, portfolio_value=100000.0)

    def test_above_limit_fails(self):
        rm = RiskManager(max_leverage=1.0)
        assert not rm.check_leverage(total_notional=150000.0, portfolio_value=100000.0)
        assert any('Leverage' in v for v in rm.violations)


class TestStopLoss:
    def test_calculate_position_stop_loss(self):
        rm = RiskManager(position_stop_loss=0.02)
        assert rm.calculate_position_stop_loss(100.0) == pytest.approx(98.0)

    def test_regression_b1_three_pct_below_entry_triggers_close(self):
        # Regression for the entry_price -> avg_entry_price crash: a position
        # 3% below entry must trigger the default 2% stop, not raise
        # AttributeError.
        rm = RiskManager()  # default position_stop_loss=0.02
        pos = _position(entry=100.0, current=97.0)
        assert rm.should_close_position(pos) is True

    def test_one_pct_below_entry_does_not_trigger(self):
        rm = RiskManager()
        pos = _position(entry=100.0, current=99.0)
        assert rm.should_close_position(pos) is False

    def test_exactly_at_stop_triggers(self):
        rm = RiskManager(position_stop_loss=0.02)
        pos = _position(entry=100.0, current=98.0)
        assert rm.should_close_position(pos) is True

    def test_zero_entry_price_is_guarded(self):
        rm = RiskManager()
        pos = _position(entry=0.0, current=10.0)
        assert rm.should_close_position(pos) is False

    def test_none_entry_price_is_guarded(self):
        rm = RiskManager()
        pos = _position(entry=100.0, current=97.0)
        pos.avg_entry_price = None  # defensive: degenerate runtime state
        assert rm.should_close_position(pos) is False


class TestCheckAllConstraints:
    def test_clean_portfolio_passes(self):
        rm = RiskManager()
        positions = {'AAPL': _position(quantity=100, current=100.0)}
        assert rm.check_all_constraints(positions, portfolio_value=100000.0,
                                        daily_pnl=500.0)
        assert rm.violations == []

    def test_violations_reset_between_calls(self):
        rm = RiskManager()
        rm.violations = ['stale violation']
        assert rm.check_all_constraints({}, 100000.0, 0.0)
        assert rm.violations == []

    def test_big_loss_and_overleverage_both_recorded(self):
        rm = RiskManager(max_daily_loss=0.05, max_leverage=1.0)
        positions = {'AAPL': _position(quantity=2000, current=100.0)}  # 200k notional
        assert not rm.check_all_constraints(positions, portfolio_value=100000.0,
                                            daily_pnl=-10000.0)
        assert len(rm.violations) == 2
        assert not rm.trading_allowed


class TestDegeneratePortfolioValueFailsClosed:
    """Zero/negative portfolio value must be an automatic violation (False +
    recorded violation), never a ZeroDivisionError — PLAN.md Phase 5 wires
    these checks into live order flow."""

    @pytest.mark.parametrize('portfolio_value', [0.0, -1000.0])
    def test_check_position_size_fails_closed(self, portfolio_value):
        rm = RiskManager()
        assert rm.check_position_size(portfolio_value, 1000.0) is False
        assert len(rm.violations) == 1
        assert 'failing closed' in rm.violations[0]

    @pytest.mark.parametrize('portfolio_value', [0.0, -1000.0])
    def test_check_sector_exposure_fails_closed(self, portfolio_value):
        rm = RiskManager()
        assert rm.check_sector_exposure({}, 'tech', 1000.0, portfolio_value) is False
        assert len(rm.violations) == 1

    @pytest.mark.parametrize('portfolio_value', [0.0, -1000.0])
    def test_check_daily_loss_limit_fails_closed_and_halts(self, portfolio_value):
        rm = RiskManager()
        assert rm.check_daily_loss_limit(-500.0, portfolio_value) is False
        assert not rm.trading_allowed
        assert len(rm.violations) == 1

    def test_check_daily_loss_limit_fails_closed_even_on_gains(self):
        # The kill-switch check cannot compute a loss percentage against a
        # degenerate base, so it fails closed regardless of the pnl sign.
        rm = RiskManager()
        assert rm.check_daily_loss_limit(500.0, 0.0) is False
        assert not rm.trading_allowed

    @pytest.mark.parametrize('portfolio_value', [0.0, -1000.0])
    def test_check_leverage_fails_closed(self, portfolio_value):
        rm = RiskManager()
        assert rm.check_leverage(1000.0, portfolio_value) is False
        assert len(rm.violations) == 1

    def test_check_all_constraints_records_violations_instead_of_crashing(self):
        rm = RiskManager()
        positions = {'AAPL': _position(quantity=100, current=100.0)}
        assert rm.check_all_constraints(positions, portfolio_value=0.0,
                                        daily_pnl=-500.0) is False
        # Both ratio checks (daily loss + leverage) fail closed.
        assert len(rm.violations) == 2
        assert not rm.trading_allowed


class TestReporting:
    def test_get_max_trade_size(self):
        rm = RiskManager(max_position_size=0.1)
        assert rm.get_max_trade_size(100000.0) == pytest.approx(10000.0)

    def test_get_report_shape(self):
        rm = RiskManager()
        report = rm.get_report()
        # 'unevaluated' is on the reporting surface so a consumer seeing
        # violations=[] can still tell that correlation/sector caps were
        # SKIPPED, not passed (no false green).
        assert set(report.keys()) == {
            'timestamp', 'trading_allowed', 'violations', 'unevaluated',
            'daily_loss', 'max_daily_loss',
        }
        assert report['trading_allowed'] is True
        assert report['violations'] == []
        assert report['unevaluated'] == []

    def test_get_report_surfaces_unevaluated_after_skipped_limits(self):
        rm = RiskManager()
        # No sector tags, no returns -> both caps unevaluated, recorded.
        rm.check_all_constraints({}, 100000.0, 0.0)
        report = rm.get_report()
        assert report['violations'] == []
        assert any('correlation' in u for u in report['unevaluated'])
        assert any('sector' in u for u in report['unevaluated'])


def _sector_pos(sector, notional):
    """A position whose asset carries a 'sector' tag (core.models.Asset has
    none, so concentration tests synthesize one)."""
    return SimpleNamespace(asset=SimpleNamespace(sector=sector),
                           current_price=notional, quantity=1)


class TestCheckCorrelation:
    """Phase 0: max_correlation was dead config — now a real, enforced check."""

    def test_perfectly_correlated_pair_violates(self):
        rm = RiskManager(max_correlation=0.8)
        a = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns = {'A': a, 'B': [2 * x + 1 for x in a]}  # corr == 1.0
        assert rm.check_correlation(returns) is False
        assert any('Correlation' in v for v in rm.violations)

    def test_anticorrelated_pair_also_violates_on_abs(self):
        rm = RiskManager(max_correlation=0.8)
        a = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns = {'A': a, 'B': [-x for x in a]}  # corr == -1.0, |corr| == 1
        assert rm.check_correlation(returns) is False

    def test_orthogonal_pair_passes(self):
        rm = RiskManager(max_correlation=0.8)
        returns = {'A': [1.0, -1.0, 1.0, -1.0, 0.0],
                   'B': [1.0, 1.0, -1.0, -1.0, 0.0]}  # corr == 0.0
        assert rm.check_correlation(returns) is True
        assert rm.violations == []

    def test_too_few_series_is_noop(self):
        rm = RiskManager(max_correlation=0.8)
        assert rm.check_correlation({'A': [0.01, 0.02, 0.03]}) is True
        assert rm.violations == []

    def test_short_series_recorded_unevaluated(self):
        rm = RiskManager(max_correlation=0.8)
        assert rm.check_correlation({'A': [0.01, 0.02], 'B': [0.01, 0.02]}) is True
        assert any('correlation' in u for u in rm.unevaluated)

    def test_zero_variance_series_recorded_unevaluated(self):
        # A flat (zero-variance) return series makes every np.corrcoef pair
        # NaN; the cap cannot be evaluated and must be RECORDED, not silently
        # passed (regression for the all-NaN-pairs silent-pass).
        rm = RiskManager(max_correlation=0.8)
        flat = [0.0, 0.0, 0.0, 0.0, 0.0]
        vary = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert rm.check_correlation({'A': flat, 'B': vary}) is True
        assert rm.violations == []
        assert any('correlation' in u for u in rm.unevaluated)


class TestCheckPortfolioSectorConcentration:
    def test_over_concentration_violates(self):
        rm = RiskManager(max_sector_exposure=0.3)
        positions = {'X': _sector_pos('tech', 40000.0)}  # 40% of 100k
        assert rm.check_portfolio_sector_concentration(positions, 100000.0) is False
        assert any('tech' in v for v in rm.violations)

    def test_within_limit_passes(self):
        rm = RiskManager(max_sector_exposure=0.3)
        positions = {'X': _sector_pos('tech', 20000.0)}  # 20%
        assert rm.check_portfolio_sector_concentration(positions, 100000.0) is True

    def test_positions_without_sector_are_noop(self):
        rm = RiskManager(max_sector_exposure=0.3)
        # core.models.Asset has no 'sector' field -> ignored, never crashes.
        positions = {'AAPL': _position(quantity=900, current=100.0)}  # 90k
        assert rm.check_portfolio_sector_concentration(positions, 100000.0) is True

    def test_fails_closed_on_nonpositive_value(self):
        rm = RiskManager()
        assert rm.check_portfolio_sector_concentration(
            {'X': _sector_pos('tech', 10000.0)}, 0.0) is False
        assert 'failing closed' in rm.violations[0]


class TestCheckAllConstraintsAdditions:
    def test_correlation_enforced_when_returns_supplied(self):
        rm = RiskManager()
        positions = {'AAPL': _position(quantity=100, current=100.0)}  # 10k, ok
        a = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns = {'A': a, 'B': [2 * x for x in a]}  # corr == 1.0 > 0.8
        assert rm.check_all_constraints(positions, 100000.0, 500.0,
                                        returns=returns) is False
        assert any('Correlation' in v for v in rm.violations)

    def test_unevaluated_recorded_when_no_data(self):
        # Sector + correlation cannot be evaluated (no tags, no returns) but
        # must NOT be silently ignored: they land in `unevaluated`, and the
        # call still passes on a clean book.
        rm = RiskManager()
        positions = {'AAPL': _position(quantity=100, current=100.0)}
        assert rm.check_all_constraints(positions, 100000.0, 500.0) is True
        assert rm.violations == []
        assert any('sector' in u for u in rm.unevaluated)
        assert any('correlation' in u for u in rm.unevaluated)
