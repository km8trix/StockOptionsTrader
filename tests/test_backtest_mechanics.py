"""Backtest integration for opt-in account and option mechanics."""

from datetime import date

import pandas as pd
import pytest

from backtesting.backtest_engine import BacktestEngine
from core.models import Asset, AssetType, Position
from desks.base import Desk
from portfolio.mechanics import (
    AccountMode,
    BorrowInventory,
    BorrowQuote,
    MarginPolicy,
    PortfolioMechanics,
)
from portfolio.option_lifecycle import OptionLifecyclePolicy
from portfolio.risk_manager import RiskManager
from strategies.base import Strategy


DAY = pd.Timestamp("2026-08-20")


class NoopStrategy(Strategy):
    def __init__(self):
        super().__init__("noop")

    def generate_signals(self, data, asset):
        return None


class NoopDesk(Desk):
    def __init__(self):
        super().__init__(
            "noop", "Noop", "", "#fff", risk_manager=RiskManager())

    def generate_intents(self, all_data, date, portfolio):
        return []

    def price_option(self, asset, underlying_frame, date, spot):
        return 1.0


def reg_t(*, quotes=(), debit_rate=0.0):
    return PortfolioMechanics(
        MarginPolicy(AccountMode.REG_T),
        borrow_inventory=BorrowInventory(quotes),
        annual_debit_rate=debit_rate,
    )


class TestMarginAndBorrowIntegration:
    def test_reg_t_allows_two_to_one_long_and_blocks_beyond_it(self):
        asset = Asset("AAA", AssetType.STOCK)
        engine = BacktestEngine(
            strategy=NoopStrategy(), initial_capital=100.0,
            commission=0.0, slippage_bps=0.0,
            portfolio_mechanics=reg_t())
        intent = {"signal": "BUY", "signal_date": DAY,
                  "days_waiting": 0}

        engine._fill_intent(asset, intent, 1.0, DAY, 2.0)

        assert engine.portfolio.get_position(asset).quantity == 200
        assert engine.portfolio.cash == -100.0
        assert engine.mechanics_log[0]["approved"] is True

        blocked = BacktestEngine(
            strategy=NoopStrategy(), initial_capital=100.0,
            commission=0.0, slippage_bps=0.0,
            portfolio_mechanics=reg_t())
        blocked._fill_intent(asset, intent, 1.0, DAY, 2.1)
        assert blocked.portfolio.positions == {}
        assert blocked.mechanics_log[0]["decision"] == (
            "INSUFFICIENT_BUYING_POWER")

    def test_short_requires_causal_locate_and_accrues_fee(self):
        asset = Asset("HTB", AssetType.STOCK)
        intent = {"signal": "SHORT", "signal_date": DAY,
                  "days_waiting": 0, "size_fraction": 0.50}
        no_locate = BacktestEngine(
            desk=NoopDesk(), initial_capital=100_000.0,
            commission=0.0, slippage_bps=0.0,
            portfolio_mechanics=reg_t())
        no_locate._fill_intent(asset, intent, 100.0, DAY, 0.5)
        assert no_locate.portfolio.positions == {}
        assert no_locate.mechanics_log[0]["decision"] == "BORROW_UNAVAILABLE"

        quote = BorrowQuote(
            "HTB", DAY.date(), 1_000, 0.36,
            valid_through=date(2026, 8, 31))
        engine = BacktestEngine(
            desk=NoopDesk(), initial_capital=100_000.0,
            commission=0.0, slippage_bps=0.0,
            portfolio_mechanics=reg_t(quotes=[quote]))
        engine._fill_intent(asset, intent, 100.0, DAY, 0.5)
        assert engine.portfolio.get_position(asset).quantity == -500
        cash_before = engine.portfolio.cash

        engine._accrue_financing(pd.Timestamp("2026-08-21"))

        assert engine.portfolio.cash == pytest.approx(cash_before - 50.0)
        assert engine.mechanics_log[-1]["borrow_fees"] == pytest.approx(50.0)

    def test_margin_debit_interest_hits_cash_causally(self):
        engine = BacktestEngine(
            strategy=NoopStrategy(), initial_capital=100.0,
            commission=0.0, slippage_bps=0.0,
            portfolio_mechanics=reg_t(debit_rate=0.36))
        engine.portfolio.cash = -1_000.0

        engine._accrue_financing(DAY)

        assert engine.portfolio.cash == pytest.approx(-1_001.0)
        assert engine.mechanics_log[-1]["debit_interest"] == 1.0

    def test_naked_short_option_is_blocked_under_mechanics(self):
        option = Asset("SPY", AssetType.PUT, 90.0, "2026-09-18")
        engine = BacktestEngine(
            desk=NoopDesk(), initial_capital=100_000.0,
            portfolio_mechanics=reg_t())
        intent = {"signal": "SHORT", "signal_date": DAY,
                  "size_fraction": 0.1, "quantity": 1}

        engine._fill_option_intent(option, intent, 1.0, DAY)

        assert engine.portfolio.positions == {}
        assert engine.mechanics_log[-1]["kind"] == "SHORT_OPTION"


class TestPhysicalOptionLifecycleIntegration:
    @staticmethod
    def _engine():
        return BacktestEngine(
            desk=NoopDesk(), initial_capital=50_000.0,
            option_lifecycle_policy=OptionLifecyclePolicy())

    @staticmethod
    def _frame(spot):
        return pd.DataFrame(
            {"open": [spot], "close": [spot], "high": [spot],
             "low": [spot], "volume": [1_000_000]},
            index=[pd.Timestamp("2026-08-21")])

    def test_long_call_exercises_into_stock_and_strike_cash(self):
        engine = self._engine()
        call = Asset("SPY", AssetType.CALL, 100.0, "2026-08-21")
        engine.portfolio.add_position(Position(
            call, 1, 2.0, 10.0, DAY))

        engine._settle_expired_options(
            {"SPY": self._frame(110.0)}, pd.Timestamp("2026-08-24"))

        stock = engine.portfolio.get_position(Asset("SPY", AssetType.STOCK))
        assert stock.quantity == 100
        assert stock.avg_entry_price == 100.0
        assert stock.current_price == 110.0
        assert engine.portfolio.get_position(call) is None
        assert engine.portfolio.cash == 40_000.0
        assert engine.mechanics_log[-1]["reason"] == "expiration_exercise"

    def test_short_put_assignment_creates_stock_even_with_margin_debit(self):
        engine = self._engine()
        put = Asset("SPY", AssetType.PUT, 100.0, "2026-08-21")
        engine.portfolio.add_position(Position(
            put, -1, 2.0, 10.0, DAY))
        engine.portfolio.cash = 1_000.0

        engine._settle_expired_options(
            {"SPY": self._frame(90.0)}, pd.Timestamp("2026-08-24"))

        stock = engine.portfolio.get_position(Asset("SPY", AssetType.STOCK))
        assert stock.quantity == 100
        assert engine.portfolio.cash == -9_000.0
        assert engine.mechanics_log[-1]["reason"] == "expiration_assignment"

    def test_ex_dividend_call_exercises_early_at_close(self):
        call = Asset("SPY", AssetType.CALL, 100.0, "2026-08-21")
        engine = BacktestEngine(
            desk=NoopDesk(), initial_capital=50_000.0,
            option_lifecycle_policy=OptionLifecyclePolicy(),
            dividend_fn=lambda symbol, on_date: 0.50)
        engine.portfolio.add_position(Position(
            call, 1, 2.0, 10.20, DAY))
        frame = pd.DataFrame(
            {"close": [110.0]}, index=[DAY])

        engine._process_early_option_lifecycle({"SPY": frame}, DAY)

        assert engine.portfolio.get_position(call) is None
        assert engine.portfolio.get_position(
            Asset("SPY", AssetType.STOCK)).quantity == 100
        assert engine.mechanics_log[-1]["reason"] == "early_exercise"

    def test_report_exposes_mechanics_only_when_enabled(self):
        engine = self._engine()
        engine.mechanics_log.append({
            "date": DAY, "event": "financing_accrual", "cash_delta": -1.0})
        report = engine._generate_report(benchmark_symbol=None)
        assert report["account_mechanics"][0]["date"] == "2026-08-20"

        legacy = BacktestEngine(strategy=NoopStrategy())
        assert "account_mechanics" not in legacy._generate_report(
            benchmark_symbol=None)
