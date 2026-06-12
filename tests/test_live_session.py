"""LiveTradingSession (E5): intents flow to the executor, risk-blocked
intents never reach it, every step is audited in order, the kill switch
halts cleanly, paper-mode parity, and the C19 reconciliation wiring."""

from __future__ import annotations

import pytest

from core.models import Asset, AssetType, OrderType
from desks.base import DeskIntent
from utils.audit import AuditLog
from utils.kill_switch import KillSwitch
from utils.live_session import LiveTradingSession

SPY = Asset("SPY", AssetType.STOCK)
QQQ = Asset("QQQ", AssetType.STOCK)


class FakeDesk:
    key = "fake"
    capital_allocation = 0.5

    def __init__(self, intents, approved=None):
        self.intents = intents
        self.approved = approved if approved is not None else list(intents)
        self.generate_called = False

    def set_clock(self, now):
        self.clock = now

    def generate_intents(self, all_data, date, portfolio):
        self.generate_called = True
        return list(self.intents)

    def apply_risk(self, intents, portfolio, all_data, date):
        return [i for i in intents if i in self.approved]


class FakePortfolio:
    def get_portfolio_value(self):
        return 100_000.0


class FakeBroker:
    def __init__(self, price=50.0):
        self.price = price
        self.orders = []

    def place_order(self, asset, order_type, quantity, limit_price):
        self.orders.append((asset, order_type, quantity, limit_price))
        return f"ORD-{len(self.orders)}"

    def get_current_price(self, symbol):
        return self.price

    def get_portfolio_status(self):
        return {"cash": 100_000.0, "positions": []}


class FakeExecutor:
    def __init__(self, on_execute=None):
        self.calls = []
        self.on_execute = on_execute

    def execute(self, side, instrument, quantity, **kwargs):
        self.calls.append((side, instrument, quantity))
        if self.on_execute is not None:
            self.on_execute(len(self.calls))
        return {"status": "filled", "avg_fill": 50.0,
                "shortfall_per_unit": 0.0, "fills": []}


@pytest.fixture
def rails(tmp_path):
    db_path = str(tmp_path / "session.db")
    audit = AuditLog(db_path, env="sandbox")
    switch = KillSwitch(db_path, audit=audit)
    return audit, switch


def make_session(desk, broker, executor, audit, switch, **kwargs):
    return LiveTradingSession(
        desk=desk, broker=broker, portfolio=FakePortfolio(),
        data_fn=lambda: {"SPY": None}, executor=executor, audit=audit,
        kill_switch=switch, **kwargs)


class TestIntentFlow:
    def test_approved_intents_reach_executor_blocked_never_do(self, rails):
        audit, switch = rails
        buy_spy = DeskIntent(SPY, "BUY", 0.05, "signal", quantity=10)
        buy_qqq = DeskIntent(QQQ, "BUY", 0.50, "too big", quantity=5)
        desk = FakeDesk([buy_spy, buy_qqq], approved=[buy_spy])
        executor = FakeExecutor()
        session = make_session(desk, FakeBroker(), executor, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "ok"
        assert result["generated"] == 2 and result["approved"] == 1
        # ONLY the approved intent was executed; the risk-blocked one
        # never reached the executor.
        assert executor.calls == [("BUY", SPY, 10)]
        assert result["reports"][0]["status"] == "filled"

    def test_every_step_audited_in_order(self, rails):
        audit, switch = rails
        buy_spy = DeskIntent(SPY, "BUY", 0.05, "signal", quantity=10)
        sell_qqq = DeskIntent(QQQ, "SELL", 0.05, "exit", quantity=5)
        desk = FakeDesk([buy_spy, sell_qqq])
        session = make_session(desk, FakeBroker(), FakeExecutor(), audit,
                               switch)
        session.evaluate_once()
        events = [e["event_type"] for e in audit.entries(limit=50,
                                                         ascending=True)]
        assert events == [
            "session_evaluate",
            "desk_intent", "desk_intent",
            "execution_start", "execution_report",
            "execution_start", "execution_report",
        ]
        intent_rows = audit.entries(event_type="desk_intent",
                                    ascending=True)
        assert intent_rows[0]["payload"]["symbol"] == "SPY"
        assert intent_rows[1]["payload"]["action"] == "SELL"

    def test_sizing_fallback_when_intent_has_no_quantity(self, rails):
        audit, switch = rails
        intent = DeskIntent(SPY, "BUY", 0.1, "signal")  # no quantity
        desk = FakeDesk([intent])
        executor = FakeExecutor()
        session = make_session(desk, FakeBroker(price=50.0), executor,
                               audit, switch)
        session.evaluate_once()
        # 100k * 0.5 allocation * 0.1 fraction = $5k at $50 -> 100 shares.
        assert executor.calls == [("BUY", SPY, 100)]

    def test_unsizable_intent_skipped_and_audited(self, rails):
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.1, "signal")])
        broker = FakeBroker(price=50.0)
        broker.price = None  # quotes dark
        executor = FakeExecutor()
        session = make_session(desk, broker, executor, audit, switch)
        result = session.evaluate_once()
        assert executor.calls == []
        assert result["reports"] == []
        assert audit.entries(event_type="execution_skipped")


class TestKillSwitchHalts:
    def test_engaged_before_evaluation_halts_cleanly(self, rails):
        audit, switch = rails
        switch.engage("manual", actor="ops")
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        session = make_session(desk, FakeBroker(), executor, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert desk.generate_called is False  # nothing ran
        assert executor.calls == []
        halted = audit.entries(event_type="session_halted")
        assert halted[0]["payload"]["reason"] == "kill_switch_engaged"

    def test_mid_loop_engagement_stops_remaining_intents(self, rails):
        audit, switch = rails
        intents = [DeskIntent(SPY, "BUY", 0.05, "a", quantity=1),
                   DeskIntent(QQQ, "BUY", 0.05, "b", quantity=2)]
        desk = FakeDesk(intents)
        # First execution trips the breaker/operator: switch engages.
        executor = FakeExecutor(
            on_execute=lambda _n: switch.engage("breaker", "circuit"))
        session = make_session(desk, FakeBroker(), executor, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert len(executor.calls) == 1  # the second intent never ran


class TestExecutionErrorHalts:
    """A broker/auth failure during execution (midnight-ET token expiry
    mid-order is THE case) is a clean, audited stop: execution_report
    with the typed reason, session_halted, remaining intents never run —
    and evaluate_once returns a halted result instead of raising."""

    def _two_intents(self):
        return [DeskIntent(SPY, "BUY", 0.05, "a", quantity=10),
                DeskIntent(QQQ, "BUY", 0.05, "b", quantity=5)]

    def test_executor_raising_auth_expired_halts_and_audits(self, rails):
        from brokers.etrade_auth import EtradeAuthExpired

        audit, switch = rails

        class RaisingExecutor:
            calls = 0

            def execute(self, side, instrument, quantity, **kwargs):
                RaisingExecutor.calls += 1
                raise EtradeAuthExpired(
                    "E*TRADE access token hit the midnight-ET hard "
                    "expiry; re-authorization required")

        desk = FakeDesk(self._two_intents())
        session = make_session(desk, FakeBroker(), RaisingExecutor(),
                               audit, switch)
        result = session.evaluate_once()  # must NOT raise
        assert result["status"] == "halted"
        assert result["reason"] == "execution_error"
        assert RaisingExecutor.calls == 1  # the second intent never ran
        assert result["reports"][0]["status"] == "error"
        assert result["reports"][0]["error_type"] == "EtradeAuthExpired"
        events = [e["event_type"] for e in audit.entries(limit=50,
                                                         ascending=True)]
        # The trail CLOSES: no dangling execution_start.
        assert events == [
            "session_evaluate", "desk_intent", "desk_intent",
            "execution_start", "execution_report", "session_halted",
        ]
        report_row = audit.entries(event_type="execution_report")[0]
        assert report_row["payload"]["status"] == "error"
        assert report_row["payload"]["error_type"] == "EtradeAuthExpired"
        halted_row = audit.entries(event_type="session_halted")[0]
        assert halted_row["payload"]["reason"] == "execution_error"
        assert "midnight-ET" in halted_row["payload"]["error"]

    def test_executor_error_report_halts_remaining_intents(self, rails):
        """The PatientExecutor's own conversion path: a terminal 'error'
        report (no exception) must halt the remaining intents too."""
        audit, switch = rails

        class ErrorReportExecutor:
            calls = 0

            def execute(self, side, instrument, quantity, **kwargs):
                ErrorReportExecutor.calls += 1
                return {"status": "error", "fills": [],
                        "error": "transport died mid-poll",
                        "error_type": "EtradeUnavailable"}

        desk = FakeDesk(self._two_intents())
        session = make_session(desk, FakeBroker(), ErrorReportExecutor(),
                               audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "execution_error"
        assert ErrorReportExecutor.calls == 1
        halted_row = audit.entries(event_type="session_halted")[0]
        assert halted_row["payload"]["error_type"] == "EtradeUnavailable"

    def test_paper_path_place_order_raising_halts_and_audits(self, rails):
        audit, switch = rails

        class BrokenBroker(FakeBroker):
            def place_order(self, asset, order_type, quantity,
                            limit_price):
                raise ConnectionError("broker socket died")

        desk = FakeDesk(self._two_intents())
        session = make_session(desk, BrokenBroker(), None, audit, switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "execution_error"
        report_row = audit.entries(event_type="execution_report")[0]
        assert report_row["payload"]["status"] == "error"
        assert report_row["payload"]["error_type"] == "ConnectionError"


class FakeGate:
    """Zero-arg daily-loss gate double: scripted results (last sticky);
    exceptions in the queue are raised."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        item = (self.results.pop(0) if len(self.results) > 1
                else self.results[0])
        if isinstance(item, Exception):
            raise item
        return item


class TestCircuitBreakerHalts:
    """E5: the daily-loss gate is evaluated at the top of EVERY
    evaluate_once(), before any intent is generated or executed."""

    def test_breach_halts_before_any_intent_runs(self, rails):
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        gate = FakeGate({"breached": True, "loss_pct": 0.025,
                         "limit_pct": 0.02})
        session = make_session(desk, FakeBroker(), executor, audit, switch,
                               circuit_breaker=gate)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "daily_loss_circuit_breaker"
        assert desk.generate_called is False  # halted BEFORE intents
        assert executor.calls == []
        halted = audit.entries(event_type="session_halted")[0]
        assert halted["payload"]["reason"] == "daily_loss_circuit_breaker"
        assert halted["payload"]["loss_pct"] == 0.025

    def test_gate_error_fails_closed(self, rails):
        """A rail that cannot be read must halt the cycle, never wave it
        through."""
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        session = make_session(
            desk, FakeBroker(), executor, audit, switch,
            circuit_breaker=FakeGate(RuntimeError("balances dark")))
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "circuit_breaker_error"
        assert executor.calls == []
        halted = audit.entries(event_type="session_halted")[0]
        assert "balances dark" in halted["payload"]["error"]

    def test_clean_gate_runs_once_and_trading_proceeds(self, rails):
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=1)])
        executor = FakeExecutor()
        gate = FakeGate({"breached": False, "loss_pct": 0.001})
        session = make_session(desk, FakeBroker(), executor, audit, switch,
                               circuit_breaker=gate)
        result = session.evaluate_once()
        assert result["status"] == "ok"
        assert gate.calls == 1
        assert executor.calls == [("BUY", SPY, 1)]

    def test_gate_auto_discovered_from_broker(self, rails):
        """LiveEtradeBroker exposes its auto-wired gate as
        broker.circuit_breaker; the session picks it up by default."""
        audit, switch = rails
        broker = FakeBroker()
        broker.circuit_breaker = FakeGate({"breached": True,
                                           "loss_pct": 0.03,
                                           "limit_pct": 0.02})
        session = make_session(FakeDesk([]), broker, FakeExecutor(), audit,
                               switch)
        result = session.evaluate_once()
        assert result["status"] == "halted"
        assert result["reason"] == "daily_loss_circuit_breaker"
        assert broker.circuit_breaker.calls == 1

    def test_explicit_none_disables_discovery(self, rails):
        audit, switch = rails
        broker = FakeBroker()
        broker.circuit_breaker = FakeGate({"breached": True})
        session = make_session(FakeDesk([]), broker, FakeExecutor(), audit,
                               switch, circuit_breaker=None)
        assert session.evaluate_once()["status"] == "ok"
        assert broker.circuit_breaker.calls == 0


class TestPaperParityAndRenew:
    def test_no_executor_routes_direct_market_orders(self, rails):
        """Paper-mode parity: the same session drives PaperTrader-style
        brokers with direct place_order calls."""
        audit, switch = rails
        desk = FakeDesk([DeskIntent(SPY, "BUY", 0.05, "x", quantity=7)])
        broker = FakeBroker()
        session = make_session(desk, broker, None, audit, switch)
        result = session.evaluate_once()
        assert broker.orders == [(SPY, OrderType.BUY, 7, None)]
        assert result["reports"][0] == {"status": "submitted",
                                        "order_id": "ORD-1"}

    def test_opportunistic_renew_called(self, rails):
        audit, switch = rails

        class FakeAuth:
            renewed = 0

            def renew(self):
                FakeAuth.renewed += 1
                return True

        desk = FakeDesk([])
        session = make_session(desk, FakeBroker(), FakeExecutor(), audit,
                               switch, auth_manager=FakeAuth())
        session.evaluate_once()
        assert FakeAuth.renewed == 1


class TestReconciliationWiring:
    def test_mismatch_engages_kill_switch_and_audits(self, rails):
        audit, switch = rails
        desk = FakeDesk([])
        broker = FakeBroker()  # broker says: no positions, 100k cash
        session = make_session(desk, broker, FakeExecutor(), audit, switch)
        result = session.run_reconciliation({"AAPL": 100}, 100_000.0)
        assert result["ok"] is False
        assert switch.engaged() is True  # C19: caller engages the switch
        assert session.last_reconciliation == result
        rows = audit.entries(event_type="reconciliation")
        assert rows and rows[0]["payload"]["ok"] is False
        # And the halted book stays halted: next evaluation refuses.
        assert session.evaluate_once()["status"] == "halted"

    def test_clean_reconciliation_leaves_switch_alone(self, rails):
        audit, switch = rails
        session = make_session(FakeDesk([]), FakeBroker(), FakeExecutor(),
                               audit, switch)
        result = session.run_reconciliation({}, 100_000.0)
        assert result["ok"] is True
        assert switch.engaged() is False
